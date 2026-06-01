# -*- coding: utf-8 -*-
"""
人物弧线与关系图（v2 新增）

分析人物在影片中的:
- 弧线（成长/堕落/扁平/转变/救赎）
- 情绪轨迹
- 关键时刻
- 人物间关系及其演变
"""
import json
import time
from pathlib import Path
from collections import defaultdict

import config
from models.schemas import (
    CharacterDeep, CharacterArc, CharacterRelation,
    Event, Beat, TranscriptSegment,
)
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("CharacterArc")

ARC_PROMPT_TEMPLATE = """你是一个专业的影视叙事分析师。请分析以下人物的角色弧线和人物间关系。

=== 人物列表 ===
{characters_info}

=== 关键事件（时间顺序） ===
{events_info}

=== 人物间共现信息 ===
{cooccurrence_info}

请分析：

一、角色弧线（每个主要人物一条）
分析每个人物从开头到结尾的变化轨迹。

二、人物关系（每对有互动的人物一条）
分析人物间的关系类型和变化。

输出 JSON 对象，只输出 JSON：
```json
{{
  "arcs": [
    {{
      "character_id": "char_000",
      "arc_type": "growth",
      "arc_description": "从怯懦逐渐变得勇敢",
      "key_moments": [0, 3, 7],
      "emotion_trajectory": [
        {{"time": 10.0, "emotion": "恐惧", "intensity": 0.8}},
        {{"time": 60.0, "emotion": "决心", "intensity": 0.6}},
        {{"time": 120.0, "emotion": "勇气", "intensity": 0.9}}
      ]
    }}
  ],
  "relations": [
    {{
      "character_a": "char_000",
      "character_b": "char_001",
      "relation_type": "romantic",
      "description": "男女主角从相识到相爱",
      "strength": 0.8,
      "evolution": ["初识时互有好感", "经历考验后关系加深", "最终在一起"]
    }}
  ]
}}
```
"""


_PLACEHOLDER_TEXTS = {
    "无", "无变化", "无明显变化", "没有变化", "没有明显变化", "暂无",
    "暂无描述", "无法判断", "无法辨认", "不确定", "未知", "none",
    "null", "n/a", "na", "-", "--",
}


def _safe_text(value, default="") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _is_placeholder_text(value) -> bool:
    text = _safe_text(value)
    if not text:
        return True
    normalized = text.strip(" \t\r\n。.!！,，;；:：").lower()
    return normalized in _PLACEHOLDER_TEXTS


def _as_text_list(value) -> list[str]:
    if value is None:
        return []
    items = value if isinstance(value, list) else [value]
    texts = []
    for item in items:
        if item is None:
            continue
        text = _safe_text(item)
        if text and not _is_placeholder_text(text):
            texts.append(text)
    return texts


def _load_profile_map(video_dir: Path) -> dict:
    profiles_path = video_dir / "character_profiles.json"
    if not profiles_path.exists():
        return {}
    try:
        data = json.loads(profiles_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning(f"读取 character_profiles.json 失败，人物关系将不使用动态档案: {e}")
        return {}
    if not isinstance(data, list):
        return {}
    return {
        item.get("character_id"): item
        for item in data
        if isinstance(item, dict) and item.get("character_id")
    }


def _valid_profile_descriptions(items, key="description", limit=5) -> list[str]:
    values = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        text = _safe_text(item.get(key, ""))
        if text and not _is_placeholder_text(text) and text not in values:
            values.append(text)
    return values[-limit:]


def _profile_context(profile: dict | None) -> list[str]:
    if not profile:
        return []

    lines = []
    names = _as_text_list(profile.get("names", []))
    if names:
        lines.append(f"  档案称呼/别名: {', '.join(names[:5])}")

    appearances = _valid_profile_descriptions(
        profile.get("appearance_changes", []),
        key="description",
        limit=4,
    )
    if appearances:
        lines.append(f"  有效外观线索: {'; '.join(appearances)}")

    actions = _valid_profile_descriptions(
        profile.get("key_actions", []),
        key="action",
        limit=6,
    )
    if actions:
        lines.append(f"  关键行为线索: {'; '.join(actions)}")

    return lines


def _profile_description_fallback(profile: dict | None) -> str:
    if not profile:
        return ""
    desc = _safe_text(profile.get("description", ""))
    if desc and not _is_placeholder_text(desc):
        return desc
    appearances = _valid_profile_descriptions(
        profile.get("appearance_changes", []),
        key="description",
        limit=1,
    )
    return appearances[-1] if appearances else ""


def _is_temp_character_id(cid: str) -> bool:
    """chunk 作用域的临时未知角色（char_tmp_*）无法跨 chunk 归并，
    不应进入人物关系图，否则会污染稳定角色的 co_appearing/relations。"""
    return str(cid or "").startswith("char_tmp_")


def _relationship_character_ids(characters: list[CharacterDeep]) -> set:
    """关系图允许的角色集合：存在稳定角色时排除临时角色；
    若全是临时角色（无脸谱降级场景），则保留全部以免关系图为空。"""
    stable = {c.character_id for c in characters if not _is_temp_character_id(c.character_id)}
    return stable if stable else {c.character_id for c in characters}


def _build_cooccurrence_info(characters: list[CharacterDeep], limit=40) -> str:
    allowed = _relationship_character_ids(characters)
    pairs = []
    co_map = defaultdict(set)
    for i, char_a in enumerate(characters):
        if char_a.character_id not in allowed:
            continue
        scenes_a = set(char_a.appearance_scenes)
        if not scenes_a:
            continue
        for char_b in characters[i + 1:]:
            if char_b.character_id not in allowed:
                continue
            scenes_b = set(char_b.appearance_scenes)
            if not scenes_b:
                continue
            shared = sorted(scenes_a & scenes_b)
            if not shared:
                continue
            pairs.append((len(shared), char_a.character_id, char_b.character_id, shared))
            co_map[char_a.character_id].add(char_b.character_id)
            co_map[char_b.character_id].add(char_a.character_id)

    for c in characters:
        if co_map.get(c.character_id):
            c.co_appearing_characters = sorted(co_map[c.character_id])

    if not pairs:
        return "（无共现数据）"

    pairs.sort(key=lambda p: (-p[0], p[1], p[2]))
    lines = []
    for count, char_a, char_b, shared in pairs[:limit]:
        sample = ", ".join(str(si) for si in shared[:12])
        suffix = "..." if len(shared) > 12 else ""
        lines.append(f"- {char_a} 与 {char_b} 共现 {count} 个 shot: {sample}{suffix}")
    return "\n".join(lines)


def _select_events_for_prompt(events: list[Event], limit=60) -> list[Event]:
    selected = []
    seen = set()

    def add(event):
        if event.event_index in seen:
            return
        selected.append(event)
        seen.add(event.event_index)

    for event in events[:20]:
        add(event)
    for event in events:
        if getattr(event, "importance", 0) >= 7 or len(event.characters) >= 2:
            add(event)

    selected.sort(key=lambda e: e.start_time)
    return selected[:limit]


def analyze_character_arcs(
    video_id: str,
    characters: list[CharacterDeep],
    events: list[Event],
    beats: list[Beat],
    transcripts: list[TranscriptSegment],
    duration: float,
) -> tuple[list[CharacterDeep], list[CharacterRelation]]:
    """
    分析人物弧线和人物间关系。

    Args:
        video_id: 视频 ID
        characters: 深度人物列表
        events: 事件列表
        beats: Beat 列表
        transcripts: 台词列表
        duration: 视频总时长

    Returns:
        (更新了 arc 字段的 CharacterDeep 列表, CharacterRelation 列表)
    """
    video_dir = config.VIDEOS_DIR / video_id
    arcs_path = video_dir / "character_arcs.json"
    relations_path = video_dir / "character_relations.json"

    # 如果已存在，直接加载
    if arcs_path.exists() and relations_path.exists():
        logger.info("人物弧线和关系已存在，直接加载")
        arcs_data = json.loads(arcs_path.read_text(encoding="utf-8"))
        rels_data = json.loads(relations_path.read_text(encoding="utf-8"))
        # 回填到 characters
        arc_map = {a["character_id"]: CharacterArc(**a) for a in arcs_data}
        for c in characters:
            if c.character_id in arc_map:
                c.arc = arc_map[c.character_id]
        return characters, [CharacterRelation(**r) for r in rels_data]

    if not characters or len(characters) < 1:
        logger.info("人物不足，跳过弧线分析")
        return characters, []

    logger.info(f"开始人物弧线分析: {len(characters)} 个人物")

    # 统计台词
    char_dialogue_count = defaultdict(int)
    for t in transcripts:
        if t.character_id:
            char_dialogue_count[t.character_id] += 1

    # 更新 dialogue_count
    for c in characters:
        c.dialogue_count = char_dialogue_count.get(c.character_id, 0)

    # 计算 importance_score
    for c in characters:
        screen_pct = c.total_screen_time / max(duration, 1) if duration > 0 else 0
        dialogue_pct = c.dialogue_count / max(len(transcripts), 1) if transcripts else 0
        event_count = sum(1 for e in events if c.character_id in e.characters)
        event_pct = event_count / max(len(events), 1) if events else 0
        c.importance_score = round(
            screen_pct * 0.4 + dialogue_pct * 0.3 + event_pct * 0.3, 3
        )
        c.key_event_indices = [
            e.event_index for e in events if c.character_id in e.characters
        ]

    profile_map = _load_profile_map(video_dir)

    # 构造 prompt
    char_lines = []
    for c in characters:
        profile = profile_map.get(c.character_id)
        description = _safe_text(c.description)
        if _is_placeholder_text(description):
            description = _profile_description_fallback(profile) or "（暂无有效外观描述）"

        lines = [
            f"- {c.character_id} ({c.display_name}): {description}",
            (
                f"  出镜: {c.total_screen_time:.0f}s, 台词: {c.dialogue_count}句, "
                f"重要性: {c.importance_score:.2f}, "
                f"出场: {c.first_appearance:.0f}s-{c.last_appearance:.0f}s"
            ),
        ]
        lines.extend(_profile_context(profile))
        if c.key_event_indices:
            lines.append(
                "  关联事件: "
                + ", ".join(str(idx) for idx in c.key_event_indices[:12])
            )
        char_lines.append("\n".join(lines))
    characters_info = "\n".join(char_lines)

    selected_events = _select_events_for_prompt(events)
    event_lines = []
    for e in selected_events:
        event_lines.append(
            f"Event {e.event_index} [{e.start_time:.0f}s-{e.end_time:.0f}s] "
            f"[{e.event_type}] 人物: {','.join(e.characters)} "
            f"重要性:{getattr(e, 'importance', 5)} — {e.description}"
        )
    events_info = "\n".join(event_lines) if event_lines else "（无事件）"

    cooccurrence_info = _build_cooccurrence_info(characters)

    prompt = ARC_PROMPT_TEMPLATE.format(
        characters_info=characters_info,
        events_info=events_info,
        cooccurrence_info=cooccurrence_info,
    )

    # LLM 调用
    relations = []
    try:
        client = get_llm_client()
        response = client.chat(prompt=prompt, temperature=0.3)
        parsed = client.parse_json(response)

        if parsed and isinstance(parsed, dict):
            # 解析弧线
            arcs_data_out = []
            for arc_item in parsed.get("arcs", []):
                cid = arc_item.get("character_id", "")
                arc = CharacterArc(
                    character_id=cid,
                    arc_type=arc_item.get("arc_type", "flat"),
                    arc_description=arc_item.get("arc_description", ""),
                    key_moments=arc_item.get("key_moments", []),
                    emotion_trajectory=arc_item.get("emotion_trajectory", []),
                )
                # 回填到 character
                for c in characters:
                    if c.character_id == cid:
                        c.arc = arc
                        break
                arcs_data_out.append(arc.model_dump())

            # 解析关系（排除 chunk 作用域临时角色，避免污染关系图）
            allowed_rel_ids = _relationship_character_ids(characters)
            for rel_item in parsed.get("relations", []):
                rel_a = rel_item.get("character_a", "")
                rel_b = rel_item.get("character_b", "")
                if rel_a not in allowed_rel_ids or rel_b not in allowed_rel_ids:
                    continue
                rel = CharacterRelation(
                    character_a=rel_a,
                    character_b=rel_b,
                    relation_type=rel_item.get("relation_type", "stranger"),
                    description=rel_item.get("description", ""),
                    strength=float(rel_item.get("strength", 0.5)),
                    evolution=rel_item.get("evolution", []),
                )
                # 填充共现 shots
                char_a = next((c for c in characters if c.character_id == rel.character_a), None)
                char_b = next((c for c in characters if c.character_id == rel.character_b), None)
                if char_a and char_b:
                    rel.co_appearance_shots = sorted(
                        set(char_a.appearance_scenes) & set(char_b.appearance_scenes)
                    )
                relations.append(rel)

            # 保存弧线
            arcs_path.write_text(
                json.dumps(arcs_data_out, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # 保存关系
            relations_path.write_text(
                json.dumps([r.model_dump() for r in relations], indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            logger.info(
                f"人物弧线分析完成: {len(arcs_data_out)} 条弧线, "
                f"{len(relations)} 条关系"
            )
        else:
            logger.warning("人物弧线 LLM 结果解析失败")

    except Exception as e:
        logger.warning(f"人物弧线分析失败: {e}")

    # 更新 characters.json
    char_path = video_dir / "characters.json"
    char_path.write_text(
        json.dumps([c.model_dump() for c in characters], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return characters, relations
