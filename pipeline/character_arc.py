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

    # 构造 prompt
    char_lines = []
    for c in characters:
        char_lines.append(
            f"- {c.character_id} ({c.display_name}): {c.description}\n"
            f"  出镜: {c.total_screen_time:.0f}s, 台词: {c.dialogue_count}句, "
            f"重要性: {c.importance_score:.2f}, "
            f"出场: {c.first_appearance:.0f}s-{c.last_appearance:.0f}s"
        )
    characters_info = "\n".join(char_lines)

    event_lines = []
    for e in events[:30]:
        event_lines.append(
            f"Event {e.event_index} [{e.start_time:.0f}s-{e.end_time:.0f}s] "
            f"[{e.event_type}] 人物: {','.join(e.characters)} — {e.description}"
        )
    events_info = "\n".join(event_lines) if event_lines else "（无事件）"

    # 共现信息
    co_lines = []
    for c in characters:
        if c.co_appearing_characters:
            co_lines.append(
                f"- {c.character_id} 与 {', '.join(c.co_appearing_characters)} 共同出现"
            )
    cooccurrence_info = "\n".join(co_lines) if co_lines else "（无共现数据）"

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

            # 解析关系
            for rel_item in parsed.get("relations", []):
                rel = CharacterRelation(
                    character_a=rel_item.get("character_a", ""),
                    character_b=rel_item.get("character_b", ""),
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
