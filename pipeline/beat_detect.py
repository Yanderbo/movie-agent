# -*- coding: utf-8 -*-
"""
剧情节拍检测（v2 新增）

将连续 shots 按叙事节拍聚合为 beats。
Beat 是介于 shot 和 story_scene 之间的叙事微单元，
例如：一段对话、一个动作序列、一个情绪转折。

使用 LLM 分析 shot 的台词、画面摘要和人物信息，判断哪些
连续 shots 属于同一个 beat。
"""
import json
import time

import config
from models.schemas import (
    Shot, Beat, TranscriptSegment, VisionSummary, Character, MinuteChunk,
)
from utils import group_consecutive
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("BeatDetect")

PRIOR_BATCH_SIZE = 30

BEAT_PROMPT_TEMPLATE = """你是一个专业的影视叙事分析师。请分析以下镜头序列，将连续镜头按"叙事节拍"（Beat）分组。

一个 Beat 是由若干连续镜头组成的叙事微单元，通常对应：
- 一段完整的对话
- 一个连续的动作序列
- 一个情绪转折过程
- 一段环境展示/空镜
- 一个蒙太奇段落

=== 镜头列表 ===
{shots_info}

=== Step 5 先验 Beat 候选（已转换为全局 shot index） ===
{prior_info}

=== 已知角色名册 ===
（characters 字段只能填写以下角色 ID；画面中出现但不在名册内的人，不要写入 characters，可在 description 中描述，不要编造新的 char_ ID）
{character_roster}

请将这些镜头分组为若干 Beat，输出 JSON 数组。每个 Beat 包含：
- beat_index: 从 {beat_offset} 开始编号
- shot_indices: 包含的镜头索引列表（必须连续）
- beat_type: 类型（setup / confrontation / resolution / transition / montage / dialogue / action / reveal）
- description: 这个节拍讲了什么（一句话）
- emotion: 主要情绪
- intensity: 戏剧强度 (0.0 - 1.0)
- characters: 涉及的人物 ID 列表

规则：
1. 相邻且叙事连贯的镜头归为同一 Beat
2. 当场景/话题/情绪发生明显转换时，开启新 Beat
3. 每个 Beat 通常包含 2-8 个镜头，但不强制
4. 所有镜头都必须被分配到某个 Beat
5. characters 字段只能填写"已知角色名册"中的 ID；无法对应名册的人不要写入 characters，只在 description 中描述，不要编造新的 char_ ID
6. Step 5 先验是参考；普通 Prior Beat 来自单个 chunk 内部，边界基本可信，除非台词/画面明显冲突，不要过度重切
7. Boundary Fused Prior 是相邻 chunk 的边界融合候选（前一个 chunk 最后 1 个 prior + 后一个 chunk 第 1 个 prior），只有这类候选需要重点判断是否合并或拆分
8. 输出 shot_indices 必须使用上方 Shot 的全局索引，不要输出 chunk 内 local index

只输出 JSON：
```json
[
  {{
    "beat_index": {beat_offset},
    "shot_indices": [{example_shot_indices}],
    "beat_type": "dialogue",
    "description": "男女主角在咖啡厅讨论计划",
    "emotion": "轻松",
    "intensity": 0.3,
    "characters": ["char_000", "char_001"]
  }}
]
```
"""


def detect_beats(
    video_id: str,
    shots: list[Shot],
    transcripts: list[TranscriptSegment],
    vision_summaries: list[VisionSummary],
    characters: list[Character],  # 运行时通常为 CharacterDeep（Character 子类），也可能为空列表
    minute_chunks: list[MinuteChunk] | None = None,
) -> list[Beat]:
    """
    将连续 shots 按叙事节拍聚合为 beats。

    Args:
        video_id: 视频 ID
        shots: 镜头列表
        transcripts: 台词列表
        vision_summaries: 画面摘要列表
        characters: 人物列表，元素为 Character 或其子类 CharacterDeep，可能为空。
            仅用于构建"已知角色名册"并校验 LLM 返回的 beat.characters；
            字段访问全部走 getattr 防御，兼容缺失 description 等字段的情形。
        minute_chunks: Step 5 的 MinuteChunk 中间结果；其中 suggested_beats
            会先转换为全局 scene_index，再作为 Step 6 的软先验。

    Returns:
        Beat 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    beats_path = video_dir / "beats.json"

    # 如果已存在，直接加载（仍需回填 shot 的 beat_index）
    if beats_path.exists():
        logger.info(f"Beat 检测结果已存在，直接加载: {beats_path}")
        data = json.loads(beats_path.read_text(encoding="utf-8"))
        loaded_beats = [Beat(**b) for b in data]
        _backfill_beat_to_shots(shots, loaded_beats, video_id)
        return loaded_beats

    logger.info(f"开始 Beat 检测: {len(shots)} 个镜头")
    client = get_llm_client()

    valid_char_ids = set()
    roster_lines = []
    for c in (characters or []):
        cid = getattr(c, "character_id", None)
        if not cid:
            continue
        valid_char_ids.add(cid)
        line = f"- {cid}: {getattr(c, 'display_name', '') or cid}"
        desc = (getattr(c, "description", "") or "").strip()
        if desc:
            line += f" — {desc[:40]}"
        roster_lines.append(line)
    character_roster = "\n".join(roster_lines) or "（暂无已知角色，characters 字段请留空）"

    trans_by_shot = {}
    for t in transcripts:
        if t.scene_index >= 0:
            trans_by_shot.setdefault(t.scene_index, []).append(t)
    vision_by_shot = {v.scene_index: v for v in vision_summaries}
    shot_map = {s.scene_index: s for s in shots}

    all_beats = []
    beat_offset = 0
    prior_candidates = _build_prior_candidates(minute_chunks, shots, shot_map)

    if prior_candidates:
        logger.info(
            f"使用 Step 5 suggested_beats 先验检测 Beat: "
            f"{len(prior_candidates)} 个候选, batch_size={PRIOR_BATCH_SIZE}"
        )
        for start in range(0, len(prior_candidates), PRIOR_BATCH_SIZE):
            batch = prior_candidates[start: start + PRIOR_BATCH_SIZE]
            batch_shots = sorted({
                si: shot_map[si]
                for candidate in batch
                for si in candidate.get("shot_indices", [])
                if si in shot_map
            }.values(), key=lambda s: s.start_time)
            logger.info(
                f"  处理先验批次: candidates {start}-{start + len(batch) - 1}, "
                f"shots {batch_shots[0].scene_index if batch_shots else '?'}-"
                f"{batch_shots[-1].scene_index if batch_shots else '?'}"
            )
            beats = _detect_batch(
                client, batch_shots, _format_prior_info(batch),
                trans_by_shot, vision_by_shot, character_roster,
                valid_char_ids, beat_offset,
                _fallback_from_priors(batch, shot_map, beat_offset),
            )
            all_beats.extend(beats)
            beat_offset += len(beats)
            time.sleep(0.5)
    else:
        logger.info("未发现可用 Step 5 beat 先验，回退到按 shot 分段检测")
        for start in range(0, len(shots), 30):
            batch_shots = shots[start: start + 30]
            logger.info(
                f"  处理段: shot {batch_shots[0].scene_index}-"
                f"{batch_shots[-1].scene_index}"
            )
            beats = _detect_batch(
                client, batch_shots,
                "（本批没有 Step 5 先验候选，请直接根据镜头内容分组）",
                trans_by_shot, vision_by_shot, character_roster,
                valid_char_ids, beat_offset,
                _fallback_beats(batch_shots, beat_offset),
            )
            all_beats.extend(beats)
            beat_offset += len(beats)
            time.sleep(0.5)

    all_beats = _finalize_beats(all_beats, shots)
    _backfill_beat_to_shots(shots, all_beats, video_id)
    beats_path.write_text(
        json.dumps([b.model_dump() for b in all_beats], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Beat 检测完成: {len(all_beats)} 个 beat")
    return all_beats


def _detect_batch(
    client, batch_shots: list[Shot], prior_info: str,
    trans_by_shot: dict, vision_by_shot: dict,
    character_roster: str, valid_char_ids: set[str],
    beat_offset: int, fallback: list[Beat],
) -> list[Beat]:
    if not batch_shots:
        return []
    prompt = BEAT_PROMPT_TEMPLATE.format(
        shots_info=_format_shots_info(batch_shots, trans_by_shot, vision_by_shot),
        prior_info=prior_info,
        beat_offset=beat_offset,
        example_shot_indices=", ".join(str(s.scene_index) for s in batch_shots[:3]),
        character_roster=character_roster,
    )
    try:
        parsed = client.parse_json(client.chat(prompt=prompt, temperature=0.3))
    except Exception as e:
        logger.warning(f"Beat 检测失败: {e}，使用兜底分组")
        return fallback
    if not parsed or not isinstance(parsed, list):
        logger.warning("Beat 检测解析失败，使用兜底分组")
        return fallback
    beats = _beats_from_parsed(parsed, batch_shots, beat_offset, valid_char_ids)
    if not beats:
        logger.warning("Beat 检测结果为空，使用兜底分组")
        return fallback
    return beats


def _build_prior_candidates(
    minute_chunks: list[MinuteChunk] | None,
    shots: list[Shot],
    shot_map: dict[int, Shot],
) -> list[dict]:
    """将 MinuteChunk.suggested_beats 转为绝对 shot index 候选。"""
    if not minute_chunks or not shots:
        return []

    chunk_groups: list[tuple[int, list[dict]]] = []
    for chunk in sorted(
        minute_chunks,
        key=lambda c: (getattr(c, "start_time", 0.0), getattr(c, "chunk_index", 0)),
    ):
        chunk_shots = list(getattr(chunk, "shot_indices", []) or [])
        suggested = getattr(chunk, "suggested_beats", []) or []
        groups = []

        for group_index, raw_group in enumerate(suggested):
            if not isinstance(raw_group, list):
                continue
            shot_indices = []
            for raw_local_index in raw_group:
                local_index = _coerce_index(raw_local_index)
                if local_index is None or not (0 <= local_index < len(chunk_shots)):
                    continue
                scene_index = chunk_shots[local_index]
                if scene_index in shot_map and scene_index not in shot_indices:
                    shot_indices.append(scene_index)
            if not shot_indices:
                continue
            groups.append({
                "kind": "regular_prior",
                "chunk_index": chunk.chunk_index,
                "group_index": group_index,
                "shot_indices": shot_indices,
            })

        chunk_groups.append((chunk.chunk_index, groups))

    candidates = []
    last_chunk = len(chunk_groups) - 1
    for index, (chunk_index, groups) in enumerate(chunk_groups):
        if not groups:
            continue

        prev_has_groups = index > 0 and bool(chunk_groups[index - 1][1])
        next_has_groups = index < last_chunk and bool(chunk_groups[index + 1][1])
        first_regular = 1 if prev_has_groups else 0
        last_regular = len(groups) - (1 if next_has_groups else 0)
        candidates.extend(groups[first_regular:last_regular])

        if next_has_groups:
            next_chunk_index, next_groups = chunk_groups[index + 1]
            tail, head = groups[-1], next_groups[0]
            candidates.append({
                "kind": "boundary_fused_prior",
                "chunk_pair": (chunk_index, next_chunk_index),
                "shot_indices": list(dict.fromkeys(
                    tail["shot_indices"] + head["shot_indices"]
                )),
                "fallback_groups": [tail["shot_indices"], head["shot_indices"]],
            })

    if candidates:
        covered = {
            si
            for candidate in candidates
            for si in candidate.get("shot_indices", [])
        }
        uncovered = [s for s in shots if s.scene_index not in covered]
        for group in group_consecutive(uncovered, lambda s: s.scene_index):
            for start in range(0, len(group), 4):
                candidates.append({
                    "kind": "uncovered_prior_gap",
                    "shot_indices": [s.scene_index for s in group[start: start + 4]],
                })

    return sorted(
        candidates,
        key=lambda c: min(
            (shot_map[si].start_time for si in c.get("shot_indices", []) if si in shot_map),
            default=0.0,
        ),
    )


def _format_shots_info(
    seg_shots: list[Shot], trans_by_shot: dict, vision_by_shot: dict,
) -> str:
    shot_lines = []
    for s in seg_shots:
        parts = [f"Shot {s.scene_index} [{s.start_time:.1f}s-{s.end_time:.1f}s]"]

        trans = trans_by_shot.get(s.scene_index, [])
        if trans:
            trans_text = " ".join([t.text[:50] for t in trans[:3]])
            speaker = trans[0].speaker or "?"
            parts.append(f"台词[{speaker}]: {trans_text}")

        vis = vision_by_shot.get(s.scene_index)
        if vis:
            parts.append(f"画面: {vis.description[:60]}")
            if vis.mood:
                parts.append(f"情绪: {vis.mood}")
            if vis.scene_type:
                parts.append(f"类型: {vis.scene_type}")

        shot_lines.append(" | ".join(parts))
    return "\n".join(shot_lines)


def _format_prior_info(candidates: list[dict]) -> str:
    if not candidates:
        return "（本批没有 Step 5 先验候选，请直接根据镜头内容分组）"

    lines = [
        "说明：所有 prior 的 shot index 均已在本地转换为全局 scene_index。"
        "普通 Prior Beat 来自单个 chunk 内部，边界基本可信；"
        "Boundary Fused Prior 是 chunk 交界处融合候选，需要重点判断。"
    ]
    for i, candidate in enumerate(candidates):
        shots_text = ", ".join(str(si) for si in candidate.get("shot_indices", []))
        kind = candidate.get("kind")
        if kind == "boundary_fused_prior":
            c0, c1 = candidate.get("chunk_pair", ("?", "?"))
            lines.append(
                f"Boundary Fused Prior {i}: chunk {c0} tail + "
                f"chunk {c1} head, shots [{shots_text}]"
            )
            lines.append(
                "  说明：这是前一个 chunk 最后 1 个 prior 与后一个 chunk "
                "第 1 个 prior 融合后的范围，请重点判断是否合并、拆分或微调。"
            )
        elif kind == "uncovered_prior_gap":
            lines.append(
                f"Uncovered Shot Group {i}: shots [{shots_text}] "
                "（Step 5 未给出 suggested_beats，按内容判断）"
            )
        else:
            lines.append(
                f"Prior Beat {i}: chunk {candidate.get('chunk_index')}, "
                f"shots [{shots_text}]"
            )
    return "\n".join(lines)


def _beats_from_parsed(
    parsed: list, seg_shots: list[Shot], beat_offset: int,
    valid_char_ids: set[str],
) -> list[Beat]:
    beats = []
    shot_map = {s.scene_index: s for s in seg_shots}
    local_to_global = {i: s.scene_index for i, s in enumerate(seg_shots)}

    for item in parsed:
        if not isinstance(item, dict):
            continue
        raw_indices = item.get("shot_indices", [])
        if not isinstance(raw_indices, list) or not raw_indices:
            continue

        mapped = []
        seen = set()
        for raw_index in raw_indices:
            scene_index = _coerce_index(raw_index)
            if scene_index is None:
                continue
            if scene_index not in shot_map:
                scene_index = local_to_global.get(scene_index, scene_index)
            if scene_index in shot_map and scene_index not in seen:
                mapped.append(scene_index)
                seen.add(scene_index)

        beat_shots = [shot_map[si] for si in mapped]
        if not beat_shots:
            continue

        raw_chars = item.get("characters", []) or []
        if valid_char_ids:
            beat_chars = [
                cid for cid in raw_chars
                if isinstance(cid, str) and cid in valid_char_ids
            ]
        else:
            beat_chars = []

        beats.append(_make_beat(
            beat_offset + len(beats), beat_shots,
            beat_type=item.get("beat_type", ""),
            description=item.get("description", ""),
            emotion=item.get("emotion", ""),
            intensity=_coerce_float(item.get("intensity"), 0.0),
            characters=beat_chars,
        ))
    return beats


def _fallback_from_priors(
    candidates: list[dict], shot_map: dict[int, Shot], offset: int,
) -> list[Beat]:
    beats = []
    for candidate in candidates:
        groups = candidate.get("fallback_groups") or [candidate.get("shot_indices", [])]
        for group in groups:
            beat_shots = [shot_map[si] for si in group if si in shot_map]
            if not beat_shots:
                continue
            beats.append(_make_beat(offset + len(beats), beat_shots, "unknown"))
    return beats


def _make_beat(
    beat_index: int,
    beat_shots: list[Shot],
    beat_type: str = "",
    description: str = "",
    emotion: str = "",
    intensity: float = 0.0,
    characters: list[str] | None = None,
) -> Beat:
    start_time = min(s.start_time for s in beat_shots)
    end_time = max(s.end_time for s in beat_shots)
    return Beat(
        beat_index=beat_index,
        start_time=start_time,
        end_time=end_time,
        duration=end_time - start_time,
        shot_indices=sorted(s.scene_index for s in beat_shots),
        beat_type=beat_type,
        description=description,
        emotion=emotion,
        intensity=intensity,
        characters=characters or [],
    )


def _coerce_index(value):
    if isinstance(value, bool):
        return None
    try:
        if isinstance(value, float) and not value.is_integer():
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finalize_beats(beats: list[Beat], shots: list[Shot]) -> list[Beat]:
    """
    规范化 beats，保证它对 shots 构成一个完整且不重叠的划分：

    1. 跨 beat 去重 shot（保留先出现者），过滤非法 shot 索引；
    2. 把未被任何 beat 覆盖的 shot 按相邻关系聚合为 ``transition`` beat；
    3. 按时间统一排序，重排 beat_index 为连续唯一值，并重算时间范围/时长。

    这样可避免 LLM 漏分镜头导致 shot 脱离叙事层级（beat_index=None）。
    """
    if not shots:
        return []
    shot_map = {s.scene_index: s for s in shots}

    # 1. 去空 + 按时间排序 + 跨 beat 去重 shot
    beats = [b for b in beats if b.shot_indices]
    beats.sort(key=lambda b: b.start_time)
    seen: set[int] = set()
    for b in beats:
        kept = [si for si in b.shot_indices if si in shot_map and si not in seen]
        seen.update(kept)
        b.shot_indices = sorted(kept)
    beats = [b for b in beats if b.shot_indices]

    # 2. 未覆盖 shot → transition beat
    uncovered = [s for s in shots if s.scene_index not in seen]
    for group in group_consecutive(uncovered, lambda s: s.scene_index):
        beats.append(_make_beat(-1, group, "transition"))

    # 3. 统一重排并重算时间范围/时长（duration 采用墙钟跨度）
    beats.sort(key=lambda b: min(shot_map[si].start_time for si in b.shot_indices))
    for i, b in enumerate(beats):
        bs = [shot_map[si] for si in b.shot_indices]
        b.beat_index = i
        b.start_time = min(s.start_time for s in bs)
        b.end_time = max(s.end_time for s in bs)
        b.duration = b.end_time - b.start_time
    return beats


def _backfill_beat_to_shots(shots: list[Shot], beats: list[Beat], video_id: str):
    """回填 shot.beat_index 并持久化到 scenes.json"""
    beat_shot_map = {}
    for b in beats:
        for si in b.shot_indices:
            beat_shot_map[si] = b.beat_index
    for s in shots:
        s.beat_index = beat_shot_map.get(s.scene_index)
    # 持久化反向链接
    scenes_json = config.VIDEOS_DIR / video_id / "scenes" / "scenes.json"
    if scenes_json.exists():
        scenes_json.write_text(
            json.dumps([s.model_dump() for s in shots], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _fallback_beats(shots: list[Shot], offset: int) -> list[Beat]:
    """当 LLM 失败时，按每 3-5 个 shot 一组做默认分组"""
    beats = []
    group_size = 4
    for i in range(0, len(shots), group_size):
        group = shots[i: i + group_size]
        beats.append(_make_beat(offset + len(beats), group, "unknown"))
    return beats
