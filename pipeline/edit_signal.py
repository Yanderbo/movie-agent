# -*- coding: utf-8 -*-
"""
剪辑信号计算（v3 — 三类信号）

EditSignal（原有 8 维）+ NarrativeSignal（叙事层）+ RecompositionSignal（二创层）

返回: (edit_signals, narrative_signals, recomposition_signals)
"""
import json
import time

import config
from models.schemas import (
    Shot, Beat, StoryScene, Event, Character, VisionSummary,
    TranscriptSegment, EditSignal, NarrativeSignal, RecompositionSignal,
)
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("EditSignal")

SIGNAL_PROMPT_TEMPLATE = """你是一个专业的影视剪辑师。请为以下视频片段计算剪辑信号。

=== 片段列表 ===
{segments_info}

请为每个片段评估以下 8 个信号（0.0 - 1.0 分）：

1. hook_score: 作为视频开头/钩子的适合度（画面冲击力、悬念、吸引力）
2. plot_importance: 对整体剧情的贡献度（核心剧情=高，过渡/日常=低）
3. emotional_intensity: 情绪表达强度（强烈情绪=高，平静=低）
4. visual_impact: 视觉冲击力（特殊构图/运镜/特效=高，普通对话=低）
5. independence_score: 片段独立性（单独观看也能理解=高，需要上下文=低）
6. continuity_dependency: 连续性依赖（必须与前后片段连看=高，可独立剪出=低）
7. boundary_quality: 剪辑边界质量（开头/结尾有自然停顿=高，在句中/动作中=低）
8. spoiler_level: 剧透程度（包含关键反转/结局=高，日常场景=低）

同时建议每个片段适合的用途（可多选）：
- hook: 适合作为开头钩子
- trailer: 适合放入预告片
- highlight: 适合作为精彩集锦
- recap: 适合用于剧情回顾
- climax_clip: 适合作为高潮片段
- character_intro: 适合用于人物介绍

输出 JSON 数组，只输出 JSON：
```json
[
  {{
    "unit_index": 0,
    "hook_score": 0.8,
    "plot_importance": 0.7,
    "emotional_intensity": 0.9,
    "visual_impact": 0.6,
    "independence_score": 0.5,
    "continuity_dependency": 0.4,
    "boundary_quality": 0.7,
    "spoiler_level": 0.3,
    "suggested_usage": ["hook", "highlight"]
  }}
]
```
"""


def _ceil_div(total: int, batch_size: int) -> int:
    if total <= 0 or batch_size <= 0:
        return 0
    return (total + batch_size - 1) // batch_size


def _elapsed(started_at: float) -> str:
    return f"{time.perf_counter() - started_at:.1f}s"


def _unit_range_label(unit_type: str, batch) -> str:
    if not batch:
        return "-"

    def unit_index(unit):
        if unit_type == "shot":
            return getattr(unit, "scene_index", "?")
        if unit_type == "beat":
            return getattr(unit, "beat_index", "?")
        if unit_type == "story_scene":
            return getattr(unit, "story_scene_index", "?")
        return "?"

    return f"{unit_index(batch[0])}-{unit_index(batch[-1])}"


def compute_edit_signals(
    video_id: str,
    shots: list[Shot],
    beats: list[Beat],
    story_scenes: list[StoryScene],
    events: list[Event],
    characters: list[Character],
    transcripts: list[TranscriptSegment],
    vision_summaries: list[VisionSummary],
) -> tuple[list[EditSignal], list[NarrativeSignal], list[RecompositionSignal]]:
    """
    为 shot / beat / story_scene 计算三类信号。

    Returns:
        (EditSignal 列表, NarrativeSignal 列表, RecompositionSignal 列表)
    """
    video_dir = config.VIDEOS_DIR / video_id
    signals_path = video_dir / "edit_signals.json"
    started_at = time.perf_counter()
    logger.info(
        "EditSignal start: "
        f"shots={len(shots or [])}, beats={len(beats or [])}, "
        f"story_scenes={len(story_scenes or [])}, events={len(events or [])}, "
        f"characters={len(characters or [])}, transcripts={len(transcripts or [])}, "
        f"vision_summaries={len(vision_summaries or [])}, "
        f"max_shots={config.EDIT_SIGNAL_MAX_SHOTS}"
    )

    # 如果已存在，直接加载（需加载全部三类信号）
    if signals_path.exists():
        logger.info(f"剪辑信号已存在，直接加载: {signals_path}")
        data = json.loads(signals_path.read_text(encoding="utf-8"))
        edit_sigs = [EditSignal(**s) for s in data]
        # 尝试加载 NarrativeSignal 和 RecompositionSignal
        ns_path = video_dir / "narrative_signals.json"
        rs_path = video_dir / "recomposition_signals.json"
        ns = []
        rs = []
        if ns_path.exists():
            ns = [NarrativeSignal(**s) for s in json.loads(ns_path.read_text(encoding="utf-8"))]
        if rs_path.exists():
            rs = [RecompositionSignal(**s) for s in json.loads(rs_path.read_text(encoding="utf-8"))]
        logger.info(
            "EditSignal cache loaded: "
            f"edit={len(edit_sigs)}, narrative={len(ns)}, recomposition={len(rs)}"
        )

        # v2→v3 升级：EditSignal 已有但 NarrativeSignal/RecompositionSignal 未计算
        if not ns or not rs:
            logger.info(
                "EditSignal cache is partial; recomputing missing: "
                f"narrative_missing={not ns}, recomposition_missing={not rs}"
            )
            client = get_llm_client()
            if not ns:
                ns = _compute_narrative_signals(
                    client, video_dir, beats, story_scenes, events, meta_duration=0,
                )
            if not rs:
                rs = _compute_recomposition_signals(
                    client, video_dir, beats, story_scenes, events,
                    transcripts, vision_summaries,
                )

        logger.info(
            "EditSignal done from cache: "
            f"edit={len(edit_sigs)}, narrative={len(ns)}, recomposition={len(rs)}, "
            f"elapsed={_elapsed(started_at)}"
        )
        return edit_sigs, ns, rs

    logger.info("开始计算剪辑信号")
    client = get_llm_client()
    logger.info("EditSignal cache miss: running fresh signal computation")

    all_signals = []

    # ── 为 beat 计算信号（beat 是剪辑的核心粒度）──
    if beats:
        phase_started = time.perf_counter()
        beat_signals = _compute_signals_for_units(
            client, "beat", beats, events, characters, transcripts, vision_summaries
        )
        all_signals.extend(beat_signals)
        logger.info(
            "EditSignal phase done: "
            f"unit_type=beat, requested={len(beats)}, produced={len(beat_signals)}, "
            f"elapsed={_elapsed(phase_started)}"
        )
    else:
        logger.info("EditSignal phase skipped: unit_type=beat, reason=no_units")

    # ── 为 story_scene 计算信号 ──
    if story_scenes:
        phase_started = time.perf_counter()
        scene_signals = _compute_signals_for_units(
            client, "story_scene", story_scenes, events, characters,
            transcripts, vision_summaries
        )
        all_signals.extend(scene_signals)
        logger.info(
            "EditSignal phase done: "
            f"unit_type=story_scene, requested={len(story_scenes)}, "
            f"produced={len(scene_signals)}, elapsed={_elapsed(phase_started)}"
        )
    else:
        logger.info("EditSignal phase skipped: unit_type=story_scene, reason=no_units")

    # ── 为代表性 shot 计算信号（有上限，避免重要事件覆盖导致全量 shot 计算）──
    important_shots = _select_representative_shots(
        shots, beats, story_scenes, events,
        max_shots=config.EDIT_SIGNAL_MAX_SHOTS,
    )
    if important_shots:
        phase_started = time.perf_counter()
        shot_signals = _compute_signals_for_units(
            client, "shot", important_shots, events, characters,
            transcripts, vision_summaries
        )
        all_signals.extend(shot_signals)
        logger.info(
            "EditSignal phase done: "
            f"unit_type=shot, requested={len(important_shots)}, "
            f"produced={len(shot_signals)}, elapsed={_elapsed(phase_started)}"
        )
    else:
        logger.info("EditSignal phase skipped: unit_type=shot, reason=no_selected_shots")

    # 保存 EditSignal
    signals_path.write_text(
        json.dumps([s.model_dump() for s in all_signals], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        f"EditSignal base signals saved: count={len(all_signals)}, "
        f"path={signals_path}, elapsed={_elapsed(started_at)}"
    )

    # ── 计算 NarrativeSignal ──
    narrative_signals = _compute_narrative_signals(
        client, video_dir, beats, story_scenes, events, meta_duration=0,
    )

    # ── 计算 RecompositionSignal ──
    recomposition_signals = _compute_recomposition_signals(
        client, video_dir, beats, story_scenes, events,
        transcripts, vision_summaries,
    )

    logger.info(
        "EditSignal complete: "
        f"edit={len(all_signals)}, narrative={len(narrative_signals)}, "
        f"recomposition={len(recomposition_signals)}, elapsed={_elapsed(started_at)}"
    )
    return all_signals, narrative_signals, recomposition_signals


def _select_representative_shots(
    shots: list[Shot],
    beats: list[Beat],
    story_scenes: list[StoryScene],
    events: list[Event],
    max_shots: int,
) -> list[Shot]:
    if not shots or max_shots <= 0:
        logger.info(
            f"shot级剪辑信号选择: total={len(shots or [])}, candidates=0, "
            "selected=0, skipped=0"
        )
        return []

    shot_map = {s.scene_index: s for s in shots}
    candidate_scores = {}
    candidate_sources = {"beat": set(), "story_scene": set(), "event": set()}
    high_event_count = 0

    def add_candidate(scene_index: int, score: float, source: str):
        if scene_index not in shot_map:
            return
        candidate_scores[scene_index] = max(candidate_scores.get(scene_index, 0.0), score)
        if source in candidate_sources:
            candidate_sources[source].add(scene_index)

    for b in beats or []:
        if not b.shot_indices:
            continue
        score = 1.0 + float(getattr(b, "intensity", 0) or 0)
        add_candidate(b.shot_indices[0], score, "beat")
        add_candidate(b.shot_indices[-1], score, "beat")

    for ss in story_scenes or []:
        if not ss.shot_indices:
            continue
        add_candidate(ss.shot_indices[0], 1.2, "story_scene")
        add_candidate(ss.shot_indices[-1], 1.2, "story_scene")

    for e in events or []:
        if getattr(e, "importance", 0) < 7:
            continue
        high_event_count += 1
        indices = [si for si in e.scene_indices if si in shot_map]
        if not indices:
            indices = [
                s.scene_index for s in shots
                if e.start_time < s.end_time and e.end_time > s.start_time
            ]
        if not indices:
            continue
        indices = sorted(dict.fromkeys(indices))
        mid = indices[len(indices) // 2]
        event_score = 2.0 + (float(getattr(e, "importance", 0) or 0) / 10.0)
        for si in {indices[0], mid, indices[-1]}:
            add_candidate(si, event_score, "event")

    if not candidate_scores:
        selected = _sample_shots_evenly(shots, min(max_shots, len(shots)))
        logger.info(
            f"shot级剪辑信号选择: total={len(shots)}, candidates=0, "
            f"selected={len(selected)}, skipped={max(0, len(shots) - len(selected))}, "
            f"limit={max_shots}, high_events={high_event_count}"
        )
        return selected

    candidate_ids = sorted(
        candidate_scores,
        key=lambda si: (-candidate_scores[si], shot_map[si].start_time),
    )

    if len(candidate_ids) <= max_shots:
        selected_ids = set(candidate_ids)
    else:
        score_quota = max(1, int(max_shots * 0.75))
        selected_ids = set(candidate_ids[:score_quota])
        remaining = [shot_map[si] for si in candidate_ids if si not in selected_ids]
        fill_count = max_shots - len(selected_ids)
        for shot in _sample_shots_evenly(remaining, fill_count):
            selected_ids.add(shot.scene_index)

    selected = [s for s in shots if s.scene_index in selected_ids]
    logger.info(
        f"shot级剪辑信号选择: total={len(shots)}, candidates={len(candidate_scores)}, "
        f"selected={len(selected)}, skipped={max(0, len(candidate_scores) - len(selected))}, "
        f"limit={max_shots}, high_events={high_event_count}, "
        f"source_candidates=beat:{len(candidate_sources['beat'])},"
        f"story_scene:{len(candidate_sources['story_scene'])},"
        f"event:{len(candidate_sources['event'])}"
    )
    return selected


def _sample_shots_evenly(shots: list[Shot], limit: int) -> list[Shot]:
    if not shots or limit <= 0:
        return []
    ordered = sorted(shots, key=lambda s: s.start_time)
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[0]]

    step = (len(ordered) - 1) / (limit - 1)
    sampled = []
    seen = set()
    for i in range(limit):
        idx = round(i * step)
        if idx in seen:
            continue
        sampled.append(ordered[idx])
        seen.add(idx)
    return sampled


def _compute_signals_for_units(
    client,
    unit_type: str,
    units,
    events: list[Event],
    characters: list[Character],
    transcripts: list[TranscriptSegment],
    vision_summaries: list[VisionSummary],
) -> list[EditSignal]:
    """为一组 unit 计算剪辑信号"""
    units = list(units or [])
    signals = []
    batch_size = 15
    total_batches = _ceil_div(len(units), batch_size)
    phase_started = time.perf_counter()
    logger.info(
        "EditSignal unit compute start: "
        f"unit_type={unit_type}, units={len(units)}, "
        f"batch_size={batch_size}, batches={total_batches}"
    )

    # 构建辅助索引
    trans_map = {}
    for t in transcripts or []:
        trans_map.setdefault(t.scene_index, []).append(t)
    vision_map = {v.scene_index: v for v in vision_summaries or []}

    for batch_start in range(0, len(units), batch_size):
        batch = units[batch_start: batch_start + batch_size]
        batch_index = batch_start // batch_size + 1
        batch_started = time.perf_counter()

        # 构造片段信息
        seg_lines = []
        for unit in batch:
            if unit_type == "shot":
                idx = unit.scene_index
                start = unit.start_time
                end = unit.end_time
                trans = trans_map.get(idx, [])
                vis = vision_map.get(idx)
            elif unit_type == "beat":
                idx = unit.beat_index
                start = unit.start_time
                end = unit.end_time
                trans = []
                for si in unit.shot_indices:
                    trans.extend(trans_map.get(si, []))
                vis_descs = [
                    vision_map[si].description
                    for si in unit.shot_indices if si in vision_map
                ]
                vis = None  # 用 vis_descs 替代
            elif unit_type == "story_scene":
                idx = unit.story_scene_index
                start = unit.start_time
                end = unit.end_time
                trans = []
                for si in unit.shot_indices:
                    trans.extend(trans_map.get(si, []))
                vis_descs = [
                    vision_map[si].description
                    for si in unit.shot_indices if si in vision_map
                ]
                vis = None
            else:
                continue

            parts = [f"[{unit_type} {idx}] {start:.1f}s-{end:.1f}s"]

            # 描述
            if hasattr(unit, "description") and unit.description:
                parts.append(f"内容: {unit.description[:80]}")
            if hasattr(unit, "beat_type") and unit.beat_type:
                parts.append(f"类型: {unit.beat_type}")
            if hasattr(unit, "plot_function") and unit.plot_function:
                parts.append(f"功能: {unit.plot_function}")

            # 台词摘要
            if trans:
                trans_text = " ".join([t.text[:30] for t in trans[:3]])
                parts.append(f"台词: {trans_text}")

            # 画面
            if unit_type == "shot" and vis:
                parts.append(f"画面: {vis.description[:60]}")
                if vis.mood:
                    parts.append(f"情绪: {vis.mood}")
            elif unit_type in ("beat", "story_scene"):
                if vis_descs:
                    parts.append(f"画面: {'; '.join(v[:40] for v in vis_descs[:3])}")

            # 人物
            chars = []
            if hasattr(unit, "characters") and unit.characters:
                chars = unit.characters
            if chars:
                parts.append(f"人物: {','.join(chars[:5])}")

            # 关联事件
            unit_events = []
            for e in events or []:
                if start < e.end_time and end > e.start_time:
                    unit_events.append(e)
            if unit_events:
                evt_desc = "; ".join(
                    [f"{e.event_type}({e.importance})" for e in unit_events[:3]]
                )
                parts.append(f"事件: {evt_desc}")

            seg_lines.append(" | ".join(parts))

        segments_info = "\n".join(seg_lines)
        prompt = SIGNAL_PROMPT_TEMPLATE.format(segments_info=segments_info)
        logger.info(
            "EditSignal batch start: "
            f"unit_type={unit_type}, batch={batch_index}/{total_batches}, "
            f"range={_unit_range_label(unit_type, batch)}, units={len(batch)}, "
            f"prompt_chars={len(prompt)}"
        )

        produced_before = len(signals)
        parsed_count = 0
        try:
            response = client.chat(prompt=prompt, temperature=0.2)
            parsed = client.parse_json(response)
            if parsed and isinstance(parsed, list):
                parsed_count = len(parsed)
                if parsed_count < len(batch):
                    logger.warning(
                        "EditSignal batch returned fewer items: "
                        f"unit_type={unit_type}, batch={batch_index}/{total_batches}, "
                        f"expected={len(batch)}, parsed={parsed_count}"
                    )
                for i, item in enumerate(parsed):
                    if i >= len(batch):
                        break
                    unit = batch[i]

                    if unit_type == "shot":
                        u_idx = unit.scene_index
                        u_start = unit.start_time
                        u_end = unit.end_time
                    elif unit_type == "beat":
                        u_idx = unit.beat_index
                        u_start = unit.start_time
                        u_end = unit.end_time
                    elif unit_type == "story_scene":
                        u_idx = unit.story_scene_index
                        u_start = unit.start_time
                        u_end = unit.end_time
                    else:
                        continue

                    signal = EditSignal(
                        unit_type=unit_type,
                        unit_index=u_idx,
                        start_time=u_start,
                        end_time=u_end,
                        hook_score=float(item.get("hook_score", 0)),
                        plot_importance=float(item.get("plot_importance", 0)),
                        emotional_intensity=float(item.get("emotional_intensity", 0)),
                        visual_impact=float(item.get("visual_impact", 0)),
                        independence_score=float(item.get("independence_score", 0)),
                        continuity_dependency=float(item.get("continuity_dependency", 0)),
                        boundary_quality=float(item.get("boundary_quality", 0)),
                        spoiler_level=float(item.get("spoiler_level", 0)),
                        suggested_usage=item.get("suggested_usage", []),
                    )
                    signals.append(signal)
            else:
                logger.warning(f"剪辑信号解析失败 ({unit_type} batch)")
        except Exception as e:
            logger.warning(f"剪辑信号计算失败 ({unit_type} batch): {e}")
        produced_count = len(signals) - produced_before
        logger.info(
            "EditSignal batch done: "
            f"unit_type={unit_type}, batch={batch_index}/{total_batches}, "
            f"parsed={parsed_count}, produced={produced_count}, "
            f"elapsed={_elapsed(batch_started)}"
        )

        time.sleep(0.5)

    logger.info(
        "EditSignal unit compute complete: "
        f"unit_type={unit_type}, produced={len(signals)}, "
        f"elapsed={_elapsed(phase_started)}"
    )
    return signals


# ═══════════════════════════════════════════════════════════════
# NarrativeSignal 计算（v3 新增）
# ═══════════════════════════════════════════════════════════════

NARRATIVE_PROMPT = """你是一个专业的叙事结构分析师。请为以下视频片段评估叙事信号。

=== 片段列表 ===
{segments_info}

为每个片段评估：
1. arc_position: 在整体叙事弧中的位置 (0-1，开头=0，结尾=1)
2. tension_level: 张力水平 (0-1)
3. information_density: 信息密度 (0-1，新信息量/叙事推进度)
4. character_focus: 主要聚焦的角色 character_id
5. narrative_function: exposition/rising_action/climax/falling_action/resolution/transition/comic_relief
6. theme_relevance: 与主题相关度 (0-1)

输出 JSON 数组，只输出 JSON：
```json
[
  {{
    "unit_index": 0,
    "arc_position": 0.2,
    "tension_level": 0.3,
    "information_density": 0.7,
    "character_focus": "char_000",
    "narrative_function": "exposition",
    "theme_relevance": 0.6
  }}
]
```
"""


def _compute_narrative_signals(
    client, video_dir, beats, story_scenes, events, meta_duration,
) -> list[NarrativeSignal]:
    """为 beat / story_scene 计算叙事信号"""
    ns_path = video_dir / "narrative_signals.json"

    all_ns = []
    units = []
    # 将 beat 和 story_scene 合并为统一列表
    for b in (beats or []):
        units.append(("beat", b.beat_index, b.start_time, b.end_time, getattr(b, "description", ""), getattr(b, "characters", [])))
    for ss in (story_scenes or []):
        units.append(("story_scene", ss.story_scene_index, ss.start_time, ss.end_time, getattr(ss, "description", ""), getattr(ss, "characters", [])))

    if ns_path.exists():
        data = json.loads(ns_path.read_text(encoding="utf-8"))
        cached = [NarrativeSignal(**s) for s in data]
        if cached or not units:
            logger.info(f"NarrativeSignal cache loaded: count={len(cached)}, path={ns_path}")
            return cached
        logger.info(
            f"NarrativeSignal cache empty with {len(units)} available units; recomputing"
        )

    if not units:
        logger.info("NarrativeSignal skipped: no beat/story_scene units")
        return []

    batch_size = 15
    total_batches = _ceil_div(len(units), batch_size)
    phase_started = time.perf_counter()
    logger.info(
        "NarrativeSignal compute start: "
        f"units={len(units)}, batch_size={batch_size}, batches={total_batches}"
    )
    for batch_start in range(0, len(units), batch_size):
        batch = units[batch_start: batch_start + batch_size]
        batch_index = batch_start // batch_size + 1
        batch_started = time.perf_counter()
        seg_lines = []
        for utype, uidx, st, et, desc, chars in batch:
            line = f"[{utype} {uidx}] {st:.1f}s-{et:.1f}s"
            if desc:
                line += f" | {desc[:80]}"
            if chars:
                line += f" | 人物: {','.join(chars[:3])}"
            seg_lines.append(line)

        prompt = NARRATIVE_PROMPT.format(segments_info="\n".join(seg_lines))
        logger.info(
            "NarrativeSignal batch start: "
            f"batch={batch_index}/{total_batches}, units={len(batch)}, "
            f"prompt_chars={len(prompt)}"
        )
        produced_before = len(all_ns)
        parsed_count = 0
        try:
            response = client.chat(prompt=prompt, temperature=0.2)
            parsed = client.parse_json(response)
            if parsed and isinstance(parsed, list):
                parsed_count = len(parsed)
                if parsed_count < len(batch):
                    logger.warning(
                        "NarrativeSignal batch returned fewer items: "
                        f"batch={batch_index}/{total_batches}, "
                        f"expected={len(batch)}, parsed={parsed_count}"
                    )
                for i, item in enumerate(parsed):
                    if i >= len(batch):
                        break
                    utype, uidx, st, et, _, _ = batch[i]
                    ns = NarrativeSignal(
                        unit_type=utype, unit_index=uidx,
                        start_time=st, end_time=et,
                        arc_position=float(item.get("arc_position", 0)),
                        tension_level=float(item.get("tension_level", 0)),
                        information_density=float(item.get("information_density", 0)),
                        character_focus=item.get("character_focus", ""),
                        narrative_function=item.get("narrative_function", ""),
                        theme_relevance=float(item.get("theme_relevance", 0)),
                    )
                    all_ns.append(ns)
            else:
                logger.warning(
                    f"NarrativeSignal parse failed: batch={batch_index}/{total_batches}"
                )
        except Exception as e:
            logger.warning(f"叙事信号计算失败: {e}")
        produced_count = len(all_ns) - produced_before
        logger.info(
            "NarrativeSignal batch done: "
            f"batch={batch_index}/{total_batches}, parsed={parsed_count}, "
            f"produced={produced_count}, elapsed={_elapsed(batch_started)}"
        )
        time.sleep(0.5)

    ns_path.write_text(
        json.dumps([s.model_dump() for s in all_ns], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        f"NarrativeSignal compute complete: count={len(all_ns)}, "
        f"path={ns_path}, elapsed={_elapsed(phase_started)}"
    )
    return all_ns


# ═══════════════════════════════════════════════════════════════
# RecompositionSignal 计算（v3 新增）
# ═══════════════════════════════════════════════════════════════

RECOMP_PROMPT = """你是一个短视频内容二次创作专家。请为以下视频片段评估二次创作价值。

=== 片段列表 ===
{segments_info}

为每个片段评估：
1. meme_potential: 梗/传播潜力 (0-1)
2. emotional_quotability: 情感引用潜力/"名场面"程度 (0-1)
3. context_freedom: 脱离上下文仍有意义的程度 (0-1)
4. remix_flexibility: 可重新组合的灵活度 (0-1)
5. platform_fit: 平台适配（douyin/bilibili/youtube 分别 0-1）
6. suggested_formats: 建议格式 (reaction/compilation/fancam/edit)

输出 JSON 数组，只输出 JSON：
```json
[
  {{
    "unit_index": 0,
    "meme_potential": 0.3,
    "emotional_quotability": 0.8,
    "context_freedom": 0.6,
    "remix_flexibility": 0.5,
    "platform_fit": {{"douyin": 0.7, "bilibili": 0.6, "youtube": 0.5}},
    "suggested_formats": ["compilation", "edit"]
  }}
]
```
"""


def _compute_recomposition_signals(
    client, video_dir, beats, story_scenes, events,
    transcripts, vision_summaries,
) -> list[RecompositionSignal]:
    """为重要片段计算二次创作信号"""
    rs_path = video_dir / "recomposition_signals.json"

    # 只为重要 beat 计算（高情绪强度 / 非 transition 类型）
    target_beats = [
        b for b in (beats or [])
        if b.intensity >= 0.5 or b.beat_type in ("confrontation", "resolution")
    ]
    target_source = "important"
    if not target_beats:
        target_beats = list(beats or [])[:10]  # 回退：取前 10 个
        target_source = "fallback_first_10"

    if rs_path.exists():
        data = json.loads(rs_path.read_text(encoding="utf-8"))
        cached = [RecompositionSignal(**s) for s in data]
        if cached or not target_beats:
            logger.info(
                f"RecompositionSignal cache loaded: count={len(cached)}, path={rs_path}"
            )
            return cached
        logger.info(
            "RecompositionSignal cache empty with "
            f"{len(target_beats)} target beats; recomputing"
        )

    trans_map = {}
    if transcripts:
        for t in transcripts:
            trans_map.setdefault(t.scene_index, []).append(t)
    vision_map = {v.scene_index: v for v in (vision_summaries or [])}

    all_rs = []
    batch_size = 10
    total_batches = _ceil_div(len(target_beats), batch_size)
    phase_started = time.perf_counter()
    logger.info(
        "RecompositionSignal compute start: "
        f"beats={len(beats or [])}, targets={len(target_beats)}, "
        f"target_source={target_source}, batch_size={batch_size}, "
        f"batches={total_batches}"
    )
    for batch_start in range(0, len(target_beats), batch_size):
        batch = target_beats[batch_start: batch_start + batch_size]
        batch_index = batch_start // batch_size + 1
        batch_started = time.perf_counter()
        seg_lines = []
        for b in batch:
            parts = [f"[beat {b.beat_index}] {b.start_time:.1f}s-{b.end_time:.1f}s"]
            if b.description:
                parts.append(f"内容: {b.description[:60]}")
            if b.emotion:
                parts.append(f"情绪: {b.emotion}")
            # 台词摘要
            beat_trans = []
            for si in b.shot_indices:
                beat_trans.extend(trans_map.get(si, []))
            if beat_trans:
                parts.append(f"台词: {' '.join(t.text[:20] for t in beat_trans[:3])}")
            seg_lines.append(" | ".join(parts))

        prompt = RECOMP_PROMPT.format(segments_info="\n".join(seg_lines))
        logger.info(
            "RecompositionSignal batch start: "
            f"batch={batch_index}/{total_batches}, units={len(batch)}, "
            f"prompt_chars={len(prompt)}"
        )
        produced_before = len(all_rs)
        parsed_count = 0
        try:
            response = client.chat(prompt=prompt, temperature=0.3)
            parsed = client.parse_json(response)
            if parsed and isinstance(parsed, list):
                parsed_count = len(parsed)
                if parsed_count < len(batch):
                    logger.warning(
                        "RecompositionSignal batch returned fewer items: "
                        f"batch={batch_index}/{total_batches}, "
                        f"expected={len(batch)}, parsed={parsed_count}"
                    )
                for i, item in enumerate(parsed):
                    if i >= len(batch):
                        break
                    b = batch[i]
                    rs = RecompositionSignal(
                        unit_type="beat", unit_index=b.beat_index,
                        start_time=b.start_time, end_time=b.end_time,
                        meme_potential=float(item.get("meme_potential", 0)),
                        emotional_quotability=float(item.get("emotional_quotability", 0)),
                        context_freedom=float(item.get("context_freedom", 0)),
                        remix_flexibility=float(item.get("remix_flexibility", 0)),
                        platform_fit=item.get("platform_fit", {}),
                        suggested_formats=item.get("suggested_formats", []),
                    )
                    all_rs.append(rs)
            else:
                logger.warning(
                    f"RecompositionSignal parse failed: batch={batch_index}/{total_batches}"
                )
        except Exception as e:
            logger.warning(f"二次创作信号计算失败: {e}")
        produced_count = len(all_rs) - produced_before
        logger.info(
            "RecompositionSignal batch done: "
            f"batch={batch_index}/{total_batches}, parsed={parsed_count}, "
            f"produced={produced_count}, elapsed={_elapsed(batch_started)}"
        )
        time.sleep(0.5)

    rs_path.write_text(
        json.dumps([s.model_dump() for s in all_rs], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        f"RecompositionSignal compute complete: count={len(all_rs)}, "
        f"path={rs_path}, elapsed={_elapsed(phase_started)}"
    )
    return all_rs
