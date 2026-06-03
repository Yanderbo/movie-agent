# -*- coding: utf-8 -*-
"""
Director Agent

基于 understand 阶段产出的 Chapter / StoryScene / Beat / Signal 信息生成
beat 级 EditPlan。Shot 只作为兼容字段和证据来源，不再作为 Director 的
主要剪辑决策单元。
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import config
from agents.prompts import DIRECTOR_PROMPT_TEMPLATE, DIRECTOR_SYSTEM_PROMPT
from agents.reviewer import review_plan
from memory.search import search_memory
from memory.store import load_memory
from models.schemas import EditClip, EditPlan, EditSignal, SearchResult, VideoMemory
from utils.llm_client import get_llm_client
from utils.logger import get_logger


LONG_VIDEO_THRESHOLD = 1800
MAX_DIRECTOR_CANDIDATES = 48
SEARCH_TOP_K = 50

logger = get_logger("Director")


@dataclass
class DirectorCandidate:
    """Director 内部使用的 beat 级候选。"""

    candidate_id: str
    beat_index: int
    start_time: float
    end_time: float
    duration: float
    shot_indices: list[int]
    summary: str
    characters: list[str] = field(default_factory=list)
    story_scene_index: int | None = None
    chapter_index: int | None = None
    score: float = 0.0
    semantic_score: float = 0.0
    edit_signal: EditSignal | None = None
    narrative_signal: Any = None
    recomposition_signal: Any = None
    transcript: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    score_reasons: list[str] = field(default_factory=list)


def run_director(
    video_id: str,
    prompt: str,
    style: str = "emotional",
    target_duration: float = 180,
    platform: str = "general",
    character_perspective: str = None,
    narrative_structure: str = "chronological",
    aspect_ratio: str = "16:9",
    max_retries: int = 3,
) -> EditPlan:
    """生成 beat 级 EditPlan。"""
    config.init_dirs()

    logger.info(f"加载 Video Memory: {video_id}")
    memory = load_memory(video_id)
    if not memory.scenes:
        raise ValueError(f"视频 {video_id} 尚未完成理解，没有场景数据")
    if not memory.beats:
        raise ValueError(f"视频 {video_id} 尚未完成 beat 检测，无法生成 beat 级剪辑方案")

    if memory.meta.duration > LONG_VIDEO_THRESHOLD:
        logger.info(
            f"视频时长 {memory.meta.duration:.0f}s 超过 {LONG_VIDEO_THRESHOLD}s，"
            "使用章节感知的 beat 级规划"
        )

    logger.info(f"检索并构造 beat 候选: \"{prompt}\"")
    search_results = search_memory(memory, prompt, top_k=SEARCH_TOP_K)
    candidates = _build_beat_candidates(
        memory=memory,
        search_results=search_results,
        user_prompt=prompt,
        style=style,
        platform=platform,
    )
    if not candidates:
        raise RuntimeError("未能构造任何 beat 级候选，无法生成 EditPlan")
    logger.info(f"Director beat 候选数: {len(candidates)}")

    director_prompt = _build_director_prompt(
        memory=memory,
        candidates=candidates,
        user_prompt=prompt,
        target_duration=target_duration,
        style=style,
        platform=platform,
        character_perspective=character_perspective,
        narrative_structure=narrative_structure,
        aspect_ratio=aspect_ratio,
    )

    client = get_llm_client()
    plan = None
    for attempt in range(max_retries):
        logger.info(f"生成 beat 级 EditPlan (尝试 {attempt + 1}/{max_retries})")
        try:
            response = client.chat(
                prompt=director_prompt,
                system_prompt=DIRECTOR_SYSTEM_PROMPT,
                temperature=0.5,
            )
            parsed = client.parse_json(response)
            if not parsed or not isinstance(parsed, dict):
                logger.warning("EditPlan 解析失败，重试")
                continue

            plan = _parse_editplan(
                data=parsed,
                video_id=video_id,
                user_prompt=prompt,
                target_duration=target_duration,
                style=style,
                narrative_structure=narrative_structure,
                character_perspective=character_perspective,
                platform=platform,
                aspect_ratio=aspect_ratio,
                candidates=candidates,
            )
            review_result = review_plan(plan, memory, prompt)
            plan.review_result = review_result

            if review_result.approved:
                logger.info(f"✅ EditPlan 审核通过 (分数: {review_result.score:.2f})")
                break

            logger.warning(
                f"❌ EditPlan 审核未通过: {review_result.feedback}\n"
                f"   问题: {review_result.issues}"
            )
            director_prompt += (
                "\n\n=== 上次方案的审核反馈 ===\n"
                f"未通过原因: {review_result.feedback}\n"
                f"具体问题: {'; '.join(review_result.issues)}\n"
                "请只从候选 candidate_id 中重新选择，并修正上述问题。"
            )
        except Exception as e:
            logger.error(f"EditPlan 生成失败: {e}")
            if attempt == max_retries - 1:
                raise

    if plan is None:
        raise RuntimeError("EditPlan 生成失败，已达最大重试次数")

    plan_path = config.EDITPLANS_DIR / f"{plan.plan_id}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"EditPlan 已保存: {plan_path}")
    return plan


def _build_director_prompt(
    memory: VideoMemory,
    candidates: list[DirectorCandidate],
    user_prompt: str,
    target_duration: float,
    style: str,
    platform: str,
    character_perspective: str,
    narrative_structure: str,
    aspect_ratio: str,
) -> str:
    char_perspective_line = ""
    char_rule = ""
    if character_perspective:
        char_perspective_line = f"- 人物视角: {character_perspective}"
        char_rule = f"9. **人物视角**: 优先选择包含 {character_perspective} 的 beat，确保该人物作为故事主线"

    return DIRECTOR_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt,
        target_duration=target_duration,
        style=style,
        platform=platform,
        aspect_ratio=aspect_ratio,
        character_perspective_line=char_perspective_line,
        video_duration=memory.meta.duration,
        width=memory.meta.width,
        height=memory.meta.height,
        characters_count=len(memory.characters),
        characters_info=_format_characters(memory),
        story_map_info=_format_story_map(memory),
        candidates_info=_format_candidates(candidates),
        events_info=_format_events(memory),
        narrative_structure=narrative_structure,
        character_rule=char_rule,
    )


def _build_beat_candidates(
    memory: VideoMemory,
    search_results: list[SearchResult],
    user_prompt: str,
    style: str,
    platform: str,
) -> list[DirectorCandidate]:
    beat_map = {b.beat_index: b for b in memory.beats}
    beat_units = {u.beat_index: u for u in memory.beat_memory_units}
    story_by_beat = _story_scene_by_beat(memory)
    chapter_by_story = _chapter_by_story_scene(memory)
    narrative_by_beat = {
        s.unit_index: s for s in memory.narrative_signals
        if s.unit_type == "beat"
    }
    edit_signal_by_beat = {
        s.unit_index: s for s in memory.edit_signals
        if s.unit_type == "beat"
    }
    recomposition_by_beat = {
        s.unit_index: s for s in memory.recomposition_signals
        if s.unit_type == "beat"
    }
    event_importance_by_beat = _event_importance_by_beat(memory)
    search_by_beat = _search_scores_by_beat(memory, search_results)
    intent = _infer_intent(user_prompt, style, platform)

    candidates = []
    for beat_index, beat in beat_map.items():
        unit = beat_units.get(beat_index)
        story_scene_index = story_by_beat.get(beat_index)
        chapter_index = chapter_by_story.get(story_scene_index)
        semantic = search_by_beat.get(beat_index, {}).get("score", 0.0)
        search_refs = search_by_beat.get(beat_index, {}).get("refs", [])
        narrative = narrative_by_beat.get(beat_index)
        recomposition = recomposition_by_beat.get(beat_index)
        edit_signal = (
            unit.edit_signal
            if unit and unit.edit_signal
            else edit_signal_by_beat.get(beat_index)
        )

        candidate = DirectorCandidate(
            candidate_id=f"beat:{beat_index}",
            beat_index=beat_index,
            start_time=beat.start_time,
            end_time=beat.end_time,
            duration=beat.duration or max(0.0, beat.end_time - beat.start_time),
            shot_indices=list(beat.shot_indices),
            summary=(unit.description if unit else beat.description) or beat.description,
            characters=list(unit.characters if unit else beat.characters),
            story_scene_index=story_scene_index,
            chapter_index=chapter_index,
            semantic_score=semantic,
            edit_signal=edit_signal,
            narrative_signal=narrative,
            recomposition_signal=recomposition,
            transcript=(unit.transcript_summary if unit else "")[:240],
        )
        candidate.score, candidate.score_reasons = _score_candidate(
            candidate=candidate,
            intent=intent,
            event_importance=event_importance_by_beat.get(beat_index, 0),
        )
        candidate.evidence_refs = _candidate_evidence_refs(candidate, search_refs)
        candidates.append(candidate)

    return _limit_candidates(candidates, MAX_DIRECTOR_CANDIDATES)


def _score_candidate(
    candidate: DirectorCandidate,
    intent: dict[str, Any],
    event_importance: int,
) -> tuple[float, list[str]]:
    sig = candidate.edit_signal
    ns = candidate.narrative_signal
    rs = candidate.recomposition_signal

    hook = getattr(sig, "hook_score", 0.0) if sig else 0.0
    plot = getattr(sig, "plot_importance", 0.0) if sig else 0.0
    emotion = getattr(sig, "emotional_intensity", 0.0) if sig else 0.0
    visual = getattr(sig, "visual_impact", 0.0) if sig else 0.0
    boundary = getattr(sig, "boundary_quality", 0.0) if sig else 0.0
    spoiler = getattr(sig, "spoiler_level", 0.0) if sig else 0.0
    tension = getattr(ns, "tension_level", 0.0) if ns else 0.0
    info_density = getattr(ns, "information_density", 0.0) if ns else 0.0
    recomposition = 0.0
    if rs:
        recomposition = max(
            getattr(rs, "meme_potential", 0.0),
            getattr(rs, "emotional_quotability", 0.0),
            getattr(rs, "context_freedom", 0.0),
        )

    task = intent["task_type"]
    if task == "trailer":
        edit_fit = 0.35 * hook + 0.25 * visual + 0.2 * tension + 0.2 * boundary
        spoiler_penalty = 0.25 * spoiler
    elif task == "recap":
        edit_fit = 0.35 * plot + 0.25 * info_density + 0.2 * boundary + 0.2 * emotion
        spoiler_penalty = 0.0
    elif task == "remix":
        edit_fit = 0.35 * recomposition + 0.25 * hook + 0.2 * emotion + 0.2 * boundary
        spoiler_penalty = 0.05 * spoiler
    else:
        edit_fit = 0.25 * hook + 0.25 * plot + 0.25 * emotion + 0.15 * visual + 0.1 * boundary
        spoiler_penalty = 0.1 * spoiler

    event_boost = min(event_importance / 10, 1.0) * 0.08
    duration_penalty = 0.08 if candidate.duration > 45 else 0.0
    score = (
        0.3 * min(candidate.semantic_score, 1.0)
        + 0.45 * edit_fit
        + 0.12 * recomposition
        + event_boost
        - spoiler_penalty
        - duration_penalty
    )
    reasons = []
    if candidate.semantic_score:
        reasons.append("semantic")
    if hook >= 0.8:
        reasons.append("hook")
    if emotion >= 0.8:
        reasons.append("emotion")
    if plot >= 0.8:
        reasons.append("plot")
    if recomposition >= 0.8:
        reasons.append("recomposition")
    if event_importance >= 7:
        reasons.append("event")
    return round(max(score, 0.0), 3), reasons or ["signal"]


def _parse_editplan(
    data: dict,
    video_id: str,
    user_prompt: str,
    target_duration: float,
    style: str,
    narrative_structure: str,
    character_perspective: str,
    platform: str,
    aspect_ratio: str,
    candidates: list[DirectorCandidate],
) -> EditPlan:
    candidate_by_id = {c.candidate_id: c for c in candidates}
    clips = []
    timeline_pos = 0.0
    skipped = 0

    items = data.get("plan_items") or data.get("clips") or []
    if not isinstance(items, list):
        logger.warning("EditPlan 输出中的 plan_items/clips 不是列表，忽略")
        items = []
    for item in items:
        if not isinstance(item, dict):
            logger.warning("EditPlan 输出中存在非对象片段，跳过")
            skipped += 1
            continue
        candidate_id = str(item.get("candidate_id") or "")
        if not candidate_id and item.get("source_beat_index") is not None:
            candidate_id = f"beat:{item.get('source_beat_index')}"

        candidate = candidate_by_id.get(candidate_id)
        if not candidate:
            logger.warning(f"未知 candidate_id: {candidate_id or '<missing>'}, 跳过")
            skipped += 1
            continue
        if not candidate.shot_indices:
            logger.warning(f"候选 {candidate_id} 缺少 shot_indices，跳过")
            skipped += 1
            continue

        speed = _clamp_speed(item.get("speed", 1.0))
        source_start = round(candidate.start_time, 3)
        source_end = round(candidate.end_time, 3)
        timeline_duration = (source_end - source_start) / speed
        evidence_refs = _dedupe(
            _as_list(item.get("evidence_refs")) + candidate.evidence_refs
        )

        clip = EditClip(
            clip_index=len(clips),
            source_unit_type="beat",
            source_scene_index=candidate.shot_indices[0],
            source_start=source_start,
            source_end=source_end,
            timeline_start=round(timeline_pos, 3),
            timeline_end=round(timeline_pos + timeline_duration, 3),
            narrative_role=item.get("narrative_role", "rising_action"),
            selection_reason=item.get("selection_reason", ""),
            characters=_as_list(item.get("characters")) or candidate.characters,
            subtitle_text=item.get("subtitle_text"),
            narration_suggestion=item.get("narration_suggestion"),
            transition_in=item.get("transition_in", "cut"),
            transition_out=item.get("transition_out", "cut"),
            speed=speed,
            audio_volume=_clamp_float(item.get("audio_volume", 1.0), 0.0, 5.0, 1.0),
            evidence_refs=evidence_refs,
            matched_transcript=item.get("matched_transcript") or candidate.transcript or None,
            matched_vision=item.get("matched_vision") or candidate.summary or None,
            edit_signal_ref=candidate.edit_signal,
            source_beat_index=candidate.beat_index,
            source_story_scene_index=candidate.story_scene_index,
        )
        clips.append(clip)
        timeline_pos += timeline_duration

    if skipped:
        logger.warning(f"跳过了 {skipped} 个无效候选")

    return EditPlan(
        plan_id=f"plan_{uuid.uuid4().hex[:8]}",
        video_id=video_id,
        title=data.get("title", f"剪辑方案 - {style}"),
        user_prompt=user_prompt,
        target_duration=target_duration,
        style=style,
        narrative_structure=data.get("narrative_structure", narrative_structure),
        character_perspective=character_perspective,
        target_platform=platform,
        aspect_ratio=aspect_ratio,
        clips=clips,
        created_at=datetime.now().isoformat(),
    )


def _format_story_map(memory: VideoMemory) -> str:
    if not memory.chapters:
        return "（无章节信息）"
    lines = []
    for ch in memory.chapters[:20]:
        title = f"《{ch.title}》" if ch.title else ""
        lines.append(
            f"- Chapter {ch.chapter_index} {title}"
            f"[{ch.start_time:.1f}-{ch.end_time:.1f}s] "
            f"{ch.chapter_type} / {ch.theme}: {ch.description[:90]}"
        )
    return "\n".join(lines)


def _format_characters(memory: VideoMemory) -> str:
    chars = sorted(
        memory.characters,
        key=lambda c: getattr(c, "total_screen_time", 0.0),
        reverse=True,
    )
    lines = []
    for c in chars[:16]:
        role = f" [{c.role}]" if c.role else ""
        lines.append(
            f"- {c.character_id} ({c.display_name}){role}: "
            f"出场 {len(c.appearance_scenes)} 镜头, {c.description[:80]}"
        )
    return "\n".join(lines) if lines else "（未识别人物）"


def _format_candidates(candidates: list[DirectorCandidate]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        sig = c.edit_signal
        ns = c.narrative_signal
        rs = c.recomposition_signal
        signal_text = ""
        if sig:
            signal_text = (
                f"hook={sig.hook_score:.2f}, plot={sig.plot_importance:.2f}, "
                f"emotion={sig.emotional_intensity:.2f}, boundary={sig.boundary_quality:.2f}, "
                f"spoiler={sig.spoiler_level:.2f}"
            )
        narrative_text = ""
        if ns:
            narrative_text = (
                f"arc={ns.arc_position:.2f}, tension={ns.tension_level:.2f}, "
                f"function={ns.narrative_function}"
            )
        recomposition_text = ""
        if rs:
            formats = ",".join(rs.suggested_formats[:3])
            recomposition_text = (
                f"meme={rs.meme_potential:.2f}, quote={rs.emotional_quotability:.2f}, "
                f"context={rs.context_freedom:.2f}, formats={formats}"
            )

        lines.append(
            "\n".join([
                f"[{i}] candidate_id={c.candidate_id} score={c.score:.2f} "
                f"time=[{c.start_time:.1f}-{c.end_time:.1f}s] duration={c.duration:.1f}s",
                f"    chapter={c.chapter_index}, story_scene={c.story_scene_index}, "
                f"shots={','.join(str(s) for s in c.shot_indices[:8])}",
                f"    summary: {c.summary[:120]}",
                f"    characters: {', '.join(c.characters[:8]) or 'none'}",
                f"    signals: {signal_text or 'none'}",
                f"    narrative: {narrative_text or 'none'}",
                f"    recomposition: {recomposition_text or 'none'}",
                f"    evidence: {', '.join(c.evidence_refs[:5])}",
            ])
        )
    return "\n".join(lines)


def _format_events(memory: VideoMemory) -> str:
    events = sorted(memory.events, key=lambda e: (-e.importance, e.start_time))[:20]
    lines = []
    for e in events:
        scenes = ",".join(str(s) for s in e.scene_indices[:8])
        lines.append(
            f"- event:{e.event_index} [{e.start_time:.1f}-{e.end_time:.1f}s] "
            f"{e.event_type} importance={e.importance} emotion={e.emotion} "
            f"scenes={scenes}: {e.description[:90]}"
        )
    return "\n".join(lines) if lines else "（无事件信息）"


def _story_scene_by_beat(memory: VideoMemory) -> dict[int, int]:
    mapping = {}
    for ss in memory.story_scenes:
        for beat_index in ss.beat_indices:
            mapping[beat_index] = ss.story_scene_index
    return mapping


def _chapter_by_story_scene(memory: VideoMemory) -> dict[int, int]:
    mapping = {}
    for ch in memory.chapters:
        for story_scene_index in ch.story_scene_indices:
            mapping[story_scene_index] = ch.chapter_index
    return mapping


def _event_importance_by_beat(memory: VideoMemory) -> dict[int, int]:
    scene_by_index = {s.scene_index: s for s in memory.scenes}
    scores = defaultdict(int)
    for event in memory.events:
        beat_indices = event.beat_indices or []
        if not beat_indices and event.scene_indices:
            for scene_index in event.scene_indices:
                scene = scene_by_index.get(scene_index)
                if scene and scene.beat_index is not None:
                    beat_indices.append(scene.beat_index)
        for beat_index in beat_indices:
            scores[beat_index] = max(scores[beat_index], event.importance)
    return dict(scores)


def _search_scores_by_beat(
    memory: VideoMemory,
    search_results: list[SearchResult],
) -> dict[int, dict[str, Any]]:
    by_scene = {s.scene_index: s for s in memory.scenes}
    result = {}
    for r in search_results:
        beat_index = r.beat_index
        if beat_index is None and r.memory_unit:
            beat_index = r.memory_unit.beat_index
        if beat_index is None:
            scene = by_scene.get(r.scene_index)
            beat_index = scene.beat_index if scene else None
        if beat_index is None:
            continue
        entry = result.setdefault(beat_index, {"score": 0.0, "refs": []})
        entry["score"] = max(entry["score"], r.score)
        entry["refs"].extend(r.source_refs)
    for entry in result.values():
        entry["refs"] = _dedupe(entry["refs"])
    return result


def _candidate_evidence_refs(
    candidate: DirectorCandidate,
    search_refs: list[str],
) -> list[str]:
    refs = [
        candidate.candidate_id,
        f"beat:{candidate.beat_index}",
        f"edit_signal:beat:{candidate.beat_index}",
    ]
    if candidate.story_scene_index is not None:
        refs.append(f"story_scene:{candidate.story_scene_index}")
    if candidate.chapter_index is not None:
        refs.append(f"chapter:{candidate.chapter_index}")
    if candidate.narrative_signal:
        refs.append(f"narrative_signal:beat:{candidate.beat_index}")
    if candidate.recomposition_signal:
        refs.append(f"recomposition_signal:beat:{candidate.beat_index}")
    refs.extend(search_refs[:3])
    return _dedupe(refs)


def _limit_candidates(
    candidates: list[DirectorCandidate],
    limit: int,
) -> list[DirectorCandidate]:
    ordered = sorted(candidates, key=lambda c: (-c.score, c.start_time))
    selected = []
    per_chapter = defaultdict(int)
    for c in ordered:
        if len(selected) >= limit:
            break
        if c.score <= 0 and len(selected) >= limit // 2:
            continue
        chapter_key = c.chapter_index if c.chapter_index is not None else -1
        if per_chapter[chapter_key] >= 5 and len(selected) >= limit // 2:
            continue
        selected.append(c)
        per_chapter[chapter_key] += 1

    if len(selected) < limit:
        selected_ids = {c.candidate_id for c in selected}
        for c in ordered:
            if c.candidate_id not in selected_ids:
                selected.append(c)
                selected_ids.add(c.candidate_id)
            if len(selected) >= limit:
                break
    return selected


def _infer_intent(user_prompt: str, style: str, platform: str) -> dict[str, Any]:
    text = f"{user_prompt} {style} {platform}".lower()
    if any(key in text for key in ("预告", "trailer", "teaser")):
        task_type = "trailer"
    elif any(key in text for key in ("解说", "复盘", "梗概", "recap")):
        task_type = "recap"
    elif any(key in text for key in ("二创", "名场面", "reaction", "remix", "梗")):
        task_type = "remix"
    else:
        task_type = "highlight"
    return {"task_type": task_type}


def _clamp_speed(value: Any) -> float:
    return _clamp_float(value, 0.25, 4.0, 1.0)


def _clamp_float(value: Any, minimum: float, maximum: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return min(max(number, minimum), maximum)


def _as_list(value: Any) -> list:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [str(value)]


def _dedupe(values: list[Any]) -> list:
    seen = set()
    result = []
    for value in values:
        if value in (None, ""):
            continue
        key = str(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result
