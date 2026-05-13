# -*- coding: utf-8 -*-
"""
Reviewer Agent（重构版）
使用 Gemini 审核 EditPlan 质量，同时进行规则校验。

核心改动：
- 新增 Grounding 校验：evidence_refs 非空、时间精度、角色一致性
- 新增事件覆盖率检查：高重要性事件是否被 EditPlan 覆盖
"""
import json

from models.schemas import EditPlan, ReviewResult, VideoMemory
from agents.prompts import REVIEWER_SYSTEM_PROMPT, REVIEWER_PROMPT_TEMPLATE
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("Reviewer")


def review_plan(
    plan: EditPlan,
    memory: VideoMemory,
    user_prompt: str,
) -> ReviewResult:
    """
    审核 EditPlan 质量。
    结合规则校验、Grounding 校验和 LLM 审核。

    Args:
        plan: 待审核的 EditPlan
        memory: Video Memory
        user_prompt: 用户原始需求

    Returns:
        ReviewResult
    """
    issues = []

    # ═══ 规则校验 ═══

    # 1. 片段数量
    if len(plan.clips) < 3:
        issues.append(f"片段数量过少: {len(plan.clips)} (最少3个)")
    if len(plan.clips) > 20:
        issues.append(f"片段数量过多: {len(plan.clips)} (最多20个)")

    # 2. 时长偏差
    actual_duration = sum(
        (c.timeline_end - c.timeline_start) for c in plan.clips
    )
    if plan.target_duration > 0:
        deviation = abs(actual_duration - plan.target_duration) / plan.target_duration
        if deviation > 0.15:
            issues.append(
                f"时长偏差过大: 实际 {actual_duration:.1f}s vs 目标 {plan.target_duration:.1f}s "
                f"(偏差 {deviation:.0%})"
            )

    # 3. 场景引用合法性
    max_scene = max((s.scene_index for s in memory.scenes), default=-1)
    for clip in plan.clips:
        if clip.source_scene_index < 0 or clip.source_scene_index > max_scene:
            issues.append(
                f"片段 {clip.clip_index} 引用了无效场景: {clip.source_scene_index}"
            )
        else:
            # 检查时间范围
            scene = next(
                (s for s in memory.scenes if s.scene_index == clip.source_scene_index),
                None,
            )
            if scene:
                if clip.source_start < scene.start_time - 0.5:
                    issues.append(
                        f"片段 {clip.clip_index} source_start ({clip.source_start:.1f}) "
                        f"早于场景起始 ({scene.start_time:.1f})"
                    )
                if clip.source_end > scene.end_time + 0.5:
                    issues.append(
                        f"片段 {clip.clip_index} source_end ({clip.source_end:.1f}) "
                        f"晚于场景结束 ({scene.end_time:.1f})"
                    )

    # 4. 叙事角色检查
    roles = [c.narrative_role for c in plan.clips]
    if "hook" not in roles and len(plan.clips) >= 3:
        issues.append("缺少 hook（开头吸引）片段")

    # 5. 连续性检查
    for i in range(len(plan.clips) - 2):
        if (plan.clips[i].narrative_role == plan.clips[i+1].narrative_role ==
                plan.clips[i+2].narrative_role):
            issues.append(
                f"片段 {i}-{i+2} 连续3个相同叙事角色: {plan.clips[i].narrative_role}"
            )

    # ═══ Grounding 校验（新增）═══
    grounding_issues = _grounding_check(plan, memory)
    issues.extend(grounding_issues)

    # ═══ 如果规则校验有严重问题，直接返回不通过 ═══
    critical_issues = [
        i for i in issues
        if "无效场景" in i or "片段数量过少" in i
    ]
    if critical_issues:
        return ReviewResult(
            approved=False,
            score=0.2,
            feedback="存在严重的结构性问题",
            issues=issues,
        )

    # ═══ LLM 审核 ═══
    try:
        llm_review = _llm_review(plan, memory, user_prompt)
        if llm_review:
            # 合并 LLM 发现的问题
            issues.extend(llm_review.get("issues", []))
            llm_approved = llm_review.get("approved", True)
            llm_score = float(llm_review.get("score", 0.7))
            llm_feedback = llm_review.get("feedback", "")

            # 综合判断
            approved = llm_approved and len(critical_issues) == 0
            return ReviewResult(
                approved=approved,
                score=llm_score,
                feedback=llm_feedback,
                issues=issues,
            )
    except Exception as e:
        logger.warning(f"LLM 审核失败，仅使用规则审核: {e}")

    # 仅规则审核的结果
    score = max(0, 1.0 - len(issues) * 0.15)
    approved = len(issues) <= 2 and score >= 0.6
    return ReviewResult(
        approved=approved,
        score=round(score, 2),
        feedback=f"规则审核: {len(issues)} 个问题",
        issues=issues,
    )


def _grounding_check(plan: EditPlan, memory: VideoMemory) -> list[str]:
    """
    Grounding 校验：验证 EditPlan 的每个 clip 是否有可追溯的证据支撑。

    检查项：
    1. evidence_refs 非空
    2. source_start/end 与 MemoryUnit 的 time_range 偏差不超过 ±0.5s
    3. clip 中声称的 characters 必须在对应 scene 的 MemoryUnit.characters 中
    4. 高重要性事件（importance ≥ 7）是否被 EditPlan 覆盖
    """
    issues = []

    # 构建 MemoryUnit 快速索引
    mu_by_scene = {}
    for mu in memory.memory_units:
        mu_by_scene[mu.scene_index] = mu

    # ── 检查 1: evidence_refs 非空 ──
    no_evidence_count = 0
    for clip in plan.clips:
        if not clip.evidence_refs:
            no_evidence_count += 1
        elif all("unchecked" in ref for ref in clip.evidence_refs):
            no_evidence_count += 1

    if no_evidence_count > 0:
        issues.append(
            f"{no_evidence_count}/{len(plan.clips)} 个片段缺少有效的 evidence_refs"
        )

    # ── 检查 2: 时间精度（与 MemoryUnit 的 time_range 对比）──
    for clip in plan.clips:
        mu = mu_by_scene.get(clip.source_scene_index)
        if not mu:
            continue

        # source_start 不应早于 MemoryUnit 的 start_time 超过 0.5s
        if clip.source_start < mu.start_time - 0.5:
            issues.append(
                f"[Grounding] 片段 {clip.clip_index}: source_start ({clip.source_start:.1f}s) "
                f"早于 MemoryUnit 起始 ({mu.start_time:.1f}s) 超过 0.5s"
            )

        # source_end 不应晚于 MemoryUnit 的 end_time 超过 0.5s
        if clip.source_end > mu.end_time + 0.5:
            issues.append(
                f"[Grounding] 片段 {clip.clip_index}: source_end ({clip.source_end:.1f}s) "
                f"晚于 MemoryUnit 结束 ({mu.end_time:.1f}s) 超过 0.5s"
            )

    # ── 检查 3: 角色一致性 ──
    for clip in plan.clips:
        if not clip.characters:
            continue
        mu = mu_by_scene.get(clip.source_scene_index)
        if not mu or not mu.characters:
            continue
        for char_id in clip.characters:
            if char_id not in mu.characters:
                issues.append(
                    f"[Grounding] 片段 {clip.clip_index}: 声称包含人物 {char_id}，"
                    f"但 MemoryUnit scene_{clip.source_scene_index} 中未出现该人物"
                )

    # ── 检查 4: 高重要性事件覆盖率 ──
    important_events = [e for e in memory.events if e.importance >= 7]
    if important_events:
        covered_scenes = {clip.source_scene_index for clip in plan.clips}
        uncovered_events = []
        for event in important_events:
            # 检查事件的 scene_indices 是否有任何一个被 EditPlan 覆盖
            event_scenes = set(event.scene_indices) if event.scene_indices else set()
            if not event_scenes:
                # 回退：用时间范围找 scene
                for s in memory.scenes:
                    if event.start_time < s.end_time and event.end_time > s.start_time:
                        event_scenes.add(s.scene_index)
            if not event_scenes & covered_scenes:
                uncovered_events.append(event)

        if uncovered_events:
            uncovered_descs = [f"{e.description[:30]}(重要性:{e.importance})" for e in uncovered_events[:3]]
            issues.append(
                f"[Grounding] {len(uncovered_events)} 个高重要性事件未被 EditPlan 覆盖: "
                f"{'; '.join(uncovered_descs)}"
            )

    return issues


def _llm_review(plan: EditPlan, memory: VideoMemory, user_prompt: str) -> dict | None:
    """使用 LLM 进行审核"""
    client = get_llm_client()

    # 简化 EditPlan 用于审核（包含证据信息）
    plan_summary = {
        "title": plan.title,
        "target_duration": plan.target_duration,
        "style": plan.style,
        "clips": [
            {
                "clip_index": c.clip_index,
                "source_scene_index": c.source_scene_index,
                "source_start": c.source_start,
                "source_end": c.source_end,
                "duration": round(c.source_end - c.source_start, 1),
                "narrative_role": c.narrative_role,
                "selection_reason": c.selection_reason[:50],
                "audio_volume": c.audio_volume,
                "evidence_refs": c.evidence_refs[:3],
                "has_transcript": bool(c.matched_transcript),
                "has_vision": bool(c.matched_vision),
            }
            for c in plan.clips
        ],
        "actual_total_duration": round(
            sum(c.timeline_end - c.timeline_start for c in plan.clips), 1
        ),
    }

    prompt = REVIEWER_PROMPT_TEMPLATE.format(
        user_prompt=user_prompt,
        target_duration=plan.target_duration,
        style=plan.style,
        editplan_json=json.dumps(plan_summary, indent=2, ensure_ascii=False),
        total_scenes=len(memory.scenes),
        video_duration=memory.meta.duration,
        max_scene_index=max((s.scene_index for s in memory.scenes), default=0),
    )

    response = client.chat(
        prompt=prompt,
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        temperature=0.2,
    )
    return client.parse_json(response)
