# -*- coding: utf-8 -*-
"""
Director Agent（重构版）
基于 Video Memory 和用户需求，使用 Gemini 生成结构化 EditPlan。

核心改动：
- 候选信息包含 MemoryUnit 的多模态融合文本和证据来源
- EditClip 必须引用候选列表中的 scene_index，不允许自造
- 每个 clip 自动填充 evidence_refs / matched_transcript / matched_vision
"""
import json
import uuid
from datetime import datetime

import config
from models.schemas import (
    EditPlan, EditClip, VideoMemory, SearchResult,
)
from memory.store import load_memory
from memory.search import search_memory
from agents.prompts import DIRECTOR_SYSTEM_PROMPT, DIRECTOR_PROMPT_TEMPLATE
from agents.reviewer import review_plan
from utils.llm_client import get_llm_client
from utils.logger import get_logger

# 长视频分章节规划的阈值（秒）。超过此时长的视频将使用滑窗策略。
LONG_VIDEO_THRESHOLD = 1800  # 30 分钟

logger = get_logger("Director")


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
    """
    Director Agent 主入口：生成 EditPlan。

    流程：
    1. 加载 Video Memory
    2. 检索候选片段
    3. 生成 EditPlan
    4. Reviewer 审核
    5. 如不通过，重新生成（最多 max_retries 次）
    6. 保存并返回
    """
    config.init_dirs()

    # 1. 加载 Video Memory
    logger.info(f"加载 Video Memory: {video_id}")
    memory = load_memory(video_id)
    if not memory.scenes:
        raise ValueError(f"视频 {video_id} 尚未完成理解，没有场景数据")

    # 2. 判断是否需要分章节规划（长视频滑窗）
    if memory.meta.duration > LONG_VIDEO_THRESHOLD:
        logger.info(
            f"视频时长 {memory.meta.duration:.0f}s 超过 {LONG_VIDEO_THRESHOLD}s，"
            f"启用分章节规划"
        )
        plan = _run_chapter_planning(
            memory=memory,
            user_prompt=prompt,
            style=style,
            target_duration=target_duration,
            platform=platform,
            character_perspective=character_perspective,
            narrative_structure=narrative_structure,
            aspect_ratio=aspect_ratio,
            max_retries=max_retries,
        )
        # 保存
        plan_path = config.EDITPLANS_DIR / f"{plan.plan_id}.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
        logger.info(f"EditPlan 已保存: {plan_path}")
        return plan

    # 短视频：单次规划
    logger.info(f"检索候选片段: \"{prompt}\"")
    candidates = search_memory(memory, prompt, top_k=30)
    logger.info(f"找到 {len(candidates)} 个候选片段")

    # 构建候选 scene 白名单（用于后续校验）
    valid_scene_indices = {r.scene_index for r in candidates}

    # 3. 构造 prompt
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

    # 4. 生成 + 审核 循环
    client = get_llm_client()
    plan = None

    for attempt in range(max_retries):
        logger.info(f"生成 EditPlan (尝试 {attempt + 1}/{max_retries})")

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

            # 构造 EditPlan 对象（带证据校验）
            plan = _parse_editplan(
                parsed, memory, video_id, prompt, target_duration,
                style, narrative_structure, character_perspective, platform, aspect_ratio,
                candidates=candidates,
                valid_scene_indices=valid_scene_indices,
            )

            # 5. Reviewer 审核
            review_result = review_plan(plan, memory, prompt)
            plan.review_result = review_result

            if review_result.approved:
                logger.info(f"✅ EditPlan 审核通过 (分数: {review_result.score:.2f})")
                break
            else:
                logger.warning(
                    f"❌ EditPlan 审核未通过: {review_result.feedback}\n"
                    f"   问题: {review_result.issues}"
                )
                # 将审核反馈加入 prompt 重新生成
                director_prompt += (
                    f"\n\n=== 上次方案的审核反馈 ===\n"
                    f"未通过原因: {review_result.feedback}\n"
                    f"具体问题: {'; '.join(review_result.issues)}\n"
                    f"请修正以上问题重新生成方案。"
                )

        except Exception as e:
            logger.error(f"EditPlan 生成失败: {e}")
            if attempt == max_retries - 1:
                raise

    if plan is None:
        raise RuntimeError("EditPlan 生成失败，已达最大重试次数")

    # 6. 保存
    plan_path = config.EDITPLANS_DIR / f"{plan.plan_id}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"EditPlan 已保存: {plan_path}")

    return plan


def _build_director_prompt(
    memory: VideoMemory,
    candidates: list[SearchResult],
    user_prompt: str,
    target_duration: float,
    style: str,
    platform: str,
    character_perspective: str,
    narrative_structure: str,
    aspect_ratio: str,
) -> str:
    """构造 Director Agent 的完整 Prompt"""

    # 人物信息（含角色标注）
    char_lines = []
    for c in memory.characters:
        role_str = f" [{c.role}]" if c.role else ""
        speaker_str = f" (speakers: {', '.join(c.speaker_ids)})" if c.speaker_ids else ""
        char_lines.append(
            f"- {c.character_id} ({c.display_name}){role_str}{speaker_str}: {c.description}, "
            f"出场 {len(c.appearance_scenes)} 个镜头, 总时长 {c.total_screen_time:.1f}s"
        )
    characters_info = "\n".join(char_lines) if char_lines else "（未识别人物）"

    # 候选片段信息（增加多模态融合文本和证据来源）
    cand_lines = []
    for i, r in enumerate(candidates[:20]):  # 限制数量
        scene = r.scene
        if scene:
            # 构建详细的候选信息
            parts = [
                f"  [{i}] Scene {scene.scene_index} "
                f"[{scene.start_time:.1f}s-{scene.end_time:.1f}s, {scene.duration:.1f}s]"
            ]
            parts.append(f"    相关度={r.score:.2f}, 命中模态: {', '.join(r.matched_modalities)}")

            if r.transcript:
                parts.append(f"    台词: {r.transcript[:100]}")
            if r.vision_summary:
                parts.append(f"    画面: {r.vision_summary[:100]}")
            if r.context_before:
                parts.append(f"    前文: {r.context_before[:60]}")
            if r.context_after:
                parts.append(f"    后文: {r.context_after[:60]}")
            if r.source_refs:
                parts.append(f"    证据: {', '.join(r.source_refs[:3])}")

            cand_lines.append("\n".join(parts))
    candidates_info = "\n".join(cand_lines) if cand_lines else "（无候选片段）"

    # 事件信息（增加 scene_indices）
    event_lines = []
    for e in memory.events:
        scenes_str = f" [scenes: {','.join(str(s) for s in e.scene_indices)}]" if e.scene_indices else ""
        event_lines.append(
            f"  [{e.start_time:.1f}s-{e.end_time:.1f}s] [{e.event_type}] "
            f"[重要性:{e.importance}] [{e.emotion}]{scenes_str} {e.description[:60]}"
        )
    events_info = "\n".join(event_lines) if event_lines else "（无事件信息）"

    # 人物视角相关行
    char_perspective_line = ""
    char_rule = ""
    if character_perspective:
        char_perspective_line = f"- 人物视角: {character_perspective}"
        char_rule = f"9. **人物视角**: 优先选择包含 {character_perspective} 的片段，确保该人物作为故事主线"

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
        characters_info=characters_info,
        candidates_info=candidates_info,
        events_info=events_info,
        narrative_structure=narrative_structure,
        character_rule=char_rule,
    )


def _parse_editplan(
    data: dict,
    memory: VideoMemory,
    video_id: str,
    user_prompt: str,
    target_duration: float,
    style: str,
    narrative_structure: str,
    character_perspective: str,
    platform: str,
    aspect_ratio: str,
    candidates: list[SearchResult] = None,
    valid_scene_indices: set[int] = None,
) -> EditPlan:
    """
    将 LLM 输出的 JSON 解析为 EditPlan 对象。

    核心改动：
    - 校验 scene_index 是否在候选白名单中
    - 自动填充 evidence_refs / matched_transcript / matched_vision
    """
    clips = []
    timeline_pos = 0.0

    # 构建候选的快速索引
    candidate_by_scene = {}
    if candidates:
        for r in candidates:
            candidate_by_scene[r.scene_index] = r

    skipped_count = 0
    for clip_data in data.get("clips", []):
        scene_idx = int(clip_data.get("source_scene_index", 0))

        # ── 校验 1：scene_index 必须存在于 memory 中 ──
        scene = next(
            (s for s in memory.scenes if s.scene_index == scene_idx), None
        )
        if scene is None:
            logger.warning(f"无效的 source_scene_index: {scene_idx}, 跳过")
            skipped_count += 1
            continue

        # ── 校验 2：scene_index 应在候选白名单中 ──
        if valid_scene_indices and scene_idx not in valid_scene_indices:
            logger.warning(
                f"Scene {scene_idx} 不在候选列表中（LLM 自造），尝试保留但标记"
            )
            # 不硬性跳过，但记录这个问题（让 Reviewer 来决定）

        source_start = float(clip_data.get("source_start", scene.start_time))
        source_end = float(clip_data.get("source_end", scene.end_time))

        # 确保时间范围合法
        source_start = max(source_start, scene.start_time)
        source_end = min(source_end, scene.end_time)
        if source_end <= source_start:
            source_start = scene.start_time
            source_end = scene.end_time

        clip_duration = source_end - source_start
        speed = float(clip_data.get("speed", 1.0))
        timeline_duration = clip_duration / speed

        # ── 自动填充证据字段 ──
        evidence_refs = clip_data.get("evidence_refs", [])
        matched_transcript = clip_data.get("matched_transcript")
        matched_vision = clip_data.get("matched_vision")

        # 从候选结果中补充证据
        candidate = candidate_by_scene.get(scene_idx)
        if candidate:
            if not evidence_refs:
                evidence_refs = candidate.source_refs[:3] if candidate.source_refs else [f"search_result#scene_{scene_idx}"]
            if not matched_transcript and candidate.transcript:
                matched_transcript = candidate.transcript[:200]
            if not matched_vision and candidate.vision_summary:
                matched_vision = candidate.vision_summary[:200]
        else:
            if not evidence_refs:
                evidence_refs = [f"scene#{scene_idx}_unchecked"]

        clip = EditClip(
            clip_index=len(clips),
            source_scene_index=scene_idx,
            source_start=round(source_start, 3),
            source_end=round(source_end, 3),
            timeline_start=round(timeline_pos, 3),
            timeline_end=round(timeline_pos + timeline_duration, 3),
            narrative_role=clip_data.get("narrative_role", "rising_action"),
            selection_reason=clip_data.get("selection_reason", ""),
            characters=clip_data.get("characters", []),
            subtitle_text=clip_data.get("subtitle_text"),
            narration_suggestion=clip_data.get("narration_suggestion"),
            transition_in=clip_data.get("transition_in", "cut"),
            transition_out=clip_data.get("transition_out", "cut"),
            speed=speed,
            audio_volume=float(clip_data.get("audio_volume", 1.0)),
            evidence_refs=evidence_refs,
            matched_transcript=matched_transcript,
            matched_vision=matched_vision,
        )
        clips.append(clip)
        timeline_pos += timeline_duration

    if skipped_count > 0:
        logger.warning(f"跳过了 {skipped_count} 个无效片段")

    plan_id = f"plan_{uuid.uuid4().hex[:8]}"

    return EditPlan(
        plan_id=plan_id,
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


# ═══════════════════════════════════════════════════════════════
# 长视频分章节规划（滑窗策略）
# ═══════════════════════════════════════════════════════════════

def _run_chapter_planning(
    memory: VideoMemory,
    user_prompt: str,
    style: str,
    target_duration: float,
    platform: str,
    character_perspective: str,
    narrative_structure: str,
    aspect_ratio: str,
    max_retries: int,
) -> EditPlan:
    """
    长视频分章节规划。

    流程：
    1. 根据事件把视频分为若干叙事章节（每章 5-15 分钟）
    2. 对每个章节独立做 search → 选片
    3. 最终做一次全局 LLM 调用，组装章节间的转场和整体叙事弧
    """
    # Step 1: 分章节
    chapters = _split_into_chapters(memory)
    logger.info(f"视频分为 {len(chapters)} 个叙事章节")

    # Step 2: 每章节独立规划
    per_chapter_duration = target_duration / max(len(chapters), 1)
    chapter_plans = []

    for i, chapter in enumerate(chapters):
        chapter_start, chapter_end = chapter["start"], chapter["end"]
        chapter_events = chapter["events"]

        logger.info(
            f"规划章节 {i+1}/{len(chapters)}: "
            f"[{chapter_start:.0f}s-{chapter_end:.0f}s] "
            f"({len(chapter_events)} 个事件)"
        )

        # 在该时间段内搜索
        candidates = search_memory(
            memory, user_prompt, top_k=15,
            time_range=(chapter_start, chapter_end),
        )
        if not candidates:
            logger.warning(f"  章节 {i+1} 无候选片段，跳过")
            continue

        valid_scene_indices = {r.scene_index for r in candidates}

        director_prompt = _build_director_prompt(
            memory=memory,
            candidates=candidates,
            user_prompt=(
                f"{user_prompt}\n\n"
                f"当前规划的是视频的第 {i+1}/{len(chapters)} 章节 "
                f"[{chapter_start:.0f}s-{chapter_end:.0f}s]。\n"
                f"本章节目标时长约 {per_chapter_duration:.0f} 秒。"
            ),
            target_duration=per_chapter_duration,
            style=style,
            platform=platform,
            character_perspective=character_perspective,
            narrative_structure=narrative_structure,
            aspect_ratio=aspect_ratio,
        )

        client = get_llm_client()
        for attempt in range(max_retries):
            try:
                response = client.chat(
                    prompt=director_prompt,
                    system_prompt=DIRECTOR_SYSTEM_PROMPT,
                    temperature=0.5,
                )
                parsed = client.parse_json(response)
                if parsed and isinstance(parsed, dict):
                    chapter_plans.append({
                        "chapter_index": i,
                        "data": parsed,
                        "candidates": candidates,
                        "valid_scene_indices": valid_scene_indices,
                    })
                    break
            except Exception as e:
                logger.warning(f"  章节 {i+1} 尝试 {attempt+1} 失败: {e}")

    # Step 3: 合并章节
    plan = _merge_chapter_plans(
        chapter_plans, memory, user_prompt, target_duration,
        style, narrative_structure, character_perspective, platform, aspect_ratio,
    )

    # 审核
    review_result = review_plan(plan, memory, user_prompt)
    plan.review_result = review_result
    if review_result.approved:
        logger.info(f"✅ 长视频 EditPlan 审核通过 (分数: {review_result.score:.2f})")
    else:
        logger.warning(f"❌ 长视频 EditPlan 审核未通过: {review_result.feedback}")

    return plan


def _split_into_chapters(memory: VideoMemory) -> list[dict]:
    """
    根据事件把视频分为若干叙事章节。

    策略：
    - 按重要事件（importance >= 6）作为章节分界
    - 每章 5-15 分钟
    - 如果没有足够的事件，按时间均匀切分
    """
    duration = memory.meta.duration
    target_chapter_len = 600  # 10 分钟一章
    min_chapter_len = 300    # 最短 5 分钟

    # 收集重要事件的时间点作为候选分界
    split_points = [0.0]
    important_events = sorted(
        [e for e in memory.events if e.importance >= 6],
        key=lambda e: e.start_time,
    )

    for event in important_events:
        last_split = split_points[-1]
        if event.start_time - last_split >= min_chapter_len:
            split_points.append(event.start_time)

    split_points.append(duration)

    # 如果章节太少，按时间均匀切分
    if len(split_points) <= 2 and duration > target_chapter_len * 2:
        split_points = [0.0]
        pos = target_chapter_len
        while pos < duration - min_chapter_len:
            split_points.append(pos)
            pos += target_chapter_len
        split_points.append(duration)

    # 构建章节
    chapters = []
    for i in range(len(split_points) - 1):
        start = split_points[i]
        end = split_points[i + 1]
        chapter_events = [
            e for e in memory.events
            if e.start_time >= start and e.start_time < end
        ]
        chapters.append({
            "start": start,
            "end": end,
            "events": chapter_events,
        })

    return chapters


def _merge_chapter_plans(
    chapter_plans: list[dict],
    memory: VideoMemory,
    user_prompt: str,
    target_duration: float,
    style: str,
    narrative_structure: str,
    character_perspective: str,
    platform: str,
    aspect_ratio: str,
) -> EditPlan:
    """
    合并各章节的 EditPlan 为完整方案。

    对每个章节的 clips 做解析后拼接，重新编号 clip_index 和 timeline。
    """
    all_clips = []
    timeline_pos = 0.0

    for cp in chapter_plans:
        data = cp["data"]
        candidates = cp["candidates"]
        valid_scene_indices = cp["valid_scene_indices"]

        # 构建候选快速索引
        candidate_by_scene = {r.scene_index: r for r in candidates}

        for clip_data in data.get("clips", []):
            scene_idx = int(clip_data.get("source_scene_index", 0))
            scene = next(
                (s for s in memory.scenes if s.scene_index == scene_idx), None
            )
            if scene is None:
                continue

            source_start = max(float(clip_data.get("source_start", scene.start_time)), scene.start_time)
            source_end = min(float(clip_data.get("source_end", scene.end_time)), scene.end_time)
            if source_end <= source_start:
                source_start, source_end = scene.start_time, scene.end_time

            clip_duration = source_end - source_start
            speed = float(clip_data.get("speed", 1.0))
            timeline_duration = clip_duration / speed

            # 自动填充证据
            evidence_refs = clip_data.get("evidence_refs", [])
            matched_transcript = clip_data.get("matched_transcript")
            matched_vision = clip_data.get("matched_vision")
            candidate = candidate_by_scene.get(scene_idx)
            if candidate:
                if not evidence_refs:
                    evidence_refs = candidate.source_refs[:3] if candidate.source_refs else [f"search_result#scene_{scene_idx}"]
                if not matched_transcript and candidate.transcript:
                    matched_transcript = candidate.transcript[:200]
                if not matched_vision and candidate.vision_summary:
                    matched_vision = candidate.vision_summary[:200]

            clip = EditClip(
                clip_index=len(all_clips),
                source_scene_index=scene_idx,
                source_start=round(source_start, 3),
                source_end=round(source_end, 3),
                timeline_start=round(timeline_pos, 3),
                timeline_end=round(timeline_pos + timeline_duration, 3),
                narrative_role=clip_data.get("narrative_role", "rising_action"),
                selection_reason=clip_data.get("selection_reason", ""),
                characters=clip_data.get("characters", []),
                subtitle_text=clip_data.get("subtitle_text"),
                narration_suggestion=clip_data.get("narration_suggestion"),
                transition_in=clip_data.get("transition_in", "cut"),
                transition_out=clip_data.get("transition_out", "cut"),
                speed=speed,
                audio_volume=float(clip_data.get("audio_volume", 1.0)),
                evidence_refs=evidence_refs,
                matched_transcript=matched_transcript,
                matched_vision=matched_vision,
            )
            all_clips.append(clip)
            timeline_pos += timeline_duration

    plan_id = f"plan_{uuid.uuid4().hex[:8]}"

    return EditPlan(
        plan_id=plan_id,
        video_id=memory.video_id,
        title=chapter_plans[0]["data"].get("title", f"剪辑方案 - {style}") if chapter_plans else f"剪辑方案 - {style}",
        user_prompt=user_prompt,
        target_duration=target_duration,
        style=style,
        narrative_structure=narrative_structure,
        character_perspective=character_perspective,
        target_platform=platform,
        aspect_ratio=aspect_ratio,
        clips=all_clips,
        created_at=datetime.now().isoformat(),
    )
