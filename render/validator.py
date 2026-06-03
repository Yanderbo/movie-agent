# -*- coding: utf-8 -*-
"""
EditPlan 校验器
在渲染前验证 EditPlan 的合法性。
"""
from pathlib import Path

from models.schemas import EditPlan, VideoMemory
from utils.logger import get_logger

logger = get_logger("Validator")

# 允许的转场类型
VALID_TRANSITIONS = {"cut", "fade_in", "fade_out", "fade", "dissolve", ""}


def validate_plan(plan: EditPlan, memory: VideoMemory) -> list[str]:
    """
    校验 EditPlan 合法性。

    Returns:
        错误列表。空列表表示校验通过。
    """
    errors = []

    # 1. 基本字段
    if not plan.clips:
        errors.append("EditPlan 没有任何片段")
        return errors

    if not plan.video_id:
        errors.append("缺少 video_id")

    # 2. 源视频存在
    source_path = memory.meta.storage_path
    if not Path(source_path).exists():
        errors.append(f"源视频不存在: {source_path}")

    video_duration = memory.meta.duration

    beat_by_index = {b.beat_index: b for b in memory.beats}

    # 3. 片段校验
    max_scene = max((s.scene_index for s in memory.scenes), default=-1)
    seen_indices = set()

    for clip in plan.clips:
        source_unit_type = getattr(clip, "source_unit_type", "shot") or "shot"

        # clip_index 唯一性
        if clip.clip_index in seen_indices:
            errors.append(f"片段 clip_index={clip.clip_index} 重复")
        seen_indices.add(clip.clip_index)

        # scene_index 范围
        if clip.source_scene_index < 0 or clip.source_scene_index > max_scene:
            errors.append(
                f"片段 {clip.clip_index}: source_scene_index={clip.source_scene_index} "
                f"超出范围 [0, {max_scene}]"
            )
            continue

        # 找到兼容 scene。beat 级片段中它代表 beat 的首个 shot。
        scene = next(
            (s for s in memory.scenes if s.scene_index == clip.source_scene_index),
            None,
        )
        if not scene:
            errors.append(f"片段 {clip.clip_index}: 找不到 scene {clip.source_scene_index}")
            continue

        # source_start >= 0
        if clip.source_start < 0:
            errors.append(f"片段 {clip.clip_index}: source_start ({clip.source_start}) < 0")

        # 时间范围
        if clip.source_start >= clip.source_end:
            errors.append(
                f"片段 {clip.clip_index}: source_start ({clip.source_start}) >= "
                f"source_end ({clip.source_end})"
            )

        # 不超出原视频 duration
        if clip.source_end > video_duration + 0.5:
            errors.append(
                f"片段 {clip.clip_index}: source_end ({clip.source_end:.1f}) "
                f"超出视频时长 ({video_duration:.1f})"
            )

        # timeline 合法性
        if clip.timeline_end <= clip.timeline_start:
            errors.append(
                f"片段 {clip.clip_index}: timeline_end ({clip.timeline_end}) "
                f"<= timeline_start ({clip.timeline_start})"
            )

        # timeline 时长与 source 时长 / speed 一致性
        source_dur = clip.source_end - clip.source_start
        timeline_dur = clip.timeline_end - clip.timeline_start
        expected_timeline = source_dur / clip.speed if clip.speed > 0 else 0
        if expected_timeline > 0 and abs(timeline_dur - expected_timeline) > 1.0:
            errors.append(
                f"片段 {clip.clip_index}: timeline 时长 ({timeline_dur:.1f}s) "
                f"与 source/speed 不一致 (预期 {expected_timeline:.1f}s)"
            )

        if source_unit_type == "beat":
            beat_index = clip.source_beat_index
            if beat_index is None:
                errors.append(f"片段 {clip.clip_index}: beat 级片段缺少 source_beat_index")
            else:
                beat = beat_by_index.get(beat_index)
                if not beat:
                    errors.append(f"片段 {clip.clip_index}: 找不到 beat {beat_index}")
                else:
                    if clip.source_scene_index not in beat.shot_indices:
                        errors.append(
                            f"片段 {clip.clip_index}: source_scene_index={clip.source_scene_index} "
                            f"不属于 beat {beat_index}"
                        )
                    if clip.source_start < beat.start_time - 1.0:
                        errors.append(
                            f"片段 {clip.clip_index}: source_start ({clip.source_start:.1f}) "
                            f"远早于 beat 起始 ({beat.start_time:.1f})"
                        )
                    if clip.source_end > beat.end_time + 1.0:
                        errors.append(
                            f"片段 {clip.clip_index}: source_end ({clip.source_end:.1f}) "
                            f"远晚于 beat 结束 ({beat.end_time:.1f})"
                        )
        else:
            if clip.source_start < scene.start_time - 1.0:
                errors.append(
                    f"片段 {clip.clip_index}: source_start ({clip.source_start:.1f}) "
                    f"远早于场景起始 ({scene.start_time:.1f})"
                )

            if clip.source_end > scene.end_time + 1.0:
                errors.append(
                    f"片段 {clip.clip_index}: source_end ({clip.source_end:.1f}) "
                    f"远晚于场景结束 ({scene.end_time:.1f})"
                )

        # 速度
        if clip.speed <= 0:
            errors.append(f"片段 {clip.clip_index}: speed={clip.speed} 非法")

        # 音量范围
        vol = clip.audio_volume
        if vol < 0 or vol > 5.0:
            errors.append(f"片段 {clip.clip_index}: audio_volume={vol} 超出合理范围 [0, 5.0]")

        # transition 取值合法
        if clip.transition_in not in VALID_TRANSITIONS:
            errors.append(f"片段 {clip.clip_index}: transition_in='{clip.transition_in}' 非法")
        if clip.transition_out not in VALID_TRANSITIONS:
            errors.append(f"片段 {clip.clip_index}: transition_out='{clip.transition_out}' 非法")

    # 4. BGM 校验
    if plan.bgm.enabled and plan.bgm.path:
        if not Path(plan.bgm.path).exists():
            errors.append(f"BGM 文件不存在: {plan.bgm.path}")
    elif plan.bgm.enabled and not plan.bgm.path:
        errors.append("BGM 已启用但未指定 bgm_path")

    # 5. target_duration 偏差
    if plan.target_duration > 0 and plan.clips:
        actual = sum(c.timeline_end - c.timeline_start for c in plan.clips)
        deviation = abs(actual - plan.target_duration) / plan.target_duration
        if deviation > 0.20:
            errors.append(
                f"总时长偏差过大: 实际 {actual:.1f}s vs 目标 {plan.target_duration:.1f}s "
                f"(偏差 {deviation:.0%})"
            )

    # 6. 时间线连续性
    for i in range(len(plan.clips) - 1):
        gap = plan.clips[i+1].timeline_start - plan.clips[i].timeline_end
        if gap > 1.0:
            logger.warning(
                f"时间线有间隙: 片段 {i} 结束 {plan.clips[i].timeline_end:.1f}s, "
                f"片段 {i+1} 开始 {plan.clips[i+1].timeline_start:.1f}s"
            )

    if errors:
        logger.error(f"EditPlan 校验失败: {len(errors)} 个错误")
        for e in errors:
            logger.error(f"  - {e}")

    return errors
