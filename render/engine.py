# -*- coding: utf-8 -*-
"""
渲染引擎
读取 EditPlan，通过 FFmpeg 执行裁剪、拼接、转场、音频处理，输出成片。
纯确定性执行，不依赖任何 LLM。
"""
import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime

import config
from models.schemas import EditPlan
from memory.store import load_memory
from render.validator import validate_plan
from render import ffmpeg_ops
from utils.logger import get_logger

logger = get_logger("RenderEngine")


def run_render(plan_id: str) -> str:
    """
    渲染入口：读取 EditPlan 并执行渲染。

    Args:
        plan_id: EditPlan ID

    Returns:
        输出视频文件路径
    """
    config.init_dirs()

    # 1. 加载 EditPlan
    plan_path = config.EDITPLANS_DIR / f"{plan_id}.json"
    if not plan_path.exists():
        raise FileNotFoundError(f"EditPlan 不存在: {plan_path}")

    plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
    plan = EditPlan(**plan_data)
    logger.info(f"加载 EditPlan: {plan.title} ({len(plan.clips)} 个片段)")

    # 2. 加载 Video Memory
    memory = load_memory(plan.video_id)
    source_video = memory.meta.storage_path

    # 3. 校验
    errors = validate_plan(plan, memory)
    if errors:
        raise ValueError(f"EditPlan 校验失败:\n" + "\n".join(f"  - {e}" for e in errors))
    logger.info("EditPlan 校验通过")

    # 4. 渲染
    render_id = f"render_{uuid.uuid4().hex[:8]}"
    render_dir = config.RENDERS_DIR / render_id
    clips_dir = render_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    output_path = str(render_dir / "output.mp4")

    try:
        result = _render_pipeline(plan, source_video, clips_dir, output_path)
        logger.info(f"✅ 渲染完成: {result}")
        return result
    except Exception as e:
        logger.error(f"渲染失败: {e}")
        raise


def _render_pipeline(
    plan: EditPlan,
    source_video: str,
    clips_dir: Path,
    output_path: str,
) -> str:
    """渲染流水线"""

    # ═══ Step 1: 裁剪每个片段 ═══
    logger.info("Step 1: 裁剪片段")
    clip_paths = []

    for clip in plan.clips:
        clip_output = str(clips_dir / f"clip_{clip.clip_index:03d}.mp4")
        logger.info(
            f"  裁剪片段 {clip.clip_index}: "
            f"scene={clip.source_scene_index}, "
            f"{clip.source_start:.1f}s-{clip.source_end:.1f}s "
            f"({clip.source_end - clip.source_start:.1f}s)"
        )

        # 精确裁剪
        ffmpeg_ops.cut_clip_precise(
            source_video, clip.source_start, clip.source_end, clip_output
        )

        # 变速处理
        if abs(clip.speed - 1.0) > 0.01:
            speed_output = str(clips_dir / f"clip_{clip.clip_index:03d}_speed.mp4")
            clip_output = ffmpeg_ops.adjust_speed(clip_output, speed_output, clip.speed)

        # 音量调整
        if clip.audio_volume != 1.0:
            vol_output = str(clips_dir / f"clip_{clip.clip_index:03d}_vol.mp4")
            ffmpeg_ops.adjust_volume(clip_output, vol_output, clip.audio_volume)
            clip_output = vol_output

        # 淡入淡出（只处理首尾片段）
        if clip.clip_index == 0 and clip.transition_in == "fade_in":
            fade_output = str(clips_dir / f"clip_{clip.clip_index:03d}_fade.mp4")
            duration = clip.source_end - clip.source_start
            clip_output = ffmpeg_ops.apply_fade(
                clip_output, fade_output, fade_in=1.0, duration=duration
            )
        if clip.clip_index == len(plan.clips) - 1 and clip.transition_out in ("fade_out", "fade"):
            fade_output = str(clips_dir / f"clip_{clip.clip_index:03d}_fadeout.mp4")
            duration = clip.source_end - clip.source_start
            clip_output = ffmpeg_ops.apply_fade(
                clip_output, fade_output, fade_out=1.5, duration=duration
            )

        clip_paths.append(clip_output)

    if not clip_paths:
        raise ValueError("没有成功裁剪的片段")

    logger.info(f"Step 1 完成: {len(clip_paths)} 个片段")

    # ═══ Step 2: 标准化片段参数（确保拼接兼容） ═══
    logger.info("Step 2: 标准化片段")
    normalized_paths = []
    for i, cp in enumerate(clip_paths):
        norm_output = str(clips_dir / f"norm_{i:03d}.mp4")
        ffmpeg_ops.normalize_clip(cp, norm_output)
        normalized_paths.append(norm_output)

    # ═══ Step 3: 拼接 ═══
    logger.info("Step 3: 拼接片段")
    joined_path = str(clips_dir.parent / "joined.mp4")
    ffmpeg_ops.concat_clips(normalized_paths, joined_path)

    # ═══ Step 4: BGM 混合（可选） ═══
    final_path = joined_path
    if plan.bgm.enabled and plan.bgm.path and Path(plan.bgm.path).exists():
        logger.info("Step 4: 混合背景音乐")
        bgm_output = str(clips_dir.parent / "with_bgm.mp4")
        final_path = ffmpeg_ops.mix_bgm(
            joined_path, plan.bgm.path, bgm_output,
            bgm_volume=plan.bgm.volume,
            fade_in=plan.bgm.fade_in,
            fade_out=plan.bgm.fade_out,
        )
    else:
        logger.info("Step 4: 跳过（无 BGM）")

    # ═══ Step 5: 输出 ═══
    logger.info("Step 5: 输出成片")
    shutil.move(final_path, output_path)

    # 清理临时文件（保留 output.mp4）
    _cleanup(clips_dir)

    logger.info(f"渲染完成: {output_path}")
    return output_path





def _cleanup(clips_dir: Path):
    """清理临时裁剪文件"""
    try:
        if clips_dir.exists():
            shutil.rmtree(clips_dir)
    except Exception as e:
        logger.warning(f"清理临时文件失败: {e}")
