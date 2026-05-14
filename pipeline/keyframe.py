# -*- coding: utf-8 -*-
"""
多帧关键帧抽取（v2）

v2 变更:
- 从每 shot 提取 1 帧升级为多帧采样
- 采样策略根据 shot 时长动态调整
- 输出 Shot.keyframe_paths（多帧）同时保留 Shot.keyframe_path（首帧兼容）
"""
import json
from pathlib import Path

import config
from models.schemas import Shot
from utils.ffmpeg_utils import extract_keyframe
from utils.logger import get_logger

logger = get_logger("MultiKeyframe")


def _get_sample_count(duration: float) -> int:
    """根据 shot 时长决定采样帧数"""
    if duration < 2.0:
        return 1
    elif duration < 5.0:
        return 2
    elif duration < 15.0:
        return 3
    else:
        return min(max(3, int(duration / 5)), config.MULTI_KEYFRAME_MAX)


def _get_sample_timestamps(start: float, end: float, count: int) -> list[float]:
    """计算均匀分布的采样时间点（避开首尾转场帧）"""
    duration = end - start
    if count == 1:
        # 单帧取中点
        return [start + duration * 0.5]

    # 避开首尾各 10% 的转场区域
    safe_start = start + duration * 0.1
    safe_end = end - duration * 0.1
    safe_duration = safe_end - safe_start

    if safe_duration <= 0:
        return [start + duration * 0.5]

    step = safe_duration / (count - 1) if count > 1 else 0
    return [round(safe_start + step * i, 3) for i in range(count)]


def extract_multi_keyframes(
    video_path: str, video_id: str, shots: list[Shot]
) -> list[Shot]:
    """
    为每个 shot 提取多帧关键帧。

    采样策略:
      duration < 2s  → 1 帧
      duration 2-5s  → 2 帧
      duration 5-15s → 3 帧
      duration > 15s → max(3, duration/5) 帧, 上限 MULTI_KEYFRAME_MAX

    Args:
        video_path: 视频文件路径
        video_id: 视频 ID
        shots: 镜头列表

    Returns:
        更新了 keyframe_path 和 keyframe_paths 的 Shot 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    keyframes_dir = video_dir / "scenes" / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    updated_shots = []
    total_extracted = 0

    for shot in shots:
        sample_count = _get_sample_count(shot.duration)
        timestamps = _get_sample_timestamps(shot.start_time, shot.end_time, sample_count)

        extracted_paths = []
        for frame_idx, ts in enumerate(timestamps):
            frame_name = f"scene_{shot.scene_index:04d}_f{frame_idx}.jpg"
            frame_path = keyframes_dir / frame_name

            if frame_path.exists():
                extracted_paths.append(str(frame_path))
                continue

            try:
                extract_keyframe(video_path, ts, str(frame_path))
                extracted_paths.append(str(frame_path))
                total_extracted += 1
            except Exception as e:
                logger.warning(
                    f"关键帧提取失败 (shot {shot.scene_index}, frame {frame_idx}): {e}"
                )

        shot.keyframe_paths = extracted_paths
        # 保持旧字段兼容: keyframe_path = 首帧（或已有的旧路径）
        if extracted_paths:
            shot.keyframe_path = extracted_paths[0]
        elif not shot.keyframe_path:
            # 回退: 尝试旧格式的单帧
            old_path = keyframes_dir / f"scene_{shot.scene_index:04d}.jpg"
            if old_path.exists():
                shot.keyframe_path = str(old_path)
                shot.keyframe_paths = [str(old_path)]

        updated_shots.append(shot)

    logger.info(
        f"多帧关键帧提取完成: 新提取 {total_extracted} 帧, "
        f"{len(shots)} 个镜头"
    )

    # 更新 scenes.json
    scenes_json = video_dir / "scenes" / "scenes.json"
    scenes_data = [s.model_dump() for s in updated_shots]
    scenes_json.write_text(
        json.dumps(scenes_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return updated_shots


# ── 向后兼容入口 ──────────────────────────────────────────────

def extract_keyframes(
    video_path: str, video_id: str, scenes: list[Shot]
) -> list[Shot]:
    """
    向后兼容入口 — 内部调用 extract_multi_keyframes。

    保留原有函数签名，方便旧代码 `from pipeline.keyframe import extract_keyframes` 调用。
    """
    return extract_multi_keyframes(video_path, video_id, scenes)
