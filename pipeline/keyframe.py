# -*- coding: utf-8 -*-
"""
关键帧抽取
在每个镜头的起始位置（偏移少许）提取关键帧图片。
"""
import json
from pathlib import Path

import config
from models.schemas import Scene
from utils.ffmpeg_utils import extract_keyframe
from utils.logger import get_logger

logger = get_logger("Keyframe")


def extract_keyframes(
    video_path: str, video_id: str, scenes: list[Scene]
) -> list[Scene]:
    """
    为每个镜头提取关键帧。
    在每个镜头起始时间偏移 0.5 秒处提取一帧（避免转场模糊帧）。

    Args:
        video_path: 视频文件路径
        video_id: 视频 ID
        scenes: 镜头列表

    Returns:
        更新了 keyframe_path 的 Scene 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    keyframes_dir = video_dir / "scenes" / "keyframes"
    keyframes_dir.mkdir(parents=True, exist_ok=True)

    updated_scenes = []
    extracted_count = 0

    for scene in scenes:
        keyframe_name = f"scene_{scene.scene_index:04d}.jpg"
        keyframe_path = keyframes_dir / keyframe_name

        if keyframe_path.exists():
            scene.keyframe_path = str(keyframe_path)
            updated_scenes.append(scene)
            continue

        # 在镜头起始偏移 0.5s 提取，避免转场帧
        offset = min(0.5, scene.duration * 0.3)
        timestamp = scene.start_time + offset

        try:
            extract_keyframe(video_path, timestamp, str(keyframe_path))
            scene.keyframe_path = str(keyframe_path)
            extracted_count += 1
        except Exception as e:
            logger.warning(f"关键帧提取失败 (scene {scene.scene_index}): {e}")
            scene.keyframe_path = None

        updated_scenes.append(scene)

    logger.info(f"关键帧提取完成: {extracted_count} 新提取, {len(scenes)} 总镜头")

    # 更新 scenes.json
    scenes_json = video_dir / "scenes" / "scenes.json"
    scenes_data = [s.model_dump() for s in updated_scenes]
    scenes_json.write_text(
        json.dumps(scenes_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return updated_scenes
