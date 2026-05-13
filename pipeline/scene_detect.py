# -*- coding: utf-8 -*-
"""
镜头切分
使用 PySceneDetect 检测镜头边界，输出场景列表。
"""
import json
from pathlib import Path

import config
from models.schemas import Scene
from utils.logger import get_logger

logger = get_logger("SceneDetect")


def detect_scenes(video_path: str, video_id: str) -> list[Scene]:
    """
    使用 PySceneDetect 检测视频中的镜头边界。

    Args:
        video_path: 视频文件路径
        video_id: 视频 ID

    Returns:
        Scene 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    scenes_dir = video_dir / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)
    scenes_json = scenes_dir / "scenes.json"

    # 如果已存在，直接加载
    if scenes_json.exists():
        logger.info(f"镜头切分结果已存在，直接加载: {scenes_json}")
        data = json.loads(scenes_json.read_text(encoding="utf-8"))
        return [Scene(**s) for s in data]

    logger.info(f"开始镜头切分: {video_path}")
    logger.info(
        f"参数: threshold={config.SCENE_DETECT_THRESHOLD}, "
        f"min_len={config.SCENE_DETECT_MIN_LEN}s"
    )

    # 使用 PySceneDetect
    from scenedetect import open_video, SceneManager
    from scenedetect.detectors import ContentDetector

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=config.SCENE_DETECT_THRESHOLD,
            min_scene_len=int(config.SCENE_DETECT_MIN_LEN * video.frame_rate),
        )
    )

    logger.info("正在分析视频帧...")
    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    logger.info(f"检测到 {len(scene_list)} 个镜头")

    # 转换为 Scene 对象
    scenes = []
    for idx, (start, end) in enumerate(scene_list):
        start_sec = start.get_seconds()
        end_sec = end.get_seconds()
        scene = Scene(
            scene_index=idx,
            start_time=round(start_sec, 3),
            end_time=round(end_sec, 3),
            duration=round(end_sec - start_sec, 3),
        )
        scenes.append(scene)

    # 保存结果
    scenes_data = [s.model_dump() for s in scenes]
    scenes_json.write_text(
        json.dumps(scenes_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"镜头切分结果已保存: {scenes_json}")

    # 打印概要
    if scenes:
        durations = [s.duration for s in scenes]
        logger.info(
            f"镜头统计: 总数={len(scenes)}, "
            f"最短={min(durations):.1f}s, "
            f"最长={max(durations):.1f}s, "
            f"平均={sum(durations)/len(durations):.1f}s"
        )

    return scenes
