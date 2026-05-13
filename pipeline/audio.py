# -*- coding: utf-8 -*-
"""
音频提取（已弃用）

.. deprecated::
    旧版流程使用整体音频提取 → 整体 ASR → 事后反查 scene。
    新版已改为 "按 shot 段提取 → 按 shot 段 ASR"，音频提取集成到
    pipeline.asr.transcribe_audio() 内部。

    保留此文件仅为向后兼容。不应在新代码中引用此模块。

原始功能：从视频中提取音频，输出 WAV 格式（适合后续 ASR 处理）。
"""
from pathlib import Path

import config
from models.schemas import VideoMeta
from utils.ffmpeg_utils import extract_audio
from utils.logger import get_logger

logger = get_logger("AudioExtract")


def extract_video_audio(meta: VideoMeta) -> str:
    """
    从视频中提取音频。

    Args:
        meta: 视频元信息

    Returns:
        音频文件路径
    """
    video_dir = config.VIDEOS_DIR / meta.video_id
    audio_path = str(video_dir / "audio.wav")

    if Path(audio_path).exists():
        logger.info(f"音频已存在，跳过提取: {audio_path}")
        return audio_path

    logger.info(f"开始提取音频: {meta.storage_path}")
    extract_audio(meta.storage_path, audio_path)
    logger.info(f"音频提取完成: {audio_path}")

    # 更新 meta
    meta.audio_path = audio_path
    meta_path = video_dir / "meta.json"
    meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    return audio_path
