# -*- coding: utf-8 -*-
"""
Video Memory 存储
JSON 文件读写操作，汇总所有理解结果。
"""
import json
from pathlib import Path

import config
from models.schemas import (
    VideoMeta, VideoMemory, Scene, TranscriptSegment,
    OCRResult, VisionSummary, Character, Event, MemoryUnit,
)
from utils.logger import get_logger

logger = get_logger("MemoryStore")


def load_meta(video_id: str) -> VideoMeta:
    """加载视频元信息"""
    meta_path = config.VIDEOS_DIR / video_id / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"视频元信息不存在: {meta_path}")
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    return VideoMeta(**data)


def load_memory(video_id: str) -> VideoMemory:
    """加载完整的 Video Memory"""
    video_dir = config.VIDEOS_DIR / video_id
    memory_path = video_dir / "memory.json"

    if memory_path.exists():
        data = json.loads(memory_path.read_text(encoding="utf-8"))
        return VideoMemory(**data)

    # 如果 memory.json 不存在，尝试从各个单独文件汇总
    return _assemble_memory(video_id)


def save_memory(memory: VideoMemory) -> str:
    """保存 Video Memory"""
    video_dir = config.VIDEOS_DIR / memory.video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    memory_path = video_dir / "memory.json"
    memory_path.write_text(
        memory.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info(f"Video Memory 已保存: {memory_path}")
    return str(memory_path)


def _assemble_memory(video_id: str) -> VideoMemory:
    """从各个 JSON 文件汇总为 Video Memory"""
    video_dir = config.VIDEOS_DIR / video_id

    meta = load_meta(video_id)

    scenes = _load_json_list(video_dir / "scenes" / "scenes.json", Scene)
    transcripts = _load_json_list(video_dir / "transcripts.json", TranscriptSegment)
    ocr_results = _load_json_list(video_dir / "ocr.json", OCRResult)
    vision_summaries = _load_json_list(video_dir / "vision.json", VisionSummary)
    characters = _load_json_list(video_dir / "characters.json", Character)
    events = _load_json_list(video_dir / "events.json", Event)

    # 加载 speaker_map（新增）
    speaker_map = {}
    speaker_map_path = video_dir / "speaker_map.json"
    if speaker_map_path.exists():
        try:
            speaker_map = json.loads(speaker_map_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"加载 speaker_map.json 失败: {e}")

    memory = VideoMemory(
        video_id=video_id,
        meta=meta,
        scenes=scenes,
        transcripts=transcripts,
        ocr_results=ocr_results,
        vision_summaries=vision_summaries,
        characters=characters,
        events=events,
        speaker_map=speaker_map,
        # memory_units 会在 memory.json 中持久化，
        # 从散文件组装时不包含，需要重新构建索引才会有
    )
    return memory


def _load_json_list(path: Path, model_cls):
    """加载 JSON 数组文件"""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [model_cls(**item) for item in data]
    except Exception as e:
        logger.warning(f"加载 {path} 失败: {e}")
    return []


def list_videos() -> list[dict]:
    """列出所有已入库的视频"""
    videos = []
    if not config.VIDEOS_DIR.exists():
        return videos
    for d in sorted(config.VIDEOS_DIR.iterdir()):
        if d.is_dir():
            meta_path = d / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    memory_exists = (d / "memory.json").exists()
                    videos.append({
                        "video_id": meta.get("video_id", d.name),
                        "filename": meta.get("filename", ""),
                        "duration": meta.get("duration", 0),
                        "status": meta.get("status", "unknown"),
                        "memory_ready": memory_exists,
                    })
                except Exception:
                    pass
    return videos
