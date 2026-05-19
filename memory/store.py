# -*- coding: utf-8 -*-
"""
Video Memory 存储（v3）
JSON 文件读写操作，汇总所有理解结果。

v3 变更:
- 支持新字段: audio_prosodies, multimodal_alignments, chapters,
  narrative_signals, recomposition_signals, chapter_memory_units
- 向后兼容旧 memory.json（通过 Optional 字段 + 默认值）
"""
import json
from pathlib import Path

import config
from models.schemas import (
    VideoMeta, VideoMemory, Shot, Scene, TranscriptSegment,
    OCRResult, VisionSummary, Character, CharacterDeep,
    Event, EventGraph, Beat, StoryScene,
    CharacterRelation, EditSignal,
    MemoryUnit, BeatMemoryUnit, SceneMemoryUnit,
    # v3 新增
    AudioProsody, MultimodalAlignment, Chapter,
    NarrativeSignal, RecompositionSignal, ChapterMemoryUnit,
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

    # Shot/Scene
    shots = _load_json_list(video_dir / "scenes" / "scenes.json", Shot)

    # 基础模态
    transcripts = _load_json_list(video_dir / "transcripts.json", TranscriptSegment)
    ocr_results = _load_json_list(video_dir / "ocr.json", OCRResult)
    vision_summaries = _load_json_list(video_dir / "vision.json", VisionSummary)

    # 人物
    # 尝试加载为 CharacterDeep，回退到 Character
    characters_deep = _load_json_list(video_dir / "characters.json", CharacterDeep)
    characters = characters_deep if characters_deep else _load_json_list(
        video_dir / "characters.json", Character
    )

    # 人物关系
    character_relations = _load_json_list(
        video_dir / "character_relations.json", CharacterRelation
    )

    # 事件
    events = _load_json_list(video_dir / "events.json", Event)

    # 事件图谱
    event_graph = None
    graph_path = video_dir / "event_graph.json"
    if graph_path.exists():
        try:
            data = json.loads(graph_path.read_text(encoding="utf-8"))
            event_graph = EventGraph(**data)
        except Exception as e:
            logger.warning(f"加载 event_graph.json 失败: {e}")

    # Beat
    beats = _load_json_list(video_dir / "beats.json", Beat)

    # StoryScene
    story_scenes = _load_json_list(video_dir / "story_scenes.json", StoryScene)

    # EditSignal
    edit_signals = _load_json_list(video_dir / "edit_signals.json", EditSignal)

    # v3 新增数据
    audio_prosodies = _load_json_list(video_dir / "audio_prosody.json", AudioProsody)
    multimodal_alignments = _load_json_list(video_dir / "multimodal_alignments.json", MultimodalAlignment)
    chapters = _load_json_list(video_dir / "chapters.json", Chapter)
    narrative_signals = _load_json_list(video_dir / "narrative_signals.json", NarrativeSignal)
    recomposition_signals = _load_json_list(video_dir / "recomposition_signals.json", RecompositionSignal)

    # speaker_map
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
        shots=shots,
        scenes=shots,  # 兼容
        transcripts=transcripts,
        ocr_results=ocr_results,
        vision_summaries=vision_summaries,
        # v3 新增
        audio_prosodies=audio_prosodies,
        multimodal_alignments=multimodal_alignments,
        # 人物
        characters=characters,
        characters_deep=characters_deep,
        character_relations=character_relations,
        speaker_map=speaker_map,
        # 叙事
        beats=beats,
        story_scenes=story_scenes,
        chapters=chapters,
        event_graph=event_graph,
        events=events,
        # 信号
        edit_signals=edit_signals,
        narrative_signals=narrative_signals,
        recomposition_signals=recomposition_signals,
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
