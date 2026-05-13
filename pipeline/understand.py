# -*- coding: utf-8 -*-
"""
视频理解主流程（重构版）

流程重排为 "先切后提"：
  ingest → scene_detect → keyframe → audio_per_shot → asr_per_shot
  → vision → character → speaker_bind → event → build_memory → indexer

核心改动：
- ASR 改为按 shot 段处理，TranscriptSegment 天然携带 scene_index
- 新增 speaker_bind 步骤，建立 speaker ↔ character 映射
- build_memory 步骤会构建 MemoryUnit 和角色判定
- 支持断点续跑
"""
import json
from pathlib import Path

import config
from models.schemas import VideoMeta, Scene
from memory.store import load_meta, save_memory, load_memory
from utils.logger import get_logger

logger = get_logger("Understand")

# 理解流程步骤定义（重排后）
STEPS = [
    "ingest",              # 1. 入库 + 元信息
    "scene_detect",        # 2. 镜头切分（提前到音频之前）
    "keyframe_extract",    # 3. 关键帧抽取
    "asr",                 # 4. 按 shot 段 ASR（不再需要先提取整体音频）
    "vision",              # 5. OCR + 画面摘要
    "character",           # 6. 人物识别
    "speaker_bind",        # 7. Speaker ↔ Character 绑定（新增）
    "event",               # 8. 事件抽取
    "build_memory",        # 9. 构建 Video Memory（含 MemoryUnit + 角色判定）
    "indexer",             # 10. 构建检索索引（含 embedding）
]


def run_understand(
    video_path: str = None,
    video_id: str = None,
    resume: bool = False,
) -> str:
    """
    运行视频理解全流程。

    Args:
        video_path: 视频文件路径（新视频）
        video_id: 视频 ID（用于 resume）
        resume: 是否从断点继续

    Returns:
        video_id
    """
    config.init_dirs()

    # 确定从哪一步开始
    start_step = 0
    meta = None

    if resume and video_id:
        # 从断点继续
        progress = _load_progress(video_id)
        completed = progress.get("completed_steps", [])
        meta = load_meta(video_id)
        for i, step in enumerate(STEPS):
            if step not in completed:
                start_step = i
                break
        else:
            logger.info(f"视频 {video_id} 所有步骤已完成")
            return video_id
        logger.info(f"从步骤 {STEPS[start_step]} 继续 (已完成: {completed})")
    elif not video_path:
        raise ValueError("必须提供 --video 或 --video-id + --resume")

    # ═══ Step 1: Ingest ═══
    if start_step <= 0:
        logger.info("=" * 50)
        logger.info("步骤 1/10: 视频入库")
        logger.info("=" * 50)
        from pipeline.ingest import ingest_video
        meta = ingest_video(video_path, video_id)
        video_id = meta.video_id
        _save_progress(video_id, "ingest")

    if meta is None:
        meta = load_meta(video_id)

    # ═══ Step 2: Scene Detect（提前到音频之前）═══
    if start_step <= 1:
        logger.info("=" * 50)
        logger.info("步骤 2/10: 镜头切分")
        logger.info("=" * 50)
        from pipeline.scene_detect import detect_scenes
        scenes = detect_scenes(meta.storage_path, video_id)
        _save_progress(video_id, "scene_detect")
    else:
        # resume 模式：从已保存的 scenes.json 加载
        scenes_json = config.VIDEOS_DIR / video_id / "scenes" / "scenes.json"
        if scenes_json.exists():
            data = json.loads(scenes_json.read_text(encoding="utf-8"))
            scenes = [Scene(**s) for s in data]
        else:
            from pipeline.scene_detect import detect_scenes
            scenes = detect_scenes(meta.storage_path, video_id)

    # ═══ Step 3: Keyframe Extract ═══
    if start_step <= 2:
        logger.info("=" * 50)
        logger.info("步骤 3/10: 关键帧抽取")
        logger.info("=" * 50)
        from pipeline.keyframe import extract_keyframes
        scenes = extract_keyframes(meta.storage_path, video_id, scenes)
        _save_progress(video_id, "keyframe_extract")

    # ═══ Step 4: ASR（按 shot 段处理）═══
    if start_step <= 3:
        logger.info("=" * 50)
        logger.info("步骤 4/10: 按镜头 ASR 语音转文字")
        logger.info("=" * 50)
        from pipeline.asr import transcribe_audio
        transcripts = transcribe_audio(
            video_id=video_id,
            video_path=meta.storage_path,
            scenes=scenes,
        )
        if not transcripts:
            logger.warning("ASR 未产生任何结果，可能是无语音内容，仍标记完成")
        _save_progress(video_id, "asr")
    else:
        transcripts = _load_cached("transcripts", video_id)

    # ═══ Step 5: Vision (OCR + 画面摘要) ═══
    if start_step <= 4:
        logger.info("=" * 50)
        logger.info("步骤 5/10: OCR + 画面摘要")
        logger.info("=" * 50)
        from pipeline.vision import analyze_keyframes
        ocr_results, vision_summaries = analyze_keyframes(video_id, scenes)
        _save_progress(video_id, "vision")
    else:
        ocr_results = _load_cached("ocr", video_id)
        vision_summaries = _load_cached("vision", video_id)

    # ═══ Step 6: Character ═══
    if start_step <= 5:
        logger.info("=" * 50)
        logger.info("步骤 6/10: 人物识别")
        logger.info("=" * 50)
        from pipeline.character import detect_characters
        characters = detect_characters(video_id, scenes)
        if characters is None:
            raise RuntimeError("人物识别处理失败，返回 None")
        _save_progress(video_id, "character")
    else:
        characters = _load_cached("characters", video_id)

    # ═══ Step 7: Speaker ↔ Character 绑定（新增）═══
    if start_step <= 6:
        logger.info("=" * 50)
        logger.info("步骤 7/10: Speaker ↔ Character 绑定")
        logger.info("=" * 50)
        from pipeline.speaker_bind import bind_speakers_to_characters
        speaker_map, transcripts, characters = bind_speakers_to_characters(
            video_id, transcripts, characters, scenes
        )
        _save_progress(video_id, "speaker_bind")
    else:
        speaker_map = _load_speaker_map(video_id)

    # ═══ Step 8: Event ═══
    if start_step <= 7:
        logger.info("=" * 50)
        logger.info("步骤 8/10: 事件抽取")
        logger.info("=" * 50)
        from pipeline.event import extract_events
        events = extract_events(
            video_id, scenes, transcripts, vision_summaries, characters, meta.duration
        )
        if events is None:
            raise RuntimeError("事件抽取处理失败，返回 None")
        _save_progress(video_id, "event")
    else:
        events = _load_cached("events", video_id)

    # ═══ Step 9: Build Memory（含 MemoryUnit + 角色判定）═══
    if start_step <= 8:
        logger.info("=" * 50)
        logger.info("步骤 9/10: 构建 Video Memory")
        logger.info("=" * 50)

        from models.schemas import VideoMemory
        from pipeline.memory_builder import build_memory_units, assign_character_roles

        # 构建 MemoryUnit 列表
        memory_units = build_memory_units(
            scenes, transcripts, ocr_results, vision_summaries, characters, events
        )

        # 角色判定
        characters = assign_character_roles(characters, transcripts, events, meta.duration)

        memory = VideoMemory(
            video_id=video_id,
            meta=meta,
            scenes=scenes,
            transcripts=transcripts if isinstance(transcripts, list) else [],
            ocr_results=ocr_results if isinstance(ocr_results, list) else [],
            vision_summaries=vision_summaries if isinstance(vision_summaries, list) else [],
            characters=characters if isinstance(characters, list) else [],
            events=events if isinstance(events, list) else [],
            memory_units=memory_units,
            speaker_map=speaker_map if isinstance(speaker_map, dict) else {},
        )
        save_memory(memory)

        # 更新 meta 状态
        meta.status = "ready"
        meta_path = config.VIDEOS_DIR / video_id / "meta.json"
        meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")

        _save_progress(video_id, "build_memory")

    # ═══ Step 10: Indexer（含 embedding）═══
    if start_step <= 9:
        logger.info("=" * 50)
        logger.info("步骤 10/10: 构建检索索引")
        logger.info("=" * 50)
        try:
            from pipeline.indexer import build_search_index
            build_search_index(video_id)
            _save_progress(video_id, "indexer")
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            raise RuntimeError(f"索引构建失败: {e}") from e

    logger.info("=" * 50)
    logger.info(f"✅ 视频理解完成! video_id={video_id}")
    memory = load_memory(video_id)
    logger.info(f"   镜头数: {len(memory.scenes)}")
    logger.info(f"   台词数: {len(memory.transcripts)}")
    logger.info(f"   人物数: {len(memory.characters)}")
    logger.info(f"   事件数: {len(memory.events)}")
    logger.info(f"   MemoryUnit 数: {len(memory.memory_units)}")
    logger.info("=" * 50)

    return video_id


# ─── 进度管理 ──────────────────────────────────────────────

def _load_progress(video_id: str) -> dict:
    """加载处理进度"""
    progress_path = config.VIDEOS_DIR / video_id / "progress.json"
    if progress_path.exists():
        return json.loads(progress_path.read_text(encoding="utf-8"))
    return {"completed_steps": []}


def _save_progress(video_id: str, step: str):
    """保存处理进度"""
    progress_path = config.VIDEOS_DIR / video_id / "progress.json"
    progress = _load_progress(video_id)
    if step not in progress["completed_steps"]:
        progress["completed_steps"].append(step)
    progress_path.write_text(
        json.dumps(progress, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_cached(data_type: str, video_id: str):
    """加载已缓存的中间结果，反序列化为 Pydantic 模型对象"""
    from models.schemas import (
        TranscriptSegment, OCRResult, VisionSummary, Character, Event
    )
    model_map = {
        "transcripts": TranscriptSegment,
        "ocr": OCRResult,
        "vision": VisionSummary,
        "characters": Character,
        "events": Event,
    }
    video_dir = config.VIDEOS_DIR / video_id
    file_map = {
        "transcripts": "transcripts.json",
        "ocr": "ocr.json",
        "vision": "vision.json",
        "characters": "characters.json",
        "events": "events.json",
    }
    filename = file_map.get(data_type)
    model_cls = model_map.get(data_type)
    if filename and model_cls:
        path = video_dir / filename
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [model_cls(**item) for item in data]
    return []


def _load_speaker_map(video_id: str) -> dict:
    """加载 speaker_map.json"""
    path = config.VIDEOS_DIR / video_id / "speaker_map.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}
