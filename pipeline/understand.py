# -*- coding: utf-8 -*-
"""
视频理解主流程（v3 — 深度多模态 + 面向剪辑决策）

层级结构: Shot → Beat → StoryScene → Chapter → EventGraph
流程（17步）:
  ingest → shot_detect → multi_keyframe → asr_windowed
  → vision → audio_analysis → character_deep → speaker_bind
  → multimodal_align → beat_detect → story_scene_detect
  → chapter_detect → event_graph → character_arc
  → edit_signal → build_memory → indexer

v3 新增步骤:
- audio_analysis: 音频韵律分析（音乐/音效/沉默/语速/音量/语音情绪）
- multimodal_align: 多模态对齐（shot/ASR/speaker/character/vision/audio）
- chapter_detect: 长视频大段落检测（StoryScene → Chapter）

v3 增强:
- vision: micro_clip（镜头运动/互动/景别）
- event: evidence + confidence
- edit_signal: 三类信号（EditSignal + NarrativeSignal + RecompositionSignal）
- memory_builder: 四层 MemoryUnit（+Chapter 级）
- indexer: 九维索引（+音频索引 + 章节索引）
"""
import json
from pathlib import Path

import config
from models.schemas import VideoMeta, Shot, Scene
from memory.store import load_meta, save_memory, load_memory
from utils.logger import get_logger

logger = get_logger("Understand")

# 理解流程步骤定义（v3）
STEPS = [
    "ingest",              # 1.  入库 + 元信息
    "shot_detect",         # 2.  镜头切分
    "multi_keyframe",      # 3.  多帧采样
    "asr_windowed",        # 4.  长窗口 ASR + 回填 shot
    "vision",              # 5.  多帧画面理解 + micro_clip
    "audio_analysis",      # 6.  音频韵律分析 🆕
    "character_deep",      # 7.  深度人物分析
    "speaker_bind",        # 8.  Speaker ↔ Character 绑定
    "multimodal_align",    # 9.  多模态对齐 🆕
    "beat_detect",         # 10. 剧情节拍检测
    "story_scene_detect",  # 11. 故事场景检测
    "chapter_detect",      # 12. 长视频大段落检测 🆕
    "event_graph",         # 13. 事件图谱构建
    "character_arc",       # 14. 人物弧线 + 关系图
    "edit_signal",         # 15. 三类信号计算
    "build_memory",        # 16. 四层 VideoMemory 构建
    "indexer",             # 17. 九维索引构建
]

# 旧步骤名 → 新步骤名映射（向后兼容 progress.json）
_STEP_ALIASES = {
    "scene_detect": "shot_detect",
    "keyframe_extract": "multi_keyframe",
    "asr": "asr_windowed",
    "character": "character_deep",
    "event": "event_graph",
}


def run_understand(
    video_path: str = None,
    video_id: str = None,
    resume: bool = False,
) -> str:
    """
    运行视频理解全流程（v3）。

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
        # 将旧步骤名转换为新名
        completed = [_STEP_ALIASES.get(s, s) for s in completed]
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

    total_steps = len(STEPS)

    # ═══ Step 1: Ingest ═══
    if start_step <= 0:
        logger.info("=" * 50)
        logger.info(f"步骤 1/{total_steps}: 视频入库")
        logger.info("=" * 50)
        from pipeline.ingest import ingest_video
        meta = ingest_video(video_path, video_id)
        video_id = meta.video_id
        _save_progress(video_id, "ingest")

    if meta is None:
        meta = load_meta(video_id)

    # ═══ Step 2: Shot Detect ═══
    if start_step <= 1:
        logger.info("=" * 50)
        logger.info(f"步骤 2/{total_steps}: 镜头切分")
        logger.info("=" * 50)
        from pipeline.scene_detect import detect_scenes
        shots = detect_scenes(meta.storage_path, video_id)
        _save_progress(video_id, "shot_detect")
    else:
        shots = _load_shots(video_id)

    # ═══ Step 3: Multi Keyframe ═══
    if start_step <= 2:
        logger.info("=" * 50)
        logger.info(f"步骤 3/{total_steps}: 多帧关键帧采样")
        logger.info("=" * 50)
        from pipeline.keyframe import extract_multi_keyframes
        shots = extract_multi_keyframes(meta.storage_path, video_id, shots)
        _save_progress(video_id, "multi_keyframe")

    # ═══ Step 4: ASR (Windowed) ═══
    if start_step <= 3:
        logger.info("=" * 50)
        logger.info(f"步骤 4/{total_steps}: 长窗口 ASR 语音转文字")
        logger.info("=" * 50)
        from pipeline.asr import transcribe_audio
        transcripts = transcribe_audio(
            video_id=video_id,
            video_path=meta.storage_path,
            scenes=shots,
        )
        if not transcripts:
            logger.warning("ASR 未产生任何结果，可能是无语音内容，仍标记完成")
        _save_progress(video_id, "asr_windowed")
    else:
        transcripts = _load_cached("transcripts", video_id)

    # ═══ Step 5: Vision (OCR + 多帧画面摘要 + micro_clip) ═══
    if start_step <= 4:
        logger.info("=" * 50)
        logger.info(f"步骤 5/{total_steps}: OCR + 多帧画面摘要 + micro_clip")
        logger.info("=" * 50)
        from pipeline.vision import analyze_keyframes
        ocr_results, vision_summaries = analyze_keyframes(video_id, shots)
        _save_progress(video_id, "vision")
    else:
        ocr_results = _load_cached("ocr", video_id)
        vision_summaries = _load_cached("vision", video_id)

    # ═══ Step 6: Audio Analysis 🆕 ═══
    if start_step <= 5:
        logger.info("=" * 50)
        logger.info(f"步骤 6/{total_steps}: 音频韵律分析")
        logger.info("=" * 50)
        from pipeline.audio_analysis import analyze_audio
        audio_prosodies = analyze_audio(
            video_id, meta.storage_path, shots,
            transcripts=transcripts, vision_summaries=vision_summaries,
        )
        _save_progress(video_id, "audio_analysis")
    else:
        audio_prosodies = _load_cached("audio_prosody", video_id)

    # ═══ Step 7: Character Deep ═══
    if start_step <= 6:
        logger.info("=" * 50)
        logger.info(f"步骤 7/{total_steps}: 深度人物分析")
        logger.info("=" * 50)
        from pipeline.character import detect_characters
        characters = detect_characters(video_id, shots)
        if characters is None:
            raise RuntimeError("人物识别处理失败，返回 None")
        _save_progress(video_id, "character_deep")
    else:
        characters = _load_cached("characters", video_id)

    # ═══ Step 8: Speaker ↔ Character 绑定 ═══
    if start_step <= 7:
        logger.info("=" * 50)
        logger.info(f"步骤 8/{total_steps}: Speaker ↔ Character 绑定")
        logger.info("=" * 50)
        from pipeline.speaker_bind import bind_speakers_to_characters
        speaker_map, transcripts, characters = bind_speakers_to_characters(
            video_id, transcripts, characters, shots
        )
        _save_progress(video_id, "speaker_bind")
    else:
        speaker_map = _load_speaker_map(video_id)

    # ═══ Step 9: Multimodal Alignment 🆕 ═══
    if start_step <= 8:
        logger.info("=" * 50)
        logger.info(f"步骤 9/{total_steps}: 多模态对齐")
        logger.info("=" * 50)
        from pipeline.multimodal_align import align_multimodal
        alignments = align_multimodal(
            video_id, shots, transcripts, vision_summaries,
            characters, speaker_map, audio_prosodies,
        )
        _save_progress(video_id, "multimodal_align")
    else:
        alignments = _load_cached("multimodal_alignments", video_id)

    # ═══ Step 10: Beat Detect ═══
    if start_step <= 9:
        logger.info("=" * 50)
        logger.info(f"步骤 10/{total_steps}: 剧情节拍检测")
        logger.info("=" * 50)
        from pipeline.beat_detect import detect_beats
        beats = detect_beats(video_id, shots, transcripts, vision_summaries, characters)
        _save_progress(video_id, "beat_detect")
    else:
        beats = _load_cached("beats", video_id)

    # ═══ Step 11: Story Scene Detect ═══
    if start_step <= 10:
        logger.info("=" * 50)
        logger.info(f"步骤 11/{total_steps}: 故事场景检测")
        logger.info("=" * 50)
        from pipeline.story_scene_detect import detect_story_scenes
        story_scenes = detect_story_scenes(video_id, shots, beats)
        _save_progress(video_id, "story_scene_detect")
    else:
        story_scenes = _load_cached("story_scenes", video_id)

    # ═══ Step 12: Chapter Detect 🆕 ═══
    if start_step <= 11:
        logger.info("=" * 50)
        logger.info(f"步骤 12/{total_steps}: 长视频大段落检测")
        logger.info("=" * 50)
        from pipeline.chapter_detect import detect_chapters
        chapters = detect_chapters(
            video_id, story_scenes, beats, shots, meta.duration,
        )
        _save_progress(video_id, "chapter_detect")
    else:
        chapters = _load_cached("chapters", video_id)

    # ═══ Step 13: Event Graph ═══
    if start_step <= 12:
        logger.info("=" * 50)
        logger.info(f"步骤 13/{total_steps}: 事件图谱构建")
        logger.info("=" * 50)
        from pipeline.event import extract_events
        events, event_graph = extract_events(
            video_id, shots, transcripts, vision_summaries, characters,
            meta.duration, beats=beats, story_scenes=story_scenes,
        )
        if events is None:
            raise RuntimeError("事件抽取处理失败，返回 None")
        _save_progress(video_id, "event_graph")
    else:
        events = _load_cached("events", video_id)
        event_graph = _load_event_graph(video_id)

    # ═══ Step 14: Character Arc ═══
    if start_step <= 13:
        logger.info("=" * 50)
        logger.info(f"步骤 14/{total_steps}: 人物弧线 + 关系图")
        logger.info("=" * 50)
        from pipeline.character_arc import analyze_character_arcs
        characters, character_relations = analyze_character_arcs(
            video_id, characters, events, beats, transcripts, meta.duration
        )
        _save_progress(video_id, "character_arc")
    else:
        character_relations = _load_cached("character_relations", video_id)

    # ═══ Step 15: Edit Signal (三类信号) ═══
    if start_step <= 14:
        logger.info("=" * 50)
        logger.info(f"步骤 15/{total_steps}: 三类信号计算")
        logger.info("=" * 50)
        from pipeline.edit_signal import compute_edit_signals
        edit_signals, narrative_signals, recomposition_signals = compute_edit_signals(
            video_id, shots, beats, story_scenes, events,
            characters, transcripts, vision_summaries,
        )
        _save_progress(video_id, "edit_signal")
    else:
        edit_signals = _load_cached("edit_signals", video_id)
        narrative_signals = _load_cached("narrative_signals", video_id)
        recomposition_signals = _load_cached("recomposition_signals", video_id)

    # ═══ Step 16: Build Memory ═══
    if start_step <= 15:
        logger.info("=" * 50)
        logger.info(f"步骤 16/{total_steps}: 构建四层 VideoMemory")
        logger.info("=" * 50)

        from models.schemas import VideoMemory, EventGraph as EG
        from pipeline.memory_builder import (
            build_memory_units, build_beat_memory_units,
            build_scene_memory_units, build_chapter_memory_units,
            assign_character_roles,
        )

        # Shot 级 MemoryUnit
        memory_units = build_memory_units(
            shots, transcripts, ocr_results, vision_summaries,
            characters, events, beats, story_scenes, edit_signals,
            chapters=chapters,
            audio_prosodies=audio_prosodies,
            alignments=alignments,
        )

        # Beat 级 MemoryUnit
        beat_memory_units = build_beat_memory_units(
            beats, shots, transcripts, vision_summaries, edit_signals,
        )

        # StoryScene 级 MemoryUnit
        scene_memory_units = build_scene_memory_units(
            story_scenes, beats, edit_signals,
        )

        # Chapter 级 MemoryUnit 🆕
        chapter_memory_units = build_chapter_memory_units(
            chapters, story_scenes, edit_signals, narrative_signals,
        )

        # 角色判定
        characters = assign_character_roles(characters, transcripts, events, meta.duration)

        # 加载 event_graph（如果还没有）
        if not isinstance(event_graph, EG):
            event_graph = _load_event_graph(video_id)

        memory = VideoMemory(
            video_id=video_id,
            meta=meta,
            # Shot 层
            shots=shots,
            scenes=shots,  # 兼容旧字段
            transcripts=transcripts if isinstance(transcripts, list) else [],
            ocr_results=ocr_results if isinstance(ocr_results, list) else [],
            vision_summaries=vision_summaries if isinstance(vision_summaries, list) else [],
            # 音频层 🆕
            audio_prosodies=audio_prosodies if isinstance(audio_prosodies, list) else [],
            # 多模态对齐层 🆕
            multimodal_alignments=alignments if isinstance(alignments, list) else [],
            # 人物层
            characters=characters if isinstance(characters, list) else [],
            characters_deep=characters if isinstance(characters, list) else [],
            character_relations=character_relations if isinstance(character_relations, list) else [],
            speaker_map=speaker_map if isinstance(speaker_map, dict) else {},
            # 叙事层
            beats=beats if isinstance(beats, list) else [],
            story_scenes=story_scenes if isinstance(story_scenes, list) else [],
            chapters=chapters if isinstance(chapters, list) else [],
            event_graph=event_graph,
            events=events if isinstance(events, list) else [],
            # 剪辑信号层
            edit_signals=edit_signals if isinstance(edit_signals, list) else [],
            narrative_signals=narrative_signals if isinstance(narrative_signals, list) else [],
            recomposition_signals=recomposition_signals if isinstance(recomposition_signals, list) else [],
            # Memory 层
            memory_units=memory_units,
            beat_memory_units=beat_memory_units,
            scene_memory_units=scene_memory_units,
            chapter_memory_units=chapter_memory_units,
        )
        save_memory(memory)

        # 更新 meta 状态
        meta.status = "ready"
        meta_path = config.VIDEOS_DIR / video_id / "meta.json"
        meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")

        _save_progress(video_id, "build_memory")

    # ═══ Step 17: Indexer ═══
    if start_step <= 16:
        logger.info("=" * 50)
        logger.info(f"步骤 17/{total_steps}: 构建九维检索索引")
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
    scenes_count = len(memory.shots or memory.scenes)
    logger.info(f"   镜头数: {scenes_count}")
    logger.info(f"   台词数: {len(memory.transcripts)}")
    logger.info(f"   人物数: {len(memory.characters)}")
    logger.info(f"   事件数: {len(memory.events)}")
    logger.info(f"   Beat 数: {len(memory.beats)}")
    logger.info(f"   StoryScene 数: {len(memory.story_scenes)}")
    logger.info(f"   Chapter 数: {len(memory.chapters)}")
    logger.info(f"   EditSignal 数: {len(memory.edit_signals)}")
    logger.info(f"   NarrativeSignal 数: {len(memory.narrative_signals)}")
    logger.info(f"   RecompositionSignal 数: {len(memory.recomposition_signals)}")
    logger.info(f"   MemoryUnit 数: {len(memory.memory_units)}")
    logger.info(f"   BeatMemoryUnit 数: {len(memory.beat_memory_units)}")
    logger.info(f"   SceneMemoryUnit 数: {len(memory.scene_memory_units)}")
    logger.info(f"   ChapterMemoryUnit 数: {len(memory.chapter_memory_units)}")
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


def _load_shots(video_id: str) -> list[Shot]:
    """加载 shots (兼容旧 scenes.json)"""
    scenes_json = config.VIDEOS_DIR / video_id / "scenes" / "scenes.json"
    if scenes_json.exists():
        data = json.loads(scenes_json.read_text(encoding="utf-8"))
        return [Shot(**s) for s in data]
    return []


def _load_cached(data_type: str, video_id: str):
    """加载已缓存的中间结果，反序列化为 Pydantic 模型对象"""
    from models.schemas import (
        TranscriptSegment, OCRResult, VisionSummary, Character, CharacterDeep,
        Event, Beat, StoryScene, EditSignal, CharacterRelation,
        AudioProsody, MultimodalAlignment, Chapter,
        NarrativeSignal, RecompositionSignal,
    )
    model_map = {
        "transcripts": TranscriptSegment,
        "ocr": OCRResult,
        "vision": VisionSummary,
        "characters": CharacterDeep,  # v2 默认用 CharacterDeep
        "events": Event,
        "beats": Beat,
        "story_scenes": StoryScene,
        "edit_signals": EditSignal,
        "character_relations": CharacterRelation,
        "audio_prosody": AudioProsody,
        "multimodal_alignments": MultimodalAlignment,
        "chapters": Chapter,
        "narrative_signals": NarrativeSignal,
        "recomposition_signals": RecompositionSignal,
    }
    file_map = {
        "transcripts": "transcripts.json",
        "ocr": "ocr.json",
        "vision": "vision.json",
        "characters": "characters.json",
        "events": "events.json",
        "beats": "beats.json",
        "story_scenes": "story_scenes.json",
        "edit_signals": "edit_signals.json",
        "character_relations": "character_relations.json",
        "audio_prosody": "audio_prosody.json",
        "multimodal_alignments": "multimodal_alignments.json",
        "chapters": "chapters.json",
        "narrative_signals": "narrative_signals.json",
        "recomposition_signals": "recomposition_signals.json",
    }
    video_dir = config.VIDEOS_DIR / video_id
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


def _load_event_graph(video_id: str):
    """加载 event_graph.json"""
    from models.schemas import EventGraph
    path = config.VIDEOS_DIR / video_id / "event_graph.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return EventGraph(**data)
    return None
