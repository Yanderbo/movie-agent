# -*- coding: utf-8 -*-
"""
视频理解主流程（v4.1 — 分钟级融合理解 + 人脸聚类优先）

层级结构: Shot → Beat → StoryScene → Chapter → EventGraph
流程（10步）:
  ingest(+压缩) → shot_detect → keyframe → face_cluster
  → minute_chunk_understand(ASR+Vision+Audio+角色)
  → beat_detect → story_scene_detect → chapter_detect
  → event_graph_and_arc → edit_signal + build_memory + indexer

v4.1 核心变更:
- 入库时自动压缩视频（>480p→480p, >10fps→10fps）
- 人脸聚类前置（Step 4），构建角色脸谱
- MinuteChunk 融合理解（Step 5），替代原 ASR+Vision+Audio+Character+SpeakerBind+MultimodalAlign
- 动态角色档案：随 chunk 处理逐步更新
- Gemini API 调用量从 ~257 次降至 ~30 次（↓87%）
"""
import json

import config
from models.schemas import Shot
from memory.store import load_meta, save_memory, load_memory
from utils.logger import get_logger

logger = get_logger("Understand")

# 理解流程步骤定义（v4.1）
STEPS = [
    "ingest",              # 1.  入库 + 压缩
    "shot_detect",         # 2.  镜头切分
    "multi_keyframe",      # 3.  多帧关键帧采样
    "face_cluster",        # 4.  人脸聚类 + 角色脸谱 🆕
    "minute_chunk",        # 5.  分钟级融合理解 🆕 (替代 ASR+vision+audio+char+speaker+align)
    "beat_detect",         # 6.  剧情节拍检测
    "story_scene_detect",  # 7.  故事场景检测
    "chapter_detect",      # 8.  长视频大段落检测
    "event_and_arc",       # 9.  事件图谱 + 人物弧线 (合并)
    "final_build",         # 10. 信号计算 + Memory构建 + 索引 (合并)
]

# 旧步骤名 → 新步骤名映射（向后兼容 progress.json）
# 旧的子步骤映射到合并步骤的前置步骤，确保合并步骤一定会重新执行
_STEP_ALIASES = {
    "scene_detect": "shot_detect",
    "keyframe_extract": "multi_keyframe",
    # 旧 ASR/Vision/Audio/Speaker 子步骤 → 退到 multi_keyframe，确保 face_cluster + minute_chunk 都重跑
    "asr": "multi_keyframe",
    "asr_windowed": "multi_keyframe",
    "vision": "multi_keyframe",
    "audio_analysis": "multi_keyframe",
    "speaker_bind": "multi_keyframe",
    "multimodal_align": "multi_keyframe",
    # 旧 character 步骤语义不同于新 face_cluster → 退到 multi_keyframe
    "character_deep": "multi_keyframe",
    "character": "multi_keyframe",
    # 旧 event/arc 子步骤 → 退到 chapter_detect，确保 event_and_arc 重跑
    "event_graph": "chapter_detect",
    "event": "chapter_detect",
    "character_arc": "chapter_detect",
    # 旧 final 子步骤 → 退到 event_and_arc，确保 final_build 重跑
    "edit_signal": "event_and_arc",
    "build_memory": "event_and_arc",
    "indexer": "event_and_arc",
}


def run_understand(
    video_path: str = None,
    video_id: str = None,
    resume: bool = False,
    until_step: int | str = None,
) -> str:
    """
    运行视频理解全流程（v4.1）。

    Args:
        video_path: 视频文件路径（新视频）
        video_id: 视频 ID（用于 resume）
        resume: 是否从断点继续
        until_step: 调试限制。完成指定步骤后停止；可传 1-10 或 STEPS 中的步骤名。

    Returns:
        video_id
    """
    config.init_dirs()
    total_steps = len(STEPS)
    until_step_index = _resolve_until_step(until_step)
    if until_step_index is not None:
        logger.info(
            f"调试限制: 将在步骤 {until_step_index + 1}/{total_steps} "
            f"{STEPS[until_step_index]} 完成后停止"
        )

    # 确定从哪一步开始
    start_step = 0
    meta = None

    if resume and video_id:
        progress = _load_progress(video_id)
        completed = progress.get("completed_steps", [])
        completed = [_STEP_ALIASES.get(s, s) for s in completed]
        meta = load_meta(video_id)
        for i, step in enumerate(STEPS):
            if step not in completed:
                start_step = i
                break
        else:
            if until_step_index is not None and until_step_index < len(STEPS) - 1:
                logger.info(
                    f"调试限制已满足: 已完成到步骤 {until_step_index + 1}/{total_steps} "
                    f"{STEPS[until_step_index]}"
                )
                return video_id
            # 验证关键产物是否存在
            video_dir = config.VIDEOS_DIR / video_id
            critical_paths = [
                video_dir / "memory.json",
                video_dir / "index" / "search_index.json",
            ]
            missing = [
                str(p.relative_to(video_dir)) for p in critical_paths
                if not p.exists()
            ]
            if missing:
                logger.warning(f"进度标记完成但缺少关键产物: {missing}，从 final_build 重跑")
                start_step = STEPS.index("final_build")
            else:
                logger.info(f"视频 {video_id} 所有步骤已完成")
                return video_id
        if until_step_index is not None and start_step > until_step_index:
            logger.info(
                f"调试限制已满足: 下一步为 {start_step + 1}/{total_steps} "
                f"{STEPS[start_step]}，已超过限制步骤 "
                f"{until_step_index + 1}/{total_steps} {STEPS[until_step_index]}"
            )
            return video_id
        logger.info(f"从步骤 {STEPS[start_step]} 继续 (已完成: {completed})")
    elif not video_path:
        raise ValueError("必须提供 --video 或 --video-id + --resume")

    # ═══ Step 1: Ingest（入库 + 压缩）═══
    if start_step <= 0:
        logger.info("=" * 50)
        logger.info(f"步骤 1/{total_steps}: 视频入库 + 压缩")
        logger.info("=" * 50)
        from pipeline.ingest import ingest_video
        meta = ingest_video(video_path, video_id)
        video_id = meta.video_id
        _save_progress(video_id, "ingest")
        if _should_stop_after(0, until_step_index):
            _log_until_step_stop(video_id, 0, total_steps)
            return video_id

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
        if _should_stop_after(1, until_step_index):
            _log_until_step_stop(video_id, 1, total_steps)
            return video_id
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
        if _should_stop_after(2, until_step_index):
            _log_until_step_stop(video_id, 2, total_steps)
            return video_id

    # ═══ Step 4: Face Cluster（人脸聚类 + 角色脸谱）═══
    if start_step <= 3:
        logger.info("=" * 50)
        logger.info(f"步骤 4/{total_steps}: 人脸聚类 + 角色脸谱构建")
        logger.info("=" * 50)
        from pipeline.face_cluster import cluster_faces
        galleries = cluster_faces(video_id, shots)
        _save_progress(video_id, "face_cluster")
        if _should_stop_after(3, until_step_index):
            _log_until_step_stop(video_id, 3, total_steps)
            return video_id
    else:
        galleries = _load_galleries(video_id)

    # ═══ Step 5: MinuteChunk Understand（分钟级融合理解）⭐ ═══
    if start_step <= 4:
        logger.info("=" * 50)
        logger.info(f"步骤 5/{total_steps}: 分钟级融合理解（ASR+Vision+Audio+角色）")
        logger.info("=" * 50)
        from pipeline.minute_chunk import run_minute_chunk_understand
        chunk_outputs = run_minute_chunk_understand(
            video_id, meta.storage_path, shots, galleries,
        )
        transcripts = chunk_outputs["transcripts"]
        ocr_results = chunk_outputs["ocr_results"]
        vision_summaries = chunk_outputs["vision_summaries"]
        audio_prosodies = chunk_outputs["audio_prosodies"]
        alignments = chunk_outputs["alignments"]
        characters = _load_cached("characters", video_id)
        speaker_map = chunk_outputs["speaker_map"]
        minute_chunks = chunk_outputs.get("chunks", [])
        _save_progress(video_id, "minute_chunk")
        if _should_stop_after(4, until_step_index):
            _log_until_step_stop(video_id, 4, total_steps)
            return video_id
    else:
        transcripts = _load_cached("transcripts", video_id)
        ocr_results = _load_cached("ocr", video_id)
        vision_summaries = _load_cached("vision", video_id)
        audio_prosodies = _load_cached("audio_prosody", video_id)
        alignments = _load_cached("multimodal_alignments", video_id)
        characters = _load_cached("characters", video_id)
        speaker_map = _load_speaker_map(video_id)
        minute_chunks = _load_cached("minute_chunks", video_id)

    # ═══ Step 6: Beat Detect ═══
    if start_step <= 5:
        logger.info("=" * 50)
        logger.info(f"步骤 6/{total_steps}: 剧情节拍检测")
        logger.info("=" * 50)
        from pipeline.beat_detect import detect_beats
        beats = detect_beats(
            video_id, shots, transcripts, vision_summaries,
            characters, minute_chunks,
        )
        _save_progress(video_id, "beat_detect")
        if _should_stop_after(5, until_step_index):
            _log_until_step_stop(video_id, 5, total_steps)
            return video_id
    else:
        # 缓存分支同样走幂等的 detect_beats：命中 beats.json 时不调用 LLM，
        # 但会回填 shot.beat_index，保持与新建分支一致的反向链接。
        from pipeline.beat_detect import detect_beats
        beats = detect_beats(
            video_id, shots, transcripts, vision_summaries,
            characters, minute_chunks,
        )

    # ═══ Step 7: Story Scene Detect ═══
    if start_step <= 6:
        logger.info("=" * 50)
        logger.info(f"步骤 7/{total_steps}: 故事场景检测")
        logger.info("=" * 50)
        from pipeline.story_scene_detect import detect_story_scenes
        story_scenes = detect_story_scenes(video_id, shots, beats)
        _save_progress(video_id, "story_scene_detect")
        if _should_stop_after(6, until_step_index):
            _log_until_step_stop(video_id, 6, total_steps)
            return video_id
    else:
        story_scenes = _load_cached("story_scenes", video_id)

    # ═══ Step 8: Chapter Detect ═══
    if start_step <= 7:
        logger.info("=" * 50)
        logger.info(f"步骤 8/{total_steps}: 长视频大段落检测")
        logger.info("=" * 50)
        from pipeline.chapter_detect import detect_chapters
        chapters = detect_chapters(
            video_id, story_scenes, beats, shots, meta.duration,
        )
        _save_progress(video_id, "chapter_detect")
        if _should_stop_after(7, until_step_index):
            _log_until_step_stop(video_id, 7, total_steps)
            return video_id
    else:
        chapters = _load_cached("chapters", video_id)

    # ═══ Step 9: Event Graph + Character Arc（合并）═══
    if start_step <= 8:
        logger.info("=" * 50)
        logger.info(f"步骤 9/{total_steps}: 事件图谱 + 人物弧线")
        logger.info("=" * 50)
        from pipeline.event import extract_events
        events, event_graph = extract_events(
            video_id, shots, transcripts, vision_summaries, characters,
            meta.duration, beats=beats, story_scenes=story_scenes,
        )
        if events is None:
            raise RuntimeError("事件抽取处理失败，返回 None")

        from pipeline.character_arc import analyze_character_arcs
        characters, character_relations = analyze_character_arcs(
            video_id, characters, events, beats, transcripts, meta.duration
        )
        _save_progress(video_id, "event_and_arc")
        if _should_stop_after(8, until_step_index):
            _log_until_step_stop(video_id, 8, total_steps)
            return video_id
    else:
        events = _load_cached("events", video_id)
        event_graph = _load_event_graph(video_id)
        character_relations = _load_cached("character_relations", video_id)

    # ═══ Step 10: EditSignal + Build Memory + Indexer（合并）═══
    if start_step <= 9:
        logger.info("=" * 50)
        logger.info(f"步骤 10/{total_steps}: 信号计算 + Memory构建 + 索引")
        logger.info("=" * 50)

        # 10a: 信号计算
        from pipeline.edit_signal import compute_edit_signals
        edit_signals, narrative_signals, recomposition_signals = compute_edit_signals(
            video_id, shots, beats, story_scenes, events,
            characters, transcripts, vision_summaries,
        )

        # 10b: 构建 VideoMemory
        from models.schemas import VideoMemory, EventGraph as EG
        from pipeline.memory_builder import (
            build_memory_units, build_beat_memory_units,
            build_scene_memory_units, build_chapter_memory_units,
            assign_character_roles,
        )

        memory_units = build_memory_units(
            shots, transcripts, ocr_results, vision_summaries,
            characters, events, beats, story_scenes, edit_signals,
            chapters=chapters,
            audio_prosodies=audio_prosodies,
            alignments=alignments,
        )
        beat_memory_units = build_beat_memory_units(
            beats, shots, transcripts, vision_summaries, edit_signals,
        )
        scene_memory_units = build_scene_memory_units(
            story_scenes, beats, edit_signals,
        )
        chapter_memory_units = build_chapter_memory_units(
            chapters, story_scenes, edit_signals, narrative_signals,
        )
        characters = assign_character_roles(characters, transcripts, events, meta.duration)

        if not isinstance(event_graph, EG):
            event_graph = _load_event_graph(video_id)

        memory = VideoMemory(
            video_id=video_id,
            meta=meta,
            shots=shots,
            scenes=shots,
            transcripts=transcripts if isinstance(transcripts, list) else [],
            ocr_results=ocr_results if isinstance(ocr_results, list) else [],
            vision_summaries=vision_summaries if isinstance(vision_summaries, list) else [],
            audio_prosodies=audio_prosodies if isinstance(audio_prosodies, list) else [],
            multimodal_alignments=alignments if isinstance(alignments, list) else [],
            characters=characters if isinstance(characters, list) else [],
            characters_deep=characters if isinstance(characters, list) else [],
            character_relations=character_relations if isinstance(character_relations, list) else [],
            speaker_map=speaker_map if isinstance(speaker_map, dict) else {},
            beats=beats if isinstance(beats, list) else [],
            story_scenes=story_scenes if isinstance(story_scenes, list) else [],
            chapters=chapters if isinstance(chapters, list) else [],
            event_graph=event_graph,
            events=events if isinstance(events, list) else [],
            edit_signals=edit_signals if isinstance(edit_signals, list) else [],
            narrative_signals=narrative_signals if isinstance(narrative_signals, list) else [],
            recomposition_signals=recomposition_signals if isinstance(recomposition_signals, list) else [],
            memory_units=memory_units,
            beat_memory_units=beat_memory_units,
            scene_memory_units=scene_memory_units,
            chapter_memory_units=chapter_memory_units,
        )
        save_memory(memory)

        meta.status = "ready"
        meta_path = config.VIDEOS_DIR / video_id / "meta.json"
        meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")

        # 10c: 构建索引
        try:
            from pipeline.indexer import build_search_index
            build_search_index(video_id)
        except Exception as e:
            logger.error(f"索引构建失败: {e}")
            raise RuntimeError(f"索引构建失败: {e}") from e

        _save_progress(video_id, "final_build")

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
    logger.info(f"   MemoryUnit 数: {len(memory.memory_units)}")
    logger.info("=" * 50)

    return video_id


# ─── 进度管理 ──────────────────────────────────────────────

def _resolve_until_step(until_step) -> int | None:
    """将调试停止步骤解析为 0-based STEPS 索引。"""
    if until_step is None:
        return None

    if isinstance(until_step, str):
        raw = until_step.strip()
        if not raw:
            return None
        if raw in STEPS:
            return STEPS.index(raw)
        try:
            value = int(raw)
        except ValueError as e:
            raise ValueError(
                "--until-step 必须是 1-10 或步骤名: "
                + ", ".join(STEPS)
            ) from e
    else:
        value = int(until_step)

    if value < 1 or value > len(STEPS):
        raise ValueError(
            f"--until-step 必须在 1-{len(STEPS)} 之间，或使用步骤名: "
            + ", ".join(STEPS)
        )
    return value - 1


def _should_stop_after(step_index: int, until_step_index: int | None) -> bool:
    return (
        until_step_index is not None
        and step_index >= until_step_index
        and step_index < len(STEPS) - 1
    )


def _log_until_step_stop(video_id: str, step_index: int, total_steps: int):
    logger.info("=" * 50)
    logger.info(
        f"调试限制触发: 已完成步骤 {step_index + 1}/{total_steps} "
        f"{STEPS[step_index]}，停止 understand。video_id={video_id}"
    )
    logger.info("=" * 50)


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
    """加载 shots (兼容旧 scenes.json)，并从 beats/story_scenes 重建反向链接"""
    scenes_json = config.VIDEOS_DIR / video_id / "scenes" / "scenes.json"
    if not scenes_json.exists():
        return []
    data = json.loads(scenes_json.read_text(encoding="utf-8"))
    shots = [Shot(**s) for s in data]

    # 从 beats.json / story_scenes.json 重建 beat_index / story_scene_index
    shot_map = {s.scene_index: s for s in shots}

    beats_path = config.VIDEOS_DIR / video_id / "beats.json"
    if beats_path.exists():
        for b in json.loads(beats_path.read_text(encoding="utf-8")):
            for si in b.get("shot_indices", []):
                if si in shot_map:
                    shot_map[si].beat_index = b.get("beat_index")

    ss_path = config.VIDEOS_DIR / video_id / "story_scenes.json"
    if ss_path.exists():
        for ss in json.loads(ss_path.read_text(encoding="utf-8")):
            for si in ss.get("shot_indices", []):
                if si in shot_map:
                    shot_map[si].story_scene_index = ss.get("story_scene_index")

    return shots


def _load_galleries(video_id: str):
    """加载角色脸谱"""
    from models.schemas import CharacterGallery
    path = config.VIDEOS_DIR / video_id / "characters" / "face_clusters.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return [CharacterGallery(**g) for g in data]
    return []


def _load_cached(data_type: str, video_id: str):
    """加载已缓存的中间结果，反序列化为 Pydantic 模型对象"""
    from models.schemas import (
        TranscriptSegment, OCRResult, VisionSummary, CharacterDeep,
        Event, Beat, StoryScene, EditSignal, CharacterRelation,
        AudioProsody, MultimodalAlignment, Chapter,
        NarrativeSignal, RecompositionSignal, MinuteChunk,
    )
    model_map = {
        "transcripts": TranscriptSegment,
        "ocr": OCRResult,
        "vision": VisionSummary,
        "characters": CharacterDeep,
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
        "minute_chunks": MinuteChunk,
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
        "minute_chunks": "minute_chunks.json",
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
