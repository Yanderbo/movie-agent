# -*- coding: utf-8 -*-
"""
多模态对齐（v3 新增）

对齐 shot、ASR、speaker、visible character、vision、audio 数据。
输出 MultimodalAlignment 记录每个 shot 的跨模态一致性。

策略：纯规则 + 轻量验证，不依赖 LLM，高效稳定。
降级策略：数据缺失时 confidence=0，不阻塞。
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

import config
from models.schemas import (
    Shot, TranscriptSegment, VisionSummary, Character,
    AudioProsody, MultimodalAlignment,
)
from utils.logger import get_logger

logger = get_logger("MultimodalAlign")


def align_multimodal(
    video_id: str,
    shots: list[Shot],
    transcripts: list[TranscriptSegment],
    vision_summaries: list[VisionSummary],
    characters: list[Character],
    speaker_map: dict,
    audio_prosodies: list[AudioProsody] = None,
) -> list[MultimodalAlignment]:
    """
    对齐每个 shot 的多模态数据。

    纯规则计算：
    1. 通过 speaker_map 建立 speaker → character 映射
    2. 通过 appearance_scenes 确定 visible_characters
    3. 交叉验证：说话者是否在画面中可见
    4. 确定 active_modalities 和 dominant_modality

    Args:
        video_id: 视频 ID
        shots: 镜头列表
        transcripts: 台词列表
        vision_summaries: 画面摘要列表
        characters: 人物列表
        speaker_map: speaker_id → character_id 映射
        audio_prosodies: 音频韵律列表（可选）

    Returns:
        MultimodalAlignment 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    align_path = video_dir / "multimodal_alignments.json"

    # 如果已存在，直接加载
    if align_path.exists():
        logger.info(f"多模态对齐结果已存在，直接加载: {align_path}")
        data = json.loads(align_path.read_text(encoding="utf-8"))
        return [MultimodalAlignment(**a) for a in data]

    if not shots:
        return []

    logger.info(f"开始多模态对齐: {len(shots)} 个 shot")

    # 构建索引
    trans_by_scene = defaultdict(list)
    for t in transcripts:
        if t.scene_index >= 0:
            trans_by_scene[t.scene_index].append(t)

    vision_by_scene = {v.scene_index: v for v in vision_summaries}

    char_by_scene = defaultdict(list)
    for c in characters:
        for si in c.appearance_scenes:
            char_by_scene[si].append(c.character_id)

    audio_by_scene = {}
    if audio_prosodies:
        for a in audio_prosodies:
            audio_by_scene[a.scene_index] = a

    alignments = []
    for shot in shots:
        si = shot.scene_index
        alignment = _align_single_shot(
            shot, trans_by_scene.get(si, []),
            vision_by_scene.get(si), char_by_scene.get(si, []),
            audio_by_scene.get(si), speaker_map,
        )
        alignments.append(alignment)

    # 保存
    align_path.write_text(
        json.dumps([a.model_dump() for a in alignments], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"多模态对齐完成: {len(alignments)} 个 shot")

    return alignments


def _align_single_shot(
    shot: Shot,
    transcripts: list[TranscriptSegment],
    vision: VisionSummary | None,
    visible_chars: list[str],
    audio: AudioProsody | None,
    speaker_map: dict,
) -> MultimodalAlignment:
    """对齐单个 shot 的多模态数据"""

    si = shot.scene_index

    # 1. speaker → character 映射
    s2c = {}
    speaking_chars = []
    for t in transcripts:
        if t.speaker and t.speaker in speaker_map:
            cid = speaker_map[t.speaker]
            s2c[t.speaker] = cid
            if cid not in speaking_chars:
                speaking_chars.append(cid)
        elif t.character_id:
            if t.character_id not in speaking_chars:
                speaking_chars.append(t.character_id)

    # 2. 确定 active modalities
    active = []
    if transcripts:
        active.append("speech")
    if vision and vision.action_description:
        active.append("visual_action")
    if vision and vision.scene_type:
        pass  # 所有有画面的 shot 都有视觉
    if audio:
        if audio.has_music:
            active.append("music")
        if audio.has_sfx:
            active.append("sfx")
        if audio.silence_ratio > 0.8:
            active.append("silence")

    # 3. 确定 dominant modality
    dominant = ""
    if audio and audio.silence_ratio > 0.8:
        dominant = "silence"
    elif transcripts and len(transcripts) >= 2:
        dominant = "speech"
    elif audio and audio.has_music and not transcripts:
        dominant = "music"
    elif vision and vision.action_description:
        dominant = "visual_action"
    elif transcripts:
        dominant = "speech"
    else:
        dominant = "visual"

    # 4. 计算对齐置信度
    confidence = 0.5  # 基准
    # 有 speaker_map 且说话者在画面中 → 高置信
    for sc in speaking_chars:
        if sc in visible_chars:
            confidence = min(confidence + 0.15, 1.0)
    # 有音频分析结果 → +0.1
    if audio:
        confidence = min(confidence + 0.1, 1.0)
    # 有画面摘要 → +0.1
    if vision:
        confidence = min(confidence + 0.1, 1.0)
    # 有台词 → +0.1
    if transcripts:
        confidence = min(confidence + 0.1, 1.0)

    # 5. 冲突检测
    notes = ""
    invisible_speakers = [c for c in speaking_chars if c not in visible_chars]
    if invisible_speakers:
        notes = f"说话者不在画面中: {','.join(invisible_speakers)}（可能是画外音/旁白）"

    return MultimodalAlignment(
        scene_index=si,
        start_time=shot.start_time,
        end_time=shot.end_time,
        speaker_to_character=s2c,
        visible_characters=visible_chars,
        speaking_characters=speaking_chars,
        active_modalities=active,
        dominant_modality=dominant,
        alignment_confidence=round(confidence, 2),
        notes=notes,
    )
