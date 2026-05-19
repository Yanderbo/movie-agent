# -*- coding: utf-8 -*-
"""
多层 MemoryUnit 构建 & 角色判定（v2）

v2 变更:
- 从单一 shot-level MemoryUnit 升级为三层: shot / beat / story_scene
- MemoryUnit 携带 edit_signal 引用
- combined_text 包含剪辑信号摘要
- 角色判定增加 importance_score 参考
"""
import json
from collections import defaultdict

import config
from models.schemas import (
    Shot, Beat, StoryScene, Chapter, TranscriptSegment, OCRResult, VisionSummary,
    Character, CharacterDeep, Event, EventGraph, EditSignal, NarrativeSignal,
    AudioProsody, MultimodalAlignment,
    MemoryUnit, BeatMemoryUnit, SceneMemoryUnit, ChapterMemoryUnit,
)
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("MemoryBuilder")

ROLE_PROMPT_TEMPLATE = """你是一个专业的影视分析师。请根据以下人物信息，判断每个人物在故事中的角色。

=== 人物列表 ===
{characters_info}

可选角色：
- male_lead: 男主角
- female_lead: 女主角
- villain: 反派
- supporting: 配角
- minor: 路人/群演

请为每个人物指定一个角色。考虑出镜时长、台词量、事件参与度等因素。
出镜最多且参与最多关键事件的通常是主角。

输出 JSON 对象，key 为 character_id，value 为角色：
```json
{{
  "char_000": "male_lead",
  "char_001": "female_lead",
  "char_002": "villain"
}}
```
"""


def build_memory_units(
    shots: list[Shot],
    transcripts: list[TranscriptSegment],
    ocr_results: list[OCRResult],
    vision_summaries: list[VisionSummary],
    characters: list[Character],
    events: list[Event],
    beats: list[Beat] = None,
    story_scenes: list[StoryScene] = None,
    edit_signals: list[EditSignal] = None,
    chapters: list[Chapter] = None,
    audio_prosodies: list[AudioProsody] = None,
    alignments: list[MultimodalAlignment] = None,
) -> list[MemoryUnit]:
    """
    将各模态数据按 shot 融合为 MemoryUnit 列表。

    v2: 同时关联 beat_index / story_scene_index / edit_signal。

    Args:
        shots: 镜头列表
        transcripts: 台词列表（已带 scene_index）
        ocr_results: OCR 结果列表
        vision_summaries: 画面摘要列表
        characters: 人物列表
        events: 事件列表
        beats: Beat 列表（可选）
        story_scenes: StoryScene 列表（可选）
        edit_signals: EditSignal 列表（可选）

    Returns:
        MemoryUnit 列表
    """
    # 按 scene_index 索引各模态数据
    trans_by_scene = defaultdict(list)
    for t in transcripts:
        if t.scene_index >= 0:
            trans_by_scene[t.scene_index].append(t)
        else:
            # 旧数据可能没有 scene_index，用时间反查
            for s in shots:
                if s.start_time <= t.start_time < s.end_time:
                    trans_by_scene[s.scene_index].append(t)
                    break

    ocr_by_scene = {}
    for o in ocr_results:
        ocr_by_scene[o.scene_index] = o

    vision_by_scene = {}
    for v in vision_summaries:
        vision_by_scene[v.scene_index] = v

    char_by_scene = defaultdict(list)
    for c in characters:
        for si in c.appearance_scenes:
            char_by_scene[si].append(c.character_id)

    # 事件与 scene 的关联：事件时间段与 scene 时间段有重叠
    events_by_scene = defaultdict(list)
    for e in events:
        for s in shots:
            if e.start_time < s.end_time and e.end_time > s.start_time:
                events_by_scene[s.scene_index].append(e)
                # 同时更新 event 的 scene_indices
                if s.scene_index not in e.scene_indices:
                    e.scene_indices.append(s.scene_index)

    # EditSignal 索引
    signal_by_shot = {}
    if edit_signals:
        for sig in edit_signals:
            if sig.unit_type == "shot":
                signal_by_shot[sig.unit_index] = sig

    # AudioProsody 索引
    audio_by_scene = {}
    if audio_prosodies:
        for a in audio_prosodies:
            audio_by_scene[a.scene_index] = a

    # MultimodalAlignment 索引
    align_by_scene = {}
    if alignments:
        for al in alignments:
            align_by_scene[al.scene_index] = al

    # Chapter 索引（shot -> chapter_index）
    chapter_by_shot = {}
    if chapters:
        for ch in chapters:
            for si in ch.shot_indices:
                chapter_by_shot[si] = ch.chapter_index

    # 构建 MemoryUnit
    memory_units = []
    for shot in shots:
        si = shot.scene_index

        scene_trans = trans_by_scene.get(si, [])
        scene_vision = vision_by_scene.get(si)
        scene_ocr = ocr_by_scene.get(si)
        scene_chars = char_by_scene.get(si, [])
        scene_events = events_by_scene.get(si, [])

        # 构建 combined_text
        text_parts = []

        # 台词
        if scene_trans:
            trans_text = " ".join([t.text for t in scene_trans])
            text_parts.append(f"台词: {trans_text}")

        # 画面描述
        if scene_vision:
            text_parts.append(f"画面: {scene_vision.description}")
            if scene_vision.mood:
                text_parts.append(f"情绪: {scene_vision.mood}")
            if scene_vision.scene_type:
                text_parts.append(f"类型: {scene_vision.scene_type}")
            if scene_vision.objects:
                text_parts.append(f"物体: {', '.join(scene_vision.objects)}")
            if scene_vision.action_description:
                text_parts.append(f"动作: {scene_vision.action_description}")

        # OCR
        if scene_ocr and scene_ocr.texts:
            text_parts.append(f"文字: {', '.join(scene_ocr.texts)}")

        # 事件
        for e in scene_events:
            text_parts.append(f"事件[{e.event_type}]: {e.description}")

        # 人物
        if scene_chars:
            text_parts.append(f"人物: {', '.join(scene_chars)}")

        combined_text = " | ".join(text_parts)

        signal = signal_by_shot.get(si)
        audio = audio_by_scene.get(si)
        align = align_by_scene.get(si)

        # 音频信息补充到 combined_text
        if audio:
            audio_parts = []
            if audio.has_music:
                audio_parts.append(f"音乐:{audio.music_mood}")
            if audio.has_sfx:
                audio_parts.append(f"音效:{','.join(audio.sfx_tags[:3])}")
            if audio.speech_emotion:
                audio_parts.append(f"语音情绪:{audio.speech_emotion}")
            if audio_parts:
                combined_text += " | 音频: " + ", ".join(audio_parts)

        unit = MemoryUnit(
            scene_index=si,
            start_time=shot.start_time,
            end_time=shot.end_time,
            duration=shot.duration,
            keyframe_path=shot.keyframe_path,
            transcripts=scene_trans,
            vision=scene_vision,
            ocr=scene_ocr,
            characters=scene_chars,
            events=scene_events,
            combined_text=combined_text,
            embedding=[],  # 在 indexer 步骤中填充
            beat_index=shot.beat_index,
            story_scene_index=shot.story_scene_index,
            edit_signal=signal,
            chapter_index=chapter_by_shot.get(si),
            audio_prosody=audio,
            alignment=align,
        )
        memory_units.append(unit)

    logger.info(f"Shot MemoryUnit 构建完成: {len(memory_units)} 个单元")
    return memory_units


def build_beat_memory_units(
    beats: list[Beat],
    shots: list[Shot],
    transcripts: list[TranscriptSegment],
    vision_summaries: list[VisionSummary],
    edit_signals: list[EditSignal] = None,
) -> list[BeatMemoryUnit]:
    """构建 Beat 级别的 MemoryUnit"""
    if not beats:
        return []

    trans_by_scene = defaultdict(list)
    for t in transcripts:
        if t.scene_index >= 0:
            trans_by_scene[t.scene_index].append(t)
    vision_by_scene = {v.scene_index: v for v in vision_summaries}

    signal_by_beat = {}
    if edit_signals:
        for sig in edit_signals:
            if sig.unit_type == "beat":
                signal_by_beat[sig.unit_index] = sig

    beat_units = []
    for beat in beats:
        # 聚合台词
        all_trans = []
        for si in beat.shot_indices:
            all_trans.extend(trans_by_scene.get(si, []))
        trans_summary = " ".join([t.text for t in all_trans[:10]])

        # 聚合画面
        vision_parts = []
        for si in beat.shot_indices:
            v = vision_by_scene.get(si)
            if v:
                vision_parts.append(v.description[:50])

        # combined_text
        parts = []
        if beat.description:
            parts.append(f"节拍: {beat.description}")
        if beat.beat_type:
            parts.append(f"类型: {beat.beat_type}")
        if trans_summary:
            parts.append(f"台词: {trans_summary[:200]}")
        if vision_parts:
            parts.append(f"画面: {'; '.join(vision_parts[:5])}")
        if beat.emotion:
            parts.append(f"情绪: {beat.emotion}")
        if beat.characters:
            parts.append(f"人物: {', '.join(beat.characters)}")

        bmu = BeatMemoryUnit(
            beat_index=beat.beat_index,
            start_time=beat.start_time,
            end_time=beat.end_time,
            duration=beat.duration,
            shot_indices=beat.shot_indices,
            beat_type=beat.beat_type,
            description=beat.description,
            emotion=beat.emotion,
            intensity=beat.intensity,
            characters=beat.characters,
            transcript_summary=trans_summary[:500],
            combined_text=" | ".join(parts),
            edit_signal=signal_by_beat.get(beat.beat_index),
        )
        beat_units.append(bmu)

    logger.info(f"Beat MemoryUnit 构建完成: {len(beat_units)} 个单元")
    return beat_units


def build_scene_memory_units(
    story_scenes: list[StoryScene],
    beats: list[Beat],
    edit_signals: list[EditSignal] = None,
) -> list[SceneMemoryUnit]:
    """构建 StoryScene 级别的 MemoryUnit"""
    if not story_scenes:
        return []

    signal_by_scene = {}
    if edit_signals:
        for sig in edit_signals:
            if sig.unit_type == "story_scene":
                signal_by_scene[sig.unit_index] = sig

    beat_map = {b.beat_index: b for b in beats} if beats else {}

    scene_units = []
    for ss in story_scenes:
        # 聚合 beat 信息
        beat_descs = []
        all_chars = set()
        for bi in ss.beat_indices:
            b = beat_map.get(bi)
            if b:
                if b.description:
                    beat_descs.append(f"[{b.beat_type}] {b.description}")
                all_chars.update(b.characters)

        parts = []
        if ss.description:
            parts.append(f"场景: {ss.description}")
        if ss.location:
            parts.append(f"地点: {ss.location}")
        if ss.plot_function:
            parts.append(f"功能: {ss.plot_function}")
        if beat_descs:
            parts.append(f"节拍: {'; '.join(beat_descs[:5])}")
        if ss.characters:
            parts.append(f"人物: {', '.join(ss.characters)}")

        smu = SceneMemoryUnit(
            story_scene_index=ss.story_scene_index,
            start_time=ss.start_time,
            end_time=ss.end_time,
            duration=ss.duration,
            beat_indices=ss.beat_indices,
            shot_indices=ss.shot_indices,
            location=ss.location,
            description=ss.description,
            characters=ss.characters or sorted(all_chars),
            plot_function=ss.plot_function,
            combined_text=" | ".join(parts),
            edit_signal=signal_by_scene.get(ss.story_scene_index),
        )
        scene_units.append(smu)

    logger.info(f"StoryScene MemoryUnit 构建完成: {len(scene_units)} 个单元")
    return scene_units


def build_chapter_memory_units(
    chapters: list[Chapter],
    story_scenes: list[StoryScene],
    edit_signals: list[EditSignal] = None,
    narrative_signals: list[NarrativeSignal] = None,
) -> list[ChapterMemoryUnit]:
    """构建 Chapter 级别的 MemoryUnit（v3 新增）"""
    if not chapters:
        return []

    signal_by_ch = {}
    if edit_signals:
        for sig in edit_signals:
            if sig.unit_type == "chapter":
                signal_by_ch[sig.unit_index] = sig

    ns_by_ch = {}
    if narrative_signals:
        for ns in narrative_signals:
            # chapter 级别的 NarrativeSignal
            if ns.unit_type == "chapter":
                ns_by_ch[ns.unit_index] = ns

    scene_map = {ss.story_scene_index: ss for ss in story_scenes} if story_scenes else {}

    chapter_units = []
    for ch in chapters:
        scene_descs = []
        all_chars = set(ch.characters)
        for ssi in ch.story_scene_indices:
            ss = scene_map.get(ssi)
            if ss:
                if ss.description:
                    scene_descs.append(f"[{ss.plot_function}] {ss.description}")
                all_chars.update(ss.characters)

        parts = []
        if ch.title:
            parts.append(f"章节: {ch.title}")
        if ch.description:
            parts.append(f"内容: {ch.description}")
        if ch.theme:
            parts.append(f"主题: {ch.theme}")
        if ch.chapter_type:
            parts.append(f"类型: {ch.chapter_type}")
        if scene_descs:
            parts.append(f"场景: {'; '.join(scene_descs[:5])}")
        if ch.characters:
            parts.append(f"人物: {', '.join(ch.characters[:5])}")
        if ch.mood_progression:
            parts.append(f"情绪: {ch.mood_progression}")

        cmu = ChapterMemoryUnit(
            chapter_index=ch.chapter_index,
            start_time=ch.start_time,
            end_time=ch.end_time,
            duration=ch.duration,
            story_scene_indices=ch.story_scene_indices,
            title=ch.title,
            description=ch.description,
            theme=ch.theme,
            chapter_type=ch.chapter_type,
            characters=ch.characters or sorted(all_chars),
            mood_progression=ch.mood_progression,
            combined_text=" | ".join(parts),
            edit_signal=signal_by_ch.get(ch.chapter_index),
            narrative_signal=ns_by_ch.get(ch.chapter_index),
        )
        chapter_units.append(cmu)

    logger.info(f"Chapter MemoryUnit 构建完成: {len(chapter_units)} 个单元")
    return chapter_units


def assign_character_roles(
    characters: list[Character],
    transcripts: list[TranscriptSegment],
    events: list[Event],
    video_duration: float,
) -> list[Character]:
    """
    调用 LLM 判定每个 character 的业务角色。

    考虑因素：出镜时长、台词量、事件参与度。

    Args:
        characters: 人物列表
        transcripts: 台词列表
        events: 事件列表
        video_duration: 视频总时长

    Returns:
        更新了 role 字段的人物列表
    """
    if not characters:
        return characters

    # 统计每个 character 的台词量
    char_transcript_count = defaultdict(int)
    for t in transcripts:
        if t.character_id:
            char_transcript_count[t.character_id] += 1

    # 统计每个 character 参与的事件数
    char_event_count = defaultdict(int)
    char_important_event_count = defaultdict(int)
    for e in events:
        for cid in e.characters:
            char_event_count[cid] += 1
            if e.importance >= 7:
                char_important_event_count[cid] += 1

    # 构造 prompt
    char_lines = []
    for c in characters:
        screen_pct = (c.total_screen_time / video_duration * 100) if video_duration > 0 else 0
        importance = ""
        if hasattr(c, "importance_score"):
            importance = f"重要性: {c.importance_score:.2f}, "
        char_lines.append(
            f"- {c.character_id} ({c.display_name}): {c.description}\n"
            f"  出镜: {c.total_screen_time:.0f}s ({screen_pct:.1f}%), "
            f"{importance}"
            f"台词: {char_transcript_count.get(c.character_id, 0)}句, "
            f"事件: {char_event_count.get(c.character_id, 0)}个 "
            f"(重要事件: {char_important_event_count.get(c.character_id, 0)}个)"
        )
    characters_info = "\n".join(char_lines)

    prompt = ROLE_PROMPT_TEMPLATE.format(characters_info=characters_info)

    try:
        client = get_llm_client()
        response = client.chat(prompt=prompt, temperature=0.2)
        parsed = client.parse_json(response)
        if parsed and isinstance(parsed, dict):
            valid_roles = {"male_lead", "female_lead", "villain", "supporting", "minor"}
            for c in characters:
                role = parsed.get(c.character_id)
                if role and role in valid_roles:
                    c.role = role
                else:
                    c.role = "supporting"
            logger.info(f"角色判定完成: {parsed}")
        else:
            logger.warning("角色判定 LLM 结果解析失败，使用默认角色")
            _fallback_role_assignment(characters)
    except Exception as e:
        logger.warning(f"角色判定 LLM 调用失败: {e}，使用默认角色")
        _fallback_role_assignment(characters)

    return characters


def _fallback_role_assignment(characters: list[Character]):
    """当 LLM 失败时，按出镜时长排序分配角色"""
    if not characters:
        return

    sorted_chars = sorted(characters, key=lambda c: c.total_screen_time, reverse=True)
    for i, c in enumerate(sorted_chars):
        if i == 0:
            c.role = "male_lead"
        elif i == 1:
            c.role = "female_lead"
        else:
            c.role = "supporting"
