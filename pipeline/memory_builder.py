# -*- coding: utf-8 -*-
"""
MemoryUnit 构建 & 角色判定

- build_memory_units(): 将各模态数据按 scene_index 融合为 MemoryUnit 列表
- assign_character_roles(): 调用 LLM 判定业务角色（male_lead / female_lead / villain / ...）
"""
import json
from collections import defaultdict

import config
from models.schemas import (
    Scene, TranscriptSegment, OCRResult, VisionSummary,
    Character, Event, MemoryUnit,
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
    scenes: list[Scene],
    transcripts: list[TranscriptSegment],
    ocr_results: list[OCRResult],
    vision_summaries: list[VisionSummary],
    characters: list[Character],
    events: list[Event],
) -> list[MemoryUnit]:
    """
    将各模态数据按 scene_index 融合为 MemoryUnit 列表。

    每个 MemoryUnit 对应一个 scene，汇聚了该 shot 内的所有模态信息，
    并生成 combined_text 用于后续 embedding。

    Args:
        scenes: 镜头列表
        transcripts: 台词列表（已带 scene_index）
        ocr_results: OCR 结果列表
        vision_summaries: 画面摘要列表
        characters: 人物列表
        events: 事件列表

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
            for s in scenes:
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
        for s in scenes:
            if e.start_time < s.end_time and e.end_time > s.start_time:
                events_by_scene[s.scene_index].append(e)
                # 同时更新 event 的 scene_indices
                if s.scene_index not in e.scene_indices:
                    e.scene_indices.append(s.scene_index)

    # 构建 MemoryUnit
    memory_units = []
    for scene in scenes:
        si = scene.scene_index

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

        unit = MemoryUnit(
            scene_index=si,
            start_time=scene.start_time,
            end_time=scene.end_time,
            duration=scene.duration,
            keyframe_path=scene.keyframe_path,
            transcripts=scene_trans,
            vision=scene_vision,
            ocr=scene_ocr,
            characters=scene_chars,
            events=scene_events,
            combined_text=combined_text,
            embedding=[],  # 在 indexer 步骤中填充
        )
        memory_units.append(unit)

    logger.info(f"MemoryUnit 构建完成: {len(memory_units)} 个单元")
    return memory_units


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
        char_lines.append(
            f"- {c.character_id} ({c.display_name}): {c.description}\n"
            f"  出镜: {c.total_screen_time:.0f}s ({screen_pct:.1f}%), "
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
