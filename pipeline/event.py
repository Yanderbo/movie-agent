# -*- coding: utf-8 -*-
"""
事件抽取
使用 Gemini 从台词 + 画面摘要 + 人物信息中抽取关键事件。
"""
import json
import time
from pathlib import Path

import config
from models.schemas import (
    TranscriptSegment, VisionSummary, Character, Event, Scene,
)
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("EventExtract")

EVENT_PROMPT_TEMPLATE = """你是一个专业的视频内容分析师。基于以下视频内容信息，提取关键事件。

视频总时长: {duration:.1f} 秒

=== 台词（部分） ===
{transcripts_text}

=== 画面摘要（部分） ===
{vision_text}

=== 已识别人物 ===
{characters_text}

请提取视频中的关键事件，每个事件代表一个有意义的叙事单元。

要求：
1. 事件按时间顺序排列
2. 每个事件覆盖一段连续时间
3. 事件类型包括：开场、对话、冲突、转折、高潮、结局、日常、回忆、独白、追逐、浪漫、搞笑、悲伤、悬疑
4. importance 用 1-10 评分，高潮和转折事件分数更高
5. 标注涉及的人物 ID
6. 描述每个事件的核心内容

输出 JSON 数组，只输出 JSON：
```json
[
  {{
    "event_index": 0,
    "start_time": 0.0,
    "end_time": 30.0,
    "event_type": "开场",
    "description": "事件描述",
    "characters": ["char_000", "char_001"],
    "emotion": "平静",
    "importance": 5
  }}
]
```
"""


def extract_events(
    video_id: str,
    scenes: list[Scene],
    transcripts: list[TranscriptSegment],
    vision_summaries: list[VisionSummary],
    characters: list[Character],
    duration: float,
) -> list[Event]:
    """
    从视频理解结果中抽取关键事件。

    Args:
        video_id: 视频 ID
        scenes: 镜头列表
        transcripts: 台词列表
        vision_summaries: 画面摘要列表
        characters: 人物列表
        duration: 视频总时长

    Returns:
        Event 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    events_path = video_dir / "events.json"

    # 如果已存在，直接加载
    if events_path.exists():
        logger.info(f"事件抽取结果已存在，直接加载: {events_path}")
        data = json.loads(events_path.read_text(encoding="utf-8"))
        return [Event(**e) for e in data]

    logger.info("开始事件抽取")

    client = get_llm_client()
    all_events = []

    # 对于短视频（< 30min），一次性处理
    # 对于长视频，分段处理
    segment_duration = 1800  # 30 分钟一段

    if duration <= segment_duration:
        events = _extract_events_segment(
            client, transcripts, vision_summaries, characters,
            0, duration, duration, offset=0,
        )
        all_events.extend(events)
    else:
        # 分段处理
        seg_start = 0.0
        seg_idx = 0
        while seg_start < duration:
            seg_end = min(seg_start + segment_duration, duration)
            logger.info(f"处理事件段 {seg_idx+1}: {seg_start:.0f}s - {seg_end:.0f}s")

            # 过滤当前时间段的内容
            seg_trans = [
                t for t in transcripts
                if t.start_time >= seg_start and t.start_time < seg_end
            ]
            seg_vision = [
                v for v in vision_summaries
                if v.timestamp >= seg_start and v.timestamp < seg_end
            ]

            events = _extract_events_segment(
                client, seg_trans, seg_vision, characters,
                seg_start, seg_end, duration, offset=0,
            )
            all_events.extend(events)

            seg_start = seg_end
            seg_idx += 1
            time.sleep(1)

    # 重新编号
    for i, event in enumerate(all_events):
        event.event_index = i

    # 为每个 event 填充 scene_indices（通过时间范围交叉）
    for event in all_events:
        event.scene_indices = []
        for scene in scenes:
            if event.start_time < scene.end_time and event.end_time > scene.start_time:
                event.scene_indices.append(scene.scene_index)

    # 保存结果
    events_path.write_text(
        json.dumps([e.model_dump() for e in all_events], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"事件抽取完成: {len(all_events)} 个事件")

    return all_events


def _extract_events_segment(
    client,
    transcripts: list[TranscriptSegment],
    vision_summaries: list[VisionSummary],
    characters: list[Character],
    start: float,
    end: float,
    total_duration: float,
    offset: float = 0,
) -> list[Event]:
    """处理一个时间段的事件抽取"""

    # 构造台词文本
    trans_lines = []
    for t in transcripts[:100]:  # 限制长度
        speaker = f"[{t.speaker}]" if t.speaker else ""
        trans_lines.append(f"[{t.start_time:.1f}s-{t.end_time:.1f}s] {speaker} {t.text}")
    transcripts_text = "\n".join(trans_lines) if trans_lines else "（无台词）"

    # 构造画面摘要文本
    vision_lines = []
    for v in vision_summaries[:50]:  # 限制长度
        vision_lines.append(
            f"[{v.timestamp:.1f}s] [{v.scene_type}] [{v.mood}] {v.description}"
        )
    vision_text = "\n".join(vision_lines) if vision_lines else "（无画面摘要）"

    # 构造人物文本
    char_lines = []
    for c in characters:
        char_lines.append(
            f"- {c.character_id} ({c.display_name}): {c.description}, "
            f"出场 {len(c.appearance_scenes)} 个镜头"
        )
    characters_text = "\n".join(char_lines) if char_lines else "（未识别人物）"

    prompt = EVENT_PROMPT_TEMPLATE.format(
        duration=total_duration,
        transcripts_text=transcripts_text,
        vision_text=vision_text,
        characters_text=characters_text,
    )

    try:
        response = client.chat(prompt=prompt, temperature=0.3)
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, list):
            logger.warning("事件抽取解析失败")
            return []

        events = []
        for item in parsed:
            event = Event(
                event_index=item.get("event_index", 0),
                start_time=float(item.get("start_time", 0)),
                end_time=float(item.get("end_time", 0)),
                event_type=item.get("event_type", ""),
                description=item.get("description", ""),
                characters=item.get("characters", []),
                emotion=item.get("emotion", ""),
                importance=int(item.get("importance", 5)),
            )
            events.append(event)

        return events

    except Exception as e:
        logger.error(f"事件抽取失败: {e}")
        return []
