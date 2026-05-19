# -*- coding: utf-8 -*-
"""
事件图谱构建（v2 — 从简单列表升级为事件图）

v2 变更:
- 事件抽取后，增加关系推理阶段
- LLM 输出 events + edges（因果/铺垫/反转/冲突升级/结果/平行）
- 输出 EventGraph 包含 nodes（EventNode）和 edges（EventEdge）
- 事件绑定到 beat_indices / story_scene_indices
"""
import json
import time
from pathlib import Path

import config
from models.schemas import (
    TranscriptSegment, VisionSummary, Character, Event, EventEdge, EventGraph,
    Shot, Beat, StoryScene,
)
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("EventGraph")

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
7. evidence: 列出支撑该事件判断的证据来源（如 "transcript:台词内容", "vision:画面描述", "audio:音效/音乐"）
8. confidence: 该事件抽取的置信度 0-1

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
    "importance": 5,
    "evidence": ["transcript:角色A说了xxx", "vision:画面中出现了xxx"],
    "confidence": 0.85
  }}
]
```
"""

EDGE_PROMPT_TEMPLATE = """你是一个专业的叙事分析师。请分析以下事件之间的关系。

=== 事件列表 ===
{events_text}

请分析事件之间的因果关系、铺垫关系、反转关系、冲突升级、结果关系和平行关系。

关系类型说明：
- cause: A 导致了 B（因果）
- foreshadow: A 为 B 埋下了伏笔（铺垫）
- reversal: B 是 A 的反转/意外
- escalation: B 是 A 的冲突升级
- resolution: B 是 A 的解决/结果
- parallel: A 和 B 是平行/对照的情节线

同时为每条关系提供：
- evidence: 支撑该关系的证据（引用具体的台词或画面）
- confidence: 关系推断的置信度 0-1
- relation_basis: 关系推断的依据说明

输出 JSON 数组，每个元素描述一条关系边。只关注重要的关系，不要过度连接。
只输出 JSON：
```json
[
  {{
    "source_event": 0,
    "target_event": 2,
    "relation_type": "cause",
    "description": "因为A发生了，所以导致了C",
    "strength": 0.8,
    "evidence": ["事件0中角色说了xxx", "事件2中画面出现了xxx"],
    "confidence": 0.75,
    "relation_basis": "角色A在事件0中的决定直接导致了事件2的冲突"
  }}
]
```
"""


def extract_events(
    video_id: str,
    scenes: list[Shot],
    transcripts: list[TranscriptSegment],
    vision_summaries: list[VisionSummary],
    characters: list[Character],
    duration: float,
    beats: list[Beat] = None,
    story_scenes: list[StoryScene] = None,
) -> tuple[list[Event], EventGraph]:
    """
    从视频理解结果中抽取关键事件，并构建事件图谱。

    v2 返回 (events, event_graph) 二元组。

    Args:
        video_id: 视频 ID
        scenes: 镜头列表
        transcripts: 台词列表
        vision_summaries: 画面摘要列表
        characters: 人物列表
        duration: 视频总时长
        beats: Beat 列表（可选）
        story_scenes: StoryScene 列表（可选）

    Returns:
        (Event 列表, EventGraph)
    """
    video_dir = config.VIDEOS_DIR / video_id
    events_path = video_dir / "events.json"
    graph_path = video_dir / "event_graph.json"

    # 如果已存在，直接加载
    if events_path.exists() and graph_path.exists():
        logger.info("事件图谱结果已存在，直接加载")
        events_data = json.loads(events_path.read_text(encoding="utf-8"))
        graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
        events = [Event(**e) for e in events_data]
        graph = EventGraph(**graph_data)
        return events, graph
    elif events_path.exists():
        # 旧数据: 只有事件没有图谱
        logger.info("旧事件数据已存在，加载后补充图谱")
        events_data = json.loads(events_path.read_text(encoding="utf-8"))
        events = [Event(**e) for e in events_data]
    else:
        events = None

    logger.info("开始事件抽取")
    client = get_llm_client()

    # ── Phase 1: 事件抽取 ──
    if events is None:
        all_events = []
        segment_duration = 1800  # 30 分钟一段

        if duration <= segment_duration:
            evts = _extract_events_segment(
                client, transcripts, vision_summaries, characters,
                0, duration, duration, offset=0,
            )
            all_events.extend(evts)
        else:
            seg_start = 0.0
            seg_idx = 0
            while seg_start < duration:
                seg_end = min(seg_start + segment_duration, duration)
                logger.info(f"处理事件段 {seg_idx+1}: {seg_start:.0f}s - {seg_end:.0f}s")

                seg_trans = [
                    t for t in transcripts
                    if t.start_time >= seg_start and t.start_time < seg_end
                ]
                seg_vision = [
                    v for v in vision_summaries
                    if v.timestamp >= seg_start and v.timestamp < seg_end
                ]

                evts = _extract_events_segment(
                    client, seg_trans, seg_vision, characters,
                    seg_start, seg_end, duration, offset=0,
                )
                all_events.extend(evts)

                seg_start = seg_end
                seg_idx += 1
                time.sleep(1)

        # 重新编号
        for i, event in enumerate(all_events):
            event.event_index = i

        events = all_events

    # 填充 scene_indices / beat_indices / story_scene_indices
    for event in events:
        if not event.scene_indices:
            event.scene_indices = []
            for scene in scenes:
                if event.start_time < scene.end_time and event.end_time > scene.start_time:
                    event.scene_indices.append(scene.scene_index)

        if beats and not event.beat_indices:
            event.beat_indices = []
            for b in beats:
                if event.start_time < b.end_time and event.end_time > b.start_time:
                    event.beat_indices.append(b.beat_index)

        if story_scenes and not event.story_scene_indices:
            event.story_scene_indices = []
            for ss in story_scenes:
                if event.start_time < ss.end_time and event.end_time > ss.start_time:
                    event.story_scene_indices.append(ss.story_scene_index)

    # 保存事件
    events_path.write_text(
        json.dumps([e.model_dump() for e in events], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Phase 2: 事件关系推理（构建图谱）──
    edges = _extract_event_edges(client, events)

    event_graph = EventGraph(nodes=events, edges=edges)

    # 保存图谱
    graph_path.write_text(
        json.dumps(event_graph.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        f"事件图谱构建完成: {len(events)} 个事件, {len(edges)} 条关系"
    )

    return events, event_graph


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
                evidence=item.get("evidence", []),
                confidence=float(item.get("confidence", 0.8)),
            )
            events.append(event)

        return events

    except Exception as e:
        logger.error(f"事件抽取失败: {e}")
        return []


def _extract_event_edges(client, events: list[Event]) -> list[EventEdge]:
    """使用 LLM 推断事件间关系"""
    if len(events) < 2:
        return []

    # 构造事件摘要
    event_lines = []
    for e in events[:30]:  # 限制数量
        event_lines.append(
            f"Event {e.event_index} [{e.start_time:.0f}s-{e.end_time:.0f}s] "
            f"[{e.event_type}] [重要性:{e.importance}] {e.description}"
        )
    events_text = "\n".join(event_lines)

    prompt = EDGE_PROMPT_TEMPLATE.format(events_text=events_text)

    try:
        response = client.chat(prompt=prompt, temperature=0.3)
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, list):
            logger.warning("事件关系推理解析失败")
            return []

        valid_indices = {e.event_index for e in events}
        edges = []
        for item in parsed:
            src = int(item.get("source_event", -1))
            tgt = int(item.get("target_event", -1))
            if src in valid_indices and tgt in valid_indices and src != tgt:
                edge = EventEdge(
                    source_event=src,
                    target_event=tgt,
                    relation_type=item.get("relation_type", ""),
                    description=item.get("description", ""),
                    strength=float(item.get("strength", 0.5)),
                    evidence=item.get("evidence", []),
                    confidence=float(item.get("confidence", 0.5)),
                    relation_basis=item.get("relation_basis", ""),
                )
                edges.append(edge)

        logger.info(f"事件关系推理完成: {len(edges)} 条关系")
        return edges

    except Exception as e:
        logger.warning(f"事件关系推理失败: {e}")
        return []
