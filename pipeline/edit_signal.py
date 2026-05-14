# -*- coding: utf-8 -*-
"""
剪辑信号计算（v2 新增）

为每个 shot / beat / story_scene 计算面向剪辑决策的 8 个信号:
- hook_score: 作为开头钩子的适合度
- plot_importance: 对整体叙事的贡献度
- emotional_intensity: 情绪表达的强烈程度
- visual_impact: 画面构图/运镜/特效的吸引力
- independence_score: 不需上下文也能理解的程度
- continuity_dependency: 必须与前后连续才有意义的程度
- boundary_quality: 作为剪辑点的自然度
- spoiler_level: 包含关键剧情信息的程度

这些信号直接服务于 DirectorAgent / ReviewerAgent 的剪辑决策。
"""
import json
import time
from pathlib import Path

import config
from models.schemas import (
    Shot, Beat, StoryScene, Event, Character, VisionSummary,
    TranscriptSegment, EditSignal,
)
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("EditSignal")

SIGNAL_PROMPT_TEMPLATE = """你是一个专业的影视剪辑师。请为以下视频片段计算剪辑信号。

=== 片段列表 ===
{segments_info}

请为每个片段评估以下 8 个信号（0.0 - 1.0 分）：

1. hook_score: 作为视频开头/钩子的适合度（画面冲击力、悬念、吸引力）
2. plot_importance: 对整体剧情的贡献度（核心剧情=高，过渡/日常=低）
3. emotional_intensity: 情绪表达强度（强烈情绪=高，平静=低）
4. visual_impact: 视觉冲击力（特殊构图/运镜/特效=高，普通对话=低）
5. independence_score: 片段独立性（单独观看也能理解=高，需要上下文=低）
6. continuity_dependency: 连续性依赖（必须与前后片段连看=高，可独立剪出=低）
7. boundary_quality: 剪辑边界质量（开头/结尾有自然停顿=高，在句中/动作中=低）
8. spoiler_level: 剧透程度（包含关键反转/结局=高，日常场景=低）

同时建议每个片段适合的用途（可多选）：
- hook: 适合作为开头钩子
- trailer: 适合放入预告片
- highlight: 适合作为精彩集锦
- recap: 适合用于剧情回顾
- climax_clip: 适合作为高潮片段
- character_intro: 适合用于人物介绍

输出 JSON 数组，只输出 JSON：
```json
[
  {{
    "unit_index": 0,
    "hook_score": 0.8,
    "plot_importance": 0.7,
    "emotional_intensity": 0.9,
    "visual_impact": 0.6,
    "independence_score": 0.5,
    "continuity_dependency": 0.4,
    "boundary_quality": 0.7,
    "spoiler_level": 0.3,
    "suggested_usage": ["hook", "highlight"]
  }}
]
```
"""


def compute_edit_signals(
    video_id: str,
    shots: list[Shot],
    beats: list[Beat],
    story_scenes: list[StoryScene],
    events: list[Event],
    characters: list[Character],
    transcripts: list[TranscriptSegment],
    vision_summaries: list[VisionSummary],
) -> list[EditSignal]:
    """
    为 shot / beat / story_scene 计算剪辑信号。

    Args:
        video_id: 视频 ID
        shots: 镜头列表
        beats: Beat 列表
        story_scenes: StoryScene 列表
        events: 事件列表
        characters: 人物列表
        transcripts: 台词列表
        vision_summaries: 画面摘要列表

    Returns:
        EditSignal 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    signals_path = video_dir / "edit_signals.json"

    # 如果已存在，直接加载
    if signals_path.exists():
        logger.info(f"剪辑信号已存在，直接加载: {signals_path}")
        data = json.loads(signals_path.read_text(encoding="utf-8"))
        return [EditSignal(**s) for s in data]

    logger.info("开始计算剪辑信号")
    client = get_llm_client()

    all_signals = []

    # ── 为 beat 计算信号（beat 是剪辑的核心粒度）──
    if beats:
        beat_signals = _compute_signals_for_units(
            client, "beat", beats, events, characters, transcripts, vision_summaries
        )
        all_signals.extend(beat_signals)

    # ── 为 story_scene 计算信号 ──
    if story_scenes:
        scene_signals = _compute_signals_for_units(
            client, "story_scene", story_scenes, events, characters,
            transcripts, vision_summaries
        )
        all_signals.extend(scene_signals)

    # ── 为重要 shot 计算信号（选择性，避免过多 API 调用）──
    # 只为每个 beat 的首尾 shot 和包含重要事件的 shot 计算
    important_shot_indices = set()
    for b in beats:
        if b.shot_indices:
            important_shot_indices.add(b.shot_indices[0])
            important_shot_indices.add(b.shot_indices[-1])
    for e in events:
        if e.importance >= 7:
            for si in e.scene_indices:
                important_shot_indices.add(si)

    important_shots = [s for s in shots if s.scene_index in important_shot_indices]
    if important_shots:
        shot_signals = _compute_signals_for_units(
            client, "shot", important_shots, events, characters,
            transcripts, vision_summaries
        )
        all_signals.extend(shot_signals)

    # 保存
    signals_path.write_text(
        json.dumps([s.model_dump() for s in all_signals], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"剪辑信号计算完成: {len(all_signals)} 个信号")

    return all_signals


def _compute_signals_for_units(
    client,
    unit_type: str,
    units,
    events: list[Event],
    characters: list[Character],
    transcripts: list[TranscriptSegment],
    vision_summaries: list[VisionSummary],
) -> list[EditSignal]:
    """为一组 unit 计算剪辑信号"""
    signals = []
    batch_size = 15

    # 构建辅助索引
    trans_map = {}
    for t in transcripts:
        trans_map.setdefault(t.scene_index, []).append(t)
    vision_map = {v.scene_index: v for v in vision_summaries}

    for batch_start in range(0, len(units), batch_size):
        batch = units[batch_start: batch_start + batch_size]

        # 构造片段信息
        seg_lines = []
        for unit in batch:
            if unit_type == "shot":
                idx = unit.scene_index
                start = unit.start_time
                end = unit.end_time
                trans = trans_map.get(idx, [])
                vis = vision_map.get(idx)
            elif unit_type == "beat":
                idx = unit.beat_index
                start = unit.start_time
                end = unit.end_time
                trans = []
                for si in unit.shot_indices:
                    trans.extend(trans_map.get(si, []))
                vis_descs = [
                    vision_map[si].description
                    for si in unit.shot_indices if si in vision_map
                ]
                vis = None  # 用 vis_descs 替代
            elif unit_type == "story_scene":
                idx = unit.story_scene_index
                start = unit.start_time
                end = unit.end_time
                trans = []
                for si in unit.shot_indices:
                    trans.extend(trans_map.get(si, []))
                vis_descs = [
                    vision_map[si].description
                    for si in unit.shot_indices if si in vision_map
                ]
                vis = None
            else:
                continue

            parts = [f"[{unit_type} {idx}] {start:.1f}s-{end:.1f}s"]

            # 描述
            if hasattr(unit, "description") and unit.description:
                parts.append(f"内容: {unit.description[:80]}")
            if hasattr(unit, "beat_type") and unit.beat_type:
                parts.append(f"类型: {unit.beat_type}")
            if hasattr(unit, "plot_function") and unit.plot_function:
                parts.append(f"功能: {unit.plot_function}")

            # 台词摘要
            if trans:
                trans_text = " ".join([t.text[:30] for t in trans[:3]])
                parts.append(f"台词: {trans_text}")

            # 画面
            if unit_type == "shot" and vis:
                parts.append(f"画面: {vis.description[:60]}")
                if vis.mood:
                    parts.append(f"情绪: {vis.mood}")
            elif unit_type in ("beat", "story_scene"):
                if vis_descs:
                    parts.append(f"画面: {'; '.join(v[:40] for v in vis_descs[:3])}")

            # 人物
            chars = []
            if hasattr(unit, "characters") and unit.characters:
                chars = unit.characters
            if chars:
                parts.append(f"人物: {','.join(chars[:5])}")

            # 关联事件
            unit_events = []
            for e in events:
                if start < e.end_time and end > e.start_time:
                    unit_events.append(e)
            if unit_events:
                evt_desc = "; ".join(
                    [f"{e.event_type}({e.importance})" for e in unit_events[:3]]
                )
                parts.append(f"事件: {evt_desc}")

            seg_lines.append(" | ".join(parts))

        segments_info = "\n".join(seg_lines)
        prompt = SIGNAL_PROMPT_TEMPLATE.format(segments_info=segments_info)

        try:
            response = client.chat(prompt=prompt, temperature=0.2)
            parsed = client.parse_json(response)
            if parsed and isinstance(parsed, list):
                for i, item in enumerate(parsed):
                    if i >= len(batch):
                        break
                    unit = batch[i]

                    if unit_type == "shot":
                        u_idx = unit.scene_index
                        u_start = unit.start_time
                        u_end = unit.end_time
                    elif unit_type == "beat":
                        u_idx = unit.beat_index
                        u_start = unit.start_time
                        u_end = unit.end_time
                    elif unit_type == "story_scene":
                        u_idx = unit.story_scene_index
                        u_start = unit.start_time
                        u_end = unit.end_time
                    else:
                        continue

                    signal = EditSignal(
                        unit_type=unit_type,
                        unit_index=u_idx,
                        start_time=u_start,
                        end_time=u_end,
                        hook_score=float(item.get("hook_score", 0)),
                        plot_importance=float(item.get("plot_importance", 0)),
                        emotional_intensity=float(item.get("emotional_intensity", 0)),
                        visual_impact=float(item.get("visual_impact", 0)),
                        independence_score=float(item.get("independence_score", 0)),
                        continuity_dependency=float(item.get("continuity_dependency", 0)),
                        boundary_quality=float(item.get("boundary_quality", 0)),
                        spoiler_level=float(item.get("spoiler_level", 0)),
                        suggested_usage=item.get("suggested_usage", []),
                    )
                    signals.append(signal)
            else:
                logger.warning(f"剪辑信号解析失败 ({unit_type} batch)")
        except Exception as e:
            logger.warning(f"剪辑信号计算失败 ({unit_type} batch): {e}")

        time.sleep(0.5)

    return signals
