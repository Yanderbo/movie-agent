# -*- coding: utf-8 -*-
"""
章节检测（v3 新增）

将连续 StoryScene 聚合为 Chapter（长视频大段落）。
Chapter 是 StoryScene 之上的最高叙事层级。

对于短视频 (< 10min)，整部视频就是一个 Chapter。
对于长视频，LLM 分析 StoryScene 序列的主题/地点/角色变化来决定章节边界。

降级策略：LLM 失败时每 3-5 个 StoryScene 一组。
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import config
from models.schemas import Shot, Beat, StoryScene, Chapter
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("ChapterDetect")

CHAPTER_PROMPT_TEMPLATE = """你是一个专业的影视叙事分析师。请将以下"故事场景"（StoryScene）序列聚合为"章节"（Chapter）。

一个 Chapter 是一个完整的叙事大段落，通常对应：
- 电影中的一个"幕"（Act）
- 一个重要的情节阶段（如：序幕、铺垫、发展、高潮、结局）
- 一个独立的叙事主题单元

视频总时长: {duration:.0f}秒

=== StoryScene 列表 ===
{scenes_info}

请将这些 StoryScene 分组为若干 Chapter。输出 JSON 数组：
- chapter_index: 从 0 开始编号
- title: 章节标题（简短有力）
- story_scene_indices: 包含的 StoryScene 索引列表（必须连续）
- description: 章节核心内容（一两句话）
- chapter_type: 叙事类型（prologue / act_1 / act_2 / act_3 / climax_act / epilogue / flashback）
- theme: 本章主题关键词
- characters: 涉及的主要人物 ID 列表
- mood_progression: 情绪走势描述（如"从紧张到释然"）

规则：
1. 一个 Chapter 通常包含 2-5 个 StoryScene
2. 叙事主题或情绪发生重大转变时，开启新 Chapter
3. 所有 StoryScene 都必须被分配到某个 Chapter

只输出 JSON：
```json
[
  {{
    "chapter_index": 0,
    "title": "命运的邂逅",
    "story_scene_indices": [0, 1, 2],
    "description": "男女主角在异国他乡意外相遇并开始合作",
    "chapter_type": "act_1",
    "theme": "相遇与信任建立",
    "characters": ["char_000", "char_001"],
    "mood_progression": "从陌生到好奇"
  }}
]
```
"""


def detect_chapters(
    video_id: str,
    story_scenes: list[StoryScene],
    beats: list[Beat],
    shots: list[Shot],
    meta_duration: float,
) -> list[Chapter]:
    """
    将 StoryScene 聚合为 Chapter。

    Args:
        video_id: 视频 ID
        story_scenes: StoryScene 列表
        beats: Beat 列表
        shots: Shot 列表
        meta_duration: 视频总时长

    Returns:
        Chapter 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    chapters_path = video_dir / "chapters.json"

    # 如果已存在，直接加载
    if chapters_path.exists():
        logger.info(f"Chapter 结果已存在，直接加载: {chapters_path}")
        data = json.loads(chapters_path.read_text(encoding="utf-8"))
        return [Chapter(**c) for c in data]

    if not story_scenes:
        logger.warning("无 StoryScene 数据，跳过 Chapter 检测")
        return []

    # 短视频：整部视频一个 Chapter
    if meta_duration < 600 or len(story_scenes) <= 3:
        logger.info(f"短视频或少量场景（{len(story_scenes)} 个），整体为一个 Chapter")
        chapters = _single_chapter(story_scenes, beats, shots)
    else:
        logger.info(f"开始 Chapter 检测: {len(story_scenes)} 个 StoryScene")
        chapters = _detect_via_llm(story_scenes, beats, shots, meta_duration)

    # 保存
    chapters_path.write_text(
        json.dumps([c.model_dump() for c in chapters], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Chapter 检测完成: {len(chapters)} 个章节")

    return chapters


def _single_chapter(
    story_scenes: list[StoryScene],
    beats: list[Beat],
    shots: list[Shot],
) -> list[Chapter]:
    """短视频：整部视频作为一个 Chapter"""
    all_beat_indices = []
    all_shot_indices = []
    all_chars = set()
    for ss in story_scenes:
        all_beat_indices.extend(ss.beat_indices)
        all_shot_indices.extend(ss.shot_indices)
        all_chars.update(ss.characters)

    ch = Chapter(
        chapter_index=0,
        title="全篇",
        start_time=story_scenes[0].start_time,
        end_time=story_scenes[-1].end_time,
        duration=story_scenes[-1].end_time - story_scenes[0].start_time,
        story_scene_indices=[ss.story_scene_index for ss in story_scenes],
        beat_indices=sorted(set(all_beat_indices)),
        shot_indices=sorted(set(all_shot_indices)),
        description="完整视频内容",
        chapter_type="act_1",
        characters=sorted(all_chars),
    )
    return [ch]


def _detect_via_llm(
    story_scenes: list[StoryScene],
    beats: list[Beat],
    shots: list[Shot],
    duration: float,
) -> list[Chapter]:
    """使用 LLM 检测 Chapter"""
    client = get_llm_client()

    # 构造 StoryScene 信息
    scene_lines = []
    for ss in story_scenes:
        parts = [
            f"StoryScene {ss.story_scene_index} [{ss.start_time:.1f}s-{ss.end_time:.1f}s]",
        ]
        if ss.location:
            parts.append(f"地点: {ss.location}")
        if ss.plot_function:
            parts.append(f"功能: {ss.plot_function}")
        if ss.description:
            parts.append(f"内容: {ss.description[:80]}")
        if ss.characters:
            parts.append(f"人物: {', '.join(ss.characters[:5])}")
        scene_lines.append(" | ".join(parts))

    scenes_info = "\n".join(scene_lines)
    prompt = CHAPTER_PROMPT_TEMPLATE.format(
        duration=duration,
        scenes_info=scenes_info,
    )

    try:
        response = client.chat(prompt=prompt, temperature=0.3)
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, list):
            logger.warning("Chapter 检测解析失败，使用默认分组")
            return _fallback_chapters(story_scenes, beats, shots)

        chapters = []
        beat_map = {b.beat_index: b for b in beats}
        for item in parsed:
            ss_indices = item.get("story_scene_indices", [])
            if not ss_indices:
                continue

            ch_scenes = [ss for ss in story_scenes if ss.story_scene_index in ss_indices]
            if not ch_scenes:
                continue

            # 收集 beat_indices 和 shot_indices
            ch_beat_indices = []
            ch_shot_indices = []
            for ss in ch_scenes:
                ch_beat_indices.extend(ss.beat_indices)
                ch_shot_indices.extend(ss.shot_indices)

            ch = Chapter(
                chapter_index=item.get("chapter_index", len(chapters)),
                title=item.get("title", ""),
                start_time=min(ss.start_time for ss in ch_scenes),
                end_time=max(ss.end_time for ss in ch_scenes),
                duration=sum(ss.duration for ss in ch_scenes),
                story_scene_indices=sorted(ss_indices),
                beat_indices=sorted(set(ch_beat_indices)),
                shot_indices=sorted(set(ch_shot_indices)),
                description=item.get("description", ""),
                chapter_type=item.get("chapter_type", ""),
                theme=item.get("theme", ""),
                characters=item.get("characters", []),
                mood_progression=item.get("mood_progression", ""),
            )
            chapters.append(ch)

        if not chapters:
            return _fallback_chapters(story_scenes, beats, shots)
        return chapters

    except Exception as e:
        logger.warning(f"Chapter 检测失败: {e}，使用默认分组")
        return _fallback_chapters(story_scenes, beats, shots)


def _fallback_chapters(
    story_scenes: list[StoryScene],
    beats: list[Beat],
    shots: list[Shot],
) -> list[Chapter]:
    """当 LLM 失败时，每 3 个 StoryScene 为一个 Chapter"""
    chapters = []
    group_size = 3
    for i in range(0, len(story_scenes), group_size):
        group = story_scenes[i: i + group_size]
        ch_beat_indices = []
        ch_shot_indices = []
        ch_chars = set()
        for ss in group:
            ch_beat_indices.extend(ss.beat_indices)
            ch_shot_indices.extend(ss.shot_indices)
            ch_chars.update(ss.characters)

        ch = Chapter(
            chapter_index=len(chapters),
            title=f"章节 {len(chapters) + 1}",
            start_time=group[0].start_time,
            end_time=group[-1].end_time,
            duration=sum(ss.duration for ss in group),
            story_scene_indices=[ss.story_scene_index for ss in group],
            beat_indices=sorted(set(ch_beat_indices)),
            shot_indices=sorted(set(ch_shot_indices)),
            characters=sorted(ch_chars),
        )
        chapters.append(ch)

    return chapters
