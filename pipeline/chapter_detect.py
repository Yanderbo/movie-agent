# -*- coding: utf-8 -*-
"""
章节检测（v3 新增）

将连续 StoryScene 聚合为 Chapter（长视频大段落）。
Chapter 是 StoryScene 之上的最高叙事层级。

对于短视频 (< 10min)，整部视频就是一个 Chapter。
对于长视频，LLM 分析 StoryScene 序列的主题/地点/角色变化来决定章节边界。

设计要点（与 beat / story_scene 对齐）:
- chapter_index 由本地自增统一编号，不信任 LLM 返回值；
- characters 从子 StoryScene 聚合，保证与 face_cluster 体系一致；
- 通过 _finalize_chapters 保证每个 StoryScene 恰好归属一个 Chapter；
- LLM 失败时每 3 个 StoryScene 一组降级。
"""
from __future__ import annotations
import json

import config
from models.schemas import Shot, Beat, StoryScene, Chapter
from prompt import CHAPTER_PROMPT_TEMPLATE
from utils import group_consecutive
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("ChapterDetect")


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
        beats: Beat 列表（保留以兼容调用方，当前聚合直接基于 StoryScene）
        shots: Shot 列表（保留以兼容调用方）
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
        chapters = [_single_chapter(story_scenes)]
    else:
        logger.info(f"开始 Chapter 检测: {len(story_scenes)} 个 StoryScene")
        chapters = _detect_via_llm(story_scenes, meta_duration)

    # 统一走 _finalize_chapters 保证覆盖 + 去重 + 索引连续
    chapters = _finalize_chapters(chapters, story_scenes)

    chapters_path.write_text(
        json.dumps([c.model_dump() for c in chapters], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"Chapter 检测完成: {len(chapters)} 个章节")
    return chapters


def _single_chapter(story_scenes: list[StoryScene]) -> Chapter:
    """短视频：整部视频作为一个 Chapter（返回单个 Chapter，由调用方包装为列表）"""
    ch = Chapter(
        chapter_index=0,
        title="全篇",
        start_time=story_scenes[0].start_time,
        end_time=story_scenes[-1].end_time,
        duration=story_scenes[-1].end_time - story_scenes[0].start_time,
        story_scene_indices=[ss.story_scene_index for ss in story_scenes],
        beat_indices=sorted({bi for ss in story_scenes for bi in ss.beat_indices}),
        shot_indices=sorted({si for ss in story_scenes for si in ss.shot_indices}),
        description="完整视频内容",
        chapter_type="act_1",
        characters=sorted({c for ss in story_scenes for c in ss.characters}),
    )
    return ch


def _detect_via_llm(
    story_scenes: list[StoryScene], duration: float,
) -> list[Chapter]:
    """使用 LLM 检测 Chapter（chapter_index 占位，由 _finalize_chapters 统一重排）"""
    client = get_llm_client()

    scene_lines = []
    for ss in story_scenes:
        parts = [f"StoryScene {ss.story_scene_index} [{ss.start_time:.1f}s-{ss.end_time:.1f}s]"]
        if ss.location:
            parts.append(f"地点: {ss.location}")
        if ss.plot_function:
            parts.append(f"功能: {ss.plot_function}")
        if ss.description:
            parts.append(f"内容: {ss.description[:80]}")
        if ss.characters:
            parts.append(f"人物: {', '.join(ss.characters[:5])}")
        scene_lines.append(" | ".join(parts))

    prompt = CHAPTER_PROMPT_TEMPLATE.format(
        duration=duration,
        scenes_info="\n".join(scene_lines),
    )
    scene_map = {ss.story_scene_index: ss for ss in story_scenes}

    try:
        response = client.chat(prompt=prompt, temperature=0.3)
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, list):
            logger.warning("Chapter 检测解析失败，使用默认分组")
            return _fallback_chapters(story_scenes)

        chapters = []
        for item in parsed:
            ss_indices = item.get("story_scene_indices", [])
            ch_scenes = [scene_map[i] for i in ss_indices if i in scene_map]
            if not ch_scenes:
                continue
            chapters.append(Chapter(
                chapter_index=-1,
                title=item.get("title", ""),
                start_time=min(ss.start_time for ss in ch_scenes),
                end_time=max(ss.end_time for ss in ch_scenes),
                duration=(
                    max(ss.end_time for ss in ch_scenes)
                    - min(ss.start_time for ss in ch_scenes)
                ),
                story_scene_indices=sorted(ss.story_scene_index for ss in ch_scenes),
                beat_indices=sorted({bi for ss in ch_scenes for bi in ss.beat_indices}),
                shot_indices=sorted({si for ss in ch_scenes for si in ss.shot_indices}),
                description=item.get("description", ""),
                chapter_type=item.get("chapter_type", ""),
                theme=item.get("theme", ""),
                characters=sorted({c for ss in ch_scenes for c in ss.characters}),
                mood_progression=item.get("mood_progression", ""),
            ))
        return chapters if chapters else _fallback_chapters(story_scenes)
    except Exception as e:
        logger.warning(f"Chapter 检测失败: {e}，使用默认分组")
        return _fallback_chapters(story_scenes)


def _finalize_chapters(
    chapters: list[Chapter], story_scenes: list[StoryScene],
) -> list[Chapter]:
    """
    规范化 Chapter，使其对 StoryScene 构成完整且不重叠的划分：

    1. 跨章节去重 StoryScene（保留先出现者），过滤非法索引；
    2. 未覆盖的 StoryScene 按相邻关系聚合为 ``transition`` 章节；
    3. 按时间统一排序，重排 chapter_index 为连续唯一值，重算时间/时长/聚合字段。
    """
    if not story_scenes:
        return chapters
    ss_map = {ss.story_scene_index: ss for ss in story_scenes}

    chapters = [c for c in chapters if c.story_scene_indices]
    chapters.sort(key=lambda c: c.start_time)
    seen: set[int] = set()
    for c in chapters:
        kept = [i for i in c.story_scene_indices if i in ss_map and i not in seen]
        seen.update(kept)
        c.story_scene_indices = sorted(kept)
    chapters = [c for c in chapters if c.story_scene_indices]

    uncovered = [ss for ss in story_scenes if ss.story_scene_index not in seen]
    for group in group_consecutive(uncovered, lambda ss: ss.story_scene_index):
        chapters.append(Chapter(
            chapter_index=-1,
            title="过渡段落",
            start_time=group[0].start_time,
            end_time=group[-1].end_time,
            duration=group[-1].end_time - group[0].start_time,
            story_scene_indices=[ss.story_scene_index for ss in group],
            beat_indices=sorted({bi for ss in group for bi in ss.beat_indices}),
            shot_indices=sorted({si for ss in group for si in ss.shot_indices}),
            chapter_type="transition",
            characters=sorted({c for ss in group for c in ss.characters}),
        ))

    chapters.sort(key=lambda c: min(ss_map[i].start_time for i in c.story_scene_indices))
    for idx, c in enumerate(chapters):
        sscs = [ss_map[i] for i in c.story_scene_indices]
        c.chapter_index = idx
        c.story_scene_indices = sorted(ss.story_scene_index for ss in sscs)
        c.beat_indices = sorted({bi for ss in sscs for bi in ss.beat_indices})
        c.shot_indices = sorted({si for ss in sscs for si in ss.shot_indices})
        c.start_time = min(ss.start_time for ss in sscs)
        c.end_time = max(ss.end_time for ss in sscs)
        c.duration = c.end_time - c.start_time
        # 始终从当前拥有的 story_scenes 重新聚合角色，防止去重后残留已不属于本章节的角色
        c.characters = sorted({ch for ss in sscs for ch in ss.characters})
    return chapters


def _fallback_chapters(story_scenes: list[StoryScene]) -> list[Chapter]:
    """当 LLM 失败时，每 3 个 StoryScene 为一个 Chapter（索引交由 finalize 统一重排）"""
    chapters = []
    group_size = 3
    for i in range(0, len(story_scenes), group_size):
        group = story_scenes[i: i + group_size]
        chapters.append(Chapter(
            chapter_index=-1,
            title=f"章节 {i // group_size + 1}",
            start_time=group[0].start_time,
            end_time=group[-1].end_time,
            duration=group[-1].end_time - group[0].start_time,
            story_scene_indices=[ss.story_scene_index for ss in group],
            beat_indices=sorted({bi for ss in group for bi in ss.beat_indices}),
            shot_indices=sorted({si for ss in group for si in ss.shot_indices}),
            characters=sorted({c for ss in group for c in ss.characters}),
        ))
    return chapters
