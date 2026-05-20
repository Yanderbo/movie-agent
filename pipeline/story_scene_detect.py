# -*- coding: utf-8 -*-
"""
故事场景检测（v2 新增）

将连续 beats 聚合为 story scenes。
StoryScene 是一段完整的叙事场景，通常对应一个地点/情境下的完整行动序列。

层级: Shot → Beat → StoryScene
"""
import json
import time
from pathlib import Path

import config
from models.schemas import Shot, Beat, StoryScene
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("StorySceneDetect")

SCENE_PROMPT_TEMPLATE = """你是一个专业的影视叙事分析师。请将以下"剧情节拍"（Beat）序列聚合为"故事场景"（StoryScene）。

一个 StoryScene 是由若干连续 Beat 组成的完整叙事场景，通常对应：
- 同一个地点/环境中发生的一系列事件
- 一段完整的情节单元（开始 → 发展 → 结束）
- 一个戏剧冲突的完整过程

=== Beat 列表 ===
{beats_info}

请将这些 Beat 分组为若干 StoryScene，输出 JSON 数组。每个 StoryScene 包含：
- story_scene_index: 从 0 开始编号
- beat_indices: 包含的 Beat 索引列表（必须连续）
- location: 场景地点/环境描述
- description: 这个场景的核心内容（一两句话）
- characters: 涉及的主要人物 ID 列表
- plot_function: 叙事功能（setup / inciting_incident / rising / climax / falling / resolution / epilogue）

规则：
1. 地点或情境发生大变化时，开启新 StoryScene
2. 一个 StoryScene 通常包含 2-6 个 Beat
3. 所有 Beat 都必须被分配到某个 StoryScene

只输出 JSON：
```json
[
  {{
    "story_scene_index": 0,
    "beat_indices": [0, 1, 2],
    "location": "咖啡厅",
    "description": "男女主角在咖啡厅重逢并讨论过去的误会",
    "characters": ["char_000", "char_001"],
    "plot_function": "inciting_incident"
  }}
]
```
"""


def detect_story_scenes(
    video_id: str,
    shots: list[Shot],
    beats: list[Beat],
) -> list[StoryScene]:
    """
    将连续 beats 聚合为 story scenes。

    Args:
        video_id: 视频 ID
        shots: 镜头列表
        beats: Beat 列表

    Returns:
        StoryScene 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    scenes_path = video_dir / "story_scenes.json"

    # 如果已存在，直接加载（仍需回填 shot 的 story_scene_index）
    if scenes_path.exists():
        logger.info(f"StoryScene 结果已存在，直接加载: {scenes_path}")
        data = json.loads(scenes_path.read_text(encoding="utf-8"))
        loaded_scenes = [StoryScene(**s) for s in data]
        # 缓存加载也需回填 shot 的反向链接
        _backfill_scene_to_shots(shots, loaded_scenes, video_id)
        return loaded_scenes

    if not beats:
        logger.warning("无 Beat 数据，跳过 StoryScene 检测")
        return []

    logger.info(f"开始 StoryScene 检测: {len(beats)} 个 beat")

    client = get_llm_client()

    # 构造 beat 信息
    beat_lines = []
    for b in beats:
        parts = [
            f"Beat {b.beat_index} [{b.start_time:.1f}s-{b.end_time:.1f}s]",
            f"类型: {b.beat_type}",
        ]
        if b.description:
            parts.append(f"内容: {b.description}")
        if b.emotion:
            parts.append(f"情绪: {b.emotion}")
        if b.characters:
            parts.append(f"人物: {', '.join(b.characters)}")
        beat_lines.append(" | ".join(parts))

    beats_info = "\n".join(beat_lines)

    prompt = SCENE_PROMPT_TEMPLATE.format(beats_info=beats_info)

    try:
        response = client.chat(prompt=prompt, temperature=0.3)
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, list):
            logger.warning("StoryScene 检测解析失败，使用默认分组")
            story_scenes = _fallback_story_scenes(beats, shots)
        else:
            story_scenes = []
            for item in parsed:
                beat_indices = item.get("beat_indices", [])
                if not beat_indices:
                    continue
                # 收集对应的 shots
                scene_beats = [b for b in beats if b.beat_index in beat_indices]
                if not scene_beats:
                    continue
                shot_indices = []
                for b in scene_beats:
                    shot_indices.extend(b.shot_indices)
                shot_indices = sorted(set(shot_indices))

                ss = StoryScene(
                    story_scene_index=item.get("story_scene_index", len(story_scenes)),
                    start_time=min(b.start_time for b in scene_beats),
                    end_time=max(b.end_time for b in scene_beats),
                    duration=sum(b.duration for b in scene_beats),
                    beat_indices=sorted(beat_indices),
                    shot_indices=shot_indices,
                    location=item.get("location", ""),
                    description=item.get("description", ""),
                    characters=item.get("characters", []),
                    plot_function=item.get("plot_function", ""),
                )
                story_scenes.append(ss)
    except Exception as e:
        logger.warning(f"StoryScene 检测失败: {e}，使用默认分组")
        story_scenes = _fallback_story_scenes(beats, shots)

    # 回填 shot 的 story_scene_index 并持久化
    _backfill_scene_to_shots(shots, story_scenes, video_id)

    # 保存 story_scenes.json
    scenes_path.write_text(
        json.dumps([s.model_dump() for s in story_scenes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"StoryScene 检测完成: {len(story_scenes)} 个故事场景")

    return story_scenes


def _backfill_scene_to_shots(
    shots: list[Shot], story_scenes: list[StoryScene], video_id: str,
):
    """回填 shot.story_scene_index 并持久化到 scenes.json"""
    shot_map = {s.scene_index: s for s in shots}
    for ss in story_scenes:
        for si in ss.shot_indices:
            if si in shot_map:
                shot_map[si].story_scene_index = ss.story_scene_index
    # 持久化反向链接
    shots_json = config.VIDEOS_DIR / video_id / "scenes" / "scenes.json"
    if shots_json.exists():
        shots_json.write_text(
            json.dumps([s.model_dump() for s in shots], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _fallback_story_scenes(beats: list[Beat], shots: list[Shot]) -> list[StoryScene]:
    """当 LLM 失败时，每 3 个 beat 为一个 story scene"""
    story_scenes = []
    group_size = 3
    for i in range(0, len(beats), group_size):
        group = beats[i: i + group_size]
        shot_indices = []
        for b in group:
            shot_indices.extend(b.shot_indices)
        shot_indices = sorted(set(shot_indices))

        ss = StoryScene(
            story_scene_index=len(story_scenes),
            start_time=group[0].start_time,
            end_time=group[-1].end_time,
            duration=sum(b.duration for b in group),
            beat_indices=[b.beat_index for b in group],
            shot_indices=shot_indices,
        )
        story_scenes.append(ss)
    return story_scenes
