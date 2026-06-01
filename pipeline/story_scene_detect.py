# -*- coding: utf-8 -*-
"""
故事场景检测（v2 新增）

将连续 beats 聚合为 story scenes。
StoryScene 是一段完整的叙事场景，通常对应一个地点/情境下的完整行动序列。

层级: Shot → Beat → StoryScene

设计要点（与 beat_detect 对齐）:
- 长视频按窗口分段调用 LLM，避免单次 prompt 过长导致尾部 beat 被截断丢失；
- story_scene_index 由本地自增统一编号，不信任 LLM 返回值，防止重复/跳号；
- characters 直接从子 beat 聚合（beat 已做角色白名单），保证与 face_cluster 体系一致；
- 通过 _finalize_story_scenes 保证每个 beat 恰好归属一个 StoryScene。
"""
import json
import time

import config
from models.schemas import Shot, Beat, StoryScene
from utils import group_consecutive
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("StorySceneDetect")

# 单次送入 LLM 的最大 beat 数量，超过则分段（防止长视频 prompt 截断）
SEGMENT_SIZE = 40

SCENE_PROMPT_TEMPLATE = """你是一个专业的影视叙事分析师。请将以下"剧情节拍"（Beat）序列聚合为"故事场景"（StoryScene）。

一个 StoryScene 是由若干连续 Beat 组成的完整叙事场景，通常对应：
- 同一个地点/环境中发生的一系列事件
- 一段完整的情节单元（开始 → 发展 → 结束）
- 一个戏剧冲突的完整过程

=== Beat 列表 ===
{beats_info}

请将这些 Beat 分组为若干 StoryScene，输出 JSON 数组。每个 StoryScene 包含：
- beat_indices: 包含的 Beat 索引列表（必须连续）
- location: 场景地点/环境描述
- description: 这个场景的核心内容（一两句话）
- plot_function: 叙事功能（setup / inciting_incident / rising / climax / falling / resolution / epilogue）

规则：
1. 地点或情境发生大变化时，开启新 StoryScene
2. 一个 StoryScene 通常包含 2-6 个 Beat
3. 所有 Beat 都必须被分配到某个 StoryScene

只输出 JSON：
```json
[
  {{
    "beat_indices": [0, 1, 2],
    "location": "咖啡厅",
    "description": "男女主角在咖啡厅重逢并讨论过去的误会",
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
        _backfill_scene_to_shots(shots, loaded_scenes, video_id)
        return loaded_scenes

    if not beats:
        logger.warning("无 Beat 数据，跳过 StoryScene 检测")
        return []

    logger.info(f"开始 StoryScene 检测: {len(beats)} 个 beat")

    client = get_llm_client()

    # 分段调用 LLM，避免长视频单次 prompt 过长
    story_scenes: list[StoryScene] = []
    for seg_start in range(0, len(beats), SEGMENT_SIZE):
        seg_beats = beats[seg_start: seg_start + SEGMENT_SIZE]
        story_scenes.extend(_detect_segment(client, seg_beats))
        if seg_start + SEGMENT_SIZE < len(beats):
            time.sleep(0.5)

    # 覆盖兜底 + 全局重排：保证每个 beat 恰好归属一个 StoryScene，索引连续唯一
    story_scenes = _finalize_story_scenes(story_scenes, beats)

    # 回填 shot 的 story_scene_index 并持久化
    _backfill_scene_to_shots(shots, story_scenes, video_id)

    scenes_path.write_text(
        json.dumps([s.model_dump() for s in story_scenes], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"StoryScene 检测完成: {len(story_scenes)} 个故事场景")
    return story_scenes


def _detect_segment(client, seg_beats: list[Beat]) -> list[StoryScene]:
    """对一段 beats 调用 LLM 聚合为 StoryScene；失败时退回默认分组。"""
    beat_lines = []
    for b in seg_beats:
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

    prompt = SCENE_PROMPT_TEMPLATE.format(beats_info="\n".join(beat_lines))

    try:
        response = client.chat(prompt=prompt, temperature=0.3)
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, list):
            logger.warning("StoryScene 检测解析失败，使用默认分组")
            return _fallback_story_scenes(seg_beats)

        beat_map = {b.beat_index: b for b in seg_beats}
        scenes = []
        for item in parsed:
            beat_indices = item.get("beat_indices", [])
            scene_beats = [beat_map[bi] for bi in beat_indices if bi in beat_map]
            if not scene_beats:
                continue
            scenes.append(_build_scene(scene_beats, item))
        return scenes if scenes else _fallback_story_scenes(seg_beats)
    except Exception as e:
        logger.warning(f"StoryScene 检测失败: {e}，使用默认分组")
        return _fallback_story_scenes(seg_beats)


def _build_scene(scene_beats: list[Beat], item: dict) -> StoryScene:
    """由一组 beat 构造 StoryScene；characters 从 beat 聚合，不信任 LLM。"""
    return StoryScene(
        story_scene_index=-1,  # 占位，最终由 _finalize_story_scenes 统一重排
        start_time=min(b.start_time for b in scene_beats),
        end_time=max(b.end_time for b in scene_beats),
        duration=(
            max(b.end_time for b in scene_beats)
            - min(b.start_time for b in scene_beats)
        ),
        beat_indices=sorted(b.beat_index for b in scene_beats),
        shot_indices=sorted({si for b in scene_beats for si in b.shot_indices}),
        location=item.get("location", ""),
        description=item.get("description", ""),
        characters=sorted({c for b in scene_beats for c in b.characters}),
        plot_function=item.get("plot_function", ""),
    )


def _finalize_story_scenes(
    scenes: list[StoryScene], beats: list[Beat],
) -> list[StoryScene]:
    """
    规范化 StoryScene，使其对 beats 构成完整且不重叠的划分：

    1. 跨场景去重 beat（保留先出现者），过滤非法 beat 索引；
    2. 未覆盖的 beat 按相邻关系聚合为 ``transition`` 场景；
    3. 按时间统一排序，重排 story_scene_index 为连续唯一值，重算时间/时长/聚合字段。
    """
    if not beats:
        return []
    beat_map = {b.beat_index: b for b in beats}

    scenes = [ss for ss in scenes if ss.beat_indices]
    scenes.sort(key=lambda ss: ss.start_time)
    seen: set[int] = set()
    for ss in scenes:
        kept = [bi for bi in ss.beat_indices if bi in beat_map and bi not in seen]
        seen.update(kept)
        ss.beat_indices = sorted(kept)
    scenes = [ss for ss in scenes if ss.beat_indices]

    uncovered = [b for b in beats if b.beat_index not in seen]
    for group in group_consecutive(uncovered, lambda b: b.beat_index):
        scenes.append(StoryScene(
            story_scene_index=-1,
            start_time=group[0].start_time,
            end_time=group[-1].end_time,
            duration=group[-1].end_time - group[0].start_time,
            beat_indices=[b.beat_index for b in group],
            shot_indices=sorted({si for b in group for si in b.shot_indices}),
            characters=sorted({c for b in group for c in b.characters}),
            plot_function="transition",
        ))

    scenes.sort(key=lambda ss: min(beat_map[bi].start_time for bi in ss.beat_indices))
    for i, ss in enumerate(scenes):
        bs = [beat_map[bi] for bi in ss.beat_indices]
        ss.story_scene_index = i
        ss.beat_indices = sorted(b.beat_index for b in bs)
        ss.shot_indices = sorted({si for b in bs for si in b.shot_indices})
        ss.start_time = min(b.start_time for b in bs)
        ss.end_time = max(b.end_time for b in bs)
        ss.duration = ss.end_time - ss.start_time
        # 始终从当前拥有的 beats 重新聚合角色，防止去重后残留已不属于本场景的 beat 角色
        ss.characters = sorted({c for b in bs for c in b.characters})
    return scenes


def _backfill_scene_to_shots(
    shots: list[Shot], story_scenes: list[StoryScene], video_id: str,
):
    """回填 shot.story_scene_index 并持久化到 scenes.json"""
    shot_map = {s.scene_index: s for s in shots}
    for ss in story_scenes:
        for si in ss.shot_indices:
            if si in shot_map:
                shot_map[si].story_scene_index = ss.story_scene_index
    shots_json = config.VIDEOS_DIR / video_id / "scenes" / "scenes.json"
    if shots_json.exists():
        shots_json.write_text(
            json.dumps([s.model_dump() for s in shots], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _fallback_story_scenes(beats: list[Beat]) -> list[StoryScene]:
    """当 LLM 失败时，每 3 个 beat 为一个 story scene（索引交由 finalize 统一重排）"""
    story_scenes = []
    group_size = 3
    for i in range(0, len(beats), group_size):
        group = beats[i: i + group_size]
        story_scenes.append(StoryScene(
            story_scene_index=-1,
            start_time=group[0].start_time,
            end_time=group[-1].end_time,
            duration=group[-1].end_time - group[0].start_time,
            beat_indices=[b.beat_index for b in group],
            shot_indices=sorted({si for b in group for si in b.shot_indices}),
            characters=sorted({c for b in group for c in b.characters}),
        ))
    return story_scenes
