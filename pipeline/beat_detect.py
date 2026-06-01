# -*- coding: utf-8 -*-
"""
剧情节拍检测（v2 新增）

将连续 shots 按叙事节拍聚合为 beats。
Beat 是介于 shot 和 story_scene 之间的叙事微单元，
例如：一段对话、一个动作序列、一个情绪转折。

使用 LLM 分析 shot 的台词、画面摘要和人物信息，判断哪些
连续 shots 属于同一个 beat。
"""
import json
import time
from pathlib import Path

import config
from models.schemas import (
    Shot, Beat, TranscriptSegment, VisionSummary, Character,
)
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("BeatDetect")

BEAT_PROMPT_TEMPLATE = """你是一个专业的影视叙事分析师。请分析以下镜头序列，将连续镜头按"叙事节拍"（Beat）分组。

一个 Beat 是由若干连续镜头组成的叙事微单元，通常对应：
- 一段完整的对话
- 一个连续的动作序列
- 一个情绪转折过程
- 一段环境展示/空镜
- 一个蒙太奇段落

=== 镜头列表 ===
{shots_info}

=== 已知角色名册 ===
（characters 字段只能填写以下角色 ID；画面中出现但不在名册内的人，用 "unknown_1"/"unknown_2" 等临时编号区分，不要编造新的 char_ ID）
{character_roster}

请将这些镜头分组为若干 Beat，输出 JSON 数组。每个 Beat 包含：
- beat_index: 从 {beat_offset} 开始编号
- shot_indices: 包含的镜头索引列表（必须连续）
- beat_type: 类型（setup / confrontation / resolution / transition / montage / dialogue / action / reveal）
- description: 这个节拍讲了什么（一句话）
- emotion: 主要情绪
- intensity: 戏剧强度 (0.0 - 1.0)
- characters: 涉及的人物 ID 列表

规则：
1. 相邻且叙事连贯的镜头归为同一 Beat
2. 当场景/话题/情绪发生明显转换时，开启新 Beat
3. 每个 Beat 通常包含 2-8 个镜头，但不强制
4. 所有镜头都必须被分配到某个 Beat
5. characters 字段只能填写"已知角色名册"中的 ID；无法对应名册的人用 "unknown_N" 临时编号，不要编造新的 char_ ID

只输出 JSON：
```json
[
  {{
    "beat_index": {beat_offset},
    "shot_indices": [{example_shot_indices}],
    "beat_type": "dialogue",
    "description": "男女主角在咖啡厅讨论计划",
    "emotion": "轻松",
    "intensity": 0.3,
    "characters": ["char_000", "char_001"]
  }}
]
```
"""


def detect_beats(
    video_id: str,
    shots: list[Shot],
    transcripts: list[TranscriptSegment],
    vision_summaries: list[VisionSummary],
    characters: list[Character],
) -> list[Beat]:
    """
    将连续 shots 按叙事节拍聚合为 beats。

    Args:
        video_id: 视频 ID
        shots: 镜头列表
        transcripts: 台词列表
        vision_summaries: 画面摘要列表
        characters: 人物列表

    Returns:
        Beat 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    beats_path = video_dir / "beats.json"

    # 如果已存在，直接加载（仍需回填 shot 的 beat_index）
    if beats_path.exists():
        logger.info(f"Beat 检测结果已存在，直接加载: {beats_path}")
        data = json.loads(beats_path.read_text(encoding="utf-8"))
        loaded_beats = [Beat(**b) for b in data]
        # 缓存加载也需回填 shot 的反向链接
        _backfill_beat_to_shots(shots, loaded_beats, video_id)
        return loaded_beats

    logger.info(f"开始 Beat 检测: {len(shots)} 个镜头")

    client = get_llm_client()
    all_beats = []

    # 构建已知角色名册（供 LLM 在 characters 字段引用，避免编造 ID）
    valid_char_ids = set()
    roster_lines = []
    for c in (characters or []):
        cid = getattr(c, "character_id", None)
        if not cid:
            continue
        valid_char_ids.add(cid)
        cname = getattr(c, "display_name", "") or cid
        cdesc = (getattr(c, "description", "") or "").strip()
        line = f"- {cid}: {cname}"
        if cdesc:
            line += f" — {cdesc[:40]}"
        roster_lines.append(line)
    character_roster = (
        "\n".join(roster_lines)
        if roster_lines
        else "（暂无已知角色，可用 unknown_1/unknown_2 临时标注）"
    )

    # 构建 shot → 台词/画面 的索引
    trans_by_shot = {}
    for t in transcripts:
        if t.scene_index >= 0:
            trans_by_shot.setdefault(t.scene_index, []).append(t)
    vision_by_shot = {v.scene_index: v for v in vision_summaries}

    # 分段处理（每次最多 30 个 shot）
    segment_size = 30
    beat_offset = 0

    for seg_start in range(0, len(shots), segment_size):
        seg_shots = shots[seg_start: seg_start + segment_size]
        logger.info(
            f"  处理段: shot {seg_shots[0].scene_index}-{seg_shots[-1].scene_index}"
        )

        # 构造 shot 信息
        shot_lines = []
        for s in seg_shots:
            parts = [f"Shot {s.scene_index} [{s.start_time:.1f}s-{s.end_time:.1f}s]"]

            # 台词
            trans = trans_by_shot.get(s.scene_index, [])
            if trans:
                trans_text = " ".join([t.text[:50] for t in trans[:3]])
                speaker = trans[0].speaker or "?"
                parts.append(f"台词[{speaker}]: {trans_text}")

            # 画面
            vis = vision_by_shot.get(s.scene_index)
            if vis:
                parts.append(f"画面: {vis.description[:60]}")
                if vis.mood:
                    parts.append(f"情绪: {vis.mood}")
                if vis.scene_type:
                    parts.append(f"类型: {vis.scene_type}")

            shot_lines.append(" | ".join(parts))

        shots_info = "\n".join(shot_lines)

        # 构建示例索引（使用当前段的实际 scene_index）
        example_indices = ", ".join(str(s.scene_index) for s in seg_shots[:3])
        prompt = BEAT_PROMPT_TEMPLATE.format(
            shots_info=shots_info,
            beat_offset=beat_offset,
            example_shot_indices=example_indices,
            character_roster=character_roster,
        )

        try:
            response = client.chat(prompt=prompt, temperature=0.3)
            parsed = client.parse_json(response)
            if not parsed or not isinstance(parsed, list):
                logger.warning("Beat 检测解析失败，使用默认分组")
                beats = _fallback_beats(seg_shots, beat_offset)
            else:
                beats = []
                # 局部索引→全局索引映射（防御 LLM 输出局部编号）
                local_to_global = {i: s.scene_index for i, s in enumerate(seg_shots)}
                for item in parsed:
                    shot_indices = item.get("shot_indices", [])
                    if not shot_indices:
                        continue
                    # 尝试全局索引，失败则尝试局部索引映射
                    shot_indices = [local_to_global.get(si, si) for si in shot_indices]
                    # 计算时间范围
                    beat_shots = [s for s in seg_shots if s.scene_index in shot_indices]
                    if not beat_shots:
                        continue
                    # 校验 LLM 返回的角色 ID：有名册时只保留名册内 ID 或 unknown_N，
                    # 无名册时保留非空字符串（无法校验），避免编造 char_ ID 污染下游。
                    raw_chars = item.get("characters", []) or []
                    if valid_char_ids:
                        beat_chars = [
                            cid for cid in raw_chars
                            if isinstance(cid, str)
                            and (cid in valid_char_ids or cid.startswith("unknown_"))
                        ]
                    else:
                        beat_chars = [
                            cid for cid in raw_chars
                            if isinstance(cid, str) and cid.strip()
                        ]
                    b = Beat(
                        # beat_index 用全局自增重编号，忽略 LLM 返回值以保证唯一
                        beat_index=beat_offset + len(beats),
                        start_time=min(s.start_time for s in beat_shots),
                        end_time=max(s.end_time for s in beat_shots),
                        duration=sum(s.duration for s in beat_shots),
                        shot_indices=sorted(shot_indices),
                        beat_type=item.get("beat_type", ""),
                        description=item.get("description", ""),
                        emotion=item.get("emotion", ""),
                        intensity=float(item.get("intensity", 0.0)),
                        characters=beat_chars,
                    )
                    beats.append(b)
        except Exception as e:
            logger.warning(f"Beat 检测失败: {e}，使用默认分组")
            beats = _fallback_beats(seg_shots, beat_offset)

        all_beats.extend(beats)
        beat_offset += len(beats)
        time.sleep(0.5)

    # 回填 shot 的 beat_index 并持久化
    _backfill_beat_to_shots(shots, all_beats, video_id)

    # 保存 beats.json
    beats_path.write_text(
        json.dumps([b.model_dump() for b in all_beats], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(f"Beat 检测完成: {len(all_beats)} 个 beat")
    return all_beats


def _backfill_beat_to_shots(shots: list[Shot], beats: list[Beat], video_id: str):
    """回填 shot.beat_index 并持久化到 scenes.json"""
    beat_shot_map = {}
    for b in beats:
        for si in b.shot_indices:
            beat_shot_map[si] = b.beat_index
    for s in shots:
        s.beat_index = beat_shot_map.get(s.scene_index)
    # 持久化反向链接
    scenes_json = config.VIDEOS_DIR / video_id / "scenes" / "scenes.json"
    if scenes_json.exists():
        scenes_json.write_text(
            json.dumps([s.model_dump() for s in shots], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _fallback_beats(shots: list[Shot], offset: int) -> list[Beat]:
    """当 LLM 失败时，按每 3-5 个 shot 一组做默认分组"""
    beats = []
    group_size = 4
    for i in range(0, len(shots), group_size):
        group = shots[i: i + group_size]
        b = Beat(
            beat_index=offset + len(beats),
            start_time=group[0].start_time,
            end_time=group[-1].end_time,
            duration=sum(s.duration for s in group),
            shot_indices=[s.scene_index for s in group],
            beat_type="unknown",
            description="",
        )
        beats.append(b)
    return beats
