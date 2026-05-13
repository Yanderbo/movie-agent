# -*- coding: utf-8 -*-
"""
Speaker ↔ Character 绑定

通过分析 ASR speaker_id 与 Character 在同一 scene 的共现关系，
建立 speaker → character 映射。最终调用 LLM 确认/修正映射。

输出：
- speaker_map.json: {"speaker_1": "char_000", ...}
- 更新 TranscriptSegment.character_id
- 更新 Character.speaker_ids
"""
import json
from collections import defaultdict
from pathlib import Path

import config
from models.schemas import TranscriptSegment, Character, Scene
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("SpeakerBind")

BIND_PROMPT_TEMPLATE = """你是一个专业的视频分析师。请根据以下信息，判断音频中的说话人与画面中的人物的对应关系。

=== 说话人信息 ===
{speakers_info}

=== 人物信息 ===
{characters_info}

=== 共现统计 ===
{cooccurrence_info}

请为每个说话人指定最可能对应的人物。如果无法确定，设为 null。

输出 JSON 对象，key 为说话人 ID，value 为人物 ID 或 null：
```json
{{
  "speaker_1": "char_000",
  "speaker_2": "char_001",
  "speaker_3": null
}}
```
"""


def bind_speakers_to_characters(
    video_id: str,
    transcripts: list[TranscriptSegment],
    characters: list[Character],
    scenes: list[Scene],
) -> tuple[dict[str, str], list[TranscriptSegment], list[Character]]:
    """
    建立 speaker ↔ character 映射。

    流程：
    1. 统计每个 scene 中出现的 speaker_id 和 character_id
    2. 根据共现频率建立候选映射
    3. 调用 LLM 确认/修正映射
    4. 回写到 TranscriptSegment.character_id 和 Character.speaker_ids

    Args:
        video_id: 视频 ID
        transcripts: 台词列表（已带 scene_index）
        characters: 人物列表
        scenes: 镜头列表

    Returns:
        (speaker_map, updated_transcripts, updated_characters)
    """
    video_dir = config.VIDEOS_DIR / video_id
    map_path = video_dir / "speaker_map.json"

    # 如果已存在，直接加载并应用
    if map_path.exists():
        logger.info(f"Speaker 映射已存在，直接加载: {map_path}")
        speaker_map = json.loads(map_path.read_text(encoding="utf-8"))
        transcripts, characters = _apply_mapping(transcripts, characters, speaker_map)
        return speaker_map, transcripts, characters

    # 收集所有 speaker_id
    all_speakers = set()
    for t in transcripts:
        if t.speaker:
            all_speakers.add(t.speaker)

    if not all_speakers:
        logger.info("未检测到任何 speaker，跳过绑定")
        map_path.write_text("{}", encoding="utf-8")
        return {}, transcripts, characters

    if not characters:
        logger.info("未识别任何人物，跳过绑定")
        map_path.write_text("{}", encoding="utf-8")
        return {}, transcripts, characters

    logger.info(f"开始 Speaker 绑定: {len(all_speakers)} 个 speaker, {len(characters)} 个人物")

    # Step 1: 统计共现关系
    # speaker_in_scene[scene_index] = set(speaker_ids)
    speaker_in_scene = defaultdict(set)
    for t in transcripts:
        if t.speaker and t.scene_index >= 0:
            speaker_in_scene[t.scene_index].add(t.speaker)

    # char_in_scene[scene_index] = set(character_ids)
    char_in_scene = defaultdict(set)
    for c in characters:
        for si in c.appearance_scenes:
            char_in_scene[si].add(c.character_id)

    # Step 2: 共现矩阵
    cooccurrence = defaultdict(lambda: defaultdict(int))
    for scene_idx in set(speaker_in_scene.keys()) & set(char_in_scene.keys()):
        for spk in speaker_in_scene[scene_idx]:
            for char_id in char_in_scene[scene_idx]:
                cooccurrence[spk][char_id] += 1

    # Step 3: 构造 prompt 信息
    # 说话人的代表台词
    speaker_samples = defaultdict(list)
    for t in transcripts:
        if t.speaker and len(speaker_samples[t.speaker]) < 3:
            speaker_samples[t.speaker].append(t.text[:50])

    speakers_info_lines = []
    for spk in sorted(all_speakers):
        samples = speaker_samples.get(spk, [])
        sample_text = " | ".join(samples) if samples else "（无台词样本）"
        speakers_info_lines.append(f"- {spk}: 台词样本: {sample_text}")
    speakers_info = "\n".join(speakers_info_lines)

    char_info_lines = []
    for c in characters:
        char_info_lines.append(
            f"- {c.character_id} ({c.display_name}): {c.description}, "
            f"出场 {len(c.appearance_scenes)} 个镜头"
        )
    characters_info = "\n".join(char_info_lines)

    cooccurrence_lines = []
    for spk in sorted(all_speakers):
        if spk in cooccurrence:
            pairs = sorted(cooccurrence[spk].items(), key=lambda x: -x[1])
            pairs_str = ", ".join([f"{cid}({cnt}次)" for cid, cnt in pairs[:5]])
            cooccurrence_lines.append(f"- {spk} 共现: {pairs_str}")
        else:
            cooccurrence_lines.append(f"- {spk} 共现: （无共现数据）")
    cooccurrence_info = "\n".join(cooccurrence_lines)

    prompt = BIND_PROMPT_TEMPLATE.format(
        speakers_info=speakers_info,
        characters_info=characters_info,
        cooccurrence_info=cooccurrence_info,
    )

    # Step 4: LLM 确认
    try:
        client = get_llm_client()
        response = client.chat(prompt=prompt, temperature=0.2)
        parsed = client.parse_json(response)
        if parsed and isinstance(parsed, dict):
            speaker_map = {
                str(k): str(v) if v else None
                for k, v in parsed.items()
            }
            # 过滤掉不存在的 character_id
            valid_char_ids = {c.character_id for c in characters}
            speaker_map = {
                k: v for k, v in speaker_map.items()
                if v is None or v in valid_char_ids
            }
        else:
            logger.warning("LLM 绑定结果解析失败，使用共现频率最高的作为映射")
            speaker_map = _fallback_mapping(all_speakers, cooccurrence)
    except Exception as e:
        logger.warning(f"LLM 绑定调用失败，使用共现频率最高的作为映射: {e}")
        speaker_map = _fallback_mapping(all_speakers, cooccurrence)

    # 去掉 value 为 None 的条目
    speaker_map = {k: v for k, v in speaker_map.items() if v is not None}

    # Step 5: 应用映射
    transcripts, characters = _apply_mapping(transcripts, characters, speaker_map)

    # 保存
    map_path.write_text(
        json.dumps(speaker_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"Speaker 绑定完成: {speaker_map}")

    # 同时更新 transcripts.json 和 characters.json
    transcript_path = video_dir / "transcripts.json"
    transcript_path.write_text(
        json.dumps([t.model_dump() for t in transcripts], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    char_path = video_dir / "characters.json"
    char_path.write_text(
        json.dumps([c.model_dump() for c in characters], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return speaker_map, transcripts, characters


def _fallback_mapping(
    speakers: set[str], cooccurrence: dict
) -> dict[str, str]:
    """当 LLM 失败时，使用共现频率最高的配对作为映射"""
    mapping = {}
    used_chars = set()
    for spk in sorted(speakers):
        if spk in cooccurrence:
            pairs = sorted(cooccurrence[spk].items(), key=lambda x: -x[1])
            for char_id, _ in pairs:
                if char_id not in used_chars:
                    mapping[spk] = char_id
                    used_chars.add(char_id)
                    break
    return mapping


def _apply_mapping(
    transcripts: list[TranscriptSegment],
    characters: list[Character],
    speaker_map: dict[str, str],
) -> tuple[list[TranscriptSegment], list[Character]]:
    """将映射应用到 transcripts 和 characters"""
    # 更新 TranscriptSegment.character_id
    for t in transcripts:
        if t.speaker and t.speaker in speaker_map:
            t.character_id = speaker_map[t.speaker]

    # 更新 Character.speaker_ids
    reverse_map = defaultdict(list)
    for spk, char_id in speaker_map.items():
        reverse_map[char_id].append(spk)
    for c in characters:
        if c.character_id in reverse_map:
            c.speaker_ids = sorted(set(c.speaker_ids + reverse_map[c.character_id]))

    return transcripts, characters
