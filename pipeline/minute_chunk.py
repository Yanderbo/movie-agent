# -*- coding: utf-8 -*-
"""
分钟级融合理解（v4.1 核心 — Step 5）

将连续 shot 拼接为 ~2-3min 的 MinuteChunk，一次性送入 Gemini
完成 ASR + Vision + Audio + 角色标注 + 跨shot分析。
替代原 step4(ASR) + step5(vision) + step6(audio) + step7(character)
     + step8(speaker_bind) + step9(multimodal_align)

策略：自底向上拼接 → Gemini 融合理解 → 自顶向下回填
"""
from __future__ import annotations
import json
import time
from pathlib import Path
from collections import defaultdict

import config
from models.schemas import (
    Shot, MinuteChunk, CharacterGallery, CharacterProfile,
    TranscriptSegment, OCRResult, VisionSummary, AudioProsody,
    MultimodalAlignment,
)
from utils.llm_client import get_llm_client
from utils.ffmpeg_utils import extract_video_segment
from utils.logger import get_logger

logger = get_logger("MinuteChunk")


# ═══════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════

CHUNK_UNDERSTAND_PROMPT = """你是一个专业的影视分析系统。请分析这段视频片段（{start:.1f}s - {end:.1f}s），该片段包含 {n_shots} 个镜头。

== 镜头边界 ==
{shot_boundaries}

{character_section}

请完成以下分析，以 JSON 格式输出：

A. **ASR 语音转录** — 逐句转录音频中的语音
   - 标注相对于片段起始的时间戳
   - 说话人用角色ID标注（如 char_000），无法匹配已知角色则用 "unknown_1", "unknown_2" 等临时编号区分不同人
   - type: dialogue / narration / voiceover

B. **逐镜头画面分析** — 对每个镜头（按上方边界），分析：
   - description: 画面描述
   - objects: 检测到的物体
   - mood: 情绪
   - scene_type: 场景类型
   - camera_motion: 镜头运动 (static/pan/tilt/zoom/tracking/handheld)
   - shot_scale: 景别 (close_up/medium/long 等)
   - action_description: 动作描述
   - ocr_texts: 画面文字
   - characters_present: 出现的角色ID列表

C. **逐镜头音频特征**
   - has_music, music_mood, has_sfx, sfx_tags
   - silence_ratio(0-1), speech_rate(slow/normal/fast)
   - volume_peak(0-1), speech_emotion

D. **角色动态更新**
   - 已知角色的新信息：新称呼/别名、形象变化、关键行为
   - 新发现的非人类实体：名称和简述（动物/机器人等）

E. **跨镜头分析**
   - narrative_continuity: 叙事脉络
   - emotion_arc: 情绪变化
   - suggested_beats: 建议的节拍分组（哪些连续镜头属于同一叙事节拍），用镜头索引表示

输出 JSON（只输出JSON）：
```json
{{
  "transcripts": [
    {{"start_time": 0.0, "end_time": 3.5, "text": "...", "speaker": "char_000", "type": "dialogue"}}
  ],
  "per_shot": [
    {{
      "scene_index": {first_shot_index},
      "vision": {{"description": "", "objects": [], "mood": "", "scene_type": "", "camera_motion": "", "shot_scale": "", "action_description": "", "ocr_texts": []}},
      "audio": {{"has_music": false, "music_mood": "", "has_sfx": false, "sfx_tags": [], "silence_ratio": 0, "speech_rate": "", "volume_peak": 0, "speech_emotion": ""}},
      "characters_present": []
    }}
  ],
  "character_updates": [
    {{"character_id": "char_000", "new_names": [], "appearance_change": "", "key_action": ""}}
  ],
  "cross_shot": {{
    "narrative_continuity": "",
    "emotion_arc": "",
    "suggested_beats": [[0, 1, 2], [3, 4]]
  }}
}}
```"""

CHARACTER_SECTION_TEMPLATE = """== 已知角色脸谱 ==
以下附图为已识别角色的参考脸谱（按 {gallery_labels} 的顺序排列）。
请在分析时用对应的角色ID标注人物。

== 已知角色档案 ==
{profiles_text}

如果画面中出现不在已知角色列表中的人物，标注为 "unknown_1", "unknown_2" 等临时编号。
如果出现非人类实体（动物/机器人等），在 character_updates 中报告。"""

NO_CHARACTER_SECTION = """== 角色信息 ==
尚无已知角色。请在分析中自行识别并命名画面中的人物。
说话人用 "unknown_1", "unknown_2" 等临时标注。
在 character_updates 中描述发现的人物外观。"""


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def run_minute_chunk_understand(
    video_id: str,
    video_path: str,
    shots: list[Shot],
    galleries: list[CharacterGallery],
) -> dict:
    """
    分钟级融合理解主流程。

    Args:
        video_id: 视频 ID
        video_path: 压缩后视频路径
        shots: 镜头列表
        galleries: 角色脸谱列表

    Returns:
        dict with keys: transcripts, ocr_results, vision_summaries,
        audio_prosodies, alignments, characters_profiles, speaker_map, chunks
    """
    video_dir = config.VIDEOS_DIR / video_id
    chunks_path = video_dir / "minute_chunks.json"

    # 检查是否已完成（通过回填产物判断）
    if _all_outputs_exist(video_dir):
        logger.info("分钟级融合理解结果已存在，直接加载")
        return _load_all_outputs(video_dir)

    logger.info(f"开始分钟级融合理解: {len(shots)} 个镜头")

    # Step 1: 构建 MinuteChunks
    chunks = build_minute_chunks(shots)
    logger.info(f"构建 {len(chunks)} 个 MinuteChunk")

    # Step 2: 初始化角色档案
    profiles = _init_profiles(galleries)

    # Step 3: 逐 chunk 处理
    client = get_llm_client()
    chunks_dir = video_dir / "chunk_segments"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_transcripts = []
    all_ocr = []
    all_vision = []
    all_audio = []
    all_alignments = []
    processed_chunks = []

    for chunk in chunks:
        logger.info(
            f"  Chunk {chunk.chunk_index}: "
            f"{chunk.start_time:.1f}s-{chunk.end_time:.1f}s "
            f"({len(chunk.shot_indices)} shots, {chunk.duration:.0f}s)"
        )

        # 提取视频片段
        segment_path = str(chunks_dir / f"chunk_{chunk.chunk_index:04d}.mp4")
        if not Path(segment_path).exists():
            try:
                extract_video_segment(
                    video_path, segment_path,
                    chunk.start_time, chunk.end_time,
                )
            except Exception as e:
                logger.warning(f"  视频片段提取失败: {e}")
                segment_path = None

        # 收集关键帧路径
        chunk_shots = [s for s in shots if s.scene_index in chunk.shot_indices]
        keyframe_paths = _collect_keyframes(chunk_shots)

        # 收集角色脸谱图片（身份脸谱 + 上一轮新增）
        gallery_paths, gallery_labels = _collect_gallery_images(
            galleries, profiles, chunk.chunk_index
        )

        # 构建 prompt
        prompt = _build_prompt(chunk, chunk_shots, profiles, gallery_labels)

        # 构建媒体文件列表：视频 + 关键帧 + 脸谱
        media_paths = []
        if segment_path and Path(segment_path).exists():
            media_paths.append(segment_path)
        media_paths.extend(keyframe_paths)
        media_paths.extend(gallery_paths)

        # 调用 Gemini
        result = _call_gemini(client, prompt, media_paths)

        if result:
            # 回填到 shot 级数据
            t, o, v, a, al = _backfill_chunk_result(
                result, chunk, chunk_shots, profiles
            )
            all_transcripts.extend(t)
            all_ocr.extend(o)
            all_vision.extend(v)
            all_audio.extend(a)
            all_alignments.extend(al)

            # 更新角色档案
            _update_profiles(profiles, result, chunk.chunk_index)

            # 保存 chunk 原始结果
            chunk.raw_transcripts = result.get("transcripts", [])
            chunk.per_shot_vision = [s.get("vision", {}) for s in result.get("per_shot", [])]
            chunk.per_shot_audio = [s.get("audio", {}) for s in result.get("per_shot", [])]
            chunk.character_updates = result.get("character_updates", [])
            chunk.cross_shot_analysis = result.get("cross_shot", {})
            chunk.suggested_beats = result.get("cross_shot", {}).get("suggested_beats", [])
        else:
            logger.warning(f"  Chunk {chunk.chunk_index} 处理失败，跳过")

        processed_chunks.append(chunk)
        time.sleep(1)

    # 补全未覆盖的 shot
    covered_indices = {v.scene_index for v in all_vision}
    for s in shots:
        if s.scene_index not in covered_indices:
            all_vision.append(VisionSummary(scene_index=s.scene_index, timestamp=s.start_time, description=""))
            all_ocr.append(OCRResult(scene_index=s.scene_index, timestamp=s.start_time))
            all_audio.append(AudioProsody(scene_index=s.scene_index))

    # 排序
    all_transcripts.sort(key=lambda t: t.start_time)
    all_ocr.sort(key=lambda o: o.scene_index)
    all_vision.sort(key=lambda v: v.scene_index)
    all_audio.sort(key=lambda a: a.scene_index)
    all_alignments.sort(key=lambda a: a.scene_index)

    # 生成 speaker_map（从角色标注直接派生）
    speaker_map = _build_speaker_map(all_transcripts)

    # 保存所有产物
    _save_all_outputs(
        video_dir, all_transcripts, all_ocr, all_vision,
        all_audio, all_alignments, profiles, speaker_map, processed_chunks,
        galleries=galleries,
    )

    logger.info(
        f"分钟级融合理解完成: {len(all_transcripts)} 句台词, "
        f"{len(all_vision)} 个画面, {len(profiles)} 个角色"
    )

    return {
        "transcripts": all_transcripts,
        "ocr_results": all_ocr,
        "vision_summaries": all_vision,
        "audio_prosodies": all_audio,
        "alignments": all_alignments,
        "character_profiles": profiles,
        "speaker_map": speaker_map,
        "chunks": processed_chunks,
    }


# ═══════════════════════════════════════════════════════════════
# Chunk 构建
# ═══════════════════════════════════════════════════════════════

def build_minute_chunks(shots: list[Shot]) -> list[MinuteChunk]:
    """以 shot 边界为切点，拼接为 ~2-3min 的 chunk。"""
    target = config.CHUNK_TARGET_DURATION
    merge_threshold = config.CHUNK_MERGE_THRESHOLD

    chunks = []
    current_shots = []
    current_duration = 0.0

    for shot in shots:
        current_shots.append(shot)
        current_duration += shot.duration

        if current_duration >= target:
            chunks.append(_make_chunk(len(chunks), current_shots, current_duration))
            current_shots = []
            current_duration = 0.0

    # 处理剩余
    if current_shots:
        if chunks and current_duration < merge_threshold:
            # 太短，合并到上一个
            last = chunks[-1]
            for s in current_shots:
                last.shot_indices.append(s.scene_index)
            last.end_time = current_shots[-1].end_time
            last.duration = last.end_time - last.start_time
        else:
            chunks.append(_make_chunk(len(chunks), current_shots, current_duration))

    return chunks


def _make_chunk(idx, shots, duration):
    return MinuteChunk(
        chunk_index=idx,
        shot_indices=[s.scene_index for s in shots],
        start_time=shots[0].start_time,
        end_time=shots[-1].end_time,
        duration=duration,
    )


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _init_profiles(galleries: list[CharacterGallery]) -> dict[str, CharacterProfile]:
    """从脸谱初始化角色档案"""
    profiles = {}
    for g in galleries:
        if g.tier == "passerby":
            continue
        profiles[g.character_id] = CharacterProfile(
            character_id=g.character_id,
            tier=g.tier,
            gallery_ref=g.character_id,
        )
    return profiles


def _collect_keyframes(chunk_shots: list[Shot]) -> list[str]:
    """收集 chunk 内所有 shot 的首帧关键帧"""
    paths = []
    for s in chunk_shots:
        kf = s.keyframe_path
        if not kf and s.keyframe_paths:
            kf = s.keyframe_paths[0] if s.keyframe_paths else None
        if kf and Path(kf).exists():
            paths.append(kf)
    return paths


def _collect_gallery_images(
    galleries: list[CharacterGallery],
    profiles: dict,
    chunk_index: int,
) -> tuple[list[str], list[str]]:
    """收集角色脸谱图片（身份识别用）+ 上一轮新增"""
    all_paths = []
    labels = []
    for g in galleries:
        if g.tier == "passerby":
            continue
        # 每角色取前2张用于身份识别（减少输入量）
        gallery_imgs = [p for p in g.gallery_paths[:2] if Path(p).exists()]
        if gallery_imgs:
            all_paths.extend(gallery_imgs)
            profile = profiles.get(g.character_id)
            name = profile.names[0] if profile and profile.names else g.character_id
            labels.append(f"{g.character_id}({name}): {len(gallery_imgs)}张")

    return all_paths, labels


def _build_prompt(chunk, chunk_shots, profiles, gallery_labels):
    """构建 Gemini prompt"""
    # 镜头边界
    boundaries = []
    for s in chunk_shots:
        boundaries.append(
            f"Shot {s.scene_index}: {s.start_time:.1f}s-{s.end_time:.1f}s ({s.duration:.1f}s)"
        )

    # 角色部分
    if profiles:
        profiles_lines = []
        for cid, p in profiles.items():
            name = p.names[0] if p.names else cid
            desc = p.description or "（暂无描述）"
            profiles_lines.append(f"- {cid} ({name}): {desc}")
        char_section = CHARACTER_SECTION_TEMPLATE.format(
            gallery_labels=", ".join(gallery_labels),
            profiles_text="\n".join(profiles_lines),
        )
    else:
        char_section = NO_CHARACTER_SECTION

    return CHUNK_UNDERSTAND_PROMPT.format(
        start=chunk.start_time,
        end=chunk.end_time,
        n_shots=len(chunk.shot_indices),
        shot_boundaries="\n".join(boundaries),
        character_section=char_section,
        first_shot_index=chunk.shot_indices[0] if chunk.shot_indices else 0,
    )


def _call_gemini(client, prompt, media_paths) -> dict | None:
    """调用 Gemini API 并解析结果"""
    try:
        if media_paths:
            response = client.chat_with_multi_media(
                prompt=prompt,
                media_paths=media_paths,
                temperature=0.2,
            )
        else:
            response = client.chat(prompt=prompt, temperature=0.2)

        parsed = client.parse_json(response)
        if parsed and isinstance(parsed, dict):
            return parsed
        logger.warning("Gemini 响应解析失败")
        return None
    except Exception as e:
        logger.error(f"Gemini 调用失败: {e}")
        return None


def _normalize_character_id(raw_id: str, profiles: dict) -> str | None:
    """将各种形式的人物标识规范化为 char_ 前缀的 ID。

    规则:
      - char_XXX → 直接返回
      - entity_XXX → 直接返回
      - unknown_N → char_tmp_unknown_N，并在 profiles 中注册
      - unknown / 空 / 其他 → None
    """
    raw_id = str(raw_id or "").strip()
    if not raw_id or raw_id == "unknown":
        return None
    if raw_id.startswith(("char_", "entity_")):
        return raw_id
    if raw_id.startswith("unknown_"):
        normalized = f"char_tmp_{raw_id}"
        if normalized not in profiles:
            profiles[normalized] = CharacterProfile(
                character_id=normalized,
                names=[raw_id],
                tier="minor",
                description="临时标注人物（无脸谱匹配）",
            )
        return normalized
    return None


def _backfill_chunk_result(result, chunk, chunk_shots, profiles):
    """将 chunk 结果回填到 shot 级数据结构"""
    transcripts = []
    ocr_results = []
    vision_summaries = []
    audio_prosodies = []
    alignments = []

    shot_map = {s.scene_index: s for s in chunk_shots}
    # 构建局部索引→全局索引映射（防御 LLM 输出局部编号）
    local_to_global = {i: s.scene_index for i, s in enumerate(chunk_shots)}

    # 回填 ASR
    for item in result.get("transcripts", []):
        seg_start = round(float(item.get("start_time", 0)) + chunk.start_time, 1)
        seg_end = round(float(item.get("end_time", 0)) + chunk.start_time, 1)
        text = item.get("text", "").strip()
        if not text:
            continue

        speaker = item.get("speaker", "unknown")
        char_id = _normalize_character_id(speaker, profiles)

        # 按中点分配 scene_index
        mid = (seg_start + seg_end) / 2.0
        scene_index = -1
        for s in chunk_shots:
            if s.start_time <= mid < s.end_time:
                scene_index = s.scene_index
                break
        if scene_index == -1 and chunk_shots:
            scene_index = min(chunk_shots, key=lambda s: abs(s.start_time - mid)).scene_index

        seg = TranscriptSegment(
            start_time=seg_start, end_time=seg_end, text=text,
            speaker=speaker, character_id=char_id,
            scene_index=scene_index,
            transcript_type=item.get("type", "dialogue"),
            cross_shot=(seg_start < shot_map.get(scene_index, chunk_shots[0]).start_time
                        if scene_index in shot_map else False),
        )
        transcripts.append(seg)

    # 回填 Vision / Audio / OCR
    for item in result.get("per_shot", []):
        si = item.get("scene_index", -1)
        # 尝试全局索引，失败则尝试局部索引映射
        if si not in shot_map and si in local_to_global:
            si = local_to_global[si]
        if si not in shot_map:
            continue
        shot = shot_map[si]

        vis = item.get("vision", {})
        vision_summaries.append(VisionSummary(
            scene_index=si, timestamp=shot.start_time,
            description=vis.get("description", ""),
            objects=vis.get("objects", []),
            mood=vis.get("mood", ""),
            scene_type=vis.get("scene_type", ""),
            camera_motion=vis.get("camera_motion", ""),
            shot_scale=vis.get("shot_scale", ""),
            action_description=vis.get("action_description", ""),
            props=vis.get("props", []),
        ))

        ocr_results.append(OCRResult(
            scene_index=si, timestamp=shot.start_time,
            texts=vis.get("ocr_texts", []),
        ))

        aud = item.get("audio", {})
        audio_prosodies.append(AudioProsody(
            scene_index=si,
            has_music=bool(aud.get("has_music", False)),
            music_mood=aud.get("music_mood", ""),
            has_sfx=bool(aud.get("has_sfx", False)),
            sfx_tags=aud.get("sfx_tags", []),
            silence_ratio=float(aud.get("silence_ratio", 0)),
            speech_rate=aud.get("speech_rate", ""),
            volume_peak=float(aud.get("volume_peak", 0)),
            speech_emotion=aud.get("speech_emotion", ""),
        ))

        # 多模态对齐（从结果直接派生）
        chars_present = []
        for c in item.get("characters_present", []):
            normalized = _normalize_character_id(c, profiles)
            if normalized:
                chars_present.append(normalized)
        shot_trans = [t for t in transcripts if t.scene_index == si]
        speaking = list({t.character_id for t in shot_trans if t.character_id})

        active = []
        if shot_trans:
            active.append("speech")
        if aud.get("has_music"):
            active.append("music")
        if aud.get("has_sfx"):
            active.append("sfx")

        dominant = "speech" if shot_trans else ("music" if aud.get("has_music") else "visual")

        alignments.append(MultimodalAlignment(
            scene_index=si, start_time=shot.start_time, end_time=shot.end_time,
            visible_characters=chars_present,
            speaking_characters=speaking,
            active_modalities=active,
            dominant_modality=dominant,
            alignment_confidence=0.8,
        ))

    return transcripts, ocr_results, vision_summaries, audio_prosodies, alignments


def _update_profiles(profiles, result, chunk_index):
    """根据 chunk 结果更新角色档案"""
    for update in result.get("character_updates", []):
        cid = update.get("character_id", "")
        if not cid:
            continue
        # 规范化 character_id（unknown_X → char_tmp_unknown_X）
        cid = _normalize_character_id(cid, profiles)
        if not cid:
            continue
        if cid in profiles:
            p = profiles[cid]
            new_names = update.get("new_names", [])
            if new_names:
                for n in new_names:
                    if n and n not in p.names:
                        p.names.append(n)
            if update.get("appearance_change"):
                p.appearance_changes.append({
                    "chunk_idx": chunk_index,
                    "description": update["appearance_change"],
                })
            if update.get("key_action"):
                p.key_actions.append({
                    "chunk_idx": chunk_index,
                    "action": update["key_action"],
                })
        else:
            # 新发现的实体
            profiles[cid] = CharacterProfile(
                character_id=cid,
                names=update.get("new_names", []),
                description=update.get("appearance_change", ""),
                tier="minor",
                is_human=not cid.startswith("entity_"),
                entity_type="other" if cid.startswith("entity_") else "human",
            )


def _build_speaker_map(transcripts) -> dict:
    """从角色标注直接生成 speaker_map"""
    speaker_map = {}
    for t in transcripts:
        if t.speaker and t.character_id:
            speaker_map[t.speaker] = t.character_id
    return speaker_map


# ═══════════════════════════════════════════════════════════════
# 持久化
# ═══════════════════════════════════════════════════════════════

def _all_outputs_exist(video_dir):
    """检查所有回填产物是否存在"""
    required = [
        "transcripts.json", "ocr.json", "vision.json",
        "audio_prosody.json", "minute_chunks.json",
        "characters.json", "speaker_map.json",
        "multimodal_alignments.json", "character_profiles.json",
    ]
    if not all((video_dir / f).exists() for f in required):
        return False

    try:
        characters = json.loads((video_dir / "characters.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(characters, list):
        return False

    stable_chars = [
        c for c in characters if str(c.get("character_id", "")).startswith("char_")
        and not str(c.get("character_id", "")).startswith("char_tmp_")
    ]
    return not stable_chars or any(c.get("appearance_scenes") for c in stable_chars)


def _load_all_outputs(video_dir):
    """加载已有产物"""
    def _load_list(filename, model_cls):
        path = video_dir / filename
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return [model_cls(**item) for item in data] if isinstance(data, list) else []
        return []

    profiles = {}
    profiles_path = video_dir / "character_profiles.json"
    if profiles_path.exists():
        data = json.loads(profiles_path.read_text(encoding="utf-8"))
        for item in data:
            p = CharacterProfile(**item)
            profiles[p.character_id] = p

    speaker_map = {}
    map_path = video_dir / "speaker_map.json"
    if map_path.exists():
        speaker_map = json.loads(map_path.read_text(encoding="utf-8"))

    return {
        "transcripts": _load_list("transcripts.json", TranscriptSegment),
        "ocr_results": _load_list("ocr.json", OCRResult),
        "vision_summaries": _load_list("vision.json", VisionSummary),
        "audio_prosodies": _load_list("audio_prosody.json", AudioProsody),
        "alignments": _load_list("multimodal_alignments.json", MultimodalAlignment),
        "character_profiles": profiles,
        "speaker_map": speaker_map,
        "chunks": _load_list("minute_chunks.json", MinuteChunk),
    }


def _save_all_outputs(
    video_dir, transcripts, ocr_results, vision_summaries,
    audio_prosodies, alignments, profiles, speaker_map, chunks,
    galleries=None,
):
    """保存所有产物"""
    def _save(filename, data_list):
        path = video_dir / filename
        path.write_text(
            json.dumps([d.model_dump() for d in data_list], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    _save("transcripts.json", transcripts)
    _save("ocr.json", ocr_results)
    _save("vision.json", vision_summaries)
    _save("audio_prosody.json", audio_prosodies)
    _save("multimodal_alignments.json", alignments)
    _save("minute_chunks.json", chunks)

    # 角色档案
    profiles_path = video_dir / "character_profiles.json"
    profiles_list = [p.model_dump() for p in profiles.values()]
    profiles_path.write_text(
        json.dumps(profiles_list, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # Speaker map
    map_path = video_dir / "speaker_map.json"
    map_path.write_text(
        json.dumps(speaker_map, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # Characters.json（从 profiles 生成，兼容下游）
    from models.schemas import CharacterDeep
    # 从 alignments 和 galleries 提取每个角色出现的 scene_index
    char_scenes = defaultdict(set)
    for al in alignments:
        for cid_vis in al.visible_characters:
            char_scenes[cid_vis].add(al.scene_index)
    if galleries:
        for g in galleries:
            char_scenes[g.character_id].update(g.appearance_scenes)
    chars = []
    for cid, p in profiles.items():
        chars.append(CharacterDeep(
            character_id=cid,
            display_name=p.names[0] if p.names else cid,
            description=p.description,
            role=None,
            appearance_scenes=sorted(char_scenes.get(cid, [])),
        ).model_dump())
    chars_path = video_dir / "characters.json"
    chars_path.write_text(
        json.dumps(chars, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    logger.info(f"所有产物已保存至 {video_dir}")
