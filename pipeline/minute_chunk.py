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

== 强制覆盖要求 ==
1. per_shot 必须输出 {n_shots} 个对象，逐一覆盖“镜头边界”中的每个 local_shot_index，不要只分析前几个镜头。
2. 每个 per_shot 对象必须同时包含 local_shot_index 和 scene_index；scene_index 必须使用镜头边界中给出的全局 scene_index。
3. 每个镜头的 vision/audio 都要填写。即使镜头很短或画面较模糊，也要给出最合理的简短描述；确实无法辨认时写“无法判断”，不要留空字符串。
4. ocr_texts 只有在画面没有文字时才可以为空数组。

A. **ASR 语音转录** — 逐句转录音频中的语音
   - start_time/end_time 必须标注相对于片段起始的时间戳，不要使用全片绝对时间戳
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

F. **角色身份合并建议**（仅在你有充分证据时才填写）
   - 如果你发现两个角色ID实际上是同一个人（同一演员），报告合并建议
   - 需要脸部特征、声音、剧情连续性、称呼等多项证据一致才能确认
   - 如果只是"看起来有点像"但不确定，不要报告

输出 JSON（只输出JSON）：
```json
{{
  "transcripts": [
    {{"start_time": 0.0, "end_time": 3.5, "text": "...", "speaker": "char_000", "type": "dialogue"}}
  ],
  "per_shot": [
    {{
      "local_shot_index": 0,
      "scene_index": {first_shot_index},
      "vision": {{"description": "简要描述该镜头画面", "objects": [], "mood": "无法判断", "scene_type": "无法判断", "camera_motion": "static", "shot_scale": "无法判断", "action_description": "简要描述该镜头动作", "ocr_texts": []}},
      "audio": {{"has_music": false, "music_mood": "无法判断", "has_sfx": false, "sfx_tags": [], "silence_ratio": 0, "speech_rate": "normal", "volume_peak": 0, "speech_emotion": "neutral"}},
      "characters_present": []
    }}
  ],
  "character_updates": [
    {{"character_id": "char_000", "new_names": [], "appearance_change": "", "key_action": ""}}
  ],
  "character_merge_suggestions": [],
  "cross_shot": {{
    "narrative_continuity": "",
    "emotion_arc": "",
    "suggested_beats": [[0, 1, 2], [3, 4]]
  }}
}}
```"""

CHARACTER_SECTION_TEMPLATE = """== 已知角色脸谱 ==
下方附件中，每张参考脸谱前都标注了对应的角色ID、序号和来源时间。
请根据脸部五官特征（而非服装或发型）匹配角色身份。

== 角色匹配规则 ==
1. 以脸部特征（五官、脸型）为主要依据，同一人可能换装/换发型。
2. 如果视频画面中的人物无法确定匹配哪个角色，标注为 "unknown_1", "unknown_2" 等临时编号，不要强行匹配。
3. 如果出现非人类实体（动物/机器人等），在 character_updates 中报告。

== 已知角色档案 ==
{profiles_text}"""

PROFILE_ONLY_CHARACTER_SECTION_TEMPLATE = """== 角色信息 ==
当前无参考脸谱图片，仅有文字档案。请结合以下档案信息识别画面中的人物。
如果无法确定匹配哪个角色，标注为 "unknown_1", "unknown_2" 等临时编号。

== 角色匹配规则 ==
1. 以脸部特征（五官、脸型）为主要依据，同一人可能换装/换发型。
2. 不确定时标注为 unknown，不要强行匹配。
3. 如果出现非人类实体（动物/机器人等），在 character_updates 中报告。

== 已知角色档案 ==
{profiles_text}"""

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

        chunk_shots = [s for s in shots if s.scene_index in chunk.shot_indices]

        # 构建结构化 gallery 记录
        gallery_records = _build_gallery_records(galleries, profiles)

        # 构建 prompt（纯文字部分）
        prompt = _build_prompt(chunk, chunk_shots, profiles, gallery_records)

        # 构建结构化 content（text-media 交错，每张图前带标签）
        content_parts = _build_structured_content(
            prompt, segment_path, gallery_records
        )

        # 调用 Gemini（使用结构化标签接口）
        result = _call_gemini_structured(client, content_parts)

        if result:
            # 回填到 shot 级数据
            t, o, v, a, al = _backfill_chunk_result(
                result, chunk, chunk_shots, profiles,
                chunk_index=chunk.chunk_index,
            )
            all_transcripts.extend(t)
            all_ocr.extend(o)
            all_vision.extend(v)
            all_audio.extend(a)
            all_alignments.extend(al)

            # 更新角色档案（含 description 回写）
            _update_profiles(profiles, result, chunk.chunk_index)

            # 处理角色合并建议（顶层独立字段）
            _process_merge_suggestions(profiles, result, chunk.chunk_index)

            # 保存 chunk 原始结果；LLM 偶尔会返回非预期类型，这里只做轻量防御。
            raw_transcripts = result.get("transcripts", [])
            if not isinstance(raw_transcripts, list):
                raw_transcripts = []
            raw_transcripts = [t for t in raw_transcripts if isinstance(t, dict)]

            raw_per_shot = result.get("per_shot", [])
            if not isinstance(raw_per_shot, list):
                raw_per_shot = []
            raw_per_shot = [s for s in raw_per_shot if isinstance(s, dict)]

            raw_updates = result.get("character_updates", [])
            if not isinstance(raw_updates, list):
                raw_updates = []

            cross_shot = result.get("cross_shot", {})
            if not isinstance(cross_shot, dict):
                cross_shot = {}

            chunk.raw_transcripts = raw_transcripts
            chunk.per_shot_vision = [s.get("vision", {}) for s in raw_per_shot]
            chunk.per_shot_audio = [s.get("audio", {}) for s in raw_per_shot]
            chunk.character_updates = raw_updates
            chunk.cross_shot_analysis = cross_shot
            chunk.suggested_beats = cross_shot.get("suggested_beats", [])
        else:
            logger.warning(f"  Chunk {chunk.chunk_index} 处理失败，跳过")

        processed_chunks.append(chunk)
        time.sleep(1)

    # 补全未覆盖的 shot
    covered_indices = {v.scene_index for v in all_vision}
    transcript_scene_indices = {t.scene_index for t in all_transcripts if t.scene_index >= 0}
    missing_indices = [s.scene_index for s in shots if s.scene_index not in covered_indices]
    if missing_indices:
        logger.warning(
            f"MinuteChunk 缺少 {len(missing_indices)} 个 shot 的 per_shot 回填，"
            f"将写入占位分析: {missing_indices[:20]}"
        )
    for s in shots:
        if s.scene_index not in covered_indices:
            all_vision.append(VisionSummary(
                scene_index=s.scene_index,
                timestamp=s.start_time,
                description="模型未返回该镜头画面分析",
            ))
            all_ocr.append(OCRResult(scene_index=s.scene_index, timestamp=s.start_time))
            all_audio.append(AudioProsody(
                scene_index=s.scene_index,
                speech_rate="unknown" if s.scene_index in transcript_scene_indices else "",
            ))

    # 排序
    all_transcripts.sort(key=lambda t: t.start_time)
    all_ocr.sort(key=lambda o: o.scene_index)
    all_vision.sort(key=lambda v: v.scene_index)
    all_audio.sort(key=lambda a: a.scene_index)
    all_alignments.sort(key=lambda a: a.scene_index)

    # 应用已确认的角色合并（canonicalization）
    identity_links = _apply_confirmed_merges(
        profiles, all_transcripts, all_alignments
    )
    if identity_links:
        links_path = video_dir / "character_identity_links.json"
        links_path.write_text(
            json.dumps(identity_links, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"已保存 {len(identity_links)} 条角色身份合并到 character_identity_links.json")

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


def _build_gallery_records(galleries, profiles):
    """构建结构化 gallery 记录，替代旧的扁平 _collect_gallery_images。

    Returns:
        list[dict]: 每个元素包含 character_id, name, description,
                    gallery_imgs, timestamps, source_shots
    """
    records = []
    for g in galleries:
        if g.tier == "passerby":
            continue
        imgs = []
        for i, p in enumerate(g.gallery_paths):
            if Path(p).exists():
                ts = g.gallery_timestamps[i] if i < len(g.gallery_timestamps) else 0.0
                src_shot = g.gallery_scene_indices[i] if i < len(g.gallery_scene_indices) else -1
                imgs.append({"path": p, "timestamp": ts, "source_shot": src_shot})
        if not imgs:
            continue

        profile = profiles.get(g.character_id)
        name = g.character_id
        desc = ""
        if profile:
            name = profile.names[0] if profile.names else g.character_id
            # 优先用 description；为空时取最近一条 appearance_change
            desc = profile.description or ""
            if not desc and profile.appearance_changes:
                desc = profile.appearance_changes[-1].get("description", "")

        records.append({
            "character_id": g.character_id,
            "name": name,
            "description": desc,
            "gallery_imgs": imgs,
            "tier": g.tier,
        })
    return records


def _build_structured_content(prompt, segment_path, gallery_records):
    """构建 text-media 交错排列的结构化 content。

    每张 gallery 图片前都插入独立的文字标签：
      char_000 / 张三 / face 1 of 4 / source shot 12 / time 35.2s
      [image]
    """
    parts = [{"type": "text", "text": prompt}]

    # 视频片段
    if segment_path and Path(segment_path).exists():
        parts.append({"type": "text", "text": "== Chunk 视频片段 =="})
        parts.append({"type": "media", "path": segment_path})

    # 逐角色、逐图带标签
    if gallery_records:
        parts.append({"type": "text", "text": "== 角色参考脸谱 =="})
    for rec in gallery_records:
        cid = rec["character_id"]
        name = rec["name"]
        n_imgs = len(rec["gallery_imgs"])
        for idx, img_info in enumerate(rec["gallery_imgs"]):
            label = (
                f"{cid} / {name} / face {idx + 1} of {n_imgs}"
                f" / source shot {img_info['source_shot']}"
                f" / time {img_info['timestamp']:.1f}s"
            )
            parts.append({"type": "text", "text": label})
            parts.append({"type": "media", "path": img_info["path"]})

    return parts


def _build_prompt(chunk, chunk_shots, profiles, gallery_records):
    """构建 Gemini prompt（纯文字部分）"""
    # 镜头边界
    boundaries = []
    for local_idx, s in enumerate(chunk_shots):
        boundaries.append(
            f"local_shot_index {local_idx} | scene_index {s.scene_index} | "
            f"abs {s.start_time:.1f}s-{s.end_time:.1f}s | "
            f"rel {s.start_time - chunk.start_time:.1f}s-{s.end_time - chunk.start_time:.1f}s | "
            f"duration {s.duration:.1f}s"
        )

    # 角色部分
    if gallery_records:
        profiles_lines = []
        for rec in gallery_records:
            cid = rec["character_id"]
            name = rec["name"]
            desc = rec["description"] or "（暂无描述）"
            # 从 profiles 补充最近的 appearance_changes 和 key_actions
            profile = profiles.get(cid)
            extra = ""
            if profile:
                recent_changes = profile.appearance_changes[-2:]
                recent_actions = profile.key_actions[-3:]
                if recent_changes:
                    changes_str = "; ".join(c.get("description", "") for c in recent_changes)
                    extra += f" | 近期外观变化: {changes_str}"
                if recent_actions:
                    actions_str = "; ".join(a.get("action", "") for a in recent_actions)
                    extra += f" | 近期行为: {actions_str}"
            profiles_lines.append(f"- {cid} ({name}): {desc}{extra}")
        char_section = CHARACTER_SECTION_TEMPLATE.format(
            profiles_text="\n".join(profiles_lines),
        )
    elif profiles:
        # 有 profiles 但没有 gallery（降级场景）— 使用专用模板
        profiles_lines = []
        for cid, p in profiles.items():
            name = p.names[0] if p.names else cid
            desc = p.description or "（暂无描述）"
            profiles_lines.append(f"- {cid} ({name}): {desc}")
        char_section = PROFILE_ONLY_CHARACTER_SECTION_TEMPLATE.format(
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


def _call_gemini_structured(client, content_parts) -> dict | None:
    """使用结构化 content（text-media 交错）调用 Gemini"""
    try:
        response = client.chat_with_labeled_media(
            content_parts=content_parts,
            temperature=0.2,
        )
        parsed = client.parse_json(response)
        if parsed and isinstance(parsed, dict):
            return parsed
        logger.warning("Gemini 响应解析失败")
        return None
    except Exception as e:
        logger.error(f"Gemini 调用失败: {e}")
        return None


def _normalize_character_id(raw_id: str, profiles: dict, chunk_index: int = -1) -> str | None:
    """将各种形式的人物标识规范化为 char_ 前缀的 ID。

    规则:
      - char_XXX → 直接返回
      - entity_XXX → 直接返回
      - unknown_N → char_tmp_chunk_XXXX_unknown_N（带 chunk 作用域），并在 profiles 中注册
      - unknown / 空 / 其他 → None
    """
    raw_id = str(raw_id or "").strip()
    if not raw_id or raw_id == "unknown":
        return None
    if raw_id.startswith(("char_", "entity_")):
        return raw_id
    if raw_id.startswith("unknown_"):
        if chunk_index >= 0:
            normalized = f"char_tmp_chunk_{chunk_index:04d}_{raw_id}"
        else:
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


def _safe_float(value, default=0.0) -> float:
    """安全解析普通浮点数，不做 0-1 置信度裁剪。"""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _safe_text(value, default="") -> str:
    """安全转成去空白字符串。"""
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _safe_bool(value, default=False) -> bool:
    """安全解析 LLM 返回的布尔值。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    if text in {"true", "yes", "y", "1", "是", "有"}:
        return True
    if text in {"false", "no", "n", "0", "否", "无", "none", "null", ""}:
        return False
    return default


def _as_str_list(value) -> list[str]:
    """兼容 LLM 将数组字段返回为字符串、对象或混合列表。"""
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = [value]

    result = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, dict):
            text = item.get("text") or item.get("name") or item.get("label") or json.dumps(item, ensure_ascii=False)
        else:
            text = str(item)
        text = text.strip()
        if text:
            result.append(text)
    return result


def _coerce_int(value) -> int | None:
    """安全解析整数索引。"""
    if value is None:
        return None
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return None


def _infer_transcript_time_mode(transcript_items, chunk) -> str:
    """推断 transcript 时间戳是相对 chunk 起点还是全局绝对时间。"""
    if chunk.start_time <= 0.5:
        return "relative"
    times = []
    for item in transcript_items:
        if not isinstance(item, dict):
            continue
        times.append(_safe_float(item.get("start_time"), -1.0))
        times.append(_safe_float(item.get("end_time"), -1.0))
    times = [t for t in times if t >= 0]
    if not times:
        return "relative"

    duration = max(chunk.duration, chunk.end_time - chunk.start_time)
    relative_score = sum(1 for t in times if 0 <= t <= duration + 1.0)
    absolute_score = sum(1 for t in times if chunk.start_time - 1.0 <= t <= chunk.end_time + 1.0)
    if absolute_score > relative_score and min(times) >= chunk.start_time - 1.0:
        return "absolute"
    return "relative"


def _chunk_time_to_abs(value, chunk, mode="relative") -> float:
    """将 LLM 返回的时间转换为全局时间。"""
    value = _safe_float(value, 0.0)
    if mode == "absolute":
        return round(value, 1)
    return round(value + chunk.start_time, 1)


def _infer_scene_index_mode(per_shot_items, shot_map, local_to_global) -> str:
    """推断 per_shot.scene_index 更像全局 scene_index 还是局部索引。"""
    raw_indices = [
        _coerce_int(item.get("scene_index"))
        for item in per_shot_items
        if isinstance(item, dict)
    ]
    raw_indices = [i for i in raw_indices if i is not None]
    local_score = sum(1 for i in raw_indices if i in local_to_global)
    global_score = sum(1 for i in raw_indices if i in shot_map)
    return "local" if local_score > global_score else "global"


def _resolve_per_shot_scene_index(item, shot_map, local_to_global, scene_index_mode) -> int | None:
    """解析 per_shot 对应的全局 scene_index，优先使用 local_shot_index 双锚点。"""
    local_idx = _coerce_int(item.get("local_shot_index"))
    if local_idx in local_to_global:
        return local_to_global[local_idx]

    scene_idx = _coerce_int(item.get("scene_index"))
    if scene_idx is None:
        return None
    if scene_index_mode == "local" and scene_idx in local_to_global:
        return local_to_global[scene_idx]
    if scene_idx in shot_map:
        return scene_idx
    if scene_idx in local_to_global:
        return local_to_global[scene_idx]
    return None


def _extract_character_id(value):
    """兼容 characters_present 中的字符串或 {character_id, confidence} 对象。"""
    if isinstance(value, dict):
        return value.get("character_id") or value.get("id")
    return value


def _backfill_chunk_result(result, chunk, chunk_shots, profiles, chunk_index=-1):
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
    transcript_items = [
        item for item in result.get("transcripts", [])
        if isinstance(item, dict)
    ]
    time_mode = _infer_transcript_time_mode(transcript_items, chunk)
    for item in transcript_items:
        seg_start = _chunk_time_to_abs(item.get("start_time", 0), chunk, time_mode)
        seg_end = _chunk_time_to_abs(item.get("end_time", 0), chunk, time_mode)
        if seg_end < seg_start:
            seg_end = seg_start
        text = _safe_text(item.get("text", ""))
        if not text:
            continue

        speaker = _safe_text(item.get("speaker", "unknown"), "unknown")
        char_id = _normalize_character_id(speaker, profiles, chunk_index=chunk_index)
        if char_id and str(speaker or "").strip().startswith("unknown_"):
            speaker = char_id

        # 按中点分配 scene_index
        mid = (seg_start + seg_end) / 2.0
        scene_index = -1
        for s in chunk_shots:
            if s.start_time <= mid < s.end_time:
                scene_index = s.scene_index
                break
        if scene_index == -1 and chunk_shots:
            scene_index = min(chunk_shots, key=lambda s: abs(s.start_time - mid)).scene_index

        shot = shot_map.get(scene_index)
        cross_shot = False
        if shot:
            cross_shot = seg_start < shot.start_time or seg_end > shot.end_time

        seg = TranscriptSegment(
            start_time=seg_start, end_time=seg_end, text=text,
            speaker=speaker, character_id=char_id,
            scene_index=scene_index,
            transcript_type=_safe_text(item.get("type", "dialogue"), "dialogue"),
            cross_shot=cross_shot,
        )
        transcripts.append(seg)

    # 回填 Vision / Audio / OCR
    per_shot_items = [
        item for item in result.get("per_shot", [])
        if isinstance(item, dict)
    ]
    scene_index_mode = _infer_scene_index_mode(
        per_shot_items, shot_map, local_to_global
    )
    per_shot_by_scene = {}

    def _per_shot_quality(item):
        vis = item.get("vision", {}) if isinstance(item.get("vision", {}), dict) else {}
        aud = item.get("audio", {}) if isinstance(item.get("audio", {}), dict) else {}
        return (
            len(str(vis.get("description", "")).strip())
            + len(str(vis.get("action_description", "")).strip())
            + len(_as_str_list(vis.get("objects", []))) * 8
            + len(_as_str_list(vis.get("ocr_texts", []))) * 8
            + (5 if aud.get("speech_emotion") else 0)
            + (5 if aud.get("music_mood") else 0)
        )

    for item in per_shot_items:
        si = _resolve_per_shot_scene_index(
            item, shot_map, local_to_global, scene_index_mode
        )
        if si is None or si not in shot_map:
            continue
        existing = per_shot_by_scene.get(si)
        if existing is None or _per_shot_quality(item) > _per_shot_quality(existing):
            per_shot_by_scene[si] = item

    missing = [s.scene_index for s in chunk_shots if s.scene_index not in per_shot_by_scene]
    if missing:
        logger.warning(
            f"  Chunk {chunk.chunk_index} per_shot 覆盖不足: "
            f"{len(per_shot_by_scene)}/{len(chunk_shots)}，缺失 scene_index={missing[:10]}"
        )

    for si in [s.scene_index for s in chunk_shots if s.scene_index in per_shot_by_scene]:
        item = per_shot_by_scene[si]
        shot = shot_map[si]

        vis = item.get("vision", {})
        if not isinstance(vis, dict):
            vis = {}
        vision_summaries.append(VisionSummary(
            scene_index=si, timestamp=shot.start_time,
            description=_safe_text(vis.get("description"), "无法判断"),
            objects=_as_str_list(vis.get("objects", [])),
            mood=_safe_text(vis.get("mood", "")),
            scene_type=_safe_text(vis.get("scene_type", "")),
            camera_motion=_safe_text(vis.get("camera_motion", "")),
            shot_scale=_safe_text(vis.get("shot_scale", "")),
            action_description=_safe_text(vis.get("action_description", "")),
            props=_as_str_list(vis.get("props", [])),
        ))

        ocr_results.append(OCRResult(
            scene_index=si, timestamp=shot.start_time,
            texts=_as_str_list(vis.get("ocr_texts", [])),
        ))

        aud = item.get("audio", {})
        if not isinstance(aud, dict):
            aud = {}
        has_music = _safe_bool(aud.get("has_music", False))
        has_sfx = _safe_bool(aud.get("has_sfx", False))
        audio_prosodies.append(AudioProsody(
            scene_index=si,
            has_music=has_music,
            music_mood=_safe_text(aud.get("music_mood", "")),
            has_sfx=has_sfx,
            sfx_tags=_as_str_list(aud.get("sfx_tags", [])),
            silence_ratio=_safe_parse_float(aud.get("silence_ratio", 0)),
            speech_rate=_safe_text(aud.get("speech_rate", "")),
            volume_peak=_safe_parse_float(aud.get("volume_peak", 0)),
            speech_emotion=_safe_text(aud.get("speech_emotion", "")),
        ))

        # 多模态对齐（从结果直接派生）
        chars_present = []
        raw_chars = item.get("characters_present", [])
        if isinstance(raw_chars, (str, dict)):
            raw_chars = [raw_chars]
        for c in raw_chars or []:
            normalized = _normalize_character_id(
                _extract_character_id(c), profiles, chunk_index=chunk_index
            )
            if normalized:
                chars_present.append(normalized)
        shot_trans = [t for t in transcripts if t.scene_index == si]
        speaking = list({t.character_id for t in shot_trans if t.character_id})

        active = []
        if shot_trans:
            active.append("speech")
        if has_music:
            active.append("music")
        if has_sfx:
            active.append("sfx")

        dominant = "speech" if shot_trans else ("music" if has_music else "visual")

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
        cid = _normalize_character_id(cid, profiles, chunk_index=chunk_index)
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
                change_desc = update["appearance_change"]
                p.appearance_changes.append({
                    "chunk_idx": chunk_index,
                    "description": change_desc,
                })
                # 回写 description — 始终保持最新外观描述
                p.description = change_desc
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


def _safe_parse_float(value, default=0.0) -> float:
    """安全解析 LLM 输出的 float 值，兼容 '', null, '90%' 等值。"""
    if value is None:
        return default
    try:
        if isinstance(value, str):
            text = value.strip()
            is_percent = text.endswith("%")
            if is_percent:
                text = text[:-1].strip()
            parsed = float(text)
            if is_percent or 1.0 < parsed <= 100.0:
                parsed /= 100.0
        else:
            parsed = float(value)
            if 1.0 < parsed <= 100.0:
                parsed /= 100.0
        return max(0.0, min(1.0, parsed))
    except (ValueError, TypeError):
        return default


def _process_merge_suggestions(profiles, result, chunk_index):
    """处理 Gemini 返回的角色合并建议（顶层独立字段）。

    将建议记录到 primary 角色的 merge_suggestions 中，
    以便后续决策是否执行 canonicalization。
    """
    for merge in result.get("character_merge_suggestions", []):
        primary = merge.get("primary", "")
        duplicate = merge.get("duplicate", "")
        confidence = _safe_parse_float(merge.get("confidence"), 0.0)
        reason = merge.get("reason", "")

        if not primary or not duplicate:
            continue
        # 跳过自身合并
        if primary == duplicate:
            continue
        # 要求两者都存在于 profiles 或已知 gallery
        if primary not in profiles or duplicate not in profiles:
            logger.warning(
                f"  合并建议被忽略: {duplicate} → {primary} "
                f"(一方或双方不在已知角色中)"
            )
            continue

        logger.info(
            f"  ⚠ Gemini 建议合并角色: {duplicate} → {primary} "
            f"(置信度: {confidence:.0%}, 原因: {reason})"
        )

        profiles[primary].merge_suggestions.append({
            "duplicate_id": duplicate,
            "reason": reason,
            "confidence": confidence,
            "chunk_index": chunk_index,
        })


_MERGE_CONFIDENCE_THRESHOLD = 0.85
_MERGE_MIN_CHUNK_REPORTS = 2


def _apply_confirmed_merges(profiles, transcripts, alignments):
    """对确认的合并建议执行 canonicalization。

    确认条件: 最高 confidence >= _MERGE_CONFIDENCE_THRESHOLD
              且 >= _MERGE_MIN_CHUNK_REPORTS 个不同 chunk 报告了同一对合并。
              正反方向会归并为同一无向 pair 后统计。

    执行:
      - transcripts 中的 character_id / speaker → primary
      - alignments 中的 visible_characters / speaking_characters → primary
      - duplicate profile 标记 merged_into = primary

    Returns:
        list[dict]: identity links 记录，保存为 character_identity_links.json
    """
    # 收集无向 pair → suggestions，避免 A→B 和 B→A 被拆开统计。
    merge_candidates = defaultdict(list)
    for cid, profile in profiles.items():
        for sug in profile.merge_suggestions:
            dup = sug.get("duplicate_id", "")
            if not dup or dup == cid or dup not in profiles:
                continue
            pair = tuple(sorted((cid, dup)))
            suggestion = dict(sug)
            suggestion["_proposed_primary"] = cid
            suggestion["_proposed_duplicate"] = dup
            merge_candidates[pair].append(suggestion)

    # 筛选确认的合并
    confirmed = {}  # duplicate → primary
    link_details = {}
    tier_rank = {"major": 0, "minor": 1, "passerby": 2}

    def primary_rank(cid: str):
        profile = profiles.get(cid)
        return (tier_rank.get(getattr(profile, "tier", ""), 3), cid)

    for pair, suggestions in merge_candidates.items():
        max_conf = max(_safe_parse_float(s.get("confidence"), 0.0) for s in suggestions)
        unique_chunks = len({s.get("chunk_index") for s in suggestions})
        if max_conf < _MERGE_CONFIDENCE_THRESHOLD or unique_chunks < _MERGE_MIN_CHUNK_REPORTS:
            continue

        primary_votes = defaultdict(int)
        for s in suggestions:
            proposed_primary = s.get("_proposed_primary")
            if proposed_primary in pair:
                primary_votes[proposed_primary] += 1

        primary = min(
            pair,
            key=lambda cid: (-primary_votes.get(cid, 0), primary_rank(cid)),
        )
        dup = pair[1] if primary == pair[0] else pair[0]
        confirmed[dup] = primary
        link_details[dup] = {
            "primary": primary,
            "duplicate": dup,
            "max_confidence": max_conf,
            "chunk_count": unique_chunks,
            "reasons": [s.get("reason", "") for s in suggestions],
        }

    def resolve_primary(cid: str) -> str:
        seen = set()
        while cid in confirmed and cid not in seen:
            seen.add(cid)
            cid = confirmed[cid]
        return cid

    confirmed = {
        dup: resolve_primary(primary)
        for dup, primary in confirmed.items()
        if dup != resolve_primary(primary)
    }

    identity_links = []
    for dup, primary in confirmed.items():
        detail = link_details.get(dup, {})
        detail["primary"] = primary
        detail["duplicate"] = dup
        identity_links.append(detail)
        logger.info(
            f"  ✓ 确认合并: {dup} → {primary} "
            f"(最高置信度: {detail.get('max_confidence', 0):.0%}, "
            f"{detail.get('chunk_count', 0)} 个 chunk 报告)"
        )

    if not confirmed:
        return []

    # 执行 canonicalization
    # 1. transcripts
    for t in transcripts:
        if t.character_id in confirmed:
            t.character_id = confirmed[t.character_id]
        if t.speaker in confirmed:
            t.speaker = confirmed[t.speaker]

    # 2. alignments
    for al in alignments:
        al.visible_characters = [
            confirmed.get(c, c) for c in al.visible_characters
        ]
        al.speaking_characters = [
            confirmed.get(c, c) for c in al.speaking_characters
        ]
        # 去重
        al.visible_characters = list(dict.fromkeys(al.visible_characters))
        al.speaking_characters = list(dict.fromkeys(al.speaking_characters))

    # 3. 标记 duplicate profile
    for dup, primary in confirmed.items():
        if dup in profiles:
            profiles[dup].merged_into = primary

    return identity_links

def _build_speaker_map(transcripts) -> dict:
    """从角色标注直接生成 speaker_map"""
    speaker_map = {}
    for t in transcripts:
        if t.speaker and t.character_id:
            speaker_key = t.speaker
            if str(speaker_key).startswith("unknown_"):
                speaker_key = t.character_id
            speaker_map[speaker_key] = t.character_id
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

    def resolve_character_id(cid: str) -> str:
        seen = set()
        while cid in profiles and profiles[cid].merged_into and cid not in seen:
            seen.add(cid)
            cid = profiles[cid].merged_into
        return cid

    # 从 alignments 和 galleries 提取每个角色出现的 scene_index
    char_scenes = defaultdict(set)
    for al in alignments:
        for cid_vis in al.visible_characters:
            char_scenes[resolve_character_id(cid_vis)].add(al.scene_index)
    if galleries:
        for g in galleries:
            char_scenes[resolve_character_id(g.character_id)].update(g.appearance_scenes)
    chars = []
    for cid, p in profiles.items():
        if p.merged_into:
            continue
        resolved_cid = resolve_character_id(cid)
        chars.append(CharacterDeep(
            character_id=resolved_cid,
            display_name=p.names[0] if p.names else cid,
            description=p.description,
            role=None,
            appearance_scenes=sorted(char_scenes.get(resolved_cid, [])),
        ).model_dump())
    chars_path = video_dir / "characters.json"
    chars_path.write_text(
        json.dumps(chars, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    logger.info(f"所有产物已保存至 {video_dir}")
