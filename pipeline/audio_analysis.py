# -*- coding: utf-8 -*-
"""
音频韵律分析（v3 新增）

识别音乐、音效、沉默、语速、音量峰值、语音情绪。
按 ASR_WINDOW_DURATION 切分窗口，LLM 分析后回填 shot。

降级策略：LLM 失败时返回空列表，不阻塞流程。
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import config
from models.schemas import Shot, AudioProsody, AudioSegment
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("AudioAnalysis")

AUDIO_PROMPT_TEMPLATE = """你是一个专业的影视音频分析师。请分析以下视频片段的音频特征。

视频时间范围: {start:.1f}s - {end:.1f}s

=== 该时段已有的台词信息 ===
{transcript_info}

=== 该时段已有的画面信息 ===
{vision_info}

请为该时间段内的每个镜头（shot）分析音频特征。镜头列表：
{shots_info}

对每个镜头，请判断：
1. has_music: 是否有背景音乐
2. music_mood: 音乐情绪（energetic/melancholic/tense/romantic/epic/calm，无音乐则为空）
3. has_sfx: 是否有明显音效
4. sfx_tags: 音效类型列表（explosion/door_slam/footsteps/rain/wind/car/gunshot 等）
5. silence_ratio: 沉默/静音占比（0-1）
6. speech_rate: 语速（slow/normal/fast，无语音则为空）
7. volume_peak: 估计的音量峰值（0-1，高潮/爆炸=高，低语=低）
8. speech_emotion: 语音情绪（calm/angry/sad/happy/fearful/surprised/neutral，无语音则为空）

输出 JSON 数组，每个元素对应一个 shot，只输出 JSON：
```json
[
  {{
    "scene_index": 0,
    "has_music": true,
    "music_mood": "tense",
    "has_sfx": false,
    "sfx_tags": [],
    "silence_ratio": 0.1,
    "speech_rate": "fast",
    "volume_peak": 0.7,
    "speech_emotion": "angry"
  }}
]
```
"""


def analyze_audio(
    video_id: str,
    video_path: str,
    shots: list[Shot],
    transcripts=None,
    vision_summaries=None,
) -> list[AudioProsody]:
    """
    分析视频音频特征，为每个 shot 生成 AudioProsody。

    按窗口分析，结合已有的台词和画面信息辅助 LLM 推断。

    Args:
        video_id: 视频 ID
        video_path: 视频文件路径
        shots: 镜头列表
        transcripts: 台词列表（可选，辅助推断）
        vision_summaries: 画面摘要列表（可选，辅助推断）

    Returns:
        AudioProsody 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    audio_path = video_dir / "audio_prosody.json"

    # 如果已存在，直接加载
    if audio_path.exists():
        logger.info(f"音频韵律分析结果已存在，直接加载: {audio_path}")
        data = json.loads(audio_path.read_text(encoding="utf-8"))
        return [AudioProsody(**a) for a in data]

    if not shots:
        logger.warning("无 shot 数据，跳过音频分析")
        return []

    logger.info(f"开始音频韵律分析: {len(shots)} 个 shot")
    client = get_llm_client()

    # 构建辅助索引
    trans_map = {}
    if transcripts:
        for t in transcripts:
            trans_map.setdefault(t.scene_index, []).append(t)
    vision_map = {}
    if vision_summaries:
        for v in vision_summaries:
            vision_map[v.scene_index] = v

    # 按窗口处理
    window_duration = config.ASR_WINDOW_DURATION
    all_prosodies = []

    total_duration = max(s.end_time for s in shots) if shots else 0
    window_start = 0.0

    while window_start < total_duration:
        window_end = min(window_start + window_duration, total_duration)

        # 找出该窗口内的 shots
        window_shots = [
            s for s in shots
            if s.start_time < window_end and s.end_time > window_start
        ]
        if not window_shots:
            window_start = window_end
            continue

        prosodies = _analyze_window(
            client, window_shots, window_start, window_end,
            trans_map, vision_map,
        )
        all_prosodies.extend(prosodies)

        window_start = window_end
        time.sleep(0.5)

    # 补全未分析到的 shot（降级为空 AudioProsody）
    analyzed_indices = {p.scene_index for p in all_prosodies}
    for s in shots:
        if s.scene_index not in analyzed_indices:
            all_prosodies.append(AudioProsody(scene_index=s.scene_index))

    all_prosodies.sort(key=lambda p: p.scene_index)

    # 保存
    audio_path.write_text(
        json.dumps([a.model_dump() for a in all_prosodies], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"音频韵律分析完成: {len(all_prosodies)} 个 shot")

    return all_prosodies


def _analyze_window(
    client,
    shots: list[Shot],
    start: float,
    end: float,
    trans_map: dict,
    vision_map: dict,
) -> list[AudioProsody]:
    """分析一个窗口内的 shot 音频特征"""

    # 构造 shots 信息
    shots_lines = []
    for s in shots:
        shots_lines.append(f"Shot {s.scene_index}: {s.start_time:.1f}s-{s.end_time:.1f}s ({s.duration:.1f}s)")
    shots_info = "\n".join(shots_lines)

    # 构造台词信息
    trans_lines = []
    for s in shots:
        seg_trans = trans_map.get(s.scene_index, [])
        for t in seg_trans[:3]:
            speaker = f"[{t.speaker}]" if t.speaker else ""
            trans_lines.append(f"[shot {s.scene_index}] {speaker} {t.text[:50]}")
    transcript_info = "\n".join(trans_lines[:30]) if trans_lines else "（无台词）"

    # 构造画面信息
    vision_lines = []
    for s in shots:
        v = vision_map.get(s.scene_index)
        if v:
            vision_lines.append(f"[shot {s.scene_index}] {v.scene_type}: {v.description[:60]}")
    vision_info = "\n".join(vision_lines[:20]) if vision_lines else "（无画面信息）"

    prompt = AUDIO_PROMPT_TEMPLATE.format(
        start=start, end=end,
        transcript_info=transcript_info,
        vision_info=vision_info,
        shots_info=shots_info,
    )

    try:
        response = client.chat(prompt=prompt, temperature=0.3)
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, list):
            logger.warning(f"音频分析窗口 {start:.0f}-{end:.0f}s 解析失败")
            return []

        prosodies = []
        valid_indices = {s.scene_index for s in shots}
        for item in parsed:
            si = item.get("scene_index", -1)
            if si not in valid_indices:
                continue
            prosody = AudioProsody(
                scene_index=si,
                has_music=bool(item.get("has_music", False)),
                music_mood=item.get("music_mood", ""),
                has_sfx=bool(item.get("has_sfx", False)),
                sfx_tags=item.get("sfx_tags", []),
                silence_ratio=float(item.get("silence_ratio", 0)),
                speech_rate=item.get("speech_rate", ""),
                volume_peak=float(item.get("volume_peak", 0)),
                speech_emotion=item.get("speech_emotion", ""),
            )
            prosodies.append(prosody)

        return prosodies

    except Exception as e:
        logger.warning(f"音频分析窗口 {start:.0f}-{end:.0f}s 失败: {e}")
        return []
