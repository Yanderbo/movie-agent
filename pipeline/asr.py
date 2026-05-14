# -*- coding: utf-8 -*-
"""
ASR 语音转文字（v2 — 长窗口模式）

v2 核心改动:
- ASR 不再按 shot 切分，改为按较长时间窗口（默认 5 分钟）提取音频
- 转写后得到带精确时间戳的句子
- 再按时间戳回填到 shot（支持跨镜头台词标记）
- 新增 transcript_type 区分对白 / 旁白 / 画外音

流程:
1. 按 ASR_WINDOW_DURATION 提取整段音频
2. 调用 Gemini Audio API 做 ASR
3. 将时间戳偏移为全局时间
4. 按 shot 边界回填 scene_index，标记跨镜头句子

仍保留对超长窗口（> ASR_CHUNK_DURATION）的内部切分处理。
"""
import json
import time
from pathlib import Path

import config
from models.schemas import Shot, TranscriptSegment
from utils.llm_client import get_llm_client
from utils.ffmpeg_utils import extract_audio_segment, get_audio_duration
from utils.logger import get_logger

logger = get_logger("ASR")

ASR_PROMPT = """你是一个专业的语音识别系统。请仔细听这段音频，将其中所有语音内容转录为文字。

要求：
1. 输出 JSON 数组格式，每个元素代表一句话/一段话
2. 每个元素包含以下字段：
   - start_time: 开始时间（秒，保留1位小数）— 相对于本段音频开头
   - end_time: 结束时间（秒，保留1位小数）— 相对于本段音频开头
   - text: 转录的文字内容
   - speaker: 说话人标识（如果能区分不同说话人，用 "speaker_1", "speaker_2" 等标识；无法区分则为 null）
   - type: 语音类型（"dialogue" 表示角色对白, "narration" 表示旁白, "voiceover" 表示画外音; 无法区分则默认 "dialogue"）
3. 按时间顺序排列
4. 只输出 JSON，不要其他内容
5. 如果某段时间没有语音，跳过即可
6. 确保时间戳尽可能准确

输出格式示例：
```json
[
  {"start_time": 0.0, "end_time": 3.5, "text": "大家好，欢迎收看", "speaker": "speaker_1", "type": "dialogue"},
  {"start_time": 4.2, "end_time": 8.1, "text": "今天我们来聊一个话题", "speaker": "speaker_1", "type": "dialogue"},
  {"start_time": 10.0, "end_time": 14.5, "text": "在那个遥远的年代...", "speaker": null, "type": "narration"}
]
```
"""


def transcribe_audio(
    video_id: str,
    video_path: str,
    scenes: list[Shot],
) -> list[TranscriptSegment]:
    """
    长窗口 ASR 转写 + 按 shot 回填。

    v2 改动: 按较长窗口（默认 5 分钟）提取音频做 ASR，
    再根据时间戳将每句话回填到对应的 shot。

    Args:
        video_id: 视频 ID
        video_path: 原始视频文件路径
        scenes: 镜头(Shot)列表

    Returns:
        TranscriptSegment 列表（全局时间戳，带 scene_index）
    """
    video_dir = config.VIDEOS_DIR / video_id
    transcript_path = video_dir / "transcripts.json"

    # 如果已存在，直接加载
    if transcript_path.exists():
        logger.info(f"ASR 结果已存在，直接加载: {transcript_path}")
        data = json.loads(transcript_path.read_text(encoding="utf-8"))
        return [TranscriptSegment(**s) for s in data]

    if not scenes:
        logger.warning("无镜头数据，跳过 ASR")
        return []

    # 计算视频总时长
    total_duration = max(s.end_time for s in scenes)
    window = config.ASR_WINDOW_DURATION
    logger.info(
        f"开始长窗口 ASR: 总时长 {total_duration:.0f}s, 窗口 {window}s, "
        f"{len(scenes)} 个镜头"
    )

    client = get_llm_client()
    audio_dir = video_dir / "audio_windows"
    audio_dir.mkdir(parents=True, exist_ok=True)

    all_segments: list[TranscriptSegment] = []

    # ── 按窗口提取音频并转写 ──
    win_start = 0.0
    win_idx = 0
    while win_start < total_duration:
        win_end = min(win_start + window, total_duration)
        win_duration = win_end - win_start

        if win_duration < 0.5:
            break

        logger.info(
            f"  窗口 {win_idx + 1}: {win_start:.0f}s - {win_end:.0f}s "
            f"({win_duration:.0f}s)"
        )

        # 提取窗口音频
        win_audio_path = str(audio_dir / f"window_{win_idx:04d}.wav")
        try:
            extract_audio_segment(
                video_path, win_audio_path,
                start_time=win_start,
                end_time=win_end,
            )
        except Exception as e:
            logger.warning(f"  窗口 {win_idx}: 音频提取失败，跳过: {e}")
            win_start = win_end
            win_idx += 1
            continue

        # 对超长窗口内部切分
        if win_duration > config.ASR_CHUNK_DURATION:
            segments = _transcribe_long_window(
                client, win_audio_path, win_start, win_end, win_duration
            )
        else:
            segments = _transcribe_window(
                client, win_audio_path, win_start, win_end
            )

        if segments is None:
            logger.warning(f"  窗口 {win_idx}: ASR 失败，跳过")
        elif segments:
            all_segments.extend(segments)

        win_start = win_end
        win_idx += 1
        time.sleep(0.5)

    # ── 按时间戳回填 scene_index ──
    all_segments = _assign_scene_indices(all_segments, scenes)

    # 按全局时间排序
    all_segments.sort(key=lambda s: s.start_time)

    # 保存结果
    data = [s.model_dump() for s in all_segments]
    transcript_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"ASR 完成: {len(all_segments)} 段台词, 保存至 {transcript_path}")

    return all_segments


def _transcribe_window(
    client, audio_path: str, win_start: float, win_end: float,
) -> list[TranscriptSegment] | None:
    """
    对单个窗口的音频做 ASR。

    Returns:
        TranscriptSegment 列表（全局时间戳），或 None 表示失败。
    """
    try:
        response = client.chat_with_media(
            prompt=ASR_PROMPT,
            media_path=audio_path,
            temperature=0.1,
        )
        parsed = client.parse_json(response)
        if parsed is None:
            logger.error(f"ASR 响应解析失败: {response[:200]}")
            return None
        if not isinstance(parsed, list):
            logger.error("ASR 响应格式不正确")
            return None

        segments = []
        for item in parsed:
            # 段内相对时间 + 窗口起始偏移 = 全局绝对时间
            seg_start = round(float(item.get("start_time", 0)) + win_start, 1)
            seg_end = round(float(item.get("end_time", 0)) + win_start, 1)

            # 时间修正：确保不超出窗口边界
            seg_start = max(seg_start, win_start)
            seg_end = min(seg_end, win_end)
            if seg_end <= seg_start:
                continue

            text = item.get("text", "")
            if not text.strip():
                continue

            seg = TranscriptSegment(
                start_time=seg_start,
                end_time=seg_end,
                text=text,
                speaker=item.get("speaker"),
                transcript_type=item.get("type", "dialogue"),
                scene_index=-1,  # 后续回填
            )
            segments.append(seg)

        logger.info(f"    识别 {len(segments)} 句")
        return segments

    except Exception as e:
        logger.error(f"ASR 处理异常: {e}")
        return None


def _transcribe_long_window(
    client, audio_path: str, win_start: float, win_end: float, win_duration: float,
) -> list[TranscriptSegment] | None:
    """处理超长窗口: 内部按 ASR_CHUNK_DURATION 切分后分别转写再合并。"""
    from utils.ffmpeg_utils import split_audio

    chunks_dir = str(Path(audio_path).parent / f"chunks_win_{win_start:.0f}")
    chunk_paths = split_audio(audio_path, chunks_dir, chunk_seconds=config.ASR_CHUNK_DURATION)
    logger.info(f"    超长窗口 ({win_duration:.0f}s)，切分为 {len(chunk_paths)} 段")

    all_segments = []
    chunk_offset = 0.0

    for i, chunk_path in enumerate(chunk_paths):
        chunk_duration = get_audio_duration(chunk_path)
        chunk_global_start = win_start + chunk_offset
        chunk_global_end = min(chunk_global_start + chunk_duration, win_end)

        segments = _transcribe_window(
            client, chunk_path, chunk_global_start, chunk_global_end
        )
        if segments is None:
            logger.warning(f"    chunk {i}: ASR 失败")
            return None
        all_segments.extend(segments)
        chunk_offset += chunk_duration
        time.sleep(0.5)

    return all_segments


def _assign_scene_indices(
    segments: list[TranscriptSegment],
    shots: list[Shot],
) -> list[TranscriptSegment]:
    """
    按时间戳将 ASR 句子回填到 shot。

    规则:
    - 句子中点落在哪个 shot 内，就分配给哪个 shot
    - 如果句子跨越了 shot 边界（start 在一个 shot，end 在另一个），
      标记 cross_shot=True，分配给中点所在的 shot
    """
    if not shots:
        return segments

    for seg in segments:
        mid = (seg.start_time + seg.end_time) / 2.0
        best_shot = None
        crosses_boundary = False

        for shot in shots:
            if shot.start_time <= mid < shot.end_time:
                best_shot = shot
                break

        # 如果中点找不到（可能在最后一帧之后），取最近的
        if best_shot is None:
            best_shot = min(shots, key=lambda s: abs(s.start_time - mid))

        seg.scene_index = best_shot.scene_index

        # 检测是否跨越 shot 边界
        if seg.start_time < best_shot.start_time or seg.end_time > best_shot.end_time:
            seg.cross_shot = True

    return segments
