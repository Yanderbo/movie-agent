# -*- coding: utf-8 -*-
"""
ASR 语音转文字（按镜头段落处理）

核心改动：ASR 从"整体提取 → 事后反查 scene"改为"按 shot 段提取 → 天然带 scene_index"。

流程：
1. 对每个 scene，从原始视频中提取该 shot 时间段的音频片段
2. 对每个片段单独调用 Gemini Audio API 做 ASR
3. 将时间戳偏移为全局时间（shot.start_time + 段内时间）
4. 每条 TranscriptSegment 天然携带 scene_index

对于超长 shot（> ASR_CHUNK_DURATION），会进一步内部切分再合并。
"""
import json
import time
from pathlib import Path

import config
from models.schemas import Scene, TranscriptSegment
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
3. 按时间顺序排列
4. 只输出 JSON，不要其他内容
5. 如果某段时间没有语音，跳过即可
6. 确保时间戳尽可能准确

输出格式示例：
```json
[
  {"start_time": 0.0, "end_time": 3.5, "text": "大家好，欢迎收看", "speaker": "speaker_1"},
  {"start_time": 4.2, "end_time": 8.1, "text": "今天我们来聊一个话题", "speaker": "speaker_1"}
]
```
"""


def transcribe_audio(
    video_id: str,
    video_path: str,
    scenes: list[Scene],
) -> list[TranscriptSegment]:
    """
    按镜头段落进行 ASR 转写。

    每个 scene 独立提取音频并转写，输出的 TranscriptSegment
    天然携带 scene_index，时间戳为全局时间。

    Args:
        video_id: 视频 ID
        video_path: 原始视频文件路径（用于按段提取音频）
        scenes: 镜头列表（需要 start_time / end_time）

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

    logger.info(f"开始按镜头 ASR 处理: {len(scenes)} 个镜头")

    client = get_llm_client()
    all_segments: list[TranscriptSegment] = []
    audio_dir = video_dir / "audio_shots"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for scene in scenes:
        shot_duration = scene.end_time - scene.start_time
        if shot_duration < 0.3:
            # 极短镜头跳过 ASR
            logger.debug(f"Scene {scene.scene_index}: 时长 {shot_duration:.1f}s 太短，跳过 ASR")
            continue

        # 1. 提取该 shot 的音频片段
        shot_audio_path = str(audio_dir / f"scene_{scene.scene_index:04d}.wav")
        try:
            extract_audio_segment(
                video_path, shot_audio_path,
                start_time=scene.start_time,
                end_time=scene.end_time,
            )
        except Exception as e:
            logger.warning(f"Scene {scene.scene_index}: 音频提取失败，跳过: {e}")
            continue

        # 2. 对于超长 shot，需要内部再切分
        if shot_duration > config.ASR_CHUNK_DURATION:
            segments = _transcribe_long_shot(
                client, shot_audio_path, scene, shot_duration
            )
        else:
            segments = _transcribe_shot(
                client, shot_audio_path, scene
            )

        if segments is None:
            logger.warning(f"Scene {scene.scene_index}: ASR 失败，跳过")
            continue

        all_segments.extend(segments)

        # API 速率控制
        time.sleep(0.5)

    # 按全局时间排序
    all_segments.sort(key=lambda s: s.start_time)

    # 保存结果
    data = [s.model_dump() for s in all_segments]
    transcript_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info(f"ASR 完成: {len(all_segments)} 段台词, 保存至 {transcript_path}")

    return all_segments


def _transcribe_shot(
    client, audio_path: str, scene: Scene,
) -> list[TranscriptSegment] | None:
    """
    对单个 shot 的音频做 ASR。

    Returns:
        TranscriptSegment 列表（全局时间戳，带 scene_index）。
        - [] 表示合法无语音内容
        - None 表示 API 调用或解析失败
    """
    try:
        response = client.chat_with_media(
            prompt=ASR_PROMPT,
            media_path=audio_path,
            temperature=0.1,
        )
        parsed = client.parse_json(response)
        if parsed is None:
            logger.error(f"ASR 响应解析失败 (scene {scene.scene_index}): {response[:200]}")
            return None
        if not isinstance(parsed, list):
            logger.error(f"ASR 响应格式不正确 (scene {scene.scene_index})")
            return None

        segments = []
        for item in parsed:
            # 段内相对时间 + shot 起始偏移 = 全局绝对时间
            seg_start = round(float(item.get("start_time", 0)) + scene.start_time, 1)
            seg_end = round(float(item.get("end_time", 0)) + scene.start_time, 1)

            # 时间修正：确保不超出 shot 边界
            seg_start = max(seg_start, scene.start_time)
            seg_end = min(seg_end, scene.end_time)
            if seg_end <= seg_start:
                continue

            seg = TranscriptSegment(
                start_time=seg_start,
                end_time=seg_end,
                text=item.get("text", ""),
                speaker=item.get("speaker"),
                scene_index=scene.scene_index,
            )
            if seg.text.strip():
                segments.append(seg)

        logger.info(f"  Scene {scene.scene_index}: 识别 {len(segments)} 句")
        return segments

    except Exception as e:
        logger.error(f"ASR 处理异常 (scene {scene.scene_index}): {e}")
        return None


def _transcribe_long_shot(
    client, audio_path: str, scene: Scene, shot_duration: float,
) -> list[TranscriptSegment] | None:
    """
    处理超长 shot 的 ASR：内部按 ASR_CHUNK_DURATION 切分后分别转写再合并。
    """
    from utils.ffmpeg_utils import split_audio

    chunks_dir = str(Path(audio_path).parent / f"chunks_scene_{scene.scene_index:04d}")
    chunk_paths = split_audio(audio_path, chunks_dir, chunk_seconds=config.ASR_CHUNK_DURATION)
    logger.info(f"  Scene {scene.scene_index}: 超长 shot ({shot_duration:.0f}s)，切分为 {len(chunk_paths)} 段")

    all_segments = []
    chunk_offset = 0.0

    for i, chunk_path in enumerate(chunk_paths):
        chunk_duration = get_audio_duration(chunk_path)

        # 构造一个虚拟 scene 来复用 _transcribe_shot 的时间偏移逻辑
        virtual_scene = Scene(
            scene_index=scene.scene_index,
            start_time=scene.start_time + chunk_offset,
            end_time=min(scene.start_time + chunk_offset + chunk_duration, scene.end_time),
            duration=chunk_duration,
        )

        segments = _transcribe_shot(client, chunk_path, virtual_scene)
        if segments is None:
            logger.warning(f"  Scene {scene.scene_index} chunk {i}: ASR 失败")
            return None
        all_segments.extend(segments)
        chunk_offset += chunk_duration
        time.sleep(0.5)

    return all_segments
