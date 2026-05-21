# -*- coding: utf-8 -*-
"""
FFmpeg / FFprobe 工具函数
封装常用的视频处理操作，所有路径配置化。
"""
import json
import subprocess
from pathlib import Path

import config
from utils.logger import get_logger

logger = get_logger("FFmpegUtils")


def _run_cmd(cmd: list[str], timeout: int | None = 600) -> subprocess.CompletedProcess:
    """运行外部命令，捕获输出"""
    logger.debug(f"执行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=timeout
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"命令失败: {e.stderr[:500]}")
        raise RuntimeError(f"命令执行失败: {e.stderr[:500]}") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"命令超时 ({timeout}s): {' '.join(cmd[:3])}...") from e


def duration_matches(
    expected: float,
    actual: float,
    tolerance_seconds: float = 5.0,
    tolerance_ratio: float = 0.01,
) -> bool:
    if expected <= 0 or actual <= 0:
        return False
    tolerance = max(tolerance_seconds, expected * tolerance_ratio)
    return abs(expected - actual) <= tolerance


def probe_video(video_path: str) -> dict:
    """
    使用 ffprobe 获取视频元信息。
    返回 ffprobe JSON 格式的完整结果。
    """
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cmd = [
        config.FFPROBE_PATH,
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]
    result = _run_cmd(cmd)
    return json.loads(result.stdout)


def get_video_info(video_path: str) -> dict:
    """
    获取视频关键信息（时长、分辨率、帧率、编码）。
    返回简化的 dict。
    """
    probe = probe_video(video_path)

    video_stream = next(
        (s for s in probe.get("streams", []) if s["codec_type"] == "video"),
        None,
    )
    if not video_stream:
        raise ValueError(f"未找到视频流: {video_path}")

    # 解析帧率
    r_frame_rate = video_stream.get("r_frame_rate", "30/1")
    num, den = map(int, r_frame_rate.split("/"))
    fps = num / den if den != 0 else 30.0

    duration = float(probe.get("format", {}).get("duration", 0))

    return {
        "duration": duration,
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": round(fps, 2),
        "codec": video_stream.get("codec_name", "unknown"),
        "file_size": int(probe.get("format", {}).get("size", 0)),
    }


def compress_video(
    src: str, dst: str,
    max_height: int = None, max_fps: int = None,
) -> dict:
    """
    压缩视频用于理解流水线（v4.1 新增）。

    仅在分辨率或帧率超过阈值时进行压缩，否则跳过。
    音频重编码为 AAC，避免源文件异常时间戳导致压缩产物截断。

    Args:
        src: 源视频路径
        dst: 输出路径
        max_height: 高度阈值，超过则缩放（默认从 config 读取）
        max_fps: 帧率阈值，超过则降帧率（默认从 config 读取）

    Returns:
        dict: {
            "compressed": bool,      # 是否实际执行了压缩
            "output_path": str,      # 输出文件路径
            "original_height": int,
            "original_fps": float,
            "compressed_height": int,
            "compressed_fps": float,
        }
    """
    if max_height is None:
        max_height = config.COMPRESS_MAX_HEIGHT
    if max_fps is None:
        max_fps = config.COMPRESS_MAX_FPS

    info = get_video_info(src)
    original_height = info["height"]
    original_fps = info["fps"]
    original_duration = info["duration"]

    need_scale = original_height > max_height
    need_fps = original_fps > max_fps

    if not need_scale and not need_fps:
        logger.info(
            f"视频无需压缩: {original_height}p @ {original_fps}fps "
            f"(阈值: {max_height}p @ {max_fps}fps)"
        )
        return {
            "compressed": False,
            "output_path": src,
            "original_height": original_height,
            "original_fps": original_fps,
            "compressed_height": original_height,
            "compressed_fps": original_fps,
        }

    Path(dst).parent.mkdir(parents=True, exist_ok=True)

    cmd = [config.FFMPEG_PATH, "-y", "-i", src]
    filters = []

    # 视频滤镜：按比例缩放，宽度自适应保持偶数
    if need_scale:
        filters.append(f"scale=-2:{max_height}")

    # 帧率
    if need_fps:
        filters.insert(0, f"fps={max_fps}")

    if filters:
        cmd.extend(["-vf", ",".join(filters)])

    # 视频编码：使用较快的预设
    cmd.extend([
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
    ])

    # 音频重编码，避免继承异常时间戳
    cmd.extend(["-c:a", "aac", "-b:a", "128k"])

    cmd.append(dst)

    compressed_height = max_height if need_scale else original_height
    compressed_fps = float(max_fps) if need_fps else original_fps

    logger.info(
        f"压缩视频: {original_height}p@{original_fps}fps → "
        f"{compressed_height}p@{compressed_fps}fps"
    )
    try:
        _run_cmd(cmd, timeout=config.FFMPEG_COMPRESS_TIMEOUT or None)
    except Exception:
        Path(dst).unlink(missing_ok=True)
        raise
    logger.info(f"视频压缩完成: {dst}")

    compressed_info = get_video_info(dst)
    if not duration_matches(original_duration, compressed_info["duration"]):
        Path(dst).unlink(missing_ok=True)
        raise RuntimeError(
            "Compressed video duration mismatch: "
            f"source={original_duration:.3f}s, compressed={compressed_info['duration']:.3f}s"
        )

    return {
        "compressed": True,
        "output_path": dst,
        "original_height": original_height,
        "original_fps": original_fps,
        "compressed_height": compressed_info["height"],
        "compressed_fps": compressed_info["fps"],
    }


def extract_video_segment(
    video_path: str, output_path: str,
    start_time: float, end_time: float,
) -> str:
    """
    从视频中截取指定时间段的视频片段（含音频）。
    用于 MinuteChunk 处理时提取分钟级视频段。

    Args:
        video_path: 源视频路径
        output_path: 输出路径
        start_time: 起始时间(秒)
        end_time: 结束时间(秒)

    Returns:
        输出文件路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError(f"时间段无效: start={start_time}, end={end_time}")

    cmd = [
        config.FFMPEG_PATH, "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-c", "copy",
        output_path,
    ]
    _run_cmd(cmd, timeout=120)
    logger.debug(f"视频段提取完成: [{start_time:.1f}s-{end_time:.1f}s] -> {output_path}")
    return output_path


def extract_audio(video_path: str, output_path: str, sample_rate: int = 16000) -> str:
    """
    从视频中提取音频。
    输出为 WAV 格式（适合 ASR）。
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.FFMPEG_PATH,
        "-y",
        "-i", video_path,
        "-vn",                # 不要视频
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",           # 单声道
        output_path,
    ]
    _run_cmd(cmd)
    logger.info(f"音频提取完成: {output_path}")
    return output_path


def extract_audio_segment(
    video_path: str, output_path: str,
    start_time: float, end_time: float,
    sample_rate: int = 16000,
) -> str:
    """
    从视频中提取指定时间段的音频。

    用于按 shot 段切分后单独提取音频，保证 ASR 时间轴
    与镜头切分结果天然对齐。

    Args:
        video_path: 视频文件路径
        output_path: 输出音频路径
        start_time: 起始时间（秒）
        end_time: 结束时间（秒）
        sample_rate: 采样率

    Returns:
        输出文件路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    duration = end_time - start_time
    if duration <= 0:
        raise ValueError(f"时间段无效: start={start_time}, end={end_time}")

    cmd = [
        config.FFMPEG_PATH,
        "-y",
        "-ss", str(start_time),
        "-i", video_path,
        "-t", str(duration),
        "-vn",                # 不要视频
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",           # 单声道
        output_path,
    ]
    _run_cmd(cmd, timeout=120)
    logger.debug(f"音频段提取完成: [{start_time:.1f}s-{end_time:.1f}s] -> {output_path}")
    return output_path


def extract_keyframe(video_path: str, timestamp: float, output_path: str) -> str:
    """
    从视频中提取指定时间点的关键帧。
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.FFMPEG_PATH,
        "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-vframes", "1",
        "-q:v", str(config.KEYFRAME_QUALITY),
        output_path,
    ]
    _run_cmd(cmd, timeout=60)
    return output_path


def cut_clip(
    source: str, start: float, end: float, output: str, re_encode: bool = False
) -> str:
    """
    裁剪视频片段。
    re_encode=False 使用 stream copy（快但可能不精确）
    re_encode=True 重新编码（慢但精确）
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    duration = end - start

    if re_encode:
        cmd = [
            config.FFMPEG_PATH, "-y",
            "-ss", str(start),
            "-i", source,
            "-t", str(duration),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            output,
        ]
    else:
        cmd = [
            config.FFMPEG_PATH, "-y",
            "-ss", str(start),
            "-i", source,
            "-t", str(duration),
            "-c", "copy",
            output,
        ]
    _run_cmd(cmd)
    return output


def concat_clips(clip_paths: list[str], output: str) -> str:
    """
    按顺序拼接多个视频片段（使用 concat demuxer）。

    .. deprecated::
        渲染阶段请使用 render.ffmpeg_ops.concat_clips()。
        此函数保留以兼容旧代码，后续版本可能移除。
    """
    import tempfile

    if not clip_paths:
        raise ValueError("片段列表不能为空")

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # 写 concat 文件
    concat_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="concat_", delete=False, encoding="utf-8"
    )
    try:
        for p in clip_paths:
            resolved = str(Path(p).resolve()).replace("\\", "/")
            escaped = resolved.replace("'", r"'\''")
            concat_file.write(f"file '{escaped}'\n")
        concat_file.close()

        cmd = [
            config.FFMPEG_PATH, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file.name,
            "-c", "copy",
            output,
        ]
        _run_cmd(cmd)
    finally:
        import os
        try:
            os.unlink(concat_file.name)
        except OSError:
            pass

    logger.info(f"拼接完成 ({len(clip_paths)} 个片段): {output}")
    return output


def get_audio_duration(audio_path: str) -> float:
    """获取音频文件时长（秒）"""
    probe = probe_video(audio_path)
    return float(probe.get("format", {}).get("duration", 0))


def split_audio(audio_path: str, output_dir: str, chunk_seconds: int = 600) -> list[str]:
    """
    将长音频按固定时长切分为多段。
    返回切分后的文件路径列表。
    """
    duration = get_audio_duration(audio_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    chunks = []
    start = 0.0
    idx = 0
    while start < duration:
        end = min(start + chunk_seconds, duration)
        chunk_path = str(Path(output_dir) / f"chunk_{idx:03d}.wav")
        cmd = [
            config.FFMPEG_PATH, "-y",
            "-ss", str(start),
            "-i", audio_path,
            "-t", str(end - start),
            "-c", "copy",
            chunk_path,
        ]
        _run_cmd(cmd, timeout=120)
        chunks.append(chunk_path)
        start = end
        idx += 1

    logger.info(f"音频切分完成: {len(chunks)} 段")
    return chunks
