# -*- coding: utf-8 -*-
"""
FFmpeg 渲染操作封装
裁剪、拼接、转场、音频混合等渲染相关的 FFmpeg 操作。
"""
import tempfile
import os
from pathlib import Path

import config
from utils.ffmpeg_utils import _run_cmd, get_audio_duration
from utils.logger import get_logger

logger = get_logger("FFmpegOps")


def _run(cmd: list[str], timeout: int = 600):
    """运行 FFmpeg 命令（委托给统一封装）"""
    logger.debug(f"执行: {' '.join(cmd[:8])}...")
    _run_cmd(cmd, timeout=timeout)


def cut_clip_precise(
    source: str, start: float, end: float, output: str
) -> str:
    """
    精确裁剪视频片段（重编码模式，帧精确）。
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    duration = end - start
    cmd = [
        config.FFMPEG_PATH, "-y",
        "-ss", f"{start:.3f}",
        "-i", source,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        "-fflags", "+genpts",
        output,
    ]
    _run(cmd)
    return output


def normalize_clip(
    input_path: str, output: str,
    width: int = None, height: int = None, fps: float = None,
) -> str:
    """
    标准化视频参数（分辨率、帧率、编码），确保拼接兼容。
    """
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    filters = []
    if width and height:
        filters.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease")
        filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
    if fps:
        filters.append(f"fps={fps}")

    cmd = [
        config.FFMPEG_PATH, "-y",
        "-i", input_path,
    ]
    if filters:
        cmd.extend(["-vf", ",".join(filters)])
    cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        output,
    ])
    _run(cmd)
    return output


def concat_clips(clip_paths: list[str], output: str) -> str:
    """使用 concat demuxer 拼接片段"""
    if not clip_paths:
        raise ValueError("片段列表为空")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

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
            "-f", "concat", "-safe", "0",
            "-i", concat_file.name,
            "-c", "copy",
            output,
        ]
        _run(cmd)
    finally:
        try:
            os.unlink(concat_file.name)
        except OSError:
            pass

    return output


def apply_fade(
    input_path: str, output: str,
    fade_in: float = 0, fade_out: float = 0, duration: float = 0,
) -> str:
    """应用淡入淡出效果"""
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    filters = []
    if fade_in > 0:
        filters.append(f"fade=t=in:st=0:d={fade_in}")
    if fade_out > 0 and duration > 0:
        fade_start = max(0, duration - fade_out)
        filters.append(f"fade=t=out:st={fade_start:.3f}:d={fade_out}")

    audio_filters = []
    if fade_in > 0:
        audio_filters.append(f"afade=t=in:st=0:d={fade_in}")
    if fade_out > 0 and duration > 0:
        fade_start = max(0, duration - fade_out)
        audio_filters.append(f"afade=t=out:st={fade_start:.3f}:d={fade_out}")

    cmd = [config.FFMPEG_PATH, "-y", "-i", input_path]
    vf = ",".join(filters) if filters else None
    af = ",".join(audio_filters) if audio_filters else None

    if vf:
        cmd.extend(["-vf", vf])
    if af:
        cmd.extend(["-af", af])
    cmd.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        output,
    ])
    _run(cmd)
    return output


def mix_bgm(
    video_path: str, bgm_path: str, output: str,
    bgm_volume: float = 0.15, fade_in: float = 2.0, fade_out: float = 3.0,
) -> str:
    """混合背景音乐"""
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # 获取视频时长以计算淡出起始时间
    video_duration = get_audio_duration(video_path)
    fade_out_start = max(0, video_duration - fade_out)

    filter_complex = (
        f"[1:a]volume={bgm_volume},"
        f"afade=t=in:st=0:d={fade_in},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out}[bgm];"
        f"[0:a][bgm]amix=inputs=2:duration=first:normalize=0[aout]"
    )

    cmd = [
        config.FFMPEG_PATH, "-y",
        "-i", video_path,
        "-i", bgm_path,
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        output,
    ]
    _run(cmd)
    return output


def adjust_volume(input_path: str, output: str, volume: float) -> str:
    """调整音频音量"""
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.FFMPEG_PATH, "-y",
        "-i", input_path,
        "-af", f"volume={volume}",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        output,
    ]
    _run(cmd)
    return output


def adjust_speed(input_path: str, output: str, speed: float) -> str:
    """调整视频速度"""
    if abs(speed - 1.0) < 0.01:
        return input_path  # 无需处理

    Path(output).parent.mkdir(parents=True, exist_ok=True)

    # video: setpts=PTS/speed; audio: atempo=speed
    vf = f"setpts=PTS/{speed}"

    # atempo 只支持 0.5-2.0，超出范围需要链式
    atempo_filters = []
    remaining = speed
    while remaining > 2.0:
        atempo_filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        atempo_filters.append("atempo=0.5")
        remaining /= 0.5
    atempo_filters.append(f"atempo={remaining:.4f}")
    af = ",".join(atempo_filters)

    cmd = [
        config.FFMPEG_PATH, "-y",
        "-i", input_path,
        "-vf", vf,
        "-af", af,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        output,
    ]
    _run(cmd)
    return output
