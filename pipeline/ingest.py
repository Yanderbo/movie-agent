# -*- coding: utf-8 -*-
"""
视频入库
- 复制视频到项目数据目录
- 使用 ffprobe 解析元信息
- 生成 video_id 和 meta.json
"""
import re
import hashlib
import shutil
import uuid
from pathlib import Path

import config
from models.schemas import VideoMeta
from utils.ffmpeg_utils import get_video_info, compress_video, duration_matches
from utils.logger import get_logger

logger = get_logger("Ingest")


def _generate_readable_video_id(video_path: str) -> str:
    """
    生成人类可读的 video_id。

    格式: {sanitized_stem}_{8位hash}
    例如: my_movie_3f7a2b1c

    sanitized_stem 取原始文件名（去掉扩展名），将非字母数字字符替换为下划线，
    然后截断到 30 个字符以避免路径过长。
    hash 部分使用文件路径 + 文件大小的 MD5 前 8 位，确保不同文件不会碰撞。
    """
    stem = Path(video_path).stem
    # 替换非字母数字和下划线的字符
    sanitized = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '_', stem)
    # 合并连续下划线
    sanitized = re.sub(r'_+', '_', sanitized).strip('_').lower()
    # 截断
    if len(sanitized) > 30:
        sanitized = sanitized[:30]
    if not sanitized:
        sanitized = "video"

    # 基于路径 + 文件大小生成 hash（保证唯一性）
    file_size = Path(video_path).stat().st_size
    hash_input = f"{Path(video_path).resolve()}:{file_size}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]

    return f"{sanitized}_{short_hash}"


def ingest_video(video_path: str, video_id: str = None) -> VideoMeta:
    """
    将视频入库：复制到数据目录，压缩（如需要），解析元信息。

    v4.1 变更：新增视频压缩步骤，生成 compressed.mp4 用于理解流水线。
    渲染阶段仍使用原始视频。

    Args:
        video_path: 原始视频文件路径
        video_id: 可选，指定 video_id；不传则自动生成

    Returns:
        VideoMeta 对象
    """
    src = Path(video_path)
    if not src.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    # 生成 video_id（人类可读：文件名 + hash）
    if not video_id:
        video_id = _generate_readable_video_id(video_path)

    # 创建视频目录
    video_dir = config.VIDEOS_DIR / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    # 复制视频到数据目录
    dest = video_dir / f"original{src.suffix}"
    if not dest.exists():
        logger.info(f"复制视频: {src} → {dest}")
        shutil.copy2(str(src), str(dest))
    else:
        logger.info(f"视频已存在，跳过复制: {dest}")

    # 解析元信息
    logger.info("解析视频元信息...")
    info = get_video_info(str(dest))

    # ── v4.1: 视频压缩 ──
    compressed_dest = video_dir / "compressed.mp4"
    compress_result = {"compressed": False}
    if compressed_dest.exists():
        logger.info(f"压缩视频已存在，跳过压缩: {compressed_dest}")
        try:
            compressed_info = get_video_info(str(compressed_dest))
            compress_result = {
                "compressed": True,
                "output_path": str(compressed_dest),
                "original_height": info["height"],
                "original_fps": info["fps"],
                "compressed_height": compressed_info["height"],
                "compressed_fps": compressed_info["fps"],
            }
        except Exception as e:
            logger.warning(f"压缩视频不可读，将重新生成: {e}")
            compressed_dest.unlink(missing_ok=True)

    if not compressed_dest.exists():
        compress_result = compress_video(
            str(dest), str(compressed_dest),
            max_height=config.COMPRESS_MAX_HEIGHT,
            max_fps=config.COMPRESS_MAX_FPS,
        )

    # 确定理解流水线使用的视频路径
    if compress_result["compressed"]:
        compressed_info = get_video_info(compress_result["output_path"])
        if not duration_matches(info["duration"], compressed_info["duration"]):
            logger.warning(
                "压缩视频时长异常，将重新生成: "
                f"source={info['duration']:.3f}s, compressed={compressed_info['duration']:.3f}s"
            )
            compressed_dest.unlink(missing_ok=True)
            compress_result = compress_video(
                str(dest), str(compressed_dest),
                max_height=config.COMPRESS_MAX_HEIGHT,
                max_fps=config.COMPRESS_MAX_FPS,
            )

    pipeline_video = str(compressed_dest) if compress_result["compressed"] else str(dest)

    meta = VideoMeta(
        video_id=video_id,
        filename=src.name,
        original_path=str(src.resolve()),
        storage_path=pipeline_video,  # v4.1: 指向压缩视频（如果有）
        duration=info["duration"],
        width=info["width"],
        height=info["height"],
        fps=info["fps"],
        codec=info["codec"],
        file_size=info["file_size"],
        status="ingested",
        # v4.1 压缩信息
        compressed_path=str(compressed_dest) if compress_result["compressed"] else "",
        is_compressed=compress_result["compressed"],
        original_height=compress_result.get("original_height", info["height"]),
        original_fps=compress_result.get("original_fps", info["fps"]),
        compressed_height=compress_result.get("compressed_height", 0),
        compressed_fps=compress_result.get("compressed_fps", 0.0),
    )

    # 保存 meta.json
    meta_path = video_dir / "meta.json"
    meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    logger.info(f"元信息已保存: {meta_path}")
    logger.info(
        f"视频入库完成: id={video_id}, "
        f"时长={meta.duration:.1f}s, "
        f"分辨率={meta.width}x{meta.height}, "
        f"帧率={meta.fps}"
    )
    if compress_result["compressed"]:
        logger.info(
            f"  压缩: {compress_result['original_height']}p@{compress_result['original_fps']}fps → "
            f"{compress_result['compressed_height']}p@{compress_result['compressed_fps']}fps"
        )

    return meta
