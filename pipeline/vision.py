# -*- coding: utf-8 -*-
"""
画面理解 + OCR
使用 Gemini Vision 对关键帧进行分析：
- OCR：识别画面中的文字
- 画面摘要：生成详细的画面描述、物体、情绪、场景类型
为节省 API 调用，OCR 和画面摘要在同一次请求中完成。
"""
import json
import time
from pathlib import Path

import config
from models.schemas import Scene, OCRResult, VisionSummary
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("Vision")

VISION_PROMPT = """你是一个专业的视频画面分析系统。请仔细观察这张视频截图，完成以下两项任务：

任务一：OCR 文字识别
识别画面中出现的所有文字内容（包括字幕、标题、标牌、屏幕文字等）。

任务二：画面摘要
对画面进行详细分析。

请以 JSON 格式输出，只输出 JSON，不要其他内容：
```json
{
  "ocr_texts": ["画面中的文字1", "画面中的文字2"],
  "description": "详细的画面描述，包含场景、人物动作、构图、光线等",
  "objects": ["检测到的物体1", "物体2"],
  "mood": "画面传达的情绪（如：紧张、温馨、悲伤、欢快、平静、激昂等）",
  "scene_type": "场景类型（如：对话、动作、空镜、过渡、特写、全景、追逐等）"
}
```
"""

BATCH_VISION_PROMPT = """你是一个专业的视频画面分析系统。以下是同一个视频中多个镜头的关键帧截图。
请为每张图片分别进行分析。

对每张图片，请分析：
1. OCR：画面中出现的文字
2. 画面描述：详细描述场景内容
3. 检测到的物体
4. 画面情绪
5. 场景类型

请以 JSON 数组格式输出，数组中每个元素对应一张图片（按顺序），只输出 JSON：
```json
[
  {
    "ocr_texts": ["文字1"],
    "description": "画面描述",
    "objects": ["物体1"],
    "mood": "情绪",
    "scene_type": "场景类型"
  },
  ...
]
```
"""


def analyze_keyframes(
    video_id: str, scenes: list[Scene], batch_size: int = 5
) -> tuple[list[OCRResult], list[VisionSummary]]:
    """
    对所有关键帧进行画面分析（OCR + 摘要）。
    支持批量处理以减少 API 调用。

    Args:
        video_id: 视频 ID
        scenes: 带 keyframe_path 的镜头列表
        batch_size: 每批处理的关键帧数量

    Returns:
        (OCR 结果列表, 画面摘要列表)
    """
    video_dir = config.VIDEOS_DIR / video_id
    ocr_path = video_dir / "ocr.json"
    vision_path = video_dir / "vision.json"

    # 如果都已存在，直接加载
    if ocr_path.exists() and vision_path.exists():
        logger.info("OCR 和画面摘要结果已存在，直接加载")
        ocr_data = json.loads(ocr_path.read_text(encoding="utf-8"))
        vision_data = json.loads(vision_path.read_text(encoding="utf-8"))
        return (
            [OCRResult(**o) for o in ocr_data],
            [VisionSummary(**v) for v in vision_data],
        )

    logger.info(f"开始画面分析: {len(scenes)} 个镜头")

    client = get_llm_client()
    all_ocr = []
    all_vision = []

    # 过滤出有关键帧的场景
    valid_scenes = [s for s in scenes if s.keyframe_path and Path(s.keyframe_path).exists()]
    logger.info(f"有效关键帧: {len(valid_scenes)}/{len(scenes)}")

    # 分批处理
    for batch_start in range(0, len(valid_scenes), batch_size):
        batch = valid_scenes[batch_start : batch_start + batch_size]
        batch_end = batch_start + len(batch)
        logger.info(f"处理批次 {batch_start // batch_size + 1}: 镜头 {batch_start}-{batch_end - 1}")

        if len(batch) == 1:
            # 单张图片
            ocr, vision = _analyze_single(client, batch[0])
            if ocr is None or vision is None:
                raise RuntimeError(
                    f"Vision 处理失败: 镜头 {batch[0].scene_index} API 调用或解析失败"
                )
            all_ocr.append(ocr)
            all_vision.append(vision)
        else:
            # 批量处理
            ocr_list, vision_list = _analyze_batch(client, batch)
            if ocr_list is None or vision_list is None:
                raise RuntimeError(
                    f"Vision 处理失败: 批次 {batch_start // batch_size + 1} "
                    f"(镜头 {batch_start}-{batch_end - 1}) API 调用或解析失败"
                )
            all_ocr.extend(ocr_list)
            all_vision.extend(vision_list)

        # API 速率控制
        time.sleep(1)

    # 保存结果
    ocr_path.write_text(
        json.dumps([o.model_dump() for o in all_ocr], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    vision_path.write_text(
        json.dumps([v.model_dump() for v in all_vision], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info(
        f"画面分析完成: {len(all_ocr)} 个 OCR, {len(all_vision)} 个画面摘要"
    )
    return all_ocr, all_vision


def _analyze_single(client, scene: Scene) -> tuple[OCRResult | None, VisionSummary | None]:
    """分析单个关键帧"""
    try:
        response = client.chat_with_media(
            prompt=VISION_PROMPT,
            media_path=scene.keyframe_path,
            temperature=0.3,
        )
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, dict):
            logger.warning(f"画面分析解析失败 (scene {scene.scene_index})")
            return None, None

        ocr = OCRResult(
            scene_index=scene.scene_index,
            timestamp=scene.start_time,
            texts=parsed.get("ocr_texts", []),
        )
        vision = VisionSummary(
            scene_index=scene.scene_index,
            timestamp=scene.start_time,
            description=parsed.get("description", ""),
            objects=parsed.get("objects", []),
            mood=parsed.get("mood", ""),
            scene_type=parsed.get("scene_type", ""),
        )
        return ocr, vision

    except Exception as e:
        logger.error(f"画面分析失败 (scene {scene.scene_index}): {e}")
        return None, None


def _analyze_batch(
    client, scenes: list[Scene]
) -> tuple[list[OCRResult], list[VisionSummary]] | tuple[None, None]:
    """
    批量分析多个关键帧。

    Returns:
        (ocr_list, vision_list) 成功时
        (None, None) API 调用异常且逐个重试仍失败时
    """
    image_paths = [s.keyframe_path for s in scenes]

    try:
        response = client.chat_with_images(
            prompt=BATCH_VISION_PROMPT,
            image_paths=image_paths,
            temperature=0.3,
        )
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, list):
            logger.warning("批量画面分析解析失败，退回逐个处理")
            # 批量解析失败 → 逐个重试（仍可能成功）
            ocr_list, vision_list = [], []
            for scene in scenes:
                ocr, vision = _analyze_single(client, scene)
                if ocr is None or vision is None:
                    logger.error(f"镜头 {scene.scene_index} 逐个重试仍失败")
                    return None, None
                ocr_list.append(ocr)
                vision_list.append(vision)
                time.sleep(0.5)
            return ocr_list, vision_list

        ocr_list = []
        vision_list = []
        for i, item in enumerate(parsed):
            if i >= len(scenes):
                break
            scene = scenes[i]
            ocr_list.append(OCRResult(
                scene_index=scene.scene_index,
                timestamp=scene.start_time,
                texts=item.get("ocr_texts", []),
            ))
            vision_list.append(VisionSummary(
                scene_index=scene.scene_index,
                timestamp=scene.start_time,
                description=item.get("description", ""),
                objects=item.get("objects", []),
                mood=item.get("mood", ""),
                scene_type=item.get("scene_type", ""),
            ))

        return ocr_list, vision_list

    except Exception as e:
        logger.error(f"批量画面分析异常: {e}")
        return None, None
