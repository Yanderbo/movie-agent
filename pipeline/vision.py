# -*- coding: utf-8 -*-
"""
画面理解 + OCR（v2 — 多帧输入）

v2 变更:
- 从单关键帧改为多帧输入，一次分析整个 shot 的多帧
- prompt 增加动作变化、表情变化、道具识别
- 新增 action_description / frame_descriptions / expression_changes / props 字段
- 对只有单帧的 shot 仍正常工作（向后兼容）

为节省 API 调用，OCR 和画面摘要在同一次请求中完成。
"""
import json
import time
from pathlib import Path

import config
from models.schemas import Shot, OCRResult, VisionSummary
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("Vision")

# ── 单帧 prompt（兼容旧模式）──
VISION_PROMPT_SINGLE = """你是一个专业的视频画面分析系统。请仔细观察这张视频截图，完成以下两项任务：

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
  "scene_type": "场景类型（如：对话、动作、空镜、过渡、特写、全景、追逐等）",
  "props": ["关键道具1", "道具2"]
}
```
"""

# ── 多帧 prompt（v2 核心）──
VISION_PROMPT_MULTI = """你是一个专业的视频画面分析系统。以下是同一个镜头（shot）内按时间顺序排列的多帧截图。
请综合分析这些帧，理解镜头内发生了什么。

请完成以下分析：
1. OCR：识别所有帧中出现的文字
2. 综合画面描述：描述这个镜头内的场景和核心内容
3. 动作/变化描述：通过对比各帧，描述镜头内发生的动作、运动和变化
4. 表情变化：如有人物，描述其表情在帧间的变化
5. 物体 & 关键道具
6. 情绪 & 场景类型
7. 镜头运动：分析镜头的运动方式（static/pan_left/pan_right/tilt_up/tilt_down/zoom_in/zoom_out/tracking/crane/handheld）
8. 人物互动：如有多人，描述人物间的互动方式（对话、肢体接触、对峙、合作等）
9. 景别：判断镜头的景别（extreme_close_up/close_up/medium_close/medium/medium_long/long/extreme_long）

请以 JSON 格式输出，只输出 JSON：
```json
{
  "ocr_texts": ["文字1", "文字2"],
  "description": "综合画面描述",
  "action_description": "动作/变化描述（从第1帧到最后1帧发生了什么）",
  "frame_descriptions": ["第1帧描述", "第2帧描述", "..."],
  "expression_changes": "人物表情变化描述（无人物则为空）",
  "objects": ["物体1", "物体2"],
  "props": ["关键道具1"],
  "mood": "整体情绪",
  "scene_type": "场景类型",
  "camera_motion": "镜头运动方式",
  "interaction_description": "人物互动描述（无互动则为空）",
  "shot_scale": "景别"
}
```
"""

# ── 批量多 shot 的 prompt ──
BATCH_VISION_PROMPT = """你是一个专业的视频画面分析系统。以下是同一个视频中多个镜头的关键帧截图。
请为每张图片分别进行分析。

对每张图片，请分析：
1. OCR：画面中出现的文字
2. 画面描述：详细描述场景内容
3. 检测到的物体
4. 画面情绪
5. 场景类型
6. 关键道具

请以 JSON 数组格式输出，数组中每个元素对应一张图片（按顺序），只输出 JSON：
```json
[
  {
    "ocr_texts": ["文字1"],
    "description": "画面描述",
    "objects": ["物体1"],
    "mood": "情绪",
    "scene_type": "场景类型",
    "props": ["道具1"]
  },
  ...
]
```
"""


def analyze_keyframes(
    video_id: str, scenes: list[Shot], batch_size: int = 5
) -> tuple[list[OCRResult], list[VisionSummary]]:
    """
    对所有关键帧进行画面分析（OCR + 摘要）。
    v2: 支持多帧输入，自动检测 shot 有几帧并选择合适的 prompt。

    Args:
        video_id: 视频 ID
        scenes: 带 keyframe_path(s) 的镜头列表
        batch_size: 单帧模式下每批处理的关键帧数量

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

    # 区分多帧 shot 和单帧 shot
    multi_frame_shots = []
    single_frame_shots = []

    for s in scenes:
        valid_paths = [p for p in (s.keyframe_paths or []) if p and Path(p).exists()]
        if len(valid_paths) >= 2:
            multi_frame_shots.append((s, valid_paths))
        elif valid_paths:
            single_frame_shots.append(s)
        elif s.keyframe_path and Path(s.keyframe_path).exists():
            single_frame_shots.append(s)

    logger.info(
        f"多帧 shot: {len(multi_frame_shots)}, 单帧 shot: {len(single_frame_shots)}"
    )

    # ── 处理多帧 shot（逐 shot 调用）──
    for shot, frame_paths in multi_frame_shots:
        logger.info(
            f"  多帧分析 shot {shot.scene_index}: {len(frame_paths)} 帧"
        )
        ocr, vision = _analyze_multi_frame(client, shot, frame_paths)
        if ocr is None or vision is None:
            raise RuntimeError(
                f"Vision 处理失败: 镜头 {shot.scene_index} 多帧分析失败"
            )
        all_ocr.append(ocr)
        all_vision.append(vision)
        time.sleep(1)

    # ── 处理单帧 shot（批量）──
    for batch_start in range(0, len(single_frame_shots), batch_size):
        batch = single_frame_shots[batch_start : batch_start + batch_size]
        batch_end = batch_start + len(batch)
        logger.info(
            f"  单帧批次 {batch_start // batch_size + 1}: "
            f"镜头 {batch[0].scene_index}-{batch[-1].scene_index}"
        )

        if len(batch) == 1:
            ocr, vision = _analyze_single(client, batch[0])
            if ocr is None or vision is None:
                raise RuntimeError(
                    f"Vision 处理失败: 镜头 {batch[0].scene_index}"
                )
            all_ocr.append(ocr)
            all_vision.append(vision)
        else:
            ocr_list, vision_list = _analyze_batch(client, batch)
            if ocr_list is None or vision_list is None:
                raise RuntimeError(
                    f"Vision 处理失败: 批次 {batch_start // batch_size + 1}"
                )
            all_ocr.extend(ocr_list)
            all_vision.extend(vision_list)

        time.sleep(1)

    # 按 scene_index 排序
    all_ocr.sort(key=lambda o: o.scene_index)
    all_vision.sort(key=lambda v: v.scene_index)

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


# ═══════════════════════════════════════════════════════════════
# 多帧分析（v2 核心）
# ═══════════════════════════════════════════════════════════════

def _analyze_multi_frame(
    client, shot: Shot, frame_paths: list[str]
) -> tuple[OCRResult | None, VisionSummary | None]:
    """分析单个 shot 的多帧关键帧 — v2 核心能力"""
    try:
        response = client.chat_with_images(
            prompt=VISION_PROMPT_MULTI,
            image_paths=frame_paths,
            temperature=0.3,
        )
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, dict):
            logger.warning(
                f"多帧分析解析失败 (shot {shot.scene_index})，退回单帧"
            )
            return _analyze_single(client, shot)

        ocr = OCRResult(
            scene_index=shot.scene_index,
            timestamp=shot.start_time,
            texts=parsed.get("ocr_texts", []),
        )
        vision = VisionSummary(
            scene_index=shot.scene_index,
            timestamp=shot.start_time,
            description=parsed.get("description", ""),
            objects=parsed.get("objects", []),
            mood=parsed.get("mood", ""),
            scene_type=parsed.get("scene_type", ""),
            camera_motion=parsed.get("camera_motion", ""),
            interaction_description=parsed.get("interaction_description", ""),
            shot_scale=parsed.get("shot_scale", ""),
            action_description=parsed.get("action_description", ""),
            frame_descriptions=parsed.get("frame_descriptions", []),
            expression_changes=parsed.get("expression_changes", ""),
            props=parsed.get("props", []),
        )
        return ocr, vision

    except Exception as e:
        logger.warning(
            f"多帧分析异常 (shot {shot.scene_index}): {e}，退回单帧"
        )
        return _analyze_single(client, shot)


# ═══════════════════════════════════════════════════════════════
# 单帧分析（兼容旧逻辑）
# ═══════════════════════════════════════════════════════════════

def _analyze_single(client, scene: Shot) -> tuple[OCRResult | None, VisionSummary | None]:
    """分析单个关键帧"""
    kf = scene.keyframe_path
    if not kf or not Path(kf).exists():
        # 尝试 keyframe_paths 的第一帧
        if scene.keyframe_paths:
            kf = next((p for p in scene.keyframe_paths if Path(p).exists()), None)
    if not kf:
        logger.warning(f"shot {scene.scene_index}: 无可用关键帧")
        return None, None

    try:
        response = client.chat_with_media(
            prompt=VISION_PROMPT_SINGLE,
            media_path=kf,
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
            props=parsed.get("props", []),
            camera_motion=parsed.get("camera_motion", ""),
            interaction_description=parsed.get("interaction_description", ""),
            shot_scale=parsed.get("shot_scale", ""),
        )
        return ocr, vision

    except Exception as e:
        logger.error(f"画面分析失败 (scene {scene.scene_index}): {e}")
        return None, None


def _analyze_batch(
    client, scenes: list[Shot]
) -> tuple[list[OCRResult], list[VisionSummary]] | tuple[None, None]:
    """
    批量分析多个关键帧（单帧模式）。

    Returns:
        (ocr_list, vision_list) 成功时
        (None, None) API 调用异常且逐个重试仍失败时
    """
    image_paths = []
    for s in scenes:
        kf = s.keyframe_path
        if not kf or not Path(kf).exists():
            if s.keyframe_paths:
                kf = next((p for p in s.keyframe_paths if Path(p).exists()), None)
        image_paths.append(kf)

    # 过滤 None
    valid_pairs = [(s, p) for s, p in zip(scenes, image_paths) if p]
    if not valid_pairs:
        return None, None
    valid_scenes, valid_paths = zip(*valid_pairs)
    valid_scenes = list(valid_scenes)
    valid_paths = list(valid_paths)

    try:
        response = client.chat_with_images(
            prompt=BATCH_VISION_PROMPT,
            image_paths=valid_paths,
            temperature=0.3,
        )
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, list):
            logger.warning("批量画面分析解析失败，退回逐个处理")
            ocr_list, vision_list = [], []
            for scene in valid_scenes:
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
            if i >= len(valid_scenes):
                break
            scene = valid_scenes[i]
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
                props=item.get("props", []),
                camera_motion=item.get("camera_motion", ""),
                interaction_description=item.get("interaction_description", ""),
                shot_scale=item.get("shot_scale", ""),
            ))

        return ocr_list, vision_list

    except Exception as e:
        logger.error(f"批量画面分析异常: {e}")
        return None, None
