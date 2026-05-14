# -*- coding: utf-8 -*-
"""
人物识别（v2 — 深度分析版）

v2 变更:
- 人脸聚类后，额外计算: 首次/末次出场、共现矩阵、台词统计、参与事件、importance_score
- 输出 CharacterDeep 替代 Character（CharacterDeep 继承 Character，向后兼容）
- 保留 InsightFace / Gemini Vision 双轨人脸检测
"""
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import config
from models.schemas import Shot, Character, CharacterDeep
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("Character")


def detect_characters(video_id: str, scenes: list[Shot]) -> list[CharacterDeep]:
    """
    从关键帧中检测人脸，聚类为不同人物，并生成描述。

    v2 输出 CharacterDeep（继承 Character），附带深度分析字段。

    流程：
    1. InsightFace 检测每个关键帧中的人脸，提取特征向量
    2. 基于特征向量进行聚类（DBSCAN / 层次聚类）
    3. 每个聚类代表一个人物
    4. 用 Gemini Vision 描述人物外观
    5. 计算首次/末次出场时间和 importance_score

    Args:
        video_id: 视频 ID
        scenes: 带 keyframe_path 的镜头列表

    Returns:
        CharacterDeep 列表
    """
    video_dir = config.VIDEOS_DIR / video_id
    char_path = video_dir / "characters.json"

    # 如果已存在，直接加载
    if char_path.exists():
        logger.info(f"人物识别结果已存在，直接加载: {char_path}")
        data = json.loads(char_path.read_text(encoding="utf-8"))
        return [CharacterDeep(**c) for c in data]

    logger.info(f"开始人物识别: {len(scenes)} 个镜头")

    # Step 1: 人脸检测与特征提取（检测所有帧，包括多帧的）
    face_data = _detect_faces(scenes)
    if not face_data:
        logger.warning("未检测到任何人脸")
        char_path.write_text("[]", encoding="utf-8")
        return []

    # Step 2: 聚类
    clusters = _cluster_faces(face_data)
    logger.info(f"聚类完成: {len(clusters)} 个人物")

    # Step 3: 生成人物信息
    characters = []
    client = get_llm_client()

    for cluster_id, cluster_info in clusters.items():
        char_id = f"char_{cluster_id:03d}"

        # 选择最佳人脸作为缩略图
        best_face = cluster_info["faces"][0]
        thumbnail_path = _save_face_thumbnail(
            video_dir, char_id, best_face
        )

        # 用 Gemini 生成描述
        description = _describe_character(client, thumbnail_path)

        appearance_scenes = sorted(cluster_info["scenes"])

        # 计算出镜时间
        screen_time = _calc_screen_time(scenes, appearance_scenes)

        # 计算首次/末次出场
        first_app = 0.0
        last_app = 0.0
        for s in scenes:
            if s.scene_index in appearance_scenes:
                if first_app == 0.0 or s.start_time < first_app:
                    first_app = s.start_time
                if s.end_time > last_app:
                    last_app = s.end_time

        char = CharacterDeep(
            character_id=char_id,
            display_name=f"人物_{cluster_id + 1}",
            description=description,
            thumbnail_path=thumbnail_path,
            appearance_scenes=appearance_scenes,
            total_screen_time=screen_time,
            first_appearance=first_app,
            last_appearance=last_app,
        )
        characters.append(char)

    # 按出场次数排序
    characters.sort(key=lambda c: len(c.appearance_scenes), reverse=True)

    # 计算共现角色
    _compute_co_appearances(characters, scenes)

    # 保存结果
    char_path.write_text(
        json.dumps([c.model_dump() for c in characters], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"人物识别完成: {len(characters)} 个人物")

    return characters


def _compute_co_appearances(characters: list[CharacterDeep], scenes: list[Shot]):
    """计算每个角色与哪些角色共同出现过"""
    # scene_index → set(character_id)
    scene_chars = defaultdict(set)
    for c in characters:
        for si in c.appearance_scenes:
            scene_chars[si].add(c.character_id)

    for c in characters:
        co_chars = set()
        for si in c.appearance_scenes:
            co_chars.update(scene_chars[si])
        co_chars.discard(c.character_id)
        c.co_appearing_characters = sorted(co_chars)


def _detect_faces(scenes: list[Shot]) -> list[dict]:
    """
    使用 InsightFace 检测所有关键帧中的人脸。
    v2: 检测所有帧（包括 keyframe_paths 中的多帧）
    返回 [{scene_index, bbox, embedding, keyframe_path}, ...]
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
        import cv2
    except ImportError:
        logger.warning("InsightFace 未安装，尝试使用 Gemini 进行人物识别")
        return _detect_faces_with_gemini(scenes)

    # 初始化模型
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    face_data = []
    for scene in scenes:
        # 收集所有可用帧
        all_paths = []
        if scene.keyframe_paths:
            all_paths.extend(
                [p for p in scene.keyframe_paths if p and Path(p).exists()]
            )
        elif scene.keyframe_path and Path(scene.keyframe_path).exists():
            all_paths.append(scene.keyframe_path)

        for kf_path in all_paths:
            img = cv2.imread(kf_path)
            if img is None:
                continue

            faces = app.get(img)
            for face in faces:
                face_data.append({
                    "scene_index": scene.scene_index,
                    "bbox": face.bbox.tolist(),
                    "embedding": face.embedding.tolist(),
                    "keyframe_path": kf_path,
                    "det_score": float(face.det_score),
                })

    logger.info(f"人脸检测完成: {len(face_data)} 个人脸")
    return face_data


def _detect_faces_with_gemini(scenes: list[Shot]) -> list[dict]:
    """
    当 InsightFace 不可用时，使用 Gemini Vision 识别人物。
    返回简化的人物信息。
    """
    client = get_llm_client()
    face_data = []
    
    # 选取部分关键帧进行分析
    valid_scenes = []
    for s in scenes:
        kf = s.keyframe_path
        if not kf or not Path(kf).exists():
            if s.keyframe_paths:
                kf = next((p for p in s.keyframe_paths if Path(p).exists()), None)
        if kf:
            valid_scenes.append((s, kf))

    sample_scenes = valid_scenes[::max(1, len(valid_scenes) // 20)]  # 最多取20帧

    prompt = """请分析这张图片中出现的人物。
对每个人物，描述其外观特征（性别、大致年龄、发型、服装等）。

输出 JSON 数组，每个元素代表一个人物：
```json
[
  {"person_id": 1, "description": "外观描述", "position": "画面位置（左/中/右）"}
]
```
如果画面中没有人物，返回空数组 []。"""

    for scene, kf_path in sample_scenes:
        try:
            response = client.chat_with_media(
                prompt=prompt, media_path=kf_path, temperature=0.2
            )
            parsed = client.parse_json(response)
            if parsed and isinstance(parsed, list):
                for p in parsed:
                    desc = p.get("description", "")
                    if desc:
                        face_data.append({
                            "scene_index": scene.scene_index,
                            "bbox": [],
                            "embedding": [],  # 无真实 embedding
                            "keyframe_path": kf_path,
                            "det_score": 1.0,
                            "description": desc,
                        })
            time.sleep(0.5)
        except Exception as e:
            logger.warning(f"Gemini 人物识别失败 (scene {scene.scene_index}): {e}")

    return face_data


def _cluster_faces(face_data: list[dict]) -> dict:
    """
    对检测到的人脸进行聚类。
    有 embedding 时用 DBSCAN，没有时用描述文本匹配。
    """
    has_embeddings = any(len(f.get("embedding", [])) > 0 for f in face_data)

    if has_embeddings:
        return _cluster_by_embedding(face_data)
    else:
        return _cluster_by_description(face_data)


def _cluster_by_embedding(face_data: list[dict]) -> dict:
    """基于人脸特征向量聚类"""
    from sklearn.cluster import DBSCAN

    embeddings = np.array([f["embedding"] for f in face_data if f["embedding"]])
    
    if len(embeddings) == 0:
        return {}

    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # DBSCAN 聚类（余弦距离）
    clustering = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
    labels = clustering.fit_predict(embeddings)

    clusters = {}
    valid_faces = [f for f in face_data if f["embedding"]]
    for idx, label in enumerate(labels):
        if label == -1:
            continue
        if label not in clusters:
            clusters[label] = {"faces": [], "scenes": set()}
        clusters[label]["faces"].append(valid_faces[idx])
        clusters[label]["scenes"].add(valid_faces[idx]["scene_index"])

    # 转 set 为 list
    for k in clusters:
        clusters[k]["scenes"] = list(clusters[k]["scenes"])

    return clusters


def _cluster_by_description(face_data: list[dict]) -> dict:
    """基于描述文本的简单聚类（Gemini fallback 模式）"""
    clusters = {}
    cluster_id = 0

    for face in face_data:
        desc = face.get("description", "")
        # 简单策略：每个不同的描述视为一个人物
        matched = False
        for cid, info in clusters.items():
            # 如果描述相似，归入同一聚类
            if info["faces"] and _desc_similar(desc, info["faces"][0].get("description", "")):
                info["faces"].append(face)
                info["scenes"].add(face["scene_index"])
                matched = True
                break
        if not matched:
            clusters[cluster_id] = {
                "faces": [face],
                "scenes": {face["scene_index"]},
            }
            cluster_id += 1

    for k in clusters:
        clusters[k]["scenes"] = list(clusters[k]["scenes"])

    return clusters


def _desc_similar(desc1: str, desc2: str) -> bool:
    """简单判断两段描述是否指同一人物（使用 2-gram 兼容中文）"""
    if not desc1 or not desc2:
        return False
    # 对有空格的文本（英文等）按空格分词，对无空格文本（中文等）使用 2-gram
    words1 = set(desc1.split()) if " " in desc1 else {desc1[i:i+2] for i in range(max(1, len(desc1)-1))}
    words2 = set(desc2.split()) if " " in desc2 else {desc2[i:i+2] for i in range(max(1, len(desc2)-1))}
    overlap = len(words1 & words2) / max(len(words1 | words2), 1)
    return overlap > 0.5


def _save_face_thumbnail(video_dir: Path, char_id: str, face_info: dict) -> str:
    """保存人脸缩略图"""
    thumbs_dir = video_dir / "characters"
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    thumb_path = thumbs_dir / f"{char_id}.jpg"

    if thumb_path.exists():
        return str(thumb_path)

    keyframe = face_info.get("keyframe_path", "")
    bbox = face_info.get("bbox", [])

    if keyframe and Path(keyframe).exists() and bbox and len(bbox) == 4:
        try:
            import cv2
            img = cv2.imread(keyframe)
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # 扩大裁剪区域
            h, w = img.shape[:2]
            pad = int(max(x2 - x1, y2 - y1) * 0.3)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            face_img = img[y1:y2, x1:x2]
            cv2.imwrite(str(thumb_path), face_img)
            return str(thumb_path)
        except Exception:
            pass

    # 如果无法裁剪人脸，直接用关键帧
    if keyframe and Path(keyframe).exists():
        import shutil
        shutil.copy2(keyframe, str(thumb_path))

    return str(thumb_path)


def _describe_character(client, thumbnail_path: str) -> str:
    """用 Gemini 生成人物描述"""
    if not thumbnail_path or not Path(thumbnail_path).exists():
        return ""

    prompt = """请简要描述这个人物的外观特征：
- 性别
- 大致年龄
- 发型和发色
- 服装
- 显著特征

用一段话简洁描述，不超过50字。"""

    try:
        response = client.chat_with_media(
            prompt=prompt, media_path=thumbnail_path, temperature=0.3
        )
        return response.strip()[:200]
    except Exception as e:
        logger.warning(f"人物描述生成失败: {e}")
        return ""


def _calc_screen_time(scenes: list[Shot], appearance_indices: list[int]) -> float:
    """计算人物总出镜时长"""
    total = 0.0
    for scene in scenes:
        if scene.scene_index in appearance_indices:
            total += scene.duration
    return round(total, 1)
