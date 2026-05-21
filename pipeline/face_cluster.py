# -*- coding: utf-8 -*-
"""
人脸聚类 + 角色脸谱构建（v4.1 新增 — Step 4）

在 Gemini 调用之前，用本地模型完成人脸检测和聚类，
构建"角色脸谱"作为后续 MinuteChunk 处理时的参考输入。

流程：
1. InsightFace 检测所有关键帧中的人脸
2. DBSCAN 聚类 → 按出现频率分层（主要角色 / 次要角色 / 路人）
3. 为每个角色选取 3-6 张代表脸（时间轴均匀采样 + 外观多样性采样）
4. 保存角色脸谱图库

输出：
- characters/face_clusters.json     聚类元数据
- characters/char_XXX_gallery/      每角色 3-6 张代表脸
"""
import json
import os
import subprocess
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

import config
from models.schemas import Shot, CharacterGallery
from utils.logger import get_logger

logger = get_logger("FaceCluster")


def cluster_faces(
    video_id: str,
    shots: list[Shot],
) -> list[CharacterGallery]:
    """
    检测关键帧人脸 → 聚类 → 构建角色脸谱。

    Args:
        video_id: 视频 ID
        shots: 带 keyframe_path(s) 的镜头列表

    Returns:
        CharacterGallery 列表（按出场频率降序排列）
    """
    video_dir = config.VIDEOS_DIR / video_id
    chars_dir = video_dir / "characters"
    clusters_path = chars_dir / "face_clusters.json"

    # 如果已存在，直接加载
    if clusters_path.exists():
        logger.info(f"人脸聚类结果已存在，直接加载: {clusters_path}")
        data = json.loads(clusters_path.read_text(encoding="utf-8"))
        return [CharacterGallery(**g) for g in data]

    chars_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 人脸检测
    face_data = _detect_all_faces(shots)
    if not face_data:
        logger.warning("未检测到任何人脸，返回空脸谱")
        clusters_path.write_text("[]", encoding="utf-8")
        return []

    # Step 2: 聚类
    clusters = _cluster_faces(face_data)
    logger.info(f"聚类完成: {len(clusters)} 个人物簇")

    # Step 3: 按频率分层 + 构建脸谱
    total_duration = max(s.end_time for s in shots) if shots else 0
    galleries = _build_galleries(
        clusters, shots, chars_dir, total_duration,
    )

    # 按出场频率排序
    galleries.sort(key=lambda g: g.total_detections, reverse=True)

    # 保存
    clusters_path.write_text(
        json.dumps([g.model_dump() for g in galleries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(
        f"角色脸谱构建完成: "
        f"{sum(1 for g in galleries if g.tier == 'major')} major, "
        f"{sum(1 for g in galleries if g.tier == 'minor')} minor, "
        f"{sum(1 for g in galleries if g.tier == 'passerby')} passerby"
    )

    return galleries


def _detect_all_faces(shots: list[Shot]) -> list[dict]:
    """
    使用 InsightFace 检测所有关键帧中的人脸。
    返回 [{scene_index, bbox, embedding, keyframe_path, timestamp, det_score}, ...]
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
        import cv2
    except ImportError:
        logger.warning(
            "InsightFace 未安装，跳过人脸聚类。"
            "后续 MinuteChunk 将由 Gemini 自行识别人物。"
        )
        return []

    # 初始化模型：优先 GPU，失败自动回退 CPU
    app = _create_face_app(FaceAnalysis)

    face_data = []
    filtered_faces = 0
    for shot in shots:
        # 收集所有可用帧
        all_paths = []
        if shot.keyframe_paths:
            all_paths.extend(
                [p for p in shot.keyframe_paths if p and Path(p).exists()]
            )
        elif shot.keyframe_path and Path(shot.keyframe_path).exists():
            all_paths.append(shot.keyframe_path)

        for kf_path in all_paths:
            try:
                img = cv2.imread(kf_path)
                if img is None:
                    continue
                faces = app.get(img)
                for face in faces:
                    if not _is_usable_face(face):
                        filtered_faces += 1
                        continue
                    face_data.append({
                        "scene_index": shot.scene_index,
                        "bbox": face.bbox.tolist(),
                        "embedding": face.embedding.tolist(),
                        "keyframe_path": kf_path,
                        "timestamp": shot.start_time,
                        "det_score": float(face.det_score),
                        "face_size": _face_size(face.bbox),
                    })
            except Exception as e:
                logger.warning(f"人脸检测失败 (shot {shot.scene_index}, {kf_path}): {e}")

    logger.info(f"人脸检测完成: {len(face_data)} 个人脸")
    if filtered_faces:
        logger.info(f"Filtered low-quality faces: {filtered_faces}")
    return face_data


def _is_usable_face(face) -> bool:
    if float(getattr(face, "det_score", 0.0)) < config.FACE_MIN_DET_SCORE:
        return False
    return _face_size(getattr(face, "bbox", [])) >= config.FACE_MIN_FACE_SIZE


def _face_size(bbox) -> float:
    if bbox is None or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, min(x2 - x1, y2 - y1))


def _create_face_app(face_analysis_cls):
    """创建 InsightFace 应用，按配置优先使用 CUDA。"""
    providers, ctx_id = _select_face_providers()
    provider_name = _provider_name(providers[0])
    try:
        app = face_analysis_cls(name="buffalo_l", providers=providers)
        app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info(f"InsightFace 后端: {provider_name}, device={ctx_id}")
        return app
    except Exception as e:
        if provider_name == "CUDAExecutionProvider":
            logger.warning(f"InsightFace GPU 初始化失败，回退 CPU: {e}")
            app = face_analysis_cls(name="buffalo_l", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=(640, 640))
            return app
        raise


def _select_face_providers() -> tuple[list, int]:
    """选择 onnxruntime provider；ctx_id=0 为 GPU，-1 为 CPU。"""
    device = config.FACE_DETECT_DEVICE
    if device == "cpu":
        return ["CPUExecutionProvider"], -1

    try:
        import onnxruntime as ort
        available = set(ort.get_available_providers())
    except Exception:
        available = set()

    if "CUDAExecutionProvider" in available:
        gpu_id = _select_cuda_device_id()
        cuda_provider = ("CUDAExecutionProvider", {"device_id": gpu_id})
        return [cuda_provider, "CPUExecutionProvider"], gpu_id

    if device in ("cuda", "gpu"):
        logger.warning("未检测到 CUDAExecutionProvider，请安装匹配 CUDA 的 onnxruntime-gpu")
    return ["CPUExecutionProvider"], -1


def _provider_name(provider) -> str:
    return provider[0] if isinstance(provider, tuple) else provider


def _select_cuda_device_id() -> int:
    raw = config.FACE_DETECT_GPU_ID
    if raw and raw != "auto":
        try:
            return max(0, int(raw))
        except ValueError:
            logger.warning(f"FACE_DETECT_GPU_ID 无效: {raw}，使用 0")
            return 0
    return _least_used_cuda_device()


def _least_used_cuda_device() -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return 0

    visible = _visible_cuda_devices()
    candidates = []
    for line in result.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            physical_id, memory_used = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if visible is None:
            candidates.append((physical_id, memory_used))
        elif physical_id in visible:
            candidates.append((visible.index(physical_id), memory_used))

    return min(candidates, key=lambda item: item[1])[0] if candidates else 0


def _visible_cuda_devices() -> list[int] | None:
    value = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    if not value:
        return None
    devices = []
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            devices.append(int(part))
    return devices or None


def _cluster_faces(face_data: list[dict]) -> dict:
    """
    基于人脸特征向量 DBSCAN 聚类。
    返回 {cluster_id: {"faces": [...], "scenes": set(...)}}
    """
    from sklearn.cluster import DBSCAN

    embeddings = np.array([f["embedding"] for f in face_data if f["embedding"]])
    if len(embeddings) == 0:
        return {}

    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms

    # DBSCAN 聚类（余弦距离）
    clustering = DBSCAN(
        eps=config.FACE_CLUSTER_EPS,
        min_samples=max(1, config.FACE_CLUSTER_MIN_SAMPLES),
        metric="cosine",
    )
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

    return _merge_close_clusters(clusters)


def _merge_close_clusters(clusters: dict) -> dict:
    threshold = config.FACE_CLUSTER_MERGE_SIM
    if threshold <= 0 or len(clusters) < 2:
        return clusters

    labels = list(clusters)
    centroids = {
        label: _normalized_centroid(info["faces"])
        for label, info in clusters.items()
    }
    component_keyframes = {
        label: _cluster_keyframes(info["faces"])
        for label, info in clusters.items()
    }
    parent = {label: label for label in labels}

    def find(label):
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    def union(left, right) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        if component_keyframes[left_root] & component_keyframes[right_root]:
            return False
        parent[right_root] = left_root
        component_keyframes[left_root].update(component_keyframes[right_root])
        return True

    merges = 0
    for i, left in enumerate(labels):
        left_centroid = centroids.get(left)
        if left_centroid is None:
            continue
        for right in labels[i + 1:]:
            right_centroid = centroids.get(right)
            if right_centroid is None:
                continue
            if float(np.dot(left_centroid, right_centroid)) >= threshold:
                merges += int(union(left, right))

    if not merges:
        return clusters

    merged = {}
    for label in labels:
        root = find(label)
        item = merged.setdefault(root, {"faces": [], "scenes": set()})
        item["faces"].extend(clusters[label]["faces"])
        item["scenes"].update(clusters[label]["scenes"])

    logger.info(f"Merged close face clusters: {len(clusters)} -> {len(merged)}")
    return merged


def _normalized_centroid(faces: list[dict]):
    embeddings = np.array([f["embedding"] for f in faces if f.get("embedding")])
    if len(embeddings) == 0:
        return None
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm else None


def _cluster_keyframes(faces: list[dict]) -> set[str]:
    return {f["keyframe_path"] for f in faces if f.get("keyframe_path")}


def _build_galleries(
    clusters: dict,
    shots: list[Shot],
    chars_dir: Path,
    total_duration: float,
) -> list[CharacterGallery]:
    """
    为每个聚类构建角色脸谱，按频率分层。

    分层规则（根据视频时长自动调整）：
    - < 10min: passerby_threshold=2
    - 10-30min: passerby_threshold=3
    - > 30min: passerby_threshold=5
    """
    import cv2

    # 动态阈值
    if total_duration < 600:
        passerby_threshold = 2
    elif total_duration < 1800:
        passerby_threshold = config.FACE_PASSERBY_MIN_APPEARANCES
    else:
        passerby_threshold = max(5, config.FACE_PASSERBY_MIN_APPEARANCES)

    total_shots = len(shots)
    major_threshold = max(10, int(total_shots * 0.05))

    galleries = []
    skipped_passerby = 0

    for cluster_id, cluster_info in clusters.items():
        faces = cluster_info["faces"]
        scenes = cluster_info["scenes"]
        n_detections = len(faces)
        n_scenes = len(scenes)

        # 分层
        if n_scenes >= major_threshold:
            tier = "major"
        elif n_scenes >= passerby_threshold:
            tier = "minor"
        else:
            tier = "passerby"

        if tier == "passerby" and not config.FACE_KEEP_PASSERBY_GALLERY:
            skipped_passerby += 1
            continue

        char_id = f"char_{cluster_id:03d}"

        # 选取代表脸
        if tier == "passerby":
            # 路人只保存 1 张（最高置信度的）
            gallery_faces = sorted(faces, key=lambda f: -f["det_score"])[:1]
        else:
            gallery_faces = _select_representative_faces(
                faces, shots,
                min_count=config.FACE_GALLERY_MIN,
                max_count=config.FACE_GALLERY_MAX,
            )

        # 保存脸谱图片
        gallery_dir = chars_dir / f"{char_id}_gallery"
        gallery_dir.mkdir(parents=True, exist_ok=True)

        gallery_paths = []
        gallery_timestamps = []

        for i, face in enumerate(gallery_faces):
            face_path = gallery_dir / f"face_{i:02d}.jpg"
            if not face_path.exists():
                _save_face_crop(face, str(face_path))
            gallery_paths.append(str(face_path))
            gallery_timestamps.append(face["timestamp"])

        # 计算聚类中心
        embeddings = np.array([f["embedding"] for f in faces])
        centroid = embeddings.mean(axis=0).tolist() if len(embeddings) > 0 else []

        gallery = CharacterGallery(
            character_id=char_id,
            gallery_paths=gallery_paths,
            gallery_timestamps=gallery_timestamps,
            total_detections=n_detections,
            appearance_scenes=sorted(scenes),
            tier=tier,
            embedding_centroid=centroid,
        )
        galleries.append(gallery)

    if skipped_passerby:
        logger.info(f"Skipped passerby clusters: {skipped_passerby}")
    return galleries


def _select_representative_faces(
    faces: list[dict],
    shots: list[Shot],
    min_count: int = 3,
    max_count: int = 6,
) -> list[dict]:
    """
    为一个角色选取代表脸，凸显形象变化。

    策略：
    1. 时间轴均匀采样 — 捕捉服装/发型随剧情变化
    2. 外观多样性采样 — 在 embedding 空间选取最远的脸
    3. 合并去重 → 保留 min_count ~ max_count 张
    """
    if len(faces) <= max_count:
        return sorted(faces, key=lambda f: f["timestamp"])

    # 策略1: 时间轴均匀采样
    sorted_by_time = sorted(faces, key=lambda f: f["timestamp"])
    time_min = sorted_by_time[0]["timestamp"]
    time_max = sorted_by_time[-1]["timestamp"]
    time_range = time_max - time_min

    n_time_samples = min(max_count, max(min_count, len(faces) // 3))
    time_selected = []

    if time_range > 0:
        for i in range(n_time_samples):
            target_time = time_min + (time_range * i / max(n_time_samples - 1, 1))
            # 找最近的且置信度最高的
            candidates = sorted(
                sorted_by_time,
                key=lambda f: (abs(f["timestamp"] - target_time), -f["det_score"]),
            )
            if candidates and candidates[0] not in time_selected:
                time_selected.append(candidates[0])
    else:
        # 所有脸在同一时刻，按置信度取
        time_selected = sorted(faces, key=lambda f: -f["det_score"])[:n_time_samples]

    # 策略2: 外观多样性采样（在embedding空间中找最远的）
    embeddings = np.array([f["embedding"] for f in faces])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_normed = embeddings / norms

    # 从置信度最高的开始，贪心选取最远的
    diversity_selected = []
    remaining_indices = list(range(len(faces)))
    # 先加入置信度最高的
    best_idx = max(remaining_indices, key=lambda i: faces[i]["det_score"])
    diversity_selected.append(best_idx)
    remaining_indices.remove(best_idx)

    while len(diversity_selected) < min_count and remaining_indices:
        # 找与已选中的所有脸距离最大的
        selected_embs = embeddings_normed[diversity_selected]
        max_min_dist = -1
        best_remaining = None
        for idx in remaining_indices:
            dists = 1 - np.dot(selected_embs, embeddings_normed[idx])
            min_dist = dists.min()
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_remaining = idx
        if best_remaining is not None:
            diversity_selected.append(best_remaining)
            remaining_indices.remove(best_remaining)

    diversity_faces = [faces[i] for i in diversity_selected]

    # 合并两组，去重
    all_selected = list(time_selected)
    for f in diversity_faces:
        if f not in all_selected:
            all_selected.append(f)

    # 截断到 max_count
    all_selected = all_selected[:max_count]

    # 按时间排序
    all_selected.sort(key=lambda f: f["timestamp"])

    return all_selected


def _save_face_crop(face_info: dict, output_path: str):
    """保存人脸裁剪图（扩大区域包含肩部）"""
    try:
        import cv2
        keyframe = face_info.get("keyframe_path", "")
        bbox = face_info.get("bbox", [])

        if not keyframe or not Path(keyframe).exists():
            return

        img = cv2.imread(keyframe)
        if img is None:
            return

        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = img.shape[:2]
            # 扩大裁剪区域（包含更多上下文）
            pad_x = int((x2 - x1) * 0.4)
            pad_y = int((y2 - y1) * 0.5)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            face_img = img[y1:y2, x1:x2]
            cv2.imwrite(output_path, face_img)
        else:
            # 无 bbox，保存整帧
            cv2.imwrite(output_path, img)
    except Exception as e:
        logger.warning(f"保存人脸裁剪失败: {e}")
