# -*- coding: utf-8 -*-
"""
人脸聚类 + 角色脸谱构建（v4.1 — Step 4）。

这个步骤位于多关键帧采样之后、MinuteChunk 理解之前。它的目标不是直接
给角色命名，而是先在本地建立稳定的 character_id 与代表脸图库，让后续
Gemini 处理每个 chunk 时能拿到“已知角色长什么样”的身份先验。

主流程：
1. 读取缓存：如果 `characters/face_clusters.json` 已存在，直接复用。
   这样断点续跑不会重复跑 InsightFace，也避免已有角色 ID 反复变化。
2. 人脸检测：遍历每个 shot 的关键帧，用 InsightFace 提取 bbox、
   det_score 和 embedding。embedding 是后续判断“是否同一个人”的核心特征。
3. 质量过滤：过滤低置信度、画面占比过小的人脸，减少远景误检和模糊脸
   对聚类的污染。尺寸阈值按关键帧短边比例计算，而不是固定像素。
4. 初始聚类：用 DBSCAN + cosine distance 将相似 embedding 聚为人物簇。
   DBSCAN 不需要预先知道角色数量，适合长视频中角色数未知的场景。
5. 拆分混簇：如果同一关键帧内一个簇出现多张脸，或簇半径过大，说明
   可能把不同人物混在一起，使用更严格的阈值做二次聚类。
6. 合并碎簇：如果同一人物因为发型、服装、光照或角度变化被拆成多个簇，
   再用簇中心相似度和代表脸桥接相似度做保守合并。
7. 构建脸谱：按出现 shot 数分为 major/minor/passerby，并为非路人角色
   保存时间均匀且外观多样的代表脸。

输出：
- `characters/face_clusters.json`：CharacterGallery 元数据。
- `characters/char_XXX_gallery/`：每个保留角色的代表脸裁剪图。
"""
import json
import os
import subprocess
from pathlib import Path
from collections import defaultdict

import numpy as np

import config
from models.schemas import Shot, CharacterGallery
from utils.logger import get_logger

logger = get_logger("FaceCluster")



# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------

def cluster_faces(
    video_id: str,
    shots: list[Shot],
) -> list[CharacterGallery]:
    """
    构建当前视频的角色脸谱。

    这是一条“本地视觉预处理”流水线，产物会作为 Step 5 MinuteChunk 的
    输入之一。后续 Gemini 不需要从零判断所有人物是否相同，而是参考这里
    生成的 char_XXX gallery 做身份识别和角色档案更新。

    Args:
        video_id: 视频 ID，用于定位 `data/videos/{video_id}/characters/`。
        shots: 已完成镜头切分和关键帧采样的镜头列表。每个 shot 至少应带
            `keyframe_path` 或 `keyframe_paths`，否则无法参与人脸检测。

    Returns:
        CharacterGallery 列表，按检测次数降序排列。列表中的 character_id
        只是稳定的内部编号，真实姓名会在后续 MinuteChunk 中逐步补全。
    """
    video_dir = config.VIDEOS_DIR / video_id
    chars_dir = video_dir / "characters"
    clusters_path = chars_dir / "face_clusters.json"

    # 1. 缓存优先：face clustering 成本高，且 character_id 会影响后续产物。
    #    断点续跑时复用已有 JSON，可以保持角色编号稳定。
    cached = _load_cached_galleries(clusters_path)
    if cached is not None:
        return cached

    chars_dir.mkdir(parents=True, exist_ok=True)

    # 2. 本地检测关键帧中的所有可用人脸，得到 bbox + embedding。
    #    如果 InsightFace 未安装或所有关键帧都没有可用脸，写入空缓存，
    #    后续 MinuteChunk 会退化为让 Gemini 自行识别人物。
    face_data = _detect_all_faces(shots)
    if not face_data:
        logger.warning("未检测到任何人脸，返回空脸谱")
        _save_galleries(clusters_path, [])
        return []

    # 3. 聚类阶段只处理 embedding，不涉及角色命名：
    #    初始 DBSCAN 负责发现人物簇，拆分/合并负责修正混簇和碎簇。
    clusters = _cluster_faces(face_data)
    logger.info(f"聚类完成: {len(clusters)} 个人物簇")

    # 4. 将人物簇转成对下游友好的 CharacterGallery：
    #    分层决定是否保留路人，代表脸图库用于 Gemini 身份参考。
    galleries = _build_galleries(clusters, shots, chars_dir)
    galleries.sort(key=lambda g: g.total_detections, reverse=True)

    # 5. 持久化最终产物。后续 understand 断点续跑会从这里加载 galleries。
    _save_galleries(clusters_path, galleries)
    _log_gallery_summary(galleries)

    return galleries


# ---------------------------------------------------------------------------
# Cache and config helpers
# ---------------------------------------------------------------------------

def _load_cached_galleries(clusters_path: Path) -> list[CharacterGallery] | None:
    """加载已构建的脸谱缓存；不存在时返回 None，让主流程继续重建。"""
    if not clusters_path.exists():
        return None
    logger.info(f"人脸聚类结果已存在，直接加载: {clusters_path}")
    data = json.loads(clusters_path.read_text(encoding="utf-8"))
    return [CharacterGallery(**g) for g in data]


def _save_galleries(clusters_path: Path, galleries: list[CharacterGallery]) -> None:
    """保存 CharacterGallery 元数据；图片文件由 `_save_gallery_faces` 单独保存。"""
    clusters_path.write_text(
        json.dumps([g.model_dump() for g in galleries], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _log_gallery_summary(galleries: list[CharacterGallery]) -> None:
    logger.info(
        f"角色脸谱构建完成: "
        f"{sum(1 for g in galleries if g.tier == 'major')} major, "
        f"{sum(1 for g in galleries if g.tier == 'minor')} minor, "
        f"{sum(1 for g in galleries if g.tier == 'passerby')} passerby"
    )




# ---------------------------------------------------------------------------
# Face detection and filtering
# ---------------------------------------------------------------------------

def _detect_all_faces(shots: list[Shot]) -> list[dict]:
    """
    使用 InsightFace 检测所有关键帧中的人脸。

    输入是 Step 3 产生的 shot 列表。函数不会直接读视频，而是读取每个 shot
    中的关键帧图片。每一张检测到且通过质量过滤的人脸都会被转成普通 dict，
    这样后续 DBSCAN、JSON 缓存和裁剪保存都不依赖 InsightFace 的对象类型。

    返回元素格式：
    {
        scene_index: 所属 shot 索引，用于统计角色出现在哪些镜头；
        bbox: 人脸框 [x1, y1, x2, y2]，用于裁剪 gallery 图片；
        embedding: InsightFace 人脸向量，用于聚类；
        keyframe_path: 来源关键帧路径，用于冲突检测和裁剪；
        timestamp: 使用 shot.start_time，代表该脸在视频时间轴上的位置；
        det_score: 检测置信度，用于过滤和优先选择高质量代表脸。
    }
    """
    try:
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
        for kf_path, kf_timestamp in _shot_keyframe_items(shot):
            try:
                img = cv2.imread(kf_path)
                if img is None:
                    continue
                faces = app.get(img)
                for face in faces:
                    if not _is_usable_face(face, img.shape):
                        filtered_faces += 1
                        continue
                    face_data.append(_face_record(face, shot, kf_path, kf_timestamp))
            except Exception as e:
                logger.warning(f"人脸检测失败 (shot {shot.scene_index}, {kf_path}): {e}")

    logger.info(f"人脸检测完成: {len(face_data)} 个人脸")
    if filtered_faces:
        logger.info(f"Filtered low-quality faces: {filtered_faces}")
    return face_data


def _shot_keyframe_paths(shot: Shot) -> list[str]:
    """
    收集一个 shot 的所有可用关键帧路径。

    `keyframe_paths` 是新版多帧采样字段，`keyframe_path` 是旧版兼容字段。
    这里同时读取并去重，避免旧数据或中间产物不完整时漏掉可用帧。
    """
    return [path for path, _ in _shot_keyframe_items(shot)]


def _shot_keyframe_items(shot: Shot) -> list[tuple[str, float]]:
    """Return existing keyframe paths with an estimated timestamp for each frame."""
    candidates = list(shot.keyframe_paths or [])
    if shot.keyframe_path:
        candidates.append(shot.keyframe_path)
    paths = []
    seen = set()
    for path in candidates:
        if not path:
            continue
        path_str = str(path)
        if path_str in seen:
            continue
        if Path(path_str).exists():
            paths.append(path_str)
            seen.add(path_str)
    total = len(paths)
    return [
        (path, _infer_keyframe_timestamp(shot, idx, total))
        for idx, path in enumerate(paths)
    ]


def _infer_keyframe_timestamp(shot: Shot, frame_index: int, frame_count: int) -> float:
    """Mirror keyframe.py sampling so multi-frame detections keep temporal order."""
    start = float(shot.start_time)
    end = float(shot.end_time)
    duration = max(0.0, end - start)
    if frame_count <= 1:
        return start + duration * 0.5

    safe_start = start + duration * 0.1
    safe_end = end - duration * 0.1
    safe_duration = safe_end - safe_start
    if safe_duration <= 0:
        return start + duration * 0.5

    step = safe_duration / max(frame_count - 1, 1)
    return round(safe_start + step * frame_index, 3)


def _face_record(face, shot: Shot, keyframe_path: str, timestamp: float | None = None) -> dict:
    """把 InsightFace 返回对象压平成后续流程稳定使用的普通 dict。"""
    return {
        "scene_index": shot.scene_index,
        "bbox": _to_list(getattr(face, "bbox", [])),
        "embedding": _to_list(getattr(face, "embedding", [])),
        "keyframe_path": keyframe_path,
        "timestamp": float(shot.start_time if timestamp is None else timestamp),
        "det_score": float(getattr(face, "det_score", 0.0)),
    }


def _to_list(value) -> list:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def _is_usable_face(face, image_shape=None) -> bool:
    """
    判断一张检测脸是否值得进入聚类。

    过滤规则有两层：
    1. det_score 过滤误检；
    2. bbox 短边过滤远景小脸。尺寸阈值按关键帧短边比例计算，并带像素兜底，
       这样同一套配置能适配 480p/1080p/4K 等不同关键帧尺寸。
    """
    if float(getattr(face, "det_score", 0.0)) < config.FACE_MIN_DET_SCORE:
        return False
    return _face_size(getattr(face, "bbox", [])) >= _face_size_threshold(image_shape)


def _face_size(bbox) -> float:
    if bbox is None or len(bbox) != 4:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return max(0.0, min(x2 - x1, y2 - y1))


def _face_size_threshold(image_shape) -> float:
    """计算人脸 bbox 短边阈值：max(画面短边比例阈值, 像素兜底阈值)。"""
    # config.py 已处理 FACE_MIN_FACE_SIZE → FACE_MIN_FACE_PIXEL_FLOOR 的兼容，
    # 此处无需再做二级 fallback。
    ratio_threshold = _image_short_side(image_shape) * config.FACE_MIN_FACE_RATIO
    return max(config.FACE_MIN_FACE_PIXEL_FLOOR, ratio_threshold)


def _image_short_side(image_shape) -> float:
    if image_shape is None or len(image_shape) < 2:
        return 0.0
    height, width = image_shape[:2]
    return float(max(0, min(int(width), int(height))))


# ---------------------------------------------------------------------------
# InsightFace runtime selection
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Embedding clustering, split, and merge
# ---------------------------------------------------------------------------

def _cluster_faces(face_data: list[dict]) -> dict:
    """
    基于人脸特征向量 DBSCAN 聚类。

    聚类分三层：
    1. 初始 DBSCAN：尽量把同一人的脸聚到一起；
    2. 拆分疑似混簇：优先避免“两个不同角色被合成一个 char”；
    3. 合并相近碎簇：缓解同一角色因为造型/角度变化被拆成多个 gallery。

    返回 {cluster_id: {"faces": [...], "scenes": set(...)}}。
    scenes 用于后续 major/minor/passerby 分层。
    """
    valid_faces = [f for f in face_data if f.get("embedding")]
    labels = _dbscan_labels(
        valid_faces,
        eps=config.FACE_CLUSTER_EPS,
        min_samples=max(1, config.FACE_CLUSTER_MIN_SAMPLES),
    )
    if labels is None:
        return {}

    clusters = {}
    for face, label in zip(valid_faces, labels):
        if label == -1:
            continue
        item = clusters.setdefault(int(label), {"faces": [], "scenes": set()})
        item["faces"].append(face)
        item["scenes"].add(face["scene_index"])

    # 先拆混簇，再保守合并，优先避免不同角色混成一个 char。
    clusters = _split_impure_clusters(clusters)
    return _merge_close_clusters(clusters)


def _dbscan_labels(faces: list[dict], eps: float, min_samples: int):
    """统一的余弦 DBSCAN 入口，避免多处重复归一化。"""
    from sklearn.cluster import DBSCAN

    embeddings = _normalized_embeddings(faces)
    if embeddings is None:
        return None
    return DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="cosine",
    ).fit_predict(embeddings)


def _split_impure_clusters(clusters: dict) -> dict:
    """
    拆分疑似混簇。

    如果同一关键帧里同一个簇出现多张脸，几乎可以判定这个簇混入了不同人。
    如果簇内半径过大，也说明成员在 embedding 空间中过于分散。两种情况都会
    触发更严格的二次 DBSCAN。
    """
    if not clusters:
        return clusters

    next_label = max(clusters) + 1
    result = {}
    split_count = 0

    for label, info in clusters.items():
        parts = _split_cluster_faces(info["faces"])
        if len(parts) == 1:
            result[label] = info
            continue

        split_count += len(parts) - 1
        for faces in parts:
            result[next_label] = {
                "faces": faces,
                "scenes": {f["scene_index"] for f in faces},
            }
            next_label += 1

    if split_count:
        logger.info(f"Split impure face clusters: {len(clusters)} -> {len(result)}")
    return result


def _split_cluster_faces(faces: list[dict]) -> list[list[dict]]:
    """对单个疑似混簇做更严格的二次聚类，并保留噪声点为独立小簇。"""
    min_samples = max(1, config.FACE_CLUSTER_MIN_SAMPLES)
    if len(faces) < min_samples * 2 or not _needs_cluster_split(faces):
        return [faces]

    labels = _dbscan_labels(
        faces,
        eps=min(config.FACE_CLUSTER_SPLIT_EPS, config.FACE_CLUSTER_EPS),
        min_samples=min_samples,
    )
    if labels is None:
        return [faces]

    grouped = defaultdict(list)
    noise = []
    for idx, label in enumerate(labels):
        if label == -1:
            noise.append(faces[idx])
        else:
            grouped[label].append(faces[idx])

    parts = [part for part in grouped.values() if part]
    if len(parts) < 2:
        return [faces]

    parts.extend([[face] for face in noise])
    return parts


def _needs_cluster_split(faces: list[dict]) -> bool:
    """判断一个簇是否需要拆分：同帧冲突或簇内 90 分位半径过大。"""
    has_conflict = _has_keyframe_conflict(faces)
    is_scattered = _cluster_radius(faces) > config.FACE_CLUSTER_MAX_RADIUS
    return has_conflict or is_scattered


def _has_keyframe_conflict(faces: list[dict]) -> bool:
    """同一关键帧内同簇多脸表示物理冲突，通常意味着不同人物被混在一起。"""
    counts = defaultdict(int)
    for face in faces:
        keyframe = face.get("keyframe_path")
        if keyframe:
            counts[keyframe] += 1
            if counts[keyframe] > 1:
                return True
    return False


def _cluster_radius(faces: list[dict]) -> float:
    """计算簇内 90 分位余弦距离，避免少量离群脸决定整个簇是否拆分。"""
    embeddings = _normalized_embeddings(faces)
    if embeddings is None or len(embeddings) < 2:
        return 0.0
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return 0.0
    distances = 1 - np.dot(embeddings, centroid / norm)
    return float(np.percentile(distances, 90))


def _merge_close_clusters(clusters: dict) -> dict:
    """
    合并疑似同一人的碎片簇。

    DBSCAN 对阈值敏感：同一角色在长视频里可能因为换发型、换装、侧脸、
    强光/暗光被拆成多个小簇。这里使用两类证据进行保守合并：
    - centroid_sim：两个簇中心向量足够相似，直接合并；
    - link_sim：两个簇的代表脸之间存在高相似连接，且簇中心不太远，合并。

    同一关键帧中同时出现过的两个簇不会合并，因为它们大概率是两个人。
    注意：不能用 scene（shot）共现来阻断合并，因为同一人在同一 shot 的
    不同关键帧中完全可能被检测到，这会导致碎簇永远无法合并。
    """
    centroid_threshold = config.FACE_CLUSTER_MERGE_SIM
    link_threshold = config.FACE_CLUSTER_MERGE_LINK_SIM
    min_centroid_sim = config.FACE_CLUSTER_MERGE_MIN_CENTROID_SIM
    if centroid_threshold <= 0 and link_threshold <= 0:
        return clusters
    if len(clusters) < 2:
        return clusters

    labels = list(clusters)
    centroids = {
        label: _normalized_centroid(info["faces"])
        for label, info in clusters.items()
    }
    merge_embeddings = {
        label: _cluster_merge_embeddings(info["faces"])
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
        # 同一关键帧中同时出现说明是物理上不同的人，禁止合并。
        if component_keyframes[left_root] & component_keyframes[right_root]:
            return False
        parent[right_root] = left_root
        component_keyframes[left_root].update(component_keyframes[right_root])
        return True

    merges = 0
    for i, left in enumerate(labels):
        left_centroid = centroids.get(left)
        left_embeddings = merge_embeddings.get(left)
        for right in labels[i + 1:]:
            right_centroid = centroids.get(right)
            right_embeddings = merge_embeddings.get(right)
            centroid_sim = _vector_similarity(left_centroid, right_centroid)
            link_sim = _top_pair_similarity(left_embeddings, right_embeddings)
            if _should_merge_clusters(
                centroid_sim,
                link_sim,
                centroid_threshold,
                link_threshold,
                min_centroid_sim,
            ):
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


def _should_merge_clusters(
    centroid_sim: float,
    link_sim: float,
    centroid_threshold: float,
    link_threshold: float,
    min_centroid_sim: float,
) -> bool:
    """集中管理合并判定，便于调参时保持 centroid/link 两条规则一致。"""
    if centroid_sim >= centroid_threshold:
        return True
    if link_threshold <= 0:
        return False
    return link_sim >= link_threshold and centroid_sim >= min_centroid_sim


def _vector_similarity(left, right) -> float:
    if left is None or right is None:
        return -1.0
    return float(np.dot(left, right))


def _top_pair_similarity(left_embeddings, right_embeddings, top_k: int = 3) -> float:
    """
    计算两个簇代表脸之间的桥接相似度。

    取 top-k 平均而不是单个最大值，可以减少偶然误匹配造成的错误合并。
    """
    if left_embeddings is None or right_embeddings is None:
        return -1.0
    sims = np.dot(left_embeddings, right_embeddings.T).reshape(-1)
    if sims.size == 0:
        return -1.0
    k = min(top_k, sims.size)
    return float(np.partition(sims, -k)[-k:].mean())


def _cluster_merge_embeddings(faces: list[dict]):
    """抽取用于合并比较的代表脸 embedding，避免大簇两两全量比较过慢。"""
    max_faces = config.FACE_CLUSTER_MERGE_MAX_FACES
    selected = _select_merge_faces(faces, max_faces)
    return _normalized_embeddings(selected)


def _select_merge_faces(faces: list[dict], max_faces: int) -> list[dict]:
    """为合并阶段选样本：高置信度脸 + 时间均匀脸，覆盖质量和时间跨度。"""
    if max_faces <= 0 or len(faces) <= max_faces:
        return faces

    score_quota = max(1, max_faces // 2)
    by_score = sorted(faces, key=lambda f: -float(f.get("det_score", 0.0)))
    high_confidence = by_score[:score_quota]
    time_samples = _time_sample_faces(faces, max_faces - len(high_confidence))
    # by_score 尾部用于补足，_dedupe_faces 会自动去除重复。
    return _dedupe_faces(high_confidence + time_samples + by_score)[:max_faces]


def _normalized_centroid(faces: list[dict]):
    """计算归一化簇中心，用于簇间余弦相似度比较和最终 JSON 输出。"""
    embeddings = _normalized_embeddings(faces)
    if embeddings is None:
        return None
    centroid = embeddings.mean(axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm else None


def _normalized_embeddings(faces: list[dict]):
    """将人脸 embedding 做 L2 归一化，使点积等价于 cosine similarity。"""
    embeddings = np.array([f["embedding"] for f in faces if f.get("embedding")])
    if len(embeddings) == 0:
        return None
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return embeddings / norms


def _cluster_keyframes(faces: list[dict]) -> set[str]:
    return {f["keyframe_path"] for f in faces if f.get("keyframe_path")}


# ---------------------------------------------------------------------------
# Character gallery construction
# ---------------------------------------------------------------------------

def _build_galleries(
    clusters: dict,
    shots: list[Shot],
    chars_dir: Path,
) -> list[CharacterGallery]:
    """
    为每个聚类构建 CharacterGallery。

    这里把纯 embedding 簇转换成下游可消费的角色脸谱：
    - tier：根据出现场景数划分 major/minor/passerby；
    - gallery_paths：代表脸裁剪图，用于 MinuteChunk 角色识别；
    - embedding_centroid：后续如需本地相似度检索，可以继续复用。
    """
    major_threshold, passerby_threshold = _gallery_thresholds(shots)
    galleries = []
    skipped_passerby = 0

    for cluster_id, cluster_info in clusters.items():
        faces = cluster_info["faces"]
        scenes = cluster_info["scenes"]
        n_detections = len(faces)
        n_scenes = len(scenes)
        tier = _cluster_tier(n_scenes, major_threshold, passerby_threshold)

        if tier == "passerby" and not config.FACE_KEEP_PASSERBY_GALLERY:
            skipped_passerby += 1
            continue

        char_id = f"char_{cluster_id:03d}"
        gallery_faces = _select_gallery_faces(faces, tier)
        (
            gallery_paths,
            gallery_timestamps,
            gallery_scene_indices,
            gallery_keyframe_paths,
        ) = _save_gallery_faces(
            gallery_faces,
            chars_dir / f"{char_id}_gallery",
        )
        centroid = _normalized_centroid(faces)

        galleries.append(CharacterGallery(
            character_id=char_id,
            gallery_paths=gallery_paths,
            gallery_timestamps=gallery_timestamps,
            gallery_scene_indices=gallery_scene_indices,
            gallery_keyframe_paths=gallery_keyframe_paths,
            total_detections=n_detections,
            appearance_scenes=sorted(scenes),
            tier=tier,
            embedding_centroid=centroid.tolist() if centroid is not None else [],
        ))

    if skipped_passerby:
        logger.info(f"Skipped passerby clusters: {skipped_passerby}")
    return galleries


def _gallery_thresholds(shots: list[Shot]) -> tuple[int, int]:
    """
    根据视频长度和 shot 数动态计算角色分层阈值。

    短视频里出现 2 次的人可能已经有意义；长视频里只出现几次更可能是路人。
    major 采用 `max(10, 总 shot 数 5%)`，避免短视频误把过多人升为主要角色。
    """
    total_duration = max((s.end_time for s in shots), default=0)
    major_threshold = max(10, int(len(shots) * 0.05))

    if total_duration < 600:
        passerby_threshold = 2
    elif total_duration < 1800:
        passerby_threshold = config.FACE_PASSERBY_MIN_APPEARANCES
    else:
        passerby_threshold = max(5, config.FACE_PASSERBY_MIN_APPEARANCES)
    return major_threshold, passerby_threshold


def _cluster_tier(
    scene_count: int,
    major_threshold: int,
    passerby_threshold: int,
) -> str:
    """根据出现的不同 shot 数给角色分层。"""
    if scene_count >= major_threshold:
        return "major"
    if scene_count >= passerby_threshold:
        return "minor"
    return "passerby"


def _select_gallery_faces(faces: list[dict], tier: str) -> list[dict]:
    """路人只保留最高置信度脸；主要/次要角色保留多张代表脸。"""
    if tier == "passerby":
        return sorted(faces, key=lambda f: -f["det_score"])[:1]
    return _select_representative_faces(
        faces,
        min_count=config.FACE_GALLERY_MIN,
        max_count=config.FACE_GALLERY_MAX,
    )


def _save_gallery_faces(
    faces: list[dict],
    gallery_dir: Path,
) -> tuple[list[str], list[float], list[int], list[str]]:
    """保存一个角色的 gallery 图片，并返回图片路径和来源元数据。"""
    gallery_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    timestamps = []
    scene_indices = []
    keyframe_paths = []
    for i, face in enumerate(faces):
        face_path = gallery_dir / f"face_{i:02d}.jpg"
        if not face_path.exists():
            _save_face_crop(face, str(face_path))
        paths.append(str(face_path))
        timestamps.append(_face_timestamp(face))
        scene_indices.append(_face_scene_index(face))
        keyframe_paths.append(str(face.get("keyframe_path", "")))
    return paths, timestamps, scene_indices, keyframe_paths


def _select_representative_faces(
    faces: list[dict],
    min_count: int = 3,
    max_count: int = 6,
) -> list[dict]:
    """
    为一个角色选取代表脸，凸显形象变化。

    策略：
    1. 先按 shot 去重，避免 gallery 被同一个近景 shot 占满。
    2. 再沿角色出现时间轴均匀采样，覆盖不同剧情阶段。
    3. 数量不足时优先补充未使用 shot、且离已选样本时间更远的脸。
    """
    if max_count <= 0:
        return []

    faces = _dedupe_faces(faces)
    if len(faces) <= max_count:
        return sorted(faces, key=_face_sort_key)

    target_count = min(max_count, len(faces))
    scene_faces = _best_faces_by_scene(faces)
    primary_pool = (
        scene_faces
        if len(scene_faces) >= min(min_count, target_count)
        else faces
    )

    selected = _time_sample_faces(primary_pool, min(target_count, len(primary_pool)))
    selected = _fill_gallery_faces(selected, faces, target_count)
    selected = selected[:target_count]
    selected.sort(key=_face_sort_key)
    return selected


def _best_faces_by_scene(faces: list[dict]) -> list[dict]:
    """Keep the best face per shot so a gallery does not collapse into one shot."""
    grouped = defaultdict(list)
    for face in faces:
        grouped[face.get("scene_index")].append(face)
    best_faces = [max(items, key=_face_quality_key) for items in grouped.values()]
    return sorted(best_faces, key=_face_sort_key)


def _fill_gallery_faces(
    selected: list[dict],
    faces: list[dict],
    target_count: int,
) -> list[dict]:
    """Fill shortages while preferring unused shots and larger temporal gaps."""
    selected = _dedupe_faces(selected)
    seen = {_face_marker(face) for face in selected}

    while len(selected) < target_count:
        candidates = [face for face in faces if _face_marker(face) not in seen]
        if not candidates:
            break
        best = max(candidates, key=lambda face: _gallery_fill_key(face, selected))
        selected.append(best)
        seen.add(_face_marker(best))

    return selected


def _gallery_fill_key(face: dict, selected: list[dict]) -> tuple:
    selected_scenes = {_face_scene_index(item) for item in selected}
    scene_is_new = _face_scene_index(face) not in selected_scenes
    return (
        int(scene_is_new),
        _min_timestamp_gap(face, selected),
        *_face_quality_key(face),
    )


def _min_timestamp_gap(face: dict, selected: list[dict]) -> float:
    if not selected:
        return float("inf")
    timestamp = _face_timestamp(face)
    return min(abs(timestamp - _face_timestamp(item)) for item in selected)


def _time_sample_faces(faces: list[dict], count: int) -> list[dict]:
    """沿视频时间轴均匀采样，覆盖角色在不同剧情阶段的造型变化。"""
    if count <= 0 or not faces:
        return []
    sorted_by_time = sorted(_dedupe_faces(faces), key=_face_sort_key)
    if len(sorted_by_time) <= count:
        return sorted_by_time

    time_min = _face_timestamp(sorted_by_time[0])
    time_max = _face_timestamp(sorted_by_time[-1])
    if time_max <= time_min:
        return _top_confidence_faces(faces, count)

    selected = []
    used = set()
    bucket_width = (time_max - time_min) / count

    for i in range(count):
        start = time_min + bucket_width * i
        end = time_max if i == count - 1 else start + bucket_width
        bucket = [
            face for face in sorted_by_time
            if _face_marker(face) not in used
            and start <= _face_timestamp(face) <= end
        ]
        if bucket:
            best = max(bucket, key=_face_quality_key)
            selected.append(best)
            used.add(_face_marker(best))

    for i in range(count):
        if len(selected) >= count:
            break
        target_time = time_min + ((time_max - time_min) * i / max(count - 1, 1))
        remaining = [
            face for face in sorted_by_time
            if _face_marker(face) not in used
        ]
        if not remaining:
            break
        best = min(
            remaining,
            key=lambda face: (
                abs(_face_timestamp(face) - target_time),
                -float(face.get("det_score", 0.0)),
            ),
        )
        selected.append(best)
        used.add(_face_marker(best))

    return sorted(_dedupe_faces(selected), key=_face_sort_key)


def _top_confidence_faces(faces: list[dict], count: int) -> list[dict]:
    """按 InsightFace 检测置信度选质量最高的脸。"""
    if count <= 0:
        return []
    return sorted(
        _dedupe_faces(faces),
        key=lambda f: -float(f.get("det_score", 0.0)),
    )[:count]


def _face_timestamp(face: dict) -> float:
    try:
        return float(face.get("timestamp", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _face_sort_key(face: dict) -> tuple:
    return (
        _face_timestamp(face),
        _face_scene_index(face),
        str(face.get("keyframe_path", "")),
        _face_bbox_key(face),
    )


def _face_scene_index(face: dict) -> int:
    try:
        return int(face.get("scene_index", -1))
    except (TypeError, ValueError):
        return -1


def _face_quality_key(face: dict) -> tuple:
    return (
        float(face.get("det_score", 0.0)),
        _face_size(face.get("bbox", [])),
    )


def _face_bbox_key(face: dict) -> tuple:
    try:
        return tuple(round(float(v), 2) for v in face.get("bbox", []))
    except (TypeError, ValueError):
        return tuple()


def _dedupe_faces(faces: list[dict]) -> list[dict]:
    """按来源关键帧 + bbox 去重，保留第一次出现的选择顺序。"""
    result = []
    seen = set()
    for face in faces:
        marker = _face_marker(face)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(face)
    return result


def _face_marker(face: dict) -> tuple:
    """构造人脸去重键；bbox 保留两位小数以兼容 float/int 表示差异。"""
    return (
        face.get("keyframe_path"),
        _face_bbox_key(face),
    )


# ---------------------------------------------------------------------------
# Image output
# ---------------------------------------------------------------------------

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
