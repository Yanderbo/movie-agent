# -*- coding: utf-8 -*-
"""
检索索引构建（重构版）

基于 MemoryUnit 构建检索索引：
1. 文本索引（关键词 + 2-gram）— 兼容旧逻辑
2. Embedding 向量索引 — 调用 Embedding API 为每个 MemoryUnit 生成向量
3. FAISS 本地向量索引 — 用于快速近似最近邻检索

输出：
- search_index.json: 文本索引（向后兼容）
- memory_units.json: 带 embedding 的 MemoryUnit 列表
- faiss.index: FAISS 向量索引文件
- id_map.json: FAISS index → scene_index 映射
"""
import json
import time
import numpy as np
from pathlib import Path

import config
from memory.store import load_memory
from utils.logger import get_logger

logger = get_logger("Indexer")

# ─── FAISS 可用性检查 ──────────────────────────────────────
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.info("FAISS 未安装，向量索引将跳过。安装方式: pip install faiss-cpu")


def build_search_index(video_id: str) -> str:
    """
    为指定视频构建检索索引。

    1. 构建传统文本索引 (search_index.json)
    2. 为每个 MemoryUnit 生成 embedding
    3. 构建 FAISS 向量索引

    Args:
        video_id: 视频 ID

    Returns:
        索引文件路径
    """
    memory = load_memory(video_id)

    index_dir = config.VIDEOS_DIR / video_id / "index"
    index_dir.mkdir(parents=True, exist_ok=True)

    # ════ Part 1: 文本索引（兼容旧逻辑）════
    text_index_path = _build_text_index(memory, index_dir)

    # ════ Part 2: Embedding + FAISS 索引 ════
    _build_embedding_index(memory, index_dir)

    return str(text_index_path)


def _build_text_index(memory, index_dir: Path) -> Path:
    """构建文本搜索索引（向后兼容旧格式）"""
    index_path = index_dir / "search_index.json"
    entries = []

    for scene in memory.scenes:
        # 从 MemoryUnit 获取数据（如果存在）
        mu = next(
            (u for u in memory.memory_units if u.scene_index == scene.scene_index),
            None,
        )

        if mu:
            # 使用 MemoryUnit 中已整合的数据
            transcript_text = " ".join([t.text for t in mu.transcripts])
            vision_desc = mu.vision.description if mu.vision else ""
            vision_mood = mu.vision.mood if mu.vision else ""
            vision_scene_type = mu.vision.scene_type if mu.vision else ""
            events_data = [
                {
                    "event_type": e.event_type,
                    "description": e.description,
                    "importance": e.importance,
                    "emotion": e.emotion,
                }
                for e in mu.events
            ]
            char_names = mu.characters
            combined_text = mu.combined_text
        else:
            # 回退：从 memory 中手动收集（兼容旧数据）
            scene_transcripts = [
                t.text for t in memory.transcripts
                if t.start_time >= scene.start_time and t.start_time < scene.end_time
            ]
            scene_vision = next(
                (v for v in memory.vision_summaries if v.scene_index == scene.scene_index),
                None,
            )
            scene_events = [
                e for e in memory.events
                if (e.start_time < scene.end_time and e.end_time > scene.start_time)
            ]
            scene_characters = [
                c.display_name for c in memory.characters
                if scene.scene_index in c.appearance_scenes
            ]

            transcript_text = " ".join(scene_transcripts)
            vision_desc = scene_vision.description if scene_vision else ""
            vision_mood = scene_vision.mood if scene_vision else ""
            vision_scene_type = scene_vision.scene_type if scene_vision else ""
            events_data = [
                {
                    "event_type": e.event_type,
                    "description": e.description,
                    "importance": e.importance,
                    "emotion": e.emotion,
                }
                for e in scene_events
            ]
            char_names = scene_characters

            # 构建 combined_text
            parts = []
            if transcript_text:
                parts.append(transcript_text)
            if vision_desc:
                parts.append(vision_desc)
            if vision_mood:
                parts.append(vision_mood)
            for e in scene_events:
                parts.append(e.description)
            combined_text = " ".join(parts)

        keywords = _extract_keywords(combined_text)

        entry = {
            "scene_index": scene.scene_index,
            "start_time": scene.start_time,
            "end_time": scene.end_time,
            "duration": scene.duration,
            "transcript": transcript_text,
            "vision_description": vision_desc,
            "vision_mood": vision_mood,
            "vision_scene_type": vision_scene_type,
            "events": events_data,
            "characters": char_names,
            "combined_text": combined_text,
            "keywords": keywords,
        }
        entries.append(entry)

    index_data = {
        "video_id": memory.video_id,
        "total_scenes": len(memory.scenes),
        "total_entries": len(entries),
        "entries": entries,
    }

    index_path.write_text(
        json.dumps(index_data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info(f"文本索引已构建: {index_path} ({len(entries)} 个条目)")
    return index_path


def _build_embedding_index(memory, index_dir: Path):
    """
    为每个 MemoryUnit 生成 embedding 并构建 FAISS 索引。

    如果 Embedding API 不可用或 FAISS 未安装，会优雅降级。
    """
    if not memory.memory_units:
        logger.info("无 MemoryUnit，跳过 embedding 索引")
        return

    # 收集所有需要 embedding 的文本
    texts = [mu.combined_text for mu in memory.memory_units]
    scene_indices = [mu.scene_index for mu in memory.memory_units]

    if not any(t.strip() for t in texts):
        logger.info("所有 MemoryUnit 文本为空，跳过 embedding 索引")
        return

    # 尝试生成 embeddings
    embeddings = _generate_embeddings(texts)

    if embeddings is None:
        logger.warning("Embedding 生成失败，跳过向量索引")
        return

    # 将 embedding 写回 MemoryUnit
    for i, mu in enumerate(memory.memory_units):
        if i < len(embeddings):
            mu.embedding = embeddings[i]

    # 保存更新后的 memory（含 embedding）
    from memory.store import save_memory
    save_memory(memory)

    # 构建 FAISS 索引
    if HAS_FAISS and embeddings:
        _build_faiss_index(embeddings, scene_indices, index_dir)


def _generate_embeddings(texts: list[str]) -> list[list[float]] | None:
    """
    调用 Embedding API 生成文本向量。

    使用 OpenAI 兼容的 /embeddings 端点。
    分批处理以避免超过 API 限制。
    """
    import requests

    api_base = config.LLM_API_BASE.rstrip("/")
    api_key = config.LLM_API_KEY
    model = config.EMBEDDING_MODEL

    if not api_key or not api_base:
        logger.warning("Embedding API 未配置，跳过")
        return None

    embeddings_url = f"{api_base}/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    all_embeddings = []
    batch_size = 20  # 每批最多 20 个文本

    for batch_start in range(0, len(texts), batch_size):
        batch_texts = texts[batch_start:batch_start + batch_size]
        # 截断过长的文本
        batch_texts = [t[:8000] if t else " " for t in batch_texts]

        payload = {
            "model": model,
            "input": batch_texts,
        }

        try:
            resp = requests.post(
                embeddings_url, headers=headers, json=payload,
                timeout=config.LLM_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.warning(f"Embedding API 返回 {resp.status_code}: {resp.text[:200]}")
                return None

            data = resp.json()
            batch_embeddings = [
                item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])
            ]
            all_embeddings.extend(batch_embeddings)

            logger.debug(f"Embedding 批次 {batch_start // batch_size + 1}: {len(batch_embeddings)} 个")
            time.sleep(0.3)

        except Exception as e:
            logger.warning(f"Embedding API 调用失败: {e}")
            return None

    logger.info(f"Embedding 生成完成: {len(all_embeddings)} 个向量")
    return all_embeddings


def _build_faiss_index(
    embeddings: list[list[float]], scene_indices: list[int], index_dir: Path
):
    """构建 FAISS 向量索引"""
    if not embeddings:
        return

    dim = len(embeddings[0])
    vectors = np.array(embeddings, dtype=np.float32)

    # L2 归一化（用于余弦相似度）
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    vectors = vectors / norms

    # 构建 IndexFlatIP（内积 = 余弦相似度，在归一化后）
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    # 保存索引
    faiss_path = str(index_dir / "faiss.index")
    faiss.write_index(index, faiss_path)

    # 保存 id 映射
    id_map = {str(i): scene_indices[i] for i in range(len(scene_indices))}
    id_map_path = index_dir / "id_map.json"
    id_map_path.write_text(
        json.dumps(id_map, indent=2), encoding="utf-8"
    )

    logger.info(f"FAISS 索引已构建: {faiss_path} (dim={dim}, n={len(embeddings)})")


def _extract_keywords(text: str) -> list[str]:
    """
    从文本中提取关键词（简单实现）。

    对中文使用 2-gram，对英文使用空格分词。
    去除过短的词和常见停用词。
    """
    if not text:
        return []

    words = set()

    # 空格分词（适合英文和混合文本）
    for w in text.split():
        w = w.strip("，。！？、：；""''（）【】,.!?:;\"'()[]")
        if len(w) >= 2:
            words.add(w.lower())

    # 对无空格的纯中文片段使用 2-gram
    parts = text.split()
    for part in parts:
        if len(part) >= 2 and not part.isascii():
            for i in range(len(part) - 1):
                gram = part[i:i+2]
                if len(gram) == 2:
                    words.add(gram)

    return sorted(words)[:100]  # 限制关键词数量
