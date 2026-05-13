# -*- coding: utf-8 -*-
"""
Video Memory 检索（重构版）

三层漏斗检索：
  Layer 1: Embedding 向量粗检 → top-50 MemoryUnit (FAISS)
  Layer 2: 关键词 / BM25 精筛 → 从中精筛 top-20
  Layer 3: LLM Reranker → 语义重排 → top-k 输出

支持：人物过滤、时间范围过滤、场景类型过滤。
SearchResult 携带完整证据链（matched_modalities, source_refs, context）。
"""
import json
import numpy as np
from pathlib import Path

import config
from models.schemas import VideoMemory, SearchResult, MemoryUnit, Scene
from utils.llm_client import get_llm_client
from utils.logger import get_logger

logger = get_logger("Search")

# ─── FAISS 可用性检查 ──────────────────────────────────────
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


def search_memory(
    memory: VideoMemory,
    query: str,
    top_k: int = 10,
    character_filter: str = None,
    time_range: tuple[float, float] = None,
    scene_type_filter: str = None,
    use_semantic: bool = True,
) -> list[SearchResult]:
    """
    在 Video Memory 中搜索相关片段（三层漏斗）。

    Args:
        memory: Video Memory
        query: 搜索查询
        top_k: 返回结果数
        character_filter: 过滤人物 ID
        time_range: (start, end) 时间范围过滤
        scene_type_filter: 场景类型过滤
        use_semantic: 是否使用 LLM Reranker

    Returns:
        SearchResult 列表，按分数降序排列
    """
    # ════ Layer 1: Embedding 向量粗检 ════
    embedding_results = _embedding_search(memory, query, top_n=50)

    # ════ Layer 2: 关键词精筛 ════
    keyword_results = []
    keyword_results.extend(_keyword_search_transcripts(memory, query))
    keyword_results.extend(_keyword_search_vision(memory, query))
    keyword_results.extend(_keyword_search_events(memory, query))
    keyword_results.extend(_index_search(memory.video_id, query))

    # 合并 Layer 1 + Layer 2
    merged = _merge_results(embedding_results, keyword_results)

    # 应用过滤器
    if character_filter:
        merged = _filter_by_character(merged, memory, character_filter)
    if time_range:
        merged = _filter_by_time(merged, memory, time_range)
    if scene_type_filter:
        merged = _filter_by_scene_type(merged, memory, scene_type_filter)

    # 去重（合并同 scene 的多模态命中）
    merged = _deduplicate_with_modality_merge(merged)

    # 排序取 top-20 进入 Layer 3
    merged.sort(key=lambda r: r.score, reverse=True)
    candidates = merged[:20]

    # ════ Layer 3: LLM Reranker ════
    if use_semantic and query and len(candidates) > 3:
        try:
            candidates = _llm_rerank(memory, query, candidates)
        except Exception as e:
            logger.warning(f"LLM Reranker 失败，使用原始排序: {e}")

    # 截取 top-k
    candidates = candidates[:top_k]

    # 填充场景信息 + 上下文
    for r in candidates:
        _enrich_result(r, memory)

    return candidates


# ═══════════════════════════════════════════════════════════════
# Layer 1: Embedding 向量检索
# ═══════════════════════════════════════════════════════════════

def _embedding_search(
    memory: VideoMemory, query: str, top_n: int = 50
) -> list[SearchResult]:
    """使用 FAISS 向量索引做粗召回"""
    if not HAS_FAISS:
        return []

    index_dir = config.VIDEOS_DIR / memory.video_id / "index"
    faiss_path = str(index_dir / "faiss.index")
    id_map_path = index_dir / "id_map.json"

    if not Path(faiss_path).exists() or not id_map_path.exists():
        return []

    try:
        # 加载索引
        index = faiss.read_index(faiss_path)
        id_map = json.loads(id_map_path.read_text(encoding="utf-8"))

        # 生成 query embedding
        query_embedding = _get_query_embedding(query)
        if query_embedding is None:
            return []

        query_vec = np.array([query_embedding], dtype=np.float32)
        # L2 归一化
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        # 检索
        k = min(top_n, index.ntotal)
        scores, indices = index.search(query_vec, k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0:
                continue
            scene_index = id_map.get(str(idx), -1)
            if scene_index < 0:
                continue

            results.append(SearchResult(
                scene_index=scene_index,
                score=float(score),
                match_type="embedding",
                snippet="",
                matched_modalities=["embedding"],
                source_refs=[f"faiss.index#{idx}"],
            ))

        logger.debug(f"Embedding 粗检: {len(results)} 个候选")
        return results

    except Exception as e:
        logger.warning(f"FAISS 检索失败: {e}")
        return []


def _get_query_embedding(query: str) -> list[float] | None:
    """获取 query 的 embedding 向量"""
    import requests

    api_base = config.LLM_API_BASE.rstrip("/")
    api_key = config.LLM_API_KEY
    model = config.EMBEDDING_MODEL

    if not api_key or not api_base:
        return None

    try:
        resp = requests.post(
            f"{api_base}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={"model": model, "input": [query[:8000]]},
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data["data"][0]["embedding"]
    except Exception as e:
        logger.warning(f"Query embedding 生成失败: {e}")
    return None


# ═══════════════════════════════════════════════════════════════
# Layer 2: 关键词检索
# ═══════════════════════════════════════════════════════════════

def _keyword_search_transcripts(memory: VideoMemory, query: str) -> list[SearchResult]:
    """在台词中搜索关键词"""
    results = []
    query_lower = query.lower()
    keywords = query_lower.split()

    for trans in memory.transcripts:
        text_lower = trans.text.lower()
        match_count = sum(1 for kw in keywords if kw in text_lower)
        if match_count > 0:
            score = match_count / len(keywords)
            scene_idx = trans.scene_index if trans.scene_index >= 0 else _find_scene_index(memory, trans.start_time)
            results.append(SearchResult(
                scene_index=scene_idx,
                score=score,
                match_type="keyword_transcript",
                snippet=trans.text[:100],
                transcript=trans.text,
                matched_modalities=["transcript"],
                source_refs=[f"transcripts.json#t{trans.start_time:.1f}"],
            ))

    return results


def _keyword_search_vision(memory: VideoMemory, query: str) -> list[SearchResult]:
    """在画面摘要中搜索关键词"""
    results = []
    query_lower = query.lower()
    keywords = query_lower.split()

    for vs in memory.vision_summaries:
        text = f"{vs.description} {vs.mood} {vs.scene_type} {' '.join(vs.objects)}"
        text_lower = text.lower()
        match_count = sum(1 for kw in keywords if kw in text_lower)
        if match_count > 0:
            score = match_count / len(keywords) * 0.8
            results.append(SearchResult(
                scene_index=vs.scene_index,
                score=score,
                match_type="keyword_vision",
                snippet=vs.description[:100],
                vision_summary=vs.description,
                matched_modalities=["vision"],
                source_refs=[f"vision.json#scene_{vs.scene_index}"],
            ))

    return results


def _keyword_search_events(memory: VideoMemory, query: str) -> list[SearchResult]:
    """在事件中搜索关键词"""
    results = []
    query_lower = query.lower()
    keywords = query_lower.split()

    for event in memory.events:
        text = f"{event.description} {event.event_type} {event.emotion}"
        text_lower = text.lower()
        match_count = sum(1 for kw in keywords if kw in text_lower)
        if match_count > 0:
            score = match_count / len(keywords) * 0.9
            score *= (1 + event.importance * 0.05)
            # 从 event 的 scene_indices 中取第一个（优先使用新字段）
            if event.scene_indices:
                scene_idx = event.scene_indices[0]
            else:
                scene_idx = _find_scene_index(memory, event.start_time)
            results.append(SearchResult(
                scene_index=scene_idx,
                score=min(score, 2.0),
                match_type="keyword_event",
                snippet=event.description[:100],
                matched_modalities=["event"],
                source_refs=[f"events.json#event_{event.event_index}"],
            ))

    return results


def _load_search_index(video_id: str) -> dict | None:
    """加载预计算的搜索索引。不存在或损坏时返回 None。"""
    index_path = config.VIDEOS_DIR / video_id / "index" / "search_index.json"
    if not index_path.exists():
        return None
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "entries" in data:
            return data
    except Exception as e:
        logger.warning(f"加载搜索索引失败，回退到原有检索: {e}")
    return None


def _index_search(video_id: str, query: str) -> list[SearchResult]:
    """使用预计算的 search_index.json 进行关键词检索"""
    index_data = _load_search_index(video_id)
    if not index_data:
        return []

    results = []
    query_lower = query.lower()
    query_keywords = set(query_lower.split())

    for part in query_lower.split():
        if len(part) >= 2 and not part.isascii():
            for i in range(len(part) - 1):
                query_keywords.add(part[i:i+2])

    if not query_keywords:
        return []

    for entry in index_data.get("entries", []):
        scene_index = entry.get("scene_index", -1)
        entry_keywords = set(entry.get("keywords", []))
        combined_text = entry.get("combined_text", "").lower()

        keyword_overlap = len(query_keywords & entry_keywords)
        text_match_count = sum(1 for kw in query_keywords if kw in combined_text)

        total_matches = keyword_overlap + text_match_count
        if total_matches > 0:
            score = min(total_matches / (len(query_keywords) * 2), 1.5) * 0.85

            # 确定命中的模态
            modalities = []
            if entry.get("transcript") and any(kw in entry["transcript"].lower() for kw in query_keywords):
                modalities.append("transcript")
            if entry.get("vision_description") and any(kw in entry["vision_description"].lower() for kw in query_keywords):
                modalities.append("vision")
            if not modalities:
                modalities.append("index")

            snippet = entry.get("vision_description", "") or entry.get("transcript", "")
            results.append(SearchResult(
                scene_index=scene_index,
                score=round(score, 3),
                match_type="index",
                snippet=snippet[:100],
                matched_modalities=modalities,
                source_refs=[f"search_index.json#scene_{scene_index}"],
            ))

    return results


# ═══════════════════════════════════════════════════════════════
# Layer 3: LLM Reranker
# ═══════════════════════════════════════════════════════════════

def _llm_rerank(
    memory: VideoMemory, query: str, candidates: list[SearchResult]
) -> list[SearchResult]:
    """
    使用 LLM 对候选结果做语义重排。

    与旧版 _semantic_search 不同：
    - 只对已筛选的候选做 rerank（不是逐场景评估）
    - 输入包含多模态融合文本
    - 输出包含解释性 relevance 分数
    """
    client = get_llm_client()

    # 构造候选摘要
    candidate_summaries = []
    for i, r in enumerate(candidates):
        # 从 MemoryUnit 获取丰富信息
        mu = next(
            (u for u in memory.memory_units if u.scene_index == r.scene_index),
            None,
        )
        if mu:
            summary = (
                f"[{i}] Scene {r.scene_index} "
                f"[{mu.start_time:.1f}s-{mu.end_time:.1f}s]: "
                f"{mu.combined_text[:200]}"
            )
        else:
            scene = r.scene
            time_str = f"[{scene.start_time:.1f}s-{scene.end_time:.1f}s]" if scene else ""
            parts = []
            if r.transcript:
                parts.append(f"台词: {r.transcript[:80]}")
            if r.vision_summary:
                parts.append(f"画面: {r.vision_summary[:80]}")
            summary = f"[{i}] Scene {r.scene_index} {time_str}: {' | '.join(parts)}"

        candidate_summaries.append(summary)

    summaries_text = "\n".join(candidate_summaries)

    prompt = f"""请评估以下视频片段与查询的相关性，并重新排序。

查询: "{query}"

候选片段:
{summaries_text}

请返回 JSON 数组，按相关性从高到低排序。每个元素包含:
- index: 候选片段的编号（方括号中的数字）
- relevance_score: 相关性分数 (0.0 - 1.0)

只返回 relevance_score > 0.2 的片段。只输出 JSON：
```json
[{{"index": 0, "relevance_score": 0.9}}]
```"""

    try:
        response = client.chat(prompt=prompt, temperature=0.2)
        parsed = client.parse_json(response)
        if not parsed or not isinstance(parsed, list):
            return candidates

        # 按 LLM 给出的分数重排
        reranked = []
        for item in parsed:
            idx = int(item.get("index", -1))
            score = float(item.get("relevance_score", 0))
            if 0 <= idx < len(candidates) and score > 0.2:
                result = candidates[idx]
                # 综合分数：原始分数 * 0.3 + LLM 分数 * 0.7
                result.score = round(result.score * 0.3 + score * 0.7, 3)
                if "semantic" not in result.matched_modalities:
                    result.matched_modalities.append("semantic")
                reranked.append(result)

        # 补回未被 LLM 评估的候选（低分追加）
        reranked_indices = {candidates.index(r) for r in reranked}
        for i, c in enumerate(candidates):
            if i not in reranked_indices:
                c.score *= 0.3  # 降权
                reranked.append(c)

        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked

    except Exception as e:
        logger.warning(f"LLM Reranker 失败: {e}")
        return candidates


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _find_scene_index(memory: VideoMemory, timestamp: float) -> int:
    """根据时间戳找到对应的场景索引"""
    for scene in memory.scenes:
        if scene.start_time <= timestamp < scene.end_time:
            return scene.scene_index
    if memory.scenes:
        closest = min(memory.scenes, key=lambda s: abs(s.start_time - timestamp))
        return closest.scene_index
    return 0


def _merge_results(
    embedding_results: list[SearchResult],
    keyword_results: list[SearchResult],
) -> list[SearchResult]:
    """合并 embedding 和关键词检索结果"""
    all_results = []
    all_results.extend(embedding_results)
    all_results.extend(keyword_results)
    return all_results


def _deduplicate_with_modality_merge(results: list[SearchResult]) -> list[SearchResult]:
    """
    去重：同一 scene 保留最高分，但合并 matched_modalities 和 source_refs。

    这比旧版的简单去重更好——它保留了"这个结果被哪些模态命中"的信息。
    """
    best: dict[int, SearchResult] = {}
    for r in results:
        if r.scene_index not in best:
            best[r.scene_index] = r
        else:
            existing = best[r.scene_index]
            # 合并模态
            for m in r.matched_modalities:
                if m not in existing.matched_modalities:
                    existing.matched_modalities.append(m)
            # 合并来源
            for ref in r.source_refs:
                if ref not in existing.source_refs:
                    existing.source_refs.append(ref)
            # 保留最高分
            if r.score > existing.score:
                existing.score = r.score
                existing.match_type = r.match_type
                existing.snippet = r.snippet or existing.snippet
            # 合并文本字段
            if r.transcript and not existing.transcript:
                existing.transcript = r.transcript
            if r.vision_summary and not existing.vision_summary:
                existing.vision_summary = r.vision_summary
    return list(best.values())


def _filter_by_character(
    results: list[SearchResult], memory: VideoMemory, char_id: str
) -> list[SearchResult]:
    """按人物过滤"""
    char = next((c for c in memory.characters if c.character_id == char_id), None)
    if not char:
        return results
    return [r for r in results if r.scene_index in char.appearance_scenes]


def _filter_by_time(
    results: list[SearchResult], memory: VideoMemory, time_range: tuple[float, float]
) -> list[SearchResult]:
    """按时间范围过滤"""
    start, end = time_range
    filtered = []
    for r in results:
        scene = next((s for s in memory.scenes if s.scene_index == r.scene_index), None)
        if scene and scene.start_time >= start and scene.end_time <= end:
            filtered.append(r)
    return filtered


def _filter_by_scene_type(
    results: list[SearchResult], memory: VideoMemory, scene_type: str
) -> list[SearchResult]:
    """按场景类型过滤"""
    matching_scenes = {
        v.scene_index for v in memory.vision_summaries
        if scene_type.lower() in v.scene_type.lower()
    }
    return [r for r in results if r.scene_index in matching_scenes]


def _enrich_result(result: SearchResult, memory: VideoMemory):
    """为搜索结果填充完整信息：scene、transcript、vision、context、memory_unit"""
    scene = next(
        (s for s in memory.scenes if s.scene_index == result.scene_index), None
    )
    if scene:
        result.scene = scene

    # 填充 MemoryUnit
    mu = next(
        (u for u in memory.memory_units if u.scene_index == result.scene_index),
        None,
    )
    if mu:
        result.memory_unit = mu

    if not result.transcript:
        if mu and mu.transcripts:
            result.transcript = " ".join([t.text for t in mu.transcripts])
        else:
            trans = [
                t.text for t in memory.transcripts
                if scene and t.start_time >= scene.start_time and t.start_time < scene.end_time
            ]
            if trans:
                result.transcript = " ".join(trans)

    if not result.vision_summary:
        if mu and mu.vision:
            result.vision_summary = mu.vision.description
        else:
            vs = next(
                (v for v in memory.vision_summaries if v.scene_index == result.scene_index),
                None,
            )
            if vs:
                result.vision_summary = vs.description

    # 填充上下文（前后 shot 摘要）
    if scene:
        prev_scene = next(
            (s for s in memory.scenes if s.scene_index == scene.scene_index - 1),
            None,
        )
        next_scene = next(
            (s for s in memory.scenes if s.scene_index == scene.scene_index + 1),
            None,
        )
        if prev_scene:
            prev_mu = next(
                (u for u in memory.memory_units if u.scene_index == prev_scene.scene_index),
                None,
            )
            if prev_mu and prev_mu.combined_text:
                result.context_before = prev_mu.combined_text[:100]
        if next_scene:
            next_mu = next(
                (u for u in memory.memory_units if u.scene_index == next_scene.scene_index),
                None,
            )
            if next_mu and next_mu.combined_text:
                result.context_after = next_mu.combined_text[:100]


# ─── CLI 入口 ──────────────────────────────────────────────

def run_search(video_id: str, query: str, top_k: int = 10) -> list[dict]:
    """搜索入口函数"""
    from memory.store import load_memory
    memory = load_memory(video_id)
    results = search_memory(memory, query, top_k=top_k)
    return [r.model_dump() for r in results]
