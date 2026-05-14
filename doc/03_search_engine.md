# 搜索引擎 (search.py)

> 文件：`memory/search.py`
> 职责：三层漏斗检索系统，从 Video Memory 中根据查询条件精准定位片段
>
> **v2 注意**：indexer 已构建 7 种索引（文本/向量/角色/事件/关系/情绪/剪辑信号），
> 当前 search.py 主要消费文本 + 向量索引，角色/事件/剪辑信号索引可在后续版本集成。

## 总体架构

```
用户查询 query
    │
    ▼
┌─────────────────────────────────────┐
│ Layer 1: Embedding 粗召回 (top-50)  │  FAISS 向量近似搜索
│  → cosine similarity ranking       │
└─────────────┬───────────────────────┘
              ▼
┌─────────────────────────────────────┐
│ Layer 2: 关键词精筛                  │  四路并行检索
│  → 台词/画面/事件/文本索引           │  + 合并去重
│  → 合并后 top-20                    │
└─────────────┬───────────────────────┘
              ▼
┌─────────────────────────────────────┐
│ Layer 3: LLM Reranker (top-k)       │  Gemini 语义重排
│  → 逐个打分 + 理由                   │  + 上下文填充
└─────────────────────────────────────┘
              ▼
         SearchResult 列表
```

## 入口函数

### `search_memory(memory, query, top_k=10, time_range=None) -> list[SearchResult]`

主检索入口。参数说明：
- `memory`：VideoMemory 对象
- `query`：用户查询文本
- `top_k`：最终返回数量
- `time_range`：可选 `(start, end)` 元组，限制检索的时间范围（用于长视频章节检索）

### `run_search(video_id, query, top_k=10) -> list[dict]`

CLI 包装，加载 memory → 调用 search_memory → 序列化为 dict 返回。

## 三层检索详解

### Layer 1: Embedding 粗召回

**实现**：`_embedding_recall()`

1. 加载 `index/faiss.index` 和 `index/id_map.json`
2. 调用 Embedding API 生成查询向量
3. L2 归一化后做 FAISS `search()`（内积搜索，等价于余弦相似度）
4. 取 top-50 个候选
5. 通过 `id_map` 转换为 `scene_index`
6. 返回 `{scene_index: score}` 字典

**降级策略**：
- FAISS 未安装 → 跳过此层
- Embedding API 不可用 → 跳过此层
- 索引文件不存在 → 跳过此层

### Layer 2: 关键词精筛

**实现**：`_keyword_search()`

四路并行检索，每路独立打分后合并：

| 路 | 数据源 | 匹配方式 |
|----|--------|----------|
| 台词 | `memory.transcripts` | `query` 子串匹配 `segment.text` |
| 画面 | `memory.vision_summaries` | `query` 子串匹配 `description` / `mood` / `scene_type` |
| 事件 | `memory.events` | `query` 子串匹配 `description` / `event_type` / `emotion` |
| 索引 | `search_index.json` | 加载预计算的关键词索引做查找 |

每路的命中会记录到 `matched_modalities` 中。

**合并去重**：同一个 `scene_index` 被多路命中时，分数累加，`matched_modalities` 合并。

### Layer 3: LLM Reranker

**实现**：`_llm_rerank()`

1. 将 Layer 1 + Layer 2 的结果合并去重，取 top-20 候选
2. 为每个候选构造摘要文本（台词 + 画面 + 事件）
3. 构造 prompt：让 LLM 对每个候选打分 0-1，并给出理由
4. 解析 LLM 响应的 JSON 数组，更新分数
5. 按新分数降序排列，取 top-k
6. 对最终结果填充上下文（`context_before` / `context_after`）

**降级策略**：LLM 不可用时直接返回 Layer 2 的结果。

## 辅助函数

### `_fill_context(result, memory)`

为 SearchResult 填充前后 shot 的摘要：
- `context_before`：前一个 scene 的台词 + 画面
- `context_after`：后一个 scene 的台词 + 画面

### 时间范围过滤

当 `time_range` 不为 None 时，在每层检索后过滤掉不在范围内的结果。用于 Director 的长视频章节式检索。

## 输出格式

每个 `SearchResult` 包含：

```python
SearchResult(
    scene_index=5,
    score=0.87,
    match_type="embedding",       # 最主要的匹配方式
    snippet="台词片段...",
    scene=Scene(...),             # 完整的 Scene (= Shot) 对象
    transcript="完整台词文本",
    vision_summary="画面描述",
    matched_modalities=["transcript", "vision", "embedding"],
    source_refs=["faiss.index#12", "transcripts.json#t120.5"],
    context_before="前一 shot 摘要",
    context_after="后一 shot 摘要",
    memory_unit=MemoryUnit(...),  # 完整的 MemoryUnit
    # v2 新增
    beat_index=3,                 # 所属 Beat
    story_scene_index=1,          # 所属 StoryScene
    edit_signal=EditSignal(...),  # 关联的剪辑信号
)
```
