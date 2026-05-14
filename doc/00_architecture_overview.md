# 系统架构总览

> 本文是所有模块文档的索引与全局架构说明。
> 各模块的详细实现文档见本目录下的专项文件。

## 文档索引

| 文件 | 涵盖模块 | 内容 |
|------|----------|------|
| [01_pipeline_understand.md](01_pipeline_understand.md) | `understand.py` + 10 个子模块 | 理解流水线 10 步详解、数据流、断点续跑 |
| [02_models_schemas.md](02_models_schemas.md) | `schemas.py` | 所有 Pydantic 数据模型定义、字段说明、向后兼容性 |
| [03_search_engine.md](03_search_engine.md) | `search.py` | 三层漏斗检索详解（Embedding + 关键词 + LLM Reranker） |
| [04_director_agent.md](04_director_agent.md) | `director.py` + `prompts.py` | Director Agent 短视频/长视频规划、证据填充、Prompt 设计 |
| [05_reviewer_agent.md](05_reviewer_agent.md) | `reviewer.py` | Reviewer Agent 三层校验（规则 + Grounding + LLM）、评分逻辑 |
| [06_render_engine.md](06_render_engine.md) | `engine.py` + `validator.py` + `ffmpeg_ops.py` | 渲染 5 步流水线、FFmpeg 操作封装 |
| [07_utils.md](07_utils.md) | `llm_client.py` + `ffmpeg_utils.py` + `logger.py` | LLM 客户端、FFmpeg 工具、日志系统 |
| [08_store_cli_config.md](08_store_cli_config.md) | `store.py` + `main.py` + `config.py` | 存储层、命令行接口、全局配置、目录结构 |

---

## 端到端数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         UNDERSTAND 阶段                                 │
│                                                                         │
│  video.mp4                                                              │
│    ├─[1 ingest]──→ meta.json                                           │
│    ├─[2 scene_detect]──→ scenes.json  ◄─── 时间轴锚点                   │
│    ├─[3 keyframe]──→ keyframes/*.jpg                                    │
│    ├─[4 asr]──→ transcripts.json  (每条带 scene_index)                  │
│    ├─[5 vision]──→ ocr.json + vision.json                              │
│    ├─[6 character]──→ characters.json                                   │
│    ├─[7 speaker_bind]──→ speaker_map.json                              │
│    │    + 更新 transcripts (character_id) + characters (speaker_ids)    │
│    ├─[8 event]──→ events.json  (每个带 scene_indices)                   │
│    ├─[9 memory_builder]──→ memory.json                                  │
│    │    含 MemoryUnit[] + 角色判定 (role)                               │
│    └─[10 indexer]──→ search_index.json + faiss.index + id_map.json     │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         SEARCH 阶段                                     │
│                                                                         │
│  query + VideoMemory                                                    │
│    ├─ Layer 1: FAISS Embedding 粗召回 (top-50)                          │
│    ├─ Layer 2: 关键词精筛 + 合并去重 (top-20)                            │
│    └─ Layer 3: LLM Reranker 语义重排 (top-k)                           │
│         │                                                               │
│         └──→ SearchResult[] (含 matched_modalities + source_refs)       │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         EDIT 阶段                                       │
│                                                                         │
│  Director Agent                                                         │
│    ├─ 构造候选信息 Prompt（含多模态融合文本 + 证据来源）                     │
│    ├─ LLM 生成 clips JSON                                              │
│    ├─ 解析 + 白名单校验 + 证据自动填充                                    │
│    └─ Reviewer Agent 审核                                               │
│         ├─ 规则校验 (6项)                                                │
│         ├─ Grounding 校验 (4项)                                          │
│         └─ LLM 审核                                                     │
│              │                                                          │
│              ├─ 通过 → 保存 EditPlan                                    │
│              └─ 不通过 → 反馈加入 prompt，重试 (最多3轮)                  │
│                                                                         │
│  长视频 (>30min): 分章节滑窗规划 → 合并                                  │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         RENDER 阶段                                     │
│                                                                         │
│  EditPlan                                                               │
│    ├─ 校验 (validator)                                                  │
│    ├─ 裁剪 (cut_clip_precise)                                          │
│    ├─ 变速/音量/淡入淡出                                                 │
│    ├─ 标准化 (normalize)                                                │
│    ├─ 拼接 (concat)                                                     │
│    └─ BGM 混合 (可选)                                                   │
│         │                                                               │
│         └──→ output.mp4                                                 │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 关键设计决策

### 1. 先切后提

> 所有模态数据必须以 `scene_index` 为锚点，不允许仅依赖时间戳模糊匹配。

ASR 不再整体提取音频后整体转写，而是先切分镜头，再按 shot 段提取音频并转写。这保证每条 `TranscriptSegment` 天然携带 `scene_index`。

### 2. MemoryUnit 多模态融合

> 检索的最小单元是 MemoryUnit，而非单一模态的 list。

MemoryUnit 将一个 shot 内的所有信息（台词、画面、OCR、人物、事件）融合为一个对象，配合预计算的 `combined_text` 和 `embedding` 实现一站式检索。

### 3. 证据驱动的 EditPlan

> Director 不允许自造片段，每个 EditClip 必须有 `evidence_refs`。

- Director 的候选列表由 search 系统提供，Prompt 明确要求只能从候选中选择
- `_parse_editplan()` 自动从 SearchResult 填充证据字段
- Reviewer 的 Grounding 校验确保证据链完整

### 4. 容错与降级

全系统设计了多层 fallback：

| 场景 | 降级策略 |
|------|----------|
| InsightFace 不可用 | Gemini Vision 替代人脸检测 |
| FAISS 不可用 | 跳过 Embedding 检索层 |
| Embedding API 不可用 | 跳过向量索引 |
| LLM Reranker 失败 | 使用 Layer 2 关键词结果 |
| LLM 角色判定失败 | 按出镜时长排序自动分配 |
| LLM 审核失败 | 仅使用规则审核结果 |

### 5. 断点续跑

每步完成后在 `progress.json` 打标记。恢复时自动跳到第一个未完成的步骤。子模块内部也有输出文件存在性检查。
