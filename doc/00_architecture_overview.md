# 系统架构总览

> 本文是所有模块文档的索引与全局架构说明。
> 各模块的详细实现文档见本目录下的专项文件。

## 文档索引

| 文件 | 涵盖模块 | 内容 |
|------|----------|------|
| [01_pipeline_understand.md](01_pipeline_understand.md) | `understand.py` + 17 个子模块 | 理解流水线 17 步详解、叙事层级、数据流、断点续跑 |
| [02_models_schemas.md](02_models_schemas.md) | `schemas.py` | 所有 Pydantic 数据模型定义（含 v3 新增：AudioProsody/MultimodalAlignment/Chapter/NarrativeSignal/RecompositionSignal） |
| [03_search_engine.md](03_search_engine.md) | `search.py` | 三层漏斗检索详解（Embedding + 关键词 + LLM Reranker） |
| [04_director_agent.md](04_director_agent.md) | `director.py` + `prompts.py` | Director Agent 短视频/长视频规划、证据填充、Prompt 设计 |
| [05_reviewer_agent.md](05_reviewer_agent.md) | `reviewer.py` | Reviewer Agent 三层校验（规则 + Grounding + LLM）、评分逻辑 |
| [06_render_engine.md](06_render_engine.md) | `engine.py` + `validator.py` + `ffmpeg_ops.py` | 渲染 5 步流水线、FFmpeg 操作封装 |
| [07_utils.md](07_utils.md) | `llm_client.py` + `ffmpeg_utils.py` + `logger.py` | LLM 客户端、FFmpeg 工具、日志系统 |
| [08_store_cli_config.md](08_store_cli_config.md) | `store.py` + `main.py` + `config.py` | 存储层（v3 多层结构）、命令行接口、全局配置、目录结构 |

---

## 端到端数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         UNDERSTAND 阶段（17步）                          │
│                                                                         │
│  video.mp4                                                              │
│    ├─[1  ingest]──→ meta.json                                           │
│    ├─[2  shot_detect]──→ scenes.json  ◄─── 时间轴锚点                   │
│    ├─[3  multi_keyframe]──→ keyframes/*_f0~f5.jpg (多帧)                │
│    ├─[4  asr_windowed]──→ transcripts.json  (长窗口ASR + 回填shot)      │
│    ├─[5  vision]──→ ocr.json + vision.json (多帧画面理解 + micro_clip)  │
│    ├─[6  audio_analysis]──→ audio_prosody.json 🆕 (音乐/音效/语速/情绪) │
│    ├─[7  character_deep]──→ characters.json (CharacterDeep)             │
│    ├─[8  speaker_bind]──→ speaker_map.json                              │
│    ├─[9  multimodal_align]──→ multimodal_alignments.json 🆕             │
│    ├─[10 beat_detect]──→ beats.json  (shot → beat 聚合)                 │
│    ├─[11 story_scene_detect]──→ story_scenes.json (beat → scene 聚合)   │
│    ├─[12 chapter_detect]──→ chapters.json 🆕 (scene → chapter 聚合)     │
│    ├─[13 event_graph]──→ events.json + event_graph.json (含证据+置信)   │
│    ├─[14 character_arc]──→ character_arcs.json + character_relations.json│
│    ├─[15 edit_signal]──→ edit_signals.json + narrative_signals.json     │
│    │                      + recomposition_signals.json (三类信号)        │
│    ├─[16 build_memory]──→ memory.json (四层MemoryUnit + 角色判定)       │
│    └─[17 indexer]──→ index/ (9种索引文件)                               │
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
│  v3 索引维度: 角色/事件/关系/情绪/剪辑信号/音频/章节                       │
│                                                                         │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         EDIT 阶段                                       │
│                                                                         │
│  Director Agent                                                         │
│    ├─ 构造候选信息 Prompt（含多模态融合文本 + 证据来源 + EditSignal）       │
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

### 1. 先切后提 → 层层聚合

> 所有模态数据必须以 `scene_index` 为锚点，不允许仅依赖时间戳模糊匹配。

v3 在此基础上进一步扩展了聚合层级：shot → beat → story_scene → **chapter** → event_graph。每个层级都有对应的 MemoryUnit。

### 2. 多层 MemoryUnit

> 检索不再局限于 shot 级别，可在 shot / beat / story_scene / **chapter** 四个粒度上进行。

| 层级 | 模型 | 粒度 | 典型用途 |
|------|------|------|---------|
| Shot | `MemoryUnit` | 单个镜头 | 精确片段定位 |
| Beat | `BeatMemoryUnit` | 叙事微单元 | 情节片段检索 |
| StoryScene | `SceneMemoryUnit` | 完整场景 | 场景级别匹配 |
| Chapter | `ChapterMemoryUnit` | 长视频大段落 | 🆕 章节级主题检索 |

### 3. 三类信号驱动的剪辑决策

> v3 将单一 EditSignal 扩展为三类信号体系。

| 信号类型 | 说明 | 服务对象 |
|----------|------|----------|
| `EditSignal` | 8 维剪辑信号（hook/剧情/情绪/视觉/独立性/连续性/边界/剧透） | Director Agent 选材 |
| `NarrativeSignal` | 🆕 叙事弧位置/张力/信息密度/叙事功能 | 结构化叙事编排 |
| `RecompositionSignal` | 🆕 梗潜力/情感引用/平台适配/二创格式 | 二次创作/短视频分发 |

### 4. 证据驱动的 EditPlan

> Director 不允许自造片段，每个 EditClip 必须有 `evidence_refs`。

v3 的 Event/EventEdge 增加了 `evidence`、`confidence`、`relation_basis` 字段，进一步强化证据链可溯源性。

### 5. 容错与降级

全系统设计了多层 fallback：

| 场景 | 降级策略 |
|------|----------|
| InsightFace 不可用 | Gemini Vision 替代人脸检测 |
| FAISS 不可用 | 跳过 Embedding 检索层 |
| Embedding API 不可用 | 跳过向量索引 |
| Beat 检测 LLM 失败 | 每 4 shot 一组默认分组 |
| StoryScene 检测 LLM 失败 | 每 3 beat 一组默认分组 |
| **Chapter 检测 LLM 失败** | 🆕 每 3 个 StoryScene 一组 |
| **音频韵律 LLM 失败** | 🆕 返回空 AudioProsody（仅保留 scene_index） |
| **叙事/二创信号 LLM 失败** | 🆕 返回空列表，不阻塞 |
| 事件关系推理 LLM 失败 | 返回空边列表（仅保留事件节点） |
| 人物弧线 LLM 失败 | 跳过弧线，保留基础人物信息 |
| 剪辑信号 LLM 失败 | 跳过该批次，不阻塞流程 |
| LLM 角色判定失败 | 按出镜时长排序自动分配 |
| LLM 审核失败 | 仅使用规则审核结果 |

### 6. 断点续跑

每步完成后在 `progress.json` 打标记。恢复时自动跳到第一个未完成的步骤。v3 通过 `_STEP_ALIASES` 映射旧步骤名（如 `scene_detect` → `shot_detect`），确保 v1/v2 旧进度文件可正常续跑。
