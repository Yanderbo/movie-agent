# 系统架构总览

> 本文是所有模块文档的索引与全局架构说明。
> 各模块的详细实现文档见本目录下的专项文件。

## 文档索引

| 文件 | 涵盖模块 | 内容 |
|------|----------|------|
| [01_pipeline_understand.md](01_pipeline_understand.md) | `understand.py` + 14 个子模块 | 理解流水线 14 步详解、叙事层级、数据流、断点续跑 |
| [02_models_schemas.md](02_models_schemas.md) | `schemas.py` | 所有 Pydantic 数据模型定义（含 v2 新增：Shot/Beat/StoryScene/EventGraph/EditSignal/CharacterDeep） |
| [03_search_engine.md](03_search_engine.md) | `search.py` | 三层漏斗检索详解（Embedding + 关键词 + LLM Reranker） |
| [04_director_agent.md](04_director_agent.md) | `director.py` + `prompts.py` | Director Agent 短视频/长视频规划、证据填充、Prompt 设计 |
| [05_reviewer_agent.md](05_reviewer_agent.md) | `reviewer.py` | Reviewer Agent 三层校验（规则 + Grounding + LLM）、评分逻辑 |
| [06_render_engine.md](06_render_engine.md) | `engine.py` + `validator.py` + `ffmpeg_ops.py` | 渲染 5 步流水线、FFmpeg 操作封装 |
| [07_utils.md](07_utils.md) | `llm_client.py` + `ffmpeg_utils.py` + `logger.py` | LLM 客户端、FFmpeg 工具、日志系统 |
| [08_store_cli_config.md](08_store_cli_config.md) | `store.py` + `main.py` + `config.py` | 存储层（v2 多层结构）、命令行接口、全局配置、目录结构 |

---

## 端到端数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         UNDERSTAND 阶段（14步）                          │
│                                                                         │
│  video.mp4                                                              │
│    ├─[1 ingest]──→ meta.json                                           │
│    ├─[2 shot_detect]──→ scenes.json  ◄─── 时间轴锚点                   │
│    ├─[3 multi_keyframe]──→ keyframes/*_f0~f5.jpg (多帧)                │
│    ├─[4 asr_windowed]──→ transcripts.json  (长窗口ASR + 回填shot)      │
│    ├─[5 vision]──→ ocr.json + vision.json (多帧画面理解)               │
│    ├─[6 character_deep]──→ characters.json (CharacterDeep)             │
│    ├─[7 speaker_bind]──→ speaker_map.json                              │
│    ├─[8 beat_detect]──→ beats.json  (shot → beat 聚合)                 │
│    ├─[9 story_scene_detect]──→ story_scenes.json (beat → scene 聚合)   │
│    ├─[10 event_graph]──→ events.json + event_graph.json (含关系边)     │
│    ├─[11 character_arc]──→ character_arcs.json + character_relations   │
│    ├─[12 edit_signal]──→ edit_signals.json (8维剪辑信号)               │
│    ├─[13 build_memory]──→ memory.json (三层MemoryUnit + 角色判定)      │
│    └─[14 indexer]──→ index/ (7种索引文件)                              │
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
│  v2 新增索引维度: 角色/事件/关系/情绪/剪辑信号                            │
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

v2 在此基础上增加了 **层层聚合** 原则：shot → beat → story_scene 逐级构建更高层叙事单元。每个层级都有对应的 MemoryUnit。

### 2. 多层 MemoryUnit

> 检索不再局限于 shot 级别，可在 shot / beat / story_scene 三个粒度上进行。

| 层级 | 模型 | 粒度 | 典型用途 |
|------|------|------|---------|
| Shot | `MemoryUnit` | 单个镜头 | 精确片段定位 |
| Beat | `BeatMemoryUnit` | 叙事微单元 | 情节片段检索 |
| StoryScene | `SceneMemoryUnit` | 完整场景 | 场景级别匹配 |

### 3. EditSignal 驱动的剪辑决策

> 不再仅凭 "相关性" 选择片段，而是综合 8 维剪辑信号做出更专业的选择。

EditSignal 为 Director Agent 提供了量化的选材依据：哪些片段适合做开头（hook_score）、哪些不能剧透（spoiler_level）、哪些可以独立剪出（independence_score）。

### 4. 证据驱动的 EditPlan

> Director 不允许自造片段，每个 EditClip 必须有 `evidence_refs`。

v2 新增 `edit_signal_ref` / `source_beat_index` / `source_story_scene_index` 字段，进一步强化证据链。

### 5. 容错与降级

全系统设计了多层 fallback：

| 场景 | 降级策略 |
|------|----------|
| InsightFace 不可用 | Gemini Vision 替代人脸检测 |
| FAISS 不可用 | 跳过 Embedding 检索层 |
| Embedding API 不可用 | 跳过向量索引 |
| Beat 检测 LLM 失败 | 每 4 shot 一组默认分组 |
| StoryScene 检测 LLM 失败 | 每 3 beat 一组默认分组 |
| 事件关系推理 LLM 失败 | 返回空边列表（仅保留事件节点） |
| 人物弧线 LLM 失败 | 跳过弧线，保留基础人物信息 |
| 剪辑信号 LLM 失败 | 跳过该批次，不阻塞流程 |
| LLM 角色判定失败 | 按出镜时长排序自动分配 |
| LLM 审核失败 | 仅使用规则审核结果 |

### 6. 断点续跑

每步完成后在 `progress.json` 打标记。恢复时自动跳到第一个未完成的步骤。v2 通过 `_STEP_ALIASES` 映射旧步骤名（如 `scene_detect` → `shot_detect`），确保旧进度文件可正常续跑。
