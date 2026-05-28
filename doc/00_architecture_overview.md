# 系统架构总览

> 本文是所有模块文档的索引与全局架构说明。
> 各模块的详细实现文档见本目录下的专项文件。

## 文档索引

| 文件 | 涵盖模块 | 内容 |
|------|----------|------|
| [01_pipeline_understand.md](01_pipeline_understand.md) | `understand.py` + v4.1 understand 子模块 | 理解流水线 10 步详解、MinuteChunk、叙事层级、数据流、断点续跑 |
| [02_models_schemas.md](02_models_schemas.md) | `schemas.py` | 所有 Pydantic 数据模型定义（含 v4.1 新增：VideoMeta 压缩字段、CharacterGallery、CharacterProfile、MinuteChunk） |
| [03_search_engine.md](03_search_engine.md) | `search.py` | 三层漏斗检索详解（Embedding + 关键词 + LLM Reranker） |
| [04_director_agent.md](04_director_agent.md) | `director.py` + `prompts.py` | Director Agent 短视频/长视频规划、证据填充、Prompt 设计 |
| [05_reviewer_agent.md](05_reviewer_agent.md) | `reviewer.py` | Reviewer Agent 三层校验（规则 + Grounding + LLM）、评分逻辑 |
| [06_render_engine.md](06_render_engine.md) | `engine.py` + `validator.py` + `ffmpeg_ops.py` | 渲染 5 步流水线、FFmpeg 操作封装 |
| [07_utils.md](07_utils.md) | `llm_client.py` + `ffmpeg_utils.py` + `logger.py` | LLM 客户端、FFmpeg 工具、日志系统 |
| [08_store_cli_config.md](08_store_cli_config.md) | `store.py` + `main.py` + `config.py` | 存储层、命令行接口、v4.1 配置、目录结构 |

---

## 端到端数据流

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         UNDERSTAND 阶段（v4.1 / 10步）                   │
│                                                                          │
│  video.mp4                                                               │
│    ├─[1  ingest]──→ original.* + compressed.mp4(按需) + meta.json        │
│    ├─[2  shot_detect]──→ scenes/scenes.json  ◄── 时间轴锚点              │
│    ├─[3  multi_keyframe]──→ scenes/keyframes/*_f0~f5.jpg                 │
│    ├─[4  face_cluster]──→ characters/face_clusters.json                  │
│    │                         + characters/char_XXX_gallery/              │
│    ├─[5  minute_chunk]──→ minute_chunks.json + character_profiles.json   │
│    │                         + transcripts/ocr/vision/audio/alignment    │
│    │                         + characters.json + speaker_map.json        │
│    ├─[6  beat_detect]──→ beats.json  (shot → beat 聚合)                  │
│    ├─[7  story_scene_detect]──→ story_scenes.json (beat → scene 聚合)    │
│    ├─[8  chapter_detect]──→ chapters.json (scene → chapter 聚合)         │
│    ├─[9  event_and_arc]──→ events.json + event_graph.json                │
│    │                         + character_arcs.json                       │
│    │                         + character_relations.json                  │
│    └─[10 final_build]──→ edit/narrative/recomposition signals            │
│                              + memory.json + index/                      │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         SEARCH 阶段                                      │
│                                                                          │
│  query + VideoMemory                                                     │
│    ├─ Layer 1: FAISS Embedding 粗召回 (top-50)                           │
│    ├─ Layer 2: 关键词精筛 + 合并去重 (top-20)                             │
│    └─ Layer 3: LLM Reranker 语义重排 (top-k)                             │
│         │                                                                │
│         └──→ SearchResult[] (含 matched_modalities + source_refs)        │
│                                                                          │
│  索引维度: 文本/向量/角色/事件/关系/情绪/剪辑信号/音频/章节                 │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         EDIT 阶段                                        │
│                                                                          │
│  Director Agent                                                          │
│    ├─ 构造候选信息 Prompt（含多模态融合文本 + 证据来源 + EditSignal）      │
│    ├─ LLM 生成 clips JSON                                                │
│    ├─ 解析 + 白名单校验 + 证据自动填充                                    │
│    └─ Reviewer Agent 审核                                                │
│         ├─ 规则校验                                                      │
│         ├─ Grounding 校验                                                │
│         └─ LLM 审核                                                      │
│              ├─ 通过 → 保存 EditPlan                                     │
│              └─ 不通过 → 反馈加入 prompt，重试                            │
│                                                                          │
│  长视频 (>30min): 分章节滑窗规划 → 合并                                   │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         RENDER 阶段                                      │
│                                                                          │
│  EditPlan                                                                │
│    ├─ 校验 (validator)                                                   │
│    ├─ 裁剪 (cut_clip_precise)                                            │
│    ├─ 变速/音量/淡入淡出                                                  │
│    ├─ 标准化 (normalize)                                                 │
│    ├─ 拼接 (concat)                                                      │
│    └─ BGM 混合 (可选)                                                    │
│         │                                                                │
│         └──→ output.mp4                                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 关键设计决策

### 1. 入库压缩降低理解成本

Step 1 会保留 `original.*` 供渲染使用，同时按需生成 `compressed.mp4` 供理解流水线使用。默认阈值是高度 `480`、帧率 `10fps`，这会显著降低后续关键帧处理、视频片段截取和多模态 API 调用成本。

### 2. 先切分，再构建角色脸谱

Shot 仍是整个系统的时间轴锚点。v4.1 在调用大模型前增加 `face_cluster`：

| 产物 | 用途 |
|------|------|
| `characters/face_clusters.json` | 保存角色聚类、代表脸来源、出现镜头、角色层级和聚类中心 |
| `characters/char_XXX_gallery/` | 每个主要/次要角色的正脸/轻微转头代表脸，用作 MinuteChunk 身份先验 |

该步骤先用 InsightFace 从 Step 3 的关键帧中提取人脸 bbox、pose/landmarks 与 embedding，并过滤低置信度、小脸、小裁剪图和明显侧脸，再用 DBSCAN 聚类。聚类后会先拆分疑似混簇（同关键帧冲突、簇半径过大），再合并疑似碎簇（簇中心相似、代表脸桥接相似、强单点桥接相似），以降低“不同人物混成一个角色”和“同一人物被拆成多个 gallery”的风险。过滤阈值使用关键帧短边比例加像素兜底，以适配不同分辨率。

`face_cluster` 只负责提供稳定的本地身份先验，不追求用传统模型解决所有跨造型/跨光照的同人归并。若后续大模型在 MinuteChunk 或角色档案更新中有充分语义证据证明两个 gallery 是同一人物，可以在更高层处理；当前 Step 4 不做这类语义聚合。

InsightFace 不可用时，该步骤返回空脸谱，MinuteChunk 会让 Gemini 自行识别人物。

### 3. MinuteChunk 替代六个独立理解步骤

v4.1 将 v3 的 `asr_windowed`、`vision`、`audio_analysis`、`character_deep`、`speaker_bind`、`multimodal_align` 合并到 `minute_chunk`。每个 chunk 使用视频片段、角色脸谱和动态角色档案作为输入，一次输出并回填；Step 3 关键帧保留给脸谱构建与后续多模态 RAG / 索引，不随 chunk 视频一起送入 Gemini：

| 回填文件 | 内容 |
|----------|------|
| `transcripts.json` | 已带 speaker 和 `character_id` 的台词 |
| `ocr.json` / `vision.json` | 逐 shot OCR 与画面摘要 |
| `audio_prosody.json` | 逐 shot 音乐、音效、沉默、语速、音量、语音情绪 |
| `multimodal_alignments.json` | 可见角色、说话角色、活跃模态和主导模态 |
| `characters.json` / `character_profiles.json` | 动态角色档案和下游兼容人物文件 |
| `speaker_map.json` | 从 speaker 标注直接派生的映射 |

角色出场回填以视觉证据为准：`characters_present` 只表示画面中真实可见且可识别的人物，不能用台词/旁白/剧情提及来补角色。`minute_chunk.py` 在回填时会同时使用 `local_shot_index` 与全局 `scene_index` 做防御，避免局部编号和全局编号混用造成 shot 错位；角色档案会保留 `appearance_changes` 历史，但“无”“无明显变化”“无法判断”等占位文本不会覆盖已有有效描述。

### 4. 层层聚合保持不变

v4.1 改的是底层理解方式，不改变高层叙事结构：

```
Shot → Beat → StoryScene → Chapter → EventGraph
```

每个层级仍可形成对应的 MemoryUnit，最终服务于检索、选材和 Reviewer Grounding。

### 5. 三类信号驱动剪辑决策

| 信号类型 | 说明 | 服务对象 |
|----------|------|----------|
| `EditSignal` | 8 维剪辑信号（hook/剧情/情绪/视觉/独立性/连续性/边界/剧透） | Director Agent 选材 |
| `NarrativeSignal` | 叙事弧位置/张力/信息密度/叙事功能 | 结构化叙事编排 |
| `RecompositionSignal` | 梗潜力/情感引用/平台适配/二创格式 | 二次创作/短视频分发 |

### 6. 证据驱动的 EditPlan

Director 不允许自造片段，每个 EditClip 必须有 `evidence_refs`。Reviewer 会检查时间、角色、事件覆盖、叙事结构和目标时长。

### 7. 容错与降级

| 场景 | 降级策略 |
|------|----------|
| 视频不需要压缩 | 直接使用原始视频作为 `storage_path` |
| InsightFace 未安装 | `face_cluster` 返回空脸谱，MinuteChunk 自行识别角色 |
| MinuteChunk 单个 chunk 解析失败 | 跳过该 chunk，后续为未覆盖 shot 补空 Vision/OCR/Audio |
| FAISS 不可用 | 跳过 Embedding 检索层 |
| Embedding API 不可用 | 跳过向量索引 |
| Beat 检测 LLM 失败 | 每 4 个 shot 一组默认分组 |
| StoryScene 检测 LLM 失败 | 每 3 个 beat 一组默认分组 |
| Chapter 检测 LLM 失败 | 每 3 个 StoryScene 一组 |
| 事件关系推理 LLM 失败 | 返回空边列表，仅保留事件节点 |
| 人物弧线 LLM 失败 | 跳过弧线，保留基础人物信息 |
| 剪辑信号 LLM 失败 | 跳过该批次，不阻塞流程 |
| LLM 审核失败 | 仅使用规则审核结果 |

### 8. 断点续跑

每步完成后在 `progress.json` 打标记。恢复时自动跳到第一个未完成的新步骤。

v4.1 通过 `_STEP_ALIASES` 映射旧步骤名：

| 旧步骤 | 新步骤 |
|--------|--------|
| `scene_detect` | `shot_detect` |
| `keyframe_extract` | `multi_keyframe` |
| `asr` / `asr_windowed` / `vision` / `audio_analysis` / `speaker_bind` / `multimodal_align` | `minute_chunk` |
| `character_deep` / `character` | `face_cluster` |
| `event_graph` / `event` / `character_arc` | `event_and_arc` |
| `edit_signal` / `build_memory` / `indexer` | `final_build` |
