# AI 长视频理解与多视角自动剪辑系统

一个纯 Python 实现的 AI 视频自动剪辑系统。当前 understand 阶段已升级为 **v4.1 的 10 步面向剪辑决策的理解流水线**：先对视频入库并按需压缩，再完成镜头切分、关键帧采样和角色脸谱构建，随后用 **MinuteChunk 分钟级融合理解** 一次性完成 ASR、画面、音频、角色标注和多模态对齐，最终构建 **Shot → Beat → StoryScene → Chapter → EventGraph** 多层叙事结构、三类剪辑信号和九维检索索引。

系统后半段由 **Director / Reviewer 双 Agent 闭环** 自动生成可验证的创意剪辑方案，并调用 FFmpeg 渲染输出成片。

## 核心特点

| 特点 | 说明 |
|------|------|
| **10 步 v4.1 理解流水线** | 从 v3 的 17 步收敛为 10 步，减少重复模型调用和中间编排成本 |
| **入库压缩** | 当高度 > 480p 或 fps > 10 时生成 `compressed.mp4` 供理解使用，渲染仍使用原始视频 |
| **角色脸谱优先** | Step 4 使用 InsightFace + DBSCAN 先构建 `CharacterGallery`，供后续 Gemini 识别角色身份 |
| **MinuteChunk 融合理解** | Step 5 将连续 shot 拼接为约 2-3 分钟 chunk，一次性完成 ASR、Vision、Audio、角色标注和对齐 |
| **多层叙事理解** | Shot → Beat → StoryScene → Chapter → EventGraph，从镜头到长视频章节全覆盖 |
| **三类剪辑信号** | EditSignal 8 维剪辑信号 + NarrativeSignal 叙事信号 + RecompositionSignal 二创信号 |
| **多层 VideoMemory** | Shot / Beat / StoryScene / Chapter 四层 MemoryUnit，融合音频与多模态对齐结果 |
| **九维检索索引** | 文本 + Embedding + 角色 + 事件 + 关系 + 情绪 + 剪辑信号 + 音频 + 章节 |
| **证据驱动剪辑** | EditClip 强制引用 `evidence_refs`，可携带 EditSignal、Beat、StoryScene 等层级引用 |
| **审核闭环** | Reviewer 进行规则校验、Grounding 校验和 LLM 审核 |
| **断点续跑** | 每步结果持久化为 JSON，`--resume` 根据 `progress.json` 继续执行，并映射旧步骤名 |

---

## 系统架构

```
用户视频
   │
   ▼
[10步 v4.1 理解流水线]
   ├─ original.* / compressed.mp4(按需)
   ├─ Shot → Beat → StoryScene → Chapter
   ├─ EventGraph + CharacterArc
   ├─ EditSignal / NarrativeSignal / RecompositionSignal
   ├─ VideoMemory 四层检索单元
   └─ 九维检索索引
   │
   ▼
Video Memory (JSON)
   │
用户需求 ──→ [Director Agent] ──多维检索──→ 候选片段 + 证据引用
                  │
                  ▼
            [Reviewer Agent]
              ├─ 规则校验
              ├─ Grounding 校验
              └─ LLM 审核
                  │
                  ▼
             EditPlan (JSON)
                  │
                  ▼
            [渲染引擎 FFmpeg]
                  │
                  ▼
             output.mp4
```

---

## 理解流水线（v4.1 / 10 步）

核心原则：**压缩降本 → 镜头锚定 → 脸谱先验 → 分钟级融合理解 → 层层聚合**。

| # | 步骤 | 模块 | 输出 | 说明 |
|---|------|------|------|------|
| 1 | `ingest` | `ingest.py` | `original.*` + 按需 `compressed.mp4` + `meta.json` | 入库、解析元信息；必要时压缩到 480p / 10fps |
| 2 | `shot_detect` | `scene_detect.py` | `scenes/scenes.json` | PySceneDetect 镜头边界检测 |
| 3 | `multi_keyframe` | `keyframe.py` | `scenes/keyframes/*.jpg` | 每个 shot 动态采样多帧关键帧 |
| 4 | `face_cluster` | `face_cluster.py` | `characters/face_clusters.json` + `char_XXX_gallery/` | 人脸检测、质量过滤、聚类、角色脸谱构建 |
| 5 | `minute_chunk` | `minute_chunk.py` | `minute_chunks.json` + 回填产物 | ASR、画面、音频、角色、speaker、对齐合并为分钟级融合理解 |
| 6 | `beat_detect` | `beat_detect.py` | `beats.json` | Shot → Beat 剧情节拍聚合 |
| 7 | `story_scene_detect` | `story_scene_detect.py` | `story_scenes.json` | Beat → StoryScene 聚合 |
| 8 | `chapter_detect` | `chapter_detect.py` | `chapters.json` | StoryScene → Chapter 长视频大段落聚合 |
| 9 | `event_and_arc` | `event.py` + `character_arc.py` | `events.json` + `event_graph.json` + `character_arcs.json` + `character_relations.json` | 事件图谱与人物弧线合并执行 |
| 10 | `final_build` | `edit_signal.py` + `memory_builder.py` + `indexer.py` | 三类信号 + `memory.json` + `index/` | 计算剪辑信号、构建 VideoMemory 和检索索引 |

### v3 → v4.1 步骤对比

| v3 (17步) | v4.1 (10步) | 说明 |
|-----------|-------------|------|
| 1. ingest | 1. ingest + 压缩 | 分辨率 > 480 → 480，fps > 10 → 10 |
| 2. shot_detect | 2. shot_detect | 不变 |
| 3. multi_keyframe | 3. keyframe | 不变 |
| 4. asr_windowed | 并入 Step 5 | ASR 并入 MinuteChunk |
| 5. vision | 并入 Step 5 | 逐 shot 画面理解由 MinuteChunk 回填 |
| 6. audio_analysis | 并入 Step 5 | 音频韵律由 MinuteChunk 随视频片段分析 |
| 7. character_deep | 拆分到 Step 4 + Step 5 | 人脸检测聚类到 Step 4，描述/命名/动态档案到 Step 5 |
| 8. speaker_bind | 并入 Step 5 | Gemini 直接用角色脸谱标注说话人 |
| 9. multimodal_align | 并入 Step 5 | 对齐结果随 chunk 回填 |
| 10. beat_detect | 6. beat_detect | 使用 Step 5 回填的台词、画面和角色信息 |
| 11. story_scene | 7. story_scene | 不变 |
| 12. chapter | 8. chapter | 不变 |
| 13. event_graph | 合并到 Step 9 | Event + CharacterArc 合并执行 |
| 14. character_arc | 合并到 Step 9 | 与事件图谱共享角色与事件上下文 |
| 15. edit_signal | 合并到 Step 10 | 与 build_memory + indexer 合并 |
| 16. build_memory | 合并到 Step 10 | 统一最终构建 |
| 17. indexer | 合并到 Step 10 | 统一最终构建 |

### 叙事层级结构

```
Shot (镜头)
 └─ Beat (节拍)
     └─ StoryScene (故事场景)
         └─ Chapter (长视频大段落)
             └─ EventGraph (事件节点 + 关系边)
```

### 数据流

```
video.mp4
   │
   ├─[1]─→ original.* + compressed.mp4(按需) + meta.json
   ├─[2]─→ scenes/scenes.json
   ├─[3]─→ scenes/keyframes/
   │
   ├─[4]─→ characters/
   │       ├── face_clusters.json
   │       ├── char_000_gallery/
   │       └── char_001_gallery/
   │
   ├─[5]─→ minute_chunks.json + character_profiles.json
   │       ├── transcripts.json
   │       ├── ocr.json + vision.json
   │       ├── audio_prosody.json
   │       ├── multimodal_alignments.json
   │       ├── characters.json
   │       └── speaker_map.json
   │
   ├─[6]─→ beats.json
   ├─[7]─→ story_scenes.json
   ├─[8]─→ chapters.json
   ├─[9]─→ events.json + event_graph.json + character_arcs.json + character_relations.json
   └─[10]→ edit_signals.json + narrative_signals.json + recomposition_signals.json
           + memory.json + index/
```

---

## MinuteChunk 融合理解

`pipeline/minute_chunk.py` 是 v4.1 的核心模块。它以 shot 边界为基础，把连续镜头拼接为约 150 秒的 chunk，截取视频片段并附带角色脸谱和前序角色档案，让 Gemini 一次输出。Step 3 关键帧保留给脸谱构建和后续多模态 RAG / 索引，不随 chunk 视频一起送入 Gemini：

- ASR 转录：逐句台词、时间戳、speaker、`character_id`
- 逐 shot 视觉摘要：描述、物体、情绪、场景类型、镜头运动、景别、OCR
- 逐 shot 音频特征：音乐、音效、沉默占比、语速、音量峰值、语音情绪
- 角色动态更新：新称呼、外观变化、关键行为、非人类实体
- 跨 shot 分析：叙事连续性、情绪弧线、建议的 beat 分组

处理完成后，chunk 结果会按 shot 时间戳回填为传统散文件，兼容下游现有的 Beat、StoryScene、Chapter、Event 和 Memory 构建逻辑。

角色回填采用保守策略：`characters_present` 只代表画面中真实可见且有足够视觉证据识别的人物；仅在台词、旁白或剧情中被提到的人物不会写入 `visible_characters`。回填时会同时检查 `local_shot_index` 和全局 `scene_index`，降低 LLM 把局部编号/全局编号混用导致的 shot 错位风险；如果模型漏掉某些 shot，会写入占位 `vision` / `ocr` / `audio` / `multimodal_alignment`，避免散文件断档。角色档案会保留 `appearance_changes` 历史，但“无”“无明显变化”“无法判断”等占位文本不会覆盖已有有效描述。

---

## 三类剪辑信号

### EditSignal（剪辑信号）

为 beat / story_scene / 重要 shot 计算 8 个面向剪辑决策的信号：

| 信号 | 含义 | 用途举例 |
|------|------|---------|
| `hook_score` | 作为开头钩子的适合度 | 选择视频开头 |
| `plot_importance` | 对整体叙事的贡献度 | 保留核心剧情 |
| `emotional_intensity` | 情绪表达强度 | 选择高潮片段 |
| `visual_impact` | 视觉冲击力 | 预告片选材 |
| `independence_score` | 片段独立性 | 可独立剪出的片段 |
| `continuity_dependency` | 连续性依赖 | 避免断裂感 |
| `boundary_quality` | 剪辑边界质量 | 选择自然剪辑点 |
| `spoiler_level` | 剧透程度 | 控制预告片剧透 |

### NarrativeSignal（叙事信号）

面向 beat / story_scene 评估 `arc_position`、`tension_level`、`information_density`、`character_focus`、`narrative_function` 和 `theme_relevance`，用于判断片段在整体叙事弧中的位置和功能。

### RecompositionSignal（二创信号）

面向重要 beat 评估 `meme_potential`、`emotional_quotability`、`context_freedom`、`remix_flexibility`、`platform_fit` 和 `suggested_formats`，用于短视频二创、名场面、反应类或混剪类选材。

---

## 三层漏斗检索

```
用户查询
    │
    ▼
Layer 1: Embedding 粗召回
    │   FAISS 向量近似检索，默认 top-50
    ▼
Layer 2: 关键词精筛
    │   台词 / 画面 / 事件 / 文本索引四路检索并去重
    ▼
Layer 3: LLM Reranker
    │   Gemini 语义重排并填充上下文
    ▼
SearchResult top-k
```

`pipeline/indexer.py` 会在 Step 10 产出文本、向量、角色、事件、关系、情绪、剪辑信号、音频、章节等索引；当前 `memory/search.py` 主链路主要消费文本索引、向量索引和 VideoMemory 中的台词/画面/事件数据。

---

## 核心数据模型

### MemoryUnit（Shot 级检索原子）

```json
{
  "scene_index": 5,
  "start_time": 120.0,
  "end_time": 135.5,
  "beat_index": 3,
  "story_scene_index": 1,
  "chapter_index": 0,
  "transcripts": [{"text": "你不要走", "character_id": "char_000", "cross_shot": false}],
  "vision": {"description": "女主在雨中追赶男主", "action_description": "从站立到奔跑", "mood": "悲伤"},
  "audio_prosody": {"has_music": true, "music_mood": "melancholic", "speech_emotion": "sad"},
  "alignment": {"dominant_modality": "speech", "alignment_confidence": 0.8},
  "edit_signal": {"hook_score": 0.85, "emotional_intensity": 0.9, "independence_score": 0.6},
  "combined_text": "台词: 你不要走 | 画面: 女主在雨中追赶男主 | 音频: 音乐:melancholic, 语音情绪:sad | ...",
  "embedding": [0.123, -0.456]
}
```

### v4.1 新增模型

| 模型 | 说明 |
|------|------|
| `CharacterGallery` | Step 4 的角色脸谱，保存代表脸路径、来源 shot/关键帧、聚类中心、出场镜头和角色层级 |
| `CharacterProfile` | Step 5 的动态角色档案，随 chunk 更新名称、外观变化和关键行为 |
| `MinuteChunk` | 分钟级理解单元，记录 chunk 时间范围、shot 列表、原始理解结果和 suggested beats |
| `VideoMeta` 压缩字段 | `compressed_path`、`is_compressed`、`original_height/fps`、`compressed_height/fps` |

---

## 项目结构

```
movie-agent/
├── main.py                  # CLI 主入口
├── config.py                # 全局配置
├── requirements.txt         # 依赖
│
├── models/
│   └── schemas.py           # Pydantic 数据模型（v4.1 含压缩/脸谱/MinuteChunk）
│
├── pipeline/                # 理解流水线（v4.1: 10步）
│   ├── understand.py        # 流水线编排器
│   ├── ingest.py            # 入库 + 压缩
│   ├── scene_detect.py      # 镜头切分
│   ├── keyframe.py          # 多帧关键帧采样
│   ├── face_cluster.py      # 人脸聚类 + 角色脸谱
│   ├── minute_chunk.py      # 分钟级融合理解
│   ├── beat_detect.py       # 剧情节拍检测
│   ├── story_scene_detect.py # 故事场景检测
│   ├── chapter_detect.py    # 长视频章节检测
│   ├── event.py             # 事件图谱构建
│   ├── character_arc.py     # 人物弧线 + 关系图
│   ├── edit_signal.py       # 三类剪辑信号计算
│   ├── memory_builder.py    # 四层 MemoryUnit 构建
│   ├── indexer.py           # 九维检索索引构建
│   ├── asr.py               # 旧独立 ASR 模块，v4.1 主链路已并入 minute_chunk
│   ├── vision.py            # 旧独立 Vision 模块，v4.1 主链路已并入 minute_chunk
│   ├── audio_analysis.py    # 旧独立音频分析模块，v4.1 主链路已并入 minute_chunk
│   ├── character.py         # 旧深度人物模块，v4.1 拆分为 face_cluster + minute_chunk
│   ├── speaker_bind.py      # 旧 speaker 绑定模块，v4.1 主链路已并入 minute_chunk
│   └── multimodal_align.py  # 旧多模态对齐模块，v4.1 主链路已并入 minute_chunk
│
├── memory/
│   ├── store.py             # Video Memory 读写
│   └── search.py            # 三层漏斗检索
│
├── agents/
│   ├── director.py          # Director Agent
│   ├── reviewer.py          # Reviewer Agent
│   └── prompts.py           # Prompt 模板
│
├── render/
│   ├── engine.py            # 渲染主流程
│   ├── validator.py         # EditPlan 校验器
│   └── ffmpeg_ops.py        # FFmpeg 原子操作
│
└── doc/                     # 模块文档
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# 可选：安装 FAISS 以获得向量检索能力
pip install faiss-cpu
```

### 2. 配置 `.env`

```bash
LLM_API_KEY="your_api_key"
LLM_API_BASE="https://your-api-endpoint/api/v1"
LLM_MODEL="turing/gemini-3.1-flash-lite-latest"
EMBEDDING_MODEL="text-embedding-004"
```

### 3. 使用方式

```bash
# 一键全流程：理解 → 生成 EditPlan → 渲染
python main.py auto --video movie.mp4 --prompt "制作一个3分钟的精彩片段合集"

# 分步执行
python main.py understand --video movie.mp4              # 理解视频（10步 v4.1）
python main.py search --video-id xxx --query "打斗场面"  # 搜索
python main.py edit --video-id xxx --prompt "爱情线剪辑" # 生成 EditPlan
python main.py render --plan-id plan_xxx                 # 渲染成片

# 断点续跑
python main.py understand --video-id xxx --resume
```

---

## 配置项

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LLM_API_KEY` | — | LLM API 密钥 |
| `LLM_API_BASE` | `https://live-turing.cn.llm.tcljd.com/api/v1` | LLM API 基础 URL |
| `LLM_MODEL` | `turing/gemini-3.1-flash-lite-latest` | LLM 模型名称 |
| `LLM_TIMEOUT` | `300` | LLM 请求超时秒数 |
| `EMBEDDING_MODEL` | `text-embedding-004` | Embedding 模型名称 |
| `FFMPEG_PATH` / `FFPROBE_PATH` | `ffmpeg` / `ffprobe` | FFmpeg 工具路径 |
| `COMPRESS_MAX_HEIGHT` | `480` | 高于此值时压缩分辨率 |
| `COMPRESS_MAX_FPS` | `10` | 高于此值时降帧率 |
| `SCENE_DETECT_THRESHOLD` | `27.0` | 镜头切分灵敏度 |
| `SCENE_DETECT_MIN_LEN` | `1.0` | 最短镜头秒数 |
| `MULTI_KEYFRAME_MAX` | `6` | 每个 shot 最大采样帧数 |
| `CHUNK_TARGET_DURATION` | `150` | MinuteChunk 目标时长秒数 |
| `CHUNK_MERGE_THRESHOLD` | `30` | 尾段低于此值时合并到前一个 chunk |
| `CHUNK_MIN_DURATION` / `CHUNK_MAX_DURATION` | `90` / `210` | 已配置，当前 chunk 构建逻辑暂未强制使用 |
| `FACE_GALLERY_MAX` / `FACE_GALLERY_MIN` | `6` / `3` | 每个角色脸谱图片数量范围 |
| `FACE_CLUSTER_EPS` | `0.42` | DBSCAN 聚类 eps |
| `FACE_CLUSTER_MERGE_STRONG_LINK_SIM` | `0.82` | 单对代表脸极高相似时的碎簇合并阈值 |
| `FACE_MIN_CROP_RATIO` / `FACE_MIN_CROP_PIXEL_FLOOR` | `0.08` / `48` | gallery 裁剪图短边过滤阈值 |
| `FACE_REJECT_SIDE_FACE` | `true` | 是否过滤侧脸，优先保留正脸/轻微转头作为身份库 |
| `FACE_PASSERBY_MIN` | `3` | 低于此出场次数视为路人 |
| `FACE_DETECT_DEVICE` | `auto` | InsightFace 推理设备：`auto` 优先 CUDA，`cuda` 指定 GPU，`cpu` 强制 CPU |
| `FACE_DETECT_GPU_ID` | `auto` | InsightFace 使用的 CUDA 设备；`auto` 选择显存占用最低的 GPU，也可指定 `0`-`7` |
| `DATA_DIR` | `./data` | 数据存储根目录 |

---

## 当前实现注意事项

- `understand --resume --video-id xxx` 依赖 `progress.json` 判断断点；如果进度文件缺失，当前代码不会自动从散文件推断完成步骤。
- v4.1 通过 `_STEP_ALIASES` 将旧步骤映射到新步骤：例如 `asr_windowed` / `vision` / `audio_analysis` / `speaker_bind` / `multimodal_align` 映射到 `minute_chunk`，`edit_signal` / `build_memory` / `indexer` 映射到 `final_build`。
- Step 5 会保存 `MinuteChunk.suggested_beats`，但当前 `beat_detect.py` 主流程仍基于回填后的台词、画面和人物信息重新让 LLM 分组；`suggested_beats` 更像后续优化入口。
- Step 5 的 `characters.json.appearance_scenes` 来自 Gemini 回填的可见角色和 Step 4 gallery 出场镜头的合并。调试角色出场异常时，应优先检查 `multimodal_alignments.json.visible_characters` 是否被误标。
- Step 9 的人物关系分析会读取 `character_profiles.json` 的有效别名、外观变化和关键行为，并根据 `characters.json.appearance_scenes` 计算共现；如果 `character_arcs.json` / `character_relations.json` 已存在，会直接加载缓存，不会自动重算。
- `face_cluster.py` 在 InsightFace 未安装时会跳过脸谱构建，由 MinuteChunk 让 Gemini 自行识别人物；InsightFace 默认 `FACE_DETECT_DEVICE=auto`，检测到 `CUDAExecutionProvider` 时使用 GPU，`FACE_DETECT_GPU_ID=auto` 会选择显存占用最低的 CUDA 设备，否则回退 CPU。
- Step 4 的人脸聚类是传统视觉模型的身份先验：会保守拆分混簇、合并高相似碎簇，并过滤小脸/侧脸，但仍可能把同一个人物拆成多个 gallery。跨 gallery 的语义归并留给后续大模型理解阶段基于剧情、台词和外观证据处理，当前不在 `face_cluster` 中实现。
- Step 10 之前只有散文件；完整四层 MemoryUnit、embedding 和索引要等 `final_build` 完成后才会出现在 `memory.json` / `index/`。

---

## 与旧版的对比

| 维度 | v1 | v2 | v3 | v4.1（当前） |
|------|----|----|----|-------------|
| **流程步骤** | 10 步 | 14 步 | 17 步 | 10 步 |
| **理解策略** | 按 shot 独立分析 | 多帧 + 层级聚合 | 音频/对齐/章节增强 | MinuteChunk 融合 ASR/Vision/Audio/角色/对齐 |
| **叙事结构** | 扁平 Shot 列表 | Shot → Beat → StoryScene → EventGraph | Shot → Beat → StoryScene → Chapter → EventGraph | 同 v3 |
| **关键帧/画面字段** | 每 shot 1 帧 | 每 shot 1-6 帧 | 同 v2，并增加镜头运动、景别、人物互动等字段 | 同 v3，用于脸谱构建与多模态 RAG / 索引 |
| **ASR / Vision / Audio** | 多为独立处理 | 独立增强 | 独立模块分步执行 | 并入 MinuteChunk，一次多模态理解后回填 |
| **人物** | 基础聚类 + 描述 | 弧线/关系图/重要性 | 进入多模态对齐 | Step 4 脸谱 + Step 5 动态角色档案 + Step 9 弧线 |
| **剪辑信号** | 无 | 8 维 EditSignal | 三类信号 | 同 v3，合并到 Step 10 |
| **MemoryUnit** | 单层 shot | 三层 shot / beat / story_scene | 四层 shot / beat / story_scene / chapter | 同 v3，合并到 Step 10 构建 |
| **检索索引** | 文本 + Embedding | 七维索引 | 九维索引 | 同 v3，合并到 Step 10 构建 |
| **调用成本** | 较低但理解浅 | 中等 | 约 257 次/30min 估算 | 约 30-33 次/30min 估算 |

---

## 外部依赖

| 依赖 | 用途 | 必须 |
|------|------|------|
| `pydantic` | 数据模型 | 是 |
| `requests` | API 调用 | 是 |
| `python-dotenv` | 环境变量 | 是 |
| `scenedetect[opencv]` | 镜头切分 | 是 |
| `opencv-python` | 图像处理、人脸裁剪 | 是 |
| `numpy` | 数值计算 | 是 |
| `scikit-learn` | DBSCAN 人脸聚类 | 是 |
| `insightface` | 人脸检测 | 可选 |
| `onnxruntime` / `onnxruntime-gpu` | InsightFace 后端；GPU 加速需安装匹配 CUDA 的 `onnxruntime-gpu` | 随 insightface / 可选 GPU |
| `faiss-cpu` | 向量检索索引 | 可选 |
| FFmpeg / FFprobe | 视频处理、压缩、片段截取 | 是 |

---

## 许可证

本项目仅供学习和研究使用。
