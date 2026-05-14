# AI 长视频理解与多视角自动剪辑系统

一个纯 Python 实现的 AI 视频自动剪辑系统。通过 **14 步面向剪辑决策的理解流水线** 全面分析长视频内容，构建 **Shot → Beat → StoryScene → EventGraph** 多层叙事结构与 **EditSignal 剪辑信号**，利用 **Director / Reviewer 双 Agent 闭环** 自动生成可验证的创意剪辑方案，最后调用 FFmpeg 渲染输出成片。

## 核心特点

| 特点 | 说明 |
|------|------|
| 🎬 **多层叙事理解** | Shot → Beat → StoryScene → EventGraph，四级叙事结构从镜头到剧情全覆盖 |
| 🖼️ **多帧画面分析** | 每个 shot 按时长动态采样 1-6 帧，捕捉动作、表情、道具变化 |
| 🎙️ **长窗口 ASR** | 按 5 分钟窗口转写，支持跨镜头台词标记和对白/旁白/画外音分类 |
| 🧑‍🤝‍🧑 **深度人物分析** | 人物弧线、关系图谱、重要性评分、台词统计、共现矩阵 |
| 🕸️ **事件图谱** | 从简单事件列表升级为因果/铺垫/反转/冲突升级关系图 |
| ✂️ **EditSignal** | 8 维剪辑信号（hook/剧情/情绪/视觉/独立性/连续性/边界/剧透）指导自动剪辑 |
| 📦 **多层 VideoMemory** | Shot/Beat/StoryScene 三层 MemoryUnit，每层携带 EditSignal |
| 🔍 **七维检索索引** | 文本 + Embedding + 角色 + 事件 + 关系 + 情绪 + 剪辑信号 |
| ✅ **证据驱动剪辑** | EditClip 强制引用 `evidence_refs`，含 EditSignal 参考 |
| 🔄 **审核闭环** | Reviewer 进行 Grounding 校验（时间精度/角色一致性/事件覆盖率） |
| 💾 **断点续跑** | 每步结果持久化为 JSON，中断后 `--resume` 可从断点继续，兼容旧进度 |
| 🤖 **全 Gemini 驱动** | 统一使用 Gemini API 的多模态能力（音频/图片/文本） |

---

## 系统架构

```
用户视频 ──→ [14步理解流水线] ──→ Video Memory (JSON)
               多层叙事结构 │              │
               ┌───────────────────────────┤
               │  Shot → Beat → StoryScene │
               │  EventGraph (因果/反转)    │
               │  EditSignal (8维剪辑信号)  │
               │  MemoryUnit × 3层          │
               │  + 七维检索索引             │
               └───────────┬───────────────┘
                           │
用户需求 ──→ [Director Agent] ──多维检索──→ 候选片段 + EditSignal
                  │                        │
                  │◄── 证据引用 ────────────┘
                  ▼
            [Reviewer Agent]
              ├─ 规则校验
              ├─ Grounding 校验（证据链/时间精度/角色一致性）
              └─ LLM 审核
                  │
                  ▼ (最多3轮修订)
             EditPlan (JSON)
               含 evidence_refs + edit_signal_ref
                  │
                  ▼
            [渲染引擎 (FFmpeg)]
                  │
                  ▼
             成片 output.mp4
```

---

## 理解流水线（14 步）

核心原则：**"先切后提"** → **"层层聚合"**  
先完成镜头切分，再逐步构建更高层叙事单元（Beat → StoryScene → EventGraph），最后计算剪辑信号。

| # | 步骤 | 模块 | 输出 | 说明 |
|---|------|------|------|------|
| 1 | `ingest` | `ingest.py` | `meta.json` | 入库 + ffprobe 解析元信息 |
| 2 | `shot_detect` | `scene_detect.py` | `scenes.json` | PySceneDetect 镜头边界检测 |
| 3 | `multi_keyframe` | `keyframe.py` | `keyframes/*.jpg` | **多帧采样**（1-6帧/shot，按时长动态） |
| 4 | `asr_windowed` | `asr.py` | `transcripts.json` | **长窗口ASR**（5分钟窗口转写 + 回填shot） |
| 5 | `vision` | `vision.py` | `ocr.json` + `vision.json` | **多帧画面理解**（动作/表情/道具变化） |
| 6 | `character_deep` | `character.py` | `characters.json` | **深度人物分析**（共现/首末出场/重要性） |
| 7 | `speaker_bind` | `speaker_bind.py` | `speaker_map.json` | Speaker ↔ Character 绑定 |
| 8 | `beat_detect` | `beat_detect.py` | `beats.json` | **新增** 剧情节拍检测（shot → beat） |
| 9 | `story_scene_detect` | `story_scene_detect.py` | `story_scenes.json` | **新增** 故事场景检测（beat → story_scene） |
| 10 | `event_graph` | `event.py` | `events.json` + `event_graph.json` | **事件图谱**（含因果/铺垫/反转关系） |
| 11 | `character_arc` | `character_arc.py` | `character_arcs.json` + `character_relations.json` | **新增** 人物弧线 + 关系图 |
| 12 | `edit_signal` | `edit_signal.py` | `edit_signals.json` | **新增** 8维剪辑信号计算 |
| 13 | `build_memory` | `memory_builder.py` | `memory.json` | 三层 MemoryUnit + 角色判定 |
| 14 | `indexer` | `indexer.py` | `index/*` | 七维检索索引构建 |

### 叙事层级结构

```
Shot (镜头)           ← 最小视觉单元，由 scene_detect 切分
 └─ Beat (节拍)       ← 2-8 个连续 shot 组成的叙事微单元
     └─ StoryScene    ← 2-6 个连续 beat 组成的完整叙事场景
         └─ EventGraph ← 事件节点 + 因果/铺垫/反转关系边
```

### 数据流

```
video.mp4
   │
   ├─[1]─→ meta.json
   ├─[2]─→ scenes/scenes.json  ←── 时间轴锚点
   ├─[3]─→ scenes/keyframes/scene_XXXX_f0.jpg ~ f5.jpg  (多帧)
   ├─[4]─→ transcripts.json  (长窗口ASR + 回填scene_index)
   ├─[5]─→ ocr.json + vision.json  (含 action_description)
   ├─[6]─→ characters.json  (CharacterDeep: 含共现/重要性)
   ├─[7]─→ speaker_map.json
   ├─[8]─→ beats.json  (shot → beat 聚合)
   ├─[9]─→ story_scenes.json  (beat → story_scene 聚合)
   ├─[10]→ events.json + event_graph.json  (含关系边)
   ├─[11]→ character_arcs.json + character_relations.json
   ├─[12]→ edit_signals.json  (8维剪辑信号)
   ├─[13]→ memory.json  (三层MemoryUnit汇总)
   └─[14]→ index/
       ├── search_index.json      (文本索引)
       ├── faiss.index            (向量索引)
       ├── id_map.json            (FAISS映射)
       ├── character_index.json   (角色索引)
       ├── event_index.json       (事件索引)
       ├── relation_index.json    (关系索引)
       ├── emotion_index.json     (情绪索引)
       └── edit_signal_index.json (剪辑信号索引)
```

---

## EditSignal（剪辑信号）

每个 shot / beat / story_scene 计算 8 个面向剪辑决策的信号：

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

---

## 三层漏斗检索

```
用户查询
    │
    ▼
┌──────────────────────┐
│ Layer 1: Embedding   │  FAISS 向量近似检索
│ 粗召回 top-50        │  (需要 faiss-cpu)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Layer 2: 关键词精筛   │  台词/画面/事件/索引
│ → 合并 + 去重 top-20  │  多模态命中记录
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Layer 3: LLM Reranker│  Gemini 语义重排
│ → 最终 top-k 输出     │  + 上下文填充
└──────────────────────┘
```

---

## 证据驱动的 Agent 闭环

### Director Agent

- **候选白名单**：`source_scene_index` 必须来自检索结果
- **证据引用**：每个 EditClip 自动填充 `evidence_refs` / `matched_transcript` / `matched_vision`
- **v2 新增**：EditClip 可引用 `edit_signal_ref` / `source_beat_index` / `source_story_scene_index`
- **长视频滑窗**：超过 30 分钟的视频，按事件分章节独立规划

### Reviewer Agent Grounding 校验

| 校验项 | 说明 |
|--------|------|
| evidence_refs 非空 | 每个 clip 必须有证据来源 |
| 时间精度 | source_start/end 与 MemoryUnit 时间偏差 ≤ ±0.5s |
| 角色一致性 | clip 中的 characters 必须在对应 MemoryUnit 中出现 |
| 事件覆盖率 | 高重要性事件（importance ≥ 7）是否被覆盖 |
| 叙事结构 | 包含 hook + climax/resolution |
| 时长偏差 | 在目标时长 ±15% 范围内 |

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
  "transcripts": [{"text": "你不要走", "character_id": "char_000", "cross_shot": false}],
  "vision": {"description": "女主在雨中追赶男主", "action_description": "从站立到奔跑", "mood": "悲伤"},
  "edit_signal": {"hook_score": 0.85, "emotional_intensity": 0.9, "independence_score": 0.6},
  "combined_text": "台词: 你不要走 | 画面: 女主在雨中追赶男主 | 动作: 从站立到奔跑 | ...",
  "embedding": [0.123, -0.456, ...]
}
```

### Beat（叙事节拍）

```json
{
  "beat_index": 3,
  "start_time": 115.0,
  "end_time": 145.0,
  "shot_indices": [4, 5, 6],
  "beat_type": "confrontation",
  "description": "男女主角在雨中追逐对峙",
  "emotion": "悲伤",
  "intensity": 0.85
}
```

### CharacterDeep（深度人物）

```json
{
  "character_id": "char_000",
  "display_name": "女主角",
  "role": "female_lead",
  "importance_score": 0.82,
  "first_appearance": 15.0,
  "last_appearance": 890.0,
  "dialogue_count": 47,
  "arc": {"arc_type": "growth", "arc_description": "从怯懦到勇敢"},
  "co_appearing_characters": ["char_001", "char_002"]
}
```

---

## 项目结构

```
movie-agent/
├── main.py                  # CLI 主入口
├── config.py                # 全局配置
├── .env                     # 环境变量
├── requirements.txt         # 依赖
│
├── models/
│   └── schemas.py           # Pydantic 数据模型（v2: 含 Shot/Beat/StoryScene/EventGraph/EditSignal）
│
├── pipeline/                # 理解流水线（14步）
│   ├── understand.py        # 流水线编排器
│   ├── ingest.py            # 入库
│   ├── scene_detect.py      # 镜头切分（PySceneDetect）
│   ├── keyframe.py          # 多帧关键帧采样
│   ├── asr.py               # 长窗口 ASR（5分钟窗口 + 回填）
│   ├── vision.py            # 多帧画面理解（OCR + 动作/表情/道具）
│   ├── character.py         # 深度人物分析（InsightFace + Gemini）
│   ├── speaker_bind.py      # Speaker ↔ Character 绑定
│   ├── beat_detect.py       # [新增] 剧情节拍检测
│   ├── story_scene_detect.py # [新增] 故事场景检测
│   ├── event.py             # 事件图谱构建
│   ├── character_arc.py     # [新增] 人物弧线 + 关系图
│   ├── edit_signal.py       # [新增] 剪辑信号计算
│   ├── memory_builder.py    # 三层 MemoryUnit 构建 + 角色判定
│   ├── indexer.py           # 七维检索索引构建
│   └── audio.py             # [已弃用] 整体音频提取
│
├── memory/                  # 存储 & 检索
│   ├── store.py             # Video Memory 读写（v2 多层结构）
│   └── search.py            # 三层漏斗检索
│
├── agents/                  # AI Agent
│   ├── director.py          # Director Agent（含长视频滑窗）
│   ├── reviewer.py          # Reviewer Agent（含 Grounding 校验）
│   └── prompts.py           # Prompt 模板
│
├── render/                  # 渲染引擎
│   ├── engine.py            # 渲染主流程
│   ├── validator.py         # EditPlan 校验器
│   └── ffmpeg_ops.py        # FFmpeg 原子操作
│
├── utils/                   # 工具库
│   ├── llm_client.py        # LLM 客户端（OpenAI 兼容）
│   ├── ffmpeg_utils.py      # FFmpeg 工具函数
│   └── logger.py            # 日志
│
├── doc/                     # 文档
│   ├── 00_architecture_overview.md
│   ├── 01_pipeline_understand.md
│   ├── 02_models_schemas.md
│   └── ...
│
└── data/                    # 运行时数据（自动生成）
    ├── videos/{video_id}/
    │   ├── meta.json, scenes/, transcripts.json, ...
    │   ├── beats.json, story_scenes.json, event_graph.json
    │   ├── character_arcs.json, character_relations.json
    │   ├── edit_signals.json, memory.json
    │   └── index/ (7种索引文件)
    ├── editplans/
    └── renders/
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
EMBEDDING_MODEL="turing/text-embedding-3-small"
```

### 3. 使用方式

```bash
# 一键全流程：理解 → 生成 EditPlan → 渲染
python main.py auto --video movie.mp4 --prompt "制作一个3分钟的精彩片段合集"

# 分步执行
python main.py understand --video movie.mp4           # 理解视频（14步）
python main.py search --video-id xxx --query "打斗场面"  # 搜索
python main.py edit --video-id xxx --prompt "爱情线剪辑" # 生成 EditPlan
python main.py render --plan-id plan_xxx               # 渲染成片

# 断点续跑（兼容 v1 旧进度）
python main.py understand --video-id xxx --resume
```

---

## 配置项

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LLM_API_KEY` | — | LLM API 密钥 |
| `LLM_API_BASE` | — | LLM API 基础 URL |
| `LLM_MODEL` | `turing/gemini-3.1-flash-lite-latest` | LLM 模型名称 |
| `EMBEDDING_MODEL` | `text-embedding-004` | Embedding 模型名称 |
| `SCENE_DETECT_THRESHOLD` | `27.0` | 镜头切分灵敏度 |
| `ASR_CHUNK_DURATION` | `600` | 超长音频内部切分长度（秒） |
| `ASR_WINDOW_DURATION` | `300` | 长窗口 ASR 窗口大小（秒） |
| `MULTI_KEYFRAME_MAX` | `6` | 每个 shot 最大采样帧数 |
| `FFMPEG_PATH` | `ffmpeg` | FFmpeg 路径 |
| `DATA_DIR` | `./data` | 数据存储根目录 |

---

## 与 v1 的对比

| 维度 | v1 | v2 |
|------|-----|-----|
| **流程步骤** | 10 步 | 14 步 |
| **叙事结构** | 扁平 Shot 列表 | Shot → Beat → StoryScene → EventGraph 四级层级 |
| **关键帧** | 每 shot 1 帧 | 每 shot 1-6 帧（按时长动态采样） |
| **ASR** | 按 shot 段提取 | 长窗口（5min）转写 + 回填，支持跨镜头/旁白 |
| **画面分析** | 单帧描述 | 多帧对比（动作/表情/道具变化） |
| **人物** | 基础聚类 + 描述 | 弧线/关系图/重要性/共现矩阵/台词统计 |
| **事件** | 扁平事件列表 | 事件图谱（因果/铺垫/反转/冲突升级关系） |
| **剪辑信号** | 无 | 8 维 EditSignal（hook/剧情/情绪/视觉/...） |
| **MemoryUnit** | 单层（shot 级） | 三层（shot / beat / story_scene） |
| **检索索引** | 文本 + Embedding | 七维（+角色/事件/关系/情绪/剪辑信号） |
| **兼容性** | — | Scene=Shot 别名，旧进度自动映射 |

---

## 外部依赖

| 依赖 | 用途 | 必须 |
|------|------|------|
| `pydantic` | 数据模型 | ✅ |
| `requests` | API 调用 | ✅ |
| `python-dotenv` | 环境变量 | ✅ |
| `scenedetect[opencv]` | 镜头切分 | ✅ |
| `opencv-python` | 图像处理 | ✅ |
| `numpy` | 数值计算 | ✅ |
| `scikit-learn` | 人脸聚类 | ✅ |
| `insightface` | 人脸检测 | ⚠️ 可选（无则用 Gemini Vision 替代） |
| `onnxruntime` | InsightFace 后端 | ⚠️ 随 insightface |
| `faiss-cpu` | 向量检索索引 | ⚠️ 可选（无则跳过 Embedding 检索层） |
| FFmpeg | 视频处理 | ✅ 系统级依赖 |

---

## 许可证

本项目仅供学习和研究使用。
