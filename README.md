# AI 长视频理解与多视角自动剪辑系统

一个纯 Python 实现的 AI 视频自动剪辑系统。通过 **10 步证据驱动的理解流水线** 全面分析长视频内容，构建以 **MemoryUnit** 为核心的多模态融合记忆，并利用 **Director / Reviewer 双 Agent 闭环** 自动生成可验证的创意剪辑方案，最后调用 FFmpeg 渲染输出成片。

## 核心特点

| 特点 | 说明 |
|------|------|
| 🧠 **深度视频理解** | 镜头切分 → ASR → OCR → 画面摘要 → 人物识别 → Speaker 绑定 → 事件抽取，7 维度全方位理解 |
| 🔗 **统一时间轴** | "先镜头切分，再提取各模态"——所有数据以 `scene_index` 为锚点物理对齐 |
| 📦 **MemoryUnit** | 以 shot 为原子的多模态融合体，汇聚台词/画面/OCR/人物/事件 + 预计算 embedding |
| 🔍 **三层漏斗检索** | Embedding 粗召回 → 关键词精筛 → LLM Reranker 语义重排 |
| ✅ **证据驱动剪辑** | EditClip 强制引用 `evidence_refs`，不允许自造片段 |
| 🔄 **审核闭环** | Reviewer 进行 Grounding 校验（时间精度、角色一致性、事件覆盖率） |
| 🎬 **长视频支持** | >30min 视频自动启用章节式滑窗规划 |
| 💾 **断点续跑** | 每步结果持久化为 JSON，中断后 `--resume` 可从断点继续 |
| 🤖 **全 Gemini 驱动** | 统一使用 Gemini API 的多模态能力（音频/图片/文本），无需部署多套模型 |

---

## 系统架构

```
用户视频 ──→ [10步理解流水线] ──→ Video Memory (JSON)
               先切后提 │              │
               ┌───────────────────────┤
               │  MemoryUnit × N       │
               │  (shot级多模态融合体)    │
               │  + Embedding 向量索引   │
               └───────────┬───────────┘
                           │
用户需求 ──→ [Director Agent] ──三层检索──→ 候选片段
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
               含 evidence_refs + matched_transcript/vision
                  │
                  ▼
            [渲染引擎 (FFmpeg)]
                  │
                  ▼
             成片 output.mp4
```

---

## 理解流水线（10 步）

流水线执行顺序经过重排，核心原则是 **"先切后提"** ——先完成镜头切分，再在每个 shot 内提取各模态数据，保证所有数据天然携带 `scene_index`。

| 步骤 | 模块 | 输入 | 输出 | 说明 |
|------|------|------|------|------|
| 1 | `ingest.py` | 视频文件 | `meta.json` | 入库 + ffprobe 解析元信息。`video_id` 使用 `文件名_hash` 格式 |
| 2 | `scene_detect.py` | 视频 | `scenes.json` | **提前到第2步**。PySceneDetect 镜头边界检测 |
| 3 | `keyframe.py` | 视频 + scenes | `keyframes/*.jpg` | 每个镜头中点抽取关键帧 |
| 4 | `asr.py` | 视频 + scenes | `transcripts.json` | **按 shot 段提取音频 → 按 shot 段 ASR**。TranscriptSegment 天然携带 `scene_index` |
| 5 | `vision.py` | 关键帧 | `ocr.json` + `vision.json` | Gemini Vision 一次调用同时完成 OCR 和画面摘要 |
| 6 | `character.py` | 关键帧 | `characters.json` | InsightFace 人脸检测聚类 + Gemini Vision 描述 |
| 7 | `speaker_bind.py` | 台词 + 人物 + scenes | `speaker_map.json` | **新增**。共现分析 + LLM 确认，建立 speaker ↔ character 映射 |
| 8 | `event.py` | 台词+画面+人物 | `events.json` | Gemini Text 从多源上下文中抽取关键事件，自动填充 `scene_indices` |
| 9 | `memory_builder.py` | 全部中间结果 | `memory.json` | **新增**。构建 MemoryUnit 多模态融合体 + LLM 角色判定（male_lead/female_lead/villain/...） |
| 10 | `indexer.py` | VideoMemory | `search_index.json` + `faiss.index` | 构建文本索引 + Embedding 向量索引 + FAISS |

### 关键数据流改动

```
旧版:  audio_extract(整体) → ASR(整体) → scene_detect → 事后反查scene
                                    ↓
新版:  scene_detect → keyframe → ASR(按shot段) → vision → character
         ↓              ↓           ↓               ↓         ↓
      scene_index    keyframe    transcript      ocr/vision  character
         ↓              ↓        +scene_index    +scene_index  +scenes
         └──────────────┴────────────┴──────────────┴─────┬────┘
                                                          ↓
                              speaker_bind → event → memory_builder → indexer
                                  ↓              ↓           ↓
                             speaker_map    scene_indices   MemoryUnit
                                                           (融合体)
```

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

SearchResult 包含:
  - matched_modalities: ["transcript", "vision", "embedding"]
  - source_refs: ["transcripts.json#t12.5", "events.json#event_3"]
  - context_before / context_after
  - memory_unit: 完整的 MemoryUnit 数据
```

---

## 证据驱动的 Agent 闭环

### Director Agent

Director Agent 生成 EditPlan 时遵循以下规则：
- **候选白名单**：`source_scene_index` 必须来自检索结果，不允许自造
- **证据引用**：每个 EditClip 自动填充 `evidence_refs` / `matched_transcript` / `matched_vision`
- **长视频滑窗**：超过 30 分钟的视频，按事件分章节独立规划，再合并为完整方案

### Reviewer Agent

Reviewer Agent 的 Grounding 校验项：

| 校验项 | 说明 |
|--------|------|
| evidence_refs 非空 | 每个 clip 必须有证据来源 |
| 时间精度 | source_start/end 与 MemoryUnit 时间偏差 ≤ ±0.5s |
| 角色一致性 | clip 中的 characters 必须在对应 scene 的 MemoryUnit 中出现 |
| 事件覆盖率 | 高重要性事件（importance ≥ 7）是否被 EditPlan 覆盖 |
| 叙事结构 | 包含 hook + climax/resolution |
| 时长偏差 | 在目标时长 ±15% 范围内 |

---

## 核心数据模型

### MemoryUnit（检索原子）

```json
{
  "scene_index": 5,
  "start_time": 120.0,
  "end_time": 135.5,
  "duration": 15.5,
  "keyframe_path": "data/videos/.../keyframes/scene_0005.jpg",
  "transcripts": [
    {"text": "你不要走", "speaker": "speaker_1", "scene_index": 5, "character_id": "char_000"}
  ],
  "vision": {"description": "女主在雨中追赶男主", "mood": "悲伤", "scene_type": "追逐"},
  "ocr": {"texts": []},
  "characters": ["char_000", "char_001"],
  "events": [{"event_type": "转折", "description": "分手后追逐", "importance": 8}],
  "combined_text": "台词: 你不要走 | 画面: 女主在雨中追赶男主 | 情绪: 悲伤 | ...",
  "embedding": [0.123, -0.456, ...]
}
```

### Character（含绑定信息）

```json
{
  "character_id": "char_000",
  "display_name": "女主角",
  "description": "年轻女性，长发，白色连衣裙",
  "role": "female_lead",
  "speaker_ids": ["speaker_1"],
  "appearance_scenes": [0, 2, 5, 8, 12],
  "total_screen_time": 245.3
}
```

### EditClip（含证据链）

```json
{
  "clip_index": 0,
  "source_scene_index": 5,
  "source_start": 120.5,
  "source_end": 132.0,
  "narrative_role": "climax",
  "selection_reason": "高情感张力的追逐场景",
  "evidence_refs": ["search_result#scene_5", "events.json#event_3"],
  "matched_transcript": "你不要走",
  "matched_vision": "女主在雨中追赶男主"
}
```

### SearchResult（含证据来源）

```json
{
  "scene_index": 5,
  "score": 0.87,
  "match_type": "embedding",
  "matched_modalities": ["transcript", "vision", "embedding", "semantic"],
  "source_refs": ["faiss.index#12", "transcripts.json#t120.5"],
  "context_before": "台词: 我们分手吧 | 画面: 咖啡厅对话",
  "context_after": "台词: (哭泣声) | 画面: 雨中独行"
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
│   └── schemas.py           # Pydantic 数据模型（VideoMemory/MemoryUnit/EditPlan/SearchResult）
│
├── pipeline/                # 理解流水线
│   ├── ingest.py            # 入库（video_id 可读化）
│   ├── scene_detect.py      # 镜头切分（PySceneDetect）
│   ├── keyframe.py          # 关键帧抽取
│   ├── asr.py               # 按 shot 段 ASR
│   ├── vision.py            # OCR + 画面摘要（Gemini Vision）
│   ├── character.py         # 人物识别（InsightFace + Gemini）
│   ├── speaker_bind.py      # Speaker ↔ Character 绑定
│   ├── event.py             # 事件抽取
│   ├── memory_builder.py    # MemoryUnit 构建 + 角色判定
│   ├── indexer.py           # 检索索引构建（文本 + Embedding + FAISS）
│   ├── audio.py             # [已弃用] 整体音频提取
│   └── understand.py        # 流水线编排器
│
├── memory/                  # 存储 & 检索
│   ├── store.py             # Video Memory 读写
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
└── data/                    # 运行时数据（自动生成）
    ├── videos/{video_id}/   # 每个视频的工作目录
    │   ├── meta.json
    │   ├── scenes/scenes.json
    │   ├── keyframes/*.jpg
    │   ├── audio_shots/*.wav
    │   ├── transcripts.json
    │   ├── ocr.json
    │   ├── vision.json
    │   ├── characters.json
    │   ├── speaker_map.json
    │   ├── events.json
    │   ├── memory.json
    │   ├── progress.json
    │   └── index/
    │       ├── search_index.json
    │       ├── faiss.index
    │       └── id_map.json
    ├── editplans/           # EditPlan JSON
    └── renders/             # 渲染输出
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
# LLM API（OpenAI 兼容格式）
LLM_API_KEY="your_api_key"
LLM_API_BASE="https://your-api-endpoint/api/v1"
LLM_MODEL="turing/gemini-3.1-flash-lite-latest"

# Embedding 模型
EMBEDDING_MODEL="turing/text-embedding-3-small"
```

### 3. 使用方式

```bash
# 一键全流程：理解 → 生成 EditPlan → 渲染
python main.py auto --video movie.mp4 --prompt "制作一个3分钟的精彩片段合集"

# 分步执行
python main.py understand --video movie.mp4           # 理解视频
python main.py search --video-id xxx --query "打斗场面"  # 搜索
python main.py edit --video-id xxx --prompt "爱情线剪辑" # 生成 EditPlan
python main.py render --plan-id plan_xxx               # 渲染成片

# 断点续跑
python main.py understand --video-id xxx --resume
```

### 4. CLI 参数

| 命令 | 参数 | 说明 |
|------|------|------|
| `understand` | `--video`, `--video-id`, `--resume` | 理解视频 |
| `search` | `--video-id`, `--query`, `--top-k` | 搜索 Video Memory |
| `edit` | `--video-id`, `--prompt`, `--style`, `--duration`, `--platform` | 生成 EditPlan |
| `show-plan` | `--plan-id` | 查看 EditPlan |
| `render` | `--plan-id` | 渲染成片 |
| `auto` | `--video`, `--prompt`, `--style`, `--duration`, `--platform` | 一键全流程 |

---

## 配置项

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `LLM_API_KEY` | — | LLM API 密钥 |
| `LLM_API_BASE` | — | LLM API 基础 URL |
| `LLM_MODEL` | `turing/gemini-3.1-flash-lite-latest` | LLM 模型名称 |
| `LLM_TIMEOUT` | `300` | API 超时（秒） |
| `EMBEDDING_MODEL` | `text-embedding-004` | Embedding 模型名称 |
| `DATA_DIR` | `./data` | 数据存储根目录 |
| `FFMPEG_PATH` | `ffmpeg` | FFmpeg 可执行文件路径 |
| `FFPROBE_PATH` | `ffprobe` | FFprobe 可执行文件路径 |
| `SCENE_DETECT_THRESHOLD` | `27.0` | 镜头切分灵敏度（越低越灵敏） |
| `SCENE_DETECT_MIN_LEN` | `1.0` | 最短镜头时长（秒） |
| `ASR_CHUNK_DURATION` | `600` | 超长 shot ASR 内部切分长度（秒） |
| `KEYFRAME_QUALITY` | `2` | 关键帧质量（FFmpeg -qscale:v） |
| `LOG_LEVEL` | `INFO` | 日志级别 |

---

## 与旧版的对比

| 维度 | 旧版 | 新版 |
|------|------|------|
| **ASR** | 整体音频提取 → 整体 ASR → 事后反查 scene | 先切分 → 按 shot 段提取音频 → 按 shot 段 ASR |
| **时间轴** | 各模态独立，事后拼凑 | 所有模态以 `scene_index` 为锚点天然对齐 |
| **Speaker** | 只有 speaker_1/2，无法对应人物 | speaker_bind 建立 speaker ↔ character 映射 |
| **人物角色** | 无业务角色 | LLM 判定 male_lead/female_lead/villain/supporting |
| **检索** | 单层关键词并列检索 | 三层漏斗（Embedding + 关键词 + LLM Reranker） |
| **检索结果** | 扁平，无证据来源 | `matched_modalities` + `source_refs` + context |
| **EditClip** | 无证据引用 | 强制 `evidence_refs` + `matched_transcript/vision` |
| **Reviewer** | 基础规则 + LLM 审核 | 新增 Grounding 校验（时间精度/角色一致性/事件覆盖） |
| **长视频** | 单次 LLM 调用，可能上下文溢出 | 分章节滑窗规划 + 全局合并 |
| **video_id** | 随机 UUID（不可读） | `文件名_8位hash`（可追溯） |
| **数据结构** | 扁平 list 堆叠 | MemoryUnit 层次化融合 |

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
