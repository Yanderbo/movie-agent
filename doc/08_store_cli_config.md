# 存储与 CLI (store.py / main.py / config.py)

> 文件：`memory/store.py`、`main.py`、`config.py`
> 职责：Video Memory 持久化、命令行接口、全局配置

---

## Video Memory 存储 (memory/store.py)

### 概述

基于 JSON 文件的读写层，所有数据持久化在 `data/videos/{video_id}/` 目录下。

当前结构支持 v4.1 understand 产物：入库压缩信息、FaceCluster 脸谱、MinuteChunk 回填文件、Shot / Beat / StoryScene / Chapter / EventGraph / 三类剪辑信号，并保留对旧版 JSON 字段的兼容。

### 主要函数

| 函数 | 功能 |
|------|------|
| `load_meta(video_id)` | 加载 `meta.json` → `VideoMeta` |
| `load_memory(video_id)` | 加载完整 Video Memory |
| `save_memory(memory)` | 保存 Video Memory 到 `memory.json` |
| `list_videos()` | 列出所有已入库视频 |

### `load_memory()` 的加载策略

```python
def load_memory(video_id) -> VideoMemory:
    if memory.json 存在:
        直接加载（包含全部多层数据）
    else:
        调用 _assemble_memory() 从散文件组装
```

### `_assemble_memory()` — 散文件组装

当 `memory.json` 不存在时（如流水线在 Step 10 `final_build` 之前中断），从以下文件逐个加载：

| 文件 | 模型类 | 说明 |
|------|--------|------|
| `scenes/scenes.json` | Shot (= Scene) | 镜头与关键帧路径 |
| `transcripts.json` | TranscriptSegment | v4.1 MinuteChunk 回填 ASR 结果 |
| `ocr.json` | OCRResult | v4.1 MinuteChunk 回填 OCR 文字 |
| `vision.json` | VisionSummary | v4.1 MinuteChunk 回填画面摘要 |
| `audio_prosody.json` | AudioProsody | v4.1 MinuteChunk 回填音频韵律 |
| `multimodal_alignments.json` | MultimodalAlignment | v4.1 MinuteChunk 回填多模态对齐 |
| `characters.json` | CharacterDeep → Character | 先尝试 CharacterDeep |
| `speaker_map.json` | dict | speaker → character |
| `beats.json` | Beat | Shot → Beat |
| `story_scenes.json` | StoryScene | Beat → StoryScene |
| `chapters.json` | Chapter | StoryScene → Chapter |
| `events.json` | Event | 事件节点 |
| `event_graph.json` | EventGraph | 事件关系图 |
| `character_relations.json` | CharacterRelation | 人物关系 |
| `edit_signals.json` | EditSignal | 8 维剪辑信号 |
| `narrative_signals.json` | NarrativeSignal | 叙事信号 |
| `recomposition_signals.json` | RecompositionSignal | 二创信号 |

注意：

- 散文件组装时 **不包含 MemoryUnit / BeatMemoryUnit / SceneMemoryUnit / ChapterMemoryUnit**，这些需要 Step 10 的 `final_build` 构建。
- 散文件组装时 **不包含 embedding 和索引文件**，这些同样由 Step 10 的 `indexer.py` 构建。
- `minute_chunks.json`、`character_profiles.json`、`characters/face_clusters.json` 是 v4.1 的中间/辅助产物，当前 `_assemble_memory()` 不会直接挂载到 `VideoMemory` 顶层。

---

## 命令行接口 (main.py)

### 命令一览

| 命令 | 函数 | 说明 |
|------|------|------|
| `understand` | `cmd_understand()` | 运行理解流水线（10步 v4.1） |
| `search` | `cmd_search()` | 搜索 Video Memory |
| `edit` | `cmd_edit()` | 生成 EditPlan |
| `show-plan` | `cmd_show_plan()` | 查看 EditPlan |
| `render` | `cmd_render()` | 渲染成片 |
| `auto` | `cmd_auto()` | 一键全流程 |

### 各命令的参数

```bash
# 理解视频（10步 v4.1）
python main.py understand --video movie.mp4
python main.py understand --video-id xxx --resume   # 断点续跑（兼容旧进度）

# 搜索
python main.py search --video-id xxx --query "打斗场面" --top-k 10

# 生成 EditPlan
python main.py edit --video-id xxx --prompt "爱情线剪辑" \
    --style emotional --duration 180 --platform bilibili

# 渲染
python main.py render --plan-id plan_xxx

# 一键全流程
python main.py auto --video movie.mp4 --prompt "3分钟精彩片段"
```

### `cmd_auto()` 三阶段

1. 调用 `run_understand(video_path)` → 返回 `video_id`
2. 调用 `run_director(video_id, prompt, ...)` → 返回 `EditPlan`
3. 调用 `run_render(plan.plan_id)` → 返回输出路径

---

## 全局配置 (config.py)

### 加载机制

```python
dotenv.load_dotenv()          # 从 .env 加载
os.getenv("KEY", "default")   # 环境变量优先
```

### 配置分组

| 分组 | 关键配置项 | 说明 |
|------|-----------|------|
| **LLM API** | `LLM_API_KEY`, `LLM_API_BASE`, `LLM_MODEL`, `LLM_TIMEOUT` | Gemini API 连接 |
| **Embedding** | `EMBEDDING_MODEL` | Embedding 模型名称 |
| **FFmpeg** | `FFMPEG_PATH`, `FFPROBE_PATH` | 可执行文件路径 |
| **视频压缩** | `COMPRESS_MAX_HEIGHT`, `COMPRESS_MAX_FPS` | v4.1: >480p→480p, >10fps→10fps |
| **镜头切分** | `SCENE_DETECT_THRESHOLD`, `SCENE_DETECT_MIN_LEN` | PySceneDetect 参数 |
| **多帧采样** | `MULTI_KEYFRAME_MAX` | 每 shot 最大采样帧数（默认6） |
| **MinuteChunk** | `CHUNK_TARGET_DURATION`, `CHUNK_MERGE_THRESHOLD`, `CHUNK_MIN/MAX_DURATION` | v4.1: 分钟级 chunk 参数 |
| **人脸聚类** | `FACE_GALLERY_MAX/MIN`, `FACE_CLUSTER_*`, `FACE_MIN_*`, `FACE_DETECT_*`, `FACE_REJECT_SIDE_FACE` | v4.1: 人脸检测、过滤、聚类、拆分/合并和脸谱参数 |
| **日志** | `LOG_DIR`, `LOG_LEVEL` | 日志存储 |
| **路径** | `DATA_DIR`, `VIDEOS_DIR`, `EDITPLANS_DIR`, `RENDERS_DIR` | 数据目录 |

### v4.1 新增配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `COMPRESS_MAX_HEIGHT` | `480` | 高于此值时压缩分辨率 |
| `COMPRESS_MAX_FPS` | `10` | 高于此值时降帧率 |
| `CHUNK_TARGET_DURATION` | `150` | MinuteChunk 目标时长(秒) |
| `CHUNK_MIN_DURATION` | `90` | MinuteChunk 最小时长(秒)，当前配置保留，构建逻辑暂未强制使用 |
| `CHUNK_MAX_DURATION` | `210` | MinuteChunk 最大时长(秒)，当前配置保留，构建逻辑暂未强制使用 |
| `CHUNK_MERGE_THRESHOLD` | `30` | 尾段低于此值合并到前一个 |
| `FACE_GALLERY_MAX` | `6` | 每个非路人角色最多保存的代表脸数量 |
| `FACE_GALLERY_MIN` | `3` | 每个非路人角色尽量保留的最少代表脸数量 |
| `FACE_CLUSTER_EPS` | `0.42` | 初始 DBSCAN 余弦距离阈值，越大越容易合并 |
| `FACE_CLUSTER_MIN_SAMPLES` | `3` | DBSCAN 成簇最少样本数 |
| `FACE_CLUSTER_SPLIT_EPS` | `0.30` | 疑似混簇二次拆分阈值 |
| `FACE_CLUSTER_MAX_RADIUS` | `0.34` | 簇内 90 分位半径上限，超过则尝试拆分 |
| `FACE_CLUSTER_MERGE_SIM` | `0.86` | 簇中心相似度合并阈值 |
| `FACE_CLUSTER_MERGE_LINK_SIM` | `0.78` | 代表脸桥接相似度合并阈值 |
| `FACE_CLUSTER_MERGE_MIN_CENTROID_SIM` | `0.62` | 使用桥接合并时要求的最低簇中心相似度 |
| `FACE_CLUSTER_MERGE_STRONG_LINK_SIM` | `0.82` | 单对代表脸极高相似时的强桥接合并阈值 |
| `FACE_CLUSTER_MERGE_STRONG_MIN_CENTROID_SIM` | `0.50` | 强桥接合并时要求的最低簇中心相似度 |
| `FACE_CLUSTER_MERGE_MAX_FACES` | `32` | 每个簇用于合并比较的最多代表脸数量 |
| `FACE_MIN_DET_SCORE` | `0.65` | InsightFace 检测置信度下限 |
| `FACE_MIN_FACE_RATIO` | `0.05` | 人脸 bbox 短边 / 关键帧短边 的比例下限 |
| `FACE_MIN_FACE_PIXEL_FLOOR` | `16` | 人脸 bbox 短边绝对像素兜底 |
| `FACE_MIN_CROP_RATIO` | `0.08` | gallery 裁剪图短边 / 关键帧短边 的比例下限 |
| `FACE_MIN_CROP_PIXEL_FLOOR` | `48` | gallery 裁剪图短边绝对像素兜底 |
| `FACE_REJECT_SIDE_FACE` | `true` | 是否过滤明显侧脸 |
| `FACE_MAX_POSE_YAW` | `35` | InsightFace pose yaw 绝对值超过该角度视为侧脸 |
| `FACE_MAX_LANDMARK_IMBALANCE` | `0.35` | 5 点 landmarks 鼻尖偏离双眼中心的最大相对比例 |
| `FACE_PASSERBY_MIN` | `3` | 中等时长视频中，低于此出现场景数视为路人 |
| `FACE_DETECT_DEVICE` | `auto` | 人脸检测设备：auto/cuda/gpu/cpu |
| `FACE_DETECT_GPU_ID` | `auto` | 自动选择或指定 CUDA device id |
| `FACE_KEEP_PASSERBY_GALLERY` | `false` | 是否保存路人脸谱 |

### `init_dirs()`

```python
def init_dirs():
    for d in [DATA_DIR, VIDEOS_DIR, EDITPLANS_DIR, RENDERS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
```

在 `run_understand()`、`run_director()`、`run_render()` 入口处调用。

---

## 数据目录结构（v4.1）

```
data/
├── videos/
│   └── {video_id}/
│       ├── meta.json                ← step 1 (含压缩信息)
│       ├── original.*               ← step 1 (渲染用)
│       ├── compressed.mp4           ← step 1 (按需生成，理解用)
│       ├── progress.json            ← 断点续跑
│       ├── scenes/
│       │   ├── scenes.json          ← step 2
│       │   └── keyframes/           ← step 3
│       ├── characters/              ← step 4 (人脸聚类)
│       │   ├── face_clusters.json
│       │   ├── char_000_gallery/
│       │   └── char_001_gallery/
│       ├── chunk_segments/          ← step 5 (临时视频片段)
│       ├── minute_chunks.json       ← step 5 (融合理解原始结果)
│       ├── character_profiles.json  ← step 5 (动态角色档案)
│       ├── transcripts.json         ← step 5 (回填, 已带 character_id)
│       ├── ocr.json                 ← step 5 (回填)
│       ├── vision.json              ← step 5 (回填)
│       ├── audio_prosody.json       ← step 5 (回填)
│       ├── multimodal_alignments.json ← step 5 (回填)
│       ├── characters.json          ← step 5 (回填)
│       ├── speaker_map.json         ← step 5 (自动生成)
│       ├── beats.json               ← step 6
│       ├── story_scenes.json        ← step 7
│       ├── chapters.json            ← step 8
│       ├── events.json              ← step 9
│       ├── event_graph.json         ← step 9
│       ├── character_arcs.json      ← step 9
│       ├── character_relations.json ← step 9
│       ├── edit_signals.json        ← step 10
│       ├── narrative_signals.json   ← step 10
│       ├── recomposition_signals.json ← step 10
│       ├── memory.json              ← step 10
│       └── index/                   ← step 10
├── editplans/
│   └── plan_xxx.json
└── renders/
    └── render_xxx/
        └── output.mp4
```
