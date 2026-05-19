# 存储与 CLI (store.py / main.py / config.py)

> 文件：`memory/store.py`、`main.py`、`config.py`
> 职责：Video Memory 持久化、命令行接口、全局配置

---

## Video Memory 存储 (memory/store.py)

### 概述

基于 JSON 文件的读写层，所有数据持久化在 `data/videos/{video_id}/` 目录下。

v3 支持多层 VideoMemory 结构（Shot / Beat / StoryScene / Chapter / EventGraph / 三类剪辑信号），并保留对旧版 JSON 字段的兼容。

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

当 `memory.json` 不存在时（如流水线在 step 16 之前中断），从以下文件逐个加载：

| 文件 | 模型类 | 说明 |
|------|--------|------|
| `scenes/scenes.json` | Shot (= Scene) | 镜头与关键帧路径 |
| `transcripts.json` | TranscriptSegment | 长窗口 ASR 结果 |
| `ocr.json` | OCRResult | OCR 文字 |
| `vision.json` | VisionSummary | 多帧画面摘要 + micro_clip |
| `audio_prosody.json` | AudioProsody | v3 音频韵律 |
| `multimodal_alignments.json` | MultimodalAlignment | v3 多模态对齐 |
| `characters.json` | CharacterDeep → Character | 先尝试 CharacterDeep |
| `speaker_map.json` | dict | speaker → character |
| `beats.json` | Beat | Shot → Beat |
| `story_scenes.json` | StoryScene | Beat → StoryScene |
| `chapters.json` | Chapter | v3 StoryScene → Chapter |
| `events.json` | Event | 事件节点 |
| `event_graph.json` | EventGraph | 事件关系图 |
| `character_relations.json` | CharacterRelation | 人物关系 |
| `edit_signals.json` | EditSignal | 8 维剪辑信号 |
| `narrative_signals.json` | NarrativeSignal | v3 叙事信号 |
| `recomposition_signals.json` | RecompositionSignal | v3 二创信号 |

注意：散文件组装时 **不包含 MemoryUnit / BeatMemoryUnit / SceneMemoryUnit / ChapterMemoryUnit**（需要 step 16 构建）和 **不包含 embedding**（需要 step 17 构建）。

---

## 命令行接口 (main.py)

### 命令一览

| 命令 | 函数 | 说明 |
|------|------|------|
| `understand` | `cmd_understand()` | 运行理解流水线（17步） |
| `search` | `cmd_search()` | 搜索 Video Memory |
| `edit` | `cmd_edit()` | 生成 EditPlan |
| `show-plan` | `cmd_show_plan()` | 查看 EditPlan |
| `render` | `cmd_render()` | 渲染成片 |
| `auto` | `cmd_auto()` | 一键全流程 |

### 各命令的参数

```bash
# 理解视频（17步）
python main.py understand --video movie.mp4
python main.py understand --video-id xxx --resume   # 断点续跑（兼容 v1 旧进度）

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
| **视频处理** | `COMPRESS_HEIGHT`, `COMPRESS_FPS` | 压缩参数 |
| **镜头切分** | `SCENE_DETECT_THRESHOLD`, `SCENE_DETECT_MIN_LEN` | PySceneDetect 参数 |
| **ASR** | `ASR_CHUNK_DURATION`, `ASR_WINDOW_DURATION` | 超长 shot 切分 + 长窗口大小 |
| **多帧采样** | `MULTI_KEYFRAME_MAX` | 每 shot 最大采样帧数（默认6） |
| **日志** | `LOG_DIR`, `LOG_LEVEL` | 日志存储 |
| **路径** | `DATA_DIR`, `VIDEOS_DIR`, `EDITPLANS_DIR`, `RENDERS_DIR` | 数据目录 |

### 关键新增配置项

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ASR_WINDOW_DURATION` | `300` | 长窗口 ASR 窗口大小（秒） |
| `MULTI_KEYFRAME_MAX` | `6` | 每个 shot 最大采样帧数 |

### `init_dirs()`

```python
def init_dirs():
    for d in [DATA_DIR, VIDEOS_DIR, EDITPLANS_DIR, RENDERS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
```

在 `run_understand()`、`run_director()`、`run_render()` 入口处调用。

---

## 数据目录结构

```
data/
├── videos/
│   └── {video_id}/
│       ├── meta.json                ← step 1
│       ├── original.mp4             ← step 1
│       ├── progress.json            ← 断点续跑
│       ├── scenes/
│       │   ├── scenes.json          ← step 2
│       │   └── keyframes/
│       │       ├── scene_0000_f0.jpg  ← step 3 (多帧)
│       │       ├── scene_0000_f1.jpg
│       │       └── ...
│       ├── audio_windows/
│       │   ├── window_0000.wav      ← step 4 (长窗口 ASR)
│       │   └── ...
│       ├── transcripts.json         ← step 4 (含 cross_shot / transcript_type)
│       ├── ocr.json                 ← step 5
│       ├── vision.json              ← step 5 (含 action_description / props / micro_clip)
│       ├── audio_prosody.json       ← step 6
│       ├── characters/
│       │   └── char_000.jpg         ← step 7
│       ├── characters.json          ← step 7 (CharacterDeep)
│       ├── speaker_map.json         ← step 8
│       ├── multimodal_alignments.json ← step 9
│       ├── beats.json               ← step 10
│       ├── story_scenes.json        ← step 11
│       ├── chapters.json            ← step 12
│       ├── events.json              ← step 13
│       ├── event_graph.json         ← step 13
│       ├── character_arcs.json      ← step 14
│       ├── character_relations.json ← step 14
│       ├── edit_signals.json        ← step 15
│       ├── narrative_signals.json   ← step 15
│       ├── recomposition_signals.json ← step 15
│       ├── memory.json              ← step 16 (四层 MemoryUnit)
│       └── index/
│           ├── search_index.json    ← step 17
│           ├── faiss.index          ← step 17
│           ├── id_map.json          ← step 17
│           ├── character_index.json ← step 17
│           ├── event_index.json     ← step 17
│           ├── relation_index.json  ← step 17
│           ├── emotion_index.json   ← step 17
│           ├── edit_signal_index.json ← step 17
│           ├── audio_index.json     ← step 17
│           └── chapter_index.json   ← step 17
├── editplans/
│   └── plan_xxx.json
└── renders/
    └── render_xxx/
        └── output.mp4
```
