# 存储与 CLI (store.py / main.py / config.py)

> 文件：`memory/store.py`、`main.py`、`config.py`
> 职责：Video Memory 持久化、命令行接口、全局配置

---

## Video Memory 存储 (memory/store.py)

### 概述

基于 JSON 文件的读写层，所有数据持久化在 `data/videos/{video_id}/` 目录下。

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
        直接加载（包含 MemoryUnit 等完整数据）
    else:
        调用 _assemble_memory() 从散文件组装
```

### `_assemble_memory()` — 散文件组装

当 `memory.json` 不存在时（如流水线在 step 9 之前中断），从以下文件逐个加载：

| 文件 | 模型类 |
|------|--------|
| `scenes/scenes.json` | Scene |
| `transcripts.json` | TranscriptSegment |
| `ocr.json` | OCRResult |
| `vision.json` | VisionSummary |
| `characters.json` | Character |
| `events.json` | Event |
| `speaker_map.json` | dict（可选） |

注意：散文件组装时 **不包含 MemoryUnit**（需要 step 9 构建）和 **不包含 embedding**（需要 step 10 构建）。

---

## 命令行接口 (main.py)

### 命令一览

| 命令 | 函数 | 说明 |
|------|------|------|
| `understand` | `cmd_understand()` | 运行理解流水线 |
| `search` | `cmd_search()` | 搜索 Video Memory |
| `edit` | `cmd_edit()` | 生成 EditPlan |
| `show-plan` | `cmd_show_plan()` | 查看 EditPlan |
| `render` | `cmd_render()` | 渲染成片 |
| `auto` | `cmd_auto()` | 一键全流程 |

### 各命令的参数

```bash
# 理解视频
python main.py understand --video movie.mp4
python main.py understand --video-id xxx --resume   # 断点续跑

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

### `cmd_search()` 输出格式

```
[Scene 5] 分数: 0.873
  时间: 120.5s - 135.0s
  模态: transcript, vision, embedding
  匹配: 你不要走...
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
| **ASR** | `ASR_CHUNK_DURATION` | 超长 shot 内部切分长度 |
| **日志** | `LOG_DIR`, `LOG_LEVEL` | 日志存储 |
| **路径** | `DATA_DIR`, `VIDEOS_DIR`, `EDITPLANS_DIR`, `RENDERS_DIR` | 数据目录 |

### `init_dirs()`

```python
def init_dirs():
    for d in [DATA_DIR, VIDEOS_DIR, EDITPLANS_DIR, RENDERS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
```

在 `run_understand()`、`run_director()`、`run_render()` 入口处调用，确保目录存在。

---

## 数据目录结构

```
data/
├── videos/
│   └── {video_id}/
│       ├── meta.json               ← step 1
│       ├── original.mp4            ← step 1
│       ├── progress.json           ← 断点续跑
│       ├── scenes/
│       │   ├── scenes.json         ← step 2
│       │   └── keyframes/
│       │       ├── scene_0000.jpg  ← step 3
│       │       └── ...
│       ├── audio_shots/
│       │   ├── shot_000.wav        ← step 4
│       │   └── ...
│       ├── transcripts.json        ← step 4
│       ├── ocr.json                ← step 5
│       ├── vision.json             ← step 5
│       ├── characters/
│       │   └── char_000.jpg        ← step 6
│       ├── characters.json         ← step 6
│       ├── speaker_map.json        ← step 7
│       ├── events.json             ← step 8
│       ├── memory.json             ← step 9
│       └── index/
│           ├── search_index.json   ← step 10
│           ├── faiss.index         ← step 10
│           └── id_map.json         ← step 10
├── editplans/
│   └── plan_xxx.json
└── renders/
    └── render_xxx/
        └── output.mp4
```
