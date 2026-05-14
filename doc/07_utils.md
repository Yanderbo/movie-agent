# 工具库 (utils/)

> 文件：`utils/llm_client.py`、`utils/ffmpeg_utils.py`、`utils/logger.py`
> 职责：为系统提供 LLM 调用、FFmpeg 操作和日志三大基础能力

---

## LLM 客户端 (llm_client.py)

### 概述

统一封装的 LLM 调用客户端，兼容 OpenAI API 格式。通过中转 API 调用 Gemini 系列模型。

### 主要方法

| 方法 | 功能 | 使用场景 |
|------|------|----------|
| `chat(prompt, system_prompt, temperature)` | 纯文本对话 | 事件抽取、角色判定、Reranker |
| `chat_with_media(prompt, media_path, temperature)` | 单个音频/视频文件 + 文本 | ASR（音频）、人物描述（图片） |
| `chat_with_images(prompt, image_paths, temperature)` | 多张图片 + 文本 | 批量画面分析 |
| `parse_json(response)` | 从 LLM 响应中提取 JSON | 所有需要结构化输出的场景 |

### 关键实现

```python
def get_llm_client() -> LLMClient
```

- 使用 `config.LLM_API_KEY`、`config.LLM_API_BASE`、`config.LLM_MODEL`
- 所有请求通过 `requests.post()` 发送到 OpenAI 兼容接口
- 超时由 `config.LLM_TIMEOUT`（默认300秒）控制

### `parse_json()` 的处理

1. 尝试直接 `json.loads(response)`
2. 失败则用正则提取 ` ```json ... ``` ` 代码块
3. 再失败则查找第一个 `{` 或 `[` 到最后一个 `}` 或 `]` 的子串
4. 仍然失败返回 `None`

### Embedding API

```python
def get_embedding(text, model=None) -> list[float]
```

调用同一 API 的 `/embeddings` 端点，生成文本向量。用于 `indexer.py` 和 `search.py`。

---

## FFmpeg 工具 (ffmpeg_utils.py)

### 概述

封装所有底层 FFmpeg/FFprobe 命令行调用，被 pipeline 和 render 共同使用。

### 主要函数

| 函数 | 功能 | 调用方 |
|------|------|--------|
| `get_video_info(path)` | ffprobe 解析元信息 | `ingest.py` |
| `probe_video(path)` | 原始 ffprobe JSON | 内部使用 |
| `extract_keyframe(video, timestamp, output)` | 提取单帧 JPEG | `keyframe.py` |
| `extract_audio(video, output)` | 提取整体音频 WAV | `audio.py`（已弃用） |
| `extract_audio_segment(video, start, duration, output)` | 提取指定时间段音频 | `asr.py`（新版） |
| `get_audio_duration(path)` | 获取音频时长 | `asr.py`、`ffmpeg_ops.py` |
| `split_audio(path, output_dir, chunk_seconds)` | 切分长音频 | `asr.py`（超长shot内部切分） |
| `concat_clips(paths, output)` | 拼接音视频 | `render/ffmpeg_ops.py` |

### `_run_cmd()` — 统一命令执行

```python
def _run_cmd(cmd: list[str], timeout: int = 300)
```

- 使用 `subprocess.run(cmd, capture_output=True, timeout=timeout)`
- 失败时记录 stderr 并抛出 `RuntimeError`
- 所有 FFmpeg 函数都通过此方法执行

### `extract_audio_segment()` — 核心新增函数

```python
def extract_audio_segment(video_path, start_time, duration, output_path)
```

```bash
ffmpeg -y -ss {start} -i {video} -t {duration}
       -vn -acodec pcm_s16le -ar 16000 -ac 1
       {output}.wav
```

- 单声道 16kHz WAV（最适合 ASR）
- 被 `asr.py` 在每个 shot 段调用

### `get_video_info()` 返回格式

```python
{
    "duration": 3600.5,
    "width": 1920,
    "height": 1080,
    "fps": 24.0,
    "codec": "h264",
    "file_size": 1073741824
}
```

---

## 日志 (logger.py)

### 使用方式

```python
from utils.logger import get_logger
logger = get_logger("ModuleName")

logger.info("消息")
logger.warning("警告")
logger.error("错误")
```

### 配置

- 日志级别由 `config.LOG_LEVEL`（默认 `INFO`）控制
- 日志文件保存在 `config.LOG_DIR`（默认 `./logs/`）
- 同时输出到控制台和文件
- 格式：`[时间] [级别] [模块名] 消息`
