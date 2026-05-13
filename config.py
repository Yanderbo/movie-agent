# -*- coding: utf-8 -*-
"""
全局配置管理
所有路径、API 密钥、参数均通过环境变量或 .env 文件配置。
"""
import os
from pathlib import Path
import dotenv

dotenv.load_dotenv()

# ─── 项目根目录 ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()

# ─── 数据存储根目录 ──────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", str(PROJECT_ROOT / "data")))
VIDEOS_DIR = DATA_DIR / "videos"
EDITPLANS_DIR = DATA_DIR / "editplans"
RENDERS_DIR = DATA_DIR / "renders"

# ─── LLM API 配置（中转 API，OpenAI 兼容格式） ─────────────
LLM_API_KEY = os.getenv("LLM_API_KEY", "sk-4PuajBlBMr71SnM7jeem2WgCe56ZiRtmrzP3NMTJLkS")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://live-turing.cn.llm.tcljd.com/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "turing/gemini-3.1-flash-lite-latest")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))

# ─── Embedding API 配置 ──────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

# ─── FFmpeg 配置 ──────────────────────────────────────────────
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.getenv("FFPROBE_PATH", "ffprobe")

# ─── 视频处理参数 ─────────────────────────────────────────────
# 压缩参数
COMPRESS_HEIGHT = int(os.getenv("COMPRESS_HEIGHT", "480"))
COMPRESS_FPS = int(os.getenv("COMPRESS_FPS", "15"))

# 镜头切分参数
SCENE_DETECT_THRESHOLD = float(os.getenv("SCENE_DETECT_THRESHOLD", "27.0"))
SCENE_DETECT_MIN_LEN = float(os.getenv("SCENE_DETECT_MIN_LEN", "1.0"))  # 最短镜头秒数

# 关键帧质量
KEYFRAME_QUALITY = int(os.getenv("KEYFRAME_QUALITY", "2"))  # FFmpeg -qscale:v

# ─── ASR 参数 ─────────────────────────────────────────────────
ASR_CHUNK_DURATION = int(os.getenv("ASR_CHUNK_DURATION", "600"))  # 音频分段长度（秒）

# ─── 日志配置 ─────────────────────────────────────────────────
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── 初始化目录 ──────────────────────────────────────────────
def init_dirs():
    """创建所有必需的目录"""
    for d in [DATA_DIR, VIDEOS_DIR, EDITPLANS_DIR, RENDERS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
