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
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://live-turing.cn.llm.tcljd.com/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "turing/gemini-3.1-flash-lite-latest")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))

# ─── Embedding API 配置 ──────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

# ─── FFmpeg 配置 ──────────────────────────────────────────────
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.getenv("FFPROBE_PATH", "ffprobe")
FFMPEG_COMPRESS_TIMEOUT = int(os.getenv("FFMPEG_COMPRESS_TIMEOUT", "0"))  # 0 = no timeout

# ─── 视频处理参数 ─────────────────────────────────────────────
# 压缩参数（v4.1: 用于理解流水线的视频压缩）
COMPRESS_MAX_HEIGHT = int(os.getenv("COMPRESS_MAX_HEIGHT", "480"))  # 高于此值时压缩
COMPRESS_MAX_FPS = int(os.getenv("COMPRESS_MAX_FPS", "10"))        # 高于此值时降帧率

# 镜头切分参数
SCENE_DETECT_THRESHOLD = float(os.getenv("SCENE_DETECT_THRESHOLD", "27.0"))
SCENE_DETECT_MIN_LEN = float(os.getenv("SCENE_DETECT_MIN_LEN", "1.0"))  # 最短镜头秒数

# 关键帧质量
KEYFRAME_QUALITY = int(os.getenv("KEYFRAME_QUALITY", "2"))  # FFmpeg -qscale:v

# ─── ASR 参数 ─────────────────────────────────────────────────
ASR_CHUNK_DURATION = int(os.getenv("ASR_CHUNK_DURATION", "600"))  # 音频分段长度（秒）
ASR_WINDOW_DURATION = int(os.getenv("ASR_WINDOW_DURATION", "300"))  # 长窗口 ASR 窗口(秒)

# ─── 多帧采样参数 ─────────────────────────────────────────────
MULTI_KEYFRAME_MAX = int(os.getenv("MULTI_KEYFRAME_MAX", "6"))  # 每个 shot 最大采样帧数

# ─── MinuteChunk 参数（v4.1 新增）────────────────────────────
CHUNK_TARGET_DURATION = int(os.getenv("CHUNK_TARGET_DURATION", "150"))  # 目标时长(秒) ~2.5min
CHUNK_MIN_DURATION = int(os.getenv("CHUNK_MIN_DURATION", "90"))         # 最小时长(秒)
CHUNK_MAX_DURATION = int(os.getenv("CHUNK_MAX_DURATION", "210"))        # 最大时长(秒)
CHUNK_MERGE_THRESHOLD = int(os.getenv("CHUNK_MERGE_THRESHOLD", "30"))   # 尾段低于此值合并到前一个

# ─── 人脸聚类参数（v4.1 新增）─────────────────────────────────
FACE_GALLERY_MAX = int(os.getenv("FACE_GALLERY_MAX", "6"))              # 每个非路人角色最多保存的代表脸数量
FACE_GALLERY_MIN = int(os.getenv("FACE_GALLERY_MIN", "3"))              # 每个非路人角色尽量保留的最少代表脸数量
FACE_CLUSTER_EPS = float(os.getenv("FACE_CLUSTER_EPS", "0.42"))         # 初始 DBSCAN 余弦距离阈值；越大越容易把脸聚到一起
FACE_CLUSTER_MIN_SAMPLES = int(os.getenv("FACE_CLUSTER_MIN_SAMPLES", "3")) # DBSCAN 成簇最少样本数；越大越容易丢掉偶发人脸
FACE_CLUSTER_SPLIT_EPS = float(os.getenv("FACE_CLUSTER_SPLIT_EPS", "0.30")) # 疑似混簇二次拆分的更严格 DBSCAN 阈值
FACE_CLUSTER_MAX_RADIUS = float(os.getenv("FACE_CLUSTER_MAX_RADIUS", "0.34")) # 簇内 90 分位余弦半径；超过则尝试拆分
FACE_CLUSTER_MERGE_SIM = float(os.getenv("FACE_CLUSTER_MERGE_SIM", "0.86")) # 簇中心余弦相似度合并阈值
FACE_CLUSTER_MERGE_LINK_SIM = float(os.getenv("FACE_CLUSTER_MERGE_LINK_SIM", "0.78")) # 代表脸跨簇 top 相似度合并阈值
FACE_CLUSTER_MERGE_MIN_CENTROID_SIM = float(os.getenv("FACE_CLUSTER_MERGE_MIN_CENTROID_SIM", "0.62")) # 使用代表脸合并时要求的最低簇中心相似度
FACE_CLUSTER_MERGE_MAX_FACES = int(os.getenv("FACE_CLUSTER_MERGE_MAX_FACES", "32")) # 每个簇用于合并比较的最多代表脸数量
FACE_MIN_DET_SCORE = float(os.getenv("FACE_MIN_DET_SCORE", "0.65"))      # InsightFace 检测置信度下限
FACE_MIN_FACE_RATIO = float(os.getenv("FACE_MIN_FACE_RATIO", "0.05"))   # 人脸 bbox 短边 / 关键帧短边 的比例下限
FACE_MIN_FACE_PIXEL_FLOOR = float(os.getenv("FACE_MIN_FACE_PIXEL_FLOOR", os.getenv("FACE_MIN_FACE_SIZE", "16"))) # 人脸 bbox 短边绝对像素兜底下限
FACE_MIN_FACE_SIZE = int(os.getenv("FACE_MIN_FACE_SIZE", str(int(FACE_MIN_FACE_PIXEL_FLOOR)))) # 兼容旧变量；新逻辑优先用 FACE_MIN_FACE_PIXEL_FLOOR
FACE_PASSERBY_MIN_APPEARANCES = int(os.getenv("FACE_PASSERBY_MIN", "3")) # 中等时长视频中，低于该出现场景数视为路人
FACE_DETECT_DEVICE = os.getenv("FACE_DETECT_DEVICE", "auto").lower()    # 人脸检测设备：auto/cuda/gpu/cpu
FACE_KEEP_PASSERBY_GALLERY = os.getenv("FACE_KEEP_PASSERBY_GALLERY", "false").lower() in ("1", "true", "yes", "on") # 是否也保存路人脸谱
FACE_DETECT_GPU_ID = os.getenv("FACE_DETECT_GPU_ID", "auto").lower()    # auto 或指定 CUDA device id

# ─── 日志配置 ─────────────────────────────────────────────────
LOG_DIR = Path(os.getenv("LOG_DIR", str(PROJECT_ROOT / "logs")))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ─── 初始化目录 ──────────────────────────────────────────────
def init_dirs():
    """创建所有必需的目录"""
    for d in [DATA_DIR, VIDEOS_DIR, EDITPLANS_DIR, RENDERS_DIR, LOG_DIR]:
        d.mkdir(parents=True, exist_ok=True)
