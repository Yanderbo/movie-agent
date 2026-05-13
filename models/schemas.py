# -*- coding: utf-8 -*-
"""
Pydantic 数据模型定义
定义系统中所有核心数据结构：视频元信息、镜头、台词、人物、事件、VideoMemory、EditPlan。
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
# 视频元信息
# ═══════════════════════════════════════════════════════════════

class VideoMeta(BaseModel):
    """视频基本元信息"""
    video_id: str
    filename: str
    original_path: str
    storage_path: str
    duration: float                   # 总时长(秒)
    width: int
    height: int
    fps: float
    codec: str
    file_size: int                    # bytes
    audio_path: Optional[str] = None
    status: str = "uploaded"          # uploaded/processing/ready/failed
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════
# 镜头 / 场景
# ═══════════════════════════════════════════════════════════════

class Scene(BaseModel):
    """单个镜头/场景"""
    scene_index: int
    start_time: float                 # 起始时间(秒)
    end_time: float                   # 结束时间(秒)
    duration: float
    keyframe_path: Optional[str] = None


# ═══════════════════════════════════════════════════════════════
# ASR 台词
# ═══════════════════════════════════════════════════════════════

class WordTimestamp(BaseModel):
    """单词级时间戳"""
    word: str
    start: float
    end: float


class TranscriptSegment(BaseModel):
    """一段台词"""
    start_time: float
    end_time: float
    text: str
    speaker: Optional[str] = None
    confidence: float = 1.0
    words: list[WordTimestamp] = Field(default_factory=list)
    scene_index: int = -1             # 所属镜头索引（-1 表示未绑定）
    character_id: Optional[str] = None  # 绑定的人物 ID（由 speaker_bind 填写）


# ═══════════════════════════════════════════════════════════════
# OCR 结果
# ═══════════════════════════════════════════════════════════════

class OCRResult(BaseModel):
    """单个场景的 OCR 结果"""
    scene_index: int
    timestamp: float
    texts: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 画面摘要
# ═══════════════════════════════════════════════════════════════

class VisionSummary(BaseModel):
    """单个场景的画面摘要"""
    scene_index: int
    timestamp: float
    description: str                  # 画面详细描述
    objects: list[str] = Field(default_factory=list)
    mood: str = ""                    # 画面情绪
    scene_type: str = ""              # 对话/动作/空镜/过渡...


# ═══════════════════════════════════════════════════════════════
# 人物
# ═══════════════════════════════════════════════════════════════

class Character(BaseModel):
    """识别到的人物"""
    character_id: str
    display_name: str
    description: str = ""
    thumbnail_path: Optional[str] = None
    appearance_scenes: list[int] = Field(default_factory=list)
    total_screen_time: float = 0.0
    speaker_ids: list[str] = Field(default_factory=list)  # 绑定的 ASR speaker 标识
    role: Optional[str] = None        # 业务角色: male_lead/female_lead/villain/supporting/...


# ═══════════════════════════════════════════════════════════════
# 事件
# ═══════════════════════════════════════════════════════════════

class Event(BaseModel):
    """抽取的事件"""
    event_index: int
    start_time: float
    end_time: float
    event_type: str                   # 对话/冲突/转折/高潮/结局/日常...
    description: str
    characters: list[str] = Field(default_factory=list)
    emotion: str = ""
    importance: int = 5               # 1-10
    scene_indices: list[int] = Field(default_factory=list)  # 该事件覆盖的镜头索引


# ═══════════════════════════════════════════════════════════════
# MemoryUnit（多模态检索原子）
# ═══════════════════════════════════════════════════════════════

class MemoryUnit(BaseModel):
    """
    一个镜头（shot）的所有模态数据的融合体，是检索的最小单元。

    每个 MemoryUnit 对应一个 scene_index，汇聚了该 shot 的台词、
    画面描述、OCR、人物和事件信息，以及预计算的语义向量。
    """
    scene_index: int
    start_time: float
    end_time: float
    duration: float
    keyframe_path: Optional[str] = None
    transcripts: list[TranscriptSegment] = Field(default_factory=list)
    vision: Optional[VisionSummary] = None
    ocr: Optional[OCRResult] = None
    characters: list[str] = Field(default_factory=list)  # character_id 列表
    events: list[Event] = Field(default_factory=list)     # 与该 shot 时间重叠的事件
    combined_text: str = ""           # 拼接后的多模态文本（用于 embedding）
    embedding: list[float] = Field(default_factory=list)  # 预计算的语义向量


# ═══════════════════════════════════════════════════════════════
# Video Memory（汇总）
# ═══════════════════════════════════════════════════════════════

class VideoMemory(BaseModel):
    """完整的视频理解结果"""
    video_id: str
    meta: VideoMeta
    scenes: list[Scene] = Field(default_factory=list)
    transcripts: list[TranscriptSegment] = Field(default_factory=list)
    ocr_results: list[OCRResult] = Field(default_factory=list)
    vision_summaries: list[VisionSummary] = Field(default_factory=list)
    characters: list[Character] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
    memory_units: list[MemoryUnit] = Field(default_factory=list)  # 预构建的检索单元
    speaker_map: dict[str, str] = Field(default_factory=dict)     # speaker_id → character_id
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════
# EditPlan
# ═══════════════════════════════════════════════════════════════

class AudioStrategy(BaseModel):
    """
    片段音频策略（已弃用）。

    .. deprecated::
        当前版本使用 EditClip.audio_volume: float 替代本类。
        保留此定义仅为向后兼容已序列化的旧 JSON 数据。
        不要在新代码中使用此类。
    """
    keep_original: bool = True
    volume: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0


class EditClip(BaseModel):
    """EditPlan 中的单个片段"""
    clip_index: int
    source_scene_index: int
    source_start: float               # 源视频起始时间(秒)
    source_end: float                 # 源视频结束时间(秒)
    timeline_start: float             # 目标时间线起始
    timeline_end: float               # 目标时间线结束
    narrative_role: str               # hook/rising_action/climax/resolution/outro
    selection_reason: str             # 选择理由
    characters: list[str] = Field(default_factory=list)
    subtitle_text: Optional[str] = None
    narration_suggestion: Optional[str] = None
    transition_in: str = "cut"
    transition_out: str = "cut"
    speed: float = 1.0
    audio_volume: float = 1.0
    # ── 证据链字段 ──
    evidence_refs: list[str] = Field(default_factory=list)  # 来源证据引用
    matched_transcript: Optional[str] = None  # 该 clip 对应的台词原文
    matched_vision: Optional[str] = None      # 该 clip 对应的画面描述


class BGMConfig(BaseModel):
    """背景音乐配置"""
    enabled: bool = False
    path: Optional[str] = None
    volume: float = 0.15
    fade_in: float = 2.0
    fade_out: float = 3.0


class ReviewResult(BaseModel):
    """审核结果"""
    approved: bool
    score: float = 0.0
    feedback: str = ""
    issues: list[str] = Field(default_factory=list)


class EditPlan(BaseModel):
    """结构化剪辑方案"""
    plan_id: str
    video_id: str
    title: str
    user_prompt: str
    target_duration: float
    style: str                        # emotional/intense/narrative/comedic...
    narrative_structure: str          # chronological/reverse/parallel/thematic
    character_perspective: Optional[str] = None
    target_platform: str = "general"  # douyin/bilibili/youtube/general
    aspect_ratio: str = "16:9"
    clips: list[EditClip] = Field(default_factory=list)
    bgm: BGMConfig = Field(default_factory=BGMConfig)
    review_result: Optional[ReviewResult] = None
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


# ═══════════════════════════════════════════════════════════════
# 搜索相关
# ═══════════════════════════════════════════════════════════════

class SearchResult(BaseModel):
    """搜索结果"""
    scene_index: int
    score: float
    match_type: str                   # keyword/semantic/character/event
    snippet: str = ""
    scene: Optional[Scene] = None
    transcript: Optional[str] = None
    vision_summary: Optional[str] = None
    # ── 证据链字段 ──
    matched_modalities: list[str] = Field(default_factory=list)   # 命中的模态列表
    source_refs: list[str] = Field(default_factory=list)          # 证据来源路径
    context_before: Optional[str] = None  # 前一个 shot 的摘要
    context_after: Optional[str] = None   # 后一个 shot 的摘要
    memory_unit: Optional[MemoryUnit] = None  # 完整的 MemoryUnit 数据
