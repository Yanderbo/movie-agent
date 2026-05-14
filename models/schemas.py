# -*- coding: utf-8 -*-
"""
Pydantic 数据模型定义（v2 — 面向剪辑决策）

层级结构: Shot → Beat → StoryScene → EventGraph
新增: EditSignal / CharacterArc / CharacterRelation / EventGraph

向后兼容:
  - Scene = Shot（别名）
  - Character 保留，CharacterDeep 扩展
  - VideoMemory.scenes 等价于 VideoMemory.shots
  - Event 保留作为 EventNode 别名
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
# Shot（原 Scene）— 最小视觉单元
# ═══════════════════════════════════════════════════════════════

class Shot(BaseModel):
    """
    单个镜头（最小视觉单元）。

    v2 变更:
    - 从 Scene 重命名为 Shot
    - keyframe_path 保留兼容，新增 keyframe_paths 支持多帧
    - 新增 beat_index / story_scene_index 向上关联
    """
    scene_index: int                  # 保留旧字段名以兼容
    start_time: float                 # 起始时间(秒)
    end_time: float                   # 结束时间(秒)
    duration: float
    keyframe_path: Optional[str] = None               # 兼容旧代码的单帧
    keyframe_paths: list[str] = Field(default_factory=list)  # 多帧路径
    beat_index: Optional[int] = None                   # 所属 Beat
    story_scene_index: Optional[int] = None            # 所属 StoryScene

    @property
    def shot_index(self) -> int:
        """shot_index 等价于 scene_index"""
        return self.scene_index


# 向后兼容别名
Scene = Shot


# ═══════════════════════════════════════════════════════════════
# Beat — 剧情节拍（由连续 shot 组成的叙事微单元）
# ═══════════════════════════════════════════════════════════════

class Beat(BaseModel):
    """
    剧情节拍 — 连续多个 shot 组成的叙事微单元。

    例如：一段对话、一个动作序列、一个情绪转折。
    粒度介于 shot 和 story_scene 之间。
    """
    beat_index: int
    start_time: float
    end_time: float
    duration: float = 0.0
    shot_indices: list[int] = Field(default_factory=list)
    beat_type: str = ""               # setup / confrontation / resolution / transition / montage
    description: str = ""
    emotion: str = ""
    intensity: float = 0.0            # 0.0 - 1.0 情绪/戏剧强度
    characters: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# StoryScene — 故事场景（由连续 beat 组成的完整场景单元）
# ═══════════════════════════════════════════════════════════════

class StoryScene(BaseModel):
    """
    故事场景 — 连续多个 beat 组成的完整叙事场景。

    通常对应一个地点/情境下的完整行动序列，
    是剪辑中 "一段完整情节" 的基本单位。
    """
    story_scene_index: int
    start_time: float
    end_time: float
    duration: float = 0.0
    beat_indices: list[int] = Field(default_factory=list)
    shot_indices: list[int] = Field(default_factory=list)
    location: str = ""
    description: str = ""
    characters: list[str] = Field(default_factory=list)
    plot_function: str = ""           # inciting_incident / rising / climax / falling / resolution / setup


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
    transcript_type: str = "dialogue"  # dialogue / narration / voiceover / subtitle
    cross_shot: bool = False           # 是否跨越镜头边界


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
    """单个场景的画面摘要（v2 支持多帧）"""
    scene_index: int
    timestamp: float
    description: str                  # 画面详细描述
    objects: list[str] = Field(default_factory=list)
    mood: str = ""                    # 画面情绪
    scene_type: str = ""              # 对话/动作/空镜/过渡...
    # ── v2 新增 ──
    action_description: str = ""      # 动作/变化描述（多帧推断）
    frame_descriptions: list[str] = Field(default_factory=list)  # 各帧独立描述
    expression_changes: str = ""      # 表情变化描述
    props: list[str] = Field(default_factory=list)  # 关键道具


# ═══════════════════════════════════════════════════════════════
# 人物（基础版 — 向后兼容）
# ═══════════════════════════════════════════════════════════════

class Character(BaseModel):
    """识别到的人物（基础版，保留向后兼容）"""
    character_id: str
    display_name: str
    description: str = ""
    thumbnail_path: Optional[str] = None
    appearance_scenes: list[int] = Field(default_factory=list)
    total_screen_time: float = 0.0
    speaker_ids: list[str] = Field(default_factory=list)  # 绑定的 ASR speaker 标识
    role: Optional[str] = None        # 业务角色: male_lead/female_lead/villain/supporting/...


# ═══════════════════════════════════════════════════════════════
# 深度人物分析（v2 扩展）
# ═══════════════════════════════════════════════════════════════

class CharacterArc(BaseModel):
    """人物弧线 — 一个角色在影片中的成长/变化轨迹"""
    character_id: str
    arc_type: str = ""                # growth / fall / flat / transformation / redemption
    arc_description: str = ""
    key_moments: list[int] = Field(default_factory=list)  # event_index 列表
    emotion_trajectory: list[dict] = Field(default_factory=list)
    # emotion_trajectory: [{"time": 60.0, "emotion": "hope", "intensity": 0.7}, ...]


class CharacterRelation(BaseModel):
    """人物间关系"""
    character_a: str                  # character_id
    character_b: str                  # character_id
    relation_type: str                # ally / rival / romantic / mentor / family / stranger / colleague
    description: str = ""
    strength: float = 0.5             # 0.0 - 1.0
    evolution: list[str] = Field(default_factory=list)  # 关系变化轨迹描述列表
    co_appearance_shots: list[int] = Field(default_factory=list)


class CharacterDeep(Character):
    """
    深度人物信息 — 继承自 Character，新增弧线、关系、重要性等字段。

    在 v2 pipeline 中替代 Character 使用。
    """
    importance_score: float = 0.0     # 0-1 综合重要性
    first_appearance: float = 0.0     # 首次出场时间(秒)
    last_appearance: float = 0.0      # 最后出场时间(秒)
    arc: Optional[CharacterArc] = None
    co_appearing_characters: list[str] = Field(default_factory=list)
    dialogue_count: int = 0
    key_event_indices: list[int] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 事件（v1 — 保留兼容）
# ═══════════════════════════════════════════════════════════════

class Event(BaseModel):
    """抽取的事件（v1 — 保留兼容别名 EventNode）"""
    event_index: int
    start_time: float
    end_time: float
    event_type: str                   # 对话/冲突/转折/高潮/结局/日常...
    description: str
    characters: list[str] = Field(default_factory=list)
    emotion: str = ""
    importance: int = 5               # 1-10
    scene_indices: list[int] = Field(default_factory=list)  # 该事件覆盖的镜头索引
    # ── v2 新增 ──
    beat_indices: list[int] = Field(default_factory=list)
    story_scene_indices: list[int] = Field(default_factory=list)


# v2 别名
EventNode = Event


class EventEdge(BaseModel):
    """事件间关系 — 事件图谱中的边"""
    source_event: int                 # event_index
    target_event: int                 # event_index
    relation_type: str                # cause / foreshadow / reversal / escalation / resolution / parallel
    description: str = ""
    strength: float = 0.5             # 0.0 - 1.0


class EventGraph(BaseModel):
    """完整事件图谱"""
    nodes: list[Event] = Field(default_factory=list)
    edges: list[EventEdge] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# EditSignal — 面向剪辑的信号
# ═══════════════════════════════════════════════════════════════

class EditSignal(BaseModel):
    """
    面向剪辑的信号 — 每个 shot / beat / story_scene 一条。

    这些信号直接服务于 DirectorAgent / ReviewerAgent 的剪辑决策。
    """
    unit_type: str                    # shot / beat / story_scene
    unit_index: int
    start_time: float
    end_time: float
    hook_score: float = 0.0           # 吸引力评分 (0-1): 作为开头钩子的适合度
    plot_importance: float = 0.0      # 剧情重要性 (0-1): 对整体叙事的贡献度
    emotional_intensity: float = 0.0  # 情绪强度 (0-1): 情绪表达的强烈程度
    visual_impact: float = 0.0        # 视觉冲击力 (0-1): 画面构图/运镜/特效的吸引力
    independence_score: float = 0.0   # 片段独立性 (0-1): 不需上下文也能理解的程度
    continuity_dependency: float = 0.0  # 连续性依赖 (0-1): 高=必须与前后连续才有意义
    boundary_quality: float = 0.0     # 剪辑边界质量 (0-1): 作为剪辑点的自然度
    spoiler_level: float = 0.0        # 剧透程度 (0-1): 包含关键剧情信息的程度
    suggested_usage: list[str] = Field(default_factory=list)
    # suggested_usage: hook / trailer / highlight / recap / climax_clip / character_intro


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
    # ── v2 新增 ──
    beat_index: Optional[int] = None
    story_scene_index: Optional[int] = None
    edit_signal: Optional[EditSignal] = None


class BeatMemoryUnit(BaseModel):
    """Beat 级别的记忆单元 — 聚合多个 shot 的信息"""
    beat_index: int
    start_time: float
    end_time: float
    duration: float = 0.0
    shot_indices: list[int] = Field(default_factory=list)
    beat_type: str = ""
    description: str = ""
    emotion: str = ""
    intensity: float = 0.0
    characters: list[str] = Field(default_factory=list)
    transcript_summary: str = ""      # beat 内台词摘要
    combined_text: str = ""
    embedding: list[float] = Field(default_factory=list)
    edit_signal: Optional[EditSignal] = None


class SceneMemoryUnit(BaseModel):
    """StoryScene 级别的记忆单元 — 聚合多个 beat 的信息"""
    story_scene_index: int
    start_time: float
    end_time: float
    duration: float = 0.0
    beat_indices: list[int] = Field(default_factory=list)
    shot_indices: list[int] = Field(default_factory=list)
    location: str = ""
    description: str = ""
    characters: list[str] = Field(default_factory=list)
    plot_function: str = ""
    combined_text: str = ""
    embedding: list[float] = Field(default_factory=list)
    edit_signal: Optional[EditSignal] = None


# ═══════════════════════════════════════════════════════════════
# Video Memory（汇总 — v2 多层结构）
# ═══════════════════════════════════════════════════════════════

class VideoMemory(BaseModel):
    """
    完整的视频理解结果（v2 — 多层结构）。

    层级: Shot → Beat → StoryScene → EventGraph
    """
    video_id: str
    meta: VideoMeta
    # ── Shot 层（原 Scene 层）──
    shots: list[Shot] = Field(default_factory=list)
    transcripts: list[TranscriptSegment] = Field(default_factory=list)
    ocr_results: list[OCRResult] = Field(default_factory=list)
    vision_summaries: list[VisionSummary] = Field(default_factory=list)
    # ── 人物层 ──
    characters: list[Character] = Field(default_factory=list)    # 基础版（兼容）
    characters_deep: list[CharacterDeep] = Field(default_factory=list)  # 深度版
    character_relations: list[CharacterRelation] = Field(default_factory=list)
    speaker_map: dict[str, str] = Field(default_factory=dict)
    # ── 叙事层 ──
    beats: list[Beat] = Field(default_factory=list)
    story_scenes: list[StoryScene] = Field(default_factory=list)
    event_graph: Optional[EventGraph] = None
    # ── 剪辑信号层 ──
    edit_signals: list[EditSignal] = Field(default_factory=list)
    # ── Memory 层 ──
    memory_units: list[MemoryUnit] = Field(default_factory=list)
    beat_memory_units: list[BeatMemoryUnit] = Field(default_factory=list)
    scene_memory_units: list[SceneMemoryUnit] = Field(default_factory=list)
    # ── 兼容旧字段 ──
    scenes: list[Shot] = Field(default_factory=list)
    events: list[Event] = Field(default_factory=list)
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
    # ── v2 新增：剪辑信号引用 ──
    edit_signal_ref: Optional[EditSignal] = None
    source_beat_index: Optional[int] = None
    source_story_scene_index: Optional[int] = None


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
    match_type: str                   # keyword/semantic/character/event/beat/story_scene/edit_signal
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
    # ── v2 新增 ──
    beat_index: Optional[int] = None
    story_scene_index: Optional[int] = None
    edit_signal: Optional[EditSignal] = None
