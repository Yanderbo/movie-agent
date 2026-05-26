# -*- coding: utf-8 -*-
"""
Pydantic 数据模型定义（v3 — 面向剪辑决策 + 深度多模态理解）

层级结构: Shot → Beat → StoryScene → Chapter → EventGraph
新增 v3: AudioProsody / MultimodalAlignment / Chapter / NarrativeSignal / RecompositionSignal
增强 v3: Event 增加 evidence+confidence / EventEdge 增加 relation_basis
         VisionSummary 增加 camera_motion / interaction / shot_scale

向后兼容:
  - Scene = Shot（别名）
  - Character 保留，CharacterDeep 扩展
  - VideoMemory.scenes 等价于 VideoMemory.shots
  - Event 保留作为 EventNode 别名
  - 所有 v3 新增字段均有默认值
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
    # ── v4.1 新增：压缩信息 ──
    compressed_path: str = ""         # 压缩后视频路径（理解流水线使用）
    is_compressed: bool = False       # 是否经过压缩
    original_height: int = 0          # 原始高度（压缩前）
    original_fps: float = 0.0         # 原始帧率（压缩前）
    compressed_height: int = 0        # 压缩后高度
    compressed_fps: float = 0.0       # 压缩后帧率


# ═══════════════════════════════════════════════════════════════
# v4.1 新增：角色脸谱 + 角色档案 + MinuteChunk
# ═══════════════════════════════════════════════════════════════

class CharacterGallery(BaseModel):
    """
    角色脸谱（v4.1 — face_cluster 步骤产物）。

    每个经聚类确认的角色保存 3-6 张代表脸图片，
    用于后续 MinuteChunk 处理时输入 Gemini 进行身份识别。
    """
    character_id: str
    gallery_paths: list[str] = Field(default_factory=list)        # 3-6张代表脸路径
    gallery_timestamps: list[float] = Field(default_factory=list) # 每张脸对应的视频时间
    gallery_scene_indices: list[int] = Field(default_factory=list) # 每张代表脸对应的 shot
    gallery_keyframe_paths: list[str] = Field(default_factory=list) # 每张代表脸来源关键帧
    total_detections: int = 0         # 总检测次数
    appearance_scenes: list[int] = Field(default_factory=list)    # 出现的 shot 列表
    tier: str = "major"               # major / minor / passerby
    embedding_centroid: list[float] = Field(default_factory=list) # 聚类中心向量


class CharacterProfile(BaseModel):
    """
    动态角色档案（v4.1 — 随 MinuteChunk 处理逐步累积）。

    在处理每个 chunk 时，Gemini 会输出角色的新信息，
    包括名称、外观变化、关键行为等，在此档案中动态更新。
    """
    character_id: str
    names: list[str] = Field(default_factory=list)                # 所有已知称呼，第一个为主名称
    description: str = ""             # 最新外观描述
    appearance_changes: list[dict] = Field(default_factory=list)  # [{chunk_idx, description}]
    key_actions: list[dict] = Field(default_factory=list)         # [{chunk_idx, action}]
    relationships_brief: list[dict] = Field(default_factory=list) # [{target_id, relation}]
    tier: str = "major"               # major / minor / passerby
    is_human: bool = True             # False = 动物/机器人等
    entity_type: str = "human"        # human / animal / robot / other
    gallery_ref: str = ""             # 关联的 CharacterGallery.character_id
    merge_suggestions: list[dict] = Field(default_factory=list)  # [{duplicate_id, confidence, reason, chunk_index}]
    merged_into: str = ""             # 如果该角色已被合并，记录目标角色ID


class MinuteChunk(BaseModel):
    """
    分钟级理解单元（v4.1 核心中间产物）。

    将连续 shot 拼接为 ~2-3min 的 chunk，一次性送入 Gemini
    完成 ASR + Vision + Audio + 角色标注 + 跨shot分析。
    处理完成后将结果回填到 shot 级数据结构。
    """
    chunk_index: int
    shot_indices: list[int] = Field(default_factory=list)
    start_time: float
    end_time: float
    duration: float
    # ── Gemini 返回的原始结果 ──
    raw_transcripts: list[dict] = Field(default_factory=list)     # ASR 原始结果
    per_shot_vision: list[dict] = Field(default_factory=list)     # 逐shot画面理解
    per_shot_audio: list[dict] = Field(default_factory=list)      # 逐shot音频特征
    character_updates: list[dict] = Field(default_factory=list)   # 角色动态更新
    cross_shot_analysis: dict = Field(default_factory=dict)       # 跨shot连续性分析
    suggested_beats: list[list[int]] = Field(default_factory=list)  # 建议的beat分组


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
    """单个场景的画面摘要（v2 支持多帧，v3 增加 micro_clip）"""
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
    # ── v3 micro_clip 新增 ──
    camera_motion: str = ""           # 镜头运动: static/pan_left/pan_right/tilt_up/tilt_down/zoom_in/zoom_out/tracking/crane/handheld
    interaction_description: str = "" # 人物间互动描述（对话、肢体接触、对峙、合作等）
    shot_scale: str = ""              # 景别: extreme_close_up/close_up/medium_close/medium/medium_long/long/extreme_long


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
    """抽取的事件（v1 兼容，v3 增加 evidence/confidence）"""
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
    # ── v3 新增 ──
    evidence: list[str] = Field(default_factory=list)   # 证据来源: ["transcript:12-15", "vision:shot_5", "audio:music_shift"]
    confidence: float = 0.8           # 事件抽取置信度 0-1


# v2 别名
EventNode = Event


class EventEdge(BaseModel):
    """事件间关系 — 事件图谱中的边（v3 增加 evidence/confidence/relation_basis）"""
    source_event: int                 # event_index
    target_event: int                 # event_index
    relation_type: str                # cause / foreshadow / reversal / escalation / resolution / parallel
    description: str = ""
    strength: float = 0.5             # 0.0 - 1.0
    # ── v3 新增 ──
    evidence: list[str] = Field(default_factory=list)    # 关系推断的依据
    confidence: float = 0.5           # 关系置信度 0-1
    relation_basis: str = ""          # 关系推断依据说明（如"因为A中角色说了XX，导致B中发生了YY"）


class EventGraph(BaseModel):
    """完整事件图谱"""
    nodes: list[Event] = Field(default_factory=list)
    edges: list[EventEdge] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 音频韵律分析（v3 新增）
# ═══════════════════════════════════════════════════════════════

class AudioSegment(BaseModel):
    """音频段落分析结果"""
    start_time: float
    end_time: float
    segment_type: str = ""            # music / sfx / silence / speech / ambient
    description: str = ""
    intensity: float = 0.0            # 0-1 音量/能量强度


class AudioProsody(BaseModel):
    """单个 shot 的音频韵律分析（v3 新增）"""
    scene_index: int
    has_music: bool = False
    music_mood: str = ""              # energetic / melancholic / tense / romantic / epic / calm
    has_sfx: bool = False
    sfx_tags: list[str] = Field(default_factory=list)  # explosion / door_slam / footsteps ...
    silence_ratio: float = 0.0        # 沉默占比 0-1
    speech_rate: str = ""             # slow / normal / fast
    volume_peak: float = 0.0          # 归一化峰值音量 0-1
    speech_emotion: str = ""          # calm / angry / sad / happy / fearful / surprised / neutral
    audio_segments: list[AudioSegment] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════
# 多模态对齐（v3 新增）
# ═══════════════════════════════════════════════════════════════

class MultimodalAlignment(BaseModel):
    """单个 shot 的多模态对齐结果（v3 新增）"""
    scene_index: int
    start_time: float
    end_time: float
    speaker_to_character: dict[str, str] = Field(default_factory=dict)  # speaker_id → character_id
    visible_characters: list[str] = Field(default_factory=list)         # 画面中可见的 character_id
    speaking_characters: list[str] = Field(default_factory=list)        # 正在说话的 character_id
    active_modalities: list[str] = Field(default_factory=list)          # speech/music/sfx/silence/visual_action/text_overlay
    dominant_modality: str = ""       # 主导模态
    alignment_confidence: float = 0.0 # 对齐置信度 0-1
    notes: str = ""                   # 对齐备注/冲突说明


# ═══════════════════════════════════════════════════════════════
# Chapter — 长视频大段落（v3 新增，StoryScene 之上）
# ═══════════════════════════════════════════════════════════════

class Chapter(BaseModel):
    """长视频大段落 — 在 StoryScene 之上的最高叙事层级（v3 新增）"""
    chapter_index: int
    title: str = ""
    start_time: float
    end_time: float
    duration: float = 0.0
    story_scene_indices: list[int] = Field(default_factory=list)
    beat_indices: list[int] = Field(default_factory=list)
    shot_indices: list[int] = Field(default_factory=list)
    description: str = ""
    chapter_type: str = ""            # prologue / act_1 / act_2 / act_3 / climax_act / epilogue / flashback
    theme: str = ""                   # 本章主题/关键词
    characters: list[str] = Field(default_factory=list)
    mood_progression: str = ""        # 情绪走势描述


# ═══════════════════════════════════════════════════════════════
# EditSignal — 面向剪辑的信号
# ═══════════════════════════════════════════════════════════════

class EditSignal(BaseModel):
    """
    面向剪辑的信号 — 每个 shot / beat / story_scene / chapter 一条。

    这些信号直接服务于 DirectorAgent / ReviewerAgent 的剪辑决策。
    """
    unit_type: str                    # shot / beat / story_scene / chapter
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


class NarrativeSignal(BaseModel):
    """叙事信号 — 衡量片段在叙事结构中的角色（v3 新增）"""
    unit_type: str                    # shot / beat / story_scene / chapter
    unit_index: int
    start_time: float
    end_time: float
    arc_position: float = 0.0         # 在整体叙事弧中的位置 0-1
    tension_level: float = 0.0        # 张力水平 0-1
    information_density: float = 0.0  # 信息密度 0-1
    character_focus: str = ""         # 主要聚焦的角色 character_id
    narrative_function: str = ""      # exposition / rising_action / climax / falling_action / resolution / transition / comic_relief
    theme_relevance: float = 0.0      # 与主题相关度 0-1


class RecompositionSignal(BaseModel):
    """二次创作信号 — 衡量片段在二次剪辑中的价值（v3 新增）"""
    unit_type: str
    unit_index: int
    start_time: float
    end_time: float
    meme_potential: float = 0.0       # 梗/传播潜力 0-1
    emotional_quotability: float = 0.0  # 情感引用潜力 0-1（"名场面"程度）
    context_freedom: float = 0.0      # 脱离上下文仍有意义的程度 0-1
    remix_flexibility: float = 0.0    # 可重新组合的灵活度 0-1
    platform_fit: dict[str, float] = Field(default_factory=dict)  # {"douyin": 0.8, "bilibili": 0.6}
    suggested_formats: list[str] = Field(default_factory=list)  # reaction / compilation / fancam / edit


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
    # ── v3 新增 ──
    chapter_index: Optional[int] = None
    audio_prosody: Optional[AudioProsody] = None
    alignment: Optional[MultimodalAlignment] = None


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
# Chapter MemoryUnit（v3 新增）
# ═══════════════════════════════════════════════════════════════

class ChapterMemoryUnit(BaseModel):
    """Chapter 级记忆单元 — 聚合多个 StoryScene 的信息（v3 新增）"""
    chapter_index: int
    start_time: float
    end_time: float
    duration: float = 0.0
    story_scene_indices: list[int] = Field(default_factory=list)
    title: str = ""
    description: str = ""
    theme: str = ""
    chapter_type: str = ""
    characters: list[str] = Field(default_factory=list)
    mood_progression: str = ""
    combined_text: str = ""
    embedding: list[float] = Field(default_factory=list)
    edit_signal: Optional[EditSignal] = None
    narrative_signal: Optional[NarrativeSignal] = None


# ═══════════════════════════════════════════════════════════════
# Video Memory（汇总 — v3 多层结构）
# ═══════════════════════════════════════════════════════════════

class VideoMemory(BaseModel):
    """
    完整的视频理解结果（v3 — 多层结构 + 深度多模态理解）。

    层级: Shot → Beat → StoryScene → Chapter → EventGraph
    """
    video_id: str
    meta: VideoMeta
    # ── Shot 层（原 Scene 层）──
    shots: list[Shot] = Field(default_factory=list)
    transcripts: list[TranscriptSegment] = Field(default_factory=list)
    ocr_results: list[OCRResult] = Field(default_factory=list)
    vision_summaries: list[VisionSummary] = Field(default_factory=list)
    # ── 音频层（v3 新增）──
    audio_prosodies: list[AudioProsody] = Field(default_factory=list)
    # ── 多模态对齐层（v3 新增）──
    multimodal_alignments: list[MultimodalAlignment] = Field(default_factory=list)
    # ── 人物层 ──
    characters: list[Character] = Field(default_factory=list)    # 基础版（兼容）
    characters_deep: list[CharacterDeep] = Field(default_factory=list)  # 深度版
    character_relations: list[CharacterRelation] = Field(default_factory=list)
    speaker_map: dict[str, str] = Field(default_factory=dict)
    # ── 叙事层 ──
    beats: list[Beat] = Field(default_factory=list)
    story_scenes: list[StoryScene] = Field(default_factory=list)
    chapters: list[Chapter] = Field(default_factory=list)        # v3 新增
    event_graph: Optional[EventGraph] = None
    # ── 剪辑信号层 ──
    edit_signals: list[EditSignal] = Field(default_factory=list)
    narrative_signals: list[NarrativeSignal] = Field(default_factory=list)          # v3 新增
    recomposition_signals: list[RecompositionSignal] = Field(default_factory=list)  # v3 新增
    # ── Memory 层 ──
    memory_units: list[MemoryUnit] = Field(default_factory=list)
    beat_memory_units: list[BeatMemoryUnit] = Field(default_factory=list)
    scene_memory_units: list[SceneMemoryUnit] = Field(default_factory=list)
    chapter_memory_units: list[ChapterMemoryUnit] = Field(default_factory=list)     # v3 新增
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
