# 数据模型 (schemas.py)

> 文件：`models/schemas.py`
> 职责：定义系统中所有核心 Pydantic 数据结构

## 模型总览

```
── 基础层 ──
VideoMeta ─────────────────── 视频元信息

── Shot 层 ──
Shot (= Scene) ────────────── 镜头（最小视觉单元，含多帧路径）
TranscriptSegment ─────────── 台词（含 cross_shot / transcript_type）
OCRResult ─────────────────── 单场景 OCR 结果
VisionSummary ─────────────── 画面摘要（v3: 含 micro_clip 字段）

── 音频层 🆕 ──
AudioSegment ──────────────── 音频精细时间段
AudioProsody ──────────────── 音频韵律（音乐/音效/沉默/语速/音量/语音情绪）

── 多模态对齐层 🆕 ──
MultimodalAlignment ───────── 跨模态一致性（speaker↔character↔visual↔audio）

── 人物层 ──
Character ─────────────────── 人物（基础版）
CharacterDeep(Character) ──── 深度人物（v2: 弧线/共现/重要性/台词数）
CharacterArc ──────────────── 人物弧线
CharacterRelation ─────────── 人物关系

── 叙事层 ──
Beat ──────────────────────── 剧情节拍（连续 shot 组成）
StoryScene ────────────────── 故事场景（连续 beat 组成）
Chapter ───────────────────── 🆕 长视频大段落（连续 StoryScene 组成）
Event (= EventNode) ───────── 事件（v3: 含 evidence + confidence）
EventEdge ─────────────────── 事件关系边（v3: 含 relation_basis）
EventGraph ────────────────── 事件图谱（节点 + 边）

── 剪辑信号层 ──
EditSignal ────────────────── 8维剪辑信号
NarrativeSignal ───────────── 🆕 叙事信号（弧位置/张力/信息密度）
RecompositionSignal ───────── 🆕 二次创作信号（梗潜力/平台适配/二创格式）

── Memory 层 ──
MemoryUnit ────────────────── Shot 级多模态融合检索原子（v3: 含 audio/alignment/chapter）
BeatMemoryUnit ────────────── Beat 级记忆单元
SceneMemoryUnit ───────────── StoryScene 级记忆单元
ChapterMemoryUnit ─────────── 🆕 Chapter 级记忆单元

VideoMemory ───────────────── 完整理解结果汇总（v3: 四层结构 + 三类信号）

── 剪辑方案层 ──
EditClip ──────────────────── 剪辑片段（含证据链 + EditSignal引用）
EditPlan ──────────────────── 结构化剪辑方案
SearchResult ──────────────── 搜索结果（含证据溯源 + beat/scene索引）
ReviewResult ──────────────── 审核结果
BGMConfig ─────────────────── 背景音乐配置
```

## 关键模型详解

### Shot（镜头）

v2 从 `Scene` 重命名为 `Shot`，通过 `Scene = Shot` 别名保持向后兼容。

| 字段 | 类型 | 说明 |
|------|------|------|
| `scene_index` | int | 镜头编号（保留旧字段名） |
| `start_time` / `end_time` | float | 时间范围 |
| `keyframe_path` | str? | 单帧路径（兼容旧代码） |
| `keyframe_paths` | list[str] | **v2** 多帧路径列表 |
| `beat_index` | int? | **v2** 所属 Beat |
| `story_scene_index` | int? | **v2** 所属 StoryScene |

`shot_index` property 等价于 `scene_index`。

### VisionSummary（画面摘要）

| 字段 | 类型 | 说明 |
|------|------|------|
| `description` | str | 画面描述 |
| `mood` | str | 情绪/氛围 |
| `scene_type` | str | 场景类型 |
| `objects` | list[str] | 可见物体 |
| `action_description` | str | **v2** 动作/变化描述 |
| `frame_descriptions` | list[str] | **v2** 各帧独立描述 |
| `expression_changes` | str | **v2** 表情变化描述 |
| `props` | list[str] | **v2** 关键道具列表 |
| `camera_motion` | str | **v3** 镜头运动（pan/tilt/zoom/static/tracking/handheld） |
| `interaction_description` | str | **v3** 人物互动描述 |
| `shot_scale` | str | **v3** 景别（close_up/medium/wide/...） |

### AudioProsody（音频韵律）🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `scene_index` | int | 对应的 shot |
| `has_music` | bool | 是否有背景音乐 |
| `music_mood` | str | 音乐情绪 |
| `has_sfx` | bool | 是否有音效 |
| `sfx_tags` | list[str] | 音效标签列表 |
| `silence_ratio` | float | 沉默占比 0-1 |
| `speech_rate` | str | 语速（slow / normal / fast） |
| `volume_peak` | float | 音量峰值 0-1 |
| `speech_emotion` | str | 语音情绪 |
| `audio_segments` | list[AudioSegment] | 精细时间段 |

### AudioSegment（音频精细时间段）🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `start_time` / `end_time` | float | 时间范围 |
| `segment_type` | str | speech/music/sfx/silence |
| `description` | str | 描述 |

### MultimodalAlignment（多模态对齐）🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `scene_index` | int | 对应的 shot |
| `start_time` / `end_time` | float | 对齐时间范围 |
| `speaker_to_character` | dict[str,str] | speaker_id → character_id 映射 |
| `visible_characters` | list[str] | 画面中可见的人物 |
| `speaking_characters` | list[str] | 正在说话的人物 |
| `active_modalities` | list[str] | 活跃模态（speech/visual_action/music/sfx/silence） |
| `dominant_modality` | str | 主导模态 |
| `alignment_confidence` | float | 对齐置信度 0-1 |
| `notes` | str | 冲突说明（如画外音检测） |

### Beat（剧情节拍）

| 字段 | 类型 | 说明 |
|------|------|------|
| `beat_index` | int | 节拍编号 |
| `shot_indices` | list[int] | 组成的 shot 列表 |
| `beat_type` | str | setup / confrontation / resolution / transition / montage |
| `description` | str | 叙事描述 |
| `emotion` | str | 主导情绪 |
| `intensity` | float | 戏剧强度 0-1 |
| `characters` | list[str] | 出场人物 |

### StoryScene（故事场景）

| 字段 | 类型 | 说明 |
|------|------|------|
| `story_scene_index` | int | 场景编号 |
| `beat_indices` | list[int] | 组成的 beat 列表 |
| `shot_indices` | list[int] | 组成的 shot 列表（汇总） |
| `location` | str | 场景地点 |
| `plot_function` | str | inciting_incident / rising / climax / falling / resolution / setup |
| `description` | str | 场景描述 |

### Chapter（长视频大段落）🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `chapter_index` | int | 章节编号 |
| `story_scene_indices` | list[int] | 包含的 StoryScene |
| `beat_indices` | list[int] | 包含的 Beat（汇总） |
| `shot_indices` | list[int] | 包含的 Shot（汇总） |
| `title` | str | 章节标题 |
| `description` | str | 章节描述 |
| `chapter_type` | str | prologue / act_1 / act_2 / act_3 / climax_act / epilogue / flashback |
| `theme` | str | 主题 |
| `characters` | list[str] | 出场人物 |
| `mood_progression` | str | 情绪走向描述 |
| `start_time` / `end_time` / `duration` | float | 时间信息 |

### Event（事件）v3 增强

| 字段 | 类型 | 说明 |
|------|------|------|
| 原有字段 | — | event_index, event_type, description, characters, importance, ... |
| `evidence` | list[str] | **v3** 支撑该事件的证据描述 |
| `confidence` | float | **v3** 事件置信度 0-1 |
| `beat_indices` | list[int] | **v2** 关联的 Beat |
| `story_scene_indices` | list[int] | **v2** 关联的 StoryScene |

### EventEdge（事件关系边）v3 增强

| 字段 | 类型 | 说明 |
|------|------|------|
| `source_event` / `target_event` | int | 源/目标事件索引 |
| `relation_type` | str | cause / foreshadow / reversal / escalation / resolution / parallel |
| `description` | str | 关系描述 |
| `evidence` | list[str] | **v3** 支撑该关系的证据 |
| `confidence` | float | **v3** 关系置信度 0-1 |
| `relation_basis` | str | **v3** 推理依据 |

### EditSignal — 剪辑信号

| 字段 | 类型 | 说明 |
|------|------|------|
| `unit_type` | str | shot / beat / story_scene / chapter（当前计算逻辑主要生成 shot/beat/story_scene） |
| `unit_index` | int | 对应的 index |
| `hook_score` | float | 钩子适合度 0-1 |
| `plot_importance` | float | 剧情重要性 0-1 |
| `emotional_intensity` | float | 情绪强度 0-1 |
| `visual_impact` | float | 视觉冲击力 0-1 |
| `independence_score` | float | 片段独立性 0-1 |
| `continuity_dependency` | float | 连续性依赖 0-1 |
| `boundary_quality` | float | 剪辑边界质量 0-1 |
| `spoiler_level` | float | 剧透程度 0-1 |
| `suggested_usage` | list[str] | hook / trailer / highlight / recap / climax_clip / character_intro |

### NarrativeSignal — 叙事信号 🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `unit_type` | str | beat / story_scene / chapter（当前计算逻辑主要生成 beat/story_scene） |
| `unit_index` | int | 对应的 index |
| `arc_position` | float | 叙事弧位置 0-1（开头=0，结尾=1） |
| `tension_level` | float | 张力水平 0-1 |
| `information_density` | float | 信息密度 0-1 |
| `character_focus` | str | 主要聚焦的角色 character_id |
| `narrative_function` | str | exposition / rising_action / climax / falling_action / resolution / transition / comic_relief |
| `theme_relevance` | float | 与主题相关度 0-1 |

### RecompositionSignal — 二次创作信号 🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `unit_type` | str | beat / story_scene（当前计算逻辑主要生成 beat） |
| `unit_index` | int | 对应的 index |
| `meme_potential` | float | 梗/传播潜力 0-1 |
| `emotional_quotability` | float | 情感引用潜力/"名场面"程度 0-1 |
| `context_freedom` | float | 脱离上下文仍有意义的程度 0-1 |
| `remix_flexibility` | float | 可重新组合的灵活度 0-1 |
| `platform_fit` | dict[str, float] | 平台适配分（douyin/bilibili/youtube） |
| `suggested_formats` | list[str] | 建议格式（reaction/compilation/fancam/edit） |

### MemoryUnit — Shot 级检索原子

| 字段 | 类型 | 说明 |
|------|------|------|
| `scene_index` | int | 镜头编号 |
| `start_time` / `end_time` | float | 时间范围 |
| `transcripts` | list[TranscriptSegment] | 该 shot 内的台词 |
| `vision` | VisionSummary? | 画面摘要 |
| `ocr` | OCRResult? | OCR 文字 |
| `characters` | list[str] | 出现的 character_id |
| `events` | list[Event] | 时间重叠的事件 |
| `combined_text` | str | 拼接后的多模态检索文本（v3: 含音频信息） |
| `embedding` | list[float] | 预计算的语义向量 |
| `beat_index` | int? | **v2** 所属 Beat |
| `story_scene_index` | int? | **v2** 所属 StoryScene |
| `edit_signal` | EditSignal? | **v2** 关联的剪辑信号 |
| `chapter_index` | int? | **v3** 所属 Chapter |
| `audio_prosody` | AudioProsody? | **v3** 音频韵律信息 |
| `alignment` | MultimodalAlignment? | **v3** 多模态对齐信息 |

### ChapterMemoryUnit — Chapter 级记忆单元 🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `chapter_index` | int | 章节编号 |
| `story_scene_indices` | list[int] | 包含的 StoryScene |
| `title` / `description` / `theme` / `chapter_type` | str | 章节信息 |
| `characters` | list[str] | 出场人物 |
| `mood_progression` | str | 情绪走向 |
| `combined_text` | str | 检索文本 |
| `edit_signal` | EditSignal? | 剪辑信号 |
| `narrative_signal` | NarrativeSignal? | 叙事信号 |

### VideoMemory — 完整理解结果

v3 多层结构：

| 字段组 | 字段 |
|--------|------|
| Shot 层 | `shots`, `transcripts`, `ocr_results`, `vision_summaries` |
| 音频层 🆕 | `audio_prosodies` |
| 对齐层 🆕 | `multimodal_alignments` |
| 人物层 | `characters`, `characters_deep`, `character_relations`, `speaker_map` |
| 叙事层 | `beats`, `story_scenes`, `chapters` 🆕, `event_graph`, `events` |
| 剪辑信号层 | `edit_signals`, `narrative_signals` 🆕, `recomposition_signals` 🆕 |
| Memory 层 | `memory_units`, `beat_memory_units`, `scene_memory_units`, `chapter_memory_units` 🆕 |
| 兼容层 | `scenes`（= shots）|

## 向后兼容性

### 类型别名

```python
Scene = Shot        # 旧代码 import Scene 可正常工作
EventNode = Event   # 旧代码 import EventNode 可正常工作
```

### 默认值策略

所有 v2/v3 新增字段均设有默认值，旧版 JSON 数据可直接反序列化：

| 字段 | 默认值 | 版本 |
|------|--------|------|
| `Shot.keyframe_paths` | `[]` | v2 |
| `Shot.beat_index` / `Shot.story_scene_index` | `None` | v2 |
| `TranscriptSegment.cross_shot` | `False` | v2 |
| `VisionSummary.action_description` | `""` | v2 |
| `VisionSummary.camera_motion` / `interaction_description` / `shot_scale` | `""` | v3 |
| `Event.evidence` | `[]` | v3 |
| `Event.confidence` | `0.8` | v3 |
| `EventEdge.evidence` | `[]` | v3 |
| `EventEdge.relation_basis` | `""` | v3 |
| `EventEdge.confidence` | `0.5` | v3 |
| `VideoMemory.audio_prosodies` / `multimodal_alignments` | `[]` | v3 |
| `VideoMemory.chapters` | `[]` | v3 |
| `VideoMemory.narrative_signals` / `recomposition_signals` | `[]` | v3 |
| `VideoMemory.chapter_memory_units` | `[]` | v3 |
| `MemoryUnit.chapter_index` | `None` | v3 |
| `MemoryUnit.audio_prosody` / `alignment` | `None` | v3 |
| `EditClip.edit_signal_ref` | `None` | v2 |
