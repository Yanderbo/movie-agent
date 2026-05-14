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
VisionSummary ─────────────── 画面摘要（v2: 含 action/expression/props）

── 人物层 ──
Character ─────────────────── 人物（基础版）
CharacterDeep(Character) ──── 深度人物（v2: 弧线/共现/重要性/台词数）
CharacterArc ──────────────── 人物弧线
CharacterRelation ─────────── 人物关系

── 叙事层 ──
Beat ──────────────────────── 剧情节拍（连续 shot 组成）
StoryScene ────────────────── 故事场景（连续 beat 组成）
Event (= EventNode) ───────── 事件（v2: 含 beat_indices / story_scene_indices）
EventEdge ─────────────────── 事件关系边
EventGraph ────────────────── 事件图谱（节点 + 边）

── 剪辑信号层 ──
EditSignal ────────────────── 8维剪辑信号

── Memory 层 ──
MemoryUnit ────────────────── Shot 级多模态融合检索原子
BeatMemoryUnit ────────────── Beat 级记忆单元
SceneMemoryUnit ───────────── StoryScene 级记忆单元
VideoMemory ───────────────── 完整理解结果汇总（多层结构）

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

### Beat（剧情节拍）🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `beat_index` | int | 节拍编号 |
| `shot_indices` | list[int] | 组成的 shot 列表 |
| `beat_type` | str | setup / confrontation / resolution / transition / montage |
| `description` | str | 叙事描述 |
| `emotion` | str | 主导情绪 |
| `intensity` | float | 戏剧强度 0-1 |
| `characters` | list[str] | 出场人物 |

### StoryScene（故事场景）🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `story_scene_index` | int | 场景编号 |
| `beat_indices` | list[int] | 组成的 beat 列表 |
| `shot_indices` | list[int] | 组成的 shot 列表（汇总） |
| `location` | str | 场景地点 |
| `plot_function` | str | inciting_incident / rising / climax / falling / resolution / setup |
| `description` | str | 场景描述 |

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
| `combined_text` | str | 拼接后的多模态检索文本 |
| `embedding` | list[float] | 预计算的语义向量 |
| `beat_index` | int? | **v2** 所属 Beat |
| `story_scene_index` | int? | **v2** 所属 StoryScene |
| `edit_signal` | EditSignal? | **v2** 关联的剪辑信号 |

### BeatMemoryUnit — Beat 级记忆单元 🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `beat_index` | int | 节拍编号 |
| `shot_indices` | list[int] | 组成的 shot |
| `beat_type` / `description` / `emotion` | str | 节拍描述 |
| `transcript_summary` | str | 台词摘要 |
| `combined_text` | str | 检索文本 |
| `edit_signal` | EditSignal? | 剪辑信号 |

### SceneMemoryUnit — StoryScene 级记忆单元 🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `story_scene_index` | int | 场景编号 |
| `beat_indices` | list[int] | 组成的 beat |
| `location` / `description` / `plot_function` | str | 场景描述 |
| `combined_text` | str | 检索文本 |
| `edit_signal` | EditSignal? | 剪辑信号 |

### EditSignal — 剪辑信号 🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `unit_type` | str | shot / beat / story_scene |
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

### CharacterDeep — 深度人物 🆕

继承自 `Character`，新增：

| 字段 | 类型 | 说明 |
|------|------|------|
| `importance_score` | float | 综合重要性 0-1 |
| `first_appearance` / `last_appearance` | float | 首/末出场时间 |
| `arc` | CharacterArc? | 人物弧线 |
| `co_appearing_characters` | list[str] | 共现角色 |
| `dialogue_count` | int | 台词数 |
| `key_event_indices` | list[int] | 关键事件索引 |

### EventGraph — 事件图谱 🆕

| 字段 | 类型 | 说明 |
|------|------|------|
| `nodes` | list[Event] | 事件节点列表 |
| `edges` | list[EventEdge] | 关系边列表 |

EventEdge 的 `relation_type`: `cause` / `foreshadow` / `reversal` / `escalation` / `resolution` / `parallel`

### TranscriptSegment — 台词

| 关键字段 | 说明 |
|----------|------|
| `scene_index` | 所属镜头索引（由 ASR 回填） |
| `character_id` | 绑定的人物 ID（由 speaker_bind 填写） |
| `cross_shot` | **v2** 是否跨越镜头边界 |
| `transcript_type` | **v2** dialogue / narration / voiceover / subtitle |

### VisionSummary — 画面摘要

v2 新增字段：

| 字段 | 说明 |
|------|------|
| `action_description` | 动作/变化描述（多帧推断） |
| `frame_descriptions` | 各帧独立描述 |
| `expression_changes` | 表情变化描述 |
| `props` | 关键道具列表 |

### EditClip — 剪辑片段

v2 新增字段：

| 字段 | 说明 |
|------|------|
| `edit_signal_ref` | 关联的 EditSignal |
| `source_beat_index` | 来源 Beat |
| `source_story_scene_index` | 来源 StoryScene |

### VideoMemory — 完整理解结果

v2 多层结构：

| 字段组 | 字段 |
|--------|------|
| Shot 层 | `shots`, `transcripts`, `ocr_results`, `vision_summaries` |
| 人物层 | `characters`, `characters_deep`, `character_relations`, `speaker_map` |
| 叙事层 | `beats`, `story_scenes`, `event_graph`, `events` |
| 剪辑信号层 | `edit_signals` |
| Memory 层 | `memory_units`, `beat_memory_units`, `scene_memory_units` |
| 兼容层 | `scenes`（= shots）|

## 向后兼容性

### 类型别名

```python
Scene = Shot        # 旧代码 import Scene 可正常工作
EventNode = Event   # 旧代码 import EventNode 可正常工作
```

### 默认值策略

所有 v2 新增字段均设有默认值，旧版 JSON 数据可直接反序列化：

| 字段 | 默认值 |
|------|--------|
| `Shot.keyframe_paths` | `[]` |
| `Shot.beat_index` / `Shot.story_scene_index` | `None` |
| `TranscriptSegment.cross_shot` | `False` |
| `VisionSummary.action_description` | `""` |
| `Event.beat_indices` / `Event.story_scene_indices` | `[]` |
| `VideoMemory.beats` / `story_scenes` / `edit_signals` | `[]` |
| `VideoMemory.beat_memory_units` / `scene_memory_units` | `[]` |
| `EditClip.edit_signal_ref` | `None` |
