# 数据模型 (schemas.py)

> 文件：`models/schemas.py`
> 职责：定义系统中所有核心 Pydantic 数据结构

## 模型总览

```
VideoMeta ─────────────────── 视频元信息
Scene ─────────────────────── 镜头/场景
TranscriptSegment ─────────── 一段台词（带 scene_index + character_id）
OCRResult ─────────────────── 单场景 OCR 结果
VisionSummary ─────────────── 单场景画面摘要
Character ─────────────────── 人物（带 speaker_ids + role）
Event ─────────────────────── 事件（带 scene_indices）
MemoryUnit ────────────────── 多模态融合检索原子
VideoMemory ───────────────── 完整理解结果汇总
EditClip ──────────────────── 剪辑片段（带证据链）
EditPlan ──────────────────── 结构化剪辑方案
SearchResult ──────────────── 搜索结果（带证据溯源）
ReviewResult ──────────────── 审核结果
BGMConfig ─────────────────── 背景音乐配置
```

## 关键模型详解

### MemoryUnit — 检索原子

MemoryUnit 是检索系统的最小单元，每个 MemoryUnit 对应一个 `scene_index`。

| 字段 | 类型 | 说明 |
|------|------|------|
| `scene_index` | int | 镜头编号 |
| `start_time` / `end_time` | float | 时间范围 |
| `transcripts` | list[TranscriptSegment] | 该 shot 内的台词 |
| `vision` | VisionSummary | 画面摘要 |
| `ocr` | OCRResult | OCR 文字 |
| `characters` | list[str] | 出现的 character_id |
| `events` | list[Event] | 时间重叠的事件 |
| `combined_text` | str | 拼接后的多模态检索文本 |
| `embedding` | list[float] | 预计算的语义向量 |

**设计意图**：将各模态数据在 shot 级别融合，使检索系统能够通过一个对象获取该时间段内的全部信息。`combined_text` 是用于 embedding 的原始文本，格式为 `台词: xxx | 画面: xxx | 情绪: xxx | ...`。

### TranscriptSegment — 台词

| 关键字段 | 说明 |
|----------|------|
| `scene_index` | 所属镜头索引（由 ASR 天然写入，不再需要事后反查） |
| `character_id` | 绑定的人物 ID（由 speaker_bind 步骤填写） |
| `speaker` | ASR 识别的说话人标识（speaker_1 等） |

### Character — 人物

| 关键字段 | 说明 |
|----------|------|
| `speaker_ids` | 绑定的 ASR speaker 标识列表 |
| `role` | 业务角色（male_lead / female_lead / villain / supporting / minor） |
| `appearance_scenes` | 出现的 scene_index 列表 |

### Event — 事件

| 关键字段 | 说明 |
|----------|------|
| `scene_indices` | 该事件覆盖的镜头索引列表（通过时间交叉计算） |
| `importance` | 重要性 1-10，Reviewer 用于检查高重要性事件覆盖率 |

### EditClip — 剪辑片段（含证据链）

| 关键字段 | 说明 |
|----------|------|
| `evidence_refs` | 证据来源引用列表（如 `search_result#scene_5`） |
| `matched_transcript` | 该片段对应的台词原文 |
| `matched_vision` | 该片段对应的画面描述 |
| `narrative_role` | hook / rising_action / climax / resolution / outro |

### SearchResult — 搜索结果（含证据溯源）

| 关键字段 | 说明 |
|----------|------|
| `matched_modalities` | 命中的模态列表（transcript/vision/embedding/event/semantic） |
| `source_refs` | 证据来源路径（如 `faiss.index#12`、`events.json#event_3`） |
| `context_before` / `context_after` | 前后 shot 的摘要（提供叙事上下文） |
| `memory_unit` | 完整的 MemoryUnit 数据 |

## 向后兼容性

所有新增字段均设有默认值：
- `scene_index = -1`（TranscriptSegment）
- `scene_indices = []`（Event）
- `speaker_ids = []`、`role = None`（Character）
- `memory_units = []`、`speaker_map = {}`（VideoMemory）
- `evidence_refs = []`（EditClip / SearchResult）

旧版 JSON 数据可以直接反序列化为新版模型，不会报错。
