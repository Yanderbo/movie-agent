# Director Agent (director.py)

> 文件：`agents/director.py`、根目录 `prompt.py`
> 职责：基于 VideoMemory、叙事层级和三类剪辑信号，生成 beat 级 EditPlan。

## 核心原则

Director 的最终剪辑单元是 **Beat**。Shot 来自镜头切换检测，粒度可能很短，只作为兼容字段、关键帧和证据来源；Chapter 和 StoryScene 用于全局结构与上下文。

```
用户需求
  → 检索命中上卷到 Beat
  → 多路构造 DirectorCandidate
  → LLM 选择 candidate_id
  → 本地解析为 beat 级 EditClip
  → Reviewer beat grounding
```

## 入口函数

```python
def run_director(
    video_id, prompt, style="emotional", target_duration=180,
    platform="general", character_perspective=None,
    narrative_structure="chronological", aspect_ratio="16:9", max_retries=3
) -> EditPlan
```

## 候选构造

Director 会先调用 `search_memory(memory, prompt, top_k=50)`，把命中的 shot 上卷到所属 beat；随后融合全量 beat 层信息构造内部候选：

```python
DirectorCandidate(
    candidate_id="beat:183",
    beat_index=183,
    start_time=2101.7,
    end_time=2106.7,
    shot_indices=[...],
    story_scene_index=38,
    chapter_index=8,
    edit_signal=EditSignal(...),
    narrative_signal=NarrativeSignal(...),
    recomposition_signal=RecompositionSignal(...),
)
```

候选来源包括：

| 来源 | 用途 |
|------|------|
| `search_memory()` | 用户语义需求相关的 shot，回溯到 beat |
| `beat_memory_units` | beat 摘要、台词、角色、EditSignal |
| `story_scenes` / `chapters` | 给 beat 补充所属故事场景和章节 |
| `edit_signals` | hook、剧情重要性、情绪、视觉、边界、剧透 |
| `narrative_signals` | 叙事弧位置、张力、信息密度、叙事功能 |
| `recomposition_signals` | 二创潜力、情感引用、脱上下文能力、平台格式 |
| `events` | 高重要事件对相关 beat 加权 |

## 本地评分

候选按任务意图加权。当前意图由 prompt/style/platform 的关键词粗分为 `trailer`、`recap`、`remix` 和 `highlight`。

- `trailer` 偏重 `hook_score`、`visual_impact`、`tension_level`、`boundary_quality`，并惩罚高剧透。
- `recap` 偏重 `plot_importance`、`information_density` 和重要事件。
- `remix` 偏重 `recomposition_signals`、`hook_score` 和情绪引用价值。
- `highlight` 使用 hook、剧情、情绪、视觉和边界的均衡分。

预告片会额外轻惩罚过长 beat，并在 prompt 中提示 3-12 秒的片段节奏、章节顺序和角色变化。解析阶段仍保持 beat 完整边界，但会对超过 12 秒的预告片 beat 自动设置最高 1.5x 的变速，避免少数长 beat 吃掉大部分时间线。

最终只把排序后的有限候选写入 prompt，避免 LLM 在 1000+ shot 中自由发散。

## Prompt 契约

Director system prompt、方案模板和审核失败重试片段统一维护在根目录 `prompt.py`。

Prompt 包含：

- 全片章节地图
- 主要人物
- beat 候选列表
- 高重要事件
- 剪辑参数

LLM 只能输出候选中的 `candidate_id`：

```json
{
  "title": "剪辑方案标题",
  "narrative_structure": "chronological",
  "plan_items": [
    {
      "candidate_id": "beat:183",
      "narrative_role": "climax",
      "selection_reason": "苏雨摘下面具的反转揭示",
      "characters": ["char_011"],
      "transition_in": "cut",
      "transition_out": "cut",
      "speed": 1.0,
      "audio_volume": 1.0,
      "evidence_refs": ["beat:183", "story_scene:38", "edit_signal:beat:183"]
    }
  ]
}
```

解析阶段不信任 LLM 自造时间；`source_start/source_end` 一律使用候选 beat 的真实边界。

## EditClip 输出

Director 输出 beat 级 `EditClip`：

```python
EditClip(
    source_unit_type="beat",
    source_beat_index=183,
    source_story_scene_index=38,
    source_scene_index=beat.shot_indices[0],  # 兼容旧字段
    source_start=beat.start_time,
    source_end=beat.end_time,
    evidence_refs=["beat:183", "story_scene:38", "edit_signal:beat:183"],
)
```

`source_scene_index` 保留给旧渲染和日志路径使用，语义上表示该 beat 的首个 shot，而不是片段只能落在单个 shot 内。

## 审核闭环

Reviewer 会检查：

- beat 是否存在
- 兼容 `source_scene_index` 是否属于该 beat
- clip 时间是否在 beat 边界内
- `evidence_refs` 是否引用了 `beat:{index}`
- 角色是否出现在 beat 或 beat 内 shot 的 MemoryUnit 中
- 高重要事件是否至少有代表性覆盖

审核未通过时，Director 会把反馈追加到 prompt 并重试，仍要求只从已有 `candidate_id` 中选择。
