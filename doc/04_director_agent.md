# Director Agent (director.py)

> 文件：`agents/director.py`
> 职责：基于 Video Memory 和用户需求，通过检索 + LLM 生成结构化 EditPlan

## 总体流程

```
加载 VideoMemory → 检索候选片段 → 构造 Prompt → LLM 生成 → 解析 + 证据填充
    → Reviewer 审核 → 不通过则加反馈重试（最多3轮） → 保存 EditPlan
```

## 入口函数

```python
def run_director(
    video_id, prompt, style="emotional", target_duration=180,
    platform="general", character_perspective=None,
    narrative_structure="chronological", aspect_ratio="16:9", max_retries=3
) -> EditPlan
```

## 短视频 vs 长视频

| 条件 | 策略 |
|------|------|
| 视频 ≤ 30min | 单次检索 + 单次 LLM 生成 |
| 视频 > 30min | 章节式滑窗规划（见下文） |

阈值由 `LONG_VIDEO_THRESHOLD = 1800` 控制。

## 短视频流程详解

### Step 1: 检索候选

```python
candidates = search_memory(memory, prompt, top_k=30)
valid_scene_indices = {r.scene_index for r in candidates}  # 白名单
```

### Step 2: 构造 Prompt

`_build_director_prompt()` 组装以下信息传给 LLM：

| 信息块 | 内容 |
|--------|------|
| 人物信息 | 每个 character 的名称、角色（male_lead/...）、speaker绑定、出场统计 |
| 候选片段 | 每个候选的 scene_index、时间范围、相关度、命中模态、台词、画面、上下文、证据 |
| 事件信息 | 事件的时间、类型、重要性、情绪、覆盖的 scene_indices |
| 剪辑参数 | 目标时长、风格、平台、画幅比、叙事结构 |

Prompt 要求 LLM：
- 只从候选列表中选择片段（不允许自造 scene_index）
- 每个片段必须包含 `evidence_refs`
- 遵循叙事结构和时长控制

### Step 3: 解析 + 证据填充

`_parse_editplan()` 的关键逻辑：

1. **校验 1**：`scene_index` 必须在 `memory.scenes` 中存在（硬性，跳过无效片段）
2. **校验 2**：`scene_index` 应在候选白名单中（软性，记录但不跳过，交给 Reviewer 决定）
3. **时间修正**：`source_start` 和 `source_end` 被钳制在 Scene 的时间范围内
4. **证据自动填充**：
   - 如果 LLM 没返回 `evidence_refs`，从候选结果中取 `source_refs`
   - 如果 LLM 没返回 `matched_transcript`，从候选结果中取 `transcript`
   - 如果 LLM 没返回 `matched_vision`，从候选结果中取 `vision_summary`
   - 不在候选中的片段标记为 `scene#X_unchecked`

### Step 4: 审核循环

```python
for attempt in range(max_retries):
    plan = 生成 + 解析
    review_result = review_plan(plan, memory, prompt)
    if review_result.approved:
        break
    else:
        director_prompt += 审核反馈  # 将反馈加入 prompt 重新生成
```

## 长视频滑窗规划

### `_run_chapter_planning()`

**Step 1: 分章节** — `_split_into_chapters()`

- 收集 `importance >= 6` 的重要事件作为章节分界点
- 分界点间距 ≥ 5 分钟
- 如果事件不够，按 10 分钟均匀切分
- 每个章节记录 `{start, end, events}`

**Step 2: 逐章节规划**

对每个章节独立执行：
1. `search_memory(memory, prompt, top_k=15, time_range=(start, end))`
2. 在 prompt 中标注当前是第几章节、本章目标时长
3. 单独调用 LLM 生成该章节的 clips

**Step 3: 合并** — `_merge_chapter_plans()`

- 将各章节的 clips 拼接
- 重新编号 `clip_index`
- 重新计算 `timeline_start / timeline_end`
- 统一填充证据

**Step 4: 审核**

整体审核一次（不循环重试，因为已经逐章节审过了）

## Prompt 模板

定义在 `agents/prompts.py`：

| 变量 | 说明 |
|------|------|
| `DIRECTOR_SYSTEM_PROMPT` | 系统角色定义 + 核心原则（不许自造片段） |
| `DIRECTOR_PROMPT_TEMPLATE` | 完整 prompt 模板，含 JSON 输出格式规范 |

LLM 输出 JSON 中每个 clip 必须包含：
```json
{
  "source_scene_index": 5,
  "source_start": 120.5,
  "source_end": 135.0,
  "narrative_role": "hook",
  "selection_reason": "...",
  "evidence_refs": ["search_result#3"],
  "matched_transcript": "...",
  "matched_vision": "..."
}
```
