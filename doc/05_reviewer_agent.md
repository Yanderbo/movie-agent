# Reviewer Agent (reviewer.py)

> 文件：`agents/reviewer.py`
> 职责：审核 EditPlan 质量，结合规则校验、Grounding 校验和 LLM 审核

## 总体流程

```
EditPlan + VideoMemory + user_prompt
    │
    ▼
┌──────────────────┐
│  规则校验         │  硬性结构检查
│  (6项)           │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  Grounding 校验   │  证据链完整性检查
│  (4项)           │
└────────┬─────────┘
         ▼
┌──────────────────┐
│  LLM 审核        │  语义层面审核
│  (Gemini)        │
└────────┬─────────┘
         ▼
    ReviewResult
```

## 入口函数

```python
def review_plan(plan: EditPlan, memory: VideoMemory, user_prompt: str) -> ReviewResult
```

返回 `ReviewResult`：
- `approved: bool` — 是否通过
- `score: float` — 0.0 ~ 1.0
- `feedback: str` — 总体评价
- `issues: list[str]` — 具体问题列表

## 校验项详解

### 规则校验（硬性）

| 序号 | 校验项 | 规则 | 严重性 |
|------|--------|------|--------|
| 1 | 片段数量 | 3 ≤ clips ≤ 20 | 🔴 严重（<3时直接拒绝） |
| 2 | 时长偏差 | 实际时长 vs 目标时长，偏差 ≤ 15% | ⚠️ 一般 |
| 3 | 场景引用 | `source_scene_index` 在 0 ~ max_scene_index 范围内 | 🔴 严重 |
| 4 | 时间合法性 | `source_start` ≥ scene.start_time - 0.5s，`source_end` ≤ scene.end_time + 0.5s | ⚠️ 一般 |
| 5 | 叙事完整 | clips 中包含 `hook` 角色（当 clips ≥ 3 时） | ⚠️ 一般 |
| 6 | 节奏多样性 | 不允许连续 3 个相同 `narrative_role` | ⚠️ 一般 |

**严重问题（无效场景 / 片段过少）直接返回 `approved=False, score=0.2`，跳过 LLM 审核。**

### Grounding 校验（新增）

`_grounding_check()` 实现以下 4 项检查：

#### 1. evidence_refs 非空

遍历每个 clip，检查：
- `evidence_refs` 列表不为空
- 不全是 `unchecked` 标记

报告：`"X/Y 个片段缺少有效的 evidence_refs"`

#### 2. 时间精度

将 clip 的 `source_start / source_end` 与对应 MemoryUnit 的时间范围对比：
- `source_start < mu.start_time - 0.5s` → 问题
- `source_end > mu.end_time + 0.5s` → 问题

#### 3. 角色一致性

检查 clip 中声称的 `characters` 是否在对应 MemoryUnit 的 `characters` 中出现。

例如：clip 说包含 `char_002`，但 MemoryUnit scene_5 的 characters 是 `[char_000, char_001]`，则报错。

#### 4. 高重要性事件覆盖率

- 收集 `importance >= 7` 的事件
- 通过 `event.scene_indices` 找到事件覆盖的 scene
- 检查 EditPlan 的 clips 是否覆盖了这些 scene
- 未覆盖的事件报告为 issue

### LLM 审核

`_llm_review()` 构造精简的 EditPlan 摘要（含证据信息）发给 Gemini：

```json
{
  "clips": [
    {
      "clip_index": 0,
      "source_scene_index": 5,
      "duration": 14.5,
      "narrative_role": "hook",
      "evidence_refs": ["search_result#scene_5"],
      "has_transcript": true,
      "has_vision": true
    }
  ]
}
```

LLM 检查的项目：时长偏差、片段数量、场景引用、时间合法性、**证据完整性**、叙事完整性、节奏多样性、连贯性。

## 评分逻辑

| 场景 | 处理 |
|------|------|
| 有严重规则问题 | `score=0.2, approved=False` |
| LLM 审核成功 | 使用 LLM 打的分，结合 `llm_approved and len(critical_issues) == 0` |
| LLM 审核失败 | fallback：`score = max(0, 1.0 - issues数 * 0.15)`，`approved = issues ≤ 2 and score ≥ 0.6` |
