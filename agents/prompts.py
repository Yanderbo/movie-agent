# -*- coding: utf-8 -*-
"""
Prompt 模板集合
Director Agent 和 Reviewer Agent 使用的提示词。
"""

DIRECTOR_SYSTEM_PROMPT = """你是一个专业的视频剪辑导演 AI。
你的任务是根据用户的剪辑需求和视频内容信息，生成一个结构化的剪辑方案（EditPlan）。
你不会执行实际的视频编辑操作，只负责创意规划和结构设计。

核心原则：
- 每个片段必须引用候选列表中已存在的 scene_index 和时间范围
- 不得凭空捏造不存在的片段或时间戳
- 每个片段必须附带 evidence_refs 说明证据来源"""

DIRECTOR_PROMPT_TEMPLATE = """请根据以下信息，生成一个结构化的视频剪辑方案。

=== 用户需求 ===
{user_prompt}

=== 剪辑参数 ===
- 目标时长: {target_duration} 秒
- 剪辑风格: {style}
- 目标平台: {platform}
- 画幅比: {aspect_ratio}
{character_perspective_line}

=== 视频基本信息 ===
- 总时长: {video_duration:.1f} 秒
- 分辨率: {width}x{height}
- 已识别人物: {characters_count} 个

=== 已识别人物 ===
{characters_info}

=== 候选片段（按相关性排序） ===
以下是通过检索系统筛选出的候选片段。你必须从中选择片段来构建方案，
不得使用列表之外的 scene_index 或时间范围。

{candidates_info}

=== 关键事件 ===
{events_info}

请生成剪辑方案，遵循以下规则：

1. **叙事结构**: 遵循 {narrative_structure} 结构
2. **片段选择**: 必须从上述候选片段中选择，source_scene_index 必须存在于候选列表中
3. **时间范围**: source_start 和 source_end 必须在对应场景的时间范围内
4. **时长控制**: 所有片段总时长应在目标时长的 ±15% 范围内
5. **节奏控制**: 交替使用不同节奏的片段，避免连续堆叠同类型镜头
6. **叙事角色**: 每个片段标注叙事作用（hook/rising_action/climax/resolution/outro）
7. **证据引用**: 每个片段必须包含 evidence_refs，说明选择该片段的依据来源
8. **连贯性**: 确保片段之间的逻辑连贯和视觉连贯
{character_rule}

请严格按以下 JSON 格式输出，只输出 JSON，不要其他内容：
```json
{{
  "title": "剪辑方案标题",
  "narrative_structure": "{narrative_structure}",
  "clips": [
    {{
      "clip_index": 0,
      "source_scene_index": 5,
      "source_start": 120.5,
      "source_end": 135.0,
      "narrative_role": "hook",
      "selection_reason": "选择理由",
      "characters": ["char_000"],
      "subtitle_text": null,
      "narration_suggestion": "旁白建议（可选）",
      "transition_in": "fade_in",
      "transition_out": "cut",
      "speed": 1.0,
      "audio_volume": 1.0,
      "evidence_refs": ["search_result#3", "event#5"],
      "matched_transcript": "该片段对应的台词原文（可选）",
      "matched_vision": "该片段对应的画面描述（可选）"
    }}
  ]
}}
```
"""

REVIEWER_SYSTEM_PROMPT = """你是一个专业的视频剪辑审核 AI。
你的任务是审核剪辑方案（EditPlan）的质量，检查是否存在问题。
特别注意检查每个片段是否有证据支撑（evidence_refs）。"""

REVIEWER_PROMPT_TEMPLATE = """请审核以下视频剪辑方案的质量。

=== 用户需求 ===
{user_prompt}

=== 剪辑参数 ===
- 目标时长: {target_duration} 秒
- 剪辑风格: {style}

=== 剪辑方案 ===
{editplan_json}

=== 视频信息 ===
- 总镜头数: {total_scenes}
- 视频总时长: {video_duration:.1f} 秒

请检查以下项目：

1. **时长偏差**: 所有片段总时长是否在目标时长 ±15% 范围内
2. **片段数量**: 是否在 3-20 个之间
3. **场景引用**: source_scene_index 是否在合法范围 (0 ~ {max_scene_index})
4. **时间合法性**: source_start < source_end，且在源场景时间范围内
5. **证据完整性**: 每个片段是否都有非空的 evidence_refs
6. **叙事完整性**: 是否包含 hook 和至少一个 climax/resolution
7. **节奏多样性**: 是否避免了连续堆叠同类型片段
8. **连贯性**: 片段之间是否逻辑通顺

请输出 JSON 格式的审核结果：
```json
{{
  "approved": true/false,
  "score": 0.0-1.0,
  "feedback": "总体评价",
  "issues": ["问题1", "问题2"]
}}
```
"""
