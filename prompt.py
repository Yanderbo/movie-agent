# -*- coding: utf-8 -*-
"""
全项目 Prompt 模板集合。

业务模块负责拼装动态上下文和解析结果；静态提示词统一在此维护。
"""

DIRECTOR_SYSTEM_PROMPT = """你是一个专业的视频剪辑导演 AI。
你的任务是根据用户的剪辑需求和视频内容信息，生成一个结构化的剪辑方案（EditPlan）。
你不会执行实际的视频编辑操作，只负责创意规划和结构设计。

核心原则：
- 每个片段必须引用候选列表中已存在的 candidate_id
- 不得凭空捏造不存在的 beat、scene 或时间戳
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

=== 剪辑节奏策略 ===
{tempo_guidance}

=== 视频基本信息 ===
- 总时长: {video_duration:.1f} 秒
- 分辨率: {width}x{height}
- 已识别人物: {characters_count} 个

=== 已识别人物 ===
{characters_info}

=== 全片故事地图 ===
{story_map_info}

=== Beat 候选片段（按导演分数排序） ===
以下候选已经由系统根据检索命中、叙事层级、剪辑信号和二创信号筛选。
你必须从中选择 candidate_id 来构建方案，不得使用列表之外的 beat 或时间范围。

{candidates_info}

=== 关键事件 ===
{events_info}

请生成剪辑方案，遵循以下规则：

1. **叙事结构**: 遵循 {narrative_structure} 结构
2. **片段选择**: 必须从上述候选片段中选择，candidate_id 必须存在于候选列表中
3. **时间范围**: 不要自造 source_start/source_end；系统会使用候选 beat 的真实时间范围
4. **时长控制**: 所有片段总时长应在目标时长的 ±15% 范围内
5. **节奏控制**: 交替使用不同节奏的片段，避免连续堆叠同类型镜头；预告片不要让单个长 beat 占据过多时长
6. **叙事角色**: 每个片段标注叙事作用（hook/rising_action/climax/resolution/outro）
7. **证据引用**: 每个片段必须包含 evidence_refs，说明选择该片段的依据来源
8. **连贯性**: 确保片段之间的逻辑连贯和视觉连贯
{character_rule}

请严格按以下 JSON 格式输出，只输出 JSON，不要其他内容：
```json
{{
  "title": "剪辑方案标题",
  "narrative_structure": "{narrative_structure}",
  "plan_items": [
    {{
      "clip_index": 0,
      "candidate_id": "beat:183",
      "narrative_role": "hook",
      "selection_reason": "选择理由",
      "characters": ["char_000"],
      "subtitle_text": null,
      "narration_suggestion": "旁白建议（可选）",
      "transition_in": "fade_in",
      "transition_out": "cut",
      "speed": 1.0,
      "audio_volume": 1.0,
      "evidence_refs": ["beat:183", "story_scene:38", "edit_signal:beat:183"]
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
3. **来源引用**: beat 级片段的 source_beat_index 和兼容 source_scene_index 是否合法
4. **时间合法性**: source_start < source_end，且在源 beat 时间范围内
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


# ═══════════════════════════════════════════════════════════════
# Understand 主链路
# ═══════════════════════════════════════════════════════════════

CHUNK_UNDERSTAND_PROMPT = """你是一个专业的影视分析系统。请分析这段视频片段（{start:.1f}s - {end:.1f}s），该片段包含 {n_shots} 个镜头。

== 镜头边界 ==
{shot_boundaries}

{character_section}

请完成以下分析，以 JSON 格式输出：

== 强制覆盖要求 ==
1. per_shot 必须输出 {n_shots} 个对象，逐一覆盖“镜头边界”中的每个 local_shot_index，不要只分析前几个镜头。
2. 每个 per_shot 对象必须同时包含 local_shot_index 和 scene_index；scene_index 必须使用镜头边界中给出的全局 scene_index。
3. 每个镜头的 vision/audio 都要填写。即使镜头很短或画面较模糊，也要给出最合理的简短描述；确实无法辨认时写“无法判断”，不要留空字符串。
4. ocr_texts 只有在画面没有文字时才可以为空数组。
5. characters_present 只填写画面中真实可见、且有足够视觉证据识别的人物/实体；仅被台词、旁白、剧情提到，或只是和参考脸谱相似但看不清脸时，不要填入该角色。无法匹配已知角色时，只有片中明确提供姓名、职位、称谓、关系或稳定剧情身份的人物才使用 unknown_N 并同步写入 character_updates；完全匿名、没有身份描述的人物不要记录到 characters_present。

A. **ASR 语音转录** — 逐句转录音频中的语音
   - start_time/end_time 必须标注相对于片段起始的时间戳，不要使用全片绝对时间戳
   - 说话人用角色ID标注（如 char_000）；无法匹配已知角色但片中有明确身份描述时，用 "unknown_1", "unknown_2" 等临时编号，并在 character_updates 补全身份；完全匿名且没有身份描述时统一写 "unknown"
   - type: dialogue / narration / voiceover

B. **逐镜头画面分析** — 对每个镜头（按上方边界），分析：
   - description: 画面描述
   - objects: 检测到的物体
   - mood: 情绪
   - scene_type: 场景类型
   - camera_motion: 镜头运动 (static/pan/tilt/zoom/tracking/handheld)
   - shot_scale: 景别 (close_up/medium/long 等)
   - action_description: 动作描述
   - ocr_texts: 画面文字
   - characters_present: 画面中可见且可识别的角色ID列表；不确定匹配哪个已知角色、但片中明确提供身份描述时使用 "unknown_1", "unknown_2" 并同步补全 character_updates；没有身份描述的匿名人物不要记录

C. **逐镜头音频特征**
   - has_music, music_mood, has_sfx, sfx_tags
   - silence_ratio(0-1), speech_rate(slow/normal/fast)
   - volume_peak(0-1), speech_emotion

D. **角色动态更新**
   - 已知角色的新信息：新称呼/别名、形象变化、关键行为
   - 对无法匹配已知角色、但片中明确出现姓名、职位、称谓、关系或稳定剧情身份的人物，沿用同一个 unknown_N，在 new_names 和 identity_description 中尽量补全；unknown_N 仅作为内部关联编号，不要把它当人物名称
   - 没有任何身份描述的匿名人物不要写入 character_updates，也不要仅凭外观或短暂出现创建角色档案
   - 新发现的非人类实体：名称和简述（动物/机器人等）

E. **跨镜头分析**
   - narrative_continuity: 叙事脉络
   - emotion_arc: 情绪变化
   - suggested_beats: 建议的节拍分组（哪些连续镜头属于同一叙事节拍），用镜头索引表示

F. **角色身份合并建议**（仅在你有充分证据时才填写）
   - 如果你发现两个角色ID实际上是同一个人（同一演员），报告合并建议
   - 需要脸部特征、声音、剧情连续性、称呼等多项证据一致才能确认
   - 如果只是"看起来有点像"但不确定，不要报告

输出 JSON（只输出JSON）：
```json
{{
  "transcripts": [
    {{"start_time": 0.0, "end_time": 3.5, "text": "...", "speaker": "char_000", "type": "dialogue"}}
  ],
  "per_shot": [
    {{
      "local_shot_index": 0,
      "scene_index": {first_shot_index},
      "vision": {{"description": "简要描述该镜头画面", "objects": [], "mood": "无法判断", "scene_type": "无法判断", "camera_motion": "static", "shot_scale": "无法判断", "action_description": "简要描述该镜头动作", "ocr_texts": []}},
      "audio": {{"has_music": false, "music_mood": "无法判断", "has_sfx": false, "sfx_tags": [], "silence_ratio": 0, "speech_rate": "normal", "volume_peak": 0, "speech_emotion": "neutral"}},
      "characters_present": []
    }}
  ],
  "character_updates": [
    {{"character_id": "char_000", "new_names": [], "identity_description": "", "appearance_change": "", "key_action": ""}}
  ],
  "character_merge_suggestions": [],
  "cross_shot": {{
    "narrative_continuity": "",
    "emotion_arc": "",
    "suggested_beats": [[0, 1, 2], [3, 4]]
  }}
}}
```"""

CHARACTER_SECTION_TEMPLATE = """== 已知角色脸谱 ==
下方附件中，每张参考脸谱前都标注了对应的角色ID、序号和来源时间。
请根据脸部五官特征（而非服装或发型）匹配角色身份。

== 角色匹配规则 ==
1. 以脸部特征（五官、脸型）为主要依据，同一人可能换装/换发型。
2. 如果人物无法确定匹配哪个已知角色，但片中明确提供姓名、职位、称谓、关系或稳定剧情身份，使用 "unknown_1", "unknown_2" 等临时内部编号，并在 character_updates 补全身份信息。
3. 如果人物既无法匹配、片中也没有任何身份描述，不要记录为角色；说话人可统一标注为 "unknown"。
4. 如果出现非人类实体（动物/机器人等），在 character_updates 中报告。

== 已知角色档案 ==
{profiles_text}"""

PROFILE_ONLY_CHARACTER_SECTION_TEMPLATE = """== 角色信息 ==
当前无参考脸谱图片，仅有文字档案。请结合以下档案信息识别画面中的人物。
如果无法确定匹配哪个角色，只有片中明确提供姓名、职位、称谓、关系或稳定剧情身份时才使用 "unknown_1", "unknown_2" 等临时内部编号，并在 character_updates 补全身份。

== 角色匹配规则 ==
1. 以脸部特征（五官、脸型）为主要依据，同一人可能换装/换发型。
2. 完全匿名、没有身份描述的人物不要记录为角色；说话人可统一标注为 "unknown"。
3. 不要强行匹配，也不要编造姓名、职位或关系。
4. 如果出现非人类实体（动物/机器人等），在 character_updates 中报告。

== 已知角色档案 ==
{profiles_text}"""

NO_CHARACTER_SECTION = """== 角色信息 ==
尚无已知角色。请优先从台词、字幕、OCR、称呼和剧情关系中识别人物的具体姓名、职位、称谓或稳定剧情身份。
只有获得这类明确身份描述时，才用 "unknown_1", "unknown_2" 等临时内部编号关联说话人、characters_present 和 character_updates，并在 new_names / identity_description 中尽量补全。
完全匿名、没有身份描述的人物无需记录为角色；其说话人统一标注为 "unknown"。不要仅凭外观或短暂出现创建人物档案。"""

BEAT_PROMPT_TEMPLATE = """你是一个专业的影视叙事分析师。请分析以下镜头序列，将连续镜头按"叙事节拍"（Beat）分组。

一个 Beat 是由若干连续镜头组成的叙事微单元，通常对应：
- 一段完整的对话
- 一个连续的动作序列
- 一个情绪转折过程
- 一段环境展示/空镜
- 一个蒙太奇段落

=== 镜头列表 ===
{shots_info}

=== Step 5 先验 Beat 候选（已转换为全局 shot index） ===
{prior_info}

=== 已知角色名册 ===
（characters 字段只能填写以下角色 ID；画面中出现但不在名册内的人，不要写入 characters，可在 description 中描述，不要编造新的 char_ ID）
{character_roster}

请将这些镜头分组为若干 Beat，输出 JSON 数组。每个 Beat 包含：
- beat_index: 从 {beat_offset} 开始编号
- shot_indices: 包含的镜头索引列表（必须连续）
- beat_type: 类型（setup / confrontation / resolution / transition / montage / dialogue / action / reveal）
- description: 这个节拍讲了什么（一句话）
- emotion: 主要情绪
- intensity: 戏剧强度 (0.0 - 1.0)
- characters: 涉及的人物 ID 列表

规则：
1. 相邻且叙事连贯的镜头归为同一 Beat
2. 当场景/话题/情绪发生明显转换时，开启新 Beat
3. 每个 Beat 通常包含 2-8 个镜头，但不强制
4. 所有镜头都必须被分配到某个 Beat
5. characters 字段只能填写"已知角色名册"中的 ID；无法对应名册的人不要写入 characters，只在 description 中描述，不要编造新的 char_ ID
6. Step 5 先验是参考；普通 Prior Beat 来自单个 chunk 内部，边界基本可信，除非台词/画面明显冲突，不要过度重切
7. Boundary Fused Prior 是相邻 chunk 的边界融合候选（前一个 chunk 最后 1 个 prior + 后一个 chunk 第 1 个 prior），只有这类候选需要重点判断是否合并或拆分
8. 输出 shot_indices 必须使用上方 Shot 的全局索引，不要输出 chunk 内 local index

只输出 JSON：
```json
[
  {{
    "beat_index": {beat_offset},
    "shot_indices": [{example_shot_indices}],
    "beat_type": "dialogue",
    "description": "男女主角在咖啡厅讨论计划",
    "emotion": "轻松",
    "intensity": 0.3,
    "characters": ["char_000", "char_001"]
  }}
]
```
"""

SCENE_PROMPT_TEMPLATE = """你是一个专业的影视叙事分析师。请将以下"剧情节拍"（Beat）序列聚合为"故事场景"（StoryScene）。

一个 StoryScene 是由若干连续 Beat 组成的完整叙事场景，通常对应：
- 同一个地点/环境中发生的一系列事件
- 一段完整的情节单元（开始 → 发展 → 结束）
- 一个戏剧冲突的完整过程

=== Beat 列表 ===
{beats_info}

请将这些 Beat 分组为若干 StoryScene，输出 JSON 数组。每个 StoryScene 包含：
- beat_indices: 包含的 Beat 索引列表（必须连续）
- location: 场景地点/环境描述
- description: 这个场景的核心内容（一两句话）
- plot_function: 叙事功能（setup / inciting_incident / rising / climax / falling / resolution / epilogue）

规则：
1. 地点或情境发生大变化时，开启新 StoryScene
2. 一个 StoryScene 通常包含 2-6 个 Beat
3. 所有 Beat 都必须被分配到某个 StoryScene

只输出 JSON：
```json
[
  {{
    "beat_indices": [0, 1, 2],
    "location": "咖啡厅",
    "description": "男女主角在咖啡厅重逢并讨论过去的误会",
    "plot_function": "inciting_incident"
  }}
]
```
"""

CHAPTER_PROMPT_TEMPLATE = """你是一个专业的影视叙事分析师。请将以下"故事场景"（StoryScene）序列聚合为"章节"（Chapter）。

一个 Chapter 是一个完整的叙事大段落，通常对应：
- 电影中的一个"幕"（Act）
- 一个重要的情节阶段（如：序幕、铺垫、发展、高潮、结局）
- 一个独立的叙事主题单元

视频总时长: {duration:.0f}秒

=== StoryScene 列表 ===
{scenes_info}

请将这些 StoryScene 分组为若干 Chapter。输出 JSON 数组：
- title: 章节标题（简短有力）
- story_scene_indices: 包含的 StoryScene 索引列表（必须连续）
- description: 章节核心内容（一两句话）
- chapter_type: 叙事类型（prologue / act_1 / act_2 / act_3 / climax_act / epilogue / flashback）
- theme: 本章主题关键词
- mood_progression: 情绪走势描述（如"从紧张到释然"）

规则：
1. 一个 Chapter 通常包含 2-5 个 StoryScene
2. 叙事主题或情绪发生重大转变时，开启新 Chapter
3. 所有 StoryScene 都必须被分配到某个 Chapter

只输出 JSON：
```json
[
  {{
    "title": "命运的邂逅",
    "story_scene_indices": [0, 1, 2],
    "description": "男女主角在异国他乡意外相遇并开始合作",
    "chapter_type": "act_1",
    "theme": "相遇与信任建立",
    "mood_progression": "从陌生到好奇"
  }}
]
```
"""

EVENT_PROMPT_TEMPLATE = """你是一个专业的视频内容分析师。基于以下视频内容信息，提取关键事件。

视频总时长: {duration:.1f} 秒
当前分析段: {segment_start:.1f}s - {segment_end:.1f}s

=== 叙事层级摘要（优先参考） ===
{hierarchy_text}

=== 台词（按时间均匀采样） ===
{transcripts_text}

=== 画面摘要（按时间均匀采样） ===
{vision_text}

=== 已识别人物 ===
{characters_text}

请提取视频中的关键事件，每个事件代表一个有意义的叙事单元。

要求：
1. 事件按时间顺序排列
2. 每个事件覆盖一段连续时间，start_time/end_time 必须使用全片绝对时间，并落在当前分析段内
3. 事件类型包括：开场、对话、冲突、转折、高潮、结局、日常、回忆、独白、追逐、浪漫、搞笑、悲伤、悬疑
4. importance 用 1-10 评分，高潮和转折事件分数更高
5. 标注涉及的人物 ID
6. 描述每个事件的核心内容
7. evidence: 列出支撑该事件判断的证据来源（如 "transcript:台词内容", "vision:画面描述", "audio:音效/音乐"）
8. confidence: 该事件抽取的置信度 0-1

输出 JSON 数组，只输出 JSON：
```json
[
  {{
    "event_index": 0,
    "start_time": 0.0,
    "end_time": 30.0,
    "event_type": "开场",
    "description": "事件描述",
    "characters": ["char_000", "char_001"],
    "emotion": "平静",
    "importance": 5,
    "evidence": ["transcript:角色A说了xxx", "vision:画面中出现了xxx"],
    "confidence": 0.85
  }}
]
```
"""

EDGE_PROMPT_TEMPLATE = """你是一个专业的叙事分析师。请分析以下事件之间的关系。

=== 事件列表 ===
{events_text}

请分析事件之间的因果关系、铺垫关系、反转关系、冲突升级、结果关系和平行关系。

关系类型说明：
- cause: A 导致了 B（因果）
- foreshadow: A 为 B 埋下了伏笔（铺垫）
- reversal: B 是 A 的反转/意外
- escalation: B 是 A 的冲突升级
- resolution: B 是 A 的解决/结果
- parallel: A 和 B 是平行/对照的情节线

同时为每条关系提供：
- evidence: 支撑该关系的证据（引用具体的台词或画面）
- confidence: 关系推断的置信度 0-1
- relation_basis: 关系推断的依据说明

输出 JSON 数组，每个元素描述一条关系边。只关注重要的关系，不要过度连接。
只输出 JSON：
```json
[
  {{
    "source_event": 0,
    "target_event": 2,
    "relation_type": "cause",
    "description": "因为A发生了，所以导致了C",
    "strength": 0.8,
    "evidence": ["事件0中角色说了xxx", "事件2中画面出现了xxx"],
    "confidence": 0.75,
    "relation_basis": "角色A在事件0中的决定直接导致了事件2的冲突"
  }}
]
```
"""

ARC_PROMPT_TEMPLATE = """你是一个专业的影视叙事分析师。请分析以下人物的角色弧线和人物间关系。

=== 人物列表 ===
{characters_info}

=== 关键事件（时间顺序） ===
{events_info}

=== 人物间共现信息 ===
{cooccurrence_info}

请分析：

一、角色弧线（每个主要人物一条）
分析每个人物从开头到结尾的变化轨迹。

二、人物关系（每对有互动的人物一条）
分析人物间的关系类型和变化。

输出 JSON 对象，只输出 JSON：
```json
{{
  "arcs": [
    {{
      "character_id": "char_000",
      "arc_type": "growth",
      "arc_description": "从怯懦逐渐变得勇敢",
      "key_moments": [0, 3, 7],
      "emotion_trajectory": [
        {{"time": 10.0, "emotion": "恐惧", "intensity": 0.8}},
        {{"time": 60.0, "emotion": "决心", "intensity": 0.6}},
        {{"time": 120.0, "emotion": "勇气", "intensity": 0.9}}
      ]
    }}
  ],
  "relations": [
    {{
      "character_a": "char_000",
      "character_b": "char_001",
      "relation_type": "romantic",
      "description": "男女主角从相识到相爱",
      "strength": 0.8,
      "evolution": ["初识时互有好感", "经历考验后关系加深", "最终在一起"]
    }}
  ]
}}
```
"""

SIGNAL_PROMPT_TEMPLATE = """你是一个专业的影视剪辑师。请为以下视频片段计算剪辑信号。

=== 片段列表 ===
{segments_info}

请为每个片段评估以下 8 个信号（0.0 - 1.0 分）：

1. hook_score: 作为视频开头/钩子的适合度（画面冲击力、悬念、吸引力）
2. plot_importance: 对整体剧情的贡献度（核心剧情=高，过渡/日常=低）
3. emotional_intensity: 情绪表达强度（强烈情绪=高，平静=低）
4. visual_impact: 视觉冲击力（特殊构图/运镜/特效=高，普通对话=低）
5. independence_score: 片段独立性（单独观看也能理解=高，需要上下文=低）
6. continuity_dependency: 连续性依赖（必须与前后片段连看=高，可独立剪出=低）
7. boundary_quality: 剪辑边界质量（开头/结尾有自然停顿=高，在句中/动作中=低）
8. spoiler_level: 剧透程度（包含关键反转/结局=高，日常场景=低）

同时建议每个片段适合的用途（可多选）：
- hook: 适合作为开头钩子
- trailer: 适合放入预告片
- highlight: 适合作为精彩集锦
- recap: 适合用于剧情回顾
- climax_clip: 适合作为高潮片段
- character_intro: 适合用于人物介绍

输出 JSON 数组，只输出 JSON：
```json
[
  {{
    "unit_index": 0,
    "hook_score": 0.8,
    "plot_importance": 0.7,
    "emotional_intensity": 0.9,
    "visual_impact": 0.6,
    "independence_score": 0.5,
    "continuity_dependency": 0.4,
    "boundary_quality": 0.7,
    "spoiler_level": 0.3,
    "suggested_usage": ["hook", "highlight"]
  }}
]
```
"""

NARRATIVE_PROMPT = """你是一个专业的叙事结构分析师。请为以下视频片段评估叙事信号。

=== 片段列表 ===
{segments_info}

为每个片段评估：
1. arc_position: 在整体叙事弧中的位置 (0-1，开头=0，结尾=1)
2. tension_level: 张力水平 (0-1)
3. information_density: 信息密度 (0-1，新信息量/叙事推进度)
4. character_focus: 主要聚焦的角色 character_id
5. narrative_function: exposition/rising_action/climax/falling_action/resolution/transition/comic_relief
6. theme_relevance: 与主题相关度 (0-1)

输出 JSON 数组，只输出 JSON：
```json
[
  {{
    "unit_index": 0,
    "arc_position": 0.2,
    "tension_level": 0.3,
    "information_density": 0.7,
    "character_focus": "char_000",
    "narrative_function": "exposition",
    "theme_relevance": 0.6
  }}
]
```
"""

RECOMP_PROMPT = """你是一个短视频内容二次创作专家。请为以下视频片段评估二次创作价值。

=== 片段列表 ===
{segments_info}

为每个片段评估：
1. meme_potential: 梗/传播潜力 (0-1)
2. emotional_quotability: 情感引用潜力/"名场面"程度 (0-1)
3. context_freedom: 脱离上下文仍有意义的程度 (0-1)
4. remix_flexibility: 可重新组合的灵活度 (0-1)
5. platform_fit: 平台适配（douyin/bilibili/youtube 分别 0-1）
6. suggested_formats: 建议格式 (reaction/compilation/fancam/edit)

输出 JSON 数组，只输出 JSON：
```json
[
  {{
    "unit_index": 0,
    "meme_potential": 0.3,
    "emotional_quotability": 0.8,
    "context_freedom": 0.6,
    "remix_flexibility": 0.5,
    "platform_fit": {{"douyin": 0.7, "bilibili": 0.6, "youtube": 0.5}},
    "suggested_formats": ["compilation", "edit"]
  }}
]
```
"""

ROLE_PROMPT_TEMPLATE = """你是一个专业的影视分析师。请根据以下人物信息，判断每个人物在故事中的角色。

=== 人物列表 ===
{characters_info}

可选角色：
- male_lead: 男主角
- female_lead: 女主角
- villain: 反派
- supporting: 配角
- minor: 路人/群演

请为每个人物指定一个角色。考虑出镜时长、台词量、事件参与度等因素。
出镜最多且参与最多关键事件的通常是主角。

输出 JSON 对象，key 为 character_id，value 为角色：
```json
{{
  "char_000": "male_lead",
  "char_001": "female_lead",
  "char_002": "villain"
}}
```
"""


# ═══════════════════════════════════════════════════════════════
# Search / Edit Agent
# ═══════════════════════════════════════════════════════════════

SEARCH_RERANK_PROMPT_TEMPLATE = """请评估以下视频片段与查询的相关性，并重新排序。

查询: "{query}"

候选片段:
{summaries_text}

请返回 JSON 数组，按相关性从高到低排序。每个元素包含:
- index: 候选片段的编号（方括号中的数字）
- relevance_score: 相关性分数 (0.0 - 1.0)

只返回 relevance_score > 0.2 的片段。只输出 JSON：
```json
[{{"index": 0, "relevance_score": 0.9}}]
```"""

DIRECTOR_RETRY_FEEDBACK_TEMPLATE = """

=== 上次方案的审核反馈 ===
未通过原因: {feedback}
具体问题: {issues}
请只从候选 candidate_id 中重新选择，并修正上述问题。"""


# ═══════════════════════════════════════════════════════════════
# 兼容模块（v4.1 主链路已由 MinuteChunk 替代）
# ═══════════════════════════════════════════════════════════════

ASR_PROMPT = """你是一个专业的语音识别系统。请仔细听这段音频，将其中所有语音内容转录为文字。

要求：
1. 输出 JSON 数组格式，每个元素代表一句话/一段话
2. 每个元素包含以下字段：
   - start_time: 开始时间（秒，保留1位小数）— 相对于本段音频开头
   - end_time: 结束时间（秒，保留1位小数）— 相对于本段音频开头
   - text: 转录的文字内容
   - speaker: 说话人标识（如果能区分不同说话人，用 "speaker_1", "speaker_2" 等标识；无法区分则为 null）
   - type: 语音类型（"dialogue" 表示角色对白, "narration" 表示旁白, "voiceover" 表示画外音; 无法区分则默认 "dialogue"）
3. 按时间顺序排列
4. 只输出 JSON，不要其他内容
5. 如果某段时间没有语音，跳过即可
6. 确保时间戳尽可能准确

输出格式示例：
```json
[
  {"start_time": 0.0, "end_time": 3.5, "text": "大家好，欢迎收看", "speaker": "speaker_1", "type": "dialogue"},
  {"start_time": 4.2, "end_time": 8.1, "text": "今天我们来聊一个话题", "speaker": "speaker_1", "type": "dialogue"},
  {"start_time": 10.0, "end_time": 14.5, "text": "在那个遥远的年代...", "speaker": null, "type": "narration"}
]
```
"""

AUDIO_PROMPT_TEMPLATE = """你是一个专业的影视音频分析师。请分析以下视频片段的音频特征。

视频时间范围: {start:.1f}s - {end:.1f}s

=== 该时段已有的台词信息 ===
{transcript_info}

=== 该时段已有的画面信息 ===
{vision_info}

请为该时间段内的每个镜头（shot）分析音频特征。镜头列表：
{shots_info}

对每个镜头，请判断：
1. has_music: 是否有背景音乐
2. music_mood: 音乐情绪（energetic/melancholic/tense/romantic/epic/calm，无音乐则为空）
3. has_sfx: 是否有明显音效
4. sfx_tags: 音效类型列表（explosion/door_slam/footsteps/rain/wind/car/gunshot 等）
5. silence_ratio: 沉默/静音占比（0-1）
6. speech_rate: 语速（slow/normal/fast，无语音则为空）
7. volume_peak: 估计的音量峰值（0-1，高潮/爆炸=高，低语=低）
8. speech_emotion: 语音情绪（calm/angry/sad/happy/fearful/surprised/neutral，无语音则为空）

输出 JSON 数组，每个元素对应一个 shot，只输出 JSON：
```json
[
  {{
    "scene_index": 0,
    "has_music": true,
    "music_mood": "tense",
    "has_sfx": false,
    "sfx_tags": [],
    "silence_ratio": 0.1,
    "speech_rate": "fast",
    "volume_peak": 0.7,
    "speech_emotion": "angry"
  }}
]
```
"""

VISION_PROMPT_SINGLE = """你是一个专业的视频画面分析系统。请仔细观察这张视频截图，完成以下两项任务：

任务一：OCR 文字识别
识别画面中出现的所有文字内容（包括字幕、标题、标牌、屏幕文字等）。

任务二：画面摘要
对画面进行详细分析。

请以 JSON 格式输出，只输出 JSON，不要其他内容：
```json
{
  "ocr_texts": ["画面中的文字1", "画面中的文字2"],
  "description": "详细的画面描述，包含场景、人物动作、构图、光线等",
  "objects": ["检测到的物体1", "物体2"],
  "mood": "画面传达的情绪（如：紧张、温馨、悲伤、欢快、平静、激昂等）",
  "scene_type": "场景类型（如：对话、动作、空镜、过渡、特写、全景、追逐等）",
  "props": ["关键道具1", "道具2"]
}
```
"""

VISION_PROMPT_MULTI = """你是一个专业的视频画面分析系统。以下是同一个镜头（shot）内按时间顺序排列的多帧截图。
请综合分析这些帧，理解镜头内发生了什么。

请完成以下分析：
1. OCR：识别所有帧中出现的文字
2. 综合画面描述：描述这个镜头内的场景和核心内容
3. 动作/变化描述：通过对比各帧，描述镜头内发生的动作、运动和变化
4. 表情变化：如有人物，描述其表情在帧间的变化
5. 物体 & 关键道具
6. 情绪 & 场景类型
7. 镜头运动：分析镜头的运动方式（static/pan_left/pan_right/tilt_up/tilt_down/zoom_in/zoom_out/tracking/crane/handheld）
8. 人物互动：如有多人，描述人物间的互动方式（对话、肢体接触、对峙、合作等）
9. 景别：判断镜头的景别（extreme_close_up/close_up/medium_close/medium/medium_long/long/extreme_long）

请以 JSON 格式输出，只输出 JSON：
```json
{
  "ocr_texts": ["文字1", "文字2"],
  "description": "综合画面描述",
  "action_description": "动作/变化描述（从第1帧到最后1帧发生了什么）",
  "frame_descriptions": ["第1帧描述", "第2帧描述", "..."],
  "expression_changes": "人物表情变化描述（无人物则为空）",
  "objects": ["物体1", "物体2"],
  "props": ["关键道具1"],
  "mood": "整体情绪",
  "scene_type": "场景类型",
  "camera_motion": "镜头运动方式",
  "interaction_description": "人物互动描述（无互动则为空）",
  "shot_scale": "景别"
}
```
"""

BATCH_VISION_PROMPT = """你是一个专业的视频画面分析系统。以下是同一个视频中多个镜头的关键帧截图。
请为每张图片分别进行分析。

对每张图片，请分析：
1. OCR：画面中出现的文字
2. 画面描述：详细描述场景内容
3. 检测到的物体
4. 画面情绪
5. 场景类型
6. 关键道具

请以 JSON 数组格式输出，数组中每个元素对应一张图片（按顺序），只输出 JSON：
```json
[
  {
    "ocr_texts": ["文字1"],
    "description": "画面描述",
    "objects": ["物体1"],
    "mood": "情绪",
    "scene_type": "场景类型",
    "props": ["道具1"]
  },
  ...
]
```
"""

BIND_PROMPT_TEMPLATE = """你是一个专业的视频分析师。请根据以下信息，判断音频中的说话人与画面中的人物的对应关系。

=== 说话人信息 ===
{speakers_info}

=== 人物信息 ===
{characters_info}

=== 共现统计 ===
{cooccurrence_info}

请为每个说话人指定最可能对应的人物。如果无法确定，设为 null。

输出 JSON 对象，key 为说话人 ID，value 为人物 ID 或 null：
```json
{{
  "speaker_1": "char_000",
  "speaker_2": "char_001",
  "speaker_3": null
}}
```
"""

CHARACTER_FACE_DETECT_PROMPT = """请分析这张图片中出现的人物。
对每个人物，描述其外观特征（性别、大致年龄、发型、服装等）。

输出 JSON 数组，每个元素代表一个人物：
```json
[
  {"person_id": 1, "description": "外观描述", "position": "画面位置（左/中/右）"}
]
```
如果画面中没有人物，返回空数组 []。"""

CHARACTER_DESCRIPTION_PROMPT = """请简要描述这个人物的外观特征：
- 性别
- 大致年龄
- 发型和发色
- 服装
- 显著特征

用一段话简洁描述，不超过50字。"""
