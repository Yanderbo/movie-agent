# API 调用与 Prompt 清单

> 代码入口：`utils/llm_client.py`、`pipeline/indexer.py`、`memory/search.py`
>
> 静态提示词唯一维护位置：仓库根目录 `prompt.py`

## 实际 HTTP 端点

| 端点 | 发起位置 | 请求输入 | 原始响应 | 调用方得到的输出 |
|------|----------|----------|----------|------------------|
| `POST {LLM_API_BASE}/chat/completions` | `LLMClient._request()` | `model`、`messages`、`temperature`、`stream=true`，可选 `max_tokens` / `response_format`；媒体以 base64 data URL 放入 `messages` | SSE 流，内容位于 `choices[0].delta.content` 或 `choices[0].text` | `chat*()` 返回拼接后的文本；业务调用通常再用 `parse_json()` 转为 `dict` / `list` |
| `POST {LLM_API_BASE}/embeddings` | `pipeline/indexer._generate_embeddings()` | `model` + 最多 20 条文本组成的 `input` | `data[]`，每项包含 `index` 和 `embedding` | 与输入顺序一致的 `list[list[float]]`，用于构建 FAISS |
| `POST {LLM_API_BASE}/embeddings` | `memory/search._get_query_embedding()` | `model` + 单条查询组成的 `input` | `data[0].embedding` | 单个 `list[float]`，用于 FAISS 查询 |
| `POST {LLM_API_BASE}/chat/completions` | `doc/deprecated/gemini_under_video.py::VideoParser.request_api()` | 外部传入 prompt + 压缩视频 base64 + reasoning 配置 | SSE 流 | 旧版独立脚本返回的解析结果字典；当前产品代码不导入此文件 |

所有 `LLMClient.chat*()` 逻辑调用在请求失败、流中断或空响应时，底层最多发送 1 次初始请求和 3 次重试请求，等待间隔为 5、10、20 秒。MinuteChunk 还会在内容无法解析时额外执行一次完整逻辑调用，因此单个 chunk 最多可能触发 8 次 HTTP 请求尝试。

## Understand 主链路

| # | 调用位置 / Prompt | 调用粒度 | 输入 | 期望输出 / 本地消费 |
|---|-------------------|----------|------|---------------------|
| 1 | `minute_chunk._call_gemini_structured()` / `CHUNK_UNDERSTAND_PROMPT` | 每个 MinuteChunk；解析失败时再调用一次 | chunk 镜头边界、角色档案、视频片段、带标签的脸谱图片 | JSON 对象：`transcripts`、`per_shot`、`character_updates`、`character_merge_suggestions`、`cross_shot`；回填 ASR/Vision/OCR/Audio/角色/对齐 |
| 2 | `beat_detect._detect_batch()` / `BEAT_PROMPT_TEMPLATE` | 每 20 个 prior 候选或 shot | shot 台词、画面摘要、Step 5 prior、正式角色名册 | Beat JSON 数组；本地校验角色白名单、连续覆盖并生成 `Beat[]` |
| 3 | `story_scene_detect._detect_segment()` / `SCENE_PROMPT_TEMPLATE` | 每 40 个 Beat | Beat 时间、类型、描述、情绪、人物摘要 | StoryScene JSON 数组；本地聚合人物并补齐未覆盖 Beat |
| 4 | `chapter_detect._detect_via_llm()` / `CHAPTER_PROMPT_TEMPLATE` | 长度不少于 600 秒且 StoryScene 多于 3 个时调用一次 | 全部 StoryScene 摘要和视频时长 | Chapter JSON 数组；本地聚合人物并补齐未覆盖 StoryScene |
| 5 | `event._extract_events_segment()` / `EVENT_PROMPT_TEMPLATE` | 每 1800 秒一段 | 分段 StoryScene/Beat 摘要、均匀采样台词/画面、人物名册、段边界 | Event JSON 数组；本地归一化时间并生成 `Event[]` |
| 6 | `event._extract_event_edges()` / `EDGE_PROMPT_TEMPLATE` | 至少 2 个事件时调用一次，最多输入 60 个事件 | 事件时间、类型、重要性和描述 | EventEdge JSON 数组；过滤无效事件索引后生成 `EventEdge[]` |
| 7 | `character_arc.analyze_character_arcs()` / `ARC_PROMPT_TEMPLATE` | 每个视频一次 | 人物档案、最多 60 个事件、人物共现信息 | JSON 对象：`arcs` + `relations`；回填角色弧并生成关系图 |
| 8 | `edit_signal._compute_signals_for_units()` / `SIGNAL_PROMPT_TEMPLATE` | shot / beat / story_scene 各自每 15 个一批 | 单元摘要、人物、相关事件、台词和画面 | 8 维 EditSignal JSON 数组；按批次位置绑定到本地单元 |
| 9 | `edit_signal._compute_narrative_signals()` / `NARRATIVE_PROMPT` | Beat + StoryScene 每 15 个一批 | 单元时间、描述、人物 | NarrativeSignal JSON 数组 |
| 10 | `edit_signal._compute_recomposition_signals()` / `RECOMP_PROMPT` | 目标 Beat 每 10 个一批 | Beat 内容、情绪和台词摘要 | RecompositionSignal JSON 数组 |
| 11 | `memory_builder.assign_character_roles()` / `ROLE_PROMPT_TEMPLATE` | 每个视频一次 | 人物出镜、重要性、台词量和事件量 | `{character_id: role}` JSON 对象；角色值限制为主角/反派/配角/路人 |
| 12 | `indexer._generate_embeddings()` | 每 20 条 MemoryUnit 文本一批 | 最长 8000 字符的 MemoryUnit 文本 | 文本向量数组，用于 `faiss.index` |

## Search / Edit

| # | 调用位置 / Prompt | 调用条件 | 输入 | 期望输出 / 本地消费 |
|---|-------------------|----------|------|---------------------|
| 13 | `search._get_query_embedding()` | FAISS 和 embedding 配置可用时，每次搜索一次 | 最长 8000 字符的用户查询 | 单个查询向量 |
| 14 | `search._llm_rerank()` / `SEARCH_RERANK_PROMPT_TEMPLATE` | 启用语义重排且候选多于 3 个时，每次搜索一次 | 用户查询 + 最多 20 个候选摘要 | `index` / `relevance_score` JSON 数组；与原始分数组合后重排 |
| 15 | `director.run_director()` / Director system + user prompt | 每次方案生成最多 `max_retries` 次，默认 3 次 | 用户需求、参数、故事地图、人物、事件、最多 48 个 Beat 候选和审核反馈 | EditPlan JSON 对象；仅接受候选中的 `candidate_id`，时间边界由本地候选回填 |
| 16 | `reviewer._llm_review()` / Reviewer system + user prompt | 规则/grounding 无严重问题时，每个 Director 方案一次 | 用户需求、剪辑参数、精简 EditPlan、视频信息 | `{approved, score, feedback, issues}` JSON 对象 |

## 兼容模块调用

以下调用点仍保留在代码中，但 v4.1 默认 Understand 主链路已由 MinuteChunk 替代。

| # | 调用位置 / Prompt | 调用粒度 | 输入 | 期望输出 / 本地消费 |
|---|-------------------|----------|------|---------------------|
| 17 | `asr._transcribe_window()` / `ASR_PROMPT` | 每个音频窗口 | 音频文件 | 带相对时间、文本、speaker、type 的 JSON 数组；转换为全片时间 |
| 18 | `audio_analysis._analyze_window()` / `AUDIO_PROMPT_TEMPLATE` | 每个时间窗口 | shot 边界、已有台词和画面摘要 | 每 shot 音频韵律 JSON 数组 |
| 19 | `vision._analyze_multi_frame()` / `VISION_PROMPT_MULTI` | 每个多关键帧 shot | 同一 shot 的多张图片 | OCR + Vision JSON 对象 |
| 20 | `vision._analyze_single()` / `VISION_PROMPT_SINGLE` | 每个单关键帧 shot或多帧失败回退 | 单张关键帧 | OCR + Vision JSON 对象 |
| 21 | `vision._analyze_batch()` / `BATCH_VISION_PROMPT` | 一批单关键帧 shot | 多张关键帧 | 按图片顺序排列的 OCR + Vision JSON 数组 |
| 22 | `character._detect_faces_with_gemini()` / `CHARACTER_FACE_DETECT_PROMPT` | 最多采样 20 张关键帧，逐帧调用 | 单张关键帧 | 画面人物外观描述 JSON 数组 |
| 23 | `character._describe_character()` / `CHARACTER_DESCRIPTION_PROMPT` | 每个需要描述的人物一次 | 人物缩略图 | 不超过 50 字的外观描述文本 |
| 24 | `speaker_bind.bind_speakers_to_characters()` / `BIND_PROMPT_TEMPLATE` | 每个视频一次 | speaker、人物和共现统计 | `{speaker_id: character_id | null}` JSON 对象 |

## Prompt 管理约定

- `prompt.py` 只保存静态 system prompt、user prompt 模板和可复用提示词片段。
- 业务模块继续负责构造镜头、事件、候选、人物等动态上下文，并负责校验和解析输出。
- 用户在 CLI 输入的剪辑需求属于运行时数据，不定义在 `prompt.py`。
- 修改输出 JSON 契约时，必须同时检查对应解析代码、`models/schemas.py` 和相关模块文档。
