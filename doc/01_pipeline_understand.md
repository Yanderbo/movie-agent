# 视频理解流水线 v4.1

## 总览

```
video.mp4
  → [1] Ingest（入库+压缩）
  → [2] Shot Detect（镜头切分）
  → [3] Keyframe（多帧关键帧采样）
  → [4] Face Cluster（人脸聚类+角色脸谱）        🆕
  → [5] MinuteChunk Understand（分钟级融合理解）  ⭐ 核心
  → [6] Beat Detect（剧情节拍检测）
  → [7] Story Scene Detect（故事场景检测）
  → [8] Chapter Detect（大段落检测）
  → [9] Event Graph + Character Arc（事件+弧线） 
  → [10] Final Build（信号+Memory+索引）
```

**vs v3 (17步)**：合并 ASR+Vision+Audio+Character+SpeakerBind+MultimodalAlign 为 MinuteChunk，
API 调用量从 ~257 次降至 ~30 次（↓87%）。

---

## 核心架构：双向层次化理解

```
自底向上（拼接聚合）                    自顶向下（回填拆分）
━━━━━━━━━━━━━━━━━━━                  ━━━━━━━━━━━━━━━━━━━
Shot (镜头切分)                        
  ↓ 按时间拼接                         
MinuteChunk (~2-3min)  ───Gemini──→  融合理解结果
                                        ↓ 按shot时间戳回填
                                     Shot级: vision/audio/ASR/角色
                                        ↓ 聚合
                                     Beat → StoryScene → Chapter
```

---

## 阶段输入/输出总表

| # | 阶段 | 主要输入 | 主要输出 | 数据如何流向下一步 |
|---|------|----------|----------|--------------------|
| 1 | Ingest | 用户传入视频路径、压缩阈值配置 | `original.*`、按需 `compressed.mp4`、`meta.json` | `meta.storage_path` 成为理解链路视频源；`original.*` 保留给 render |
| 2 | Shot Detect | `meta.storage_path`、镜头切分阈值 | `scenes/scenes.json` | Shot 时间边界成为后续关键帧、角色、ASR/视觉回填、叙事聚合的统一锚点 |
| 3 | Keyframe | `meta.storage_path`、`scenes/scenes.json` | `scenes/keyframes/*.jpg`、shot keyframe 路径 | 关键帧进入 Step 4 脸谱构建；不随 MinuteChunk 视频片段直接送入 Gemini |
| 4 | Face Cluster | keyframes、shots、人脸检测/聚类配置 | `characters/face_clusters.json`、`characters/char_XXX_gallery/` | 角色脸谱作为 Step 5 的身份先验；InsightFace 不可用时输出空脸谱 |
| 5 | MinuteChunk Understand | `meta.storage_path`、shots、face galleries、前序 `character_profiles.json` | `minute_chunks.json`、`transcripts.json`、`vision.json`、`ocr.json`、`audio_prosody.json`、`multimodal_alignments.json`、`characters.json`、`speaker_map.json`、可选 `character_identity_links.json` | 下游以回填后的 shot 级多模态散文件为基础；正式角色名册已过滤低证据临时角色 |
| 6 | Beat Detect | shots、`transcripts.json`、`vision.json`、`characters.json` | `beats.json`、回写 `scenes/scenes.json` 的 `beat_index` | Beat 成为 StoryScene、Event、Memory 和信号计算的基础叙事单元；缓存命中也会回填反向链接 |
| 7 | Story Scene Detect | beats、shots、台词/画面摘要、角色信息 | `story_scenes.json`、回写 `scenes/scenes.json` 的 `story_scene_index` | StoryScene 提供更高层剧情上下文，供 Chapter、Event 和 EditSignal 使用；缓存命中也会回填反向链接 |
| 8 | Chapter Detect | story_scenes、beats、shots、角色/情绪摘要 | `chapters.json` | Chapter 提供长视频大段落结构，供事件抽取、MemoryUnit 和检索索引使用 |
| 9 | Event Graph + Character Arc | shots、transcripts、vision、beats、story_scenes、chapters、characters、`character_profiles.json` | `events.json`、`event_graph.json`、`character_arcs.json`、`character_relations.json` | 事件、关系和人物弧线进入 final build，并作为 Director / Reviewer 的叙事证据 |
| 10 | Final Build | Step 1-9 所有散文件、三类信号配置、embedding 配置 | `edit_signals.json`、`narrative_signals.json`、`recomposition_signals.json`、`memory.json`、`index/` | Search 读取 `memory.json` / `index/`；Director / Reviewer 使用 Memory、事件、角色和信号证据；索引失败会中断 final_build |

边界说明：Step 5 `minute_chunk` 是 understand 主链路中需要视频片段做多模态理解的步骤；Step 6-10 不再打开原视频或压缩视频读取 shot 画面。后续视觉依据来自 `vision.json` / `ocr.json` / `multimodal_alignments.json`，需要引用画面文件时使用 Step 3 产出的 `Shot.keyframe_paths`（兼容字段 `keyframe_path` 仍保留）。

---

## Step 1: Ingest（入库 + 压缩）

**模块**：`pipeline/ingest.py`

**流程**：
1. 复制视频到 `data/videos/{video_id}/original.mp4`
2. 检测分辨率和帧率
3. 如果高度 > 480p 或帧率 > 10fps → 生成 `compressed.mp4`
   - 缩放：`scale=-2:480`
   - 降帧：`-r 10`
   - 音频保留原始：`-c:a copy`
4. 后续所有理解步骤使用 `meta.storage_path`（压缩时为 `compressed.mp4`，未压缩时为入库原始视频）
5. 渲染阶段使用 `original.mp4`

**输出**：`original.*`、按需生成的 `compressed.mp4`、`meta.json`（含 `compressed_path`, `is_compressed` 等新字段）

---

## Step 2: Shot Detect（镜头切分）

**模块**：`pipeline/scene_detect.py`

基于 `meta.storage_path` 使用 PySceneDetect 进行镜头切分；如果 Step 1 发生压缩，则该路径指向 `compressed.mp4`，否则指向入库后的原始视频。
输出 `scenes/scenes.json`。

---

## Step 3: Keyframe（多帧关键帧采样）

**模块**：`pipeline/keyframe.py`

为每个 shot 采样多帧关键帧（最多6帧），基于 `meta.storage_path`。
关键帧用于 Step 4 脸谱构建，以及后续多模态 RAG / 索引；不作为 Step 5 MinuteChunk 的 Gemini 输入。

---

## Step 4: Face Cluster（人脸聚类 + 角色脸谱）🆕

**模块**：`pipeline/face_cluster.py`

**目标**：

在调用 Gemini 之前，用本地视觉模型先完成“同一个人是谁”的基础归并，生成稳定的 `char_XXX` 脸谱库。这个步骤不负责给角色起真实姓名；真实姓名、别名、行为和关系会在 Step 5 `MinuteChunk` 中结合剧情逐步补全。

**流程**：

1. **缓存读取**
   - 如果 `characters/face_clusters.json` 已存在，直接加载并返回。
   - 这样断点续跑不会重复跑 InsightFace，也能保持 `char_XXX` 编号稳定。

2. **关键帧人脸检测**
   - 遍历每个 shot 的 `keyframe_paths` 和兼容字段 `keyframe_path`。
   - InsightFace 输出 `bbox`、`det_score`、`embedding`。
   - `embedding` 是人脸聚类的核心特征；`bbox` 用于保存 gallery 裁剪图。
   - `FACE_DETECT_DEVICE=auto` 时优先使用 `CUDAExecutionProvider`，不可用则回退 CPU。
   - `FACE_DETECT_GPU_ID=auto` 时自动选择显存占用最低的 CUDA 设备，也可指定具体 GPU 编号。

3. **人脸质量过滤**
   - 先用 `FACE_MIN_DET_SCORE` 过滤低置信度误检。
   - 再用 `max(关键帧短边 * FACE_MIN_FACE_RATIO, FACE_MIN_FACE_PIXEL_FLOOR)` 过滤过小人脸。
   - 再用 `max(关键帧短边 * FACE_MIN_CROP_RATIO, FACE_MIN_CROP_PIXEL_FLOOR)` 过滤裁剪后仍过小的 gallery 候选。
   - 默认开启 `FACE_REJECT_SIDE_FACE`：优先用 InsightFace `pose` 的 yaw 判断侧脸，缺失或不明显时再用 5 点 landmarks 的鼻尖偏移作为辅助。
   - 这样不再固定使用单一像素阈值，而是能适配 480p、1080p、4K 等不同关键帧尺寸。

4. **初始 DBSCAN 聚类**
   - 对归一化后的人脸 embedding 使用 DBSCAN，距离度量为 cosine。
   - `FACE_CLUSTER_EPS` 控制“多近算同一个人”；越大越容易合并。
   - `FACE_CLUSTER_MIN_SAMPLES` 控制成簇最小样本数；越大越保守。

5. **拆分疑似混簇**
   - 如果同一关键帧中同一个簇出现多张脸，说明不同人物可能被混在一起。
   - 如果簇内 90 分位半径超过 `FACE_CLUSTER_MAX_RADIUS`，说明簇太分散。
   - 触发后使用更严格的 `FACE_CLUSTER_SPLIT_EPS` 做二次聚类。

6. **合并疑似碎簇**
   - 用 `FACE_CLUSTER_MERGE_SIM` 判断簇中心是否足够相似。
   - 用 `FACE_CLUSTER_MERGE_LINK_SIM` 判断两个簇的代表脸之间是否存在高相似“桥接”。
   - 使用代表脸合并时，还要求簇中心至少达到 `FACE_CLUSTER_MERGE_MIN_CENTROID_SIM`，防止误合并。
   - 对单对代表脸极高相似的碎簇，使用 `FACE_CLUSTER_MERGE_STRONG_LINK_SIM` 和更低的 `FACE_CLUSTER_MERGE_STRONG_MIN_CENTROID_SIM` 做强桥接合并。
   - 如果两个簇在同一关键帧中同时出现过，不合并。
   - 这一步主要缓解同一人物因换发型、换装、光照变化被拆成多个 gallery 的问题，但不会保证完全消除碎 gallery。

7. **角色分层**
   - **major**：出现 shot 数 ≥ `max(10, 总 shot 数 * 0.05)`。
   - **minor**：未达到 major，但出现 shot 数达到路人阈值。
   - **passerby**：低于路人阈值，默认不保存 gallery。
   - 路人阈值随视频长度变化：短视频 `<10min` 为 2；中等视频使用 `FACE_PASSERBY_MIN`；长视频 `>30min` 至少为 5。

8. **代表脸选择与保存**
   - 路人只保留最高置信度 1 张，且默认不保存。
   - 主要/次要角色保留 `FACE_GALLERY_MIN` 到 `FACE_GALLERY_MAX` 张。
   - 选脸策略：先按 shot 去重，避免同一近景 shot 占满 gallery；再沿角色出现时间轴均匀采样；数量不足时优先补充未使用 shot、且离已选样本时间更远的脸。
   - 保存到 `characters/char_XXX_gallery/face_XX.jpg`，元数据写入 `characters/face_clusters.json`，并记录每张代表脸的时间戳、来源 shot 和来源关键帧。

**关键参数**：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `FACE_MIN_DET_SCORE` | `0.65` | InsightFace 检测置信度下限 |
| `FACE_MIN_FACE_RATIO` | `0.05` | 人脸 bbox 短边占关键帧短边的比例下限 |
| `FACE_MIN_FACE_PIXEL_FLOOR` | `16` | 人脸 bbox 短边绝对像素兜底 |
| `FACE_MIN_CROP_RATIO` | `0.08` | gallery 裁剪图短边占关键帧短边的比例下限 |
| `FACE_MIN_CROP_PIXEL_FLOOR` | `48` | gallery 裁剪图短边绝对像素兜底 |
| `FACE_REJECT_SIDE_FACE` | `true` | 是否过滤明显侧脸 |
| `FACE_MAX_POSE_YAW` | `35` | pose yaw 绝对值超过该角度视为侧脸 |
| `FACE_MAX_LANDMARK_IMBALANCE` | `0.35` | 鼻尖相对双眼中心偏移超过该比例视为侧脸 |
| `FACE_CLUSTER_EPS` | `0.42` | 初始 DBSCAN 余弦距离阈值 |
| `FACE_CLUSTER_MIN_SAMPLES` | `3` | DBSCAN 成簇最少样本数 |
| `FACE_CLUSTER_SPLIT_EPS` | `0.30` | 疑似混簇二次拆分阈值 |
| `FACE_CLUSTER_MAX_RADIUS` | `0.34` | 簇内 90 分位半径上限 |
| `FACE_CLUSTER_MERGE_SIM` | `0.86` | 簇中心相似度合并阈值 |
| `FACE_CLUSTER_MERGE_LINK_SIM` | `0.78` | 代表脸桥接相似度合并阈值 |
| `FACE_CLUSTER_MERGE_MIN_CENTROID_SIM` | `0.62` | 桥接合并时要求的最低簇中心相似度 |
| `FACE_CLUSTER_MERGE_STRONG_LINK_SIM` | `0.82` | 单对代表脸极高相似时的强桥接合并阈值 |
| `FACE_CLUSTER_MERGE_STRONG_MIN_CENTROID_SIM` | `0.50` | 强桥接合并时要求的最低簇中心相似度 |
| `FACE_CLUSTER_MERGE_MAX_FACES` | `32` | 每个簇用于合并比较的最多代表脸数量 |
| `FACE_GALLERY_MIN/MAX` | `3 / 6` | 每个非路人角色的代表脸数量范围 |
| `FACE_KEEP_PASSERBY_GALLERY` | `false` | 是否保存路人脸谱 |

**参数来源**：

正式参数在 `config.py` 中定义，并可由 `.env` 覆盖；`face_cluster.py` 只读取这些集中配置，不单独维护业务默认值。

**输出**：

| 路径 | 内容 |
|------|------|
| `characters/face_clusters.json` | `CharacterGallery` 列表，包含 `character_id`、gallery 路径、gallery 时间戳、gallery 来源 shot/关键帧、出现 shot、tier、embedding centroid |
| `characters/char_XXX_gallery/face_XX.jpg` | 代表脸裁剪图，裁剪时会扩大 bbox 以包含头发、肩部和部分衣着上下文；裁剪后仍过小则丢弃 |

**边界**：

- Step 4 是传统视觉模型阶段，目标是产出“足够稳定、足够干净”的角色脸谱先验，而不是最终人物真值。
- 同一人物仍可能因极端造型、遮挡、光照或年龄/妆造变化被拆成多个 gallery；当前不在 `face_cluster` 中用语义规则强行聚合。
- 后续如果在 MinuteChunk / 动态角色档案更新中获得充分证据证明两个 gallery 是同一人物，可以在更高层做角色级聚合；该能力目前仅作为后续扩展方向。

**降级**：

InsightFace 不可用时跳过，写入空脸谱，由 Step 5 的 Gemini 自行识别；GPU 后端初始化失败时自动回退 CPU。当前本地深度模型主要是 InsightFace；LLM 与 Embedding 走远程 API，DBSCAN / PySceneDetect / FFmpeg 不涉及本地模型上 GPU。

---

## Step 5: MinuteChunk Understand（分钟级融合理解）⭐

**模块**：`pipeline/minute_chunk.py`

**替代原 v3 的**: ASR + Vision + Audio + Character + SpeakerBind + MultimodalAlign

### 5.1 Chunk 构建（自底向上）
- 以 shot 边界为切点，拼接为 ~2-3min 的 chunk
- 当前代码按 `CHUNK_TARGET_DURATION`（默认 150s）累积 shot
- 尾段 < `CHUNK_MERGE_THRESHOLD`（默认 30s）时合并到前一个 chunk
- `CHUNK_MIN_DURATION` / `CHUNK_MAX_DURATION` 已在配置中保留，但当前 `build_minute_chunks()` 暂未强制使用这两个边界

### 5.2 每个 Chunk 的 Gemini 输入
| 输入 | 来源 |
|------|------|
| 视频片段 | 从 `meta.storage_path` 截取 |
| 角色脸谱 | Step 4 的 gallery（身份识别 + 上一轮新增） |
| 角色档案 | 前序 chunk 累积的角色信息 |

说明：Step 3 的关键帧不作为 MinuteChunk Gemini 输入；关键帧主要用于 Step 4 脸谱构建，以及后续多模态 RAG / 索引使用。

调用容错：`minute_chunk.py` 会先用通用增强 JSON 解析器处理 Gemini 响应，支持代码围栏未闭合或 JSON 后带说明文字的情况。解析失败时，会用相同调用再重试一次；只有两次都失败时才跳过该 chunk，并由后续占位逻辑补齐未覆盖 shot。

### 5.3 Gemini 一次性输出
- **A. ASR 转录** — 逐句，已用角色ID标注说话人
- **B. 逐 shot 画面分析** — description/objects/mood/camera/OCR
- **C. 逐 shot 音频特征** — music/sfx/emotion/speech_rate
- **D. 角色动态更新** — 新称呼/形象变化/关键行为
- **E. 跨 shot 分析** — 叙事连续性/情绪弧线/beat 建议

`characters_present` 必须只填写画面中真实可见、且有足够视觉证据识别的人物/实体。仅被台词、旁白、剧情提到，或只是和参考脸谱相似但看不清脸时，不应填入已知角色；无法确定身份时使用 `unknown_1` 等临时编号。

### 5.4 自顶向下回填
将 chunk 结果按 shot 时间戳拆分回填：
- `transcripts.json` — 已带 character_id
- `vision.json` / `ocr.json`
- `audio_prosody.json`
- `multimodal_alignments.json`
- `characters.json` — 动态更新
- `speaker_map.json` — 自动生成
- `character_identity_links.json` — 角色身份合并记录（有合并时生成）

回填时会结合 `local_shot_index` 与全局 `scene_index` 解析 per-shot 结果，防止 LLM 把局部编号和全局编号混用。如果模型漏掉某些 shot，会写入占位 `vision` / `ocr` / `audio` / `multimodal_alignment`，避免散文件断档。`characters.json.appearance_scenes` 由 `multimodal_alignments.json.visible_characters` 和 Step 4 gallery 的出场镜头合并而来，因此角色出场异常时优先检查 `visible_characters` 的误标。

### 5.5 动态角色档案
- 每处理完一个 chunk，更新角色档案：新称呼、形象变化、关键行为
- 下一个 chunk 的 prompt 中包含最新的角色档案
- 允许根据剧情发展修改角色名称、增加别名
- `appearance_change` 会保留到 `appearance_changes` 历史；其中“无”“无明显变化”“无法判断”等占位文本不会覆盖已有 `description`，也不会作为 Step 9 关系分析的有效外观线索
- `key_action` / `new_names` 中的占位文本会被过滤，避免污染角色别名和关键行为

### 5.6 临时角色收敛与正式准入

LLM 返回的 `unknown_N` 会先被规范化为带 chunk 作用域的 `char_tmp_chunk_XXXX_unknown_N`，避免不同 chunk 的临时人物互相覆盖。保存正式 `characters.json` 前，`minute_chunk.py` 会基于出现场景、台词数、可见/说话共现、相邻 chunk 名称和 `character_profiles.json` 描述做二次收敛：

- 能匹配到稳定 `char_XXX` 的临时身份，会统一 canonical 到该正式角色。
- 不能匹配稳定角色但跨 chunk 证据一致的临时身份，会合并为同一个高证据临时角色。
- 低证据 `char_tmp_chunk_*` 默认不进入正式 `characters.json` / `speaker_map.json`，但仍保留在 `character_profiles.json`、transcript 和 alignment 原始记录中，方便排查。
- 临时角色进入正式名册的最低证据是：至少 2 个 chunk，或至少 8 个出场 shot，或至少 6 条台词。

缓存命中时也会执行同一收敛逻辑，并重写 `characters.json`、`speaker_map.json`、`character_profiles.json` 和 `character_identity_links.json` 等派生产物，避免旧缓存继续污染下游。

### 5.7 特殊情况
| 情况 | 处理 |
|------|------|
| 无人脸片段 | ASR标注 "unknown_1" 等临时编号，视觉只分析场景 |
| 非人类角色 | 报告为"非人类实体"，简单记录 |
| 角色换装 | 脸谱含多时段脸，Gemini参考匹配 |

---

## Step 6: Beat Detect（剧情节拍检测）

**模块**：`pipeline/beat_detect.py`

利用 Step 5 回填后的 `transcripts.json`、`vision.json` 和 `characters.json` 进行分组：将连续 shots 按 30 个一段送入 LLM，聚合为叙事节拍 Beat，输出 `beats.json` 并回填 `shot.beat_index`。

**输入 / 输出**：

| 项 | 内容 |
|----|------|
| 输入 | `scenes/scenes.json`、`transcripts.json`、`vision.json`、正式 `characters.json` |
| 输出 | `beats.json`、回写后的 `scenes/scenes.json` (`shot.beat_index`) |
| 缓存 | `beats.json` 已存在时不调用 LLM，但仍执行 `_backfill_beat_to_shots()` |
| 降级 | LLM 解析失败时 `_fallback_beats()` 按每 4 个 shot 一组生成默认 beat，再进入 `_finalize_beats()` |

关键约束（v4.1.1 修复）：

- **角色名册注入与校验**：`characters` 列表会被转成"已知角色名册"写入 prompt（`char_id: 名字 — 描述`）。LLM 在 `beat.characters` 中只能引用名册内的 ID，画面里出现但不在名册的人可以在描述中写成 `unknown_N`，但不能写入 `Beat.characters`。回填时按白名单过滤，丢弃编造的 `char_` ID；当名册为空时返回空角色列表。这避免了 beat 角色 ID 与 Step 4 face_cluster 的 `char_xxx` 体系错配。
- **全覆盖 + 不重叠划分（`_finalize_beats`）**：分段 LLM 结果汇总后统一规范化——跨 beat 去重 shot（保留先出现者）、过滤越界/幻觉 shot 索引、把 LLM 漏分的 shot 按相邻关系聚合为 `transition` beat。保证每个 shot 恰好归属一个 beat，杜绝 `beat_index=None` 的孤儿镜头脱离叙事层级。
- **beat_index 全局唯一且按时间连续**：规范化阶段按时间统一排序并重排 `beat_index` 为 `0..N-1`，忽略 LLM 返回值，防止跨段重复索引破坏 `shot → beat` 反向链接和下游 story_scene 关联。
- **duration 采用墙钟跨度**：`beat.duration = end_time - start_time`（与 StoryScene / Chapter 统一），不再用子 shot 时长求和，避免漏分/非连续时口径漂移。
- **缓存分支幂等**：`understand.py` 的 Step 6 续跑分支同样调用 `detect_beats()`；命中 `beats.json` 时不触发 LLM，但仍会回填 `shot.beat_index`，与新建分支行为一致。
- LLM 解析失败时回退到"每 4 个 shot 一组"的默认分组（`_fallback_beats`），同样进入 `_finalize_beats` 规范化。

说明：MinuteChunk 原始结果中会保存 `suggested_beats`，但当前 `detect_beats()` 主入口尚未直接读取 `minute_chunks.json`，因此 `suggested_beats` 更像后续优化入口；现阶段 Beat 仍由 `beat_detect.py` 基于回填后的台词、画面和人物信息重新让 LLM 判断。

---

## Step 7: Story Scene Detect（故事场景检测）

**模块**：`pipeline/story_scene_detect.py`

将连续 Beat 聚合为故事场景 StoryScene，输出 `story_scenes.json` 并回填 `shot.story_scene_index`。

**输入 / 输出**：

| 项 | 内容 |
|----|------|
| 输入 | `beats.json`、`scenes/scenes.json` |
| 输出 | `story_scenes.json`、回写后的 `scenes/scenes.json` (`shot.story_scene_index`) |
| 缓存 | `story_scenes.json` 已存在时不调用 LLM，但仍执行 `_backfill_scene_to_shots()` |
| 降级 | LLM 解析失败时 `_fallback_story_scenes()` 按每 3 个 beat 一组生成默认 StoryScene |

关键约束（v4.1.1 修复，与 Beat 对齐）：

- **分段调用，长视频不丢尾**：beats 按 `SEGMENT_SIZE`（默认 40）分窗送入 LLM（`_detect_segment`），避免一次性把全部 beat 塞进单个 prompt 导致超长截断、尾部 beat 静默丢失。
- **story_scene_index 本地自增**：不再信任 LLM 返回的 `story_scene_index`（旧实现 `item.get("story_scene_index", ...)` 有重复/跳号风险，会污染 `memory_builder` 按索引做 key 的 scene 单元和 EditSignal 映射）。最终索引由 `_finalize_story_scenes` 统一重排为 `0..N-1`。
- **characters 从子 Beat 聚合**：StoryScene 的 `characters` 直接取所属 beat 的并集，而非采信 LLM 返回值。由于 beat 已做角色白名单，这保证了 StoryScene 角色 ID 与 face_cluster 的 `char_xxx` 体系一致，不会重新引入幻觉 ID。
- **全覆盖兜底（`_finalize_story_scenes`）**：跨场景去重 beat、过滤非法 beat 索引、把漏分的 beat 聚合为 `transition` 场景，保证每个 beat 恰好归属一个 StoryScene。
- **duration 采用墙钟跨度**：`story_scene.duration = end_time - start_time`。
- LLM 解析失败时按"每 3 个 beat 一组"降级（`_fallback_story_scenes`），同样进入规范化。

---

## Step 8: Chapter Detect（大段落检测）

**模块**：`pipeline/chapter_detect.py`

将连续 StoryScene 聚合为 Chapter（长视频大段落）。短视频（`< 600s` 或 StoryScene `≤ 3`）整体作为一个 Chapter；否则走 LLM 分组。

**输入 / 输出**：

| 项 | 内容 |
|----|------|
| 输入 | `story_scenes.json`、`beats.json`、`scenes/scenes.json`、`meta.duration` |
| 输出 | `chapters.json` |
| 缓存 | `chapters.json` 已存在时直接加载 |
| 降级 | LLM 解析失败时 `_fallback_chapters()` 按每 3 个 StoryScene 一组生成默认 Chapter |

关键约束（v4.1.1 修复，与 Beat / StoryScene 对齐）：

- **chapter_index 本地自增**：不再信任 LLM 返回的 `chapter_index`；最终由 `_finalize_chapters` 统一重排为 `0..N-1`，避免索引碰撞破坏 `memory_builder` 按 `chapter_index` 做 key 的 chapter 单元。
- **characters 从子 StoryScene 聚合**：Chapter 的 `characters` 取所属 StoryScene 的并集，不采信 LLM 返回值，保持角色 ID 体系一致。
- **全覆盖兜底（`_finalize_chapters`）**：跨章节去重 StoryScene、过滤非法索引、把漏分的 StoryScene 聚合为 `transition` 章节，保证每个 StoryScene 恰好归属一个 Chapter。
- **duration 口径统一**：单章节路径与 LLM 路径均使用 `end_time - start_time`（旧实现 LLM 路径用子时长求和、单章节路径用墙钟跨度，二者不一致）。
- 移除了旧实现中未使用的 `beat_map` 死代码；LLM 解析失败时按"每 3 个 StoryScene 一组"降级（`_fallback_chapters`），同样进入规范化。

---

## Step 9: Event Graph + Character Arc

**模块**：`pipeline/event.py` + `pipeline/character_arc.py`

合并为一步执行：先抽取事件和事件关系图，再分析人物弧线和人物关系。

**输入 / 输出**：

| 项 | 内容 |
|----|------|
| 输入 | shots、`transcripts.json`、`vision.json`、`beats.json`、`story_scenes.json`、`chapters.json`、`characters.json`、`character_profiles.json` |
| 输出 | `events.json`、`event_graph.json`、`character_arcs.json`、`character_relations.json`；并更新 `characters.json` 的弧线、台词数、重要性和 `key_event_indices` |
| 事件缓存 | `events.json` + `event_graph.json` 同时存在时直接加载；只有 `events.json` 时会加载事件并补建图谱 |
| 弧线缓存 | `character_arcs.json` + `character_relations.json` 同时存在时直接加载并回填到 characters |

事件抽取按 30 分钟分段执行，但不再只看每段开头材料。Prompt 会优先输入 StoryScene / Beat 层级摘要，并对台词与画面做时间均匀采样；同时显式传入 `segment_start` / `segment_end`，要求 LLM 输出全片绝对时间。解析后会把事件时间归一化并 clamp 到当前段范围，避免后续分段的事件错误落到 0 秒附近。

事件关系图不再只取 `events[:30]`，而是保留全部高重要事件，再按时间均匀补齐到固定上限。人物弧线的事件输入也改为开头/中段/结尾均匀采样，并优先保留高重要、多人物事件，减少“只看影片开头”的人物弧线偏差。

人物弧线/关系分析会读取 Step 5 的 `character_profiles.json`，补充有效别名、外观变化和关键行为；同时根据 `characters.json.appearance_scenes` 计算人物共现，作为关系推断证据。`events.json`、`event_graph.json`、`character_arcs.json` 与 `character_relations.json` 已存在时会直接加载缓存，如需用新的事件/关系逻辑重算，需要删除对应文件或从 `event_and_arc` 前置步骤重新跑。

---

## Step 10: Final Build（信号 + Memory + 索引）

**模块**：`pipeline/edit_signal.py` + `pipeline/memory_builder.py` + `pipeline/indexer.py`

合并为一步：
1. 计算三类信号（EditSignal / NarrativeSignal / RecompositionSignal）
2. 构建四层 VideoMemory（Shot / Beat / StoryScene / Chapter）
3. 构建检索索引

EditSignal 会覆盖全部 beat 和 story_scene；shot 级信号只覆盖代表性关键镜头，默认受 `EDIT_SIGNAL_MAX_SHOTS=240` 限制。候选来自 beat 首尾 shot、重要事件首尾/中点 shot、story_scene 首尾 shot；超过上限时按事件重要度、beat intensity 和时间覆盖裁剪。未计算 shot 信号的 MemoryUnit 保持 `edit_signal=None`，不影响 Memory 和索引构建。

**输入 / 输出**：

| 子步骤 | 输入 | 输出 / 行为 |
|--------|------|-------------|
| 10a 信号计算 | beats、story_scenes、代表性 shots、events、characters、transcripts、vision | `edit_signals.json`、`narrative_signals.json`、`recomposition_signals.json` |
| 10b Memory 构建 | Step 1-9 散文件 + 三类信号 | `memory.json`，包含 `memory_units`、`beat_memory_units`、`scene_memory_units`、`chapter_memory_units`；同时把 `meta.status` 置为 `ready` |
| 10c 索引构建 | `memory.json` | `index/search_index.json`、可选 `faiss.index` / `id_map.json`、`character_index.json`、`event_index.json`、`relation_index.json`、`emotion_index.json`、`edit_signal_index.json`、`audio_index.json`、`chapter_index.json` |

信号缓存有升级路径：如果 `edit_signals.json` 已存在，会直接加载 EditSignal；若 `narrative_signals.json` 或 `recomposition_signals.json` 缺失，会只补算缺失类型。RecompositionSignal 只面向重要 beat 计算（`intensity >= 0.5` 或指定剧情类型），没有候选时回退到前 10 个 beat。角色业务身份判定会跳过低证据 `char_tmp_`，并在 LLM 失败时按出镜时长做默认主配角分配。

**信号日志与排查**：

| 日志片段 | 含义 |
|----------|------|
| `EditSignal start` | Step 10a 输入规模，包括 shots / beats / story_scenes / events / transcripts / vision_summaries 和 `max_shots` |
| `EditSignal cache loaded` / `EditSignal cache is partial` | 三类信号缓存命中情况；部分缓存会只补算缺失类型 |
| `NarrativeSignal cache empty ... recomputing` / `RecompositionSignal cache empty ... recomputing` | 对空数组缓存的自愈：当仍有可计算单元时，不把空文件当作有效完成结果 |
| `shot级剪辑信号选择` | shot 级代表镜头选择摘要：总 shot、候选数、选中数、跳过数、上限、高重要事件数、候选来源 |
| `EditSignal batch start/done` | beat / story_scene / shot 的每批 LLM 调用进度、unit 范围、prompt 字符数、解析数、产出数和耗时 |
| `NarrativeSignal batch start/done` | 叙事信号每批 LLM 调用进度和耗时 |
| `RecompositionSignal compute start` | 二创信号目标 beat 数量和来源：`important` 或 `fallback_first_10` |
| `EditSignal complete` | Step 10a 三类信号全部结束；如果之后仍慢，通常进入 Memory 构建或索引构建 |

排查 Step 10 慢时，先看 `EditSignal complete` 是否出现。若未出现，慢点在 10a 信号 LLM batch；若已出现但 Step 10 未结束，重点检查 10b Memory 构建和 10c embedding / indexer。Step 10a 不读取 shot 视频或关键帧图片，它只消费 Step 5 的文本化多模态摘要和层级结构。

索引层是 final_build 的硬要求：embedding API 或 FAISS 不可用会跳过对应向量层，但 `build_search_index()` 自身抛错时 `understand.py` 会中断并抛出 `RuntimeError`，不会标记 `final_build` 完成。

---

## 产物目录数据流

前面的总表展示每步输入/输出的语义关系；这里补充最终落盘目录视角，便于排查某一步是否已经产出完整散文件。

```
video.mp4
   │
   ├─[1]─→ original.* + compressed.mp4(按需) + meta.json
   ├─[2]─→ scenes/scenes.json
   ├─[3]─→ scenes/keyframes/
   │
   ├─[4]─→ characters/
   │       ├── face_clusters.json
   │       ├── char_000_gallery/ (3-6张脸)
   │       └── char_001_gallery/
   │
   ├─[5]─→ minute_chunks.json + character_profiles.json
   │       ├── transcripts.json (已带character_id)
   │       ├── ocr.json + vision.json
   │       ├── audio_prosody.json
   │       ├── multimodal_alignments.json
   │       ├── characters.json
   │       ├── speaker_map.json
   │       └── character_identity_links.json
   │
   ├─[6]─→ beats.json
   ├─[7]─→ story_scenes.json
   ├─[8]─→ chapters.json
   ├─[9]─→ events.json + event_graph.json
   │       + character_arcs.json + character_relations.json
   └─[10]→ edit_signals.json + narrative_signals.json
           + recomposition_signals.json + memory.json + index/
```

---

## Gemini API 调用量对比（30min视频, 200 shot）

| 步骤 | v3 调用 | v4.1 调用 |
|------|---------|-----------|
| ASR | 6 | 0 (并入 chunk) |
| Vision | ~200 | 0 |
| Audio | ~6 | 0 |
| Character | ~10 | 0 |
| SpeakerBind | 1 | 0 |
| **MinuteChunk** | — | **~12** |
| Beat | 7 | 0-2 |
| StoryScene | 1 | 1-2 (按 40 beat 分窗) |
| Chapter | 1 | 1 |
| Event+Arc | 4 | 2-3 |
| EditSignal | ~15 | ~8 |
| NarrativeSignal | ~3 | ~3 |
| RecompSignal | ~3 | ~3 |
| **总计** | **~257** | **~30-33 (↓87%)** |

---

## 断点续跑与兼容

`understand.py` 使用 `progress.json` 记录已完成步骤。v4.1 的新步骤名如下：

```python
[
  "ingest",
  "shot_detect",
  "multi_keyframe",
  "face_cluster",
  "minute_chunk",
  "beat_detect",
  "story_scene_detect",
  "chapter_detect",
  "event_and_arc",
  "final_build",
]
```

旧进度文件通过 `_STEP_ALIASES` 兼容：

| 旧步骤 | 映射到 | 说明 |
|--------|--------|------|
| `scene_detect` | `shot_detect` | 直接映射 |
| `keyframe_extract` | `multi_keyframe` | 直接映射 |
| `asr` / `asr_windowed` / `vision` / `audio_analysis` / `speaker_bind` / `multimodal_align` | `multi_keyframe` | 退到更前，确保 `face_cluster` + `minute_chunk` 都重跑 |
| `character_deep` / `character` | `multi_keyframe` | 旧语义不同，退到更前 |
| `event_graph` / `event` / `character_arc` | `chapter_detect` | 退到前置步骤，确保 `event_and_arc` 重跑 |
| `edit_signal` / `build_memory` / `indexer` | `event_and_arc` | 退到前置步骤，确保 `final_build` 重跑 |

恢复逻辑将旧子步骤映射到合并步骤的**前置步骤**，而非合并步骤本身，避免部分完成被误判为全部完成。
当所有步骤标记完成时，还会验证关键产物（`memory.json` 与 `index/search_index.json`）是否存在。

调试时可使用 `python main.py understand --video movie.mp4 --until-step 5` 或 `--until-step minute_chunk`，流水线会在指定步骤完成并写入 `progress.json` 后停止。该限制不会标记后续步骤完成；下一次使用 `--resume` 会从第一个未完成步骤继续执行。

Step 6/7 完成后会回写 `scenes/scenes.json`，持久化 `beat_index` / `story_scene_index` 反向链接。
`_load_shots()` 加载时还会从 `beats.json` / `story_scenes.json` 防御性重建这些链接。

---

## 当前实现注意事项

- `face_cluster.py` 在 InsightFace 未安装时会跳过，返回空脸谱；此时 MinuteChunk prompt 会用 `unknown_1` 等临时标注，`_normalize_character_id()` 会统一将其转为 chunk 作用域的 `char_tmp_chunk_XXXX_unknown_X`，避免不同 chunk 的临时人物互相覆盖，并在 speaker、characters_present、character_updates 三个渠道保持一致。正式写入 `characters.json` 前还会做证据阈值和 canonicalization，低证据临时角色不会进入下游正式名册。
- `face_cluster.py` 会优先读取 `characters/face_clusters.json` 缓存。修改人脸聚类阈值后，如需重新生成角色脸谱，需要删除该缓存及对应 gallery 目录，或从 face cluster 前置步骤重新跑。
- 人脸聚类参数以 `config.py` / `.env` 为准；修改阈值后需要清理旧 `face_clusters.json` 才会重新生成脸谱。
- `minute_chunk.py` 的已有产物检查包含 9 个文件（含 `characters.json`, `speaker_map.json`, `multimodal_alignments.json`, `character_profiles.json`）；缓存命中时仍会对角色身份做收敛，并重写正式角色与 speaker 映射派生产物。
- Step 6/7 无论是新计算还是缓存加载，都会通过 `_backfill_beat_to_shots()` / `_backfill_scene_to_shots()` 回填 shot 的反向链接并持久化到 `scenes/scenes.json`。
- Step 6/7/8 都通过 `_finalize_*` 规范化保证层级是「完整且不重叠的划分」：LLM 漏分的 shot/beat/story_scene 会被聚合成 `transition` 单元补回，索引按时间重排为连续唯一值。因此 `beats.json` / `story_scenes.json` / `chapters.json` 中可能出现 `beat_type="transition"` 或 `plot_function="transition"` 或 `chapter_type="transition"` 的兜底单元，属于预期行为。
- Step 7/8 的 `characters` 一律从子层（beat / story_scene）聚合，不采信 LLM 返回的角色列表，以保持与 Step 4 face_cluster 的 `char_xxx` 体系一致。
- Beat / StoryScene / Chapter 的 `duration` 统一为 `end_time - start_time`（墙钟跨度），不是子单元时长求和。
- `utils.group_consecutive()` 是 Step 6/7/8 共用的「相邻分组」工具，用于把未覆盖的单元按序号连续性聚合为过渡单元。
- Step 9 的事件、事件关系和人物弧线 prompt 会使用时间均匀采样与层级摘要；已有事件/关系/弧线产物仍按缓存优先读取，如需验证新采样逻辑，需要删除对应 JSON 或从 `event_and_arc` 前置步骤重跑。
- Step 10 的 shot 级 EditSignal 是代表镜头抽样，不再默认覆盖全部 shot；可通过 `EDIT_SIGNAL_MAX_SHOTS` 调整上限。日志中的 `source_candidates` 可用于判断候选主要来自 beat 边界、story_scene 边界还是高重要事件。
- Step 10 之前只有散文件；完整四层 MemoryUnit、embedding 和检索索引需要 `final_build` 完成后才具备。
- Prompt 的镜头边界同时给出 `local_shot_index` 和全局 `scene_index`，并要求 `per_shot` 覆盖每个 local shot；回填时会处理两者冲突，并按整体局部/全局索引模式推断作为防御。
- Step 6-10 不重新读取 shot 原视频；最终 `MemoryUnit` 会保留 `keyframe_path` 和 `keyframe_paths`，供后续检索、选材或展示引用关键帧。
