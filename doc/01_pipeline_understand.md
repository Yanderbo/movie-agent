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
文件更小，后续上传 Gemini 更快。

---

## Step 4: Face Cluster（人脸聚类 + 角色脸谱）🆕

**模块**：`pipeline/face_cluster.py`

**流程**：
1. **InsightFace 检测**所有关键帧中的人脸
   - `FACE_DETECT_DEVICE=auto` 时优先使用 `CUDAExecutionProvider`，不可用则回退 CPU
   - `FACE_DETECT_GPU_ID=auto` 时自动选择显存占用最低的 CUDA 设备，也可指定具体 GPU 编号
2. **DBSCAN 聚类**（余弦距离）
3. **频率分层**（根据视频时长自动调整阈值）：
   - **主要角色 (major)**: 出现 ≥ 总shot数的5% 或 ≥ 10次
   - **次要角色 (minor)**: 出现 3-9次
   - **路人 (passerby)**: 出现 < 3次，不建档
4. **角色脸谱构建**：每角色 3-6 张代表脸
   - 时间轴均匀采样 → 捕捉服装/发型变化
   - Embedding 多样性采样 → 捕捉不同表情/角度
5. 保存到 `characters/char_XXX_gallery/`

**降级**：InsightFace 不可用时跳过，由 Step 5 的 Gemini 自行识别；GPU 后端初始化失败时自动回退 CPU。当前本地深度模型主要是 InsightFace；LLM 与 Embedding 走远程 API，DBSCAN / PySceneDetect / FFmpeg 不涉及本地模型上 GPU。

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
| 关键帧图片 | Step 3 的 keyframes |
| 角色脸谱 | Step 4 的 gallery（身份识别 + 上一轮新增） |
| 角色档案 | 前序 chunk 累积的角色信息 |

### 5.3 Gemini 一次性输出
- **A. ASR 转录** — 逐句，已用角色ID标注说话人
- **B. 逐 shot 画面分析** — description/objects/mood/camera/OCR
- **C. 逐 shot 音频特征** — music/sfx/emotion/speech_rate
- **D. 角色动态更新** — 新称呼/形象变化/关键行为
- **E. 跨 shot 分析** — 叙事连续性/情绪弧线/beat 建议

### 5.4 自顶向下回填
将 chunk 结果按 shot 时间戳拆分回填：
- `transcripts.json` — 已带 character_id
- `vision.json` / `ocr.json`
- `audio_prosody.json`
- `multimodal_alignments.json`
- `characters.json` — 动态更新
- `speaker_map.json` — 自动生成

### 5.5 动态角色档案
- 每处理完一个 chunk，更新角色档案：新称呼、形象变化、关键行为
- 下一个 chunk 的 prompt 中包含最新的角色档案
- 允许根据剧情发展修改角色名称、增加别名

### 5.6 特殊情况
| 情况 | 处理 |
|------|------|
| 无人脸片段 | ASR标注 "unknown_1" 等临时编号，视觉只分析场景 |
| 非人类角色 | 报告为"非人类实体"，简单记录 |
| 角色换装 | 脸谱含多时段脸，Gemini参考匹配 |

---

## Step 6: Beat Detect（剧情节拍检测）

**模块**：`pipeline/beat_detect.py`

利用 Step 5 回填后的 `transcripts.json`、`vision.json` 和 `characters.json` 进行分组。

说明：MinuteChunk 原始结果中会保存 `suggested_beats`，但当前 `detect_beats()` 主入口尚未直接读取 `minute_chunks.json`，因此 `suggested_beats` 更像后续优化入口；现阶段 Beat 仍由 `beat_detect.py` 基于回填后的台词、画面和人物信息重新让 LLM 判断。

---

## Step 7: Story Scene Detect

**模块**：`pipeline/story_scene_detect.py`

不变。将连续 beat 分组为故事场景。

---

## Step 8: Chapter Detect

**模块**：`pipeline/chapter_detect.py`

不变。将连续 story_scene 分组为大段落。

---

## Step 9: Event Graph + Character Arc

**模块**：`pipeline/event.py` + `pipeline/character_arc.py`

合并为一步执行：先抽取事件和事件关系图，再分析人物弧线和人物关系。
角色档案（来自 Step 5）提供丰富的角色发展信息。

---

## Step 10: Final Build（信号 + Memory + 索引）

**模块**：`pipeline/edit_signal.py` + `pipeline/memory_builder.py` + `pipeline/indexer.py`

合并为一步：
1. 计算三类信号（EditSignal / NarrativeSignal / RecompositionSignal）
2. 构建四层 VideoMemory（Shot / Beat / StoryScene / Chapter）
3. 构建检索索引

---

## 数据流总览

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
   │       └── speaker_map.json
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
| StoryScene | 1 | 1 |
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

Step 6/7 完成后会回写 `scenes/scenes.json`，持久化 `beat_index` / `story_scene_index` 反向链接。
`_load_shots()` 加载时还会从 `beats.json` / `story_scenes.json` 防御性重建这些链接。

---

## 当前实现注意事项

- `face_cluster.py` 在 InsightFace 未安装时会跳过，返回空脸谱；此时 MinuteChunk prompt 会用 `unknown_1` 等临时标注，`_normalize_character_id()` 会统一将其转为 `char_tmp_unknown_X`，并在 speaker、characters_present、character_updates 三个渠道保持一致。
- `minute_chunk.py` 的已有产物检查包含 9 个文件（含 `characters.json`, `speaker_map.json`, `multimodal_alignments.json`, `character_profiles.json`）。
- Step 6/7 无论是新计算还是缓存加载，都会通过 `_backfill_beat_to_shots()` / `_backfill_scene_to_shots()` 回填 shot 的反向链接并持久化到 `scenes/scenes.json`。
- Step 10 之前只有散文件；完整四层 MemoryUnit、embedding 和检索索引需要 `final_build` 完成后才具备。
- Prompt 示例中的 `scene_index` / `shot_indices` 使用动态占位符，回填时还有局部→全局索引映射作为防御。
