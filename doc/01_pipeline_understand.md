# 理解流水线 (understand.py)

> 文件：`pipeline/understand.py`
> 职责：编排 17 步理解流水线，将一个视频文件从原始输入转化为多层结构化 Video Memory

## 总体流程

```
ingest → shot_detect → multi_keyframe → asr_windowed
  → vision → audio_analysis → character_deep → speaker_bind
  → multimodal_align → beat_detect → story_scene_detect
  → chapter_detect → event_graph → character_arc
  → edit_signal → build_memory → indexer
```

核心设计原则：
1. **先切后提** ——先完成镜头切分（step 2），再在每个 shot 内提取各模态数据
2. **层层聚合** ——从 shot 到 beat 到 story_scene 到 chapter，逐级构建更高层叙事单元
3. **多模态对齐** ——v3 新增跨模态一致性验证，确保 speaker / character / vision / audio 数据对齐
4. **三类信号驱动** ——EditSignal + NarrativeSignal + RecompositionSignal 为下游提供多维量化依据

### 叙事层级结构

```
Shot (镜头)             ← 最小视觉单元，由 scene_detect 切分
 └─ Beat (节拍)         ← 2-8 个连续 shot 组成的叙事微单元
     └─ StoryScene      ← 2-6 个连续 beat 组成的完整叙事场景
         └─ Chapter     ← 🆕 多个 StoryScene 组成的长视频大段落
             └─ EventGraph ← 事件节点 + 因果/铺垫/反转关系边（含证据+置信度）
```

## 入口函数

```python
def run_understand(video_path, video_id=None, resume=False) -> str:
```

- `video_path`：新视频文件路径（首次处理必传）
- `video_id` + `resume=True`：从断点继续（按 `progress.json` 中记录的已完成步骤确定起始位置）

## 17 步详解

---

### Step 1: Ingest（入库）

**模块**：`pipeline/ingest.py`

**流程**：
1. 校验文件存在性
2. 生成人类可读的 `video_id`：取文件名 → 去特殊字符 → 截断30字符 → 拼接 `_` + 路径MD5前8位
3. 在 `data/videos/{video_id}/` 下创建工作目录
4. 复制源视频到 `original.mp4`（如已存在则跳过）
5. 调用 `ffprobe` 解析元信息（时长、分辨率、帧率、编码、文件大小）
6. 写出 `meta.json`

**输出**：`meta.json`（VideoMeta 对象序列化）

---

### Step 2: Shot Detect（镜头切分）

**模块**：`pipeline/scene_detect.py`

**流程**：
1. 使用 PySceneDetect 的 `ContentDetector` 检测镜头边界
2. 参数：`threshold`（默认27.0）、`min_scene_len`（默认1.0秒）
3. 构建 `Shot` 对象（含 `scene_index`、`start_time`、`end_time`、`duration`）
4. 写出 `scenes/scenes.json`

**输出**：`scenes/scenes.json`

**关键点**：此步骤**必须在 ASR 和 Vision 之前完成**，后续所有模态数据都以 `scene_index` 为锚点

---

### Step 3: Multi Keyframe（多帧关键帧采样）

**模块**：`pipeline/keyframe.py`

**流程**：
1. 根据 shot 时长动态决定采样帧数：
   - `<2s` → 1帧, `2-5s` → 2帧, `5-15s` → 3帧, `>15s` → 4-6帧
2. 采样时间点避开首尾各 10% 区域（避免转场模糊帧）
3. 调用 `ffmpeg -ss {timestamp} -vframes 1` 提取 JPEG
4. 首帧赋给 `Shot.keyframe_path`（向后兼容），全部帧路径存入 `Shot.keyframe_paths`
5. 重写 `scenes.json`

**输出**：`scenes/keyframes/scene_XXXX_f0~f5.jpg` + 更新后的 `scenes.json`

---

### Step 4: ASR Windowed（长窗口语音转文字）

**模块**：`pipeline/asr.py`

**流程**：
1. 按 `ASR_WINDOW_DURATION`（默认 300s）切分音频窗口
2. 对每个窗口整体提取音频 → 调用 Gemini Audio API 转写
3. 解析带时间戳的转写结果，转换为全局绝对时间
4. **回填 shot**：按句子中点落入哪个 shot 分配 `scene_index`
5. 标记跨镜头台词（`cross_shot=True`）
6. 识别台词类型（`dialogue` / `narration` / `voiceover`）
7. 写出 `transcripts.json`

**输出**：`transcripts.json`

**关键点**：
- 长窗口避免了因镜头切分导致的语音中断
- `cross_shot=True` 标记帮助后续判断台词归属

---

### Step 5: Vision（OCR + 多帧画面理解 + micro_clip）

**模块**：`pipeline/vision.py`

**v3 增强**：新增 micro_clip 字段（镜头运动、互动描述、景别）。

**流程**：
1. 分批处理（默认 `batch_size=5`）
2. 多帧 shot：发送多张帧图片给 Gemini Vision，分析动作变化和表情变化
3. 单帧 shot：分析单帧（传统模式）
4. 返回结果包含 v3 新增字段：`camera_motion`、`interaction_description`、`shot_scale`
5. 写出 `ocr.json` 和 `vision.json`

**输出**：`ocr.json`、`vision.json`

**v3 micro_clip 字段**：
| 字段 | 说明 |
|------|------|
| `camera_motion` | 镜头运动（pan/tilt/zoom/static/tracking/handheld） |
| `interaction_description` | 人物互动描述 |
| `shot_scale` | 景别（extreme_close_up/close_up/medium_close/medium/medium_long/long/extreme_long） |

---

### Step 6: Audio Analysis（音频韵律分析）🆕

**模块**：`pipeline/audio_analysis.py`

**流程**：
1. 按 ASR 窗口时长组织时间窗口和窗口内 shot
2. 构造每段 shot 内的台词/画面上下文 prompt
3. LLM 基于上下文推断音频韵律特征：音乐、音效、沉默比例、语速、音量峰值、语音情绪
4. 回填 `AudioProsody` 对象到每个 shot
5. 写出 `audio_prosody.json`

**输出**：`audio_prosody.json`（AudioProsody 列表）

**当前实现说明**：`audio_analysis.py` 当前没有像 ASR 那样抽取音频文件并传入多模态接口，而是根据台词、画面摘要和 shot 时间范围让 LLM 推断音频韵律；因此产物更适合作为剪辑辅助信号，而不是精确 DSP 分析结果。

**AudioProsody 字段**：
| 字段 | 说明 |
|------|------|
| `has_music` | 是否有背景音乐 |
| `music_mood` | 音乐情绪（tense/romantic/cheerful/...） |
| `has_sfx` | 是否有音效 |
| `sfx_tags` | 音效标签列表 |
| `silence_ratio` | 沉默占比 0-1 |
| `speech_rate` | 语速（slow/normal/fast） |
| `volume_peak` | 音量峰值 0-1 |
| `speech_emotion` | 语音情绪 |
| `audio_segments` | AudioSegment 列表（精细时间段） |

**降级策略**：LLM 失败时返回空 AudioProsody（仅保留 `scene_index`），不阻塞流程

---

### Step 7: Character Deep（深度人物分析）

**模块**：`pipeline/character.py`

**流程**：
1. 人脸检测：InsightFace（优先）或 Gemini Vision（降级）
2. 多帧人脸检测：遍历 `keyframe_paths` 中所有帧
3. 聚类 + 描述生成
4. 计算深度字段：`first_appearance`, `last_appearance`, `co_appearing_characters`
5. 写出 `characters.json`

**输出**：`characters.json`（`CharacterDeep` 对象列表）

---

### Step 8: Speaker Bind（Speaker ↔ Character 绑定）

**模块**：`pipeline/speaker_bind.py`

**流程**：
1. 收集 ASR 中的 `speaker_id`
2. 统计共现矩阵
3. LLM 确认映射
4. 回写到 `TranscriptSegment.character_id` 和 `Character.speaker_ids`

**输出**：`speaker_map.json`、更新后的 `transcripts.json` 和 `characters.json`

---

### Step 9: Multimodal Alignment（多模态对齐）🆕

**模块**：`pipeline/multimodal_align.py`

**流程**（纯规则，无 LLM 依赖）：
1. 通过 `speaker_map` 建立 speaker → character 映射
2. 通过 `appearance_scenes` 确定每个 shot 的 `visible_characters`
3. 交叉验证：说话者是否在画面中可见（检测画外音/旁白）
4. 确定 `active_modalities`（speech/visual_action/music/sfx/silence）
5. 确定 `dominant_modality`（主导模态）
6. 计算 `alignment_confidence`（基于多模态数据完整度）
7. 检测冲突（说话者不在画面中 → 可能是画外音/旁白）
8. 写出 `multimodal_alignments.json`

**输出**：`multimodal_alignments.json`（MultimodalAlignment 列表）

**降级策略**：数据缺失时 `confidence=0.5`（基准），不阻塞

---

### Step 10: Beat Detect（剧情节拍检测）

**模块**：`pipeline/beat_detect.py`

**流程**：
1. 每次取 30 个 shot 为一组，构造 prompt
2. 将每 shot 的台词摘要、画面描述、人物、情绪、时长等信息发送给 LLM
3. LLM 返回分组建议：哪些连续 shot 属于同一个 beat
4. 构建 `Beat` 对象（含 `beat_type`, `description`, `emotion`, `intensity`）
5. **回写** `Shot.beat_index`，建立双向关联
6. 写出 `beats.json`

**Beat 类型**：`setup` / `confrontation` / `resolution` / `transition` / `montage`

**降级策略**：LLM 失败时，每 4 个 shot 自动分为一组

---

### Step 11: Story Scene Detect（故事场景检测）

**模块**：`pipeline/story_scene_detect.py`

**流程**：
1. 将 beat 列表发送给 LLM，分析哪些连续 beat 属于同一个故事场景
2. 构建 `StoryScene` 对象（含 `location`, `plot_function`, `characters`）
3. 汇总 `shot_indices`
4. **回写** `Shot.story_scene_index`
5. 写出 `story_scenes.json`

**Plot Function**：`inciting_incident` / `rising` / `climax` / `falling` / `resolution` / `setup`

**降级策略**：LLM 失败时，每 3 个 beat 自动分为一组

---

### Step 12: Chapter Detect（长视频大段落检测）🆕

**模块**：`pipeline/chapter_detect.py`

**流程**：
1. 短视频（< 10min）：整部视频作为一个 Chapter
2. 长视频：将 StoryScene 列表发送给 LLM，按主题/地点/角色变化分析章节边界
3. 构建 `Chapter` 对象（含 `title`, `theme`, `chapter_type`, `mood_progression`）
4. 汇总 `story_scene_indices`, `beat_indices`, `shot_indices`
5. 写出 `chapters.json`

**Chapter 类型**：`prologue` / `act_1` / `act_2` / `act_3` / `climax_act` / `epilogue` / `flashback`

**降级策略**：LLM 失败时，每 3 个 StoryScene 自动分为一组

---

### Step 13: Event Graph（事件图谱构建）v3 增强

**模块**：`pipeline/event.py`

**v3 增强**：Event 增加 `evidence`（证据）和 `confidence`（置信度），EventEdge 增加 `relation_basis`。

**流程**：
1. 事件抽取：LLM 从多模态上下文中抽取事件
2. 事件关系推理（`_extract_event_edges`）
   - 关系类型：`cause` / `foreshadow` / `reversal` / `escalation` / `resolution` / `parallel`
   - 每条关系包含 `evidence` 和 `relation_basis`（推理依据）
3. 填充 `Event.beat_indices` 和 `Event.story_scene_indices`
4. 构建 `EventGraph`（含 `nodes` + `edges`）
5. 写出 `events.json` 和 `event_graph.json`

**输出**：`events.json`、`event_graph.json`

---

### Step 14: Character Arc（人物弧线 + 关系图）

**模块**：`pipeline/character_arc.py`

**流程**：
1. 计算每个角色的 `importance_score`（出镜 40% + 台词 30% + 事件 30%）
2. 统计 `dialogue_count`
3. LLM 分析人物弧线（`arc_type`: growth / fall / transformation / ...）
4. LLM 分析人物间关系（`relation_type`: ally / rival / romantic / ...）
5. 计算共现 shot 列表
6. 更新 `characters.json`，写出 `character_relations.json`

**输出**：更新后的 `characters.json`、`character_relations.json`

---

### Step 15: Edit Signal（三类信号计算）v3 升级

**模块**：`pipeline/edit_signal.py`

**v3 变更**：从单一 EditSignal 升级为三类信号体系。

**子流程 A：EditSignal（8维剪辑信号）**
- 为每个 beat / story_scene / 重要 shot 计算 8 维信号
- 每批 15 个单元发送给 LLM 评估

**子流程 B：NarrativeSignal（叙事信号）** 🆕
- 分析叙事弧位置（arc_position）、张力水平（tension_level）、信息密度
- 判定叙事功能（exposition/climax/resolution/...）

**子流程 C：RecompositionSignal（二次创作信号）** 🆕
- 评估梗潜力（meme_potential）、情感引用价值（emotional_quotability）
- 计算平台适配分（douyin/bilibili/youtube）
- 建议二创格式（reaction/compilation/fancam/edit）

**输出**：`edit_signals.json`、`narrative_signals.json`、`recomposition_signals.json`

**降级策略**：各子流程独立降级，LLM 失败返回空列表，不互相阻塞

**v2→v3 兼容**：如果已有 `edit_signals.json` 但无新信号文件，自动补算 NarrativeSignal 和 RecompositionSignal

---

### Step 16: Build Memory（构建四层 VideoMemory）v3 升级

**模块**：`pipeline/memory_builder.py`

**v3 变更**：从三层 MemoryUnit 升级为四层。

**子流程 A：Shot 级 MemoryUnit**
- 融合台词、画面、OCR、人物、事件、EditSignal
- 🆕 新增 `chapter_index`、`audio_prosody`、`alignment` 字段
- 🆕 `combined_text` 中补充音频信息（音乐/音效/语音情绪）

**子流程 B：Beat 级 BeatMemoryUnit**
- 聚合 beat 内所有 shot 的台词和画面

**子流程 C：StoryScene 级 SceneMemoryUnit**
- 聚合 story_scene 内所有 beat 的描述

**子流程 D：Chapter 级 ChapterMemoryUnit** 🆕
- 聚合 chapter 内所有 StoryScene 的描述/主题/人物
- 关联 NarrativeSignal

**子流程 E：角色判定**
- 统计台词量、事件参与度、出镜时长、`importance_score`
- LLM 判定 `male_lead` / `female_lead` / `villain` / `supporting` / `minor`

**最终**：组装 `VideoMemory`（含四层 MemoryUnit + 三类信号 + 音频/对齐数据），写出 `memory.json`

---

### Step 17: Indexer（构建九维检索索引）v3 升级

**模块**：`pipeline/indexer.py`

**v3 变更**：从 7 种索引扩展为 9 种。

| # | 索引 | 文件 | 说明 |
|---|------|------|------|
| 1 | 文本索引 | `search_index.json` | 关键词 + 2-gram（兼容 v1） |
| 2 | 向量索引 | `faiss.index` + `id_map.json` | Embedding 余弦相似度 |
| 3 | 角色索引 | `character_index.json` | character_id → shots/beats/scenes |
| 4 | 事件索引 | `event_index.json` | event_type → 按重要性排序 |
| 5 | 关系索引 | `relation_index.json` | char_a\|char_b → 关系/共现 |
| 6 | 情绪索引 | `emotion_index.json` | emotion → shots/beats 列表 |
| 7 | 剪辑信号索引 | `edit_signal_index.json` | signal_type → top-N + by_usage 反向索引 |
| 8 | 音频索引 | `audio_index.json` | 🆕 music:mood / sfx:tag / speech_emotion → shots |
| 9 | 章节索引 | `chapter_index.json` | 🆕 chapter_index → {title, theme, scenes, characters} |

---

## 断点续跑机制

- 每步完成后调用 `_save_progress(video_id, step_name)`，写入 `progress.json`
- resume 时读取 `progress.json`，找到第一个未完成的步骤作为起始点
- **v2/v3 向后兼容**：`_STEP_ALIASES` 将旧步骤名映射为新步骤名
- 当前实现依赖 `progress.json` 判断断点；如果进度文件缺失，不会自动从散文件推断完成步骤
- `beat_detect` / `story_scene_detect` 会在内存对象上回填 `Shot.beat_index` / `Shot.story_scene_index`，中断续跑时建议检查 `scenes/scenes.json` 是否包含这些回填字段
- LLM 分组结果按现有输出落盘；建议检查 `beats.json`、`story_scenes.json`、`chapters.json` 是否完整覆盖对应下层索引

```python
_STEP_ALIASES = {
    "scene_detect": "shot_detect",
    "keyframe_extract": "multi_keyframe",
    "asr": "asr_windowed",
    "character": "character_deep",
    "event": "event_graph",
}
```

## 数据流示意

```
video.mp4
   │
   ├─[1]─→ meta.json
   ├─[2]─→ scenes/scenes.json  ←── 时间轴锚点
   ├─[3]─→ scenes/keyframes/scene_XXXX_f0~f5.jpg  (多帧)
   │
   ├─[4]─→ transcripts.json  (长窗口ASR + scene_index回填 + cross_shot)
   ├─[5]─→ ocr.json + vision.json  (多帧: action/expression/props + micro_clip)
   ├─[6]─→ audio_prosody.json  🆕 (音乐/音效/沉默/语速/音量/语音情绪)
   ├─[7]─→ characters.json  (CharacterDeep: 共现/首末出场/重要性)
   ├─[8]─→ speaker_map.json
   ├─[9]─→ multimodal_alignments.json  🆕 (跨模态一致性)
   │
   ├─[10]→ beats.json  (shot → beat 聚合)      ← 叙事层级构建
   ├─[11]→ story_scenes.json  (beat → scene)    ← 叙事层级构建
   ├─[12]→ chapters.json  🆕 (scene → chapter)  ← 叙事层级构建
   ├─[13]→ events.json + event_graph.json        ← 事件图谱（含证据+置信度）
   ├─[14]→ character_relations.json              ← 人物关系图
   ├─[15]→ edit_signals.json + narrative_signals.json    ← 三类信号
   │       + recomposition_signals.json
   │
   ├─[16]→ memory.json  (四层MemoryUnit汇总 + 三类信号 + 角色判定)
   └─[17]→ index/
       ├── search_index.json      (文本)
       ├── faiss.index + id_map   (向量)
       ├── character_index.json   (角色)
       ├── event_index.json       (事件)
       ├── relation_index.json    (关系)
       ├── emotion_index.json     (情绪)
       ├── edit_signal_index.json (剪辑信号)
       ├── audio_index.json       🆕 (音频)
       └── chapter_index.json     🆕 (章节)
```
