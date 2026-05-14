# 理解流水线 (understand.py)

> 文件：`pipeline/understand.py`
> 职责：编排 14 步理解流水线，将一个视频文件从原始输入转化为多层结构化 Video Memory

## 总体流程

```
ingest → shot_detect → multi_keyframe → asr_windowed → vision → character_deep
  → speaker_bind → beat_detect → story_scene_detect → event_graph
  → character_arc → edit_signal → build_memory → indexer
```

核心设计原则：
1. **先切后提** ——先完成镜头切分（step 2），再在每个 shot 内提取各模态数据
2. **层层聚合** ——从 shot 到 beat 到 story_scene，逐级构建更高层叙事单元
3. **信号驱动** ——计算 EditSignal 为 DirectorAgent 提供量化选材依据

### 叙事层级结构

```
Shot (镜头)           ← 最小视觉单元，由 scene_detect 切分
 └─ Beat (节拍)       ← 2-8 个连续 shot 组成的叙事微单元
     └─ StoryScene    ← 2-6 个连续 beat 组成的完整叙事场景
         └─ EventGraph ← 事件节点 + 因果/铺垫/反转关系边
```

## 入口函数

```python
def run_understand(video_path, video_id=None, resume=False) -> str:
```

- `video_path`：新视频文件路径（首次处理必传）
- `video_id` + `resume=True`：从断点继续（按 `progress.json` 中记录的已完成步骤确定起始位置）

## 14 步详解

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

**v2 变更**：从每 shot 单帧升级为多帧动态采样。

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

**v2 变更**：从按 shot 段逐个转写，改为按 5 分钟长窗口整体转写 + 按时间戳回填 shot。

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

### Step 5: Vision（OCR + 多帧画面理解）

**模块**：`pipeline/vision.py`

**v2 变更**：支持多帧输入，输出动作/表情/道具变化描述。

**流程**：
1. 分批处理（默认 `batch_size=5`）
2. 多帧 shot：发送多张帧图片给 Gemini Vision，分析动作变化和表情变化
3. 单帧 shot：分析单帧（传统模式）
4. 返回结果包含 v2 新增字段：`action_description`, `frame_descriptions`, `expression_changes`, `props`
5. 写出 `ocr.json` 和 `vision.json`

**输出**：`ocr.json`、`vision.json`

---

### Step 6: Character Deep（深度人物分析）

**模块**：`pipeline/character.py`

**v2 变更**：输出 `CharacterDeep`（继承自 `Character`），新增深度字段。

**流程**：
1. 人脸检测：InsightFace（优先）或 Gemini Vision（降级）
2. 多帧人脸检测：遍历 `keyframe_paths` 中所有帧
3. 聚类 + 描述生成
4. 计算深度字段：`first_appearance`, `last_appearance`, `co_appearing_characters`
5. 写出 `characters.json`

**输出**：`characters.json`（`CharacterDeep` 对象列表）

---

### Step 7: Speaker Bind（Speaker ↔ Character 绑定）

**模块**：`pipeline/speaker_bind.py`（未修改）

**流程**：
1. 收集 ASR 中的 `speaker_id`
2. 统计共现矩阵
3. LLM 确认映射
4. 回写到 `TranscriptSegment.character_id` 和 `Character.speaker_ids`

**输出**：`speaker_map.json`、更新后的 `transcripts.json` 和 `characters.json`

---

### Step 8: Beat Detect（剧情节拍检测）🆕

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

### Step 9: Story Scene Detect（故事场景检测）🆕

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

### Step 10: Event Graph（事件图谱构建）🆕 升级

**模块**：`pipeline/event.py`

**v2 变更**：从扁平事件列表升级为事件图谱（含关系边）。

**流程**：
1. 事件抽取（同 v1）：LLM 从多模态上下文中抽取事件
2. **新增**：事件关系推理（`_extract_event_edges`）
   - 关系类型：`cause` / `foreshadow` / `reversal` / `escalation` / `resolution` / `parallel`
   - LLM 分析事件间的因果、铺垫、反转等关系
3. 填充 `Event.beat_indices` 和 `Event.story_scene_indices`
4. 构建 `EventGraph`（含 `nodes` + `edges`）
5. 写出 `events.json` 和 `event_graph.json`

**输出**：`events.json`、`event_graph.json`

---

### Step 11: Character Arc（人物弧线 + 关系图）🆕

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

### Step 12: Edit Signal（剪辑信号计算）🆕

**模块**：`pipeline/edit_signal.py`

**流程**：
1. 为每个 **beat** 计算 8 维剪辑信号（核心粒度）
2. 为每个 **story_scene** 计算信号
3. 为 **重要 shot**（beat 首尾 + 高重要性事件所在 shot）计算信号
4. 每批 15 个单元发送给 LLM 评估
5. 写出 `edit_signals.json`

**8 维信号**：`hook_score` / `plot_importance` / `emotional_intensity` / `visual_impact` / `independence_score` / `continuity_dependency` / `boundary_quality` / `spoiler_level`

**建议用途标签**：`hook` / `trailer` / `highlight` / `recap` / `climax_clip` / `character_intro`

---

### Step 13: Build Memory（构建多层 VideoMemory）升级

**模块**：`pipeline/memory_builder.py`

**v2 变更**：从单层 MemoryUnit 升级为三层。

**子流程 A：Shot 级 MemoryUnit**
- 同 v1，但新增 `beat_index`, `story_scene_index`, `edit_signal` 字段
- `combined_text` 新增 `动作:` 字段

**子流程 B：Beat 级 BeatMemoryUnit** 🆕
- 聚合 beat 内所有 shot 的台词和画面

**子流程 C：StoryScene 级 SceneMemoryUnit** 🆕
- 聚合 story_scene 内所有 beat 的描述

**子流程 D：角色判定**
- 统计台词量、事件参与度、出镜时长、`importance_score`
- LLM 判定 `male_lead` / `female_lead` / `villain` / `supporting` / `minor`

**最终**：组装 `VideoMemory`，写出 `memory.json`

---

### Step 14: Indexer（构建七维检索索引）升级

**模块**：`pipeline/indexer.py`

**v2 变更**：从 3 种索引扩展为 7 种。

| # | 索引 | 文件 | 说明 |
|---|------|------|------|
| 1 | 文本索引 | `search_index.json` | 关键词 + 2-gram（兼容 v1） |
| 2 | 向量索引 | `faiss.index` + `id_map.json` | Embedding 余弦相似度 |
| 3 | 角色索引 | `character_index.json` | character_id → shots/beats/scenes |
| 4 | 事件索引 | `event_index.json` | event_type → 按重要性排序 |
| 5 | 关系索引 | `relation_index.json` | char_a\|char_b → 关系/共现 |
| 6 | 情绪索引 | `emotion_index.json` | emotion → shots/beats 列表 |
| 7 | 剪辑信号索引 | `edit_signal_index.json` | signal_type → top-N + by_usage 反向索引 |

---

## 断点续跑机制

- 每步完成后调用 `_save_progress(video_id, step_name)`，写入 `progress.json`
- resume 时读取 `progress.json`，找到第一个未完成的步骤作为起始点
- **v2 向后兼容**：`_STEP_ALIASES` 将旧步骤名映射为新步骤名

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
   ├─[5]─→ ocr.json + vision.json  (多帧: action/expression/props)
   ├─[6]─→ characters.json  (CharacterDeep: 共现/首末出场/重要性)
   ├─[7]─→ speaker_map.json
   │
   ├─[8]─→ beats.json  (shot → beat 聚合)     ← 叙事层级构建
   ├─[9]─→ story_scenes.json  (beat → scene)   ← 叙事层级构建
   ├─[10]→ events.json + event_graph.json       ← 事件图谱
   ├─[11]→ character_relations.json             ← 人物关系图
   ├─[12]→ edit_signals.json  (8维×3层)        ← 剪辑信号
   │
   ├─[13]→ memory.json  (三层MemoryUnit汇总 + 角色判定)
   └─[14]→ index/
       ├── search_index.json      (文本)
       ├── faiss.index + id_map   (向量)
       ├── character_index.json   (角色)
       ├── event_index.json       (事件)
       ├── relation_index.json    (关系)
       ├── emotion_index.json     (情绪)
       └── edit_signal_index.json (剪辑信号)
```
