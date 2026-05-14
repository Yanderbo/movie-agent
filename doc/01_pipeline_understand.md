# 理解流水线 (understand.py)

> 文件：`pipeline/understand.py`
> 职责：编排 10 步理解流水线，将一个视频文件从原始输入转化为结构化的 Video Memory

## 总体流程

```
ingest → scene_detect → keyframe → asr → vision → character
  → speaker_bind → event → build_memory → indexer
```

核心设计原则：**先切后提** ——先完成镜头切分（step 2），再在每个 shot 内提取各模态数据，保证所有数据天然以 `scene_index` 为锚点对齐。

## 入口函数

```python
def run_understand(video_path, video_id=None, resume=False) -> str:
```

- `video_path`：新视频文件路径（首次处理必传）
- `video_id` + `resume=True`：从断点继续（按 `progress.json` 中记录的已完成步骤确定起始位置）

## 10 步详解

---

### Step 1: Ingest（入库）

**模块**：`pipeline/ingest.py`

**流程**：
1. 校验文件存在性
2. 生成人类可读的 `video_id`：取文件名 → 去特殊字符 → 截断30字符 → 拼接 `_` + 路径MD5前8位
   - 例：`my_movie_3f7a2b1c`
3. 在 `data/videos/{video_id}/` 下创建工作目录
4. 复制源视频到 `original.mp4`（如已存在则跳过）
5. 调用 `ffprobe` 解析元信息（时长、分辨率、帧率、编码、文件大小）
6. 写出 `meta.json`

**输出**：`meta.json`（VideoMeta 对象序列化）

**关键点**：
- `video_id` 不再使用 UUID，改用 `文件名_hash` 提高可读性
- `original_path` 记录原始位置，`storage_path` 记录副本位置

---

### Step 2: Scene Detect（镜头切分）

**模块**：`pipeline/scene_detect.py`

**流程**：
1. 使用 PySceneDetect 的 `ContentDetector` 检测镜头边界
2. 参数：`threshold`（默认27.0，越低越灵敏）、`min_scene_len`（默认1.0秒）
3. 遍历检测到的 `(start, end)` 对，构建 `Scene` 对象（含 `scene_index`、`start_time`、`end_time`、`duration`）
4. 写出 `scenes/scenes.json`

**输出**：`scenes/scenes.json`

**关键点**：
- 此步骤**必须在 ASR 和 Vision 之前完成**，因为后续所有模态数据都以 `scene_index` 为锚点
- 结果会被后续所有步骤复用

---

### Step 3: Keyframe Extract（关键帧抽取）

**模块**：`pipeline/keyframe.py`

**流程**：
1. 遍历每个 `Scene`，在 `start_time + offset` 位置提取一帧
2. `offset = min(0.5s, duration * 0.3)`——避免提取到转场模糊帧
3. 调用 `ffmpeg -ss {timestamp} -vframes 1` 提取 JPEG
4. 保存到 `scenes/keyframes/scene_XXXX.jpg`
5. 更新 `Scene.keyframe_path` 字段
6. 重写 `scenes.json`（包含 `keyframe_path`）

**输出**：`scenes/keyframes/*.jpg` + 更新后的 `scenes.json`

**关键点**：
- 已存在的关键帧会跳过（支持断点续跑）
- 关键帧质量由 `config.KEYFRAME_QUALITY` 控制

---

### Step 4: ASR（按 shot 段语音转文字）

**模块**：`pipeline/asr.py`

**流程**：
1. 遍历每个 `Scene`
2. 对每个 scene，用 `extract_audio_segment()` 从原始视频中提取该 shot 时间段的音频片段（WAV 格式）
3. 将音频片段发送给 Gemini Audio API 做 ASR
4. 解析 JSON 响应，将段内相对时间转换为全局绝对时间：`global_time = scene.start_time + segment_relative_time`
5. 每条 `TranscriptSegment` 天然携带 `scene_index`
6. 对于超长 shot（>ASR_CHUNK_DURATION），内部再切分为多段分别转写后合并
7. 全部结果按全局时间排序，写出 `transcripts.json`

**输出**：`transcripts.json`、`audio_shots/*.wav`

**关键点**：
- 不再提取整体音频 → 改为按 shot 段提取，保证时间轴天然对齐
- 极短镜头（<0.3s）跳过 ASR
- 时间修正：`seg_start = max(seg_start, scene.start_time)`，防止超出 shot 边界
- API 速率控制：每个 shot 间隔 0.5s

---

### Step 5: Vision（OCR + 画面摘要）

**模块**：`pipeline/vision.py`

**流程**：
1. 过滤出有关键帧的场景
2. 分批处理（默认 `batch_size=5`）：
   - 单张：调用 `chat_with_media()` 分析单帧
   - 多张：调用 `chat_with_images()` 批量分析
3. Gemini Vision 一次调用同时返回 OCR 文字 + 画面描述 + 检测物体 + 情绪 + 场景类型
4. 批量解析失败时自动退回逐个处理
5. 写出 `ocr.json` 和 `vision.json`

**输出**：`ocr.json`（OCRResult列表）、`vision.json`（VisionSummary列表）

**关键点**：
- 合并 OCR 和画面摘要到一次 API 调用，减少消耗
- 批量处理失败后有逐个重试的 fallback
- `scene_index` 由关键帧对应的 Scene 直接确定

---

### Step 6: Character（人物识别）

**模块**：`pipeline/character.py`

**流程**：
1. **人脸检测**：优先使用 InsightFace（`buffalo_l` 模型），提取人脸 bbox + embedding
2. **Gemini Fallback**：InsightFace 不可用时，用 Gemini Vision 分析关键帧中的人物外观
3. **聚类**：
   - 有 embedding 时：DBSCAN（余弦距离，eps=0.5）
   - 无 embedding 时：基于描述文本的 2-gram 相似度匹配
4. **描述生成**：每个聚类选最佳人脸缩略图，用 Gemini 生成 50 字以内的外观描述
5. 按出场次数降序排列
6. 写出 `characters.json`

**输出**：`characters.json`、`characters/*.jpg`（人脸缩略图）

**关键点**：
- `appearance_scenes` 记录出现在哪些 scene
- `total_screen_time` 按出现 scene 的 duration 累加
- 缩略图会做边界扩展（pad 30%）以包含更多面部周围信息

---

### Step 7: Speaker Bind（Speaker ↔ Character 绑定）

**模块**：`pipeline/speaker_bind.py`

**流程**：
1. 收集所有 ASR 中出现的 `speaker_id`
2. 统计共现矩阵：哪些 `speaker_id` 和 `character_id` 在同一个 scene 中出现
3. 构造 prompt 包含：说话人台词样本、人物描述、共现统计
4. 调用 LLM 确认/修正映射
5. LLM 失败时 fallback：按共现频率最高的一对一配对
6. 回写映射到：
   - `TranscriptSegment.character_id`
   - `Character.speaker_ids`
   - `speaker_map.json`

**输出**：`speaker_map.json`、更新后的 `transcripts.json` 和 `characters.json`

**关键点**：
- 使用贪心策略避免一个 character 被多个 speaker 重复绑定
- 无 speaker 或无 character 时安全跳过

---

### Step 8: Event（事件抽取）

**模块**：`pipeline/event.py`

**流程**：
1. 对于短视频（<30min）一次性处理；长视频按 30min 分段
2. 将台词、画面摘要、人物信息组织为 prompt 上下文
3. 调用 Gemini 抽取事件列表（event_type、时间范围、涉及人物、情绪、重要性1-10）
4. 解析响应，构建 `Event` 对象
5. 重新编号 `event_index`
6. **填充 `scene_indices`**：遍历每个 event 和 scene，将时间范围有交叉的 scene 加入
7. 写出 `events.json`

**输出**：`events.json`

**关键点**：
- `scene_indices` 实现了事件与镜头的双向关联
- 事件类型涵盖：对话、冲突、转折、高潮、结局、日常、回忆、独白等

---

### Step 9: Build Memory（构建 Video Memory）

**模块**：`pipeline/memory_builder.py`

**子流程 A：构建 MemoryUnit**

1. 将各模态数据按 `scene_index` 分组索引
2. 对每个 Scene 创建一个 `MemoryUnit`，融合：
   - 该 shot 内的台词列表
   - 该 shot 的画面摘要
   - 该 shot 的 OCR 结果
   - 该 shot 出现的人物 ID
   - 与该 shot 时间重叠的事件
3. 生成 `combined_text`：将所有模态拼接为一段检索文本
   - 格式：`台词: xxx | 画面: xxx | 情绪: xxx | 类型: xxx | 事件[转折]: xxx | 人物: char_000`
4. `embedding` 字段留空，在 step 10 由 indexer 填充

**子流程 B：角色判定**

1. 统计每个 character 的台词量、事件参与度、出镜时长占比
2. 构造 prompt 让 LLM 判定业务角色（male_lead / female_lead / villain / supporting / minor）
3. LLM 失败时 fallback：按出镜时长排序，第一名 = male_lead，第二名 = female_lead

**最终**：将所有数据组装为 `VideoMemory` 对象，调用 `save_memory()` 写出 `memory.json`

**输出**：`memory.json`

---

### Step 10: Indexer（构建检索索引）

**模块**：`pipeline/indexer.py`

**流程**：

1. **文本索引**（Part 1）：
   - 遍历每个 Scene / MemoryUnit
   - 提取 `combined_text`，生成关键词列表（空格分词 + 中文 2-gram）
   - 写出 `index/search_index.json`

2. **Embedding 向量**（Part 2）：
   - 收集所有 MemoryUnit 的 `combined_text`
   - 分批（每批20个）调用 Embedding API 生成向量
   - 将向量写回 MemoryUnit.embedding，更新 `memory.json`

3. **FAISS 索引**（Part 3）：
   - 将所有 embedding 做 L2 归一化
   - 构建 `IndexFlatIP`（内积索引，归一化后等价于余弦相似度）
   - 写出 `index/faiss.index` 和 `index/id_map.json`

**输出**：`index/search_index.json`、`index/faiss.index`、`index/id_map.json`

**关键点**：
- FAISS 未安装时优雅降级（跳过向量索引）
- Embedding API 不可用时也优雅降级
- `id_map.json` 记录 FAISS 内部索引号 → scene_index 的映射

---

## 断点续跑机制

- 每步完成后调用 `_save_progress(video_id, step_name)`，写入 `progress.json`
- resume 时读取 `progress.json`，找到第一个未完成的步骤作为起始点
- 每个子模块内部也有缓存检查（如 `transcripts.json` 已存在则直接加载）

## 数据流示意

```
video.mp4
   │
   ├─[1]─→ meta.json
   │
   ├─[2]─→ scenes/scenes.json  ←─── 所有后续步骤的时间轴锚点
   │
   ├─[3]─→ scenes/keyframes/*.jpg
   │           ↓
   ├─[4]─→ transcripts.json  (每条带 scene_index)
   │           ↓
   ├─[5]─→ ocr.json + vision.json  (每条带 scene_index)
   │           ↓
   ├─[6]─→ characters.json  (每个带 appearance_scenes)
   │           ↓
   ├─[7]─→ speaker_map.json  (speaker_id → character_id)
   │        + 更新 transcripts.json (character_id)
   │        + 更新 characters.json (speaker_ids)
   │           ↓
   ├─[8]─→ events.json  (每个带 scene_indices)
   │           ↓
   ├─[9]─→ memory.json  (汇总 + MemoryUnit + 角色判定)
   │           ↓
   └─[10]→ index/search_index.json + faiss.index + id_map.json
```
