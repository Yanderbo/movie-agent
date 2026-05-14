# 渲染引擎 (render/)

> 文件：`render/engine.py`、`render/validator.py`、`render/ffmpeg_ops.py`
> 职责：读取 EditPlan，通过 FFmpeg 执行裁剪、拼接、转场、音频处理，输出成片

## 总体流程

```
EditPlan JSON
    │
    ▼
┌──────────────┐
│  validator    │  校验 EditPlan 合法性
└──────┬───────┘
       ▼
┌──────────────┐
│  Step 1      │  精确裁剪每个 clip
│  cut_clip    │
└──────┬───────┘
       ▼
┌──────────────┐
│  Step 2      │  标准化参数（分辨率/帧率/编码）
│  normalize   │
└──────┬───────┘
       ▼
┌──────────────┐
│  Step 3      │  拼接所有片段
│  concat      │
└──────┬───────┘
       ▼
┌──────────────┐
│  Step 4      │  混合背景音乐（可选）
│  mix_bgm     │
└──────┬───────┘
       ▼
   output.mp4
```

## 入口函数

```python
def run_render(plan_id: str) -> str
```

1. 加载 `editplans/{plan_id}.json`
2. 加载 VideoMemory 获取源视频路径
3. 调用 `validate_plan()` 校验
4. 执行渲染流水线
5. 返回输出视频路径

## 渲染流水线详解

### Step 1: 裁剪 (cut_clip_precise)

```python
ffmpeg -y -ss {start} -i {source} -t {duration}
       -c:v libx264 -preset fast -crf 20
       -c:a aac -b:a 192k
       -avoid_negative_ts make_zero -fflags +genpts
       {output}
```

- 使用重编码模式（而非 `-c copy`），确保帧精确
- 每个 clip 独立裁剪为 `clip_XXX.mp4`

裁剪后还会按需进行：
- **变速处理**：`setpts=PTS/speed` + `atempo=speed`（atempo 限制 0.5-2.0，超出时链式组合）
- **音量调整**：`volume={value}` 滤镜
- **淡入淡出**：首个 clip 可 fade_in，末尾 clip 可 fade_out

### Step 2: 标准化 (normalize_clip)

统一所有片段的参数，确保拼接兼容：
- 视频编码：libx264
- 音频编码：aac，192k，44100Hz，双声道
- 可选分辨率缩放 + 黑边填充

### Step 3: 拼接 (concat_clips)

使用 FFmpeg concat demuxer：
```python
# 生成 concat.txt
file '/path/to/norm_000.mp4'
file '/path/to/norm_001.mp4'
...

ffmpeg -y -f concat -safe 0 -i concat.txt -c copy {output}
```

### Step 4: BGM 混合 (mix_bgm)

```python
filter_complex = "[1:a]volume={bgm_vol},afade=in:...,afade=out:...[bgm];
                  [0:a][bgm]amix=inputs=2:duration=first:normalize=0[aout]"
```

- 仅当 `plan.bgm.enabled=True` 且 BGM 文件存在时执行
- BGM 音量默认 0.15（相对原音 15%）
- 自动添加 BGM 的淡入 (2s) 和淡出 (3s)

## FFmpeg 操作封装

`render/ffmpeg_ops.py` 提供以下原子操作：

| 函数 | 功能 |
|------|------|
| `cut_clip_precise()` | 精确裁剪（重编码） |
| `normalize_clip()` | 标准化参数 |
| `concat_clips()` | concat demuxer 拼接 |
| `apply_fade()` | 淡入淡出（视频+音频） |
| `mix_bgm()` | BGM 混合 |
| `adjust_volume()` | 音量调整 |
| `adjust_speed()` | 变速处理 |

所有函数都委托给 `utils/ffmpeg_utils.py` 的 `_run_cmd()` 执行子进程。

## 校验器 (validator.py)

`validate_plan()` 在渲染前做最后一道防线：
- 检查源视频文件是否存在
- 校验每个 clip 的 `source_scene_index` 是否在合法范围
- 校验 `source_start < source_end`
- 校验时间范围是否在源场景内
- 返回错误列表，非空则拒绝渲染

## 临时文件处理

- 所有临时文件（裁剪片段、标准化片段、拼接中间文件）保存在 `renders/{render_id}/clips/`
- 渲染完成后自动清理 `clips/` 目录
- 最终输出保留在 `renders/{render_id}/output.mp4`
