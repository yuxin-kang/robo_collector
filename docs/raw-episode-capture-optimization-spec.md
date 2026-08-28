# 多相机采集 Raw Episode 与可审计数据管线优化 Spec

状态：Final
日期：2026-08-26
适用项目：`robo_collector`
相关参考：[Robo-DM 论文](https://arxiv.org/abs/2505.15558)、[Robo-DM 仓库](https://github.com/BerkeleyAutomation/robodm)

## 1. 目标

在不破坏现有 Unitree G1、双路 RealSense、ZMQ、ROS、LeRobot、GR00T 和 OpenPI 流程的前提下，将采集系统升级为：

1. 原始数据可追溯、可重放、可重新对齐和重新编码。
2. 实时采集路径不依赖逐帧 MP4 编码。
3. 相机、机器人状态和动作的时间语义明确，误差可量化。
4. 丢帧、过期、乱序、重连和转换失败均可审计。
5. LeRobot、GR00T、OpenPI 等格式从同一个原始 Episode 派生。
6. 进程崩溃或机器重启后，未完成 Episode 能被识别并安全恢复或丢弃。

## 2. 非目标

- 第一阶段不整体替换现有 ROS、Unitree 或 ZMQ 架构。
- 第一阶段不直接引入 Robo-DM 的 EBML 容器、mmap 解码缓存或训练侧数据加载器。
- 不把网络接收时间冒充硬件曝光时间。
- 不承诺跨设备的物理级同步；如果需要亚毫秒级同步，必须另行引入硬件触发、PTP 或专用同步方案。
- 不在采集线程中为了“补齐帧数”复制旧图像，也不静默丢弃无法对齐的数据。

## 3. 当前实现与问题

### 3.1 当前数据路径

```text
RealSense
  -> server_realsense.py：取最新帧、JPEG 编码、ZMQ PUB
  -> client.py：ZMQ SUB、CONFLATE、JPEG 解码为 RGB
  -> CameraFrameCache：保留最新完整相机 bundle
  -> collector_node.py：按定时器取最新状态和相机帧
  -> LeRobotV21Writer：同步写 MP4 + Parquet
  -> EpisodeSaveWorker：Episode 结束后执行保存/收尾
```

### 3.2 已确认的风险

| 风险 | 当前行为 | 影响 |
|---|---|---|
| 时间戳语义不足 | `server_realsense.py` 在取帧后用 `time.time()` 生成时间戳 | 无法证明是设备采集/曝光时间 |
| 最新帧语义 | 每路相机和缓存均保留最新帧 | 中间帧可能被跳过，当前没有完整丢帧统计 |
| 主动丢帧 | `client.py` 默认使用 `CONFLATE` 和较小 `RCVHWM` | 适合低延迟，不适合默认保证完整采集 |
| 二次处理 | 服务端 JPEG，客户端解码 RGB，采集节点再用 OpenCV 写 `mp4v` | CPU、内存复制和二次有损压缩开销较大 |
| 固定时间轴 | LeRobot `timestamp` 由 `frame_index / fps` 生成 | 采集间隔异常时，训练时间轴与真实时间不一致 |
| 质量信息分散 | 有 session、sequence、年龄和 skew 校验，但缺少 Episode 级汇总 | 难以筛选和定位坏数据 |

现有实现中需要保持的优点包括：Episode 生命周期、异常丢弃、`.inprogress` 暂存、原子提交、校验和、启动恢复、命令幂等和下游转换测试。

## 4. 目标架构

```text
                 ┌──────────────────────────────┐
                 │ RealSense / Robot sources     │
                 │ device timestamp + sequence   │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │ Camera-side capture spool     │
                 │ complete-source target        │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │ Camera/State transport        │
                 │ raw payload + provenance      │
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │ RawEpisodeRecorder            │
                 │ bounded queue + durable chunks│
                 └──────────────┬───────────────┘
                                │ STOP / finalize
                 ┌──────────────▼───────────────┐
                 │ EpisodeMaterializer           │
                 │ alignment + MP4/Parquet export│
                 └──────────────┬───────────────┘
                                │
                 ┌──────────────▼───────────────┐
                 │ EpisodeQualityGate            │
                 │ report + READY/REVIEW/REJECT  │
                 └──────────────┬───────────────┘
                                │
              LeRobot / GR00T / OpenPI / optional Robo-DM
```

当前目标架构中，相机侧 spool 是完整采集源；主机侧 Raw recorder 是传输层观测源。迁移阶段允许先上线主机侧 shadow recording，但其 manifest 必须标记 `source_scope=transport_observed`，不能声称恢复了传输前已经被 `latest` 或 `CONFLATE` 丢弃的帧。MP4、Parquet 和转换数据都是可删除、可重建的派生物。

## 5. 数据契约

### 5.1 相机数据包

新增 v4 时不能直接改变 `CameraFrameCache` 的输入结构。协议层分为两步：

```text
wire packet v3/v4
  -> decode_envelope()
  -> NormalizedPacket
  -> optional decode_images()
```

`CameraFrameCache` 只依赖 `NormalizedPacket`。v3 继续作为默认发送协议；客户端先支持 v3/v4 双读，完成部署后再通过能力开关启用 v4。v3 中没有设备时间戳的帧必须标记为 `timestamp_quality=host_after_capture`，不能被补写成 device timestamp。

v4 的逻辑结构如下（实际传输仍使用 msgpack binary）：

```json
{
  "schema": "robo_collector_camera.v4",
  "session_id": "camera-session-id",
  "streams": {
    "head": {
      "sequence": 1234,
      "payload": "<jpeg bytes>",
      "payload_encoding": "image/jpeg",
      "timestamps": {
        "device": 1710000000123.4,
        "device_unit": "ms",
        "device_clock_domain": "realsense:<serial>",
        "server_wall": 1710000000.456,
        "server_monotonic": 12345.678
      },
      "timestamp_quality": "device"
    }
  },
  "metadata": {
    "cameras": {
      "head": {"serial": "camera-serial", "device_info": {}}
    },
    "width": 640,
    "height": 480,
    "fps": 30,
    "payload_encoding": "jpeg"
  }
}
```

标准化后的 `NormalizedCameraFrame` 至少包含：`stream`、`sequence`、原始 `payload`、`payload_encoding`、`timestamp_quality`、设备/服务端/接收端时间戳和 `clock_domain`。v3 到标准化结构的映射为：`images[name] -> payload`、`sequences[name] -> sequence`、`timestamps[name] -> server_wall`。

要求：

- 优先使用 RealSense 提供的设备时间戳；如果设备时间不可用，必须标记 `timestamp_quality=host_after_capture`。
- `sequence` 必须是设备/采集流内单调递增的整数。
- `session_id` 在相机服务重启时改变。
- 保留原始 JPEG bytes；解码后的 RGB 只作为预览、质检或下游物化输入。
- 不跨不同 `clock_domain` 直接比较绝对时间；必须先进行时钟映射或使用本地 monotonic 接收时间。
- 协议版本切换期间，v3/v4 不能混用同一 packet 解析路径；不支持的版本必须显式报错。

### 5.2 Raw Episode 目录

Raw Episode 的目录布局在相机侧和主机侧保持一致，`source_scope` 区分它是完整采集源还是传输观测副本。相机侧由于使用 `RSP1` framed-msgpack，manifest 使用独立的 `robo_collector.camera_spool.v1` schema；主机任务 Raw 使用 `robo_collector.raw_episode.v1`/`RER1` JSON-base64。collector 只接受经校验的相机源快照并转换到主机 Raw，不能把两种 record 编码直接互读。具体 chunk 编码可以复用 msgpack，第一阶段不新增第三方依赖：

```text
raw_episodes/<episode_id>/
├── manifest.inprogress.json
├── camera/
│   ├── head/chunk-000000.msgpack
│   └── ego_view/chunk-000000.msgpack
├── robot/state-000000.jsonl
├── events/events-000000.jsonl
└── checksums.json
```

关闭 raw 记录后，先 fsync chunk 和目录，再原子地将 manifest 变为 `status=RAW_CLOSED`：

```text
raw_episodes/<episode_id>/
├── manifest.json
├── camera/...
├── robot/...
├── events/...
└── checksums.json
```

`manifest.json` 是唯一持久状态源，不再同时用 `COMPLETE`、`PUBLISHED` 等标记文件表达另一套状态。它至少包含：

```json
{
  "schema": "robo_collector.raw_episode.v1",
  "episode_id": "...",
  "status": "RAW_CLOSED",
  "source_scope": "transport_observed",
  "capture_plane": "host_receiver",
  "task_prompt": "...",
  "collector_git_commit": "...",
  "config_hash": "...",
  "start_wall_time": 0.0,
  "end_wall_time": 0.0,
  "streams": {
    "head": {
      "serial": "...",
      "frame_count": 0,
      "first_sequence": 0,
      "last_sequence": 0,
      "sequence_gap_count": 0,
      "clock_domain": "...",
      "timestamp_quality": "device",
      "producer_gap_count": 0,
      "transport_gap_count": 0,
      "selection_gap_count": 0
    }
  },
  "materialization_jobs": [],
  "artifacts": {},
  "quality": {},
  "termination": {
    "command": "STOP",
    "reason": "user_stop"
  }
}
```

### 5.3 每帧 provenance

相机和状态记录均需要保留以下字段的可用子集：

```text
episode_id
stream_name
sequence
clock_domain
device_timestamp
server_wall_timestamp
receive_monotonic_timestamp
record_monotonic_timestamp
selected_dataset_timestamp
alignment_residual_sec
selection_policy
```

`selected_dataset_timestamp` 和 `alignment_residual_sec` 是物化阶段产生的派生字段，不能覆盖原始时间戳。

## 6. 时间对齐策略

### 6.1 采集阶段

采集阶段只保存原始事件及其时间戳，不进行不可逆插值。

- 完整采集模式应在相机侧先写入 spool，再更新 `latest` 并发送 ZMQ；这样 source-side spool 才能覆盖传输前的帧。
- 迁移阶段如果只在主机接收端写 Raw，必须标记 `source_scope=transport_observed`，不能声称恢复了 `latest` 或 `CONFLATE` 丢失的帧。
- `latest` 继续用于预览和低延迟模式，但每层都要记录上一 sequence、当前 sequence 和缺口分类。
- 相机、状态和事件的原始时间戳必须原样保存；采集阶段不对跨 clock domain 的绝对时间做未经校准的比较。

缺口统一分为三类：

```text
producer_gap  ：源设备/相机服务产生的 sequence 本身不连续
transport_gap  ：已发送但接收端未收到的 sequence
selection_gap  ：已收到，但物化时没有满足时间窗口的样本
```

### 6.2 物化阶段

默认使用配置中第一路 RGB stream 的真实采集时间戳作为目标时间轴。每一帧参考 RGB 都对应一行候选输出，不能根据 `output_fps` 重新合成目标时刻：

```text
t_k = timestamp(reference_rgb[k])
```

对每个参考 RGB 时刻：

- 参考 RGB 直接作为该行图像；其他相机图像使用相对于参考 RGB 时刻的有界最近邻，不在图像之间插值。
- 机器人状态使用相对于参考 RGB 时刻的有界最近邻；因此最终保存的机器人状态是对齐 RGB，而不是让 RGB 去追随机器人频率。
- 动作默认使用参考 RGB 时刻之前最近的有效动作，即 zero-order hold。
- 标准 LeRobot 物化使用 `strict` 策略：任意必需相机在窗口内没有有效帧时，丢弃整个目标采样点，不复制旧图像或生成黑帧。
- `strict` 策略下，Parquet 行数与每路视频帧数必须一致；被丢弃的目标采样点只进入 QC 报告，并计为 `selection_gap`。
- 可选的 `sparse` 策略仅用于诊断导出：保留行并写入 `camera_valid_<stream>=false`，不作为标准 LeRobot 训练数据发布。
- LeRobot 的 `timestamp` 仍按成功写入行的 `frame_index / output_fps` 生成，以满足固定帧率视频和训练接口；每行另外保存真实 RGB 目标时间戳、来源 sequence 和 `alignment_residual_sec`，保证对齐过程可复现。

### 6.3 同步质量

每个 Episode 统计并输出：

```text
camera-camera skew: P50 / P95 / P99 / max
state-camera skew: P50 / P95 / P99 / max
per-stream timestamp gap
clock offset/drift estimate（如果有）
source/collector clock-mapping uncertainty；缺失或超过配置阈值进入 `REVIEW`
```

阈值由采集配置决定，不在代码中用固定的 100 ms 代表所有任务。项目文档目前已经将现有机制定义为 best effort，因此新实现不得把该校验描述成硬件同步保证。

## 7. 模块改动范围

### 7.1 `src/camera/robo_collector_camera/`

- 增加 `decode_envelope(v3|v4) -> NormalizedPacket`，保留 v3/v4 双读；v3 仍为默认发送协议，v4 通过能力开关启用。
- 记录设备时间戳、时钟域和服务器 monotonic 时间。
- 增加 raw payload 读取接口或等价的延迟解码接口，避免采集热路径强制解码 JPEG。
- 相机侧 spool 在“完整采集模式”中先于 latest 聚合和 ZMQ 发送落盘；如果暂时无法部署，只能提供 transport-observed Raw。
- 为每个 stream 暴露产生帧数、发送帧数、sequence gap 和重启次数。
- 保留当前 PUB/SUB 传输；`CONFLATE` 改为显式的 `preview`/`recording` 策略，禁止用无限缓存代替可靠记录。

### 7.2 `src/robo_collector/robo_collector/camera_cache.py`

- 将“最新帧选择”和“丢帧统计”分开。
- 记录每一路的上一 sequence、缺口长度、乱序、重复、过期和 session 切换。
- 在 bundle 中保留原始 payload 和完整 timestamp provenance。
- cache 输入统一使用 `NormalizedPacket`，不直接依赖 v3/v4 wire shape。
- `recording` 模式下，接收队列必须有界；队列溢出时返回明确失败或标记 Episode，而不是静默丢弃。

### 7.3 `collector_node.py` / `save_worker.py`

- `RECORDING` 阶段只向 `RawEpisodeRecorder` 写入 raw frame、状态和事件。
- `STOP` 后先关闭 raw 并生成 durable materialization job，再由 `EpisodeMaterializer` 生成训练格式。
- 现有 `EpisodeSaveWorker` 可以继续一次执行一个任务；必须增加 durable job 状态、启动扫描和 at-least-once 重试，`SimpleQueue` 只用于进度通知，不能作为恢复依据。
- 以 `source_manifest_hash + conversion_config_hash + output_schema_version` 形成幂等键；输出使用临时目录和原子提交。
- 现有 `RecordStateMachine` 的 START/STOP/DISCARD/FAILED 语义保持不变；`OPEN` 等持久阶段不直接替换运行时枚举。
- Gesture trigger 只负责发起命令和附加任务信息，不直接参与视频编码。

### 7.4 `lerobot_dataset.py`

- 第一阶段保留现有 `LeRobotV21Writer`，新增从 Raw Episode 读取并生成 LeRobot 的 materializer。
- 第二阶段关闭实时 `OpenCvVideoSink.write()`，使 `mp4v` 不再位于采集热路径。
- 继续使用现有 `.inprogress`、原子提交、校验和和恢复逻辑。
- 输出中保留源 Episode ID、转换器版本、时间选择策略和源时间戳列。
- 既有数据集的 camera keys、shape、fps、robot type 和 feature schema 校验继续有效。

### 7.5 下游转换器

`gr00t_converter.py` 和 `pi05_converter.py` 均应接受已发布的 LeRobot materialization 或 Raw Episode manifest，并写入：

```text
source_episode_id
source_manifest_hash
converter_version
conversion_config_hash
output_schema_version
```

转换失败只能留下失败产物或临时目录，不能修改 Raw Episode，也不能让部分输出被当成成功数据。

## 8. 视频物化策略

实时路径保存原始 JPEG 后，Episode 结束再编码 MP4。编码器必须通过实际数据 benchmark 选择，不直接照搬 Robo-DM 论文的 CRF 或 codec 参数。

建议顺序：

1. 先用当前 `mp4v` 生成兼容基线，并验证帧数、顺序和画面一致性。
2. 评估 H.264、AV1 的文件大小、CPU 时间、解码速度和下游兼容性。
3. RGB 默认选择兼容性和吞吐更好的方案；AV1 作为归档或大规模存储候选。
4. 深度数据如需要近似无损，单独评估 PNG/FFV1，不与 RGB 共用质量阈值。

如果异步编码仍然需要队列：

- 队列必须有最大容量。
- 队列满时优先阻塞/失败并标记 Episode，而不是静默丢帧。
- 记录 enqueue、encode、flush、commit 各阶段耗时。
- 采集线程不等待耗时不可控的编码器。

## 9. Episode 生命周期与恢复

### 9.1 唯一持久状态

`manifest.json` 中的 `status` 是唯一持久状态源；`COMPLETE`、`PUBLISHED` 等标记文件不再表达另一套生命周期。状态转移为：

```text
OPEN
  -> RAW_CLOSED
  -> MATERIALIZING
  -> MATERIALIZED
  -> QC
  -> READY | REVIEW | REJECT

OPEN -> DISCARDED
OPEN -> QUARANTINED
RAW_CLOSED -> MATERIALIZATION_FAILED -> RAW_CLOSED（重试）
MATERIALIZING -> MATERIALIZATION_FAILED
```

状态含义：

- `OPEN`：仍在接收数据，只有一个 active Episode。
- `RAW_CLOSED`：raw chunk 已关闭、fsync，manifest 已原子提交，可开始物化。
- `MATERIALIZING`：存在正在执行的 durable job。
- `MATERIALIZED`：派生物已原子提交并通过结构检查，尚未完成 QC。
- `QC`：正在运行质量检查。
- `READY`：通过自动发布阈值，可进入下游数据集索引；`REVIEW` 和 `REJECT` 不进入默认训练索引。
- `DISCARDED`：用户或安全策略主动放弃；默认保留 raw 并记录原因。
- `QUARANTINED`：raw 结构损坏或无法安全封存，需要人工处理。

### 9.2 运行时状态映射

现有 `RecordStateMachine` 的 `IDLE/RECORDING/NEED_TO_SAVE/SAVING/FAILED/DISCARD` 保持不变，只映射到持久状态：

| 运行时状态 | 持久动作 |
|---|---|
| `IDLE -> START` | 创建 `OPEN` manifest |
| `RECORDING` | 追加 raw chunk、状态和事件 |
| `STOP` / `NEED_TO_SAVE` | 关闭 raw，提交 `RAW_CLOSED` |
| `SAVING` | 执行 `MATERIALIZING` durable job |
| 保存成功 | `MATERIALIZED -> QC -> READY/REVIEW/REJECT` |
| `FAILED` / `DISCARD` | `DISCARDED` 或 `QUARANTINED` |

### 9.3 崩溃恢复

- `OPEN`：启动时校验 chunk 边界和已写前缀；首版不承诺续写。可完整读取的前缀封存为 `RAW_CLOSED` 并标记 `termination.reason=process_crash`，无法验证的 Episode 进入 `QUARANTINED`。
- `RAW_CLOSED`：创建或恢复 materialization job。
- `MATERIALIZING`：根据输入 manifest hash 和转换配置重新执行；旧临时目录先隔离，不能覆盖已提交产物。
- `MATERIALIZED`/`QC`：从已有派生物继续校验；校验失败回到 `RAW_CLOSED` 重试。
- 所有状态更新、manifest、chunk 和派生物提交均使用临时文件/目录、fsync 和原子 rename。
- `DISCARD` 默认只清理未提交的派生物；raw 的删除由明确的保留策略执行。

### 9.4 Durable materialization job

每个 job 写入 manifest 的 `materialization_jobs`，至少包含：

```text
job_id
source_manifest_hash
conversion_config_hash
output_schema_version
status
attempt_count
last_error
updated_at
```

`EpisodeSaveWorker` 可以继续一次执行一个任务，但必须在启动时扫描未完成 job，并按幂等键执行 at-least-once 重试。进度队列只用于 UI/状态上报，不能作为恢复依据。

## 10. Episode 质量门

### 10.1 必检项

- manifest、chunk、checksums 可读取。
- 所有被 Raw recorder 接受的记录都有 episode/session/stream/sequence 和所需的时间戳字段；缺字段的记录被拒绝并计入错误统计。
- 时间戳在各自 clock domain 内单调或明确标记异常。
- `strict` 输出中 Parquet 行数与每路视频帧数一致；`sparse` 诊断输出不进入标准训练索引。
- 每路已提交视频可完整解码，且视频帧顺序与物化索引一致。
- producer、transport、selection 三类 gap 均有计数和来源，超过配置阈值时不能自动进入 `READY`。
- skew、状态年龄、相机年龄和写入 deadline 均有统计值。
- camera source snapshot 必须包含 chunk prefix、稳定性和 stream high-watermark；
  snapshot 不稳定或与接收端已观察水位不一致时不得自动发布。
- 完整采集模式必须记录 source/collector clock-mapping uncertainty；缺失或超过
  配置阈值时只能进入 `REVIEW`，不能把 START/STOP 窗口当作物理同步保证。
- 状态维度、关节顺序、图像尺寸和 schema 与配置一致。
- 如果采集源的 `source_scope=transport_observed`，需要完整采集保证的配置不得将其标记为 `READY`，只能进入 `REVIEW` 或 `REJECT`。

### 10.2 质量状态

```text
READY   ：满足自动发布阈值
REVIEW  ：存在可接受但需要人工确认的缺口/同步误差
REJECT  ：损坏、严重丢帧、无法对齐或转换失败
DISCARDED：用户或安全策略主动丢弃
```

质量门不能覆盖或删除原始记录；所有规则和阈值写入 `quality.json`。

默认输出策略为 `strict`。`sparse` 只用于分析传输和对齐问题；它必须显式记录缺失相机及原因，不能通过复制上一帧、黑帧或修改视频帧率伪造完整训练样本。

## 11. 兼容与迁移

### 阶段 0：只增加观测

- 继续使用现有 LeRobot writer。
- 增加 producer/transport/selection sequence gap、接收延迟、写入耗时和 Episode QC 统计。
- 不改变输出字段和默认 codec。

### 阶段 1：Raw shadow recording

- 在主机接收端旁路保存 `source_scope=transport_observed` 的 Raw Episode，并验证存储、恢复和指标开销。
- 现有 MP4/Parquet 仍作为主输出。
- 对比 raw 与现有输出的帧数、顺序、时间戳和任务元数据。
- 不把主机 Raw 宣称为完整相机事实源；继续明确记录 ZMQ/CONFLATE 造成的观测缺口。

### 阶段 2：Raw-first materialization

- 部署相机侧 capture spool，或部署等价的逐流有界可靠记录链路，并将 `source_scope` 切换为 `camera_capture`。
- 相机客户端完成 v3/v4 双读，cache 只接收 `NormalizedPacket`；默认发送协议仍为 v3，完成回归后才启用 v4。
- 采集热路径不再实时写 MP4。
- LeRobot 从 Raw Episode 生成，输出 schema 尽量与现有数据兼容。
- 通过回放和帧级对比测试后切换默认路径；首版不承诺崩溃后继续追加同一个 Episode。

### 阶段 3：质量门和多格式派生

- QC 结果参与 READY/REVIEW/REJECT 发布。
- GR00T/OpenPI 转换统一从已发布 Episode 派生。
- 根据训练侧 benchmark 决定是否增加 Robo-DM 归档或解码缓存。

## 12. 验收标准

### 12.1 固定验收配置

功能和性能验收使用同一套可复现配置：

```text
robot: Unitree G1
camera: 2 路 RealSense RGB（head + ego_view）
resolution/fps: 640x480@30 FPS
depth: off（深度另做 codec 测试）
functional set: 10 个 Episode × 60 秒
soak set: 连续 60 分钟
max_raw_queue_depth: 128 packets（必须是有限值）
max_pending_materialization_jobs: 1
```

验收数据需保存测试硬件、相机序列号、配置文件 hash、代码 commit 和测试开始/结束时间。所有 hash 至少覆盖：canonical manifest、每个 raw chunk、每个已提交派生物；编码器版本和转换配置也必须写入 manifest。

### 正确性

- 新 Raw schema 中，缺少必需 session/sequence/clock/timestamp 的记录数必须为 0；不满足条件的记录必须被拒绝并计数。
- Raw Episode 重放时，raw payload、状态顺序和事件顺序与 manifest 中的记录顺序一致。
- Raw 到 LeRobot、GR00T、OpenPI 的每个输出都能反查 source manifest hash、转换配置 hash 和 schema 版本。
- `strict` Episode 的 Parquet 行数与每路视频帧数一致；`sparse` 结果不得进入标准训练索引。
- 同一 input manifest hash、转换配置 hash 和工具版本再次物化时，逻辑结果必须一致：帧数、顺序、source sequence、选帧策略和时间戳完全一致；MP4 字节 hash 如因编码器版本变化不同，必须记录新的 encoder version 和 artifact hash。

### 可靠性

- 在 `OPEN`、关闭 raw、物化、QC、提交各阶段注入进程退出，重启后都能将 Episode 归类为 `RAW_CLOSED`、可重试、`QUARANTINED` 或 `DISCARDED`，不能遗留无状态半成品。
- 断电/异常退出不会把未完成产物标记为 `READY`。
- 队列深度始终不超过 128 packets；队列满、磁盘不足、视频编码失败均会显式失败或进入隔离，不静默成功。
- 对预先注入的 producer/transport/selection 缺口，分类结果必须与注入点一致；未解释 gap 数必须为 0。

### 性能

- 在固定验收配置下，采集热路径 RSS 不得随 Episode 时长线性增长；队列深度和磁盘剩余空间均有监控。
- 相对阶段 0 同硬件基线，采集热路径 CPU p95 增幅不超过 10%，RSS 增幅不超过 20%；无法满足时必须记录原因并调整配置预算。
- collector timer deadline miss rate 不超过 0.1%，或不超过阶段 0 基线加 0.1 个百分点，以较宽者为准。
- 10 个 60 秒 Episode 的物化任务在 20 分钟内完成；超过时必须保留积压、失败原因和可重试 job 状态。

### 质量可观测性

- 每个 `RAW_CLOSED` 或之后的 Episode 都生成质量报告；报告缺失时不能进入 `READY`。
- camera-camera 和 state-camera skew 能计算 P50/P95/P99/max，并与配置阈值比较；该指标不等价于物理捕获同步。
- producer/transport/selection gap、重复、乱序、过期、重连和转换失败均能按 Episode 查询。

## 13. 测试计划

### 单元测试

- v3/v4 packet 编解码和向后兼容。
- timestamp normalization、clock domain 和 alignment residual。
- sequence gap、重复、乱序、session 切换。
- 有界队列满载行为。
- Raw manifest 原子提交和 checksum 校验。
- 相机 source snapshot 的窗口截取、稳定性和 high-watermark 校验。
- clock-mapping uncertainty 缺失/超阈值时的 fail-closed QC。
- materializer 幂等和部分输出清理。

### 故障测试

- 相机服务重启。
- ZMQ 连接断开、消息缺失和消息乱序。
- 单路相机停止更新。
- 状态源停止或时间戳异常。
- 进程在每个生命周期状态中被杀死。
- 磁盘不足和 MP4 编码失败。
- QC 失败后重新物化，不修改 raw。

### 集成/性能测试

- 双路 640x480@30 FPS 连续采集。
- 记录 10/60/300 秒 Episode 的 CPU、内存、磁盘和端到端延迟。
- 比较 `latest/conflate`、有界完整队列和 camera-side raw spool 三种策略。
- 比较 `mp4v`、H.264、AV1 的体积、编码时间、解码时间和下游训练兼容性。

## 14. 关键取舍

| 方案 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| 保持现有实时 MP4 | 改动小 | 编码阻塞采集、二次压缩、难重建 | 仅作为迁移基线 |
| 相机侧 Raw spool + 结束后物化 | 可重放、低热路径负担、格式可重建 | 占用更多原始磁盘，需要相机侧存储与恢复 | 推荐主方案 |
| 直接换 Robo-DM 容器 | 统一多模态容器、利于后续加载 | 引入格式和依赖迁移风险，不能解决同步/丢帧 | 后续评估 |
| 继续默认 CONFLATE | 低延迟、内存可控 | 不能保证完整帧 | 预览模式保留，录制模式显式配置 |
| 关闭 CONFLATE 并无限缓存 | 理论上少丢帧 | 内存失控、延迟不断增长 | 禁止 |

## 15. 实施优先级

```text
P0  时间戳契约 + sequence gap 统计 + Episode provenance
P0  相机侧 Raw spool（无条件时先做 transport-observed shadow recording）
P0  Raw Episode 目录、唯一状态 manifest 和原子恢复
P1  采集热路径与 MP4 物化解耦
P1  Episode QC 和 READY/REVIEW/REJECT
P1  LeRobot/GR00T/OpenPI 统一派生
P2  codec benchmark、Robo-DM 归档、训练侧 cache/mmap
```

第一版实现应优先保证“事实可追溯”和“失败可识别”，不要同时改协议、编码器、训练加载器和所有下游格式。每个阶段都必须保留现有输出作为回归基线，并用实际采集数据完成帧数、顺序、时间戳和性能对比。
