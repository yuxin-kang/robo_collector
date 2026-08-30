# Robo Collector 项目面试讲解文档

> 适用项目：`robo_collector`
> 当前实现：Unitree G1、双路 RealSense RGB、ROS 2、ZMQ、LeRobot v2.1 风格数据集
> 文档依据：仓库当前实现（`HEAD=64a8741`）

这份文档的目标不是让你逐文件背代码，而是把项目讲成一个完整的工程问题：

```text
为什么要做 → 数据链路怎么设计 → 最容易出错的地方是什么
→ 我如何定位问题 → 为什么这样解决 → 还有哪些边界和后续工作
```

面试时应始终以自己实际参与的内容为准。下面的表述以当前仓库实现为依据，
其中“建议说法”可以直接口述。

---

## 1. 先用 30 秒讲清楚项目

### 推荐开场

> 这是一个面向 Unitree G1 遥操作的数据采集系统，负责把 StepIt/ROS 里的机器人状态、动作和 IMU 数据，与两路 RealSense RGB 图像采集并对齐，最终生成 LeRobot 风格的 Parquet 和 MP4 数据集，并支持进一步转换成 GR00T 和 OpenPI pi0.5 所需的格式。
>
> 我把系统拆成相机采集、机器人状态归一化、Episode 生命周期管理、原始数据落盘、停止后的数据物化和质量门几个部分。采集热路径只保存有界的原始记录，不在每个采集 tick 上做不可控的视频编码；Episode 停止后再根据原始时间戳做对齐、写训练格式和 QC。这样遇到丢帧、进程崩溃或转换失败时，能够知道数据发生了什么，并且可以从 Raw Episode 重试，而不是只得到一个看起来完整但无法验证的数据集。

### 一句话定位

> 我做的不是一个简单的“相机转 MP4”脚本，而是一条带有时间语义、丢帧审计、生命周期状态和失败恢复能力的机器人多模态数据管线。

### 两行简历版本

```text
搭建 Unitree G1 双路 RealSense RGB 采集链路，完成 ROS2 状态归一化、ZMQ 远程传输、帧缓存和多源时间对齐。
设计 Raw-first Episode 数据管线，实现异常安全丢弃、可恢复物化、LeRobot Parquet/MP4 输出及 GR00T/OpenPI pi0.5 转换。
```

---

## 2. 面试官需要听到的项目背景

### 2.1 业务/任务目标

机器人模仿学习的数据不是单一图像，而是同一时刻附近的多模态样本，至少包括：

- Unitree G1 的关节位置、速度、力矩、控制量和 IMU；
- StepIt 产生的目标关节位置、对齐后的目标、动作和策略状态；
- `head` 和 `ego_view` 两路 RealSense RGB；
- 任务描述、Episode ID、帧序号、源时间戳和质量信息。

仓库默认的关键维度是：

| 数据 | 当前契约 |
| --- | --- |
| G1 关节自由度 | 29 |
| `aligned_target_pos` | 45 维 |
| StepIt 原始 observation | 1545 维 |
| 策略输入扁平化状态 | 1110 维 |
| 相机 | `head` + `ego_view`，默认 RGB，常用 640×480@30 FPS |

### 2.2 真正的工程难点

面试时不要只说“我用了 ZMQ 和 OpenCV”。真正有价值的难点是：

1. 相机、机器人状态和动作来自不同线程、不同消息源，时间戳语义不完全相同。
2. ZMQ PUB/SUB 和 latest-frame 策略适合低延迟，但天然不能保证接收端看到每一帧。
3. 采集期间如果直接编码 MP4，编码耗时会反过来影响采集线程。
4. STOP、DISCARD、进程退出和相机线程停止可能同时发生，容易产生最后几帧或 manifest 不完整的问题。
5. 一个文件存在不代表 Episode 真正保存成功，Parquet、MP4、manifest 和 QC 可能处于不同阶段。
6. 数据一旦被错误覆盖或错误标记为成功，后续训练很难再判断问题来自采集、传输还是转换。

因此，设计重点是“可追溯、可检测、可恢复”，而不是承诺“任何情况下都不丢帧”。

---

## 3. 整体架构怎么讲

### 3.1 组件关系

```text
                         ┌──────────────────────────┐
                         │ StepIt / ROS2 sources    │
                         │ JointState, IMU, fields  │
                         └────────────┬─────────────┘
                                      │
                                      ▼
                         ┌──────────────────────────┐
                         │ robo_state_node          │
                         │ validate + normalize     │
                         │ publish /robo_state/...  │
                         └────────────┬─────────────┘
                                      │
                                      │ typed RoboStateSample
                                      │
┌───────────────────┐                 ▼
│ RealSense devices │       ┌──────────────────────────┐
│ head + ego_view   │──────▶│ LeRobotCollectorNode     │
└─────────┬─────────┘       │ command + cache + record │
          │                  └──────┬─────────┬─────────┘
          ▼                         │         │
┌───────────────────┐               │         │
│ camera server     │               │         │ START/STOP/DISCARD
│ timestamp/seq     │               │         ▼
│ JPEG + ZMQ PUB    │               │  ┌───────────────┐
│ optional RawSpool │               │  │ RawEpisode    │
└─────────┬─────────┘               │  │ manifest+CRC  │
          │                         │  └──────┬────────┘
          ▼                         │         │ STOP
┌───────────────────┐               │         ▼
│ camera client     │───────────────┘  ┌───────────────┐
│ normalize packet  │                  │ Materializer  │
└─────────┬─────────┘                  │ disk index    │
          ▼                            │ align + write │
┌───────────────────┐                  └──────┬────────┘
│ CameraFrameCache  │                         │
│ latest bundle     │                         ▼
│ bounded callback  │                  ┌───────────────┐
│ gap/skew metrics  │                  │ QC Gate       │
└───────────────────┘                  │ READY/REVIEW  │
                                       │ /REJECT       │
                                       └──────┬────────┘
                                              │
                         ┌────────────────────┴──────────────────┐
                         ▼                                       ▼
                LeRobot Parquet/MP4                    GR00T/OpenPI pi0.5
```

### 3.2 组件职责

| 组件 | 主要职责 | 面试中的关键点 |
| --- | --- | --- |
| `robo_state_node.py` | 订阅 StepIt/ROS2 输入，校验维度和新鲜度，发布统一状态样本 | 先统一数据契约，避免 collector 直接理解一堆上游 topic |
| `state_builder.py` | 解析关节、拼接策略字段、记录源时间戳和缺失字段 | 29 DOF、45 维 target、1110 维 policy state 都有显式校验 |
| `server_realsense.py` | 读取多路 RealSense、编码 RGB、添加 sequence/时间戳、发布 ZMQ | 设备时间不可用时明确标记 `host_after_capture` |
| `client.py` | 兼容 v3/v4 wire envelope，标准化成统一 packet | wire 协议和 cache 输入解耦 |
| `camera_cache.py` | 保留最新完整相机 bundle，管理有界回调队列和 gap/skew 统计 | latest 用于低延迟选择，Raw callback 负责审计收到的包 |
| `collector_node.py` | 控制 Episode、接收状态和相机数据、触发物化与发布状态 | command lock + lifecycle lock + 状态机 |
| `raw_episode.py` | 原始相机/状态/事件的耐久记录和生命周期 manifest | 原始事实和派生训练文件分离 |
| `raw_materializer.py` | 从 Raw Episode 建时间索引、对齐、生成 LeRobot | STOP 后处理，严格选帧，不伪造图像 |
| `lerobot_dataset.py` | 写 Parquet、MP4，处理临时目录、原子提交和恢复 | 输出是派生物，不是唯一事实源 |
| `episode_quality.py` | 汇总 gap、skew、年龄、完整性和 source binding | 只有 `READY` 才算训练侧可发布 |
| `gr00t_converter.py` / `pi05_converter.py` | 将已发布数据或合法 Raw 源转换成下游格式 | 临时目录 + provenance + hash，失败不污染 Raw |

---

## 4. 按数据流讲一遍实现

### 4.1 机器人状态：先归一化，再交给采集器

`robo_state_node.py` 订阅：

- `JointState`；
- `Imu`；
- StepIt 下的目标、动作、observation 和策略字段；
- StepIt 状态诊断消息。

它不会直接把上游数组原样转发，而是通过 `RoboStateAssembler` 做几件事：

1. 检查向量维度和数值是否为有限值。
2. 根据关节名后缀分出真实关节、控制目标和增益数据。
3. 检查 29 个关节的 key 是否一致、是否重复、顺序是否稳定。
4. 缓存每个输入的时间戳，并检查最大年龄和必需输入之间的 skew。
5. 生成统一的 `RoboStateSample`，同时保留 `source_timestamp_names` 和 `source_timestamps_sec`。

一个容易被问到的点是：`Float32MultiArray` 没有 ROS Header，怎么办？

推荐回答：

> 对带 Header 的消息，我优先使用消息源时间；对没有 Header 的 StepIt 数组，在 callback 接收时记录本地时间，并在输出里保留这个时间语义。它不是硬件产生时间，而是 host receive time，所以后续不会把它冒充成物理采样时间。

### 4.2 相机服务：相机时间、服务端时间和 sequence 一起保留

相机端 `RealSenseReader` 对每路相机做以下事情：

1. 调用 RealSense pipeline 读取 RGB，深度是可选路径。
2. 读取设备 frame number 作为流内 sequence；若设备值异常，使用受控的 fallback。
3. 记录设备 timestamp、timestamp domain、server wall time 和 server monotonic time。
4. 使用 JPEG 保存 RGB payload，默认通过 ZMQ PUB 发布 composed packet。
5. 计算相机源端的 `producer_gap_count`。
6. 如果开启相机侧 Raw Spool，先保存原始 payload，再更新 latest，再进入 ZMQ 发布路径。

当前协议策略是：

- v3 仍然是默认发送协议，保持部署兼容；
- v4 是显式 opt-in，增加更完整的 envelope/provenance；
- client 同时具备 v3/v4 标准化读取能力；
- 不支持的 schema、缺失 session、非有限时间戳或非法 payload 会显式拒绝。

面试表达：

> 我没有把网络接收时间伪装成设备曝光时间，而是同时保存 device、server 和 receive 三类时间，并给每一帧一个流内 sequence。这样即使最终需要用 host 时间做对齐，原始证据仍然存在。

### 4.3 ZMQ 客户端和缓存：低延迟视图与可靠记录分开

`CameraClient` 先把 v3/v4/兼容旧 mapping 转成统一的 `NormalizedPacket`，其 frame 中包含：

- stream name；
- 原始 encoded payload；
- payload encoding；
- sequence；
- device/server/receive 时间戳；
- clock domain；
- timestamp quality。

采集模式下，`CameraFrameCache` 使用有界接收策略，默认 callback queue capacity 为 128；
预览模式可以使用 `CONFLATE` 追求最新帧和低延迟。二者的语义不同：

```text
preview    ：允许跳过中间帧，关注“现在画面是什么”
recording  ：队列有限，但不能静默丢弃；满载要暴露为失败/质量问题
```

缓存有两条逻辑路径：

```text
每个有效 packet ───────────────▶ raw callback ─▶ RawEpisodeRecorder
          │
          └────────────────────▶ latest complete bundle ─▶ record tick
```

这样，最新 bundle 被判定为重复、乱序、过期或 skew 超限时，原始收到的 packet 仍然可以进入审计路径。

缓存还统计：

- `producer_gap`：源 sequence 本身不连续；
- `publisher_gap`：相机服务 latest 聚合/发送时跳过的 sequence；
- `transport_gap`：发送端有证据、接收端没有收到；
- `unattributed_gap`：现有证据无法归因；
- duplicate、reorder、session restart；
- stale/expired、queue overflow 和 callback error；
- camera-camera skew 的 P50/P95/P99/max。

### 4.4 Episode 录制：Raw-first 是主路径

当前 `recording_mode` 默认是 `raw_v1`；`raw_first` 是兼容别名，`dual_write` 用于迁移/对比，`mcap_first` 是 gated release candidate。

Raw-first 的核心原则是：

```text
采集期：记录原始 payload、状态、事件和 provenance
停止后：关闭 Raw → 创建 durable job → 对齐 → 生成 Parquet/MP4 → QC
```

Raw Episode 使用以下形式保存：

- `manifest.inprogress.json` / `manifest.json`；
- 相机和机器人状态分目录写 chunk；
- JSON record 带长度和 CRC32 framing；
- 相机 payload 以 Base64 保留原始 bytes；
- checksums 和 record error counters；
- Episode、session、sequence、clock domain、时间戳和 gap 分类。

相机侧 spool 使用自己的 `RSP1` framed-msgpack 格式；主机 task Raw 使用 `RER1` JSON/base64 格式。
两者目录结构相似，但 schema 不相同，导入时必须经过校验适配，不能直接把文件当成同一种格式读取。

### 4.5 物化：RGB 时间轴 + 有界最近邻

`RawEpisodeMaterializer` 不把所有 Raw record 一次性加载到 Python list，而是建立临时 SQLite 时间索引，
每次只查询当前输出帧需要的 state/camera 候选，控制内存上界。

默认物化逻辑可以这样讲：第一路配置的 RGB stream 是参考时间轴，每一帧参考 RGB 都对应一行候选输出：

```text
t_k = timestamp(reference_rgb[k])
```

对每个参考 RGB 时刻 `t_k`：

- 参考 RGB 直接作为该行图像，其他相机在 residual 上限内做有界最近邻，不在图像之间插值；
- 机器人状态在 residual 上限内做最近邻选择，最终是机器人状态对齐 RGB；
- 动作：默认使用 `t_k` 之前最近的有效动作，即 zero-order hold；
- 必需相机或状态缺失：strict 策略下丢弃该目标采样点；
- 不复制上一帧、不补黑帧、不为了凑行数伪造样本；
- 将选择到的 source sequence、source timestamp、residual 和 selection policy 写入输出。

一个重要实现细节是：LeRobot 输出的 `timestamp` 仍使用成功写入的连续 `dataset frame index / output_fps`，
以满足固定帧率视频和训练接口；真实 RGB 目标时间保留在 `alignment_target_source_timestamp`，并同时记录来源 sequence。

### 4.6 QC 和输出发布

物化完成不等于可以训练。`EpisodeQualityGate` 会检查：

- Raw manifest、chunk、checksum 是否可读；
- 必需 provenance 是否齐全；
- Parquet 行数和各路视频帧数是否一致；
- producer/publisher/transport/selection gap 是否超过配置阈值；
- duplicate、reorder、session restart、timestamp anomaly；
- camera-camera skew、state-camera skew、state/camera age；
- camera source snapshot、high watermark 和 clock mapping evidence；
- 输出文件的结构、解码和 hash 证据；
- source scope 是否真的满足“完整采集”的要求。

最终状态为：

| 状态 | 含义 | 是否进入默认训练索引 |
| --- | --- | --- |
| `READY` | 通过自动质量门 | 是 |
| `REVIEW` | 有可解释但需要人工确认的问题 | 否 |
| `REJECT` | 损坏、严重缺失、无法对齐或物化失败 | 否 |
| `DISCARDED` | 用户或安全策略主动放弃 | 否 |
| `QUARANTINED` | Raw 结构或证据损坏，等待人工处理 | 否 |

转换器只接受合格源，输出中带有：

```text
source_episode_id
source_manifest_hash
converter_version
conversion_config_hash
output_schema_version
```

GR00T 和 OpenPI pi0.5 的转换均在临时目录中完成，成功后才原子替换到目标目录；失败时不能把半成品当成成功数据。

---

## 5. Episode 生命周期：面试时按这个顺序讲

### 5.1 持久状态和运行时状态的分工

运行时的 `RecordStateMachine` 负责实时命令语义：

```text
IDLE → RECORDING → NEED_TO_SAVE → SAVING → IDLE
                     └──────────────→ FAILED / DISCARD
```

Raw manifest 负责跨进程、跨重启的持久状态：

```text
OPEN
  → RAW_CLOSED
  → MATERIALIZING
  → MATERIALIZED
  → QC
  → READY / REVIEW / REJECT
```

`manifest.json` 是持久状态的唯一来源；不会通过某个 MP4 文件是否存在来推断“已经保存成功”。

### 5.2 START

收到 START 时：

1. 校验 task prompt、Episode ID 和 command fingerprint。
2. 清理上一 Episode 的 camera cache window，避免 idle 期间的 gap 泄漏到新 Episode。
3. 建立 Raw Episode，写入 `OPEN` manifest、采集配置、git commit 和 config hash。
4. 记录 START event，重置 state/camera sequence 和统计窗口。
5. 后续 camera callback 和 state callback 才能写入这个 recorder。

命令带 `command_id`，同一个命令重试时会被 receipt ledger 识别为 duplicate replay；
如果同一个 `command_id` 的 payload 变了，会被拒绝，避免控制端重试造成两个不同动作。

### 5.3 RECORDING

采集过程中：

- callback 路径接收并校验原始相机 packet；
- 状态回调保存原始状态和接收时间；
- record tick 只使用新鲜、完整、满足 skew 的 state + camera bundle；
- camera cache、callback queue、episode 时长、帧数和磁盘空间都有上限；
- 达到安全限制或发生不可恢复写入错误时，进入安全丢弃，而不是继续制造不完整数据。

### 5.4 STOP

STOP 是一个明确的事务边界：

1. command lock 串行化 STOP 和并发 START/DISCARD。
2. 先等待 callback queue 中已接收的 packet 排空。
3. 对 `camera_capture` 模式，按 START/STOP 的 server-wall-time 窗口 snapshot 相机侧 spool。
4. 关闭 Raw recorder，刷盘并原子发布 `RAW_CLOSED` manifest。
5. 创建 durable materialization job，记录 source manifest hash、conversion config hash 和 schema version。
6. 后台 worker 执行物化，ROS timer 只负责读取进度和结果。

### 5.5 SAVE / QC

物化成功后还要经过质量门：

- `READY`：才调用运行时状态机的 `mark_saved()`；
- `REVIEW` 或 `REJECT`：保持明确的失败/人工处理路径，不能在 UI 上显示“保存成功”；
- 持久化结果、质量报告或 manifest 更新失败：直接报告失败，不能吞异常后继续标记成功。

### 5.6 DISCARD

DISCARD 会记录原因并将 Raw Episode 标记为 `DISCARDED`。默认保留 Raw 作为审计证据，
只清理未提交的派生临时产物；真正删除原始数据属于单独的保留策略，不在采集控制命令里隐式完成。

### 5.7 进程崩溃和重启

启动时扫描 Raw root：

- `OPEN`：校验可读前缀，安全封存或隔离；首版不承诺在原 Episode 上继续追加；
- `RAW_CLOSED`：创建或恢复物化 job；
- `MATERIALIZING`：按 durable job 重新执行；
- `MATERIALIZED/QC`：重新验证 artifact、provenance 和质量证据；
- chunk、manifest 或证据不可验证：进入 `QUARANTINED`，不覆盖原始证据。

恢复未结束前，新的 START 会被拒绝，避免恢复线程和实时 writer 同时持有 dataset lock。

---

## 6. 重点问题、定位思路和最终解决方案

下面这张表是面试最有价值的部分。回答时用“现象 → 根因 → 修改 → 验证”的顺序。

| 问题 | 根因 | 最终方案 | 面试表达 |
| --- | --- | --- | --- |
| ZMQ/latest 可能跳帧 | latest 只保留最新 bundle，PUB/SUB 也可能存在传输缺口 | 每路 sequence + producer/publisher/transport gap 分类；相机侧可选 Raw Spool 在发送前保存 payload；主机 Raw 明确标记 `transport_observed` | “我没有把接收端副本冒充完整相机事实源，先区分可观察范围，再用 spool 覆盖传输前的帧。” |
| 相机停止和 spool 关闭竞态 | 读取线程可能还在 `wait_for_frames()` 或 append，服务端已经 close spool | `RealSenseReader.stop()` 先停止 pipeline，再 join capture thread；确认线程退出后再关闭 spool；`RawSpool.append()` 和 `close()` 共享生命周期锁 | “我把停止顺序定义成 pipeline stop → thread join → spool close，并用锁保护 append/close 的临界区。” |
| 采集线程被 MP4 编码拖慢 | JPEG 解码、OpenCV 编码和磁盘写入都可能阻塞采集 hot path | 默认 `raw_v1`，采集期只写 Raw；STOP 后由 `EpisodeSaveWorker` 异步物化 Parquet/MP4 | “把高延迟编码从实时采集路径移出，代价是需要更多 Raw 磁盘和后台任务管理。” |
| 状态和相机时间不能直接比较 | 不同源可能使用设备时钟、ROS 时间、server wall time 或接收时间 | 保存原始时间戳和 clock domain；采集 gate 使用本地 monotonic receive time；camera source 导入使用有证据的时钟映射并记录 uncertainty | “这是可量化的 best-effort 对齐，不是硬件级同步。” |
| Episode 崩溃后只剩半个文件 | 只有 MP4/Parquet 文件，没有唯一生命周期状态 | Raw manifest 作为持久状态源；chunk framing + CRC32；manifest、checksums、临时目录和原子 rename；启动扫描恢复 job | “重启时根据 durable state 恢复或隔离，而不是根据文件名猜状态。” |
| QC/持久化失败仍显示保存成功 | 结果函数吞异常，调用方无条件 `mark_saved()` | `_mark_raw_materialization_succeeded()` 返回质量状态并对持久化错误抛出；只有 `READY` 才 `mark_saved()` | “把保存成功从‘函数返回了’改成‘产物、证据、QC 都通过’。” |
| 丢目标点后输出 timestamp 不连续 | 使用原始 target frame index 生成 LeRobot timestamp | 用成功写入的 dense dataset frame index；同时保留原始 target timestamp 和 selection metadata | “训练时间轴描述输出行，source timestamp 描述真实来源，两者不混用。” |
| manifest 损坏被当成空状态 | JSON 解析失败时返回 `{}`，可能创建新状态覆盖旧证据 | 解析失败显式抛错；保留损坏文件并 quarantine | “对持久化元数据采用 fail-closed，而不是把损坏当成首次运行。” |
| extra metadata 覆盖核心字段 | 任意额外字段都能覆盖 episode、sequence、时间戳等身份信息 | 保留字段集合；核心 identity/timestamp/alignment 字段只能由 recorder/materializer 生成 | “元数据扩展可以加字段，但不能篡改数据身份和 provenance。” |
| manifest 每帧重写影响实时性 | 每帧 checksum/fsync/manifest rename 产生大量小 I/O | 相机服务默认按记录数或时间 checkpoint，chunk fsync 分组；close 时强制 flush 和 checksum | “实时路径用有界 durability budget，关闭边界再做完整耐久确认。” |
| 内存随 Episode 变大 | 直接在 Python 容器中保留全部 Raw record 或物化候选 | callback queue 有界；物化使用临时 SQLite 时间索引，逐帧查询候选 | “把大数据放到磁盘索引，内存只保留当前处理窗口。” |
| 转换产生半成品 | 直接往最终输出目录写，异常后目录看起来像成功 | converter 使用 staging directory，完成 hash/provenance 后原子替换；同一 source/config/version 可复用 | “转换器是从已发布源派生的幂等消费者，不修改 Raw。” |

### 6.1 其中最值得展开的 P0 问题：相机停止竞态

如果面试官问“你修过的最严重 bug 是什么”，可以这样回答：

> 最严重的是相机 capture thread 和 Raw Spool close 的停止竞态。原来的风险是主线程以为服务要结束，先关闭 spool，但读取线程还可能从 RealSense 取到最后一帧并调用 append，结果可能出现最后帧丢失、manifest 统计落后或 checksum 不完整。
>
> 我把停止流程改成三步：第一步设置 stop event 并停止 RealSense pipeline，让阻塞的 SDK 调用返回；第二步 join 读取线程并确认它已经退出；第三步才关闭 spool。与此同时，append、checkpoint 和 close 使用同一把生命周期锁，保证不会出现 append 与 close 同时修改同一份状态。最后在相机 acceptance test 中验证了 pipeline stop 发生在 thread join 之前。

这个回答体现的是并发时序、资源所有权和持久化边界，而不只是“加了一把锁”。

### 6.2 P1 问题：为什么不能无条件 `mark_saved()`

推荐回答：

> Episode 保存其实有多个阶段：Raw 已关闭、物化完成、artifact 可读、provenance 一致、QC 通过。任何一个阶段失败，都不能仅凭 worker 返回就显示成功。现在 materializer 会返回质量状态，collector 只在 `READY` 时调用 `mark_saved()`；`REVIEW`、`REJECT` 或持久化异常会进入明确的失败路径，并保留失败原因和 Raw 证据。

### 6.3 “丢帧以后怎样恢复”的准确回答

不要说“我能自动恢复所有丢帧”。准确说法是：

> 我能恢复已经可靠写入 Raw Spool、但后续物化或进程阶段失败的数据；也能通过 sequence 和 gap 分类发现丢帧。对于相机端已经丢失、且没有 camera-side Raw Spool 保存的帧，系统不能凭空恢复，只能把它标记为 transport/unattributed gap，并让 QC 阻止它进入 READY。完整采集场景需要在相机发送前开启 camera-side spool，并在 STOP 时按时间窗口绑定到 task Episode。

这不是缺点掩饰，而是数据系统必须遵守的事实边界。

---

## 7. 时间对齐应该怎样解释

### 7.1 三类时间

可以把时间分成三层：

| 时间 | 用途 | 是否可直接当成物理采样时间 |
| --- | --- | --- |
| device timestamp | RealSense 设备源时间 | 取决于设备 timestamp domain 和映射 |
| server wall/monotonic | 相机服务捕获/处理时的 host 证据 | 不是曝光时间 |
| receive monotonic | collector 主机收到消息的本地时间 | 适合做本机新鲜度和线程间 gate |

机器人状态也有类似差异：ROS message header、StepIt 字段 callback receive time、collector record time 并不等价。

### 7.2 当前的两阶段策略

采集阶段只保留原始事件，不做不可逆插值：

```text
记录 source timestamp + sequence + clock domain + receive/record time
```

物化阶段以参考 RGB 建立目标时间轴：

```text
RGB source timestamps → robot/camera bounded nearest neighbor → residual → strict/drop → 输出
```

采集时使用本地 monotonic receive time 做 state-camera 新鲜度和 skew gate，避免要求机器人和相机主机墙上时钟完全同步；
相机内部的 inter-camera skew 使用相机服务的共同 server-wall 语义检查。

### 7.3 不能过度承诺的地方

面试官如果问“你们是不是严格同步”，回答：

> 不是硬件触发意义上的严格同步。当前实现以参考 RGB 的真实时间戳为目标，把机器人状态和动作对齐到 RGB，并提供残差、来源和不确定度。网络传输、服务端编码和不同设备时钟映射不包含在本地 receive-time gate 的物理保证里。如果任务需要亚毫秒级 capture-time 对齐，还需要 PTP、硬件 trigger 或专用同步方案。

---

## 8. 为什么选择 Raw-first，而不是直接写 MP4

### 直接写 MP4 的问题

- 编码可能阻塞采集线程；
- JPEG 解码后再编码 MP4 有额外 CPU 和内存复制；
- 中途崩溃时，视频、Parquet 和状态记录可能不一致；
- 一旦时间对齐策略改变，已经写好的 MP4 很难重建；
- 发生 QC 失败时，缺少原始输入和选择证据。

### Raw-first 的收益

- 原始相机 payload 保持不变，可以重新解码、重新对齐、重新编码；
- 状态、图像、事件和 provenance 在同一个 Episode 中可追溯；
- MP4/Parquet/GR00T/OpenPI 都是从同一源派生；
- 物化失败只需要重试 job，不必重新操作机器人；
- 可以把 QC 的失败原因定位到 source、transport、selection 或 output 阶段。

### 代价和取舍

- 需要更多磁盘；
- 需要管理 Raw manifest、job 和恢复；
- STOP 后有异步物化延迟；
- 相机端完整采集需要部署 spool，并解决 source/task 的绑定。

可以用一句话总结：

> 我用磁盘空间和后台处理复杂度，换取了实时路径稳定性和数据可重建能力。

---

## 9. 借鉴 Robo-DM 时怎样说

当前仓库可以说借鉴了 Robo-DM 的“原始事实与训练派生物分离、便于回放/重建”的思路，
但不要说已经把 Robo-DM 的所有技术都搬进来了。

当前已落地的方向：

- Raw payload 和训练格式解耦；
- Episode 级 provenance、质量信息和可恢复状态；
- 停止后物化，支持同一源派生多种下游格式；
- 有界内存和磁盘上的中间索引。

当前没有作为第一阶段实现的内容：

- Robo-DM 的 EBML 容器；
- 训练侧 mmap 解码缓存；
- Robo-DM 专用的训练 DataLoader；
- 经过 benchmark 选择的新视频 codec。

被问到“为什么没直接引入 EBML/mmap”时，可以答：

> 当时优先解决的是采集事实的完整性、时间语义和失败恢复。如果协议、容器和训练加载器同时迁移，问题很难定位，也会增加部署风险。当前先用依赖较少的 Raw chunk、manifest、CRC 和 SQLite 物化索引把边界建立起来；后续再根据数据规模和训练 I/O benchmark 决定是否引入 EBML 或 mmap，而不是为了追求技术名词直接替换现有链路。

---

## 10. 面试时的 2 分钟完整回答

下面这段可以按自己的语气练习：

> 我的项目是给 Unitree G1 遥操作采集训练数据。输入端一边是 StepIt/ROS2 的关节、IMU、目标和策略字段，另一边是两路 RealSense RGB。我先通过 `robo_state_node` 把上游状态统一成带维度校验和时间戳的 `RoboStateSample`；相机端由独立 server 读取多路 RealSense，给每帧保留 sequence、设备/服务端时间和 JPEG payload，再通过 ZMQ 发到采集主机。
>
> 采集器不直接把所有事情塞在一个 callback 里。相机 client 先把不同协议标准化成统一 envelope，`CameraFrameCache` 用有界队列接收，同时维护最新完整 bundle 和 producer、publisher、transport gap 统计。Episode 开始后，我把相机原始包、机器人状态和命令事件写进 Raw Episode；默认的 `raw_v1` 模式只做有界原始落盘，不在实时路径里做 MP4 编码。
>
> STOP 时先把已经收到的相机 callback 排空，再关闭 Raw，创建持久化的 materialization job。物化阶段用磁盘上的 SQLite 时间索引，以第一路 RGB 的真实时间戳建立目标时间轴，把机器人状态和其他相机对齐到每个 RGB 帧，对动作使用 before-or-equal 的 zero-order hold；如果必需数据超出 residual 窗口，就丢弃该 RGB 目标点并计为 selection gap，不复制旧图像。然后生成 LeRobot Parquet/MP4，写入 source sequence、residual 和 selection policy，最后经过质量门，只有 READY 才能算保存成功。
>
> 我重点解决过几个可靠性问题：相机停止时先停 pipeline、再确认读取线程退出、最后关 spool，避免最后帧和 manifest 竞态；持久化或 QC 失败不能再被标记成 saved；物化丢点后 timestamp 用成功输出的连续行号而不是原始目标序号；损坏的 manifest 采用 fail-closed 并 quarantine；相机 spool 则用 checkpoint 和分组 fsync 降低每帧 I/O。最终 Raw 是事实源，Parquet、MP4、GR00T 和 OpenPI 都是可验证的派生物。

---

## 11. 常见追问和建议答案

### Q1：为什么相机和机器人不直接放在一个进程里？

> 相机端通常靠近硬件，机器人状态和 collector 可能在另一台主机。拆开后硬件驱动、网络传输和数据编排的故障边界更清晰，也便于相机服务独立重启。代价是跨主机时间和传输丢帧必须显式建模，所以我保留了 session、sequence、timestamp provenance 和 gap 统计。

### Q2：为什么用 ZMQ PUB/SUB？

> 现有系统需要一对多的实时相机分发和低延迟预览，PUB/SUB 接入简单、吞吐高、和 ROS2 解耦。它不是持久化队列，所以我没有把它当成可靠存储：录制模式使用有界策略并记录 transport gap，完整采集则在相机端先写 Raw Spool。

### Q3：为什么不把 ZMQ 的 HWM 调得无限大？

> 无限缓存只是把丢帧问题变成延迟和内存失控。采集系统更重要的是有明确资源上限：队列满时暴露失败，保留 gap 证据，并让 QC 阻止坏 Episode 进入训练。

### Q4：你怎么知道丢的是哪一层？

> 每路相机有流内 sequence，服务端计算 producer gap 和 publisher gap，接收端结合 packet sequence 计算 transport gap，物化阶段再统计 selection gap。四类缺口分别对应源端、发布端、网络/接收端和对齐选择阶段，不能只看一个总计数。

### Q5：如果网络已经丢了帧，你的系统能恢复吗？

> 只从接收端 Raw 不能恢复传输前已经丢掉的帧。接收端能做的是识别和分类。要覆盖这部分，必须在相机 server 的发送前路径开启 camera-side spool，并在 STOP 时把相机 source session 按时间窗口绑定到 task Episode。即便有 spool，也会对 snapshot 稳定性、high watermark 和 clock mapping 做 QC。

### Q6：为什么不用上一帧补齐？

> 对训练数据而言，静默复制会把“缺失观测”伪装成“真实观测”，而且会破坏质量统计。默认 strict 策略宁愿丢弃这个输出目标点并报告 selection gap；诊断场景可以设计 sparse 输出，但不能把 sparse 结果混进标准训练索引。

### Q7：如何保证重启后不会重复生成或覆盖数据？

> job 使用 source manifest hash、conversion config hash 和 output schema version 作为幂等依据；输出先写 staging 目录，验证后原子提交。启动时扫描 manifest 和 durable job，而不是依赖内存队列或进度日志。已有产物会先验证 provenance 和 artifact evidence，匹配才复用。

### Q8：为什么一个 MP4 存在了，还不能说明保存成功？

> 因为 MP4 只是一个派生 artifact，可能缺帧、和 Parquet 行数不一致、没有 provenance，或者 QC 还没完成。系统将 `MATERIALIZED`、`QC` 和 `READY` 分开，只有 manifest 状态、质量报告和产物证据都一致才允许进入训练索引。

### Q9：你们的时间对齐是硬同步吗？

> 不是。当前是保留来源时间并用本地 monotonic receive time 做采集 gate，停止后以固定 FPS 做有界最近邻和 residual 统计。如果是物理级同步需求，需要 PTP、硬件 trigger 或相机/机器人统一时钟；当前实现不会把网络接收时间宣传成曝光时间。

### Q10：为什么物化索引使用 SQLite，而不是直接读成 list？

> 主要是控制内存上界和支持长 Episode。Raw record 可能包含大图像 payload，全部加载到 Python list 会让 RSS 随时长线性增长。SQLite 临时索引只在当前输出时间点查询 state 和各路 camera 候选，最终仍然逐帧写入 writer。

### Q11：如何验证状态维度和关节顺序？

> 在状态归一化阶段检查长度、有限值、重复 key 和三类关节字段的集合一致性；在 writer 阶段继续检查 joint names 和 dataset canonical ordering。这样不会把错位的 29 维向量当成合法数据写入训练集。

### Q12：你做了哪些测试？

> 测试覆盖了 camera v3/v4 解析、时间戳 provenance、队列溢出、stop 顺序、spool checksum 和 snapshot、Raw Episode 生命周期、重启恢复、job 幂等、物化对齐、dense timestamp、QC gate、LeRobot writer 以及 GR00T/OpenPI 转换。当前仓库最近一次验证中，camera 测试为 29 passed，collector 测试为 258 passed，另有 48 个目标测试通过，并通过了 Python compileall。ROS2 真机和长时间 soak 属于部署验收，需要在有对应硬件和 ROS 环境时提供实测数据。

---

## 12. 个人贡献怎么讲，避免像在背模块名

面试官问“这个项目你具体做了什么”，不要只回答“我写了 collector_node.py”。可以按下面四个层次说：

### 12.1 我负责的主线

> 我负责把相机采集、机器人状态、Episode 控制和训练数据输出串成一条可运行的数据链路，重点处理多源时间、帧缺口、异常结束和后处理恢复问题。

### 12.2 我做的关键设计

> 我把采集和物化分成两个阶段：采集期保存原始事实，停止后再做对齐和格式转换；用 manifest 管理持久化生命周期，用 sequence 和 timestamp provenance 管理数据可信度，用 QC 状态决定是否允许进入训练。

### 12.3 我解决的工程问题

> 我解决了相机线程停止与 spool close 的竞态、物化/QC 失败仍被标记保存成功、丢点后输出 timestamp 不连续、损坏 manifest 被误判为空状态、metadata 覆盖核心身份字段，以及逐帧 fsync 造成的 I/O 开销问题。

### 12.4 我做出的取舍

> 我保留了 v3 和 legacy 兼容路径，避免一次性切换协议和输出；把 raw-first 设为主路径，把完整相机事实源绑定设为显式配置；暂时没有引入 EBML、mmap 和训练侧 DataLoader，先用更小的依赖解决采集可靠性和数据可审计性。

如果某一部分不是你独立完成的，就把“我负责”改成“这个仓库采用”，把自己的实际贡献讲具体，不要把所有文件都包装成个人成果。

---

## 13. 代码定位清单

面试前建议至少打开下面这些文件，知道每个文件解决什么问题：

| 主题 | 文件 |
| --- | --- |
| 项目运行方式和限制 | [`README.md`](../README.md) |
| 总体优化设计和验收标准 | [`raw-episode-capture-optimization-spec.md`](raw-episode-capture-optimization-spec.md) |
| RealSense 读取、时间戳、sequence、ZMQ 发布 | [`server_realsense.py`](../src/camera/robo_collector_camera/server_realsense.py) |
| v3/v4 packet 标准化 | [`client.py`](../src/camera/robo_collector_camera/client.py) |
| 相机 Raw Spool、CRC、checkpoint、snapshot | [`raw_spool.py`](../src/camera/robo_collector_camera/raw_spool.py) |
| bounded cache、gap/skew、callback queue | [`camera_cache.py`](../src/robo_collector/robo_collector/camera_cache.py) |
| Episode 命令和主链路 | [`collector_node.py`](../src/robo_collector/robo_collector/collector_node.py) |
| runtime state machine 和 command ledger | [`collector_state.py`](../src/robo_collector/robo_collector/collector_state.py) |
| Raw Episode manifest、chunk、恢复和 job | [`raw_episode.py`](../src/robo_collector/robo_collector/raw_episode.py) |
| RGB 参考时间轴对齐和物化 | [`raw_materializer.py`](../src/robo_collector/robo_collector/raw_materializer.py) |
| LeRobot Parquet/MP4 writer | [`lerobot_dataset.py`](../src/robo_collector/robo_collector/lerobot_dataset.py) |
| QC 和 READY/REVIEW/REJECT | [`episode_quality.py`](../src/robo_collector/robo_collector/episode_quality.py) |
| GR00T 转换 | [`gr00t_converter.py`](../src/robo_collector/robo_collector/gr00t_converter.py) |
| OpenPI pi0.5 转换 | [`pi05_converter.py`](../src/robo_collector/robo_collector/pi05_converter.py) |
| 状态归一化和维度校验 | [`state_builder.py`](../src/robo_state/robo_state/state_builder.py) |
| 手势触发和命令生成 | [`gesture_trigger_node.py`](../src/robo_collector/robo_collector/gesture_trigger_node.py) |

---

## 14. 面试时不要说过头的四句话

把下面的说法换成更准确的版本：

| 不建议说 | 建议说 |
| --- | --- |
| “系统完全不丢帧。” | “系统能检测、分类和审计丢帧；开启 camera-side spool 后可以覆盖发送前的源数据。” |
| “相机和机器人严格同步。” | “系统保留多源时间并做带 residual 的 best-effort 对齐；不是硬件级同步。” |
| “所有异常都能自动恢复。” | “已落盘且有证据的 Raw/物化任务可以恢复；没有保存的帧无法凭空恢复。” |
| “MP4 生成了就保存成功。” | “只有 artifact、provenance 和 QC 都通过并进入 READY，才算训练侧发布成功。” |

这四个边界说清楚，反而会让面试官觉得你理解数据系统，而不是只会堆功能。

---

## 15. 当前验证证据和剩余边界

### 已有的代码级验证

当前仓库最近一次验证包括：

- camera 测试：`29 passed`，包含 6 个 subtests；
- `robo_collector` 测试：`258 passed`，包含 10 个 subtests；
- 重点目标测试：`48 passed`，包含 3 个 subtests；
- Python `compileall` 通过；
- `git diff --check` 通过。

### 仍需用部署实测回答的问题

不要把单元测试说成真机验收。真实部署还应测量：

- 双路 640×480@30 FPS 的长时间采集；
- 相机断连/重连；
- collector 进程在 OPEN、STOP、物化、QC、提交各阶段退出；
- 10 个 60 秒 Episode 和至少一次长时间 soak；
- CPU、RSS、磁盘增长、队列深度和 deadline miss；
- MP4 编码时间、文件大小、解码兼容性；
- source spool 与 host receiver 之间的窗口绑定和 clock mapping。

如果面试官继续追问量化收益，而你没有实测数字，就直接说：

> 当前仓库已经完成了结构化实现和自动化测试，但 CPU、RSS、丢帧率和长时间 soak 的最终数字需要在目标 G1、相机和 ROS2 部署环境中采集，不能用本地单元测试结果替代。

---

## 15.1 Phase 6/7 MCAP 工具和 rollout 说法

可以这样描述运维链路：`info` 查看 MCAP 摘要，`doctor --group` 校验指定
canonical MCAP 的结构和 checksum；Episode 的 manifest、source fence 和质量状态
由 ingestion/QC 发布门统一校验。`recover` 从可恢复前缀生成新 artifact，
`replay` 检查事件顺序，`migration` 比较 Raw-v1/MCAP，`benchmark` 记录本地
诊断指标。所有命令对路径错误、checksum 损坏、非 READY 或不完整 source
证据 fail-closed，不覆盖 sealed `*.mcap`。

shadow rollout 的验收是逐 Episode 对比 accepted frontier、sequence/gap、
时间戳、选帧数、manifest/config hash 和最终质量状态；`raw_v1` 仍是默认，
`dual_write` 只用于迁移证据，`mcap_first` 是 release candidate。单元测试、
本地 benchmark 或短时 harness 不能声称完成真实硬件 soak；面试中应明确说明
仍需在目标 G1、相机和 ROS2 环境采集 process-kill、reconnect、10×60 秒及
60 分钟 soak 的部署证据。

## 16. 最后记忆版

面试前只记住这五句话也可以：

1. **先统一状态和相机的数据契约，再做采集。**
2. **采集期保留 Raw，停止后才做对齐和 MP4/Parquet 物化。**
3. **所有数据都保留 sequence、时间来源和 gap 分类，避免“看起来完整”。**
4. **manifest 是唯一持久状态，只有 QC 通过的 READY 才能发布。**
5. **能恢复的是已经落盘的数据；没有被保存的帧只能检测丢失，不能凭空恢复。**
