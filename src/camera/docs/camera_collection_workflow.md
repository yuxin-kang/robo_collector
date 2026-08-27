# 相机采集流程

本文档记录 `robo_collector/src/camera` 当前相机链路。完整遥操作数据流见 `../../../docs/teleop_data_collection_workflow.md`。

## 目录

```text
src/camera/
  scripts/
    setup_camera_env.sh
    run_realsense_server.sh
    run_camera_viewer.sh
    test_camera_client.sh
  robo_collector_camera/
    client.py
    raw_spool.py
    server_realsense.py
    viewer.py
```

## Robot Side: RealSense 发布端

在连接 RealSense 的机器人端运行：

```bash
cd /home/kyx/robot/vla/robo_collector/src/camera
bash scripts/setup_camera_env.sh --server
source .venv_camera/bin/activate
bash scripts/run_realsense_server.sh --list-devices
```

双 RealSense RGB 发布：

```bash
bash scripts/run_realsense_server.sh \
  --camera head:<D405_SERIAL> \
  --camera ego_view:<D435I_SERIAL> \
  --port 5555 \
  --width 640 --height 480 --fps 30 \
  --jpeg-quality 80 \
  --no-depth
```

完整采集模式可在相机侧先写入有界 Raw spool，再更新 latest 并发送 ZMQ：

```bash
bash scripts/run_realsense_server.sh \
  --camera head:<D405_SERIAL> \
  --camera ego_view:<D435I_SERIAL> \
  --raw-spool-dir /data/robo_collector/camera_spools \
  --packet-schema v3
```

`v3` 是默认发送协议；`v4` 仅在客户端完成回归后通过
`--packet-schema v4` 显式启用。相机侧 spool 是完整采集源候选，主机从
ZMQ 保存的 Raw Episode 仍必须标记为 `source_scope=transport_observed`，
不能补回传输前被 latest、队列或网络丢掉的帧。

兼容单相机旧路径：

```bash
bash scripts/run_realsense_server.sh --serial <SERIAL> --port 5555 --no-depth
```

## Host Side: 相机客户端

在采集主机运行：

```bash
cd /home/kyx/robot/vla/robo_collector/src/camera
bash scripts/setup_camera_env.sh --client
source .venv_camera/bin/activate
bash scripts/test_camera_client.sh --host 192.168.123.164 --port 5555
```

打开 viewer：

```bash
bash scripts/run_camera_viewer.sh --host 192.168.123.164 --port 5555
```

## Python 调用入口

```python
from robo_collector_camera.client import CameraClient

camera = CameraClient("192.168.123.164", 5555)
packet = camera.read_envelope(timeout_ms=10)
```

协议先经过 `decode_envelope(v3|v4)` 标准化为同一结构；需要图像时再解码。
录制模式关闭 `CONFLATE` 并使用有界接收队列，预览模式才使用
`CONFLATE` 追求低延迟。标准化 frame 保留原始 payload、sequence、设备/服务端/
接收端时间戳、`clock_domain` 和 `timestamp_quality`。

解码后的调用入口仍兼容旧代码：

```python
packet = camera.read(timeout_ms=10)
```

返回数据中当前约定：

- `images["head"]`：D405 平视 RGB JPEG 解码后的 `uint8 [H, W, 3]`。
- `images["ego_view"]`：D435i 俯视 RGB JPEG 解码后的 `uint8 [H, W, 3]`。
- `timestamps[...]`：v3 各路服务端时间戳，单位为秒；缺少设备时间时标记为
  `host_after_capture`。
- `metadata["cameras"]`：每路 RealSense 的序列号、设备信息和采集配置。
- envelope 还包含每路 `sequence`、原始 JPEG payload、时间戳来源和 session。

## 与机器人状态采集的对齐点

相机数据和 ROS2 数据在 Raw Episode 中统一封装，但不跨 clock domain 直接比较
绝对时间：

- 相机侧使用 packet 内的 `timestamps`。
- ROS2 `JointState`、`Imu` 使用 header timestamp。
- `/stepit/field/*` 这类 `Float32MultiArray` 没有 header，应在 callback 中记录接收时间。
- 每条记录保存采集时间、来源 topic/stream、schema 版本、sequence 和原始维度。
- `producer_gap` 表示相机源 sequence 不连续，`publisher_gap` 表示相机服务在
  latest 聚合时跳过的 sequence，`transport_gap` 表示已发送但主机未收到，
  `unattributed_gap` 表示当前证据无法归因；物化时无法在窗口内选帧的样本另计为
  `selection_gap`。
- 采集阶段只保存原始事件；固定时间轴、选帧和 MP4/Parquet 生成在 Episode 关闭后执行。

相机侧 spool 在关闭时写入 chunk SHA-256。若采集主机能访问 spool 目录并配置
`--raw-source-scope camera_capture --camera-raw-spool-root <mounted-path>`，collector
会按 task 的 START/STOP server-wall-time 窗口导入 source-side 记录，并把 source
session、manifest hash、窗口和导入帧数写入任务 Raw manifest；否则只能使用并标记为
`transport_observed` 的接收端 shadow recording。

相机 spool 的 `RSP1` framed-msgpack chunk 是 source-side 格式；collector 导入时
会把经校验的前缀适配成主机 Raw Episode 的 `RER1` JSON/base64 记录。该适配不会
覆盖源端 sequence、clock domain 或原始时间戳，任务 manifest 还会记录每个 source
session 的 hash 和时钟映射。
