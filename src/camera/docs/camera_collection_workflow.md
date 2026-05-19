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
packet = camera.read(timeout_ms=10)
```

返回数据中当前约定：

- `images["head"]`：D405 平视 RGB JPEG 解码后的 `uint8 [H, W, 3]`。
- `images["ego_view"]`：D435i 俯视 RGB JPEG 解码后的 `uint8 [H, W, 3]`。
- `timestamps[...]`：各路图像时间戳，单位为秒。
- `metadata["cameras"]`：每路 RealSense 的序列号、设备信息和采集配置。

## 与机器人状态采集的对齐点

后续新增状态采集时，相机数据和 ROS2 数据需要统一封装：

- 相机侧使用 packet 内的 `timestamps`。
- ROS2 `JointState`、`Imu` 使用 header timestamp。
- `/stepit/field/*` 这类 `Float32MultiArray` 没有 header，应在 callback 中记录接收时间。
- 每条记录保存采集时间、来源 topic/stream、schema 版本和原始维度。
