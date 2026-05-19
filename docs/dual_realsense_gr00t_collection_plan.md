# 双 RealSense RGB 采集与 GR00T 风格保存计划

本文档记录双 RealSense RGB 采集改造计划。目标是让机器人端发布一个 composed camera server payload，collector 侧按 GR00T/LeRobot v2.1 风格保存两路 RGB video feature。

参考依据：[GR00T Data Collection](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html)。其中 camera server 在机器人端运行，通过 ZMQ 发布 JPEG 图像；多相机保存为独立 `videos/observation.images.<camera>/` 目录；ZMQ payload 结构包含 `timestamps` 和 `images`。

## Summary

- 按 GR00T/SONIC 做法改成机器人端一个 composed camera server：同一条 ZMQ msgpack payload 同时发布两路 RGB 图像。
- 数据集保存为 LeRobot v2.1/GR00T 风格：每个相机一个 video feature、一个独立 MP4 目录，parquet 每帧引用对应视频 timestamp。
- 字段命名：
  - RealSense D405，平视相机：`observation.images.head`
  - RealSense D435i，俯视相机：`observation.images.ego_view`

## Key Changes

### 相机 Server

- 将 `server_realsense.py` 从单相机固定 `ego_view` 改成多相机 composed server。
- 新增 CLI：
  - `--list-devices`
  - `--camera head:<D405_SERIAL>`
  - `--camera ego_view:<D435I_SERIAL>`
- ZMQ payload：

```python
{
    "schema": "robo_collector_camera.v2",
    "timestamps": {"head": <float seconds>, "ego_view": <float seconds>},
    "images": {"head": <jpeg bytes>, "ego_view": <jpeg bytes>},
    "metadata": {"cameras": {...}},
}
```

- 每个 RealSense 一个 pipeline/thread。
- Server 每个 publish cycle 只在两路都有新 RGB 帧时发布一条合成消息。

### Collector 与 Writer

- `CameraFrameCache` 改为多流缓存，默认要求 `head,ego_view` 两路都存在。
- 新增 ROS 参数 `camera_streams:=head,ego_view`。
- 保留旧 `camera_stream` 单相机兼容。
- Writer 支持多 `camera_keys`，每个 episode 写两个视频：

```text
videos/observation.images.head/episode_000000.mp4
videos/observation.images.ego_view/episode_000000.mp4
```

- Parquet 每帧包含：

```python
"observation.images.head": {
    "path": "videos/observation.images.head/episode_000000.mp4",
    "timestamp": frame_index / fps,
}
"observation.images.ego_view": {
    "path": "videos/observation.images.ego_view/episode_000000.mp4",
    "timestamp": frame_index / fps,
}
```

- `meta/info.json` 和 `meta/modality.json` 同时声明 `head` 与 `ego_view` 两个 RGB video features。

### 启动与文档

相机 server 示例：

```bash
src/camera/scripts/run_realsense_server.sh \
  --camera head:<D405_SERIAL> \
  --camera ego_view:<D435I_SERIAL> \
  --port 5555 \
  --width 640 --height 480 --fps 30 \
  --jpeg-quality 80 \
  --no-depth
```

采集 launcher 默认：

```bash
bash scripts/launch_data_collection.sh \
  --camera-host 192.168.123.164 \
  --camera-port 5555 \
  --camera-streams head,ego_view \
  --root-output-dir outputs \
  --fps 50
```

Viewer 自动 tiled 显示所有 RGB streams，不再写死只显示 `ego_view`。

## Test Plan

### Unit Tests

- Camera client 能解码同一 payload 中的 `head` 和 `ego_view`。
- Writer 保存两个 MP4 path、两个 parquet video reference、两个 `info.json.features`、两个 `modality.json.observation.images`。
- Metadata 写入失败时两个 partial video 和 parquet 都能被 DISCARD 清理。
- Collector 在缺 `head` 或缺 `ego_view` 时不写帧并发布 WARN。
- 单相机旧路径 `camera_stream:=ego_view` 仍可用。

### Integration Tests

- Fake camera server 发布 `head` + `ego_view` 两路 RGB，fake `/robo_state/sample` 发布状态，跑两个 episode。
- 验证输出目录、parquet 两列 video reference、两个 MP4 可由 PyAV 打开。
- 确认 `action.joint_position` 仍是 29 维，task prompt 写入不变。

## Assumptions

- 只采集 RGB，不保存 depth。
- D405 固定映射为 `head`，D435i 固定映射为 `ego_view`。
- 相机发布方式采用 GR00T 一致的 ZMQ msgpack，不新增 ROS2 `sensor_msgs/Image` topic。
- Collector 仍按 50Hz 保存；RealSense server 默认 30Hz，collector 复用最新完整相机 bundle。
