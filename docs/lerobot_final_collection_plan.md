# Robo Collector 最终 LeRobot 采集计划

本文档固定最终 LeRobot 数据采集层的目标、接口、输出格式、启动方式和验证计划。该层只负责订阅已经规整验证过的机器人状态样本，读取 RealSense RGB 图像，并按 episode 写入 LeRobot v2.1 风格数据集。

参考格式和流程：[GR00T Data Collection](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html)。

## 目标

- 订阅已验证的 `/robo_state/sample`，不再直接订阅 `/stepit/*`。
- 从 RealSense ZMQ camera server 读取 `ego_view` RGB 帧。
- 以 50Hz 对齐最新机器人状态和 RGB 图像，写入 LeRobot v2.1 风格数据集。
- 采集程序启动后默认空等，不创建 episode、不写数据。
- 只有收到 ROS2 `START` 指令才开始记录 episode。
- 收到 `STOP` 后保存当前 episode 并回到等待下一个 `START` 的循环。
- 录制中收到 `DISCARD` 时丢弃当前 episode 并回到等待状态。
- 只保存本项目真实可采的数据字段，不伪造 SONIC/SMPL/planner 字段。

## 新增 ROS2 Package

### `src/robo_collector_msgs`

定义采集控制消息：

```text
std_msgs/Header header

uint8 START=1
uint8 STOP=2
uint8 DISCARD=3

uint8 command
string task_prompt
string episode_id
```

### `src/robo_collector`

最终 LeRobot collector 节点，使用 `ament_cmake_python` 构建，避免 `setup.py` 构建问题。

主要入口：

```bash
ros2 run robo_collector lerobot_collector_node
```

## ROS2 接口

### 输入 Topic

| Topic | Type | 说明 |
| --- | --- | --- |
| `/robo_state/sample` | `robo_state_msgs/msg/RoboStateSample` | 已验证、规整后的机器人状态样本 |
| `/robo_collector/record_command` | `robo_collector_msgs/msg/RecordCommand` | episode 录制控制命令 |

### 输出 Topic

| Topic | Type | 说明 |
| --- | --- | --- |
| `/robo_collector/status` | `diagnostic_msgs/msg/DiagnosticStatus` | collector 状态、当前 episode、帧数和告警 |

## Collector 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `robo_state_topic` | `/robo_state/sample` | 统一机器人状态输入 |
| `record_command_topic` | `/robo_collector/record_command` | 录制控制命令 topic |
| `status_topic` | `/robo_collector/status` | collector 状态 topic |
| `camera_host` | `192.168.123.164` | camera server 地址 |
| `camera_port` | `5555` | camera server 端口 |
| `camera_stream` | `ego_view` | 采集的 RGB stream 名称 |
| `dataset_name` | 自动时间戳 | 输出数据集名 |
| `root_output_dir` | `outputs` | 数据集根输出目录 |
| `fps` | `50` | episode 写入频率 |
| `max_state_age_sec` | `0.2` | 状态样本最大允许 age |
| `max_camera_age_sec` | `0.2` | 图像帧最大允许 age |

## 状态机

Collector 使用四个状态：

| 状态 | 行为 |
| --- | --- |
| `IDLE` | 缓存最新 `/robo_state/sample` 和相机帧，不创建数据集，不写 episode |
| `RECORDING` | 收到 `START(task_prompt=...)` 后打开 episode，每 20ms 写入一帧 |
| `NEED_TO_SAVE` | 收到 `STOP` 后保存 episode，关闭视频 writer，更新 metadata，再回到 `IDLE` |
| `DISCARD` | 录制中收到 `DISCARD` 后删除当前未保存 episode，再回到 `IDLE` |

状态转移：

```text
IDLE -- START(task_prompt 非空) --> RECORDING
RECORDING -- STOP --> NEED_TO_SAVE -- save ok --> IDLE
RECORDING -- DISCARD --> DISCARD -- discard ok --> IDLE
NEED_TO_SAVE -- DISCARD --> DISCARD -- discard ok --> IDLE
```

约束：

- `START` 必须携带非空 `task_prompt`，否则拒绝开始并发布 `WARN`。
- `IDLE` 下收到 `STOP` 或 `DISCARD` 不产生数据。
- `RECORDING` 下缺少新鲜 state 或 camera frame 时跳过该 tick，并在 status 中发布 `WARN`。
- 保存失败时保留未完成 episode 的上下文，发布 `ERROR` 供人工处理。

## LeRobot 输出结构

每个数据集目录形如：

```text
outputs/<dataset_name>/
  data/
    train-000000.parquet
    train-000001.parquet
  videos/
    observation.images.ego_view/
      episode_000000.mp4
      episode_000001.mp4
  meta/
    info.json
    modality.json
    episodes.jsonl
    tasks.jsonl
```

### 数据字段

| 字段 | 维度 | 来源 |
| --- | --- | --- |
| `observation.images.ego_view` | decoded `[H, W, 3]` | MP4 video feature，parquet 中按 `{path, timestamp}` 引用 camera server `ego_view` RGB |
| `observation.state.joint_position` | `29` | `RoboStateSample.robot_state.joint_pos` |
| `observation.state.joint_velocity` | `29` | `RoboStateSample.robot_state.joint_vel` |
| `observation.state.joint_torque` | `29` | `RoboStateSample.robot_state.joint_torque` |
| `observation.state.imu_angular_velocity` | `3` | `RoboStateSample.imu.angular_velocity` |
| `observation.state.imu_linear_acceleration` | `3` | `RoboStateSample.imu.linear_acceleration` |
| `observation.state.projected_gravity_or_quat` | `4` | 当前保存 IMU orientation quaternion |
| `action.joint_position` | `29` | `RoboStateSample.target_joint_pos` |
| `action.policy_action` | `29` | `RoboStateSample.action` |
| `annotation.human.action.task_description` | string | `RecordCommand.task_prompt` |

说明：

- v1 只保存 RGB `ego_view`，不保存 depth。
- `projected_gravity_or_quat` 当前使用四元数直接落盘；如后续确认 projected gravity 更稳定，可在保持字段语义清晰的前提下迁移。
- 每帧保存 `timestamp`、`frame_index`、`episode_index`、`index`、`task_index`，用于 LeRobot 数据集索引。

## 依赖环境

新增脚本：

```bash
bash scripts/setup_data_collection_env.sh
```

行为：

- 创建 `.venv_data_collection --system-site-packages`，保证 `rclpy` 等 ROS Python 包仍可 import。
- 默认安装 collector 运行所需的轻量依赖：`opencv-python`、`numpy`、`pyzmq`、`msgpack`、`pyarrow`。
- editable 安装 `src/camera/`，供 collector 复用 `CameraClient`。

如需使用官方 LeRobot loader、转换或校验工具，再显式安装大依赖：

```bash
bash scripts/setup_data_collection_env.sh --with-lerobot
```

`--with-lerobot` 会额外安装 `lerobot` 和 `av`，并可能下载 PyTorch/CUDA 大 wheel；现场采集本身不依赖这一步。

## 一键启动

新增脚本：

```bash
bash scripts/launch_data_collection.sh \
  --camera-host 192.168.123.164 \
  --camera-port 5555 \
  --root-output-dir outputs \
  --fps 50
```

tmux panes：

| Pane | 命令 |
| --- | --- |
| 0 | `ros2 run robo_state robo_state_node` |
| 1 | `ros2 run robo_collector lerobot_collector_node ...` |
| 2 | `src/camera/scripts/run_camera_viewer.sh --host ... --port ...` |

该脚本不启动 StepIt、XRT、全身控制或 camera server；这些进程仍由人工按现场流程启动。

## 手动控制命令

开始一段 episode：

```bash
ros2 topic pub --once /robo_collector/record_command \
  robo_collector_msgs/msg/RecordCommand \
  "{command: 1, task_prompt: 'pick up the red cup', episode_id: 'manual_001'}"
```

保存当前 episode：

```bash
ros2 topic pub --once /robo_collector/record_command \
  robo_collector_msgs/msg/RecordCommand \
  "{command: 2}"
```

丢弃当前 episode：

```bash
ros2 topic pub --once /robo_collector/record_command \
  robo_collector_msgs/msg/RecordCommand \
  "{command: 3}"
```

查看 collector 状态：

```bash
ros2 topic echo /robo_collector/status
ros2 topic hz /robo_collector/status
```

## 测试计划

### 单元测试

- `RecordCommand` 状态机：`IDLE -> RECORDING -> NEED_TO_SAVE -> IDLE`。
- 未收到 `START` 时不创建 dataset、不写 episode。
- `START` 注入 `task_prompt`，每帧 task annotation 一致。
- `STOP` 保存 episode 并写出 metadata。
- `DISCARD` 不保留 episode 文件和 metadata。
- 缺 state 或缺 camera 时跳过帧，并在 status 中报告 `WARN`。

### 集成测试

- fake `/robo_state/sample` publisher + fake camera client 跑 2 个 episode。
- 验证 LeRobot 目录结构、parquet、mp4、`meta/info.json`、`meta/modality.json`、`episodes.jsonl`、`tasks.jsonl` 存在。
- `ros2 topic pub ... START` 后开始写帧。
- `ros2 topic pub ... STOP` 后保存并等待下一次 `START`。
- `ros2 topic hz /robo_collector/status` 和日志能看到 `IDLE/RECORDING/NEED_TO_SAVE`。

### 现场验证

- episode 帧率约 50Hz。
- `observation.images.ego_view` MP4 可被 LeRobot/pyav 读取。
- `action.joint_position` 为 29 维。
- `annotation.human.action.task_description` 写入 `tasks.jsonl` 并通过 frame `task_index` 关联。
- 程序长时间不退出时可连续采多个 episode。

## 假设

- `/robo_state/sample` 是唯一机器人状态源。
- 相机 server 由人工手动启动，collector 只连接 `camera_host:camera_port`。
- 每个 episode 的语言信息由 `START` 命令携带。
- 空 `task_prompt` 表示无效 episode，collector 拒绝开始。
- 输出格式目标是 SONIC 风格的 LeRobot v2.1 数据集结构，但字段集合适配本项目真实可采数据。
