# 遥操作数据采集系统流程

本文档记录当前 G1 遥操作采集系统的运行入口、数据流和后续机器人状态采集接口。目标是在写状态采集代码前，把已验证的启动命令和 ROS2 话题固定下来，便于后续 review。

## 机器与目录

- 远端开发机：`kyx@10.12.120.125`
- 远端项目根目录：`/home/kyx/robot/vla`
- 本仓库目录：`/home/kyx/robot/vla/robo_collector`
- StepIt 工作区：`/home/kyx/robot/vla/stepit_ws`
- XRT retargeting 工作区：`/home/kyx/robot/vla/xrt_retargeting`
- XRoboToolkit PC service：`/home/kyx/robot/vla/XRoboToolkit-PC-Service`

## 总体数据流

```text
Pico / XRoboToolkit
  -> /xrobotoolkit/body_state, /xrobotoolkit/joy, ...
  -> xrt_retargeting
  -> /retargeting/retargeted
  -> stepit_ros2
  -> robot command + /stepit/* ROS2 topics

RealSense camera
  -> robo_collector/src/camera ZMQ publisher
  -> host-side collector/client
```

当前需要新增的采集代码应从 StepIt 的 ROS2 发布端读取机器人状态和目标关节角。相机链路已经在 `src/camera/README.md` 中记录，后续采集节点需要把相机帧和 ROS2 状态按时间戳对齐。

## 启动顺序

推荐按下面顺序启动。每个长运行命令放在独立终端里。

### 1. 登录远端机器

```bash
ssh kyx@10.12.120.125
cd /home/kyx/robot/vla
```

### 2. 启动 XRT retargeting

```bash
cd /home/kyx/robot/vla/xrt_retargeting
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch xrt_retargeting xrt_retargeting.launch.py \
  actual_human_height:=1.7 \
  init_max_iterations:=10 \
  max_iterations:=1 \
  downsample_factor:=2 \
  timestep:=0.02 \
  visualize:=true
```

作用：

- 启动 `xrobotoolkit_state_publisher`，把 XRoboToolkit 数据发布到 `/xrobotoolkit/*`。
- 启动 `xrt_retargeting`，订阅 `/xrobotoolkit/body_state`。
- 发布 `/retargeting/retargeted`，供 StepIt 使用。

### 3. 启动 StepIt 遥操作

```bash
cd /home/kyx/robot/vla/stepit_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

STEPIT_JOY_NAME=pico \
  bash src/stepit/scripts/run.sh \
    models/unitree_g1_xtrack_0429grav_teleop/real.conf.sh \
    src/stepit/config/run/ros.conf.sh \
    -f nnrtapi@tensorrt \
    -c joystick \
    -f joystick@ros2 \
    -v 3 \
    -- --ros-args -p joy_topic:=/xrobotoolkit/joy
```

作用：

- 订阅 `/xrobotoolkit/joy`。
- 订阅 `/retargeting/retargeted`。
- 在 ROS2 namespace `/stepit` 下发布机器人状态、策略输出和低层命令相关话题。

### 4. 启动相机发布端

在接 RealSense 的机器人端运行：

```bash
cd /home/kyx/robot/vla/robo_collector/src/camera
bash scripts/setup_camera_env.sh --server
source .venv_camera/bin/activate
bash scripts/run_realsense_server.sh --port 5555
```

只采 RGB 时：

```bash
bash scripts/run_realsense_server.sh --port 5555 --no-depth
```

### 5. 测试相机客户端

在采集主机运行：

```bash
cd /home/kyx/robot/vla/robo_collector/src/camera
bash scripts/setup_camera_env.sh --client
source .venv_camera/bin/activate
bash scripts/test_camera_client.sh --host 192.168.123.164 --port 5555
```

可视化检查：

```bash
bash scripts/run_camera_viewer.sh --host 192.168.123.164 --port 5555
```

## 已验证的 ROS2 图

2026-05-18 在 `10.12.120.125` 上观察到两个主命令正在运行，ROS2 节点包括：

- `/xrobotoolkit_state_publisher`
- `/xrt_retargeting`
- `/xrt_retargeting_visualizer`
- `/stepit/stepit_ros2`
- `/raycaster_publisher`

常用检查命令：

```bash
source /opt/ros/humble/setup.bash
source /home/kyx/robot/vla/stepit_ws/install/setup.bash
source /home/kyx/robot/vla/xrt_retargeting/install/setup.bash

ros2 node list
ros2 topic list -t
ros2 node info /stepit/stepit_ros2
ros2 node info /xrt_retargeting
```

## 后续采集接口候选

优先采集这些 topic：

| Topic | Type | 当前用途 |
| --- | --- | --- |
| `/stepit/joint_states` | `sensor_msgs/msg/JointState` | 机器人关节状态、命令、gain 和期望 torque 的合并发布 |
| `/stepit/imu` | `sensor_msgs/msg/Imu` | IMU 姿态、角速度、线加速度 |
| `/stepit/status` | `diagnostic_msgs/msg/DiagnosticStatus` | StepIt 运行状态 |
| `/stepit/field/last_target_joint_pos` | `std_msgs/msg/Float32MultiArray` | position actuator 的目标关节角，当前观测为 29 维，约 50 Hz |
| `/stepit/field/action` | `std_msgs/msg/Float32MultiArray` | 策略 action，当前观测为 29 维 |
| `/retargeting/retargeted` | `xrt_retargeting_msgs/msg/RetargetedConfiguration` | XRT 输出给 StepIt 的 retargeted qpos，可作为上游对照 |
| `/xrobotoolkit/joy` | `sensor_msgs/msg/Joy` | 手柄输入，可用于对齐操作者动作 |

`/stepit/joint_states` 的消息结构来自 StepIt 源码 `plugin/ros2_base/src/publisher.cpp`：

- 前 `dof` 项：真实电机关节状态，name 后缀为 `_joint`。
- 接着是 foot force 项，当前观测为 `LL_FOOT`、`LR_FOOT`。
- 接着 `dof` 项：下发给机器人低层的命令，name 后缀为 `_cmd`。
- 最后 `dof` 项：控制 gain 和期望 torque，name 后缀为 `_gain`。

因此后续采集代码不要只靠固定数组下标解析 `/stepit/joint_states`；应优先按 `name` 字段拆分 `_joint`、`_cmd`、`_gain`。

`/stepit/field/last_target_joint_pos` 来自 StepIt 源码 `plugin/policy_neuro/src/actuator.cpp`，由 `PositionActuator` 注册并发布。它更适合作为“目标关节角”的直接采集源。

## 快速验证命令

检查 topic 类型：

```bash
ros2 topic list -t
```

检查当前关节状态：

```bash
ros2 topic echo --once /stepit/joint_states
```

检查目标关节角：

```bash
ros2 topic echo --once /stepit/field/last_target_joint_pos
ros2 topic hz /stepit/field/last_target_joint_pos --window 10
```

检查 StepIt 低层状态频率：

```bash
ros2 topic hz /stepit/joint_states --window 10
```

当前观测：

- `/stepit/joint_states` 约 500 Hz。
- `/stepit/field/last_target_joint_pos` 约 50 Hz。

## 后续实现建议

新增机器人状态采集代码时，建议做成 `robo_collector` 内的 ROS2 subscriber 或独立 Python 进程：

- 订阅 `/stepit/joint_states`、`/stepit/imu`、`/stepit/status`。
- 订阅 `/stepit/field/last_target_joint_pos` 作为目标关节角。
- 可选订阅 `/stepit/field/action`、`/retargeting/retargeted`、`/xrobotoolkit/joy` 用于调试和训练数据补充。
- 使用 ROS2 header timestamp；没有 header 的 `Float32MultiArray` 需要在 callback 中记录本机接收时间。
- 与相机 `CameraClient.read()` 返回的 `timestamps` 做统一时间戳封装。
- 存储时保留 topic 名、ROS2 type、数据维度、采集频率和解析版本，避免后续 StepIt 配置变化时无法回放。

## 待确认事项

- 目标训练数据是否需要采集 `action`，还是只需要 `last_target_joint_pos`。
- `last_target_joint_pos` 的 29 个关节顺序是否始终与 StepIt `spec.joint_names` 一致。源码上符合 actuator 的 `policy_spec.dof` 顺序，但采集代码应保存关节名映射。
- 相机主机 IP `192.168.123.164` 是否固定，或需要做成配置项。
