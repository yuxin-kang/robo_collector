# StepIt ROS2 遥操作接口

本文档记录 `/home/kyx/robot/vla/stepit_ws` 中 StepIt 遥操作命令和后续数据采集需要关注的 ROS2 输出。

## 启动命令

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

运行后观察到实际进程会通过 `ros2 run stepit_ros2 stepit` 启动，并 remap 到 namespace `/stepit`。

## 节点输入输出

检查命令：

```bash
source /opt/ros/humble/setup.bash
source /home/kyx/robot/vla/stepit_ws/install/setup.bash
source /home/kyx/robot/vla/xrt_retargeting/install/setup.bash

ros2 node info /stepit/stepit_ros2
```

当前观察到 `/stepit/stepit_ros2` 订阅：

- `/xrobotoolkit/joy`：`sensor_msgs/msg/Joy`
- `/retargeting/retargeted`：`xrt_retargeting_msgs/msg/RetargetedConfiguration`

当前观察到 `/stepit/stepit_ros2` 发布：

- `/stepit/joint_states`：`sensor_msgs/msg/JointState`
- `/stepit/imu`：`sensor_msgs/msg/Imu`
- `/stepit/status`：`diagnostic_msgs/msg/DiagnosticStatus`
- `/stepit/field/*`：`std_msgs/msg/Float32MultiArray`

## 后续采集的关键 topic

| Topic | Type | 说明 |
| --- | --- | --- |
| `/stepit/joint_states` | `sensor_msgs/msg/JointState` | 机器人低层状态和命令的合并消息 |
| `/stepit/imu` | `sensor_msgs/msg/Imu` | 机器人 IMU |
| `/stepit/status` | `diagnostic_msgs/msg/DiagnosticStatus` | StepIt 状态 |
| `/stepit/field/last_target_joint_pos` | `std_msgs/msg/Float32MultiArray` | position actuator 当前目标关节角 |
| `/stepit/field/action` | `std_msgs/msg/Float32MultiArray` | 策略 action |
| `/stepit/field/joint_pos` | `std_msgs/msg/Float32MultiArray` | 策略侧关节位置 field |
| `/stepit/field/joint_vel` | `std_msgs/msg/Float32MultiArray` | 策略侧关节速度 field |

当前实测：

- `/stepit/joint_states` 约 500 Hz。
- `/stepit/field/last_target_joint_pos` 约 50 Hz，消息为 29 维 float array。
- `/stepit/field/action` 当前观测为 29 维 float array。

## `joint_states` 结构

源码位置：

```text
/home/kyx/robot/vla/stepit_ws/src/stepit/plugin/ros2_base/src/publisher.cpp
```

`publishLowLevel()` 将一个 `JointState` 消息分成四段：

1. `spec.dof` 个真实关节状态，name 后缀为 `_joint`。
2. `spec.foot_names.size()` 个 foot force 项，例如 `LL_FOOT`、`LR_FOOT`。
3. `spec.dof` 个低层命令，name 后缀为 `_cmd`。
4. `spec.dof` 个控制 gain / desired torque，name 后缀为 `_gain`。

采集代码应按 `name` 字段解析，不要硬编码数组切片。这样即使模型 DOF 或 foot 名称变化，回放和训练数据也能保持可解释。

## `last_target_joint_pos` 来源

源码位置：

```text
/home/kyx/robot/vla/stepit_ws/src/stepit/plugin/policy_neuro/src/actuator.cpp
```

`PositionActuator` 注册：

```text
last_target_joint_pos
```

并在 `setLowCmd()` 中把策略 action 经过 scale/bias 转成 `target_joint_pos_`。因此它是后续“机器人关节目标角度”的首选采集 topic。

## 调试命令

```bash
ros2 topic echo --once /stepit/joint_states
ros2 topic echo --once /stepit/field/last_target_joint_pos
ros2 topic echo --once /stepit/field/action
ros2 topic hz /stepit/joint_states --window 10
ros2 topic hz /stepit/field/last_target_joint_pos --window 10
```

## 后续实现注意

- `Float32MultiArray` 没有 header，采集节点要自行记录 ROS2 接收时间。
- 保存 `joint_states.name` 到每条 episode 的 metadata 中。
- 同时保存 `/stepit/field/last_target_joint_pos` 和 `/stepit/joint_states` 中 `_cmd` 段，便于确认两者是否一致。
- 训练数据 schema 中区分 `measured_joint_pos`、`target_joint_pos`、`cmd_joint_pos`、`action`。
