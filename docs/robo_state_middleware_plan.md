# Robo State 中间层实现计划

本文档记录 `src/robo_state/` 中间层的目标、ROS2 接口、消息 schema、节点行为和验收方式。该模块只负责把 StepIt 分散发布的 ROS2 topic 规整成统一强类型状态样本，不负责存储 episode，不处理相机数据。

## 目标

新增两个 ROS2 package：

- `src/robo_state`：Python 节点 package，订阅 StepIt 原始 topic，校验维度并缓存最新状态。
- `src/robo_state_msgs`：自定义 msg package，定义统一机器人状态样本。

`robo_state` 以 `/stepit/field/last_target_joint_pos` 作为约 50 Hz 主时钟。每次收到目标关节角时，节点检查缓存中的必需字段，拼接策略输入，计算 StepIt observation 的 L2 差值，并发布一个统一样本。

后续统一采集模块只需要订阅 `/robo_state/sample`，再与相机数据一起做对齐和落盘。

## Workspace Layout

仓库根目录作为 colcon workspace。ROS2 package 和非 ROS2 相机工具都放在 `src/` 下：

```text
robo_collector/
  src/
    camera/
    robo_state/
    robo_state_msgs/
  docs/
```

从仓库根目录执行 `colcon build`，不要进入单个 package 目录构建。

## 不做的事

- 不写数据文件。
- 不保存 episode。
- 不处理相机。
- 不决定最终训练数据存储格式。

## Public ROS2 Interface

主状态 topic：

```text
/robo_state/sample
```

类型：

```text
robo_state_msgs/msg/RoboStateSample
```

健康状态 topic：

```text
/robo_state/status
```

类型：

```text
diagnostic_msgs/msg/DiagnosticStatus
```

## Message Design

### `RoboStateSample.msg`

```text
std_msgs/Header header

robo_state_msgs/PolicyState policy_state
robo_state_msgs/RobotLowState robot_state
sensor_msgs/Imu imu

float32[29] target_joint_pos
float32[45] aligned_target_pos
float32[29] action
float32[1545] stepit_observation
float32 observation_l2_error

string[] missing_optional_fields
```

### `PolicyState.msg`

策略输入字段按以下顺序拼接为 `flattened`：

```text
float32[90] relative_ori_6d
float32[45] motion_anchor_lin_vel_b
float32[45] motion_anchor_ang_vel_b
float32[30] ang_vel_history
float32[30] gravity_history
float32[290] joint_pos_rel_history
float32[290] joint_vel_history
float32[290] action_history

float32[1110] flattened
```

### `RobotLowState.msg`

从 `/stepit/joint_states` 解析得到：

```text
string[29] joint_names

float32[29] joint_pos
float32[29] joint_vel
float32[29] joint_torque

float32[29] cmd_joint_pos
float32[29] cmd_joint_vel
float32[29] cmd_joint_torque

float32[29] kp
float32[29] kd
float32[29] desired_torque

string[] foot_names
float32[] foot_force
```

解析规则：

- `*_joint`：真实机器人关节状态，进入 `joint_pos`、`joint_vel`、`joint_torque`。
- `*_cmd`：低层命令，进入 `cmd_joint_pos`、`cmd_joint_vel`、`cmd_joint_torque`。
- `*_gain`：控制 gain 和期望 torque，进入 `kp`、`kd`、`desired_torque`。
- 不带上述后缀的 name 视为 foot force 项，name 进入 `foot_names`，`effort` 进入 `foot_force`。

解析必须按 `name` 后缀执行，不能硬编码数组切片。

## 输入 Topic

```text
/stepit/joint_states
/stepit/imu
/stepit/status
/stepit/field/last_target_joint_pos
/stepit/field/aligned_target_pos
/stepit/field/action
/stepit/field/observation
/stepit/field/relative_ori_6d
/stepit/field/motion_anchor_lin_vel_b
/stepit/field/motion_anchor_ang_vel_b
/stepit/field/ang_vel_history
/stepit/field/gravity_history
/stepit/field/joint_pos_rel_history
/stepit/field/joint_vel_history
/stepit/field/action_history
```

## 节点行为

每次收到 `/stepit/field/aligned_target_pos`：

1. 校验 `aligned_target_pos` 为 45 维。
2. 检查必需字段是否齐全且未超过 `max_cache_age_sec`。
3. 校验所有 StepIt field 的维度。
4. 按固定顺序拼接 8 个策略状态输入字段为 `PolicyState.flattened`，总维度必须为 1110。
5. 保留 `/stepit/field/observation` 1545 维 actor 原始输入用于调试。
6. 发布 `/robo_state/sample`。

错误处理：

- 必需字段缺失或过期时，不发布 sample，只更新 `/robo_state/status`。
- 字段维度不符时，发布 error status，并拒绝发布错误 sample。
- `/stepit/status` 的 WARN/ERROR 等级需要透传到 `/robo_state/status`，便于现场排查。

## 参数

```text
stepit_ns:=/stepit
output_topic:=/robo_state/sample
status_topic:=/robo_state/status
publish_only_when_complete:=true
validate_observation:=true
max_cache_age_sec:=0.2
```

## 启动命令

构建：

```bash
cd /home/kyx/robot/vla/robo_collector

source /opt/ros/humble/setup.bash
colcon build --packages-select robo_state_msgs robo_state --symlink-install
```

如果之前用旧布局或旧 build type 构建过，先清理生成目录，避免 colcon 使用缓存：

```bash
rm -rf build install log
```

启动：

```bash
cd /home/kyx/robot/vla/robo_collector

source /opt/ros/humble/setup.bash
source /home/kyx/robot/vla/stepit_ws/install/setup.bash
source /home/kyx/robot/vla/robo_collector/install/setup.bash

ros2 run robo_state robo_state_node
```

带参数示例：

```bash
ros2 run robo_state robo_state_node --ros-args \
  -p stepit_ns:=/stepit \
  -p output_topic:=/robo_state/sample \
  -p status_topic:=/robo_state/status \
  -p publish_only_when_complete:=true \
  -p validate_observation:=true \
  -p max_cache_age_sec:=0.2
```

## Test Plan

单元测试：

- 8 个策略状态输入字段拼接顺序必须等于 1110 维。
- `JointState` 按 `*_joint`、`*_cmd`、`*_gain`、foot force 正确拆分。
- 缺字段时不发布 sample。
- 维度错误时发布 error status。
- `/stepit/field/observation` 1545 维原始 actor 输入可保留用于调试。

现场 smoke test：

```bash
ros2 topic echo --once /robo_state/sample
ros2 topic hz /robo_state/sample --window 10
ros2 topic echo --once /robo_state/status
```

## 验收标准

- `/robo_state/sample` 稳定约 50 Hz。
- 每条 sample 有 1110 维 `policy_state.flattened`。
- 每条 sample 有 45 维 `aligned_target_pos`。
- 每条 sample 有 29 维 `target_joint_pos`。
- `RobotLowState.joint_names` 顺序稳定并写入消息。
- 字段缺失或维度错误时不会发布错误 sample。
- 节点不创建数据文件。

## Assumptions

- ROS2 package 名固定为 `robo_state`，源码路径为 `src/robo_state/`。
- 使用自定义 ROS2 msg，方便后续统一采集模块强类型订阅。
- StepIt 的 `last_target_joint_pos`、`action` 和低层状态 DOF 当前为 29。
- `robo_state` 只做中间层发布，不落盘、不接相机。
- 后续统一采集模块负责订阅 `/robo_state/sample` 和相机数据，并决定最终存储格式。
