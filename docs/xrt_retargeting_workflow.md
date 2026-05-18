# XRT Retargeting 遥操作流程

本文档记录 `/home/kyx/robot/vla/xrt_retargeting` 在当前遥操作采集系统中的运行方式和 ROS2 数据流。

## 启动命令

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

## Launch 行为

`xrt_retargeting.launch.py` 会：

- include `xrt_bridge/launch/state_publisher.launch.py`。
- 启动 `xrobotoolkit_state_publisher`，默认 topic prefix 为 `xrobotoolkit`。
- 启动 `xrt_retargeting_node`。
- 当 `visualize:=true` 时启动 `xrt_retargeting_visualizer`。

## 当前节点

运行后观察到：

- `/xrobotoolkit_state_publisher`
- `/xrt_retargeting`
- `/xrt_retargeting_visualizer`

检查命令：

```bash
source /opt/ros/humble/setup.bash
source /home/kyx/robot/vla/xrt_retargeting/install/setup.bash

ros2 node list
ros2 node info /xrt_retargeting
```

## Topic 数据流

`/xrt_retargeting` 订阅：

- `/xrobotoolkit/body_state`：`xrt_bridge/msg/BodyState`

`/xrt_retargeting` 发布：

- `/retargeting/retargeted`：`xrt_retargeting_msgs/msg/RetargetedConfiguration`
- `/retargeting/frames`：`xrt_retargeting_msgs/msg/RetargetingFrames`

XRoboToolkit bridge 还会发布：

- `/xrobotoolkit/app_state`
- `/xrobotoolkit/head_state`
- `/xrobotoolkit/controller_state`
- `/xrobotoolkit/joy`
- `/xrobotoolkit/hand_state`
- `/xrobotoolkit/body_state`
- `/xrobotoolkit/motion_state`

StepIt 当前使用：

- `/xrobotoolkit/joy`
- `/retargeting/retargeted`

## 与采集代码的关系

后续机器人状态采集主接口应以 StepIt 发布的 `/stepit/*` 为准。XRT retargeting 侧建议按需额外采集：

- `/retargeting/retargeted`：记录人体动作 retarget 后的机器人 qpos，用于和 StepIt 目标角做对照。
- `/xrobotoolkit/joy`：记录操作者手柄输入。
- `/xrobotoolkit/body_state`：如需要训练或回放人体姿态，可额外保存。

## 调试命令

```bash
ros2 topic list -t
ros2 topic echo --once /xrobotoolkit/joy
ros2 topic echo --once /xrobotoolkit/body_state
ros2 topic echo --once /retargeting/retargeted
ros2 topic hz /retargeting/retargeted --window 10
```

如果 `/retargeting/retargeted` 没有数据，先检查：

- XRoboToolkit 服务是否在发送 body tracking 数据。
- `/xrobotoolkit/body_state` 是否有消息。
- `actual_human_height`、`init_max_iterations`、`max_iterations`、`downsample_factor`、`timestep` 参数是否与当前测试命令一致。
