# YAML 采集字段配置计划

本文档固定 Robo Collector 通过 YAML 配置选择机器人采集字段的目标、接口、实现范围和验证计划。

## Summary

- 新增一个 YAML 配置文件，用来选择保存哪些机器人字段。
- 配置只控制 `target` 和 `state` 数值字段；相机视频、timestamp、frame/episode/task metadata 始终保存。
- 字段按整组向量选择，不做单个维度过滤；例如 `joint_pos_rel_history` 永远是完整 290 维。
- 配置字段必须是支持全集的子集；启动时校验，非法字段直接报错退出。

## Public Interface

新增 ROS 参数和 launcher 参数：

```bash
-p field_config_path:=configs/collection_fields.yml
```

```bash
bash scripts/launch_data_collection.sh \
  --field-config configs/collection_fields.yml \
  --camera-host 192.168.123.164 \
  --camera-streams head,ego_view
```

YAML 格式固定为友好分组名。当前策略采集配置为：

```yaml
target:
  - aligned_target_pos

state:
  - relative_ori_6d
  - motion_anchor_lin_vel_b
  - motion_anchor_ang_vel_b
  - ang_vel_history
  - gravity_history
  - joint_pos_rel_history
  - joint_vel_history
  - action_history
```

字段映射固定为：

```text
target.joint_position -> action.joint_position
target.aligned_target_pos -> action.aligned_target_pos
state.joint_position -> observation.state.joint_position
state.joint_velocity -> observation.state.joint_velocity
state.joint_torque -> observation.state.joint_torque
state.imu_angular_velocity -> observation.state.imu_angular_velocity
state.imu_linear_acceleration -> observation.state.imu_linear_acceleration
state.projected_gravity_or_quat -> observation.state.projected_gravity_or_quat
state.relative_ori_6d -> observation.state.relative_ori_6d
state.motion_anchor_lin_vel_b -> observation.state.motion_anchor_lin_vel_b
state.motion_anchor_ang_vel_b -> observation.state.motion_anchor_ang_vel_b
state.ang_vel_history -> observation.state.ang_vel_history
state.gravity_history -> observation.state.gravity_history
state.joint_pos_rel_history -> observation.state.joint_pos_rel_history
state.joint_vel_history -> observation.state.joint_vel_history
state.action_history -> observation.state.action_history
```

## Implementation Changes

新增字段配置加载/校验模块：

- 读取 YAML。
- 要求顶层只允许 `target` 和 `state`。
- 两个 key 都必须存在，且列表不能为空。
- 拒绝未知字段、重复字段、非 list/string 类型。
- 如果未传 `field_config_path`，保持旧兼容行为：保存现有全部机器人字段，包括 `action.policy_action`。

修改 writer：

- `LeRobotV21Writer` 接收可选 `field_selection`。
- parquet row 只写配置选中的数值字段。
- `meta/info.json.features` 只声明配置选中的数值字段。
- `meta/modality.json` 的 `action` / `observation.state` 只声明配置选中的字段。
- 相机字段、annotation、timestamp、frame_index、episode_index、index、task_index 仍固定写入。

修改 collector/launcher：

- `collector_node` 新增参数 `field_config_path`。
- 启动时读取配置；配置错误时 fail fast，并在终端输出英文错误。
- `launch_data_collection.sh` 新增 `--field-config` 并传给 ROS 参数。
- `scripts/setup_data_collection_env.sh` 增加 `PyYAML`；`package.xml` 增加 `python3-yaml` 运行依赖。

新增示例配置：

- `configs/collection_fields.yml`
- 默认示例保存 `aligned_target_pos` 作为训练输出目标，保存 8 个策略状态输入字段。

## Test Plan

配置加载单测：

- 合法 YAML 能加载并映射到正确 parquet key。
- 未知字段报错。
- 重复字段报错。
- 缺少 `target` 或 `state` 报错。
- 空列表报错。
- 非 list/string 报错。

writer 单测：

- 配置为保存 `target.aligned_target_pos` 和 8 个策略状态输入字段时，parquet 只包含这些配置字段。
- `info.json.features` 只包含选中的数值字段和固定 metadata/image 字段。
- `modality.json` 只包含选中的 `action` / `observation.state` 字段。
- 未传配置时保持当前行为，现有测试继续通过。

集成/脚本检查：

- `bash -n scripts/launch_data_collection.sh scripts/setup_data_collection_env.sh`
- `py_compile` 新增模块和改动模块。
- 本地 `robo_collector` 单测全通过。
- ROS 环境里 `colcon build --symlink-install --packages-select robo_collector` 通过。

## Assumptions

- YAML 只控制机器人字段，不控制双相机视频保存。
- v1 不支持按单个关节过滤，避免破坏 29 维 feature shape 和 joint names。
- `target.aligned_target_pos` 表示当前策略训练输出目标，即 `action.aligned_target_pos`。
- `target.joint_position` 仍可用于兼容旧机器人目标关节角度，即 `action.joint_position`。
- `action.policy_action` 不放进 YAML v1；未传配置时仍保留它以兼容已有数据格式。
