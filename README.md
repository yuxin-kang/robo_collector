# Robo Collector

Robo Collector 是 G1 遥操作数据采集工作区。它把 StepIt 发布的机器人状态规整成 `/robo_state/sample`，再和 RealSense `ego_view` RGB 图像对齐，按 episode 写成 LeRobot v2.1 风格数据集。

## 目录

```text
robo_collector/
  src/
    camera/                 # RealSense ZMQ server/client/viewer，非 ROS package
    robo_state_msgs/         # RoboStateSample 等状态消息
    robo_state/              # StepIt -> /robo_state/sample 中间层
    robo_collector_msgs/     # RecordCommand 控制消息
    robo_collector/          # LeRobot collector 节点
  scripts/
    setup_data_collection_env.sh
    launch_data_collection.sh
  docs/
  outputs/
```

## 1. 准备 Python 环境

在采集主机的仓库根目录运行：

```bash
cd /home/kyx/robot/vla/robo_collector

source /opt/ros/humble/setup.bash
bash scripts/setup_data_collection_env.sh
source .venv_data_collection/bin/activate
```

默认只安装采集运行所需依赖：

```text
opencv-python, numpy, pyzmq, msgpack, pyarrow
```

`rclpy` 通过 `--system-site-packages` 从系统 ROS 环境继承。正常采集不需要安装官方 `lerobot`。只有需要官方 loader、转换或训练工具时才运行：

```bash
bash scripts/setup_data_collection_env.sh --with-lerobot
```

## 2. 编译 ROS2 包

```bash
cd /home/kyx/robot/vla/robo_collector

source /opt/ros/humble/setup.bash
source .venv_data_collection/bin/activate

colcon build --symlink-install \
  --packages-select robo_collector_msgs robo_state_msgs robo_state robo_collector

source install/setup.bash
```

确认包已注册：

```bash
ros2 pkg list | grep -E 'robo_collector|robo_state'
```

## 3. 启动外部依赖

采集脚本不会启动 StepIt、XRT、全身控制或 camera server。现场需要先手动启动这些进程。

### XRT 和 StepIt

按现场流程先启动：

- XRT retargeting
- StepIt 遥操作
- 全身控制/机器人侧控制链路

确认 StepIt 正在发布：

```bash
ros2 node list
ros2 topic hz /stepit/field/last_target_joint_pos
```

### RealSense Camera Server

在连接 RealSense 的机器上运行：

```bash
cd /home/kyx/robot/vla/robo_collector/src/camera

bash scripts/setup_camera_env.sh --server
source .venv_camera/bin/activate

bash scripts/run_realsense_server.sh --port 5555 --no-depth
```

在采集主机可测试相机：

```bash
cd /home/kyx/robot/vla/robo_collector/src/camera
source .venv_camera/bin/activate

bash scripts/test_camera_client.sh --host 192.168.123.164 --port 5555
bash scripts/run_camera_viewer.sh --host 192.168.123.164 --port 5555
```

## 4. 一键启动采集相关进程

回到仓库根目录：

```bash
cd /home/kyx/robot/vla/robo_collector

source /opt/ros/humble/setup.bash
source .venv_data_collection/bin/activate
source install/setup.bash

bash scripts/launch_data_collection.sh \
  --camera-host 192.168.123.164 \
  --camera-port 5555 \
  --root-output-dir outputs \
  --fps 50
```

该脚本启动一个 tmux session：`robo_data_collection`。

| Pane | 进程 |
| --- | --- |
| 0 | `ros2 run robo_state robo_state_node` |
| 1 | `ros2 run robo_collector lerobot_collector_node ...` |
| 2 | `src/camera/scripts/run_camera_viewer.sh ...` |

进入 tmux：

```bash
tmux attach -t robo_data_collection
```

退出 tmux 但不关闭程序：按 `Ctrl-b`，再按 `d`。

确认 collector 正常：

```bash
ros2 node list | grep collector
ros2 topic list | grep robo_collector
ros2 topic echo --once /robo_collector/status
```

应看到：

```text
/lerobot_collector_node
/robo_collector/record_command
/robo_collector/status
```

## 5. 控制 episode

开始录制：

```bash
ros2 topic pub --once /robo_collector/record_command \
  robo_collector_msgs/msg/RecordCommand \
  "{command: 1, task_prompt: 'shake hands', episode_id: 'manual_001'}"
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

查看状态：

```bash
ros2 topic echo /robo_collector/status
```

## 6. 输出数据

保存后数据写到：

```text
outputs/<dataset_name>/
  data/
    train-000000.parquet
  videos/
    observation.images.ego_view/
      episode_000000.mp4
  meta/
    info.json
    modality.json
    episodes.jsonl
    tasks.jsonl
```

检查最近一次采集：

```bash
latest=$(ls -td outputs/robo_collector_* | head -1)
find "$latest" -maxdepth 4 -type f | sort
cat "$latest/meta/episodes.jsonl"
cat "$latest/meta/tasks.jsonl"
```

检查 parquet 内容：

```bash
source .venv_data_collection/bin/activate

python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sorted(Path("outputs").glob("robo_collector_*"))[-1])
table = pq.read_table(root / "data/train-000000.parquet")
row = table.slice(0, 1).to_pylist()[0]

print("dataset:", root)
print("rows:", table.num_rows)
print("columns:", table.column_names)
print("task:", row["annotation.human.action.task_description"])
print("video_ref:", row["observation.images.ego_view"])
print("joint_position_len:", len(row["observation.state.joint_position"]))
print("policy_action_len:", len(row["action.policy_action"]))
PY
```

## 7. 关闭程序

关闭采集 tmux session：

```bash
tmux kill-session -t robo_data_collection
```

确认关闭：

```bash
ros2 node list | grep -E 'collector|robo_state'
```

## 8. 常见问题

### `Waiting for at least 1 matching subscription(s)...`

说明 collector 没有运行，或者当前终端没有 source workspace。

检查：

```bash
ros2 node list | grep collector
ros2 topic list | grep robo_collector
```

如果没有 `/lerobot_collector_node`，查看 tmux collector pane：

```bash
tmux list-panes -t robo_data_collection -F '#{pane_index}: dead=#{pane_dead} cmd=#{pane_current_command}'
tmux capture-pane -t robo_data_collection:0.1 -p -S -160
```

### `colcon build --packages-select robo_collector` 找不到 msg 包

`robo_collector` 依赖消息包，首次或清理后需要一起编译：

```bash
colcon build --symlink-install \
  --packages-select robo_collector_msgs robo_state_msgs robo_state robo_collector
```

### camera viewer 有 `QFontDatabase` 警告

这是 OpenCV Qt 字体警告，viewer 仍可连接相机；不影响 collector 写数据。

### `setup_data_collection_env.sh` 下载 `torch` 失败

默认脚本不会安装 `lerobot`，不会下载 PyTorch/CUDA 大包。只有使用 `--with-lerobot` 时才会拉这些大依赖。现场采集不需要 `--with-lerobot`。

### `status` 显示缺少 state 或 camera

collector 已收到 START，但跳过写帧：

- `missing robo_state sample`：检查 `/robo_state/sample` 和 StepIt。
- `missing camera frame`：检查 camera server 地址、端口和 stream。
- `stale camera frame`：检查相机发布频率和网络。

## 9. 开发验证

```bash
python3 -m py_compile src/camera/robo_collector_camera/*.py
python3 -m py_compile src/robo_collector/robo_collector/*.py

PYTHONPATH=src/robo_collector python3 -m unittest discover -s src/robo_collector/test -q
PYTHONPATH=src/robo_state python3 -m unittest discover -s src/robo_state/test -q

bash -n scripts/setup_data_collection_env.sh scripts/launch_data_collection.sh src/camera/scripts/*.sh
```
