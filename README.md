# Robo Collector

G1 遥操作数据采集工作区。采集链路把 StepIt 状态整理成 `/robo_state/sample`，再和 RealSense `head`、`ego_view` 图像对齐，保存为 LeRobot v2.1 风格 dataset。

当前默认字段配置在 [configs/collection_fields.yml](configs/collection_fields.yml)：

- target: `action.aligned_target_pos`，45 维
- state: policy 输入状态，合计 1110 维
- camera、timestamp、episode/frame/task metadata 固定保存

## Setup

```bash
cd /home/kyx/robot/vla/robo_collector

source /opt/ros/humble/setup.bash
bash scripts/setup_data_collection_env.sh
source .venv_data_collection/bin/activate

colcon build --symlink-install \
  --packages-select robo_state_msgs robo_collector_msgs robo_state robo_collector

source install/setup.bash
```

如果你用 zsh，把 `setup.bash` 换成 `setup.zsh`。

确认消息已更新：

```bash
ros2 interface show robo_state_msgs/msg/RoboStateSample | grep aligned_target_pos
ros2 interface show robo_state_msgs/msg/PolicyState | grep flattened
```

期望看到：

```text
float32[45] aligned_target_pos
float32[1110] flattened
```

## Launch

启动前需要先手动启动 StepIt、XRT、机器人控制链路和 RealSense camera server。

```bash
bash scripts/launch_data_collection.sh \
  --field-config configs/collection_fields.yml \
  --camera-host 192.168.123.164 \
  --camera-port 5555 \
  --camera-streams head,ego_view \
  --root-output-dir outputs \
  --fps 50
```

脚本会创建 tmux session：

```bash
tmux attach -t robo_data_collection
```

查看状态：

```bash
ros2 topic echo --once /robo_collector/status
```

## Record

开始：

```bash
ros2 topic pub --once /robo_collector/record_command \
  robo_collector_msgs/msg/RecordCommand \
  "{command: 1, task_prompt: 'Shake hand with somebody'}"
```

保存：

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

同一次 launch 下，多次 START/STOP 会追加到同一个 dataset：

```text
outputs/robo_collector_YYYYMMDD_HHMMSS/
  data/train-000000.parquet
  data/train-000001.parquet
  videos/observation.images.head/episode_000000.mp4
  videos/observation.images.head/episode_000001.mp4
  videos/observation.images.ego_view/episode_000000.mp4
  videos/observation.images.ego_view/episode_000001.mp4
  meta/
```

如果想每次录制都用新的 dataset，重启 launch 脚本，或传不同的 `--dataset-name`。

## Check Data

```bash
latest=$(ls -td outputs/robo_collector_* | head -1)
cat "$latest/meta/episodes.jsonl"
```

检查 parquet 字段长度：

```bash
python - <<'PY'
from pathlib import Path
import pyarrow.parquet as pq

root = Path(sorted(Path("outputs").glob("robo_collector_*"))[-1])
table = pq.read_table(root / "data/train-000000.parquet")
row = table.slice(0, 1).to_pylist()[0]

keys = [
    "action.aligned_target_pos",
    "observation.state.relative_ori_6d",
    "observation.state.motion_anchor_lin_vel_b",
    "observation.state.motion_anchor_ang_vel_b",
    "observation.state.ang_vel_history",
    "observation.state.gravity_history",
    "observation.state.joint_pos_rel_history",
    "observation.state.joint_vel_history",
    "observation.state.action_history",
]

print("dataset:", root)
print("rows:", table.num_rows)
for key in keys:
    print(key, len(row[key]))
PY
```

## Convert To GR00T

现有采集输出不会被原地改写。新增脚本会读取一个已有 dataset，并在新的目标目录下生成 Isaac-GR00T 兼容数据集：

```bash
python scripts/convert_outputs_to_gr00t.py \
  --source-root outputs \
  --dataset-name robo_collector_YYYYMMDD_HHMMSS \
  --dest-root exports \
  --output-name robo_collector_YYYYMMDD_HHMMSS_gr00t \
  --action-source aligned_target_pos
```

参数说明：

- `--source-root`：已有数据集所在父目录
- `--dataset-name`：已有数据集文件夹名字
- `--dest-root`：转换后数据集输出父目录
- `--output-name`：转换后数据集目录名；默认 `<dataset-name>_gr00t`
- `--action-source`：GR00T `action` 列映射来源，可选 `aligned_target_pos`、`policy_action`、`joint_position`
  - `policy_action` 只有在 source parquet 本身包含 `action.policy_action` 时可用；默认 YAML 采集配置不会写这个列

当前转换器面向本项目现有 split-field source schema：

- 读取 `observation.state.relative_ori_6d` 等拆分状态列
- 重新拼成 GR00T 所需的单列 `observation.state`
- 把选中的 action 列重写成单列 `action`
- 复制视频到 `videos/chunk-000/observation.images.<camera>/episode_*.mp4`
- 生成 GR00T 风格 `meta/modality.json`

如果 source dataset 缺少必需状态列，或缺少选中的 action 列，脚本会直接报错退出。

## Notes

- `Waiting for at least 1 matching subscription(s)...` 通常表示 collector 没启动或当前终端没 source workspace。
- `tmux capture-pane -t robo_data_collection:0.1 -p -S -160` 可以看 collector 日志。
- 默认不安装官方 `lerobot`，现场采集不需要它；只有训练/转换工具需要时才运行 `bash scripts/setup_data_collection_env.sh --with-lerobot`。
- 关闭采集：`tmux kill-session -t robo_data_collection`。
