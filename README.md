# Robo Collector

[![Ubuntu 22.04 / 24.04](https://img.shields.io/badge/Ubuntu-22.04%20%2F%2024.04-blue.svg?logo=ubuntu)](https://ubuntu.com/)
[![ROS 2 Humble / Jazzy](https://img.shields.io/badge/ROS%202-Humble%20%2F%20Jazzy-blue.svg)](https://docs.ros.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Robo Collector is a ROS 2 data-collection workspace for Unitree G1 teleoperation.
It normalizes StepIt robot state into `/robo_state/sample`, aligns it with
RealSense RGB streams such as `head` and `ego_view`, and stores episodes in a
LeRobot v2.1-style dataset. A conversion utility is also provided for exporting
existing datasets into an Isaac-GR00T-compatible layout.

The default field configuration is:

| Group | Fields |
| --- | --- |
| `target` | `action.aligned_target_pos`, 45 dimensions |
| `state` | Policy input state fields, 1110 dimensions in total |
| metadata | Camera references, timestamps, episode/frame indices, and task metadata |

## Repository Layout

```text
robo_collector/
  configs/collection_fields.yml
  configs/gesture_trigger_plan.example.yml
  scripts/
    setup_data_collection_env.sh
    launch_data_collection.sh
    convert_outputs_to_gr00t.py
  src/
    camera/                 # RealSense ZMQ publisher/client
    robo_state_msgs/        # Typed state sample messages
    robo_state/             # StepIt-to-RoboState normalization node
    robo_collector_msgs/    # Recording command messages
    robo_collector/         # Episode writer and collector node
```

## Setup

Supported deployment targets:

| Ubuntu | ROS 2 |
| --- | --- |
| 22.04 | Humble |
| 24.04 | Jazzy |

The setup examples and launch script resolve ROS in this order:

1. `ROS_SETUP_PATH`, if exported and points to a valid `setup.bash`
2. `/opt/ros/$ROS_DISTRO/setup.bash`, if `ROS_DISTRO` is exported
3. Ubuntu default mapping: 22.04 -> `humble`, 24.04 -> `jazzy`
4. the only installed distro under `/opt/ros`

If you keep multiple ROS distros under `/opt/ros` on another Ubuntu version,
export either `ROS_SETUP_PATH` or `ROS_DISTRO` before building or launching.

```bash
git clone https://github.com/yuxin-kang/robo_collector.git
cd robo_collector

source "$(bash scripts/resolve_ros_setup.sh)"

bash scripts/setup_data_collection_env.sh
source .venv_data_collection/bin/activate

colcon build --symlink-install \
  --packages-select robo_state_msgs robo_collector_msgs robo_state robo_collector

source install/setup.bash
```

If you use `zsh`, source `/opt/ros/$ROS_DISTRO/setup.zsh` and `install/setup.zsh`
for your interactive shell. `launch_data_collection.sh` still sources
`setup.bash` internally because it launches worker panes through `bash`.

Verify that the generated message interfaces contain the expected fields:

```bash
ros2 interface show robo_state_msgs/msg/RoboStateSample | grep aligned_target_pos
ros2 interface show robo_state_msgs/msg/PolicyState | grep flattened
```

Expected output:

```text
float32[45] aligned_target_pos
float32[1110] flattened
```

## Camera Setup

The camera module lives in [`src/camera`](src/camera). On the robot-side machine
connected to the RealSense cameras:

```bash
cd /path/to/robo_collector/src/camera
bash scripts/setup_camera_env.sh --server
source .venv_camera/bin/activate
bash scripts/run_realsense_server.sh --list-devices
```

Start the dual-camera RGB publisher:

```bash
bash scripts/run_realsense_server.sh \
  --camera head:<D405_SERIAL> \
  --camera ego_view:<D435I_SERIAL> \
  --port 5555 \
  --width 640 --height 480 --fps 30 \
  --jpeg-quality 80 \
  --no-depth
```

On the collection host, test the client or open the viewer:

```bash
cd /path/to/robo_collector/src/camera
bash scripts/setup_camera_env.sh --client
source .venv_camera/bin/activate
bash scripts/test_camera_client.sh --host 192.168.123.164 --port 5555
bash scripts/run_camera_viewer.sh --host 192.168.123.164 --port 5555
```

## Launch

Before launching Robo Collector, start the external teleoperation stack: StepIt,
XRT retargeting, robot control, and the RealSense camera server.

```bash
bash scripts/launch_data_collection.sh \
  --field-config configs/collection_fields.yml \
  --camera-host 192.168.123.164 \
  --camera-port 5555 \
  --camera-streams head,ego_view \
  --root-output-dir outputs \
  --fps 30
```

The recommended `configs/collection_fields.yml` also stores
`observation.state.joint_position` and `action.policy_action`, so the same
source dataset satisfies the default inputs of both the GR00T and OpenPI pi0.5
converters.

The launch script creates a tmux session:

```bash
tmux attach -t robo_data_collection
```

Check collector status:

```bash
ros2 topic echo --once /robo_collector/status
```

## Recording Episodes

Start a new episode:

```bash
ros2 topic pub --once /robo_collector/record_command \
  robo_collector_msgs/msg/RecordCommand \
  "{command: 1, task_prompt: 'Shake hand with somebody'}"
```

Stop and save the current episode:

```bash
ros2 topic pub --once /robo_collector/record_command \
  robo_collector_msgs/msg/RecordCommand \
  "{command: 2}"
```

Discard the current episode:

```bash
ros2 topic pub --once /robo_collector/record_command \
  robo_collector_msgs/msg/RecordCommand \
  "{command: 3}"
```

Multiple `START`/`STOP` cycles in the same launch append episodes to the same
dataset:

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

Restart the launch script or pass a different `--dataset-name` when you want a
new dataset directory.

## Gesture-Triggered Multi-Trial Recording

For repeated trials of the same task prompt, run the standalone
`gesture_trigger_node`. It reads a plan, watches `/robo_state/sample`, publishes
the existing `RecordCommand` messages, and only advances progress after the
planned `episode_id` appears in `meta/episodes.jsonl` with a non-empty saved
trajectory.

Start the normal collector first, then launch the trigger node in another shell:

```bash
ros2 run robo_collector gesture_trigger_node --ros-args \
  -p plan_path:=$PWD/configs/gesture_trigger_plan.example.yml
```

Plan fields:

- `dataset_root`: optional fixed dataset root. Leave it empty to discover the
  active dataset root from `/robo_collector/status`.
- `collector.fps`: recording frame rate used to enforce the configured
  `max_tail_frames` bound.
- `gesture_source`: which vector to read from `/robo_state/sample`.
- `references` and `conditions`: calibration targets plus trigger thresholds.
- `tasks`: one or more prompts, each with its own `target_trials`.

The node publishes operator status on `/gesture_trigger/status` and writes a
non-authoritative cache log to
`<dataset_root>/meta/gesture_trigger_progress.json`. On restart, it rebuilds
progress from `meta/episodes.jsonl` instead of trusting the progress log.

## Convert to Isaac-GR00T

The converter reads an existing Robo Collector dataset and writes a new
Isaac-GR00T-compatible dataset. It does not modify the source dataset in place.

```bash
python scripts/convert_outputs_to_gr00t.py \
  --source-root outputs \
  --dataset-name robo_collector_YYYYMMDD_HHMMSS \
  --dest-root exports \
  --output-name robo_collector_YYYYMMDD_HHMMSS_gr00t \
  --action-source aligned_target_pos
```

Arguments:

- `--source-root`: parent directory of the source dataset.
- `--dataset-name`: source dataset directory name.
- `--dest-root`: parent directory for converted datasets.
- `--output-name`: converted dataset directory name; defaults to
  `<dataset-name>_gr00t`.
- `--action-source`: source column for the single GR00T `action` vector. Choices
  are `aligned_target_pos`, `policy_action`, and `joint_position`.

The converter currently targets this project's split-field source schema:

- Reads state columns such as `observation.state.relative_ori_6d`.
- Reconstructs the single `observation.state` column required by GR00T.
- Rewrites the selected action source into a single `action` column.
- Copies videos to `videos/chunk-000/observation.images.<camera>/episode_*.mp4`.
- Generates GR00T-style `meta/modality.json`.

The script exits with an error if the source dataset lacks required state
columns or the selected action column.

## Convert to OpenPI pi0.5

The pi0.5 converter reads an existing Robo Collector dataset and writes a new
OpenPI-friendly LeRobot v2.1 dataset for the `pi05_g1_finetune` data path.

```bash
python scripts/convert_outputs_to_pi05.py \
  --source-root outputs \
  --dataset-name robo_collector_YYYYMMDD_HHMMSS \
  --dest-root exports \
  --output-name robo_collector_YYYYMMDD_HHMMSS_pi05
```

Arguments:

- `--source-root`: parent directory of the source dataset.
- `--dataset-name`: source dataset directory name.
- `--dest-root`: parent directory for converted datasets.
- `--output-name`: converted dataset directory name; defaults to
  `<dataset-name>_pi05`.
- `--state-key`: 29-dim source state column; defaults to
  `observation.state.joint_position`.
- `--action-key`: 29-dim source action column; defaults to
  `action.policy_action`.
- `--history-window-index`: window to extract when the selected vector column is
  a flat history vector whose length is a multiple of 29; defaults to `-1`.

The converter writes compact OpenPI keys `head_image`, `ego_image`, `state`,
`actions`, and `task_index`. Images are decoded from the source videos and
embedded as PNG-backed Hugging Face image columns in parquet, matching the
OpenPI G1 LeRobot training layout. It exits with an error if the source dataset
lacks the required head/ego camera streams or selected 29-dim vector columns.

## Acknowledgement

Robo Collector uses [StepIt](https://github.com/chengruiz/stepit) as the
teleoperation/control framework and as the source of robot state, policy, and
target topics consumed by the ROS 2 collection pipeline.

This project is also inspired by the dataset conventions and tooling from
[LeRobot](https://github.com/huggingface/lerobot) and
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T).

## License

This project is released under the [MIT License](LICENSE).
