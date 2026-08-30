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
| metadata | Fixed-rate timeline, source timestamps, camera references, episode/frame indices, and task metadata |

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
ros2 interface show robo_state_msgs/msg/RoboStateSample | grep source_timestamp_names
ros2 interface show robo_state_msgs/msg/PolicyState | grep flattened
ros2 interface show robo_collector_msgs/msg/RecordCommand | grep -E 'command_id|force'
```

Expected output:

```text
float32[45] aligned_target_pos
string[] source_timestamp_names
float32[1110] flattened
string command_id
bool force
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

To prepare complete-source capture, enable the camera-side bounded Raw spool.
It is written before latest-frame aggregation and ZMQ publication, so it can be
recovered independently after a server restart:

```bash
bash scripts/run_realsense_server.sh \
  --camera head:<D405_SERIAL> \
  --camera ego_view:<D435I_SERIAL> \
  --raw-spool-dir /data/robo_collector/camera_spools \
  --packet-schema v3
```

`v3` remains the default wire protocol; `v4` is an explicit opt-in after the
normalized-envelope path has been validated. A host Raw Episode made from
received ZMQ packets is still `transport_observed`; it does not recover frames
that were lost before reception. To use the spool as the task's complete
camera source, mount the camera spool directory on the collection host and
pass `--camera-raw-spool-root` together with
`--raw-source-scope camera_capture`. The collector snapshots source records in
the START/STOP server-wall-time window and imports them into the task Raw
Episode. Without that mount/parameter, `camera_capture` is explicitly
downgraded to `transport_observed`.

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
  --reference-camera-stream head \
  --camera-stream-rate head=30 \
  --camera-stream-rate ego_view=30 \
  --root-output-dir outputs \
  --max-episode-duration-sec 600 \
  --max-episode-frames 18000 \
  --min-free-disk-bytes 2147483648 \
  --max-camera-clock-mapping-uncertainty-sec 0.05 \
  --recording-mode raw_v1 \
  --raw-episode-root outputs/.raw_episodes \
  --raw-source-scope transport_observed \
  --camera-callback-queue-size 128
```

`reference_camera_stream` selects the RGB timeline used for aligned output.
Supply one `--camera-stream-rate STREAM=HZ` for every configured stream when
the acquisition rates are known; the values are configuration evidence, not a
substitute for timestamps or measured counts. The robot observation rate is
measured independently from received state callbacks and is never inferred
from a camera rate. The old `--fps HZ` flag remains an opt-in compatibility
fallback for deployments that have not migrated, but the launch script no
longer assumes either 30 Hz camera or 50 Hz robot data.

`raw_v1` writes bounded Raw v1 records during `RECORDING`; MP4/Parquet are
materialized after `STOP`, then checked by the Episode quality gate. The
deprecated `raw_first` spelling is accepted as an alias for `raw_v1` during
rollout and emits a warning. The host collector intentionally keeps
`transport_observed` until the camera-side spool is linked to the same task
Episode; complete-source capture must not be claimed from a receiver-only
recording.

Capture modes are explicit:

| Mode | Required capture sinks | Save/publication rule |
| --- | --- | --- |
| `raw_v1` | Raw v1 | Current rollout default; materialize and quality-check after STOP. |
| `dual_write` | Raw v1 and landing MCAP | Migration evidence only; any sink fault quarantines the episode and cannot report save success. |
| `mcap_first` | landing MCAP | Opt-in release candidate; seal and validate MCAP before any ready publication. |

`mcap_first` is **not** the default. Promote it only after the release gate
below passes on the deployment hardware.

MCAP candidates use the same side-effect-free `EpisodeQualityGate` entry point
as Raw v1 manifests. When a manifest includes `canonical` (or
`canonical_artifacts`) evidence, the gate validates both immutable
`camera_mcap` and `robot_mcap` files with the internal structural validator,
checks the recorded SHA-256 values, and consumes `content_qc` status. A
canonical `READY` candidate is rejected unless both modality groups and
content QC pass; structural failures are fail-closed and never repair or
rewrite a sealed MCAP. Raw v1 manifests without canonical evidence retain
their existing validation path.

For a mounted camera spool, use these raw options instead:

```bash
  --raw-source-scope camera_capture \
  --camera-raw-spool-root /data/robo_collector/camera_spools
```

The raw manifest records source session(s), source manifest hash, capture
window, imported frame counts, and the measured source/collector clock mapping
uncertainty. The threshold is configurable with
`--max-camera-clock-mapping-uncertainty-sec`; missing or exceeded mapping
evidence stays `REVIEW`. A source with `REVIEW` or `REJECT` QC, or a
transport-only source when complete capture is required, is not accepted by
the GR00T/OpenPI raw-input path.

### Phase 6/7 MCAP operations (local, fail-closed)

The repository also provides an operator CLI for inspecting and validating an
Episode without changing the sealed source artifacts:

```bash
python -m robo_collector.mcap_tool info <mcap-file>
python -m robo_collector.mcap_tool doctor <mcap-file> --group camera
python -m robo_collector.mcap_tool recover --attempt 1 <episode-dir>
python -m robo_collector.mcap_tool replay <manifest.json> --format json
python -m robo_collector.mcap_tool migration <manifest.json>
python -m robo_collector.mcap_tool benchmark <mcap-file> --iterations 1
```

Run these commands from an activated workspace (or set `PYTHONPATH` to the
Python package). Use `--group robot` for the robot stream. `info` and `doctor` are read-only; `recover` writes only an
explicitly named recovery artifact and never replaces a sealed `*.mcap`.
Replay and migration preserve source ordering and report parity differences
instead of silently repairing them. Commands fail closed for missing paths,
non-`READY` manifests, checksum failures, or incomplete source evidence.
Benchmark output is diagnostic only and is not a hardware acceptance result.

For shadow rollout, run the same Episode through Raw v1 and MCAP and compare
accepted frontiers, per-stream sequences, timestamps, selected frame counts,
manifest/config hashes, and terminal quality state. A parity report is
evidence for promotion, not a replacement for the process-kill, reconnect, and
soak gates below. The launch script keeps `raw_v1` as the default; select
`dual_write` for migration evidence and `mcap_first` only after these gates
pass.

### Recovery and acceptance evidence

`raw_v1` is the rollout default recording mode; `raw_first` is only its
deprecated compatibility alias. At startup the
collector scans `<raw-episode-root>` and retries pending or partially committed
materialization jobs. Until that scan completes, `START` is rejected. If the
scan or recovery coordinator fails, the node stays fail-closed and publishes an
`ERROR` status; repair the raw root and restart the node rather than deleting
the raw Episode.

Every sealed Episode must have `manifest.json`, `checksums.json`,
`quality.json`, and a durable materialization job record. The expected commit
order is `RAW_CLOSED -> MATERIALIZING -> MATERIALIZED -> QC ->
READY/REVIEW/REJECT`; only `READY` enters the default training index. A crash
during QC or publication is recovered by revalidating artifact hashes and
quality evidence, so `READY` must never be inferred from an MP4/Parquet file
alone.

For each deployment acceptance run, archive the command-line configuration,
camera serials, source/collector clock evidence, code commit, start/end time,
queue-depth and disk/RSS/CPU status, deadline-miss counters, materialization
job history, and the results of the dual-camera reconnect, process-kill,
10 x 60-second, and 60-minute soak tests. These measurements are deployment
evidence; the unit tests do not replace them.

Before promoting `mcap_first`, run the same capture once in `dual_write` and
verify all of the following against one episode ID:

1. Raw v1 and MCAP terminal accepted frontiers match, with no missing or
   unclassified per-sink disposition.
2. Every camera source fence is bound to one spool generation/session and has
   START/STOP evidence plus accepted, written, and durable high-watermarks.
3. The sealed MCAP validates against its expected inventory and source fences;
   Raw materialization and MCAP-derived output agree on selected RGB/state
   counts and reference stream.
4. Injected Raw-sink and MCAP-sink failures both quarantine `dual_write`; a
   fault must never leave a READY pointer or saved-success status.
5. Restart, reconnect, disk-full/write-failure, process-kill, soak, and normal
   STOP runs retain their manifests, checksums, quality results, and operator
   command line. Hardware results remain release evidence outside the unit
   suite.

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

An empty `episode_id` keeps direct terminal control convenient and targets the
currently active episode. Automated controllers should send the `episode_id`
returned in collector status; mismatched STOP/DISCARD commands are rejected.
They should also reuse one stable `command_id` across retries and confirm the
matching `last_command_*` status fields. `force: true` is reserved for an
explicit operator override. If a duration, frame-count, or free-disk limit is
reached, the collector fail-closes with `SAFETY DISCARD`; it never saves a
known-incomplete episode.

State/camera admission uses local monotonic receive-time skew, so robot and
camera hosts do not need synchronized wall clocks for the gate. Original state
and camera source timestamps are still stored as float64 audit columns;
inter-camera skew is checked in the camera server's shared clock domain.
This is a best-effort receive-time gate, not a hard bound on physical capture
alignment: server encoding and network delay are outside that bound. Experiments
that require capture-time guarantees should add hardware/PTP synchronization and
validate the stored source timestamps with a measured clock-offset budget.

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

Datasets created before `robo_collector_schema_version: 1` used different
timeline semantics and are intentionally rejected for append. Start a new
`dataset_name`; no automatic in-place migration is provided because inventing
source timestamps for old rows would silently corrupt alignment provenance.

Protocol fields in `RecordCommand`, `RoboStateSample`, and camera packet v3 must
be deployed together. After updating, clean-rebuild the ROS workspace and
restart both the camera server and collector; mixed message hashes or a v2
camera server are not wire-compatible.

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
- `collector.stop_confirm_timeout_sec`: STOP acknowledgement deadline; STOP is
  retried idempotently for the current episode.
- `collector.save_confirm_timeout_sec`: rolling save watchdog. Only a newer
  collector `save_progress_seq` renews this deadline, so repeated status messages
  from a stuck save cannot keep the attempt alive indefinitely. Configure it
  above the longest expected video-close, validation, fsync, rename, or metadata
  transaction phase because those system calls cannot report intermediate
  progress.
- `collector.max_save_wait_sec`: maximum time the gesture controller waits for
  metadata reconciliation, even while progress continues. It does not cancel an
  in-flight durable writer transaction.
- `collector.discard_confirm_timeout_sec`: fail-closed DISCARD acknowledgement
  deadline; DISCARD is retried for the owned episode before the trigger pauses.
- `collector.max_recording_duration_sec`: gesture-side fail-closed watchdog; an
  incomplete attempt is discarded if it runs too long or gesture samples go
  stale.
- `gesture_source`: which vector to read from `/robo_state/sample`.
- `references` and `conditions`: calibration targets plus trigger thresholds.
- `tasks`: one or more prompts, each with its own `target_trials`.

The node publishes operator status on `/gesture_trigger/status` and writes a
non-authoritative cache log to
`<dataset_root>/meta/gesture_trigger_progress.json`. On restart, it rebuilds
progress from `meta/episodes.jsonl` instead of trusting the progress log.
New episode rows include SHA-256/size integrity records. Recovery verifies those
records before marking a gesture trial complete; dataset metadata updates use a
durable transaction journal, and only one writer may own a dataset at a time.
Episode finalization runs on one background worker. Collector status remains
responsive in `SAVING` mode and exposes `save_phase`, `save_elapsed_sec`, and
`save_progress_seq`/`save_progress_age_sec`. `DISCARD` is rejected while that
durable commit is in flight; a failed save transitions to `FAILED`, where
episode-scoped `DISCARD` remains available. On node shutdown, the collector waits
for `save_shutdown_grace_sec` (default `10.0`) before warning, then continues to
wait for an in-flight save so the writer cannot outlive the node. If the process
is externally terminated, staging data remains available for startup recovery.

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

Both converters preserve per-Episode source provenance. The aggregate index is
`meta/raw_provenance.json`; immutable raw-materialization entries are also
written under `meta/raw_provenance/<source_episode_id>.json`. Converters reject
raw Episodes that are not QC `READY`.

## Acknowledgement

Robo Collector uses [StepIt](https://github.com/chengruiz/stepit) as the
teleoperation/control framework and as the source of robot state, policy, and
target topics consumed by the ROS 2 collection pipeline.

This project is also inspired by the dataset conventions and tooling from
[LeRobot](https://github.com/huggingface/lerobot) and
[Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T).

## License

This project is released under the [MIT License](LICENSE).
