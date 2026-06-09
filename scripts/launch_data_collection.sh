#!/usr/bin/env bash
set -euo pipefail

SESSION="robo_data_collection"
CAMERA_HOST="192.168.123.164"
CAMERA_PORT="5555"
CAMERA_STREAM=""
CAMERA_STREAMS="head,ego_view"
ROOT_OUTPUT_DIR="outputs"
DATASET_NAME=""
FIELD_CONFIG=""
FPS="50"
PRINT_ROS_SETUP=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session)
      SESSION="$2"
      shift 2
      ;;
    --camera-host)
      CAMERA_HOST="$2"
      shift 2
      ;;
    --camera-port)
      CAMERA_PORT="$2"
      shift 2
      ;;
    --camera-stream)
      CAMERA_STREAM="$2"
      CAMERA_STREAMS=""
      shift 2
      ;;
    --camera-streams)
      CAMERA_STREAMS="$2"
      shift 2
      ;;
    --root-output-dir)
      ROOT_OUTPUT_DIR="$2"
      shift 2
      ;;
    --dataset-name)
      DATASET_NAME="$2"
      shift 2
      ;;
    --field-config)
      FIELD_CONFIG="$2"
      shift 2
      ;;
    --fps)
      FPS="$2"
      shift 2
      ;;
    --print-ros-setup)
      PRINT_ROS_SETUP=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."

resolve_ros_setup() {
  local ros_root="${ROBO_COLLECTOR_ROS_ROOT:-/opt/ros}"
  local setup_path=""
  local candidates=()
  local candidate=""

  if [[ -n "${ROS_SETUP_PATH:-}" ]]; then
    if [[ -f "${ROS_SETUP_PATH}" ]]; then
      printf '%s\n' "${ROS_SETUP_PATH}"
      return 0
    fi

    echo "ROS_SETUP_PATH points to a missing file: ${ROS_SETUP_PATH}" >&2
    return 1
  fi

  if [[ -n "${ROS_DISTRO:-}" ]]; then
    setup_path="${ros_root}/${ROS_DISTRO}/setup.bash"
    if [[ -f "${setup_path}" ]]; then
      printf '%s\n' "${setup_path}"
      return 0
    fi

    echo "ROS_DISTRO is set to '${ROS_DISTRO}', but ${setup_path} does not exist." >&2
    return 1
  fi

  for candidate in "${ros_root}"/*/setup.bash; do
    [[ -f "${candidate}" ]] || continue
    candidates+=("${candidate}")
  done

  if [[ "${#candidates[@]}" -eq 1 ]]; then
    printf '%s\n' "${candidates[0]}"
    return 0
  fi

  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "Unable to find a ROS 2 setup script under ${ros_root}." >&2
  else
    echo "Multiple ROS 2 setup scripts were found under ${ros_root}." >&2
    printf '  %s\n' "${candidates[@]}" >&2
  fi

  echo "Export ROS_SETUP_PATH=/opt/ros/<distro>/setup.bash or ROS_DISTRO=<distro> before launching." >&2
  return 1
}

ROS_SETUP="$(resolve_ros_setup)"

if [[ "$PRINT_ROS_SETUP" == "1" ]]; then
  printf '%s\n' "$ROS_SETUP"
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for launch_data_collection.sh" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists" >&2
  exit 1
fi

COMMON='cd '"$(printf '%q' "$(pwd)")"'; source '"$(printf '%q' "${ROS_SETUP}")"'; if [[ -f install/setup.bash ]]; then source install/setup.bash; fi; if [[ -f .venv_data_collection/bin/activate ]]; then source .venv_data_collection/bin/activate; fi'

COLLECTOR_ARGS=(
  "--ros-args"
  "-p" "camera_host:=${CAMERA_HOST}"
  "-p" "camera_port:=${CAMERA_PORT}"
  "-p" "root_output_dir:=${ROOT_OUTPUT_DIR}"
  "-p" "fps:=${FPS}"
)

if [[ -n "$CAMERA_STREAMS" ]]; then
  COLLECTOR_ARGS+=("-p" "camera_streams:=${CAMERA_STREAMS}")
else
  COLLECTOR_ARGS+=("-p" "camera_stream:=${CAMERA_STREAM}")
fi

if [[ -n "$DATASET_NAME" ]]; then
  COLLECTOR_ARGS+=("-p" "dataset_name:=${DATASET_NAME}")
fi

if [[ -n "$FIELD_CONFIG" ]]; then
  COLLECTOR_ARGS+=("-p" "field_config_path:=${FIELD_CONFIG}")
fi

quote_args() {
  printf ' %q' "$@"
}

COLLECTOR_CMD="ros2 run robo_collector lerobot_collector_node$(quote_args "${COLLECTOR_ARGS[@]}")"
VIEWER_CMD="src/camera/scripts/run_camera_viewer.sh$(quote_args --host "${CAMERA_HOST}" --port "${CAMERA_PORT}")"

STATE_PANE=$(
  tmux new-session -d -s "$SESSION" -n collector -P -F "#{pane_id}" \
    "bash -lc '$COMMON; ros2 run robo_state robo_state_node'"
)
tmux set-option -t "$SESSION" remain-on-exit on >/dev/null
COLLECTOR_PANE=$(
  tmux split-window -t "$STATE_PANE" -h -P -F "#{pane_id}" \
    "bash -lc '$COMMON; $COLLECTOR_CMD'"
)
tmux split-window -t "$COLLECTOR_PANE" -v \
  "bash -lc '$COMMON; $VIEWER_CMD'"
tmux select-layout -t "$SESSION:collector" tiled >/dev/null

echo "Using ROS setup: $ROS_SETUP"
echo "Started tmux session '$SESSION'. Attach with:"
echo "  tmux attach -t $SESSION"
echo "Collector pane:"
echo "  tmux capture-pane -t '$COLLECTOR_PANE' -p -S -160"
echo
echo "Manual START example:"
echo "  ros2 topic pub --once /robo_collector/record_command robo_collector_msgs/msg/RecordCommand \"{command: 1, task_prompt: 'your task prompt'}\""
echo "Manual STOP example:"
echo "  ros2 topic pub --once /robo_collector/record_command robo_collector_msgs/msg/RecordCommand \"{command: 2}\""
