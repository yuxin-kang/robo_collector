#!/usr/bin/env bash
set -euo pipefail

SESSION="robo_data_collection"
CAMERA_HOST="192.168.123.164"
CAMERA_PORT="5555"
CAMERA_STREAM="ego_view"
ROOT_OUTPUT_DIR="outputs"
DATASET_NAME=""
FPS="50"

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
    --fps)
      FPS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required for launch_data_collection.sh" >&2
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists" >&2
  exit 1
fi

COMMON='cd '"$(printf '%q' "$(pwd)")"'; if [[ -f /opt/ros/humble/setup.bash ]]; then source /opt/ros/humble/setup.bash; fi; if [[ -f install/setup.bash ]]; then source install/setup.bash; fi; if [[ -f .venv_data_collection/bin/activate ]]; then source .venv_data_collection/bin/activate; fi'

COLLECTOR_ARGS=(
  "--ros-args"
  "-p" "camera_host:=${CAMERA_HOST}"
  "-p" "camera_port:=${CAMERA_PORT}"
  "-p" "camera_stream:=${CAMERA_STREAM}"
  "-p" "root_output_dir:=${ROOT_OUTPUT_DIR}"
  "-p" "fps:=${FPS}"
)

if [[ -n "$DATASET_NAME" ]]; then
  COLLECTOR_ARGS+=("-p" "dataset_name:=${DATASET_NAME}")
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

echo "Started tmux session '$SESSION'. Attach with:"
echo "  tmux attach -t $SESSION"
echo "Collector pane:"
echo "  tmux capture-pane -t '$COLLECTOR_PANE' -p -S -160"
echo
echo "Manual START example:"
echo "  ros2 topic pub --once /robo_collector/record_command robo_collector_msgs/msg/RecordCommand \"{command: 1, task_prompt: 'your task prompt'}\""
echo "Manual STOP example:"
echo "  ros2 topic pub --once /robo_collector/record_command robo_collector_msgs/msg/RecordCommand \"{command: 2}\""
