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
FPS=""
REFERENCE_CAMERA_STREAM=""
CAMERA_STREAM_RATES_HZ=()
MAX_EPISODE_DURATION_SEC="600.0"
MAX_EPISODE_FRAMES="18000"
MIN_FREE_DISK_BYTES="2147483648"
MAX_CAMERA_CLOCK_MAPPING_UNCERTAINTY_SEC="0.05"
RECORDING_MODE="raw_v1"
RAW_EPISODE_ROOT=""
RAW_SOURCE_SCOPE="transport_observed"
CAMERA_RAW_SPOOL_ROOT=""
CAMERA_CALLBACK_QUEUE_SIZE="128"
PRINT_ROS_SETUP=0
PRINT_COLLECTOR_COMMAND=0

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
    --reference-camera-stream)
      REFERENCE_CAMERA_STREAM="$2"
      shift 2
      ;;
    --camera-stream-rate)
      CAMERA_STREAM_RATES_HZ+=("$2")
      shift 2
      ;;
    --max-episode-duration-sec)
      MAX_EPISODE_DURATION_SEC="$2"
      shift 2
      ;;
    --max-episode-frames)
      MAX_EPISODE_FRAMES="$2"
      shift 2
      ;;
    --min-free-disk-bytes)
      MIN_FREE_DISK_BYTES="$2"
      shift 2
      ;;
    --max-camera-clock-mapping-uncertainty-sec)
      MAX_CAMERA_CLOCK_MAPPING_UNCERTAINTY_SEC="$2"
      shift 2
      ;;
    --recording-mode)
      RECORDING_MODE="$2"
      shift 2
      ;;
    --raw-episode-root)
      RAW_EPISODE_ROOT="$2"
      shift 2
      ;;
    --raw-source-scope)
      RAW_SOURCE_SCOPE="$2"
      shift 2
      ;;
    --camera-raw-spool-root)
      CAMERA_RAW_SPOOL_ROOT="$2"
      shift 2
      ;;
    --camera-callback-queue-size)
      CAMERA_CALLBACK_QUEUE_SIZE="$2"
      shift 2
      ;;
    --print-ros-setup)
      PRINT_ROS_SETUP=1
      shift
      ;;
    --print-collector-command)
      PRINT_COLLECTOR_COMMAND=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."

# Phase 6/7 rollout contract: raw_v1 remains the safe default; dual_write is
# shadow/migration evidence and mcap_first is an opt-in release candidate.
# MCAP inspection/recovery/replay tools run after STOP, never in this hot path.
case "$RECORDING_MODE" in
  raw_v1|dual_write|mcap_first)
    ;;
  raw_first)
    echo "warning: recording mode 'raw_first' is deprecated; use 'raw_v1'" >&2
    ;;
  *)
    echo "Invalid --recording-mode '$RECORDING_MODE'; expected raw_v1, dual_write, mcap_first, or deprecated raw_first" >&2
    exit 2
    ;;
esac

CONFIGURED_CAMERA_STREAMS=()
if [[ -n "$CAMERA_STREAMS" ]]; then
  IFS=',' read -r -a CONFIGURED_CAMERA_STREAMS <<< "$CAMERA_STREAMS"
else
  CONFIGURED_CAMERA_STREAMS+=("$CAMERA_STREAM")
fi
for index in "${!CONFIGURED_CAMERA_STREAMS[@]}"; do
  stream="${CONFIGURED_CAMERA_STREAMS[$index]}"
  stream="${stream#"${stream%%[![:space:]]*}"}"
  stream="${stream%"${stream##*[![:space:]]}"}"
  if [[ -z "$stream" ]]; then
    echo "Configured camera streams must not contain empty entries" >&2
    exit 2
  fi
  CONFIGURED_CAMERA_STREAMS[index]="$stream"
done

if [[ -z "$REFERENCE_CAMERA_STREAM" ]]; then
  REFERENCE_CAMERA_STREAM="${CONFIGURED_CAMERA_STREAMS[0]}"
fi
reference_is_configured=0
for stream in "${CONFIGURED_CAMERA_STREAMS[@]}"; do
  if [[ "$stream" == "$REFERENCE_CAMERA_STREAM" ]]; then
    reference_is_configured=1
  fi
done
if [[ "$reference_is_configured" != "1" ]]; then
  echo "Reference camera stream '$REFERENCE_CAMERA_STREAM' is not in the configured stream set" >&2
  exit 2
fi

declare -A SEEN_CAMERA_RATES=()
for entry in "${CAMERA_STREAM_RATES_HZ[@]}"; do
  if [[ ! "$entry" =~ ^([A-Za-z0-9_.-]+)=([0-9]+([.][0-9]+)?)$ ]]; then
    echo "Invalid --camera-stream-rate '$entry'; expected STREAM=POSITIVE_HZ" >&2
    exit 2
  fi
  stream="${BASH_REMATCH[1]}"
  rate_hz="${BASH_REMATCH[2]}"
  if [[ "$rate_hz" =~ ^0+([.]0+)?$ ]]; then
    echo "Invalid --camera-stream-rate '$entry'; expected STREAM=POSITIVE_HZ" >&2
    exit 2
  fi
  if [[ -n "${SEEN_CAMERA_RATES[$stream]:-}" ]]; then
    echo "Duplicate --camera-stream-rate for '$stream'" >&2
    exit 2
  fi
  SEEN_CAMERA_RATES[$stream]=1
done

if [[ ${#CAMERA_STREAM_RATES_HZ[@]} -gt 0 ]]; then
  for stream in "${CONFIGURED_CAMERA_STREAMS[@]}"; do
    if [[ -z "${SEEN_CAMERA_RATES[$stream]:-}" ]]; then
      echo "Missing --camera-stream-rate for configured stream '$stream'" >&2
      exit 2
    fi
  done
  for stream in "${!SEEN_CAMERA_RATES[@]}"; do
    configured=0
    for candidate in "${CONFIGURED_CAMERA_STREAMS[@]}"; do
      if [[ "$candidate" == "$stream" ]]; then
        configured=1
      fi
    done
    if [[ "$configured" != "1" ]]; then
      echo "Camera rate provided for unconfigured stream '$stream'" >&2
      exit 2
    fi
  done
fi

if [[ -n "$FPS" ]]; then
  if [[ ! "$FPS" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "$FPS" =~ ^0+([.]0+)?$ ]]; then
    echo "Invalid --fps '$FPS'; expected a positive legacy fallback rate" >&2
    exit 2
  fi
  echo "warning: --fps is a legacy fallback; prefer one --camera-stream-rate STREAM=HZ per configured stream" >&2
fi

CAMERA_STREAM_RATES_VALUE=""
if [[ ${#CAMERA_STREAM_RATES_HZ[@]} -gt 0 ]]; then
  CAMERA_STREAM_RATES_VALUE="$(IFS=,; printf '%s' "${CAMERA_STREAM_RATES_HZ[*]}")"
fi

COLLECTOR_ARGS=(
  "--ros-args"
  "-p" "camera_host:=${CAMERA_HOST}"
  "-p" "camera_port:=${CAMERA_PORT}"
  "-p" "root_output_dir:=${ROOT_OUTPUT_DIR}"
  "-p" "reference_camera_stream:=${REFERENCE_CAMERA_STREAM}"
  "-p" "max_episode_duration_sec:=${MAX_EPISODE_DURATION_SEC}"
  "-p" "max_episode_frames:=${MAX_EPISODE_FRAMES}"
  "-p" "min_free_disk_bytes:=${MIN_FREE_DISK_BYTES}"
  "-p" "max_camera_clock_mapping_uncertainty_sec:=${MAX_CAMERA_CLOCK_MAPPING_UNCERTAINTY_SEC}"
  "-p" "recording_mode:=${RECORDING_MODE}"
  "-p" "raw_source_scope:=${RAW_SOURCE_SCOPE}"
  "-p" "camera_raw_spool_root:=${CAMERA_RAW_SPOOL_ROOT}"
  "-p" "camera_callback_queue_size:=${CAMERA_CALLBACK_QUEUE_SIZE}"
)

if [[ -n "$CAMERA_STREAM_RATES_VALUE" ]]; then
  COLLECTOR_ARGS+=("-p" "camera_stream_rates_hz:=${CAMERA_STREAM_RATES_VALUE}")
fi

if [[ -n "$FPS" ]]; then
  COLLECTOR_ARGS+=("-p" "fps:=${FPS}")
fi

if [[ -n "$RAW_EPISODE_ROOT" ]]; then
  COLLECTOR_ARGS+=("-p" "raw_episode_root:=${RAW_EPISODE_ROOT}")
fi

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

if [[ "$PRINT_COLLECTOR_COMMAND" == "1" ]]; then
  printf '%s\n' "$COLLECTOR_CMD"
  exit 0
fi

resolve_ros_setup() {
  bash scripts/resolve_ros_setup.sh
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
