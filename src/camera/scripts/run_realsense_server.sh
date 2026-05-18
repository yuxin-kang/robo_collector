#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f ".venv_camera/bin/activate" ]]; then
  source .venv_camera/bin/activate
fi

python -m robo_collector_camera.server_realsense "$@"

