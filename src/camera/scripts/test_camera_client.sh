#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f ".venv_camera/bin/activate" ]]; then
  source .venv_camera/bin/activate
fi

PYTHON_BIN="${PYTHON:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    PYTHON_BIN="python3"
  fi
fi

"$PYTHON_BIN" - "$@" <<'PY'
import argparse
import time

from robo_collector_camera.client import CameraClient

parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True)
parser.add_argument("--port", type=int, default=5555)
parser.add_argument("--timeout-ms", type=int, default=1000)
args = parser.parse_args()

client = CameraClient(args.host, args.port)
try:
    for _ in range(100):
        packet = client.read(timeout_ms=args.timeout_ms)
        if packet is None:
            print("No frame")
            continue
        print("schema:", packet.get("schema"))
        print("timestamps:", packet["timestamps"])
        print("metadata:", packet["metadata"])
        print("images:", {k: (v.shape, str(v.dtype)) for k, v in packet["images"].items()})
        break
        time.sleep(0.1)
    else:
        raise SystemExit("No camera frame received")
finally:
    client.close()
PY
