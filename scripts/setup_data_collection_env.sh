#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv_data_collection"
WITH_LEROBOT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --with-lerobot)
      WITH_LEROBOT=1
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$(dirname "$0")/.."

create_venv() {
  local venv_dir="$1"

  if python3 -m venv --system-site-packages "$venv_dir" && [[ -f "$venv_dir/bin/activate" ]]; then
    return 0
  fi

  echo
  echo "python3 -m venv failed; falling back to user-space virtualenv."
  echo "This avoids requiring sudo/python3-venv on locked-down machines."
  rm -rf "$venv_dir"

  if python3 -m virtualenv --version >/dev/null 2>&1; then
    python3 -m virtualenv --system-site-packages "$venv_dir"
    [[ -f "$venv_dir/bin/activate" ]]
    return
  fi

  python3 -m pip install --user -U virtualenv
  python3 -m virtualenv --system-site-packages "$venv_dir"
  [[ -f "$venv_dir/bin/activate" ]]
}

if [[ ! -x "$VENV_DIR/bin/python" || ! -f "$VENV_DIR/bin/activate" ]]; then
  rm -rf "$VENV_DIR"
  create_venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install \
  opencv-python \
  numpy \
  pyzmq \
  msgpack \
  PyYAML \
  pyarrow

if [[ "$WITH_LEROBOT" == "1" ]]; then
  python -m pip install lerobot av
fi

python -m pip install -e src/camera

echo
echo "Data collection environment ready."
if [[ "$WITH_LEROBOT" != "1" ]]; then
  echo "Skipped optional lerobot install. Use --with-lerobot when you need the official loader/tools."
fi
echo "Activate with:"
echo "  source $(pwd)/$VENV_DIR/bin/activate"
