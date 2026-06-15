#!/usr/bin/env bash
set -euo pipefail

ros_root="${ROBO_COLLECTOR_ROS_ROOT:-/opt/ros}"

setup_for_distro() {
  local distro="$1"
  printf '%s/%s/setup.bash\n' "$ros_root" "$distro"
}

ubuntu_default_distro() {
  local os_release="${ROBO_COLLECTOR_OS_RELEASE:-/etc/os-release}"
  local version_id=""

  if [[ ! -f "$os_release" ]]; then
    return 1
  fi

  version_id="$(
    . "$os_release"
    printf '%s\n' "${VERSION_ID:-}"
  )"

  case "$version_id" in
    22.04) printf '%s\n' "humble" ;;
    24.04) printf '%s\n' "jazzy" ;;
    *) return 1 ;;
  esac
}

resolve_ros_setup() {
  local setup_path=""
  local distro=""
  local candidates=()
  local candidate=""

  if [[ -n "${ROS_SETUP_PATH:-}" ]]; then
    if [[ -f "$ROS_SETUP_PATH" ]]; then
      printf '%s\n' "$ROS_SETUP_PATH"
      return 0
    fi

    echo "ROS_SETUP_PATH points to a missing file: $ROS_SETUP_PATH" >&2
    return 1
  fi

  if [[ -n "${ROS_DISTRO:-}" ]]; then
    setup_path="$(setup_for_distro "$ROS_DISTRO")"
    if [[ -f "$setup_path" ]]; then
      printf '%s\n' "$setup_path"
      return 0
    fi

    echo "ROS_DISTRO is set to '$ROS_DISTRO', but $setup_path does not exist." >&2
    return 1
  fi

  if distro="$(ubuntu_default_distro)"; then
    setup_path="$(setup_for_distro "$distro")"
    if [[ -f "$setup_path" ]]; then
      printf '%s\n' "$setup_path"
      return 0
    fi

    echo "Ubuntu maps to ROS_DISTRO='$distro', but $setup_path does not exist." >&2
    return 1
  fi

  for candidate in "$ros_root"/*/setup.bash; do
    [[ -f "$candidate" ]] || continue
    candidates+=("$candidate")
  done

  if [[ "${#candidates[@]}" -eq 1 ]]; then
    printf '%s\n' "${candidates[0]}"
    return 0
  fi

  if [[ "${#candidates[@]}" -eq 0 ]]; then
    echo "Unable to find a ROS 2 setup script under $ros_root." >&2
  else
    echo "Multiple ROS 2 setup scripts were found under $ros_root." >&2
    printf '  %s\n' "${candidates[@]}" >&2
  fi

  echo "Export ROS_SETUP_PATH=/opt/ros/<distro>/setup.bash or ROS_DISTRO=<distro> before launching." >&2
  return 1
}

resolve_ros_setup
