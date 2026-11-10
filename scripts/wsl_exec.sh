#!/usr/bin/env bash
# Run one command inside WSL with the project environment loaded
# (ROS 2 Jazzy, ~/venvs/uav-sitl, project PYTHONPATH, Gazebo resource paths).
#
# Normally called from Windows by wsl.ps1. The command arrives in $UAV_CMD
# (forwarded through WSLENV so Windows quoting cannot mangle it) and is run by
# bash from the project root. Without $UAV_CMD, the positional arguments are
# executed as-is. If $UAV_LOG is set, output is also appended to that file.
set -o pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

if ! source "$PROJECT_ROOT/scripts/setup_wsl_env.sh" >/dev/null; then
  echo "[wsl] scripts/setup_wsl_env.sh failed; run '.\\wsl.ps1 check'." >&2
  exit 1
fi
export PYTHONUNBUFFERED=1

if [[ -n "${UAV_CMD:-}" ]]; then
  cmd=(bash -c "$UAV_CMD")
elif (($#)); then
  cmd=("$@")
else
  echo "[wsl] nothing to run" >&2
  exit 2
fi

if [[ -n "${UAV_LOG:-}" ]]; then
  mkdir -p "$(dirname -- "$UAV_LOG")"
  {
    echo "=== $(date -Is) ${UAV_CMD:-$*}"
    echo "=== git $(git rev-parse --short HEAD 2>/dev/null) on $(git branch --show-current 2>/dev/null)"
  } >>"$UAV_LOG"
  echo "[wsl] logging to $UAV_LOG"
  "${cmd[@]}" 2>&1 | tee -a "$UAV_LOG"
  status=${PIPESTATUS[0]}
  echo "=== exit $status at $(date -Is)" >>"$UAV_LOG"
  exit "$status"
fi
exec "${cmd[@]}"
