#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/setup_wsl_env.sh"

PX4_GZ_WORLD="${PX4_GZ_WORLD:-default}"
PX4_GZ_WORLD_SDF="${PX4_GZ_WORLD_SDF:-}"
PX4_ROOTFS="$PX4_DIR/build/px4_sitl_default/rootfs"
PX4_BINARY="$PX4_DIR/build/px4_sitl_default/bin/px4"
if [[ ! -x "$PX4_BINARY" ]]; then
  echo "PX4 SITL has not been built. Run bash scripts/setup_wsl_sitl.sh first." >&2
  exit 1
fi
cd "$PX4_ROOTFS"
source ./gz_env.sh
# Project worlds (e.g. worlds/m5_obstacle_world_1.sdf from
# scripts/generate_m5_sitl_worlds.py) take precedence over PX4's built-in ones.
if [[ -z "$PX4_GZ_WORLD_SDF" && -f "$PROJECT_ROOT/worlds/${PX4_GZ_WORLD}.sdf" ]]; then
  PX4_GZ_WORLD_SDF="$PROJECT_ROOT/worlds/${PX4_GZ_WORLD}.sdf"
fi
PX4_GZ_WORLD_SDF="${PX4_GZ_WORLD_SDF:-$PX4_GZ_WORLDS/${PX4_GZ_WORLD}.sdf}"
export PX4_SIM_MODEL="${PX4_SIM_MODEL:-x500_lidar}"
export PX4_SYS_AUTOSTART="${PX4_SYS_AUTOSTART:-4012}"
export PX4_GZ_WORLD
export PX4_GZ_STANDALONE=1

GZ_SIM_ARGS=(-r)
if [[ "${HEADLESS:-1}" != 0 ]]; then
  GZ_SIM_ARGS+=(-s)
fi
if [[ ! -f "$PX4_GZ_WORLD_SDF" ]]; then
  echo "Gazebo world file not found: $PX4_GZ_WORLD_SDF" >&2
  exit 1
fi
setsid gz sim "${GZ_SIM_ARGS[@]}" "$PX4_GZ_WORLD_SDF" &
GZ_PID=$!
cleanup() {
  trap - EXIT INT TERM
  if kill -0 -- "-$GZ_PID" 2>/dev/null; then
    kill -INT -- "-$GZ_PID" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 -- "-$GZ_PID" 2>/dev/null || break
      sleep 0.2
    done
    kill -TERM -- "-$GZ_PID" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$GZ_PID" 2>/dev/null || true
  fi
  wait "$GZ_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

ready=0
for _ in $(seq 1 60); do
  if gz service -l 2>/dev/null | grep -q "/world/${PX4_GZ_WORLD}/create"; then
    ready=1
    break
  fi
  sleep 0.5
done
if [[ "$ready" != 1 ]]; then
  echo "Gazebo did not expose world ${PX4_GZ_WORLD}." >&2
  exit 1
fi

echo "Starting PX4 SITL with model=$PX4_SIM_MODEL autostart=$PX4_SYS_AUTOSTART world=$PX4_GZ_WORLD ($PX4_GZ_WORLD_SDF)"
if [[ -t 0 ]]; then
  "$PX4_BINARY" -i 0
else
  # No terminal on stdin (launched from wsl.ps1, a background job or a test
  # harness): run without the pxh shell so PX4 does not exit on EOF.
  "$PX4_BINARY" -d -i 0
fi
