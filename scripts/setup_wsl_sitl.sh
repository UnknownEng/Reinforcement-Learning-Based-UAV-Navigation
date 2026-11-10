#!/usr/bin/env bash
# Build the ROS package (symlink install, so source edits on the Windows drive
# take effect without rebuilding) and prepare the PX4 x500_lidar airframe.
#   bash scripts/setup_wsl_sitl.sh             # ROS package + PX4 build
#   bash scripts/setup_wsl_sitl.sh --ros-only  # ROS package only (fast)
set -eo pipefail

ROS_ONLY=0
[[ "${1:-}" == --ros-only ]] && ROS_ONLY=1

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/projects/PX4-Autopilot}"
ROS_BUILD_ROOT="${UAV_ROS_BUILD_ROOT:-$HOME/.cache/uav_autonomy/ros2}"
ROS_INSTALL_ROOT="${UAV_ROS_INSTALL_ROOT:-$HOME/.local/share/uav_autonomy/ros2_install}"
ROS_LOG_ROOT="${UAV_ROS_LOG_ROOT:-$HOME/.cache/uav_autonomy/ros2_log}"
AIRFRAME_SOURCE="$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes/4012_gz_x500_lidar"
AIRFRAME_TARGET="$PX4_DIR/build/px4_sitl_default/etc/init.d-posix/airframes/4012_gz_x500_lidar"

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy is not installed in this WSL distro." >&2
  exit 1
fi
if [[ ! -d "$PX4_DIR" ]]; then
  echo "PX4 checkout not found at $PX4_DIR." >&2
  exit 1
fi

source /opt/ros/jazzy/setup.bash
if [[ -f "$HOME/venvs/uav-sitl/bin/activate" ]]; then
  source "$HOME/venvs/uav-sitl/bin/activate"
fi

colcon --log-base "$ROS_LOG_ROOT" build \
  --symlink-install \
  --base-paths "$PROJECT_ROOT/ros2_ws/src" \
  --build-base "$ROS_BUILD_ROOT/build" \
  --install-base "$ROS_INSTALL_ROOT" \
  --packages-select uav_autonomy

if ((ROS_ONLY)); then
  echo "ROS package uav_autonomy is built (symlink install at $ROS_INSTALL_ROOT)."
  exit 0
fi

if [[ ! -f "$AIRFRAME_SOURCE" ]]; then
  sed \
    -e 's/@name Gazebo x500/@name Gazebo x500 with LiDAR/' \
    -e 's/PX4_SIM_MODEL=${PX4_SIM_MODEL:=x500}/PX4_SIM_MODEL=${PX4_SIM_MODEL:=x500_lidar}/' \
    "$PX4_DIR/ROMFS/px4fmu_common/init.d-posix/airframes/4001_gz_x500" \
    > "$AIRFRAME_SOURCE"
fi

(
  source "$HOME/venvs/uav-sitl/bin/activate"
  cd "$PX4_DIR"
  make -j4 px4_sitl_default
)
if [[ -d "$(dirname "$AIRFRAME_TARGET")" ]]; then
  install -m 0644 "$AIRFRAME_SOURCE" "$AIRFRAME_TARGET"
fi

echo "ROS package and x500_lidar airframe are ready. In each WSL shell, run: source $PROJECT_ROOT/scripts/setup_wsl_env.sh"
