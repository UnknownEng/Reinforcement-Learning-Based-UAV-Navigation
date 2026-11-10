#!/usr/bin/env bash
# Source this file from Ubuntu WSL to set up the project ROS/Gazebo environment.
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PX4_DIR="${PX4_DIR:-$HOME/projects/PX4-Autopilot}"
ROS_INSTALL_ROOT="${UAV_ROS_INSTALL_ROOT:-$HOME/.local/share/uav_autonomy/ros2_install}"
PROJECT_LINK="$HOME/proj"

if [[ ! -e "$PROJECT_LINK" && ! -L "$PROJECT_LINK" ]]; then
  ln -s "$PROJECT_ROOT" "$PROJECT_LINK"
elif [[ "$(cd -- "$PROJECT_LINK" && pwd -P)" != "$(cd -- "$PROJECT_ROOT" && pwd -P)" ]]; then
  echo "$PROJECT_LINK already points to a different project; refusing to change it." >&2
  return 1 2>/dev/null || exit 1
fi

for pair in \
  "$PROJECT_ROOT/PX4-Autopilot:$PX4_DIR" \
  "$PROJECT_ROOT/venv:$HOME/venvs/uav-sitl"; do
  link="${pair%%:*}"
  target="${pair#*:}"
  if [[ ! -e "$link" && ! -L "$link" ]]; then
    ln -s "$target" "$link"
  elif [[ "$(cd -- "$link" && pwd -P)" != "$(cd -- "$target" && pwd -P)" ]]; then
    echo "$link already exists and does not point to $target; refusing to change it." >&2
    return 1 2>/dev/null || exit 1
  fi
done

source /opt/ros/jazzy/setup.bash
if [[ -f "$HOME/venvs/uav-sitl/bin/activate" ]]; then
  source "$HOME/venvs/uav-sitl/bin/activate"
fi
if [[ -f "$ROS_INSTALL_ROOT/setup.bash" ]]; then
  source "$ROS_INSTALL_ROOT/setup.bash"
fi

export PROJECT_ROOT PX4_DIR
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GZ_SIM_RESOURCE_PATH="$PROJECT_ROOT/models:$PX4_DIR/Tools/simulation/gz/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
printf 'Project: %s\nPX4: %s\nROS: %s\nGazebo: %s\n' "$PROJECT_ROOT" "$PX4_DIR" "$(command -v ros2)" "$(command -v gz)"
