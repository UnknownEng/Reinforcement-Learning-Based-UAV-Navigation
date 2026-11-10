#!/usr/bin/env bash
# Health check of the WSL simulation toolchain. Run via `.\wsl.ps1 check`
# (which has already sourced setup_wsl_env.sh). Read-only.
fail=0
ok()   { printf '  [ok]   %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; fail=1; }

echo "WSL: $(lsb_release -ds 2>/dev/null), $(nproc) CPUs, $(free -g | awk '/Mem:/{print $2}') GB RAM, user $(whoami)"
echo "Project: $PROJECT_ROOT (branch $(git branch --show-current 2>/dev/null))"

[[ "${ROS_DISTRO:-}" == jazzy ]] && ok "ROS 2 $ROS_DISTRO" || bad "ROS 2 Jazzy not sourced"
command -v gz >/dev/null && ok "Gazebo Sim $(gz sim --versions 2>/dev/null | head -1)" || bad "gz not found"

px4_bin="$PX4_DIR/build/px4_sitl_default/bin/px4"
if [[ -x "$px4_bin" ]]; then
  ok "PX4 $(git -C "$PX4_DIR" describe --tags 2>/dev/null) built at $PX4_DIR"
else
  bad "PX4 SITL not built ($px4_bin); run: .\\wsl.ps1 run \"bash scripts/setup_wsl_sitl.sh\""
fi
airframe="$PX4_DIR/build/px4_sitl_default/etc/init.d-posix/airframes/4012_gz_x500_lidar"
[[ -f "$airframe" ]] && ok "x500_lidar airframe 4012 installed" || bad "airframe 4012_gz_x500_lidar missing; run setup_wsl_sitl.sh"
[[ -d "$PROJECT_ROOT/models/x500_lidar" ]] && ok "x500_lidar Gazebo model in project models/" || bad "models/x500_lidar missing"

[[ "$(readlink -f "$HOME/proj")" == "$(readlink -f "$PROJECT_ROOT")" ]] \
  && ok "~/proj -> $PROJECT_ROOT (default paths of the ROS nodes resolve)" \
  || bad "~/proj does not point at this checkout"

if [[ "$(command -v python)" == "$HOME/venvs/uav-sitl/"* ]]; then
  ok "venv $(python --version 2>&1) at ~/venvs/uav-sitl"
else
  bad "venv ~/venvs/uav-sitl not active (python=$(command -v python))"
fi
python - <<'PY' || fail=1
import importlib
mods = ["numpy", "gymnasium", "torch", "stable_baselines3", "pymavlink", "rclpy"]
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  [ok]   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"  [FAIL] {m}: {e}")
        raise SystemExit(1)
import gymnasium as gym, rl_env  # noqa: F401
envs = sorted(k for k in gym.registry if k.startswith("UAVFastNav"))
print(f"  [ok]   rl_env imported from {rl_env.__file__.rsplit('/', 2)[0]} -> {', '.join(envs)}")
PY

if prefix="$(ros2 pkg prefix uav_autonomy 2>/dev/null)"; then
  ok "ROS package uav_autonomy at $prefix"
  exes="$(ros2 pkg executables uav_autonomy 2>/dev/null | awk '{print $2}')"
  missing=""
  for e in $(sed -n "s/^ *'\([a-z0-9_]*\) = .*/\1/p" "$PROJECT_ROOT/ros2_ws/src/uav_autonomy/setup.py"); do
    grep -qx "$e" <<<"$exes" || missing+=" $e"
  done
  [[ -z "$missing" ]] && ok "all setup.py entry points installed" \
    || bad "installed package is stale, missing:$missing -> run: .\\wsl.ps1 build"
  if [[ -L "$prefix/share/uav_autonomy/launch/m5_sitl.launch.py" ]]; then
    ok "symlink install: edits to ROS launch/config/python apply without rebuild"
  else
    warn "copy install: rebuild (.\\wsl.ps1 build) after every ROS package edit"
  fi
else
  bad "ROS package uav_autonomy not built; run: .\\wsl.ps1 build"
fi

n_worlds=$(ls "$PROJECT_ROOT"/worlds/m5_obstacle_world_*.sdf 2>/dev/null | wc -l)
((n_worlds == 3)) && ok "M5 SITL worlds generated (worlds/)" \
  || warn "M5 SITL worlds not generated; run: .\\wsl.ps1 run \"python scripts/generate_m5_sitl_worlds.py\""

if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  gpu="$(nvidia-smi -L | head -1 | cut -d'(' -f1)"
  python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null \
    && ok "$gpu usable by torch" || warn "$gpu present, but torch is CPU-only (fine for 64x64 MLP PPO)"
fi

running="$(pgrep -af 'bin/px4|gz sim' | grep -v pgrep)"
[[ -n "$running" ]] && warn "simulator already running:"$'\n'"$running"

((fail)) && { echo "Toolchain check: FAILED"; exit 1; }
echo "Toolchain check: OK"
