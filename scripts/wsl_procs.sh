#!/usr/bin/env bash
# List or stop the project's simulation processes inside WSL.
#   bash scripts/wsl_procs.sh status
#   bash scripts/wsl_procs.sh stop
# The patterns live in this file, not on the command line, so pgrep/pkill -f
# never match the shell running this script.
# Note: 'gz sim' matches any Gazebo instance in this distro, including ones
# started for other projects.
SIM_PATTERNS=(
  'build/px4_sitl_default/bin/px4'
  'gz sim'
  'ros2 launch uav_autonomy'
  'uav_autonomy/lib/uav_autonomy/'
)
WORK_PATTERN='python[0-9.]* .*(training/|experiments/|train_|evaluate|retrain)|python[0-9.]* -m pytest'

case "${1:-status}" in
  stop)
    found=0
    for pat in "${SIM_PATTERNS[@]}"; do
      if pkill -INT -f "$pat"; then found=1; echo "SIGINT  -> $pat"; fi
    done
    if ((found)); then
      for _ in $(seq 1 10); do
        alive=0
        for pat in "${SIM_PATTERNS[@]}"; do pgrep -f "$pat" >/dev/null && alive=1; done
        ((alive)) || break
        sleep 0.5
      done
      for pat in "${SIM_PATTERNS[@]}"; do
        pkill -KILL -f "$pat" && echo "SIGKILL -> $pat (did not exit on SIGINT)"
      done
      echo "simulation stopped"
    else
      echo "nothing was running"
    fi
    ;;
  status)
    echo "--- simulation processes"
    sims=""
    for pat in "${SIM_PATTERNS[@]}"; do sims+="$(pgrep -af "$pat")"$'\n'; done
    sims="$(sort -u <<<"$sims" | sed '/^$/d' | cut -c1-160)"
    echo "${sims:-(none)}"
    echo "--- training / evaluation / tests"
    work="$(pgrep -af "$WORK_PATTERN" | grep -v 'bash -c' | cut -c1-160)"
    echo "${work:-(none)}"
    echo "--- load"; uptime; free -h | sed -n 1,2p
    echo "--- recent logs"; ls -t logs/*.log 2>/dev/null | head -5 || echo "(none)"
    ;;
  pause-training|resume-training)
    # Freeze / continue training processes (e.g. so PX4 SITL gets the CPU to itself).
    sig=STOP; [[ "$1" == resume-training ]] && sig=CONT
    # Anchored at the start of the command line so wrapper shells ("bash -c ... python training/...")
    # that merely mention a training script are never frozen.
    pids="$(pgrep -f '^python[0-9.]* .*training/' || true)"
    if [[ -z "$pids" ]]; then echo "no training processes"; exit 0; fi
    kill -"$sig" $pids && echo "SIG$sig -> $(echo $pids | wc -w) training processes"
    ;;
  *)
    echo "usage: $0 status|stop|pause-training|resume-training" >&2; exit 2 ;;
esac
