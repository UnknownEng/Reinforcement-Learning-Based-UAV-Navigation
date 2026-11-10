"""
run_m0_baseline.py
------------------
M0: the ROS 2 baseline flight (px4_mavlink_bridge + test_flight_baseline) in PX4 SITL,
judged by the corrected pass criteria (hover >= 1 m, >= 2 m forward flight, touchdown,
acknowledged disarm). Usage (WSL):  python -m uav_sitl.run_m0_baseline --runs 3
"""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from uav_sitl.sim_launcher import SimLauncher

REPORT = Path.home() / "proj/tests/telemetry/milestone1_flight_telemetry.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--out", default="logs/sitl_trials/m0_baseline.json")
    args = ap.parse_args()
    results = []
    for k in range(args.runs):
        sim = SimLauncher(world="default", model="x500_lidar", log_path=Path(f"logs/sitl_trials/sitl_m0_{k}.log"),
                          extra_env={"PX4_SIM_MODEL": "x500_lidar"})
        bridge = None
        rec = {"run": k}
        try:
            sim.start()
            if not sim.wait_ready(240):
                raise RuntimeError("sim_failed_to_start")
            if REPORT.exists():
                REPORT.unlink()
            bridge = subprocess.Popen(["ros2", "run", "uav_autonomy", "px4_mavlink_bridge"], start_new_session=True,
                                      stdout=open(f"logs/sitl_trials/m0_bridge_{k}.log", "w"), stderr=subprocess.STDOUT)
            time.sleep(5.0)
            t0 = time.monotonic()
            test = subprocess.run(["ros2", "run", "uav_autonomy", "test_flight_baseline"], capture_output=True,
                                  text=True, timeout=400)
            rec["wall_s"] = round(time.monotonic() - t0, 1)
            if REPORT.exists():
                data = json.load(open(REPORT))
                rec.update(result="PASS" if data["success"] else "FAIL",
                           metrics={k2: v for k2, v in data["metrics"].items() if k2 != "telemetry"})
            else:
                rec.update(result="no_report", tail=test.stdout[-800:])
        except subprocess.TimeoutExpired:
            rec["result"] = "test_timeout"
        except RuntimeError as exc:
            rec["result"] = str(exc)
        finally:
            if bridge is not None:
                os.killpg(bridge.pid, signal.SIGINT)
                try:
                    bridge.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(bridge.pid, signal.SIGKILL)
            sim.stop()
        results.append(rec)
        print(f"[m0] run {k + 1}/{args.runs}: {rec.get('result')}  {rec.get('metrics', {}).get('pass_checks')}", flush=True)
    json.dump(results, open(args.out, "w"), indent=1)
    print(f"[m0] {sum(r.get('result') == 'PASS' for r in results)}/{len(results)} PASS -> {args.out}")


if __name__ == "__main__":
    main()
