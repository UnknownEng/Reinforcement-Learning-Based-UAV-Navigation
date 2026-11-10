"""
px4_step_response.py
--------------------
M1 calibration data: fly the PX4 x500 (SITL) through a fixed sequence of
heading-frame velocity steps and record the response at ~50 Hz.

Usage (WSL, project env):  python -m uav_sitl.px4_step_response --out logs/px4_step_response.json
"""

import argparse
import json
import time
from pathlib import Path

from uav_sitl.gz_link import GzLink
from uav_sitl.px4_link import PX4Link
from uav_sitl.sim_launcher import SimLauncher

# (duration_s, vx, vy, vz, yaw_rate) in the heading frame (FLU)
SEQUENCE = [
    (3.0, 0.0, 0.0, 0.0, 0.0),
    (4.0, 1.0, 0.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0, 0.0),
    (5.0, 3.0, 0.0, 0.0, 0.0), (5.0, 0.0, 0.0, 0.0, 0.0),
    (4.0, 0.0, 2.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0, 0.0),
    (4.0, -2.0, 0.0, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0, 0.0),
    (3.0, 0.0, 0.0, 1.0, 0.0), (3.0, 0.0, 0.0, 0.0, 0.0),
    (3.0, 0.0, 0.0, -1.0, 0.0), (3.0, 0.0, 0.0, 0.0, 0.0),
    (4.0, 0.0, 0.0, 0.0, 0.8), (2.0, 0.0, 0.0, 0.0, 0.0),
    (5.0, 2.0, 0.0, 0.0, 0.5), (4.0, 0.0, 0.0, 0.0, 0.0),
    (4.0, 2.1, 2.1, 0.0, 0.0), (4.0, 0.0, 0.0, 0.0, 0.0),
    (3.0, 3.0, -1.5, 0.5, -0.6), (4.0, 0.0, 0.0, 0.0, 0.0),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="logs/px4_step_response.json")
    ap.add_argument("--altitude", type=float, default=5.0)
    ap.add_argument("--no-launch", action="store_true", help="use an already running SITL")
    args = ap.parse_args()

    launcher = None if args.no_launch else SimLauncher(world="default", log_path=Path("logs/sitl_m1_calib.log"))
    if launcher:
        launcher.start()
        if not launcher.wait_ready(180):
            launcher.stop()
            raise SystemExit("PX4 did not become ready")
    link = PX4Link(log=False)
    gz = GzLink(scan_topic=None)
    try:
        link.connect()
        if not gz.wait_truth(30):
            raise SystemExit("no Gazebo ground-truth odometry")
        if not link.offboard_takeoff(args.altitude):
            raise SystemExit("takeoff failed")
        link.wait_for(lambda t: abs(t.vel_enu[2]) < 0.1, 10.0)
        time.sleep(2.0)
        link.samples.clear()
        gz.truth_log.clear()
        cmd_log = []
        link.log_enabled = True
        gz.log_truth = True
        t0 = time.monotonic()
        for dur, vx, vy, vz, yr in SEQUENCE:
            link.set_velocity(vx, vy, vz, yr)
            cmd_log.append((time.monotonic() - t0, vx, vy, vz, yr))
            time.sleep(dur)
        link.log_enabled = False
        gz.log_truth = False
        samples = [dict(s, t=s["t"] - t0) for s in link.samples]
        truth = [{"t": s.wall - t0, "sim_t": s.stamp, "pos": s.pos.tolist(), "vel": s.vel.tolist(),
                  "quat": s.quat.tolist(), "rates": s.rates.tolist()} for s in gz.truth_log]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"sequence": SEQUENCE, "altitude": args.altitude, "cmd_log": cmd_log,
                       "samples": samples, "truth": truth}, f)
        print(f"recorded {len(samples)} PX4 samples and {len(truth)} ground-truth samples "
              f"over {samples[-1]['t']:.1f}s -> {args.out}")
        link.land()
    finally:
        link.close()
        if launcher:
            launcher.stop()


if __name__ == "__main__":
    main()
