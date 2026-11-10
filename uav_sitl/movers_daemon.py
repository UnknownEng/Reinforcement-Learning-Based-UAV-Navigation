"""
movers_daemon.py
----------------
Moves the SITL hazards in Gazebo from its own process, so the vehicle's control loop never shares a
Python interpreter (GIL) with blocking Gazebo service calls. Positions follow SitlMovers.advance_to
(fixed 0.05 s steps on absolute sim time from the same seed), the same deterministic function the
control process evaluates for its ground-truth collision checks.

Started by uav_sitl/run_trials.py and uav_sitl/run_missions.py; stops when killed.
"""

import argparse
import time

from uav_sitl.gz_link import GzLink
from uav_sitl.movers import SitlMovers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", required=True)
    ap.add_argument("--model", required=True, help="vehicle model name, only used to read the sim clock")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--avoid", type=float, nargs=2, default=(0.0, 0.0))
    args = ap.parse_args()
    gz = GzLink(model=args.model, world=args.world, scan_topic=None, range_topic=None)
    movers = SitlMovers(args.seed, args.n, avoid_xy=tuple(args.avoid))
    while True:
        tr = gz.get_truth()
        if tr is not None and tr.stamp >= movers.t_sim + movers.STEP:
            movers.update(gz, tr.stamp)
        time.sleep(0.005)


if __name__ == "__main__":
    main()
