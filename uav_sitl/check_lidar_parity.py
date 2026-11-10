"""
check_lidar_parity.py
---------------------
Sim-to-SITL sensor parity: fly the x500_lidar3d to a hover inside an exported forest and
compare Gazebo's 7 x 90 LiDAR grid with rl_env.lidar3d.scan() of the same world at the
ground-truth pose. Reports range agreement and feature agreement (the policy input).

Usage (WSL):  python -m uav_sitl.check_lidar_parity --seed 77
"""

import argparse
import json
import time

import numpy as np

from rl_env.lidar3d import DEFAULT_SPEC, features, filter_ground, scan
from rl_env.observation import quat_to_rotation_matrix
from rl_env.world3d import generate_forest
from uav_sitl.gz_link import GzLink
from uav_sitl.policy_runtime import sanitize_ranges
from uav_sitl.px4_link import PX4Link
from uav_sitl.run_trials import takeoff_to_height
from uav_sitl.sim_launcher import SimLauncher
from uav_sitl.world_sdf import export


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=77)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--out", default="logs/lidar_parity.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    world = generate_forest(rng, 20.0, "dense", keep_out=[(0.0, 0.0, 1.8)])
    name = f"parity_{args.seed}"
    export(world, name, (0, 0, 0), (5, 0, 2))
    sim = SimLauncher(world=name, model="x500_lidar3d", log_path=f"logs/sitl_{name}.log",
                      extra_env={"PX4_SIM_MODEL": "x500_lidar3d"})
    sim.start()
    assert sim.wait_ready(240), "SITL did not start"
    px4 = PX4Link(); px4.connect(60)
    gz = GzLink(model="x500_lidar3d_0", world=name, scan_topic="/uav/sensors/scan3d")
    gz.wait_truth(30)
    res = takeoff_to_height(px4, gz, 2.0)
    if res != "ok":
        tel = px4.snapshot()
        raise SystemExit(f"takeoff failed at step '{res}'; armed={tel.armed} mode={tel.main_mode} "
                         f"range={gz.range_down} px4: {tel.status_texts[-6:]}")
    spec = DEFAULT_SPEC
    rows = []
    for k in range(args.samples):
        px4.set_velocity(yaw_rate=0.5 if k % 2 else -0.5)     # vary heading between samples
        time.sleep(0.6)
        px4.set_velocity()
        time.sleep(0.4)
        sc, tr = gz.get_scan(), gz.get_truth()
        R = quat_to_rotation_matrix(tr.quat).astype(np.float64)
        g = sanitize_ranges(sc.ranges, spec.min_range, spec.max_range).reshape(len(spec.elevations_deg), spec.n_azimuth)
        g = filter_ground(g, tr.pos, R, spec)
        s = scan(world.local(tr.pos[:2], 11.0), tr.pos, R, spec)
        fg, fs = features(g, spec), features(s, spec)
        both = (g < spec.max_range) & (s < spec.max_range)
        rows.append({"grid_shape": list(sc.ranges.shape), "vertical_count_ok": sc.ranges.shape[0] == 7,
                     "hit_agreement": float(np.mean((g < spec.max_range) == (s < spec.max_range))),
                     "range_mae_common_hits": float(np.mean(np.abs(g[both] - s[both]))) if both.any() else None,
                     "feature_mae": float(np.mean(np.abs(fg - fs))),
                     "feature_max_err": float(np.max(np.abs(fg - fs)))})
        raw_last = {"gazebo_raw": sanitize_ranges(sc.ranges, spec.min_range, spec.max_range).tolist(),
                    "sim": s.tolist(), "pos": tr.pos.tolist(), "quat": tr.quat.tolist()}
    px4.close()
    sim.stop()
    summary = {k: float(np.mean([r[k] for r in rows if r[k] is not None])) for k in
               ("hit_agreement", "range_mae_common_hits", "feature_mae", "feature_max_err")}
    summary["grid_shape"] = rows[0]["grid_shape"]
    gr = np.array(raw_last["gazebo_raw"]).reshape(7, 90)
    summary["gazebo_min_range_per_row"] = np.round(gr.min(axis=1), 2).tolist()
    summary["gazebo_hits_below_0.6m_per_row"] = (gr < 0.6).sum(axis=1).tolist()
    json.dump({"summary": summary, "samples": rows, "last": raw_last}, open(args.out, "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
