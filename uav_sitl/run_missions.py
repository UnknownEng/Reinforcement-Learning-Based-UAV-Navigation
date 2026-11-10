"""
run_missions.py
---------------
M10: full autonomous missions in PX4 SITL + Gazebo: preflight -> arm -> takeoff ->
multi-waypoint navigation through an unknown forest with moving hazards -> landing,
with fail-safes (uav_sitl/mission.py).

Usage (WSL):  python -m uav_sitl.run_missions --model experiments/world/dr/model.zip --missions 5
"""

import argparse
import json
import math
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from rl_env.world3d import free_space_connected, generate_forest
from uav_sitl.gz_link import GzLink
from uav_sitl.mission import Mission, MissionConfig
from uav_sitl.movers import SitlMovers
from uav_sitl.px4_link import PX4Link
from uav_sitl.sim_launcher import SimLauncher
from uav_sitl.world_sdf import export
from training.train_world_policy import shield_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def make_mission(seed: int, n_waypoints: int = 3, half: float = 20.0):
    rng = np.random.default_rng(seed)
    for _ in range(50):
        world = generate_forest(rng, half, str(rng.choice(["open", "medium", "dense"])), keep_out=[(0.0, 0.0, 1.8)])
        wps, prev, ok = [], np.zeros(2), True
        for _ in range(n_waypoints):
            for _ in range(40):
                d, ang = rng.uniform(8.0, 14.0), rng.uniform(-math.pi, math.pi)
                p = prev + d * np.array([math.cos(ang), math.sin(ang)])
                z = rng.uniform(1.8, 3.2)
                if np.all(np.abs(p) < half - 3.0) and world.distance(np.array([[p[0], p[1], z]]))[0] > 1.0 \
                        and free_space_connected(world, prev, p, inflate=0.45)[0]:
                    wps.append([float(p[0]), float(p[1]), float(z)])
                    prev = p
                    break
            else:
                ok = False
                break
        if ok:
            return world, wps
    raise RuntimeError("no feasible mission")


def fly_mission(seed: int, world, wps, n_movers: int, model: str, shield: bool, gui: bool, log_dir: Path,
                cbf=False) -> dict:
    movers = SitlMovers(seed, n_movers, avoid_xy=(0.0, 0.0)) if n_movers > 0 else None
    name = f"mission_{seed}"
    export(world, name, (0, 0, 0), wps[-1], decorate=movers.add_to_sdf if movers else None)
    sim = SimLauncher(world=name, model="x500_lidar3d", headless=not gui,
                      log_path=log_dir / f"sitl_{name}.log", extra_env={"PX4_SIM_MODEL": "x500_lidar3d"})
    rec = {"seed": seed, "waypoint_list": wps, "waypoints": len(wps), "movers": n_movers, "forest": world.meta}
    px4, stop, mover_daemon = None, threading.Event(), None
    t0 = time.monotonic()
    try:
        sim.start()
        if not sim.wait_ready(240):
            rec["final_state"] = "sim_failed_to_start"
        else:
            px4 = PX4Link()
            px4.connect(60)
            gz = GzLink(model="x500_lidar3d_0", world=name, scan_topic="/uav/sensors/scan3d")
            mission = Mission(px4, gz, model, "50d", MissionConfig(waypoints=wps),
                              shield=shield_config() if shield else None, world=world, cbf=cbf)
            if movers is not None:
                # hazards move from a separate process (no GIL sharing with the control loop); this thread
                # only evaluates the same deterministic positions to log the true clearance to them
                mover_daemon = subprocess.Popen(movers.daemon_argv(name, "x500_lidar3d_0"), stdout=subprocess.DEVNULL,
                                                stderr=open(log_dir / f"sitl_{name}.movers.log", "w"))

                def tick():
                    while not stop.is_set():
                        tr = gz.get_truth()
                        if tr is not None:
                            movers.advance_to(tr.stamp)
                            if mission.airborne_for_movers(float(tr.pos[2])):
                                d = float(movers.as_world().distance((tr.pos + np.array([0, 0, 0.24]))[None])[0]) - 0.35
                                mission.min_clear_movers = min(mission.min_clear_movers, d)
                                mission.min_truth_clearance = min(mission.min_clear_static, mission.min_clear_movers)
                        time.sleep(0.05)
                threading.Thread(target=tick, daemon=True).start()
            rec.update(mission.run())
    except Exception as exc:
        rec.update(final_state="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        stop.set()
        if mover_daemon is not None:
            mover_daemon.terminate()
            try:
                mover_daemon.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mover_daemon.kill()
        if px4 is not None:
            px4.close()
        sim.stop()
    rec["wall_s"] = round(time.monotonic() - t0, 1)
    return rec


def policy_flew(rec: dict) -> bool:
    return any(e["state"] == "NAVIGATE" for e in rec.get("events", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--missions", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=8100)
    ap.add_argument("--movers", type=int, default=3)
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--cbf", action="store_true", help="also apply the 3-D CBF filter after the shield")
    ap.add_argument("--cbf-moving", action="store_true", help="CBF filter with the moving-obstacle term (M6)")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cbf_moving:
        args.cbf = "moving"
    out = Path(args.out or PROJECT_ROOT / "logs" / "sitl_trials" / f"missions_{int(time.time())}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for i in range(args.missions):
        seed = args.seed0 + i
        world, wps = make_mission(seed)
        # Relaunch only when PX4/Gazebo failed before the policy ever had control (start-up, arming,
        # takeoff); a mission in which the policy flew is never repeated.
        for attempt in range(1, 4):
            rec = fly_mission(seed, world, wps, args.movers, args.model, not args.no_shield, args.gui, out.parent,
                              args.cbf)
            rec["launch_attempts"] = attempt
            if policy_flew(rec) or attempt == 3:
                break
            why = rec.get("error") or (rec.get("events") or [{"msg": rec.get("final_state")}])[-1]["msg"]
            print(f"[mission] seed {seed}: launch attempt {attempt} failed before the policy flew ({why}); "
                  f"relaunching", flush=True)
        results.append(rec)
        legs = rec.get("legs", [])
        print(f"[mission] {i + 1}/{args.missions} seed {seed}: {rec.get('final_state')}  "
              f"waypoints {rec.get('waypoints_reached', 0)}/{len(wps)}  min_clear={rec.get('min_truth_clearance')} "
              f"(static {rec.get('min_clear_static')}, movers {rec.get('min_clear_movers')})  "
              f"legs={[(l['result'], l['duration_s']) for l in legs]}", flush=True)
        json.dump({"model": args.model, "shield": not args.no_shield, "cbf": args.cbf, "missions": results},
                  open(out, "w"), indent=1)
    # Obstacles are visual-only in Gazebo, so contact is judged from ground truth: any hull contact fails a mission.
    done = sum(r.get("final_state") == "DONE" and r.get("waypoints_reached") == r["waypoints"]
               and (r.get("min_truth_clearance") is None or r["min_truth_clearance"] > 0.0) for r in results)
    print(f"[mission] fully successful missions (all waypoints, landed, no contact): {done}/{len(results)} -> {out}")


if __name__ == "__main__":
    main()
