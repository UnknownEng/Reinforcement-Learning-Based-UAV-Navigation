"""
run_trials.py
-------------
Honest closed-loop validation of learned policies in PX4 SITL + Gazebo.

Each trial: generate a scenario with the same code used in training (start moved to
PX4's spawn point at the origin), export it to Gazebo, boot PX4, take off, hand
control to the policy (no scripted waypoints), and judge the outcome:
  success    PX4's own estimate is within the goal tolerance at low speed
  collision  Gazebo ground-truth pose comes within the vehicle radius of any obstacle,
             or the vehicle tips over / hits the ground
  timeout    sim-time horizon exceeded
Every attempt is recorded; nothing is retried or overwritten.

Usage (WSL):
  python -m uav_sitl.run_trials --suite m3 --model <zip> --trials 10
  python -m uav_sitl.run_trials --suite forest --model <zip> --trials 10 --shield
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rl_env.safety_shield import SafetyShieldConfig
from rl_env.world3d import World3D, generate_forest, generate_urban, generate_warehouse
from uav_sitl.gz_link import GzLink
from uav_sitl.movers import SitlMovers
from uav_sitl.policy_runtime import PolicyRuntime, sanitize_ranges
from uav_sitl.px4_link import PX4_MAIN_MODE_OFFBOARD, PX4Link
from uav_sitl.sim_launcher import SimLauncher
from uav_sitl.world_sdf import export, world_from_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUITES = {
    # name: obs mode, drone model, scan topic, horizon (s sim), goal tolerance (m)
    "m3": dict(obs="18d", model="x500_lidar", scan="/uav/sensors/scan", horizon=25.0, tol=0.4, radius=0.35),
    "m4": dict(obs="26d", model="x500_lidar", scan="/uav/sensors/scan", horizon=25.0, tol=0.4, radius=0.35),
    "m5": dict(obs="26d", model="x500_lidar", scan="/uav/sensors/scan", horizon=35.0, tol=0.4, radius=0.35),
    "forest": dict(obs="50d", model="x500_lidar3d", scan="/uav/sensors/scan3d", horizon=50.0, tol=0.5, radius=0.35),
    "urban": dict(obs="50d", model="x500_lidar3d", scan="/uav/sensors/scan3d", horizon=50.0, tol=0.5, radius=0.35),
    "warehouse": dict(obs="50d", model="x500_lidar3d", scan="/uav/sensors/scan3d", horizon=50.0, tol=0.5,
                      radius=0.35),
    "forest_movers": dict(obs="50d", model="x500_lidar3d", scan="/uav/sensors/scan3d", horizon=50.0, tol=0.5,
                          radius=0.35),
}


def _shift(world: World3D, dx: float, dy: float) -> World3D:
    w = World3D(world.cyl.copy(), world.sph.copy(), world.box.copy(), world.half_size, world.kind, world.meta,
                world.cyl_tag, world.sph_tag, world.box_tag, world.cap.copy(), world.cap_tag)
    for arr, cols in ((w.cyl, (0, 1)), (w.sph, (0, 1)), (w.box, (0, 1)), (w.cap, (0, 1, 3, 4))):
        if len(arr):
            for c in cols:
                arr[:, c] -= dx if c in (0, 3) else dy
    return w


def make_scenario(suite: str, seed: int):
    """Returns (world or None, goal_xyz, geofence_offset_xy) with the start at the origin."""
    rng = np.random.default_rng(seed)
    if suite == "m3":
        d, ang = rng.uniform(2.5, 8.0), rng.uniform(-math.pi, math.pi)
        return None, np.array([d * math.cos(ang), d * math.sin(ang), rng.uniform(1.5, 4.5)]), (0.0, 0.0)
    if suite in ("m4", "m5"):
        if suite == "m4":
            from rl_env.uav_fast_env_v1 import UAVFastEnvV1
            env = UAVFastEnvV1()
            env.reset(seed=seed, options={"curriculum_stage": int(rng.integers(2, 4))})
        else:
            from rl_env.uav_fast_env_v2 import UAVFastEnvV2, UAVObstacleEnvConfigV2
            env = UAVFastEnvV2(UAVObstacleEnvConfigV2(curriculum_stage=3))
            env.reset(seed=seed)
        sx, sy = float(env.dynamics.state.pos_world[0]), float(env.dynamics.state.pos_world[1])
        cyl = np.array([(x - sx, y - sy, r, 0.0, 10.0) for x, y, r in env.obstacles]).reshape(-1, 5)
        world = World3D(cyl, np.zeros((0, 4)), np.zeros((0, 7)), 12.0, "poles",
                        cyl_tag=np.array(["pole"] * len(cyl)))
        goal = np.array(env.goal_world, dtype=np.float64) - np.array([sx, sy, 0.0])
        return world, goal, (sx, sy)
    half = 20.0
    forest = lambda: generate_forest(rng, half, str(rng.choice(["open", "medium", "dense"])),
                                     keep_out=[(0.0, 0.0, 1.8)])
    kind_rng_preset = {"forest": forest, "forest_movers": forest,
                       "urban": lambda: generate_urban(rng, half, keep_out=[(0.0, 0.0, 1.8)]),
                       "warehouse": lambda: generate_warehouse(rng, half, keep_out=[(0.0, 0.0, 1.8)])}
    from rl_env.world3d import free_space_connected
    for _ in range(30):
        d, ang = rng.uniform(8.0, 18.0), rng.uniform(-math.pi, math.pi)
        goal = np.array([d * math.cos(ang), d * math.sin(ang), rng.uniform(1.6, 3.4)])
        if np.any(np.abs(goal[:2]) > half - 2.0):
            continue
        world = kind_rng_preset[suite]()
        if world.distance(goal[None])[0] < 0.9:
            continue
        ok, _ = free_space_connected(world, np.zeros(2), goal[:2], inflate=0.45)
        if ok:
            return world, goal, (0.0, 0.0)
    raise RuntimeError("could not sample a feasible scenario")


def takeoff_to_height(px4: PX4Link, gz: GzLink, height: float, timeout: float = 40.0) -> str:
    """Arm in OFFBOARD and climb until the downward rangefinder (not the drifting baro) reads `height`.
    Returns "ok" or the name of the step that failed."""
    if not px4.wait_for(lambda t: t.have_position, 30.0):
        return "no_position"
    for attempt in range(3):
        if px4.start_offboard(timeout=15.0) and px4.arm(timeout=15.0):
            break
        time.sleep(2.0)
    else:
        return "offboard_or_arm_refused"
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        tel = px4.snapshot()
        h = gz.height_above_ground(tel.roll, tel.pitch)
        h = tel.pos_enu[2] if h is None else h
        if h >= height - 0.1:
            px4.set_velocity()
            return "ok"
        px4.set_velocity(vz=min(1.0, max(0.3, height - h)))
        time.sleep(0.05)
    px4.set_velocity()
    return "climb_timeout"


def run_trial(suite: str, seed: int, model_path: str, shield: bool, out_dir: Path, gui: bool = False,
              cbf=False, record_dir: Path = None) -> dict:
    cfg = SUITES[suite]
    # --record: same airframe and LiDAR plus recording-only cameras (models/x500_lidar3d_cam)
    gz_model = cfg["model"] + ("_cam" if record_dir is not None and cfg["model"] == "x500_lidar3d" else "")
    world, goal, (ox, oy) = make_scenario(suite, seed)
    world_name = "default" if world is None else f"{suite}_trial_{seed}"
    movers = None
    if suite == "forest_movers":
        movers = SitlMovers(seed, int(np.random.default_rng(seed + 1).integers(2, 6)), avoid_xy=(0.0, 0.0))
    if world is not None:
        export(world, world_name, (0.0, 0.0, 0.0), goal, decorate=movers.add_to_sdf if movers else None)
    shield_cfg = None
    if shield:
        g = 8.0 if suite == "m5" else 19.5
        shield_cfg = SafetyShieldConfig(geofence_x=(-g - ox, g - ox), geofence_y=(-g - oy, g - oy),
                                        geofence_z=(0.25, 7.0 if suite in ("forest", "urban", "warehouse") else 6.0))
    runtime = PolicyRuntime(model_path, cfg["obs"], goal, shield=shield_cfg, cbf=cbf)
    log_path = out_dir / f"sitl_{suite}_{seed}.log"
    rec = {"suite": suite, "seed": seed, "model": model_path, "shield": shield, "cbf": cbf, "goal": goal.tolist(),
           "world": world_name, "obstacles": None if world is None else world.counts(),
           "movers": 0 if movers is None else len(movers.names())}
    launcher = SimLauncher(world=world_name, model=gz_model, autostart=4012, headless=not gui,
                           log_path=log_path, extra_env={"PX4_SIM_MODEL": gz_model})
    px4 = gz = None
    mover_daemon = cam_daemon = None
    rdir = None if record_dir is None else Path(record_dir) / f"{suite}_{seed}"
    log = {k: [] for k in ("stamp", "truth_pos", "truth_quat", "nav_pos", "vel", "ranges_raw", "ranges_used",
                           "raw_action", "action", "cmd", "safety_active")}
    t_wall0 = time.monotonic()
    try:
        launcher.start()
        if not launcher.wait_ready(240):
            rec.update(outcome="sim_failed_to_start")
            return rec
        px4 = PX4Link()
        px4.connect(60)
        gz = GzLink(model=f"{gz_model}_0", world=world_name, scan_topic=cfg["scan"])
        if rdir is not None:
            rdir.mkdir(parents=True, exist_ok=True)
            cam_daemon = subprocess.Popen([sys.executable, "-m", "uav_sitl.camera_recorder", "--out", str(rdir)],
                                          stdout=subprocess.DEVNULL, stderr=open(rdir / "camera_recorder.log", "w"))
        if movers is not None:
            # hazards move from a separate process (no GIL sharing with this control loop)
            mover_daemon = subprocess.Popen(movers.daemon_argv(world_name, f"{gz_model}_0"),
                                            stdout=subprocess.DEVNULL, stderr=open(log_path.with_suffix('.movers.log'), 'w'))
        if not gz.wait_truth(30):
            rec.update(outcome="no_ground_truth")
            return rec
        takeoff = takeoff_to_height(px4, gz, 2.0)
        if takeoff != "ok":
            rec.update(outcome="takeoff_failed", takeoff_step=takeoff,
                       px4_messages=px4.snapshot().status_texts[-8:])
            return rec
        t0 = gz.get_truth().stamp
        last_scan, traj, n_shield, min_clear, steps = -1.0, [], 0, float("inf"), 0
        outcome = "timeout"
        # Stale-perception checks run on SIM time (PX4 and Gazebo are lockstepped, so a slow Gazebo frame is
        # not a stall): hold position after 0.3 s without a new scan, give up after 1.0 s. A simulator that
        # stops advancing altogether for 30 s of wall time is reported separately.
        stall_wall, stall_sim = time.monotonic(), t0
        while True:
            truth = gz.get_truth()
            if truth.stamp - t0 > cfg["horizon"]:
                break
            scan = gz.get_scan()
            if scan is None or scan.stamp <= last_scan:
                gap = truth.stamp - stall_sim
                if gap > 0.3:
                    px4.set_velocity()
                if gap > 1.0 or time.monotonic() - stall_wall > 30.0:
                    outcome = "sensor_stall" if gap > 1.0 else "sim_frozen"
                    rec["stall"] = {"sim_gap_s": round(gap, 2), "wall_gap_s": round(time.monotonic() - stall_wall, 1)}
                    break
                time.sleep(0.005)
                continue
            stall_wall, stall_sim = time.monotonic(), truth.stamp
            last_scan = scan.stamp
            tel = px4.snapshot()
            if tel.main_mode != PX4_MAIN_MODE_OFFBOARD:
                # PX4 took the vehicle away from the policy (e.g. a failsafe after a SITL sensor timeout):
                # not a policy outcome, recorded with PX4's own messages.
                outcome = "px4_failsafe"
                rec["px4_messages"] = tel.status_texts[-8:]
                break
            # Navigation state as on a real low-flying drone: horizontal position/velocity and attitude
            # from PX4's estimator, height above ground from the downward rangefinder (baro drifts ~1 m).
            h = gz.height_above_ground(tel.roll, tel.pitch)
            nav_pos = (tel.pos_enu[0], tel.pos_enu[1], h if h is not None else tel.pos_enu[2])
            step = runtime.step(nav_pos, tel.vel_enu, tel.quat_enu_flu, tel.rates_flu, scan)
            px4.set_velocity(*step.cmd)
            steps += 1
            n_shield += int(step.shield_active)
            if rdir is not None:
                for k, v in (("stamp", truth.stamp), ("truth_pos", truth.pos), ("truth_quat", truth.quat),
                             ("nav_pos", nav_pos), ("vel", tel.vel_enu),
                             ("ranges_raw", sanitize_ranges(scan.ranges, 0.10, 10.0).reshape(7, -1)),
                             ("ranges_used", runtime.last_ranges), ("raw_action", step.raw_action),
                             ("action", step.action), ("cmd", step.cmd), ("safety_active", step.shield_active)):
                    log[k].append(np.array(v, dtype=np.float32) if k != "stamp" else float(v))
            # ground-truth safety checks
            if world is not None:
                hull = (truth.pos + np.array([0.0, 0.0, 0.24]))[None]   # x500 base_link above model origin
                c = float(world.distance(hull)[0]) - cfg["radius"]
                if movers is not None:
                    movers.advance_to(truth.stamp)   # same deterministic positions the daemon pushes
                    c = min(c, float(movers.as_world().distance(hull)[0]) - cfg["radius"])
                min_clear = min(min_clear, c)
                if c <= 0.0:
                    outcome = "collision"
                    break
            roll = math.atan2(2 * (truth.quat[0] * truth.quat[1] + truth.quat[2] * truth.quat[3]),
                              1 - 2 * (truth.quat[1] ** 2 + truth.quat[2] ** 2))
            if abs(roll) > math.radians(70) or (truth.pos[2] < 0.15 and steps > 20):
                outcome = "crash"
                break
            traj.append([round(truth.stamp - t0, 2)] + [round(float(v), 3) for v in truth.pos] +
                        [round(float(v), 3) for v in tel.pos_enu] + [round(float(v), 3) for v in step.cmd])
            nav_dist = float(np.linalg.norm(np.asarray(nav_pos) - goal))
            if nav_dist <= cfg["tol"] and float(np.linalg.norm(tel.vel_enu)) <= 0.6:
                outcome = "success"
                break
        truth = gz.get_truth()
        tel = px4.snapshot()
        h = gz.height_above_ground(tel.roll, tel.pitch)
        nav_end = np.array([tel.pos_enu[0], tel.pos_enu[1], h if h is not None else tel.pos_enu[2]])
        path = np.array([t[1:4] for t in traj]) if traj else np.zeros((1, 3))
        rec.update(outcome=outcome, sim_time_s=round(truth.stamp - t0, 2), steps=steps,
                   truth_dist_to_goal=round(float(np.linalg.norm(truth.pos - goal)), 3),
                   nav_dist_to_goal=round(float(np.linalg.norm(nav_end - goal)), 3),
                   nav_minus_truth_xyz=[round(float(a - b), 3) for a, b in zip(nav_end, truth.pos)],
                   ekf_dist_to_goal=round(float(np.linalg.norm(np.asarray(tel.pos_enu) - goal)), 3),
                   ekf_minus_truth_xyz=[round(float(a - b), 3) for a, b in zip(tel.pos_enu, truth.pos)],
                   path_length=round(float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))), 2),
                   straight_dist=round(float(np.linalg.norm(goal - np.array([0, 0, 2.0]))), 2),
                   min_truth_clearance=None if world is None else round(min_clear, 3),
                   shield_rate=round(n_shield / max(steps, 1), 3), wall_s=round(time.monotonic() - t_wall0, 1),
                   trajectory=traj[::3])
        px4.set_velocity()
        if rdir is not None and log["stamp"]:
            np.savez_compressed(rdir / "flight.npz", **{k: np.array(v) for k, v in log.items()})
            json.dump({**{k: v for k, v in rec.items() if k != "trajectory"}, "t0": t0, "gz_model": gz_model,
                       "mover_seed": seed if movers is not None else None,
                       "n_movers": len(movers.names()) if movers is not None else 0},
                      open(rdir / "flight.json", "w"), indent=1, default=float)
        return rec
    except Exception as exc:  # record, don't hide
        rec.update(outcome="error", error=f"{type(exc).__name__}: {exc}")
        return rec
    finally:
        for daemon in (mover_daemon, cam_daemon):
            if daemon is not None:
                daemon.terminate()
                try:
                    daemon.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    daemon.kill()
        if px4 is not None:
            px4.close()
        launcher.stop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES), required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--shield", action="store_true")
    ap.add_argument("--cbf", action="store_true", help="also apply the 3-D CBF filter (50d suites)")
    ap.add_argument("--cbf-moving", action="store_true", help="CBF filter with the moving-obstacle term (M6)")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--record", default=None, metavar="DIR",
                    help="fly the camera variant and save camera frames + LiDAR + trajectory per trial under DIR")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cbf_moving:
        args.cbf = "moving"
    out_dir = PROJECT_ROOT / "logs" / "sitl_trials"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = Path(args.out or out_dir / f"{args.suite}_{'shield_' if args.shield else ''}{int(time.time())}.json")
    results = []
    for i in range(args.trials):
        # Launch failures (PX4/Gazebo did not start, or the sensor stack glitched before takeoff
        # completed) happen before the policy ever has control: relaunch up to twice and record it.
        # Flights where the policy had control are never retried.
        launch_attempts = 0
        while True:
            launch_attempts += 1
            r = run_trial(args.suite, args.seed0 + i, args.model, args.shield, out_dir, args.gui, args.cbf,
                          Path(args.record) if args.record else None)
            if r["outcome"] not in ("takeoff_failed", "sim_failed_to_start", "no_ground_truth") or launch_attempts >= 3:
                break
        r["launch_attempts"] = launch_attempts
        results.append(r)
        print(f"[{args.suite}] trial {i + 1}/{args.trials} seed {r['seed']}: {r['outcome']}"
              + (f"  t={r.get('sim_time_s')}s  truth_err={r.get('truth_dist_to_goal')}m"
                 f"  min_clear={r.get('min_truth_clearance')}  shield={r.get('shield_rate')}" if 'steps' in r else
                 f"  {r.get('error', '')}"), flush=True)
        json.dump({"suite": args.suite, "model": args.model, "shield": args.shield, "cbf": args.cbf,
                   "trials": results},
                  open(out, "w"), indent=1)
    n = len(results)
    counts = {k: sum(r["outcome"] == k for r in results) for k in sorted({r["outcome"] for r in results})}
    print(f"[{args.suite}] {counts}  success {100 * counts.get('success', 0) / n:.0f}%  -> {out}")


if __name__ == "__main__":
    main()
