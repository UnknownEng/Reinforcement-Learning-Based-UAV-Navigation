"""
fit_dynamics.py
---------------
M1: fit the fast simulator's cascade parameters to PX4 SITL ground truth.

Each command segment of the recorded step-response flight is replayed in the
fast simulator, starting from the true Gazebo state at the segment start, and
the simulated world velocity and roll/pitch are compared with ground truth.

Usage:  python -m uav_sitl.fit_dynamics --log logs/px4_step_response.json --out logs/m1_fit.json
"""

import argparse
import json
import math

import numpy as np
from scipy.optimize import least_squares

from rl_env.action import CommandedVelocities
from rl_env.config import UAVEnvConfig
from rl_env.dynamics import FastUAVDynamics

PARAMS = ["tau_velocity_xy", "vel_i_xy", "tau_velocity_z", "vel_i_z", "tau_attitude", "tau_thrust",
          "drag_xy", "drag_z", "latency", "tau_yaw"]
X0 = [0.556, 0.4, 0.25, 2.0, 0.10, 0.03, 0.10, 0.10, 0.05, 0.12]
LO = [0.15, 0.0, 0.05, 0.0, 0.02, 0.01, 0.0, 0.0, 0.0, 0.02]   # lags >= dt (explicit-Euler stability)
HI = [1.50, 3.0, 1.00, 8.0, 0.40, 0.20, 1.0, 1.0, 0.20, 0.60]


def quat_to_euler(q):
    qw, qx, qy, qz = q
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def load(path):
    d = json.load(open(path))
    tr = d["truth"]
    t = np.array([s["t"] for s in tr])
    vel = np.array([s["vel"] for s in tr])
    pos = np.array([s["pos"] for s in tr])
    eul = np.array([quat_to_euler(s["quat"]) for s in tr])
    rates = np.array([s["rates"] for s in tr])
    # Smooth finite-difference velocity a little (pose stream is exact but stamps jitter)
    k = 3
    vel_s = np.vstack([np.convolve(vel[:, i], np.ones(k) / k, mode="same") for i in range(3)]).T
    cmds = d["cmd_log"]
    return t, pos, vel_s, eul, rates, cmds


def apply(dyn, x):
    p = dict(zip(PARAMS, x))
    dyn.tau_velocity_xy, dyn.vel_i_xy = p["tau_velocity_xy"], p["vel_i_xy"]
    dyn.tau_velocity_z, dyn.vel_i_z = p["tau_velocity_z"], p["vel_i_z"]
    dyn.tau_attitude, dyn.tau_thrust = p["tau_attitude"], p["tau_thrust"]
    dyn.drag_xy, dyn.drag_z = p["drag_xy"], p["drag_z"]
    dyn.set_latency(max(p["latency"], 0.0))
    dyn.config.tau_yaw = p["tau_yaw"]


def simulate_segments(x, data, legacy=False):
    t, pos, vel, eul, rates, cmds = data
    cfg = UAVEnvConfig(enable_noise=False)
    dyn = FastUAVDynamics(cfg)
    apply(dyn, x)
    if legacy:  # approximation of the pre-M1 model: fast first-order lag, accel capped at 3 / 2.5 m/s^2
        dyn.max_horizontal_accel, dyn.max_vertical_accel = 3.0, 2.5
    sim_vel, sim_att, idx_all = [], [], []
    edges = [c[0] for c in cmds] + [t[-1] + 1e-6]
    for i, c in enumerate(cmds):
        idx = np.where((t >= edges[i]) & (t < edges[i + 1]))[0]
        if len(idx) < 5:
            continue
        j0 = idx[0]
        roll0, pitch0, yaw0 = eul[j0]
        dyn.set_state(pos[j0], vel[j0], roll0, pitch0, yaw0, yaw_rate=rates[j0][2])
        prev = cmds[i - 1] if i > 0 else (0, 0, 0, 0, 0)
        dyn.cmd_queue.clear()
        for _ in range(dyn.latency_substeps):
            dyn.cmd_queue.append(CommandedVelocities(*prev[1:5]))
        cmd = CommandedVelocities(*c[1:5])
        t_sim = t[j0]
        out_v, out_a, ts = [dyn.state.vel_world.copy()], [(dyn.state.roll, dyn.state.pitch)], [t_sim]
        dt = cfg.dt_substep
        while t_sim < t[idx[-1]]:
            dyn.cmd_queue.append(cmd)
            delayed = dyn.cmd_queue.popleft()
            dyn._substep(delayed, dt)
            t_sim += dt
            out_v.append(dyn.state.vel_world.copy())
            out_a.append((dyn.state.roll, dyn.state.pitch))
            ts.append(t_sim)
        ts, out_v, out_a = np.array(ts), np.array(out_v), np.array(out_a)
        for j in idx:
            sim_vel.append([np.interp(t[j], ts, out_v[:, k]) for k in range(3)])
            sim_att.append([np.interp(t[j], ts, out_a[:, k]) for k in range(2)])
            idx_all.append(j)
    return np.array(idx_all), np.array(sim_vel), np.array(sim_att)


def residuals(x, data, att_weight=3.0):
    idx, sv, sa = simulate_segments(x, data)
    _, _, vel, eul, _, _ = data
    rv = (sv - vel[idx]).ravel()
    ra = att_weight * (sa - eul[idx, :2]).ravel()
    return np.concatenate([rv, ra])


def metrics(x, data, legacy=False):
    idx, sv, sa = simulate_segments(x, data, legacy=legacy)
    _, _, vel, eul, _, _ = data
    ev = sv - vel[idx]
    ea = np.degrees(sa - eul[idx, :2])
    return {"vel_rmse_xy": float(np.sqrt(np.mean(ev[:, :2] ** 2))), "vel_rmse_z": float(np.sqrt(np.mean(ev[:, 2] ** 2))),
            "tilt_rmse_deg": float(np.sqrt(np.mean(ea ** 2))), "samples": int(len(idx))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="logs/px4_step_response.json")
    ap.add_argument("--out", default="logs/m1_fit.json")
    ap.add_argument("--fix-integrators", action="store_true",
                    help="hold the velocity-loop I gains at PX4 defaults (0.4 xy, 2.0 z); short step "
                         "segments cannot identify them, but they matter for wind rejection")
    args = ap.parse_args()
    data = load(args.log)
    if args.fix_integrators:
        for name, val in (("vel_i_xy", 0.4), ("vel_i_z", 2.0)):
            i = PARAMS.index(name)
            X0[i], LO[i], HI[i] = val, val - 1e-6, val + 1e-6

    legacy_x = [0.20, 0.0, 0.15, 0.0, 0.02, 0.01, 0.0, 0.0, 0.05, 0.12]
    before_legacy = metrics(legacy_x, data, legacy=True)
    before_default = metrics(X0, data)
    fit = least_squares(residuals, X0, bounds=(LO, HI), args=(data,), x_scale="jac", diff_step=1e-3, max_nfev=200)
    x = np.array(fit.x)
    # Latency is quantized to the 10 ms sub-step, so finite differences cannot see it: grid-search it.
    i_lat = PARAMS.index("latency")
    grid = []
    for lat in np.arange(0.0, 0.161, 0.01):
        x_try = x.copy()
        x_try[i_lat] = lat
        grid.append((float(np.sum(residuals(x_try, data) ** 2)), float(lat)))
    x[i_lat] = min(grid)[1]
    after = metrics(x, data)
    fit.x = x
    result = {"params": dict(zip(PARAMS, [round(float(v), 4) for v in fit.x])),
              "old_model_approx": before_legacy, "px4_default_gains": before_default, "fitted": after,
              "latency_grid_sse": grid, "cost": float(fit.cost), "nfev": int(fit.nfev)}
    json.dump(result, open(args.out, "w"), indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
