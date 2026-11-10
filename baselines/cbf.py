"""
cbf.py
------
Control Barrier Function (CBF) safety filter for ANY nominal action on UAVWorldNav-v0
(the RL policy, APF, DWA, ...), Milestone 9.

Information used -- ONBOARD SENSING ONLY: the observation (attitude obs[9:13], measured velocity
obs[3:6], altitude obs[13]) and the raw LiDAR grid (`ranges` argument or env.last_ranges);
without it, the 24 sector minima obs[18:42] placed at the sector centres. Returns are measured from
the collision-hull centre (env.config.hull_center_dz above pos_world), and the blind region below
the raised LiDAR is filled (common.onboard_points: beam_fill, blind_fill).

Model and constraints (heading frame, vehicle-centred; u = commanded velocity [vx, vy, vz]):
  * single integrator x' = u, the obstacle returns p_i are static;
  * barrier per return  h_i = |p_i^+| - d_safe,   d_safe = r_uav + safety_margin,
    evaluated at the predicted relative position p_i^+ = p_i - v_meas * prediction_s: the measured
    velocity carries the vehicle on for the transport delay and the velocity-loop lag before a
    new command acts (predictor-based CBF for input delay); prediction_s = 0 gives the plain CBF;
  * CBF condition dh_i/dt >= -alpha h_i with dh_i/dt = -n_i . u, n_i = p_i^+ / |p_i^+|, i.e. the
    linear constraint  n_i . u <= alpha h_i;
  * optional ground barrier on the altitude (u_z >= -alpha_z (z^+ - z_floor)), and the command
    limits (|u_z| <= v_z, |u_xy| <= v_xy as a circumscribed polygon).
Only returns whose constraint can bind (alpha h_i below the largest command speed) are kept, the
`max_constraints` most critical ones at most.
QP: min ||u - u_nom||^2 s.t. A u <= b, solved exactly as a least-distance program with the
Lawson-Hanson NNLS active-set method (scipy.optimize.nnls); `solver="slsqp"` uses
scipy.optimize.minimize(SLSQP) instead. A nominal action that satisfies every constraint is
returned unchanged. Infeasible QP (e.g. squeezed between returns inside d_safe): fall back to
braking (zero velocity command). The yaw-rate channel is passed through.
"""

from dataclasses import dataclass
import math
import time
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from scipy.optimize import minimize, nnls

from baselines.common import (VehicleLimits, action_to_velocity, frame_from_obs, hull_center_dz, make_config,
                              onboard_points)


@dataclass
class CBFConfig:
    uav_radius: float = 0.35
    safety_margin: float = 0.25          # d_safe = 0.60 m from the vehicle centre to a return
    alpha: float = 1.0                   # class-K gain (1/s)
    prediction_s: float = 0.4            # delay/lag compensation horizon (s); 0 = plain single integrator
    max_constraints: int = 200
    ground_barrier: bool = True
    z_floor: float = 0.3
    alpha_z: float = 1.0
    polygon_sides: int = 16
    ground_margin: float = 0.15
    beam_fill: bool = True               # see common.lidar_grid_points
    blind_fill: Tuple[float, ...] = (0.3, 0.0, -0.3)   # see common.onboard_points (hull-centre heights)
    solver: str = "ldp"                  # "ldp" (NNLS active set) or "slsqp"
    tol: float = 1e-6
    # M6 moving obstacles: per LiDAR sector, the obstacle's own radial velocity is estimated from the
    # measured range rate (obs[42:50]) plus the vehicle's radial velocity; an approaching obstacle
    # (estimate below -mover_deadband) tightens its barrier constraint:  n.u <= alpha h + v_rad.
    moving_obstacles: bool = False
    mover_deadband: float = 0.3          # m/s, ignores ego-motion/sector-switching noise
    mover_range_gate: float = 1.0        # m, only returns near the sector's closest range get its rate
    sensor_above_hull: float = 0.40      # LiDAR origin above the hull centre (0.64 - 0.24 m)

    @property
    def d_safe(self) -> float:
        return self.uav_radius + self.safety_margin


def solve_projection_qp(u_nom: np.ndarray, A: np.ndarray, b: np.ndarray, solver: str = "ldp",
                        tol: float = 1e-6) -> Tuple[np.ndarray, bool]:
    """
    min ||u - u_nom||^2  s.t.  A u <= b.  Returns (u, feasible).

    "ldp": with x = u - u_nom this is the least-distance program min ||x|| s.t. G x >= h, G = -A,
    h = A u_nom - b, solved via NNLS (Lawson & Hanson 1974, ch. 23): y = argmin_{y>=0} ||E y - f||
    with E = [G^T; h^T], f = e_{n+1}; r = E y - f; x = -r[:n] / r[n]; ||r|| = 0 <=> infeasible.
    """
    u_nom = np.asarray(u_nom, dtype=np.float64)
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if not len(b) or np.all(A @ u_nom <= b + tol):
        return u_nom.copy(), True
    n = len(u_nom)
    if solver == "slsqp":
        res = minimize(lambda u: float(np.sum((u - u_nom) ** 2)), np.zeros(n), jac=lambda u: 2.0 * (u - u_nom),
                       constraints=[{"type": "ineq", "fun": lambda u: b - A @ u, "jac": lambda u: -A}],
                       method="SLSQP", options={"ftol": 1e-12, "maxiter": 200})
        u = np.asarray(res.x, dtype=np.float64)
        return u, bool(res.success) and bool(np.all(A @ u <= b + 1e-5))
    h = A @ u_nom - b
    E = np.vstack([-A.T, h[None, :]])
    f = np.zeros(n + 1)
    f[-1] = 1.0
    y, _ = nnls(E, f, maxiter=50 * E.shape[1])
    r = E @ y - f
    if np.linalg.norm(r) < 1e-10 or abs(r[-1]) < 1e-12:
        return u_nom.copy(), False
    u = u_nom - r[:n] / r[-1]
    return u, bool(np.all(A @ u <= b + 1e-5))


class CBFSafetyFilter:
    """
    `filter_action(obs, action, ranges=None, env=None, perception_metadata=None) -> (action, info)`,
    same call style as rl_env.safety_shield.DeterministicSafetyShield. Onboard sensing only.
    """

    def __init__(self, config: Optional[CBFConfig] = None, env=None, **overrides):
        self.cfg = make_config(CBFConfig, config, **overrides)
        self.limits = VehicleLimits.from_env(env)
        self.hull_dz = hull_center_dz(env)
        k = np.arange(self.cfg.polygon_sides) * 2.0 * math.pi / self.cfg.polygon_sides
        # circumscribed polygon: every |u_xy| <= v_xy (what the env accepts) satisfies it
        self._poly = np.column_stack([np.cos(k), np.sin(k), np.zeros_like(k)])
        self.n_calls = 0
        self.n_interventions = 0
        self.n_infeasible = 0

    def constraints(self, obs: np.ndarray, points_heading: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Obstacle and ground rows (A, b) of the CBF condition, heading frame."""
        c = self.cfg
        obs = np.asarray(obs, dtype=np.float64)
        fr = frame_from_obs(obs)
        v_meas = fr.R_heading @ obs[3:6]
        u_bound = math.hypot(self.limits.v_xy, self.limits.v_z) + 1e-3
        rows, rhs = [np.zeros((0, 3))], [np.zeros(0)]
        min_h = math.inf
        if len(points_heading):
            pp = points_heading - v_meas * c.prediction_s
            d = np.linalg.norm(pp, axis=1)
            h = d - c.d_safe
            min_h = float(h.min())
            slack = c.alpha * h
            if c.moving_obstacles and len(obs) >= 50:
                slack = slack + self.approach_speed(obs, points_heading, pp / np.maximum(d, 1e-9)[:, None], v_meas)
            keep = np.nonzero(slack < u_bound)[0]
            if len(keep) > c.max_constraints:
                keep = keep[np.argsort(slack[keep])[:c.max_constraints]]
            rows.append(pp[keep] / np.maximum(d[keep], 1e-9)[:, None])
            rhs.append(slack[keep])
        alt = float(obs[13])
        if c.ground_barrier and alt > c.z_floor:
            hz = alt + v_meas[2] * c.prediction_s - c.z_floor
            if c.alpha_z * hz < self.limits.v_z + 1e-3:
                rows.append(np.array([[0.0, 0.0, -1.0]]))
                rhs.append(np.array([c.alpha_z * hz]))
        return np.vstack(rows), np.concatenate(rhs), {"min_h": min_h, "v_meas": v_meas}

    def approach_speed(self, obs: np.ndarray, pts: np.ndarray, n: np.ndarray, v_meas: np.ndarray) -> np.ndarray:
        """Per return: min(0, estimated radial velocity of the obstacle itself + deadband), heading frame.
        Range rate r' = n.(v_obs - v_ego)  =>  n.v_obs = r' + n.v_ego (8 level sectors, sector k centred at
        45k deg CCW from the nose; only returns within mover_range_gate of the sector's closest range)."""
        c = self.cfg
        rates = np.clip(np.asarray(obs[42:50], dtype=np.float64), -5.0, 5.0)
        level = np.asarray(obs[18:26], dtype=np.float64)
        k = (np.floor((np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) + 22.5) / 45.0).astype(int)) % 8
        sensor_range = np.linalg.norm(pts - np.array([0.0, 0.0, c.sensor_above_hull]), axis=1)
        v_rad = rates[k] + n @ v_meas
        gated = np.abs(sensor_range - level[k]) <= c.mover_range_gate
        return np.where(gated, np.minimum(0.0, v_rad + c.mover_deadband), 0.0)

    def filter_action(self, obs: np.ndarray, action: np.ndarray, ranges: Optional[np.ndarray] = None, env=None,
                      perception_metadata: Optional[Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        t0 = time.perf_counter()
        c, lim = self.cfg, self.limits
        self.n_calls += 1
        obs = np.asarray(obs, dtype=np.float64)
        raw = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float32).reshape(-1), nan=0.0, posinf=1.0,
                                    neginf=-1.0), -1.0, 1.0).astype(np.float32)
        if raw.shape[0] != 4:
            raise ValueError(f"Action dimension mismatch: expected 4, got {raw.shape[0]}")
        u_nom, _ = action_to_velocity(raw, lim)                # yaw rate is passed through
        fr = frame_from_obs(obs)
        if env is not None:
            self.hull_dz = hull_center_dz(env)
        pts, valid, source = onboard_points(obs, env, ranges=ranges, R=fr.R_heading, ground_margin=c.ground_margin,
                                            beam_fill=c.beam_fill, blind_fill=c.blind_fill, hull_dz=self.hull_dz)
        A_obs, b_obs, meta = self.constraints(obs, pts[valid])
        info: Dict[str, Any] = {"intervened": False, "infeasible": False, "fallback": None, "source": source,
                                "n_constraints": int(len(b_obs)), "n_active": 0, "min_h": meta["min_h"],
                                "u_nom": u_nom, "u_safe": u_nom, "solve_ms": 0.0}
        if isinstance(perception_metadata, dict):
            info["sensor_stale"] = bool(perception_metadata.get("is_sensor_stale", False))
        if not len(b_obs) or np.all(A_obs @ u_nom <= b_obs + c.tol):
            info["solve_ms"] = (time.perf_counter() - t0) * 1e3
            return raw.copy(), info

        A = np.vstack([A_obs, self._poly, [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])
        b = np.concatenate([b_obs, np.full(len(self._poly), lim.v_xy), [lim.v_z, lim.v_z]])
        u, ok = solve_projection_qp(u_nom, A, b, c.solver, c.tol)
        if not ok:
            u = np.zeros(3)                                  # brake
            info.update(infeasible=True, fallback="brake")
            self.n_infeasible += 1
        self.n_interventions += 1
        h = math.hypot(u[0], u[1])
        if h > lim.v_xy:
            u[:2] *= lim.v_xy / h
        u[2] = float(np.clip(u[2], -lim.v_z, lim.v_z))
        out = np.array([u[0] / lim.v_xy, u[1] / lim.v_xy, u[2] / lim.v_z, raw[3]], dtype=np.float64)
        info.update(intervened=True, u_safe=u, n_active=int(np.sum(A_obs @ u >= b_obs - 1e-4)),
                    solve_ms=(time.perf_counter() - t0) * 1e3)
        return np.clip(out, -1.0, 1.0).astype(np.float32), info


class CBFSafetyWrapper(gym.Wrapper):
    """Applies CBFSafetyFilter to every action of a UAVWorldNav-v0 env (e.g. around an RL policy)."""

    def __init__(self, env: gym.Env, cbf: Optional[CBFSafetyFilter] = None):
        super().__init__(env)
        self.cbf = cbf if cbf is not None else CBFSafetyFilter(env=env)
        self._obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs = np.asarray(obs, dtype=np.float32).copy()
        return obs, info

    def step(self, action):
        safe, cinfo = self.cbf.filter_action(self._obs, action, env=self.env)
        obs, reward, terminated, truncated, info = self.env.step(safe)
        self._obs = np.asarray(obs, dtype=np.float32).copy()
        info["cbf_intervened"] = cinfo["intervened"]
        info["cbf_infeasible"] = cinfo["infeasible"]
        info["cbf_action"] = safe
        return obs, reward, terminated, truncated, info
