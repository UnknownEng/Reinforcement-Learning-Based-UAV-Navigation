"""
dwa.py
------
Dynamic Window Approach (Fox, Burgard & Thrun 1997), holonomic 3-D variant for UAVWorldNav-v0
(Milestone 9).

Information used -- ONBOARD SENSING ONLY:
  * the observation: goal offset obs[0:3], measured velocity obs[3:6], attitude obs[9:13],
    altitude obs[13];
  * the raw LiDAR range grid env.last_ranges (7 x 90) when an env is passed to `act`, otherwise
    the 24 sector minima obs[18:42]. Returns become an obstacle point set in vehicle-centred,
    world-aligned axes (rotated by the attitude);
  * the vehicle's own nominal velocity-loop parameters (env.config: tau_velocity_xy/z,
    action_latency_s, tau_attitude) for the rollout model. No map, no absolute position.
Returns are measured from the collision-hull centre (env.config.hull_center_dz above pos_world), and
the blind region below the raised LiDAR is filled (common.onboard_points: beam_fill, blind_fill).

Each decision (10 Hz):
  1. Dynamic window: commands reachable from the measured velocity v_c within `window_s` at the
     acceleration limits, v_c +/- a_max * window_s per axis (horizontal a_max_xy, vertical a_max_z),
     intersected with the command limits, sampled on an n_xy x n_xy x n_z grid, plus the
     goal-directed and the zero command clipped into the window.
  2. Rollout over `horizon_s`: the command is held constant. By default the vehicle's response to it
     follows the first-order velocity loop with the PX4-fitted time constants and latency
     (x(t) from v_c toward the command); response_tau_xy = response_tau_z = latency_s = 0 gives
     ideal constant-velocity rollouts. (Without the lag model DWA trusts commands the vehicle cannot
     follow yet and runs into obstacles at speed.)
  3. Admissibility: the rollout must stay farther than r_uav + margin from every LiDAR return, and
     the vehicle must still be able to stop: the emergency-stop path starting after one decision
     period (velocity decays with the loop time constants; ideal mode: |v| t_react + |v|^2/2a_brake)
     must be collision-free too. Rollouts may not leave the hard altitude limits. Returns already
     inside the inflated radius (after a disturbance) only forbid approaching them further.
     Clearances come from a KD-tree over the returns, queried at rollout samples every rollout_dt
     (the threshold is raised by half the sample spacing so gaps between samples are covered).
  4. Score: w_goal * progress + w_clear * clearance + w_speed * speed - w_alt * band violation, where
     progress = reduction of the goal distance at the rollout end / (v_max T) and clearance = the
     rollout's minimum distance to the returns minus the inflation, saturated at clear_max.
  5. No admissible command: brake (zero velocity command, outside the window: emergency stop).
Recovery from DWA's local minima (like the recovery behaviours of ROS move_base): when the goal
distance stops shrinking, a temporary sub-goal `escape_dist` to the side of the obstacles replaces
the goal for `escape_steps`. It is stored in goal-relative coordinates (vehicle position relative to
the goal = -R obs[0:3]), so it needs no absolute positioning either.
Yaw: proportional toward the goal.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from baselines.common import (EpisodeTracker, ProgressMonitor, VehicleLimits, choose_escape_sign, escape_tangent,
                              frame_from_obs, hull_center_dz, make_config, onboard_points, unwrap,
                              velocity_to_action, yaw_rate_towards)


@dataclass
class DWAConfig:
    # dynamic window
    a_max_xy: float = 3.0            # m/s^2
    a_max_z: float = 2.0             # m/s^2
    window_s: float = 0.4            # s -> +/-1.2 m/s horizontally, +/-0.8 m/s vertically
    n_xy: int = 7
    n_z: int = 3
    # rollouts
    horizon_s: float = 1.5
    rollout_dt: float = 0.15
    decision_dt: float = 0.1
    response_tau_xy: Optional[float] = None   # None: env.config.tau_velocity_xy (0.602 s); 0: ideal
    response_tau_z: Optional[float] = None    # None: env.config.tau_velocity_z (0.166 s)
    latency_s: Optional[float] = None         # None: env action latency + attitude lag (0.18 s)
    brake_samples: int = 6
    t_react: float = 0.4             # ideal mode only: stopping distance |v| t_react + |v|^2 / (2 a_brake)
    a_brake: float = 2.5
    # safety
    uav_radius: float = 0.35
    margin: float = 0.25
    # objective
    w_goal: float = 1.0
    w_clear: float = 0.25
    w_speed: float = 0.05
    w_alt: float = 0.5
    clear_max: float = 1.5
    speed_bonus_min_dist: float = 2.0
    k_goal: float = 1.0              # gain of the extra goal-directed candidate (1/s)
    # altitude
    z_band: Tuple[float, float] = (1.0, 4.5)
    z_hard: Tuple[float, float] = (0.5, 5.5)
    takeoff_alt: float = 0.8
    # misc
    k_yaw: float = 1.5
    ground_margin: float = 0.15
    beam_fill: bool = True           # see common.lidar_grid_points
    blind_fill: Tuple[float, ...] = (0.3, 0.0, -0.3)   # see common.onboard_points (hull-centre heights)
    stuck_window: int = 30
    stuck_progress: float = 0.4
    escape_steps: int = 30
    escape_dist: float = 4.0
    seed: Optional[int] = None


class DWAController:
    """DWA policy: `act(obs, env=None) -> action (4,)` in [-1, 1]. Onboard sensing only."""

    name = "dwa"
    uses_map = False

    def __init__(self, env=None, config: Optional[DWAConfig] = None, **overrides):
        self.cfg = c = make_config(DWAConfig, config, **overrides)
        self.limits = VehicleLimits.from_env(env)
        ecfg = getattr(unwrap(env), "config", None)
        pick = lambda v, name, default: float(v) if v is not None else float(getattr(ecfg, name, default))
        self.tau = np.array([pick(c.response_tau_xy, "tau_velocity_xy", 0.602)] * 2
                            + [pick(c.response_tau_z, "tau_velocity_z", 0.166)])
        self.latency = (float(c.latency_s) if c.latency_s is not None else
                        float(getattr(ecfg, "action_latency_s", 0.08)) + float(getattr(ecfg, "tau_attitude", 0.10)))
        self._t = np.arange(1, int(round(c.horizon_s / c.rollout_dt)) + 1) * c.rollout_dt
        self.hull_dz = hull_center_dz(env)
        self._tracker = EpisodeTracker()
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        c = self.cfg
        self.rng = np.random.default_rng(c.seed if seed is None else seed)
        self.monitor = ProgressMonitor(c.stuck_window, c.stuck_progress)
        self._escape_left = 0
        self._escape_sign = 0
        self._escape_start_dist = math.inf
        self._subgoal = np.zeros(3)
        self.n_escapes = 0
        self.n_brakes = 0
        self.last_info: Dict[str, Any] = {}
        self._fresh = True          # reset() called since the last act()

    @property
    def r_infl(self) -> float:
        return self.cfg.uav_radius + self.cfg.margin

    # ------------------------------------------------------------- model
    def response(self, V: np.ndarray, v_c: np.ndarray, t: np.ndarray):
        """Positions (N, K, 3) and velocities (N, K, 3) at times t (K,) under the held commands V (N, 3)."""
        tp = np.maximum(t - self.latency, 0.0)[:, None]                     # (K, 1) time since it acts
        hold = np.minimum(t, self.latency)[:, None]
        tau = self.tau[None, :]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            decay = np.where(tau > 0, np.exp(-tp / np.where(tau > 0, tau, 1.0)), 0.0)   # (K, 3)
        f = np.where(tau > 0, tau * (1.0 - decay), 0.0)
        dv = (v_c[None, :] - V)[:, None, :]                                 # (N, 1, 3)
        pos = v_c[None, None, :] * hold[None] + V[:, None, :] * tp[None] + dv * f[None]
        vel = np.where((t[:, None] < self.latency)[None], v_c[None, None, :], V[:, None, :] + dv * decay[None])
        return pos, vel

    def candidates(self, v_c: np.ndarray, target: np.ndarray, alt: float) -> np.ndarray:
        """Velocity samples (N, 3), local frame, inside the dynamic window and the command limits."""
        c, lim = self.cfg, self.limits
        dv, dvz = c.a_max_xy * c.window_s, c.a_max_z * c.window_s
        lo = np.array([max(v_c[0] - dv, -lim.v_xy), max(v_c[1] - dv, -lim.v_xy), max(v_c[2] - dvz, -lim.v_z)])
        hi = np.array([min(v_c[0] + dv, lim.v_xy), min(v_c[1] + dv, lim.v_xy), min(v_c[2] + dvz, lim.v_z)])
        hi = np.maximum(hi, lo)
        X, Y, Z = np.meshgrid(np.linspace(lo[0], hi[0], c.n_xy), np.linspace(lo[1], hi[1], c.n_xy),
                              np.linspace(lo[2], hi[2], c.n_z), indexing="ij")
        g = c.k_goal * np.asarray(target, dtype=np.float64)
        gh = math.hypot(g[0], g[1])
        if gh > lim.v_xy:
            g[:2] *= lim.v_xy / gh
        g[2] = float(np.clip(g[2], -lim.v_z, lim.v_z))
        V = np.vstack([np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1),
                       np.clip(np.stack([g, np.zeros(3)]), lo, hi)])
        h = np.hypot(V[:, 0], V[:, 1])
        V[:, :2] *= np.minimum(1.0, lim.v_xy / np.maximum(h, 1e-9))[:, None]
        if alt < c.takeoff_alt:                         # climb before translating
            V[:, :2] *= float(np.clip((alt - 0.2) / max(c.takeoff_alt - 0.2, 1e-3), 0.0, 1.0))
        return V

    # ----------------------------------------------------------- scoring
    def evaluate(self, V: np.ndarray, P: np.ndarray, v_c: np.ndarray, target: np.ndarray,
                 alt: float) -> Dict[str, np.ndarray]:
        """Admissibility and score of the candidates V (N, 3) given obstacle points P (M, 3), local frame."""
        c, lim = self.cfg, self.limits
        r_infl, n, T = self.r_infl, len(V), float(self._t[-1])
        pos, _ = self.response(V, v_c, self._t)                               # (N, K, 3)
        p1, v1 = self.response(V, v_c, np.array([c.decision_dt]))
        p1, v1 = p1[:, 0], v1[:, 0]
        if np.all(self.tau > 0):                                              # emergency stop after one period
            b_end = p1 + v1 * (self.latency + self.tau)[None, :]
        else:
            s1 = np.linalg.norm(v1, axis=1)
            s_stop = s1 * c.t_react + s1 ** 2 / (2.0 * c.a_brake)
            b_end = p1 + v1 / np.maximum(s1, 1e-9)[:, None] * s_stop[:, None]
        u = np.linspace(0.0, 1.0, c.brake_samples)
        brake = p1[:, None, :] + u[None, :, None] * (b_end - p1)[:, None, :]  # (N, Kb, 3)

        gaps = np.linalg.norm(np.diff(np.concatenate([np.zeros((n, 1, 3)), pos], axis=1), axis=1), axis=2).max(axis=1)
        r_roll = np.sqrt(r_infl ** 2 + 0.25 * gaps ** 2)
        r_brake = np.sqrt(r_infl ** 2 + 0.25 * (np.linalg.norm(b_end - p1, axis=1) / (c.brake_samples - 1)) ** 2)
        d_roll = np.full(n, np.inf)
        d_brake = np.full(n, np.inf)
        ok_inside = np.ones(n, dtype=bool)
        if len(P):
            pp = np.einsum("ij,ij->i", P, P)
            inside = pp < r_infl ** 2
            if not np.all(inside):
                tree = cKDTree(P[~inside], balanced_tree=False, compact_nodes=False)
                q = np.concatenate([pos.reshape(-1, 3), brake.reshape(-1, 3)])
                d, _ = tree.query(q, k=1, distance_upper_bound=r_infl + c.clear_max + 0.05)
                k = pos.shape[1]
                d_roll = d[:n * k].reshape(n, k).min(axis=1)
                d_brake = d[n * k:].reshape(n, -1).min(axis=1)
            if np.any(inside):
                nin = P[inside] / np.sqrt(np.maximum(pp[inside], 1e-12))[:, None]
                ok_inside = (V @ nin.T <= 0.05).all(axis=1)
        clearance = np.clip(d_roll - r_infl, 0.0, c.clear_max)
        z = alt + pos[:, :, 2]
        z_ok = (z.min(axis=1) >= min(alt, c.z_hard[0])) & (z.max(axis=1) <= max(alt, c.z_hard[1]))
        admissible = (d_roll > r_roll) & (d_brake > r_brake) & ok_inside & z_ok
        z_end = z[:, -1]
        band_violation = np.maximum(0.0, c.z_band[0] - z_end) + np.maximum(0.0, z_end - c.z_band[1])
        d0 = float(np.linalg.norm(target))
        progress = (d0 - np.linalg.norm(target[None, :] - pos[:, -1], axis=1)) / (lim.v_xy * T)
        speed = np.hypot(V[:, 0], V[:, 1]) / lim.v_xy if d0 > c.speed_bonus_min_dist else np.zeros(n)
        score = (c.w_goal * progress + c.w_clear * clearance / c.clear_max + c.w_speed * speed
                 - c.w_alt * band_violation)
        return {"score": score, "admissible": admissible, "clearance": clearance, "progress": progress,
                "rollout": pos}

    def _away_direction(self, P: np.ndarray) -> np.ndarray:
        """Horizontal direction pointing away from the nearby returns (1/clearance weighted)."""
        if not len(P):
            return np.zeros(2)
        r = np.linalg.norm(P, axis=1)
        near = r < 3.0
        if not np.any(near):
            return np.zeros(2)
        w = 1.0 / np.maximum(r[near] - self.cfg.uav_radius, 0.05)
        return -((P[near, :2] / r[near][:, None]) * w[:, None]).sum(axis=0)

    # ---------------------------------------------------------------- act
    def act(self, obs: np.ndarray, env=None) -> np.ndarray:
        if env is not None:
            self.hull_dz = hull_center_dz(env)
            if self._tracker.new_episode(env) and not self._fresh:     # env reset without reset()
                self.reset()
        self._fresh = False
        c, lim = self.cfg, self.limits
        obs = np.asarray(obs, dtype=np.float64)
        fr = frame_from_obs(obs)
        g = fr.R @ obs[0:3]
        v_c = fr.R @ obs[3:6]
        alt = float(obs[13])
        dist = float(np.linalg.norm(g))
        pts, valid, source = onboard_points(obs, env, R=fr.R, ground_margin=c.ground_margin, beam_fill=c.beam_fill,
                                            blind_fill=c.blind_fill, hull_dz=self.hull_dz)
        P = pts[valid]
        reach = math.hypot(lim.v_xy, lim.v_z) * c.horizon_s + self.r_infl + c.clear_max + 0.5
        if len(P):
            P = P[np.einsum("ij,ij->i", P, P) < reach * reach]

        # ---- recovery sub-goal (goal-relative coordinates: vehicle at x_v = -g)
        x_v = -g
        if self._escape_left == 0 and self.monitor.update(dist):
            away = self._away_direction(P)
            if self._escape_sign != 0 and dist > self._escape_start_dist - 0.5:
                self._escape_sign = -self._escape_sign
            else:
                self._escape_sign = choose_escape_sign(away, g[:2], self.rng)
            t = escape_tangent(away, g[:2], self._escape_sign)
            z_goal = float(np.clip(alt, *c.z_band)) - alt
            self._subgoal = x_v + np.array([t[0] * c.escape_dist, t[1] * c.escape_dist, z_goal])
            self._escape_start_dist = dist
            self._escape_left = c.escape_steps
            self.n_escapes += 1
        target = g
        escaping = self._escape_left > 0
        if escaping:
            target = self._subgoal - x_v
            self._escape_left -= 1
            if np.linalg.norm(target) < 0.7:
                self._escape_left = 0
            if self._escape_left == 0:
                self.monitor.reset()

        V = self.candidates(v_c, target, alt)
        ev = self.evaluate(V, P, v_c, target, alt)
        adm = ev["admissible"]
        if np.any(adm):
            i = int(np.argmax(np.where(adm, ev["score"], -np.inf)))
            v = V[i].copy()
            braking = False
        else:
            i, v, braking = -1, np.zeros(3), True
            self.n_brakes += 1
        yaw_rate = yaw_rate_towards(g, fr, lim, c.k_yaw)
        self.last_info = {"source": source, "n_candidates": len(V), "n_admissible": int(adm.sum()),
                          "n_points": len(P), "braking": braking, "escaping": escaping, "v_cmd": v.copy(),
                          "score": float(ev["score"][i]) if i >= 0 else None}
        return velocity_to_action(v, yaw_rate, fr, lim)
