"""
apf.py
------
Artificial Potential Field (APF) baseline for UAVWorldNav-v0 (Milestone 9).

Information used -- ONBOARD SENSING ONLY:
  * the observation: goal offset in the body frame obs[0:3], attitude quaternion obs[9:13],
    altitude obs[13];
  * the raw LiDAR range grid env.last_ranges (7 x 90) when an env is passed to `act`, otherwise
    the 24 sector minima obs[18:42].
No map, no absolute position, no knowledge of moving obstacles beyond what the LiDAR returns.
Returns are measured from the collision-hull centre (env.config.hull_center_dz above pos_world), and
the blind region below the raised LiDAR is filled (common.onboard_points: beam_fill, blind_fill).

Velocity field, in vehicle-centred world-aligned axes (g = goal offset, p_j = LiDAR return):
  v_att = k_att * g, saturated at the command limits (conic well beyond v_max / k_att)
  v_rep = s_goal * sum_j eta * (1/c_j - 1/rho0) / c_j^2 * (-p_j / |p_j|)      for c_j < rho0
          Khatib's repulsive gradient, one term per azimuth column j of the LiDAR (its closest
          return, so a tall trunk and a short stump weigh alike), c_j = |p_j| - r_uav is the hull
          clearance; s_goal = min(1, |g| / goal_fade_dist) fades repulsion at the goal (Ge & Cui 2000,
          "goals non-reachable with obstacles nearby").
  v = v_att + v_rep (+ v_escape), then
    * speed limit from the clearance ahead (|v| <= k_speed (c_front - c_stop)): a standard addition
      for real vehicles, since the PX4 velocity loop needs ~2 m to stop from 3 m/s;
    * vertical: altitude kept in [z_floor, z_ceil]; horizontal motion suppressed below takeoff_alt.
  Yaw: proportional toward the goal (the LiDAR is 360 deg, yaw only affects the logged attitude).

Local minima: when the goal distance shrinks by less than `stuck_progress` m over `stuck_window`
steps, or the field stalls (|v| < stall_speed),
  * escape (standard tangential + random perturbation): for `escape_steps` steps the controller
    circulates around the obstacles with a tangential velocity (perpendicular to the repulsive
    force, on the goal's side first, the other side after a failed attempt) plus a random horizontal
    perturbation, with the attraction halved;
  * virtual obstacle (Park & Lee 2003): a repulsive marker is left where the vehicle got stuck, so
    the minimum is filled in and not revisited. Markers live in goal-relative coordinates (vehicle
    position relative to the goal = -R obs[0:3]), i.e. they need no absolute positioning either.
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from baselines.common import (EpisodeTracker, ProgressMonitor, VehicleLimits, choose_escape_sign, escape_tangent,
                              frame_from_obs, hull_center_dz, make_config, onboard_points, velocity_to_action,
                              vertical_limits, yaw_rate_towards)


@dataclass
class APFConfig:
    # attraction
    k_att: float = 1.0               # 1/s, horizontal
    k_att_z: float = 1.0             # 1/s, vertical
    # repulsion
    uav_radius: float = 0.35
    rho0: float = 1.6                # influence distance (hull clearance, m)
    eta: float = 0.25                # repulsive gain per LiDAR column (m^3/s)
    c_floor: float = 0.05            # clearance floor inside the 1/c terms (m)
    rep_max: float = 6.0             # cap on |v_rep| (m/s)
    rep_z_scale: float = 0.5         # vertical share of the repulsion
    goal_fade_dist: float = 1.5      # repulsion fades linearly inside this goal distance
    ground_margin: float = 0.15      # returns below this height are treated as ground
    beam_fill: bool = True           # see common.lidar_grid_points
    blind_fill: Tuple[float, ...] = (0.3, 0.0, -0.3)   # see common.onboard_points (hull-centre heights)
    # speed scaling from the clearance ahead
    speed_scaling: bool = True
    k_speed: float = 1.2             # 1/s
    c_stop: float = 0.3              # m
    v_min: float = 0.3               # m/s, floor of the clearance speed limit
    front_cone_deg: float = 50.0
    # altitude
    z_floor: float = 0.6
    z_ceil: float = 5.0
    k_alt: float = 1.0
    takeoff_alt: float = 0.8
    # yaw
    k_yaw: float = 1.5
    # local-minimum escape
    stuck_window: int = 25
    stuck_progress: float = 0.35
    stall_speed: float = 0.25
    stall_steps: int = 8
    escape_steps: int = 25
    escape_speed: float = 1.5
    escape_noise: float = 0.3
    noise_hold_steps: int = 5
    virtual_gain: float = 1.5        # m/s at a virtual obstacle, fading linearly to 0 at virtual_radius
    virtual_radius: float = 2.5      # m
    max_virtual: int = 40
    seed: Optional[int] = None


class APFController:
    """APF policy: `act(obs, env=None) -> action (4,)` in [-1, 1]. Onboard sensing only."""

    name = "apf"
    uses_map = False

    def __init__(self, env=None, config: Optional[APFConfig] = None, **overrides):
        self.cfg = make_config(APFConfig, config, **overrides)
        self.limits = VehicleLimits.from_env(env)
        self.hull_dz = hull_center_dz(env)
        self._tracker = EpisodeTracker()
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        c = self.cfg
        self.rng = np.random.default_rng(c.seed if seed is None else seed)
        self.monitor = ProgressMonitor(c.stuck_window, c.stuck_progress)
        self._stall = 0
        self._escape_left = 0
        self._escape_sign = 0
        self._escape_start_dist = math.inf
        self._noise = np.zeros(3)
        self._virtual = np.zeros((0, 3))
        self.n_escapes = 0
        self.last_info: Dict[str, Any] = {}
        self._fresh = True          # reset() called since the last act()

    # --------------------------------------------------------------- forces
    def repulsion(self, pts: np.ndarray, valid: np.ndarray):
        """Repulsive velocity (3,), per-column clearances (n_cols,) and column points (n_cols, 3)."""
        c = self.cfg
        r = np.linalg.norm(pts, axis=-1)
        clr = np.where(valid, r - c.uav_radius, np.inf)
        rows = np.argmin(clr, axis=0)
        cols = np.arange(clr.shape[1])
        c_col, p_col, r_col = clr[rows, cols], pts[rows, cols], r[rows, cols]
        near = c_col < c.rho0
        v_rep = np.zeros(3)
        if np.any(near):
            cc = np.maximum(c_col[near], c.c_floor)
            mag = c.eta * (1.0 / cc - 1.0 / c.rho0) / cc ** 2
            u = p_col[near] / np.maximum(r_col[near], 1e-6)[:, None]
            v_rep = -(mag[:, None] * u).sum(axis=0)
            v_rep[2] *= c.rep_z_scale
            n = float(np.linalg.norm(v_rep))
            if n > c.rep_max:
                v_rep *= c.rep_max / n
        return v_rep, c_col, p_col

    def _virtual_repulsion(self, x_v: np.ndarray) -> np.ndarray:
        """Horizontal push away from the virtual obstacles (goal-relative vehicle position x_v)."""
        c = self.cfg
        if not len(self._virtual):
            return np.zeros(3)
        rel = x_v[None, :2] - self._virtual[:, :2]
        d = np.hypot(rel[:, 0], rel[:, 1])
        m = (d < c.virtual_radius) & (d > 1e-3)
        if not np.any(m):
            return np.zeros(3)
        w = c.virtual_gain * (1.0 - d[m] / c.virtual_radius) / d[m]
        out = np.zeros(3)
        out[:2] = (rel[m] * w[:, None]).sum(axis=0)
        return out

    def _attraction(self, g: np.ndarray) -> np.ndarray:
        c, lim = self.cfg, self.limits
        v = np.array([c.k_att * g[0], c.k_att * g[1], c.k_att_z * g[2]])
        h = math.hypot(v[0], v[1])
        if h > lim.v_xy:
            v[:2] *= lim.v_xy / h
        v[2] = float(np.clip(v[2], -lim.v_z, lim.v_z))
        return v

    # ------------------------------------------------------------------ act
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
        dist = float(np.linalg.norm(g))
        alt = float(obs[13])
        pts, valid, source = onboard_points(obs, env, R=fr.R, ground_margin=c.ground_margin, beam_fill=c.beam_fill,
                                            blind_fill=c.blind_fill, hull_dz=self.hull_dz)

        v_att = self._attraction(g)
        v_rep, c_col, p_col = self.repulsion(pts, valid)
        v_rep *= min(1.0, dist / c.goal_fade_dist)
        v_virt = self._virtual_repulsion(-g)
        v = v_att + v_rep + v_virt

        # ---- local-minimum detection and escape
        stuck = self.monitor.update(dist)
        self._stall = self._stall + 1 if (float(np.linalg.norm(v)) < c.stall_speed and dist > 1.0) else 0
        if self._escape_left == 0 and dist > 1.0 and (stuck or self._stall >= c.stall_steps):
            if self._escape_sign != 0 and dist > self._escape_start_dist - 0.5:
                self._escape_sign = -self._escape_sign          # previous attempt failed: other side
            else:
                self._escape_sign = choose_escape_sign(v_rep[:2], g[:2], self.rng)
            self._escape_start_dist = dist
            self._escape_left = c.escape_steps
            self._stall = 0
            self.n_escapes += 1
            self._virtual = np.vstack([self._virtual, -g[None, :]])[-c.max_virtual:]
        escaping = self._escape_left > 0
        if escaping:
            if (c.escape_steps - self._escape_left) % c.noise_hold_steps == 0:
                ang = self.rng.uniform(-math.pi, math.pi)
                self._noise = c.escape_noise * lim.v_xy * np.array([math.cos(ang), math.sin(ang), 0.0])
            t = escape_tangent(v_rep[:2], g[:2], self._escape_sign)
            v = 0.5 * v_att + v_rep + v_virt + np.array([t[0], t[1], 0.0]) * c.escape_speed + self._noise
            self._escape_left -= 1
            if self._escape_left == 0:
                self.monitor.reset()

        # ---- speed limit from the clearance ahead
        speed = float(np.linalg.norm(v))
        c_front = math.inf
        if c.speed_scaling and speed > 1e-6 and np.any(np.isfinite(c_col)):
            fin = np.isfinite(c_col)
            dirs = p_col[fin] / np.maximum(np.linalg.norm(p_col[fin], axis=1), 1e-6)[:, None]
            ahead = dirs @ (v / speed) > math.cos(math.radians(c.front_cone_deg))
            if np.any(ahead):
                c_front = float(c_col[fin][ahead].min())
                cap = float(np.clip(c.k_speed * (c_front - c.c_stop), c.v_min, lim.v_xy))
                if speed > cap:
                    v *= cap / speed

        # ---- altitude handling
        if alt < c.takeoff_alt:
            v[:2] *= float(np.clip((alt - 0.2) / max(c.takeoff_alt - 0.2, 1e-3), 0.0, 1.0))
            if g[2] > 0.0:
                v[2] = max(v[2], 0.8)
        vz_lo, vz_hi = vertical_limits(alt, c.z_floor, c.z_ceil, c.k_alt)
        v[2] = float(np.clip(v[2], vz_lo, max(vz_hi, vz_lo)))

        yaw_rate = yaw_rate_towards(g, fr, lim, c.k_yaw)
        self.last_info = {"source": source, "v_att": v_att, "v_rep": v_rep, "v_virtual": v_virt, "v_cmd": v.copy(),
                          "escaping": escaping, "c_front": c_front, "dist": dist}
        return velocity_to_action(v, yaw_rate, fr, lim)
