"""
uav_world_env.py
----------------
UAVWorldNav-v0: 3-D navigation in procedurally generated forests (and urban /
warehouse / pole worlds) with a multi-channel LiDAR, PX4-calibrated dynamics,
optional moving obstacles (M6) and domain randomization (M7).

Observation (50,):
  [0:18]   proprioception, identical to the 18-D interface (goal & velocity in body
           frame, body rates, attitude quaternion, altitude, previous action)
  [18:26]  LiDAR level band (-10..+10 deg), min range per 45 deg sector   (shield-compatible)
  [26:34]  LiDAR down band  (-30, -20 deg)
  [34:42]  LiDAR up band    (+20, +30 deg)
  [42:50]  range rate of the level band (m/s; negative = closing), for moving obstacles
Action (4,): heading-frame [vx, vy, vz, yaw_rate] in [-1, 1] (same as M2-M5).
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from rl_env.action import ActionHandler
from rl_env.config import UAVEnvConfig
from rl_env.dynamics import FastUAVDynamics
from rl_env.lidar3d import DEFAULT_SPEC, Lidar3DSpec, add_noise, features, scan
from rl_env.observation import ObservationHandler, quat_to_rotation_matrix
from rl_env.world3d import (World3D, free_space_connected, generate_forest, generate_poles, generate_urban,
                            generate_warehouse)

OBS_DIM = 50


@dataclass
class UAVWorldEnvConfig(UAVEnvConfig):
    world_kinds: Tuple[str, ...] = ("forest",)
    forest_presets: Tuple[str, ...] = ("open", "medium", "dense")
    half_size: float = 20.0
    goal_dist_range: Tuple[float, float] = (8.0, 20.0)
    goal_alt_range: Tuple[float, float] = (1.6, 3.4)
    start_clear_radius: float = 1.8
    goal_clear_radius: float = 1.2
    max_episode_steps: int = 450
    uav_radius: float = 0.35
    hull_center_dz: float = 0.24           # body centre above the reference point (x500 base_link)
    ceiling: float = 8.0
    goal_tolerance_dist: float = 0.5
    goal_tolerance_vel: float = 0.6
    check_feasibility: bool = True
    lidar_noise_std: float = 0.02
    lidar_dropout: float = 0.0
    # moving obstacles (M6)
    n_movers_range: Tuple[int, int] = (0, 0)
    mover_kinds: Tuple[str, ...] = ("walker", "flyer")
    walker_speed: Tuple[float, float] = (0.6, 1.6)
    flyer_speed: Tuple[float, float] = (0.8, 2.5)
    # domain randomization (M7)
    domain_randomization: bool = False
    # rewards
    reward_progress_weight: float = 2.5
    reward_success: float = 100.0
    reward_collision_penalty: float = 50.0
    reward_time_penalty: float = 0.02
    reward_action_smoothness: float = 0.05
    reward_clearance_weight: float = 0.3
    clearance_safe_dist: float = 1.0
    altitude_band: Tuple[float, float] = (0.8, 5.0)
    reward_altitude_weight: float = 0.05


class Movers:
    """Moving obstacles: walkers (people-sized cylinders on the ground) and flyers (spheres in the air)."""

    def __init__(self, rng: np.random.Generator, cfg: UAVWorldEnvConfig, n: int, avoid_xy: np.ndarray):
        self.rng, self.cfg = rng, cfg
        half = cfg.half_size - 1.0
        self.kind, self.pos, self.goal, self.speed, self.radius = [], [], [], [], []
        for _ in range(n):
            kind = str(rng.choice(cfg.mover_kinds))
            for _ in range(50):
                xy = rng.uniform(-half, half, size=2)
                if np.linalg.norm(xy - avoid_xy) > 5.0:
                    break
            z = 0.0 if kind == "walker" else rng.uniform(1.5, 4.0)
            self.kind.append(kind)
            self.pos.append(np.array([xy[0], xy[1], z]))
            self.goal.append(self._new_goal(kind))
            lo, hi = cfg.walker_speed if kind == "walker" else cfg.flyer_speed
            self.speed.append(rng.uniform(lo, hi))
            self.radius.append(rng.uniform(0.25, 0.35) if kind == "walker" else rng.uniform(0.3, 0.5))

    def _new_goal(self, kind):
        half = self.cfg.half_size - 1.0
        xy = self.rng.uniform(-half, half, size=2)
        z = 0.0 if kind == "walker" else self.rng.uniform(1.5, 4.0)
        return np.array([xy[0], xy[1], z])

    def step(self, dt: float) -> None:
        for i in range(len(self.pos)):
            d = self.goal[i] - self.pos[i]
            dist = np.linalg.norm(d)
            if dist < 0.3:
                self.goal[i] = self._new_goal(self.kind[i])
                continue
            self.pos[i] = self.pos[i] + d / dist * min(self.speed[i] * dt, dist)

    def velocities(self) -> np.ndarray:
        out = []
        for i in range(len(self.pos)):
            d = self.goal[i] - self.pos[i]
            n = np.linalg.norm(d)
            out.append(d / n * self.speed[i] if n > 1e-6 else np.zeros(3))
        return np.array(out).reshape(-1, 3)

    def primitives(self) -> Tuple[np.ndarray, np.ndarray]:
        cyl, sph = [], []
        for k, p, r in zip(self.kind, self.pos, self.radius):
            if k == "walker":
                cyl.append((p[0], p[1], r, 0.0, 1.8))
            else:
                sph.append((p[0], p[1], p[2], r))
        return np.array(cyl).reshape(-1, 5), np.array(sph).reshape(-1, 4)


class UAVWorldEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 10}

    def __init__(self, config: Optional[UAVWorldEnvConfig] = None, lidar: Lidar3DSpec = DEFAULT_SPEC):
        super().__init__()
        self.config = config if config is not None else UAVWorldEnvConfig()
        self.lidar = lidar
        self._dirs_body = lidar.dirs_body()
        self.action_handler = ActionHandler(self.config)
        self.obs18 = ObservationHandler(self.config)
        self.dynamics = FastUAVDynamics(self.config)
        self.action_space = self.action_handler.action_space
        low = np.concatenate([self.obs18.observation_space.low, np.zeros(24), np.full(8, -5.0)]).astype(np.float32)
        high = np.concatenate([self.obs18.observation_space.high, np.full(24, lidar.max_range),
                               np.full(8, 5.0)]).astype(np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.world: Optional[World3D] = None
        self.movers: Optional[Movers] = None
        self.last_obs = None
        self.last_meta = None
        self.last_pos = np.zeros(3, dtype=np.float32)
        self.dr_params: Dict[str, float] = {}

    def set_curriculum(self, world_kinds=None, forest_presets=None, n_movers_range=None,
                       domain_randomization=None) -> None:
        """Change the scenario distribution between episodes (used by training curricula)."""
        if world_kinds is not None:
            self.config.world_kinds = tuple(world_kinds)
        if forest_presets is not None:
            self.config.forest_presets = tuple(forest_presets)
        if n_movers_range is not None:
            self.config.n_movers_range = tuple(n_movers_range)
        if domain_randomization is not None:
            self.config.domain_randomization = bool(domain_randomization)

    # ------------------------------------------------------------------ reset
    def _make_world(self, kind: str, preset: str, keep_out) -> World3D:
        rng, h = self.np_random, self.config.half_size
        if kind == "forest":
            return generate_forest(rng, h, preset=preset, keep_out=keep_out)
        if kind == "urban":
            return generate_urban(rng, h, keep_out=keep_out)
        if kind == "warehouse":
            return generate_warehouse(rng, h, keep_out=keep_out)
        if kind == "poles":
            return generate_poles(rng, h, n=int(rng.integers(6, 12)), keep_out=keep_out)
        return World3D.empty(h)

    def _apply_domain_randomization(self) -> None:
        rng, dyn, cfg = self.np_random, self.dynamics, self.config
        if not cfg.domain_randomization:
            self.dr_params = {}
            self.lidar_noise, self.lidar_dropout = cfg.lidar_noise_std, cfg.lidar_dropout
            for name in ("tau_velocity_xy", "tau_velocity_z", "tau_attitude", "tau_thrust", "drag_xy",
                         "drag_z", "thrust_to_weight"):
                setattr(dyn, name, float(getattr(cfg, {"drag_xy": "drag_coeff_xy",
                                                       "drag_z": "drag_coeff_z"}.get(name, name))))
            dyn.set_latency(cfg.action_latency_s)
            dyn.wind_velocity = np.zeros(3)
            dyn.wind_disturbance_std = cfg.wind_disturbance_std
            return
        # M7: mass +/-20 % (thrust-to-weight and drag per unit mass), inertia +/-30 % (attitude lag),
        # motor lag, controller tuning, latency 20-140 ms, sensor noise / dropout, steady wind + gusts.
        mass_scale = rng.uniform(0.8, 1.2)
        inertia_scale = rng.uniform(0.7, 1.3)
        p = {
            "mass_scale": mass_scale,
            "thrust_to_weight": cfg.thrust_to_weight / mass_scale,
            "tau_attitude": cfg.tau_attitude * inertia_scale,
            "tau_thrust": rng.uniform(0.010, 0.060),
            "tau_velocity_xy": cfg.tau_velocity_xy * rng.uniform(0.8, 1.25),
            "tau_velocity_z": cfg.tau_velocity_z * rng.uniform(0.8, 1.25),
            "drag_xy": cfg.drag_coeff_xy * rng.uniform(0.5, 1.5) / mass_scale,
            "drag_z": cfg.drag_coeff_z * rng.uniform(0.5, 1.5) / mass_scale,
            "latency_s": rng.uniform(0.02, 0.14),
            "noise_pos_std": rng.uniform(0.01, 0.08),
            "noise_vel_std": rng.uniform(0.02, 0.12),
            "lidar_noise_std": rng.uniform(0.01, 0.06),
            "lidar_dropout": rng.uniform(0.0, 0.10),
            "wind_speed": rng.uniform(0.0, 5.0),
            "gust_std": rng.uniform(0.05, 0.40),
        }
        for name in ("thrust_to_weight", "tau_attitude", "tau_thrust", "tau_velocity_xy", "tau_velocity_z",
                     "drag_xy", "drag_z", "noise_pos_std", "noise_vel_std"):
            setattr(dyn, name, float(p[name]))
        dyn.set_latency(p["latency_s"])
        wdir = rng.uniform(-math.pi, math.pi)
        dyn.wind_velocity = np.array([math.cos(wdir), math.sin(wdir), 0.0]) * p["wind_speed"]
        dyn.wind_disturbance_std = p["gust_std"]
        self.lidar_noise, self.lidar_dropout = p["lidar_noise_std"], p["lidar_dropout"]
        self.dr_params = p

    def _apply_overrides(self, ov: Dict[str, float]) -> None:
        """Pin individual physical parameters (M7 robustness sweeps). Keys:
        latency_s, wind_speed (m/s, random direction), gust_std, mass_scale, inertia_scale,
        tau_thrust, lidar_noise_std, lidar_dropout, noise_pos_std, noise_vel_std."""
        dyn, cfg, rng = self.dynamics, self.config, self.np_random
        if "latency_s" in ov:
            dyn.set_latency(float(ov["latency_s"]))
        if "wind_speed" in ov:
            ang = rng.uniform(-math.pi, math.pi)
            dyn.wind_velocity = np.array([math.cos(ang), math.sin(ang), 0.0]) * float(ov["wind_speed"])
        if "gust_std" in ov:
            dyn.wind_disturbance_std = float(ov["gust_std"])
        if "mass_scale" in ov:
            m = float(ov["mass_scale"])
            dyn.thrust_to_weight = cfg.thrust_to_weight / m
            dyn.drag_xy, dyn.drag_z = cfg.drag_coeff_xy / m, cfg.drag_coeff_z / m
        if "inertia_scale" in ov:
            dyn.tau_attitude = cfg.tau_attitude * float(ov["inertia_scale"])
        if "tau_thrust" in ov:
            dyn.tau_thrust = float(ov["tau_thrust"])
        for k in ("noise_pos_std", "noise_vel_std"):
            if k in ov:
                setattr(dyn, k, float(ov[k]))
        if "lidar_noise_std" in ov:
            self.lidar_noise = float(ov["lidar_noise_std"])
        if "lidar_dropout" in ov:
            self.lidar_dropout = float(ov["lidar_dropout"])
        self.dr_params = dict(self.dr_params, **{k: float(v) for k, v in ov.items()})

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        cfg, rng = self.config, self.np_random
        opts = options or {}
        self.dynamics.rng = rng
        kind = opts.get("world_kind") or str(rng.choice(cfg.world_kinds))
        preset = opts.get("forest_preset") or str(rng.choice(cfg.forest_presets))
        half = cfg.half_size
        for attempt in range(20):
            start = rng.uniform(-half + 3.0, half - 3.0, size=2)
            for _ in range(50):
                d = rng.uniform(*cfg.goal_dist_range)
                ang = rng.uniform(-math.pi, math.pi)
                goal_xy = start + d * np.array([math.cos(ang), math.sin(ang)])
                if np.all(np.abs(goal_xy) < half - 2.0):
                    break
            goal = np.array([goal_xy[0], goal_xy[1], rng.uniform(*cfg.goal_alt_range)])
            keep = [(start[0], start[1], cfg.start_clear_radius), (goal[0], goal[1], cfg.goal_clear_radius)]
            world = opts.get("world") or self._make_world(kind, preset, keep)
            if not cfg.check_feasibility or "world" in opts:
                break
            ok, _ = free_space_connected(world, start, goal_xy, inflate=cfg.uav_radius + 0.1)
            if ok and world.distance(goal[None])[0] > cfg.goal_clear_radius - 0.3:
                break
        self.world, self.kind, self.preset = world, kind, preset
        self._cull_cache = None
        self.feasibility_attempts = attempt + 1
        self.goal_world = goal.astype(np.float32)
        self._apply_domain_randomization()
        if "dynamics_overrides" in opts:
            self._apply_overrides(opts["dynamics_overrides"])
        yaw0 = float(opts.get("start_yaw", rng.uniform(-math.pi, math.pi)))
        self.dynamics.reset(x0=float(start[0]), y0=float(start[1]), z0=0.0, yaw0=yaw0)
        n_lo, n_hi = opts.get("n_movers_range", cfg.n_movers_range)
        n_movers = int(rng.integers(n_lo, n_hi + 1)) if n_hi > 0 else 0
        self.movers = Movers(rng, cfg, n_movers, start) if n_movers else None
        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        st = self.dynamics.get_state()
        self.last_pos = st.pos_world.copy()
        self.prev_dist = float(np.linalg.norm(st.pos_world - self.goal_world))
        self.initial_dist = self.prev_dist
        self.path_length = 0.0
        self.min_clearance = float("inf")
        self.hull_clearance = float("inf")
        self.prev_level = None
        obs = self._observe()
        return obs, self._info(st, reason="reset")

    # ------------------------------------------------------------- sensing
    def _world_now(self, center_xy, radius, z_window=None) -> World3D:
        # Re-cull the full world only when the vehicle has moved > 1 m from the last cull centre;
        # the cached set covers radius + 1.5 m, so it always contains everything within `radius`.
        cache = getattr(self, "_cull_cache", None)
        if cache is None or cache[1] < radius or np.hypot(*(np.asarray(center_xy) - cache[0])) > 1.0:
            cache = (np.array(center_xy, dtype=np.float64), radius + 1.5,
                     self.world.local(center_xy, radius + 1.5))
            self._cull_cache = cache
        local = cache[2].local(center_xy, radius, z_window)
        if self.movers is not None:
            cyl, sph = self.movers.primitives()
            local = local.with_extra(cyl, sph)
        return local

    def _observe(self) -> np.ndarray:
        st = self.dynamics.get_noisy_state() if self.config.enable_noise else self.dynamics.get_state()
        R = quat_to_rotation_matrix(st.quat).astype(np.float64)
        true = self.dynamics.state
        z = float(true.pos_world[2])
        lw = self._world_now(true.pos_world[:2], self.lidar.max_range + 0.5,
                             (z - 0.6 * self.lidar.max_range, z + 0.6 * self.lidar.max_range))
        ranges = scan(lw, true.pos_world, quat_to_rotation_matrix(true.quat).astype(np.float64), self.lidar,
                      self._dirs_body)
        ranges = add_noise(ranges, self.np_random, self.lidar_noise, self.lidar_dropout, self.lidar.max_range)
        feat = features(ranges, self.lidar)
        level = feat[:8]
        rate = np.zeros(8, dtype=np.float32) if self.prev_level is None else \
            np.clip((level - self.prev_level) / self.config.dt_policy, -5.0, 5.0)
        self.prev_level = level.copy()
        proprio = self.obs18.compute_observation(st.pos_world, st.vel_world, st.quat, st.body_rates,
                                                 self.goal_world, self.prev_action, ground_z=0.0)
        obs = np.concatenate([proprio, feat, rate]).astype(np.float32)
        obs = np.clip(np.nan_to_num(obs, nan=0.0), self.observation_space.low, self.observation_space.high)
        self.last_obs = obs.copy()
        self.last_ranges = ranges
        self.last_meta = {"is_sensor_stale": False, "sensor_age_s": 0.0}
        return obs

    # ---------------------------------------------------------------- step
    def step(self, action):
        cfg = self.config
        self.step_count += 1
        act = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float32).flatten()), -1.0, 1.0)
        prev_pos = self.dynamics.state.pos_world.astype(np.float64).copy()
        st = self.dynamics.step(self.action_handler.scale_action(act))
        if self.movers is not None:
            self.movers.step(cfg.dt_policy)
        pos = st.pos_world.astype(np.float64)
        self.path_length += float(np.linalg.norm(pos - prev_pos))
        self.last_pos = st.pos_world.copy()

        # 3-D collision check along the step (start, middle, end) against static + moving obstacles
        pts = np.stack([prev_pos + f * (pos - prev_pos) for f in (0.33, 0.67, 1.0)])
        pts[:, 2] += cfg.hull_center_dz
        near = self._world_now(pos[:2], 2.5)
        clearance = float(near.distance(pts).min()) if (len(near.cyl) + len(near.sph) + len(near.box)
                                                        + len(near.cap)) else 10.0
        hull_clearance = clearance - cfg.uav_radius
        self.min_clearance = min(self.min_clearance, hull_clearance)
        self.hull_clearance = hull_clearance
        collided = hull_clearance <= 0.0
        self.collision_with = self._nearest_tag(pts[-1]) if collided else None

        dist = float(np.linalg.norm(pos - self.goal_world))
        speed = float(np.linalg.norm(st.vel_world))
        out = abs(pos[0]) > cfg.half_size or abs(pos[1]) > cfg.half_size or pos[2] > cfg.ceiling
        ground_crash = st.ground_impact_speed > cfg.ground_crash_speed
        success = dist <= cfg.goal_tolerance_dist and speed <= cfg.goal_tolerance_vel and not collided
        timeout = self.step_count >= cfg.max_episode_steps

        obs = self._observe()
        level = obs[18:26]
        r = cfg.reward_progress_weight * (self.prev_dist - dist) - cfg.reward_time_penalty
        r -= cfg.reward_action_smoothness * float(np.sum((act - self.prev_action) ** 2))
        deficit = np.maximum(0.0, (cfg.clearance_safe_dist - level) / cfg.clearance_safe_dist)
        r -= cfg.reward_clearance_weight * float(np.sum(deficit ** 2))
        lo, hi = cfg.altitude_band
        if self.step_count > 20 and not (lo <= pos[2] <= hi):
            r -= cfg.reward_altitude_weight
        if success:
            r += cfg.reward_success
        if collided or out or ground_crash:
            r -= cfg.reward_collision_penalty
        self.prev_dist = dist
        self.prev_action = act.copy()

        reason = ("goal_reached" if success else "obstacle_collision" if collided else "ground_crash"
                  if ground_crash else "out_of_bounds" if out else "episode_timeout" if timeout else "active")
        terminated = bool(success or collided or out or ground_crash)
        truncated = bool(timeout and not terminated)
        return obs, float(r), terminated, truncated, self._info(st, reason)

    def _nearest_tag(self, point) -> str:
        """Semantic type of the obstacle closest to `point` (trunk, sapling, branch, foliage, bush, log, mover...)."""
        p = np.asarray(point, dtype=np.float64)
        w = self.world
        cands = []

        def tag_of(tags, i):
            return str(tags[i]) if tags is not None and len(tags) > i else "?"

        if len(w.cyl):
            c = w.cyl
            dxy = np.hypot(p[0] - c[:, 0], p[1] - c[:, 1]) - c[:, 2]
            dz = np.maximum(c[:, 3] - p[2], p[2] - c[:, 4])
            d = np.hypot(np.maximum(dxy, 0), np.maximum(dz, 0)) + np.minimum(np.maximum(dxy, dz), 0)
            i = int(np.argmin(d)); cands.append((float(d[i]), tag_of(w.cyl_tag, i)))
        if len(w.sph):
            d = np.linalg.norm(p - w.sph[:, :3], axis=1) - w.sph[:, 3]
            i = int(np.argmin(d)); cands.append((float(d[i]), tag_of(w.sph_tag, i)))
        if len(w.box):
            b = w.box
            cy, sy = np.cos(b[:, 6]), np.sin(b[:, 6])
            rx, ry = p[0] - b[:, 0], p[1] - b[:, 1]
            q = np.stack([np.abs(cy * rx + sy * ry) - b[:, 3], np.abs(-sy * rx + cy * ry) - b[:, 4],
                          np.abs(p[2] - b[:, 2]) - b[:, 5]], axis=1)
            d = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
            i = int(np.argmin(d)); cands.append((float(d[i]), tag_of(w.box_tag, i)))
        if len(w.cap):
            k = w.cap
            a, ba = k[:, :3], k[:, 3:6] - k[:, :3]
            h = np.clip(((p - a) * ba).sum(axis=1) / np.maximum((ba * ba).sum(axis=1), 1e-12), 0, 1)
            d = np.linalg.norm(p - a - h[:, None] * ba, axis=1) - k[:, 6]
            i = int(np.argmin(d)); cands.append((float(d[i]), tag_of(w.cap_tag, i)))
        if self.movers is not None and len(self.movers.pos):
            cyl, sph = self.movers.primitives()
            mw = World3D(cyl, sph, np.zeros((0, 7)), w.half_size)
            cands.append((float(mw.distance_numpy(p[None])[0]), "mover"))
        return min(cands)[1] if cands else "?"

    def _info(self, st, reason: str) -> Dict[str, Any]:
        return {"step": self.step_count, "termination_reason": reason, "is_success": reason == "goal_reached",
                "is_collision": reason == "obstacle_collision", "dist_to_goal": self.prev_dist,
                "path_length": self.path_length, "initial_dist": self.initial_dist,
                "min_hull_clearance_m": self.min_clearance,
                "hull_clearance_m": getattr(self, "hull_clearance", float("inf")),
                "world_kind": self.kind, "forest_preset": self.preset,
                "n_movers": 0 if self.movers is None else len(self.movers.pos),
                "feasibility_attempts": self.feasibility_attempts, "domain_randomization": dict(self.dr_params),
                "collision_with": getattr(self, "collision_with", None),
                "pos_world": st.pos_world.copy()}
