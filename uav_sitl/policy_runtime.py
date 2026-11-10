"""
policy_runtime.py
-----------------
Runs a trained policy in PX4 SITL with the *same* observation code used in training:

  18d  open-space interface (M2/M3)               rl_env.observation.ObservationHandler
  26d  planar 72-ray LiDAR (M4/M5)                rl_env.observation_26d.ObservationHandler26D
  50d  7-channel LiDAR (forest/urban/warehouse)   rl_env.lidar3d features + range rates

State comes from PX4's own estimator (LOCAL_POSITION_NED / ATTITUDE), as it would on a
real vehicle; the LiDAR comes from Gazebo. One policy step is taken per new scan (10 Hz).
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from stable_baselines3 import PPO

from rl_env.action import ActionHandler
from rl_env.config import UAVEnvConfig
from rl_env.lidar3d import DEFAULT_SPEC, features, filter_ground
from rl_env.observation import ObservationHandler, quat_to_rotation_matrix
from rl_env.observation_26d import ObservationHandler26D
from rl_env.safety_shield import DeterministicSafetyShield, SafetyShieldConfig


@dataclass
class StepRecord:
    obs: np.ndarray
    raw_action: np.ndarray
    action: np.ndarray
    cmd: tuple
    shield_active: bool


def sanitize_ranges(ranges: np.ndarray, min_range: float, max_range: float) -> np.ndarray:
    """Gazebo reports +inf for 'no return' and -inf for 'below minimum range'."""
    r = np.asarray(ranges, dtype=np.float64).copy()
    r[np.isposinf(r) | np.isnan(r) | (r > max_range)] = max_range
    r[np.isneginf(r) | (r < min_range)] = min_range
    return r


class PolicyRuntime:
    def __init__(self, model_path: str, obs_mode: str, goal_world, shield: Optional[SafetyShieldConfig] = None,
                 config: Optional[UAVEnvConfig] = None, cbf=False):
        assert obs_mode in ("18d", "26d", "50d")
        self.model = PPO.load(model_path, device="cpu")
        self.obs_mode = obs_mode
        self.goal = np.asarray(goal_world, dtype=np.float32)
        self.config = config or UAVEnvConfig()
        self.actions = ActionHandler(self.config)
        self.obs18 = ObservationHandler(self.config)
        self.obs26 = ObservationHandler26D(self.config) if obs_mode == "26d" else None
        self.shield = DeterministicSafetyShield(shield) if shield is not None else None
        self.cbf = None
        if cbf:
            # 3-D CBF filter (baselines.cbf) on the raw LiDAR grid, applied after the rule shield (50d only)
            assert obs_mode == "50d", "the CBF filter needs the 7-channel LiDAR grid"
            from baselines.cbf import CBFSafetyFilter
            from rl_env.uav_world_env import UAVWorldEnvConfig
            self.cbf = CBFSafetyFilter(moving_obstacles=(cbf == "moving"))   # cbf: True or "moving" (M6 term)
            self.cbf.hull_dz = UAVWorldEnvConfig().hull_center_dz
        self.last_ranges = None
        self.n_cbf = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.prev_level = None
        self.n_steps = 0
        self.n_shield = 0

    def observe(self, pos, vel, quat, rates, scan) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float32)
        vel = np.asarray(vel, dtype=np.float32)
        quat = np.asarray(quat, dtype=np.float32)
        rates = np.asarray(rates, dtype=np.float32)
        if self.obs_mode == "18d":
            return self.obs18.compute_observation(pos, vel, quat, rates, self.goal, self.prev_action)
        if self.obs_mode == "26d":
            ranges = sanitize_ranges(scan.ranges.reshape(-1), 0.10, 10.0)
            obs, _ = self.obs26.compute_observation_with_metadata(
                pos_world=pos, vel_world=vel, quat=quat, rates_body=rates, goal_world=self.goal,
                prev_action=self.prev_action, raw_scan=(ranges, scan.angle_min, scan.angle_step))
            return obs
        spec = DEFAULT_SPEC
        ranges = sanitize_ranges(scan.ranges, spec.min_range, spec.max_range).reshape(len(spec.elevations_deg),
                                                                                      spec.n_azimuth)
        R = quat_to_rotation_matrix(quat).astype(np.float64)
        ranges = filter_ground(ranges, pos, R, spec)
        self.last_ranges = ranges
        feat = features(ranges, spec)
        level = feat[:8]
        rate = np.zeros(8, dtype=np.float32) if self.prev_level is None else \
            np.clip((level - self.prev_level) / 0.1, -5.0, 5.0)
        self.prev_level = level.copy()
        proprio = self.obs18.compute_observation(pos, vel, quat, rates, self.goal, self.prev_action)
        return np.concatenate([proprio, feat, rate]).astype(np.float32)

    def step(self, pos, vel, quat, rates, scan, position_world=None) -> StepRecord:
        obs = self.observe(pos, vel, quat, rates, scan)
        raw = np.asarray(self.model.predict(obs, deterministic=True)[0], dtype=np.float32)
        act, active = raw, False
        if self.shield is not None:
            meta = {"position_world": np.asarray(position_world if position_world is not None else pos)}
            act, diag = self.shield.filter_action(obs[:26] if len(obs) >= 26 else obs, raw, perception_metadata=meta)
            active = bool(diag.is_intervened)
        if self.cbf is not None:
            act, cinfo = self.cbf.filter_action(obs, act, ranges=self.last_ranges)
            self.n_cbf += int(cinfo["intervened"])
            active = active or bool(cinfo["intervened"])
        cmd = self.actions.scale_action(act)
        self.prev_action = np.clip(act, -1.0, 1.0).astype(np.float32)
        self.n_steps += 1
        self.n_shield += int(active)
        return StepRecord(obs, raw, act, (cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate), active)
