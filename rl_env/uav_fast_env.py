"""
uav_fast_env.py
---------------
High-throughput, lightweight Gymnasium environment for 3D autonomous UAV navigation.
Adheres strictly to the Gymnasium API and shares the identical 18D observation
and 4D action interface. It runs as a standalone Python simulation on Windows.
"""

from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np

from rl_env.action import ActionHandler, CommandedVelocities
from rl_env.config import UAVEnvConfig
from rl_env.dynamics import FastUAVDynamics, UAVState
from rl_env.observation import ObservationHandler
from rl_env.reward import RewardCalculator, RewardComponents
from rl_env.termination import TerminationChecker, TerminationStatus


class UAVFastEnv(gym.Env):
    """
    Fast, lightweight 3D UAV navigation environment.
    Observations: 18D vector (relative goal, linear & angular velocities, orientation quaternion,
                              relative altitude, previous normalized action).
    Actions: 4D vector in [-1, +1]^4 (body forward velocity, lateral velocity, vertical velocity, yaw rate).
    """

    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 10
    }

    def __init__(
        self,
        config: Optional[UAVEnvConfig] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.config = config if config is not None else UAVEnvConfig()
        self.render_mode = render_mode

        # Core handlers
        self.action_handler = ActionHandler(self.config)
        self.obs_handler = ObservationHandler(self.config)
        self.reward_calc = RewardCalculator(self.config)
        self.term_checker = TerminationChecker(self.config)

        # Standard Gymnasium spaces
        self.action_space = self.action_handler.action_space
        self.observation_space = self.obs_handler.observation_space

        # Dynamics simulation model (seeded in reset)
        self.dynamics = FastUAVDynamics(self.config)

        # Episode tracking states
        self.step_count: int = 0
        self.goal_world: np.ndarray = np.zeros(3, dtype=np.float32)
        self.prev_action: np.ndarray = np.zeros(4, dtype=np.float32)
        self.prev_dist: float = 0.0
        self.initial_dist: float = 0.0
        self.path_points: List[np.ndarray] = []
        self.cumulative_reward: float = 0.0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the UAV environment for a new episode.
        Returns:
            observation (np.ndarray): Initial 18D observation vector.
            info (dict): Diagnostic metadata.
        """
        super().reset(seed=seed)
        self.dynamics.rng = self.np_random

        opts = options or {}

        # 1. Spawn UAV position and orientation
        if "start_pos" in opts:
            start_pos = np.array(opts["start_pos"], dtype=np.float32)
            x0, y0, z0 = float(start_pos[0]), float(start_pos[1]), float(start_pos[2])
        else:
            x0 = float(self.np_random.uniform(self.config.start_pos_xy_range[0], self.config.start_pos_xy_range[1]))
            y0 = float(self.np_random.uniform(self.config.start_pos_xy_range[0], self.config.start_pos_xy_range[1]))
            z0 = float(self.config.start_altitude)

        if "start_yaw" in opts:
            yaw0 = float(opts["start_yaw"])
        elif self.config.randomize_start_yaw:
            yaw0 = float(self.np_random.uniform(-np.pi, np.pi))
        else:
            yaw0 = 0.0

        # Reset dynamics
        state = self.dynamics.reset(x0=x0, y0=y0, z0=z0, yaw0=yaw0)

        # 2. Spawn 3D Navigation Goal
        if "goal_pos" in opts:
            self.goal_world = np.array(opts["goal_pos"], dtype=np.float32)
        else:
            # Random distance and azimuth relative to start position
            goal_dist = self.np_random.uniform(self.config.min_goal_dist, self.config.max_goal_dist)
            goal_azimuth = self.np_random.uniform(-np.pi, np.pi)
            gx = x0 + goal_dist * np.cos(goal_azimuth)
            gy = y0 + goal_dist * np.sin(goal_azimuth)

            # Clamp goal within safe interior of arena geofence
            margin = 1.0
            gx = np.clip(gx, self.config.arena_x[0] + margin, self.config.arena_x[1] - margin)
            gy = np.clip(gy, self.config.arena_y[0] + margin, self.config.arena_y[1] - margin)
            gz = float(self.np_random.uniform(self.config.goal_altitude_range[0], self.config.goal_altitude_range[1]))
            self.goal_world = np.array([gx, gy, gz], dtype=np.float32)

        # 3. Reset internal bookkeeping
        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.last_pos = state.pos_world.copy()
        self.path_length = 0.0
        self.cumulative_reward = 0.0

        dist = float(np.linalg.norm(state.pos_world - self.goal_world))
        self.prev_dist = dist
        self.initial_dist = dist

        # 4. Compute initial observation
        obs_state = self.dynamics.get_noisy_state() if self.config.enable_noise else state
        obs = self.obs_handler.compute_observation(
            pos_world=obs_state.pos_world,
            vel_world=obs_state.vel_world,
            quat=obs_state.quat,
            rates_body=obs_state.body_rates,
            goal_world=self.goal_world,
            prev_action=self.prev_action,
            ground_z=0.0
        )

        info = {
            "pos_world": state.pos_world.copy(),
            "goal_world": self.goal_world.copy(),
            "dist_to_goal": dist,
            "step": 0
        }

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one policy control step (0.1s at 10 Hz).
        """
        self.step_count += 1

        # 1. Parse and physically scale action
        cmd = self.action_handler.scale_action(action)
        act_norm = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)

        # 2. Advance physics via internal sub-stepping
        state = self.dynamics.step(cmd)
        step_disp = float(np.linalg.norm(state.pos_world - self.last_pos))
        self.path_length += step_disp
        self.last_pos = state.pos_world.copy()

        # 3. Distance & Navigation state
        dist_curr = float(np.linalg.norm(state.pos_world - self.goal_world))

        # 4. Termination & Truncation Checks
        term_status: TerminationStatus = self.term_checker.check(
            state=state,
            dist_to_goal=dist_curr,
            step_count=self.step_count
        )

        # 5. Reward Calculation
        reward_comp: RewardComponents = self.reward_calc.compute_reward(
            dist_prev=self.prev_dist,
            dist_curr=dist_curr,
            action_curr=act_norm,
            action_prev=self.prev_action,
            is_success=term_status.is_success,
            is_out_of_bounds=term_status.is_out_of_bounds,
            is_crashed=term_status.is_crashed
        )
        self.cumulative_reward += reward_comp.total

        # 6. Compute Observation (with sensor noise if enabled)
        obs_state = self.dynamics.get_noisy_state() if self.config.enable_noise else state
        obs = self.obs_handler.compute_observation(
            pos_world=obs_state.pos_world,
            vel_world=obs_state.vel_world,
            quat=obs_state.quat,
            rates_body=obs_state.body_rates,
            goal_world=self.goal_world,
            prev_action=act_norm,
            ground_z=0.0
        )

        # 7. Update history
        self.prev_action = act_norm.copy()
        self.prev_dist = dist_curr

        # 8. Diagnostic info dictionary
        path_len = self.path_length

        info: Dict[str, Any] = {
            "step": self.step_count,
            "dist_to_goal": dist_curr,
            "pos_world": state.pos_world.copy(),
            "vel_world": state.vel_world.copy(),
            "goal_world": self.goal_world.copy(),
            "roll_deg": float(np.degrees(state.roll)),
            "pitch_deg": float(np.degrees(state.pitch)),
            "yaw_deg": float(np.degrees(state.yaw)),
            "is_success": term_status.is_success,
            "is_crashed": term_status.is_crashed,
            "is_out_of_bounds": term_status.is_out_of_bounds,
            "termination_reason": term_status.reason,
            "path_length": path_len,
            "cumulative_reward": self.cumulative_reward,
            **reward_comp.to_dict()
        }

        if term_status.terminated or term_status.truncated:
            info["terminal_observation"] = obs.copy()
            info["final_distance"] = dist_curr

        return obs, reward_comp.total, term_status.terminated, term_status.truncated, info

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            state = self.dynamics.get_state()
            s = (
                f"[Step {self.step_count:3d}] "
                f"Pos: [{state.pos_world[0]:5.2f}, {state.pos_world[1]:5.2f}, {state.pos_world[2]:5.2f}] m | "
                f"Goal: [{self.goal_world[0]:5.2f}, {self.goal_world[1]:5.2f}, {self.goal_world[2]:5.2f}] m | "
                f"Dist: {self.prev_dist:5.2f} m | "
                f"R/P/Y: [{np.degrees(state.roll):4.1f}, {np.degrees(state.pitch):4.1f}, {np.degrees(state.yaw):4.1f}] deg | "
                f"Reward: {self.cumulative_reward:6.1f}"
            )
            return s
        return None

    def close(self):
        pass
