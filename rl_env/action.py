"""
action.py
---------
Action space definition, physical scaling, and validation for the UAV environment.
"""

from dataclasses import dataclass
import gymnasium as gym
import numpy as np

from rl_env.config import UAVEnvConfig


@dataclass
class CommandedVelocities:
    vx: float        # Forward body velocity (m/s, FLU)
    vy: float        # Leftward body velocity (m/s, FLU)
    vz: float        # Upward body velocity (m/s, FLU)
    yaw_rate: float  # CCW yaw rate (rad/s, FLU)

    def to_array(self) -> np.ndarray:
        return np.array([self.vx, self.vy, self.vz, self.yaw_rate], dtype=np.float32)


class ActionHandler:
    def __init__(self, config: UAVEnvConfig):
        self.config = config
        self.action_dim = 4

        # Continuous action space bounded to [-1.0, +1.0]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32
        )

        # Scale factors
        self.scale_factors = np.array([
            self.config.max_horizontal_velocity,  # vx
            self.config.max_horizontal_velocity,  # vy
            self.config.max_vertical_velocity,    # vz
            self.config.max_yaw_rate              # yaw_rate
        ], dtype=np.float32)

    def scale_action(self, action: np.ndarray) -> CommandedVelocities:
        """
        Scale normalized action a in [-1, +1]^4 to physical velocities.
        Handles NaN/Inf gracefully by clamping and sanitizing.
        """
        act = np.asarray(action, dtype=np.float32).flatten()
        if act.shape[0] != self.action_dim:
            raise ValueError(f"Action dimension mismatch: expected {self.action_dim}, got {act.shape[0]}")

        # Sanitize against non-finite values
        if not np.all(np.isfinite(act)):
            act = np.nan_to_num(act, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clamp to [-1.0, 1.0]
        act_clamped = np.clip(act, -1.0, 1.0)

        # Physical scaling
        scaled = act_clamped * self.scale_factors

        # Horizontal speed limit applies to the combined vx/vy speed (like PX4's
        # MPC_XY_VEL_MAX), so diagonal commands cannot exceed it.
        h_speed = float(np.hypot(scaled[0], scaled[1]))
        if h_speed > self.config.max_horizontal_velocity:
            scaled[:2] *= self.config.max_horizontal_velocity / h_speed

        return CommandedVelocities(
            vx=float(scaled[0]),
            vy=float(scaled[1]),
            vz=float(scaled[2]),
            yaw_rate=float(scaled[3])
        )

    def unscale_action(self, physical_cmd: CommandedVelocities) -> np.ndarray:
        """
        Convert physical command to normalized action [-1, 1].
        Useful for classical baseline controllers and testing.
        """
        arr = physical_cmd.to_array()
        normalized = arr / self.scale_factors
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
