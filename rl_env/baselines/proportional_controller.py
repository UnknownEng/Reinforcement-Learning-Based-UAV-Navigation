"""
proportional_controller.py
--------------------------
Baseline 2: Classical Proportional / Waypoint Guidance Controller.
Operates directly on the 18D POMDP observation vector:
  - Generates proportional body-frame velocity commands: v_body = K_p * r_goal,body
  - Generates proportional yaw rate commands to orient vehicle along line-of-sight: yaw_rate = K_psi * atan2(y, x)
  - Scales commands to normalized action [-1, 1]^4.
"""

from typing import Optional
import numpy as np

from rl_env.action import ActionHandler, CommandedVelocities
from rl_env.config import UAVEnvConfig


class ProportionalWaypointController:
    """
    Classical P-guidance controller for waypoint navigation.
    """

    def __init__(
        self,
        config: Optional[UAVEnvConfig] = None,
        kp_xy: float = 1.2,
        kp_z: float = 1.2,
        k_yaw: float = 1.5
    ):
        self.config = config if config is not None else UAVEnvConfig()
        self.action_handler = ActionHandler(self.config)
        self.kp_xy = kp_xy
        self.kp_z = kp_z
        self.k_yaw = k_yaw

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Compute normalized action from observation vector.
        Observation layout:
          obs[0:3] = goal position in vehicle body FLU frame [x_b, y_b, z_b]
        """
        goal_body = observation[0:3]
        xg_b, yg_b, zg_b = float(goal_body[0]), float(goal_body[1]), float(goal_body[2])

        # Proportional velocity commands in body frame
        vx_cmd = np.clip(self.kp_xy * xg_b, -self.config.max_horizontal_velocity, self.config.max_horizontal_velocity)
        vy_cmd = np.clip(self.kp_xy * yg_b, -self.config.max_horizontal_velocity, self.config.max_horizontal_velocity)
        vz_cmd = np.clip(self.kp_z * zg_b, -self.config.max_vertical_velocity, self.config.max_vertical_velocity)

        # Proportional heading alignment (orient towards target in horizontal plane)
        horiz_dist = np.hypot(xg_b, yg_b)
        if horiz_dist > 0.2:
            yaw_err = np.arctan2(yg_b, xg_b)
            yaw_rate_cmd = np.clip(self.k_yaw * yaw_err, -self.config.max_yaw_rate, self.config.max_yaw_rate)
        else:
            yaw_rate_cmd = 0.0

        physical_cmd = CommandedVelocities(
            vx=float(vx_cmd),
            vy=float(vy_cmd),
            vz=float(vz_cmd),
            yaw_rate=float(yaw_rate_cmd)
        )

        # Return normalized action in [-1, +1]^4
        return self.action_handler.unscale_action(physical_cmd)

    def reset(self):
        pass
