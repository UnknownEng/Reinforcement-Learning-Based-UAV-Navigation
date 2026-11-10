"""
termination.py
--------------
Explicit termination and truncation conditions for the UAV environment:
  - Success: Goal reached and stabilized
  - Failure: Geofence violation, hard ground crash, excessive tilt
  - Truncation: Maximum episode step timeout
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from rl_env.config import UAVEnvConfig
from rl_env.dynamics import UAVState


@dataclass
class TerminationStatus:
    terminated: bool
    truncated: bool
    is_success: bool
    is_out_of_bounds: bool
    is_crashed: bool
    reason: str


class TerminationChecker:
    def __init__(self, config: UAVEnvConfig):
        self.config = config

    def check(
        self,
        state: UAVState,
        dist_to_goal: float,
        step_count: int,
        max_episode_steps: Optional[int] = None
    ) -> TerminationStatus:
        speed = float(np.linalg.norm(state.vel_world))
        pos = state.pos_world
        roll = abs(state.roll)
        pitch = abs(state.pitch)

        # 1. Check Success: Within goal tolerance, low speed, upright attitude
        if dist_to_goal <= self.config.goal_tolerance_dist:
            if speed <= self.config.goal_tolerance_vel and roll < np.radians(30.0) and pitch < np.radians(30.0):
                return TerminationStatus(
                    terminated=True,
                    truncated=False,
                    is_success=True,
                    is_out_of_bounds=False,
                    is_crashed=False,
                    reason="goal_reached_and_stabilized"
                )

        # 2. Check Excessive Tilt Angle
        if roll > self.config.max_tilt_angle_rad or pitch > self.config.max_tilt_angle_rad:
            return TerminationStatus(
                terminated=True,
                truncated=False,
                is_success=False,
                is_out_of_bounds=False,
                is_crashed=True,
                reason="excessive_tilt"
            )

        # 3. Check Geofence Violation
        x_min, x_max = self.config.arena_x
        y_min, y_max = self.config.arena_y
        z_min, z_max = self.config.arena_z

        if (pos[0] < x_min or pos[0] > x_max or
            pos[1] < y_min or pos[1] > y_max or
            pos[2] > z_max):
            return TerminationStatus(
                terminated=True,
                truncated=False,
                is_success=False,
                is_out_of_bounds=True,
                is_crashed=False,
                reason="geofence_violation"
            )

        # 4. Check Ground Crash: touchdown faster than ground_crash_speed during the
        #    last step. The dynamics records the impact speed before the floor
        #    constraint zeroes the vertical velocity.
        impact_speed = float(getattr(state, "ground_impact_speed", 0.0))
        if impact_speed > getattr(self.config, "ground_crash_speed", 1.0):
            return TerminationStatus(
                terminated=True,
                truncated=False,
                is_success=False,
                is_out_of_bounds=False,
                is_crashed=True,
                reason="ground_crash"
            )

        # 5. Check Truncation (Timeout)
        limit = max_episode_steps if max_episode_steps is not None else self.config.max_episode_steps
        if step_count >= limit:
            return TerminationStatus(
                terminated=False,
                truncated=True,
                is_success=False,
                is_out_of_bounds=False,
                is_crashed=False,
                reason="episode_timeout"
            )

        # Normal continuation
        return TerminationStatus(
            terminated=False,
            truncated=False,
            is_success=False,
            is_out_of_bounds=False,
            is_crashed=False,
            reason="active"
        )
