"""
reward.py
---------
Decomposed, interpretable reward function for autonomous 3D UAV navigation.
All weights and parameters are loaded from UAVEnvConfig.
"""

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from rl_env.config import UAVEnvConfig


@dataclass
class RewardComponents:
    progress: float
    success: float
    smoothness_penalty: float
    boundary_penalty: float
    time_penalty: float
    total: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'reward_progress': self.progress,
            'reward_success': self.success,
            'penalty_smoothness': self.smoothness_penalty,
            'penalty_boundary': self.boundary_penalty,
            'penalty_time': self.time_penalty,
            'reward_total': self.total
        }


class RewardCalculator:
    def __init__(self, config: UAVEnvConfig):
        self.config = config

    def compute_reward(
        self,
        dist_prev: float,
        dist_curr: float,
        action_curr: np.ndarray,
        action_prev: np.ndarray,
        is_success: bool,
        is_out_of_bounds: bool,
        is_crashed: bool
    ) -> RewardComponents:
        """
        Compute decomposed reward for the current timestep.
        """
        # 1. Progress reward: proportional to reduction in distance to goal
        delta_d = dist_prev - dist_curr
        progress_reward = float(self.config.reward_progress_weight * delta_d)

        # 2. Goal success terminal bonus
        success_reward = float(self.config.reward_success if is_success else 0.0)

        # 3. Action smoothness penalty: penalizes high jerk / abrupt action changes
        action_diff = action_curr - action_prev
        action_jerk_norm_sq = float(np.sum(np.square(action_diff)))
        smoothness_penalty = float(self.config.reward_action_smoothness * action_jerk_norm_sq)

        # 4. Boundary / Crash penalty
        boundary_penalty = 0.0
        if is_out_of_bounds or is_crashed:
            boundary_penalty = float(self.config.reward_collision_penalty)

        # 5. Small constant time step penalty
        time_penalty = float(self.config.reward_time_penalty)

        # Total scalar reward
        total = progress_reward + success_reward - smoothness_penalty - boundary_penalty - time_penalty

        # Sanitize against non-finite rewards
        if not np.isfinite(total):
            total = 0.0

        return RewardComponents(
            progress=progress_reward,
            success=success_reward,
            smoothness_penalty=smoothness_penalty,
            boundary_penalty=boundary_penalty,
            time_penalty=time_penalty,
            total=total
        )
