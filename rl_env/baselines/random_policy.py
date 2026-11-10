"""
random_policy.py
----------------
Baseline 1: Uniform Random Policy.
Samples action uniformly from the action space [-1.0, +1.0]^4.
Serves as lower-bound performance baseline.
"""

from typing import Optional
import gymnasium as gym
import numpy as np


class RandomPolicy:
    def __init__(self, action_space: gym.spaces.Box, seed: Optional[int] = None):
        self.action_space = action_space
        if seed is not None:
            self.action_space.seed(seed)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Sample an action uniformly from the action space [-1, 1]^4.
        """
        return self.action_space.sample()

    def reset(self):
        pass
