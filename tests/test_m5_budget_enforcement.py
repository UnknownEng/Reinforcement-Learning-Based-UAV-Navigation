"""
test_m5_budget_enforcement.py
-----------------------------
Focused unit tests proving strict budget enforcement in SB3 PPO:
1. Step budget cap: Ensures training terminates at or below the exact step cap (no rollout overshoot).
2. Wall-clock budget cap: Ensures training terminates when time budget is exceeded.
"""

import time
import unittest
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class StrictBudgetCallback(BaseCallback):
    """Callback enforcing hard ceilings on environment steps and wall-clock duration."""

    def __init__(self, max_timesteps: int = 100, max_wall_clock_s: float = 60.0, verbose: int = 0):
        super().__init__(verbose)
        self.max_timesteps = int(max_timesteps)
        self.max_wall_clock_s = float(max_wall_clock_s)
        self.start_time: float = 0.0
        self.halt_reason: str = "none"

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()

    def _on_step(self) -> bool:
        # Enforce step ceiling
        if self.num_timesteps >= self.max_timesteps:
            self.halt_reason = "step_limit"
            return False

        # Enforce wall-clock ceiling
        if (time.perf_counter() - self.start_time) >= self.max_wall_clock_s:
            self.halt_reason = "time_limit"
            return False

        return True


class TestBudgetEnforcement(unittest.TestCase):
    """Test suite proving strict budget ceilings."""

    def setUp(self):
        # Lightweight dummy environment for fast unit testing
        self.env = gym.make("Pendulum-v1")

    def tearDown(self):
        self.env.close()

    def test_strict_step_budget_prevents_rollout_overshoot(self):
        """
        Verify that when max_timesteps = 100 with rollout n_steps = 64,
        training terminates at exactly 100 steps instead of overshooting to 128 (2 x 64).
        """
        target_steps = 100
        n_steps = 64
        cb = StrictBudgetCallback(max_timesteps=target_steps, max_wall_clock_s=30.0)

        model = PPO("MlpPolicy", self.env, n_steps=n_steps, verbose=0, device="cpu")
        model.learn(total_timesteps=200, callback=cb)

        self.assertEqual(model.num_timesteps, target_steps,
                         f"Final steps ({model.num_timesteps}) must not exceed target ({target_steps})")
        self.assertEqual(cb.halt_reason, "step_limit")

    def test_strict_wall_clock_budget_enforcement(self):
        """Verify that training terminates when wall-clock ceiling is reached."""
        time_limit_s = 0.5
        cb = StrictBudgetCallback(max_timesteps=1000000, max_wall_clock_s=time_limit_s)

        model = PPO("MlpPolicy", self.env, n_steps=64, verbose=0, device="cpu")
        t0 = time.perf_counter()
        model.learn(total_timesteps=1000000, callback=cb)
        elapsed = time.perf_counter() - t0

        self.assertEqual(cb.halt_reason, "time_limit")
        self.assertLess(elapsed, time_limit_s + 0.5, "Training must halt shortly after time ceiling")


if __name__ == "__main__":
    unittest.main()
