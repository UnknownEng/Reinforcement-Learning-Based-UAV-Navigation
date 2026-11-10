"""
test_uav_fast_env_v2.py
-----------------------
Unit tests for UAVFastEnvV2 (UAVFastNav-v2):
- Verification of 26-D observation and 4-D action contracts
- Gym registry check (gym.make("UAVFastNav-v2"))
- Reset behavior and deterministic route validation in info dict
- Curriculum stage setting and obstacle distribution
- Step execution and reward decomposition
- Dynamic feasibility disclaimer check
"""

import math
import unittest
import gymnasium as gym
import numpy as np

import rl_env
from rl_env.uav_fast_env_v2 import UAVFastEnvV2, UAVObstacleEnvConfigV2


class TestUAVFastEnvV2(unittest.TestCase):
    """
    Unit test suite for UAVFastNav-v2 environment.
    
    DISCLAIMER:
    2D connectivity does not establish dynamic flight feasibility. While 2D BFS in inflated free-space
    guarantees the geometric existence of a collision-free corridor of width >= 2*(r_uav + margin),
    actual flight success depends on UAV acceleration limits, actuation lag, velocity saturation,
    attitude tilt limits, and closed-loop control tracking error.
    """

    def setUp(self):
        self.config = UAVObstacleEnvConfigV2()
        self.env = gym.make("UAVFastNav-v2")

    def tearDown(self):
        self.env.close()

    def test_disclaimer_present(self):
        """Verify the explicit dynamic flight feasibility disclaimer is documented."""
        doc = UAVFastEnvV2.__doc__
        self.assertIsNotNone(doc)
        self.assertIn("2D connectivity does not establish dynamic flight feasibility", doc)

    def test_contracts_preserved(self):
        """Verify strict preservation of 26-D observation and 4-D action contracts."""
        obs_space = self.env.observation_space
        act_space = self.env.action_space

        self.assertEqual(obs_space.shape, (26,))
        self.assertEqual(obs_space.dtype, np.float32)

        self.assertEqual(act_space.shape, (4,))
        self.assertEqual(act_space.dtype, np.float32)
        np.testing.assert_allclose(act_space.low, -1.0)
        np.testing.assert_allclose(act_space.high, 1.0)

    def test_reset_route_validation_info(self):
        """Verify reset() validates route traversability and returns validation info."""
        obs, info = self.env.reset(seed=123)
        self.assertEqual(obs.shape, (26,))
        self.assertTrue(np.all(np.isfinite(obs)))

        self.assertIn("route_traversable", info)
        self.assertIn("route_min_clearance", info)
        self.assertIn("route_length", info)
        self.assertIn("route_attempts", info)
        self.assertIn("curriculum_stage", info)

        self.assertTrue(info["route_traversable"])
        self.assertGreaterEqual(info["route_min_clearance"], 0.50)

    def test_curriculum_stages(self):
        """Verify obstacle counts across curriculum stages 0, 1, 2, 3."""
        # Stage 0: Free space (0 obstacles)
        self.env.unwrapped.set_curriculum_stage(0)
        obs, info = self.env.reset(seed=201)
        self.assertEqual(info["obstacle_count"], 0)

        # Stage 1: Sparse obstacles (1 to 3)
        self.env.unwrapped.set_curriculum_stage(1)
        obs, info = self.env.reset(seed=202)
        self.assertGreaterEqual(info["obstacle_count"], 1)
        self.assertLessEqual(info["obstacle_count"], 3)
        self.assertTrue(info["route_traversable"])

        # Stage 2: Clutter (3 to 5)
        self.env.unwrapped.set_curriculum_stage(2)
        obs, info = self.env.reset(seed=203)
        self.assertGreaterEqual(info["obstacle_count"], 3)
        self.assertLessEqual(info["obstacle_count"], 5)
        self.assertTrue(info["route_traversable"])

        # Invalid stage raises ValueError
        with self.assertRaises(ValueError):
            self.env.unwrapped.set_curriculum_stage(5)

    def test_step_execution_and_heading_reward_decomposition(self):
        """Verify step execution, reward decomposition, and absence of NaNs/Infs."""
        obs, _ = self.env.reset(seed=301)
        total_steps = 25

        for step_i in range(total_steps):
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)

            self.assertEqual(obs.shape, (26,))
            self.assertTrue(np.all(np.isfinite(obs)), f"Non-finite obs at step {step_i}")
            self.assertTrue(math.isfinite(reward), f"Non-finite reward at step {step_i}")

            # Check decomposed reward components
            rc = info["reward_components"]
            self.assertIn("reward_heading_align", rc)
            self.assertIn("penalty_speed_gate", rc)
            self.assertIn("reward_progress", rc)
            self.assertIn("penalty_clearance", rc)

            self.assertTrue(math.isfinite(rc["reward_heading_align"]))
            self.assertTrue(math.isfinite(rc["penalty_speed_gate"]))
            self.assertLessEqual(rc["penalty_speed_gate"], 0.0, "Speed gate penalty must be <= 0")

            if terminated or truncated:
                obs, _ = self.env.reset()

    def test_options_active_regime_propagation(self):
        """Verify passing active_regime in options updates both forced_regime and active_regime."""
        self.env.unwrapped.set_curriculum_stage(3)
        obs, info = self.env.reset(seed=401, options={"active_regime": 3})
        self.assertEqual(self.env.unwrapped.forced_regime, 3)
        self.assertEqual(self.env.unwrapped.active_regime, 3)
        self.assertEqual(self.env.unwrapped.current_max_episode_steps, 300)

    def test_options_max_episode_steps_override(self):
        """Verify passing max_episode_steps in options locks the step limit."""
        obs, info = self.env.reset(seed=402, options={"max_episode_steps": 250})
        self.assertEqual(self.env.unwrapped.current_max_episode_steps, 250)


if __name__ == "__main__":
    unittest.main()
