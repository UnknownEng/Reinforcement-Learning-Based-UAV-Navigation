"""
test_gate_b2_smoke.py
---------------------
Controlled Smoke Test Suite for Milestone 4 Phase 4B Gate B2.

Validates:
  1. 26-D PPO Network Architecture Initialization:
       - Policy network input layer: W_0 in R^{64 x 26}
       - Value network input layer: W_0 in R^{64 x 26}
       - Action output layer: 4 continuous actions in [-1, +1]
       - Rejection of 18-D input vectors by the 26-D network
  2. Forward Inference Behavior (deterministic and stochastic passes).
  3. Bounded Closed-Loop Policy Rollout Smoke Test (200 steps on UAVFastNav-v1, zero NaNs/Infs).
  4. Single-Batch Gradient Flow & Loss Computation (without weight updates or training).
  5. Multi-Stage Curriculum & Procedural Reset Smoke Test (Stages 0, 1, 2, 3).
  6. Collision Termination & Terminal Penalty Invariants.
  7. Continuous Quadratic Clearance Penalty Activation.
  8. Strict Baseline Protections: Checkpoint SHA-256 intact, zero training, zero diff on frozen files.
"""

import hashlib
import math
import os
import subprocess
import unittest
from typing import Dict, List, Tuple
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

import rl_env
from rl_env.config import UAVEnvConfig
from rl_env.observation_26d import ObservationHandler26D, SectorStatus
from rl_env.uav_fast_env_v1 import UAVFastEnvV1, UAVObstacleEnvConfig


EXPECTED_CHECKPOINT_HASH = "8d7a24eda075cbbeb60c4a6df07029c3a38fd770ce436a6daf96feff85300dfc"
CHECKPOINT_PATH = "/home/kali/proj/experiments/milestone2/models/ppo_uav_final.zip"
EXPECTED_FROZEN_TAG_COMMIT = "31c81c77aaea82ac35a34fb102e3128257b09b30"


class TestGateB2Smoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Instantiate a reference 26-D PPO model on UAVFastNav-v1 (untrained)
        cls.env = gym.make("UAVFastNav-v1")
        cls.model = PPO(
            policy="MlpPolicy",
            env=cls.env,
            n_steps=64,
            batch_size=32,
            policy_kwargs=dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
                activation_fn=torch.nn.Tanh
            ),
            seed=42,
            device="cpu"
        )

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_01_ppo_26d_network_architecture(self):
        """Verify SB3 PPO initializes with 26-D input layer (W_0 in R^{64 x 26}) and rejects 18-D inputs."""
        policy = self.model.policy

        # Inspect MLP extractor layers
        pi_in = policy.mlp_extractor.policy_net[0]
        vf_in = policy.mlp_extractor.value_net[0]

        self.assertIsInstance(pi_in, torch.nn.Linear)
        self.assertIsInstance(vf_in, torch.nn.Linear)

        # 26 inputs -> 64 hidden units
        self.assertEqual(pi_in.in_features, 26, f"Policy input features {pi_in.in_features} != 26")
        self.assertEqual(pi_in.out_features, 64)
        self.assertEqual(vf_in.in_features, 26, f"Value input features {vf_in.in_features} != 26")
        self.assertEqual(vf_in.out_features, 64)

        # Output dimensions: action_net (64 -> 4) and value_net (64 -> 1)
        self.assertEqual(policy.action_net.in_features, 64)
        self.assertEqual(policy.action_net.out_features, 4)
        self.assertEqual(policy.value_net.in_features, 64)
        self.assertEqual(policy.value_net.out_features, 1)

        # Rejection test: Passing an 18-D observation tensor into the 26-D policy must fail
        obs_18d = torch.randn(1, 18, dtype=torch.float32)
        with self.assertRaises(RuntimeError):
            _ = policy(obs_18d)

    def test_02_deterministic_and_stochastic_inference(self):
        """Verify forward inference under deterministic and stochastic modes outputs valid actions in [-1, +1]^4."""
        obs, info = self.env.reset(seed=101)
        self.assertEqual(obs.shape, (26,))

        # Deterministic inference
        act_det1, _ = self.model.predict(obs, deterministic=True)
        act_det2, _ = self.model.predict(obs, deterministic=True)

        self.assertEqual(act_det1.shape, (4,))
        self.assertTrue(np.all(np.isfinite(act_det1)))
        self.assertTrue(np.all(act_det1 >= -1.0) and np.all(act_det1 <= 1.0))
        # Deterministic outputs must match identically
        np.testing.assert_array_equal(act_det1, act_det2)

        # Stochastic inference
        act_stoch, _ = self.model.predict(obs, deterministic=False)
        self.assertEqual(act_stoch.shape, (4,))
        self.assertTrue(np.all(np.isfinite(act_stoch)))
        self.assertTrue(np.all(act_stoch >= -1.0) and np.all(act_stoch <= 1.0))

    def test_03_bounded_closed_loop_rollout_smoke(self):
        """Execute a safe, bounded rollout (200 steps) on UAVFastNav-v1 with the 26-D policy without errors."""
        obs, info = self.env.reset(seed=202)
        total_steps = 0
        episodes = 0
        max_steps = 200

        while total_steps < max_steps:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, step_info = self.env.step(action)
            total_steps += 1

            # Assert complete numerical validity on every single step
            self.assertEqual(obs.shape, (26,))
            self.assertTrue(np.all(np.isfinite(obs)), f"Non-finite observation at step {total_steps}!")
            self.assertTrue(math.isfinite(reward), f"Non-finite reward at step {total_steps}!")
            self.assertEqual(len(step_info["clearances_m"]), 8)
            self.assertTrue(all(math.isfinite(c) for c in step_info["clearances_m"]))

            if terminated or truncated:
                episodes += 1
                obs, info = self.env.reset()

        self.assertEqual(total_steps, max_steps)
        self.assertGreaterEqual(episodes, 1)

    def test_04_single_batch_gradient_flow(self):
        """
        Verify forward and backward gradient computation on a single synthetic batch of shape (32, 26).
        Confirms analytical learnability without modifying weights, running training loops, or saving checkpoints.
        """
        batch_size = 32
        obs_tensor = torch.randn(batch_size, 26, dtype=torch.float32)
        act_tensor = torch.randn(batch_size, 4, dtype=torch.float32)

        # Forward evaluation
        values, log_prob, entropy = self.model.policy.evaluate_actions(obs_tensor, act_tensor)

        self.assertEqual(values.shape, (batch_size, 1))
        self.assertEqual(log_prob.shape, (batch_size,))
        self.assertEqual(entropy.shape, (batch_size,))

        # Compute synthetic surrogate loss
        loss = values.mean() + log_prob.mean() - 0.01 * entropy.mean()
        self.assertTrue(torch.isfinite(loss))

        # Backward pass
        self.model.policy.zero_grad()
        loss.backward()

        # Check gradients on 26-D input layer
        pi_grad = self.model.policy.mlp_extractor.policy_net[0].weight.grad
        vf_grad = self.model.policy.mlp_extractor.value_net[0].weight.grad

        self.assertIsNotNone(pi_grad)
        self.assertIsNotNone(vf_grad)
        self.assertEqual(pi_grad.shape, (64, 26))
        self.assertEqual(vf_grad.shape, (64, 26))
        self.assertTrue(torch.all(torch.isfinite(pi_grad)), "NaN in policy input gradients!")
        self.assertTrue(torch.all(torch.isfinite(vf_grad)), "NaN in value input gradients!")

        # Confirm non-zero gradient flow into all 26 feature columns
        pi_grad_per_feature = torch.sum(torch.abs(pi_grad), dim=0)
        self.assertTrue(torch.all(pi_grad_per_feature > 0.0), "Dead input features in 26-D policy layer!")

        # Clean up gradients: zero out and do NOT call optimizer.step()
        self.model.policy.zero_grad()

    def test_05_curriculum_stages_reset_behavior(self):
        """Verify UAVFastNav-v1 across all 4 curriculum stages (Stages 0, 1, 2, 3) and seed determinism."""
        expected_ranges = {
            0: (0, 0),      # Free space
            1: (1, 1),      # Single obstacle
            2: (3, 4),      # Corridor
            3: (6, 8)       # Dense maze
        }

        for stage, (min_obs, max_obs) in expected_ranges.items():
            obs, info = self.env.reset(seed=300 + stage, options={"curriculum_stage": stage})
            obs_count = info["obstacle_count"]
            self.assertGreaterEqual(
                obs_count, min_obs,
                f"Stage {stage} spawned {obs_count} obstacles, expected at least {min_obs}"
            )
            self.assertLessEqual(
                obs_count, max_obs,
                f"Stage {stage} spawned {obs_count} obstacles, expected at most {max_obs}"
            )

            if stage == 0:
                # In free space, all 8 clearances must be clear (10.0m)
                np.testing.assert_array_equal(info["clearances_m"], np.full(8, 10.0, dtype=np.float32))

        # Seed determinism test
        obs_1, info_1 = self.env.reset(seed=4242, options={"curriculum_stage": 2})
        obs_2, info_2 = self.env.reset(seed=4242, options={"curriculum_stage": 2})
        np.testing.assert_array_equal(obs_1, obs_2)
        self.assertEqual(info_1["obstacles"], info_2["obstacles"])

    def test_06_collision_termination_and_terminal_penalty(self):
        """Verify obstacle collision termination and terminal penalty (-50.0)."""
        # Place cylinder directly at (0.45, 0.0), R = 0.20 -> surface at 0.25m <= 0.35m (collision)
        obs, info = self.env.reset(options={
            "start_pos": [0.0, 0.0, 1.5],
            "start_yaw": 0.0,
            "goal_pos": [5.0, 0.0, 1.5],
            "obstacles": [(0.45, 0.0, 0.20)]
        })

        self.assertLessEqual(info["min_surface_dist_m"], 0.35)

        # Stepping while in collision must immediately terminate with crash
        next_obs, reward, terminated, truncated, step_info = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertTrue(step_info["is_crashed"])
        self.assertTrue(step_info["is_obstacle_collision"])
        self.assertIn("obstacle_collision", step_info["termination_reason"])
        self.assertLessEqual(reward, -45.0)

    def test_07_quadratic_clearance_penalty_behavior(self):
        """Verify quadratic clearance penalty formula: R_clear = -w_clear * sum(max(0, (d_safe - d_i)/d_safe)^2)."""
        env_u = self.env.unwrapped

        # Clearance >= 1.50m -> zero penalty
        r_clear = env_u.compute_obstacle_reward(
            dist_prev=3.0, dist_curr=3.0,
            action_curr=np.zeros(4), action_prev=np.zeros(4),
            clearances=np.full(8, 2.0, dtype=np.float32),
            is_success=False, is_out_of_bounds=False,
            is_ground_crash=False, is_obstacle_collision=False
        )
        self.assertEqual(r_clear.clearance_penalty, 0.0)

        # Clearance = 0.75m -> deficit = 0.50 -> penalty = 0.50 * (0.50)^2 = 0.125
        clear_test = np.full(8, 2.0, dtype=np.float32)
        clear_test[0] = 0.75
        r_close = env_u.compute_obstacle_reward(
            dist_prev=3.0, dist_curr=3.0,
            action_curr=np.zeros(4), action_prev=np.zeros(4),
            clearances=clear_test,
            is_success=False, is_out_of_bounds=False,
            is_ground_crash=False, is_obstacle_collision=False
        )
        self.assertAlmostEqual(r_close.clearance_penalty, 0.125, places=5)

    def test_08_strict_baseline_and_checkpoint_invariants(self):
        """Verify baseline protections: checkpoint hash intact, frozen tag intact, adapters untouched, UAVFastNav-v0 18-D."""
        # 1. Checkpoint SHA-256
        self.assertTrue(os.path.exists(CHECKPOINT_PATH))
        hasher = hashlib.sha256()
        with open(CHECKPOINT_PATH, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()
        self.assertEqual(actual_hash, EXPECTED_CHECKPOINT_HASH)

        # 2. Frozen tag commit
        tag_cmd = subprocess.run(["git", "rev-parse", "milestone4-phase4a-frozen^{commit}"], capture_output=True, text=True)
        self.assertEqual(tag_cmd.returncode, 0)
        self.assertEqual(tag_cmd.stdout.strip(), EXPECTED_FROZEN_TAG_COMMIT)

        # 3. Production adapters diff
        diff_cmd = subprocess.run(
            ["git", "diff", "milestone4-phase4a-frozen", "--", "ppo_adapter.py", "ppo_inference_adapter.py", "rl_env/observation.py"],
            capture_output=True, text=True
        )
        self.assertEqual(diff_cmd.returncode, 0)
        self.assertEqual(len(diff_cmd.stdout.strip()), 0)

        # 4. UAVFastNav-v0 remains 18-D
        env_v0 = gym.make("UAVFastNav-v0")
        obs_v0, _ = env_v0.reset(seed=999)
        self.assertEqual(obs_v0.shape, (18,))
        self.assertEqual(env_v0.observation_space.shape, (18,))
        env_v0.close()


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGateB2Smoke)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    import sys
    res = run_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
