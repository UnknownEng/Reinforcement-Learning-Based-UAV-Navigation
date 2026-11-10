"""
test_policy_compatibility.py
----------------------------
Gate B0 Policy Compatibility and Isolation Verification.

Confirms:
1. Checkpoint integrity (SHA-256 matches 8d7a24eda075cbbeb60c4a6df07029c3a38fd770ce436a6daf96feff85300dfc).
2. Existing PPO checkpoint expects strictly 18-D observations.
3. Network input layer weight is strictly W_0 in R^{64 x 18}.
4. Passing an 18-D observation succeeds and returns action in [-1, +1]^4.
5. Passing an experimental 26-D observation strictly raises a dimension error (cannot be silently loaded).
6. Production flight adapters (ppo_adapter.py and ppo_inference_adapter.py) are 100% isolated and unchanged.
"""

import hashlib
import os
import subprocess
import sys
import unittest
import numpy as np
import torch
from stable_baselines3 import PPO

from rl_env.config import UAVEnvConfig
from rl_env.observation import ObservationHandler
from rl_env.observation_26d import ObservationHandler26D


EXPECTED_PPO_SHA256 = "8d7a24eda075cbbeb60c4a6df07029c3a38fd770ce436a6daf96feff85300dfc"
PPO_MODEL_PATH = "/home/kali/proj/experiments/milestone2/models/ppo_uav_final.zip"


class TestPolicyCompatibility(unittest.TestCase):
    def test_01_checkpoint_hash_integrity(self):
        """Verify the frozen checkpoint SHA-256 matches the baseline exactly."""
        self.assertTrue(os.path.isfile(PPO_MODEL_PATH), f"Model file not found at {PPO_MODEL_PATH}")

        sha256 = hashlib.sha256()
        with open(PPO_MODEL_PATH, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest()

        self.assertEqual(
            actual_hash,
            EXPECTED_PPO_SHA256,
            f"Checkpoint SHA-256 mismatch! Got {actual_hash}, expected {EXPECTED_PPO_SHA256}"
        )

    def test_02_policy_observation_dimension_invariants(self):
        """Verify the loaded PPO model strictly expects 18-D input."""
        model = PPO.load(PPO_MODEL_PATH, device="cpu")

        # 1. Gymnasium observation space in policy
        self.assertEqual(model.observation_space.shape, (18,))

        # 2. PyTorch neural network input weight matrix
        # Policy net first linear layer
        w_pi_0 = model.policy.mlp_extractor.policy_net[0].weight
        b_pi_0 = model.policy.mlp_extractor.policy_net[0].bias
        self.assertEqual(
            tuple(w_pi_0.shape),
            (64, 18),
            f"Policy net input weight shape is {w_pi_0.shape}, expected (64, 18)"
        )
        self.assertEqual(tuple(b_pi_0.shape), (64,))

        # Value net first linear layer
        w_vf_0 = model.policy.mlp_extractor.value_net[0].weight
        b_vf_0 = model.policy.mlp_extractor.value_net[0].bias
        self.assertEqual(
            tuple(w_vf_0.shape),
            (64, 18),
            f"Value net input weight shape is {w_vf_0.shape}, expected (64, 18)"
        )
        self.assertEqual(tuple(b_vf_0.shape), (64,))

    def test_03_inference_with_18d_observation(self):
        """Verify inference succeeds on valid 18-D observation."""
        model = PPO.load(PPO_MODEL_PATH, device="cpu")
        handler_18d = ObservationHandler(UAVEnvConfig())

        obs_18 = handler_18d.compute_observation(
            pos_world=np.zeros(3, dtype=np.float32),
            vel_world=np.zeros(3, dtype=np.float32),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            rates_body=np.zeros(3, dtype=np.float32),
            goal_world=np.array([5.0, 0.0, 1.5], dtype=np.float32),
            prev_action=np.zeros(4, dtype=np.float32),
            ground_z=0.0
        )

        self.assertEqual(obs_18.shape, (18,))
        action, _states = model.predict(obs_18, deterministic=True)

        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.all(action >= -1.0))
        self.assertTrue(np.all(action <= 1.0))

    def test_04_rejection_of_26d_observation_by_18d_policy(self):
        """Verify passing a 26-D observation into the 18-D model strictly fails."""
        model = PPO.load(PPO_MODEL_PATH, device="cpu")
        handler_26d = ObservationHandler26D(UAVEnvConfig())

        obs_26 = handler_26d.compute_observation(
            pos_world=np.zeros(3, dtype=np.float32),
            vel_world=np.zeros(3, dtype=np.float32),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            rates_body=np.zeros(3, dtype=np.float32),
            goal_world=np.array([5.0, 0.0, 1.5], dtype=np.float32),
            prev_action=np.zeros(4, dtype=np.float32),
            clearances=[5.0] * 8,
            ground_z=0.0
        )
        self.assertEqual(obs_26.shape, (26,))

        # Feeding 26D vector into 18D model must raise ValueError / RuntimeError
        with self.assertRaises((ValueError, RuntimeError)) as ctx:
            model.predict(obs_26, deterministic=True)

        # Confirm the exception is indeed a dimension mismatch error
        err_msg = str(ctx.exception)
        self.assertTrue(
            "26" in err_msg or "18" in err_msg or "shape" in err_msg or "size mismatch" in err_msg,
            f"Unexpected error message: {err_msg}"
        )

    def test_05_production_adapters_isolation(self):
        """Verify that production adapters remain 100% identical to the frozen baseline."""
        adapter_files = [
            "ros2_ws/src/uav_autonomy/uav_autonomy/ppo_inference_adapter.py"
        ]
        # Also check ppo_adapter.py if present
        if os.path.exists("/home/kali/proj/ros2_ws/src/uav_autonomy/uav_autonomy/ppo_adapter.py"):
            adapter_files.append("ros2_ws/src/uav_autonomy/uav_autonomy/ppo_adapter.py")

        cmd = [
            "git", "diff", "milestone4-phase4a-frozen..HEAD", "--"
        ] + adapter_files

        res = subprocess.run(cmd, capture_output=True, text=True, cwd="/home/kali/proj")
        self.assertEqual(res.returncode, 0)
        self.assertEqual(
            res.stdout.strip(),
            "",
            f"Production adapters were modified! Diff:\n{res.stdout}"
        )

        # Ensure adapter does not import observation_26d or use 26-D
        for rel_path in adapter_files:
            abs_path = os.path.join("/home/kali/proj", rel_path)
            with open(abs_path, "r") as f:
                content = f.read()
                self.assertNotIn("observation_26d", content)
                self.assertNotIn("ObservationHandler26D", content)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPolicyCompatibility)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    res = run_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
