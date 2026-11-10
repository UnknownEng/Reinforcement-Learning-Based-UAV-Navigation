"""
test_m5_stage3.py
------------------
Comprehensive Unit Test Suite for Milestone 5 Stage 3 Redesign.
Validates:
1. SafetyShieldWrapper observation caching on reset and step
2. SafetyShieldWrapper intervention on unsafe actions
3. SafetyShieldWrapper certified fail-safe zero-velocity hover on None/NaN/Inf obs
4. Regime 1 (Dense Clutter) obstacle count, radius, and pairwise gap >= 1.10m
5. Regime 2 (Narrow Gates) aperture width in [1.00m, 1.30m] and keepout zones
6. Regime 3 (Sequential Slaloms) barrier stages and alternating lateral openings
7. Action divergence quadratic penalty computation (R_div = -0.25 * ||a_raw - a_shield||^2)
8. Domain randomization parameter ranges (tau_xy, a_max, p_drop, sigma_lidar, etc.)
9. Geometry validator traversability across all regimes
10. Contract preservation: Box(26,) observation space and Box(4,) action space
11. Checkpoint compatibility with [64, 64] MLP architecture
12. Cryptographic immutability of frozen checkpoints
13. CPU inference and shielding latency within budget (< 1.0 ms)
"""

import hashlib
import math
import os
import time
import unittest
from typing import List, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

import rl_env
from rl_env.action import CommandedVelocities
from rl_env.safety_shield import (
    DeterministicSafetyShield,
    SafetyShieldConfig,
    SafetyShieldWrapper
)
from rl_env.uav_fast_env_v2 import UAVFastEnvV2, UAVObstacleEnvConfigV2


class TestM5Stage3Redesign(unittest.TestCase):
    """Exhaustive validation test suite for Milestone 5 Stage 3."""

    def setUp(self):
        self.config = UAVObstacleEnvConfigV2(
            curriculum_stage=3,
            reward_align_weight=0.10,
            reward_speed_gate_weight=0.10,
            reward_reverse_weight=0.35,
            reward_action_divergence_weight=0.25,
            min_path_clearance=0.45,
            safety_margin=0.10,
            grid_resolution=0.10,
            max_resample_attempts=50,
            enable_domain_randomization=True
        )

    # -------------------------------------------------------------------------
    # Test 1: SafetyShieldWrapper Observation Caching on Reset
    # -------------------------------------------------------------------------
    def test_wrapper_caches_obs_on_reset(self):
        """Verify SafetyShieldWrapper intercepts and caches valid observation on reset."""
        raw_env = UAVFastEnvV2(config=self.config)
        wrapper = SafetyShieldWrapper(raw_env)

        obs, info = wrapper.reset(seed=401)
        self.assertIsNotNone(wrapper.current_obs, "Wrapper current_obs must not be None after reset")
        self.assertEqual(wrapper.current_obs.shape, (26,), "Cached observation must have shape (26,)")
        np.testing.assert_allclose(wrapper.current_obs, obs, err_msg="Cached obs must match reset return")
        self.assertIsNotNone(wrapper.current_meta, "Wrapper current_meta must be cached on reset")
        self.assertEqual(wrapper.total_interventions, 0, "Interventions must reset to 0")
        self.assertEqual(wrapper.fail_safe_events, 0, "Fail-safe events must reset to 0")

        # Verify environment-level last_obs is also set (Layer A defense)
        self.assertIsNotNone(raw_env.last_obs, "Environment last_obs must be set on reset")
        np.testing.assert_allclose(raw_env.last_obs, obs, err_msg="Environment last_obs must match obs")

        raw_env.close()

    # -------------------------------------------------------------------------
    # Test 2: SafetyShieldWrapper Interception and Modification
    # -------------------------------------------------------------------------
    def test_wrapper_intercepts_and_modifies_unsafe_action(self):
        """Verify wrapper intercepts unsafe actions (e.g. reverse into rear obstacle) and modifies them."""
        raw_env = UAVFastEnvV2(config=self.config)
        wrapper = SafetyShieldWrapper(raw_env)

        # Place an obstacle directly behind the spawn point: spawn=(0,0), rear obs at (0, -0.60)
        options = {
            "start_pos": [0.0, 0.0, 2.0],
            "start_yaw": 0.0,  # Facing +X, so rear is -X
            "goal_pos": [5.0, 0.0, 2.0],
            "obstacles": [(-0.60, 0.0, 0.20)]  # Obstacle directly behind at X = -0.60m (distance 0.60m)
        }
        obs, _ = wrapper.reset(seed=402, options=options)

        # Command full reverse flight: vx = -1.0 (towards rear obstacle)
        unsafe_action = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        next_obs, reward, term, trunc, info = wrapper.step(unsafe_action)

        self.assertIsNotNone(wrapper.last_shielded_action)
        self.assertGreater(wrapper.total_interventions, 0, "Shield must intervene against rear obstacle")
        self.assertTrue(info["is_shield_intervened"], "info must report is_shield_intervened=True")
        self.assertGreater(info["shield_action_divergence"], 0.0, "Divergence must be positive when intervened")

        # Commanded vx must be clamped to prevent reverse collision
        shielded_vx = wrapper.last_shielded_action[0]
        self.assertGreater(shielded_vx, -0.2, f"Shielded vx ({shielded_vx}) should be clamped away from -1.0")

        # Divergence penalty must be subtracted from reward
        self.assertGreater(info["shield_divergence_penalty"], 0.0)

        raw_env.close()

    # -------------------------------------------------------------------------
    # Test 3: SafetyShieldWrapper Certified Fail-Safe on None / NaN / Inf
    # -------------------------------------------------------------------------
    def test_wrapper_failsafe_zero_velocity_on_none_or_nan(self):
        """Verify wrapper triggers emergency zero-velocity hover when obs is None, NaN, or Inf."""
        raw_env = UAVFastEnvV2(config=self.config)
        wrapper = SafetyShieldWrapper(raw_env)
        wrapper.reset(seed=403)

        arbitrary_action = np.array([0.8, -0.5, 0.3, -0.2], dtype=np.float32)

        # Case A: Observation is None
        wrapper.current_obs = None
        raw_env.last_obs = None
        action_out = wrapper.action(arbitrary_action)
        self.assertEqual(wrapper.fail_safe_events, 1, "Fail-safe event counter must increment on None")
        # Action handler unscale: zero velocity vx=0, vy=0, vz=0 maps to 0.0 in action space
        np.testing.assert_allclose(action_out[:3], [0.0, 0.0, 0.0], atol=1e-5,
                                  err_msg="Zero velocity command must be issued on None obs")

        # Case B: Observation contains NaN
        nan_obs = np.zeros(26, dtype=np.float32)
        nan_obs[3] = np.nan
        wrapper.current_obs = nan_obs
        action_out_nan = wrapper.action(arbitrary_action)
        self.assertEqual(wrapper.fail_safe_events, 2, "Fail-safe event counter must increment on NaN")
        self.assertTrue(np.all(np.isfinite(action_out_nan)), "Action must be finite even with NaN obs")
        np.testing.assert_allclose(action_out_nan[:3], [0.0, 0.0, 0.0], atol=1e-5,
                                  err_msg="Zero velocity command must be issued on NaN obs")

        # Case C: Observation contains Inf
        inf_obs = np.zeros(26, dtype=np.float32)
        inf_obs[18] = np.inf
        wrapper.current_obs = inf_obs
        action_out_inf = wrapper.action(arbitrary_action)
        self.assertEqual(wrapper.fail_safe_events, 3, "Fail-safe event counter must increment on Inf")
        self.assertTrue(np.all(np.isfinite(action_out_inf)), "Action must be finite even with Inf obs")
        np.testing.assert_allclose(action_out_inf[:3], [0.0, 0.0, 0.0], atol=1e-5)

        raw_env.close()

    # -------------------------------------------------------------------------
    # Test 4: Regime 1 (Dense Clutter Forest) Obstacle Count & Gap >= 1.10m
    # -------------------------------------------------------------------------
    def test_regime1_clutter_clearance_gte_1_10m(self):
        """
        Verify Regime 1 generates 7-10 obstacles with radii in [0.20, 0.32]m,
        and pairwise surface-to-surface gap >= 1.10m across 50 sample episodes.
        """
        env = UAVFastEnvV2(config=self.config)

        for ep in range(50):
            seed = 1000 + ep
            obs, info = env.reset(seed=seed, options={"active_regime": 1})
            obstacles = env.obstacles

            self.assertEqual(info["active_regime"], 1, f"Episode {ep} active_regime should be 1")
            self.assertGreaterEqual(len(obstacles), 7, f"Episode {ep}: N_obs {len(obstacles)} < 7")
            self.assertLessEqual(len(obstacles), 10, f"Episode {ep}: N_obs {len(obstacles)} > 10")

            # Check radii
            for ox, oy, r in obstacles:
                self.assertGreaterEqual(r, 0.20 - 1e-5, f"Radius {r} < 0.20m")
                self.assertLessEqual(r, 0.32 + 1e-5, f"Radius {r} > 0.32m")

            # Check pairwise surface-to-surface clearances
            n_obs = len(obstacles)
            for i in range(n_obs):
                for j in range(i + 1, n_obs):
                    dist_centers = math.hypot(obstacles[i][0] - obstacles[j][0],
                                              obstacles[i][1] - obstacles[j][1])
                    surf_gap = dist_centers - obstacles[i][2] - obstacles[j][2]
                    self.assertGreaterEqual(
                        surf_gap, 1.10 - 1e-4,
                        f"Ep {ep}: Inter-obstacle surface gap {surf_gap:.3f}m < 1.10m between {i} and {j}"
                    )

        env.close()

    # -------------------------------------------------------------------------
    # Test 17: Regimes 1 and 2 Retain 200-Step Limit
    # -------------------------------------------------------------------------
        env.close()
    def test_regime2_gate_aperture_between_1_00m_and_1_30m(self):
        """
        Verify Regime 2 generates dual gate posts with aperture width in [1.00m, 1.30m]
        and keeps the approach/departure zone clear across 50 sample episodes.
        """
        env = UAVFastEnvV2(config=self.config)

        for ep in range(50):
            seed = 2000 + ep
            obs, info = env.reset(seed=seed, options={"active_regime": 2})
            obstacles = env.obstacles

            self.assertEqual(info["active_regime"], 2)
            self.assertGreaterEqual(len(obstacles), 2, "Gate must have at least 2 primary posts")

            # First two obstacles are the primary gate posts (r = 0.25m)
            p1, p2 = obstacles[0], obstacles[1]
            self.assertAlmostEqual(p1[2], 0.25, delta=1e-4, msg="Post 1 radius must be 0.25m")
            self.assertAlmostEqual(p2[2], 0.25, delta=1e-4, msg="Post 2 radius must be 0.25m")

            center_dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
            aperture_width = center_dist - p1[2] - p2[2]

            self.assertGreaterEqual(aperture_width, 1.00 - 1e-4,
                                    f"Ep {ep}: Aperture width {aperture_width:.3f}m < 1.00m")
            self.assertLessEqual(aperture_width, 1.30 + 1e-4,
                                 f"Ep {ep}: Aperture width {aperture_width:.3f}m > 1.30m")

            # If flanking cylinders exist, verify lateral offset >= 2.0m from gate center
            gate_center = np.array([(p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0])
            for flank in obstacles[2:]:
                flank_dist = math.hypot(flank[0] - gate_center[0], flank[1] - gate_center[1])
                self.assertGreaterEqual(flank_dist, 2.0, f"Flanking cylinder too close to gate center: {flank_dist:.2f}m")

        env.close()

    # -------------------------------------------------------------------------
    # Test 17: Regimes 1 and 2 Retain 200-Step Limit
    # -------------------------------------------------------------------------
        env.close()
    def test_regime3_slalom_alternating_openings(self):
        """
        Verify Regime 3 generates 2 or 3 barrier stages with alternating lateral opening offsets
        (Stage 1 Left > 0, Stage 2 Right < 0) across 50 sample episodes.
        """
        env = UAVFastEnvV2(config=self.config)

        for ep in range(50):
            seed = 3000 + ep
            obs, info = env.reset(seed=seed, options={"active_regime": 3})
            obstacles = env.obstacles

            self.assertEqual(info["active_regime"], 3)
            # Each stage has 2 posts, so total obstacles = 4 (2 stages) or 6 (3 stages)
            self.assertIn(len(obstacles), (4, 6), f"Ep {ep}: Obstacle count {len(obstacles)} not in (4, 6)")

            # Check post radii
            for ox, oy, r in obstacles:
                self.assertAlmostEqual(r, 0.25, delta=1e-4)

            # Analyze Stage 1 and Stage 2 window centers relative to flight path
            spawn_xy = info["pos_world"][:2]
            goal_xy = info["goal_world"][:2]
            path_vec = goal_xy - spawn_xy
            path_len = np.linalg.norm(path_vec)
            d_unit = path_vec / max(0.1, path_len)
            u_trans = np.array([-d_unit[1], d_unit[0]])

            # Stage 1 window center
            w1_center = np.array([(obstacles[0][0] + obstacles[1][0]) / 2.0,
                                  (obstacles[0][1] + obstacles[1][1]) / 2.0])
            lat_offset_1 = float(np.dot(w1_center - spawn_xy, u_trans))

            # Stage 2 window center
            w2_center = np.array([(obstacles[2][0] + obstacles[3][0]) / 2.0,
                                  (obstacles[2][1] + obstacles[3][1]) / 2.0])
            lat_offset_2 = float(np.dot(w2_center - spawn_xy, u_trans))

            # Stage 1 must be Left (positive lateral offset > 0.5)
            self.assertGreater(lat_offset_1, 0.5, f"Ep {ep}: Stage 1 lateral offset {lat_offset_1:.2f} should be Left (>0.5)")
            # Stage 2 must be Right (negative lateral offset < -0.5)
            self.assertLess(lat_offset_2, -0.5, f"Ep {ep}: Stage 2 lateral offset {lat_offset_2:.2f} should be Right (<-0.5)")

        env.close()

    # -------------------------------------------------------------------------
    # Test 17: Regimes 1 and 2 Retain 200-Step Limit
    # -------------------------------------------------------------------------
        env.close()
    def test_action_divergence_quadratic_penalty_calculation(self):
        """
        Verify R_div = -w_div * ||a_raw - a_shield||^2:
        - When a_raw == a_shield, penalty is 0.0
        - When ||a_raw - a_shield||^2 = 1.0, penalty is exactly -0.25 * 1.0 = -0.25
        """
        env = UAVFastEnvV2(config=self.config)
        env.reset(seed=407)

        # 1. Zero divergence test
        a_safe = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        r_comp_zero = env.compute_obstacle_reward_v2(
            dist_prev=5.0, dist_curr=4.9,
            action_curr=a_safe, action_prev=a_safe,
            clearances=np.full(8, 5.0, dtype=np.float32),
            goal_body=np.array([5.0, 0.0, 0.0], dtype=np.float32),
            vel_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            is_success=False, is_out_of_bounds=False,
            is_ground_crash=False, is_obstacle_collision=False,
            raw_action=a_safe,
            shielded_action=a_safe
        )
        self.assertEqual(r_comp_zero.action_divergence_penalty, 0.0, "Zero divergence must incur zero penalty")

        # 2. Known divergence test: a_raw = [1.0, 0, 0, 0], a_shield = [0.0, 0, 0, 0] -> diff^2 = 1.0
        a_raw = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        a_shield = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        r_comp_div = env.compute_obstacle_reward_v2(
            dist_prev=5.0, dist_curr=4.9,
            action_curr=a_shield, action_prev=a_shield,
            clearances=np.full(8, 5.0, dtype=np.float32),
            goal_body=np.array([5.0, 0.0, 0.0], dtype=np.float32),
            vel_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            is_success=False, is_out_of_bounds=False,
            is_ground_crash=False, is_obstacle_collision=False,
            raw_action=a_raw,
            shielded_action=a_shield
        )
        expected_penalty = 0.25 * 1.0  # w_div = 0.25
        self.assertAlmostEqual(r_comp_div.action_divergence_penalty, expected_penalty, delta=1e-5)
        # Total reward must reflect subtraction of divergence penalty
        diff_in_total = r_comp_zero.total - r_comp_div.total
        self.assertAlmostEqual(diff_in_total, expected_penalty, delta=1e-5)

        # 3. Test through SafetyShieldWrapper.step()
        wrapper = SafetyShieldWrapper(env, w_div=0.25)
        wrapper.reset(seed=407)
        # Simulate step with raw action and verified divergence
        wrapper.last_raw_action = a_raw
        wrapper.last_shielded_action = a_shield
        # Call step: wrapper calculates divergence penalty
        obs, reward, term, trunc, info = wrapper.step(a_safe)
        self.assertIn("shield_action_divergence", info)
        self.assertIn("shield_divergence_penalty", info)

        env.close()

    # -------------------------------------------------------------------------
    # Test 17: Regimes 1 and 2 Retain 200-Step Limit
    # -------------------------------------------------------------------------
        env.close()
    def test_domain_randomization_parameter_ranges(self):
        """
        Verify that domain randomization draws parameters strictly within approved bounds:
        - tau_xy in [0.45, 0.75] s   (+/-25 % around the PX4-fitted 0.60 s)
        - a_max in [4.00, 8.00] m/s^2 (acceleration-setpoint cap; PX4 tilt limit ~9.8)
        - p_drop in [0.00, 0.15]
        - sigma_lidar in [0.01, 0.08] m
        - noise_pos in [0.01, 0.05] m
        - noise_vel in [0.02, 0.08] m/s
        - wind_std in [0.05, 0.15] m/s^2
        """
        env = UAVFastEnvV2(config=self.config)

        tau_samples = []
        accel_samples = []
        dropout_samples = []
        noise_samples = []

        for ep in range(100):
            seed = 5000 + ep
            obs, info = env.reset(seed=seed, options={"enable_domain_randomization": True})

            tau = env.dynamics.tau_velocity_xy
            accel = env.dynamics.max_horizontal_accel
            p_drop = env.active_lidar_dropout
            sig_lidar = env.active_lidar_noise

            tau_samples.append(tau)
            accel_samples.append(accel)
            dropout_samples.append(p_drop)
            noise_samples.append(sig_lidar)

            self.assertGreaterEqual(tau, 0.45 - 1e-5, f"tau_xy {tau} < 0.45s")
            self.assertLessEqual(tau, 0.75 + 1e-5, f"tau_xy {tau} > 0.75s")

            self.assertGreaterEqual(accel, 4.00 - 1e-5, f"a_max {accel} < 4.00 m/s^2")
            self.assertLessEqual(accel, 8.00 + 1e-5, f"a_max {accel} > 8.00 m/s^2")

            self.assertGreaterEqual(p_drop, 0.00 - 1e-5, f"p_drop {p_drop} < 0.0")
            self.assertLessEqual(p_drop, 0.15 + 1e-5, f"p_drop {p_drop} > 0.15")

            self.assertGreaterEqual(sig_lidar, 0.01 - 1e-5, f"sig_lidar {sig_lidar} < 0.01m")
            self.assertLessEqual(sig_lidar, 0.08 + 1e-5, f"sig_lidar {sig_lidar} > 0.08m")

        # Verify variance across samples (proving active stochastic draws)
        self.assertGreater(np.var(tau_samples), 1e-4, "tau_xy must be actively randomized")
        self.assertGreater(np.var(accel_samples), 1e-4, "a_max must be actively randomized")
        self.assertGreater(np.var(dropout_samples), 1e-4, "p_drop must be actively randomized")
        self.assertGreater(np.var(noise_samples), 1e-4, "sig_lidar must be actively randomized")

        env.close()

    # -------------------------------------------------------------------------
    # Test 17: Regimes 1 and 2 Retain 200-Step Limit
    # -------------------------------------------------------------------------
        env.close()
    def test_09_geometry_validator_no_resample_exhaustion(self):
        """
        Verify that 2D BFS validator successfully discovers traversable routes
        across 30 consecutive episodes of each regime (90 total) without exceptions.
        """
        env = UAVFastEnvV2(config=self.config)

        for regime_id in (1, 2, 3):
            traversable_count = 0
            for ep in range(30):
                seed = 6000 + regime_id * 100 + ep
                obs, info = env.reset(seed=seed, options={"active_regime": regime_id})
                if info.get("route_traversable", False):
                    traversable_count += 1

            pass_rate = traversable_count / 30.0
            self.assertGreaterEqual(
                pass_rate, 0.95,
                f"Regime {regime_id} route traversability rate ({pass_rate:.1%}) < 95%"
            )

        env.close()

    # -------------------------------------------------------------------------
    # Test 17: Regimes 1 and 2 Retain 200-Step Limit
    # -------------------------------------------------------------------------
        env.close()
    def test_10_contract_preservation_box26_box4(self):
        """Verify UAVFastNav-v2 with SafetyShieldWrapper preserves Box(26,) and Box(4,) contracts."""
        raw_env = UAVFastEnvV2(config=self.config)
        wrapper = SafetyShieldWrapper(raw_env)

        # Check spaces
        self.assertEqual(wrapper.observation_space.shape, (26,))
        self.assertEqual(wrapper.action_space.shape, (4,))
        np.testing.assert_allclose(wrapper.action_space.low, -1.0)
        np.testing.assert_allclose(wrapper.action_space.high, 1.0)

        obs, info = wrapper.reset(seed=701)
        self.assertEqual(obs.shape, (26,))
        self.assertTrue(np.all(np.isfinite(obs)))

        act = wrapper.action_space.sample()
        next_obs, r, term, trunc, info = wrapper.step(act)
        self.assertEqual(next_obs.shape, (26,))
        self.assertTrue(np.all(np.isfinite(next_obs)))
        self.assertTrue(isinstance(r, float))
        self.assertTrue(isinstance(term, bool))
        self.assertTrue(isinstance(trunc, bool))

        raw_env.close()

    # -------------------------------------------------------------------------
    # Test 11: Checkpoint Compatibility with [64, 64] MLP Architecture
    # -------------------------------------------------------------------------
    def test_11_checkpoint_compatibility_and_parameter_count(self):
        """
        Verify that ppo_m5_stage2b_remediated_64x64.zip loads cleanly into
        the redesigned Stage 3 environment and executes forward inference.
        """
        ckpt_path = "experiments/milestone5/models/ppo_m5_stage2b_remediated_64x64.zip"
        self.assertTrue(os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}")

        raw_env = UAVFastEnvV2(config=self.config)
        wrapper = SafetyShieldWrapper(raw_env)

        model = PPO.load(ckpt_path, env=wrapper, device="cpu")
        self.assertIsNotNone(model.policy)

        # Confirm policy network architecture is [64, 64]
        net_arch = model.policy.net_arch
        # In SB3, net_arch is either [64, 64] or dict(pi=[64, 64], vf=[64, 64])
        if isinstance(net_arch, dict):
            self.assertEqual(net_arch["pi"], [64, 64])
            self.assertEqual(net_arch["vf"], [64, 64])
        else:
            self.assertEqual(net_arch, [64, 64])

        # Test forward prediction
        obs, _ = wrapper.reset(seed=801)
        action, _ = model.predict(obs, deterministic=True)
        self.assertEqual(action.shape, (4,))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertTrue(np.all(action >= -1.0 - 1e-5) and np.all(action <= 1.0 + 1e-5))

        raw_env.close()

    # -------------------------------------------------------------------------
    # Test 12: Cryptographic Immutability of Frozen Baseline Checkpoints
    # -------------------------------------------------------------------------
    def test_12_pre_training_baseline_hashes_immutable(self):
        """Verify that all frozen model checkpoints remain bit-identical."""
        expected_hashes = {
            "experiments/milestone5/models/ppo_m5_stage2b_remediated_64x64.zip":
                "c2197f7052e90fa016cf90319677834197fc7139f367631843a6f36b8f5b2ae0",
            "experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip":
                "06907c9fa636e5f268ca55750107918370e445313ad1ac01207c15a4c5935308",
            "experiments/milestone2/models/best_model.zip":
                "a5a239a5f318bd18c3578a82c108a23841502f9dc30fe541ec5ae08fb5bd1a1f"
        }

        for path, expected_hash in expected_hashes.items():
            if os.path.exists(path):
                hasher = hashlib.sha256()
                with open(path, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                computed = hasher.hexdigest()
                self.assertEqual(computed, expected_hash, f"Checkpoint modified: {path}")

    # -------------------------------------------------------------------------
    # Test 13: CPU Inference & Shielding Latency Budget
    # -------------------------------------------------------------------------
    def test_13_cpu_inference_latency_budget(self):
        """
        Verify that policy forward inference + safety shield action filtering
        achieves mean latency < 0.80 ms and P95 latency < 1.00 ms on CPU across 500 steps.
        """
        ckpt_path = "experiments/milestone5/models/ppo_m5_stage2b_remediated_64x64.zip"
        if not os.path.exists(ckpt_path):
            self.skipTest("Checkpoint not available for latency profiling")

        raw_env = UAVFastEnvV2(config=self.config)
        wrapper = SafetyShieldWrapper(raw_env)
        model = PPO.load(ckpt_path, device="cpu")
        model.policy.eval()

        obs, _ = wrapper.reset(seed=901)
        durations = []

        # Warm-up (50 steps)
        for _ in range(50):
            with torch.no_grad():
                act, _ = model.predict(obs, deterministic=True)
            _ = wrapper.shield.filter_action(obs, act)

        # Profile 500 steps of pure policy inference + shield filtering (excluding sim step overhead)
        for _ in range(500):
            t0 = time.perf_counter()
            with torch.no_grad():
                act, _ = model.predict(obs, deterministic=True)
            shielded_act, _ = wrapper.shield.filter_action(obs, act)
            dt_ms = (time.perf_counter() - t0) * 1000.0
            durations.append(dt_ms)

            # Step environment to get new obs
            obs, _, term, trunc, _ = wrapper.step(act)
            if term or trunc:
                obs, _ = wrapper.reset()

        mean_ms = float(np.mean(durations))
        p95_ms = float(np.percentile(durations, 95))

        print(f"\n[LATENCY PROFILE] CPU Policy+Shield: Mean={mean_ms:.3f} ms, P95={p95_ms:.3f} ms")
        self.assertLess(mean_ms, 0.80, f"Mean latency {mean_ms:.3f}ms exceeds 0.80ms budget")
        self.assertLess(p95_ms, 1.00, f"P95 latency {p95_ms:.3f}ms exceeds 1.00ms budget")

        raw_env.close()

    # -------------------------------------------------------------------------
    # Test 14: Action Divergence Normalized Bounds & Units Verification
    # -------------------------------------------------------------------------
    def test_14_action_divergence_normalized_bounds_and_units(self):
        """
        Verify action divergence operates strictly on normalized Box(4,) actions [-1, 1]^4,
        is dimensionless, and reaches up to 3.25 under severe horizontal reversals.
        """
        # Case 1: severe reversal in horizontal plane: raw=(1.0, 1.0, 0, 0), shield=(-0.5, 0.0, 0, 0)
        a_raw = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
        a_shield = np.array([-0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        expected_div = (1.0 - (-0.5))**2 + (1.0 - 0.0)**2  # 1.5^2 + 1.0^2 = 2.25 + 1.00 = 3.25
        div = float(np.sum(np.square(a_raw - a_shield)))
        self.assertAlmostEqual(div, 3.25, delta=1e-5)
        self.assertAlmostEqual(expected_div, 3.25, delta=1e-5)

        # Case 2: Divergence with w_div = 0.10
        wrapper = SafetyShieldWrapper(UAVFastEnvV2(config=self.config), w_div=0.10)
        self.assertAlmostEqual(wrapper.w_div, 0.10, places=4)

    # -------------------------------------------------------------------------
    # Test 15: Stagnation Penalty When Hovering Far from Goal
    # -------------------------------------------------------------------------
    def test_15_stagnation_penalty_when_hovering(self):
        """
        Verify that hovering (speed < 0.15 m/s) far from goal (dist > 1.0m) incurs
        stagnation penalty of 0.05 per step, while active forward motion incurs 0.0.
        """
        env = UAVFastEnvV2(config=self.config)
        env.reset(seed=501)

        # Hovering state: vel = [0.0, 0.0, 0.0], dist = 4.0m
        r_hover = env.compute_obstacle_reward_v2(
            dist_prev=4.0, dist_curr=4.0,
            action_curr=np.zeros(4, dtype=np.float32), action_prev=np.zeros(4, dtype=np.float32),
            clearances=np.full(8, 5.0, dtype=np.float32),
            goal_body=np.array([4.0, 0.0, 0.0], dtype=np.float32),
            vel_body=np.array([0.05, 0.0, 0.0], dtype=np.float32),
            is_success=False, is_out_of_bounds=False,
            is_ground_crash=False, is_obstacle_collision=False
        )
        self.assertAlmostEqual(r_hover.stagnation_penalty, 0.05, places=4)

        # Active forward motion: vel = [1.0, 0.0, 0.0]
        r_active = env.compute_obstacle_reward_v2(
            dist_prev=4.0, dist_curr=3.9,
            action_curr=np.zeros(4, dtype=np.float32), action_prev=np.zeros(4, dtype=np.float32),
            clearances=np.full(8, 5.0, dtype=np.float32),
            goal_body=np.array([3.9, 0.0, 0.0], dtype=np.float32),
            vel_body=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            is_success=False, is_out_of_bounds=False,
            is_ground_crash=False, is_obstacle_collision=False
        )
        self.assertAlmostEqual(r_active.stagnation_penalty, 0.0, places=4)

        env.close()

    # -------------------------------------------------------------------------
    # Test 17: Regimes 1 and 2 Retain 200-Step Limit
    # -------------------------------------------------------------------------
        env.close()
    def test_16_regime2_tilt_bounded_to_15_degrees(self):
        """
        Verify that Regime 2 gates are tilted within [-15.0 deg, +15.0 deg],
        preserving traversable projected width >= 0.96m.
        """
        env = UAVFastEnvV2(config=self.config)

        for ep in range(50):
            obs, info = env.reset(seed=3000 + ep, options={"active_regime": 2})
            obstacles = env.obstacles
            p1, p2 = obstacles[0], obstacles[1]
            gate_vec = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            gate_len = np.linalg.norm(gate_vec)

            # Spawn to goal vector
            path_vec = env.goal_world[:2] - env.dynamics.state.pos_world[:2]
            path_len = np.linalg.norm(path_vec)
            d_unit = path_vec / path_len
            u_trans = np.array([-d_unit[1], d_unit[0]])

            # Angle between gate axis and transverse axis
            cos_tilt = abs(np.dot(gate_vec / gate_len, u_trans))
            tilt_deg = math.degrees(math.acos(np.clip(cos_tilt, -1.0, 1.0)))
            self.assertLessEqual(tilt_deg, 15.01, f"Ep {ep}: Gate tilt {tilt_deg:.2f} deg > 15 deg")

        env.close()

    # -------------------------------------------------------------------------
    # Test 17: Regimes 1 and 2 Retain 200-Step Limit
    # -------------------------------------------------------------------------
    def test_17_regimes_1_and_2_retain_200_step_limit(self):
        """
        Verify that Regimes 1 and 2 (as well as Stages 0, 1, 2) retain their
        strict 200-step episode horizon.
        """
        env = UAVFastEnvV2(config=self.config)

        # 1. Test Regime 1 (Clutter)
        obs, info = env.reset(seed=4001, options={"active_regime": 1})
        self.assertEqual(info["max_episode_steps"], 200)
        self.assertEqual(env.current_max_episode_steps, 200)
        zero_action = np.zeros(4, dtype=np.float32)

        for step in range(1, 250):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            if terminated or truncated:
                break

        self.assertEqual(env.step_count, 200)
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(info["termination_reason"], "episode_timeout")
        self.assertEqual(info["step"], 200)
        self.assertEqual(info["max_episode_steps"], 200)

        # 2. Test Regime 2 (Gates)
        obs, info = env.reset(seed=4002, options={"active_regime": 2})
        self.assertEqual(info["max_episode_steps"], 200)
        self.assertEqual(env.current_max_episode_steps, 200)

        for step in range(1, 250):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            if terminated or truncated:
                break

        self.assertEqual(env.step_count, 200)
        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertEqual(info["termination_reason"], "episode_timeout")

        # 3. Test Curriculum Stages 0, 1, 2
        for stage in [0, 1, 2]:
            obs, info = env.reset(seed=4003 + stage, options={"curriculum_stage": stage})
            self.assertEqual(info["max_episode_steps"], 200)
            self.assertEqual(env.current_max_episode_steps, 200)

        env.close()

    # -------------------------------------------------------------------------
    # Test 18: Regime 3 Receives Intended 300-Step Limit
    # -------------------------------------------------------------------------
    def test_18_regime_3_receives_300_step_limit(self):
        """
        Verify that Regime 3 (Sequential Slaloms) receives the intended
        300-step episode horizon and does not truncate prematurely at step 200.
        """
        env = UAVFastEnvV2(config=self.config)
        obs, info = env.reset(seed=5001, options={"active_regime": 3})

        self.assertEqual(info["max_episode_steps"], 300)
        self.assertEqual(env.current_max_episode_steps, 300)
        self.assertEqual(env.active_regime, 3)

        zero_action = np.zeros(4, dtype=np.float32)

        # Step up to 200 and assert no premature truncation
        for step in range(1, 201):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            self.assertFalse(truncated, f"Premature truncation at step {step} in Regime 3")
            self.assertFalse(terminated, f"Unexpected termination at step {step}")

        self.assertEqual(env.step_count, 200)
        self.assertEqual(info["step"], 200)
        self.assertEqual(info["max_episode_steps"], 300)
        self.assertEqual(info["termination_reason"], "active")

        # Continue stepping to 300
        for step in range(201, 350):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            if terminated or truncated:
                break

        self.assertEqual(env.step_count, 300)
        self.assertTrue(truncated, "Expected truncation at step 300")
        self.assertFalse(terminated)
        self.assertEqual(info["termination_reason"], "episode_timeout")
        self.assertEqual(info["step"], 300)
        self.assertEqual(info["max_episode_steps"], 300)

        env.close()

    # -------------------------------------------------------------------------
    # Test 19: Environment Terminates Correctly at Configured Custom Limit
    # -------------------------------------------------------------------------
    def test_19_environment_terminates_correctly_at_configured_limit(self):
        """
        Verify that custom episode limits in configuration are strictly honored.
        """
        custom_cfg = UAVObstacleEnvConfigV2(
            curriculum_stage=3,
            max_episode_steps_default=120,
            max_episode_steps_slalom=240
        )
        env = UAVFastEnvV2(config=custom_cfg)
        zero_action = np.zeros(4, dtype=np.float32)

        # Regime 1 custom limit (120)
        obs, info = env.reset(seed=6001, options={"active_regime": 1})
        self.assertEqual(info["max_episode_steps"], 120)
        for _ in range(200):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            if terminated or truncated:
                break
        self.assertEqual(env.step_count, 120)
        self.assertTrue(truncated)

        # Regime 3 custom limit (240)
        obs, info = env.reset(seed=6002, options={"active_regime": 3})
        self.assertEqual(info["max_episode_steps"], 240)
        for _ in range(300):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            if terminated or truncated:
                break
        self.assertEqual(env.step_count, 240)
        self.assertTrue(truncated)

        env.close()

    # -------------------------------------------------------------------------
    # Test 20: Goal Success and Collision Independent of Timeout
    # -------------------------------------------------------------------------
    def test_20_goal_success_and_collision_independent_of_timeout(self):
        """
        Verify that goal success and obstacle collision terminate the episode
        with terminated=True, truncated=False regardless of horizon.
        """
        env = UAVFastEnvV2(config=self.config)

        # 1. Goal Success: Spawn drone directly at goal
        obs, info = env.reset(seed=7001, options={
            "active_regime": 3,
            "start_pos": [0.0, 0.0, 1.5],
            "goal_pos": [0.05, 0.0, 1.5],
            "obstacles": []
        })
        zero_action = np.zeros(4, dtype=np.float32)
        # Step once while stationary at goal
        obs, reward, terminated, truncated, info = env.step(zero_action)

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["termination_reason"], "goal_reached_and_stabilized")
        self.assertLess(env.step_count, 300)

        # 2. Obstacle Collision: Command full speed into an obstacle
        obs, info = env.reset(seed=7002, options={
            "active_regime": 1,
            "start_pos": [0.0, 0.0, 1.5],
            "obstacles": [(1.0, 0.0, 0.3)]
        })
        # Command forward flight towards obstacle at x=1.0
        fwd_action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        collided = False
        for _ in range(50):
            obs, reward, terminated, truncated, info = env.step(fwd_action)
            if terminated or truncated:
                collided = True
                break

        self.assertTrue(collided)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["is_crashed"])
        self.assertIn("obstacle_collision", info["termination_reason"])
        self.assertLess(env.step_count, 200)

        env.close()

    # -------------------------------------------------------------------------
    # Test 21: Reset Between Regimes Does Not Leak Horizon
    # -------------------------------------------------------------------------
    def test_21_reset_between_regimes_no_horizon_leakage(self):
        """
        Verify that alternating resets between Regime 3 (300 steps) and Regimes 1/2
        (200 steps) strictly resets the horizon with zero state leakage.
        """
        env = UAVFastEnvV2(config=self.config)
        zero_action = np.zeros(4, dtype=np.float32)

        sequence = [3, 1, 3, 2, 3, 1]
        expected_limits = [300, 200, 300, 200, 300, 200]

        for i, (regime, exp_limit) in enumerate(zip(sequence, expected_limits)):
            obs, info = env.reset(seed=8000 + i, options={"active_regime": regime})
            self.assertEqual(info["max_episode_steps"], exp_limit, f"Cycle {i}: wrong reset info limit")
            self.assertEqual(env.current_max_episode_steps, exp_limit, f"Cycle {i}: wrong property limit")
            self.assertEqual(env.active_regime, regime, f"Cycle {i}: wrong active regime")

            for _ in range(exp_limit + 50):
                obs, reward, terminated, truncated, info = env.step(zero_action)
                if terminated or truncated:
                    break

            self.assertEqual(env.step_count, exp_limit, f"Cycle {i} (Regime {regime}): terminated at {env.step_count} instead of {exp_limit}")
            self.assertTrue(truncated)
            self.assertFalse(terminated)
            self.assertEqual(info["termination_reason"], "episode_timeout")

        env.close()

    # -------------------------------------------------------------------------
    # Test 22: Episode Metrics Report Correct Termination Reason and Step Count
    # -------------------------------------------------------------------------
    def test_22_episode_metrics_report_correct_termination_reason_and_steps(self):
        """
        Verify that telemetry in info accurately reflects step count, horizon, and reason.
        """
        env = UAVFastEnvV2(config=self.config)
        zero_action = np.zeros(4, dtype=np.float32)

        obs, info = env.reset(seed=9001, options={"active_regime": 3})
        self.assertIn("max_episode_steps", info)
        self.assertEqual(info["max_episode_steps"], 300)
        self.assertEqual(info["step"], 0)

        for s in range(1, 301):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            self.assertEqual(info["step"], s)
            self.assertEqual(info["max_episode_steps"], 300)
            self.assertEqual(info["active_regime"], 3)
            if truncated:
                self.assertEqual(s, 300)
                self.assertEqual(info["termination_reason"], "episode_timeout")

        env.close()

    # -------------------------------------------------------------------------
    # Test 23: Full Wrapper Stack (Gymnasium TimeLimit + SafetyShieldWrapper)
    # -------------------------------------------------------------------------
    def test_23_full_wrapper_stack_integration(self):
        """
        Verify that gym.make('UAVFastNav-v2') wrapped with SafetyShieldWrapper
        correctly respects the 200-step limit for Regimes 1/2 and 300-step limit
        for Regime 3 through all wrapper layers.
        """
        raw_env = gym.make("UAVFastNav-v2", config=self.config)
        shielded_env = SafetyShieldWrapper(raw_env, w_div=0.10)
        zero_action = np.zeros(4, dtype=np.float32)

        # 1. Regime 1 via wrapped stack -> 200 steps
        obs, info = shielded_env.reset(seed=9101, options={"active_regime": 1})
        self.assertEqual(info["max_episode_steps"], 200)
        for _ in range(250):
            obs, reward, terminated, truncated, info = shielded_env.step(zero_action)
            if terminated or truncated:
                break
        self.assertEqual(raw_env.unwrapped.step_count, 200)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "episode_timeout")

        # 2. Regime 3 via wrapped stack -> 300 steps
        obs, info = shielded_env.reset(seed=9102, options={"active_regime": 3})
        self.assertEqual(info["max_episode_steps"], 300)
        for _ in range(350):
            obs, reward, terminated, truncated, info = shielded_env.step(zero_action)
            if terminated or truncated:
                break
        self.assertEqual(raw_env.unwrapped.step_count, 300)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "episode_timeout")

        shielded_env.close()

    # -------------------------------------------------------------------------
    # Test 24: Mid-Episode Mutation Does Not Change Active Horizon
    # -------------------------------------------------------------------------
    def test_24_mid_episode_curriculum_or_regime_mutation_does_not_change_horizon(self):
        """
        Verify that mid-episode mutations to curriculum_stage or forced_regime
        do not leak or mutate the active locked episode horizon.
        """
        env = UAVFastEnvV2(config=self.config)
        zero_action = np.zeros(4, dtype=np.float32)

        # Reset in Regime 3 -> locked to 300 steps
        obs, info = env.reset(seed=9201, options={"active_regime": 3})
        self.assertEqual(env.current_max_episode_steps, 300)
        self.assertEqual(info["max_episode_steps"], 300)

        for step in range(1, 101):
            obs, reward, terminated, truncated, info = env.step(zero_action)

        # Attempt mid-episode mutation
        env.set_curriculum_stage(1)
        env.forced_regime = 1

        # Assert active episode horizon remains strictly 300
        self.assertEqual(env.current_max_episode_steps, 300)

        # Step through step 200: must not truncate
        for step in range(101, 201):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            self.assertFalse(truncated, f"Mid-episode mutation caused premature truncation at step {step}")

        # Continue to step 300
        for step in range(201, 350):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            if terminated or truncated:
                break

        self.assertEqual(env.step_count, 300)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "episode_timeout")

        # Subsequent reset: now honors curriculum stage 1 -> 200 steps
        obs, info = env.reset(seed=9202)
        self.assertEqual(env.current_max_episode_steps, 200)
        self.assertEqual(info["max_episode_steps"], 200)

        env.close()

    # -------------------------------------------------------------------------
    # Test 25: Minimum Clearance Metric Semantics and Hull Clearance
    # -------------------------------------------------------------------------
    def test_25_clearance_metric_semantics_and_hull_clearance(self):
        """
        Verify the exact mathematical definition of clearance metrics:
          - min_surface_dist_m: Euclidean distance from UAV center to obstacle surface
          - min_hull_clearance_m: Physical margin from UAV hull (radius 0.35m) to surface
          - Standoff boundary: min_surface_dist_m = 0.85m -> min_hull_clearance_m = 0.50m
          - Collision boundary: min_surface_dist_m <= 0.35m -> min_hull_clearance_m <= 0.00m
        """
        env = UAVFastEnvV2(config=self.config)

        # 1. Test exact standoff boundary (center dist = 1.35m, radius = 0.50m -> surface dist = 0.85m)
        obs, info = env.reset(seed=9301, options={
            "curriculum_stage": 1,
            "start_pos": [0.0, 0.0, 1.5],
            "goal_pos": [5.0, 0.0, 1.5],
            "obstacles": [(1.35, 0.0, 0.50)]
        })

        expected_surf_dist = 1.35 - 0.50  # 0.85 m
        expected_hull_dist = 0.85 - 0.35  # 0.50 m
        self.assertAlmostEqual(info["min_surface_dist_m"], expected_surf_dist, places=3)
        self.assertAlmostEqual(info["min_hull_clearance_m"], expected_hull_dist, places=3)

        # Step once with zero action (hovering at spawn)
        zero_action = np.zeros(4, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(zero_action)
        self.assertAlmostEqual(info["min_surface_dist_m"], expected_surf_dist, places=2)
        self.assertAlmostEqual(info["min_hull_clearance_m"], expected_hull_dist, places=2)
        self.assertFalse(info["is_crashed"])

        # 2. Test collision boundary penetration (center dist = 0.84m, radius = 0.50m -> surface dist = 0.34m <= 0.35m)
        obs, info = env.reset(seed=9302, options={
            "curriculum_stage": 1,
            "enable_domain_randomization": False,
            "start_pos": [0.0, 0.0, 1.5],
            "goal_pos": [5.0, 0.0, 1.5],
            "obstacles": [(0.84, 0.0, 0.50)]
        })
        self.assertAlmostEqual(info["min_surface_dist_m"], 0.34, places=2)
        self.assertAlmostEqual(info["min_hull_clearance_m"], -0.01, places=2)

        # Step: collision triggers immediately because surface_dist <= 0.35m (hull contact/penetration)
        obs, reward, terminated, truncated, info = env.step(zero_action)
        self.assertTrue(terminated)
        self.assertTrue(info["is_crashed"])
        self.assertIn("obstacle_collision", info["termination_reason"])

        env.close()


if __name__ == "__main__":
    unittest.main()
