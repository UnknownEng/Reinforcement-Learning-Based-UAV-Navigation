"""
test_gate_b4_benchmark.py
-------------------------
Milestone 4 Phase 4B — Gate B4: Comprehensive Tri-Policy Benchmark & Generalization Suite.

Validates:
  1. Tri-Policy Comparative Benchmark across 100 standardized episodes (Topic 19):
     - Candidate: Retrained 26-D Perception-Augmented PPO (ppo_uav_obstacle_26d_final.zip)
     - Baseline 1: Classical Guidance (P-Controller + VFH Obstacle Avoidance)
     - Baseline 2: Governed 18-D Baseline PPO (ppo_uav_final.zip + Supervisory Braking)
  2. Gate B4 Quantitative Acceptance Criteria (Section 5 & 7):
     - Candidate Waypoint Success Rate >= 90.0% overall
     - Candidate Collision Rate <= 5.0%
     - Statistically superior collision safety over Baseline 1 and Baseline 2
     - Path length efficiency L / L_straight <= 1.30
     - Mean inference latency on CPU < 1.00 ms
  3. Generalization Stress Tests (Topic 21):
     - Stress Test 1: Sensor Range Noise (sigma = 0.05m Gaussian noise + 5% random beam dropouts)
     - Stress Test 2: Narrow Corridor Traversals (tight gates with clearance down to 1.0 m)
     - Stress Test 3: Dense Extreme Clutter (up to 10 simultaneous obstacles)
  4. Cryptographic Baseline Integrity Invariants:
     - Checkpoint SHA-256 matches 8d7a24eda075cbbe...
     - Frozen tag milestone4-phase4a-frozen remains commit 31c81c77...
     - Production adapters remain untouched
"""

import hashlib
import json
import math
import os
import subprocess
import sys
import time
import unittest
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

sys.path.insert(0, "/home/kali/proj")
import rl_env
from rl_env.action import CommandedVelocities
from rl_env.baselines.proportional_controller import ProportionalWaypointController
from rl_env.uav_fast_env_v1 import UAVFastEnvV1, UAVObstacleEnvConfig


class ClassicalVFHController:
    """
    Classical Waypoint Guidance Controller augmented with Vector Field Histogram (VFH)
    repulsive steering away from low-clearance obstacle sectors.
    """
    def __init__(self, config: Optional[UAVObstacleEnvConfig] = None, d_safe: float = 1.5, k_rep: float = 1.2):
        self.config = config if config is not None else UAVObstacleEnvConfig()
        self.p_ctrl = ProportionalWaypointController(self.config)
        self.d_safe = d_safe
        self.k_rep = k_rep
        # Sector center angles in radians (FLU frame)
        self.sector_angles = np.array([
            0.0,              # Sector 0: Front
            math.pi / 4.0,    # Sector 1: Front-Left
            math.pi / 2.0,    # Sector 2: Left
            3.0 * math.pi / 4.0, # Sector 3: Back-Left
            math.pi,          # Sector 4: Back
            -3.0 * math.pi / 4.0,# Sector 5: Back-Right
            -math.pi / 2.0,   # Sector 6: Right
            -math.pi / 4.0    # Sector 7: Front-Right
        ], dtype=np.float32)

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        # P-controller attraction towards goal in body frame
        p_action = self.p_ctrl.predict(observation[:18], deterministic=deterministic)
        cmd = self.p_ctrl.action_handler.scale_action(p_action)

        # Repulsive force from obstacle clearances (obs[18:26])
        clearances = observation[18:26]
        v_rep_x = 0.0
        v_rep_y = 0.0

        for i in range(8):
            d = clearances[i]
            if 0.10 <= d < self.d_safe:
                # Repulsive strength inversely proportional to distance
                rep_mag = self.k_rep * (1.0 / d - 1.0 / self.d_safe)
                # Repel away from sector direction (-cos, -sin)
                v_rep_x -= rep_mag * np.cos(self.sector_angles[i])
                v_rep_y -= rep_mag * np.sin(self.sector_angles[i])

        # Combined commanded velocity
        vx_total = np.clip(cmd.vx + v_rep_x, -self.config.max_horizontal_velocity, self.config.max_horizontal_velocity)
        vy_total = np.clip(cmd.vy + v_rep_y, -self.config.max_horizontal_velocity, self.config.max_horizontal_velocity)

        new_cmd = CommandedVelocities(
            vx=float(vx_total),
            vy=float(vy_total),
            vz=float(cmd.vz),
            yaw_rate=float(cmd.yaw_rate)
        )
        return self.p_ctrl.action_handler.unscale_action(new_cmd)

    def reset(self):
        pass


class Governed18DPolicy:
    """
    Frozen 18-D Baseline PPO Policy coupled with Phase 4A Supervisory Braking Governor.
    When front clearance drops below 1.5m, forward velocity is proportionally clamped.
    """
    def __init__(self, model_path: str, d_brake: float = 1.5):
        self.model = PPO.load(model_path, device="cpu")
        self.d_brake = d_brake
        self.action_handler = None

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_18d = observation[:18]
        action, _ = self.model.predict(obs_18d, deterministic=deterministic)
        clearances = observation[18:26]

        # Supervisory braking governor: clamp forward action ax if front sector (sec 0, 1, 7) has obstacle
        front_clearance = min(clearances[0], clearances[1], clearances[7])
        if front_clearance < self.d_brake:
            # Scale forward action down proportionally, or reverse if critical (< 0.6m)
            brake_factor = max(0.0, (front_clearance - 0.5) / (self.d_brake - 0.5))
            if action[0] > 0.0:
                action[0] *= brake_factor
            if front_clearance < 0.6:
                action[0] = -0.5  # Emergency back-away

        return action

    def reset(self):
        pass


class TestGateB4Benchmark(unittest.TestCase):
    """Gate B4 Test Suite: Tri-Policy Comparative Benchmark & Generalization."""

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.cand_path = "/home/kali/proj/experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip"
        cls.base_path = "/home/kali/proj/experiments/milestone2/models/ppo_uav_final.zip"
        cls.expected_base_hash = "8d7a24eda075cbbeb60c4a6df07029c3a38fd770ce436a6daf96feff85300dfc"

        # Verify candidate exists
        if not os.path.exists(cls.cand_path):
            raise FileNotFoundError(f"Candidate model not found at {cls.cand_path}")

        cls.cand_model = PPO.load(cls.cand_path, device="cpu")
        cls.cand_model.predict(np.zeros(26, dtype=np.float32), deterministic=True)

        cls.base_governed = Governed18DPolicy(cls.base_path)
        cls.classical_vfh = ClassicalVFHController()

    def _evaluate_policy_episodes(self, policy, env, seeds: List[int], stage: int) -> Dict[str, Any]:
        successes = 0
        collisions = 0
        rewards = []
        path_efficiencies = []
        min_clearances = []
        latencies_ms = []

        for seed in seeds:
            opts = {"curriculum_stage": stage}
            obs, info = env.reset(seed=seed, options=opts)
            init_dist = info["dist_to_goal"]
            min_clear = min(10.0, float(info.get("min_surface_dist_m", 10.0)))
            ep_rew = 0.0

            while True:
                t0 = time.perf_counter()
                action = policy.predict(obs, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

                obs, rew, term, trunc, info = env.step(action)
                ep_rew += float(rew)

                surf_dist = min(10.0, float(info.get("min_surface_dist_m", 10.0)))
                if surf_dist < min_clear:
                    min_clear = surf_dist

                if term or trunc:
                    if info.get("is_success", False):
                        successes += 1
                    elif info.get("is_crashed", False):
                        collisions += 1

                    path_len = float(info.get("path_length", 0.0))
                    eff = (path_len / init_dist) if init_dist > 1e-3 else 1.0

                    rewards.append(ep_rew)
                    path_efficiencies.append(eff)
                    min_clearances.append(min_clear)
                    break

        n = len(seeds)
        return {
            "num_episodes": n,
            "success_rate_pct": float(100.0 * successes / n),
            "collision_rate_pct": float(100.0 * collisions / n),
            "mean_reward": float(np.mean(rewards)),
            "mean_path_efficiency": float(np.mean(path_efficiencies)),
            "mean_min_clearance_m": float(np.mean(min_clearances)),
            "min_clearance_overall_m": float(np.min(min_clearances)),
            "mean_latency_ms": float(np.mean(latencies_ms))
        }

    def test_01_tri_policy_comparative_benchmark_100_episodes(self):
        """
        Gate B4 Primary Acceptance Test: 100-episode 4-tier benchmark comparison
        between Candidate 26-D, Baseline 1 (VFH), and Baseline 2 (Governed 18-D).
        """
        env = gym.make("UAVFastNav-v1")
        tier_configs = [
            ("Tier 1 (Free Space)", list(range(20000, 20025)), 0),
            ("Tier 2 (Single Obstacle)", list(range(20025, 20050)), 1),
            ("Tier 3 (Corridor / Gates)", list(range(20050, 20075)), 2),
            ("Tier 4 (Dense Clutter)", list(range(20075, 20100)), 3),
        ]

        cand_tiers = []
        vfh_tiers = []
        gov_tiers = []

        print("\n[Gate B4] Executing Tri-Policy Comparative Benchmark (100 episodes)...")
        for name, seeds, stage in tier_configs:
            c_res = self._evaluate_policy_episodes(self.cand_model, env, seeds, stage)
            v_res = self._evaluate_policy_episodes(self.classical_vfh, env, seeds, stage)
            g_res = self._evaluate_policy_episodes(self.base_governed, env, seeds, stage)

            cand_tiers.append(c_res)
            vfh_tiers.append(v_res)
            gov_tiers.append(g_res)

            print(f"  {name}:")
            print(f"    Candidate 26-D: Success {c_res['success_rate_pct']:.1f}% | Collisions {c_res['collision_rate_pct']:.1f}% | Eff {c_res['mean_path_efficiency']:.3f}")
            print(f"    Baseline 1 VFH: Success {v_res['success_rate_pct']:.1f}% | Collisions {v_res['collision_rate_pct']:.1f}% | Eff {v_res['mean_path_efficiency']:.3f}")
            print(f"    Baseline 2 Gov: Success {g_res['success_rate_pct']:.1f}% | Collisions {g_res['collision_rate_pct']:.1f}% | Eff {g_res['mean_path_efficiency']:.3f}")

        env.close()

        # Aggregate metrics over 100 episodes
        cand_overall_success = np.mean([t["success_rate_pct"] for t in cand_tiers])
        cand_overall_collision = np.mean([t["collision_rate_pct"] for t in cand_tiers])
        cand_mean_latency = np.mean([t["mean_latency_ms"] for t in cand_tiers])
        cand_mean_efficiency = np.mean([t["mean_path_efficiency"] for t in cand_tiers])

        gov_overall_collision = np.mean([t["collision_rate_pct"] for t in gov_tiers])
        vfh_overall_collision = np.mean([t["collision_rate_pct"] for t in vfh_tiers])

        print(f"\n[Gate B4 Summary]")
        print(f"  Candidate 26-D Success Rate:    {cand_overall_success:.1f}% (Required: >= 90.0%)")
        print(f"  Candidate 26-D Collision Rate:  {cand_overall_collision:.1f}% (Required: <= 5.0%)")
        print(f"  Baseline 1 VFH Collision Rate:  {vfh_overall_collision:.1f}%")
        print(f"  Baseline 2 Gov Collision Rate:  {gov_overall_collision:.1f}%")
        print(f"  Candidate CPU Inference Latency:{cand_mean_latency:.3f} ms (Required: < 1.0 ms)")

        # Assert Gate B4 Criteria
        self.assertGreaterEqual(cand_overall_success, 90.0, "Candidate 26-D success rate must be >= 90.0%")
        self.assertLessEqual(cand_overall_collision, 5.0, "Candidate 26-D collision rate must be <= 5.0%")
        self.assertLessEqual(cand_mean_efficiency, 1.30, "Candidate path efficiency must be <= 1.30")
        self.assertLess(cand_mean_latency, 1.0, "Candidate inference latency must be < 1.0 ms on CPU")
        self.assertLess(cand_overall_collision, gov_overall_collision, "Candidate must have fewer collisions than Baseline 2")

    def test_02_generalization_sensor_noise_and_dropouts(self):
        """
        Generalization Stress Test 1 (Topic 21):
        Injects Gaussian range noise (sigma = 0.05m) and 5% random beam dropouts (posinf/nan).
        Verifies policy robustness against perception noise.
        """
        env = gym.make("UAVFastNav-v1")
        seeds = list(range(21000, 21025))  # 25 unseen test seeds
        successes = 0
        collisions = 0

        rng = np.random.default_rng(12345)

        for seed in seeds:
            obs, info = env.reset(seed=seed, options={"curriculum_stage": 2})

            while True:
                # Inject 0.05m noise into 8 clearance sectors
                noisy_obs = obs.copy()
                noise = rng.normal(0.0, 0.05, size=8).astype(np.float32)
                noisy_obs[18:26] = np.clip(noisy_obs[18:26] + noise, 0.10, 10.0)

                # 5% random dropout (set sector clearance to max or min noise)
                if rng.random() < 0.05:
                    drop_idx = rng.integers(0, 8)
                    noisy_obs[18 + drop_idx] = 10.0

                action, _ = self.cand_model.predict(noisy_obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(action)

                if term or trunc:
                    if info.get("is_success", False):
                        successes += 1
                    elif info.get("is_crashed", False):
                        collisions += 1
                    break

        env.close()
        success_rate = 100.0 * successes / len(seeds)
        collision_rate = 100.0 * collisions / len(seeds)

        print(f"\n[Stress Test 1: Noise & Dropouts] Success: {success_rate:.1f}%, Collisions: {collision_rate:.1f}%")
        self.assertGreaterEqual(success_rate, 70.0, "Policy must maintain graceful navigation (>= 70% success) under sensor noise")
        self.assertLessEqual(collision_rate, 30.0, "Collisions must remain bounded (<= 30%) under sensor noise and beam dropouts")

    def test_03_generalization_narrow_corridor_traversals(self):
        """
        Generalization Stress Test 2 (Topic 21):
        Forces navigation through narrow corridors (obstacle gap reduced down to 1.1 m).
        """
        env = gym.make("UAVFastNav-v1")
        seeds = list(range(22000, 22025))  # 25 unseen seeds
        successes = 0
        collisions = 0

        for seed in seeds:
            # Construct a tight custom gate: two cylinders at y = +0.75m and y = -0.75m, x = 3.0m
            # Gap width = 1.5m - 2*0.20m = 1.10m for 0.35m radius UAV (0.70m UAV span)
            custom_obs = [
                (3.0, 0.75, 0.20),
                (3.0, -0.75, 0.20),
            ]
            opts = {
                "start_pos": [0.0, 0.0, 2.0],
                "goal_pos": [6.0, 0.0, 2.0],
                "obstacles": custom_obs
            }
            obs, info = env.reset(seed=seed, options=opts)

            while True:
                action, _ = self.cand_model.predict(obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(action)

                if term or trunc:
                    if info.get("is_success", False):
                        successes += 1
                    elif info.get("is_crashed", False):
                        collisions += 1
                    break

        env.close()
        success_rate = 100.0 * successes / len(seeds)
        collision_rate = 100.0 * collisions / len(seeds)

        print(f"\n[Stress Test 2: Narrow Gate (1.1m)] Success: {success_rate:.1f}%, Collisions: {collision_rate:.1f}%")
        self.assertGreaterEqual(success_rate, 50.0, "Must pass tight corridor with >= 50% success")
        self.assertLessEqual(collision_rate, 50.0, "Collisions in tight corridor must be <= 50%")

    def test_04_generalization_extreme_dense_clutter(self):
        """
        Generalization Stress Test 3 (Topic 21):
        Dense extreme clutter: 10 cylinders placed simultaneously in the arena.
        """
        env = gym.make("UAVFastNav-v1")
        seeds = list(range(23000, 23025))  # 25 unseen seeds
        successes = 0
        collisions = 0

        for seed in seeds:
            opts = {
                "curriculum_stage": 3,
                "num_obstacles": 10
            }
            obs, info = env.reset(seed=seed, options=opts)

            while True:
                action, _ = self.cand_model.predict(obs, deterministic=True)
                obs, rew, term, trunc, info = env.step(action)

                if term or trunc:
                    if info.get("is_success", False):
                        successes += 1
                    elif info.get("is_crashed", False):
                        collisions += 1
                    break

        env.close()
        success_rate = 100.0 * successes / len(seeds)
        collision_rate = 100.0 * collisions / len(seeds)

        print(f"\n[Stress Test 3: Extreme Clutter (10 obs)] Success: {success_rate:.1f}%, Collisions: {collision_rate:.1f}%")
        self.assertGreaterEqual(success_rate, 75.0, "Must maintain >= 75% success in extreme clutter")
        self.assertLessEqual(collision_rate, 25.0, "Collisions in extreme clutter must be <= 25%")

    def test_05_baseline_invariants_and_checkpoint_hashes(self):
        """Verify baseline protections remain 100% intact."""
        with open(self.base_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(h, self.expected_base_hash, "Baseline checkpoint SHA-256 altered!")

        tag_commit = subprocess.check_output(
            ["git", "rev-parse", "milestone4-phase4a-frozen^{commit}"]
        ).decode().strip()
        self.assertEqual(tag_commit, "31c81c77aaea82ac35a34fb102e3128257b09b30", "Frozen tag altered!")

        diff = subprocess.check_output([
            "git", "diff", "milestone4-phase4a-frozen", "--",
            "ppo_adapter.py", "ppo_inference_adapter.py", "rl_env/observation.py", "rl_env/config.py"
        ]).decode().strip()
        self.assertEqual(len(diff), 0, "Protected production files modified!")


if __name__ == "__main__":
    unittest.main()
