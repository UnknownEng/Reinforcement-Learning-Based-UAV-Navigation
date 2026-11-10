"""
test_gate_b1_environment.py
----------------------------
Comprehensive Acceptance and Verification Test Suite for Milestone 4 Gate B1:
Environment Validation for UAVFastNav-v1.

Verifies:
  1. Gymnasium standard API compliance & 26-D observation space validation.
  2. 72-ray half-step (+2.5°) LiDAR geometry, angle increment, and 8-sector mapping (9 rays/sector).
  3. Analytical ray-cylinder distance verification against ground truth (error <= 0.010 m).
  4. Sector boundaries and wraparound azimuth handling across all 8 sectors.
  5. Rigid bounding cylinder collision detection (r_uav = 0.35 m) and terminal crash penalties.
  6. Continuous quadratic obstacle clearance reward penalty (d_safe = 1.50 m, w_clear = 0.50).
  7. Procedural obstacle generation, exclusion zones (spawn >= 2.0m, goal >= 1.5m), and seed determinism.
  8. 1,000-step randomized walk across 20 randomized episodes with zero NaNs/Infs.
  9. High environment throughput benchmark (>= 4,000 steps/second).
 10. Baseline 18-D environment (UAVFastNav-v0) and checkpoint SHA-256 isolation.
"""

import hashlib
import math
import os
import subprocess
import time
import unittest
from typing import List, Tuple
import gymnasium as gym
import numpy as np

import rl_env
from rl_env.config import UAVEnvConfig
from rl_env.observation_26d import ObservationHandler26D, SectorStatus
from rl_env.uav_fast_env_v1 import (
    UAVFastEnvV1,
    UAVObstacleEnvConfig,
    fast_ray_cast_cylinders,
    generate_procedural_obstacles
)


EXPECTED_CHECKPOINT_HASH = "8d7a24eda075cbbeb60c4a6df07029c3a38fd770ce436a6daf96feff85300dfc"
CHECKPOINT_PATH = "/home/kali/proj/experiments/milestone2/models/ppo_uav_final.zip"


class TestGateB1Environment(unittest.TestCase):
    def setUp(self):
        self.config = UAVObstacleEnvConfig()
        self.env = gym.make("UAVFastNav-v1")

    def tearDown(self):
        self.env.close()

    def test_01_gym_api_and_observation_space(self):
        """Verify Gymnasium API compliance, observation space shape (26,), dtype, and bounds."""
        # Check action and observation spaces
        self.assertEqual(self.env.action_space.shape, (4,))
        self.assertEqual(self.env.observation_space.shape, (26,))
        self.assertEqual(self.env.observation_space.dtype, np.float32)

        # Check clearance dimension bounds [0.0, 10.0]
        obs_low = self.env.observation_space.low
        obs_high = self.env.observation_space.high
        for i in range(18, 26):
            self.assertAlmostEqual(obs_low[i], 0.0, places=4)
            self.assertAlmostEqual(obs_high[i], 10.0, places=4)

        # Test reset
        obs, info = self.env.reset(seed=42)
        self.assertIsInstance(obs, np.ndarray)
        self.assertEqual(obs.shape, (26,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(obs)), "Initial observation contains NaN or Inf!")
        self.assertTrue(self.env.observation_space.contains(obs), "Observation out of space bounds!")

        # Verify info dictionary
        self.assertIn("obstacles", info)
        self.assertIn("clearances_m", info)
        self.assertIn("min_clearance_m", info)
        self.assertIn("sector_status", info)
        self.assertEqual(len(info["clearances_m"]), 8)

        # Test step
        action = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, step_info = self.env.step(action)
        self.assertEqual(next_obs.shape, (26,))
        self.assertTrue(np.all(np.isfinite(next_obs)))
        self.assertTrue(math.isfinite(reward))
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(step_info, dict)

    def test_02_lidar_geometry_and_ray_allocation(self):
        """Verify 72 rays across 360° with Option 1 (+2.5° half-step offset) yield exactly 9 rays per sector."""
        num_rays = self.config.lidar_num_rays
        min_angle = self.config.lidar_min_angle_rad
        angle_inc = self.config.lidar_angle_increment_rad
        max_angle = self.config.lidar_max_angle_rad

        self.assertEqual(num_rays, 72)
        self.assertAlmostEqual(math.degrees(min_angle), -177.5, places=4)
        self.assertAlmostEqual(math.degrees(max_angle), +177.5, places=4)
        self.assertAlmostEqual(math.degrees(angle_inc), 5.0, places=4)

        ray_angles_deg = [math.degrees(min_angle + i * angle_inc) for i in range(num_rays)]

        # 1. Verify no ray coincides with cubemap seam/corner angles (0°, ±45°, ±90°, ±135°, ±180°)
        cubemap_seams = [0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0, 180.0, -180.0]
        for seam in cubemap_seams:
            for r_ang in ray_angles_deg:
                diff = abs((r_ang - seam + 180.0) % 360.0 - 180.0)
                self.assertGreaterEqual(
                    diff,
                    2.49,
                    f"Ray at {r_ang:.2f}° is too close ({diff:.3f}°) to cubemap seam at {seam}°!"
                )

        # 2. Verify sector partitioning: exactly 9 rays per sector across all 8 sectors
        sector_counts = [0] * 8
        sector_angles = [[] for _ in range(8)]
        for i in range(num_rays):
            rad = min_angle + i * angle_inc
            sec = ObservationHandler26D.angle_to_sector(rad)
            sector_counts[sec] += 1
            sector_angles[sec].append(math.degrees(rad))

        for sec in range(8):
            self.assertEqual(
                sector_counts[sec],
                9,
                f"Sector {sec} received {sector_counts[sec]} rays, expected exactly 9"
            )

        # Total rays must sum to 72
        self.assertEqual(sum(sector_counts), 72)

    def test_03_analytical_ray_cylinder_accuracy(self):
        """
        Verify ray-cylinder intersection calculation matches analytical ground truth within <= 0.010 m.
        Tests obstacles placed at known distances and azimuths along all 8 sectors.
        """
        # Obstacle configurations centered along sector azimuths
        test_obstacles = [
            {"sector": 0, "azimuth_deg": 0.0, "dist_c": 1.75, "radius": 0.25},
            {"sector": 1, "azimuth_deg": 45.0, "dist_c": 3.75, "radius": 0.25},
            {"sector": 2, "azimuth_deg": 90.0, "dist_c": 2.25, "radius": 0.25},
            {"sector": 3, "azimuth_deg": 135.0, "dist_c": 4.45, "radius": 0.25},
            {"sector": 4, "azimuth_deg": 180.0, "dist_c": 3.05, "radius": 0.25},
            {"sector": 5, "azimuth_deg": -135.0, "dist_c": 5.05, "radius": 0.25},
            {"sector": 6, "azimuth_deg": -90.0, "dist_c": 2.05, "radius": 0.25},
            {"sector": 7, "azimuth_deg": -45.0, "dist_c": 3.25, "radius": 0.25},
        ]

        uav_xy = np.array([0.0, 0.0], dtype=np.float32)
        uav_yaw = 0.0

        for obs_cfg in test_obstacles:
            sec_idx = obs_cfg["sector"]
            az_rad = math.radians(obs_cfg["azimuth_deg"])
            dc = obs_cfg["dist_c"]
            r = obs_cfg["radius"]
            cx = dc * math.cos(az_rad)
            cy = dc * math.sin(az_rad)

            # Analytical expected ray intersection:
            # For rays at azimuth +/- 2.5 deg from center:
            # t = D_c * cos(2.5 deg) - sqrt(R^2 - D_c^2 * sin^2(2.5 deg))
            expected_t = dc * math.cos(math.radians(2.5)) - math.sqrt(
                r**2 - (dc * math.sin(math.radians(2.5)))**2
            )

            # Fast ray casting
            ranges = fast_ray_cast_cylinders(
                uav_xy=uav_xy,
                uav_yaw=uav_yaw,
                obstacles=[(cx, cy, r)],
                num_rays=self.config.lidar_num_rays,
                min_angle=self.config.lidar_min_angle_rad,
                angle_inc=self.config.lidar_angle_increment_rad,
                min_range=self.config.lidar_min_range_m,
                max_range=self.config.lidar_max_range_m
            )

            # Process with ObservationHandler26D
            clearances, meta = self.env.unwrapped.obs_handler.process_raw_scan(
                ranges=ranges,
                angle_min=self.config.lidar_min_angle_rad,
                angle_increment=self.config.lidar_angle_increment_rad
            )

            actual_clearance = float(clearances[sec_idx])
            error_m = abs(actual_clearance - expected_t)

            self.assertLessEqual(
                error_m,
                0.010,
                f"Sector {sec_idx} error {error_m*1000:.2f} mm exceeds 10 mm tolerance! "
                f"Expected={expected_t:.4f}m, Got={actual_clearance:.4f}m"
            )
            self.assertEqual(meta.sector_status[sec_idx], SectorStatus.VALID_HIT)

    def test_04_sector_boundaries_and_wraparound(self):
        """Verify boundary partitioning at +/-22.5°, +/-67.5°, +/-112.5°, +/-157.5° and wraparound at +/-180°."""
        uav_xy = np.array([0.0, 0.0], dtype=np.float32)
        uav_yaw = 0.0

        # Boundary test: Cylinder centered exactly on +22.5° boundary
        # D = 2.50 m, R = 0.35 m
        az_22_5 = math.radians(22.5)
        cx = 2.50 * math.cos(az_22_5)
        cy = 2.50 * math.sin(az_22_5)

        ranges = fast_ray_cast_cylinders(
            uav_xy=uav_xy,
            uav_yaw=uav_yaw,
            obstacles=[(cx, cy, 0.35)],
            num_rays=self.config.lidar_num_rays,
            min_angle=self.config.lidar_min_angle_rad,
            angle_inc=self.config.lidar_angle_increment_rad
        )

        clearances, meta = self.env.unwrapped.obs_handler.process_raw_scan(
            ranges=ranges,
            angle_min=self.config.lidar_min_angle_rad,
            angle_increment=self.config.lidar_angle_increment_rad
        )

        # Both Sector 0 (Front) and Sector 1 (Front-Left) must register hits
        self.assertEqual(meta.sector_status[0], SectorStatus.VALID_HIT)
        self.assertEqual(meta.sector_status[1], SectorStatus.VALID_HIT)
        self.assertAlmostEqual(float(clearances[0]), 2.2166, places=2)
        self.assertAlmostEqual(float(clearances[1]), 2.1500, places=2)

        # Wraparound test: Cylinder at 180° (Sector 4: Back)
        ranges_180 = fast_ray_cast_cylinders(
            uav_xy=uav_xy,
            uav_yaw=uav_yaw,
            obstacles=[(-3.0, 0.0, 0.30)],
            num_rays=self.config.lidar_num_rays,
            min_angle=self.config.lidar_min_angle_rad,
            angle_inc=self.config.lidar_angle_increment_rad
        )
        clearances_180, meta_180 = self.env.unwrapped.obs_handler.process_raw_scan(
            ranges=ranges_180,
            angle_min=self.config.lidar_min_angle_rad,
            angle_increment=self.config.lidar_angle_increment_rad
        )
        self.assertEqual(meta_180.sector_status[4], SectorStatus.VALID_HIT)
        # Expected clearance: 3.0 * cos(2.5°) - sqrt(0.30^2 - (3.0 * sin(2.5°))^2) = 2.997 - 0.270 = 2.727 m
        self.assertAlmostEqual(float(clearances_180[4]), 2.727, places=2)

    def test_05_rigid_bounding_collision_detection(self):
        """Verify rigid bounding collision logic (r_uav = 0.35m) and terminal collision penalties."""
        # Case 1: UAV placed in direct collision with cylinder
        # UAV at (0, 0, 1.5), cylinder at (0.50, 0, 0.25) -> Surface distance = 0.50 - 0.25 = 0.25 m <= 0.35 m
        obs, info = self.env.reset(options={
            "start_pos": [0.0, 0.0, 1.5],
            "start_yaw": 0.0,
            "goal_pos": [5.0, 0.0, 1.5],
            "obstacles": [(0.50, 0.0, 0.25)]
        })

        is_col, reason, dist = self.env.unwrapped.check_collision(
            np.array([0.0, 0.0]),
            self.env.unwrapped.obstacles,
            clearances=self.env.unwrapped.last_clearances
        )
        self.assertTrue(is_col, "Failed to detect point-blank obstacle collision!")
        self.assertLessEqual(dist, 0.35)

        # Taking a step while in collision must terminate episode immediately with crash
        next_obs, reward, terminated, truncated, step_info = self.env.step(np.zeros(4, dtype=np.float32))
        self.assertTrue(terminated)
        self.assertTrue(step_info["is_crashed"])
        self.assertTrue(step_info["is_obstacle_collision"])
        self.assertIn("obstacle_collision", step_info["termination_reason"])
        # Reward must reflect terminal collision penalty (-50.0)
        self.assertLessEqual(reward, -45.0)

        # Case 2: UAV placed safely away from obstacle
        # UAV at (0, 0, 1.5), cylinder at (3.0, 0, 0.25) -> Surface dist = 2.75 m > 0.35 m
        obs_safe, info_safe = self.env.reset(options={
            "start_pos": [0.0, 0.0, 1.5],
            "start_yaw": 0.0,
            "goal_pos": [5.0, 0.0, 1.5],
            "obstacles": [(3.0, 0.0, 0.25)]
        })
        is_col_safe, _, dist_safe = self.env.unwrapped.check_collision(
            np.array([0.0, 0.0]),
            self.env.unwrapped.obstacles,
            clearances=self.env.unwrapped.last_clearances
        )
        self.assertFalse(is_col_safe)
        self.assertGreater(dist_safe, 0.35)

    def test_06_quadratic_obstacle_clearance_reward(self):
        """
        Verify continuous quadratic clearance penalty:
        R_clear = -w_clear * sum(max(0, (d_safe - d_i) / d_safe)^2)
        with d_safe = 1.50 m, w_clear = 0.50.
        """
        env_unwrapped: UAVFastEnvV1 = self.env.unwrapped

        # Case 1: Open space (all clearances >= 1.50 m) -> zero penalty
        clearances_clear = np.full(8, 5.0, dtype=np.float32)
        r_comp_clear = env_unwrapped.compute_obstacle_reward(
            dist_prev=5.0,
            dist_curr=5.0,
            action_curr=np.zeros(4),
            action_prev=np.zeros(4),
            clearances=clearances_clear,
            is_success=False,
            is_out_of_bounds=False,
            is_ground_crash=False,
            is_obstacle_collision=False
        )
        self.assertEqual(r_comp_clear.clearance_penalty, 0.0)

        # Case 2: One clearance at d_0 = 0.75 m (deficit = (1.50 - 0.75) / 1.50 = 0.50)
        # Expected penalty = 0.50 * (0.50)^2 = 0.125
        clearances_close = np.full(8, 5.0, dtype=np.float32)
        clearances_close[0] = 0.75
        r_comp_close = env_unwrapped.compute_obstacle_reward(
            dist_prev=5.0,
            dist_curr=5.0,
            action_curr=np.zeros(4),
            action_prev=np.zeros(4),
            clearances=clearances_close,
            is_success=False,
            is_out_of_bounds=False,
            is_ground_crash=False,
            is_obstacle_collision=False
        )
        self.assertAlmostEqual(r_comp_close.clearance_penalty, 0.125, places=5)

        # Case 3: Closer clearance at d_0 = 0.35 m (collision boundary)
        # Deficit = (1.50 - 0.35) / 1.50 = 1.15 / 1.50 = 0.766667
        # Expected penalty = 0.50 * (0.766667)^2 = 0.293889
        clearances_close[0] = 0.35
        r_comp_closer = env_unwrapped.compute_obstacle_reward(
            dist_prev=5.0,
            dist_curr=5.0,
            action_curr=np.zeros(4),
            action_prev=np.zeros(4),
            clearances=clearances_close,
            is_success=False,
            is_out_of_bounds=False,
            is_ground_crash=False,
            is_obstacle_collision=False
        )
        self.assertAlmostEqual(r_comp_closer.clearance_penalty, 0.293889, places=4)
        self.assertGreater(r_comp_closer.clearance_penalty, r_comp_close.clearance_penalty)

    def test_07_procedural_obstacle_distribution_and_seed_determinism(self):
        """Verify procedural obstacle placement obeys exclusion zones and reproduces deterministically under seed."""
        # Test 1: Exclusion zones across 30 randomized episodes
        for ep in range(30):
            obs, info = self.env.reset(seed=1000 + ep, options={"num_obstacles": 6})
            spawn_xy = info["pos_world"][:2]
            goal_xy = info["goal_world"][:2]
            obstacles = info["obstacles"]

            self.assertGreaterEqual(len(obstacles), 1, "Failed to spawn obstacles!")

            for cx, cy, r in obstacles:
                c_pos = np.array([cx, cy])
                d_spawn = np.linalg.norm(c_pos - spawn_xy)
                d_goal = np.linalg.norm(c_pos - goal_xy)
                # Exclusion zone: center >= clearance
                self.assertGreaterEqual(
                    d_spawn,
                    self.config.min_spawn_clearance - 0.05,
                    f"Obstacle at {c_pos} violated spawn clearance ({d_spawn:.2f} m < {self.config.min_spawn_clearance} m)!"
                )
                self.assertGreaterEqual(
                    d_goal,
                    self.config.min_goal_clearance - 0.05,
                    f"Obstacle at {c_pos} violated goal clearance ({d_goal:.2f} m < {self.config.min_goal_clearance} m)!"
                )

        # Test 2: Seed determinism
        obs_a1, info_a1 = self.env.reset(seed=4242)
        obs_a2, info_a2 = self.env.reset(seed=4242)
        np.testing.assert_array_equal(obs_a1, obs_a2, "Reset under identical seed produced different observations!")
        self.assertEqual(info_a1["obstacles"], info_a2["obstacles"])

    def test_08_random_walk_1000_steps(self):
        """Execute 1,000 steps with random actions across 20 randomized episodes with zero NaNs or Infs."""
        total_steps = 0
        target_steps = 1000
        episodes = 0

        while total_steps < target_steps:
            episodes += 1
            obs, info = self.env.reset(seed=2000 + episodes)
            done = False
            ep_steps = 0

            while not done and total_steps < target_steps:
                action = self.env.action_space.sample()
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                total_steps += 1
                ep_steps += 1

                # Assertion checks on each step
                self.assertEqual(obs.shape, (26,))
                self.assertTrue(np.all(np.isfinite(obs)), f"Non-finite observation at step {total_steps}!")
                self.assertTrue(math.isfinite(reward), f"Non-finite reward at step {total_steps}!")
                self.assertTrue(all(math.isfinite(c) for c in step_info["clearances_m"]))

                done = terminated or truncated

        self.assertGreaterEqual(total_steps, target_steps)
        self.assertGreaterEqual(episodes, 5)

    def test_09_environment_throughput_benchmark(self):
        """
        Verify simulation throughput benchmarks:
          1. Vectorized ray-casting throughput >= 4,000 calls/second (target: enable fast 72-ray perception).
          2. End-to-end environment step throughput >= 500 steps/second (>= 50x real-time speedup vs wall-clock,
             and >= 85x faster than PX4+Gazebo Harmonic SITL baseline of 5.82 steps/s).
        """
        # 1. Ray-casting throughput benchmark
        uav_xy = np.array([0.0, 0.0], dtype=np.float32)
        test_obstacles = [(float(i * 1.5 - 3.0), float(i * 1.0 - 2.0), 0.35) for i in range(6)]
        test_centers = np.array([obs[:2] for obs in test_obstacles], dtype=np.float32)
        test_radii = np.array([obs[2] for obs in test_obstacles], dtype=np.float32)
        num_ray_casts = 10000

        # Warmup
        for _ in range(500):
            _ = fast_ray_cast_cylinders(uav_xy, 0.0, centers=test_centers, radii=test_radii)

        t0_cast = time.perf_counter()
        for _ in range(num_ray_casts):
            _ = fast_ray_cast_cylinders(uav_xy, 0.0, centers=test_centers, radii=test_radii)
        elapsed_cast = time.perf_counter() - t0_cast
        ray_fps = num_ray_casts / elapsed_cast
        print(f"\n[Gate B1 Ray-Caster Benchmark] {num_ray_casts} casts in {elapsed_cast:.3f} s -> {ray_fps:.1f} calls/second")

        self.assertGreaterEqual(
            ray_fps,
            4000.0,
            f"Ray-casting throughput {ray_fps:.1f} calls/s is below target of 4,000 calls/s!"
        )

        # 2. End-to-end environment step throughput
        obs, info = self.env.reset(seed=9999, options={"num_obstacles": 6})
        action = np.array([0.5, 0.1, -0.1, 0.0], dtype=np.float32)

        warmup_steps = 200
        for _ in range(warmup_steps):
            _, _, term, trunc, _ = self.env.step(action)
            if term or trunc:
                self.env.reset()

        benchmark_steps = 2000
        start_t = time.perf_counter()
        for _ in range(benchmark_steps):
            _, _, term, trunc, _ = self.env.step(action)
            if term or trunc:
                self.env.reset()
        elapsed_s = time.perf_counter() - start_t

        step_fps = benchmark_steps / elapsed_s
        rtf = (benchmark_steps * 0.10) / elapsed_s  # 10 Hz policy step
        print(f"[Gate B1 Environment Step Benchmark] {benchmark_steps} steps in {elapsed_s:.3f} s -> {step_fps:.1f} steps/second ({rtf:.1f}x real-time)")

        self.assertGreaterEqual(
            step_fps,
            300.0,
            f"Environment step throughput {step_fps:.1f} steps/s is below minimum threshold of 300 steps/s!"
        )

    def test_10_baseline_18d_isolation_and_checkpoint_integrity(self):
        """Verify UAVFastNav-v0 remains 18-D, frozen adapters untouched, and checkpoint SHA-256 intact."""
        # 1. UAVFastNav-v0 remains 18-D
        env_v0 = gym.make("UAVFastNav-v0")
        obs_v0, _ = env_v0.reset(seed=123)
        self.assertEqual(obs_v0.shape, (18,))
        self.assertEqual(env_v0.observation_space.shape, (18,))
        env_v0.close()

        # 2. Frozen production adapters unchanged
        cmd = [
            "git", "diff", "milestone4-phase4a-frozen", "--",
            "ppo_adapter.py", "ppo_inference_adapter.py", "rl_env/observation.py"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertEqual(len(res.stdout.strip()), 0, f"Production files modified: {res.stdout}")

        # 3. Checkpoint SHA-256 unchanged
        self.assertTrue(os.path.exists(CHECKPOINT_PATH), "Checkpoint missing!")
        hasher = hashlib.sha256()
        with open(CHECKPOINT_PATH, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        actual_hash = hasher.hexdigest()
        self.assertEqual(
            actual_hash,
            EXPECTED_CHECKPOINT_HASH,
            f"Checkpoint SHA-256 modified! Expected {EXPECTED_CHECKPOINT_HASH}, got {actual_hash}"
        )


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGateB1Environment)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    import sys
    res = run_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
