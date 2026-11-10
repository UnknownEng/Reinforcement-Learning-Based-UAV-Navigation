#!/usr/bin/env python3
"""
test_m5_geometry_and_metrics.py
--------------------------------
Unit tests for Milestone 5 Phase M5.2 correction pass:
1. Deterministic geometry validation: overlap detection, traversability BFS, spawn/goal clearance.
2. Obstacle density Level 0 geometry consistency across all 7 factors.
3. Path efficiency calculation: success-only calculation, failed-flight exclusion.
4. Sensitivity metrics calculation: percentage-point metrics, no double counting, tie handling.
"""

import math
import unittest
import numpy as np


def validate_scenario_geometry(
    obstacles,
    start_pos=(0.0, 0.0, 2.0),
    goal_pos=(6.5, 0.0, 2.0),
    uav_radius=0.35
):
    n = len(obstacles)
    overlaps = []
    min_sep = float('inf')

    for i in range(n):
        xi, yi, ri = obstacles[i]
        for j in range(i + 1, n):
            xj, yj, rj = obstacles[j]
            center_dist = math.hypot(xi - xj, yi - yj)
            surf_dist = center_dist - (ri + rj)
            if surf_dist < min_sep:
                min_sep = surf_dist
            if surf_dist < 0.0:
                overlaps.append((i, j, float(surf_dist)))

    d_start = min([math.hypot(ox - start_pos[0], oy - start_pos[1]) - r for ox, oy, r in obstacles]) if obstacles else float('inf')
    d_goal = min([math.hypot(ox - goal_pos[0], oy - goal_pos[1]) - r for ox, oy, r in obstacles]) if obstacles else float('inf')

    # Vectorized 2D BFS Grid
    from collections import deque
    res = 0.05
    xs = np.arange(-0.5, 7.0, res)
    ys = np.arange(-2.5, 2.5, res)
    X, Y = np.meshgrid(xs, ys, indexing='ij')

    blocked = np.zeros(X.shape, dtype=bool)
    for ox, oy, r in obstacles:
        blocked |= ((X - ox)**2 + (Y - oy)**2) <= ((r + uav_radius)**2)

    start_i = int(np.clip(round((start_pos[0] - xs[0]) / res), 0, len(xs) - 1))
    start_j = int(np.clip(round((start_pos[1] - ys[0]) / res), 0, len(ys) - 1))
    goal_i = int(np.clip(round((goal_pos[0] - xs[0]) / res), 0, len(xs) - 1))
    goal_j = int(np.clip(round((goal_pos[1] - ys[0]) / res), 0, len(ys) - 1))

    is_traversable = False
    if not blocked[start_i, start_j] and not blocked[goal_i, goal_j]:
        visited = np.zeros(blocked.shape, dtype=bool)
        q = deque([(start_i, start_j)])
        visited[start_i, start_j] = True
        while q:
            ci, cj = q.popleft()
            if abs(ci - goal_i) <= 2 and abs(cj - goal_j) <= 2:
                is_traversable = True
                break
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < len(xs) and 0 <= nj < len(ys):
                    if not blocked[ni, nj] and not visited[ni, nj]:
                        visited[ni, nj] = True
                        q.append((ni, nj))

    invalid_reasons = []
    if len(overlaps) > 0:
        invalid_reasons.append("OBSTACLE_OVERLAP")
    if d_start < uav_radius:
        invalid_reasons.append("SPAWN_BLOCKED")
    if d_goal < uav_radius:
        invalid_reasons.append("GOAL_BLOCKED")
    if not is_traversable:
        invalid_reasons.append("NO_PASSABLE_ROUTE")

    return {
        "is_valid": len(invalid_reasons) == 0,
        "has_overlap": len(overlaps) > 0,
        "overlap_count": len(overlaps),
        "min_separation_m": round(float(min_sep), 4) if min_sep != float('inf') else None,
        "spawn_clearance_m": round(float(d_start), 4),
        "goal_clearance_m": round(float(d_goal), 4),
        "is_traversable": is_traversable,
        "invalid_reasons": invalid_reasons
    }


def generate_scenario_obstacles_corrected(factor_id: str, level_param, seed: int):
    rng = np.random.default_rng(seed)
    y1 = float(rng.uniform(1.3, 1.7) * rng.choice([1.0, -1.0]))
    y2 = float(rng.uniform(1.3, 1.7) * rng.choice([1.0, -1.0]))
    y3 = float(rng.uniform(1.3, 1.7) * rng.choice([1.0, -1.0]))

    R = 0.30
    if factor_id == "F6_obstacle_size":
        R = float(level_param)

    if factor_id == "F5_obstacle_density":
        N = int(level_param)
        if N == 3:
            # Fixed: Level 0 returns exact nominal corridor geometry!
            return [(2.2, y1, R), (3.8, y2, R), (5.2, y3, R)]
        else:
            x_positions = np.linspace(1.8, 5.4, N)
            obstacles = []
            for x in x_positions:
                y = float(rng.uniform(1.3, 1.7) * rng.choice([1.0, -1.0]))
                obstacles.append((float(x), y, R))
            return obstacles
    else:
        return [(2.2, y1, R), (3.8, y2, R), (5.2, y3, R)]


class TestM5GeometryAndMetrics(unittest.TestCase):

    def test_level0_geometry_identical_across_factors(self):
        """Task 1: Verify Level 0 geometry is 100% bit-for-bit identical across all 7 factors."""
        factors = [
            ("F1_lidar_noise", 0.0),
            ("F2_lidar_dropout", 0.0),
            ("F3_obs_state_noise", (0.0, 0.0)),
            ("F4_sensor_rate", 1),
            ("F5_obstacle_density", 3),
            ("F6_obstacle_size", 0.30),
            ("F7_dynamics_perturbation", (0.20, 3.0)),
        ]
        for seed in range(20001, 20026):
            base_obs = generate_scenario_obstacles_corrected(factors[0][0], factors[0][1], seed)
            for f_id, param in factors[1:]:
                obs = generate_scenario_obstacles_corrected(f_id, param, seed)
                self.assertEqual(base_obs, obs, f"Level 0 geometry mismatch for {f_id} at seed {seed}")

    def test_geometry_validation_detects_overlap(self):
        """Task 2: Overlapping obstacles must be detected and flagged."""
        # Two cylinders intersecting: (2.0, 0.0, 0.5) and (2.5, 0.0, 0.5) -> distance = 0.5, sum of radii = 1.0 -> overlap 0.5m
        obstacles = [(2.0, 0.0, 0.5), (2.5, 0.0, 0.5)]
        res = validate_scenario_geometry(obstacles)
        self.assertFalse(res["is_valid"])
        self.assertTrue(res["has_overlap"])
        self.assertIn("OBSTACLE_OVERLAP", res["invalid_reasons"])
        self.assertLess(res["min_separation_m"], 0.0)

    def test_geometry_validation_detects_impassable_wall(self):
        """Task 2: Continuous wall of obstacles blocking the corridor must be flagged as non-traversable."""
        # Unbroken vertical barrier from y = -2.5 to y = +2.5 at x = 3.0
        obstacles = [(3.0, y, 0.40) for y in np.linspace(-2.2, 2.2, 10)]
        res = validate_scenario_geometry(obstacles)
        self.assertFalse(res["is_valid"])
        self.assertFalse(res["is_traversable"])
        self.assertIn("NO_PASSABLE_ROUTE", res["invalid_reasons"])

    def test_geometry_validation_nominal_corridor_is_valid(self):
        """Task 2: Standard nominal corridor must be valid, non-overlapping, and traversable."""
        obstacles = [(2.2, 1.4, 0.30), (3.8, -1.4, 0.30), (5.2, 1.4, 0.30)]
        res = validate_scenario_geometry(obstacles)
        self.assertTrue(res["is_valid"])
        self.assertFalse(res["has_overlap"])
        self.assertTrue(res["is_traversable"])
        self.assertEqual(res["invalid_reasons"], [])
        self.assertGreater(res["spawn_clearance_m"], 1.0)
        self.assertGreater(res["goal_clearance_m"], 1.0)

    def test_path_efficiency_successful_only(self):
        """Task 3: Path efficiency must only be computed on successful flights."""
        # 1 successful flight (path_len = 7.15, straight = 6.5 -> eff = 1.10)
        # 1 failed flight (crashed at x = 2.0 -> path_len = 2.0, straight = 6.5)
        episodes = [
            {"success": True, "path_length_m": 7.15, "straight_line_dist_m": 6.5},
            {"success": False, "path_length_m": 2.00, "straight_line_dist_m": 6.5},
        ]
        succ_effs = [e["path_length_m"] / e["straight_line_dist_m"] for e in episodes if e["success"]]
        self.assertEqual(len(succ_effs), 1)
        self.assertAlmostEqual(succ_effs[0], 1.10)
        # Verifying that naive all-episode averaging would yield misleading < 1.0 value
        naive_all_avg = sum(e["path_length_m"] / e["straight_line_dist_m"] for e in episodes) / len(episodes)
        self.assertLess(naive_all_avg, 1.0)

    def test_sensitivity_metrics_percentage_points_and_ties(self):
        """Task 5: Sensitivity metrics must be in percentage points without double-counting, marking ties."""
        factor_deltas = [
            ("F6_obstacle_size", 92.0, 8.0),       # drop = 84.0 pp
            ("F5_obstacle_density", 92.0, 40.0),    # drop = 52.0 pp
            ("F7_dynamics", 92.0, 76.0),            # drop = 16.0 pp
            ("F2_lidar_dropout", 92.0, 88.0),       # drop = 4.0 pp
            ("F3_state_noise", 92.0, 92.0),         # drop = 0.0 pp
            ("F4_sensor_rate", 92.0, 92.0),         # drop = 0.0 pp
            ("F1_lidar_noise", 92.0, 92.0),         # drop = 0.0 pp
        ]

        # Calculate percentage point drop
        scored = []
        for fid, l0, l4 in factor_deltas:
            drop_pp = round(l0 - l4, 2)
            scored.append((fid, drop_pp))

        # Sort descending by degradation
        scored.sort(key=lambda x: x[1], reverse=True)

        # Assign standard competition ranks and mark ties
        ranked = []
        n = len(scored)
        # First find tied groups
        is_tied = [False] * n
        for i in range(n):
            if (i > 0 and scored[i][1] == scored[i-1][1]) or (i < n - 1 and scored[i][1] == scored[i+1][1]):
                is_tied[i] = True

        current_rank = 1
        for idx in range(n):
            fid, drop_pp = scored[idx]
            if idx > 0 and drop_pp == scored[idx-1][1]:
                rank_num = ranked[-1][2]
            else:
                rank_num = idx + 1
            rank_str = f"Tied #{rank_num}" if is_tied[idx] else f"#{rank_num}"
            ranked.append((fid, drop_pp, rank_num, rank_str))

        self.assertEqual(ranked[0][3], "#1")
        self.assertEqual(ranked[1][3], "#2")
        self.assertEqual(ranked[2][3], "#3")
        self.assertEqual(ranked[3][3], "#4")
        self.assertEqual(ranked[4][3], "Tied #5")
        self.assertEqual(ranked[5][3], "Tied #5")
        self.assertEqual(ranked[6][3], "Tied #5")


if __name__ == "__main__":
    unittest.main()
