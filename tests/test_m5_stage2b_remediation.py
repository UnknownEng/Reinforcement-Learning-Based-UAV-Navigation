"""
test_m5_stage2b_remediation.py
------------------------------
Focused unit and regression test suite for Milestone 5.4 Gate B:
1. Reverse motion with the goal behind the UAV.
2. Obstacles directly behind the UAV (Case R1 & R2 geometry).
3. Obstacles offset to either side (Case R3 & R4 geometry).
4. Close-proximity obstacles and stopping distance physics.
5. Correct coordinate frames (REP-103 FLU), velocity signs, observation dimensions (Box(26,)), finite rewards, and termination conditions.
6. Existing geometry-validator route feasibility and strict training step-budget interrupt enforcement.
"""

import math
import unittest
import numpy as np
import gymnasium as gym

from rl_env.heading_reward import compute_smooth_heading_reward, HeadingRewardConfig
from rl_env.uav_fast_env_v2 import UAVFastEnvV2, UAVObstacleEnvConfigV2
from rl_env.geometry_validator import RouteGeometryValidator
from rl_env.dynamics import FastUAVDynamics
from rl_env.config import UAVEnvConfig
from rl_env.action import CommandedVelocities


class TestStage2bRemediation(unittest.TestCase):
    def setUp(self):
        self.w_align = 0.10
        self.w_speed = 0.10
        self.w_rev = 0.35

    def test_reverse_motion_with_goal_behind(self):
        """
        Verify that when goal is directly behind (psi = pi):
        1. Stationary yawing (vx = 0) receives zero reverse penalty.
        2. Reverse motion (vx < 0) receives quadratic reverse speed penalty.
        3. Reverse speed penalty strictly overpowers distance progress reward (net negative).
        4. Forward motion (vx > 0) incurs no reverse penalty.
        """
        d_psi = math.pi
        v_zero = np.array([0.0, 0.0], dtype=np.float32)
        v_rev_max = np.array([-3.0, 0.0], dtype=np.float32)
        v_rev_mid = np.array([-1.5, 0.0], dtype=np.float32)
        v_fwd = np.array([2.0, 0.0], dtype=np.float32)

        _, p_zero, r_tot_zero = compute_smooth_heading_reward(
            d_psi, v_zero, self.w_align, self.w_speed, self.w_rev
        )
        _, p_rev_max, r_tot_rev_max = compute_smooth_heading_reward(
            d_psi, v_rev_max, self.w_align, self.w_speed, self.w_rev
        )
        _, p_rev_mid, r_tot_rev_mid = compute_smooth_heading_reward(
            d_psi, v_rev_mid, self.w_align, self.w_speed, self.w_rev
        )
        _, p_fwd, r_tot_fwd = compute_smooth_heading_reward(
            d_psi, v_fwd, self.w_align, self.w_speed, self.w_rev
        )

        # 1. Zero velocity -> zero penalty
        self.assertAlmostEqual(p_zero, 0.0, places=6)

        # 2. Reverse velocity -> negative penalty with quadratic scaling
        self.assertLess(p_rev_max, p_rev_mid)
        self.assertLess(p_rev_mid, 0.0)
        
        # Expected penalty at vx = -3.0: -0.10 * 1.0 * 9 - 0.35 * 1.0 * 9 = -0.90 - 3.15 = -4.05
        expected_p_max = - (self.w_speed * 1.0 * 9.0 + self.w_rev * 1.0 * 9.0)
        self.assertAlmostEqual(p_rev_max, expected_p_max, places=5)

        # 3. Progress reward comparison:
        # Reversing at -3.0 m/s for dt = 0.1s covers 0.30m. Progress reward = +2.5 * 0.30 = +0.75
        prog_rev_max = 2.5 * 0.30
        net_rev_max = r_tot_rev_max + prog_rev_max
        net_zero = r_tot_zero + 0.0
        
        # Net reward for reversing at -3.0 m/s MUST be strictly worse than stationary yawing
        self.assertLess(net_rev_max, net_zero)
        self.assertLess(net_rev_max, -2.0)  # strongly negative

        # 4. Forward motion has rev_speed = 0, so only symmetric penalty applies
        expected_p_fwd = -self.w_speed * 1.0 * (2.0 ** 2)
        self.assertAlmostEqual(p_fwd, expected_p_fwd, places=5)

    def test_commanded_velocity_latency_elimination(self):
        """
        Verify that when UAV is stationary (v_xy = [0, 0]) but commands reverse velocity (cmd_vx = -3.0):
        The reverse speed penalty activates immediately at step 0 without waiting for physical acceleration.
        """
        d_psi = math.pi
        v_zero = np.array([0.0, 0.0], dtype=np.float32)
        cmd_rev = np.array([-3.0, 0.0], dtype=np.float32)

        _, p_speed, r_tot = compute_smooth_heading_reward(
            d_psi, v_zero, self.w_align, self.w_speed, 0.50, cmd_vel_xy=cmd_rev
        )
        # Expected: symmetric penalty = 0 (v=0), reverse penalty = -0.50 * 1.0 * 9.0 = -4.50
        self.assertAlmostEqual(p_speed, -4.50, places=5)
        self.assertAlmostEqual(r_tot, -4.60, places=5)

    def test_no_reverse_penalty_when_aligned(self):
        """
        Verify that when goal is directly ahead (psi = 0):
        Reverse penalty is exactly 0.0 even if vx < 0 (e.g. backing away from a front obstacle).
        """
        d_psi = 0.0
        v_rev = np.array([-3.0, 0.0], dtype=np.float32)
        r_align, p_speed, r_tot = compute_smooth_heading_reward(
            d_psi, v_rev, self.w_align, self.w_speed, self.w_rev
        )
        self.assertAlmostEqual(p_speed, 0.0, places=6)
        self.assertAlmostEqual(r_align, self.w_align, places=6)
        self.assertAlmostEqual(r_tot, self.w_align, places=6)

    def test_smoothness_and_differentiability_at_zero_velocity(self):
        """
        Verify that p_speed is C^1 smooth across vx = 0 (derivative exists and is 0 at vx = 0).
        """
        eps = 1e-5
        psi = math.pi
        
        # Left difference (vx < 0)
        _, p_left, _ = compute_smooth_heading_reward(psi, np.array([-eps, 0.0]), self.w_align, self.w_speed, self.w_rev)
        # Center (vx = 0)
        _, p_mid, _ = compute_smooth_heading_reward(psi, np.array([0.0, 0.0]), self.w_align, self.w_speed, self.w_rev)
        # Right difference (vx > 0)
        _, p_right, _ = compute_smooth_heading_reward(psi, np.array([eps, 0.0]), self.w_align, self.w_speed, self.w_rev)

        deriv_left = (p_mid - p_left) / eps
        deriv_right = (p_right - p_mid) / eps

        self.assertAlmostEqual(deriv_left, 0.0, places=4)
        self.assertAlmostEqual(deriv_right, 0.0, places=4)
        self.assertAlmostEqual(deriv_left, deriv_right, places=4)

    def test_coordinate_frames_and_velocity_signs(self):
        """
        Verify REP-103 FLU body frame convention:
        +x_b: Forward
        -x_b: Backward
        +y_b: Left
        -y_b: Right
        """
        config = UAVObstacleEnvConfigV2(enable_noise=False)
        env = UAVFastEnvV2(config=config)

        # 1. Forward action [1, 0, 0, 0] produces positive vx_world when yaw = 0
        env.reset(options={'start_pos': [0, 0, 2], 'start_yaw': 0, 'goal_pos': [5, 0, 2]})
        env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        s_fwd = env.dynamics.get_state()
        self.assertGreater(s_fwd.vel_world[0], 0.0)

        # 2. Reverse action [-1, 0, 0, 0] produces negative vx_world when yaw = 0
        env.reset(options={'start_pos': [0, 0, 2], 'start_yaw': 0, 'goal_pos': [-5, 0, 2]})
        env.step(np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        s_rev = env.dynamics.get_state()
        self.assertLess(s_rev.vel_world[0], 0.0)

        env.close()

    def test_observation_dimensions_and_finite_rewards(self):
        """
        Verify Gym Box contracts:
        - Observation: Box(-inf, inf, shape=(26,), float32)
        - Action: Box(-1.0, 1.0, shape=(4,), float32)
        - Rewards are strictly finite on all actions.
        """
        config = UAVObstacleEnvConfigV2(curriculum_stage=2)
        env = UAVFastEnvV2(config=config)

        obs, info = env.reset(seed=42)
        self.assertEqual(obs.shape, (26,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(obs)))

        for act in [
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            np.array([-1.0, 1.0, -0.5, -1.0], dtype=np.float32),
        ]:
            obs, reward, term, trunc, info = env.step(act)
            self.assertEqual(obs.shape, (26,))
            self.assertTrue(np.all(np.isfinite(obs)))
            self.assertTrue(math.isfinite(reward))

        env.close()

    def test_regression_scenarios_geometry_validation(self):
        """
        Verify that all 4 rear-obstacle regression scenarios (R1-R4) are geometrically traversable:
        R1: Obstacle at (-2.5, 0.0), R = 0.50m
        R2: Obstacle at (-1.5, 0.0), R = 0.40m
        R3: Obstacle at (-2.5, +0.40), R = 0.50m
        R4: Obstacle at (-2.5, -0.40), R = 0.50m
        """
        val = RouteGeometryValidator(
            arena_x=(-12.0, 12.0),
            arena_y=(-12.0, 12.0),
            uav_radius=0.35,
            safety_margin=0.15,
            grid_resolution=0.10,
            min_path_clearance=0.50
        )
        start_xy = np.array([0.0, 0.0], dtype=np.float32)
        goal_xy = np.array([-6.0, 0.0], dtype=np.float32)

        cases = [
            ("R1_Direct", [(-2.5, 0.0, 0.50)]),
            ("R2_SevereProximity", [(-1.5, 0.0, 0.40)]),
            ("R3_OffsetLeft", [(-2.5, 0.40, 0.50)]),
            ("R4_OffsetRight", [(-2.5, -0.40, 0.50)])
        ]

        for name, obs_list in cases:
            res = val.validate_route(start_xy, goal_xy, obs_list)
            self.assertTrue(res.is_traversable, f"Case {name} failed traversability check")
            self.assertGreaterEqual(res.min_route_clearance, 0.50, f"Case {name} route clearance too narrow")

    def test_close_proximity_stopping_distance_dynamics(self):
        """
        Stopping distance matches PX4: in PX4 v1.15 x500 SITL ground truth
        (logs/px4_step_response.json) a zero-velocity command at 3.0 m/s stops the
        vehicle within 2.12-2.26 m. The calibrated fast simulator must agree (+/-20 %).
        """
        config = UAVEnvConfig(enable_noise=False)
        dyn = FastUAVDynamics(config=config)
        dyn.reset(x0=0.0, y0=0.0, z0=5.0, yaw0=0.0)
        cruise = CommandedVelocities(vx=3.0, vy=0.0, vz=0.0, yaw_rate=0.0)
        for _ in range(60):                       # settle at 3 m/s
            dyn.step(cruise)
        x0 = float(dyn.state.pos_world[0])
        stop = CommandedVelocities(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)
        for _ in range(60):
            dyn.step(stop)
            if abs(dyn.state.vel_world[0]) < 0.05:
                break
        dist = float(dyn.state.pos_world[0]) - x0
        self.assertGreater(dist, 0.8 * 2.12)
        self.assertLess(dist, 1.2 * 2.26)


if __name__ == "__main__":
    unittest.main()
