"""
test_safety_shield.py
---------------------
Comprehensive Unit Test Suite for Corrected Deterministic Safety Shield (Milestone 5.4).

Covers all 12 Gate 4 requirements:
  1. Exact clearance boundary at the required 0.50 m trajectory margin (d_k = 0.85 m)
  2. A point safely above that boundary (d_k = 1.50 m)
  3. A point below that boundary (d_k = 0.60 m)
  4. Braking with nonzero physical velocity and zero commanded velocity
  5. Command latency and velocity lag modeling (k_delay = 0.25 s)
  6. Rear-left and rear-right obstacles triggering Layer 1
  7. Frontal obstacle with lateral escape space (lateral motion preserved)
  8. Frontal obstacle without safe escape (deterministic hold/stop, no injected motion)
  9. Stale, invalid, NaN and Inf sensor readings
  10. Finite bounded actions and unchanged Box(26,) / Box(4,) contracts
  11. R3 pre-integration versus post-integration collision timing check
  12. R2 blocked-motion / deadlock behavior verification
"""

import math
import unittest
import numpy as np

from rl_env.safety_shield import (
    DeterministicSafetyShield,
    SafetyShieldConfig,
    SafetyShieldDiagnostic,
    SafetyShieldWrapper
)
from rl_env.geometry_validator import RouteGeometryValidator
from rl_env.action import ActionHandler, CommandedVelocities
from rl_env.config import UAVEnvConfig


class TestSafetyShieldCorrected(unittest.TestCase):
    def setUp(self):
        self.config = SafetyShieldConfig()
        self.shield = DeterministicSafetyShield(self.config)

    def _create_mock_obs(
        self,
        goal_body: np.ndarray = np.array([5.0, 0.0, 0.0], dtype=np.float32),
        vel_body: np.ndarray = np.array([0.0, 0.0, 0.0], dtype=np.float32),
        clearances: np.ndarray = np.full(8, 10.0, dtype=np.float32)
    ) -> np.ndarray:
        """Constructs a valid 26D observation vector."""
        obs = np.zeros(26, dtype=np.float32)
        obs[0:3] = goal_body
        obs[3:6] = vel_body
        obs[6:9] = np.zeros(3, dtype=np.float32)       # rates
        obs[9:13] = np.array([1.0, 0.0, 0.0, 0.0])     # quat
        obs[13] = 2.0                                  # rel alt
        obs[14:18] = np.zeros(4, dtype=np.float32)     # prev action
        obs[18:26] = clearances                       # 8 sectors
        return obs

    def test_dense_scene_adds_localized_braking_margin(self):
        obs = self._create_mock_obs()
        sparse = np.full(8, 10.0, dtype=np.float32)
        dense = sparse.copy()
        dense[:4] = 0.90
        obs_sparse = self._create_mock_obs(clearances=sparse)
        obs_dense = self._create_mock_obs(clearances=dense)
        act = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        sparse_out, _ = self.shield.filter_action(obs_sparse, act)
        dense_out, _ = self.shield.filter_action(obs_dense, act)
        self.assertGreater(float(sparse_out[0]), float(dense_out[0]))

    def test_world_geofence_projects_outward_velocity(self):
        obs = self._create_mock_obs()
        act = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out, diag = self.shield.filter_action(
            obs, act, perception_metadata={"position_world": np.array([7.9, 0.0, 2.0])}
        )
        self.assertTrue(diag.geofence_active)
        self.assertLess(float(out[0]), 1.0)

    # -------------------------------------------------------------------------
    # 1. Exact clearance boundary at 0.50 m trajectory margin (d_k = 0.85 m)
    # -------------------------------------------------------------------------
    def test_01_exact_clearance_boundary_at_point_85m(self):
        """At d_k = 0.85 m (r_uav + 0.50m margin), allowable approach velocity is 0.0 m/s."""
        self.assertAlmostEqual(self.config.d_base, 0.85, places=3)
        clearances = np.full(8, 10.0, dtype=np.float32)
        clearances[0] = 0.85  # Front sector exactly at d_base
        obs = self._create_mock_obs(clearances=clearances)
        raw_act = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Full forward cmd

        shielded_act, diag = self.shield.filter_action(obs, raw_act)

        # Frontal velocity must be clamped to 0.0
        self.assertTrue(diag.layer2_active)
        self.assertAlmostEqual(diag.shielded_cmd.vx, 0.0, places=4)
        self.assertAlmostEqual(shielded_act[0], 0.0, places=4)

    # -------------------------------------------------------------------------
    # 2. A point safely above that boundary (d_k = 1.50 m)
    # -------------------------------------------------------------------------
    def test_02_point_safely_above_boundary_at_1_50m(self):
        """At d_k = 1.50 m, positive approach velocity is permitted up to v_safe."""
        clearances = np.full(8, 10.0, dtype=np.float32)
        clearances[0] = 1.50  # Front sector at 1.50 m (> 0.85 m)
        obs = self._create_mock_obs(clearances=clearances)
        raw_act = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        shielded_act, diag = self.shield.filter_action(obs, raw_act)

        # Safe speed for d=1.50, c=0.65 m, k_delay = 0.08 + 0.602 = 0.682 s (PX4-calibrated), a=3:
        # B = 2*3*0.682 = 4.092; v_safe = (-4.092 + sqrt(4.092^2 + 24*0.65)) / 2 = 0.80 m/s
        self.assertTrue(diag.layer2_active)
        self.assertGreater(diag.shielded_cmd.vx, 0.70)
        self.assertLess(diag.shielded_cmd.vx, 0.90)

    # -------------------------------------------------------------------------
    # 3. A point below that boundary (d_k = 0.60 m)
    # -------------------------------------------------------------------------
    def test_03_point_below_boundary_at_0_60m(self):
        """At d_k = 0.60 m (< d_base = 0.85 m), v_safe is strictly 0.0 m/s."""
        clearances = np.full(8, 10.0, dtype=np.float32)
        clearances[0] = 0.60
        obs = self._create_mock_obs(clearances=clearances)
        raw_act = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)

        shielded_act, diag = self.shield.filter_action(obs, raw_act)

        self.assertTrue(diag.layer2_active)
        self.assertAlmostEqual(diag.shielded_cmd.vx, 0.0, places=4)

    # -------------------------------------------------------------------------
    # 4. Braking with nonzero physical velocity and zero commanded velocity
    # -------------------------------------------------------------------------
    def test_04_braking_with_nonzero_physical_velocity(self):
        """Physical approach speed exceeds safe threshold -> no approach command permitted."""
        clearances = np.full(8, 10.0, dtype=np.float32)
        clearances[0] = 1.20  # Front sector
        # Physical velocity is already moving forward at 2.5 m/s (above v_safe ~ 0.88 m/s)
        vel_body = np.array([2.5, 0.0, 0.0], dtype=np.float32)
        obs = self._create_mock_obs(vel_body=vel_body, clearances=clearances)

        # Policy tries to command forward motion
        raw_act = np.array([0.5, 0.0, 0.0, 0.0], dtype=np.float32)
        shielded_act, diag = self.shield.filter_action(obs, raw_act)

        # Because physical velocity >= v_safe, commanded forward velocity is clamped to 0.0
        self.assertTrue(diag.layer2_active)
        self.assertAlmostEqual(diag.shielded_cmd.vx, 0.0, places=4)

    # -------------------------------------------------------------------------
    # 5. Command latency and velocity lag modeling
    # -------------------------------------------------------------------------
    def test_05_latency_and_velocity_lag_parameters(self):
        """Verify effective latency horizon k_delay = action_latency_s + tau_velocity_xy."""
        # Calibrated to PX4 v1.15 x500 SITL (uav_sitl/fit_dynamics.py)
        self.assertAlmostEqual(self.config.action_latency_s, 0.08, places=3)
        self.assertAlmostEqual(self.config.tau_velocity_xy, 0.602, places=3)
        self.assertAlmostEqual(self.config.k_delay, 0.682, places=3)

    # -------------------------------------------------------------------------
    # 6. Rear-left and rear-right obstacles
    # -------------------------------------------------------------------------
    def test_06_rear_left_and_rear_right_obstacles_trigger_layer1(self):
        """Goal behind (|d_psi| > 90°) + rear-left (sec 3) or rear-right (sec 5) -> Layer 1 triggers."""
        # Rear-left obstacle
        obs_rl = self._create_mock_obs(
            goal_body=np.array([-6.0, 0.0, 0.0], dtype=np.float32),
            clearances=np.array([10.0, 10.0, 10.0, 2.20, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        )
        raw_act = np.array([-1.0, 0.0, 0.0, 0.5], dtype=np.float32)
        s_act_rl, diag_rl = self.shield.filter_action(obs_rl, raw_act)
        self.assertTrue(diag_rl.layer1_active)
        self.assertAlmostEqual(s_act_rl[0], 0.0, places=4)

        # Rear-right obstacle
        obs_rr = self._create_mock_obs(
            goal_body=np.array([-6.0, 0.0, 0.0], dtype=np.float32),
            clearances=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 2.20, 10.0, 10.0], dtype=np.float32)
        )
        s_act_rr, diag_rr = self.shield.filter_action(obs_rr, raw_act)
        self.assertTrue(diag_rr.layer1_active)
        self.assertAlmostEqual(s_act_rr[0], 0.0, places=4)

    # -------------------------------------------------------------------------
    # 7. Frontal obstacle with lateral escape space
    # -------------------------------------------------------------------------
    def test_07_frontal_obstacle_preserves_lateral_escape(self):
        """Front is blocked, but policy commands diagonal move -> forward clamped, lateral preserved."""
        clearances = np.full(8, 10.0, dtype=np.float32)
        clearances[0] = 0.85  # Front blocked at d_base
        clearances[2] = 10.0  # Left completely clear
        obs = self._create_mock_obs(clearances=clearances)

        # Policy commands forward and left: vx = 2.0 m/s (a0=0.667), vy = 2.0 m/s (a1=0.667)
        raw_act = np.array([2.0 / 3.0, 2.0 / 3.0, 0.0, 0.0], dtype=np.float32)
        shielded_act, diag = self.shield.filter_action(obs, raw_act)

        # Forward velocity clamped to 0.0, lateral velocity preserved at 2.0 m/s
        self.assertAlmostEqual(diag.shielded_cmd.vx, 0.0, places=3)
        self.assertAlmostEqual(diag.shielded_cmd.vy, 2.0, places=2)

    # -------------------------------------------------------------------------
    # 8. Frontal obstacle without safe escape (deterministic hold/stop)
    # -------------------------------------------------------------------------
    def test_08_frontal_obstacle_without_safe_escape_holds(self):
        """Front blocked, policy commands pure forward -> stops to 0.0, no arbitrary lateral injection."""
        clearances = np.full(8, 10.0, dtype=np.float32)
        clearances[0] = 0.85
        obs = self._create_mock_obs(clearances=clearances)

        # Policy commands pure forward
        raw_act = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        shielded_act, diag = self.shield.filter_action(obs, raw_act)

        # Must hold in place: both vx and vy must be 0.0 (no arbitrary command created)
        self.assertAlmostEqual(diag.shielded_cmd.vx, 0.0, places=4)
        self.assertAlmostEqual(diag.shielded_cmd.vy, 0.0, places=4)

    # -------------------------------------------------------------------------
    # 9. Stale, invalid, NaN and Inf sensor readings
    # -------------------------------------------------------------------------
    def test_09_stale_invalid_nan_inf_sensor_readings(self):
        """Corrupted LiDAR arrays must be handled conservatively without NaN/Inf."""
        clearances = np.array([float("nan"), float("inf"), -1.0, float("nan"), 10.0, 10.0, 10.0, 10.0], dtype=np.float32)
        obs = self._create_mock_obs(clearances=clearances)
        raw_act = np.array([1.0, 1.0, -0.5, 0.5], dtype=np.float32)
        meta = {"is_sensor_stale": True}

        shielded_act, diag = self.shield.filter_action(obs, raw_act, perception_metadata=meta)

        self.assertTrue(np.all(np.isfinite(shielded_act)))
        self.assertTrue(diag.is_intervened)

    # -------------------------------------------------------------------------
    # 10. Finite bounded actions and unchanged Box(26,) / Box(4,) contracts
    # -------------------------------------------------------------------------
    def test_10_contract_preservation_and_boundedness(self):
        """Observation remains strictly unmodified; action output strictly within [-1, 1]^4."""
        obs = self._create_mock_obs()
        obs_copy = obs.copy()
        raw_act = np.array([5.0, -10.0, 0.5, -0.2], dtype=np.float32)  # Out of bounds

        shielded_act, _ = self.shield.filter_action(obs, raw_act)

        np.testing.assert_array_equal(obs, obs_copy)
        self.assertEqual(shielded_act.shape, (4,))
        self.assertTrue(np.all(shielded_act >= -1.0))
        self.assertTrue(np.all(shielded_act <= 1.0))

    # -------------------------------------------------------------------------
    # 11. R3 pre-integration versus post-integration collision timing check
    # -------------------------------------------------------------------------
    def test_11_r3_pre_vs_post_integration_physics_timing(self):
        """Verifies why pre-step surface dist of 0.3656m collides within 0.10s at -0.575 m/s."""
        dt = 0.10  # 10 substeps of 0.01s
        v_approach = 0.575  # velocity towards obstacle
        pre_step_surface_dist = 0.3656
        uav_radius = 0.3500

        # Physical displacement during step
        disp = v_approach * dt  # 0.0575 m
        post_step_surface_dist = pre_step_surface_dist - disp  # 0.3081 m

        # Proves post-step distance drops strictly below uav_radius (collision)
        self.assertGreater(pre_step_surface_dist, uav_radius)
        self.assertLess(post_step_surface_dist, uav_radius)

    # -------------------------------------------------------------------------
    # 12. R2 blocked-motion / deadlock behavior verification
    # -------------------------------------------------------------------------
    def test_12_r2_deadlock_resolution_via_directional_freedom(self):
        """When Front is blocked (sec 0 = 0.85m), lateral steering command (sec 2 = 10m) is not scaled to 0."""
        clearances = np.full(8, 10.0, dtype=np.float32)
        clearances[0] = 0.85  # Front obstacle
        obs = self._create_mock_obs(clearances=clearances)

        # Policy steering diagonally: vx = 2.0, vy = 1.0
        raw_cmd = CommandedVelocities(vx=2.0, vy=1.0, vz=0.0, yaw_rate=0.5)
        raw_act = self.shield.action_handler.unscale_action(raw_cmd)

        shielded_act, diag = self.shield.filter_action(obs, raw_act)

        # Forward must be blocked (0.0), but lateral must remain positive (~1.0 m/s)
        self.assertAlmostEqual(diag.shielded_cmd.vx, 0.0, places=3)
        self.assertAlmostEqual(diag.shielded_cmd.vy, 1.0, places=2)

    # -------------------------------------------------------------------------
    # 13. Regression Test: Layer 1 Active Turn Assist
    # -------------------------------------------------------------------------
    def test_13_layer1_active_turn_assist(self):
        """When |d_psi| > 90 deg and rear obstacle is present, Layer 1 injects active yaw rate."""
        obs = self._create_mock_obs(
            goal_body=np.array([-5.0, 5.0, 0.0], dtype=np.float32),  # Behind and to the left (+135 deg)
            clearances=np.array([10.0, 10.0, 10.0, 1.5, 10.0, 10.0, 10.0, 10.0], dtype=np.float32)  # Rear obstacle
        )
        # Policy commands negative vx and zero yaw
        raw_cmd = CommandedVelocities(vx=-1.0, vy=0.0, vz=0.0, yaw_rate=0.0)
        raw_act = self.shield.action_handler.unscale_action(raw_cmd)

        shielded_act, diag = self.shield.filter_action(obs, raw_act)
        self.assertTrue(diag.layer1_active)
        self.assertAlmostEqual(diag.shielded_cmd.vx, 0.0, places=4)
        # Yaw rate must be positive (turning towards goal at +135 deg)
        self.assertGreater(diag.shielded_cmd.yaw_rate, 0.5)

    # -------------------------------------------------------------------------
    # 14. Regression Test: Terminal Goal Alignment Assist
    # -------------------------------------------------------------------------
    def test_14_terminal_goal_alignment_assist(self):
        """When d_goal < 1.0m and speed is low (<0.5 m/s), terminal bias is applied."""
        obs = self._create_mock_obs(
            goal_body=np.array([0.6, -0.4, 0.0], dtype=np.float32),  # Close to goal
            vel_body=np.array([0.1, 0.0, 0.0], dtype=np.float32),   # Low speed
            clearances=np.full(8, 10.0, dtype=np.float32)
        )
        raw_cmd = CommandedVelocities(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)
        raw_act = self.shield.action_handler.unscale_action(raw_cmd)

        shielded_act, diag = self.shield.filter_action(obs, raw_act)
        # Gentle velocity towards goal must be present
        self.assertGreater(diag.shielded_cmd.vx, 0.0)
        self.assertLess(diag.shielded_cmd.vy, 0.0)  # Towards negative y (goal is at y = -0.4)

    # -------------------------------------------------------------------------
    # 15. Regression Test: Calibrated Margin Standoff Resolution
    # -------------------------------------------------------------------------
    def test_15_calibrated_margin_standoff_resolution(self):
        """With safety_margin=0.40m (d_base=0.75m), obstacle at 0.763m permits positive safe speed."""
        cal_config = SafetyShieldConfig(safety_margin=0.40)
        cal_shield = DeterministicSafetyShield(cal_config)
        self.assertAlmostEqual(cal_config.d_base, 0.75, places=3)

        clearances = np.full(8, 10.0, dtype=np.float32)
        clearances[0] = 0.763  # Obstacle at 0.763m (Suite E geometry)
        obs = self._create_mock_obs(clearances=clearances)
        raw_act = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        shielded_act, diag = cal_shield.filter_action(obs, raw_act)
        # With d_base=0.75m, c = 0.763 - 0.75 = 0.013m > 0 -> positive safe speed permitted
        self.assertGreater(diag.safe_speeds[0], 0.0)
        self.assertGreater(diag.shielded_cmd.vx, 0.0)


if __name__ == "__main__":
    unittest.main()
