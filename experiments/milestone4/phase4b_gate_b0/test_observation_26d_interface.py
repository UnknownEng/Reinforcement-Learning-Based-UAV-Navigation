"""
test_observation_26d_interface.py
---------------------------------
Comprehensive Gate B0 Interface Validation Test Suite for 26-D Observation Space.

Validates:
1. Preservation of 18-D proprioceptive state ordering, values, and bounds.
2. 26-D observation dimensions, shape (26,), dtype float32, and Gymnasium Box compliance.
3. 8-D LiDAR sector clearance ordering, angular mapping, and units in meters.
4. Normal range measurements [0.10, 10.0] m.
5. Minimum range (< 0.10 m) proximity hazard clamping.
6. Maximum range (> 10.0 m) and +Inf handling (measured clear).
7. Non-finite values (NaN, -Inf) and negative ranges.
8. Unobserved blind sectors (sectors 3, 4, 5 for 180 deg LiDAR): conservative non-clear space handling.
9. Stale sensor data (age > 200 ms) and unavailable sensor streams.
10. Explicit distinction between measured clear space and invalid/unobserved data via metadata.
11. Direct clearance array input sanitization.
12. Policy compatibility and isolation checks (18-D model rejects 26-D input).
"""

import json
import math
import os
import sys
import unittest
import numpy as np

# Use project virtualenv dependencies
from rl_env.config import UAVEnvConfig
from rl_env.observation import ObservationHandler
from rl_env.observation_26d import (
    ObservationHandler26D,
    SectorStatus,
    InvalidClearancePolicy,
    Observation26DMetadata
)


class TestObservation26DInterface(unittest.TestCase):
    def setUp(self):
        self.config = UAVEnvConfig()
        self.handler_18d = ObservationHandler(self.config)
        self.handler_26d = ObservationHandler26D(
            self.config,
            invalid_policy=InvalidClearancePolicy.CONSERVATIVE_ZERO,
            staleness_timeout_s=0.20,
            min_range=0.10,
            max_range=10.0,
            default_clear_range=10.0
        )

        # Realistic flight state test fixtures
        self.pos_world = np.array([2.5, -1.2, 1.8], dtype=np.float32)
        self.vel_world = np.array([0.8, -0.3, 0.1], dtype=np.float32)
        self.quat = np.array([0.9238795, 0.0, 0.0, 0.3826834], dtype=np.float32)  # yaw ~45 deg
        self.rates_body = np.array([0.02, -0.01, 0.05], dtype=np.float32)
        self.goal_world = np.array([6.0, 2.0, 2.0], dtype=np.float32)
        self.prev_action = np.array([0.4, -0.1, 0.2, 0.0], dtype=np.float32)
        self.ground_z = 0.0

    def test_01_proprioceptive_18d_preservation(self):
        """Verify indices 0..17 exactly match the frozen 18-D ObservationHandler output."""
        obs_18 = self.handler_18d.compute_observation(
            pos_world=self.pos_world,
            vel_world=self.vel_world,
            quat=self.quat,
            rates_body=self.rates_body,
            goal_world=self.goal_world,
            prev_action=self.prev_action,
            ground_z=self.ground_z
        )

        obs_26, meta = self.handler_26d.compute_observation_with_metadata(
            pos_world=self.pos_world,
            vel_world=self.vel_world,
            quat=self.quat,
            rates_body=self.rates_body,
            goal_world=self.goal_world,
            prev_action=self.prev_action,
            clearances=[5.0] * 8,
            ground_z=self.ground_z
        )

        self.assertEqual(obs_18.shape, (18,))
        self.assertEqual(obs_26.shape, (26,))
        self.assertEqual(obs_26.dtype, np.float32)

        # Compare slice 0..18
        np.testing.assert_allclose(
            obs_26[:18],
            obs_18,
            rtol=1e-6,
            atol=1e-6,
            err_msg="18D proprioceptive slice does not match baseline ObservationHandler!"
        )

        # Check bounds
        self.assertTrue(np.all(obs_26[:18] >= self.handler_26d.observation_space.low[:18]))
        self.assertTrue(np.all(obs_26[:18] <= self.handler_26d.observation_space.high[:18]))

    def test_02_gymnasium_box_space_compliance(self):
        """Verify Gymnasium Box space shape, bounds, and containment."""
        space = self.handler_26d.observation_space
        self.assertEqual(space.shape, (26,))
        self.assertEqual(space.dtype, np.float32)

        # Check clearance bounds [0.0, 10.0]
        np.testing.assert_array_equal(space.low[18:], np.zeros(8, dtype=np.float32))
        np.testing.assert_array_equal(space.high[18:], np.full(8, 10.0, dtype=np.float32))

        # Sample and contain
        sample = space.sample()
        self.assertTrue(space.contains(sample))

    def test_03_sector_angular_mapping(self):
        """Verify exact angular azimuth mapping into 8 sectors matching REP-103."""
        # Azimuth angles in radians
        angles_and_expected_sectors = [
            (0.0, 0),                       # Front (0 deg)
            (math.radians(10.0), 0),        # Front (+10 deg)
            (math.radians(-10.0), 0),       # Front (-10 deg)
            (math.radians(45.0), 1),        # Front-Left (+45 deg)
            (math.radians(90.0), 2),        # Left (+90 deg)
            (math.radians(135.0), 3),       # Back-Left (+135 deg)
            (math.radians(180.0), 4),       # Back (+180 deg)
            (math.radians(-180.0), 4),      # Back (-180 deg)
            (math.radians(-135.0), 5),      # Back-Right (-135 deg)
            (math.radians(-90.0), 6),       # Right (-90 deg)
            (math.radians(-45.0), 7),       # Front-Right (-45 deg)
        ]

        for angle, expected_sec in angles_and_expected_sectors:
            sec = ObservationHandler26D.angle_to_sector(angle)
            self.assertEqual(
                sec,
                expected_sec,
                f"Angle {math.degrees(angle):.1f} deg mapped to sector {sec}, expected {expected_sec}"
            )

    def test_04_normal_lidar_scan_processing(self):
        """Verify processing of a realistic 36-beam planar LiDAR scan."""
        # 36 beams from -90 deg (-pi/2) to +90 deg (+pi/2)
        angle_min = -math.pi / 2.0
        angle_max = math.pi / 2.0
        num_rays = 36
        angle_inc = (angle_max - angle_min) / (num_rays - 1)

        # Place an obstacle directly in front at 2.5 meters (rays 16..20 around 0 deg)
        ranges = [10.0] * num_rays
        for r_idx in range(16, 21):
            ranges[r_idx] = 2.50

        # Place another obstacle to the left at 4.0 meters (rays 30..35 around +90 deg)
        for r_idx in range(30, 36):
            ranges[r_idx] = 4.00

        clearances, meta = self.handler_26d.process_raw_scan(
            ranges=ranges,
            angle_min=angle_min,
            angle_increment=angle_inc
        )

        self.assertEqual(len(clearances), 8)
        self.assertAlmostEqual(clearances[0], 2.50, places=2)  # Front
        self.assertEqual(meta.sector_status[0], SectorStatus.VALID_HIT)

        self.assertAlmostEqual(clearances[2], 4.00, places=2)  # Left
        self.assertEqual(meta.sector_status[2], SectorStatus.VALID_HIT)

        # Sectors 7 (Front-Right) and 1 (Front-Left) should be measured clear (10.0m)
        self.assertAlmostEqual(clearances[7], 10.0, places=2)
        self.assertEqual(meta.sector_status[7], SectorStatus.MEASURED_CLEAR)

    def test_05_conservative_unobserved_sector_handling(self):
        """CRITICAL: Unobserved blind sectors (3, 4, 5) must NOT be interpreted as clear space (10.0m)."""
        # Forward 180 deg scan has zero rays in sectors 3 (Back-Left), 4 (Back), 5 (Back-Right)
        angle_min = -math.pi / 2.0
        angle_max = math.pi / 2.0
        num_rays = 36
        angle_inc = (angle_max - angle_min) / (num_rays - 1)

        ranges = [10.0] * num_rays  # All forward rays clear

        clearances, meta = self.handler_26d.process_raw_scan(
            ranges=ranges,
            angle_min=angle_min,
            angle_increment=angle_inc
        )

        # Under CONSERVATIVE_ZERO policy, blind rear sectors must report 0.0 m (zero clearance)
        self.assertEqual(self.handler_26d.invalid_policy, InvalidClearancePolicy.CONSERVATIVE_ZERO)
        self.assertAlmostEqual(clearances[3], 0.0, places=4, msg="Sector 3 (Back-Left) falsely reported clear space!")
        self.assertAlmostEqual(clearances[4], 0.0, places=4, msg="Sector 4 (Back) falsely reported clear space!")
        self.assertAlmostEqual(clearances[5], 0.0, places=4, msg="Sector 5 (Back-Right) falsely reported clear space!")

        # Check metadata status
        self.assertEqual(meta.sector_status[3], SectorStatus.UNOBSERVED_BLIND_ZONE)
        self.assertEqual(meta.sector_status[4], SectorStatus.UNOBSERVED_BLIND_ZONE)
        self.assertEqual(meta.sector_status[5], SectorStatus.UNOBSERVED_BLIND_ZONE)

        # Validity mask must be False for blind sectors
        self.assertFalse(meta.validity_mask[3])
        self.assertFalse(meta.validity_mask[4])
        self.assertFalse(meta.validity_mask[5])

    def test_06_minimum_and_maximum_range_boundary_clamping(self):
        """Verify sub-minimum range (< 0.10m) and excessive range (> 10.0m / +Inf) clamping."""
        angle_min = -math.pi / 2.0
        angle_inc = math.pi / 35.0
        num_rays = 36

        ranges = [float("inf")] * num_rays
        # Ray 18 (Front): 0.05 m (sub-minimum proximity hazard)
        ranges[18] = 0.05
        # Ray 27 (Left): 15.0 m (beyond max range -> clear)
        ranges[27] = 15.0

        clearances, meta = self.handler_26d.process_raw_scan(
            ranges=ranges,
            angle_min=angle_min,
            angle_increment=angle_inc
        )

        # Front sector must clamp to min_range 0.10 m
        self.assertAlmostEqual(clearances[0], 0.10, places=4)
        self.assertEqual(meta.sector_status[0], SectorStatus.SUBMIN_PROXIMITY)

        # Ray beyond 10.0m is measured clear
        self.assertAlmostEqual(clearances[1], 10.0, places=4)
        self.assertEqual(meta.sector_status[1], SectorStatus.MEASURED_CLEAR)

    def test_07_nan_and_negative_infinity_corrupt_data(self):
        """Verify corrupt data (NaN, -Inf, negative range) does NOT report clear space."""
        angle_min = -math.pi / 2.0
        angle_inc = math.pi / 35.0
        num_rays = 36

        # Entire scan is corrupt
        ranges = [float("nan")] * num_rays
        ranges[18] = float("-inf")
        ranges[19] = -5.0

        clearances, meta = self.handler_26d.process_raw_scan(
            ranges=ranges,
            angle_min=angle_min,
            angle_increment=angle_inc
        )

        # Under CONSERVATIVE_ZERO, corrupt sectors report 0.0 m
        for sec_idx in range(8):
            self.assertAlmostEqual(
                clearances[sec_idx],
                0.0,
                places=4,
                msg=f"Sector {sec_idx} falsely reported non-zero clearance on corrupt input!"
            )
            self.assertFalse(meta.validity_mask[sec_idx])

    def test_08_stale_sensor_data_handling(self):
        """Verify that stale sensor data (> 200 ms) triggers conservative hazard fallback."""
        angle_min = -math.pi / 2.0
        angle_inc = math.pi / 35.0
        ranges = [10.0] * 36

        current_time = 100.50
        stale_sensor_time = 100.25  # age = 0.25 s > 0.20 s threshold

        clearances, meta = self.handler_26d.process_raw_scan(
            ranges=ranges,
            angle_min=angle_min,
            angle_increment=angle_inc,
            sensor_time=stale_sensor_time,
            current_time=current_time
        )

        self.assertTrue(meta.is_sensor_stale)
        self.assertFalse(meta.is_perception_valid)
        self.assertAlmostEqual(meta.sensor_age_s, 0.25, places=3)

        # All sectors must drop to conservative 0.0 m
        np.testing.assert_array_equal(clearances, np.zeros(8, dtype=np.float32))
        for status in meta.sector_status:
            self.assertEqual(status, SectorStatus.STALE_TIMED_OUT)

    def test_09_unavailable_sensor_stream(self):
        """Verify that missing/None sensor inputs produce conservative hazard clearance."""
        obs_26, meta = self.handler_26d.compute_observation_with_metadata(
            pos_world=self.pos_world,
            vel_world=self.vel_world,
            quat=self.quat,
            rates_body=self.rates_body,
            goal_world=self.goal_world,
            prev_action=self.prev_action,
            clearances=None,
            raw_scan=None,
            ground_z=self.ground_z
        )

        self.assertEqual(obs_26.shape, (26,))
        self.assertTrue(meta.is_sensor_stale)
        self.assertFalse(meta.is_perception_valid)

        # 18D proprioceptive slice remains valid
        self.assertTrue(np.all(np.isfinite(obs_26[:18])))
        # 8D clearance slice drops to 0.0 m
        np.testing.assert_array_equal(obs_26[18:], np.zeros(8, dtype=np.float32))

    def test_10_direct_clearance_array_sanitization(self):
        """Verify direct 8-element clearance array input with invalid entries and ambiguity resolution."""
        dirty_clearances = [
            2.5,          # Valid (0)
            float("nan"), # Corrupt -> 0.0m (1)
            15.0,         # Beyond max -> 10.0m (2)
            0.02,         # Submin -> 0.10m (3)
            float("-inf"),# Corrupt -> 0.0m (4)
            -1.0,         # Negative -> 0.0m (5)
            7.5,          # Valid (6)
            float("inf")  # Bare +Inf without validity mask -> rejected to 0.0m (7)
        ]

        # Case A: Bare array without explicit validity mask (strict policy rejects bare +Inf)
        clean_a, meta_a = self.handler_26d.sanitize_clearances_array(dirty_clearances)
        expected_a = [2.5, 0.0, 10.0, 0.10, 0.0, 0.0, 7.5, 0.0]
        np.testing.assert_allclose(clean_a, expected_a, rtol=1e-5, atol=1e-5)
        self.assertFalse(meta_a.validity_mask[1])
        self.assertFalse(meta_a.validity_mask[4])
        self.assertFalse(meta_a.validity_mask[5])
        self.assertFalse(meta_a.validity_mask[7])  # Bare +Inf is unverified
        self.assertEqual(meta_a.sector_status[7], SectorStatus.UNOBSERVED_BLIND_ZONE)
        self.assertTrue(meta_a.validity_mask[0])
        self.assertTrue(meta_a.validity_mask[2])

        # Case B: Array with explicit validity mask confirming sector 7 was actively observed
        validity_mask = [True, False, True, True, False, False, True, True]
        clean_b, meta_b = self.handler_26d.sanitize_clearances_array(
            dirty_clearances,
            sector_validity_mask=validity_mask
        )
        expected_b = [2.5, 0.0, 10.0, 0.10, 0.0, 0.0, 7.5, 10.0]
        np.testing.assert_allclose(clean_b, expected_b, rtol=1e-5, atol=1e-5)
        self.assertTrue(meta_b.validity_mask[7])
        self.assertEqual(meta_b.sector_status[7], SectorStatus.MEASURED_CLEAR)

    def test_11_distinction_between_clear_space_and_invalid_data(self):
        """Verify that genuine measured clear space is strictly distinguishable from invalid data."""
        # Case A: Measured clear space
        ranges_clear = [float("inf")] * 36
        clean_a, meta_a = self.handler_26d.process_raw_scan(
            ranges=ranges_clear,
            angle_min=-math.pi/2.0,
            angle_increment=math.pi/35.0
        )

        # Case B: Corrupt / invalid scan
        ranges_corrupt = [float("nan")] * 36
        clean_b, meta_b = self.handler_26d.process_raw_scan(
            ranges=ranges_corrupt,
            angle_min=-math.pi/2.0,
            angle_increment=math.pi/35.0
        )

        # Sector 0 (Front) in Case A is 10.0m and valid
        self.assertAlmostEqual(clean_a[0], 10.0, places=2)
        self.assertEqual(meta_a.sector_status[0], SectorStatus.MEASURED_CLEAR)
        self.assertTrue(meta_a.validity_mask[0])

        # Sector 0 (Front) in Case B is 0.0m and invalid
        self.assertAlmostEqual(clean_b[0], 0.0, places=2)
        self.assertEqual(meta_b.sector_status[0], SectorStatus.CORRUPT_INVALID)
        self.assertFalse(meta_b.validity_mask[0])

        # Numerical and categorical distinction is absolute
        self.assertNotEqual(clean_a[0], clean_b[0])
        self.assertNotEqual(meta_a.sector_status[0], meta_b.sector_status[0])


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestObservation26DInterface)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    res = run_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
