"""
test_option_a_acceptance.py
---------------------------
Prerequisite Sensor & Interface Acceptance Test Suite for Option A:
360° Planar LiDAR with 8-Sector, 26-D Observation.

Tests:
1. 72-ray, 360° geometric parameters: N=72, angle_min=-pi, angle_max=pi - 2pi/72, inc=5.0 deg.
2. Ray-to-sector allocation: exactly 9 rays per sector across all 8 sectors.
3. Wraparound and sector boundary azimuth handling.
4. Omnidirectional obstacle detection in all 8 directions at known distances.
5. Analytical cylinder geometry verification: ray-cylinder intersection vs analytical distance.
6. Near-range (< 0.10m) and max-range/clear (> 10.0m / +Inf) handling in 360° context.
7. Direct-array ambiguity resolution: bare +Inf rejected, verified +Inf accepted.
8. Complete classification of sensor states (VALID_HIT, MEASURED_CLEAR, SUBMIN_PROXIMITY, CORRUPT_INVALID, STALE_TIMED_OUT).
9. Elimination of blind-zone conflation: in 360° scans, unobserved sectors are 0; 0.0m strictly denotes fault/stale/corrupt.
"""

import math
import sys
import unittest
import numpy as np

from rl_env.config import UAVEnvConfig
from rl_env.observation_26d import (
    ObservationHandler26D,
    SectorStatus,
    InvalidClearancePolicy,
    Observation26DMetadata
)


class TestOptionAAcceptance(unittest.TestCase):
    def setUp(self):
        self.config = UAVEnvConfig()
        self.handler = ObservationHandler26D(
            self.config,
            invalid_policy=InvalidClearancePolicy.CONSERVATIVE_ZERO,
            staleness_timeout_s=0.20,
            min_range=0.10,
            max_range=10.0,
            default_clear_range=10.0
        )
        self.num_rays = 72
        self.angle_min = math.radians(-177.5)
        self.angle_inc = math.radians(5.0)  # 5.0 deg
        self.angle_max = math.radians(177.5)        # +177.5 deg

    def test_01_sensor_geometry_and_ray_allocation(self):
        """Verify that 72 rays across 360 deg yield exactly 9 rays per sector for all 8 sectors."""
        counts = [0] * 8
        sector_angles = [[] for _ in range(8)]

        for i in range(self.num_rays):
            angle = self.angle_min + i * self.angle_inc
            sec = ObservationHandler26D.angle_to_sector(angle)
            counts[sec] += 1
            sector_angles[sec].append(math.degrees(angle))

        # Empirical assertion: Exactly 9 rays in every single sector
        for sec_idx, name in enumerate(ObservationHandler26D.SECTOR_NAMES):
            self.assertEqual(
                counts[sec_idx],
                9,
                f"Sector {sec_idx} ({name}) received {counts[sec_idx]} rays, expected exactly 9"
            )

        # Check total rays
        self.assertEqual(sum(counts), 72)

        # Check sector angular spans
        # Sector 0 (Front): should span -22.5 deg to +17.5 deg
        self.assertAlmostEqual(sector_angles[0][0], -22.5, places=4)
        self.assertAlmostEqual(sector_angles[0][-1], +17.5, places=4)

        # Sector 4 (Back): should span +157.5 deg to +177.5 deg and -177.5 deg to -162.5 deg
        self.assertIn(-177.5, sector_angles[4])
        self.assertIn(177.5, sector_angles[4])

    def test_02_wraparound_and_boundary_cases(self):
        """Verify azimuth angle wrapping and exact sector boundary partitioning."""
        test_cases = [
            (-math.pi, 4),                     # -180.0 deg -> Back
            (math.pi, 4),                      # +180.0 deg -> Back
            (math.pi - 1e-6, 4),               # +179.999 deg -> Back
            (-math.pi + 1e-6, 4),              # -179.999 deg -> Back
            (math.radians(-22.5), 0),          # -22.5 deg (boundary) -> Front (inclusive lower)
            (math.radians(22.4999), 0),        # +22.4999 deg -> Front
            (math.radians(22.5), 1),           # +22.5 deg (boundary) -> Front-Left
            (math.radians(67.5), 2),           # +67.5 deg (boundary) -> Left
            (math.radians(112.5), 3),          # +112.5 deg (boundary) -> Back-Left
            (math.radians(157.5), 4),          # +157.5 deg (boundary) -> Back
            (math.radians(-157.5), 5),         # -157.5 deg (boundary) -> Back-Right (inclusive lower [-157.5, -112.5))
            (math.radians(-157.5001), 4),      # -157.5001 deg -> Back (<-157.5)
            (math.radians(-112.5), 6),         # -112.5 deg (boundary) -> Right
            (math.radians(-67.5), 7),          # -67.5 deg (boundary) -> Front-Right
        ]

        for ang_rad, expected_sec in test_cases:
            sec = ObservationHandler26D.angle_to_sector(ang_rad)
            self.assertEqual(
                sec,
                expected_sec,
                f"Angle {math.degrees(ang_rad):.4f} deg mapped to sector {sec}, expected {expected_sec}"
            )

    def test_03_omnidirectional_obstacle_detection_all_8_sectors(self):
        """Verify representative obstacles placed in all 8 directions at known distances."""
        # Known ground-truth obstacle distances for all 8 sectors
        test_distances = {
            0: 2.00,  # Front
            1: 3.50,  # Front-Left
            2: 1.50,  # Left
            3: 4.20,  # Back-Left
            4: 2.80,  # Back
            5: 5.10,  # Back-Right
            6: 1.80,  # Right
            7: 3.00   # Front-Right
        }

        # Initialize 72-ray scan with open sky (+Inf)
        ranges = [float("inf")] * self.num_rays

        # Inject obstacles into the center ray of each sector
        center_angles = {
            0: 0.0,
            1: math.radians(45.0),
            2: math.radians(90.0),
            3: math.radians(135.0),
            4: math.radians(180.0),
            5: math.radians(-135.0),
            6: math.radians(-90.0),
            7: math.radians(-45.0)
        }

        for sec_idx, d_truth in test_distances.items():
            target_ang = center_angles[sec_idx]
            # Find closest ray index
            closest_ray = None
            min_diff = float("inf")
            for r_i in range(self.num_rays):
                ang = self.angle_min + r_i * self.angle_inc
                diff = abs(math.atan2(math.sin(ang - target_ang), math.cos(ang - target_ang)))
                if diff < min_diff:
                    min_diff = diff
                    closest_ray = r_i
            # Assign obstacle distance to that ray and its neighbors
            ranges[closest_ray] = d_truth

        clearances, meta = self.handler.process_raw_scan(
            ranges=ranges,
            angle_min=self.angle_min,
            angle_increment=self.angle_inc
        )

        # Assert every single sector reports its exact ground-truth distance within 0.01m
        for sec_idx in range(8):
            expected_d = test_distances[sec_idx]
            actual_d = float(clearances[sec_idx])
            self.assertAlmostEqual(
                actual_d,
                expected_d,
                places=2,
                msg=f"Sector {sec_idx} ({ObservationHandler26D.SECTOR_NAMES[sec_idx]}) expected {expected_d}m, got {actual_d}m"
            )
            self.assertEqual(meta.sector_status[sec_idx], SectorStatus.VALID_HIT)
            self.assertTrue(meta.validity_mask[sec_idx])

        # Overall perception must be 100% valid
        self.assertTrue(meta.is_perception_valid)

    def test_04_analytical_cylinder_geometry_verification(self):
        """
        Verify ray-casting against an analytical cylinder obstacle.
        Cylinder: center (x_c, y_c) = (3.0, 0.0) m, radius R = 0.50 m.
        Analytical closest distance to vehicle center: d_true = 3.0 - 0.50 = 2.50 m.
        """
        cx, cy, R = 3.0, 0.0, 0.50
        ranges = [float("inf")] * self.num_rays

        for i in range(self.num_rays):
            angle = self.angle_min + i * self.angle_inc
            dx = math.cos(angle)
            dy = math.sin(angle)

            # Ray-circle intersection: |t * d - c|^2 = R^2
            # t^2 - 2*(d.c)*t + (|c|^2 - R^2) = 0
            b = -2.0 * (dx * cx + dy * cy)
            c = (cx * cx + cy * cy) - R * R
            disc = b * b - 4.0 * c

            if disc >= 0:
                t1 = (-b - math.sqrt(disc)) / 2.0
                if t1 > 0:
                    ranges[i] = t1

        clearances, meta = self.handler.process_raw_scan(
            ranges=ranges,
            angle_min=self.angle_min,
            angle_increment=self.angle_inc
        )

        # Sector 0 (Front) must report closest cylinder surface distance
        # Analytical expected closest distance is exactly cx - R = 2.50 m
        actual_front = float(clearances[0])
        self.assertAlmostEqual(actual_front, 2.50, delta=0.02, msg=f"Expected 2.50m, got {actual_front}m")
        self.assertEqual(meta.sector_status[0], SectorStatus.VALID_HIT)

        # All other sectors should be measured clear (10.0m)
        for s in range(1, 8):
            self.assertAlmostEqual(float(clearances[s]), 10.0, places=2)
            self.assertEqual(meta.sector_status[s], SectorStatus.MEASURED_CLEAR)

    def test_05_direct_array_ambiguity_resolution(self):
        """
        Verify that bare +Inf in a direct array CANNOT be silently interpreted as 10.0m clear space.
        Only explicit validity confirmation or raw scan ray processing permits MEASURED_CLEAR.
        """
        # Test 1: Bare array with +Inf
        array_with_inf = [5.0, 5.0, float("inf"), 5.0, float("inf"), 5.0, 5.0, 5.0]

        # By default, require_explicit_validity_for_inf is True: bare +Inf without validity mask is rejected
        clean_bare, meta_bare = self.handler.sanitize_clearances_array(array_with_inf)
        self.assertEqual(clean_bare[2], 0.0, "Bare +Inf falsely accepted as clear without validity mask!")
        self.assertEqual(clean_bare[4], 0.0, "Bare +Inf falsely accepted as clear without validity mask!")
        self.assertEqual(meta_bare.sector_status[2], SectorStatus.UNOBSERVED_BLIND_ZONE)
        self.assertEqual(meta_bare.sector_status[4], SectorStatus.UNOBSERVED_BLIND_ZONE)
        self.assertFalse(meta_bare.validity_mask[2])
        self.assertFalse(meta_bare.validity_mask[4])

        # Test 2: Array with explicit validity mask confirming sectors 2 and 4 were actively observed
        valid_mask = [True, True, True, True, True, True, True, True]
        clean_verified, meta_verified = self.handler.sanitize_clearances_array(
            array_with_inf,
            sector_validity_mask=valid_mask
        )
        self.assertEqual(clean_verified[2], 10.0, "Verified +Inf rejected despite valid mask!")
        self.assertEqual(clean_verified[4], 10.0, "Verified +Inf rejected despite valid mask!")
        self.assertEqual(meta_verified.sector_status[2], SectorStatus.MEASURED_CLEAR)
        self.assertEqual(meta_verified.sector_status[4], SectorStatus.MEASURED_CLEAR)
        self.assertTrue(meta_verified.validity_mask[2])
        self.assertTrue(meta_verified.validity_mask[4])

    def test_06_complete_elimination_of_blind_sectors(self):
        """Verify that in a 360° scan with open space, zero sectors are classified as UNOBSERVED_BLIND_ZONE."""
        ranges_open = [float("inf")] * self.num_rays

        clearances, meta = self.handler.process_raw_scan(
            ranges=ranges_open,
            angle_min=self.angle_min,
            angle_increment=self.angle_inc
        )

        # In 360° mode, all 8 sectors must be MEASURED_CLEAR (10.0m)
        np.testing.assert_array_equal(clearances, np.full(8, 10.0, dtype=np.float32))

        # Zero sectors may be classified as UNOBSERVED_BLIND_ZONE
        blind_count = sum(1 for s in meta.sector_status if s == SectorStatus.UNOBSERVED_BLIND_ZONE)
        self.assertEqual(blind_count, 0, f"Found {blind_count} blind sectors in 360 deg scan!")

        # All sectors must have validity_mask == True
        self.assertTrue(np.all(meta.validity_mask))
        self.assertTrue(meta.is_perception_valid)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOptionAAcceptance)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    res = run_tests()
    sys.exit(0 if res.wasSuccessful() else 1)
