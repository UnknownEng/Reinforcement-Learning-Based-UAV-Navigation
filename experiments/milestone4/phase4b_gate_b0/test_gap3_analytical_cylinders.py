"""
test_gap3_analytical_cylinders.py
---------------------------------
Gap 3: Multi-Angle Analytical Cylinder Tests

Comprehensive analytical test suite evaluating the 8-sector 26-D perception model
against an independent mathematical ray-circle intersection model across:
- All 8 directional sectors (including lateral, rear, and wraparound).
- Multiple cylinder radii in the specified [0.25, 0.80] m range.
- Multiple obstacle center distances [1.5, 7.5] m (128 parametric combinations).
- Non-central azimuths within each sector.
- Distant obstacle miss behavior (open paths).
- Sub-minimum proximity hazard clamping (< min_range).
- Multiple simultaneous obstacles across separate sectors without cross-talk.
- Tangent grazing ray numerical stability.
"""

import math
import unittest
from typing import Dict, List, Optional, Tuple
import numpy as np

from rl_env.observation_26d import ObservationHandler26D, SectorStatus, InvalidClearancePolicy


def ray_circle_intersection_analytical(
    ray_origin: Tuple[float, float],
    ray_angle_rad: float,
    circle_center: Tuple[float, float],
    circle_radius: float
) -> Optional[float]:
    """
    Independent ground-truth ray-circle intersection solver.
    Solves: t^2 - 2(d . c_rel) t + ||c_rel||^2 - R^2 = 0
    where d = (cos theta, sin theta), c_rel = circle_center - ray_origin.
    """
    ox, oy = ray_origin
    cx, cy = circle_center
    dx = math.cos(ray_angle_rad)
    dy = math.sin(ray_angle_rad)

    rel_x = cx - ox
    rel_y = cy - oy

    b = dx * rel_x + dy * rel_y
    c = rel_x**2 + rel_y**2 - circle_radius**2
    disc = b**2 - c

    if disc < 0.0:
        return None

    sqrt_disc = math.sqrt(max(0.0, disc))
    t1 = b - sqrt_disc
    t2 = b + sqrt_disc

    if t1 > 0.0:
        return t1
    elif t2 > 0.0:
        return t2
    return None


class TestGap3AnalyticalCylinders(unittest.TestCase):
    def setUp(self):
        self.handler = ObservationHandler26D(
            invalid_policy=InvalidClearancePolicy.CONSERVATIVE_ZERO,
            min_range=0.10,
            max_range=10.0,
            default_clear_range=10.0
        )
        self.num_rays = 72
        self.angle_min = -math.pi
        self.angle_inc = 2.0 * math.pi / 72.0

        # Sector definitions: (center_angle_deg, sector_index, name)
        self.sectors = [
            (0.0,    0, "front"),
            (45.0,   1, "front_left"),
            (90.0,   2, "left"),
            (135.0,  3, "back_left"),
            (180.0,  4, "back"),
            (-135.0, 5, "back_right"),
            (-90.0,  6, "right"),
            (-45.0,  7, "front_right")
        ]

    def _ray_angles(self) -> List[float]:
        return [self.angle_min + i * self.angle_inc for i in range(self.num_rays)]

    def _generate_scan_from_obstacles(
        self,
        obstacles: List[Tuple[Tuple[float, float], float]]
    ) -> List[float]:
        """
        Calculates ground-truth 72-ray ranges for one or more circular obstacles.
        obstacles: list of ((cx, cy), radius)
        """
        ranges = []
        for ang in self._ray_angles():
            closest_t = float('inf')
            for (cx, cy), r in obstacles:
                t = ray_circle_intersection_analytical((0.0, 0.0), ang, (cx, cy), r)
                if t is not None and t < closest_t:
                    closest_t = t
            ranges.append(closest_t)
        return ranges

    def test_3_1_full_parametric_sweep_128_configurations(self):
        """
        Case 3.1: Parametric sweep across all 8 sectors, 4 radii in [0.25, 0.80]m,
        and 4 distances in [1.5, 7.5]m. Total = 8 * 4 * 4 = 128 configurations.
        Verifies:
          - Target sector clearance exactly equals D - R (central ray strikes normal).
          - Sector status is VALID_HIT.
          - Untargeted sectors report 10.0m with MEASURED_CLEAR.
          - Absolute numerical error against analytical model < 1e-4 m.
        """
        radii = [0.25, 0.40, 0.60, 0.80]
        distances = [1.50, 3.00, 5.00, 7.50]

        total_tested = 0

        for az_deg, sec_idx, sec_name in self.sectors:
            az_rad = math.radians(az_deg)
            for r in radii:
                for d in distances:
                    total_tested += 1
                    cx = d * math.cos(az_rad)
                    cy = d * math.sin(az_rad)

                    ranges = self._generate_scan_from_obstacles([((cx, cy), r)])

                    clearances, meta = self.handler.process_raw_scan(
                        ranges=ranges,
                        angle_min=self.angle_min,
                        angle_increment=self.angle_inc
                    )

                    expected_clearance = d - r

                    # Target sector verification
                    actual_clearance = clearances[sec_idx]
                    self.assertAlmostEqual(
                        actual_clearance,
                        expected_clearance,
                        places=4,
                        msg=f"Failed in Sector {sec_idx} ({sec_name}) for d={d}, r={r}"
                    )
                    self.assertEqual(meta.sector_status[sec_idx], SectorStatus.VALID_HIT)

                    # Untargeted sectors verification
                    for other_sec in range(8):
                        if other_sec == sec_idx:
                            continue
                        # Angular width: arcsin(r/d)
                        ang_half_deg = math.degrees(math.asin(r / d))
                        # If angular width does not reach other sector boundary (22.5 deg from center):
                        if ang_half_deg < 20.0:
                            self.assertEqual(
                                clearances[other_sec],
                                10.0,
                                msg=f"Leakage into Sector {other_sec} from target {sec_name}"
                            )
                            self.assertEqual(meta.sector_status[other_sec], SectorStatus.MEASURED_CLEAR)

        self.assertEqual(total_tested, 128)

    def test_3_2_non_central_azimuths_in_all_sectors(self):
        """
        Case 3.2: Places cylinders at non-central azimuths (+10° and -10° offset) within each sector.
        D = 3.00 m, R = 0.30 m.
        Verifies that closest ray within the sector is properly aggregated.
        """
        d = 3.00
        r = 0.30

        for az_deg, sec_idx, sec_name in self.sectors:
            for offset_deg in [-10.0, 10.0]:
                test_az_deg = az_deg + offset_deg
                test_az_rad = math.radians(test_az_deg)
                cx = d * math.cos(test_az_rad)
                cy = d * math.sin(test_az_rad)

                ranges = self._generate_scan_from_obstacles([((cx, cy), r)])

                clearances, meta = self.handler.process_raw_scan(
                    ranges=ranges,
                    angle_min=self.angle_min,
                    angle_increment=self.angle_inc
                )

                # Ground-truth sector minimum from ray-circle analytical calculation
                sec_ray_hits = []
                for ray_idx, ang in enumerate(self._ray_angles()):
                    if self.handler.angle_to_sector(ang) == sec_idx:
                        if ranges[ray_idx] < 10.0:
                            sec_ray_hits.append(ranges[ray_idx])

                self.assertTrue(len(sec_ray_hits) > 0)
                expected_sec_min = min(sec_ray_hits)

                self.assertAlmostEqual(clearances[sec_idx], expected_sec_min, places=4)
                self.assertEqual(meta.sector_status[sec_idx], SectorStatus.VALID_HIT)

    def test_3_3_complete_miss_beyond_range(self):
        """
        Case 3.3: Cylinder placed at distance 12.0m, radius 0.50m (closest distance 11.5m > 10.0m max_range).
        All rays are infinite/miss.
        All 8 sectors must report default_clear_range (10.0m) and MEASURED_CLEAR.
        """
        cx, cy = 12.0, 0.0
        r = 0.50

        ranges = self._generate_scan_from_obstacles([((cx, cy), r)])

        clearances, meta = self.handler.process_raw_scan(
            ranges=ranges,
            angle_min=self.angle_min,
            angle_increment=self.angle_inc
        )

        for sec in range(8):
            self.assertEqual(clearances[sec], 10.0)
            self.assertEqual(meta.sector_status[sec], SectorStatus.MEASURED_CLEAR)
        self.assertTrue(meta.is_perception_valid)

    def test_3_4_submin_proximity_hazard_handling(self):
        """
        Case 3.4: Cylinder placed at D = 0.30 m, R = 0.25 m in Sector 0 (closest point 0.05m < 0.10m min_range).
        Sector 0 must be clamped to min_range (0.10 m) with SUBMIN_PROXIMITY status.
        """
        cx, cy = 0.30, 0.0
        r = 0.25

        ranges = self._generate_scan_from_obstacles([((cx, cy), r)])

        clearances, meta = self.handler.process_raw_scan(
            ranges=ranges,
            angle_min=self.angle_min,
            angle_increment=self.angle_inc
        )

        self.assertAlmostEqual(clearances[0], 0.10, places=5)
        self.assertEqual(meta.sector_status[0], SectorStatus.SUBMIN_PROXIMITY)

        # Lateral and rear sectors remain completely clear
        for sec in [2, 3, 4, 5, 6]:
            self.assertAlmostEqual(clearances[sec], 10.0, places=5)
            self.assertEqual(meta.sector_status[sec], SectorStatus.MEASURED_CLEAR)

    def test_3_5_multiple_simultaneous_obstacles(self):
        """
        Case 3.5: Multiple obstacles placed simultaneously across separate sectors:
          - Sector 0 (Front): D = 2.00 m, R = 0.30 m -> clearance 1.70 m
          - Sector 3 (Back-Left): D = 4.00 m, R = 0.50 m -> clearance 3.50 m
          - Sector 6 (Right): D = 3.50 m, R = 0.40 m -> clearance 3.10 m
        Verifies all three sectors report their exact clearances with VALID_HIT,
        and intermediate sectors remain MEASURED_CLEAR without cross-talk.
        """
        obs_list = [
            # Sector 0 (0 deg)
            ((2.00 * math.cos(0.0), 2.00 * math.sin(0.0)), 0.30),
            # Sector 3 (135 deg)
            ((4.00 * math.cos(math.radians(135.0)), 4.00 * math.sin(math.radians(135.0))), 0.50),
            # Sector 6 (-90 deg)
            ((3.50 * math.cos(math.radians(-90.0)), 3.50 * math.sin(math.radians(-90.0))), 0.40)
        ]

        ranges = self._generate_scan_from_obstacles(obs_list)

        clearances, meta = self.handler.process_raw_scan(
            ranges=ranges,
            angle_min=self.angle_min,
            angle_increment=self.angle_inc
        )

        # Expected clearances: D - R
        self.assertAlmostEqual(clearances[0], 1.70, places=4)
        self.assertEqual(meta.sector_status[0], SectorStatus.VALID_HIT)

        self.assertAlmostEqual(clearances[3], 3.50, places=4)
        self.assertEqual(meta.sector_status[3], SectorStatus.VALID_HIT)

        self.assertAlmostEqual(clearances[6], 3.10, places=4)
        self.assertEqual(meta.sector_status[6], SectorStatus.VALID_HIT)

        # Sectors 1, 2, 4, 5, 7 must be clear
        for sec in [1, 2, 4, 5, 7]:
            self.assertEqual(clearances[sec], 10.0)
            self.assertEqual(meta.sector_status[sec], SectorStatus.MEASURED_CLEAR)

    def test_3_6_tangent_grazing_ray_stability(self):
        """
        Case 3.6: Ray exactly tangent to cylinder edge (discriminant = 0).
        Verifies numerical stability, lack of NaN/complex values, and clean boundary behavior.
        """
        # Place cylinder so ray at 0 deg is tangent to its edge
        # Ray along +X: y = 0.
        # Cylinder at cx = 3.0, cy = 0.5, R = 0.5 -> tangent at (3.0, 0.0)
        cx, cy = 3.0, 0.5
        r = 0.5

        t = ray_circle_intersection_analytical((0.0, 0.0), 0.0, (cx, cy), r)
        self.assertIsNotNone(t)
        self.assertAlmostEqual(t, 3.0, places=5)
        self.assertFalse(math.isnan(t))


if __name__ == "__main__":
    unittest.main()
