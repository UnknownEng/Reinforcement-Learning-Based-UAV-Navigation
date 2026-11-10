"""
test_gap2_boundary_straddling.py
--------------------------------
Gap 2: Boundary-Straddling Obstacle Verification

Tests cylinders centered near sector boundaries (e.g. +22.5°, sector boundaries across all 8 sectors,
and wraparound across ±180°). Calculates which actual sensor rays intersect each cylinder,
verifies analytical intersection distances, tests adjacent sector clearances and minimum aggregation,
and confirms that adjacent sectors only report hits when rays geometrically intersect.
"""

import math
import unittest
from typing import Dict, List, Optional, Tuple
import numpy as np

from rl_env.observation_26d import ObservationHandler26D, SectorStatus, InvalidClearancePolicy


def ray_circle_intersection(
    ray_origin: Tuple[float, float],
    ray_angle_rad: float,
    circle_center: Tuple[float, float],
    circle_radius: float
) -> Optional[float]:
    """
    Computes exact analytical ray-circle intersection distance.
    Ray: p(t) = ray_origin + t * [cos(theta), sin(theta)], t > 0.
    Circle: ||p - circle_center||^2 = circle_radius^2.
    Returns closest positive intersection distance t, or None if ray misses or touches behind origin.
    """
    ox, oy = ray_origin
    cx, cy = circle_center
    dx = math.cos(ray_angle_rad)
    dy = math.sin(ray_angle_rad)

    # Relative center vector: c - o
    rel_x = cx - ox
    rel_y = cy - oy

    # Quadratic equation: t^2 - 2*(d . rel)*t + ||rel||^2 - R^2 = 0
    b = dx * rel_x + dy * rel_y
    c = rel_x**2 + rel_y**2 - circle_radius**2
    discriminant = b**2 - c

    if discriminant < 0.0:
        return None  # Ray misses circle

    sqrt_disc = math.sqrt(discriminant)
    t1 = b - sqrt_disc
    t2 = b + sqrt_disc

    if t1 > 0.0:
        return t1
    elif t2 > 0.0:
        return t2
    return None


class TestGap2BoundaryStraddling(unittest.TestCase):
    def setUp(self):
        self.handler = ObservationHandler26D(
            invalid_policy=InvalidClearancePolicy.CONSERVATIVE_ZERO,
            min_range=0.10,
            max_range=10.0,
            default_clear_range=10.0
        )
        self.num_rays = 72
        self.angle_min = -math.pi
        self.angle_inc = 2.0 * math.pi / 72.0  # 5.0 deg

    def _ray_angles(self) -> List[float]:
        return [self.angle_min + i * self.angle_inc for i in range(self.num_rays)]

    def _compute_scan_for_cylinder(
        self,
        center_dist: float,
        azimuth_deg: float,
        radius: float
    ) -> Tuple[List[float], Dict[int, float]]:
        """Generates 72-ray range array and dict of ray_index -> hit_distance for a cylinder."""
        az_rad = math.radians(azimuth_deg)
        cx = center_dist * math.cos(az_rad)
        cy = center_dist * math.sin(az_rad)

        ranges = []
        hits = {}
        for i, ang in enumerate(self._ray_angles()):
            t = ray_circle_intersection((0.0, 0.0), ang, (cx, cy), radius)
            if t is not None and 0.10 <= t < 10.0:
                ranges.append(t)
                hits[i] = t
            else:
                ranges.append(float('inf'))
        return ranges, hits

    def test_2_1_symmetric_boundary_straddle_22_5_deg(self):
        """
        Case 2.1: Cylinder centered exactly on +22.5° boundary between Sector 0 (Front) and Sector 1 (Front-Left).
        D = 2.50 m, R = 0.35 m.
        Angular half-width: arcsin(0.35 / 2.50) = 8.049°.
        Spans azimuths [14.45°, 30.55°].
        Affected rays:
          - Sector 0: Ray 39 (+15°), Ray 40 (+20°)
          - Sector 1: Ray 41 (+25°), Ray 42 (+30°)
        """
        center_dist = 2.50
        azimuth_deg = 22.5
        radius = 0.35

        ranges, hits = self._compute_scan_for_cylinder(center_dist, azimuth_deg, radius)

        # Verify exactly rays 39, 40, 41, 42 hit
        self.assertEqual(sorted(hits.keys()), [39, 40, 41, 42])

        # Verify analytical symmetry
        # Ray 40 (+20°) and Ray 41 (+25°) are 2.5° from center
        self.assertAlmostEqual(hits[40], hits[41], places=5)
        # Ray 39 (+15°) and Ray 42 (+30°) are 7.5° from center
        self.assertAlmostEqual(hits[39], hits[42], places=5)

        # Expected distance for Ray 40/41 (2.5° offset):
        # b = 2.5 * cos(2.5°) = 2.497621
        # disc = 0.35^2 - (2.5 * sin(2.5°))^2 = 0.1225 - 0.0118916 = 0.1106084
        # t = 2.497621 - sqrt(0.1106084) = 2.165042 m
        expected_min_clearance = center_dist * math.cos(math.radians(2.5)) - math.sqrt(
            radius**2 - (center_dist * math.sin(math.radians(2.5)))**2
        )
        self.assertAlmostEqual(hits[40], expected_min_clearance, places=5)

        # Process with ObservationHandler26D
        clearances, meta = self.handler.process_raw_scan(
            ranges=ranges,
            angle_min=self.angle_min,
            angle_increment=self.angle_inc
        )

        # Sector 0 (Front) minimum clearance
        self.assertAlmostEqual(clearances[0], expected_min_clearance, places=4)
        self.assertEqual(meta.sector_status[0], SectorStatus.VALID_HIT)

        # Sector 1 (Front-Left) minimum clearance
        self.assertAlmostEqual(clearances[1], expected_min_clearance, places=4)
        self.assertEqual(meta.sector_status[1], SectorStatus.VALID_HIT)

        # Other sectors must be clear
        for sec in range(2, 8):
            self.assertEqual(clearances[sec], 10.0)
            self.assertEqual(meta.sector_status[sec], SectorStatus.MEASURED_CLEAR)

    def test_2_2_asymmetric_boundary_straddle(self):
        """
        Case 2.2: Cylinder centered at +21.0° (1.5° inside Sector 0 from +22.5° boundary).
        D = 2.50 m, R = 0.35 m.
        Angular half-width: 8.049° -> span [12.95°, 29.05°].
        Affected rays:
          - Ray 39 (+15°, Sector 0): delta = 6.0°
          - Ray 40 (+20°, Sector 0): delta = 1.0° (closest ray)
          - Ray 41 (+25°, Sector 1): delta = 4.0°
          - Ray 42 (+30°): MISS (> 29.05°)
        """
        center_dist = 2.50
        azimuth_deg = 21.0
        radius = 0.35

        ranges, hits = self._compute_scan_for_cylinder(center_dist, azimuth_deg, radius)

        # Verify rays 39, 40, 41 hit, and 42 misses
        self.assertEqual(sorted(hits.keys()), [39, 40, 41])

        # Sector 0 min ray is Ray 40 (1.0° offset)
        expected_sec0 = center_dist * math.cos(math.radians(1.0)) - math.sqrt(
            radius**2 - (center_dist * math.sin(math.radians(1.0)))**2
        )
        # Sector 1 min ray is Ray 41 (4.0° offset)
        expected_sec1 = center_dist * math.cos(math.radians(4.0)) - math.sqrt(
            radius**2 - (center_dist * math.sin(math.radians(4.0)))**2
        )

        clearances, meta = self.handler.process_raw_scan(
            ranges=ranges,
            angle_min=self.angle_min,
            angle_increment=self.angle_inc
        )

        self.assertAlmostEqual(clearances[0], expected_sec0, places=4)
        self.assertAlmostEqual(clearances[1], expected_sec1, places=4)
        self.assertLess(clearances[0], clearances[1])  # Sector 0 is closer
        self.assertEqual(meta.sector_status[0], SectorStatus.VALID_HIT)
        self.assertEqual(meta.sector_status[1], SectorStatus.VALID_HIT)

    def test_2_3_boundary_adjacent_single_sector_hit(self):
        """
        Case 2.3: Cylinder placed close to +22.5° boundary but completely inside Sector 1.
        Azimuth = +25.0°, D = 3.00 m, R = 0.10 m.
        Angular radius = arcsin(0.10 / 3.00) = 1.91° -> span [23.09°, 26.91°].
        Ray 40 (+20°) is outside span (misses).
        Ray 41 (+25°) is inside span (hits).
        Ray 42 (+30°) is outside span (misses).
        Sector 1 must report VALID_HIT at 2.90 m.
        Sector 0 must report MEASURED_CLEAR (10.0 m) without cross-talk leakage.
        """
        center_dist = 3.00
        azimuth_deg = 25.0
        radius = 0.10

        ranges, hits = self._compute_scan_for_cylinder(center_dist, azimuth_deg, radius)

        # Only Ray 41 should hit
        self.assertEqual(list(hits.keys()), [41])
        self.assertAlmostEqual(hits[41], 2.90, places=4)

        clearances, meta = self.handler.process_raw_scan(
            ranges=ranges,
            angle_min=self.angle_min,
            angle_increment=self.angle_inc
        )

        # Sector 1 detected obstacle
        self.assertAlmostEqual(clearances[1], 2.90, places=4)
        self.assertEqual(meta.sector_status[1], SectorStatus.VALID_HIT)

        # Sector 0 MUST NOT report an obstacle
        self.assertEqual(clearances[0], 10.0)
        self.assertEqual(meta.sector_status[0], SectorStatus.MEASURED_CLEAR)

    def test_2_4_wraparound_boundary_straddle_180_deg(self):
        """
        Case 2.4: Cylinder centered at 180° (-180°) straddling the array wraparound boundary.
        D = 3.00 m, R = 0.40 m.
        Angular radius = arcsin(0.40 / 3.00) = 7.66°.
        Span wraps across -180° / +180°: [-180°, -172.34°] and [+172.34°, +180°].
        Affected rays:
          - Ray 0 (-180°): center hit at 2.60 m
          - Ray 1 (-175°): hit at analytical distance
          - Ray 71 (+175°): hit at analytical distance
        All three rays belong to Sector 4 (Back).
        Verify Sector 4 minimum clearance is exactly 2.60 m across the wraparound index seam.
        """
        center_dist = 3.00
        azimuth_deg = 180.0
        radius = 0.40

        ranges, hits = self._compute_scan_for_cylinder(center_dist, azimuth_deg, radius)

        # Rays 0, 1, 71 must hit
        self.assertIn(0, hits)
        self.assertIn(1, hits)
        self.assertIn(71, hits)
        self.assertAlmostEqual(hits[0], 2.60, places=4)
        self.assertAlmostEqual(hits[1], hits[71], places=5)

        clearances, meta = self.handler.process_raw_scan(
            ranges=ranges,
            angle_min=self.angle_min,
            angle_increment=self.angle_inc
        )

        # Sector 4 (Back) aggregates rays 0, 1, 71 to 2.60 m
        self.assertAlmostEqual(clearances[4], 2.60, places=4)
        self.assertEqual(meta.sector_status[4], SectorStatus.VALID_HIT)

        # All other sectors must be clear
        for sec in [0, 1, 2, 3, 5, 6, 7]:
            self.assertEqual(clearances[sec], 10.0)
            self.assertEqual(meta.sector_status[sec], SectorStatus.MEASURED_CLEAR)

    def test_2_5_all_sector_boundaries_straddling(self):
        """
        Case 2.5: Systematic evaluation across all remaining sector boundary angles:
          - 67.5°  (between Sector 1 and 2)
          - 112.5° (between Sector 2 and 3)
          - 157.5° (between Sector 3 and 4)
          - -157.5° (between Sector 4 and 5)
          - -112.5° (between Sector 5 and 6)
          - -67.5°  (between Sector 6 and 7)
          - -22.5°  (between Sector 7 and 0)
        For each boundary, place cylinder of D=2.50m, R=0.35m.
        Verify both adjacent sectors report VALID_HIT with clearances matching analytical values.
        """
        boundary_cases = [
            (67.5, (1, 2)),
            (112.5, (2, 3)),
            (157.5, (3, 4)),
            (-157.5, (4, 5)),
            (-112.5, (5, 6)),
            (-67.5, (6, 7)),
            (-22.5, (7, 0))
        ]

        center_dist = 2.50
        radius = 0.35
        expected_min = center_dist * math.cos(math.radians(2.5)) - math.sqrt(
            radius**2 - (center_dist * math.sin(math.radians(2.5)))**2
        )

        for b_deg, (sec_a, sec_b) in boundary_cases:
            with self.subTest(boundary_deg=b_deg, sectors=(sec_a, sec_b)):
                ranges, hits = self._compute_scan_for_cylinder(center_dist, b_deg, radius)

                # Exactly 4 rays hit (2 on each side of boundary)
                self.assertEqual(len(hits), 4)

                clearances, meta = self.handler.process_raw_scan(
                    ranges=ranges,
                    angle_min=self.angle_min,
                    angle_increment=self.angle_inc
                )

                # Both adjacent sectors must match analytical minimum
                self.assertAlmostEqual(clearances[sec_a], expected_min, places=4)
                self.assertAlmostEqual(clearances[sec_b], expected_min, places=4)
                self.assertEqual(meta.sector_status[sec_a], SectorStatus.VALID_HIT)
                self.assertEqual(meta.sector_status[sec_b], SectorStatus.VALID_HIT)


if __name__ == "__main__":
    unittest.main()
