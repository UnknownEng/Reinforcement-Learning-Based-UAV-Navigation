"""
test_m5_geometry_validator.py
-----------------------------
Unit tests for RouteGeometryValidator:
- Narrow passages
- Blocked routes
- Boundary clearance
- Resampling exhaustion
- Dynamic flight feasibility disclaimer presence
"""

import unittest
import numpy as np
from rl_env.geometry_validator import RouteGeometryValidator, ValidationResult


class TestRouteGeometryValidator(unittest.TestCase):
    """
    Test suite for 2D BFS RouteGeometryValidator.
    
    DISCLAIMER:
    2D connectivity does not establish dynamic flight feasibility. While 2D BFS in inflated free-space
    guarantees the geometric existence of a collision-free corridor of width >= 2*(r_uav + margin),
    actual flight success depends on UAV acceleration limits, actuation lag, velocity saturation,
    attitude tilt limits, and closed-loop control tracking error.
    """

    def setUp(self):
        self.validator = RouteGeometryValidator(
            arena_x=(-10.0, 10.0),
            arena_y=(-10.0, 10.0),
            uav_radius=0.35,
            safety_margin=0.15,  # r_inflate = 0.50 m
            grid_resolution=0.10,
            min_path_clearance=0.50,
            max_resample_attempts=10  # shorter for test
        )

    def test_disclaimer_present(self):
        """Verify the explicit disclaimer is documented on the validator class."""
        doc = RouteGeometryValidator.__doc__
        self.assertIsNotNone(doc)
        self.assertIn("2D connectivity does not establish dynamic flight feasibility", doc)

    def test_narrow_passage(self):
        """
        Verify that a passage narrower than 2*r_inflate (1.00m) is marked impassable,
        whereas a wider passage (1.40m) is traversable.
        """
        start = np.array([0.0, -3.0], dtype=np.float32)
        goal = np.array([0.0, 3.0], dtype=np.float32)

        # Narrow passage: Two cylinders at x = -0.75 and x = +0.75, radius = 0.40m each
        # Center-to-center = 1.50m. Surface gap = 1.50 - 2*0.40 = 0.70m (< 1.00m required for 2*0.50m)
        narrow_obstacles = [
            (-0.75, 0.0, 0.40),
            (0.75, 0.0, 0.40)
        ]
        res_narrow = self.validator.validate_route(start, goal, narrow_obstacles)
        # Note: If arena is wide, BFS could go around. So let's block the flanks too, or check through-gap.
        # Let's add barrier walls on flanks to force passage through center
        flank_obstacles = [
            (-3.0, 0.0, 1.5),
            (3.0, 0.0, 1.5),
            (-7.0, 0.0, 2.5),
            (7.0, 0.0, 2.5),
        ]
        res_chokepoint_narrow = self.validator.validate_route(start, goal, narrow_obstacles + flank_obstacles)
        self.assertFalse(res_chokepoint_narrow.is_traversable,
                         f"Narrow gap of 0.70m should be impassable! Reason: {res_chokepoint_narrow.reason}")

        # Wide passage: gap = 1.40m (centers at -1.10 and +1.10, radius 0.40m -> surface gap = 2.20 - 0.80 = 1.40m)
        wide_obstacles = [
            (-1.10, 0.0, 0.40),
            (1.10, 0.0, 0.40)
        ] + flank_obstacles
        res_wide = self.validator.validate_route(start, goal, wide_obstacles)
        self.assertTrue(res_wide.is_traversable,
                        f"Wide gap of 1.40m should be traversable! Reason: {res_wide.reason}")
        self.assertGreaterEqual(res_wide.min_route_clearance, 0.50)

    def test_blocked_route(self):
        """Verify that a complete wall of obstacles spanning the arena returns is_traversable=False."""
        start = np.array([0.0, -4.0], dtype=np.float32)
        goal = np.array([0.0, 4.0], dtype=np.float32)

        # Continuous wall of overlapping cylinders at y = 0.0 spanning from x = -10 to +10
        wall = [(float(x), 0.0, 0.60) for x in np.arange(-9.0, 9.5, 1.0)]

        res = self.validator.validate_route(start, goal, wall)
        self.assertFalse(res.is_traversable)
        self.assertEqual(res.reason, "no_connected_route")

    def test_boundary_clearance(self):
        """
        Verify that an obstacle placed adjacent to the arena boundary wall
        leaving an impassable perimeter gap is properly rejected.
        """
        start = np.array([8.0, -3.0], dtype=np.float32)
        goal = np.array([8.0, 3.0], dtype=np.float32)

        # Arena wall at x = 10.0.
        # Place cylinder at x = 9.30, radius = 0.40m.
        # Distance to boundary = 10.0 - 9.30 - 0.40 = 0.30m (< r_inflate = 0.50m)
        boundary_obstacle = [(9.30, 0.0, 0.40)]

        # Also block inner passage so UAV must try boundary gap
        inner_wall = [(float(x), 0.0, 0.60) for x in np.arange(-9.0, 8.5, 1.0)]
        all_obs = boundary_obstacle + inner_wall

        res = self.validator.validate_route(start, goal, all_obs)
        self.assertFalse(res.is_traversable,
                         "Perimeter passage with 0.30m boundary clearance should be rejected")

    def test_resampling_exhaustion(self):
        """
        Verify that resampling exhaustion occurs and raises RuntimeError after max attempts,
        or returns failure when raise_on_exhaustion=False.
        """
        start = np.array([0.0, -3.0], dtype=np.float32)
        goal = np.array([0.0, 3.0], dtype=np.float32)

        # Pathological generator that always returns a solid blocking wall
        def pathological_generator():
            return [(float(x), 0.0, 0.60) for x in np.arange(-9.0, 9.5, 1.0)]

        # 1. With raise_on_exhaustion=True
        with self.assertRaises(RuntimeError) as ctx:
            self.validator.validate_and_generate_obstacles(
                generator_fn=pathological_generator,
                start_xy=start,
                goal_xy=goal,
                raise_on_exhaustion=True
            )
        self.assertIn("Resampling exhaustion", str(ctx.exception))
        self.assertIn(str(self.validator.max_resample_attempts), str(ctx.exception))

        # 2. With raise_on_exhaustion=False
        obs, res = self.validator.validate_and_generate_obstacles(
            generator_fn=pathological_generator,
            start_xy=start,
            goal_xy=goal,
            raise_on_exhaustion=False
        )
        self.assertFalse(res.is_traversable)
        self.assertEqual(res.attempts, self.validator.max_resample_attempts)

    def test_diagonal_corner_cutting_prevention(self):
        """
        Verify that 8-connected BFS does not cut diagonally across touching obstacle corners.
        Scenario: Two obstacles touch diagonally across orthogonal grid axes, forming a pinch point.
        Without corner cutting prevention, an 8-connected diagonal step would squeeze through the diagonal vertex.
        With corner cutting prevention, the route is recognized as impassable.
        """
        start = np.array([-2.0, -2.0], dtype=np.float32)
        goal = np.array([2.0, 2.0], dtype=np.float32)

        # Place obstacles whose inflated circles (radius + 0.50m) touch along the diagonal
        # Obstacle A: center at (-0.60, 0.60), radius = 0.35m -> inflated radius = 0.85m
        # Obstacle B: center at (0.60, -0.60), radius = 0.35m -> inflated radius = 0.85m
        # The center-to-center distance is sqrt(1.2^2 + (-1.2)^2) = sqrt(2.88) = 1.697m
        # Sum of inflated radii is 0.85 + 0.85 = 1.70m!
        # They touch exactly at (0, 0) diagonally!
        touching_obstacles = [
            (-0.60, 0.60, 0.35),
            (0.60, -0.60, 0.35)
        ]
        # Also flank barriers so the only attempted passage is the diagonal pinch point
        flanks = [
            (-3.0, 3.0, 2.0),
            (3.0, -3.0, 2.0),
            (-5.0, 5.0, 3.0),
            (5.0, -5.0, 3.0),
            (-8.0, 0.0, 3.0),
            (8.0, 0.0, 3.0),
            (0.0, -8.0, 3.0),
            (0.0, 8.0, 3.0),
            (-4.0, 0.0, 2.0),
            (4.0, 0.0, 2.0),
            (0.0, -4.0, 2.0),
            (0.0, 4.0, 2.0)
        ]
        res = self.validator.validate_route(start, goal, touching_obstacles + flanks)
        self.assertFalse(res.is_traversable,
                         "BFS should not cut diagonally across touching inflated obstacle corners")

    def test_segment_clearance_vs_node_clearance(self):
        """
        Verify that continuous segment evaluation accurately measures clearance along
        the line segment between waypoints, detecting obstacles that are closer to the
        segment midpoint than to either discrete waypoint.
        """
        # Validator with 0.20m resolution to space out waypoints
        val = RouteGeometryValidator(
            arena_x=(-10.0, 10.0),
            arena_y=(-10.0, 10.0),
            grid_resolution=0.20,
            min_path_clearance=0.50
        )
        start = np.array([0.0, -2.0], dtype=np.float32)
        goal = np.array([0.0, 2.0], dtype=np.float32)

        # Place cylinder at x = 0.95, y = 0.0 with radius 0.40m
        # Path runs along x = 0.0, so at y = 0.0, distance to center is 0.95m
        # Surface distance is 0.95 - 0.40 = 0.55m
        obs = [(0.95, 0.0, 0.40)]
        res = val.validate_route(start, goal, obs)
        self.assertTrue(res.is_traversable)
        self.assertAlmostEqual(res.min_route_clearance, 0.55, delta=0.05)

    def test_segment_boundary_clearance(self):
        """Verify that segments approaching arena boundaries maintain continuous clearance along the chord."""
        val = RouteGeometryValidator(
            arena_x=(-5.0, 5.0),
            arena_y=(-5.0, 5.0),
            min_path_clearance=0.50
        )
        # Start near boundary
        start = np.array([4.4, -2.0], dtype=np.float32)
        goal = np.array([4.4, 2.0], dtype=np.float32)

        # Path runs at x = 4.4, boundary at x = 5.0 -> clearance is 5.0 - 4.4 = 0.60m >= 0.50m
        res = val.validate_route(start, goal, [])
        self.assertTrue(res.is_traversable)
        self.assertAlmostEqual(res.min_route_clearance, 0.60, delta=0.05)

        # Path runs at x = 4.6, boundary at x = 5.0 -> clearance is 5.0 - 4.6 = 0.40m < 0.50m
        start_tight = np.array([4.6, -2.0], dtype=np.float32)
        goal_tight = np.array([4.6, 2.0], dtype=np.float32)
        res_tight = val.validate_route(start_tight, goal_tight, [])
        self.assertFalse(res_tight.is_traversable)


if __name__ == "__main__":
    unittest.main()
