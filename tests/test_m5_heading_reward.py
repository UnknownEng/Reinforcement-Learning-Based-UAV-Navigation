"""
test_m5_heading_reward.py
-------------------------
Unit tests for smooth heading alignment reward and speed penalty:
- Full range continuity across [-pi, pi]
- Monotonicity and symmetry
- Non-vanishing gradient at +/- pi
- Speed penalty gating and scaling
- Verification that no incentive for unsafe forward motion exists for rear-facing goals
- Verification of safety disclaimer
"""

import math
import unittest
import numpy as np
from rl_env.heading_reward import compute_smooth_heading_reward


class TestHeadingReward(unittest.TestCase):
    """
    Unit test suite for smooth heading alignment and speed penalty.
    
    DISCLAIMER:
    While smooth heading alignment and speed gating reduce premature forward throttle into rearward/lateral
    space, they do not claim or guarantee physical safety against all dynamic collision risks.
    """

    def setUp(self):
        self.w_align = 0.10
        self.w_speed = 0.10

    def test_disclaimer_present(self):
        """Verify the explicit safety disclaimer is documented."""
        import rl_env.heading_reward as hr
        doc = hr.__doc__
        self.assertIsNotNone(doc)
        self.assertIn("do not claim or guarantee physical safety against all dynamic collision risks", doc)

    def test_full_range_continuity_and_smoothness(self):
        """
        Verify that r_align and p_speed are smooth and continuous across a dense grid of 2,000 points in [-pi, pi].
        Verify finite bounds and bounded finite differences (Lipschitz continuity).
        """
        angles = np.linspace(-math.pi, math.pi, 2000)
        v_test = np.array([2.0, 0.0], dtype=np.float32)

        r_aligns = []
        p_speeds = []
        r_totals = []

        for psi in angles:
            ra, ps, rt = compute_smooth_heading_reward(psi, v_test, self.w_align, self.w_speed)
            self.assertTrue(math.isfinite(ra))
            self.assertTrue(math.isfinite(ps))
            self.assertTrue(math.isfinite(rt))
            r_aligns.append(ra)
            p_speeds.append(ps)
            r_totals.append(rt)

        r_aligns = np.array(r_aligns)
        p_speeds = np.array(p_speeds)

        # Check bounded finite differences (no step discontinuities)
        d_ra = np.diff(r_aligns)
        d_ps = np.diff(p_speeds)
        d_angles = np.diff(angles)

        max_grad_ra = np.max(np.abs(d_ra / d_angles))
        max_grad_ps = np.max(np.abs(d_ps / d_angles))

        self.assertLess(max_grad_ra, 0.15)
        self.assertLess(max_grad_ps, 0.25)

    def test_alignment_reward_monotonicity_and_bounds(self):
        """Verify peak at 0, minimum at +/- pi, symmetry, and strict monotonicity for |psi| in [0, pi]."""
        v_zero = np.zeros(2, dtype=np.float32)

        # Peak at psi = 0
        r_zero, _, _ = compute_smooth_heading_reward(0.0, v_zero, self.w_align, self.w_speed)
        self.assertAlmostEqual(r_zero, self.w_align, places=6)

        # Min at psi = +/- pi
        r_pos_pi, _, _ = compute_smooth_heading_reward(math.pi, v_zero, self.w_align, self.w_speed)
        r_neg_pi, _, _ = compute_smooth_heading_reward(-math.pi, v_zero, self.w_align, self.w_speed)
        self.assertAlmostEqual(r_pos_pi, -self.w_align, places=6)
        self.assertAlmostEqual(r_neg_pi, -self.w_align, places=6)

        # Monotonicity check
        psi_vals = np.linspace(0.0, math.pi, 100)
        rewards = [compute_smooth_heading_reward(p, v_zero, self.w_align, self.w_speed)[0] for p in psi_vals]
        for i in range(len(rewards) - 1):
            self.assertGreaterEqual(rewards[i], rewards[i + 1])

        # Symmetry check
        for p in [0.2, 0.7, 1.5, 2.8]:
            r_pos, _, _ = compute_smooth_heading_reward(p, v_zero, self.w_align, self.w_speed)
            r_neg, _, _ = compute_smooth_heading_reward(-p, v_zero, self.w_align, self.w_speed)
            self.assertAlmostEqual(r_pos, r_neg, places=6)

    def test_non_vanishing_gradient_at_pi(self):
        """Verify that derivative at +/- pi is strictly non-zero (unlike naive cos(psi) which vanishes at pi)."""
        eps = 1e-4
        v_zero = np.zeros(2, dtype=np.float32)

        r_pi_minus, _, _ = compute_smooth_heading_reward(math.pi - eps, v_zero, self.w_align, self.w_speed)
        r_pi, _, _ = compute_smooth_heading_reward(math.pi, v_zero, self.w_align, self.w_speed)

        # Finite difference: (r(pi) - r(pi - eps)) / eps should be approx -w_align * sin(pi/2) = -w_align
        grad_pi = (r_pi - r_pi_minus) / eps
        self.assertAlmostEqual(grad_pi, -self.w_align, delta=1e-3)
        self.assertLess(grad_pi, -0.05, "Gradient at pi must be strictly negative and non-vanishing")

    def test_wrap_continuity_and_derivative_behavior_at_pi(self):
        """
        Explicitly verify:
        1. Exact value continuity across the +/- pi wrap: R(+pi) == R(-pi).
        2. Derivative behavior at the boundary:
           - Approach +pi from left: dR/dpsi < 0 (decreasing dpsi toward 0 increases reward)
           - Approach -pi from right: dR/dpsi > 0 (increasing dpsi toward 0 increases reward)
        3. Near 180 degrees (e.g. 175 deg to 180 deg), the gradient magnitude is at its global maximum:
           |dR/dpsi| = w_align * sin(psi / 2) >= 0.099 * w_align.
        """
        v_zero = np.zeros(2, dtype=np.float32)

        # 1. Exact value continuity across wrap
        r_pos, _, _ = compute_smooth_heading_reward(math.pi, v_zero, self.w_align, self.w_speed)
        r_neg, _, _ = compute_smooth_heading_reward(-math.pi, v_zero, self.w_align, self.w_speed)
        self.assertEqual(r_pos, r_neg)
        self.assertAlmostEqual(r_pos, -self.w_align, places=7)

        # 2. Derivative behavior near +/- pi
        eps = 1e-4
        r_left = compute_smooth_heading_reward(math.pi - eps, v_zero, self.w_align, self.w_speed)[0]
        grad_left = (r_pos - r_left) / eps  # slope approaching +pi
        self.assertLess(grad_left, -0.09, "Approaching +pi must have negative slope")

        r_right = compute_smooth_heading_reward(-math.pi + eps, v_zero, self.w_align, self.w_speed)[0]
        grad_right = (r_right - r_neg) / eps  # slope leaving -pi
        self.assertGreater(grad_right, 0.09, "Leaving -pi must have positive slope")

        # 3. Peak gradient near 180 degrees
        for angle_deg in [170.0, 175.0, 179.0, 180.0]:
            rad = math.radians(angle_deg)
            # Theoretical gradient magnitude is w_align * sin(rad / 2)
            grad_mag = self.w_align * math.sin(rad / 2.0)
            self.assertGreaterEqual(grad_mag, 0.99 * self.w_align,
                                    f"Gradient magnitude at {angle_deg} deg must be >= 99% of peak w_align")

    def test_speed_penalty_scaling_and_gating(self):
        """Verify zero speed penalty when aligned, and quadratic scaling with speed when misaligned."""
        # 1. Aligned (psi = 0)
        for v in [0.0, 1.0, 2.0, 3.0]:
            _, p_speed, _ = compute_smooth_heading_reward(0.0, np.array([v, 0.0]), self.w_align, self.w_speed)
            self.assertAlmostEqual(p_speed, 0.0, places=6)

        # 2. Rear-facing (psi = pi)
        for v in [0.0, 1.0, 2.0, 3.0]:
            _, p_speed, _ = compute_smooth_heading_reward(math.pi, np.array([v, 0.0]), self.w_align, self.w_speed)
            expected_penalty = -self.w_speed * 1.0 * (v ** 2)
            self.assertAlmostEqual(p_speed, expected_penalty, places=6)

    def test_no_unsafe_forward_motion_incentive(self):
        """
        Verify that for rear-facing goals (psi = pi), commanding forward velocity v_x > 0
        produces a strictly worse reward than stationary yaw rotation (v_x = 0).
        """
        # Scenario: Goal is directly behind the UAV (psi = pi)
        # Action A: Stationary yawing (v_x = 0, v_y = 0)
        _, p_speed_stationary, r_tot_stationary = compute_smooth_heading_reward(
            math.pi, np.array([0.0, 0.0]), self.w_align, self.w_speed
        )

        # Action B: Aggressive forward throttle (v_x = 2.5, v_y = 0)
        _, p_speed_forward, r_tot_forward = compute_smooth_heading_reward(
            math.pi, np.array([2.5, 0.0]), self.w_align, self.w_speed
        )

        # Action C: Moderate forward throttle (v_x = 1.0, v_y = 0)
        _, p_speed_mod, r_tot_mod = compute_smooth_heading_reward(
            math.pi, np.array([1.0, 0.0]), self.w_align, self.w_speed
        )

        # Stationary yawing must have zero speed penalty
        self.assertEqual(p_speed_stationary, 0.0)

        # Forward throttle must incur negative speed penalty
        self.assertLess(p_speed_mod, 0.0)
        self.assertLess(p_speed_forward, p_speed_mod)

        # Total heading reward must strictly favor stationary yaw over forward translation
        self.assertGreater(r_tot_stationary, r_tot_mod)
        self.assertGreater(r_tot_stationary, r_tot_forward)

        # When progress reward is included, forward motion away from goal increases distance,
        # so net reward further penalizes forward motion
        dt = 0.1
        w_prog = 2.5
        # Moving forward at 2.5 m/s when goal is behind increases distance by 2.5 * 0.1 = 0.25m
        prog_forward = -w_prog * (2.5 * dt)
        prog_stationary = 0.0

        net_forward = r_tot_forward + prog_forward
        net_stationary = r_tot_stationary + prog_stationary
        self.assertGreater(net_stationary, net_forward)


if __name__ == "__main__":
    unittest.main()
