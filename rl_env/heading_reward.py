"""
heading_reward.py
-----------------
Smooth Heading Alignment Reward and Smooth Speed Penalty Engine for Fast UAV Navigation.

Formulation:
  Heading Error:
    Δψ = atan2(y_{g,b}, x_{g,b}) ∈ [-π, π]

  Smooth Heading Alignment Reward:
    R_align(Δψ) = w_align * (2 * cos(Δψ / 2) - 1) ∈ [-w_align, +w_align]

  Smooth Rearward/Misalignment Speed Penalty:
    P_speed(v_xy, Δψ) = -w_speed * sin²(Δψ / 2) * ||v_xy||²

  Total Heading Control Reward:
    R_head = R_align + P_speed

DISCLAIMER:
While smooth heading alignment and speed gating reduce premature forward throttle into rearward/lateral
space, they do not claim or guarantee physical safety against all dynamic collision risks.
"""

from dataclasses import dataclass
import math
from typing import Optional, Tuple
import numpy as np


@dataclass
class HeadingRewardConfig:
    """Configuration for smooth heading alignment and speed penalty."""
    reward_align_weight: float = 0.10
    reward_speed_gate_weight: float = 0.10
    reward_reverse_weight: float = 0.50


def compute_smooth_heading_reward(
    heading_error_rad: float,
    vel_body_xy: np.ndarray,
    weight_align: float = 0.10,
    weight_speed: float = 0.10,
    weight_reverse: float = 0.50,
    cmd_vel_xy: Optional[np.ndarray] = None
) -> Tuple[float, float, float]:
    """
    Computes smooth heading alignment reward, speed penalty, and directional reverse penalty.
    
    Args:
        heading_error_rad: Heading error Δψ in body frame [-π, π].
                           0 = goal directly ahead (+x_b).
                           ±π = goal directly behind (-x_b).
        vel_body_xy: Horizontal velocity in body frame [vx_b, vy_b] (m/s).
        cmd_vel_xy: Optional commanded velocity in body frame [cmd_vx, cmd_vy] (m/s).
                    When provided, eliminates the credit assignment delay from actuator lag
                    so reverse commands are immediately penalized at t=0 before physical motion begins.
        weight_align: Scaling coefficient for alignment reward.
        weight_speed: Scaling coefficient for symmetric speed penalty under misalignment.
        weight_reverse: Scaling coefficient for asymmetric reverse speed penalty (vx < 0).
        
    Returns:
        (r_align, p_speed, r_total)
    """
    # Wrap heading error to [-pi, pi]
    d_psi = float((heading_error_rad + math.pi) % (2.0 * math.pi) - math.pi)

    # 1. Smooth C^∞ Heading Alignment Reward: [-w_align, +w_align]
    # At d_psi = 0: 2*cos(0) - 1 = +1 -> +w_align
    # At d_psi = ±pi: 2*cos(±pi/2) - 1 = -1 -> -w_align
    # Derivative at ±pi is non-zero (-w_align * sin(±pi/2) = ∓w_align)
    r_align = float(weight_align * (2.0 * math.cos(d_psi / 2.0) - 1.0))

    # 2. Smooth C^∞ Speed Penalty Gating Function: [0, 1]
    # At d_psi = 0: sin^2(0) = 0 (zero speed penalty when aligned)
    # At d_psi = ±pi: sin^2(±pi/2) = 1 (maximum speed penalty when rear-facing)
    g_speed = float(math.sin(d_psi / 2.0) ** 2)

    vx = float(vel_body_xy[0])
    vy = float(vel_body_xy[1])
    speed_sq = vx * vx + vy * vy

    # Symmetric speed penalty under misalignment
    p_symmetric = float(-weight_speed * g_speed * speed_sq)

    # Asymmetric reverse speed penalty: active only when reversing (vx < 0) under misalignment.
    # Evaluates both physical velocity and commanded intent to eliminate latency blindspots.
    rev_speed = max(0.0, -vx)
    if cmd_vel_xy is not None:
        cmd_rev = max(0.0, -float(cmd_vel_xy[0]))
        rev_speed = max(rev_speed, cmd_rev)

    p_reverse = float(-weight_reverse * g_speed * (rev_speed ** 2))

    p_speed = float(p_symmetric + p_reverse)

    r_total = float(r_align + p_speed)
    return r_align, p_speed, r_total
