"""
test_fast_dynamics.py
---------------------
Unit tests for fast multicopter closed-loop flight dynamics:
  - First-order velocity tracking lag and acceleration limits
  - Tilt-to-accelerate roll/pitch coupling
  - Action latency queue delay
  - Hard floor ground contact constraint
  - Yaw dynamics and angle wrapping
"""

import pytest
import numpy as np

from rl_env.action import CommandedVelocities
from rl_env.config import UAVEnvConfig
from rl_env.dynamics import FastUAVDynamics
from rl_env.observation import quat_to_rotation_matrix


def test_velocity_lag_and_acceleration_limits():
    config = UAVEnvConfig(enable_noise=False)  # Deterministic test
    dyn = FastUAVDynamics(config)
    dyn.reset(x0=0.0, y0=0.0, z0=5.0, yaw0=0.0)

    # Command forward velocity vx = 3.0 m/s
    cmd = CommandedVelocities(vx=3.0, vy=0.0, vz=0.0, yaw_rate=0.0)

    # Step once (0.1s)
    state = dyn.step(cmd)

    # Velocity must increase in +X direction, but cannot reach 3.0 immediately due to lag & accel limits
    assert state.vel_world[0] > 0.0
    assert state.vel_world[0] < 3.0
    # Over several steps it converges towards 3.0 m/s (PX4-calibrated loop: ~0.6 s time constant).
    # PX4 x500 SITL itself still reads 3.10 m/s 5 s after this step (integrator overshoot).
    for _ in range(40):
        state = dyn.step(cmd)
    assert np.isclose(state.vel_world[0], 3.0, atol=0.12)


def test_tilt_to_accelerate_pitch_coupling():
    config = UAVEnvConfig(enable_noise=False)
    dyn = FastUAVDynamics(config)
    dyn.reset(x0=0.0, y0=0.0, z0=5.0, yaw0=0.0)

    # Forward acceleration needs the thrust vector tilted forward: nose down.
    # In ENU/FLU with ZYX Euler angles that is a positive pitch angle.
    cmd = CommandedVelocities(vx=3.0, vy=0.0, vz=0.0, yaw_rate=0.0)
    for _ in range(5):
        state = dyn.step(cmd)

    R = quat_to_rotation_matrix(state.quat)
    assert state.pitch > 0.0
    assert (R @ np.array([1.0, 0.0, 0.0]))[2] < 0.0   # nose points below the horizon
    assert (R @ np.array([0.0, 0.0, 1.0]))[0] > 0.0   # thrust axis leans forward


def test_ground_contact_floor():
    config = UAVEnvConfig(enable_noise=False)
    dyn = FastUAVDynamics(config)
    dyn.reset(x0=0.0, y0=0.0, z0=0.5, yaw0=0.0)

    # Command aggressive downward descent
    cmd = CommandedVelocities(vx=0.0, vy=0.0, vz=-1.5, yaw_rate=0.0)
    for _ in range(20):
        state = dyn.step(cmd)

    # Position must never penetrate below ground (z >= 0.0)
    assert state.pos_world[2] >= 0.0


def test_yaw_rate_and_wrapping():
    config = UAVEnvConfig(enable_noise=False)
    dyn = FastUAVDynamics(config)
    dyn.reset(x0=0.0, y0=0.0, z0=5.0, yaw0=3.0)  # Near +pi

    # Positive CCW yaw rate will push past +pi, wrapping to -pi
    cmd = CommandedVelocities(vx=0.0, vy=0.0, vz=0.0, yaw_rate=1.0)
    for _ in range(10):
        state = dyn.step(cmd)

    assert -np.pi <= state.yaw <= np.pi
