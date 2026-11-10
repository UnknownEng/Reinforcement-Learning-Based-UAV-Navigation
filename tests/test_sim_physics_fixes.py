"""
test_sim_physics_fixes.py
-------------------------
Regression tests for the simulator physics corrections:
  - tilt direction matches the thrust needed for the acceleration
  - velocity commands are in the heading frame (yaw only), so tilt cannot cause climb
  - body_rates are true body angular rates (consistent with the quaternion motion)
  - horizontal speed / acceleration limits apply to the combined vx, vy
  - hard touchdowns are detected as ground crashes
  - observation quaternions are sign-canonical (qw >= 0)
"""

import math

import numpy as np

from rl_env.action import ActionHandler, CommandedVelocities
from rl_env.config import UAVEnvConfig
from rl_env.dynamics import FastUAVDynamics
from rl_env.observation import (
    ObservationHandler,
    canonical_unit_quaternion,
    euler_to_quaternion,
    quat_to_rotation_matrix,
)
from rl_env.observation_26d import ObservationHandler26D
from rl_env.termination import TerminationChecker


def _dyn(z0=5.0, yaw0=0.0):
    dyn = FastUAVDynamics(UAVEnvConfig(enable_noise=False))
    dyn.reset(x0=0.0, y0=0.0, z0=z0, yaw0=yaw0)
    return dyn


def _thrust_axis(state):
    return quat_to_rotation_matrix(state.quat) @ np.array([0.0, 0.0, 1.0])


def test_left_acceleration_tilts_thrust_left():
    dyn = _dyn()
    for _ in range(3):
        state = dyn.step(CommandedVelocities(vx=0.0, vy=3.0, vz=0.0, yaw_rate=0.0))
    assert state.roll < 0.0                 # right side down
    assert _thrust_axis(state)[1] > 0.05    # thrust leans toward +y (left)


def test_tilt_follows_acceleration_direction_at_any_heading():
    for yaw0 in (0.0, 1.0, -2.0, 3.0):
        dyn = _dyn(yaw0=yaw0)
        prev_vel = dyn.get_state().vel_world.copy()
        state = dyn.step(CommandedVelocities(vx=3.0, vy=-1.0, vz=0.0, yaw_rate=0.0))
        accel_xy = (state.vel_world - prev_vel)[:2]
        thrust_xy = _thrust_axis(state)[:2]
        cos_angle = float(np.dot(accel_xy, thrust_xy) / (np.linalg.norm(accel_xy) * np.linalg.norm(thrust_xy)))
        assert cos_angle > 0.99, f"yaw0={yaw0}: thrust tilt not aligned with acceleration"


def test_forward_command_does_not_climb():
    # Previously the command was rotated by the full tilted attitude, so a pure
    # forward command produced ~0.9 m/s of climb.
    dyn = _dyn(z0=5.0)
    for _ in range(12):
        state = dyn.step(CommandedVelocities(vx=3.0, vy=0.0, vz=0.0, yaw_rate=0.0))
    assert abs(state.pos_world[2] - 5.0) < 0.02
    assert abs(state.vel_world[2]) < 0.02


def test_command_is_relative_to_heading():
    dyn = _dyn(yaw0=math.pi / 2.0)  # facing world +y (north)
    for _ in range(20):
        state = dyn.step(CommandedVelocities(vx=2.0, vy=0.0, vz=0.0, yaw_rate=0.0))
    assert abs(state.vel_world[0]) < 0.05
    assert abs(state.vel_world[1] - 2.0) < 0.05


def test_body_rates_match_quaternion_motion():
    # Combined yaw rotation + acceleration: body rates must equal the angular
    # velocity implied by consecutive attitudes, omega = vee(log(R_k^T R_k+1)) / dt.
    dyn = _dyn()
    cfg = dyn.config
    cmd = CommandedVelocities(vx=3.0, vy=1.5, vz=0.5, yaw_rate=0.8)
    for _ in range(3):
        dyn.step(cmd)
    for _ in range(5):
        R0 = quat_to_rotation_matrix(dyn.state.quat).astype(np.float64)
        dyn._substep(cmd, cfg.dt_substep)
        R1 = quat_to_rotation_matrix(dyn.state.quat).astype(np.float64)
        dR = R0.T @ R1
        angle = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1.0) / 2.0)))
        axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
        omega = axis * (angle / (2.0 * math.sin(angle))) / cfg.dt_substep if angle > 1e-9 else np.zeros(3)
        np.testing.assert_allclose(dyn.state.body_rates, omega, atol=0.03)


def test_diagonal_speed_limited_to_max_horizontal_velocity():
    handler = ActionHandler(UAVEnvConfig())
    cmd = handler.scale_action(np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32))
    assert math.isclose(math.hypot(cmd.vx, cmd.vy), 3.0, rel_tol=1e-5)
    # Direction is preserved, and in-limit commands are untouched.
    assert math.isclose(cmd.vx, cmd.vy, rel_tol=1e-6)
    small = handler.scale_action(np.array([0.3, -0.4, 0.5, 0.2], dtype=np.float32))
    assert math.isclose(small.vx, 0.9, rel_tol=1e-5) and math.isclose(small.vy, -1.2, rel_tol=1e-5)


def test_diagonal_acceleration_norm_limited():
    dyn = _dyn()
    prev = dyn.get_state().vel_world.copy()
    dyn._substep(CommandedVelocities(vx=3.0, vy=3.0, vz=0.0, yaw_rate=0.0), 0.01)
    accel = (dyn.state.vel_world - prev)[:2] / 0.01
    assert np.linalg.norm(accel) <= dyn.config.max_horizontal_accel + 1e-3


def test_hard_touchdown_is_ground_crash_soft_is_not():
    config = UAVEnvConfig(enable_noise=False)
    checker = TerminationChecker(config)

    dyn = _dyn(z0=3.0)
    status = None
    for step in range(60):
        state = dyn.step(CommandedVelocities(vx=0.0, vy=0.0, vz=-1.5, yaw_rate=0.0))
        status = checker.check(state=state, dist_to_goal=10.0, step_count=step + 1)
        if status.terminated:
            break
    assert status.reason == "ground_crash" and status.is_crashed

    # Resting on the ground with a descent command must not count as a crash.
    dyn = _dyn(z0=0.0)
    for step in range(20):
        state = dyn.step(CommandedVelocities(vx=0.0, vy=0.0, vz=-1.5, yaw_rate=0.0))
        status = checker.check(state=state, dist_to_goal=10.0, step_count=step + 1)
        assert status.reason != "ground_crash"


def test_horizontal_acceleration_requires_tilt():
    # Causality (M1): thrust acts along the actual body z-axis, so a vehicle whose
    # attitude cannot change cannot accelerate sideways.
    dyn = _dyn(z0=5.0)
    dyn.tau_attitude = 1e6
    for _ in range(10):
        state = dyn.step(CommandedVelocities(vx=3.0, vy=0.0, vz=0.0, yaw_rate=0.0))
    assert np.linalg.norm(state.vel_world[:2]) < 0.01


def test_step_response_matches_px4_x500():
    # PX4 v1.15 x500 SITL ground truth (logs/px4_step_response.json): a 3 m/s forward
    # step is within 10 % after ~1.24 s with ~24 deg peak pitch.
    dyn = _dyn(z0=5.0)
    t, t10, peak_pitch = 0.0, None, 0.0
    for _ in range(40):
        state = dyn.step(CommandedVelocities(vx=3.0, vy=0.0, vz=0.0, yaw_rate=0.0))
        t += 0.1
        peak_pitch = max(peak_pitch, math.degrees(state.pitch))
        if t10 is None and abs(state.vel_world[0] - 3.0) < 0.3:
            t10 = t
    assert 1.0 <= t10 <= 1.6, t10
    assert 20.0 <= peak_pitch <= 30.0, peak_pitch


def test_quaternion_sign_is_canonical_in_observations():
    q = euler_to_quaternion(0.05, -0.1, -2.5)
    q_neg = -q
    np.testing.assert_allclose(canonical_unit_quaternion(q_neg), canonical_unit_quaternion(q), atol=1e-6)
    assert canonical_unit_quaternion(q_neg)[0] >= 0.0

    args = dict(
        pos_world=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        vel_world=np.array([0.5, -0.2, 0.1], dtype=np.float32),
        rates_body=np.zeros(3, dtype=np.float32),
        goal_world=np.array([4.0, -1.0, 2.0], dtype=np.float32),
        prev_action=np.zeros(4, dtype=np.float32),
    )
    h18 = ObservationHandler(UAVEnvConfig())
    np.testing.assert_allclose(h18.compute_observation(quat=q_neg, **args),
                               h18.compute_observation(quat=q, **args), atol=1e-6)

    h26 = ObservationHandler26D(UAVEnvConfig())
    o_pos, _ = h26.compute_observation_with_metadata(quat=q, clearances=[5.0] * 8, **args)
    o_neg, _ = h26.compute_observation_with_metadata(quat=q_neg, clearances=[5.0] * 8, **args)
    np.testing.assert_allclose(o_neg, o_pos, atol=1e-6)
