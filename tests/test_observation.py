"""
test_observation.py
-------------------
Unit tests for 18D observation space computation, coordinate transformations,
and numerical safety.
"""

import pytest
import numpy as np

from rl_env.config import UAVEnvConfig
from rl_env.observation import ObservationHandler, euler_to_quaternion, quat_to_rotation_matrix


def test_quat_to_rotation_matrix():
    # Identity quaternion [1, 0, 0, 0] must yield 3x3 identity matrix
    q_id = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    R_id = quat_to_rotation_matrix(q_id)
    np.testing.assert_allclose(R_id, np.eye(3), atol=1e-6)

    # 90 degree yaw rotation (+Z axis): ENU East (+X) rotates to Body -Y (or World North +Y becomes Body +X)
    # R_b^w maps body to world. If yaw = +90 deg, Body X (+forward) points North (+Y in world).
    q_yaw90 = euler_to_quaternion(0.0, 0.0, np.pi / 2.0)
    R_yaw90 = quat_to_rotation_matrix(q_yaw90)
    body_fwd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    world_fwd = R_yaw90 @ body_fwd
    np.testing.assert_allclose(world_fwd, [0.0, 1.0, 0.0], atol=1e-6)

    # Orthogonality: R @ R.T == I and det(R) == 1
    np.testing.assert_allclose(R_yaw90 @ R_yaw90.T, np.eye(3), atol=1e-6)
    assert np.isclose(np.linalg.det(R_yaw90), 1.0, atol=1e-6)


def test_euler_to_quaternion_roundtrip():
    roll, pitch, yaw = 0.1, -0.2, 0.7
    q = euler_to_quaternion(roll, pitch, yaw)
    assert np.isclose(np.linalg.norm(q), 1.0, atol=1e-6)


def test_observation_computation_and_dimension():
    config = UAVEnvConfig()
    obs_handler = ObservationHandler(config)

    pos_w = np.array([2.0, 3.0, 4.0], dtype=np.float32)
    vel_w = np.array([0.5, -0.5, 0.1], dtype=np.float32)
    q = euler_to_quaternion(0.0, 0.0, 0.0)  # zero rotation
    rates = np.array([0.01, -0.02, 0.05], dtype=np.float32)
    goal_w = np.array([5.0, 7.0, 6.0], dtype=np.float32)
    prev_act = np.array([0.2, -0.3, 0.4, -0.5], dtype=np.float32)

    obs = obs_handler.compute_observation(
        pos_world=pos_w,
        vel_world=vel_w,
        quat=q,
        rates_body=rates,
        goal_world=goal_w,
        prev_action=prev_act,
        ground_z=0.0
    )

    # Dimensions
    assert obs.shape == (18,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))

    # For zero rotation: goal_body == goal_world - pos_world
    expected_rel_goal = goal_w - pos_w
    np.testing.assert_allclose(obs[0:3], expected_rel_goal, atol=1e-6)

    # vel_body == vel_world
    np.testing.assert_allclose(obs[3:6], vel_w, atol=1e-6)

    # rates_body
    np.testing.assert_allclose(obs[6:9], rates, atol=1e-6)

    # quaternion
    np.testing.assert_allclose(obs[9:13], q, atol=1e-6)

    # relative altitude
    assert np.isclose(obs[13], 4.0, atol=1e-6)

    # previous action
    np.testing.assert_allclose(obs[14:18], prev_act, atol=1e-6)


def test_observation_nan_inf_safety():
    config = UAVEnvConfig()
    obs_handler = ObservationHandler(config)

    # Provide NaN / Inf inputs
    pos_w = np.array([np.nan, 3.0, np.inf], dtype=np.float32)
    vel_w = np.array([0.5, np.nan, 0.1], dtype=np.float32)
    q = np.array([np.nan, 0.0, 0.0, 0.0], dtype=np.float32)
    rates = np.zeros(3, dtype=np.float32)
    goal_w = np.zeros(3, dtype=np.float32)
    prev_act = np.zeros(4, dtype=np.float32)

    obs = obs_handler.compute_observation(
        pos_world=pos_w,
        vel_world=vel_w,
        quat=q,
        rates_body=rates,
        goal_world=goal_w,
        prev_action=prev_act,
        ground_z=0.0
    )

    assert obs.shape == (18,)
    assert np.all(np.isfinite(obs))
    assert obs_handler.observation_space.contains(obs)
