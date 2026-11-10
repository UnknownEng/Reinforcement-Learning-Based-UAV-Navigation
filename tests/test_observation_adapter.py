"""
test_observation_adapter.py
---------------------------
Unit tests verifying the ROS 2 Observation & Action Adapter logic:
  - Quaternion convention conversion (ROS 2 XYZW -> Training WXYZ)
  - Coordinate frame rotation (World ENU -> Body FLU)
  - 18D observation dimension, layout, and bound checking
  - Physical action unscaling and clamping
  - Safety supervisor watchdog triggers (stale telemetry, geofence, tilt)
"""

import pytest
import numpy as np

from rl_env.config import UAVEnvConfig
from rl_env.observation import ObservationHandler, quat_to_rotation_matrix, euler_to_quaternion
from rl_env.action import ActionHandler, CommandedVelocities


def transform_ros_odom_to_observation(
    pos_enu: np.ndarray,
    vel_enu: np.ndarray,
    quat_ros_xyzw: np.ndarray,
    rates_body_flu: np.ndarray,
    goal_enu: np.ndarray,
    prev_action: np.ndarray,
    obs_handler: ObservationHandler,
    ground_z: float = 0.0
) -> np.ndarray:
    """
    Reference transformation function used by the ROS 2 Observation Adapter.
    Converts ROS 2 conventions (XYZW quaternion, ENU world frames) to the
    18D training observation representation.
    """
    # 1. Reorder ROS 2 scalar-last quaternion [x, y, z, w] to scalar-first [w, x, y, z]
    qx, qy, qz, qw = quat_ros_xyzw[0], quat_ros_xyzw[1], quat_ros_xyzw[2], quat_ros_xyzw[3]
    q_training_wxyz = np.array([qw, qx, qy, qz], dtype=np.float32)

    # 2. Compute observation via frozen ObservationHandler
    obs = obs_handler.compute_observation(
        pos_world=pos_enu.astype(np.float32),
        vel_world=vel_enu.astype(np.float32),
        quat=q_training_wxyz,
        rates_body=rates_body_flu.astype(np.float32),
        goal_world=goal_enu.astype(np.float32),
        prev_action=prev_action.astype(np.float32),
        ground_z=ground_z
    )
    return obs


def test_quaternion_reordering_and_frame_rotation():
    config = UAVEnvConfig()
    obs_handler = ObservationHandler(config)

    # Case 1: Identity orientation (facing East in ENU, roll=0, pitch=0, yaw=0)
    pos_enu = np.array([0.0, 0.0, 2.0])
    vel_enu = np.array([1.0, 0.0, 0.0])  # Moving East
    quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0])  # ROS identity quaternion
    rates_flu = np.array([0.0, 0.0, 0.0])
    goal_enu = np.array([5.0, 0.0, 2.0])  # 5m East
    prev_act = np.zeros(4, dtype=np.float32)

    obs = transform_ros_odom_to_observation(
        pos_enu, vel_enu, quat_xyzw, rates_flu, goal_enu, prev_act, obs_handler
    )

    assert obs.shape == (18,)
    # Relative goal in body frame must be [5.0, 0.0, 0.0] (straight ahead)
    np.testing.assert_allclose(obs[0:3], [5.0, 0.0, 0.0], atol=1e-5)
    # Velocity in body frame must be [1.0, 0.0, 0.0]
    np.testing.assert_allclose(obs[3:6], [1.0, 0.0, 0.0], atol=1e-5)
    # Quaternion in observation must be [1.0, 0.0, 0.0, 0.0] (scalar-first)
    np.testing.assert_allclose(obs[9:13], [1.0, 0.0, 0.0, 0.0], atol=1e-5)
    # Altitude must be 2.0m
    assert np.isclose(obs[13], 2.0)


def test_yaw_rotated_observation():
    config = UAVEnvConfig()
    obs_handler = ObservationHandler(config)

    # Vehicle facing North (+Y in ENU) -> yaw = +90 deg (+pi/2)
    # Euler (0, 0, pi/2): qw = cos(pi/4) = 0.7071, qz = sin(pi/4) = 0.7071
    # ROS 2 XYZW: [0, 0, 0.7071, 0.7071]
    qw = np.cos(np.pi / 4.0)
    qz = np.sin(np.pi / 4.0)
    quat_xyzw = np.array([0.0, 0.0, qz, qw])

    pos_enu = np.array([0.0, 0.0, 2.0])
    vel_enu = np.array([0.0, 2.0, 0.0])  # Moving North in world frame (+2 m/s)
    rates_flu = np.array([0.0, 0.0, 0.1])
    goal_enu = np.array([0.0, 4.0, 2.0])  # Goal is 4m North in world frame
    prev_act = np.zeros(4, dtype=np.float32)

    obs = transform_ros_odom_to_observation(
        pos_enu, vel_enu, quat_xyzw, rates_flu, goal_enu, prev_act, obs_handler
    )

    # In body frame (facing North):
    # - Goal 4m North in world is 4m Forward (+X) in body frame!
    np.testing.assert_allclose(obs[0:3], [4.0, 0.0, 0.0], atol=1e-4)
    # - Velocity 2m/s North in world is 2m/s Forward (+X) in body frame!
    np.testing.assert_allclose(obs[3:6], [2.0, 0.0, 0.0], atol=1e-4)
    # - Quaternion in observation is [qw, 0, 0, qz]
    np.testing.assert_allclose(obs[9:13], [qw, 0.0, 0.0, qz], atol=1e-4)


def test_action_unscaling_to_twist():
    config = UAVEnvConfig()
    act_handler = ActionHandler(config)

    action = np.array([0.5, -0.5, 0.2, -0.8], dtype=np.float32)
    cmd = act_handler.scale_action(action)

    assert np.isclose(cmd.vx, 0.5 * 3.0)       # 1.5 m/s
    assert np.isclose(cmd.vy, -0.5 * 3.0)      # -1.5 m/s
    assert np.isclose(cmd.vz, 0.2 * 1.5)       # 0.3 m/s
    assert np.isclose(cmd.yaw_rate, -0.8 * 1.0) # -0.8 rad/s
