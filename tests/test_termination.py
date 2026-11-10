"""
test_termination.py
-------------------
Unit tests for termination and truncation logic:
  - Goal arrival and stabilization
  - Geofence boundary violation
  - Excessive tilt angle
  - Hard ground crash
  - Episode step timeout truncation
"""

import pytest
import numpy as np

from rl_env.config import UAVEnvConfig
from rl_env.dynamics import UAVState
from rl_env.observation import euler_to_quaternion
from rl_env.termination import TerminationChecker, TerminationStatus


def create_mock_state(x=0.0, y=0.0, z=2.0, vx=0.0, vy=0.0, vz=0.0, roll=0.0, pitch=0.0, yaw=0.0):
    q = euler_to_quaternion(roll, pitch, yaw)
    return UAVState(
        pos_world=np.array([x, y, z], dtype=np.float32),
        vel_world=np.array([vx, vy, vz], dtype=np.float32),
        roll=float(roll),
        pitch=float(pitch),
        yaw=float(yaw),
        body_rates=np.zeros(3, dtype=np.float32),
        quat=q
    )


def test_goal_success_stabilized():
    config = UAVEnvConfig()
    checker = TerminationChecker(config)

    # Within tolerance (0.2m <= 0.40m), low velocity (0.1m/s <= 0.50m/s), upright
    state = create_mock_state(x=5.0, y=5.0, z=2.0, vx=0.05, vy=-0.05, vz=0.0)
    status = checker.check(state=state, dist_to_goal=0.2, step_count=50)

    assert status.terminated is True
    assert status.truncated is False
    assert status.is_success is True
    assert status.reason == "goal_reached_and_stabilized"


def test_goal_near_but_unstable():
    config = UAVEnvConfig()
    checker = TerminationChecker(config)

    # Within distance tolerance (0.2m) but flying too fast (2.0 m/s > 0.50m/s)
    state = create_mock_state(x=5.0, y=5.0, z=2.0, vx=1.5, vy=1.0, vz=0.5)
    status = checker.check(state=state, dist_to_goal=0.2, step_count=50)

    assert status.terminated is False
    assert status.is_success is False


def test_geofence_violation():
    config = UAVEnvConfig()
    checker = TerminationChecker(config)

    # Outside arena X (+15.0m > +12.0m)
    state = create_mock_state(x=15.0, y=0.0, z=2.0)
    status = checker.check(state=state, dist_to_goal=5.0, step_count=20)

    assert status.terminated is True
    assert status.is_out_of_bounds is True
    assert status.reason == "geofence_violation"

    # Above maximum ceiling (+14.0m > +12.0m)
    state_high = create_mock_state(x=0.0, y=0.0, z=14.0)
    status_high = checker.check(state=state_high, dist_to_goal=5.0, step_count=20)

    assert status_high.terminated is True
    assert status_high.is_out_of_bounds is True


def test_excessive_tilt_failure():
    config = UAVEnvConfig()
    checker = TerminationChecker(config)

    # Excessive roll angle (70 deg > 60 deg max)
    state = create_mock_state(x=0.0, y=0.0, z=2.0, roll=np.radians(70.0))
    status = checker.check(state=state, dist_to_goal=5.0, step_count=20)

    assert status.terminated is True
    assert status.is_crashed is True
    assert status.reason == "excessive_tilt"


def test_step_timeout_truncation():
    config = UAVEnvConfig()
    checker = TerminationChecker(config)

    state = create_mock_state(x=0.0, y=0.0, z=2.0)
    status = checker.check(state=state, dist_to_goal=5.0, step_count=200)

    assert status.terminated is False
    assert status.truncated is True
    assert status.reason == "episode_timeout"
