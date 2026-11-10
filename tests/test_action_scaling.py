"""
test_action_scaling.py
----------------------
Unit tests for action space scaling, unscaling, saturation, and robustness.
"""

import pytest
import numpy as np

from rl_env.config import UAVEnvConfig
from rl_env.action import ActionHandler, CommandedVelocities


def test_action_scaling_extremes():
    config = UAVEnvConfig()
    handler = ActionHandler(config)

    # Per-axis upper limits
    cmd_x = handler.scale_action(np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32))
    assert np.isclose(cmd_x.vx, 3.0)
    assert np.isclose(cmd_x.vz, 1.5)
    assert np.isclose(cmd_x.yaw_rate, 1.0)
    cmd_y = handler.scale_action(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    assert np.isclose(cmd_y.vy, 3.0)

    # Per-axis lower limits
    cmd_neg = handler.scale_action(np.array([-1.0, 0.0, -1.0, -1.0], dtype=np.float32))
    assert np.isclose(cmd_neg.vx, -3.0)
    assert np.isclose(cmd_neg.vz, -1.5)
    assert np.isclose(cmd_neg.yaw_rate, -1.0)

    # Diagonal: combined horizontal speed is limited to max_horizontal_velocity
    for act in ([1.0, 1.0, 1.0, 1.0], [-1.0, -1.0, -1.0, -1.0]):
        cmd = handler.scale_action(np.array(act, dtype=np.float32))
        assert np.isclose(np.hypot(cmd.vx, cmd.vy), 3.0)
        assert np.isclose(abs(cmd.vx), abs(cmd.vy))
        assert np.isclose(abs(cmd.vz), 1.5)

    # Zero
    act_zero = np.zeros(4, dtype=np.float32)
    cmd_zero = handler.scale_action(act_zero)
    assert np.isclose(cmd_zero.vx, 0.0)
    assert np.isclose(cmd_zero.vy, 0.0)
    assert np.isclose(cmd_zero.vz, 0.0)
    assert np.isclose(cmd_zero.yaw_rate, 0.0)


def test_action_clamping_and_nan_safety():
    config = UAVEnvConfig()
    handler = ActionHandler(config)

    # Actions outside [-1, 1] must be saturated
    act_over = np.array([5.0, -10.0, 2.5, -3.0], dtype=np.float32)
    cmd = handler.scale_action(act_over)
    # Saturated to [1, -1] per axis, then the combined horizontal speed to 3 m/s
    assert np.isclose(np.hypot(cmd.vx, cmd.vy), 3.0)
    assert cmd.vx > 0.0 and cmd.vy < 0.0
    assert np.isclose(cmd.vz, 1.5)
    assert np.isclose(cmd.yaw_rate, -1.0)

    # Actions containing NaN / Inf
    act_nan = np.array([np.nan, np.inf, -np.inf, 0.5], dtype=np.float32)
    cmd_nan = handler.scale_action(act_nan)
    assert np.isfinite(cmd_nan.vx)
    assert np.isfinite(cmd_nan.vy)
    assert np.isfinite(cmd_nan.vz)
    assert np.isfinite(cmd_nan.yaw_rate)


def test_action_roundtrip_unscaling():
    config = UAVEnvConfig()
    handler = ActionHandler(config)

    orig_action = np.array([0.5, -0.25, 0.75, -0.6], dtype=np.float32)
    cmd = handler.scale_action(orig_action)
    recovered_action = handler.unscale_action(cmd)
    np.testing.assert_allclose(orig_action, recovered_action, atol=1e-6)


def test_action_dimension_mismatch():
    config = UAVEnvConfig()
    handler = ActionHandler(config)

    with pytest.raises(ValueError):
        handler.scale_action(np.array([1.0, 2.0, 3.0]))  # 3 elements instead of 4
