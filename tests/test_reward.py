"""
test_reward.py
--------------
Unit tests for decomposed reward components, signs, weights, and numerical safety.
"""

import pytest
import numpy as np

from rl_env.config import UAVEnvConfig
from rl_env.reward import RewardCalculator, RewardComponents


def test_progress_reward_sign():
    config = UAVEnvConfig()
    calc = RewardCalculator(config)
    zero_act = np.zeros(4, dtype=np.float32)

    # Moving closer: dist goes from 5.0m to 4.0m (+1.0m progress)
    rew_close = calc.compute_reward(
        dist_prev=5.0,
        dist_curr=4.0,
        action_curr=zero_act,
        action_prev=zero_act,
        is_success=False,
        is_out_of_bounds=False,
        is_crashed=False
    )
    # Progress reward = 2.5 * 1.0 = 2.5, time penalty = 0.02, total = 2.48
    assert rew_close.progress > 0.0
    assert np.isclose(rew_close.progress, 2.5, atol=1e-5)
    assert np.isclose(rew_close.total, 2.5 - config.reward_time_penalty, atol=1e-5)

    # Moving farther: dist goes from 4.0m to 5.0m (-1.0m progress)
    rew_away = calc.compute_reward(
        dist_prev=4.0,
        dist_curr=5.0,
        action_curr=zero_act,
        action_prev=zero_act,
        is_success=False,
        is_out_of_bounds=False,
        is_crashed=False
    )
    assert rew_away.progress < 0.0
    assert np.isclose(rew_away.progress, -2.5, atol=1e-5)


def test_success_bonus():
    config = UAVEnvConfig()
    calc = RewardCalculator(config)
    zero_act = np.zeros(4, dtype=np.float32)

    rew = calc.compute_reward(
        dist_prev=0.5,
        dist_curr=0.2,
        action_curr=zero_act,
        action_prev=zero_act,
        is_success=True,
        is_out_of_bounds=False,
        is_crashed=False
    )
    assert rew.success == config.reward_success
    assert rew.total > 90.0


def test_action_smoothness_penalty():
    config = UAVEnvConfig()
    calc = RewardCalculator(config)

    # Smooth action transition (identical actions)
    act1 = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    rew_smooth = calc.compute_reward(
        dist_prev=5.0,
        dist_curr=5.0,
        action_curr=act1,
        action_prev=act1,
        is_success=False,
        is_out_of_bounds=False,
        is_crashed=False
    )
    assert np.isclose(rew_smooth.smoothness_penalty, 0.0, atol=1e-6)

    # Abrupt action jerk (flip from -1 to +1)
    act_neg = np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    act_pos = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    rew_jerk = calc.compute_reward(
        dist_prev=5.0,
        dist_curr=5.0,
        action_curr=act_pos,
        action_prev=act_neg,
        is_success=False,
        is_out_of_bounds=False,
        is_crashed=False
    )
    # diff = [2, 2, 2, 2], diff^2 = [4, 4, 4, 4], sum = 16. penalty = 0.05 * 16 = 0.8
    assert np.isclose(rew_jerk.smoothness_penalty, 0.05 * 16.0, atol=1e-5)


def test_boundary_penalty():
    config = UAVEnvConfig()
    calc = RewardCalculator(config)
    zero_act = np.zeros(4, dtype=np.float32)

    rew_oob = calc.compute_reward(
        dist_prev=5.0,
        dist_curr=5.0,
        action_curr=zero_act,
        action_prev=zero_act,
        is_success=False,
        is_out_of_bounds=True,
        is_crashed=False
    )
    assert rew_oob.boundary_penalty == config.reward_collision_penalty
    assert rew_oob.total < -40.0
