"""
test_gym_api.py
---------------
Verify Gymnasium API compliance using gymnasium.utils.env_checker.check_env.
"""

import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import rl_env


def test_gym_api_compliance():
    """Verify that UAVFastNav-v0 satisfies all Gymnasium standard checks."""
    env = gym.make("UAVFastNav-v0")
    # check_env raises exceptions or warnings if any Gymnasium API invariant is broken
    check_env(env.unwrapped)
    env.close()


def test_reset_signature_and_seed_reproducibility():
    """Verify reset returns (obs, info) and seeded resets produce identical trajectories."""
    env = gym.make("UAVFastNav-v0")
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)

    assert obs1.shape == (18,)
    assert isinstance(info1, dict)
    assert "dist_to_goal" in info1
    assert "pos_world" in info1
    assert "goal_world" in info1

    # Seeded runs must produce identical initial states and goals
    assert (obs1 == obs2).all()
    assert (info1["pos_world"] == info2["pos_world"]).all()
    assert (info1["goal_world"] == info2["goal_world"]).all()

    # Stepping with identical actions produces identical subsequent states
    act = env.action_space.sample()
    next_obs1, rew1, term1, trunc1, _ = env.step(act)

    env.reset(seed=42)
    next_obs2, rew2, term2, trunc2, _ = env.step(act)

    assert (next_obs1 == next_obs2).all()
    assert rew1 == rew2
    assert term1 == term2
    assert trunc1 == trunc2

    env.close()


def test_step_signature_and_bounds():
    """Verify 5-tuple step output and bounds adherence."""
    env = gym.make("UAVFastNav-v0")
    obs, info = env.reset(seed=101)

    for _ in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (18,)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

        if terminated or truncated:
            break

    env.close()
