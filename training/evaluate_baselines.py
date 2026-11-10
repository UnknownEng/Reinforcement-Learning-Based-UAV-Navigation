"""
evaluate_baselines.py
---------------------
Evaluate Random Policy and Classical Proportional Waypoint Controller
over 100 standardized, reproducible test episodes (fixed seeds 1000..1099).
Outputs comprehensive metrics and comparison tables.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
import gymnasium as gym
import numpy as np

import rl_env
from rl_env.baselines import RandomPolicy, ProportionalWaypointController

MILESTONE2_DIR = Path(__file__).resolve().parents[1] / "experiments" / "milestone2"


def evaluate_policy(
    policy,
    env: gym.Env,
    num_episodes: int = 100,
    start_seed: int = 1000
) -> Dict[str, Any]:
    rewards: List[float] = []
    episode_lengths: List[int] = []
    final_distances: List[float] = []
    path_lengths: List[float] = []
    path_efficiencies: List[float] = []
    successes: int = 0
    crashes: int = 0
    out_of_bounds: int = 0
    timeouts: int = 0

    for ep_idx in range(num_episodes):
        seed = start_seed + ep_idx
        obs, info = env.reset(seed=seed)
        policy.reset()

        ep_rew = 0.0
        step = 0
        init_dist = info["dist_to_goal"]

        while True:
            action = policy.predict(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            ep_rew += rew
            step += 1

            if terminated or truncated:
                if info.get("is_success", False):
                    successes += 1
                elif info.get("is_crashed", False):
                    crashes += 1
                elif info.get("is_out_of_bounds", False):
                    out_of_bounds += 1
                elif truncated:
                    timeouts += 1

                final_dist = info.get("final_distance", info.get("dist_to_goal", 0.0))
                path_len = info.get("path_length", 0.0)
                efficiency = (path_len / init_dist) if init_dist > 1e-3 else 1.0

                rewards.append(ep_rew)
                episode_lengths.append(step)
                final_distances.append(final_dist)
                path_lengths.append(path_len)
                path_efficiencies.append(efficiency)
                break

    metrics = {
        "num_episodes": num_episodes,
        "success_rate_pct": float(100.0 * successes / num_episodes),
        "crash_rate_pct": float(100.0 * crashes / num_episodes),
        "out_of_bounds_rate_pct": float(100.0 * out_of_bounds / num_episodes),
        "timeout_rate_pct": float(100.0 * timeouts / num_episodes),
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "mean_episode_length_steps": float(np.mean(episode_lengths)),
        "std_episode_length_steps": float(np.std(episode_lengths)),
        "mean_episode_duration_s": float(np.mean(episode_lengths) * 0.10),
        "mean_final_distance_m": float(np.mean(final_distances)),
        "std_final_distance_m": float(np.std(final_distances)),
        "mean_path_length_m": float(np.mean(path_lengths)),
        "std_path_length_m": float(np.std(path_lengths)),
        "mean_path_efficiency_ratio": float(np.mean(path_efficiencies)),
    }
    return metrics


def run_all_baselines(num_episodes: int = 100):
    env = gym.make("UAVFastNav-v0")
    print(f"=== Evaluating Baselines over {num_episodes} seeded episodes ===")

    # 1. Random Policy
    print("\n[1/2] Evaluating Random Policy...")
    rand_pol = RandomPolicy(env.action_space, seed=42)
    rand_metrics = evaluate_policy(rand_pol, env, num_episodes=num_episodes, start_seed=1000)
    print(f"  Success Rate:  {rand_metrics['success_rate_pct']:.1f}%")
    print(f"  Mean Reward:   {rand_metrics['mean_reward']:.2f} +/- {rand_metrics['std_reward']:.2f}")
    print(f"  Final Dist:    {rand_metrics['mean_final_distance_m']:.2f} m")

    # 2. Classical Proportional Waypoint Controller
    print("\n[2/2] Evaluating Classical Proportional Controller...")
    prop_ctrl = ProportionalWaypointController(env.unwrapped.config)
    prop_metrics = evaluate_policy(prop_ctrl, env, num_episodes=num_episodes, start_seed=1000)
    print(f"  Success Rate:  {prop_metrics['success_rate_pct']:.1f}%")
    print(f"  Mean Reward:   {prop_metrics['mean_reward']:.2f} +/- {prop_metrics['std_reward']:.2f}")
    print(f"  Final Dist:    {prop_metrics['mean_final_distance_m']:.2f} m")
    print(f"  Mean Steps:    {prop_metrics['mean_episode_length_steps']:.1f} ({prop_metrics['mean_episode_duration_s']:.2f} s)")

    env.close()

    results = {
        "evaluation_episodes": num_episodes,
        "seed_range": [1000, 1000 + num_episodes - 1],
        "random_policy": rand_metrics,
        "proportional_controller": prop_metrics
    }

    out_path = MILESTONE2_DIR / "baseline_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nBaseline results written to: {out_path}")
    return results


if __name__ == "__main__":
    run_all_baselines(num_episodes=100)
