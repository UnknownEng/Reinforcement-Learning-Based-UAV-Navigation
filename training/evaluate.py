"""
evaluate.py
-----------
Evaluate a trained PPO model (or checkpoint) over 100 standardized test episodes
using seeds 1000..1099, matching baseline evaluation exactly.
Outputs detailed evaluation statistics and compares with baseline metrics.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict
import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO

import rl_env
from training.evaluate_baselines import evaluate_policy

MILESTONE2_DIR = Path(__file__).resolve().parents[1] / "experiments" / "milestone2"


class PPOPolicyWrapper:
    """Wraps SB3 PPO model to expose .predict(obs, deterministic) interface."""
    def __init__(self, model: PPO):
        self.model = model

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action

    def reset(self):
        pass


def evaluate_ppo(
    model_path: str = str(MILESTONE2_DIR / "models" / "ppo_uav_final.zip"),
    num_episodes: int = 100,
    start_seed: int = 1000,
    output_path: str = str(MILESTONE2_DIR / "ppo_evaluation.json")
) -> Dict[str, Any]:
    print(f"Loading trained PPO model from: {model_path}")
    model = PPO.load(model_path, device="cpu")
    policy_wrapper = PPOPolicyWrapper(model)

    env = gym.make("UAVFastNav-v0")
    print(f"Evaluating PPO policy over {num_episodes} seeded episodes (seeds {start_seed}..{start_seed + num_episodes - 1})...")

    ppo_metrics = evaluate_policy(
        policy=policy_wrapper,
        env=env,
        num_episodes=num_episodes,
        start_seed=start_seed
    )
    env.close()

    print("\n=== PPO Evaluation Results ===")
    print(f"Success Rate:    {ppo_metrics['success_rate_pct']:.1f}%")
    print(f"Crash Rate:      {ppo_metrics['crash_rate_pct']:.1f}%")
    print(f"OOB Rate:        {ppo_metrics['out_of_bounds_rate_pct']:.1f}%")
    print(f"Timeout Rate:    {ppo_metrics['timeout_rate_pct']:.1f}%")
    print(f"Mean Reward:     {ppo_metrics['mean_reward']:.2f} +/- {ppo_metrics['std_reward']:.2f}")
    print(f"Mean Final Dist: {ppo_metrics['mean_final_distance_m']:.2f} +/- {ppo_metrics['std_final_distance_m']:.2f} m")
    print(f"Mean Steps:      {ppo_metrics['mean_episode_length_steps']:.1f} ({ppo_metrics['mean_episode_duration_s']:.2f} s)")
    print(f"Mean Path Len:   {ppo_metrics['mean_path_length_m']:.2f} m (Efficiency: {ppo_metrics['mean_path_efficiency_ratio']:.3f})")

    results = {
        "model_path": model_path,
        "evaluation_episodes": num_episodes,
        "seed_range": [start_seed, start_seed + num_episodes - 1],
        "metrics": ppo_metrics
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to: {output_path}")

    # Generate combined 3-way comparison table
    baseline_path = MILESTONE2_DIR / "baseline_comparison.json"
    if os.path.exists(baseline_path):
        with open(baseline_path, "r") as f:
            base_data = json.load(f)

        comparison_path = MILESTONE2_DIR / "tri_policy_comparison.json"
        tri_comparison = {
            "evaluation_episodes": num_episodes,
            "seed_range": [start_seed, start_seed + num_episodes - 1],
            "random_policy": base_data["random_policy"],
            "proportional_controller": base_data["proportional_controller"],
            "ppo_policy": ppo_metrics
        }
        with open(comparison_path, "w") as f:
            json.dump(tri_comparison, f, indent=2)
        print(f"Consolidated tri-policy comparison saved to: {comparison_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=str(MILESTONE2_DIR / "models" / "ppo_uav_final.zip"))
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    evaluate_ppo(model_path=args.model, num_episodes=args.episodes)
