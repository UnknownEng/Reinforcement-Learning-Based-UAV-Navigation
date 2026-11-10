"""
train_ppo.py
------------
Train Proximal Policy Optimization (PPO) agent on the UAVFastNav-v0 environment
using Stable-Baselines3.
Features:
  - Small MLP architecture (pi=[64, 64], vf=[64, 64])
  - Fully reproducible (fixed random seeds for Torch, NumPy, and Gym)
  - TensorBoard logging
  - Periodic evaluation callbacks and best-model checkpoints
  - Resource usage telemetry (CPU, RAM, wall-clock time)
"""

import argparse
import json
import os
import time
from pathlib import Path
import psutil
import torch
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

import rl_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MILESTONE2_DIR = PROJECT_ROOT / "experiments" / "milestone2"


def train_ppo(
    total_timesteps: int = 100000,
    seed: int = 42,
    log_dir: str = str(MILESTONE2_DIR / "tb_logs"),
    model_dir: str = str(MILESTONE2_DIR / "models"),
    eval_freq: int = 10000,
    n_eval_episodes: int = 20,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.005,
    vf_coef: float = 0.5,
    max_grad_norm: float = 0.5
):
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate training environment wrapped with Monitor for episode statistics
    raw_env = gym.make("UAVFastNav-v0")
    env = Monitor(raw_env, filename=os.path.join(log_dir, "monitor.csv"))
    env.reset(seed=seed)

    # Separate evaluation environment with independent seed
    eval_raw_env = gym.make("UAVFastNav-v0")
    eval_env = Monitor(eval_raw_env)
    eval_env.reset(seed=seed + 100)

    # Network architecture: small MLP (pi=[64, 64], vf=[64, 64])
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=torch.nn.Tanh
    )

    print("=" * 60)
    print("Initializing PPO Training on UAVFastNav-v0")
    print(f"Total Timesteps:     {total_timesteps:,}")
    print(f"Random Seed:         {seed}")
    print(f"Policy Network:      MLP pi=[64, 64], vf=[64, 64]")
    print(f"Learning Rate:       {learning_rate}")
    print(f"Rollout n_steps:     {n_steps} (Batch: {batch_size}, Epochs: {n_epochs})")
    print(f"Discount gamma:      {gamma} (GAE lambda: {gae_lambda})")
    print(f"Clip Range:          {clip_range}")
    print(f"Device:              cpu")
    print(f"TensorBoard Dir:     {log_dir}")
    print(f"Model Checkpoint Dir:{model_dir}")
    print("=" * 60)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
        seed=seed,
        device="cpu",
        verbose=1
    )

    # Evaluation and checkpoint callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000, eval_freq),
        save_path=model_dir,
        name_prefix="ppo_uav_step"
    )

    process = psutil.Process(os.getpid())
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)
    t_start = time.perf_counter()

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=False
    )

    t_end = time.perf_counter()
    wall_time_s = t_end - t_start
    ram_final_mb = process.memory_info().rss / (1024 * 1024)
    training_steps_per_sec = total_timesteps / wall_time_s

    # Save final model
    final_model_path = os.path.join(model_dir, "ppo_uav_final.zip")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    metadata = {
        "algorithm": "PPO",
        "library": "Stable-Baselines3",
        "total_timesteps": total_timesteps,
        "seed": seed,
        "wall_time_seconds": wall_time_s,
        "steps_per_second": training_steps_per_sec,
        "ram_initial_mb": ram_initial_mb,
        "ram_final_mb": ram_final_mb,
        "ram_delta_mb": ram_final_mb - ram_initial_mb,
        "hyperparameters": {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": "Tanh",
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "max_grad_norm": max_grad_norm,
            "device": "cpu"
        },
        "final_model_path": final_model_path
    }

    meta_path = MILESTONE2_DIR / "training_metadata.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Training metadata written to: {meta_path}")

    env.close()
    eval_env.close()
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ppo(total_timesteps=args.timesteps, seed=args.seed)
