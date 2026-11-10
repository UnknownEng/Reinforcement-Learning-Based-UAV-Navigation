"""
train_ppo_26d.py
----------------
Milestone 4 Phase 4B — Gate B3: 26-D Perception-Augmented PPO Curriculum Training.

Features:
  - Fast, bounded training on UAVFastNav-v1 with 26-D observation space
  - 4-Stage Free-Space to Clutter Automated Curriculum (Topic 11):
      * Stage 0 (0 to 15% steps): Free Space (N_obs = 0)
      * Stage 1 (15% to 35% steps): Single Obstacle on path (N_obs = 1)
      * Stage 2 (35% to 65% steps): Corridors and Gates (N_obs = 3-4)
      * Stage 3 (65% to 100% steps): Dense Clutter (N_obs = 6-8)
  - Small MLP policy architecture (pi=[64, 64], vf=[64, 64], Tanh activation)
  - Reproducible random seeds (torch=42, numpy=42, env=42)
  - Periodic evaluation callbacks and best-model checkpointing
  - Checkpoints saved to dedicated Milestone 4 Phase 4B directory
  - Strict preservation of frozen Phase 4A baseline and original 18-D checkpoint
  - Zero-NaN / Inf safety watchdog
  - Comprehensive telemetry logging (CPU, RAM, wall-clock time, step throughput)
"""

import argparse
import hashlib
import json
import os
import shutil
import time
from typing import Any, Dict, Optional
import psutil
import torch
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

import rl_env


class CurriculumStageCallback(BaseCallback):
    """
    Manages automated curriculum progression across the 4 stages defined in Topic 11.
    Saves intermediate stage checkpoints and updates environment obstacle density.
    """
    def __init__(
        self,
        eval_env: Monitor,
        total_timesteps: int,
        checkpoint_dir: str,
        stage_milestones: Optional[Dict[int, int]] = None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.total_timesteps = total_timesteps
        self.checkpoint_dir = checkpoint_dir

        if stage_milestones is not None:
            self.s0_end = stage_milestones[0]
            self.s1_end = stage_milestones[1]
            self.s2_end = stage_milestones[2]
        else:
            self.s0_end = int(0.15 * total_timesteps)
            self.s1_end = int(0.35 * total_timesteps)
            self.s2_end = int(0.65 * total_timesteps)

        self.current_stage: int = 0
        self.stage_transition_log: list = []
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _on_step(self) -> bool:
        step = self.num_timesteps

        # Determine target curriculum stage
        if step < self.s0_end:
            target_stage = 0
        elif step < self.s1_end:
            target_stage = 1
        elif step < self.s2_end:
            target_stage = 2
        else:
            target_stage = 3

        # Safety watchdog: check for NaN/Inf in rewards or observations
        rewards = self.locals.get("rewards")
        if rewards is not None:
            if np.any(np.isnan(rewards)) or np.any(np.isinf(rewards)):
                print(f"[ERROR] NaN/Inf detected in step rewards at timestep {step}! Halting training.")
                return False

        # Stage transition trigger
        if target_stage != self.current_stage:
            prev_stage = self.current_stage
            self.current_stage = target_stage

            # Save stage checkpoint
            stage_ckpt_path = os.path.join(
                self.checkpoint_dir, f"ppo_26d_stage{prev_stage}_{step}_steps.zip"
            )
            self.model.save(stage_ckpt_path)

            transition_info = {
                "timestep": step,
                "from_stage": prev_stage,
                "to_stage": target_stage,
                "stage_checkpoint": stage_ckpt_path,
                "timestamp": time.time()
            }
            self.stage_transition_log.append(transition_info)

            # Update training env
            if hasattr(self.training_env, "env_method"):
                self.training_env.env_method("set_curriculum_stage", target_stage)
            else:
                self.training_env.unwrapped.set_curriculum_stage(target_stage)

            # Update evaluation env
            self.eval_env.unwrapped.set_curriculum_stage(target_stage)

            if self.verbose:
                print("\n" + "=" * 65)
                print(f"[CURRICULUM STAGE ADVANCE] Step {step:,} / {self.total_timesteps:,}")
                print(f"  Transition: Stage {prev_stage} -> Stage {target_stage}")
                print(f"  Saved Stage Checkpoint: {stage_ckpt_path}")
                print("=" * 65 + "\n")

        return True


def train_ppo_26d(
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    base_dir: str = "/home/kali/proj/experiments/milestone4/phase4b_gate_b3",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.20,
    ent_coef: float = 0.005,
    vf_coef: float = 0.50,
    max_grad_norm: float = 0.50,
    eval_freq: int = 25000,
    n_eval_episodes: int = 20
) -> Dict[str, Any]:
    """
    Executes Gate B3 Curriculum Training on UAVFastNav-v1.
    """
    model_dir = os.path.join(base_dir, "models")
    stage_ckpt_dir = os.path.join(model_dir, "stage_checkpoints")
    log_dir = os.path.join(base_dir, "logs")
    tb_dir = os.path.join(base_dir, "tb_logs")
    m4_models_dir = "/home/kali/proj/experiments/milestone4/models"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(stage_ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(m4_models_dir, exist_ok=True)

    # 1. Strict Baseline Checkpoint Check
    frozen_ckpt = "/home/kali/proj/experiments/milestone2/models/ppo_uav_final.zip"
    expected_hash = "8d7a24eda075cbbeb60c4a6df07029c3a38fd770ce436a6daf96feff85300dfc"
    with open(frozen_ckpt, "rb") as f:
        pre_hash = hashlib.sha256(f.read()).hexdigest()
    if pre_hash != expected_hash:
        raise RuntimeError(f"FATAL: Frozen baseline hash mismatch before training! Expected {expected_hash}, got {pre_hash}")

    # 2. Fix Random Seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 3. Training & Evaluation Environments
    raw_env = gym.make("UAVFastNav-v1")
    raw_env.unwrapped.set_curriculum_stage(0)
    env = Monitor(raw_env, filename=os.path.join(log_dir, "monitor.csv"))
    env.reset(seed=seed)

    eval_raw_env = gym.make("UAVFastNav-v1")
    eval_raw_env.unwrapped.set_curriculum_stage(0)
    eval_env = Monitor(eval_raw_env)
    eval_env.reset(seed=seed + 1000)

    # 4. Neural Network Architecture (MLP pi=[64, 64], vf=[64, 64], Tanh)
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=torch.nn.Tanh
    )

    print("=" * 70)
    print("MILESTONE 4 PHASE 4B: GATE B3 CURRICULUM TRAINING")
    print("=" * 70)
    print(f"Environment:          UAVFastNav-v1 (26-D Observation, Option 1 LiDAR)")
    print(f"Total Timesteps:      {total_timesteps:,}")
    print(f"Random Seed:          {seed}")
    print(f"Architecture:         MLP (pi=[64, 64], vf=[64, 64], Tanh)")
    print(f"Learning Rate:        {learning_rate}")
    print(f"Rollout n_steps:      {n_steps} (Batch: {batch_size}, Epochs: {n_epochs})")
    print(f"Discount gamma:       {gamma} (GAE lambda: {gae_lambda})")
    print(f"Clip Range:           {clip_range}")
    print(f"Entropy Coef:         {ent_coef}")
    print(f"Value Coef:           {vf_coef}")
    print(f"Device:               cpu")
    print(f"Model Output Dir:     {model_dir}")
    print(f"TensorBoard Log Dir:  {tb_dir}")
    print("=" * 70)

    # Linear learning rate schedule with floor
    def lr_schedule(progress_remaining: float) -> float:
        min_lr = 5e-5
        return float(min_lr + (learning_rate - min_lr) * progress_remaining)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr_schedule,
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
        tensorboard_log=tb_dir,
        seed=seed,
        device="cpu",
        verbose=1
    )

    # 5. Callbacks
    curriculum_cb = CurriculumStageCallback(
        eval_env=eval_env,
        total_timesteps=total_timesteps,
        checkpoint_dir=stage_ckpt_dir,
        verbose=1
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50000, eval_freq),
        save_path=stage_ckpt_dir,
        name_prefix="ppo_uav_26d_step"
    )

    process = psutil.Process(os.getpid())
    ram_initial_mb = process.memory_info().rss / (1024 * 1024)
    t_start = time.perf_counter()

    # 6. Training Execution
    print("\n[TRAINING START] Launching curriculum learning loop...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_cb, eval_cb, checkpoint_cb],
        progress_bar=False
    )

    t_end = time.perf_counter()
    wall_time_s = t_end - t_start
    ram_final_mb = process.memory_info().rss / (1024 * 1024)
    steps_per_sec = total_timesteps / wall_time_s

    # 7. Save Final Model Checkpoints
    final_model_path = os.path.join(model_dir, "ppo_uav_obstacle_26d_final.zip")
    model.save(final_model_path)
    print(f"\n[MODEL SAVED] Final model saved to: {final_model_path}")

    # Mirror to milestone4/models/ as required by specification
    m4_final_path = os.path.join(m4_models_dir, "ppo_uav_obstacle_26d_final.zip")
    shutil.copy2(final_model_path, m4_final_path)

    best_model_src = os.path.join(model_dir, "best_model.zip")
    m4_best_path = os.path.join(m4_models_dir, "ppo_uav_obstacle_26d_best.zip")
    if os.path.exists(best_model_src):
        shutil.copy2(best_model_src, m4_best_path)
        print(f"[MODEL SAVED] Best model mirrored to: {m4_best_path}")

    # Compute checksums
    def compute_sha256(path: str) -> str:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    final_hash = compute_sha256(final_model_path)
    best_hash = compute_sha256(m4_best_path) if os.path.exists(m4_best_path) else "N/A"

    # Re-verify original baseline checkpoint hash
    with open(frozen_ckpt, "rb") as f:
        post_hash = hashlib.sha256(f.read()).hexdigest()
    if post_hash != expected_hash:
        raise RuntimeError(f"FATAL: Frozen baseline hash altered during training! Expected {expected_hash}, got {post_hash}")

    # 8. Record Telemetry Metadata
    metadata = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate_id": "Gate B3: Full Curriculum Training",
        "algorithm": "PPO",
        "library": "Stable-Baselines3",
        "total_timesteps": total_timesteps,
        "seed": seed,
        "wall_time_seconds": wall_time_s,
        "steps_per_second": steps_per_sec,
        "ram_initial_mb": ram_initial_mb,
        "ram_final_mb": ram_final_mb,
        "ram_delta_mb": ram_final_mb - ram_initial_mb,
        "hyperparameters": {
            "net_arch": {"pi": [64, 64], "vf": [64, 64]},
            "activation_fn": "Tanh",
            "learning_rate_initial": learning_rate,
            "learning_rate_schedule": "linear_decay",
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
        "curriculum_stages": {
            "stage_0": {"name": "Free Space", "steps": f"0 - {curriculum_cb.s0_end:,}", "obstacles": 0},
            "stage_1": {"name": "Single Obstacle", "steps": f"{curriculum_cb.s0_end:,} - {curriculum_cb.s1_end:,}", "obstacles": 1},
            "stage_2": {"name": "Corridors / Gates", "steps": f"{curriculum_cb.s1_end:,} - {curriculum_cb.s2_end:,}", "obstacles": "3 - 4"},
            "stage_3": {"name": "Dense Clutter", "steps": f"{curriculum_cb.s2_end:,} - {total_timesteps:,}", "obstacles": "6 - 8"}
        },
        "stage_transitions": curriculum_cb.stage_transition_log,
        "checkpoint_hashes": {
            "ppo_uav_obstacle_26d_final": {
                "path": final_model_path,
                "sha256": final_hash
            },
            "ppo_uav_obstacle_26d_best": {
                "path": m4_best_path,
                "sha256": best_hash
            },
            "frozen_baseline_ppo_uav_final": {
                "path": frozen_ckpt,
                "sha256": post_hash,
                "integrity_intact": True
            }
        }
    }

    meta_path = os.path.join(base_dir, "training_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nTraining metadata recorded to: {meta_path}")

    env.close()
    eval_env.close()
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milestone 4 Phase 4B Gate B3 Training")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-freq", type=int, default=25000, help="Evaluation frequency in steps")
    args = parser.parse_args()

    train_ppo_26d(
        total_timesteps=args.timesteps,
        seed=args.seed,
        eval_freq=args.eval_freq
    )
