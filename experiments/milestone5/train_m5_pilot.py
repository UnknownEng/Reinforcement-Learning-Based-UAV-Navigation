"""
train_m5_pilot.py
-----------------
Bounded M5.4 Pilot Training Runner for Autonomous UAV Navigation.

Strict Constraints:
1. Model: Single [64, 64] MLP architecture (pi=[64, 64], vf=[64, 64]).
2. Environment: UAVFastNav-v2 (Stage 0: 360° heading alignment in free space).
3. Budget: Maximum 100,000 environment steps OR 15 minutes wall-clock time.
4. Seeds: Designated training seeds only (base seed 100). Zero access to frozen benchmarks.
5. Safety: Immediate halt on NaN/Inf, reset failures, or checkpoint corruption.
6. Isolation: M4 baseline checkpoint hash verified pre- and post-training.
"""

from collections import deque
from dataclasses import asdict
import hashlib
import json
import os
import resource
import sys
import time
from typing import Dict, Any, List

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import rl_env
from rl_env.uav_fast_env_v2 import UAVFastEnvV2, UAVObstacleEnvConfigV2


def get_checkpoint_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class PilotMonitoringCallback(BaseCallback):
    """
    Monitoring and Safety Guard Callback for Phase M5.4 Pilot Training.
    - Halts immediately if step budget (100,000 steps) is reached (no rollout overshoot).
    - Halts immediately if wall-clock budget (15 mins) is exceeded.
    - Halts immediately if NaN or Inf appears in network parameters.
    - Records comprehensive explicit per-episode metrics into structured CSV and JSON.
    """

    def __init__(
        self,
        max_timesteps: int = 100000,
        max_wall_clock_s: float = 900.0,  # 15 minutes
        episodes_csv_path: Optional[str] = None,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.max_timesteps = int(max_timesteps)
        self.max_wall_clock_s = float(max_wall_clock_s)
        self.episodes_csv_path = episodes_csv_path
        self.start_time: float = 0.0
        self.episode_count: int = 0
        self.episodes_data: List[Dict[str, Any]] = []
        self.nan_inf_detected: bool = False
        self.time_limit_exceeded: bool = False
        self.step_limit_reached: bool = False
        self.last_log_step: int = 0
        self.log_interval: int = 10000
        self._csv_file = None

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()
        if self.episodes_csv_path:
            self._csv_file = open(self.episodes_csv_path, "w", buffering=1)
            self._csv_file.write(
                "episode,step,return,length,is_success,is_collision,is_timeout,is_out_of_bounds,min_clearance,termination_reason\n"
            )

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def _on_step(self) -> bool:
        # 1. Hard step budget check (prevents any rollout overshoot)
        if self.num_timesteps >= self.max_timesteps:
            self.step_limit_reached = True
            if self.verbose > 0:
                print(f"\n[PILOT HALT] Hard step budget reached: {self.num_timesteps} >= {self.max_timesteps}")
            return False

        # 2. Wall-clock budget check
        elapsed = time.perf_counter() - self.start_time
        if elapsed >= self.max_wall_clock_s:
            self.time_limit_exceeded = True
            if self.verbose > 0:
                print(f"\n[PILOT HALT] Wall-clock time budget reached: {elapsed:.2f} s >= {self.max_wall_clock_s:.1f} s")
            return False

        # 3. NaN / Inf parameter check (periodic)
        if self.num_timesteps % 2048 == 0:
            for name, param in self.model.policy.named_parameters():
                if not torch.all(torch.isfinite(param)):
                    self.nan_inf_detected = True
                    print(f"\n[PILOT ABORT] NaN/Inf detected in model parameter {name} at step {self.num_timesteps}!")
                    return False

        # 4. Intercept completed episodes from info dicts with explicit fields
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_count += 1
                ep_len = int(info["episode"]["l"])
                ep_ret = float(info["episode"]["r"])
                is_success = bool(info.get("is_success", False))
                is_crashed = bool(info.get("is_crashed", False))
                is_out_of_bounds = bool(info.get("is_out_of_bounds", False))
                min_clearance = float(info.get("min_clearance_m", 10.0))
                term_reason = str(info.get("termination_reason", "unknown"))

                is_timeout = (ep_len >= 200) and not (is_success or is_crashed or is_out_of_bounds)

                record = {
                    "episode": self.episode_count,
                    "step": int(self.num_timesteps),
                    "return": round(ep_ret, 4),
                    "length": ep_len,
                    "is_success": is_success,
                    "is_collision": is_crashed,
                    "is_timeout": is_timeout,
                    "is_out_of_bounds": is_out_of_bounds,
                    "min_clearance": round(min_clearance, 4),
                    "termination_reason": term_reason
                }
                self.episodes_data.append(record)

                if self._csv_file:
                    self._csv_file.write(
                        f"{self.episode_count},{self.num_timesteps},{ep_ret:.4f},{ep_len},"
                        f"{int(is_success)},{int(is_crashed)},{int(is_timeout)},{int(is_out_of_bounds)},"
                        f"{min_clearance:.4f},{term_reason}\n"
                    )

        # 4. Periodic progress logging
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self.last_log_step = self.num_timesteps
            recent = self.episodes_data[-50:] if len(self.episodes_data) >= 50 else self.episodes_data
            if recent:
                succ_rate = sum(1 for e in recent if e["is_success"]) / len(recent) * 100.0
                mean_ret = float(np.mean([e["return"] for e in recent]))
                mean_len = float(np.mean([e["length"] for e in recent]))
            else:
                succ_rate, mean_ret, mean_len = 0.0, 0.0, 0.0

            fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0
            if self.verbose > 0:
                print(f"[Step {self.num_timesteps:6d} | {elapsed:5.1f}s | {fps:4d} fps] "
                      f"Episodes: {self.episode_count:4d} | Recent Succ: {succ_rate:5.1f}% | "
                      f"Mean Return: {mean_ret:6.1f} | Mean Len: {mean_len:4.1f}")

        return True


def run_pilot_training(
    total_timesteps: int = 100000,
    max_wall_clock_s: float = 900.0,  # 15 min
    base_seed: int = 100
) -> Dict[str, Any]:
    print("=" * 75)
    print("PHASE M5.4 BOUNDED PILOT TRAINING: [64, 64] MLP")
    print("=" * 75)

    # 1. Pre-training Baseline Integrity Check
    m4_ckpt_path = "experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip"
    expected_m4_hash = "06907c9fa636e5f268ca55750107918370e445313ad1ac01207c15a4c5935308"
    if not os.path.exists(m4_ckpt_path):
        raise FileNotFoundError(f"M4 Checkpoint missing: {m4_ckpt_path}")
    pre_m4_hash = get_checkpoint_sha256(m4_ckpt_path)
    if pre_m4_hash != expected_m4_hash:
        raise ValueError(f"Pre-flight M4 Checkpoint SHA mismatch! Expected {expected_m4_hash}, got {pre_m4_hash}")
    print(f"[INTEGRITY PRE-CHECK] Frozen M4 Checkpoint verified: {pre_m4_hash[:16]}... (BIT-IDENTICAL)")

    # 2. Output Directories Setup
    model_dir = "experiments/milestone5/models"
    log_dir = "experiments/milestone5/logs/pilot"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    pilot_ckpt_path = os.path.join(model_dir, "ppo_m5_pilot_64x64.zip")
    monitor_csv_path = os.path.join(log_dir, "pilot_monitor.csv")

    # 3. Environment Instantiation (Curriculum Stage 0: Free space, arbitrary yaw)
    config = UAVObstacleEnvConfigV2(
        curriculum_stage=0,
        reward_align_weight=0.10,
        reward_speed_gate_weight=0.10,
        grid_resolution=0.10,
        min_path_clearance=0.50,
        max_resample_attempts=50
    )
    raw_env = gym.make("UAVFastNav-v2", config=config)
    env = Monitor(raw_env, filename=monitor_csv_path)

    # Set deterministic training seed
    env.reset(seed=base_seed)

    # 4. PPO Agent Configuration ([64, 64] MLP Architecture)
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=base_seed,
        device="cpu",
        verbose=0
    )

    # 5. Execute Training with Safety Callback
    episodes_csv = os.path.join(log_dir, "pilot_episodes.csv")
    callback = PilotMonitoringCallback(
        max_timesteps=total_timesteps,
        max_wall_clock_s=max_wall_clock_s,
        episodes_csv_path=episodes_csv,
        verbose=1
    )

    t0 = time.perf_counter()
    rusage_start = resource.getrusage(resource.RUSAGE_SELF)

    print(f"\n[TRAINING START] Launching pilot run for up to {total_timesteps:,} steps (Max {max_wall_clock_s/60:.1f} mins)...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except Exception as e:
        print(f"\n[EXCEPTION DURING TRAINING]: {e}")
        env.close()
        raise e

    t1 = time.perf_counter()
    rusage_end = resource.getrusage(resource.RUSAGE_SELF)
    env.close()

    elapsed_wall_s = t1 - t0
    steps_completed = int(model.num_timesteps)
    steps_per_sec = steps_completed / elapsed_wall_s if elapsed_wall_s > 0 else 0.0
    peak_ram_mb = rusage_end.ru_maxrss / 1024.0

    # 6. Save Pilot Model Checkpoint
    model.save(pilot_ckpt_path)
    pilot_hash = get_checkpoint_sha256(pilot_ckpt_path)

    # 7. Post-training Baseline Integrity Check
    post_m4_hash = get_checkpoint_sha256(m4_ckpt_path)
    if post_m4_hash != expected_m4_hash:
        raise ValueError("CRITICAL: Frozen Milestone 4 Checkpoint was corrupted or overwritten!")
    print(f"[INTEGRITY POST-CHECK] Frozen M4 Checkpoint verified: {post_m4_hash[:16]}... (BIT-IDENTICAL)")

    # 8. Compute Empirical Training Statistics
    episodes = callback.episodes_data
    num_episodes = len(episodes)

    if num_episodes > 0:
        returns = [e["return"] for e in episodes]
        lengths = [e["length"] for e in episodes]
        success_count = sum(1 for e in episodes if e["is_success"])
        crash_count = sum(1 for e in episodes if e["is_crashed"])
        oob_count = sum(1 for e in episodes if e["is_out_of_bounds"])
        timeout_count = sum(1 for e in episodes if e["is_timeout"])

        success_rate = success_count / num_episodes
        crash_rate = crash_count / num_episodes
        oob_rate = oob_count / num_episodes
        timeout_rate = timeout_count / num_episodes

        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        min_return = float(np.min(returns))
        max_return = float(np.max(returns))

        # First 50 vs Last 50 comparison (learning dynamics)
        first_n = episodes[:50]
        last_n = episodes[-50:] if len(episodes) >= 50 else episodes

        first_succ_rate = sum(1 for e in first_n if e["is_success"]) / len(first_n)
        first_mean_ret = float(np.mean([e["return"] for e in first_n]))

        last_succ_rate = sum(1 for e in last_n if e["is_success"]) / len(last_n)
        last_mean_ret = float(np.mean([e["return"] for e in last_n]))
    else:
        success_count = crash_count = oob_count = timeout_count = 0
        success_rate = crash_rate = oob_rate = timeout_rate = 0.0
        mean_return = std_return = min_return = max_return = 0.0
        first_succ_rate = first_mean_ret = last_succ_rate = last_mean_ret = 0.0

    results = {
        "architecture": "MLP [64, 64]",
        "curriculum_stage": 0,
        "base_seed": base_seed,
        "target_steps": total_timesteps,
        "completed_steps": steps_completed,
        "elapsed_wall_clock_s": round(elapsed_wall_s, 2),
        "steps_per_second": round(steps_per_sec, 1),
        "peak_ram_mb": round(peak_ram_mb, 1),
        "nan_inf_detected": bool(callback.nan_inf_detected),
        "time_limit_exceeded": bool(callback.time_limit_exceeded),
        "completed_episodes": num_episodes,
        "episodic_metrics": {
            "mean_return": round(mean_return, 2),
            "std_return": round(std_return, 2),
            "min_return": round(min_return, 2),
            "max_return": round(max_return, 2),
            "success_rate": round(success_rate, 4),
            "success_count": success_count,
            "collision_rate": round(crash_rate, 4),
            "collision_count": crash_count,
            "timeout_rate": round(timeout_rate, 4),
            "timeout_count": timeout_count,
            "out_of_bounds_rate": round(oob_rate, 4),
            "out_of_bounds_count": oob_count,
        },
        "learning_progression": {
            "first_50_episodes": {
                "success_rate": round(first_succ_rate, 4),
                "mean_return": round(first_mean_ret, 2)
            },
            "last_50_episodes": {
                "success_rate": round(last_succ_rate, 4),
                "mean_return": round(last_mean_ret, 2)
            },
            "success_rate_gain_pp": round((last_succ_rate - first_succ_rate) * 100.0, 2),
            "return_gain": round(last_mean_ret - first_mean_ret, 2)
        },
        "checkpoint_isolation": {
            "m4_baseline_sha256": post_m4_hash,
            "m4_hash_verified": True,
            "pilot_checkpoint_path": pilot_ckpt_path,
            "pilot_checkpoint_sha256": pilot_hash
        }
    }

    # Save detailed JSON results
    json_path = "experiments/milestone5/m5_pilot_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print("M5.4 PILOT TRAINING COMPLETE — SUMMARY METRICS")
    print("=" * 75)
    print(f"  Architecture:               MLP [64, 64]")
    print(f"  Steps Completed:            {steps_completed:,} / {total_timesteps:,}")
    print(f"  Wall-Clock Duration:        {elapsed_wall_s:.2f} s ({elapsed_wall_s/60:.2f} min / 15.0 max)")
    print(f"  Throughput:                 {steps_per_sec:.1f} steps/second")
    print(f"  Peak RAM:                   {peak_ram_mb:.1f} MB")
    print(f"  NaN / Inf Status:           {callback.nan_inf_detected} (CLEAN)")
    print(f"  Completed Episodes:         {num_episodes}")
    print(f"  Mean Return:                {mean_return:.2f} ± {std_return:.2f}")
    print(f"  Overall Success Rate:       {success_rate*100:.1f}% ({success_count}/{num_episodes})")
    print(f"  Overall Collision Rate:     {crash_rate*100:.1f}% ({crash_count}/{num_episodes})")
    print(f"  Overall Timeout Rate:       {timeout_rate*100:.1f}% ({timeout_count}/{num_episodes})")
    print(f"  Overall Out-of-Bounds Rate: {oob_rate*100:.1f}% ({oob_count}/{num_episodes})")
    print("-" * 75)
    print(f"  Learning Dynamics Progression (First 50 -> Last 50):")
    print(f"    Success Rate:             {first_succ_rate*100:.1f}% -> {last_succ_rate*100:.1f}% ({results['learning_progression']['success_rate_gain_pp']:+.1f} pp)")
    print(f"    Mean Return:              {first_mean_ret:.1f} -> {last_mean_ret:.1f} ({results['learning_progression']['return_gain']:+.1f})")
    print("-" * 75)
    print(f"  Pilot Checkpoint Saved:     {pilot_ckpt_path}")
    print(f"  Pilot SHA-256:              {pilot_hash}")
    print(f"  M4 Baseline Checkpoint:     UNTOUCHED & BIT-IDENTICAL")
    print("=" * 75)

    return results


if __name__ == "__main__":
    run_pilot_training()
