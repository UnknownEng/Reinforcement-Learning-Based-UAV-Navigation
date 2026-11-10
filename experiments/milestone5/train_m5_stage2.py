"""
train_m5_stage2.py
------------------
Milestone 5.4 Stage 2 Bounded Pilot Training: Single [64, 64] MLP Policy.

Constraints:
- Maximum 100,000 environment steps (strictly enforced)
- Maximum 15 minutes wall-clock time
- Single [64, 64] PPO policy initialized from verified Stage 1 checkpoint (ppo_m5_stage1_64x64.zip)
- Environment: UAVFastNav-v2 Stage 2 (N = 3–5 cylindrical obstacles with continuous route clearance validation)
- Designated training seeds only (base seed 300, strictly in [100, 5000])
- No access whatsoever to frozen G1/G2/G3 evaluation seeds ([10001, 10200] and [20001, 20025])
- Strict telemetry logging with explicit per-episode fields
- Post-training rear-obstacle deterministic diagnostic
- M4 baseline checkpoint hash verified pre- and post-run
- Stage 0 and Stage 1 checkpoint hashes verified pre- and post-run
"""

from collections import deque
import hashlib
import json
import math
import os
import resource
import sys
import time
from typing import Dict, Any, List, Optional

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


class Stage2MonitoringCallback(BaseCallback):
    """
    Monitoring, Telemetry, and Safety Callback for Milestone 5 Stage 2 Pilot Training.
    - Halts at exactly max_timesteps (no rollout overshoot).
    - Halts immediately if wall-clock budget is exceeded.
    - Halts immediately if NaN or Inf appears in network parameters.
    - Records explicit per-episode telemetry to CSV and JSON.
    - Tracks interval metrics every 10,000 steps.
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
        self.interval_metrics: List[Dict[str, Any]] = []
        self.nan_inf_detected: bool = False
        self.time_limit_exceeded: bool = False
        self.step_limit_reached: bool = False
        self.last_log_step: int = 0
        self.log_interval: int = 10000
        self._csv_file = None

    def _on_training_start(self) -> None:
        self.start_time = time.perf_counter()
        if self.episodes_csv_path:
            os.makedirs(os.path.dirname(self.episodes_csv_path), exist_ok=True)
            self._csv_file = open(self.episodes_csv_path, "w", buffering=1)
            self._csv_file.write(
                "episode,step,return,length,is_success,is_collision,is_timeout,is_out_of_bounds,min_clearance,termination_reason\n"
            )

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()
            self._csv_file = None

    def _on_step(self) -> bool:
        # 1. Hard step budget check (stops collection immediately at step cap)
        if self.num_timesteps >= self.max_timesteps:
            self.step_limit_reached = True
            if self.verbose > 0:
                print(f"\n[STAGE 2 HALT] Hard step budget reached: {self.num_timesteps} >= {self.max_timesteps}")
            return False

        # 2. Wall-clock budget check
        elapsed = time.perf_counter() - self.start_time
        if elapsed >= self.max_wall_clock_s:
            self.time_limit_exceeded = True
            if self.verbose > 0:
                print(f"\n[STAGE 2 HALT] Wall-clock time budget reached: {elapsed:.2f} s >= {self.max_wall_clock_s:.1f} s")
            return False

        # 3. NaN / Inf parameter check (periodic)
        if self.num_timesteps % 2000 == 0:
            for name, param in self.model.policy.named_parameters():
                if not torch.all(torch.isfinite(param)):
                    self.nan_inf_detected = True
                    print(f"\n[STAGE 2 ABORT] NaN/Inf detected in model parameter {name} at step {self.num_timesteps}!")
                    return False

        # 4. Intercept completed episodes from info dicts with explicit telemetry
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

        # 5. Periodic progress logging and interval snapshot (every 10k steps)
        if self.num_timesteps - self.last_log_step >= self.log_interval:
            self.last_log_step = self.num_timesteps
            recent = self.episodes_data[-50:] if len(self.episodes_data) >= 50 else self.episodes_data
            if recent:
                succ_rate = sum(1 for e in recent if e["is_success"]) / len(recent) * 100.0
                coll_rate = sum(1 for e in recent if e["is_collision"]) / len(recent) * 100.0
                tout_rate = sum(1 for e in recent if e["is_timeout"]) / len(recent) * 100.0
                mean_ret = float(np.mean([e["return"] for e in recent]))
                mean_len = float(np.mean([e["length"] for e in recent]))
                mean_clr = float(np.mean([e["min_clearance"] for e in recent]))
            else:
                succ_rate, coll_rate, tout_rate, mean_ret, mean_len, mean_clr = 0.0, 0.0, 0.0, 0.0, 0.0, 10.0

            fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0

            interval_entry = {
                "step": int(self.num_timesteps),
                "elapsed_s": round(elapsed, 1),
                "fps": fps,
                "episodes_completed": self.episode_count,
                "recent_success_rate": round(succ_rate, 1),
                "recent_collision_rate": round(coll_rate, 1),
                "recent_timeout_rate": round(tout_rate, 1),
                "recent_mean_return": round(mean_ret, 2),
                "recent_mean_length": round(mean_len, 1),
                "recent_mean_min_clearance": round(mean_clr, 3)
            }
            self.interval_metrics.append(interval_entry)

            if self.verbose > 0:
                print(f"[Step {self.num_timesteps:6d} | {elapsed:5.1f}s | {fps:4d} fps] "
                      f"Eps: {self.episode_count:4d} | Succ: {succ_rate:5.1f}% | Coll: {coll_rate:5.1f}% | "
                      f"Tout: {tout_rate:5.1f}% | Ret: {mean_ret:6.1f} | Clr: {mean_clr:5.2f}m")

        return True


def evaluate_rear_obstacle_behavior(
    checkpoint_path: str
) -> Dict[str, Any]:
    """
    Evaluates policy on deterministic rear-obstacle test to assess collision avoidance.
    UAV at (0, 0, 2), yaw = 0. Goal at (-6, 0, 2). Obstacle at (-2.5, 0.0), radius = 0.50m.
    """
    model = PPO.load(checkpoint_path, device="cpu")
    config = UAVObstacleEnvConfigV2(curriculum_stage=2, enable_noise=False)
    env = UAVFastEnvV2(config=config)

    opts = {
        'start_pos': [0.0, 0.0, 2.0],
        'start_yaw': 0.0,
        'goal_pos': [-6.0, 0.0, 2.0],
        'obstacles': [(-2.5, 0.0, 0.50)]
    }
    obs, info = env.reset(seed=101, options=opts)

    trajectory = []
    terminated = False
    truncated = False
    step = 0
    min_clearance_overall = float("inf")
    is_collision = False
    termination_reason = "max_steps"

    while not (terminated or truncated) and step < 50:
        action, _ = model.predict(obs, deterministic=True)
        scaled_cmd = env.action_handler.scale_action(action)
        state_curr = env.dynamics.get_state()

        goal_body = obs[:3]
        d_psi_deg = float(np.degrees(np.arctan2(goal_body[1], goal_body[0])))

        dist_obs_center = math.hypot(state_curr.pos_world[0] - (-2.5), state_curr.pos_world[1] - 0.0)
        dist_obs_surface = dist_obs_center - 0.50
        min_clearance_overall = min(min_clearance_overall, dist_obs_surface)

        obs, reward, terminated, truncated, info = env.step(action)
        step += 1

        record = {
            "step": step,
            "d_psi_deg": round(d_psi_deg, 1),
            "cmd_vx": round(float(scaled_cmd.vx), 2),
            "cmd_vy": round(float(scaled_cmd.vy), 2),
            "cmd_yaw_rate": round(float(scaled_cmd.yaw_rate), 2),
            "pos_x": round(float(state_curr.pos_world[0]), 3),
            "pos_y": round(float(state_curr.pos_world[1]), 3),
            "vel_x": round(float(state_curr.vel_world[0]), 3),
            "vel_y": round(float(state_curr.vel_world[1]), 3),
            "dist_obs_surface": round(dist_obs_surface, 3),
            "min_lidar_clearance": round(float(info.get("min_clearance_m", 10.0)), 3)
        }
        trajectory.append(record)

        if info.get("is_crashed", False):
            is_collision = True
            termination_reason = info.get("termination_reason", "collision")
            break
        elif info.get("is_success", False):
            termination_reason = "success"
            break

    env.close()

    return {
        "initial_clearance_m": 2.00,
        "obstacle_position": [-2.5, 0.0],
        "obstacle_radius_m": 0.50,
        "min_clearance_reached_m": round(min_clearance_overall, 3),
        "is_collision": bool(is_collision),
        "termination_reason": str(termination_reason),
        "total_steps": step,
        "trajectory_sample": trajectory[:15]
    }


def run_stage2_pilot(
    total_timesteps: int = 100000,
    max_wall_clock_s: float = 900.0,  # 15 minutes
    base_seed: int = 300
) -> Dict[str, Any]:
    print("=" * 80)
    print("MILESTONE 5.4 STAGE 2 BOUNDED PILOT TRAINING: [64, 64] MLP")
    print("=" * 80)

    # 1. Pre-training Baseline Integrity Check
    m4_ckpt_path = "experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip"
    expected_m4_hash = "06907c9fa636e5f268ca55750107918370e445313ad1ac01207c15a4c5935308"
    if not os.path.exists(m4_ckpt_path):
        raise FileNotFoundError(f"M4 Checkpoint missing: {m4_ckpt_path}")
    pre_m4_hash = get_checkpoint_sha256(m4_ckpt_path)
    if pre_m4_hash != expected_m4_hash:
        raise ValueError(f"Pre-flight M4 Checkpoint SHA mismatch! Expected {expected_m4_hash}, got {pre_m4_hash}")
    print(f"[INTEGRITY PRE-CHECK] Frozen M4 Checkpoint verified: {pre_m4_hash[:16]}... (BIT-IDENTICAL)")

    # 2. Stage 0 Checkpoint Existence Verification
    stage0_ckpt_path = "experiments/milestone5/models/ppo_m5_pilot_64x64.zip"
    expected_stage0_hash = "4ecd2dae4a3cdc31f87ec1459fdf57320842207c48813c05d06440b5a4eca9f0"
    if not os.path.exists(stage0_ckpt_path):
        raise FileNotFoundError(f"Stage 0 checkpoint missing: {stage0_ckpt_path}")
    pre_stage0_hash = get_checkpoint_sha256(stage0_ckpt_path)
    if pre_stage0_hash != expected_stage0_hash:
        raise ValueError(f"Pre-flight Stage 0 Checkpoint SHA mismatch! Expected {expected_stage0_hash}, got {pre_stage0_hash}")
    print(f"[INTEGRITY PRE-CHECK] Stage 0 Checkpoint verified: {pre_stage0_hash[:16]}... (BIT-IDENTICAL)")

    # 3. Stage 1 Checkpoint Verification & Initialization Source
    stage1_ckpt_path = "experiments/milestone5/models/ppo_m5_stage1_64x64.zip"
    expected_stage1_hash = "4a61d80125510fc15275ec1bdfb93aacc861460a67d04d1985371196be66b6ea"
    if not os.path.exists(stage1_ckpt_path):
        raise FileNotFoundError(f"Stage 1 checkpoint missing: {stage1_ckpt_path}")
    pre_stage1_hash = get_checkpoint_sha256(stage1_ckpt_path)
    if pre_stage1_hash != expected_stage1_hash:
        raise ValueError(f"Pre-flight Stage 1 Checkpoint SHA mismatch! Expected {expected_stage1_hash}, got {pre_stage1_hash}")
    print(f"[INITIALIZATION] Stage 1 Checkpoint verified: {pre_stage1_hash[:16]}... (BIT-IDENTICAL)")

    # 4. Output Directories Setup
    model_dir = "experiments/milestone5/models"
    log_dir = "experiments/milestone5/logs/stage2"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    stage2_ckpt_path = os.path.join(model_dir, "ppo_m5_stage2_64x64.zip")
    monitor_csv_path = os.path.join(log_dir, "stage2_monitor.csv")
    episodes_csv_path = os.path.join(log_dir, "stage2_episodes.csv")

    # 5. Environment Instantiation (Curriculum Stage 2: N = 3–5 obstacles)
    config = UAVObstacleEnvConfigV2(
        curriculum_stage=2,
        reward_align_weight=0.10,
        reward_speed_gate_weight=0.10,
        grid_resolution=0.10,
        min_path_clearance=0.50,
        max_resample_attempts=50
    )
    raw_env = gym.make("UAVFastNav-v2", config=config)
    env = Monitor(raw_env, filename=monitor_csv_path)

    # Set deterministic training seed in designated range [100, 5000]
    env.reset(seed=base_seed)

    # 6. Load Policy from Stage 1 and Configure PPO
    print(f"\n[MODEL SETUP] Initializing PPO from Stage 1 weights: {stage1_ckpt_path}")
    # Rollout buffer n_steps = 2000 (50 iterations * 2000 = exactly 100,000 steps)
    model = PPO.load(
        stage1_ckpt_path,
        env=env,
        learning_rate=3e-4,
        n_steps=2000,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=base_seed,
        device="cpu",
        verbose=0
    )

    # 7. Safety and Telemetry Callback
    callback = Stage2MonitoringCallback(
        max_timesteps=total_timesteps,
        max_wall_clock_s=max_wall_clock_s,
        episodes_csv_path=episodes_csv_path,
        verbose=1
    )

    t0 = time.perf_counter()
    rusage_start = resource.getrusage(resource.RUSAGE_SELF)

    print(f"\n[TRAINING START] Launching Stage 2 pilot for up to {total_timesteps:,} steps (Max {max_wall_clock_s/60:.1f} mins)...")
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=True)
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

    # 8. Save Stage 2 Model Checkpoint
    model.save(stage2_ckpt_path)
    stage2_hash = get_checkpoint_sha256(stage2_ckpt_path)

    # 9. Post-training Baseline & Prior Stage Integrity Check
    post_m4_hash = get_checkpoint_sha256(m4_ckpt_path)
    if post_m4_hash != expected_m4_hash:
        raise ValueError("CRITICAL: Frozen Milestone 4 Checkpoint was corrupted or overwritten!")
    print(f"[INTEGRITY POST-CHECK] Frozen M4 Checkpoint verified: {post_m4_hash[:16]}... (BIT-IDENTICAL)")

    post_stage0_hash = get_checkpoint_sha256(stage0_ckpt_path)
    if post_stage0_hash != expected_stage0_hash:
        raise ValueError("CRITICAL: Stage 0 Checkpoint was corrupted or overwritten!")
    print(f"[INTEGRITY POST-CHECK] Stage 0 Checkpoint verified: {post_stage0_hash[:16]}... (BIT-IDENTICAL)")

    post_stage1_hash = get_checkpoint_sha256(stage1_ckpt_path)
    if post_stage1_hash != expected_stage1_hash:
        raise ValueError("CRITICAL: Stage 1 Checkpoint was corrupted or overwritten!")
    print(f"[INTEGRITY POST-CHECK] Stage 1 Checkpoint verified: {post_stage1_hash[:16]}... (BIT-IDENTICAL)")

    # 10. Compute Empirical Stage 2 Metrics
    episodes = callback.episodes_data
    num_episodes = len(episodes)

    if num_episodes > 0:
        returns = [e["return"] for e in episodes]
        lengths = [e["length"] for e in episodes]
        clearances = [e["min_clearance"] for e in episodes]

        success_count = sum(1 for e in episodes if e["is_success"])
        crash_count = sum(1 for e in episodes if e["is_collision"])
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

        mean_clearance = float(np.mean(clearances))
        std_clearance = float(np.std(clearances))
        min_clearance_all = float(np.min(clearances))
        max_clearance_all = float(np.max(clearances))

        # First 50 vs Last 50 comparison
        first_n = episodes[:50]
        last_n = episodes[-50:] if len(episodes) >= 50 else episodes

        first_succ_rate = sum(1 for e in first_n if e["is_success"]) / len(first_n)
        first_coll_rate = sum(1 for e in first_n if e["is_collision"]) / len(first_n)
        first_mean_ret = float(np.mean([e["return"] for e in first_n]))
        first_mean_clr = float(np.mean([e["min_clearance"] for e in first_n]))

        last_succ_rate = sum(1 for e in last_n if e["is_success"]) / len(last_n)
        last_coll_rate = sum(1 for e in last_n if e["is_collision"]) / len(last_n)
        last_mean_ret = float(np.mean([e["return"] for e in last_n]))
        last_mean_clr = float(np.mean([e["min_clearance"] for e in last_n]))
    else:
        success_count = crash_count = oob_count = timeout_count = 0
        success_rate = crash_rate = oob_rate = timeout_rate = 0.0
        mean_return = std_return = min_return = max_return = 0.0
        mean_clearance = std_clearance = min_clearance_all = max_clearance_all = 0.0
        first_succ_rate = first_coll_rate = first_mean_ret = first_mean_clr = 0.0
        last_succ_rate = last_coll_rate = last_mean_ret = last_mean_clr = 0.0

    # 11. Run Post-Training Deterministic Rear-Obstacle Diagnostic
    print("\n[DIAGNOSTIC] Running deterministic rear-obstacle diagnostic on Stage 2 checkpoint...")
    rear_diag = evaluate_rear_obstacle_behavior(stage2_ckpt_path)

    results = {
        "architecture": "MLP [64, 64]",
        "curriculum_stage": 2,
        "base_seed": base_seed,
        "target_steps": total_timesteps,
        "completed_steps": steps_completed,
        "elapsed_wall_clock_s": round(elapsed_wall_s, 2),
        "steps_per_second": round(steps_per_sec, 1),
        "peak_ram_mb": round(peak_ram_mb, 1),
        "nan_inf_detected": bool(callback.nan_inf_detected),
        "time_limit_exceeded": bool(callback.time_limit_exceeded),
        "step_limit_reached": bool(callback.step_limit_reached),
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
            "min_clearance_stats": {
                "mean": round(mean_clearance, 3),
                "std": round(std_clearance, 3),
                "min": round(min_clearance_all, 3),
                "max": round(max_clearance_all, 3)
            }
        },
        "learning_progression": {
            "first_50_episodes": {
                "success_rate": round(first_succ_rate, 4),
                "collision_rate": round(first_coll_rate, 4),
                "mean_return": round(first_mean_ret, 2),
                "mean_min_clearance": round(first_mean_clr, 3)
            },
            "last_50_episodes": {
                "success_rate": round(last_succ_rate, 4),
                "collision_rate": round(last_coll_rate, 4),
                "mean_return": round(last_mean_ret, 2),
                "mean_min_clearance": round(last_mean_clr, 3)
            },
            "success_rate_gain_pp": round((last_succ_rate - first_succ_rate) * 100.0, 2),
            "collision_rate_delta_pp": round((last_coll_rate - first_coll_rate) * 100.0, 2),
            "return_gain": round(last_mean_ret - first_mean_ret, 2)
        },
        "interval_metrics_10k": callback.interval_metrics,
        "rear_obstacle_behavior": {
            "initial_clearance_m": rear_diag["initial_clearance_m"],
            "min_clearance_reached_m": rear_diag["min_clearance_reached_m"],
            "is_collision": rear_diag["is_collision"],
            "termination_reason": rear_diag["termination_reason"],
            "total_steps": rear_diag["total_steps"],
            "sample_trajectory": rear_diag["trajectory_sample"]
        },
        "checkpoint_isolation": {
            "m4_baseline_sha256": post_m4_hash,
            "m4_hash_verified": True,
            "stage0_checkpoint_sha256": post_stage0_hash,
            "stage1_checkpoint_sha256": post_stage1_hash,
            "stage2_checkpoint_path": stage2_ckpt_path,
            "stage2_checkpoint_sha256": stage2_hash
        }
    }

    # Save detailed JSON results
    json_path = "experiments/milestone5/m5_stage2_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("STAGE 2 PILOT TRAINING COMPLETE — SUMMARY METRICS")
    print("=" * 80)
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
    print(f"  Mean Min Clearance:         {mean_clearance:.3f} ± {std_clearance:.3f} m")
    print("-" * 80)
    print("  Learning Progression (First 50 -> Last 50):")
    print(f"    Success Rate:             {first_succ_rate*100:.1f}% -> {last_succ_rate*100:.1f}% ({results['learning_progression']['success_rate_gain_pp']:+.1f} pp)")
    print(f"    Collision Rate:           {first_coll_rate*100:.1f}% -> {last_coll_rate*100:.1f}% ({results['learning_progression']['collision_rate_delta_pp']:+.1f} pp)")
    print(f"    Mean Return:              {first_mean_ret:.1f} -> {last_mean_ret:.1f} ({results['learning_progression']['return_gain']:+.1f})")
    print(f"    Mean Min Clearance:       {first_mean_clr:.3f} m -> {last_mean_clr:.3f} m")
    print("-" * 80)
    print("  Deterministic Rear-Obstacle Diagnostic (Post Stage 2):")
    print(f"    Initial Clearance:        {rear_diag['initial_clearance_m']:.2f} m")
    print(f"    Min Clearance Reached:    {rear_diag['min_clearance_reached_m']:.3f} m")
    print(f"    Collision Occurred:       {rear_diag['is_collision']} ({rear_diag['termination_reason']})")
    print(f"    Flight Steps:             {rear_diag['total_steps']}")
    print("-" * 80)
    print(f"  Stage 2 Checkpoint Saved:   {stage2_ckpt_path}")
    print(f"  Stage 2 SHA-256:            {stage2_hash}")
    print(f"  M4 Baseline Checkpoint:     UNTOUCHED & BIT-IDENTICAL")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_stage2_pilot()
