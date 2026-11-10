"""
smoke_test_m5_env.py
--------------------
Bounded Smoke Test & Resource Budget Verification for UAVFastNav-v2 (Milestone 5 Phase M5.4).

Measures and verifies:
1. Wall-clock execution time and steps/second throughput.
2. Peak RAM consumption via getrusage.
3. Environment reset success rate and procedural route validation.
4. Complete absence of NaN / Inf in observations, rewards, and info metrics.
5. Checkpoint isolation of frozen Milestone 4 baseline.
"""

import hashlib
import os
import resource
import sys
import time
from typing import Dict, Any
import gymnasium as gym
import numpy as np

import rl_env
from rl_env.uav_fast_env_v2 import UAVFastEnvV2, UAVObstacleEnvConfigV2


def get_checkpoint_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_smoke_test(
    num_episodes: int = 15,
    max_steps_per_episode: int = 200,
    max_wall_clock_s: float = 30.0,
    max_ram_mb: float = 1000.0,
    base_seed: int = 9001
) -> Dict[str, Any]:
    print("=" * 70)
    print("PHASE M5.4 BOUNDED SMOKE TEST & RESOURCE PROFILING")
    print("=" * 70)

    # 0. Measure Baseline Import Footprint
    rusage_baseline = resource.getrusage(resource.RUSAGE_SELF)
    baseline_ram_mb = rusage_baseline.ru_maxrss / 1024.0

    # 1. Baseline Checkpoint Isolation Verification
    m4_ckpt_path = "experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip"
    expected_m4_hash = "06907c9fa636e5f268ca55750107918370e445313ad1ac01207c15a4c5935308"
    if not os.path.exists(m4_ckpt_path):
        raise FileNotFoundError(f"M4 Checkpoint missing: {m4_ckpt_path}")
    actual_m4_hash = get_checkpoint_sha256(m4_ckpt_path)
    if actual_m4_hash != expected_m4_hash:
        raise ValueError(f"M4 Checkpoint SHA mismatch! Expected {expected_m4_hash}, got {actual_m4_hash}")
    print(f"[CHECKPOINT ISOLATION] M4 Checkpoint SHA-256 verified: {actual_m4_hash[:16]}... (BIT-IDENTICAL)")

    # 2. Environment Initialization
    config = UAVObstacleEnvConfigV2(
        reward_align_weight=0.10,
        reward_speed_gate_weight=0.10,
        grid_resolution=0.10,
        min_path_clearance=0.50,
        max_resample_attempts=50
    )
    env = gym.make("UAVFastNav-v2", config=config)

    t_start = time.perf_counter()
    rusage_start = resource.getrusage(resource.RUSAGE_SELF)

    total_steps = 0
    nan_inf_count = 0
    resets_attempted = 0
    resets_successful = 0
    reset_durations = []
    episode_lengths = []
    terminations_reasons = []

    # Cycle through curriculum stages 0, 1, 2, 3 across episodes
    stages = [0, 1, 2, 3]

    print(f"[SMOKE EXECUTION] Running {num_episodes} episodes across curriculum stages...")

    for ep_idx in range(num_episodes):
        ep_seed = base_seed + ep_idx
        stage = stages[ep_idx % len(stages)]
        env.unwrapped.set_curriculum_stage(stage)

        resets_attempted += 1
        t_rst0 = time.perf_counter()
        try:
            obs, info = env.reset(seed=ep_seed)
            t_rst1 = time.perf_counter()
            reset_durations.append(t_rst1 - t_rst0)
            resets_successful += 1
        except Exception as e:
            print(f"Reset failed on episode {ep_idx} (seed {ep_seed}): {e}")
            continue

        if not np.all(np.isfinite(obs)):
            nan_inf_count += 1
            print(f"NaN/Inf detected in reset observation! Ep {ep_idx}")

        ep_step = 0
        terminated = False
        truncated = False

        while not (terminated or truncated) and ep_step < max_steps_per_episode:
            # Action sampling with occasional full throttle and pure yaw
            if ep_step % 4 == 0:
                action = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # full forward throttle
            elif ep_step % 4 == 1:
                action = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)  # pure yaw
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_step += 1
            total_steps += 1

            if not np.all(np.isfinite(obs)):
                nan_inf_count += 1
            if not np.isfinite(reward):
                nan_inf_count += 1
            rc = info.get("reward_components", {})
            for k, v in rc.items():
                if not np.isfinite(v):
                    nan_inf_count += 1

        episode_lengths.append(ep_step)
        terminations_reasons.append(info.get("termination_reason", "unknown"))

    t_end = time.perf_counter()
    rusage_end = resource.getrusage(resource.RUSAGE_SELF)

    elapsed_wall_s = t_end - t_start
    steps_per_sec = total_steps / elapsed_wall_s if elapsed_wall_s > 0 else 0.0
    # On Linux ru_maxrss is in kilobytes
    peak_ram_mb = rusage_end.ru_maxrss / 1024.0
    incremental_ram_mb = max(0.0, peak_ram_mb - baseline_ram_mb)

    env.close()

    # Verify M4 Checkpoint Isolation Post-Test
    post_m4_hash = get_checkpoint_sha256(m4_ckpt_path)
    if post_m4_hash != expected_m4_hash:
        raise ValueError("M4 Checkpoint altered during test execution!")

    results = {
        "num_episodes": num_episodes,
        "total_steps": total_steps,
        "elapsed_wall_clock_s": round(elapsed_wall_s, 3),
        "steps_per_second": round(steps_per_sec, 1),
        "baseline_ram_mb": round(baseline_ram_mb, 2),
        "peak_ram_mb": round(peak_ram_mb, 2),
        "incremental_ram_mb": round(incremental_ram_mb, 2),
        "resets_attempted": resets_attempted,
        "resets_successful": resets_successful,
        "reset_success_rate": round(resets_successful / resets_attempted if resets_attempted > 0 else 0.0, 4),
        "mean_reset_ms": round(float(np.mean(reset_durations)) * 1000.0, 2) if reset_durations else 0.0,
        "nan_inf_count": nan_inf_count,
        "checkpoint_isolation_verified": True,
        "m4_checkpoint_sha256": post_m4_hash,
    }

    print("-" * 70)
    print("SMOKE TEST RESULTS SUMMARY:")
    print(f"  Total Steps Executed:       {total_steps}")
    print(f"  Wall-Clock Time:           {elapsed_wall_s:.3f} s (Budget: < {max_wall_clock_s:.1f} s)")
    print(f"  Throughput:                {steps_per_sec:.1f} steps/second")
    print(f"  Process Peak RAM:          {peak_ram_mb:.1f} MB (PyTorch Baseline: {baseline_ram_mb:.1f} MB)")
    print(f"  Incremental Env RAM:       {incremental_ram_mb:.2f} MB")
    print(f"  Reset Success Rate:        {results['reset_success_rate'] * 100:.1f}% ({resets_successful}/{resets_attempted})")
    print(f"  Mean Reset Duration:       {results['mean_reset_ms']:.2f} ms")
    print(f"  NaN / Inf Occurrences:     {nan_inf_count} (Must be 0)")
    print(f"  M4 Baseline Checkpoint:    UNTOUCHED & BIT-IDENTICAL")
    print("=" * 70)

    # Assert budget constraints
    assert elapsed_wall_s <= max_wall_clock_s, f"Wall-clock budget exceeded: {elapsed_wall_s:.2f}s > {max_wall_clock_s}s"
    assert peak_ram_mb <= max_ram_mb, f"RAM budget exceeded: {peak_ram_mb:.1f}MB > {max_ram_mb}MB"
    assert incremental_ram_mb <= 50.0, f"Incremental RAM budget exceeded: {incremental_ram_mb:.1f}MB > 50.0MB"
    assert nan_inf_count == 0, f"NaN/Inf detected: {nan_inf_count}"
    assert results["reset_success_rate"] == 1.0, f"Reset success rate < 1.0: {results['reset_success_rate']}"

    print("ALL BUDGET AND SAFETY CHECKS PASSED.")
    return results


if __name__ == "__main__":
    res = run_smoke_test()
