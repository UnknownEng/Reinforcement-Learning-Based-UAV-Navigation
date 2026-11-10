"""
benchmark_env.py
----------------
Benchmark throughput and latency of the UAVFastNav-v0 environment.
Measures:
  - Total steps, elapsed wall-clock time
  - Steps per second (Hz)
  - Real-time factor (RTF) = (steps * 0.1s) / wall_time
  - Latency per step (mean, min, max, p99)
"""

import json
import os
import time
from pathlib import Path
import psutil
import numpy as np
import gymnasium as gym

import rl_env

MILESTONE2_DIR = Path(__file__).resolve().parents[1] / "experiments" / "milestone2"


def run_benchmark(total_steps: int = 20000, seed: int = 42):
    print(f"--- Running UAVFastNav-v0 Benchmark ({total_steps} steps) ---")
    process = psutil.Process(os.getpid())
    ram_start_mb = process.memory_info().rss / (1024 * 1024)

    env = gym.make("UAVFastNav-v0")
    obs, info = env.reset(seed=seed)

    step_latencies = []
    t_start = time.perf_counter()

    episodes = 0
    for step in range(total_steps):
        action = env.action_space.sample()
        t0 = time.perf_counter()
        obs, reward, terminated, truncated, info = env.step(action)
        t1 = time.perf_counter()
        step_latencies.append((t1 - t0) * 1000.0)  # ms

        if terminated or truncated:
            episodes += 1
            obs, info = env.reset()

    t_end = time.perf_counter()
    env.close()

    elapsed_wall_s = t_end - t_start
    steps_per_sec = total_steps / elapsed_wall_s
    simulated_time_s = total_steps * 0.10  # 10 Hz policy step = 0.1s
    rtf = simulated_time_s / elapsed_wall_s

    latencies_arr = np.array(step_latencies)
    mean_lat_ms = float(np.mean(latencies_arr))
    p50_lat_ms = float(np.median(latencies_arr))
    p95_lat_ms = float(np.percentile(latencies_arr, 95))
    p99_lat_ms = float(np.percentile(latencies_arr, 99))
    max_lat_ms = float(np.max(latencies_arr))

    ram_end_mb = process.memory_info().rss / (1024 * 1024)

    benchmark_results = {
        "benchmark_environment": "UAVFastNav-v0",
        "policy_frequency_hz": 10.0,
        "physics_substep_hz": 100.0,
        "substeps_per_step": 10,
        "total_steps_executed": total_steps,
        "episodes_completed": episodes,
        "elapsed_wall_time_s": elapsed_wall_s,
        "simulated_time_s": simulated_time_s,
        "throughput_steps_per_sec": steps_per_sec,
        "real_time_factor_rtf": rtf,
        "latency_step_mean_ms": mean_lat_ms,
        "latency_step_p50_ms": p50_lat_ms,
        "latency_step_p95_ms": p95_lat_ms,
        "latency_step_p99_ms": p99_lat_ms,
        "latency_step_max_ms": max_lat_ms,
        "ram_start_mb": ram_start_mb,
        "ram_end_mb": ram_end_mb
    }

    print("\n=== Benchmark Summary ===")
    print(f"Total Steps:        {total_steps:,}")
    print(f"Wall-Clock Time:    {elapsed_wall_s:.2f} s")
    print(f"Throughput:         {steps_per_sec:.1f} steps/s")
    print(f"Real-Time Factor:   {rtf:.1f}x real-time")
    print(f"Step Latency:       mean={mean_lat_ms:.3f} ms | p95={p95_lat_ms:.3f} ms | p99={p99_lat_ms:.3f} ms")
    print(f"RAM Usage:          {ram_end_mb:.1f} MB (Delta: {ram_end_mb - ram_start_mb:+.1f} MB)")

    out_path = MILESTONE2_DIR / "env_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"Results written to: {out_path}")
    return benchmark_results


if __name__ == "__main__":
    run_benchmark(total_steps=20000, seed=42)
