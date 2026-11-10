"""
evaluate_gate_b3.py
-------------------
Milestone 4 Phase 4B — Gate B3: Standardized 100-Episode Benchmark Evaluation.

Evaluates the trained 26-D perception-augmented PPO policy against:
  - 100 standardized test episodes across 4 difficulty tiers (Topic 18):
      * Tier 1 (Free Space): Seeds 20000..20024 (25 episodes, N_obs = 0)
      * Tier 2 (Single Obstacle): Seeds 20025..20049 (25 episodes, N_obs = 1)
      * Tier 3 (Corridor / Gates): Seeds 20050..20074 (25 episodes, N_obs = 3-4)
      * Tier 4 (Dense Clutter): Seeds 20075..20099 (25 episodes, N_obs = 6-8)
  - Computes all quantitative metrics from Topic 20:
      1. Waypoint Success Rate (%) [Target >= 90%]
      2. Obstacle Collision Rate (%) [Target < 5%]
      3. Minimum Obstacle Clearance (m) [Target >= 0.40m]
      4. Path Length (m) and Path Efficiency (L / L_straight) [Target <= 1.30]
      5. Mission Duration (s) [Target <= 15.0s]
      6. Terminal Arrival Error (m) [Target <= 0.40m]
      7. Terminal Arrival Speed (m/s) [Target <= 0.50m/s]
      8. Inference Latency on CPU (ms) [Target < 1.0ms]
  - Performs baseline comparison against Frozen 18-D PPO (Baseline 2).
  - Records cryptographic checksums and validates baseline non-interference.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, "/home/kali/proj")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

import rl_env


def evaluate_tier_episodes(
    policy,
    env: gym.Env,
    seeds: List[int],
    stage: int,
    is_18d: bool = False
) -> Dict[str, Any]:
    """
    Evaluates policy over a specific list of seeds and curriculum stage.
    """
    ep_rewards: List[float] = []
    ep_lengths: List[int] = []
    final_dists: List[float] = []
    final_speeds: List[float] = []
    min_clearances: List[float] = []
    path_lengths: List[float] = []
    path_efficiencies: List[float] = []
    latencies_ms: List[float] = []

    successes = 0
    collisions = 0
    out_of_bounds = 0
    timeouts = 0

    for seed in seeds:
        opts = {"curriculum_stage": stage}
        obs, info = env.reset(seed=seed, options=opts)
        ep_rew = 0.0
        step = 0
        min_clear = min(10.0, float(info.get("min_surface_dist_m", 10.0)))
        init_dist = info["dist_to_goal"]

        while True:
            # Measure single-step forward inference latency
            t0 = time.perf_counter()
            if is_18d:
                # 18-D policy receives only proprioceptive slice
                obs_input = obs[:18]
                action, _ = policy.predict(obs_input, deterministic=True)
            else:
                action, _ = policy.predict(obs, deterministic=True)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            obs, rew, terminated, truncated, info = env.step(action)
            ep_rew += float(rew)
            step += 1

            # Track minimum distance to obstacle surface
            surf_dist = min(10.0, float(info.get("min_surface_dist_m", 10.0)))
            if surf_dist < min_clear:
                min_clear = surf_dist

            if terminated or truncated:
                final_dist = float(info.get("final_distance", info.get("dist_to_goal", 0.0)))
                final_vel = info.get("final_velocity", np.zeros(3))
                speed = float(np.linalg.norm(final_vel))
                path_len = float(info.get("path_length", 0.0))
                eff = (path_len / init_dist) if init_dist > 1e-3 else 1.0

                if info.get("is_success", False):
                    successes += 1
                elif info.get("is_crashed", False):
                    collisions += 1
                elif info.get("is_out_of_bounds", False):
                    out_of_bounds += 1
                elif truncated:
                    timeouts += 1

                ep_rewards.append(ep_rew)
                ep_lengths.append(step)
                final_dists.append(final_dist)
                final_speeds.append(speed)
                min_clearances.append(min_clear)
                path_lengths.append(path_len)
                path_efficiencies.append(eff)
                break

    n = len(seeds)
    return {
        "num_episodes": n,
        "success_rate_pct": float(100.0 * successes / n),
        "collision_rate_pct": float(100.0 * collisions / n),
        "out_of_bounds_rate_pct": float(100.0 * out_of_bounds / n),
        "timeout_rate_pct": float(100.0 * timeouts / n),
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "mean_final_distance_m": float(np.mean(final_dists)),
        "mean_final_speed_mps": float(np.mean(final_speeds)),
        "mean_min_clearance_m": float(np.mean(min_clearances)),
        "min_clearance_overall_m": float(np.min(min_clearances)),
        "mean_path_length_m": float(np.mean(path_lengths)),
        "mean_path_efficiency": float(np.mean(path_efficiencies)),
        "mean_duration_s": float(np.mean(ep_lengths) * 0.10),
        "mean_inference_latency_ms": float(np.mean(latencies_ms)),
        "max_inference_latency_ms": float(np.max(latencies_ms))
    }


def run_gate_b3_benchmark(
    candidate_model_path: str = "/home/kali/proj/experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip",
    baseline_model_path: str = "/home/kali/proj/experiments/milestone2/models/ppo_uav_final.zip",
    output_path: str = "/home/kali/proj/experiments/milestone4/phase4b_gate_b3/gate_b3_evaluation_results.json"
) -> Dict[str, Any]:
    """
    Executes the 100-episode 4-tier benchmark comparison between 26-D Candidate and 18-D Baseline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.set_num_threads(1)  # Real-time single-thread CPU execution matching flight executor

    # 1. Baseline Integrity Checks
    expected_baseline_hash = "8d7a24eda075cbbeb60c4a6df07029c3a38fd770ce436a6daf96feff85300dfc"
    with open(baseline_model_path, "rb") as f:
        actual_base_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_base_hash != expected_baseline_hash:
        raise RuntimeError(f"Baseline hash corrupted! Expected {expected_baseline_hash}, got {actual_base_hash}")

    with open(candidate_model_path, "rb") as f:
        cand_hash = hashlib.sha256(f.read()).hexdigest()

    # 2. Load Models & Perform Warmup
    print("=" * 70)
    print("MILESTONE 4 PHASE 4B: GATE B3 BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"Loading Candidate 26-D Model: {candidate_model_path} (SHA-256: {cand_hash[:16]}...)")
    cand_model = PPO.load(candidate_model_path, device="cpu")
    # Single warmup pass to allocate internal PyTorch inference buffers
    cand_model.predict(np.zeros(26, dtype=np.float32), deterministic=True)

    print(f"Loading Baseline 18-D Model:  {baseline_model_path} (SHA-256: {actual_base_hash[:16]}...)")
    base_model = PPO.load(baseline_model_path, device="cpu")
    base_model.predict(np.zeros(18, dtype=np.float32), deterministic=True)

    # 3. Define Standardized 100-Episode Tiers (Topic 18)
    tiers = {
        "Tier 1 (Free Space)": {
            "seeds": list(range(20000, 20025)),
            "stage": 0,
            "description": "25 episodes in unobstructed open space (N_obs = 0)"
        },
        "Tier 2 (Single Obstacle)": {
            "seeds": list(range(20025, 20050)),
            "stage": 1,
            "description": "25 episodes with 1 obstacle on flight path (N_obs = 1)"
        },
        "Tier 3 (Corridor / Gates)": {
            "seeds": list(range(20050, 20075)),
            "stage": 2,
            "description": "25 episodes in 3-4 cylinder corridors / gates"
        },
        "Tier 4 (Dense Clutter)": {
            "seeds": list(range(20075, 20100)),
            "stage": 3,
            "description": "25 episodes in dense clutter (6-8 cylinders)"
        }
    }

    env = gym.make("UAVFastNav-v1")

    candidate_tier_results: Dict[str, Any] = {}
    baseline_tier_results: Dict[str, Any] = {}

    print("\n--- Evaluating Candidate 26-D Obstacle-Aware PPO ---")
    for tier_name, tier_cfg in tiers.items():
        print(f"  Evaluating {tier_name} (seeds {tier_cfg['seeds'][0]}..{tier_cfg['seeds'][-1]})...")
        res = evaluate_tier_episodes(
            policy=cand_model,
            env=env,
            seeds=tier_cfg["seeds"],
            stage=tier_cfg["stage"],
            is_18d=False
        )
        candidate_tier_results[tier_name] = res
        print(f"    Success: {res['success_rate_pct']:.1f}% | Collisions: {res['collision_rate_pct']:.1f}% | Min Clearance: {res['mean_min_clearance_m']:.2f}m")

    print("\n--- Evaluating Baseline 18-D PPO (Blind Proprioception) ---")
    for tier_name, tier_cfg in tiers.items():
        print(f"  Evaluating {tier_name} (seeds {tier_cfg['seeds'][0]}..{tier_cfg['seeds'][-1]})...")
        res = evaluate_tier_episodes(
            policy=base_model,
            env=env,
            seeds=tier_cfg["seeds"],
            stage=tier_cfg["stage"],
            is_18d=True
        )
        baseline_tier_results[tier_name] = res
        print(f"    Success: {res['success_rate_pct']:.1f}% | Collisions: {res['collision_rate_pct']:.1f}% | Min Clearance: {res['mean_min_clearance_m']:.2f}m")

    env.close()

    # 4. Compute Aggregate 100-Episode Metrics
    def aggregate_metrics(tier_dict: Dict[str, Any]) -> Dict[str, Any]:
        total_ep = sum(t["num_episodes"] for t in tier_dict.values())
        tot_success = sum(t["success_rate_pct"] * t["num_episodes"] / 100.0 for t in tier_dict.values())
        tot_collision = sum(t["collision_rate_pct"] * t["num_episodes"] / 100.0 for t in tier_dict.values())
        tot_oob = sum(t["out_of_bounds_rate_pct"] * t["num_episodes"] / 100.0 for t in tier_dict.values())
        tot_timeout = sum(t["timeout_rate_pct"] * t["num_episodes"] / 100.0 for t in tier_dict.values())

        return {
            "total_episodes": total_ep,
            "overall_success_rate_pct": float(100.0 * tot_success / total_ep),
            "overall_collision_rate_pct": float(100.0 * tot_collision / total_ep),
            "overall_oob_rate_pct": float(100.0 * tot_oob / total_ep),
            "overall_timeout_rate_pct": float(100.0 * tot_timeout / total_ep),
            "mean_reward": float(np.mean([t["mean_reward"] for t in tier_dict.values()])),
            "mean_final_distance_m": float(np.mean([t["mean_final_distance_m"] for t in tier_dict.values()])),
            "mean_min_clearance_m": float(np.mean([t["mean_min_clearance_m"] for t in tier_dict.values()])),
            "min_clearance_overall_m": float(min(t["min_clearance_overall_m"] for t in tier_dict.values())),
            "mean_path_length_m": float(np.mean([t["mean_path_length_m"] for t in tier_dict.values()])),
            "mean_path_efficiency": float(np.mean([t["mean_path_efficiency"] for t in tier_dict.values()])),
            "mean_duration_s": float(np.mean([t["mean_duration_s"] for t in tier_dict.values()])),
            "mean_inference_latency_ms": float(np.mean([t["mean_inference_latency_ms"] for t in tier_dict.values()]))
        }

    cand_agg = aggregate_metrics(candidate_tier_results)
    base_agg = aggregate_metrics(baseline_tier_results)

    # 5. Gate B3 Pass/Fail Acceptance Checks
    pass_success_rate = cand_agg["overall_success_rate_pct"] >= 90.0
    pass_collision_rate = cand_agg["overall_collision_rate_pct"] <= 5.0
    pass_min_clearance = cand_agg["mean_min_clearance_m"] >= 0.40
    pass_latency = cand_agg["mean_inference_latency_ms"] < 1.00
    pass_baseline_safety = cand_agg["overall_collision_rate_pct"] < base_agg["overall_collision_rate_pct"]

    gate_b3_verdict = (
        pass_success_rate and
        pass_collision_rate and
        pass_min_clearance and
        pass_latency
    )

    print("\n" + "=" * 70)
    print("GATE B3 EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Overall Success Rate:      {cand_agg['overall_success_rate_pct']:.1f}% (Target >= 90.0%) -> {'PASS' if pass_success_rate else 'FAIL'}")
    print(f"Overall Collision Rate:    {cand_agg['overall_collision_rate_pct']:.1f}% (Target <= 5.0%)  -> {'PASS' if pass_collision_rate else 'FAIL'}")
    print(f"Mean Obstacle Clearance:   {cand_agg['mean_min_clearance_m']:.2f} m (Target >= 0.40 m) -> {'PASS' if pass_min_clearance else 'FAIL'}")
    print(f"Mean Path Efficiency:      {cand_agg['mean_path_efficiency']:.3f} (Target <= 1.30)")
    print(f"Mean Inference Latency:    {cand_agg['mean_inference_latency_ms']:.3f} ms (Target < 1.0 ms) -> {'PASS' if pass_latency else 'FAIL'}")
    print(f"Baseline Collision Rate:   {base_agg['overall_collision_rate_pct']:.1f}% (Candidate Safer: {'PASS' if pass_baseline_safety else 'FAIL'})")
    print(f"GATE B3 OVERALL VERDICT:   {'PASS' if gate_b3_verdict else 'FAIL'}")
    print("=" * 70)

    # 6. Save JSON Evidence
    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate_id": "Gate B3: Curriculum Training Evaluation",
        "verdict": "PASS" if gate_b3_verdict else "FAIL",
        "acceptance_criteria": {
            "success_rate_ge_90_pct": {
                "required": 90.0,
                "actual": cand_agg["overall_success_rate_pct"],
                "passed": pass_success_rate
            },
            "collision_rate_le_5_pct": {
                "required": 5.0,
                "actual": cand_agg["overall_collision_rate_pct"],
                "passed": pass_collision_rate
            },
            "min_clearance_ge_0_40m": {
                "required": 0.40,
                "actual": cand_agg["mean_min_clearance_m"],
                "passed": pass_min_clearance
            },
            "inference_latency_lt_1ms": {
                "required": 1.00,
                "actual": cand_agg["mean_inference_latency_ms"],
                "passed": pass_latency
            },
            "safer_than_frozen_18d_baseline": {
                "candidate_collision_rate": cand_agg["overall_collision_rate_pct"],
                "baseline_collision_rate": base_agg["overall_collision_rate_pct"],
                "passed": pass_baseline_safety
            }
        },
        "candidate_26d": {
            "model_path": candidate_model_path,
            "sha256": cand_hash,
            "overall_metrics": cand_agg,
            "tier_metrics": candidate_tier_results
        },
        "baseline_18d": {
            "model_path": baseline_model_path,
            "sha256": actual_base_hash,
            "overall_metrics": base_agg,
            "tier_metrics": baseline_tier_results
        }
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nBenchmark evidence written to: {output_path}")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Gate B3 Model")
    parser.add_argument("--candidate", type=str, default="/home/kali/proj/experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip")
    parser.add_argument("--baseline", type=str, default="/home/kali/proj/experiments/milestone2/models/ppo_uav_final.zip")
    parser.add_argument("--output", type=str, default="/home/kali/proj/experiments/milestone4/phase4b_gate_b3/gate_b3_evaluation_results.json")
    args = parser.parse_args()

    run_gate_b3_benchmark(
        candidate_model_path=args.candidate,
        baseline_model_path=args.baseline,
        output_path=args.output
    )
