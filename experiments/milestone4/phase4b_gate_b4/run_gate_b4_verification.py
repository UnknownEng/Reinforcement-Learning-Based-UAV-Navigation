"""
run_gate_b4_verification.py
---------------------------
Milestone 4 Phase 4B — Gate B4: Comprehensive Standalone Benchmark & Stress Evaluation.

Generates gate_b4_verification_results.json comparing:
  1. Candidate: Retrained 26-D PPO (ppo_uav_obstacle_26d_final.zip)
  2. Baseline 1: Classical Guidance (P-Controller + VFH)
  3. Baseline 2: Frozen 18-D PPO + Phase 4A Supervisory Brake (ppo_uav_final.zip)
Across:
  - 100 standardized test benchmark episodes across 4 tiers
  - Topic 21 Stress Tests:
      * Stress Test 1: Sensor noise (0.05m) & random beam dropouts (5%)
      * Stress Test 2: Narrow corridor / gate (1.10m gap)
      * Stress Test 3: Extreme dense clutter (10 obstacles)
  - Baseline invariants and cryptographic hash verification
"""

import datetime
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

sys.path.insert(0, "/home/kali/proj")

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

import rl_env
from experiments.milestone4.phase4b_gate_b4.test_gate_b4_benchmark import (
    ClassicalVFHController,
    Governed18DPolicy,
)


def run_benchmark():
    print("=" * 70)
    print("Milestone 4 Phase 4B — Gate B4: Standalone Benchmark & Stress Evaluation")
    print("=" * 70)

    # 1. Baseline Invariant Checks
    base_model_path = "/home/kali/proj/experiments/milestone2/models/ppo_uav_final.zip"
    expected_base_hash = "8d7a24eda075cbbeb60c4a6df07029c3a38fd770ce436a6daf96feff85300dfc"
    cand_model_path = "/home/kali/proj/experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip"

    with open(base_model_path, "rb") as f:
        base_hash = hashlib.sha256(f.read()).hexdigest()
    assert base_hash == expected_base_hash, f"Baseline hash mismatch: {base_hash}"

    with open(cand_model_path, "rb") as f:
        cand_hash = hashlib.sha256(f.read()).hexdigest()

    tag_commit = subprocess.check_output(
        ["git", "rev-parse", "milestone4-phase4a-frozen^{commit}"]
    ).decode().strip()
    assert tag_commit == "31c81c77aaea82ac35a34fb102e3128257b09b30", "Frozen tag mismatch"

    diff = subprocess.check_output([
        "git", "diff", "milestone4-phase4a-frozen", "--",
        "ppo_adapter.py", "ppo_inference_adapter.py", "rl_env/observation.py", "rl_env/config.py"
    ]).decode().strip()
    assert len(diff) == 0, "Protected production files modified!"

    torch.set_num_threads(1)
    cand_model = PPO.load(cand_model_path, device="cpu")

    vfh = ClassicalVFHController()
    gov18d = Governed18DPolicy(base_model_path)

    # Warmup
    dummy_26d = np.zeros(26, dtype=np.float32)
    dummy_26d[18:26] = 5.0
    for _ in range(10):
        cand_model.predict(dummy_26d, deterministic=True)

    # Standardized 100-episode benchmark across 4 tiers
    tier_definitions = [
        {"tier": 1, "name": "Free Space", "seeds": list(range(20000, 20025)), "stage": 0},
        {"tier": 2, "name": "Single Obstacle", "seeds": list(range(20025, 20050)), "stage": 1},
        {"tier": 3, "name": "Corridor / Gates", "seeds": list(range(20050, 20075)), "stage": 2},
        {"tier": 4, "name": "Dense Clutter", "seeds": list(range(20075, 20100)), "stage": 3},
    ]

    env = gym.make("UAVFastNav-v1")

    controllers = {
        "candidate_26d": {"name": "Candidate 26-D PPO", "type": "cand"},
        "baseline1_vfh": {"name": "Baseline 1 Classical VFH", "type": "vfh"},
        "baseline2_gov18d": {"name": "Baseline 2 Governed 18-D PPO", "type": "gov18d"}
    }

    tier_results = {c: {} for c in controllers}
    overall_results = {c: {
        "successes": 0, "collisions": 0, "timeouts": 0, "geofence": 0,
        "clearances": [], "path_efficiencies": [], "mission_durations": [],
        "arrival_errors": [], "latencies_ms": []
    } for c in controllers}

    print("\n[Executing 100-Episode Comparative Benchmark]")
    for t_def in tier_definitions:
        tier_num = t_def["tier"]
        tier_name = t_def["name"]
        seeds = t_def["seeds"]
        stage = t_def["stage"]

        print(f"\n--- Tier {tier_num}: {tier_name} (25 episodes) ---")

        for c_key, c_info in controllers.items():
            succ = 0
            coll = 0
            to = 0
            geo = 0
            clearances = []
            efficiencies = []
            durations = []
            errors = []
            latencies = []

            for seed in seeds:
                obs, info = env.reset(seed=seed, options={"curriculum_stage": stage})
                min_clear = min(10.0, float(info.get("min_surface_dist_m", 10.0)))
                step_in_ep = 0

                while True:
                    t0 = time.perf_counter()
                    if c_info["type"] == "cand":
                        action, _ = cand_model.predict(obs, deterministic=True)
                    elif c_info["type"] == "vfh":
                        action = vfh.predict(obs, deterministic=True)
                    elif c_info["type"] == "gov18d":
                        action = gov18d.predict(obs, deterministic=True)
                    t1 = time.perf_counter()
                    latencies.append((t1 - t0) * 1000.0)

                    obs, rew, term, trunc, info = env.step(action)
                    step_in_ep += 1
                    sd = min(10.0, float(info.get("min_surface_dist_m", 10.0)))
                    if sd < min_clear:
                        min_clear = sd

                    if term or trunc:
                        init_d = max(0.1, info.get("init_distance", 5.0))
                        path_len = info.get("path_length", init_d)
                        final_d = info.get("final_distance", info.get("dist_to_goal", 0.0))
                        steps = step_in_ep

                        clearances.append(min_clear)
                        efficiencies.append(path_len / init_d)
                        durations.append(steps * 0.10)
                        errors.append(final_d)

                        if info.get("is_success", False):
                            succ += 1
                        elif info.get("is_crashed", False):
                            coll += 1
                        elif trunc:
                            to += 1
                        else:
                            geo += 1
                        break

            n_eps = len(seeds)
            s_rate = 100.0 * succ / n_eps
            c_rate = 100.0 * coll / n_eps
            mean_eff = float(np.mean(efficiencies))

            tier_results[c_key][f"tier_{tier_num}"] = {
                "name": tier_name,
                "episodes": n_eps,
                "success_rate_pct": s_rate,
                "collision_rate_pct": c_rate,
                "mean_clearance_m": float(np.mean(clearances)),
                "path_efficiency": mean_eff,
                "mean_mission_duration_s": float(np.mean(durations)),
                "mean_terminal_error_m": float(np.mean(errors)),
                "mean_latency_ms": float(np.mean(latencies))
            }

            overall_results[c_key]["successes"] += succ
            overall_results[c_key]["collisions"] += coll
            overall_results[c_key]["timeouts"] += to
            overall_results[c_key]["geofence"] += geo
            overall_results[c_key]["clearances"].extend(clearances)
            overall_results[c_key]["path_efficiencies"].extend(efficiencies)
            overall_results[c_key]["mission_durations"].extend(durations)
            overall_results[c_key]["arrival_errors"].extend(errors)
            overall_results[c_key]["latencies_ms"].extend(latencies)

            print(f"  {c_info['name']:<30} | Success: {s_rate:5.1f}% | Collisions: {c_rate:5.1f}% | Eff: {mean_eff:5.3f}")

    # Compile overall benchmark metrics
    benchmark_summary = {}
    for c_key, c_info in controllers.items():
        res = overall_results[c_key]
        n_tot = 100
        benchmark_summary[c_key] = {
            "controller_name": c_info["name"],
            "episodes": n_tot,
            "success_rate_pct": float(100.0 * res["successes"] / n_tot),
            "collision_rate_pct": float(100.0 * res["collisions"] / n_tot),
            "timeout_rate_pct": float(100.0 * res["timeouts"] / n_tot),
            "geofence_rate_pct": float(100.0 * res["geofence"] / n_tot),
            "mean_min_clearance_m": float(np.mean(res["clearances"])),
            "mean_path_efficiency": float(np.mean(res["path_efficiencies"])),
            "mean_mission_duration_s": float(np.mean(res["mission_durations"])),
            "mean_arrival_error_m": float(np.mean(res["arrival_errors"])),
            "mean_latency_ms": float(np.mean(res["latencies_ms"])),
            "p99_latency_ms": float(np.percentile(res["latencies_ms"], 99))
        }

    # Stress Test 1: Noise & Dropouts
    print("\n[Executing Topic 21 Stress Test 1: Sensor Noise & Dropouts]")
    rng = np.random.default_rng(12345)
    stress1_seeds = list(range(21000, 21025))
    s1_succ, s1_coll = 0, 0
    for seed in stress1_seeds:
        obs, info = env.reset(seed=seed, options={"curriculum_stage": 2})
        while True:
            noisy_obs = obs.copy()
            noise = rng.normal(0.0, 0.05, size=8).astype(np.float32)
            noisy_obs[18:26] = np.clip(noisy_obs[18:26] + noise, 0.10, 10.0)
            if rng.random() < 0.05:
                drop_idx = rng.integers(0, 8)
                noisy_obs[18 + drop_idx] = 10.0
            action, _ = cand_model.predict(noisy_obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            if term or trunc:
                if info.get("is_success", False):
                    s1_succ += 1
                elif info.get("is_crashed", False):
                    s1_coll += 1
                break
    stress1_succ_rate = 100.0 * s1_succ / len(stress1_seeds)
    stress1_coll_rate = 100.0 * s1_coll / len(stress1_seeds)
    print(f"  Stress Test 1: Success {stress1_succ_rate:.1f}% | Collisions {stress1_coll_rate:.1f}%")

    # Stress Test 2: Narrow Gate (1.1m)
    print("\n[Executing Topic 21 Stress Test 2: Narrow Gate (1.1m gap)]")
    stress2_seeds = list(range(22000, 22025))
    s2_succ, s2_coll = 0, 0
    for seed in stress2_seeds:
        custom_obs = [(3.0, 0.75, 0.20), (3.0, -0.75, 0.20)]
        opts = {"start_pos": [0.0, 0.0, 2.0], "goal_pos": [6.0, 0.0, 2.0], "obstacles": custom_obs}
        obs, info = env.reset(seed=seed, options=opts)
        while True:
            action, _ = cand_model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            if term or trunc:
                if info.get("is_success", False):
                    s2_succ += 1
                elif info.get("is_crashed", False):
                    s2_coll += 1
                break
    stress2_succ_rate = 100.0 * s2_succ / len(stress2_seeds)
    stress2_coll_rate = 100.0 * s2_coll / len(stress2_seeds)
    print(f"  Stress Test 2: Success {stress2_succ_rate:.1f}% | Collisions {stress2_coll_rate:.1f}%")

    # Stress Test 3: Extreme Clutter (10 obstacles)
    print("\n[Executing Topic 21 Stress Test 3: Extreme Dense Clutter (10 obstacles)]")
    stress3_seeds = list(range(23000, 23025))
    s3_succ, s3_coll = 0, 0
    for seed in stress3_seeds:
        opts = {"curriculum_stage": 3, "num_obstacles": 10}
        obs, info = env.reset(seed=seed, options=opts)
        while True:
            action, _ = cand_model.predict(obs, deterministic=True)
            obs, rew, term, trunc, info = env.step(action)
            if term or trunc:
                if info.get("is_success", False):
                    s3_succ += 1
                elif info.get("is_crashed", False):
                    s3_coll += 1
                break
    stress3_succ_rate = 100.0 * s3_succ / len(stress3_seeds)
    stress3_coll_rate = 100.0 * s3_coll / len(stress3_seeds)
    print(f"  Stress Test 3: Success {stress3_succ_rate:.1f}% | Collisions {stress3_coll_rate:.1f}%")

    env.close()

    # Pass Criteria Check
    cand_summary = benchmark_summary["candidate_26d"]
    pass_criteria = {
        "success_rate_ge_90": cand_summary["success_rate_pct"] >= 90.0,
        "collision_rate_le_5": cand_summary["collision_rate_pct"] <= 5.0,
        "mean_clearance_ge_04": cand_summary["mean_min_clearance_m"] >= 0.40,
        "path_efficiency_le_130": cand_summary["mean_path_efficiency"] <= 1.30,
        "latency_le_1ms": cand_summary["p99_latency_ms"] < 1.0,
        "baseline_beat": cand_summary["collision_rate_pct"] < benchmark_summary["baseline2_gov18d"]["collision_rate_pct"],
        "stress_test1_pass": stress1_succ_rate >= 70.0,
        "stress_test2_pass": stress2_succ_rate >= 50.0,
        "stress_test3_pass": stress3_succ_rate >= 75.0,
        "frozen_invariants_pass": True
    }
    all_passed = all(pass_criteria.values())
    verdict = "PASS" if all_passed else "FAIL"

    output_data = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gate_id": "Gate B4: Standalone Benchmark & Stress Evaluation",
        "verdict": verdict,
        "pass_criteria": pass_criteria,
        "baseline_integrity": {
            "baseline_sha256": base_hash,
            "expected_baseline_sha256": expected_base_hash,
            "candidate_sha256": cand_hash,
            "frozen_tag_commit": tag_commit,
            "protected_files_diff_zero": len(diff) == 0
        },
        "tri_policy_summary": benchmark_summary,
        "tier_breakdown": tier_results,
        "topic_21_stress_tests": {
            "stress_test_1_noise_and_dropouts": {
                "noise_std_m": 0.05,
                "dropout_prob": 0.05,
                "episodes": len(stress1_seeds),
                "success_rate_pct": stress1_succ_rate,
                "collision_rate_pct": stress1_coll_rate,
                "pass": stress1_succ_rate >= 70.0
            },
            "stress_test_2_narrow_gate_1_1m": {
                "corridor_gap_m": 1.10,
                "uav_wingspan_m": 0.70,
                "episodes": len(stress2_seeds),
                "success_rate_pct": stress2_succ_rate,
                "collision_rate_pct": stress2_coll_rate,
                "pass": stress2_succ_rate >= 50.0
            },
            "stress_test_3_extreme_clutter_10_obs": {
                "num_obstacles": 10,
                "episodes": len(stress3_seeds),
                "success_rate_pct": stress3_succ_rate,
                "collision_rate_pct": stress3_coll_rate,
                "pass": stress3_succ_rate >= 75.0
            }
        }
    }

    out_json = "/home/kali/proj/experiments/milestone4/phase4b_gate_b4/gate_b4_verification_results.json"
    with open(out_json, "w") as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 70)
    print(f"Gate B4 Verification Verdict: {verdict}")
    print(f"Results written to: {out_json}")
    print("=" * 70)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    run_benchmark()
