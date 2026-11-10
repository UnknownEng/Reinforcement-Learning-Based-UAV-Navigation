#!/usr/bin/env python3
"""
evaluate_m5_shielded_generalization.py
----------------------------------------
Milestone 5 — Phase M5.5 Shielded Held-Out Generalization Evaluator.

Evaluates the shielded Stage 2b candidate policy:
  Checkpoint: experiments/milestone5/models/ppo_m5_stage2b_remediated_64x64.zip
  Safety Shield: rl_env.safety_shield.DeterministicSafetyShield
on the exact 8 held-out generalization challenge suites across deterministic
seeds 10001–10200 (200 episodes).

Protocol Fidelity:
- Reuses exact procedural suite generation from evaluate_m5_generalization_baseline.py.
- Applies the verified DeterministicSafetyShield configuration (d_base=0.85m, k_delay=0.25s).
- Evaluates CPU inference latency including both policy forward pass and shield projection.
- Pre- and post-evaluation SHA-256 integrity checks are enforced on all checkpoints.
- Zero training, zero fine-tuning, zero weight modifications.
"""

import os
import sys
import time
import json
import math
import hashlib
import argparse
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO

import rl_env
from rl_env.safety_shield import DeterministicSafetyShield, SafetyShieldConfig
from experiments.milestone5.evaluate_m5_generalization_baseline import (
    generate_heldout_suite_config,
    get_file_sha256
)

# Authoritative Checkpoint Paths & Expected SHA-256 Hashes
STAGE2B_MODEL_PATH = "/home/kali/proj/experiments/milestone5/models/ppo_m5_stage2b_remediated_64x64.zip"
EXPECTED_STAGE2B_HASH = "c2197f7052e90fa016cf90319677834197fc7139f367631843a6f36b8f5b2ae0"

M4_MODEL_PATH = "/home/kali/proj/experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip"
EXPECTED_M4_HASH = "06907c9fa636e5f268ca55750107918370e445313ad1ac01207c15a4c5935308"

# Default Output Paths (Strictly isolated from baseline artifacts)
DEFAULT_OUT_JSON = "/home/kali/proj/experiments/milestone5/m5_shielded_generalization_results.json"
DEFAULT_OUT_MD = "/home/kali/proj/experiments/milestone5/m5_shielded_generalization_report.md"

# Standard Suite Generator Function
def get_evaluation_suites(seed_start: int = 10001, seed_count: int = 200) -> List[Tuple[str, str, range]]:
    """Generates 8 balanced challenge suites across the designated seed range."""
    per_suite = max(1, seed_count // 8)
    names = [
        ('A', 'Unseen Obstacle Positions'),
        ('B', 'Unseen Obstacle Arrangements'),
        ('C', 'Different Obstacle Spacing'),
        ('D', 'Randomized Start/Goal Configurations'),
        ('E', 'Asymmetric Obstacle Layouts'),
        ('F', 'Increased Obstacle Density'),
        ('G', 'Narrow Passages'),
        ('H', 'Multiple Sequential Obstacles'),
    ]
    suites = []
    current_seed = seed_start
    for s_id, s_name in names:
        suites.append((s_id, s_name, range(current_seed, current_seed + per_suite)))
        current_seed += per_suite
    return suites


def run_episode(
    env: gym.Env,
    model: PPO,
    shield: DeterministicSafetyShield,
    suite_id: str,
    suite_name: str,
    seed: int
) -> Dict[str, Any]:
    """Executes a single shielded evaluation episode with exact protocol tracking."""
    cfg = generate_heldout_suite_config(suite_id, seed)
    opts = {
        'start_pos': cfg['start_pos'],
        'start_yaw': cfg['start_yaw'],
        'goal_pos': cfg['goal_pos'],
        'obstacles': cfg['obstacles']
    }

    obs, info = env.reset(seed=seed, options=opts)

    p0 = np.array(cfg['start_pos'], dtype=np.float32)
    g0 = np.array(cfg['goal_pos'], dtype=np.float32)
    straight_line_dist = float(np.linalg.norm(g0 - p0))

    traj_positions = [p0.copy()]
    step_latencies_total = []
    step_latencies_policy = []
    step_latencies_shield = []
    min_surface_clearance = float(info.get("min_surface_dist_m", 10.0))
    min_lidar_clearance = 10.0
    speeds = []
    watchdog_trips = 0
    nan_inf_found = False

    shield_interventions = 0
    layer1_interventions = 0
    layer2_interventions = 0

    step_count = 0
    done = False

    while not done:
        curr_pos = traj_positions[-1]
        x, y, z = curr_pos[0], curr_pos[1], curr_pos[2]
        if abs(x) > 8.0 or abs(y) > 8.0 or z > 6.0:
            watchdog_trips += 1
        if z < 0.25:
            watchdog_trips += 1

        if np.isnan(obs).any() or np.isinf(obs).any():
            nan_inf_found = True

        # Check planar LiDAR ray clearances in obs[18:26]
        raw_clearances = np.asarray(obs[18:26], dtype=np.float32)
        if len(raw_clearances) == 8:
            step_min_ray = float(np.min(raw_clearances))
            if step_min_ray < min_lidar_clearance:
                min_lidar_clearance = step_min_ray

        # Policy Inference (CPU)
        t0 = time.perf_counter()
        with torch.inference_mode():
            raw_action, _ = model.predict(obs, deterministic=True)
        t1 = time.perf_counter()
        policy_lat_ms = (t1 - t0) * 1000.0

        if np.isnan(raw_action).any() or np.isinf(raw_action).any():
            nan_inf_found = True

        # Safety Shield Action Interception
        t2 = time.perf_counter()
        meta = info.get("perception_metadata", None)
        if not isinstance(meta, dict):
            meta = {}
        else:
            meta = dict(meta)
        # The shield needs the pose for its independent world-frame geofence.
        # info belongs to the current observation, before this action is applied.
        meta["position_world"] = np.asarray(curr_pos, dtype=np.float32).copy()
        action, diag = shield.filter_action(obs, raw_action, perception_metadata=meta)
        t3 = time.perf_counter()
        shield_lat_ms = (t3 - t2) * 1000.0

        if diag.is_intervened:
            shield_interventions += 1
        if diag.layer1_active:
            layer1_interventions += 1
        if diag.layer2_active:
            layer2_interventions += 1

        step_latencies_policy.append(policy_lat_ms)
        step_latencies_shield.append(shield_lat_ms)
        step_latencies_total.append(policy_lat_ms + shield_lat_ms)

        obs, reward, term, trunc, info = env.step(action)
        step_count += 1
        done = term or trunc

        pos_k = np.array(info.get("pos_world", [x, y, z]), dtype=np.float32)
        traj_positions.append(pos_k)

        vel_k = info.get("vel_world", np.zeros(3))
        speed = float(np.linalg.norm(vel_k))
        speeds.append(speed)
        if speed > 4.5:
            watchdog_trips += 1

        c_surf = float(info.get("min_surface_dist_m", 10.0))
        if c_surf < min_surface_clearance:
            min_surface_clearance = c_surf

    traj_arr = np.array(traj_positions)
    path_len = float(np.sum(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1)))
    path_eff = float(path_len / max(0.1, straight_line_dist))
    duration_s = float(step_count * 0.10)

    term_reason = info.get("termination_reason", "UNKNOWN")
    min_hull_dist = float(min_surface_clearance - 0.35)
    is_success = bool(info.get("is_success", False)) or (term_reason == "goal_reached_and_stabilized")
    is_collision = bool(info.get("is_obstacle_collision", False)) or bool(info.get("is_crashed", False)) or ("collision" in term_reason)
    is_phys_hull_collision = bool(min_hull_dist <= 0.0)
    is_lidar_breach = bool(min_lidar_clearance <= 0.35)

    mean_lat_total = float(np.mean(step_latencies_total)) if step_latencies_total else 0.0
    p95_lat_total = float(np.percentile(step_latencies_total, 95)) if step_latencies_total else 0.0
    mean_lat_policy = float(np.mean(step_latencies_policy)) if step_latencies_policy else 0.0
    mean_lat_shield = float(np.mean(step_latencies_shield)) if step_latencies_shield else 0.0

    max_spd = float(np.max(speeds)) if speeds else 0.0
    mean_spd = float(np.mean(speeds)) if speeds else 0.0

    return {
        'seed': seed,
        'suite_id': suite_id,
        'suite_name': suite_name,
        'start_pos': cfg['start_pos'],
        'start_yaw_rad': round(float(cfg['start_yaw']), 4),
        'goal_pos': cfg['goal_pos'],
        'num_obstacles': len(cfg['obstacles']),
        'success': is_success,
        'collision': is_collision,
        'environment_collision_flag': is_collision,
        'physical_hull_collision': is_phys_hull_collision,
        'lidar_clearance_breach': is_lidar_breach,
        'timeout': bool(step_count >= 200 and not is_success and not is_collision and not is_phys_hull_collision),
        # A watchdog/boundary breach is still a breach if the episode later
        # recovers or reaches its goal. Never hide it behind episode outcome.
        'out_of_bounds': bool(watchdog_trips > 0),
        'termination_reason': term_reason,
        'min_ground_truth_clearance_m': round(min_surface_clearance, 4),
        'min_hull_clearance_m': round(min_hull_dist, 5),
        'min_lidar_clearance_m': round(min_lidar_clearance, 5),
        'path_length_m': round(path_len, 3),
        'straight_line_dist_m': round(straight_line_dist, 3),
        'path_efficiency': round(path_eff, 4),
        'episode_duration_sim_s': round(duration_s, 2),
        'step_count': step_count,
        'mean_inference_latency_ms': round(mean_lat_total, 4),
        'p95_inference_latency_ms': round(p95_lat_total, 4),
        'step_latency_samples_ms': [round(float(x), 6) for x in step_latencies_total],
        'policy_mean_latency_ms': round(mean_lat_policy, 4),
        'shield_mean_latency_ms': round(mean_lat_shield, 4),
        'max_speed_mps': round(max_spd, 3),
        'mean_speed_mps': round(mean_spd, 3),
        'watchdog_trips': watchdog_trips,
        'nan_inf_detected': nan_inf_found,
        'shield_interventions': shield_interventions,
        'layer1_interventions': layer1_interventions,
        'layer2_interventions': layer2_interventions,
    }


def generate_markdown_report(data: dict, md_path: str):
    """Generates a structured analytical markdown report for shielded generalization."""
    om = data['overall_metrics']
    ss = data['suite_summaries']
    shield_cfg = data['shield_configuration']

    seeds = [ep['seed'] for ep in data.get('episodes', [])]
    seed_range_str = f"Seeds {min(seeds)}–{max(seeds)}" if seeds else "N/A"

    md = f"""# Milestone 5 — Phase M5.5: Shielded Held-Out Generalization Evaluation Report

**Evaluation Timestamp**: {data['timestamp']}  
**Evaluated Candidate Checkpoint**: `{data['model_path']}`  
**Candidate Checkpoint SHA-256 (Pre-Eval)**: `{data['checkpoint_sha256_pre']}`  
**Candidate Checkpoint SHA-256 (Post-Eval)**: `{data['checkpoint_sha256_post']}`  
**Checkpoint Immutability Verified**: **{data['checkpoint_sha256_pre'] == data['checkpoint_sha256_post']}**  
**Safety Shield Active**: **{data['safety_shield_active']}**  
**Shield Base Clearance ($d_{{\\text{{base}}}}$)**: `{shield_cfg['d_base_m']} m` (Hull: 0.35m + Margin: 0.50m)  
**Shield Effective Delay ($k_{{\\text{{delay}}}}$)**: `{shield_cfg['k_delay_s']} s`  

---

## 1. Executive Summary & Gate G1 Verdict

Phase M5.5 evaluates the **shielded candidate policy** (`{os.path.basename(data['model_path'])}` + `DeterministicSafetyShield`) on the held-out environmental configurations ({seed_range_str}).

A total of **{om['total_episodes']} episodes** were executed across {len(ss)} distinct challenge suites.

### Primary Headline Metrics vs. Gate G1 Criteria
| Metric | Prescribed Threshold | Measured Shielded Value | Status |
| :--- | :---: | :---: | :---: |
| **Success Rate** | $\\ge 80.0\\%$ | **{om['success_rate_pct']}%** ({om['successful_episodes']}/{om['total_episodes']}) | **{"PASS" if om['success_rate_pct'] >= 80.0 else "FAIL"}** |
| **Environment Collision Rate** | $\\le 15.0\\%$ | **{om['collision_rate_pct']}%** ({om['collision_episodes']}/{om['total_episodes']}) | **{"PASS" if om['collision_rate_pct'] <= 15.0 else "FAIL"}** |
| **Physical Hull Collisions ($d_{{\\text{{hull}}}} \\le 0$)** | Exactly $0$ | **{om.get('physical_hull_collision_episodes', 0)}** | **{"PASS" if om.get('physical_hull_collision_episodes', 0) == 0 else "FAIL"}** |
| **LiDAR Clearance Breaches ($c \\le r_{{\\text{{uav}}}}$)** | Informational | **{om.get('lidar_clearance_breach_episodes', 0)}** | — |
| **Timeout Rate** | $\\le 10.0\\%$ | **{round(100.0 * om['timeout_episodes'] / om['total_episodes'], 1)}%** ({om['timeout_episodes']}/{om['total_episodes']}) | **{"PASS" if om['timeout_episodes'] / max(1, om['total_episodes']) <= 0.10 else "FAIL"}** |
| **Out-of-Bounds Rate** | $0.0\\%$ | **{round(100.0 * om['out_of_bounds_episodes'] / om['total_episodes'], 1)}%** ({om['out_of_bounds_episodes']}/{om['total_episodes']}) | **{"PASS" if om['out_of_bounds_episodes'] == 0 else "FAIL"}** |
| **Mean Clearance** | Informational | **{om['clearance_stats_m']['mean']} m** (Min: **{om['clearance_stats_m']['min']} m**) | — |
| **Min Hull Clearance** | $> 0.0\\text{{ m}}$ | **{om['clearance_stats_m'].get('min_hull_clearance', round(om['clearance_stats_m']['min'] - 0.35, 5))} m** | **{"PASS" if om['clearance_stats_m'].get('min_hull_clearance', om['clearance_stats_m']['min'] - 0.35) > 0 else "FAIL"}** |
| **Mean Path Efficiency** | $\\le 1.30$ | **{om['path_efficiency_stats']['mean']}** | **{"PASS" if om['path_efficiency_stats']['mean'] <= 1.30 else "FAIL"}** |
| **Combined Latency** | Mean $<0.80$ ms; P95 $<1.00$ ms | Mean: **{om['latency_stats_ms']['mean']} ms** (P95: **{om['latency_stats_ms']['p95']} ms**) | **{"PASS" if om['latency_stats_ms']['mean'] < 0.80 and om['latency_stats_ms']['p95'] < 1.00 else "FAIL"}** |
| **Watchdog Trips** | $0$ | **{om['total_watchdog_interventions']}** | **{"PASS" if om['total_watchdog_interventions'] == 0 else "FAIL"}** |
| **NaN / Inf Detections** | $0$ | **{om['nan_inf_occurrences']}** | **{"PASS" if om['nan_inf_occurrences'] == 0 else "FAIL"}** |

---

## 2. Disaggregated Performance by Challenge Suite

| Suite | Challenge Name | Episodes | Success | Success % | Env Coll | Hull Coll | LiDAR Breach | Min Clearance | Mean Eff | Mean Latency | Shield Interv |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for s_id, s_info in ss.items():
        md += f"| **{s_id}** | {s_info['suite_name']} | {s_info['total_episodes']} | {s_info['success_count']} | **{s_info['success_rate_pct']}%** | {s_info['collision_count']} | {s_info.get('physical_hull_collision_count', 0)} | {s_info.get('lidar_clearance_breach_count', 0)} | {s_info['min_clearance_m']} m | {s_info['mean_path_efficiency']} | {s_info['mean_latency_ms']} ms | {s_info.get('total_shield_interventions', 0)} |\n"

    md += f"""| **ALL**| **Combined Evaluation Set** | **{om['total_episodes']}** | **{om['successful_episodes']}** | **{om['success_rate_pct']}%** | **{om['collision_episodes']}** | **{om.get('physical_hull_collision_episodes', 0)}** | **{om.get('lidar_clearance_breach_episodes', 0)}** | **{om['clearance_stats_m']['min']} m** | **{om['path_efficiency_stats']['mean']}** | **{om['latency_stats_ms']['mean']} ms** | **{om.get('total_shield_interventions', 0)}** |

---

## 3. Comparison: Unshielded M4 Baseline vs. Shielded Stage 2b

| Evaluation Phase | Policy / System | Success Rate | Collision Rate | Timeout Rate | Min Clearance |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **M5.1 Baseline** | Frozen M4 (Unshielded) | 12.5% | 87.5% | 0.0% | 0.016 m |
| **M5.5 Candidate** | Stage 2b + Deterministic Shield | **{om['success_rate_pct']}%** | **{om['collision_rate_pct']}%** | **{round(100.0 * om['timeout_episodes'] / om['total_episodes'], 1)}%** | **{om['clearance_stats_m']['min']} m** |
| **Net Improvement** | Safety Shield Impact | **{round(om['success_rate_pct'] - 12.5, 1):+}%** | **{round(om['collision_rate_pct'] - 87.5, 1):+}%** | — | — |

---

## 4. Checkpoint Immutability & Integrity Proof
- **Stage 2b Checkpoint SHA-256**: `{data['checkpoint_sha256_post']}` (VERIFIED BIT-IDENTICAL)
- **M4 Baseline Checkpoint SHA-256**: `{data['m4_baseline_sha256']}` (VERIFIED BIT-IDENTICAL)
- Exactly zero gradient updates or training steps occurred during this evaluation.
"""
    with open(md_path, "w") as f:
        f.write(md)


def main():
    parser = argparse.ArgumentParser(description="Milestone 5 Shielded Held-Out Generalization Evaluator")
    parser.add_argument("--model", type=str, default=STAGE2B_MODEL_PATH, help="Path to policy checkpoint")
    parser.add_argument("--expected-hash", type=str, default=None, help="Expected SHA-256 hash of the candidate model")
    parser.add_argument("--seed-start", type=int, default=10001, help="Starting integer seed for suite generation (default: 10001)")
    parser.add_argument("--seed-count", type=int, default=200, help="Total number of episodes across 8 suites (default: 200)")
    parser.add_argument("--safety-margin", type=float, default=0.55, help="Additional obstacle-surface margin beyond the 0.35 m hull (m)")
    parser.add_argument("--smoke-test", action="store_true", help="Execute minimal non-benchmark smoke test")
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUT_JSON, help="Output JSON path")
    parser.add_argument("--output-md", type=str, default=DEFAULT_OUT_MD, help="Output Markdown report path")
    args = parser.parse_args()

    print("=" * 80)
    print("Milestone 5 — Phase M5.5: Shielded Held-Out Generalization Evaluation")
    print(f"Candidate: {args.model}")
    print(f"Seed Allocation: {args.seed_start} to {args.seed_start + args.seed_count - 1} ({args.seed_count} episodes)")
    print("=" * 80)

    # 1. Pre-Evaluation Integrity Verification
    print("\n[Integrity Check] Verifying candidate and baseline checkpoints before evaluation...")
    assert os.path.exists(args.model), f"Candidate model missing at {args.model}"
    pre_sha = get_file_sha256(args.model)
    expected_hash = args.expected_hash if args.expected_hash is not None else EXPECTED_STAGE2B_HASH
    assert pre_sha == expected_hash, f"Candidate model SHA mismatch! Expected {expected_hash}, got {pre_sha}"
    print(f"  Candidate Checkpoint SHA-256 (pre-eval): {pre_sha} (VERIFIED)")

    assert os.path.exists(M4_MODEL_PATH), f"M4 model missing at {M4_MODEL_PATH}"
    m4_sha_pre = get_file_sha256(M4_MODEL_PATH)
    assert m4_sha_pre == EXPECTED_M4_HASH, f"M4 Checkpoint SHA mismatch! Expected {EXPECTED_M4_HASH}, got {m4_sha_pre}"
    print(f"  Milestone 4 Checkpoint SHA-256:         {m4_sha_pre} (VERIFIED BIT-IDENTICAL)")

    # Keep CPU inference deterministic and avoid oversized OpenMP pools on WSL.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    # 2. Load Policy on CPU
    print(f"\n[Model Load] Loading candidate policy on CPU: {args.model}")
    model = PPO.load(args.model, device="cpu")
    model.policy.eval()

    # 3. Initialize Safety Shield with validated R1-R4 configuration
    shield = DeterministicSafetyShield(SafetyShieldConfig(safety_margin=args.safety_margin))
    print(f"[Safety Shield Active] d_base={shield.config.d_base:.2f}m, k_delay={shield.config.k_delay:.2f}s, directional projection=True")

    # 4. Initialize Environment
    env = gym.make("UAVFastNav-v1")

    # Determine execution suites
    if args.smoke_test:
        print("\n[MODE: SMOKE TEST] Running 2 non-benchmark test episodes (Seeds: 99901, 99902)...")
        suites = [
            ('A', 'Unseen Obstacle Positions (Smoke)', [99901]),
            ('B', 'Unseen Obstacle Arrangements (Smoke)', [99902]),
        ]
        out_json_path = args.output_json if args.output_json != DEFAULT_OUT_JSON else "/home/kali/proj/experiments/milestone5/m5_shielded_generalization_smoke.json"
        out_md_path = args.output_md if args.output_md != DEFAULT_OUT_MD else "/home/kali/proj/experiments/milestone5/m5_shielded_generalization_smoke.md"
    else:
        suites = get_evaluation_suites(args.seed_start, args.seed_count)
        out_json_path = args.output_json
        out_md_path = args.output_md

    total_episodes_planned = sum(len(seeds) for _, _, seeds in suites)
    print(f"[Evaluation Plan] Executing {total_episodes_planned} episodes across {len(suites)} suites...")

    episode_records = []
    suite_summaries = {}
    t_start_all = time.time()

    for s_id, s_name, seed_range in suites:
        print(f"\n--- Running Suite {s_id}: {s_name} (Seeds {min(seed_range)} - {max(seed_range)}) ---")
        s_records = []

        for seed in seed_range:
            rec = run_episode(env, model, shield, s_id, s_name, seed)
            s_records.append(rec)
            episode_records.append(rec)

            status_sym = "[PASS]" if rec['success'] else "[FAIL]"
            print(f"  {status_sym} Seed {seed}: reason={rec['termination_reason']} | "
                  f"steps={rec['step_count']} ({rec['episode_duration_sim_s']}s) | "
                  f"min_clr={rec['min_ground_truth_clearance_m']:.3f}m | "
                  f"eff={rec['path_efficiency']:.2f} | lat={rec['mean_inference_latency_ms']:.3f}ms | "
                  f"interventions={rec['shield_interventions']}")

        # Suite aggregate
        s_succ = sum(1 for r in s_records if r['success'])
        s_coll = sum(1 for r in s_records if r['collision'])
        s_phys_coll = sum(1 for r in s_records if r.get('physical_hull_collision', False))
        s_lidar_breach = sum(1 for r in s_records if r.get('lidar_clearance_breach', False))
        s_clrs = [r['min_ground_truth_clearance_m'] for r in s_records]
        s_hull_clrs = [r['min_hull_clearance_m'] for r in s_records]
        s_effs = [r['path_efficiency'] for r in s_records if r['success']]
        s_lats = [r['mean_inference_latency_ms'] for r in s_records]
        s_interv = sum(r['shield_interventions'] for r in s_records)

        suite_summaries[s_id] = {
            'suite_name': s_name,
            'total_episodes': len(s_records),
            'success_count': s_succ,
            'success_rate_pct': round(100.0 * s_succ / len(s_records), 1),
            'collision_count': s_coll,
            'collision_rate_pct': round(100.0 * s_coll / len(s_records), 1),
            'physical_hull_collision_count': s_phys_coll,
            'lidar_clearance_breach_count': s_lidar_breach,
            'min_clearance_m': round(float(np.min(s_clrs)), 4) if s_clrs else 0.0,
            'min_hull_clearance_m': round(float(np.min(s_hull_clrs)), 5) if s_hull_clrs else 0.0,
            'mean_path_efficiency': round(float(np.mean(s_effs)), 3) if s_effs else 0.0,
            'mean_latency_ms': round(float(np.mean(s_lats)), 4) if s_lats else 0.0,
            'total_shield_interventions': s_interv
        }

    t_end_all = time.time()
    total_wall_s = round(t_end_all - t_start_all, 2)
    env.close()

    # Post-Evaluation Integrity Verification
    print("\n[Integrity Check] Verifying candidate checkpoint after evaluation...")
    post_sha = get_file_sha256(args.model)
    assert post_sha == expected_hash, f"Candidate model SHA mismatch post-eval! Expected {expected_hash}, got {post_sha}"
    assert pre_sha == post_sha, "Candidate model hash changed during evaluation!"
    print(f"  Candidate Checkpoint SHA-256 (post-eval): {post_sha} (UNCHANGED, BIT-IDENTICAL)")

    # Overall Metrics Calculation
    n_total = len(episode_records)
    n_succ = sum(1 for r in episode_records if r['success'])
    n_coll = sum(1 for r in episode_records if r['collision'])
    n_phys_coll = sum(1 for r in episode_records if r.get('physical_hull_collision', False))
    n_lidar_breach = sum(1 for r in episode_records if r.get('lidar_clearance_breach', False))
    n_tout = sum(1 for r in episode_records if r['timeout'])
    n_oob = sum(1 for r in episode_records if r['out_of_bounds'])
    all_clrs = [r['min_ground_truth_clearance_m'] for r in episode_records]
    all_hull_clrs = [r['min_hull_clearance_m'] for r in episode_records]
    all_effs = [r['path_efficiency'] for r in episode_records if r['success']]
    all_lats = [float(x) for r in episode_records for x in r.get('step_latency_samples_ms', [])]
    all_trips = sum(r['watchdog_trips'] for r in episode_records)
    nan_infs = sum(1 for r in episode_records if r['nan_inf_detected'])
    all_interv = sum(r['shield_interventions'] for r in episode_records)

    overall_metrics = {
        'total_episodes': n_total,
        'successful_episodes': n_succ,
        'success_rate_pct': round(100.0 * n_succ / n_total, 2) if n_total else 0.0,
        'collision_episodes': n_coll,
        'collision_rate_pct': round(100.0 * n_coll / n_total, 2) if n_total else 0.0,
        'physical_hull_collision_episodes': n_phys_coll,
        'physical_hull_collision_rate_pct': round(100.0 * n_phys_coll / n_total, 2) if n_total else 0.0,
        'lidar_clearance_breach_episodes': n_lidar_breach,
        'lidar_clearance_breach_rate_pct': round(100.0 * n_lidar_breach / n_total, 2) if n_total else 0.0,
        'timeout_episodes': n_tout,
        'out_of_bounds_episodes': n_oob,
        'clearance_stats_m': {
            'min': round(float(np.min(all_clrs)), 4) if all_clrs else 0.0,
            'mean': round(float(np.mean(all_clrs)), 4) if all_clrs else 0.0,
            'min_hull_clearance': round(float(np.min(all_hull_clrs)), 5) if all_hull_clrs else 0.0,
        },
        'path_efficiency_stats': {
            'mean': round(float(np.mean(all_effs)), 3) if all_effs else 0.0,
        },
        'latency_stats_ms': {
            'mean': round(float(np.mean(all_lats)), 4) if all_lats else 0.0,
            'p95': round(float(np.percentile(all_lats, 95)), 4) if all_lats else 0.0,
        },
        'total_watchdog_interventions': all_trips,
        'nan_inf_occurrences': nan_infs,
        'total_shield_interventions': all_interv,
        'total_wall_clock_s': total_wall_s
    }

    # Construct final payload
    payload = {
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'model_path': args.model,
        'checkpoint_sha256_pre': pre_sha,
        'checkpoint_sha256_post': post_sha,
        'm4_baseline_sha256': m4_sha_pre,
        'safety_shield_active': True,
        'shield_configuration': {
            'd_base_m': shield.config.d_base,
            'safety_margin_m': shield.config.safety_margin,
            'dense_scene_extra_margin_m': shield.config.dense_scene_extra_margin_m,
            'dense_scene_sector_count': shield.config.dense_scene_sector_count,
            'k_delay_s': shield.config.k_delay,
            'a_max_mps2': shield.config.max_horizontal_accel,
            'directional_projection': True,
            'physical_velocity_feedback': True
        },
        'is_smoke_test': args.smoke_test,
        'overall_metrics': overall_metrics,
        'suite_summaries': suite_summaries,
        'episodes': episode_records
    }

    with open(out_json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[Artifact] Saved results to: {out_json_path}")

    generate_markdown_report(payload, out_md_path)
    print(f"[Artifact] Saved markdown report to: {out_md_path}")

    # Summary Output
    print("\n" + "=" * 80)
    print(f"SHIELDED GENERALIZATION SUMMARY ({'SMOKE TEST' if args.smoke_test else 'FULL BENCHMARK'})")
    print(f"  Success Rate:    {overall_metrics['success_rate_pct']}% ({n_succ}/{n_total})")
    print(f"  Collision Rate:  {overall_metrics['collision_rate_pct']}% ({n_coll}/{n_total})")
    print(f"  Min Clearance:   {overall_metrics['clearance_stats_m']['min']} m")
    print(f"  Mean Latency:    {overall_metrics['latency_stats_ms']['mean']} ms (Policy + Shield)")
    print(f"  Interventions:   {all_interv} steps intervened by safety shield")
    print(f"  Wall Time:       {total_wall_s:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
