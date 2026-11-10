#!/usr/bin/env python3
"""
evaluate_m5_shielded_robustness.py
------------------------------------
Milestone 5 — Phase M5.5 Shielded Controlled Robustness Evaluator.

Evaluates the shielded Stage 2b candidate policy:
  Checkpoint: experiments/milestone5/models/ppo_m5_stage2b_remediated_64x64.zip
  Safety Shield: rl_env.safety_shield.DeterministicSafetyShield
across 7 isolated disturbance factors, with 5 severity levels each, on 25
deterministic seeds (20001–20025) for a total of 875 controlled test episodes.

Protocol Fidelity:
- Reuses exact factor definitions, severity levels, seed ranges, and scenario generation
  from evaluate_m5_robustness_corrected.py.
- Preserves the deterministic geometry validation (overlap detection, clearance checks, BFS).
- Applies DeterministicSafetyShield to policy action commands on every simulation step.
- Tracks separate path efficiency for successful flights vs. failed flight lengths.
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
from typing import Any, Dict, List, Tuple, Optional
from collections import deque
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO

import rl_env
from rl_env.uav_fast_env_v1 import UAVFastEnvV1
from rl_env.safety_shield import DeterministicSafetyShield, SafetyShieldConfig
from experiments.milestone5.evaluate_m5_robustness_corrected import (
    FACTORS,
    SEEDS,
    generate_scenario_obstacles,
    validate_scenario_geometry,
    assign_ranks_with_ties,
    get_file_sha256
)

# Authoritative Checkpoint Paths & Expected SHA-256 Hashes
STAGE2B_MODEL_PATH = "/home/kali/proj/experiments/milestone5/models/ppo_m5_stage2b_remediated_64x64.zip"
EXPECTED_STAGE2B_HASH = "c2197f7052e90fa016cf90319677834197fc7139f367631843a6f36b8f5b2ae0"

M4_MODEL_PATH = "/home/kali/proj/experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip"
EXPECTED_M4_HASH = "06907c9fa636e5f268ca55750107918370e445313ad1ac01207c15a4c5935308"

# Default Output Paths (Strictly isolated from baseline artifacts)
DEFAULT_OUT_JSON = "/home/kali/proj/experiments/milestone5/m5_shielded_robustness_results.json"
DEFAULT_OUT_MD = "/home/kali/proj/experiments/milestone5/m5_shielded_robustness_report.md"


def run_episode(
    env: UAVFastEnvV1,
    model: PPO,
    shield: DeterministicSafetyShield,
    factor_id: str,
    level_dict: Dict[str, Any],
    seed: int
) -> Dict[str, Any]:
    """Executes a single shielded robustness evaluation episode."""
    level_idx = level_dict["idx"]
    level_param = level_dict["param"]

    # Obstacle layout
    obstacles = generate_scenario_obstacles(factor_id, level_param, seed)

    # Geometry Validation
    geom_val = validate_scenario_geometry(obstacles)

    opts = {
        "start_pos": [0.0, 0.0, 2.0],
        "start_yaw": 0.0,
        "goal_pos": [6.5, 0.0, 2.0],
        "obstacles": obstacles
    }

    # Dynamics setup
    tau_nom = 0.20
    acc_nom = 3.0
    if factor_id == "F7_dynamics_perturbation":
        tau, a_max = level_param
        env.dynamics.config.tau_velocity_xy = float(tau)
        env.dynamics.config.max_horizontal_accel = float(a_max)
    else:
        env.dynamics.config.tau_velocity_xy = tau_nom
        env.dynamics.config.max_horizontal_accel = acc_nom

    obs, info = env.reset(seed=seed, options=opts)

    p0 = np.array(opts["start_pos"], dtype=np.float32)
    g0 = np.array(opts["goal_pos"], dtype=np.float32)
    straight_line_dist = float(np.linalg.norm(g0 - p0))

    traj_positions = [p0.copy()]
    step_latencies = []
    min_surface_clearance = float(info.get("min_surface_dist_m", 10.0))
    speeds = []
    watchdog_trips = 0
    nan_inf_found = False

    shield_interventions = 0
    layer1_interventions = 0
    layer2_interventions = 0

    step_count = 0
    done = False
    held_clearances = None

    # Step-level noise generator
    step_rng = np.random.default_rng(seed + 1000 + level_idx * 50)

    while not done and step_count < 150:
        s = env.dynamics.get_state()
        raw_ranges = env.last_raw_ranges.copy()

        # Disturbance injection into observation
        p_noise = np.zeros(3, dtype=np.float32)
        v_noise = np.zeros(3, dtype=np.float32)

        if factor_id == "F1_lidar_noise":
            sig = float(level_param)
            if sig > 0.0:
                noise = step_rng.normal(0.0, sig, size=len(raw_ranges)).astype(np.float32)
                raw_ranges = np.clip(raw_ranges + noise, 0.10, 10.0)

        elif factor_id == "F2_lidar_dropout":
            p_drop = float(level_param)
            if p_drop > 0.0:
                mask = step_rng.uniform(0.0, 1.0, size=len(raw_ranges)) < p_drop
                raw_ranges[mask] = 10.0  # Missing returns report clear space (10.0m)

        elif factor_id == "F3_obs_state_noise":
            sp, sv = level_param
            if sp > 0.0:
                p_noise = step_rng.normal(0.0, sp, size=3).astype(np.float32)
            if sv > 0.0:
                v_noise = step_rng.normal(0.0, sv, size=3).astype(np.float32)

        # Compute the unified 26-D observation for policy
        obs_policy, meta = env.obs_handler.compute_observation_with_metadata(
            pos_world=s.pos_world + p_noise,
            vel_world=s.vel_world + v_noise,
            quat=s.quat,
            rates_body=s.body_rates,
            goal_world=env.goal_world,
            prev_action=env.prev_action,
            raw_scan=(raw_ranges, env.config.lidar_min_angle_rad, env.config.lidar_angle_increment_rad),
            ground_z=0.0
        )

        # Factor 4: Clearance Staleness (Zero-Order Hold)
        if factor_id == "F4_sensor_rate":
            K = int(level_param)
            if step_count % K == 0 or held_clearances is None:
                held_clearances = obs_policy[18:26].copy()
            else:
                obs_policy[18:26] = held_clearances.copy()

        # Check bounds / watchdog
        curr_pos = traj_positions[-1]
        x, y, z = curr_pos[0], curr_pos[1], curr_pos[2]
        if abs(x) > 9.0 or abs(y) > 9.0 or z > 6.0:
            watchdog_trips += 1
        if z < 0.20:
            watchdog_trips += 1

        if np.isnan(obs_policy).any() or np.isinf(obs_policy).any():
            nan_inf_found = True

        # Policy Inference (CPU)
        t0 = time.perf_counter()
        with torch.inference_mode():
            raw_action, _ = model.predict(obs_policy, deterministic=True)
        t1 = time.perf_counter()
        policy_lat_ms = (t1 - t0) * 1000.0

        if np.isnan(raw_action).any() or np.isinf(raw_action).any():
            nan_inf_found = True

        # Safety Shield Action Interception
        t2 = time.perf_counter()
        meta = dict(meta) if isinstance(meta, dict) else {}
        meta["position_world"] = np.asarray(s.pos_world, dtype=np.float32).copy()
        action, diag = shield.filter_action(obs_policy, raw_action, perception_metadata=meta)
        t3 = time.perf_counter()
        shield_lat_ms = (t3 - t2) * 1000.0

        if diag.is_intervened:
            shield_interventions += 1
        if diag.layer1_active:
            layer1_interventions += 1
        if diag.layer2_active:
            layer2_interventions += 1

        step_latencies.append(policy_lat_ms + shield_lat_ms)

        # Step Environment
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

    # Restore nominal dynamics parameters
    env.dynamics.config.tau_velocity_xy = tau_nom
    env.dynamics.config.max_horizontal_accel = acc_nom

    traj_arr = np.array(traj_positions)
    path_len = float(np.sum(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1)))
    duration_s = float(step_count * 0.10)

    term_reason = info.get("termination_reason", "UNKNOWN")
    is_success = bool(info.get("is_success", False)) or (term_reason == "goal_reached_and_stabilized")
    is_collision = bool(info.get("is_obstacle_collision", False)) or bool(info.get("is_crashed", False)) or ("collision" in term_reason)

    if is_success:
        path_eff = round(float(path_len / max(0.1, straight_line_dist)), 4)
        failed_path_len = None
        failed_prog_frac = None
    else:
        path_eff = None
        failed_path_len = round(path_len, 3)
        failed_prog_frac = round(float(path_len / max(0.1, straight_line_dist)), 4)

    mean_lat = float(np.mean(step_latencies)) if step_latencies else 0.0
    med_lat = float(np.median(step_latencies)) if step_latencies else 0.0
    p95_lat = float(np.percentile(step_latencies, 95)) if step_latencies else 0.0
    max_spd = float(np.max(speeds)) if speeds else 0.0
    mean_spd = float(np.mean(speeds)) if speeds else 0.0

    return {
        "seed": seed,
        "factor_id": factor_id,
        "level_idx": level_idx,
        "level_label": level_dict["label"],
        "level_desc": level_dict["desc"],
        "geometry_validation": geom_val,
        "success": is_success,
        "collision": is_collision,
        "termination_reason": term_reason,
        "step_count": step_count,
        "episode_duration_sim_s": round(duration_s, 2),
        "min_ground_truth_clearance_m": round(min_surface_clearance, 4),
        "path_length_m": round(path_len, 3),
        "straight_line_dist_m": round(straight_line_dist, 3),
        "path_efficiency_successful": path_eff,
        "failed_path_length_m": failed_path_len,
        "failed_progress_fraction": failed_prog_frac,
        "mean_inference_latency_ms": round(mean_lat, 4),
        "median_inference_latency_ms": round(med_lat, 4),
        "p95_inference_latency_ms": round(p95_lat, 4),
        "step_latency_samples_ms": [round(float(x), 6) for x in step_latencies],
        "max_speed_mps": round(max_spd, 3),
        "mean_speed_mps": round(mean_spd, 3),
        "watchdog_trips": watchdog_trips,
        "nan_inf_detected": nan_inf_found,
        "shield_interventions": shield_interventions,
        "layer1_interventions": layer1_interventions,
        "layer2_interventions": layer2_interventions,
    }


def generate_markdown_report(data: Dict[str, Any], ranked_by_peak: List[Dict[str, Any]], out_path: str):
    """Generates a structured analytical markdown report for shielded robustness."""
    meta = data["metadata"]
    f_sums = data["factor_summaries"]

    lines = []
    lines.append("# Milestone 5 — Phase M5.5: Shielded Controlled Robustness Report")
    lines.append("")
    lines.append("## 1. Executive Summary & Verification State")
    lines.append("")
    lines.append(f"- **Phase**: Milestone 5 — Phase M5.5: Controlled Robustness Testing (Shielded)")
    lines.append(f"- **Evaluation Timestamp**: `{meta['timestamp']}`")
    lines.append(f"- **Total Test Episodes**: {meta['total_episodes']}")
    lines.append(f"- **Candidate Model Checkpoint**: `{meta['model_path']}`")
    lines.append(f"- **Pre-Evaluation SHA-256**: `{meta['model_sha256_pre']}`")
    lines.append(f"- **Post-Evaluation SHA-256**: `{meta['model_sha256_post']}`")
    lines.append(f"- **M4 Baseline Checkpoint SHA-256**: `{meta['m4_baseline_sha256']}` (VERIFIED BIT-IDENTICAL)")
    lines.append(f"- **Checkpoint Integrity**: **PASS** (Bit-identical, 0 modifications, strictly frozen)")
    lines.append(f"- **Safety Shield**: `DeterministicSafetyShield` (Active on all steps)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Shielded Robustness Sensitivity Ranking")
    lines.append("")
    lines.append("Ranked by **Peak Degradation** (maximum drop in success rate relative to Level 0):")
    lines.append("")
    lines.append("| Rank | Factor Name | Domain | Level 0 Succ | Level 4 Succ | Endpoint Drop | Peak Drop (Worst Level) | Shield Interv |")
    lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |")

    for rf in ranked_by_peak:
        interv = sum(lvl.get("shield_interventions", 0) for lvl in rf["levels"])
        lines.append(f"| **{rf['rank_str']}** | **{rf['factor_name']}** | {rf['domain']} | "
                     f"{rf['l0_success_pct']:.1f}% | {rf['l4_success_pct']:.1f}% | "
                     f"{rf['degradation_endpoint_pp']:+.1f} pp | "
                     f"**{rf['peak_degradation_pp']:+.1f} pp** ({rf['peak_level_label']}) | {interv} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Decision Gate G2 & G3 Compliance Summary")
    lines.append("")

    # Compute Gate G2 (Level 0 nominal corridor)
    l0_succs = [f_sums[f["id"]]["levels"][0]["success_rate_pct"] for f in FACTORS if f["id"] in f_sums]
    mean_l0_succ = float(np.mean(l0_succs)) if l0_succs else 0.0
    g2_pass = mean_l0_succ >= 90.0

    # Compute Gate G3-A (Level 2 aggregate)
    l2_succs = [f_sums[f["id"]]["levels"][2]["success_rate_pct"] for f in FACTORS if f["id"] in f_sums and len(f_sums[f["id"]]["levels"]) > 2]
    mean_l2_succ = float(np.mean(l2_succs)) if l2_succs else 0.0
    g3a_pass = mean_l2_succ >= 85.0

    # Compute Gate G3-B (Locked per-factor Level 2 floor constraints)
    g3b_thresholds = {
        "F1_lidar_noise": (23, 25, 92.0),
        "F2_lidar_dropout": (22, 25, 88.0),
        "F3_obs_state_noise": (22, 25, 88.0),
        "F4_sensor_rate": (22, 25, 88.0),
        "F5_obstacle_density": (20, 25, 80.0),
        "F6_obstacle_size": (15, 25, 60.0),
        "F7_dynamics_perturbation": (20, 25, 80.0),
    }

    g3b_results = {}
    g3b_overall_pass = True
    for f_id, (min_succ, total_ep, min_pct) in g3b_thresholds.items():
        if f_id in f_sums and len(f_sums[f_id]["levels"]) > 2:
            lvl2 = f_sums[f_id]["levels"][2]
            succ_cnt = lvl2.get("success_count", int(round(lvl2["success_rate_pct"] * total_ep / 100.0)))
            succ_pct = lvl2["success_rate_pct"]
            passed = (succ_cnt >= min_succ) or (succ_pct >= min_pct)
            if not passed:
                g3b_overall_pass = False
            g3b_results[f_id] = {
                "name": f_sums[f_id]["factor_name"],
                "succ_cnt": succ_cnt,
                "total_ep": total_ep,
                "succ_pct": succ_pct,
                "req_cnt": min_succ,
                "req_pct": min_pct,
                "pass": passed
            }

    lines.append(f"| Decision Gate | Prescribed Threshold | Measured Shielded Value | Status |")
    lines.append(f"| :--- | :---: | :---: | :---: |")
    lines.append(f"| **Gate G2 (Corridor Regression)** | Success Rate $\\ge 90.0\\%$ on L0 | **{mean_l0_succ:.1f}%** | **{'PASS' if g2_pass else 'FAIL'}** |")
    lines.append(f"| **Gate G3-A (Aggregate L2 Robustness)** | Mean Success $\\ge 85.0\\%$ across 7 L2 factors | **{mean_l2_succ:.1f}%** | **{'PASS' if g3a_pass else 'FAIL'}** |")
    lines.append(f"| **Gate G3-B (Per-Factor Floor Constraints)** | All 7 factor floors met | **{'ALL PASSED' if g3b_overall_pass else 'DEFICIT DETECTED'}** | **{'PASS' if g3b_overall_pass else 'FAIL'}** |")
    lines.append("")
    lines.append("### Gate G3-B Per-Factor Breakdown (Level 2 Moderate Robustness)")
    lines.append("")
    lines.append("| Factor ID | Factor Description | Required Threshold | Measured Success | Status |")
    lines.append("| :--- | :--- | :---: | :---: | :---: |")
    for f_id, res in g3b_results.items():
        lines.append(f"| **{f_id[:2]}** | {res['name']} | $\\ge {res['req_cnt']}/{res['total_ep']}$ ({res['req_pct']:.0f}%) | **{res['succ_cnt']}/{res['total_ep']} ({res['succ_pct']:.1f}%)** | **{'PASS' if res['pass'] else 'FAIL'}** |")

    lines.append("")
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Milestone 5 Shielded Controlled Robustness Evaluator")
    parser.add_argument("--model", type=str, default=STAGE2B_MODEL_PATH, help="Path to policy checkpoint")
    parser.add_argument("--expected-hash", type=str, default=None, help="Expected SHA-256 hash of the candidate model")
    parser.add_argument("--safety-margin", type=float, default=0.55, help="Obstacle-surface margin beyond the UAV hull (m)")
    parser.add_argument("--smoke-test", action="store_true", help="Execute minimal non-benchmark smoke test")
    parser.add_argument("--output-json", type=str, default=DEFAULT_OUT_JSON, help="Output JSON path")
    parser.add_argument("--output-md", type=str, default=DEFAULT_OUT_MD, help="Output Markdown report path")
    args = parser.parse_args()

    print("=" * 80)
    print("Milestone 5 — Phase M5.5: Shielded Controlled Robustness Evaluation")
    print(f"Candidate: {args.model}")
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

    # 2. Load Policy on CPU
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(f"\n[Model Load] Loading candidate policy on CPU: {args.model}")
    model = PPO.load(args.model, device="cpu")
    model.policy.eval()

    # 3. Initialize Safety Shield with validated R1-R4 configuration
    shield = DeterministicSafetyShield(SafetyShieldConfig(safety_margin=args.safety_margin))
    print(f"[Safety Shield Active] d_base={shield.config.d_base:.2f}m, k_delay={shield.config.k_delay:.2f}s, directional projection=True")

    # 4. Initialize Environment
    env = gym.make("UAVFastNav-v1").unwrapped

    if args.smoke_test:
        print("\n[MODE: SMOKE TEST] Running minimal smoke test across 1 factor x 2 levels x 2 seeds...")
        test_factors = [FACTORS[0]]  # F1_lidar_noise
        test_levels = test_factors[0]["levels"][:2]  # L0 and L1
        test_seeds = [99901, 99902]
        out_json_path = args.output_json if args.output_json != DEFAULT_OUT_JSON else "/home/kali/proj/experiments/milestone5/m5_shielded_robustness_smoke.json"
        out_md_path = args.output_md if args.output_md != DEFAULT_OUT_MD else "/home/kali/proj/experiments/milestone5/m5_shielded_robustness_smoke.md"
    else:
        test_factors = FACTORS
        test_seeds = SEEDS
        out_json_path = args.output_json
        out_md_path = args.output_md

    total_episodes_planned = len(test_factors) * (len(test_levels) if args.smoke_test else 5) * len(test_seeds)
    print(f"[Evaluation Plan] Executing {total_episodes_planned} episodes...")

    all_episode_records = []
    factor_summaries = {}
    t_start_all = time.time()
    ep_counter = 0

    for factor in test_factors:
        f_id = factor["id"]
        f_name = factor["name"]
        print(f"\n{'='*75}")
        print(f"Factor: {f_id} — {f_name} ({factor['domain']})")
        print(f"{'='*75}")

        f_records = []
        level_summaries = []
        levels_to_run = test_levels if args.smoke_test else factor["levels"]

        for lvl in levels_to_run:
            l_idx = lvl["idx"]
            l_lbl = lvl["label"]
            l_desc = lvl["desc"]

            lvl_records = []
            succ_count = 0
            coll_count = 0
            clearances_succ = []
            clearances_all = []
            succ_efficiencies = []
            failed_lengths = []
            latencies_mean = []
            latencies_median = []
            invalid_geom_count = 0
            geom_reasons_counter = {}
            interventions_lvl = 0

            for seed in test_seeds:
                rec = run_episode(env, model, shield, f_id, lvl, seed)
                lvl_records.append(rec)
                f_records.append(rec)
                all_episode_records.append(rec)
                ep_counter += 1

                if rec["success"]:
                    succ_count += 1
                    succ_efficiencies.append(rec["path_efficiency_successful"])
                    clearances_succ.append(rec["min_ground_truth_clearance_m"])
                if rec["collision"]:
                    coll_count += 1
                    failed_lengths.append(rec["failed_path_length_m"])
                clearances_all.append(rec["min_ground_truth_clearance_m"])
                latencies_mean.append(rec["mean_inference_latency_ms"])
                latencies_median.append(rec["median_inference_latency_ms"])
                interventions_lvl += rec["shield_interventions"]

                gval = rec["geometry_validation"]
                if not gval["is_valid"]:
                    invalid_geom_count += 1

            n_ep = len(lvl_records)
            succ_rate = round(100.0 * succ_count / n_ep, 2)
            coll_rate = round(100.0 * coll_count / n_ep, 2)
            min_c_all = float(np.min(clearances_all)) if clearances_all else 0.0
            mean_c_all = float(np.mean(clearances_all)) if clearances_all else 0.0
            min_c_succ = float(np.min(clearances_succ)) if clearances_succ else None
            mean_c_succ = float(np.mean(clearances_succ)) if clearances_succ else None
            mean_eff_succ = float(np.mean(succ_efficiencies)) if succ_efficiencies else None
            mean_len_failed = float(np.mean(failed_lengths)) if failed_lengths else None
            mean_lat = float(np.mean(latencies_mean))
            med_lat = float(np.median(latencies_median))

            lvl_sum = {
                "level_idx": l_idx,
                "level_label": l_lbl,
                "description": l_desc,
                "episodes": n_ep,
                "success_count": succ_count,
                "success_rate_pct": succ_rate,
                "collision_count": coll_count,
                "collision_rate_pct": coll_rate,
                "invalid_geometry_count": invalid_geom_count,
                "min_clearance_all_m": round(min_c_all, 4),
                "mean_clearance_all_m": round(mean_c_all, 4),
                "min_clearance_successful_m": round(min_c_succ, 4) if min_c_succ is not None else None,
                "mean_clearance_successful_m": round(mean_c_succ, 4) if mean_c_succ is not None else None,
                "mean_path_efficiency_successful_only": round(mean_eff_succ, 4) if mean_eff_succ is not None else None,
                "mean_path_length_failed_m": round(mean_len_failed, 3) if mean_len_failed is not None else None,
                "mean_latency_ms": round(mean_lat, 4),
                "median_latency_ms": round(med_lat, 4),
                "shield_interventions": interventions_lvl
            }
            level_summaries.append(lvl_sum)

            eff_str = f"{mean_eff_succ:.3f}" if mean_eff_succ is not None else "N/A"
            clr_succ_str = f"{min_c_succ:.3f}m" if min_c_succ is not None else "N/A"
            print(f"  [{l_lbl}] {l_desc:<45} | Succ: {succ_count:>2}/{n_ep} ({succ_rate:>5.1f}%) | "
                  f"Coll: {coll_count:>2}/{n_ep} ({coll_rate:>5.1f}%) | "
                  f"SuccEff: {eff_str:>5} | SuccMinClr: {clr_succ_str:>7} | LatMed: {med_lat:>5.3f}ms | Interv: {interventions_lvl}")

        l0_succ = level_summaries[0]["success_rate_pct"]
        l4_succ = level_summaries[-1]["success_rate_pct"]
        succ_delta_pp = round(l4_succ - l0_succ, 2)
        degradation_endpoint_pp = round(l0_succ - l4_succ, 2)

        degradations_all_levels = [round(l0_succ - lvl["success_rate_pct"], 2) for lvl in level_summaries]
        peak_degradation_pp = max(degradations_all_levels)
        peak_level_idx = degradations_all_levels.index(peak_degradation_pp)

        factor_summaries[f_id] = {
            "factor_id": f_id,
            "factor_name": f_name,
            "domain": factor["domain"],
            "unit": factor["unit"],
            "levels": level_summaries,
            "l0_success_pct": l0_succ,
            "l4_success_pct": l4_succ,
            "success_delta_pp": succ_delta_pp,
            "degradation_endpoint_pp": degradation_endpoint_pp,
            "peak_degradation_pp": peak_degradation_pp,
            "peak_level_idx": peak_level_idx,
            "peak_level_label": level_summaries[peak_level_idx]["level_label"]
        }

    t_end_all = time.time()
    total_duration_s = round(t_end_all - t_start_all, 2)
    env.close()

    # Post-Evaluation Integrity Verification
    print("\n[Integrity Check] Verifying candidate checkpoint after evaluation...")
    post_sha = get_file_sha256(args.model)
    assert post_sha == expected_hash, f"Candidate model SHA mismatch post-eval! Expected {expected_hash}, got {post_sha}"
    assert pre_sha == post_sha, "Candidate model hash changed during evaluation!"
    print(f"  Candidate Checkpoint SHA-256 (post-eval): {post_sha} (UNCHANGED, BIT-IDENTICAL)")

    # Sensitivity Rankings
    ranked_by_endpoint = assign_ranks_with_ties(list(factor_summaries.values()), key="degradation_endpoint_pp")
    ranked_by_peak = assign_ranks_with_ties(list(factor_summaries.values()), key="peak_degradation_pp")

    json_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_path": args.model,
            "model_sha256_pre": pre_sha,
            "model_sha256_post": post_sha,
            "m4_baseline_sha256": m4_sha_pre,
            "safety_shield_active": True,
            "shield_configuration": {
                "d_base_m": shield.config.d_base,
                "safety_margin_m": shield.config.safety_margin,
                "dense_scene_extra_margin_m": shield.config.dense_scene_extra_margin_m,
                "dense_scene_sector_count": shield.config.dense_scene_sector_count,
                "k_delay_s": shield.config.k_delay,
                "a_max_mps2": shield.config.max_horizontal_accel,
                "directional_projection": True,
                "physical_velocity_feedback": True
            },
            "is_smoke_test": args.smoke_test,
            "total_episodes": ep_counter,
            "seeds": test_seeds,
            "execution_time_s": total_duration_s
        },
        "sensitivity_ranking_peak": ranked_by_peak,
        "sensitivity_ranking_endpoint": ranked_by_endpoint,
        "factor_summaries": factor_summaries,
        "episodes": all_episode_records
    }

    with open(out_json_path, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"\n[Artifact] Saved machine-readable results to: {out_json_path}")

    generate_markdown_report(json_payload, ranked_by_peak, out_md_path)
    print(f"[Artifact] Saved markdown report to: {out_md_path}")

    print("\n" + "=" * 80)
    print(f"SHIELDED ROBUSTNESS SUMMARY ({'SMOKE TEST' if args.smoke_test else 'FULL BENCHMARK'})")
    print(f"  Total Episodes:  {ep_counter}")
    print(f"  Execution Time:  {total_duration_s:.1f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()
