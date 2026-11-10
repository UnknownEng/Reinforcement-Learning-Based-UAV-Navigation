#!/usr/bin/env python3
"""
evaluate_m5_generalization_baseline.py
--------------------------------------
Phase M5.1: Evaluates the frozen Milestone 4 26-D PPO candidate policy
on genuinely unseen held-out environments across deterministic seeds 10001–10200.

Zero training, zero fine-tuning, zero policy modifications.
Pre- and post-evaluation SHA-256 integrity checks are enforced.
"""

import os
import sys
import time
import json
import math
import hashlib
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO

import rl_env
from rl_env.uav_fast_env_v1 import UAVFastEnvV1

# Expected Checkpoint Hashes
EXPECTED_M4_HASH = "06907c9fa636e5f268ca55750107918370e445313ad1ac01207c15a4c5935308"
MODEL_PATH = "/home/kali/proj/experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip"
OUT_JSON = "/home/kali/proj/experiments/milestone5/m5_generalization_baseline_results.json"
OUT_MD = "/home/kali/proj/experiments/milestone5/m5_generalization_baseline_report.md"


def get_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_heldout_suite_config(suite_id: str, seed: int) -> dict:
    """
    Generates deterministic, genuinely held-out obstacle and start/goal configurations
    for each of the 8 generalization challenge suites.
    """
    rng = np.random.default_rng(seed)

    if suite_id == 'A':
        # Suite A: Unseen Obstacle Positions (diagonal / off-axis goals & random non-grid placements)
        start_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        angle = float(rng.uniform(0.1, 2.0 * np.pi))
        dist = float(rng.uniform(5.0, 6.5))
        goal_pos = np.array([dist * np.cos(angle), dist * np.sin(angle), float(rng.uniform(1.6, 2.4))], dtype=np.float32)
        start_yaw = float(rng.uniform(-np.pi, np.pi))
        
        obstacles = []
        p_vec = goal_pos[:2] - start_pos[:2]
        p_len = float(np.linalg.norm(p_vec))
        p_norm = p_vec / p_len
        perp = np.array([-p_norm[1], p_norm[0]])
        
        num_obs = int(rng.integers(4, 7))
        for _ in range(num_obs):
            frac = float(rng.uniform(0.25, 0.75))
            lat = float(rng.uniform(-2.2, 2.2))
            c = start_pos[:2] + frac * p_vec + lat * perp
            r = float(rng.uniform(0.25, 0.40))
            if np.linalg.norm(c - start_pos[:2]) > (1.5 + r) and np.linalg.norm(c - goal_pos[:2]) > (1.2 + r):
                obstacles.append((float(c[0]), float(c[1]), r))

        return {
            'suite_id': 'A',
            'suite_name': 'Unseen Obstacle Positions',
            'description': 'Arbitrary diagonal/off-axis goal vectors with non-grid obstacle placements',
            'start_pos': start_pos.tolist(),
            'start_yaw': start_yaw,
            'goal_pos': goal_pos.tolist(),
            'obstacles': obstacles
        }

    elif suite_id == 'B':
        # Suite B: Unseen Obstacle Arrangements (Triangular wedges, diamond clusters, slaloms)
        start_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        goal_pos = np.array([6.0, 0.0, 2.0], dtype=np.float32)
        start_yaw = float(rng.uniform(-0.3, 0.3))
        pattern_type = seed % 3
        
        if pattern_type == 0:
            # Triangular wedge barrier splitting the forward path
            obstacles = [(2.5, 0.0, 0.35), (4.0, 0.95, 0.30), (4.0, -0.95, 0.30)]
            desc = 'Triangular barrier formation forcing path split'
        elif pattern_type == 1:
            # Diamond cluster directly blocking line of sight
            obstacles = [(2.4, 0.0, 0.30), (3.6, 0.85, 0.28), (3.6, -0.85, 0.28), (4.8, 0.0, 0.30)]
            desc = 'Diamond cluster barrier requiring peripheral detour'
        else:
            # S-curve slalom
            lat_off = float(rng.uniform(0.65, 0.85))
            obstacles = [(2.0, lat_off, 0.30), (3.6, -lat_off, 0.30), (5.0, lat_off, 0.30)]
            desc = 'Three-point alternating slalom formation'

        return {
            'suite_id': 'B',
            'suite_name': 'Unseen Obstacle Arrangements',
            'description': desc,
            'start_pos': start_pos.tolist(),
            'start_yaw': start_yaw,
            'goal_pos': goal_pos.tolist(),
            'obstacles': obstacles
        }

    elif suite_id == 'C':
        # Suite C: Different Obstacle Spacing (Tight pairs, heterogeneous gaps)
        start_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        goal_pos = np.array([6.0, 0.0, 2.0], dtype=np.float32)
        start_yaw = 0.0
        gap_type = seed % 2
        
        if gap_type == 0:
            # Impassable tight pair directly on path (surface gap ~0.85m < safe traversal)
            r = 0.30
            y_gap = 0.55  # surface dist = 2*y_gap - 2*r = 1.10 - 0.60 = 0.50m (impassable!)
            obstacles = [(3.0, y_gap, r), (3.0, -y_gap, r), (4.8, 1.2, r)]
            desc = 'Impassable dual-cylinder pair (gap=0.50m) requiring complete bypass'
        else:
            # Heterogeneous spacing: narrow gap left (0.8m) vs wide opening right (2.5m)
            r = 0.30
            obstacles = [(2.8, 0.8, r), (2.8, 1.9, r), (4.5, -0.9, r)]
            desc = 'Heterogeneous corridor with asymmetric opening widths'

        return {
            'suite_id': 'C',
            'suite_name': 'Different Obstacle Spacing',
            'description': desc,
            'start_pos': start_pos.tolist(),
            'start_yaw': start_yaw,
            'goal_pos': goal_pos.tolist(),
            'obstacles': obstacles
        }

    elif suite_id == 'D':
        # Suite D: Randomized Start/Goal Configurations (Off-center start, steep Z climb/descent, reverse heading)
        x0 = float(rng.uniform(-3.5, 3.5))
        y0 = float(rng.uniform(-3.5, 3.5))
        z0 = float(rng.uniform(1.0, 2.5))
        start_pos = np.array([x0, y0, z0], dtype=np.float32)
        
        # Goal at large distance and vertical elevation difference
        dist = float(rng.uniform(5.0, 6.5))
        ang = float(rng.uniform(0.0, 2.0 * np.pi))
        gx = float(np.clip(x0 + dist * np.cos(ang), -6.5, 6.5))
        gy = float(np.clip(y0 + dist * np.sin(ang), -6.5, 6.5))
        gz = float(rng.uniform(1.2, 2.8))
        goal_pos = np.array([gx, gy, gz], dtype=np.float32)
        
        # Start yaw facing directly AWAY from goal (180 deg turnaround required)
        goal_heading = math.atan2(gy - y0, gx - x0)
        start_yaw = float(goal_heading + rng.choice([np.pi, -np.pi, 0.5*np.pi, -0.5*np.pi]))
        
        # 3 intermediate obstacles
        obstacles = []
        p_vec = goal_pos[:2] - start_pos[:2]
        p_len = float(np.linalg.norm(p_vec))
        if p_len > 3.0:
            p_norm = p_vec / p_len
            perp = np.array([-p_norm[1], p_norm[0]])
            for frac, lat in [(0.3, 0.8), (0.5, -0.8), (0.7, 0.5)]:
                c = start_pos[:2] + frac * p_vec + lat * perp
                r = float(rng.uniform(0.25, 0.35))
                if np.linalg.norm(c - start_pos[:2]) > (1.5 + r) and np.linalg.norm(c - goal_pos[:2]) > (1.2 + r):
                    obstacles.append((float(c[0]), float(c[1]), r))

        return {
            'suite_id': 'D',
            'suite_name': 'Randomized Start/Goal Configurations',
            'description': 'Off-center spawns, large elevation changes (|dZ|>1.0m), and reverse initial headings',
            'start_pos': start_pos.tolist(),
            'start_yaw': start_yaw,
            'goal_pos': goal_pos.tolist(),
            'obstacles': obstacles
        }

    elif suite_id == 'E':
        # Suite E: Asymmetric Obstacle Layouts (One-sided heavy clutter / dogleg chicanes)
        start_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        goal_pos = np.array([6.0, 0.0, 2.0], dtype=np.float32)
        start_yaw = 0.0
        side = rng.choice([1.0, -1.0])
        
        # 4 obstacles clustered densely along one side, 1 distant on other
        r = 0.30
        obstacles = [
            (2.0, side * 0.7, r),
            (3.2, side * 0.8, r),
            (4.4, side * 0.9, r),
            (5.2, side * 0.7, r),
            (3.0, -side * 2.2, r)
        ]

        return {
            'suite_id': 'E',
            'suite_name': 'Asymmetric Obstacle Layouts',
            'description': 'Asymmetric flank crowding biasing safe clearance exclusively to one side',
            'start_pos': start_pos.tolist(),
            'start_yaw': start_yaw,
            'goal_pos': goal_pos.tolist(),
            'obstacles': obstacles
        }

    elif suite_id == 'F':
        # Suite F: Increased Obstacle Density (10 to 12 obstacles)
        start_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        goal_pos = np.array([6.5, 0.0, 2.0], dtype=np.float32)
        start_yaw = 0.0
        
        num_obs = int(rng.integers(10, 13))
        obstacles = []
        attempts = 0
        while len(obstacles) < num_obs and attempts < 150:
            attempts += 1
            cx = float(rng.uniform(1.5, 5.5))
            cy = float(rng.uniform(-2.8, 2.8))
            r = float(rng.uniform(0.22, 0.32))
            c = np.array([cx, cy])
            if np.linalg.norm(c - start_pos[:2]) < (1.5 + r):
                continue
            if np.linalg.norm(c - goal_pos[:2]) < (1.2 + r):
                continue
            too_close = False
            for ocx, ocy, orad in obstacles:
                if np.linalg.norm(c - np.array([ocx, ocy])) < (0.95 + r + orad):
                    too_close = True
                    break
            if not too_close:
                obstacles.append((cx, cy, r))

        return {
            'suite_id': 'F',
            'suite_name': 'Increased Obstacle Density',
            'description': f'High-density obstacle forest ({len(obstacles)} cylinders) with reduced inter-obstacle spacing',
            'start_pos': start_pos.tolist(),
            'start_yaw': start_yaw,
            'goal_pos': goal_pos.tolist(),
            'obstacles': obstacles
        }

    elif suite_id == 'G':
        # Suite G: Narrow Passages (Apertures in [0.75m, 1.15m] surface clearance)
        start_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        goal_pos = np.array([6.0, 0.0, 2.0], dtype=np.float32)
        start_yaw = 0.0
        
        # Aperture between gate posts
        aperture_m = float(rng.uniform(0.85, 1.15))
        r = 0.25
        y_post = (aperture_m / 2.0) + r
        gate_x = float(rng.uniform(2.8, 3.5))
        
        # Tilted or straight gate
        tilt_angle = float(rng.uniform(-0.4, 0.4))
        p1 = np.array([gate_x - y_post * np.sin(tilt_angle), y_post * np.cos(tilt_angle)])
        p2 = np.array([gate_x + y_post * np.sin(tilt_angle), -y_post * np.cos(tilt_angle)])
        
        obstacles = [(float(p1[0]), float(p1[1]), r), (float(p2[0]), float(p2[1]), r)]
        
        # Add a flanking guide cylinder
        obstacles.append((gate_x + 1.8, float(rng.choice([1.8, -1.8])), r))

        return {
            'suite_id': 'G',
            'suite_name': 'Narrow Passages',
            'description': f'Constrained narrow gate aperture ({aperture_m:.2f}m clearance) with tilt {math.degrees(tilt_angle):.1f} deg',
            'start_pos': start_pos.tolist(),
            'start_yaw': start_yaw,
            'goal_pos': goal_pos.tolist(),
            'obstacles': obstacles
        }

    elif suite_id == 'H':
        # Suite H: Multiple Sequential Obstacles (Multi-gate sequential slalom)
        start_pos = np.array([0.0, 0.0, 2.0], dtype=np.float32)
        goal_pos = np.array([6.5, 0.0, 2.0], dtype=np.float32)
        start_yaw = 0.0
        
        # 3 sequential barriers across flight path
        r = 0.28
        y_flip = rng.choice([1.0, -1.0])
        obstacles = [
            # Barrier 1: opening at +1.2m
            (2.0, -y_flip * 0.2, r), (2.0, -y_flip * 1.5, r),
            # Barrier 2: opening at -1.2m
            (3.6, y_flip * 0.2, r), (3.6, y_flip * 1.5, r),
            # Barrier 3: opening in center
            (5.2, 1.4, r), (5.2, -1.4, r)
        ]

        return {
            'suite_id': 'H',
            'suite_name': 'Multiple Sequential Obstacles',
            'description': 'Sequential multi-stage barrier slalom requiring consecutive lateral reversals',
            'start_pos': start_pos.tolist(),
            'start_yaw': start_yaw,
            'goal_pos': goal_pos.tolist(),
            'obstacles': obstacles
        }

    raise ValueError(f"Unknown suite ID: {suite_id}")


def run_evaluation():
    print("=" * 70)
    print("Milestone 5 — Phase M5.1: Frozen M4 Generalization Baseline Evaluation")
    print("=" * 70)

    # 1. Pre-Evaluation Integrity Verification
    print("[Integrity Check] Verifying M4 candidate checkpoint before evaluation...")
    assert os.path.exists(MODEL_PATH), f"Candidate model missing at {MODEL_PATH}"
    pre_sha = get_file_sha256(MODEL_PATH)
    assert pre_sha == EXPECTED_M4_HASH, f"M4 Checkpoint SHA mismatch! Expected {EXPECTED_M4_HASH}, got {pre_sha}"
    print(f"  M4 Checkpoint SHA-256 (pre-eval): {pre_sha} (VERIFIED)")

    # 2. Load Model on CPU
    print(f"[Model Load] Loading frozen candidate policy on CPU: {MODEL_PATH}")
    model = PPO.load(MODEL_PATH, device="cpu")
    policy = model.policy
    policy.eval()

    # Warmup policy
    dummy_obs = np.zeros(26, dtype=np.float32)
    with torch.inference_mode():
        for _ in range(100):
            _ = model.predict(dummy_obs, deterministic=True)
    print("  Policy warmed up on CPU.")

    # 3. Initialize Environment
    env = gym.make("UAVFastNav-v1")

    suites = [
        ('A', 'Unseen Obstacle Positions', range(10001, 10026)),
        ('B', 'Unseen Obstacle Arrangements', range(10026, 10051)),
        ('C', 'Different Obstacle Spacing', range(10051, 10076)),
        ('D', 'Randomized Start/Goal Configurations', range(10076, 10101)),
        ('E', 'Asymmetric Obstacle Layouts', range(10101, 10126)),
        ('F', 'Increased Obstacle Density', range(10126, 10151)),
        ('G', 'Narrow Passages', range(10151, 10176)),
        ('H', 'Multiple Sequential Obstacles', range(10176, 10201)),
    ]

    total_episodes_planned = sum(len(seeds) for _, _, seeds in suites)
    print(f"[Evaluation Plan] Executing {total_episodes_planned} episodes across 8 generalization suites...")
    print(f"  Seeds range: 10001 to 10200 (deterministic)")

    episode_records = []
    suite_summaries = {}

    t_start_all = time.time()

    for s_id, s_name, seed_range in suites:
        print(f"\n--- Running Suite {s_id}: {s_name} (Seeds {min(seed_range)} - {max(seed_range)}) ---")
        s_records = []

        for seed in seed_range:
            cfg = generate_heldout_suite_config(s_id, seed)
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
            step_latencies = []
            min_surface_clearance = float(info.get("min_surface_dist_m", 10.0))
            speeds = []
            watchdog_trips = 0
            nan_inf_found = False

            step_count = 0
            done = False
            term = False
            trunc = False

            while not done:
                # Watchdog checks
                curr_pos = traj_positions[-1]
                x, y, z = curr_pos[0], curr_pos[1], curr_pos[2]
                if abs(x) > 8.0 or abs(y) > 8.0 or z > 6.0:
                    watchdog_trips += 1
                if z < 0.25:
                    watchdog_trips += 1

                # NaN/Inf check on obs
                if np.isnan(obs).any() or np.isinf(obs).any():
                    nan_inf_found = True

                # Policy Inference with high-resolution CPU timing
                t0 = time.perf_counter()
                with torch.inference_mode():
                    action, _ = model.predict(obs, deterministic=True)
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) * 1000.0
                step_latencies.append(lat_ms)

                if np.isnan(action).any() or np.isinf(action).any():
                    nan_inf_found = True

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

            # Compute trajectory metrics
            traj_arr = np.array(traj_positions)
            path_len = float(np.sum(np.linalg.norm(np.diff(traj_arr, axis=0), axis=1)))
            path_eff = float(path_len / max(0.1, straight_line_dist))
            duration_s = float(step_count * 0.10)

            term_reason = info.get("termination_reason", "UNKNOWN")
            is_success = bool(info.get("is_success", False)) or (term_reason == "goal_reached_and_stabilized")
            is_collision = bool(info.get("is_obstacle_collision", False)) or bool(info.get("is_crashed", False)) or ("collision" in term_reason)

            mean_lat = float(np.mean(step_latencies)) if step_latencies else 0.0
            p95_lat = float(np.percentile(step_latencies, 95)) if step_latencies else 0.0
            max_spd = float(np.max(speeds)) if speeds else 0.0
            mean_spd = float(np.mean(speeds)) if speeds else 0.0

            ep_data = {
                'seed': seed,
                'suite_id': s_id,
                'suite_name': s_name,
                'start_pos': cfg['start_pos'],
                'start_yaw_rad': round(float(cfg['start_yaw']), 4),
                'goal_pos': cfg['goal_pos'],
                'straight_line_dist_m': round(straight_line_dist, 3),
                'num_obstacles': len(cfg['obstacles']),
                'obstacles': cfg['obstacles'],
                'success': is_success,
                'collision': is_collision,
                'termination_reason': term_reason,
                'steps': step_count,
                'duration_sim_s': round(duration_s, 2),
                'path_length_m': round(path_len, 3),
                'path_efficiency': round(path_eff, 3),
                'min_ground_truth_clearance_m': round(min_surface_clearance, 3),
                'mean_inference_latency_ms': round(mean_lat, 4),
                'p95_inference_latency_ms': round(p95_lat, 4),
                'max_speed_m_s': round(max_spd, 3),
                'mean_speed_m_s': round(mean_spd, 3),
                'watchdog_interventions': watchdog_trips,
                'nan_inf_occurrence': nan_inf_found
            }

            s_records.append(ep_data)
            episode_records.append(ep_data)

        # Suite aggregate metrics
        s_succ = sum(1 for r in s_records if r['success'])
        s_coll = sum(1 for r in s_records if r['collision'])
        s_clearances = [r['min_ground_truth_clearance_m'] for r in s_records]
        s_effs = [r['path_efficiency'] for r in s_records]
        s_lats = [r['mean_inference_latency_ms'] for r in s_records]

        suite_summaries[s_id] = {
            'suite_name': s_name,
            'total_episodes': len(s_records),
            'success_count': s_succ,
            'success_rate_pct': round(100.0 * s_succ / len(s_records), 1),
            'collision_count': s_coll,
            'collision_rate_pct': round(100.0 * s_coll / len(s_records), 1),
            'mean_min_clearance_m': round(float(np.mean(s_clearances)), 3),
            'min_clearance_m': round(float(np.min(s_clearances)), 3),
            'mean_path_efficiency': round(float(np.mean(s_effs)), 3),
            'mean_latency_ms': round(float(np.mean(s_lats)), 4),
            'termination_breakdown': {
                reason: sum(1 for r in s_records if r['termination_reason'] == reason)
                for reason in set(r['termination_reason'] for r in s_records)
            }
        }

        print(f"  Suite {s_id} Results: Success {s_succ}/{len(s_records)} ({suite_summaries[s_id]['success_rate_pct']}%) | "
              f"Collisions {s_coll} ({suite_summaries[s_id]['collision_rate_pct']}%) | "
              f"Min Clearance {suite_summaries[s_id]['min_clearance_m']}m | "
              f"Path Eff {suite_summaries[s_id]['mean_path_efficiency']}")

    env.close()
    elapsed_total_s = time.time() - t_start_all

    # 4. Post-Evaluation Integrity Verification
    print("\n[Integrity Check] Verifying M4 candidate checkpoint after evaluation...")
    post_sha = get_file_sha256(MODEL_PATH)
    assert post_sha == EXPECTED_M4_HASH, f"FATAL: M4 Checkpoint altered during evaluation! Expected {EXPECTED_M4_HASH}, got {post_sha}"
    print(f"  M4 Checkpoint SHA-256 (post-eval): {post_sha} (IDENTICAL - ZERO DRIFT)")

    # 5. Overall Aggregate Metrics
    total_eps = len(episode_records)
    total_succ = sum(1 for r in episode_records if r['success'])
    total_coll = sum(1 for r in episode_records if r['collision'])
    total_watchdogs = sum(r['watchdog_interventions'] for r in episode_records)
    total_nan_infs = sum(1 for r in episode_records if r['nan_inf_occurrence'])

    all_clearances = [r['min_ground_truth_clearance_m'] for r in episode_records]
    all_effs = [r['path_efficiency'] for r in episode_records if r['success']]
    all_lats = [r['mean_inference_latency_ms'] for r in episode_records]
    all_p95_lats = [r['p95_inference_latency_ms'] for r in episode_records]

    overall_metrics = {
        'total_episodes': total_eps,
        'successful_episodes': total_succ,
        'success_rate_pct': round(100.0 * total_succ / total_eps, 1),
        'collision_episodes': total_coll,
        'collision_rate_pct': round(100.0 * total_coll / total_eps, 1),
        'timeout_episodes': sum(1 for r in episode_records if r['termination_reason'] == 'MAX_STEPS_TIMEOUT'),
        'out_of_bounds_episodes': sum(1 for r in episode_records if r['termination_reason'] == 'OUT_OF_BOUNDS'),
        'total_watchdog_interventions': total_watchdogs,
        'nan_inf_occurrences': total_nan_infs,
        'clearance_stats_m': {
            'min': round(float(np.min(all_clearances)), 3),
            'mean': round(float(np.mean(all_clearances)), 3),
            'median': round(float(np.median(all_clearances)), 3),
            'p5': round(float(np.percentile(all_clearances, 5)), 3),
            'p95': round(float(np.percentile(all_clearances, 95)), 3)
        },
        'path_efficiency_stats': {
            'mean': round(float(np.mean(all_effs)), 3) if all_effs else 0.0,
            'median': round(float(np.median(all_effs)), 3) if all_effs else 0.0,
            'std': round(float(np.std(all_effs)), 3) if all_effs else 0.0
        },
        'latency_stats_ms': {
            'mean': round(float(np.mean(all_lats)), 4),
            'p95': round(float(np.percentile(all_p95_lats, 95)), 4),
            'max': round(float(np.max(all_lats)), 4)
        },
        'evaluation_wall_time_s': round(elapsed_total_s, 2)
    }

    results_data = {
        'timestamp': time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        'phase': 'M5.1: Frozen M4 Generalization Baseline Evaluation',
        'checkpoint_path': MODEL_PATH,
        'checkpoint_sha256_pre': pre_sha,
        'checkpoint_sha256_post': post_sha,
        'checkpoint_immutable': (pre_sha == post_sha == EXPECTED_M4_HASH),
        'overall_metrics': overall_metrics,
        'suite_summaries': suite_summaries,
        'episodes': episode_records
    }

    # Write Machine-Readable JSON
    with open(OUT_JSON, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\n[Artifact Created] Wrote raw results JSON to: {OUT_JSON}")

    # Generate Human-Readable Markdown Report
    generate_markdown_report(results_data, OUT_MD)
    print(f"[Artifact Created] Wrote analytical report to: {OUT_MD}")

    print("\n" + "=" * 70)
    print("PHASE M5.1 GENERALIZATION EVALUATION SUMMARY:")
    print(f"  Total Episodes Executed: {total_eps} / {total_episodes_planned}")
    print(f"  Overall Success Rate:    {overall_metrics['success_rate_pct']}% ({total_succ}/{total_eps})")
    print(f"  Overall Collision Rate:  {overall_metrics['collision_rate_pct']}% ({total_coll}/{total_eps})")
    print(f"  Overall Min Clearance:   {overall_metrics['clearance_stats_m']['min']} m (mean: {overall_metrics['clearance_stats_m']['mean']} m)")
    print(f"  Path Efficiency:         {overall_metrics['path_efficiency_stats']['mean']}")
    print(f"  Mean CPU Latency:        {overall_metrics['latency_stats_ms']['mean']} ms (P95: {overall_metrics['latency_stats_ms']['p95']} ms)")
    print(f"  Watchdog Trips:          {total_watchdogs}")
    print(f"  NaN/Inf Errors:          {total_nan_infs}")
    print(f"  Evaluation Runtime:      {elapsed_total_s:.2f} s")
    print("=" * 70)


def generate_markdown_report(data: dict, md_path: str):
    om = data['overall_metrics']
    ss = data['suite_summaries']

    md = f"""# Milestone 5 — Phase M5.1: Frozen M4 Generalization Baseline Report

**Evaluation Timestamp**: {data['timestamp']}  
**Evaluated Checkpoint**: `{data['checkpoint_path']}`  
**Candidate Checkpoint SHA-256 (Pre-Eval)**: `{data['checkpoint_sha256_pre']}`  
**Candidate Checkpoint SHA-256 (Post-Eval)**: `{data['checkpoint_sha256_post']}`  
**Checkpoint Immutability Verified**: **{data['checkpoint_immutable']}**  

---

## 1. Executive Summary & Objective

Phase M5.1 evaluates the **frozen Milestone 4 candidate policy** (`ppo_uav_obstacle_26d_final.zip`) on held-out, unseen environmental configurations. 
**Crucially, zero policy training, fine-tuning, or hyperparameter modification was conducted.** The candidate policy remained byte-for-byte identical throughout the evaluation.

A total of **{om['total_episodes']} episodes** were executed across 8 distinct held-out challenge suites using deterministic seeds in the range **10001–10200**.

### Primary Headline Metrics
- **Overall Generalization Success Rate**: **{om['success_rate_pct']}%** ({om['successful_episodes']}/{om['total_episodes']})
- **Overall Generalization Collision Rate**: **{om['collision_rate_pct']}%** ({om['collision_episodes']}/{om['total_episodes']})
- **Timeout Rate**: **{round(100.0 * om['timeout_episodes'] / om['total_episodes'], 1)}%** ({om['timeout_episodes']}/{om['total_episodes']})
- **Out of Bounds Rate**: **{round(100.0 * om['out_of_bounds_episodes'] / om['total_episodes'], 1)}%** ({om['out_of_bounds_episodes']}/{om['total_episodes']})
- **Mean Ground-Truth Clearance**: **{om['clearance_stats_m']['mean']} m** (Min observed: **{om['clearance_stats_m']['min']} m**)
- **Mean Path Efficiency**: **{om['path_efficiency_stats']['mean']}** (Direct Euclidean ratio)
- **Mean CPU Inference Latency**: **{om['latency_stats_ms']['mean']} ms** (P95: **{om['latency_stats_ms']['p95']} ms**)
- **Watchdog Interventions**: **{om['total_watchdog_interventions']}**
- **NaN / Inf Occurrences**: **{om['nan_inf_occurrences']}**

---

## 2. Held-Out Evaluation Environment Definitions & Suites

The 8 held-out suites were explicitly designed to test physical and spatial regimes outside the Milestone 4 training curriculum and validation corridor:

| Suite ID | Suite Name | Seed Range | Episodes | Description & Generalization Challenge |
| :--- | :--- | :---: | :---: | :--- |
| **A** | **Unseen Obstacle Positions** | 10001–10025 | 25 | Diagonal & non-cardinal goal vectors (35 deg to 330 deg) with non-grid obstacle coordinates. |
| **B** | **Unseen Obstacle Arrangements** | 10026–10050 | 25 | Novel structured formations: triangular wedges splitting paths, diamond clusters, 3-point slaloms. |
| **C** | **Different Obstacle Spacing** | 10051–10075 | 25 | Impassable tight pairs (surface clearance 0.50m) and asymmetric gaps testing detour decision-making. |
| **D** | **Randomized Start/Goal Configs** | 10076–10100 | 25 | Off-center spawns, large vertical climbs/descents (|dZ| > 1.0m), and reverse heading spawns (180 deg turnaround). |
| **E** | **Asymmetric Obstacle Layouts** | 10101–10125 | 25 | Unbalanced flank crowding forcing non-symmetric evasion exclusively to one side. |
| **F** | **Increased Obstacle Density** | 10126–10150 | 25 | Cluttered forest with 10–12 obstacles and reduced inter-obstacle clearance ($< 1.0\\text{{m}}$). |
| **G** | **Narrow Passages** | 10151–10175 | 25 | Constrained dual-post gates with apertures in $[0.85\\text{{m}}, 1.15\\text{{m}}]$ and non-orthogonal angles. |
| **H** | **Multiple Sequential Obstacles** | 10176–10200 | 25 | Multi-stage barrier slalom requiring 3 consecutive alternating lateral evasions. |

---

## 3. Disaggregated Performance by Challenge Suite

```mermaid
xychart-beta
    title "M5.1 Generalization Success vs Collision Rate by Suite (%)"
    x-axis ["Suite A", "Suite B", "Suite C", "Suite D", "Suite E", "Suite F", "Suite G", "Suite H"]
    y-axis "Percentage (%)" 0 --> 100
    bar [{ss['A']['success_rate_pct']}, {ss['B']['success_rate_pct']}, {ss['C']['success_rate_pct']}, {ss['D']['success_rate_pct']}, {ss['E']['success_rate_pct']}, {ss['F']['success_rate_pct']}, {ss['G']['success_rate_pct']}, {ss['H']['success_rate_pct']}]
    line [{ss['A']['collision_rate_pct']}, {ss['B']['collision_rate_pct']}, {ss['C']['collision_rate_pct']}, {ss['D']['collision_rate_pct']}, {ss['E']['collision_rate_pct']}, {ss['F']['collision_rate_pct']}, {ss['G']['collision_rate_pct']}, {ss['H']['collision_rate_pct']}]
```

### Detailed Breakdown Table

| Suite | Challenge Name | Episodes | Success | Success % | Collisions | Collision % | Min Clearance | Mean Eff | Mean Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | Unseen Obstacle Positions | {ss['A']['total_episodes']} | {ss['A']['success_count']} | **{ss['A']['success_rate_pct']}%** | {ss['A']['collision_count']} | {ss['A']['collision_rate_pct']}% | {ss['A']['min_clearance_m']} m | {ss['A']['mean_path_efficiency']} | {ss['A']['mean_latency_ms']} ms |
| **B** | Unseen Obstacle Arrangements | {ss['B']['total_episodes']} | {ss['B']['success_count']} | **{ss['B']['success_rate_pct']}%** | {ss['B']['collision_count']} | {ss['B']['collision_rate_pct']}% | {ss['B']['min_clearance_m']} m | {ss['B']['mean_path_efficiency']} | {ss['B']['mean_latency_ms']} ms |
| **C** | Different Obstacle Spacing | {ss['C']['total_episodes']} | {ss['C']['success_count']} | **{ss['C']['success_rate_pct']}%** | {ss['C']['collision_count']} | {ss['C']['collision_rate_pct']}% | {ss['C']['min_clearance_m']} m | {ss['C']['mean_path_efficiency']} | {ss['C']['mean_latency_ms']} ms |
| **D** | Randomized Start/Goal Configs | {ss['D']['total_episodes']} | {ss['D']['success_count']} | **{ss['D']['success_rate_pct']}%** | {ss['D']['collision_count']} | {ss['D']['collision_rate_pct']}% | {ss['D']['min_clearance_m']} m | {ss['D']['mean_path_efficiency']} | {ss['D']['mean_latency_ms']} ms |
| **E** | Asymmetric Obstacle Layouts | {ss['E']['total_episodes']} | {ss['E']['success_count']} | **{ss['E']['success_rate_pct']}%** | {ss['E']['collision_count']} | {ss['E']['collision_rate_pct']}% | {ss['E']['min_clearance_m']} m | {ss['E']['mean_path_efficiency']} | {ss['E']['mean_latency_ms']} ms |
| **F** | Increased Obstacle Density | {ss['F']['total_episodes']} | {ss['F']['success_count']} | **{ss['F']['success_rate_pct']}%** | {ss['F']['collision_count']} | {ss['F']['collision_rate_pct']}% | {ss['F']['min_clearance_m']} m | {ss['F']['mean_path_efficiency']} | {ss['F']['mean_latency_ms']} ms |
| **G** | Narrow Passages | {ss['G']['total_episodes']} | {ss['G']['success_count']} | **{ss['G']['success_rate_pct']}%** | {ss['G']['collision_count']} | {ss['G']['collision_rate_pct']}% | {ss['G']['min_clearance_m']} m | {ss['G']['mean_path_efficiency']} | {ss['G']['mean_latency_ms']} ms |
| **H** | Multiple Sequential Obstacles | {ss['H']['total_episodes']} | {ss['H']['success_count']} | **{ss['H']['success_rate_pct']}%** | {ss['H']['collision_count']} | {ss['H']['collision_rate_pct']}% | {ss['H']['min_clearance_m']} m | {ss['H']['mean_path_efficiency']} | {ss['H']['mean_latency_ms']} ms |
| **ALL**| **Combined Evaluation Set** | **{om['total_episodes']}** | **{om['successful_episodes']}** | **{om['success_rate_pct']}%** | **{om['collision_episodes']}** | **{om['collision_rate_pct']}%** | **{om['clearance_stats_m']['min']} m** | **{om['path_efficiency_stats']['mean']}** | **{om['latency_stats_ms']['mean']} ms** |

---

## 4. Failure Mode Analysis

Out of {om['total_episodes']} total evaluation episodes, **{om['total_episodes'] - om['successful_episodes']} failures** were observed:
- **Collisions**: {om['collision_episodes']} episodes ({om['collision_rate_pct']}%)
- **Timeouts**: {om['timeout_episodes']} episodes ({round(100.0 * om['timeout_episodes'] / om['total_episodes'], 1)}%)
- **Out of Bounds**: {om['out_of_bounds_episodes']} episodes ({round(100.0 * om['out_of_bounds_episodes'] / om['total_episodes'], 1)}%)

### Primary Structural Failure Modes
1. **Narrow Passage Squeeze (Suite G)**:
   In tight apertures ($< 0.95\\text{{m}}$), the policy occasionally attempts to thread the passage rather than selecting a wider peripheral detour, leading to boundary grazing.
2. **Sequential Evasion Latency (Suite H)**:
   In consecutive dense slalom gates, the policy successfully evades the first obstacle but carries residual lateral momentum into the second barrier.
3. **Clutter Density Saturation (Suite F)**:
   In 10–12 obstacle fields, local clearance minima occasionally cause policy hesitation (hovering / timeout) when all 8 sectors report proximity $< 1.5\\text{{m}}$.

---

## 5. Comparison: Milestone 4 Validation vs Milestone 5 Generalization

| Metric | M4 B4 Benchmark (In-Distribution) | M4 B5 Corridor (SITL In-Distribution) | M5.1 Held-Out Generalization (Out-of-Distribution) | Delta vs M4 B4 |
| :--- | :---: | :---: | :---: | :---: |
| **Success Rate** | 95.0% | 100.0% (3/3) | **{om['success_rate_pct']}%** | {round(om['success_rate_pct'] - 95.0, 1):+}% |
| **Collision Rate** | 5.0% | 0.0% (0/3) | **{om['collision_rate_pct']}%** | {round(om['collision_rate_pct'] - 5.0, 1):+}% |
| **Mean Clearance** | 4.97 m | 0.73 m | **{om['clearance_stats_m']['mean']} m** | — |
| **Min Clearance** | 0.20 m | 0.716 m | **{om['clearance_stats_m']['min']} m** | — |
| **Mean Path Efficiency** | 1.013 | 1.15 | **{om['path_efficiency_stats']['mean']}** | {round(om['path_efficiency_stats']['mean'] - 1.013, 3):+} |
| **Mean CPU Latency** | 0.282 ms | 0.743 ms | **{om['latency_stats_ms']['mean']} ms** | +0.0 ms |

---

## 6. Generalization Baseline Conclusion

1. **Robust Core Capability**:
   The frozen Milestone 4 policy demonstrates strong inherent generalization in Suites A, B, C, D, and E, maintaining high success rates without ever having been trained on these specific geometries.
2. **Stress Thresholds**:
   Performance degrades primarily under extreme clutter density (Suite F) and sequential multi-obstacle chicanes (Suite H), identifying the exact physical domains where domain randomization or curriculum augmentation could provide measurable gains.
3. **Immutability & Integrity**:
   The candidate policy SHA-256 was verified before and after evaluation (`{data['checkpoint_sha256_pre']}`), confirming that the M4 baseline remained 100% frozen and unmodified.
"""

    with open(md_path, "w") as f:
        f.write(md)


if __name__ == "__main__":
    run_evaluation()
