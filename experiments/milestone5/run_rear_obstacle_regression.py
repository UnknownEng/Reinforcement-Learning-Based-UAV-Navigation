"""
run_rear_obstacle_regression.py
--------------------------------
Milestone 5.4 Frozen Stage 2 Rear-Obstacle Regression Gate.

Executes the four deterministic rear-obstacle test cases against the existing Stage 2 checkpoint:
  1. Direct rear trap: obstacle at (-2.5, 0.0), radius 0.50 m
  2. Severe proximity trap: obstacle at (-1.5, 0.0), radius 0.40 m
  3. Lateral offset left: obstacle at (-2.5, +0.4), radius 0.50 m
  4. Lateral offset right: obstacle at (-2.5, -0.4), radius 0.50 m

Records per-step telemetry, computes all acceptance criteria, and exports JSON results.
"""

import argparse
import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch
from stable_baselines3 import PPO

import rl_env
from rl_env.uav_fast_env_v2 import UAVFastEnvV2, UAVObstacleEnvConfigV2
from rl_env.observation_26d import ObservationHandler26D
from rl_env.safety_shield import DeterministicSafetyShield, SafetyShieldConfig


def get_checkpoint_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def run_regression_case(
    case_id: str,
    case_name: str,
    obstacle_spec: Tuple[float, float, float],
    model: PPO,
    env: UAVFastEnvV2,
    shield: Optional[DeterministicSafetyShield] = None,
    max_steps: int = 100  # 10.0 seconds at 10 Hz
) -> Dict[str, Any]:
    ox, oy, orad = obstacle_spec
    opts = {
        'start_pos': [0.0, 0.0, 2.0],
        'start_yaw': 0.0,
        'goal_pos': [-6.0, 0.0, 2.0],
        'obstacles': [(ox, oy, orad)]
    }

    obs, info = env.reset(seed=101, options=opts)
    
    # Capture initial LiDAR sectors
    initial_lidar_clearances = {
        ObservationHandler26D.SECTOR_NAMES[i]: round(float(obs[18 + i]), 4)
        for i in range(8)
    }

    trajectory: List[Dict[str, Any]] = []
    terminated = False
    truncated = False
    step = 0
    min_center_dist = float("inf")
    min_surface_dist = float("inf")
    min_trajectory_clearance = float("inf")
    min_lidar_clearance = float("inf")
    
    is_collision = False
    termination_reason = "max_steps_exceeded"
    
    # Tracking criteria
    exceeded_reverse_speed_during_misalignment = False
    max_reverse_speed_misaligned = 0.0
    max_cmd_reverse_speed_misaligned = 0.0
    
    t0 = time.perf_counter()

    while not (terminated or truncated) and step < max_steps:
        # Policy action
        raw_action, _ = model.predict(obs, deterministic=True)
        raw_scaled_cmd = env.action_handler.scale_action(raw_action)
        state_curr = env.dynamics.get_state()

        if shield is not None:
            shielded_action, diag = shield.filter_action(
                obs=obs,
                action=raw_action,
                perception_metadata=getattr(env, "last_meta", None)
            )
            action_to_apply = shielded_action
            scaled_cmd = diag.shielded_cmd
        else:
            action_to_apply = raw_action
            scaled_cmd = raw_scaled_cmd
            diag = None
        
        # Heading error from observation
        goal_body = obs[:3]
        d_psi_rad = math.atan2(float(goal_body[1]), float(goal_body[0]))
        d_psi_deg = math.degrees(d_psi_rad)
        abs_d_psi_deg = abs(d_psi_deg)
        
        # Body frame velocity
        R_b_w = env.dynamics.state.quat
        # From obs[3:6]: linear velocity in body FLU frame
        vx_b = float(obs[3])
        vy_b = float(obs[4])
        vz_b = float(obs[5])
        
        # Distance to obstacle center & surface
        dist_center = math.hypot(state_curr.pos_world[0] - ox, state_curr.pos_world[1] - oy)
        dist_surface = dist_center - orad
        dist_trajectory = dist_surface - env.config.uav_radius  # clearance beyond UAV radius
        
        min_center_dist = min(min_center_dist, dist_center)
        min_surface_dist = min(min_surface_dist, dist_surface)
        min_trajectory_clearance = min(min_trajectory_clearance, dist_trajectory)
        
        # Reverse speed check when misalignment > 90 deg
        reverse_speed_b = -vx_b  # positive when moving backward
        cmd_reverse_speed = -float(scaled_cmd.vx)
        
        if abs_d_psi_deg > 90.0:
            if reverse_speed_b > 1.0:
                exceeded_reverse_speed_during_misalignment = True
            max_reverse_speed_misaligned = max(max_reverse_speed_misaligned, reverse_speed_b)
            max_cmd_reverse_speed_misaligned = max(max_cmd_reverse_speed_misaligned, cmd_reverse_speed)

        # Pre-step observation LiDAR sectors
        pre_lidar_sectors = [round(float(c), 4) for c in obs[18:26]]
        
        # Stopping distance estimate based on current physical body speed
        v_speed = math.hypot(vx_b, vy_b)
        a_max_dyn = 3.0
        k_delay_dyn = 0.25
        stopping_dist_est = 0.85 + k_delay_dyn * v_speed + (v_speed ** 2) / (2.0 * a_max_dyn)

        # Environment step
        obs, reward, terminated, truncated, info = env.step(action_to_apply)
        step += 1
        
        # Post-step state and clearances
        state_post = env.dynamics.state
        post_dist_center = math.hypot(state_post.pos_world[0] - ox, state_post.pos_world[1] - oy)
        post_dist_surface = post_dist_center - orad
        post_dist_trajectory = post_dist_surface - env.config.uav_radius
        
        min_center_dist = min(min_center_dist, post_dist_center)
        min_surface_dist = min(min_surface_dist, post_dist_surface)
        min_trajectory_clearance = min(min_trajectory_clearance, post_dist_trajectory)
        
        lidar_clr = float(info.get("min_clearance_m", 10.0))
        min_lidar_clearance = min(min_lidar_clearance, lidar_clr)
        
        rec = {
            "step": step,
            "d_psi_deg": round(d_psi_deg, 2),
            "raw_cmd_vx": round(float(raw_scaled_cmd.vx), 3),
            "cmd_vx": round(float(scaled_cmd.vx), 3),
            "cmd_vy": round(float(scaled_cmd.vy), 3),
            "cmd_vz": round(float(scaled_cmd.vz), 3),
            "cmd_yaw_rate": round(float(scaled_cmd.yaw_rate), 3),
            "vel_body_vx": round(vx_b, 3),
            "vel_body_vy": round(vy_b, 3),
            "vel_body_vz": round(vz_b, 3),
            "pos_x": round(float(state_curr.pos_world[0]), 4),
            "pos_y": round(float(state_curr.pos_world[1]), 4),
            "pos_z": round(float(state_curr.pos_world[2]), 4),
            "vel_world_x": round(float(state_curr.vel_world[0]), 4),
            "vel_world_y": round(float(state_curr.vel_world[1]), 4),
            "vel_world_z": round(float(state_curr.vel_world[2]), 4),
            "dist_center": round(dist_center, 4),
            "dist_surface": round(dist_surface, 4),
            "dist_trajectory_clearance": round(dist_trajectory, 4),
            "post_pos_x": round(float(state_post.pos_world[0]), 4),
            "post_pos_y": round(float(state_post.pos_world[1]), 4),
            "post_dist_center": round(post_dist_center, 4),
            "post_dist_surface": round(post_dist_surface, 4),
            "post_dist_trajectory_clearance": round(post_dist_trajectory, 4),
            "min_lidar_clearance": round(lidar_clr, 4),
            "lidar_sectors_8d": pre_lidar_sectors,
            "stopping_distance_est": round(stopping_dist_est, 4),
            "shield_intervened": diag.is_intervened if diag else False,
            "shield_interventions": diag.interventions if diag else [],
            "shield_scale_factor": round(float(diag.scale_factor), 3) if diag else 1.0
        }
        trajectory.append(rec)
        
        if info.get("is_crashed", False):
            is_collision = True
            termination_reason = info.get("termination_reason", "collision")
            break
        elif info.get("is_success", False):
            termination_reason = "success"
            break
        elif info.get("is_out_of_bounds", False):
            termination_reason = "out_of_bounds"
            break

    elapsed_s = time.perf_counter() - t0
    
    # Acceptance criteria checks
    pass_no_collision = not is_collision
    pass_surface_clearance = min_surface_dist >= 0.35  # UAV radius
    pass_trajectory_clearance = min_trajectory_clearance >= 0.50  # 0.50 m beyond UAV radius
    pass_reverse_speed = max_reverse_speed_misaligned <= 1.0  # reverse speed <= 1.0 m/s when yaw err > 90 deg
    pass_duration = (step <= max_steps) and (termination_reason == "success")
    
    all_passed = (
        pass_no_collision and
        pass_surface_clearance and
        pass_trajectory_clearance and
        pass_reverse_speed and
        pass_duration
    )
    
    return {
        "case_id": case_id,
        "case_name": case_name,
        "obstacle_spec": {
            "center_x": ox,
            "center_y": oy,
            "radius": orad,
            "initial_center_dist": round(math.hypot(ox, oy), 3),
            "initial_surface_dist": round(math.hypot(ox, oy) - orad, 3),
            "initial_physical_clearance": round(math.hypot(ox, oy) - orad - env.config.uav_radius, 3)
        },
        "initial_lidar_sectors": initial_lidar_clearances,
        "first_action_cmd": {
            "raw_cmd_vx": trajectory[0].get("raw_cmd_vx") if trajectory else None,
            "cmd_vx": trajectory[0]["cmd_vx"] if trajectory else None,
            "cmd_vy": trajectory[0]["cmd_vy"] if trajectory else None,
            "cmd_vz": trajectory[0]["cmd_vz"] if trajectory else None,
            "cmd_yaw_rate": trajectory[0]["cmd_yaw_rate"] if trajectory else None,
            "shield_intervened": trajectory[0].get("shield_intervened", False) if trajectory else False
        },
        "metrics": {
            "min_center_dist": round(min_center_dist, 4),
            "min_surface_dist": round(min_surface_dist, 4),
            "min_trajectory_clearance": round(min_trajectory_clearance, 4),
            "min_lidar_clearance": round(min_lidar_clearance, 4),
            "max_reverse_speed_misaligned": round(max_reverse_speed_misaligned, 3),
            "max_cmd_reverse_speed_misaligned": round(max_cmd_reverse_speed_misaligned, 3),
            "steps_to_termination": step,
            "elapsed_time_s": round(step * 0.1, 2),  # simulated time
            "wall_clock_s": round(elapsed_s, 4),
            "is_collision": is_collision,
            "termination_reason": termination_reason
        },
        "criteria_checks": {
            "criterion_1_zero_collision": {
                "passed": pass_no_collision,
                "detail": f"Collision occurred: {is_collision} ({termination_reason})"
            },
            "criterion_2a_surface_clearance_gte_0_35m": {
                "passed": pass_surface_clearance,
                "threshold": 0.35,
                "measured": round(min_surface_dist, 4),
                "detail": f"Min surface clearance: {min_surface_dist:.4f} m (Required >= 0.350 m)"
            },
            "criterion_2b_trajectory_clearance_gte_0_50m": {
                "passed": pass_trajectory_clearance,
                "threshold": 0.50,
                "measured": round(min_trajectory_clearance, 4),
                "detail": f"Min trajectory clearance: {min_trajectory_clearance:.4f} m (Required >= 0.500 m)"
            },
            "criterion_3_reverse_speed_misaligned_lte_1_0ms": {
                "passed": pass_reverse_speed,
                "threshold": 1.0,
                "measured": round(max_reverse_speed_misaligned, 3),
                "detail": f"Max reverse speed when |d_psi| > 90 deg: {max_reverse_speed_misaligned:.3f} m/s (Required <= 1.000 m/s)"
            },
            "criterion_4_completed_within_10s": {
                "passed": pass_duration,
                "threshold_steps": max_steps,
                "threshold_s": 10.0,
                "measured_steps": step,
                "measured_s": round(step * 0.1, 2),
                "detail": f"Steps: {step} ({step*0.1:.2f} s), Reason: {termination_reason}"
            }
        },
        "case_verdict": "PASS" if all_passed else "FAIL",
        "trajectory": trajectory
    }


def main():
    parser = argparse.ArgumentParser(description="Rear-Obstacle Regression Gate")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/milestone5/models/ppo_m5_stage2_64x64.zip",
        help="Path to checkpoint model to evaluate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/milestone5/m5_stage2_rear_obstacle_regression.json",
        help="Path to output JSON summary"
    )
    parser.add_argument(
        "--gate-name",
        type=str,
        default="Frozen Stage 2 Rear-Obstacle Regression Gate",
        help="Name of the regression gate"
    )
    parser.add_argument(
        "--shield",
        action="store_true",
        help="Enable deterministic hierarchical safety shield"
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"REAR-OBSTACLE REGRESSION GATE: {args.gate_name}")
    print("=" * 80)

    ckpt_path = args.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint missing: {ckpt_path}")
        
    hash_pre = get_checkpoint_sha256(ckpt_path)
    print(f"[CHECKPOINT PRE-VERIFY] {ckpt_path}")
    print(f"  SHA-256: {hash_pre}")

    # Baseline checkpoints check
    m4_path = "experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip"
    expected_m4_hash = "06907c9fa636e5f268ca55750107918370e445313ad1ac01207c15a4c5935308"
    hash_m4_pre = get_checkpoint_sha256(m4_path)
    assert hash_m4_pre == expected_m4_hash, "M4 baseline hash mismatch!"

    # Load policy and env
    print(f"\n[ENVIRONMENT SETUP] Initializing UAVFastNav-v2 with weights from {ckpt_path}...")
    model = PPO.load(ckpt_path, device="cpu")
    config = UAVObstacleEnvConfigV2(curriculum_stage=2, enable_noise=False)
    env = UAVFastEnvV2(config=config)

    shield = DeterministicSafetyShield() if args.shield else None
    if shield is not None:
        print("[SAFETY SHIELD ACTIVE] Deterministic Hierarchical Combined Safety Shield enabled.")

    test_cases = [
        ("R1", "Direct Rear Trap", (-2.5, 0.0, 0.50)),
        ("R2", "Severe Proximity Trap", (-1.5, 0.0, 0.40)),
        ("R3", "Lateral Offset Left Trap", (-2.5, 0.40, 0.50)),
        ("R4", "Lateral Offset Right Trap", (-2.5, -0.40, 0.50))
    ]

    all_case_results = []
    
    for cid, cname, ospec in test_cases:
        print(f"\n--- Running Regression Case {cid}: {cname} ---")
        print(f"  Obstacle: ({ospec[0]}, {ospec[1]}), Radius: {ospec[2]} m")
        res = run_regression_case(cid, cname, ospec, model, env, shield=shield)
        print(f"  Verdict: {res['case_verdict']}")
        print(f"  Termination: {res['metrics']['termination_reason']} at step {res['metrics']['steps_to_termination']}")
        print(f"  Min Surface Dist: {res['metrics']['min_surface_dist']:.4f} m")
        print(f"  Max Reverse Speed (|d_psi| > 90 deg): {res['metrics']['max_reverse_speed_misaligned']:.3f} m/s")
        for crit_name, crit in res["criteria_checks"].items():
            status_str = "PASS" if crit["passed"] else "FAIL"
            print(f"    [{status_str}] {crit_name}: {crit['detail']}")
        all_case_results.append(res)

    env.close()

    # Post-check hashes
    hash_post = get_checkpoint_sha256(ckpt_path)
    hash_m4_post = get_checkpoint_sha256(m4_path)
    assert hash_post == hash_pre, "Evaluated checkpoint was modified during test!"
    assert hash_m4_post == expected_m4_hash, "Milestone 4 checkpoint was modified during test!"
    print(f"\n[CHECKPOINT POST-VERIFY]")
    print(f"  Evaluated Checkpoint SHA-256: {hash_post} (BIT-IDENTICAL)")
    print(f"  Milestone 4 SHA-256: {hash_m4_post} (BIT-IDENTICAL)")

    # Overall Verdict
    all_passed = all(r["case_verdict"] == "PASS" for r in all_case_results)
    overall_verdict = "PASS" if all_passed else "FAIL"
    print(f"\n================================================================================")
    print(f"OVERALL REGRESSION GATE VERDICT: {overall_verdict}")
    print(f"================================================================================")

    # Save JSON summary
    json_path = args.output
    summary_data = {
        "gate_name": args.gate_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256_pre": hash_pre,
        "checkpoint_sha256_post": hash_post,
        "m4_baseline_sha256": hash_m4_post,
        "overall_verdict": overall_verdict,
        "cases": all_case_results
    }
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"[ARTIFACT] Machine-readable telemetry saved to {json_path}")


if __name__ == "__main__":
    main()
