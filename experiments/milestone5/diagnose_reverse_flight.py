"""
diagnose_reverse_flight.py
--------------------------
Deterministic Diagnostic for Reverse-Flight Behavior and Rear Obstacle Navigation.

Objectives:
1. Verify the environment's body-frame action-to-motion mapping (REP-103 FLU).
2. Evaluate the saved Stage 0 policy on a deterministic scenario with an obstacle directly behind the UAV.
3. Record step-by-step action commands, motion, and minimum clearance to the rear obstacle.
"""

import math
from typing import Dict, Any, List
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO

from rl_env.uav_fast_env_v2 import UAVFastEnvV2, UAVObstacleEnvConfigV2


def verify_body_frame_mapping() -> Dict[str, Any]:
    """Verify body-frame action to motion mapping with pure directional commands."""
    config = UAVObstacleEnvConfigV2(enable_noise=False)
    env = UAVFastEnvV2(config=config)

    # 1. Forward throttle: action [1.0, 0, 0, 0]
    env.reset(options={'start_pos': [0.0, 0.0, 2.0], 'start_yaw': 0.0, 'goal_pos': [5.0, 0.0, 2.0]})
    obs, _, _, _, _ = env.step(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    state_fwd = env.dynamics.get_state()
    fwd_valid = (state_fwd.vel_world[0] > 0.0) and (abs(state_fwd.vel_world[1]) < 0.01)

    # 2. Reverse throttle: action [-1.0, 0, 0, 0]
    env.reset(options={'start_pos': [0.0, 0.0, 2.0], 'start_yaw': 0.0, 'goal_pos': [-5.0, 0.0, 2.0]})
    obs, _, _, _, _ = env.step(np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    state_rev = env.dynamics.get_state()
    rev_valid = (state_rev.vel_world[0] < 0.0) and (abs(state_rev.vel_world[1]) < 0.01)

    # 3. Lateral Left throttle: action [0, 1.0, 0, 0]
    env.reset(options={'start_pos': [0.0, 0.0, 2.0], 'start_yaw': 0.0, 'goal_pos': [0.0, 5.0, 2.0]})
    obs, _, _, _, _ = env.step(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    state_left = env.dynamics.get_state()
    left_valid = (state_left.vel_world[1] > 0.0) and (abs(state_left.vel_world[0]) < 0.01)

    # 4. Positive Yaw Rate: action [0, 0, 0, 1.0]
    env.reset(options={'start_pos': [0.0, 0.0, 2.0], 'start_yaw': 0.0, 'goal_pos': [5.0, 0.0, 2.0]})
    obs, _, _, _, _ = env.step(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
    state_yaw = env.dynamics.get_state()
    yaw_valid = (state_yaw.body_rates[2] > 0.0) and (state_yaw.yaw > 0.0)

    env.close()

    mapping_results = {
        "forward_command_motion_valid": bool(fwd_valid),
        "reverse_command_motion_valid": bool(rev_valid),
        "left_command_motion_valid": bool(left_valid),
        "yaw_rate_command_motion_valid": bool(yaw_valid),
        "forward_vel_x": round(float(state_fwd.vel_world[0]), 3),
        "reverse_vel_x": round(float(state_rev.vel_world[0]), 3),
        "left_vel_y": round(float(state_left.vel_world[1]), 3),
        "yaw_rate_r": round(float(state_yaw.body_rates[2]), 3),
    }
    return mapping_results


def run_rear_obstacle_diagnostic(
    checkpoint_path: str = "experiments/milestone5/models/ppo_m5_pilot_64x64.zip"
) -> Dict[str, Any]:
    """Run saved Stage 0 pilot policy on scenario with obstacle directly behind UAV."""
    model = PPO.load(checkpoint_path, device="cpu")

    config = UAVObstacleEnvConfigV2(
        curriculum_stage=0,
        enable_noise=False
    )
    env = UAVFastEnvV2(config=config)

    # UAV at (0, 0, 2), yaw = 0 (facing +x)
    # Goal at (-6.0, 0, 2) (180 deg dead rear)
    # Obstacle directly behind at (-2.5, 0.0), radius = 0.50m
    rear_obstacle = [(-2.5, 0.0, 0.50)]
    opts = {
        'start_pos': [0.0, 0.0, 2.0],
        'start_yaw': 0.0,
        'goal_pos': [-6.0, 0.0, 2.0],
        'obstacles': rear_obstacle
    }

    obs, info = env.reset(seed=101, options=opts)
    trajectory: List[Dict[str, Any]] = []

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

        # Distance to rear obstacle center (-2.5, 0.0)
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
        "initial_surface_distance_m": 2.00,
        "obstacle_position": [-2.5, 0.0],
        "obstacle_radius_m": 0.50,
        "min_clearance_reached_m": round(min_clearance_overall, 3),
        "is_collision": bool(is_collision),
        "termination_reason": str(termination_reason),
        "total_steps": step,
        "trajectory_sample": trajectory[:15]  # first 15 steps
    }


def main():
    print("=" * 70)
    print("REVERSE-FLIGHT BEHAVIOR & REAR OBSTACLE DIAGNOSTIC")
    print("=" * 70)

    # 1. Action-to-motion mapping
    mapping = verify_body_frame_mapping()
    print("[MAPPING CHECK]")
    for k, v in mapping.items():
        print(f"  {k}: {v}")
    all_mapping_ok = all(v is True for k, v in mapping.items() if "valid" in k)
    print(f"  -> Body FLU Mapping Integrity: {'PASS' if all_mapping_ok else 'FAIL'}")

    # 2. Rear obstacle diagnostic
    print("\n[REAR OBSTACLE DIAGNOSTIC WITH SAVED STAGE 0 POLICY]")
    diag = run_rear_obstacle_diagnostic()
    print(f"  Initial Clearance to Obstacle: {diag['initial_surface_distance_m']:.2f} m")
    print(f"  Min Clearance Reached:         {diag['min_clearance_reached_m']:.3f} m")
    print(f"  Collision Occurred:            {diag['is_collision']} ({diag['termination_reason']})")
    print(f"  Steps to Termination:          {diag['total_steps']}")

    print("\n  Trajectory Progression (First 10 Steps):")
    print("  Step | d_psi (deg) | cmd_vx | cmd_vy | cmd_yaw | pos_x  | pos_y  | dist_to_obs")
    for r in diag["trajectory_sample"][:10]:
        print(f"  {r['step']:4d} | {r['d_psi_deg']:11.1f} | {r['cmd_vx']:6.2f} | {r['cmd_vy']:6.2f} | "
              f"{r['cmd_yaw_rate']:7.2f} | {r['pos_x']:6.2f} | {r['pos_y']:6.2f} | {r['dist_obs_surface']:11.3f} m")
    print("=" * 70)


if __name__ == "__main__":
    main()
