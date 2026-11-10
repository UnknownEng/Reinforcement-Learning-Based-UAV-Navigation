"""
run_safety_shield_generalization_diagnostics.py
-----------------------------------------------
Executes 13 deterministic generalization safety diagnostic scenarios for the
Hierarchical Combined Safety Shield, verifying both safety enforcement and
absence of false interventions across edge cases.
"""

import json
import math
import os
import sys
import time
from typing import Dict, Any, List, Tuple

import numpy as np

import rl_env
from rl_env.safety_shield import (
    DeterministicSafetyShield,
    SafetyShieldConfig,
    SafetyShieldDiagnostic
)
from rl_env.observation_26d import Observation26DMetadata, SectorStatus


def create_base_obs(
    goal_body: Tuple[float, float, float] = (5.0, 0.0, 0.0),
    vel_body: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    clearances: List[float] = None
) -> np.ndarray:
    """Constructs a synthetic 26D observation vector."""
    obs = np.zeros(26, dtype=np.float32)
    obs[0:3] = goal_body
    obs[3:6] = vel_body
    obs[6:10] = [1.0, 0.0, 0.0, 0.0]  # identity quat
    obs[10:14] = 0.0                  # body rates
    obs[14:18] = 0.0                  # prev action
    if clearances is None:
        obs[18:26] = 10.0             # open space
    else:
        obs[18:26] = np.array(clearances, dtype=np.float32)
    return obs


def run_all_diagnostics() -> Dict[str, Any]:
    shield = DeterministicSafetyShield(SafetyShieldConfig())
    results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # GEN-01: Open-space rear goal (N=0) -> verify no false intervention
    # -------------------------------------------------------------------------
    obs_1 = create_base_obs(goal_body=(-6.0, 0.0, 0.0), clearances=[10.0]*8)
    act_1 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_act_1, diag_1 = shield.filter_action(obs_1, act_1)
    pass_1 = (not diag_1.is_intervened) and (abs(s_act_1[0] - (-1.0)) < 1e-5)
    results.append({
        "test_id": "GEN-01",
        "name": "Open-space rear goal reverse flight (no false intervention)",
        "passed": bool(pass_1),
        "details": {
            "is_intervened": diag_1.is_intervened,
            "shielded_act": [float(x) for x in s_act_1],
            "interventions": diag_1.interventions
        }
    })

    # -------------------------------------------------------------------------
    # GEN-02: Rear obstacle farther than stopping threshold (x = -4.0m)
    # -------------------------------------------------------------------------
    # Distance to obstacle surface: 4.0 - 0.5 (radius) = 3.5m > 3.0m threshold
    clearances_2 = [10.0, 10.0, 10.0, 10.0, 3.65, 10.0, 10.0, 10.0]
    obs_2 = create_base_obs(goal_body=(-6.0, 0.0, 0.0), clearances=clearances_2)
    act_2 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_act_2, diag_2 = shield.filter_action(obs_2, act_2)
    pass_2 = (not diag_2.layer1_active) and (not diag_2.layer2_active)
    results.append({
        "test_id": "GEN-02",
        "name": "Rear obstacle outside threshold (d=3.65m > 3.0m)",
        "passed": bool(pass_2),
        "details": {
            "is_intervened": diag_2.is_intervened,
            "layer1_active": diag_2.layer1_active,
            "layer2_active": diag_2.layer2_active,
            "shielded_act": [float(x) for x in s_act_2]
        }
    })

    # -------------------------------------------------------------------------
    # GEN-03: Rear obstacle inside stopping threshold (x = -2.8m, d = 2.45m < 3.0m)
    # -------------------------------------------------------------------------
    clearances_3 = [10.0, 10.0, 10.0, 10.0, 2.45, 10.0, 10.0, 10.0]
    obs_3 = create_base_obs(goal_body=(-6.0, 0.0, 0.0), clearances=clearances_3)
    act_3 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_act_3, diag_3 = shield.filter_action(obs_3, act_3)
    pass_3 = diag_3.layer1_active and (s_act_3[0] >= 0.0)
    results.append({
        "test_id": "GEN-03",
        "name": "Rear obstacle inside threshold (d=2.45m < 3.0m)",
        "passed": bool(pass_3),
        "details": {
            "layer1_active": diag_3.layer1_active,
            "shielded_cmd_vx": diag_3.shielded_cmd.vx,
            "interventions": diag_3.interventions
        }
    })

    # -------------------------------------------------------------------------
    # GEN-04: Rear-left obstacle (sector 3 BL obstructed, d = 2.2m)
    # -------------------------------------------------------------------------
    clearances_4 = [10.0, 10.0, 10.0, 2.20, 10.0, 10.0, 10.0, 10.0]
    obs_4 = create_base_obs(goal_body=(-6.0, 0.0, 0.0), clearances=clearances_4)
    act_4 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_act_4, diag_4 = shield.filter_action(obs_4, act_4)
    pass_4 = diag_4.layer1_active and (s_act_4[0] >= 0.0)
    results.append({
        "test_id": "GEN-04",
        "name": "Rear-left obstacle in sector 3 (d=2.20m)",
        "passed": bool(pass_4),
        "details": {
            "layer1_active": diag_4.layer1_active,
            "shielded_cmd_vx": diag_4.shielded_cmd.vx,
            "interventions": diag_4.interventions
        }
    })

    # -------------------------------------------------------------------------
    # GEN-05: Rear-right obstacle (sector 5 BR obstructed, d = 2.2m)
    # -------------------------------------------------------------------------
    clearances_5 = [10.0, 10.0, 10.0, 10.0, 10.0, 2.20, 10.0, 10.0]
    obs_5 = create_base_obs(goal_body=(-6.0, 0.0, 0.0), clearances=clearances_5)
    act_5 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_act_5, diag_5 = shield.filter_action(obs_5, act_5)
    pass_5 = diag_5.layer1_active and (s_act_5[0] >= 0.0)
    results.append({
        "test_id": "GEN-05",
        "name": "Rear-right obstacle in sector 5 (d=2.20m)",
        "passed": bool(pass_5),
        "details": {
            "layer1_active": diag_5.layer1_active,
            "shielded_cmd_vx": diag_5.shielded_cmd.vx,
            "interventions": diag_5.interventions
        }
    })

    # -------------------------------------------------------------------------
    # GEN-06: Front obstacle while moving forward (sector 0, d = 1.0m, a0 = +1.0)
    # -------------------------------------------------------------------------
    clearances_6 = [1.00, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    obs_6 = create_base_obs(goal_body=(5.0, 0.0, 0.0), clearances=clearances_6)
    act_6 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_act_6, diag_6 = shield.filter_action(obs_6, act_6)
    pass_6 = diag_6.layer2_active and (s_act_6[0] < 0.60) and (diag_6.scale_factor < 0.60)
    results.append({
        "test_id": "GEN-06",
        "name": "Front obstacle dynamic braking (sector 0, d=1.0m, raw cmd=+3.0m/s)",
        "passed": bool(pass_6),
        "details": {
            "layer2_active": diag_6.layer2_active,
            "scale_factor": diag_6.scale_factor,
            "shielded_cmd_vx": diag_6.shielded_cmd.vx,
            "interventions": diag_6.interventions
        }
    })

    # -------------------------------------------------------------------------
    # GEN-07: Side obstacle while translating laterally (sector 2 Left, d = 0.8m, a1 = +1.0)
    # -------------------------------------------------------------------------
    clearances_7 = [10.0, 10.0, 0.80, 10.0, 10.0, 10.0, 10.0, 10.0]
    obs_7 = create_base_obs(goal_body=(5.0, 0.0, 0.0), clearances=clearances_7)
    act_7 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    s_act_7, diag_7 = shield.filter_action(obs_7, act_7)
    pass_7 = diag_7.layer2_active and (s_act_7[1] < 0.50)
    results.append({
        "test_id": "GEN-07",
        "name": "Side obstacle lateral dynamic braking (sector 2, d=0.8m, raw vy=+3.0m/s)",
        "passed": bool(pass_7),
        "details": {
            "layer2_active": diag_7.layer2_active,
            "scale_factor": diag_7.scale_factor,
            "shielded_cmd_vy": diag_7.shielded_cmd.vy,
            "interventions": diag_7.interventions
        }
    })

    # -------------------------------------------------------------------------
    # GEN-08: Multi-sector obstacle configuration (Front d=1.2m, Left d=0.9m)
    # -------------------------------------------------------------------------
    clearances_8 = [1.20, 10.0, 0.90, 10.0, 2.00, 10.0, 10.0, 10.0]
    obs_8 = create_base_obs(goal_body=(3.0, 3.0, 0.0), clearances=clearances_8)
    act_8 = np.array([0.8, 0.8, 0.0, 0.5], dtype=np.float32)
    s_act_8, diag_8 = shield.filter_action(obs_8, act_8)
    pass_8 = diag_8.layer2_active and (diag_8.scale_factor < 0.70)
    results.append({
        "test_id": "GEN-08",
        "name": "Multi-sector simultaneous constraints (Front d=1.2m, Left d=0.9m)",
        "passed": bool(pass_8),
        "details": {
            "layer2_active": diag_8.layer2_active,
            "scale_factor": diag_8.scale_factor,
            "shielded_cmd": {
                "vx": diag_8.shielded_cmd.vx,
                "vy": diag_8.shielded_cmd.vy,
                "yaw_rate": diag_8.shielded_cmd.yaw_rate
            }
        }
    })

    # -------------------------------------------------------------------------
    # GEN-09: Low commanded velocity approach (v = 0.2 m/s, d = 1.00m)
    # -------------------------------------------------------------------------
    clearances_9 = [1.00, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    obs_9 = create_base_obs(goal_body=(5.0, 0.0, 0.0), clearances=clearances_9)
    # 0.2 m/s commanded forward
    act_9 = np.array([0.20 / 3.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_act_9, diag_9 = shield.filter_action(obs_9, act_9)
    # At 0.2m/s, required distance is 0.85 + 0.25(0.2) + 0.04/6 = 0.907m < 1.00m -> safe without scaling
    pass_9 = (diag_9.scale_factor >= 0.999)
    results.append({
        "test_id": "GEN-09",
        "name": "Low-velocity approach (cmd_vx=0.20m/s, d=1.00m -> no intervention needed)",
        "passed": bool(pass_9),
        "details": {
            "scale_factor": diag_9.scale_factor,
            "is_intervened": diag_9.is_intervened,
            "shielded_cmd_vx": diag_9.shielded_cmd.vx
        }
    })

    # -------------------------------------------------------------------------
    # GEN-10: High velocity approach (v = 3.0 m/s, d = 1.2m)
    # -------------------------------------------------------------------------
    clearances_10 = [1.20, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
    obs_10 = create_base_obs(goal_body=(5.0, 0.0, 0.0), clearances=clearances_10)
    act_10 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_act_10, diag_10 = shield.filter_action(obs_10, act_10)
    pass_10 = diag_10.layer2_active and (diag_10.shielded_cmd.vx < 1.80)
    results.append({
        "test_id": "GEN-10",
        "name": "High-velocity approach aggressive braking (cmd_vx=3.0m/s, d=1.2m)",
        "passed": bool(pass_10),
        "details": {
            "scale_factor": diag_10.scale_factor,
            "shielded_cmd_vx": diag_10.shielded_cmd.vx,
            "interventions": diag_10.interventions
        }
    })

    # -------------------------------------------------------------------------
    # GEN-11: Heading error boundary check (88° vs 92° with rear obstacle at d=2.0m)
    # -------------------------------------------------------------------------
    # 88 deg: x_g = cos(88°) = 0.0349, y_g = sin(88°) = 0.9994
    obs_11a = create_base_obs(goal_body=(0.0349, 0.9994, 0.0), clearances=[10.0, 10.0, 10.0, 10.0, 2.0, 10.0, 10.0, 10.0])
    act_11 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_act_11a, diag_11a = shield.filter_action(obs_11a, act_11)

    # 92 deg: x_g = cos(92°) = -0.0349, y_g = sin(92°) = 0.9994
    obs_11b = create_base_obs(goal_body=(-0.0349, 0.9994, 0.0), clearances=[10.0, 10.0, 10.0, 10.0, 2.0, 10.0, 10.0, 10.0])
    s_act_11b, diag_11b = shield.filter_action(obs_11b, act_11)

    pass_11 = (not diag_11a.layer1_active) and diag_11b.layer1_active and (s_act_11b[0] >= 0.0)
    results.append({
        "test_id": "GEN-11",
        "name": "Heading error boundary discontinuity test (|d_psi|=88° vs 92°)",
        "passed": bool(pass_11),
        "details": {
            "at_88_deg_layer1": diag_11a.layer1_active,
            "at_88_deg_cmd_vx": diag_11a.shielded_cmd.vx,
            "at_92_deg_layer1": diag_11b.layer1_active,
            "at_92_deg_cmd_vx": diag_11b.shielded_cmd.vx
        }
    })

    # -------------------------------------------------------------------------
    # GEN-12: Heading error near 180° (pure rear goal, d_psi = 179°)
    # -------------------------------------------------------------------------
    # 179 deg: x_g = -0.9998, y_g = 0.0175
    obs_12 = create_base_obs(goal_body=(-0.9998, 0.0175, 0.0), clearances=[10.0, 10.0, 10.0, 10.0, 1.5, 10.0, 10.0, 10.0])
    act_12 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_act_12, diag_12 = shield.filter_action(obs_12, act_12)
    pass_12 = diag_12.layer1_active and (s_act_12[0] >= 0.0)
    results.append({
        "test_id": "GEN-12",
        "name": "Pure rear goal alignment (|d_psi|=179°, rear d=1.5m)",
        "passed": bool(pass_12),
        "details": {
            "layer1_active": diag_12.layer1_active,
            "heading_error_deg": diag_12.heading_error_deg,
            "shielded_cmd_vx": diag_12.shielded_cmd.vx
        }
    })

    # -------------------------------------------------------------------------
    # GEN-13: Sensor stale / corrupt fail-safe
    # -------------------------------------------------------------------------
    obs_13 = create_base_obs(goal_body=(-5.0, 0.0, 0.0), clearances=[10.0]*8)
    meta_stale = {"is_sensor_stale": True}
    act_13 = np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    s_act_13, diag_13 = shield.filter_action(obs_13, act_13, perception_metadata=meta_stale)
    pass_13 = diag_13.layer1_active and (s_act_13[0] >= 0.0)
    results.append({
        "test_id": "GEN-13",
        "name": "Perception staleness conservative fallback (is_sensor_stale=True)",
        "passed": bool(pass_13),
        "details": {
            "layer1_active": diag_13.layer1_active,
            "is_intervened": diag_13.is_intervened,
            "interventions": diag_13.interventions
        }
    })

    total_tests = len(results)
    passed_tests = sum(1 for r in results if r["passed"])
    overall_pass = (passed_tests == total_tests)

    output_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": total_tests - passed_tests,
        "overall_pass": overall_pass,
        "results": results
    }

    return output_data


if __name__ == "__main__":
    out = run_all_diagnostics()
    out_path = "experiments/milestone5/m5_stage2b_safety_shield_generalization.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print("=" * 80)
    print(f"SAFETY SHIELD GENERALIZATION DIAGNOSTICS: {out['passed_tests']}/{out['total_tests']} PASSED")
    print("=" * 80)
    for r in out["results"]:
        status = "[PASS]" if r["passed"] else "[FAIL]"
        print(f"  {status} {r['test_id']}: {r['name']}")
    print(f"\nArtifact exported to: {out_path}")
    if not out["overall_pass"]:
        sys.exit(1)
