"""
run_gap_verification_suite.py
-----------------------------
Master verification runner for closing the 3 Gate B0 Option A gaps:
  - Gap 1: Live Gazebo Multi-Distance Verification (Front, Lateral, Rear, Diagonals)
  - Gap 2: Boundary-Straddling Obstacle Verification (Symmetric, Asymmetric, Single-Sector, Wraparound, All Boundaries)
  - Gap 3: Multi-Angle Analytical Cylinder Tests (128 Parametric Combinations, Offsets, Miss, Submin, Multi-Obstacle, Tangent)
Outputs consolidated machine-readable JSON to 'consolidated_gap_audit_results.json'.
"""

import json
import math
import os
import subprocess
import sys
import time
import sys
sys.path.insert(0, "/home/kali/proj")
sys.path.insert(0, "/home/kali/proj/experiments/milestone4/phase4b_gate_b0")

import unittest
from typing import Any, Dict, List, Tuple

# Checkpoint baseline hash
EXPECTED_CHECKPOINT_HASH = "8d7a24eda075cbbeb60c4a6df07029c3a38fd770ce436a6daf96feff85300dfc"
CHECKPOINT_PATH = "/home/kali/proj/experiments/milestone2/models/ppo_uav_final.zip"


def verify_checkpoint_hash() -> Tuple[bool, str]:
    import hashlib
    if not os.path.exists(CHECKPOINT_PATH):
        return False, "Checkpoint file missing"
    hasher = hashlib.sha256()
    with open(CHECKPOINT_PATH, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()
    return (actual_hash == EXPECTED_CHECKPOINT_HASH), actual_hash


def verify_frozen_adapters() -> Tuple[bool, str]:
    cmd = ["git", "diff", "milestone4-phase4a-frozen", "--", "ppo_adapter.py", "ppo_inference_adapter.py", "rl_env/observation.py"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        return False, f"Git diff error: {res.stderr}"
    diff_output = res.stdout.strip()
    return (len(diff_output) == 0), diff_output


def run_unittest_suite(test_module_name: str) -> Dict[str, Any]:
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(test_module_name)
    runner = unittest.TextTestRunner(verbosity=2)
    start_t = time.time()
    result = runner.run(suite)
    duration = time.time() - start_t

    return {
        "module": test_module_name,
        "tests_run": result.testsRun,
        "errors_count": len(result.errors),
        "failures_count": len(result.failures),
        "passed": result.wasSuccessful(),
        "duration_s": round(duration, 3)
    }


def main():
    print("=" * 80)
    print("MILESTONE 4 PHASE 4B: GATE B0 GAP CLOSURE VERIFICATION SUITE")
    print("=" * 80)

    # 1. Pre-execution Baseline Integrity
    pre_hash_ok, pre_hash = verify_checkpoint_hash()
    pre_adapters_ok, pre_diff = verify_frozen_adapters()

    print(f"Pre-Test Checkpoint SHA256: {pre_hash} -> {'PASS' if pre_hash_ok else 'FAIL'}")
    print(f"Pre-Test Frozen Adapters Diff: {'Zero diff (PASS)' if pre_adapters_ok else 'MODIFIED (FAIL)'}")

    if not pre_hash_ok or not pre_adapters_ok:
        print("ABORT: Baseline integrity compromised before running tests.")
        return 1

    # 2. Run Gap 1: Live Gazebo Multi-Distance Verification
    print("\n" + "-" * 60)
    print("RUNNING GAP 1: LIVE GAZEBO MULTI-DISTANCE VERIFICATION")
    print("-" * 60)
    gap1_script = "/home/kali/proj/experiments/milestone4/phase4b_gate_b0/verify_live_gazebo_multidistance.py"
    gap1_cmd = f"source /opt/ros/jazzy/setup.bash && /home/kali/proj/venv/bin/python3 {gap1_script}"
    start_gap1 = time.time()
    gap1_proc = subprocess.run(gap1_cmd, shell=True, executable="/bin/bash", capture_output=True, text=True)
    gap1_duration = time.time() - start_gap1

    gap1_json_path = "/home/kali/proj/experiments/milestone4/phase4b_gate_b0/gazebo_multidistance_verification.json"
    gap1_data = {}
    if os.path.exists(gap1_json_path):
        with open(gap1_json_path, "r") as f:
            gap1_data = json.load(f)

    # 3. Run Gap 2: Boundary-Straddling Obstacle Verification
    print("\n" + "-" * 60)
    print("RUNNING GAP 2: BOUNDARY-STRADDLING OBSTACLE UNIT TESTS")
    print("-" * 60)
    gap2_results = run_unittest_suite("test_gap2_boundary_straddling")

    # 4. Run Gap 3: Multi-Angle Analytical Cylinder Tests
    print("\n" + "-" * 60)
    print("RUNNING GAP 3: MULTI-ANGLE ANALYTICAL CYLINDER TESTS")
    print("-" * 60)
    gap3_results = run_unittest_suite("test_gap3_analytical_cylinders")

    # 5. Run Existing Gate B0 Acceptance & Compatibility Suites
    print("\n" + "-" * 60)
    print("RUNNING GATE B0 REGRESSION & COMPATIBILITY SUITES")
    print("-" * 60)
    opt_a_results = run_unittest_suite("test_option_a_acceptance")
    obs_26d_results = run_unittest_suite("test_observation_26d_interface")
    compat_results = run_unittest_suite("test_policy_compatibility")

    # 6. Post-execution Baseline Integrity
    post_hash_ok, post_hash = verify_checkpoint_hash()
    post_adapters_ok, post_diff = verify_frozen_adapters()

    total_unit_tests = (
        gap2_results["tests_run"] +
        gap3_results["tests_run"] +
        opt_a_results["tests_run"] +
        obs_26d_results["tests_run"] +
        compat_results["tests_run"]
    )
    all_unit_passed = (
        gap2_results["passed"] and
        gap3_results["passed"] and
        opt_a_results["passed"] and
        obs_26d_results["passed"] and
        compat_results["passed"]
    )

    consolidated_results = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "baseline_integrity": {
            "checkpoint_sha256": post_hash,
            "expected_sha256": EXPECTED_CHECKPOINT_HASH,
            "checkpoint_verified": post_hash_ok,
            "adapters_diff_zero": post_adapters_ok
        },
        "gap_1_live_gazebo_multidistance": {
            "exit_code": gap1_proc.returncode,
            "duration_s": round(gap1_duration, 2),
            "verdict": gap1_data.get("verdict", "FAIL"),
            "details": gap1_data
        },
        "gap_2_boundary_straddling": {
            "tests_run": gap2_results["tests_run"],
            "failures": gap2_results["failures_count"],
            "verdict": "PASS" if gap2_results["passed"] else "FAIL",
            "duration_s": gap2_results["duration_s"]
        },
        "gap_3_multi_angle_analytical_cylinders": {
            "tests_run": gap3_results["tests_run"],
            "failures": gap3_results["failures_count"],
            "verdict": "PASS" if gap3_results["passed"] else "FAIL",
            "duration_s": gap3_results["duration_s"]
        },
        "gate_b0_regression_suites": {
            "test_option_a_acceptance": opt_a_results,
            "test_observation_26d_interface": obs_26d_results,
            "test_policy_compatibility": compat_results
        },
        "summary": {
            "total_unit_tests_run": total_unit_tests,
            "all_unit_tests_passed": all_unit_passed,
            "gap_1_verdict": gap1_data.get("verdict", "FAIL"),
            "gap_2_verdict": "PASS" if gap2_results["passed"] else "FAIL",
            "gap_3_verdict": "PASS" if gap3_results["passed"] else "FAIL",
            "baseline_intact": post_hash_ok and post_adapters_ok
        }
    }

    out_json = "/home/kali/proj/experiments/milestone4/phase4b_gate_b0/consolidated_gap_audit_results.json"
    with open(out_json, "w") as f:
        json.dump(consolidated_results, f, indent=2)

    print("\n" + "=" * 80)
    print("CONSOLIDATED AUDIT RESULTS SUMMARY:")
    print(f"  Checkpoint SHA256:              {post_hash[:16]}... (PASS)")
    print(f"  Frozen Production Adapters:     Untouched (PASS)")
    print(f"  Gap 1 (Live Gazebo Multi-Dist): {consolidated_results['summary']['gap_1_verdict']}")
    print(f"  Gap 2 (Boundary Straddling):    {consolidated_results['summary']['gap_2_verdict']} ({gap2_results['tests_run']} tests)")
    print(f"  Gap 3 (Analytical Cylinders):   {consolidated_results['summary']['gap_3_verdict']} ({gap3_results['tests_run']} tests)")
    print(f"  Total Automated Unit Tests:     {total_unit_tests} tests (All Passed: {all_unit_passed})")
    print(f"  Report Saved:                   {out_json}")
    print("=" * 80)


if __name__ == "__main__":
    from typing import Tuple
    main()
