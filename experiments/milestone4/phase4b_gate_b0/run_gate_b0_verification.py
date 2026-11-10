"""
run_gate_b0_verification.py
---------------------------
Orchestrates and records the full Milestone 4 Phase 4B Gate B0 validation suite.
Outputs:
  - experiments/milestone4/phase4b_gate_b0/gate_b0_verification_results.json
"""

import hashlib
import json
import os
import subprocess
import sys
import time
import unittest
import numpy as np

# Import test cases
import test_observation_26d_interface
import test_policy_compatibility


def get_git_info():
    branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).decode().strip()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    status = subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
    return {
        "branch": branch,
        "commit": commit,
        "dirty": bool(status)
    }


def compute_sha256(filepath):
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def main():
    print("=" * 70)
    print("EXECUTING MILESTONE 4 PHASE 4B — GATE B0 INTERFACE VALIDATION")
    print("=" * 70)

    start_time = time.time()

    # 1. Integrity Check: Checkpoint SHA-256
    model_path = "/home/kali/proj/experiments/milestone2/models/ppo_uav_final.zip"
    expected_hash = "8d7a24eda075cbbeb60c4a6df07029c3a38fd770ce436a6daf96feff85300dfc"
    actual_hash = compute_sha256(model_path)
    hash_match = (actual_hash == expected_hash)
    print(f"Checkpoint SHA-256: {actual_hash} [Match: {hash_match}]")

    # 2. Frozen Baseline Tag Verification
    tag_commit = subprocess.check_output(["git", "rev-parse", "milestone4-phase4a-frozen^{commit}"]).decode().strip()
    expected_tag_commit = "31c81c7f99990b7936a113ecaa68c07c4bc4713c"
    tag_match = tag_commit.startswith(expected_tag_commit[:7])
    print(f"Freeze Tag milestone4-phase4a-frozen: {tag_commit[:7]} [Match: {tag_match}]")

    # 3. Production Adapter Diff Verification
    cmd = [
        "git", "diff", "milestone4-phase4a-frozen..HEAD", "--",
        "ros2_ws/src/uav_autonomy/uav_autonomy/ppo_inference_adapter.py"
    ]
    diff_output = subprocess.check_output(cmd).decode().strip()
    adapters_untouched = (len(diff_output) == 0)
    print(f"Production Adapter Zero-Diff: {adapters_untouched}")

    # 4. Run Test Suite 1: Interface Validation
    loader = unittest.TestLoader()
    suite1 = loader.loadTestsFromModule(test_observation_26d_interface)
    runner1 = unittest.TextTestRunner(verbosity=2)
    res1 = runner1.run(suite1)

    # 5. Run Test Suite 2: Policy Compatibility
    suite2 = loader.loadTestsFromModule(test_policy_compatibility)
    runner2 = unittest.TextTestRunner(verbosity=2)
    res2 = runner2.run(suite2)

    total_tests = res1.testsRun + res2.testsRun
    total_failures = len(res1.failures) + len(res2.failures)
    total_errors = len(res1.errors) + len(res2.errors)

    elapsed_s = time.time() - start_time
    git_info = get_git_info()

    verdict = "PASS" if (
        total_failures == 0 and total_errors == 0 and hash_match and tag_match and adapters_untouched
    ) else "FAIL"

    results_data = {
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate": "GATE_B0",
        "verdict": verdict,
        "git_state": git_info,
        "checkpoint_integrity": {
            "model_path": model_path,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "match": hash_match
        },
        "freeze_baseline_integrity": {
            "tag": "milestone4-phase4a-frozen",
            "tag_commit": tag_commit,
            "expected_commit": expected_tag_commit,
            "match": tag_match
        },
        "adapter_isolation": {
            "ppo_inference_adapter_diff_lines": len(diff_output.splitlines()),
            "adapters_untouched": adapters_untouched
        },
        "test_suites": {
            "test_observation_26d_interface": {
                "tests_run": res1.testsRun,
                "failures": len(res1.failures),
                "errors": len(res1.errors),
                "success": res1.wasSuccessful()
            },
            "test_policy_compatibility": {
                "tests_run": res2.testsRun,
                "failures": len(res2.failures),
                "errors": len(res2.errors),
                "success": res2.wasSuccessful()
            }
        },
        "summary": {
            "total_tests": total_tests,
            "total_failures": total_failures,
            "total_errors": total_errors,
            "elapsed_seconds": round(elapsed_s, 3)
        },
        "conservative_sensor_handling_policy": {
            "default_policy": "CONSERVATIVE_ZERO",
            "unobserved_clearance_m": 0.0,
            "corrupt_clearance_m": 0.0,
            "stale_clearance_m": 0.0,
            "measured_clear_m": 10.0,
            "submin_proximity_m": 0.10,
            "silently_interpret_missing_as_clear": False,
            "metadata_validity_distinction": True
        }
    }

    out_path = "/home/kali/proj/experiments/milestone4/phase4b_gate_b0/gate_b0_verification_results.json"
    with open(out_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print("\n" + "=" * 70)
    print(f"GATE B0 VALIDATION COMPLETE: VERDICT = {verdict}")
    print(f"Total Tests Executed: {total_tests} | Failures: {total_failures} | Errors: {total_errors}")
    print(f"Saved JSON results to: {out_path}")
    print("=" * 70)

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
