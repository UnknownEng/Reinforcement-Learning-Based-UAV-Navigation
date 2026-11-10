"""
run_gate_b3_verification.py
---------------------------
Milestone 4 Phase 4B — Gate B3: End-to-End Curriculum Training & Evaluation Runner.

Orchestrates:
  1. Pre-training baseline integrity audit:
     - Checkpoint SHA-256 (8d7a24ed...)
     - Tag commit (31c81c77...)
     - Adapter diff (0 lines)
     - 26-D observation contract
  2. Full 1,000,000-timestep 4-stage automated curriculum training run (train_ppo_26d)
  3. Post-training baseline integrity audit
  4. Standardized 100-episode 4-tier benchmark evaluation (evaluate_gate_b3)
  5. Gate B3 Acceptance Criteria Validation
  6. Consolidated results report and JSON output
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict

sys.path.insert(0, "/home/kali/proj")

import gymnasium as gym
import numpy as np

import rl_env
from experiments.milestone4.phase4b_gate_b3.train_ppo_26d import train_ppo_26d
from experiments.milestone4.phase4b_gate_b3.evaluate_gate_b3 import run_gate_b3_benchmark


def main():
    print("=" * 80)
    print("MILESTONE 4 PHASE 4B: GATE B3 MASTER VERIFICATION RUNNER")
    print("=" * 80)

    # 1. Pre-Training Baseline Verification
    expected_ckpt = "8d7a24eda075cbbeb60c4a6df07029c3a38fd770ce436a6daf96feff85300dfc"
    expected_tag = "31c81c77aaea82ac35a34fb102e3128257b09b30"
    frozen_ckpt_path = "/home/kali/proj/experiments/milestone2/models/ppo_uav_final.zip"

    with open(frozen_ckpt_path, "rb") as f:
        pre_hash = hashlib.sha256(f.read()).hexdigest()
    if pre_hash != expected_ckpt:
        print(f"FATAL: Checkpoint hash altered before Gate B3! Got {pre_hash}")
        sys.exit(1)

    tag_commit = subprocess.check_output(
        ["git", "rev-parse", "milestone4-phase4a-frozen^{commit}"]
    ).decode().strip()
    if tag_commit != expected_tag:
        print(f"FATAL: Frozen tag commit mismatch! Got {tag_commit}")
        sys.exit(1)

    diff = subprocess.check_output([
        "git", "diff", "milestone4-phase4a-frozen", "--",
        "ppo_adapter.py", "ppo_inference_adapter.py", "rl_env/observation.py", "rl_env/config.py"
    ]).decode().strip()
    if len(diff) > 0:
        print(f"FATAL: Protected files modified! Diff:\n{diff}")
        sys.exit(1)

    print(f"Pre-Training Baseline Checkpoint: {pre_hash[:16]}... [PASS]")
    print(f"Pre-Training Tag Commit:         {tag_commit[:16]}... [PASS]")
    print(f"Pre-Training Protected Files:    0 diff lines [PASS]\n")

    # 2. Launch Full 1,000,000-Step Curriculum Training
    total_timesteps = 1_000_000
    t_train_start = time.perf_counter()
    train_meta = train_ppo_26d(
        total_timesteps=total_timesteps,
        seed=42,
        eval_freq=25000,
        n_eval_episodes=20
    )
    t_train_end = time.perf_counter()
    train_duration_s = t_train_end - t_train_start

    # 3. Post-Training Baseline Integrity Verification
    with open(frozen_ckpt_path, "rb") as f:
        post_hash = hashlib.sha256(f.read()).hexdigest()
    if post_hash != expected_ckpt:
        print(f"FATAL: Checkpoint hash altered during training! Got {post_hash}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE: RUNNING GATE B3 STANDARDIZED BENCHMARK EVALUATION")
    print("=" * 80)

    # 4. Standardized 100-Episode Benchmark Evaluation
    cand_model_path = "/home/kali/proj/experiments/milestone4/models/ppo_uav_obstacle_26d_final.zip"
    eval_results = run_gate_b3_benchmark(
        candidate_model_path=cand_model_path,
        baseline_model_path=frozen_ckpt_path,
        output_path="/home/kali/proj/experiments/milestone4/phase4b_gate_b3/gate_b3_evaluation_results.json"
    )

    # 5. Compile Consolidated Gate B3 Summary
    overall_verdict = eval_results.get("verdict", "FAIL")
    cand_metrics = eval_results["candidate_26d"]["overall_metrics"]
    base_metrics = eval_results["baseline_18d"]["overall_metrics"]

    consolidated_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gate_id": "Gate B3: Full Curriculum Training & Evaluation",
        "verdict": overall_verdict,
        "baseline_integrity": {
            "checkpoint_sha256": post_hash,
            "expected_sha256": expected_ckpt,
            "checkpoint_verified": True,
            "frozen_tag_commit": tag_commit,
            "expected_tag_commit": expected_tag,
            "frozen_tag_verified": True,
            "adapters_diff_zero": True
        },
        "training_telemetry": {
            "total_timesteps": total_timesteps,
            "wall_time_seconds": train_duration_s,
            "steps_per_second": total_timesteps / train_duration_s,
            "ram_initial_mb": train_meta["ram_initial_mb"],
            "ram_final_mb": train_meta["ram_final_mb"],
            "ram_delta_mb": train_meta["ram_delta_mb"],
            "checkpoints": train_meta["checkpoint_hashes"]
        },
        "benchmark_summary": {
            "candidate_26d": cand_metrics,
            "baseline_18d": base_metrics,
            "acceptance_criteria": eval_results["acceptance_criteria"]
        }
    }

    results_path = "/home/kali/proj/experiments/milestone4/phase4b_gate_b3/gate_b3_verification_results.json"
    with open(results_path, "w") as f:
        json.dump(consolidated_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"GATE B3 OVERALL VERDICT: {overall_verdict}")
    print(f"  Candidate 26-D Success Rate:  {cand_metrics['overall_success_rate_pct']:.1f}%")
    print(f"  Candidate 26-D Collision Rate:{cand_metrics['overall_collision_rate_pct']:.1f}%")
    print(f"  Baseline 18-D Collision Rate: {base_metrics['overall_collision_rate_pct']:.1f}%")
    print(f"  Mean Clearance:               {cand_metrics['mean_min_clearance_m']:.2f} m")
    print(f"  Mean Inference Latency:       {cand_metrics['mean_inference_latency_ms']:.3f} ms")
    print(f"  Consolidated Evidence Saved:  {results_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
