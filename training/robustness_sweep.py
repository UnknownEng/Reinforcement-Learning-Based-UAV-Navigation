"""
robustness_sweep.py
-------------------
M7: one-factor-at-a-time robustness sweeps on medium forests. Each physical parameter is
pinned to a series of values (everything else nominal) and success / collision rates are
measured on the same seeds, for one or more policies (e.g. trained without vs with domain
randomization).

Usage (WSL):
  python training/robustness_sweep.py --policies nominal=experiments/world/movers/model.zip \
      dr=experiments/world/dr/model.zip --episodes 40 --out logs/m7_sweep.json
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

SWEEPS = {
    "latency_s": [0.02, 0.08, 0.14, 0.20],
    "wind_speed": [0.0, 2.0, 4.0, 6.0, 8.0],
    "mass_scale": [0.7, 0.85, 1.0, 1.15, 1.3],
    "inertia_scale": [0.6, 1.0, 1.5, 2.0],
    "lidar_dropout": [0.0, 0.1, 0.2, 0.3],
    "lidar_noise_std": [0.0, 0.05, 0.10, 0.20],
    "noise_pos_std": [0.0, 0.05, 0.10, 0.20],
}


def run(job):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from rl_env.safety_shield import DeterministicSafetyShield
    from rl_env.uav_world_env import UAVWorldEnv, UAVWorldEnvConfig
    from training.train_world_policy import shield_config

    label, path, factor, value, seeds, shield = job
    model = PPO.load(path, device="cpu")
    env = UAVWorldEnv(UAVWorldEnvConfig(world_kinds=("forest",), forest_presets=("medium",)))
    sh = DeterministicSafetyShield(shield_config()) if shield else None
    out = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed, options={"dynamics_overrides": {factor: value}})
        while True:
            act = model.predict(obs, deterministic=True)[0]
            if sh is not None:
                act, _ = sh.filter_action(obs, act, {"position_world": env.last_pos})
            obs, r, term, trunc, info = env.step(act)
            if term or trunc:
                break
        out.append(info["termination_reason"])
    return label, factor, value, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", required=True, help="label=path entries")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=700_000)
    ap.add_argument("--factors", default="all")
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    policies = dict(p.split("=", 1) for p in args.policies)
    factors = list(SWEEPS) if args.factors == "all" else args.factors.split(",")
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    jobs = [(lab, path, f, v, seeds, not args.no_shield) for lab, path in policies.items() for f in factors
            for v in SWEEPS[f]]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = pool.map(run, jobs)
    table = {}
    for lab, f, v, outs in res:
        n = len(outs)
        table.setdefault(f, {}).setdefault(str(v), {})[lab] = {
            "success_pct": round(100 * outs.count("goal_reached") / n, 1),
            "collision_pct": round(100 * outs.count("obstacle_collision") / n, 1),
            "timeout_pct": round(100 * outs.count("episode_timeout") / n, 1)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"policies": policies, "episodes": args.episodes, "shield": not args.no_shield,
               "wall_s": round(time.time() - t0), "table": table}, open(args.out, "w"), indent=1)
    for f, rows in table.items():
        print(f"== {f}")
        for v, cell in rows.items():
            print(f"   {v:>6}: " + "   ".join(f"{lab} {c['success_pct']:5.1f}% ok / {c['collision_pct']:4.1f}% crash"
                                            for lab, c in cell.items()))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
