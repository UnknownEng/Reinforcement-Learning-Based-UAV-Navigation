"""
eval_world_policy.py
--------------------
Fixed-seed evaluation on UAVWorldNav-v0 scenarios, for learned policies and the M9
baselines, with or without the safety shield.

Scenarios (seen in training: forest open/medium/dense; unseen: very_dense, urban, warehouse):
  forest_open, forest_medium, forest_dense, forest_very_dense, urban, warehouse, poles,
  forest_movers (1-6 moving obstacles), forest_dr (domain randomization), forest_movers_dr

Usage (WSL):
  python training/eval_world_policy.py --policy experiments/world/forest/model.zip --scenarios all --episodes 100
  python training/eval_world_policy.py --baseline apf --scenarios forest_medium --episodes 50
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

SCENARIOS = {
    "forest_open": dict(world_kinds=("forest",), forest_presets=("open",)),
    "forest_medium": dict(world_kinds=("forest",), forest_presets=("medium",)),
    "forest_dense": dict(world_kinds=("forest",), forest_presets=("dense",)),
    "forest_very_dense": dict(world_kinds=("forest",), forest_presets=("very_dense",)),
    "urban": dict(world_kinds=("urban",)),
    "warehouse": dict(world_kinds=("warehouse",)),
    "poles": dict(world_kinds=("poles",)),
    "forest_movers": dict(world_kinds=("forest",), forest_presets=("open", "medium", "dense"), n_movers_range=(1, 6)),
    "forest_dr": dict(world_kinds=("forest",), forest_presets=("open", "medium", "dense"), domain_randomization=True),
    "forest_movers_dr": dict(world_kinds=("forest",), forest_presets=("open", "medium", "dense"),
                             n_movers_range=(1, 6), domain_randomization=True),
}


def run_scenario(job):
    import torch
    torch.set_num_threads(1)
    from rl_env.safety_shield import DeterministicSafetyShield
    from rl_env.uav_world_env import UAVWorldEnv, UAVWorldEnvConfig
    from training.train_world_policy import shield_config

    scenario, policy_path, baseline, shield, cbf, seeds, *rest = job
    cbf_params = rest[0] if rest else {}
    env = UAVWorldEnv(UAVWorldEnvConfig())
    env.set_curriculum(**{k: v for k, v in SCENARIOS[scenario].items()})
    rule_shield = DeterministicSafetyShield(shield_config()) if shield else None
    cbf_filter = None
    if cbf:
        from baselines.cbf import CBFSafetyFilter
        cbf_filter = CBFSafetyFilter(moving_obstacles=(cbf == "moving"), **cbf_params)
    if policy_path:
        from stable_baselines3 import PPO
        model = PPO.load(policy_path, device="cpu")
        controller = None
    elif baseline == "goal":
        class GoalController:   # straight-at-the-goal; only meaningful with --cbf / --shield
            def act(self, obs, env=None):
                g = obs[0:3]
                d = max(float(np.hypot(g[0], g[1])), 1e-6)
                sp = min(3.0, d)
                return np.clip([g[0] / d * sp / 3.0, g[1] / d * sp / 3.0, g[2] / 1.5,
                                np.arctan2(g[1], g[0])], -1.0, 1.0).astype(np.float32)
        controller = GoalController()
    else:
        from baselines.runner import make_controller
        controller = make_controller(baseline, env)
    episodes = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        if controller is not None and hasattr(controller, "reset"):
            controller.reset(seed)
        steps, interventions, t_compute = 0, 0, 0.0
        while True:
            t0 = time.perf_counter()
            if controller is None:
                act = model.predict(obs, deterministic=True)[0]
            else:
                act = controller.act(obs, env)
            if rule_shield is not None:
                act, diag = rule_shield.filter_action(obs, act, {"position_world": env.last_pos,
                                                               "is_sensor_stale": False})
                interventions += int(diag.is_intervened)
            if cbf_filter is not None:
                act, cinfo = cbf_filter.filter_action(obs, act, env=env)
                interventions += int(cinfo.get("intervened", False))
            t_compute += time.perf_counter() - t0
            obs, r, term, trunc, info = env.step(act)
            steps += 1
            if term or trunc:
                break
        episodes.append({"seed": seed, "outcome": info["termination_reason"], "steps": steps,
                         "path_length": round(info["path_length"], 2), "initial_dist": round(info["initial_dist"], 2),
                         "min_hull_clearance": round(float(info["min_hull_clearance_m"]), 3),
                         "intervention_rate": round(interventions / steps, 3),
                         "compute_ms": round(1e3 * t_compute / steps, 3), "kind": info["world_kind"],
                         "preset": info["forest_preset"], "n_movers": info["n_movers"],
                         "collision_with": info.get("collision_with")})
    return scenario, episodes


def summarize(eps):
    n = len(eps)
    rate = lambda k: round(100.0 * sum(e["outcome"] == k for e in eps) / n, 1)
    succ = [e for e in eps if e["outcome"] == "goal_reached"]
    return {"episodes": n, "success_pct": rate("goal_reached"), "collision_pct": rate("obstacle_collision"),
            "timeout_pct": rate("episode_timeout"), "out_of_bounds_pct": rate("out_of_bounds"),
            "ground_crash_pct": rate("ground_crash"),
            "path_efficiency_median": round(float(np.median([e["path_length"] / max(e["initial_dist"], 1e-6)
                                                               for e in succ])), 3) if succ else None,
            "time_to_goal_median_s": round(float(np.median([e["steps"] * 0.1 for e in succ])), 1) if succ else None,
            "min_hull_clearance_p5": round(float(np.percentile([e["min_hull_clearance"] for e in eps], 5)), 3),
            "intervention_rate_mean": round(float(np.mean([e["intervention_rate"] for e in eps])), 3),
            "collisions_by_type": {t: sum(e.get("collision_with") == t for e in eps)
                                   for t in sorted({e.get("collision_with") for e in eps if e.get("collision_with")})},
            "compute_ms_mean": round(float(np.mean([e["compute_ms"] for e in eps])), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--scenarios", default="all")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=500_000)
    ap.add_argument("--shield", action="store_true", help="apply the rule-based safety shield")
    ap.add_argument("--cbf", action="store_true", help="apply the CBF safety filter (baselines.cbf)")
    ap.add_argument("--cbf-moving", action="store_true",
                    help="CBF filter with the moving-obstacle (range-rate) term, M6; implies --cbf")
    ap.add_argument("--cbf-param", action="append", default=[], metavar="NAME=VALUE",
                    help="override a CBFConfig field, e.g. mover_deadband=0.6 (sensitivity studies)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert bool(args.policy) != bool(args.baseline), "give exactly one of --policy / --baseline"
    names = list(SCENARIOS) if args.scenarios == "all" else args.scenarios.split(",")
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    # split each scenario's seeds into chunks so all workers stay busy
    chunks = max(1, args.workers // max(1, len(names)) * 2)
    cbf = "moving" if args.cbf_moving else bool(args.cbf)
    cbf_params = {k: float(v) for k, v in (item.split("=", 1) for item in args.cbf_param)}
    jobs = [(s, args.policy, args.baseline, args.shield, cbf, seeds[i::chunks], cbf_params)
            for s in names for i in range(chunks)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        results = pool.map(run_scenario, jobs)
    merged = {}
    for name, eps in results:
        merged.setdefault(name, []).extend(eps)
    report = {"policy": args.policy, "baseline": args.baseline, "shield": args.shield, "cbf": cbf,
              "cbf_params": cbf_params,
              "seeds": [seeds[0], seeds[-1]], "wall_s": round(time.time() - t0, 1),
              "summary": {k: summarize(v) for k, v in merged.items()}, "episodes": merged}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)
    for k, s in report["summary"].items():
        print(f"{k:18s} success {s['success_pct']:5.1f}%  collision {s['collision_pct']:5.1f}%  timeout "
              f"{s['timeout_pct']:5.1f}%  eff {s['path_efficiency_median']}  p5 clearance {s['min_hull_clearance_p5']}"
              f"  interventions {s['intervention_rate_mean']}  {s['compute_ms_mean']} ms", flush=True)
    print(f"saved {args.out} ({report['wall_s']} s)")


if __name__ == "__main__":
    main()
