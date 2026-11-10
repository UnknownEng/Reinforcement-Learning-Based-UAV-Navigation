"""
runner.py
---------
Factory and episode evaluation for the Milestone-9 classical baselines on UAVWorldNav-v0.

  make_controller(name, env, **options)   "apf", "dwa", "rrt_star", "informed_rrt_star"
  evaluate(controller_name, env_config, seeds, wrap_cbf=False, ...) -> list of per-episode dicts
  summarize(results) -> aggregate dict

Importing this module creates no environment and runs nothing. Command line (small runs only):
  python -m baselines.runner --controller dwa --seeds 0:5 [--cbf] [--world-kind forest --preset dense]
"""

import argparse
import json
import math
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import numpy as np

from baselines.apf import APFController
from baselines.cbf import CBFConfig, CBFSafetyFilter
from baselines.dwa import DWAController
from baselines.rrt_star import RRTStarController

CONTROLLER_NAMES = ("apf", "dwa", "rrt_star", "informed_rrt_star")
INFORMATION = {"apf": "onboard LiDAR + observation", "dwa": "onboard LiDAR + observation",
               "rrt_star": "global map (env.world) + onboard LiDAR for replanning",
               "informed_rrt_star": "global map (env.world) + onboard LiDAR for replanning"}


def make_controller(name: str, env=None, **options):
    """Controller with `act(obs, env) -> action (4,)` and `reset(seed=None)`; options override its config."""
    key = name.lower().replace("-", "_").replace("*", "_star").strip("_")
    if key == "apf":
        return APFController(env, **options)
    if key == "dwa":
        return DWAController(env, **options)
    if key in ("rrt_star", "rrtstar"):
        return RRTStarController(env, informed=False, **options)
    if key in ("informed_rrt_star", "informed_rrtstar"):
        return RRTStarController(env, informed=True, **options)
    raise ValueError(f"unknown controller {name!r}; expected one of {CONTROLLER_NAMES}")


def evaluate(controller_name: str, env_config=None, seeds: Iterable[int] = (0,), wrap_cbf: bool = False, *,
             reset_options: Union[None, Dict[str, Any], Callable[[int], Dict[str, Any]]] = None,
             controller_kwargs: Optional[Dict[str, Any]] = None, cbf_config: Optional[CBFConfig] = None,
             max_steps: Optional[int] = None, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Run one episode per seed and return per-episode records:
      outcome (info["termination_reason"]), success, collision, steps, path_length, initial_dist,
      final_dist, path_efficiency (initial_dist / path_length on success, else 0),
      min_hull_clearance_m, mean/max compute time per decision in ms (controller + optional CBF;
      for RRT* the initial plan is part of the first decision, also reported as plan_time_ms), ...
    `reset_options` is passed to env.reset (a dict, or a function of the seed).
    """
    from rl_env.uav_world_env import UAVWorldEnv, UAVWorldEnvConfig

    env = UAVWorldEnv(env_config if env_config is not None else UAVWorldEnvConfig())
    ctrl = make_controller(controller_name, env, **(controller_kwargs or {}))
    cbf = CBFSafetyFilter(cbf_config, env=env) if wrap_cbf else None
    results = []
    for seed in seeds:
        seed = int(seed)
        opts = reset_options(seed) if callable(reset_options) else (dict(reset_options) if reset_options else None)
        obs, info = env.reset(seed=seed, options=opts)
        ctrl.reset(seed=seed)
        times, n_interv, n_infeas = [], 0, 0
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            t0 = time.perf_counter()
            action = ctrl.act(obs, env)
            if cbf is not None:
                action, cinfo = cbf.filter_action(obs, action, env=env)
                n_interv += int(cinfo["intervened"])
                n_infeas += int(cinfo["infeasible"])
            times.append(time.perf_counter() - t0)
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        success = info["termination_reason"] == "goal_reached"
        plen, d0 = float(info["path_length"]), float(info["initial_dist"])
        t_ms = np.asarray(times) * 1e3
        rec = {
            "controller": getattr(ctrl, "name", controller_name), "seed": seed, "wrap_cbf": bool(wrap_cbf),
            "uses_map": bool(getattr(ctrl, "uses_map", False)),
            "outcome": info["termination_reason"], "success": success,
            "collision": info["termination_reason"] == "obstacle_collision",
            "steps": steps, "path_length": plen, "initial_dist": d0,
            "final_dist": float(np.linalg.norm(np.asarray(info["pos_world"]) - env.goal_world)),
            "path_efficiency": d0 / max(plen, d0, 1e-9) if success else 0.0,
            "min_hull_clearance_m": float(info["min_hull_clearance_m"]),
            "mean_compute_ms": float(t_ms.mean()) if len(t_ms) else 0.0,
            "max_compute_ms": float(t_ms.max()) if len(t_ms) else 0.0,
            "world_kind": info.get("world_kind"), "forest_preset": info.get("forest_preset"),
            "n_movers": info.get("n_movers", 0),
        }
        if hasattr(ctrl, "plan_time_s"):
            rec.update(plan_time_ms=1e3 * ctrl.plan_time_s, n_plans=ctrl.n_plans, n_replans=ctrl.n_replans,
                       n_plan_failures=ctrl.n_plan_failures)
        if hasattr(ctrl, "n_escapes"):
            rec["n_escapes"] = ctrl.n_escapes
        if cbf is not None:
            rec.update(cbf_interventions=n_interv, cbf_infeasible=n_infeas)
        results.append(rec)
        if verbose:
            print(f"[{rec['controller']}{'+cbf' if wrap_cbf else ''}] seed={seed} {rec['outcome']} "
                  f"steps={steps} clr={rec['min_hull_clearance_m']:.2f} {rec['mean_compute_ms']:.1f} ms/dec")
    return results


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Success / collision / timeout rates and mean metrics over episodes."""
    if not results:
        return {}
    n = len(results)
    outcomes: Dict[str, int] = {}
    for r in results:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    succ = [r for r in results if r["success"]]
    mean = lambda key, rs: float(np.mean([r[key] for r in rs])) if rs else math.nan
    return {"episodes": n, "success_rate": len(succ) / n,
            "collision_rate": outcomes.get("obstacle_collision", 0) / n,
            "timeout_rate": outcomes.get("episode_timeout", 0) / n, "outcomes": outcomes,
            "mean_steps_success": mean("steps", succ), "mean_path_efficiency": mean("path_efficiency", results),
            "mean_min_hull_clearance_m": mean("min_hull_clearance_m", results),
            "mean_compute_ms": mean("mean_compute_ms", results),
            "max_compute_ms": max(r["max_compute_ms"] for r in results)}


def _parse_seeds(text: str) -> List[int]:
    if ":" in text:
        a, b = text.split(":")
        return list(range(int(a), int(b)))
    return [int(s) for s in text.split(",") if s]


def main(argv=None) -> None:
    from rl_env.uav_world_env import UAVWorldEnvConfig

    ap = argparse.ArgumentParser(description="Evaluate a classical baseline on UAVWorldNav-v0")
    ap.add_argument("--controller", default="dwa", choices=CONTROLLER_NAMES)
    ap.add_argument("--seeds", default="0:5", help="'a:b' (range) or 's1,s2,...'")
    ap.add_argument("--cbf", action="store_true", help="wrap the controller with the CBF safety filter")
    ap.add_argument("--world-kind", default=None)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--movers", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the per-episode records as JSON")
    args = ap.parse_args(argv)
    cfg = UAVWorldEnvConfig()
    if args.world_kind:
        cfg.world_kinds = (args.world_kind,)
    if args.preset:
        cfg.forest_presets = (args.preset,)
    if args.movers:
        cfg.n_movers_range = (args.movers, args.movers)
    res = evaluate(args.controller, cfg, _parse_seeds(args.seeds), args.cbf, verbose=True)
    print(json.dumps(summarize(res), indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2, default=float)


if __name__ == "__main__":
    main()
