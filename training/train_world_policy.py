"""
train_world_policy.py
---------------------
PPO training on UAVWorldNav-v0 (3-D worlds, 7-channel LiDAR, PX4-calibrated dynamics).

Stages (each can start from the previous stage's checkpoint):
  forest   M8 base policy: procedural forests, density curriculum open -> medium -> dense
  movers   M6: forests with 1-6 moving obstacles (walkers on the ground, flyers in the air)
  dr       M7: forests + movers with full domain randomization (mass, inertia, motor and
           controller time constants, drag, 20-140 ms latency, sensor noise/dropout, wind)

The policy is trained behind the deterministic safety shield (as in M5) unless --no-shield.

Usage (WSL):
  python training/train_world_policy.py --stage forest --steps 1500000 --n-envs 4
  python training/train_world_policy.py --stage movers --init experiments/world/forest/model.zip --steps 600000
"""

import argparse
import json
import subprocess
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from rl_env.safety_shield import DeterministicSafetyShield, SafetyShieldConfig, SafetyShieldWrapper
from rl_env.uav_world_env import UAVWorldEnv, UAVWorldEnvConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGES = {
    # name: (initial curriculum, schedule [(fraction, kwargs for set_curriculum)])
    "forest": (dict(world_kinds=("forest",), forest_presets=("open",), n_movers_range=(0, 0),
                    domain_randomization=False),
               [(0.25, dict(forest_presets=("open", "medium"))),
                (0.55, dict(forest_presets=("open", "medium", "dense")))]),
    "movers": (dict(world_kinds=("forest",), forest_presets=("open", "medium", "dense"), n_movers_range=(1, 3),
                    domain_randomization=False),
               [(0.4, dict(n_movers_range=(1, 6)))]),
    "dr": (dict(world_kinds=("forest",), forest_presets=("open", "medium", "dense"), n_movers_range=(0, 4),
                domain_randomization=True), []),
}


def shield_config(half: float = 20.0) -> SafetyShieldConfig:
    g = half - 0.5
    return SafetyShieldConfig(geofence_x=(-g, g), geofence_y=(-g, g), geofence_z=(0.25, 7.5))


class LagrangianCost(gym.Wrapper):
    """PPO-Lagrangian (Ray et al., 2019): a per-step cost c = 1 while the hull is within `margin` of an
    obstacle (collisions included); the agent is trained on r - lam * c, with lam set by dual ascent
    (LagrangeMultiplier callback) so that the mean episode cost approaches the cost limit."""

    def __init__(self, env, margin: float):
        super().__init__(env)
        self.margin, self.lam = margin, 0.0

    def set_lambda(self, lam: float):
        self.lam = float(lam)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        c = float(info.get("hull_clearance_m", np.inf) < self.margin)
        info["cost"] = c
        return obs, r - self.lam * c, term, trunc, info


def make_env_fn(init_curriculum: dict, shield: bool, monitor_csv: str, reward_overrides: dict = None,
                cost_margin: float = None):
    def _f():
        env = UAVWorldEnv(UAVWorldEnvConfig(**(reward_overrides or {})))
        env.set_curriculum(**init_curriculum)
        if cost_margin is not None:
            env = LagrangianCost(env, cost_margin)
        if shield:
            env = SafetyShieldWrapper(env, shield=DeterministicSafetyShield(shield_config()), w_div=0.10)
        return Monitor(env, filename=monitor_csv, info_keywords=("termination_reason",))
    return _f


class LagrangeMultiplier(BaseCallback):
    """Dual ascent on lam after every rollout: lam <- clip(lam + lr * (mean episode cost - limit), 0, lam_max)."""

    def __init__(self, n_envs: int, cost_limit: float, lr: float, lam_max: float = 20.0):
        super().__init__()
        self.cost_limit, self.lr, self.lam_max = cost_limit, lr, lam_max
        self.lam, self.ep_cost, self.finished, self.history = 0.0, np.zeros(n_envs), [], []

    def _on_step(self) -> bool:
        for i, (done, info) in enumerate(zip(self.locals["dones"], self.locals["infos"])):
            self.ep_cost[i] += info.get("cost", 0.0)
            if done:
                self.finished.append(self.ep_cost[i])
                self.ep_cost[i] = 0.0
        return True

    def _on_rollout_end(self) -> None:
        if not self.finished:
            return
        jc = float(np.mean(self.finished))
        self.finished = []
        self.lam = float(np.clip(self.lam + self.lr * (jc - self.cost_limit), 0.0, self.lam_max))
        self.training_env.env_method("set_lambda", self.lam)
        self.history.append({"steps": self.num_timesteps, "episode_cost": round(jc, 3), "lambda": round(self.lam, 3)})
        if len(self.history) % 6 == 0:
            print(f"[lagrange] {self.num_timesteps:,} steps: episode cost {jc:.2f} (limit {self.cost_limit}), "
                  f"lambda {self.lam:.2f}", flush=True)


class CurriculumProgress(BaseCallback):
    def __init__(self, total: int, schedule, log_every: int = 25_000):
        super().__init__()
        self.total, self.schedule, self.log_every = total, list(schedule), log_every
        self.applied = set()
        self.outcomes = deque(maxlen=200)
        self.t0 = time.time()
        self.next_log = log_every
        self.history = []

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None and not np.all(np.isfinite(rewards)):
            print("[train] non-finite reward; stopping", flush=True)
            return False
        frac = self.num_timesteps / self.total
        for i, (f, kwargs) in enumerate(self.schedule):
            if frac >= f and i not in self.applied:
                self.training_env.env_method("set_curriculum", **kwargs)
                self.applied.add(i)
                print(f"[train] {self.num_timesteps:,} steps: curriculum -> {kwargs}", flush=True)
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.outcomes.append(info.get("termination_reason", "?"))
        if self.num_timesteps >= self.next_log:
            self.next_log += self.log_every
            n = max(len(self.outcomes), 1)
            rates = {k: round(100 * sum(o == k for o in self.outcomes) / n, 1)
                     for k in ("goal_reached", "obstacle_collision", "episode_timeout", "out_of_bounds",
                               "ground_crash")}
            sps = self.num_timesteps / max(time.time() - self.t0, 1e-6)
            self.history.append({"steps": self.num_timesteps, **rates, "steps_per_s": round(sps)})
            print(f"[train] {self.num_timesteps:>9,}/{self.total:,}  last {n} eps: success {rates['goal_reached']:5.1f}%  "
                  f"collision {rates['obstacle_collision']:5.1f}%  timeout {rates['episode_timeout']:5.1f}%  "
                  f"{sps:5.0f} steps/s", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), required=True)
    ap.add_argument("--steps", type=int, default=1_500_000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--init", default=None, help="checkpoint to continue from")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--reward", action="append", default=[], metavar="FIELD=VALUE",
                    help="override a UAVWorldEnvConfig reward field (ablations), e.g. reward_clearance_weight=0")
    ap.add_argument("--lagrangian", action="store_true", help="PPO-Lagrangian: constrain near-miss cost")
    ap.add_argument("--cost-margin", type=float, default=0.15, help="hull clearance (m) below which cost = 1")
    ap.add_argument("--cost-limit", type=float, default=1.0, help="target mean cost per episode")
    ap.add_argument("--lambda-lr", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(1)
    out = Path(args.out or PROJECT_ROOT / "experiments" / "world" / (args.stage + ("_noshield" if args.no_shield else "")))
    out.mkdir(parents=True, exist_ok=True)
    init_cur, schedule = STAGES[args.stage]
    overrides = {}
    for item in args.reward:
        field, value = item.split("=", 1)
        if not field.startswith("reward_") or not hasattr(UAVWorldEnvConfig, field):
            ap.error(f"unknown reward field {field!r}")
        overrides[field] = float(value)
    fns = [make_env_fn(init_cur, not args.no_shield, str(out / f"monitor_{i}.csv"), overrides,
                       args.cost_margin if args.lagrangian else None) for i in range(args.n_envs)]
    venv = SubprocVecEnv(fns, start_method="fork") if args.n_envs > 1 else DummyVecEnv(fns)
    venv.seed(args.seed)
    n_steps = max(256, 4096 // args.n_envs)
    common = dict(learning_rate=3e-4, n_steps=n_steps, batch_size=256, n_epochs=10, gamma=0.99, gae_lambda=0.95,
                  clip_range=0.2, ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, seed=args.seed, device="cpu",
                  verbose=0)
    if args.init:
        model = PPO.load(args.init, env=venv, **common)
    else:
        model = PPO("MlpPolicy", venv, policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]),
                                                          activation_fn=torch.nn.Tanh), **common)
    cb = CurriculumProgress(args.steps, schedule)
    ckpt = CheckpointCallback(save_freq=max(1, 100_000 // args.n_envs), save_path=str(out / "checkpoints"),
                              name_prefix=args.stage)
    callbacks = [cb, ckpt]
    lagrange = LagrangeMultiplier(args.n_envs, args.cost_limit, args.lambda_lr) if args.lagrangian else None
    if lagrange is not None:
        callbacks.append(lagrange)
    t0 = time.time()
    model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=True)
    model.save(out / "model.zip")
    git = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=PROJECT_ROOT).stdout.strip()
    json.dump({"stage": args.stage, "steps": int(model.num_timesteps), "wall_s": round(time.time() - t0),
               "init": args.init, "shield": not args.no_shield, "n_envs": args.n_envs, "git_head": git,
               "reward_overrides": overrides,
               "lagrangian": None if lagrange is None else {"cost_margin": args.cost_margin,
                                                            "cost_limit": args.cost_limit,
                                                            "lambda_lr": args.lambda_lr, "history": lagrange.history},
               "curriculum": {"initial": init_cur, "schedule": schedule}, "history": cb.history},
              open(out / "training_summary.json", "w"), indent=1, default=str)
    print(f"[train] saved {out / 'model.zip'} after {model.num_timesteps:,} steps in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
