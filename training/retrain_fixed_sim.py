"""
retrain_fixed_sim.py
--------------------
Retrain the M2, M4 and M5 policies on the corrected simulator (tilt direction,
heading-frame commands, body rates, combined speed/acceleration limits, ground
crash, canonical quaternion).

Each pipeline reuses the hyperparameters, environment configs, seeds and
curricula of the original training scripts:
  M2  training/train_ppo.py                               (UAVFastNav-v0, 100k steps)
  M4  experiments/milestone4/phase4b_gate_b3/train_ppo_26d.py (UAVFastNav-v1, 1M steps,
      4-stage curriculum at 15/35/65 %, linear LR 3e-4 -> 5e-5)
  M5  experiments/milestone5/train_m5_{pilot,stage1,stage2b,stage3}.py
      pilot (stage 0, 100k) -> stage 1 (100k) -> stage 2b (stage 2 + reverse
      penalty, 100k) -> stage 3 (regimes + domain randomisation + safety shield).
      The original stage 3 went through several partial runs (~55k steps in
      total); here it is one run of --m5-stage3-steps (default 60k) from stage 2b.

Outputs go to --out-dir (default experiments/retrain_fixed_sim/) and never touch
the original checkpoints.

Usage (in WSL, from the project root, e.g. via .\\wsl.ps1 run):
  python training/retrain_fixed_sim.py --only m5
  python training/retrain_fixed_sim.py --only m4 --scale 0.01 --out-dir /tmp/smoke
"""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

import rl_env  # noqa: F401  (registers UAVFastNav-v0/v1/v2)
from rl_env.safety_shield import SafetyShieldWrapper
from rl_env.uav_fast_env_v2 import UAVObstacleEnvConfigV2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_ENVS = 1  # set from --n-envs; >1 runs that many simulators in parallel processes (same 2048-step batch)


def make_env(factory, seed: int, monitor_csv: Path):
    """Single Monitor-wrapped env (n_envs=1, the original setup) or a SubprocVecEnv of N_ENVS copies."""
    if N_ENVS == 1:
        env = Monitor(factory(), filename=str(monitor_csv))
        env.reset(seed=seed)
        return env
    stem = monitor_csv.with_suffix("")
    fns = [lambda i=i: Monitor(factory(), filename=f"{stem}_{i}.csv") for i in range(N_ENVS)]
    venv = SubprocVecEnv(fns, start_method="fork")
    venv.seed(seed)
    return venv


def rollout(n_steps_total: int) -> int:
    return max(64, n_steps_total // N_ENVS)

PPO_COMMON = dict(batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                  ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5, device="cpu", verbose=0)
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=torch.nn.Tanh)


class ProgressCallback(BaseCallback):
    """Prints progress and tracks rolling success; stops on NaN/Inf rewards."""

    def __init__(self, tag: str, total: int, every: int = 10_000):
        super().__init__()
        self.tag, self.total, self.every = tag, total, every
        self.outcomes = deque(maxlen=100)
        self.episodes = 0
        self.t0 = time.time()
        self.next_print = every

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None and not np.all(np.isfinite(rewards)):
            print(f"[{self.tag}] NaN/Inf reward at step {self.num_timesteps}; stopping", flush=True)
            return False
        for done, info in zip(self.locals.get("dones", []), self.locals.get("infos", [])):
            if done:
                self.episodes += 1
                self.outcomes.append(1.0 if info.get("is_success") else 0.0)
        if self.num_timesteps >= self.next_print:
            self.next_print += self.every
            rate = 100.0 * float(np.mean(self.outcomes)) if self.outcomes else float("nan")
            print(f"[{self.tag}] {self.num_timesteps:>8,}/{self.total:,} steps  {self.episodes:5d} eps  "
                  f"rolling success {rate:5.1f}%  {time.time() - self.t0:6.0f}s", flush=True)
        return True

    def summary(self) -> dict:
        return {"episodes": self.episodes,
                "rolling_success_last100": float(np.mean(self.outcomes)) if self.outcomes else None,
                "wall_s": round(time.time() - self.t0, 1)}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finish(model: PPO, cb: ProgressCallback, path: Path, record: dict) -> dict:
    model.save(path)
    record.update(cb.summary(), steps=int(model.num_timesteps), checkpoint=str(path), sha256=sha256(path))
    print(f"[{cb.tag}] saved {path} ({record['steps']:,} steps, {record['wall_s']}s)", flush=True)
    return record


def train_m2(out: Path, scale: float) -> dict:
    """Mirror of training/train_ppo.py defaults."""
    steps, seed = max(2048, int(100_000 * scale)), 42
    models, logs = out / "m2" / "models", out / "m2" / "logs"
    models.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    env = make_env(lambda: gym.make("UAVFastNav-v0"), seed, logs / "monitor.csv")
    eval_env = Monitor(gym.make("UAVFastNav-v0"))
    eval_env.reset(seed=seed + 100)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=rollout(2048), policy_kwargs=POLICY_KWARGS,
                seed=seed, **PPO_COMMON)
    cb = ProgressCallback("M2", steps)
    ev = EvalCallback(eval_env, best_model_save_path=str(models), log_path=str(logs),
                      eval_freq=10_000, n_eval_episodes=20, deterministic=True)
    model.learn(total_timesteps=steps, callback=[cb, ev])
    return finish(model, cb, models / "ppo_uav_final.zip", {"pipeline": "M2", "env": "UAVFastNav-v0"})


def _m4_curriculum_callback_cls():
    """Reuse the exact curriculum callback from the original M4 training script."""
    src = PROJECT_ROOT / "experiments" / "milestone4" / "phase4b_gate_b3" / "train_ppo_26d.py"
    spec = importlib.util.spec_from_file_location("train_ppo_26d_original", src)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CurriculumStageCallback


def train_m4(out: Path, scale: float) -> dict:
    """Mirror of experiments/milestone4/phase4b_gate_b3/train_ppo_26d.py defaults."""
    steps, seed = max(4096, int(1_000_000 * scale)), 42
    models, logs = out / "m4" / "models", out / "m4" / "logs"
    models.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    env = make_env(lambda: gym.make("UAVFastNav-v1"), seed, logs / "monitor.csv")
    eval_env = Monitor(gym.make("UAVFastNav-v1"))
    eval_env.reset(seed=seed + 1000)

    lr0, lr_min = 3e-4, 5e-5
    model = PPO("MlpPolicy", env, learning_rate=lambda p: float(lr_min + (lr0 - lr_min) * p), n_steps=rollout(2048),
                policy_kwargs=POLICY_KWARGS, seed=seed, **PPO_COMMON)
    curriculum = _m4_curriculum_callback_cls()(eval_env=eval_env, total_timesteps=steps,
                                               checkpoint_dir=str(models / "stage_checkpoints"), verbose=1)
    cb = ProgressCallback("M4", steps, every=25_000)
    ev = EvalCallback(eval_env, best_model_save_path=str(models), log_path=str(logs),
                      eval_freq=25_000, n_eval_episodes=20, deterministic=True)
    model.learn(total_timesteps=steps, callback=[curriculum, cb, ev])
    record = {"pipeline": "M4", "env": "UAVFastNav-v1",
              "curriculum_transitions": [{k: v for k, v in t.items() if k != "timestamp"}
                                         for t in curriculum.stage_transition_log]}
    return finish(model, cb, models / "ppo_uav_obstacle_26d_final.zip", record)


def _m5_env(config: UAVObstacleEnvConfigV2, monitor_csv: Path, shield: bool, seed: int):
    def factory():
        env = gym.make("UAVFastNav-v2", config=config)
        return SafetyShieldWrapper(env, w_div=0.10) if shield else env
    return make_env(factory, seed, monitor_csv)


def train_m5(out: Path, scale: float, stage3_steps: int, stage3_init: str = None) -> list:
    """Mirror of the M5 pilot -> stage1 -> stage2b -> stage3 scripts.
    With stage3_init, only stage 3 runs (continued from that checkpoint) and is saved as stage3_ext."""
    models, logs = out / "m5" / "models", out / "m5" / "logs"
    models.mkdir(parents=True, exist_ok=True); logs.mkdir(parents=True, exist_ok=True)
    stages = [
        # name, seed, steps, config kwargs, shield, init checkpoint
        ("pilot", 100, 100_000, dict(curriculum_stage=0, reward_align_weight=0.10, reward_speed_gate_weight=0.10,
                                     grid_resolution=0.10, max_resample_attempts=50), False, None),
        ("stage1", 200, 100_000, dict(curriculum_stage=1, reward_align_weight=0.10, reward_speed_gate_weight=0.10,
                                      grid_resolution=0.10, max_resample_attempts=50), False, "pilot"),
        ("stage2b", 350, 100_000, dict(curriculum_stage=2, reward_align_weight=0.10, reward_speed_gate_weight=0.10,
                                       reward_reverse_weight=0.50, grid_resolution=0.10,
                                       max_resample_attempts=50), False, "stage1"),
        ("stage3", 402, stage3_steps, dict(curriculum_stage=3, reward_align_weight=0.10,
                                           reward_speed_gate_weight=0.10, reward_reverse_weight=0.35,
                                           reward_action_divergence_weight=0.10, min_path_clearance=0.45,
                                           safety_margin=0.10, grid_resolution=0.10, max_resample_attempts=50,
                                           enable_domain_randomization=True), True, "stage2b"),
    ]
    records, paths = [], {}
    if stage3_init:
        name, seed, _, cfg, shield, _ = stages[-1]
        stages = [("stage3_ext", seed + 1, stage3_steps, cfg, shield, "given")]
        paths["given"] = Path(stage3_init)
    for name, seed, full_steps, cfg, shield, init in stages:
        steps = max(2000, int(full_steps * scale))
        env = _m5_env(UAVObstacleEnvConfigV2(**cfg), logs / f"{name}_monitor.csv", shield, seed)
        if init is None:
            model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=rollout(2048), policy_kwargs=POLICY_KWARGS,
                        seed=seed, **PPO_COMMON)
        else:
            model = PPO.load(paths[init], env=env, learning_rate=3e-4, n_steps=rollout(2000), seed=seed,
                             **PPO_COMMON)
        cb = ProgressCallback(f"M5-{name}", steps)
        model.learn(total_timesteps=steps, callback=cb, reset_num_timesteps=True)
        paths[name] = models / f"ppo_m5_{name}_64x64.zip"
        records.append(finish(model, cb, paths[name], {"pipeline": "M5", "stage": name, "seed": seed,
                                                        "init": str(paths[init]) if init else None,
                                                        "safety_shield": shield, "env_config": cfg}))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=["m2", "m4", "m5"], required=True)
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "experiments" / "retrain_fixed_sim"))
    parser.add_argument("--scale", type=float, default=1.0, help="Fraction of the original step budgets (smoke tests)")
    parser.add_argument("--m5-stage3-steps", type=int, default=60_000)
    parser.add_argument("--m5-stage3-init", default=None, help="continue only stage 3 from this checkpoint")
    parser.add_argument("--n-envs", type=int, default=1,
                        help="parallel simulator processes (1 = original single-env setup)")
    args = parser.parse_args()

    torch.set_num_threads(1)
    global N_ENVS
    N_ENVS = max(1, args.n_envs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    git_rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                             cwd=PROJECT_ROOT).stdout.strip()
    sim_dirty = subprocess.run(["git", "status", "--porcelain", "rl_env"], capture_output=True, text=True,
                               cwd=PROJECT_ROOT).stdout.strip() != ""
    t0 = time.time()
    if args.only == "m2":
        records = [train_m2(out, args.scale)]
    elif args.only == "m4":
        records = [train_m4(out, args.scale)]
    else:
        records = train_m5(out, args.scale, max(2000, int(args.m5_stage3_steps * args.scale)), args.m5_stage3_init)
    summary = {"pipeline": args.only, "scale": args.scale, "n_envs": N_ENVS, "git_head": git_rev,
               "rl_env_uncommitted_changes": sim_dirty, "total_wall_s": round(time.time() - t0, 1),
               "runs": records}
    with open(out / f"{args.only}_summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"[{args.only.upper()}] done in {summary['total_wall_s']}s -> {out / (args.only + '_summary.json')}",
          flush=True)


if __name__ == "__main__":
    main()
