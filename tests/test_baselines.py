"""
Fast tests for the Milestone-9 classical baselines (baselines/): APF, DWA, (Informed) RRT*,
the CBF safety filter and the runner.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:                  # `baselines` is a top-level package of the repo
    sys.path.insert(0, str(ROOT))

from baselines import (CBFSafetyFilter, RRTStarConfig, RRTStarPlanner, evaluate, make_controller,  # noqa: E402
                       path_length, solve_projection_qp)
from baselines.common import VehicleLimits, action_to_velocity  # noqa: E402
from rl_env.uav_world_env import Movers, UAVWorldEnv, UAVWorldEnvConfig  # noqa: E402
from rl_env.world3d import World3D  # noqa: E402

CFG = UAVWorldEnvConfig(check_feasibility=False, enable_noise=False)
FAST_RRT = dict(max_iters=300, max_time_s=30.0, replan_max_iters=300, replan_time_s=30.0)   # iteration budgets


def _world(cyl=(), half=20.0):
    return World3D(np.array(cyl, dtype=np.float64).reshape(-1, 5), np.zeros((0, 4)), np.zeros((0, 7)), half,
                   cap=np.zeros((0, 7)))


def _run(name, env, obs, steps, **kw):
    ctrl = make_controller(name, env, **kw)
    info = {}
    for _ in range(steps):
        a = ctrl.act(obs, env)
        assert a.shape == (4,) and a.dtype == np.float32
        assert np.all(np.isfinite(a)) and np.all(np.abs(a) <= 1.0)
        obs, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return ctrl, obs, info


# ------------------------------------------------------------------ controllers
@pytest.fixture(scope="module")
def dense_forest():
    env = UAVWorldEnv(CFG)
    env.reset(seed=7, options={"world_kind": "forest", "forest_preset": "dense"})
    return env.world


@pytest.mark.parametrize("name", ["apf", "dwa", "rrt_star", "informed_rrt_star"])
def test_actions_valid_in_forest(name, dense_forest):
    env = UAVWorldEnv(CFG)
    obs, _ = env.reset(seed=7, options={"world": dense_forest})     # same start/goal as the generated forest
    kw = FAST_RRT if "rrt" in name else {}
    ctrl, obs, _ = _run(name, env, obs, 6, **kw)
    assert ctrl.uses_map == ("rrt" in name)
    if not ctrl.uses_map:                       # onboard methods also run on the observation alone
        a = make_controller(name).act(obs)
        assert a.shape == (4,) and np.all(np.abs(a) <= 1.0)


@pytest.mark.parametrize("name", ["apf", "dwa", "rrt_star"])
def test_reaches_goal_in_empty_world(name):
    env = UAVWorldEnv(CFG)
    obs, _ = env.reset(seed=11, options={"world": World3D.empty(20.0)})
    _, _, info = _run(name, env, obs, 450, **(FAST_RRT if "rrt" in name else {}))
    assert info["termination_reason"] == "goal_reached"


@pytest.mark.parametrize("informed", [False, True])
def test_rrt_star_detours_around_tall_cylinder(informed):
    world = _world([(0.0, 0.0, 1.0, 0.0, 15.0)], half=10.0)             # cylinder taller than the band
    planner = RRTStarPlanner(world, RRTStarConfig(informed=informed, seed=0, max_iters=400, max_time_s=30.0))
    start, goal = np.array([-6.0, 0.0, 2.0]), np.array([6.0, 0.0, 2.0])
    res = planner.plan(start, goal)
    assert res.success and not res.direct
    path = res.path
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)
    dense = np.vstack([np.linspace(path[i], path[i + 1], 200) for i in range(len(path) - 1)])
    assert world.distance(dense).min() > planner.thr                   # r_uav + margin everywhere
    assert np.all((dense[:, 2] >= 1.0 - 1e-9) & (dense[:, 2] <= 4.5 + 1e-9))
    assert 12.3 < path_length(path) < 15.0                             # around it (straight = 12 m)
    assert path_length(path) <= res.raw_cost + 1e-9                    # smoothing never lengthens


def test_rrt_star_replans_around_unmapped_obstacle():
    env = UAVWorldEnv(CFG)
    obs, _ = env.reset(seed=11, options={"world": World3D.empty(20.0)})
    ctrl = make_controller("rrt_star", env, **FAST_RRT)
    action = ctrl.act(obs, env)                                         # plans the straight path
    mid = 0.5 * (ctrl.path[-2] + ctrl.path[-1]) + np.array([0.0, 0.0, CFG.hull_center_dz])
    movers = Movers(np.random.default_rng(0), CFG, 1, np.array([100.0, 100.0]))
    movers.kind, movers.pos, movers.goal, movers.speed, movers.radius = ["flyer"], [mid], [mid + 5.0], [0.0], [0.5]
    env.movers = movers                                                 # static, not in env.world
    for _ in range(450):
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            break
        action = ctrl.act(obs, env)
    assert info["termination_reason"] == "goal_reached"
    assert ctrl.n_replans >= 1 and info["min_hull_clearance_m"] > 0.0


# -------------------------------------------------------------------------- CBF
def _hover(world, xyz=(0.0, 0.0, 2.0), yaw=0.6, vel=(0.0, 0.0, 0.0)):
    """Env (no LiDAR noise) holding the vehicle at xyz / yaw / vel, with a matching observation."""
    env = UAVWorldEnv(UAVWorldEnvConfig(check_feasibility=False, enable_noise=False, lidar_noise_std=0.0))
    env.reset(seed=0, options={"world": world})
    env.dynamics.set_state(np.array(xyz), np.array(vel), 0.0, 0.0, yaw)
    return env, env._observe()


def _world_velocity(action, yaw):
    v, _ = action_to_velocity(action, VehicleLimits())
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def test_cbf_leaves_action_unchanged_when_far():
    rng = np.random.default_rng(0)
    for world in (World3D.empty(20.0), _world([(9.0, 0.0, 0.3, 0.0, 15.0)])):
        env, obs = _hover(world)
        cbf = CBFSafetyFilter(env=env)
        for a in rng.uniform(-1.2, 1.2, size=(40, 4)).astype(np.float32):
            out, info = cbf.filter_action(obs, a, env=env)
            assert not info["intervened"]
            assert np.array_equal(out, np.clip(a, -1.0, 1.0))


@pytest.mark.parametrize("vel", [(0.0, 0.0, 0.0), (0.8, 0.0, 0.0)])
def test_cbf_never_increases_approach_speed(vel):
    yaw, r_cyl, d_surface = 0.6, 0.3, 1.0                             # obstacle surface 1 m from the hull
    world = _world([(d_surface + r_cyl, 0.0, r_cyl, 0.0, 15.0)])
    env, obs = _hover(world, yaw=yaw, vel=vel)
    cbf = CBFSafetyFilter(env=env)
    n = np.array([1.0, 0.0])                                           # direction to the obstacle (world)
    bound = cbf.cfg.alpha * (d_surface - cbf.cfg.d_safe)
    rng = np.random.default_rng(1)
    actions = rng.uniform(-1.0, 1.0, size=(150, 4)).astype(np.float32)
    toward = np.array([math.cos(-yaw), math.sin(-yaw), 0.0, 0.0], dtype=np.float32)  # full speed at it
    n_interventions = 0
    for a in np.vstack([toward, actions]):
        out, info = cbf.filter_action(obs, a, env=env)
        assert not info["infeasible"]
        app_nom = n @ _world_velocity(a, yaw)[:2]
        app_out = n @ _world_velocity(out, yaw)[:2]
        assert app_out <= app_nom + 1e-6                               # never closes in faster
        assert app_out <= max(bound, 0.0) + 0.05                       # dh/dt >= -alpha h
        assert out[3] == pytest.approx(np.clip(a[3], -1, 1))          # yaw rate passed through
        n_interventions += info["intervened"]
    assert n_interventions > 10


def test_cbf_brakes_when_infeasible():
    world = _world([(0.0, 0.45, 0.1, 0.0, 15.0), (0.0, -0.45, 0.1, 0.0, 15.0)])   # squeezed in a gap
    env, obs = _hover(world, yaw=0.0)
    out, info = CBFSafetyFilter(env=env).filter_action(obs, np.array([0.5, 0.3, 0.2, 0.4], np.float32), env=env)
    assert info["infeasible"] and info["fallback"] == "brake"
    np.testing.assert_allclose(out[:3], 0.0, atol=1e-7)
    assert out[3] == pytest.approx(0.4)


def test_projection_qp_matches_slsqp_and_detects_infeasibility():
    rng = np.random.default_rng(2)
    for _ in range(10):
        A = rng.normal(size=(12, 3))
        b = A @ rng.normal(size=3) + rng.uniform(0.0, 0.5, size=12)    # feasible by construction
        u0 = 3.0 * rng.normal(size=3)
        u1, ok1 = solve_projection_qp(u0, A, b, "ldp")
        u2, ok2 = solve_projection_qp(u0, A, b, "slsqp")
        assert ok1 and ok2 and np.all(A @ u1 <= b + 1e-6)
        np.testing.assert_allclose(u1, u2, atol=1e-4)
    _, ok = solve_projection_qp(np.zeros(3), np.array([[1.0, 0, 0], [-1.0, 0, 0]]), np.array([-1.0, -1.0]))
    assert not ok


# ----------------------------------------------------------------------- runner
def test_runner_evaluate_and_factory():
    with pytest.raises(ValueError):
        make_controller("potential_field_v2")
    empty = {"world": World3D.empty(20.0)}
    res = evaluate("dwa", CFG, seeds=[11], reset_options=empty)
    res += evaluate("apf", CFG, seeds=[11], wrap_cbf=True, reset_options=empty)
    for rec in res:
        assert rec["outcome"] == "goal_reached" and rec["success"]
        for key in ("steps", "path_length", "initial_dist", "min_hull_clearance_m", "mean_compute_ms"):
            assert np.isfinite(rec[key])
        assert rec["mean_compute_ms"] > 0.0 and rec["path_length"] >= rec["initial_dist"] - 0.6
    assert "cbf_interventions" in res[1]


def test_cbf_moving_obstacle_term():
    """M6: a return whose range shrinks faster than the vehicle's own motion explains is treated as an
    approaching obstacle and tightens the constraint; a static return seen while flying toward it does not."""
    filt = CBFSafetyFilter(moving_obstacles=True)
    obs = np.zeros(50)
    obs[18:26] = 10.0
    obs[18] = 3.0                                    # sector 0 (ahead): closest return 3 m from the sensor
    pts = np.array([[3.0, 0.0, 0.40]])               # that return, relative to the hull centre
    n = pts / np.linalg.norm(pts, axis=1)[:, None]
    v_ego = np.array([1.0, 0.0, 0.0])                # flying toward it at 1 m/s
    obs[42] = -float(n[0] @ v_ego)                   # range rate of a static object
    assert filt.approach_speed(obs, pts, n, v_ego)[0] == pytest.approx(0.0)
    obs[42] = -float(n[0] @ v_ego) - 1.5             # same, plus an obstacle closing at 1.5 m/s
    assert filt.approach_speed(obs, pts, n, v_ego)[0] == pytest.approx(-(1.5 * 1.0 - 0.3), abs=0.05)
    far = np.array([[8.0, 0.0, 0.40]])               # a return 5 m behind the sector minimum is not gated in
    assert filt.approach_speed(obs, far, far / np.linalg.norm(far), v_ego)[0] == 0.0
