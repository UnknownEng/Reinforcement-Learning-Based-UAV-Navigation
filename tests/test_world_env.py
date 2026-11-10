"""
Tests for the 3-D worlds (rl_env/world3d.py), the multi-channel LiDAR (rl_env/lidar3d.py)
and the UAVWorldNav-v0 environment (rl_env/uav_world_env.py).
"""

import math

import numpy as np
import pytest

from rl_env.lidar3d import DEFAULT_SPEC, features, scan
from rl_env.uav_world_env import OBS_DIM, UAVWorldEnv, UAVWorldEnvConfig
from rl_env.world3d import (World3D, free_space_connected, generate_forest, generate_urban,
                            generate_warehouse)


def _single(cyl=None, sph=None, box=None, cap=None):
    z = lambda n: np.zeros((0, n))
    return World3D(np.array(cyl or z(5)).reshape(-1, 5), np.array(sph or z(4)).reshape(-1, 4),
                   np.array(box or z(7)).reshape(-1, 7), 30.0, cap=np.array(cap or z(7)).reshape(-1, 7))


O = np.array([0.0, 0.0, 2.0])
X = np.array([[1.0, 0.0, 0.0]])


@pytest.mark.parametrize("world,expected", [
    (_single(sph=[(5.0, 0.0, 2.0, 1.0)]), 4.0),
    (_single(cyl=[(5.0, 0.0, 0.5, 0.0, 10.0)]), 4.5),
    (_single(cyl=[(5.0, 0.0, 0.5, 0.0, 1.0)]), 10.0),              # passes over a short stump
    (_single(box=[(5.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0)]), 4.0),
    (_single(box=[(5.0, 0.0, 2.0, 1.0, 1.0, 1.0, math.pi / 4)]), 5.0 - math.sqrt(2.0)),
    (_single(cap=[(5.0, -3.0, 2.0, 5.0, 3.0, 2.0, 0.2)]), 4.8),     # horizontal branch across the ray
])
def test_ray_hits_analytic(world, expected):
    assert world.raycast(O, X, 10.0)[0] == pytest.approx(expected, abs=1e-9)
    assert world.raycast_numpy(O, X, 10.0)[0] == pytest.approx(expected, abs=1e-9)


def test_cylinder_cap_and_inside():
    w = _single(cyl=[(0.0, 0.0, 0.5, 0.0, 1.0)])
    assert w.raycast(np.array([0.0, 0.0, 3.0]), np.array([[0.0, 0.0, -1.0]]), 10.0)[0] == pytest.approx(2.0)
    assert w.raycast(np.array([0.0, 0.0, 0.5]), X, 10.0)[0] == 0.0          # origin inside
    assert w.distance(np.array([[0.0, 0.0, 0.5]]))[0] < 0.0


def test_jit_matches_numpy_on_random_forest():
    rng = np.random.default_rng(1)
    w = generate_forest(rng, 15.0, "dense")
    for _ in range(3):
        o = np.array([rng.uniform(-10, 10), rng.uniform(-10, 10), rng.uniform(1.5, 4.0)])
        d = rng.normal(size=(400, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        loc = w.local(o[:2], 11.0)
        np.testing.assert_allclose(loc.raycast(o, d, 10.0), loc.raycast_numpy(o, d, 10.0), atol=1e-9)
        pts = o + rng.normal(size=(200, 3))
        np.testing.assert_allclose(loc.distance(pts), loc.distance_numpy(pts), atol=1e-9)


def test_forest_is_realistic():
    rng = np.random.default_rng(2)
    w = generate_forest(rng, 20.0, "medium")                # 40 x 40 m = 0.16 ha at ~700 stems/ha
    assert 80 <= w.meta["n_mature"] + w.meta["n_saplings"] <= 140
    trunks = w.cyl[w.cyl_tag == "trunk"]
    assert 0.05 <= np.median(trunks[:, 2]) <= 0.25           # radius, i.e. 10-50 cm diameter
    assert np.all(trunks[:, 4] >= 12.0)                      # mature trees are tall
    assert np.any(w.cap_tag == "branch") and np.any(w.sph_tag == "bush")
    # something to hit inside the 1.5-4.5 m flight band besides trunks (saplings, foliage, branches)
    band = w.local(np.zeros(2), 30.0, (1.5, 4.5))
    assert len(band.sph) + len(band.cap) > 20


def test_lidar_bands_see_obstacles_above_and_below():
    R = np.eye(3)
    s = DEFAULT_SPEC.mount_offset_body[2] + O[2]               # sensor height (vehicle at 2 m, 0.64 m mast)
    level = features(scan(_single(cyl=[(3.0, 0.0, 0.3, 0.0, 10.0)]), O, R))
    assert level[0] < 3.0 and np.all(level[1:8] >= 9.9)      # front sector of the level band
    up = features(scan(_single(sph=[(4.0, 0.0, s + 4.0 * math.tan(math.radians(25)), 0.6)]), O, R))
    assert up[16] < 6.0 and up[0] >= 9.9                     # seen only by the up band (~25 deg up)
    down = features(scan(_single(sph=[(3.0, 0.0, s - 3.0 * math.tan(math.radians(25)), 0.4)]), O, R))
    assert down[8] < 4.0 and down[0] >= 9.9                  # seen only by the down band (~25 deg down)
    left = features(scan(_single(cyl=[(0.0, 3.0, 0.3, 0.0, 10.0)]), O, R))
    assert left[2] < 3.0                                     # sector 2 = left (CCW numbering)


def test_feasibility_detects_blocking_wall():
    wall = _single(box=[(0.0, 0.0, 5.0, 0.3, 30.0, 5.0, 0.0)])   # 60 m long, 10 m high wall at x = 0
    ok, _ = free_space_connected(wall, np.array([-5.0, 0.0]), np.array([5.0, 0.0]))
    assert not ok
    gap = _single(box=[(0.0, -15.0, 5.0, 0.3, 14.0, 5.0, 0.0), (0.0, 15.0, 5.0, 0.3, 14.0, 5.0, 0.0)])
    ok, _ = free_space_connected(gap, np.array([-5.0, 0.0]), np.array([5.0, 0.0]))
    assert ok                                                # 2 m doorway between the halves


def test_env_api_and_collision():
    env = UAVWorldEnv(UAVWorldEnvConfig(check_feasibility=False, enable_noise=False))
    trunk = _single(cyl=[(3.0, 0.0, 0.4, 0.0, 15.0)])
    obs, info = env.reset(seed=0, options={"world": trunk, "start_yaw": 0.0})
    assert obs.shape == (OBS_DIM,) and np.all(np.isfinite(obs))
    env.dynamics.reset(x0=0.0, y0=0.0, z0=2.0, yaw0=0.0)
    env.goal_world = np.array([8.0, 0.0, 2.0], dtype=np.float32)
    reasons = []
    for _ in range(80):
        obs, r, term, trunc, info = env.step(np.array([1.0, 0.0, 0.0, 0.0], np.float32))
        reasons.append(info["termination_reason"])
        if term or trunc:
            break
    assert info["termination_reason"] == "obstacle_collision"
    assert info["pos_world"][0] < 3.0                        # stopped at the trunk, not past it


def test_env_success_in_empty_world_and_generators():
    env = UAVWorldEnv(UAVWorldEnvConfig(check_feasibility=False, enable_noise=False))
    obs, info = env.reset(seed=1, options={"world": World3D.empty(20.0)})
    for _ in range(450):
        g = obs[0:3]
        a = np.clip([g[0] / 2.0, g[1] / 2.0, g[2] / 1.0, math.atan2(g[1], g[0])], -1, 1).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    assert info["termination_reason"] == "goal_reached"
    for gen in (generate_urban, generate_warehouse):
        w = gen(np.random.default_rng(3), 20.0)
        assert len(w.box) > 5


def test_movers_move_and_domain_randomization_ranges():
    env = UAVWorldEnv(UAVWorldEnvConfig(n_movers_range=(3, 3), domain_randomization=True, check_feasibility=False))
    env.reset(seed=4)
    p0 = np.array(env.movers.pos).copy()
    for _ in range(10):
        env.step(np.zeros(4, np.float32))
    assert np.linalg.norm(np.array(env.movers.pos) - p0, axis=1).min() > 0.3
    dr = env.dr_params
    assert 0.8 <= dr["mass_scale"] <= 1.2 and 0.02 <= dr["latency_s"] <= 0.14 and 0.0 <= dr["wind_speed"] <= 5.0
    assert env.dynamics.latency_substeps == max(1, round(dr["latency_s"] / 0.01))
