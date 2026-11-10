"""Unit tests for the M10 mission fail-safe layer and the PPO-Lagrangian cost wrapper (no simulator needed)."""

import math

import gymnasium as gym
import numpy as np
import pytest

from uav_sitl.mission import Mission, MissionConfig


def _mission():
    return Mission(px4=None, gz=None, model_path="unused", obs_mode="50d", cfg=MissionConfig(waypoints=[[5, 0, 2]]))


def _obs(level=None, down=None, up=None, far=10.0):
    obs = np.full(50, far, dtype=np.float32)
    for lo, vals in ((18, level), (26, down), (34, up)):
        for k, r in (vals or {}).items():
            obs[lo + k] = r
    return obs


def test_brake_inactive_when_clear():
    cmd, braked = _mission()._emergency_brake((1.0, 0.5, -0.3, 0.2), _obs())
    assert not braked
    assert cmd == pytest.approx((1.0, 0.5, -0.3, 0.2))


def test_brake_cancels_only_motion_toward_the_obstacle():
    # obstacle dead ahead (sector 0): forward motion removed, sideways motion and yaw kept
    cmd, braked = _mission()._emergency_brake((1.5, 0.4, 0.0, 0.3), _obs(level={0: 0.3}))
    assert braked
    assert cmd[0] == pytest.approx(0.0, abs=1e-9)
    assert cmd[1] == pytest.approx(0.4)
    assert cmd[3] == pytest.approx(0.3)


def test_brake_allows_backing_away():
    cmd, braked = _mission()._emergency_brake((-1.0, 0.0, 0.0, 0.0), _obs(level={0: 0.3}))
    assert braked
    assert cmd[0] == pytest.approx(-1.0)


def test_brake_diagonal_sector_removes_projection():
    # obstacle front-left (sector 1, 45 deg): command straight left keeps only the component orthogonal to it
    cmd, _ = _mission()._emergency_brake((0.0, 1.0, 0.0, 0.0), _obs(level={1: 0.2}))
    u = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4)])
    assert float(np.array(cmd[:2]) @ u) == pytest.approx(0.0, abs=1e-9)
    assert cmd[0] == pytest.approx(-0.5)
    assert cmd[1] == pytest.approx(0.5)


def test_brake_vertical_bands():
    cmd, _ = _mission()._emergency_brake((0.0, 0.0, -0.8, 0.0), _obs(down={4: 0.3}))
    assert cmd[2] == 0.0                       # no descending onto something below
    cmd, _ = _mission()._emergency_brake((0.0, 0.0, 0.8, 0.0), _obs(up={2: 0.3}))
    assert cmd[2] == 0.0                       # no climbing into a branch above
    cmd, _ = _mission()._emergency_brake((0.0, 0.0, 0.8, 0.0), _obs(down={4: 0.3}))
    assert cmd[2] == pytest.approx(0.8)        # climbing away from something below is fine


def test_runtime_cbf_layer_blocks_motion_into_close_return():
    """PolicyRuntime(cbf=True) on a synthetic 7x90 scan with a wall 0.6 m ahead at every elevation: a nominal
    full-speed-forward command must be cut by the CBF layer (alpha * h ~ 0 at the wall)."""
    from pathlib import Path
    from types import SimpleNamespace

    from rl_env.lidar3d import DEFAULT_SPEC
    from uav_sitl.policy_runtime import PolicyRuntime

    model = Path(__file__).resolve().parents[1] / "experiments" / "world" / "forest" / "model.zip"
    if not model.exists():
        pytest.skip("forest model not trained")
    spec = DEFAULT_SPEC
    az = np.degrees(spec.azimuths_rad())
    ranges = np.full((len(spec.elevations_deg), spec.n_azimuth), spec.max_range)
    ahead = np.abs(az) < 25.0
    for i, el in enumerate(spec.elevations_deg):
        ranges[i, ahead] = 0.6 / max(math.cos(math.radians(el)), 0.2)
    scan = SimpleNamespace(ranges=ranges.reshape(-1), stamp=1.0)
    rt = PolicyRuntime(str(model), "50d", goal_world=(10.0, 0.0, 2.0), cbf=True)
    rt.model = SimpleNamespace(predict=lambda obs, deterministic=True: (np.array([1.0, 0.0, 0.0, 0.0]), None))
    step = rt.step((0.0, 0.0, 2.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0), scan)
    assert step.cmd[0] <= 0.25            # nominal 3 m/s straight into the wall is cut to (near) zero
    assert rt.n_cbf == 1


class _StubEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,))
    action_space = gym.spaces.Box(-1, 1, (1,))

    def __init__(self, clearances):
        self.clearances = list(clearances)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 1.0, False, False, {"hull_clearance_m": self.clearances.pop(0)}


def test_lagrangian_cost_wrapper():
    from training.train_world_policy import LagrangianCost
    env = LagrangianCost(_StubEnv([0.5, 0.1, -0.05]), margin=0.15)
    env.reset()
    env.set_lambda(2.0)
    rewards, costs = [], []
    for _ in range(3):
        _, r, _, _, info = env.step(np.zeros(1))
        rewards.append(r)
        costs.append(info["cost"])
    assert costs == [0.0, 1.0, 1.0]
    assert rewards == pytest.approx([1.0, -1.0, -1.0])


def test_sitl_movers_positions_are_a_function_of_sim_time():
    """The mover daemon and the control process step their own SitlMovers copies at different moments;
    collision checks are only valid if both land on identical positions for the same sim time."""
    pytest.importorskip("gz.transport13")
    from uav_sitl.movers import SitlMovers
    a, b = SitlMovers(8301, 4), SitlMovers(8301, 4)
    for t in np.arange(0.0, 20.0, 0.013):          # irregular, fine-grained calls (daemon)
        a.advance_to(float(t))
    for t in (3.7, 11.2, 19.99):                   # a few coarse calls (control loop)
        b.advance_to(t)
    a.advance_to(19.99)
    assert np.allclose(np.array(a.movers.pos), np.array(b.movers.pos))
    assert a.t_sim == pytest.approx(b.t_sim)
