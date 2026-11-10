"""
common.py
---------
Shared helpers for the Milestone-9 classical navigation baselines.

Frames (same conventions as rl_env):
  world    ENU, fixed.
  body     FLU, attached to the vehicle; obs[0:3] (goal) and obs[3:6] (velocity) live here.
  heading  body frame rotated by yaw only; the action's velocity commands live here.
  local    world-aligned (ENU) axes centred on the vehicle. The onboard baselines reason in this
           frame: getting there needs only the attitude quaternion obs[9:13], so it remains
           onboard information (no absolute position, no map).

Observation layout of UAVWorldNav-v0 (rl_env/uav_world_env.py):
  [0:3] goal (body)  [3:6] velocity (body)  [6:9] rates  [9:13] quaternion  [13] altitude
  [14:18] previous action  [18:26] level band  [26:34] down band  [34:42] up band  [42:50] range rate
"""

from collections import deque
from dataclasses import dataclass, fields, replace
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rl_env.lidar3d import DEFAULT_SPEC, Lidar3DSpec
from rl_env.observation import quat_to_rotation_matrix

SECTOR_AZ = np.radians(np.arange(8) * 45.0)            # sector centres: 0 = front, CCW
BAND_EL = np.radians(np.array([0.0, -25.0, 25.0]))      # level, down, up band centres (obs order)


def unwrap(env):
    """The bare UAVWorldEnv behind gym wrappers (None stays None)."""
    return None if env is None else getattr(env, "unwrapped", env)


def make_config(cls, config=None, **overrides):
    """Instantiate a config dataclass, or copy `config` with `overrides` applied."""
    if config is None:
        return cls(**overrides)
    return replace(config, **overrides) if overrides else config


def split_overrides(overrides: Dict[str, Any], *classes) -> Tuple[Dict[str, Any], ...]:
    """Route keyword overrides to the config dataclass that defines them."""
    out = tuple({} for _ in classes)
    for key, value in overrides.items():
        for i, cls in enumerate(classes):
            if key in {f.name for f in fields(cls)}:
                out[i][key] = value
                break
        else:
            raise TypeError(f"unknown option {key!r} for {[c.__name__ for c in classes]}")
    return out


@dataclass(frozen=True)
class VehicleLimits:
    """Command limits of the action scaling (rl_env.action.ActionHandler)."""
    v_xy: float = 3.0          # combined horizontal speed cap (m/s)
    v_z: float = 1.5           # vertical speed cap (m/s)
    yaw_rate: float = 1.0      # rad/s

    @classmethod
    def from_env(cls, env) -> "VehicleLimits":
        cfg = getattr(unwrap(env), "config", None)
        if cfg is None:
            return cls()
        return cls(float(cfg.max_horizontal_velocity), float(cfg.max_vertical_velocity), float(cfg.max_yaw_rate))


@dataclass
class Frame:
    """Vehicle attitude decoded from an observation."""
    R: np.ndarray        # body -> world/local (3, 3)
    yaw: float           # rad, ENU

    def world_to_heading(self, v: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1], v[2]], dtype=np.float64)

    def heading_to_world(self, v: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]], dtype=np.float64)

    @property
    def R_heading(self) -> np.ndarray:
        """body -> heading (tilt only)."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]) @ self.R


def frame_from_obs(obs: np.ndarray) -> Frame:
    R = quat_to_rotation_matrix(np.asarray(obs[9:13], dtype=np.float64)).astype(np.float64)
    return Frame(R=R, yaw=math.atan2(R[1, 0], R[0, 0]))


def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------------------------------------------ LiDAR
_DIRS: Dict[Tuple[Lidar3DSpec, bool], np.ndarray] = {}


def ray_dirs_body(spec: Lidar3DSpec = DEFAULT_SPEC, beam_fill: bool = False) -> np.ndarray:
    """
    Cached unit ray directions, body FLU, shape (rows, n_az, 3); rows = n_el in the order of
    env.last_ranges. With beam_fill each ring is split into three rows at -half, 0, +half the ring
    spacing (rows = 3 n_el).
    """
    key = (spec, bool(beam_fill))
    d = _DIRS.get(key)
    if d is None:
        az = spec.azimuths_rad()
        el = np.radians(np.asarray(spec.elevations_deg, dtype=np.float64))
        if beam_fill:
            half = 0.5 * float(np.min(np.diff(el))) if len(el) > 1 else 0.0
            el = (el[:, None] + np.array([-half, 0.0, half])[None, :]).ravel()
        ce, se = np.cos(el)[:, None], np.sin(el)[:, None]
        d = np.stack([ce * np.cos(az)[None, :], ce * np.sin(az)[None, :], np.broadcast_to(se, (len(el), len(az)))],
                     axis=2)
        d = _DIRS[key] = np.ascontiguousarray(d, dtype=np.float64)
    return d


def lidar_grid_points(ranges: np.ndarray, R: np.ndarray, spec: Lidar3DSpec = DEFAULT_SPEC,
                      beam_fill: bool = False):
    """
    Returns of the raw range grid (n_el, n_az) as points (rows, n_az, 3) relative to the vehicle
    centre, rotated by R (body -> target frame), and a validity mask (a return closer than max range).

    beam_fill: the rings are 10 deg apart, so a surface between two rings (e.g. the underside of a
    foliage blob at flight height) is not sampled and point-cloud clearances come out optimistic.
    With beam_fill each return is also placed at +/- half the ring spacing (nearest-neighbour
    filling of the scan; rows = 3 n_el).
    """
    shape = (len(spec.elevations_deg), spec.n_azimuth)
    r = np.asarray(ranges, dtype=np.float64).reshape(shape)
    if beam_fill:
        r = np.repeat(r, 3, axis=0)
    d = ray_dirs_body(spec, beam_fill)
    pts = (np.asarray(spec.mount_offset_body, dtype=np.float64) + r[..., None] * d) @ R.T
    valid = np.isfinite(r) & (r < spec.max_range - 1e-3)
    return pts, valid


def sector_points(obs: np.ndarray, R: np.ndarray, spec: Lidar3DSpec = DEFAULT_SPEC):
    """Fallback point set (3, 8, 3) from the 24 sector minima obs[18:42], placed at the sector centres."""
    r = np.asarray(obs[18:42], dtype=np.float64).reshape(3, 8)          # level, down, up
    el, az = BAND_EL[:, None], SECTOR_AZ[None, :]
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el) * np.ones_like(az)], axis=-1)
    pts = (np.asarray(spec.mount_offset_body, dtype=np.float64) + r[..., None] * d) @ R.T
    valid = np.isfinite(r) & (r < spec.max_range - 1e-3)
    return pts, valid


def hull_center_dz(env) -> float:
    """Height of the collision-hull centre above the reference point pos_world (env.config.hull_center_dz)."""
    return float(getattr(getattr(unwrap(env), "config", None), "hull_center_dz", 0.0))


def onboard_points(obs: np.ndarray, env=None, ranges: Optional[np.ndarray] = None, R: Optional[np.ndarray] = None,
                   ground_margin: float = 0.15, spec: Optional[Lidar3DSpec] = None, beam_fill: bool = False,
                   blind_fill: Tuple[float, ...] = (), hull_dz: Optional[float] = None):
    """
    Onboard obstacle points relative to the collision-hull centre (the reference point pos_world
    raised by hull_dz, default env.config.hull_center_dz). R maps body axes to a frame whose third
    axis is world-vertical (local or heading frame; default local).

    Uses the raw grid (`ranges`, else env.last_ranges) and falls back to the sector minima in the
    observation. Returns at or below the ground plane are dropped, as a ground-segmentation stage
    would (the fast simulator has no ground returns; PX4/Gazebo does). See lidar_grid_points for
    beam_fill.

    blind_fill (heights relative to the hull centre, e.g. (0.3, 0.0, -0.3)): the LiDAR sits above the
    hull and cannot see below its lowest ring, so close to the vehicle the hull's own height band is
    unobserved and a trunk 0.4 m away only shows up 0.4 m higher. For each azimuth column the
    horizontally closest return is assumed to continue downward into that blind region (true for
    trunks, stems, posts and walls; conservative for overhanging objects): copies of it are added at
    the listed heights that lie below both the return and the lowest visible height there.
    Returns (points (rows, cols, 3), valid (rows, cols), source).
    """
    base = unwrap(env)
    spec = spec or getattr(base, "lidar", DEFAULT_SPEC)
    R = frame_from_obs(obs).R if R is None else R
    dz = hull_center_dz(env) if hull_dz is None else float(hull_dz)
    if ranges is None and base is not None:
        ranges = getattr(base, "last_ranges", None)
    if ranges is not None:
        pts, valid = lidar_grid_points(ranges, R, spec, beam_fill)
        low = ray_dirs_body(spec)[0] @ R.T                                   # lowest ring, (n_az, 3)
        source = "lidar"
    else:
        pts, valid = sector_points(obs, R, spec)
        el = math.radians(min(spec.elevations_deg))
        low = np.stack([math.cos(el) * np.cos(SECTOR_AZ), math.cos(el) * np.sin(SECTOR_AZ),
                        np.full(8, math.sin(el))], axis=1) @ R.T
        source = "sectors"
    pts = pts - np.array([0.0, 0.0, dz])
    alt_hull = float(obs[13]) + dz
    if ground_margin is not None:
        valid = valid & (alt_hull + pts[..., 2] > ground_margin)
    if blind_fill and np.any(valid):
        rho = np.where(valid, np.hypot(pts[..., 0], pts[..., 1]), np.inf)
        rows = np.argmin(rho, axis=0)
        cols = np.arange(rho.shape[1])
        pc, rc = pts[rows, cols], rho[rows, cols]
        mount_h = float((R @ np.asarray(spec.mount_offset_body, dtype=np.float64))[2]) - dz
        tan_low = low[:, 2] / np.maximum(np.hypot(low[:, 0], low[:, 1]), 1e-6)
        with np.errstate(invalid="ignore"):
            top = np.minimum(pc[:, 2], mount_h + rc * tan_low)                 # lowest observed height
        extra_p, extra_v = [], []
        for z in blind_fill:
            extra_p.append(np.column_stack([pc[:, 0], pc[:, 1], np.full(len(cols), float(z))]))
            ok = np.isfinite(rc) & (z < top)
            if ground_margin is not None:
                ok &= alt_hull + z > ground_margin
            extra_v.append(ok)
        pts = np.concatenate([pts, np.stack(extra_p)], axis=0)
        valid = np.concatenate([valid, np.stack(extra_v)], axis=0)
    return pts, valid, source


# ----------------------------------------------------------------- actions
def velocity_to_action(v_local: np.ndarray, yaw_rate: float, frame: Frame, limits: VehicleLimits) -> np.ndarray:
    """Local (world-aligned) velocity + yaw rate -> normalized heading-frame action in [-1, 1]^4."""
    v = frame.world_to_heading(np.asarray(v_local, dtype=np.float64))
    h = math.hypot(v[0], v[1])
    if h > limits.v_xy:
        v[:2] *= limits.v_xy / h
    a = np.array([v[0] / limits.v_xy, v[1] / limits.v_xy, v[2] / limits.v_z, yaw_rate / limits.yaw_rate])
    return np.clip(np.nan_to_num(a, nan=0.0), -1.0, 1.0).astype(np.float32)


def action_to_velocity(action: np.ndarray, limits: VehicleLimits) -> Tuple[np.ndarray, float]:
    """Inverse of the env's ActionHandler.scale_action: heading-frame velocity (3,) and yaw rate."""
    a = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float64).reshape(-1)[:4], nan=0.0,
                              posinf=1.0, neginf=-1.0), -1.0, 1.0)
    v = np.array([a[0] * limits.v_xy, a[1] * limits.v_xy, a[2] * limits.v_z])
    h = math.hypot(v[0], v[1])
    if h > limits.v_xy:
        v[:2] *= limits.v_xy / h
    return v, float(a[3] * limits.yaw_rate)


def yaw_rate_towards(direction_local: np.ndarray, frame: Frame, limits: VehicleLimits, gain: float = 1.5,
                     min_dist: float = 0.7) -> float:
    """P yaw-rate command that turns the nose toward a local (world-aligned) direction."""
    if math.hypot(direction_local[0], direction_local[1]) < min_dist:
        return 0.0
    err = wrap_angle(math.atan2(direction_local[1], direction_local[0]) - frame.yaw)
    return float(np.clip(gain * err, -limits.yaw_rate, limits.yaw_rate))


def vertical_limits(alt: float, z_floor: float, z_ceil: float, gain: float = 1.0) -> Tuple[float, float]:
    """(vz_min, vz_max) keeping the vehicle inside [z_floor, z_ceil]; no descent below the floor."""
    return -gain * max(0.0, alt - z_floor), gain * max(0.0, z_ceil - alt)


# ------------------------------------------------------- episode utilities
class ProgressMonitor:
    """Flags a local minimum: less than `min_progress` m of goal-distance reduction over `window` steps."""

    def __init__(self, window: int = 25, min_progress: float = 0.35, ignore_within: float = 1.0):
        self.window, self.min_progress, self.ignore_within = int(window), float(min_progress), float(ignore_within)
        self.hist: deque = deque(maxlen=self.window)

    def reset(self) -> None:
        self.hist.clear()

    def update(self, dist: float) -> bool:
        self.hist.append(float(dist))
        if dist < self.ignore_within or len(self.hist) < self.window:
            return False
        return (self.hist[0] - dist) < self.min_progress


class EpisodeTracker:
    """Detects that the env was reset (new world / goal, or step counter went back)."""

    def __init__(self):
        self._key = None
        self._step = -1

    def new_episode(self, env) -> bool:
        base = unwrap(env)
        if base is None:
            return False
        goal = getattr(base, "goal_world", None)
        key = (id(getattr(base, "world", None)), None if goal is None else tuple(np.round(np.asarray(goal), 4)))
        step = int(getattr(base, "step_count", 0))
        new = key != self._key or step < self._step
        self._key, self._step = key, step
        return new


def escape_tangent(away_xy: np.ndarray, goal_xy: np.ndarray, sign: int) -> np.ndarray:
    """
    Unit horizontal tangent for circulating around the obstacles: `away_xy` points away from them,
    sign +1 = counter-clockwise circulation, -1 = clockwise. Falls back to the goal direction.
    """
    n = np.asarray(away_xy, dtype=np.float64)
    nn = math.hypot(n[0], n[1])
    if nn < 1e-6:
        g = np.asarray(goal_xy, dtype=np.float64)
        gn = math.hypot(g[0], g[1])
        n = -g / gn if gn > 1e-6 else np.array([1.0, 0.0])
        nn = 1.0
    n = n / nn
    return sign * np.array([-n[1], n[0]])


def choose_escape_sign(away_xy: np.ndarray, goal_xy: np.ndarray, rng: np.random.Generator) -> int:
    """Circulate toward the side the goal is on; random if the goal is straight behind the obstacle."""
    t = escape_tangent(away_xy, goal_xy, +1)
    g = np.asarray(goal_xy, dtype=np.float64)
    gn = math.hypot(g[0], g[1])
    side = float(t @ g) / gn if gn > 1e-6 else 0.0
    if abs(side) < 0.1:
        return int(rng.choice([-1, 1]))
    return 1 if side > 0 else -1
