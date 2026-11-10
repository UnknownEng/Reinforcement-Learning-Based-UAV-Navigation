"""
rrt_star.py
-----------
Global sampling-based planning baseline for UAVWorldNav-v0 (Milestone 9):
RRT* (Karaman & Frazzoli 2011) and Informed RRT* (Gammell, Srinivasa & Barfoot 2014) in 3-D,
shortcut smoothing, and a pure-pursuit path follower that replans when the path is blocked.

Information used -- GLOBAL MAP KNOWLEDGE (privileged; allowed for this baseline only):
  * the static obstacle map env.world (World3D) and the goal position env.goal_world;
  * the vehicle position, estimated as goal_world - R obs[0:3], i.e. the same (noisy) navigation
    estimate the observation is built from;
  * onboard LiDAR (env.last_ranges) only to notice returns the map does not explain (moving or
    unmapped obstacles) on the path ahead, and to add them to the map before replanning.

Planner (RRTStarPlanner):
  * state space: x, y inside the world bounds (minus xy_margin), z in the altitude band [1.0, 4.5] m;
  * collision checking: world.distance(p + hull offset) > r_uav + margin at points sampled every
    edge_resolution (0.1 m) along each edge, i.e. for the collision-hull centre the env checks
    (env.config.hull_center_dz above the planned reference point); the threshold is raised by half
    the sample spacing so the check also holds between samples (distance fields are 1-Lipschitz);
  * goal bias, steering step, rewiring radius min(gamma (log n / n)^(1/3), rewire_max) with gamma
    from the free-volume formula, lazy choose-parent, batched rewiring checks, cost propagation;
  * informed=True: once a solution of cost c_best exists, samples are drawn uniformly from the
    prolate spheroid {x : |x - start| + |x - goal| <= c_best} (Informed RRT*);
  * budget: max_iters and max_time_s (whichever comes first); stops early when the solution is
    within near_optimal_tol of the straight-line lower bound;
  * smoothing: greedy farthest-visible shortcutting, then random shortcuts between points on the path.
Follower (RRTStarController.act): pure pursuit on the waypoint path with a speed-dependent
lookahead, corner and arrival speed limits, heading-frame velocity commands, yaw toward the
lookahead point; vertical takeoff to the band first. Replans (rate limited) when unexplained LiDAR
returns block the path ahead or the vehicle deviates from the path; hovers while blocked.
"""

from dataclasses import dataclass, replace
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from baselines.common import (EpisodeTracker, VehicleLimits, frame_from_obs, hull_center_dz, make_config,
                              onboard_points, split_overrides, unwrap, velocity_to_action, yaw_rate_towards)
from rl_env import raycast_jit as _jit
from rl_env.lidar3d import DEFAULT_SPEC
from rl_env.world3d import World3D


@dataclass
class RRTStarConfig:
    uav_radius: float = 0.35
    margin: float = 0.25                 # clearance threshold = uav_radius + margin
    edge_resolution: float = 0.1         # m between collision samples along an edge
    z_band: Tuple[float, float] = (1.0, 4.5)
    xy_margin: float = 1.0               # stay this far inside the world bounds
    step_size: float = 1.5               # steering distance (m)
    goal_bias: float = 0.1
    rewire_max: float = 3.0              # cap on the rewiring radius (m)
    gamma_factor: float = 1.2            # gamma = factor * gamma_RRT*
    max_iters: int = 1500
    max_time_s: float = 3.0
    near_optimal_tol: float = 0.01
    informed: bool = False
    shortcut_iters: int = 60
    hull_dz: float = 0.0                 # collision hull centre above the planned (reference) point
    seed: Optional[int] = None


@dataclass
class PlanResult:
    success: bool
    path: np.ndarray                     # smoothed waypoints (K, 3), start .. goal
    raw_path: np.ndarray                 # tree path before smoothing
    raw_cost: float
    cost: float
    iterations: int
    n_nodes: int
    time_s: float
    first_solution_iter: int = -1
    first_solution_cost: float = math.inf
    direct: bool = False


def path_length(path: np.ndarray) -> float:
    p = np.asarray(path, dtype=np.float64)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


def _dedupe(points: List[np.ndarray], tol: float = 1e-6) -> np.ndarray:
    out = [np.asarray(points[0], dtype=np.float64)]
    for p in points[1:]:
        if np.linalg.norm(p - out[-1]) > tol:
            out.append(np.asarray(p, dtype=np.float64))
    return np.array(out)


class RRTStarPlanner:
    """RRT* / Informed RRT* over a World3D map. Uses GLOBAL MAP KNOWLEDGE (see module docstring)."""

    def __init__(self, world: World3D, config: Optional[RRTStarConfig] = None, **overrides):
        self.cfg = c = make_config(RRTStarConfig, config, **overrides)
        self.thr = c.uav_radius + c.margin
        h = world.half_size - c.xy_margin
        self.lo = np.array([-h, -h, c.z_band[0]], dtype=np.float64)
        self.hi = np.array([h, h, c.z_band[1]], dtype=np.float64)
        self.hull = np.array([0.0, 0.0, float(c.hull_dz)])
        pad = self.thr + 1.0
        z_lo, z_hi = c.z_band[0] + c.hull_dz - pad, c.z_band[1] + c.hull_dz + pad
        self.world = world.local(np.zeros(2), 2.0 * world.half_size, (z_lo, z_hi))
        # World3D.distance with the culled arrays converted once (it re-converts them on every call)
        self._arrays = self.world._arrays() if _jit.HAVE_NUMBA else None
        self.rng = np.random.default_rng(c.seed)
        vol = float(np.prod(self.hi - self.lo))
        self.gamma = c.gamma_factor * 2.0 * (4.0 / 3.0) ** (1.0 / 3.0) * (vol / (4.0 / 3.0 * math.pi)) ** (1.0 / 3.0)
        self.n_checked_points = 0

    # ------------------------------------------------------- collision checks
    def _distance(self, pts: np.ndarray) -> np.ndarray:
        """world.distance of the hull centres of the planned points pts (P, 3)."""
        q = np.ascontiguousarray(pts + self.hull, dtype=np.float64)
        self.n_checked_points += len(q)
        return _jit.distance(q, *self._arrays) if self._arrays is not None else self.world.distance(q)

    def clearance(self, points: np.ndarray) -> np.ndarray:
        """Hull-centre clearance (distance to the nearest obstacle surface) at planned points."""
        return self._distance(np.atleast_2d(np.asarray(points, dtype=np.float64)))

    def edges_free(self, A: np.ndarray, B: np.ndarray, thr: Optional[float] = None) -> np.ndarray:
        """
        Collision-free flags (E,) of the straight edges A[i] -> B[i] (A may be a single point): hull
        clearance > thr (default r_uav + margin) at samples every edge_resolution, plus half the
        sample spacing so the bound also holds between samples.
        """
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        res = self.cfg.edge_resolution
        lim = (self.thr if thr is None else thr) + 0.5 * res
        if A.ndim == 1 and B.ndim == 1:                                      # single-edge fast path
            n = max(int(math.ceil(math.sqrt(float((B - A) @ (B - A))) / res)) + 1, 2)
            pts = A + np.linspace(0.0, 1.0, n)[:, None] * (B - A)
            return np.array([self._distance(pts).min() > lim])
        A, B = np.atleast_2d(A), np.atleast_2d(B)
        if len(A) == 1 and len(B) > 1:
            A = np.repeat(A, len(B), axis=0)
        seg = B - A
        n = np.maximum(np.ceil(np.linalg.norm(seg, axis=1) / res).astype(np.int64) + 1, 2)
        offs = np.zeros(len(n), dtype=np.int64)
        offs[1:] = np.cumsum(n)[:-1]
        eid = np.repeat(np.arange(len(n)), n)
        s = (np.arange(int(n.sum())) - offs[eid]) / (n[eid] - 1)
        pts = A[eid] + s[:, None] * seg[eid]
        return np.minimum.reduceat(self._distance(pts), offs) > lim

    def _free_anchor(self, p: np.ndarray, max_shift: float = 1.5) -> Optional[np.ndarray]:
        """p itself if it is clear, else the closest clear point within max_shift (inside the bounds)."""
        need = self.thr + self.cfg.edge_resolution
        if self.clearance(p)[0] > need:
            return p.copy()
        ang = np.radians(np.arange(0, 360, 30))
        cand = [p + np.array([r * math.cos(a), r * math.sin(a), dz]) for r in np.arange(0.25, max_shift + 1e-9, 0.25)
                for dz in (0.0, 0.3, -0.3) for a in ang]
        cand = np.array([q for q in cand if np.all(q >= self.lo) and np.all(q <= self.hi)]).reshape(-1, 3)
        if not len(cand):
            return None
        ok = self.clearance(cand) > need
        if not np.any(ok):
            return None
        cand = cand[ok]
        return cand[np.argmin(np.linalg.norm(cand - p, axis=1))]

    # -------------------------------------------------------------- sampling
    def _radius(self, n: int) -> float:
        if n < 2:
            return self.cfg.rewire_max
        return min(self.gamma * (math.log(n) / n) ** (1.0 / 3.0), self.cfg.rewire_max)

    def _setup_ellipsoid(self, start: np.ndarray, goal: np.ndarray) -> None:
        self._c_min = float(np.linalg.norm(goal - start))
        self._center = 0.5 * (start + goal)
        a1 = (goal - start) / max(self._c_min, 1e-9)
        helper = np.array([0.0, 0.0, 1.0]) if abs(a1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        b2 = np.cross(a1, helper)
        b2 /= np.linalg.norm(b2)
        self._C = np.column_stack([a1, b2, np.cross(a1, b2)])

    def _sample_informed(self, c_best: float) -> Optional[np.ndarray]:
        r1 = 0.5 * c_best
        r2 = 0.5 * math.sqrt(max(c_best * c_best - self._c_min * self._c_min, 0.0))
        radii = np.array([r1, r2, r2])
        for _ in range(30):
            x = self.rng.normal(size=3)
            x *= self.rng.random() ** (1.0 / 3.0) / max(np.linalg.norm(x), 1e-12)
            p = self._C @ (radii * x) + self._center
            if np.all(p >= self.lo) and np.all(p <= self.hi):
                return p
        return None

    def _sample(self, it: int, goal: np.ndarray, c_best: float) -> np.ndarray:
        u = self._u[it]                                    # pre-drawn uniforms: 3 position + 1 goal bias
        if self._goal_idx < 0 and u[3] < self.cfg.goal_bias:
            return goal.copy()
        if self.cfg.informed and math.isfinite(c_best):
            p = self._sample_informed(c_best)
            if p is not None:
                return p
        return self.lo + u[:3] * (self.hi - self.lo)

    # ------------------------------------------------------------------ tree
    def _insert(self, x: np.ndarray, i_near: int) -> int:
        """Add x (edge from i_near already checked): choose the best parent, then rewire the neighbours."""
        n = self._n
        nodes, cost, parent, children = self._nodes, self._cost, self._parent, self._children
        dn = np.sqrt(np.einsum("ij,ij->i", nodes[:n] - x, nodes[:n] - x))
        near = np.nonzero(dn <= self._radius(n))[0]
        best_p, best_c = i_near, cost[i_near] + dn[i_near]
        if len(near):
            c_via = cost[near] + dn[near]
            for j in np.argsort(c_via):                       # lazy: cheapest first, stop at first free
                if c_via[j] >= best_c - 1e-9:
                    break
                if self.edges_free(nodes[near[j]], x)[0]:
                    best_p, best_c = int(near[j]), float(c_via[j])
                    break
        new = n
        nodes[new], parent[new], cost[new] = x, best_p, best_c
        children[best_p].append(new)
        self._n = n + 1
        if len(near):
            cand = near[(best_c + dn[near] < cost[near] - 1e-9) & (near != best_p)]
            if len(cand):
                free = self.edges_free(x, nodes[cand])
                for idx in cand[free]:
                    new_c = best_c + dn[idx]
                    if new_c < cost[idx] - 1e-9:
                        children[parent[idx]].remove(idx)
                        parent[idx] = new
                        children[new].append(int(idx))
                        delta = cost[idx] - new_c
                        stack = [int(idx)]
                        while stack:                           # propagate the cost decrease
                            k = stack.pop()
                            cost[k] -= delta
                            stack.extend(children[k])
        return new

    def _tree_path(self, idx: int) -> np.ndarray:
        out = []
        while idx >= 0:
            out.append(self._nodes[idx].copy())
            idx = int(self._parent[idx])
        return np.array(out[::-1])

    # ------------------------------------------------------------- planning
    def plan(self, start: np.ndarray, goal: np.ndarray, max_iters: Optional[int] = None,
             max_time_s: Optional[float] = None) -> PlanResult:
        """Plan from start to goal (3-D, world frame). The returned path starts at start, ends at goal."""
        c = self.cfg
        t0 = time.perf_counter()
        max_iters = c.max_iters if max_iters is None else int(max_iters)
        max_time_s = c.max_time_s if max_time_s is None else float(max_time_s)
        start_true = np.asarray(start, dtype=np.float64).copy()
        goal_true = np.asarray(goal, dtype=np.float64).copy()
        s = self._free_anchor(np.clip(start_true, self.lo, self.hi))
        g = self._free_anchor(np.clip(goal_true, self.lo, self.hi))
        empty = np.zeros((0, 3))

        def finish(raw: np.ndarray, ok: bool, iters: int, direct: bool = False) -> PlanResult:
            if not ok:
                return PlanResult(False, empty, empty, math.inf, math.inf, iters, getattr(self, "_n", 0),
                                  time.perf_counter() - t0, self._first_iter, self._first_cost)
            smooth = self.shortcut(raw)
            full = _dedupe([start_true] + list(smooth) + [goal_true])
            raw_full = _dedupe([start_true] + list(raw) + [goal_true])
            return PlanResult(True, full, raw_full, path_length(raw_full), path_length(full), iters,
                              getattr(self, "_n", 0), time.perf_counter() - t0, self._first_iter,
                              self._first_cost, direct)

        self._first_iter, self._first_cost, self._goal_idx, self._n = -1, math.inf, -1, 0
        if s is None or g is None:
            return finish(empty, False, 0)
        self._setup_ellipsoid(s, g)
        if self._c_min < 1e-9 or self.edges_free(s, g)[0]:
            self._first_iter, self._first_cost = 0, self._c_min
            return finish(np.array([s, g]), True, 0, direct=True)

        cap = max_iters + 2
        self._nodes = np.empty((cap, 3))
        self._parent = np.full(cap, -1, dtype=np.int64)
        self._cost = np.zeros(cap)
        self._children: List[List[int]] = [[] for _ in range(cap)]
        self._nodes[0] = s
        self._n = 1
        self._u = self.rng.random((max_iters + 1, 4))
        it = 0
        for it in range(1, max_iters + 1):
            if (it & 7) == 0 and time.perf_counter() - t0 > max_time_s:
                break
            c_best = self._cost[self._goal_idx] if self._goal_idx >= 0 else math.inf
            if c_best <= (1.0 + c.near_optimal_tol) * self._c_min:
                break
            x_rand = self._sample(it, g, c_best)
            diff = self._nodes[:self._n] - x_rand
            d2 = np.einsum("ij,ij->i", diff, diff)
            i_near = int(np.argmin(d2))
            d = math.sqrt(d2[i_near])
            if d < 1e-9 or self._n >= cap - 1:
                continue
            x_near = self._nodes[i_near]
            x_new = x_rand if d <= c.step_size else x_near + (x_rand - x_near) * (c.step_size / d)
            is_goal = self._goal_idx < 0 and np.linalg.norm(x_new - g) < 1e-9
            if not self.edges_free(self._nodes[i_near], x_new)[0]:
                continue
            new = self._insert(x_new, i_near)
            if is_goal:
                self._goal_idx = new
            elif self._goal_idx < 0 and np.linalg.norm(g - x_new) <= c.step_size and self._n < cap \
                    and self.edges_free(x_new, g)[0]:
                self._goal_idx = self._insert(g.copy(), new)
            if self._goal_idx >= 0 and self._first_iter < 0:
                self._first_iter, self._first_cost = it, float(self._cost[self._goal_idx])
        if self._goal_idx < 0:
            return finish(empty, False, it)
        return finish(self._tree_path(self._goal_idx), True, it)

    # ------------------------------------------------------------ smoothing
    def shortcut(self, path: np.ndarray, iters: Optional[int] = None) -> np.ndarray:
        """Greedy farthest-visible shortcutting, then random shortcuts between points on the path."""
        P = [np.asarray(p, dtype=np.float64) for p in path]
        if len(P) <= 2:
            return np.array(P)
        out, i = [P[0]], 0
        while i < len(P) - 1:
            cands = np.arange(len(P) - 1, i, -1)
            free = self.edges_free(P[i], np.array([P[j] for j in cands]))
            j = int(cands[np.argmax(free)]) if np.any(free) else i + 1
            out.append(P[j])
            i = j
        for _ in range(self.cfg.shortcut_iters if iters is None else iters):
            if len(out) < 3:
                break
            seg = np.linalg.norm(np.diff(np.array(out), axis=0), axis=1)
            cum = np.concatenate([[0.0], np.cumsum(seg)])
            s1, s2 = np.sort(self.rng.uniform(0.0, cum[-1], size=2))
            i1 = min(int(np.searchsorted(cum, s1, side="right")) - 1, len(out) - 2)
            i2 = min(int(np.searchsorted(cum, s2, side="right")) - 1, len(out) - 2)
            if i1 == i2 or s2 - s1 < 0.2:
                continue
            q1 = out[i1] + (s1 - cum[i1]) / max(seg[i1], 1e-12) * (out[i1 + 1] - out[i1])
            q2 = out[i2] + (s2 - cum[i2]) / max(seg[i2], 1e-12) * (out[i2 + 1] - out[i2])
            if self.edges_free(q1, q2)[0]:
                out = out[:i1 + 1] + [q1, q2] + out[i2 + 1:]
        return _dedupe(out, tol=1e-3)


def plan_path(world: World3D, start, goal, informed: bool = False, **overrides) -> PlanResult:
    """Convenience wrapper: plan once with (Informed) RRT* on `world`."""
    return RRTStarPlanner(world, informed=informed, **overrides).plan(np.asarray(start), np.asarray(goal))


# ------------------------------------------------------------------- follower
class PurePursuit:
    """Arc-length bookkeeping and lookahead on a 3-D polyline."""

    def __init__(self, path: np.ndarray):
        self.path = _dedupe(list(np.asarray(path, dtype=np.float64)))
        if len(self.path) == 1:
            self.path = np.vstack([self.path, self.path + 1e-3])
        self.seg = np.diff(self.path, axis=0)
        self.seg_len = np.linalg.norm(self.seg, axis=1)
        self.cum = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.total = float(self.cum[-1])
        self.s = 0.0

    def point_at(self, s: float) -> np.ndarray:
        s = float(np.clip(s, 0.0, self.total))
        k = min(max(int(np.searchsorted(self.cum, s, side="right")) - 1, 0), len(self.seg) - 1)
        return self.path[k] + (s - self.cum[k]) / max(self.seg_len[k], 1e-12) * self.seg[k]

    def project(self, pos: np.ndarray, back: float = 1.0, ahead: float = 4.0) -> Tuple[float, float]:
        """Closest point within a window around the current progress; updates and returns (s, distance)."""
        idx = np.nonzero((self.cum[1:] >= self.s - back) & (self.cum[:-1] <= self.s + ahead))[0]
        if not len(idx):
            idx = np.arange(len(self.seg))
        A, d, L = self.path[idx], self.seg[idx], self.seg_len[idx]
        t = np.clip(np.einsum("ij,ij->i", pos - A, d) / np.maximum(L * L, 1e-12), 0.0, 1.0)
        dist = np.linalg.norm(A + t[:, None] * d - pos, axis=1)
        k = int(np.argmin(dist))
        self.s = float(self.cum[idx[k]] + t[k] * L[k])
        return self.s, float(dist[k])

    def next_corner(self) -> Tuple[float, float]:
        """(distance to the next vertex, turn angle there in rad); (inf, 0) on the last segment."""
        k = min(max(int(np.searchsorted(self.cum, self.s, side="right")) - 1, 0), len(self.seg) - 1)
        if k >= len(self.seg) - 1:
            return math.inf, 0.0
        a, b = self.seg[k] / max(self.seg_len[k], 1e-12), self.seg[k + 1] / max(self.seg_len[k + 1], 1e-12)
        return float(self.cum[k + 1] - self.s), float(math.acos(float(np.clip(a @ b, -1.0, 1.0))))

    def sample_ahead(self, skip: float, length: float, step: float = 0.2) -> np.ndarray:
        s = np.arange(self.s + skip, min(self.s + skip + length, self.total) + 1e-9, step)
        return np.array([self.point_at(v) for v in s]).reshape(-1, 3)


@dataclass
class FollowerConfig:
    lookahead_min: float = 1.0
    lookahead_time: float = 0.5          # lookahead = min + time * speed
    lookahead_max: float = 2.5
    corner_lookahead: float = 0.4        # the lookahead point may run at most this far past the next vertex
    k_arrive: float = 0.9                # 1/s, arrival P gain
    a_dec: float = 1.5                   # m/s^2, speed <= sqrt(2 a_dec s_remaining)
    corner_speed_min: float = 1.0
    corner_slow_gain: float = 1.0        # extra speed allowed per metre before a corner (1/s)
    k_yaw: float = 1.5
    takeoff_alt: float = 1.3             # first waypoint above the start inside the band
    replan: bool = True
    block_check_dist: float = 4.0        # look this far along the path for blocking returns (m)
    block_clearance: float = 0.5         # an unexplained return closer than this to the path blocks it
    block_skip: float = 0.3
    unexplained_dist: float = 0.2        # a return farther than this from the map is not in the map
    deviation_replan: float = 1.5
    replan_interval_steps: int = 10
    replan_max_iters: int = 800
    replan_time_s: float = 1.0
    sensed_range: float = 6.0
    sensed_voxel: float = 0.1
    sensed_radius: float = 0.1           # sensed return -> sphere of this radius + half the LiDAR ring gap
                                         # at its range (the surface between rings is not observed)
    ground_margin: float = 0.15


class RRTStarController:
    """
    (Informed) RRT* planner + pure-pursuit follower as a policy: `act(obs, env) -> action (4,)`.
    Uses GLOBAL MAP KNOWLEDGE (env.world, env.goal_world); `env` is required.
    The current waypoint path (K, 3) is in `.path`; planning statistics in `.plan_result`.
    """

    uses_map = True

    def __init__(self, env=None, informed: Optional[bool] = None, planner_config: Optional[RRTStarConfig] = None,
                 follower_config: Optional[FollowerConfig] = None, **overrides):
        p_over, f_over = split_overrides(overrides, RRTStarConfig, FollowerConfig)
        self.planner_cfg = make_config(RRTStarConfig, planner_config, **p_over)
        if informed is not None:
            self.planner_cfg = replace(self.planner_cfg, informed=bool(informed))
        self.cfg = make_config(FollowerConfig, follower_config, **f_over)
        self.name = "informed_rrt_star" if self.planner_cfg.informed else "rrt_star"
        self.limits = VehicleLimits.from_env(env)
        self.hull_dz = hull_center_dz(env)
        self._tracker = EpisodeTracker()
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        self._seed = self.planner_cfg.seed if seed is None else seed
        self.follower: Optional[PurePursuit] = None
        self.path: Optional[np.ndarray] = None
        self.plan_result: Optional[PlanResult] = None
        self.plan_time_s = 0.0
        self.n_plans = 0
        self.n_replans = 0
        self.n_plan_failures = 0
        self._since_plan = 10 ** 9
        self._budget_scale = 1.0
        self.last_info: Dict[str, Any] = {}
        self._fresh = True          # reset() called since the last act()

    # ------------------------------------------------------------ planning
    def _takeoff_point(self, planner: RRTStarPlanner, start: np.ndarray) -> np.ndarray:
        lo, hi = self.planner_cfg.z_band
        z0 = float(np.clip(max(self.cfg.takeoff_alt, lo), lo, hi))
        for z in (z0, z0 + 0.4, z0 + 0.8, lo + 0.1):
            p = np.array([start[0], start[1], min(z, hi)])
            if planner.clearance(p)[0] > planner.thr + 0.15:
                return p
        return np.array([start[0], start[1], z0])

    def _unexplained_returns(self, base, P_world: np.ndarray) -> np.ndarray:
        if not len(P_world):
            return P_world
        return P_world[base.world.distance(P_world) > self.cfg.unexplained_dist]

    def plan(self, env, start: np.ndarray, sensed: Optional[np.ndarray] = None) -> PlanResult:
        """(Re)plan from `start` to env.goal_world on env.world (+ sensed returns as spheres)."""
        base = unwrap(env)
        c = self.cfg
        world = base.world
        if sensed is not None and len(sensed):
            keys = np.floor(sensed / c.sensed_voxel).astype(np.int64)
            _, first = np.unique(keys, axis=0, return_index=True)
            pts = sensed[np.sort(first)]
            spec = getattr(base, "lidar", DEFAULT_SPEC)
            el = np.radians(np.asarray(spec.elevations_deg, dtype=np.float64))
            gap = float(np.min(np.diff(el))) if len(el) > 1 else 0.0
            sensor = np.asarray(start, dtype=np.float64) + np.array([0.0, 0.0, spec.mount_offset_body[2]])
            radius = c.sensed_radius + 0.5 * np.linalg.norm(pts - sensor, axis=1) * math.tan(gap)
            world = world.with_extra(sph=np.column_stack([pts, radius]))
        seed = None if self._seed is None else int(self._seed) * 1000 + self.n_plans
        cfg = replace(self.planner_cfg, seed=seed, hull_dz=self.hull_dz)
        if self.n_plans > 0:
            cfg = replace(cfg, max_iters=c.replan_max_iters, max_time_s=c.replan_time_s)
        cfg = replace(cfg, max_iters=int(cfg.max_iters * self._budget_scale),
                      max_time_s=cfg.max_time_s * self._budget_scale)
        t0 = time.perf_counter()
        planner = RRTStarPlanner(world, cfg)
        start = np.asarray(start, dtype=np.float64)
        prefix = []
        if start[2] < cfg.z_band[0] - 0.05:
            prefix = [start]
            start = self._takeoff_point(planner, start)
        res = planner.plan(start, np.asarray(base.goal_world, dtype=np.float64))
        self.plan_time_s += time.perf_counter() - t0
        self.n_plans += 1
        self._since_plan = 0
        if res.success:
            path = _dedupe(prefix + list(res.path))
            self.path, self.plan_result = path, res
            self.follower = PurePursuit(path)
            self._budget_scale = 1.0
        else:
            self.n_plan_failures += 1
            self._budget_scale = min(self._budget_scale * 2.0, 4.0)
        return res

    # ------------------------------------------------------------ sensing
    def _path_blocked(self, base, P_world: np.ndarray, hull_pos: np.ndarray) -> bool:
        """Is an unmapped return closer than block_clearance to the hull-centre path ahead?"""
        c = self.cfg
        if self.follower is None or not len(P_world):
            return False
        Q = self.follower.sample_ahead(c.block_skip, c.block_check_dist)
        if not len(Q):
            return False
        Q = Q + np.array([0.0, 0.0, self.hull_dz])
        rel = P_world - hull_pos
        P = P_world[np.einsum("ij,ij->i", rel, rel) < (c.block_check_dist + c.block_skip + 1.0) ** 2]
        if not len(P):
            return False
        d2 = ((P[:, None, :] - Q[None, :, :]) ** 2).sum(axis=-1).min(axis=1)
        close = P[d2 < c.block_clearance ** 2]
        return bool(len(close)) and bool(len(self._unexplained_returns(base, close)))

    # ----------------------------------------------------------------- act
    def act(self, obs: np.ndarray, env=None) -> np.ndarray:
        base = unwrap(env)
        if base is None or getattr(base, "world", None) is None:
            raise ValueError("RRTStarController needs the env (it plans on env.world)")
        self.hull_dz = hull_center_dz(env)
        if self._tracker.new_episode(env) and not self._fresh:         # env reset without reset()
            self.reset()
        self._fresh = False
        c, lim = self.cfg, self.limits
        obs = np.asarray(obs, dtype=np.float64)
        fr = frame_from_obs(obs)
        goal = np.asarray(base.goal_world, dtype=np.float64)
        pos = goal - fr.R @ obs[0:3]
        self._since_plan += 1
        can_plan = self._since_plan >= c.replan_interval_steps
        if self.follower is None and can_plan:
            self.plan(env, pos)
        if self.follower is None:
            return velocity_to_action(np.zeros(3), 0.0, fr, lim)

        _, dev = self.follower.project(pos)
        blocked = False
        if c.replan:
            pts, valid, _ = onboard_points(obs, env, R=fr.R, ground_margin=c.ground_margin, hull_dz=self.hull_dz)
            hull_pos = pos + np.array([0.0, 0.0, self.hull_dz])
            P_world = hull_pos + pts[valid]                                  # where the returns are
            blocked = self._path_blocked(base, P_world, hull_pos)
            if (blocked or dev > c.deviation_replan) and can_plan:
                rel = P_world - hull_pos
                sensed = P_world[np.einsum("ij,ij->i", rel, rel) < c.sensed_range ** 2]
                res = self.plan(env, pos, self._unexplained_returns(base, sensed))
                self.n_replans += 1
                if res.success:
                    _, dev = self.follower.project(pos)
                    blocked = self._path_blocked(base, P_world, hull_pos)
        if blocked:
            self.last_info = {"blocked": True, "deviation": dev}
            return velocity_to_action(np.zeros(3), 0.0, fr, lim)

        v_meas = float(np.linalg.norm(obs[3:6]))
        pp = self.follower
        look = float(np.clip(c.lookahead_min + c.lookahead_time * v_meas, c.lookahead_min, c.lookahead_max))
        s_corner, ang = pp.next_corner()
        if math.isfinite(s_corner):                    # do not cut the corner (its vertex hugs an obstacle)
            look = max(min(look, s_corner + c.corner_lookahead), 0.3)
        target = pp.point_at(pp.s + look)
        to_t = target - pos
        d_t = float(np.linalg.norm(to_t))
        s_eff = max(pp.total - pp.s, float(np.linalg.norm(goal - pos)))
        speed = min(lim.v_xy, c.k_arrive * s_eff, math.sqrt(2.0 * c.a_dec * s_eff))
        if math.isfinite(s_corner):
            v_corner = max(c.corner_speed_min, lim.v_xy * (1.0 - ang / math.pi))
            speed = min(speed, v_corner + c.corner_slow_gain * s_corner)
        v = to_t / d_t * speed if d_t > 1e-6 else np.zeros(3)
        yaw_rate = yaw_rate_towards(to_t, fr, lim, c.k_yaw, min_dist=0.3)
        self.last_info = {"blocked": False, "deviation": dev, "s": pp.s, "s_total": pp.total, "v_cmd": v}
        return velocity_to_action(v, yaw_rate, fr, lim)
