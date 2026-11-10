"""
geometry_validator.py
---------------------
Deterministic 2D Route Feasibility and Geometry Validation Engine for Fast UAV Navigation.

Features:
- Obstacle inflation by UAV bounding radius (r_uav = 0.35m) plus safety margin (m_safe = 0.15m).
- 8-connected grid Breadth-First Search (BFS) in inflated free-space.
- Full arena boundary containment and boundary clearance verification.
- Continuous path clearance evaluation along the entire discovered trajectory.
- Procedural layout resampling with explicit exhaustion bounds.

DISCLAIMER:
2D connectivity does not establish dynamic flight feasibility. While 2D BFS in inflated free-space
guarantees the geometric existence of a collision-free corridor of width >= 2*(r_uav + margin),
actual flight success depends on UAV acceleration limits, actuation lag, velocity saturation,
attitude tilt limits, and closed-loop control tracking error.
"""

from dataclasses import dataclass, field
from collections import deque
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class ValidationResult:
    """Detailed outcome of a geometric route validation query."""
    is_traversable: bool
    reason: str
    min_route_clearance: float
    route_coords: Optional[np.ndarray] = None
    route_length: float = 0.0
    grid_cells_searched: int = 0
    attempts: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_traversable": bool(self.is_traversable),
            "reason": str(self.reason),
            "min_route_clearance": float(self.min_route_clearance),
            "route_length": float(self.route_length),
            "grid_cells_searched": int(self.grid_cells_searched),
            "attempts": int(self.attempts),
        }


class RouteGeometryValidator:
    """
    Deterministic 2D BFS Route Feasibility Validator.
    
    DISCLAIMER:
    2D connectivity does not establish dynamic flight feasibility. While 2D BFS in inflated free-space
    guarantees the geometric existence of a collision-free corridor of width >= 2*(r_uav + margin),
    actual flight success depends on UAV acceleration limits, actuation lag, velocity saturation,
    attitude tilt limits, and closed-loop control tracking error.
    """

    def __init__(
        self,
        arena_x: Tuple[float, float] = (-12.0, 12.0),
        arena_y: Tuple[float, float] = (-12.0, 12.0),
        uav_radius: float = 0.35,
        safety_margin: float = 0.10,
        grid_resolution: float = 0.10,
        min_path_clearance: float = 0.45,
        max_resample_attempts: int = 50
    ):
        self.arena_x = arena_x
        self.arena_y = arena_y
        self.uav_radius = float(uav_radius)
        self.safety_margin = float(safety_margin)
        self.r_inflate = self.uav_radius + self.safety_margin  # 0.45 m
        self.grid_resolution = float(grid_resolution)
        self.min_path_clearance = float(min_path_clearance)
        self.max_resample_attempts = int(max_resample_attempts)

    def validate_route(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        obstacles: Sequence[Tuple[float, float, float]]
    ) -> ValidationResult:
        """
        Validates whether a traversable route exists between start_xy and goal_xy
        in the presence of cylindrical obstacles and arena boundaries.
        
        Steps:
        1. Checks start and goal boundary containment (inflated by r_inflate).
        2. Checks start and goal clearance from obstacles (inflated by r_inflate).
        3. Constructs a local bounded 2D occupancy grid with obstacle & boundary inflation.
        4. Executes 8-connected BFS from start to goal.
        5. Computes Euclidean surface clearance along the discovered path.
        """
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])

        # 1. Boundary containment check for start and goal
        ax_min, ax_max = self.arena_x[0] + self.r_inflate, self.arena_x[1] - self.r_inflate
        ay_min, ay_max = self.arena_y[0] + self.r_inflate, self.arena_y[1] - self.r_inflate

        if not (ax_min <= sx <= ax_max and ay_min <= sy <= ay_max):
            return ValidationResult(
                is_traversable=False,
                reason="start_boundary_violation",
                min_route_clearance=0.0
            )

        if not (ax_min <= gx <= ax_max and ay_min <= gy <= ay_max):
            return ValidationResult(
                is_traversable=False,
                reason="goal_boundary_violation",
                min_route_clearance=0.0
            )

        # 2. Obstacle collision check for start and goal
        for ox, oy, orad in obstacles:
            d_start = math.hypot(sx - ox, sy - oy)
            if d_start < (orad + self.r_inflate):
                return ValidationResult(
                    is_traversable=False,
                    reason=f"start_obstructed_by_obstacle(d={d_start:.3f}<{orad+self.r_inflate:.3f})",
                    min_route_clearance=0.0
                )
            d_goal = math.hypot(gx - ox, gy - oy)
            if d_goal < (orad + self.r_inflate):
                return ValidationResult(
                    is_traversable=False,
                    reason=f"goal_obstructed_by_obstacle(d={d_goal:.3f}<{orad+self.r_inflate:.3f})",
                    min_route_clearance=0.0
                )

        # 3. Determine search bounding box
        # Include start, goal, and all obstacles with margins
        all_x = [sx, gx] + [obs[0] - obs[2] - self.r_inflate for obs in obstacles] + [obs[0] + obs[2] + self.r_inflate for obs in obstacles]
        all_y = [sy, gy] + [obs[1] - obs[2] - self.r_inflate for obs in obstacles] + [obs[1] + obs[2] + self.r_inflate for obs in obstacles]

        margin = 1.5
        box_x_min = max(self.arena_x[0], min(all_x) - margin)
        box_x_max = min(self.arena_x[1], max(all_x) + margin)
        box_y_min = max(self.arena_y[0], min(all_y) - margin)
        box_y_max = min(self.arena_y[1], max(all_y) + margin)

        res = self.grid_resolution
        nx = int(math.ceil((box_x_max - box_x_min) / res))
        ny = int(math.ceil((box_y_max - box_y_min) / res))

        # Discrete coordinates for grid
        xs = box_x_min + (np.arange(nx, dtype=np.float32) + 0.5) * res
        ys = box_y_min + (np.arange(ny, dtype=np.float32) + 0.5) * res
        X, Y = np.meshgrid(xs, ys, indexing='ij')

        # 4. Occupancy mask (True = blocked / impassable)
        blocked = np.zeros((nx, ny), dtype=bool)

        # Arena boundary inflation
        blocked[X < (self.arena_x[0] + self.r_inflate)] = True
        blocked[X > (self.arena_x[1] - self.r_inflate)] = True
        blocked[Y < (self.arena_y[0] + self.r_inflate)] = True
        blocked[Y > (self.arena_y[1] - self.r_inflate)] = True

        # Obstacle inflation
        for ox, oy, orad in obstacles:
            inflated_r_sq = (orad + self.r_inflate) ** 2
            dist_sq = (X - ox) ** 2 + (Y - oy) ** 2
            blocked[dist_sq <= inflated_r_sq] = True

        # Discretize start and goal to grid coordinates
        s_ix = int(np.clip(round((sx - box_x_min) / res - 0.5), 0, nx - 1))
        s_iy = int(np.clip(round((sy - box_y_min) / res - 0.5), 0, ny - 1))
        g_ix = int(np.clip(round((gx - box_x_min) / res - 0.5), 0, nx - 1))
        g_iy = int(np.clip(round((gy - box_y_min) / res - 0.5), 0, ny - 1))

        if blocked[s_ix, s_iy]:
            return ValidationResult(
                is_traversable=False,
                reason="start_grid_cell_blocked",
                min_route_clearance=0.0
            )

        if blocked[g_ix, g_iy]:
            return ValidationResult(
                is_traversable=False,
                reason="goal_grid_cell_blocked",
                min_route_clearance=0.0
            )

        # 5. 8-connected BFS in flat integer array
        n_total = nx * ny
        visited = blocked.flatten().copy()  # blocked cells are marked as already visited

        start_flat = s_ix * ny + s_iy
        goal_flat = g_ix * ny + g_iy

        visited[start_flat] = True
        parent = np.full(n_total, -1, dtype=np.int32)

        queue = deque([start_flat])
        found = False
        cells_searched = 0

        # Neighbor movements (8-connectivity)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while queue:
            curr = queue.popleft()
            cells_searched += 1

            if curr == goal_flat:
                found = True
                break

            cx = curr // ny
            cy = curr % ny

            for dx, dy in moves:
                ncx = cx + dx
                ncy = cy + dy
                if 0 <= ncx < nx and 0 <= ncy < ny:
                    # Diagonal corner-cutting prevention:
                    # Do not cut diagonally between or past orthogonally adjacent blocked cells
                    if dx != 0 and dy != 0:
                        if blocked[ncx, cy] or blocked[cx, ncy]:
                            continue

                    nidx = ncx * ny + ncy
                    if not visited[nidx]:
                        visited[nidx] = True
                        parent[nidx] = curr
                        queue.append(nidx)

        if not found:
            return ValidationResult(
                is_traversable=False,
                reason="no_connected_route",
                min_route_clearance=0.0,
                grid_cells_searched=cells_searched
            )

        # 6. Reconstruct path
        path_indices = []
        curr = goal_flat
        while curr != -1:
            path_indices.append(curr)
            curr = parent[curr]
        path_indices.reverse()

        # Convert path indices to metric coordinates
        path_coords = np.zeros((len(path_indices), 2), dtype=np.float32)
        for idx_step, flat_idx in enumerate(path_indices):
            c_x = flat_idx // ny
            c_y = flat_idx % ny
            path_coords[idx_step, 0] = xs[c_x]
            path_coords[idx_step, 1] = ys[c_y]

        # Calculate path length
        diffs = np.diff(path_coords, axis=0)
        route_length = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))

        # 7. Validate minimum clearance along entire continuous route segments
        min_clearance = float("inf")
        num_pts = len(path_coords)

        if num_pts == 1:
            pt = path_coords[0]
            d_obs = float("inf")
            for ox, oy, orad in obstacles:
                surf_d = math.hypot(pt[0] - ox, pt[1] - oy) - orad
                if surf_d < d_obs:
                    d_obs = surf_d
            d_bound = min(
                pt[0] - self.arena_x[0],
                self.arena_x[1] - pt[0],
                pt[1] - self.arena_y[0],
                self.arena_y[1] - pt[1]
            )
            min_clearance = min(d_obs, d_bound)
        else:
            for k in range(num_pts - 1):
                p1 = path_coords[k]
                p2 = path_coords[k + 1]
                v = p2 - p1
                l2 = float(v[0] * v[0] + v[1] * v[1])

                # Continuous obstacle surface clearance along line segment [p1, p2]
                d_obs_seg = float("inf")
                for ox, oy, orad in obstacles:
                    if l2 < 1e-8:
                        dist_center = math.hypot(p1[0] - ox, p1[1] - oy)
                    else:
                        t = max(0.0, min(1.0, float((ox - p1[0]) * v[0] + (oy - p1[1]) * v[1]) / l2))
                        closest_x = p1[0] + t * v[0]
                        closest_y = p1[1] + t * v[1]
                        dist_center = math.hypot(closest_x - ox, closest_y - oy)

                    surf_d = dist_center - orad
                    if surf_d < d_obs_seg:
                        d_obs_seg = surf_d

                # Continuous boundary clearance along line segment [p1, p2]
                d_bound_seg = min(
                    min(p1[0], p2[0]) - self.arena_x[0],
                    self.arena_x[1] - max(p1[0], p2[0]),
                    min(p1[1], p2[1]) - self.arena_y[0],
                    self.arena_y[1] - max(p1[1], p2[1])
                )

                c_seg = min(d_obs_seg, d_bound_seg)
                if c_seg < min_clearance:
                    min_clearance = c_seg

        if min_clearance < self.min_path_clearance:
            return ValidationResult(
                is_traversable=False,
                reason=f"clearance_insufficient({min_clearance:.3f}<{self.min_path_clearance:.3f})",
                min_route_clearance=min_clearance,
                route_coords=path_coords,
                route_length=route_length,
                grid_cells_searched=cells_searched
            )

        return ValidationResult(
            is_traversable=True,
            reason="valid",
            min_route_clearance=min_clearance,
            route_coords=path_coords,
            route_length=route_length,
            grid_cells_searched=cells_searched
        )

    def validate_and_generate_obstacles(
        self,
        generator_fn: Callable[[], List[Tuple[float, float, float]]],
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        raise_on_exhaustion: bool = True
    ) -> Tuple[List[Tuple[float, float, float]], ValidationResult]:
        """
        Repeatedly invokes generator_fn until a layout with a verified traversable route is produced,
        or until max_resample_attempts is reached.
        
        If max_resample_attempts is exceeded and raise_on_exhaustion is True, raises RuntimeError.
        Otherwise returns the last generated layout and failed ValidationResult.
        """
        last_result = ValidationResult(is_traversable=False, reason="uninitialized", min_route_clearance=0.0)
        last_obstacles: List[Tuple[float, float, float]] = []

        for attempt in range(1, self.max_resample_attempts + 1):
            obstacles = generator_fn()
            res = self.validate_route(start_xy, goal_xy, obstacles)
            res.attempts = attempt
            if res.is_traversable:
                return obstacles, res
            last_result = res
            last_obstacles = obstacles

        if raise_on_exhaustion:
            raise RuntimeError(
                f"Resampling exhaustion: failed to generate a traversable layout after "
                f"{self.max_resample_attempts} attempts. Last reason: {last_result.reason}"
            )

        return last_obstacles, last_result
