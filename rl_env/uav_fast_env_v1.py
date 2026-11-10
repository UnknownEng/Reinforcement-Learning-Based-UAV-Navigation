"""
uav_fast_env_v1.py
------------------
Upgraded High-Throughput Gymnasium Environment for Milestone 4 Phase 4B (Gate B1).
Integrates:
  - 26-D Unified Observation Space (18-D proprioception + 8-sector directional clearances)
  - 72-Ray Planar LiDAR Sensor Simulation with Option 1 (+2.5° half-step offset)
  - Vectorized 2D Ray-Cylinder Intersection Engine
  - Procedural Cylindrical Obstacle Registry with Multi-Stage Curriculum Support
  - Continuous Quadratic Obstacle Clearance Reward Penalty
  - Rigid Bounding Cylinder UAV Collision Detection (r_uav = 0.35 m)
  - Adherence to standard Gymnasium API and REP-103 FLU coordinates
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import gymnasium as gym
import numpy as np

from rl_env.action import ActionHandler, CommandedVelocities
from rl_env.config import UAVEnvConfig
from rl_env.dynamics import FastUAVDynamics, UAVState
from rl_env.observation_26d import (
    ObservationHandler26D,
    Observation26DMetadata,
    SectorStatus,
    InvalidClearancePolicy
)
from rl_env.termination import TerminationChecker, TerminationStatus


@dataclass
class UAVObstacleEnvConfig(UAVEnvConfig):
    """Configuration for UAVFastNav-v1 with procedural obstacles and 360° LiDAR."""
    # --- Procedural Obstacle Geometry & Distribution ---
    default_num_obstacles: int = 4
    min_obstacle_radius: float = 0.25      # Minimum cylinder radius (meters)
    max_obstacle_radius: float = 0.60      # Maximum cylinder radius (meters)
    min_spawn_clearance: float = 2.0       # Minimum distance from UAV spawn to obstacle (m)
    min_goal_clearance: float = 1.5        # Minimum distance from Goal to obstacle (m)
    min_inter_obstacle_dist: float = 1.5   # Center-to-center minimum clearance between obstacles (m)
    adversarial_prob: float = 0.6          # Probability of placing obstacle directly on path

    # --- Vehicle Physical Footprint ---
    uav_radius: float = 0.35               # Bounding collision radius (X500 frame + propellers) (m)

    # --- 360° Planar LiDAR Specification (Option 1 Half-Step Offset) ---
    lidar_num_rays: int = 72
    lidar_min_angle_rad: float = math.radians(-177.5)   # -177.5 deg
    lidar_max_angle_rad: float = math.radians(177.5)    # +177.5 deg
    lidar_angle_increment_rad: float = math.radians(5.0)  # 5.0 deg
    lidar_min_range_m: float = 0.10
    lidar_max_range_m: float = 10.0
    lidar_default_clear_m: float = 10.0

    # --- Obstacle Reward Weights ---
    reward_clearance_weight: float = 0.50  # Quadratic clearance penalty multiplier
    clearance_safe_dist: float = 1.50      # Distance threshold below which clearance is penalized (m)
    reward_obstacle_collision_penalty: float = 50.0  # Terminal penalty for obstacle crash


@dataclass
class ObstacleRewardComponents:
    progress: float
    success: float
    clearance_penalty: float
    smoothness_penalty: float
    boundary_penalty: float
    time_penalty: float
    total: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "reward_progress": float(self.progress),
            "reward_success": float(self.success),
            "penalty_clearance": float(self.clearance_penalty),
            "penalty_smoothness": float(self.smoothness_penalty),
            "penalty_boundary": float(self.boundary_penalty),
            "penalty_time": float(self.time_penalty),
            "reward_total": float(self.total),
        }


RAY_BODY_ANGLES_72 = (math.radians(-177.5) + np.arange(72, dtype=np.float32) * math.radians(5.0)).astype(np.float32)


def fast_ray_cast_cylinders(
    uav_xy: np.ndarray,
    uav_yaw: float,
    obstacles: Optional[Sequence[Tuple[float, float, float]]] = None,
    centers: Optional[np.ndarray] = None,
    radii: Optional[np.ndarray] = None,
    num_rays: int = 72,
    min_angle: float = math.radians(-177.5),
    angle_inc: float = math.radians(5.0),
    min_range: float = 0.10,
    max_range: float = 10.0
) -> np.ndarray:
    """
    Vectorized 2D ray-cylinder intersection calculation across all LiDAR beams.
    Ray origin: uav_xy (world frame).
    Ray angles: uav_yaw + (min_angle + i * angle_inc) (world frame).
    Returns:
        np.ndarray of shape (num_rays,) with range distances in [min_range, max_range].
        Unobstructed rays return max_range (10.0 m).
    """
    if num_rays == 72 and abs(min_angle - math.radians(-177.5)) < 1e-5 and abs(angle_inc - math.radians(5.0)) < 1e-5:
        ray_world_angles = uav_yaw + RAY_BODY_ANGLES_72
    else:
        ray_world_angles = uav_yaw + (min_angle + np.arange(num_rays, dtype=np.float32) * angle_inc)

    cos_a = np.cos(ray_world_angles)
    sin_a = np.sin(ray_world_angles)

    if centers is None:
        if obstacles is None or len(obstacles) == 0:
            return np.full(num_rays, max_range, dtype=np.float32)
        centers = np.array([obs[:2] for obs in obstacles], dtype=np.float32)
        radii = np.array([obs[2] for obs in obstacles], dtype=np.float32)
    elif len(centers) == 0:
        return np.full(num_rays, max_range, dtype=np.float32)

    rel_x = centers[:, 0] - uav_xy[0]
    rel_y = centers[:, 1] - uav_xy[1]

    # Vectorized outer dot product: b = cos(theta)*rel_x + sin(theta)*rel_y
    b = np.outer(cos_a, rel_x) + np.outer(sin_a, rel_y)
    rel_sq = rel_x**2 + rel_y**2
    disc = b**2 - (rel_sq[None, :] - radii[None, :]**2)

    valid_disc = disc >= 0.0
    sqrt_disc = np.sqrt(np.maximum(0.0, disc))
    t1 = b - sqrt_disc
    t2 = b + sqrt_disc

    hits = np.where(
        valid_disc & (t1 > 0.0),
        t1,
        np.where(valid_disc & (t2 > 0.0), min_range, np.inf)
    )
    raw_ranges = np.min(hits, axis=1)

    return np.clip(raw_ranges, min_range, max_range).astype(np.float32)


def generate_procedural_obstacles(
    rng: np.random.Generator,
    num_obstacles: int,
    spawn_xy: np.ndarray,
    goal_xy: np.ndarray,
    arena_x: Tuple[float, float],
    arena_y: Tuple[float, float],
    min_radius: float = 0.25,
    max_radius: float = 0.60,
    min_spawn_clearance: float = 2.0,
    min_goal_clearance: float = 1.5,
    min_inter_dist: float = 1.5,
    adversarial_prob: float = 0.6
) -> List[Tuple[float, float, float]]:
    """
    Generates procedural vertical cylinders respecting spawn, goal, and inter-obstacle separation.
    Optionally places an adversarial obstacle directly along the flight corridor.
    """
    if num_obstacles <= 0:
        return []

    obstacles: List[Tuple[float, float, float]] = []
    margin = 1.5
    ax_min, ax_max = arena_x[0] + margin, arena_x[1] - margin
    ay_min, ay_max = arena_y[0] + margin, arena_y[1] - margin

    # 1. Adversarial On-Path Placement
    if adversarial_prob > 0.0 and rng.random() < adversarial_prob:
        path_vec = goal_xy - spawn_xy
        path_len = float(np.linalg.norm(path_vec))
        if path_len > (min_spawn_clearance + min_goal_clearance + 0.5):
            path_dir = path_vec / path_len
            perp_dir = np.array([-path_dir[1], path_dir[0]])
            # Sample along path fraction
            frac = float(rng.uniform(0.35, 0.65))
            lat_jitter = float(rng.uniform(-0.35, 0.35))
            adv_pos = spawn_xy + frac * path_vec + lat_jitter * perp_dir
            adv_r = float(rng.uniform(min_radius, min_radius + 0.10))

            # Verify bounds
            if ax_min <= adv_pos[0] <= ax_max and ay_min <= adv_pos[1] <= ay_max:
                d_spawn = float(np.linalg.norm(adv_pos - spawn_xy))
                d_goal = float(np.linalg.norm(adv_pos - goal_xy))
                if d_spawn >= (min_spawn_clearance + adv_r) and d_goal >= (min_goal_clearance + adv_r):
                    obstacles.append((float(adv_pos[0]), float(adv_pos[1]), adv_r))

    # 2. General Randomized Cylinder Placement
    max_attempts = 100
    attempts = 0
    while len(obstacles) < num_obstacles and attempts < max_attempts:
        attempts += 1
        cx = float(rng.uniform(ax_min, ax_max))
        cy = float(rng.uniform(ay_min, ay_max))
        r = float(rng.uniform(min_radius, max_radius))
        c_pos = np.array([cx, cy])

        # Check separation from spawn and goal
        if np.linalg.norm(c_pos - spawn_xy) < (min_spawn_clearance + r):
            continue
        if np.linalg.norm(c_pos - goal_xy) < (min_goal_clearance + r):
            continue

        # Check inter-obstacle separation
        too_close = False
        for ocx, ocy, orad in obstacles:
            if np.linalg.norm(c_pos - np.array([ocx, ocy])) < (min_inter_dist + r + orad):
                too_close = True
                break
        if too_close:
            continue

        obstacles.append((cx, cy, r))

    return obstacles


class UAVFastEnvV1(gym.Env):
    """
    UAVFastNav-v1: 26-D Procedural Obstacle Gymnasium Environment for PPO Retraining.
    Observations (26D):
      [0:3]   Relative goal position in body FLU frame (m)
      [3:6]   Linear velocity in body FLU frame (m/s)
      [6:9]   Angular rates in body FLU frame (rad/s)
      [9:13]  Unit attitude quaternion [qw, qx, qy, qz]
      [13]    Relative altitude above ground datum (m)
      [14:18] Previous normalized action [-1, 1]^4
      [18:26] 8-sector directional obstacle clearances (m)
    Actions (4D):
      [ax, ay, az, ayaw] in [-1, +1]^4 (body forward, lateral, vertical, yaw rate)
    """

    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 10
    }

    def __init__(
        self,
        config: Optional[UAVObstacleEnvConfig] = None,
        render_mode: Optional[str] = None
    ):
        super().__init__()
        self.config = config if config is not None else UAVObstacleEnvConfig()
        self.render_mode = render_mode

        # Core handlers
        self.action_handler = ActionHandler(self.config)
        self.obs_handler = ObservationHandler26D(
            self.config,
            invalid_policy=InvalidClearancePolicy.CONSERVATIVE_ZERO,
            staleness_timeout_s=0.20,
            min_range=self.config.lidar_min_range_m,
            max_range=self.config.lidar_max_range_m,
            default_clear_range=self.config.lidar_default_clear_m
        )
        self.dynamics = FastUAVDynamics(self.config)
        self.term_checker = TerminationChecker(self.config)

        # Standard Gymnasium spaces
        self.action_space = self.action_handler.action_space
        self.observation_space = self.obs_handler.observation_space

        # Episode tracking states
        self.step_count: int = 0
        self.goal_world: np.ndarray = np.zeros(3, dtype=np.float32)
        self.prev_action: np.ndarray = np.zeros(4, dtype=np.float32)
        self.prev_dist: float = 0.0
        self.initial_dist: float = 0.0
        self.last_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self.path_length: float = 0.0
        self.cumulative_reward: float = 0.0

        # Obstacles and LiDAR state
        self.obstacles: List[Tuple[float, float, float]] = []
        self.last_clearances: np.ndarray = np.full(8, self.config.lidar_default_clear_m, dtype=np.float32)
        self.last_raw_ranges: np.ndarray = np.full(self.config.lidar_num_rays, self.config.lidar_default_clear_m, dtype=np.float32)
        self.last_meta: Optional[Observation26DMetadata] = None

        # Curriculum tracking
        self.curriculum_stage: int = 0

    def set_curriculum_stage(self, stage: int) -> None:
        """Sets active curriculum stage in [0, 3]."""
        if stage not in (0, 1, 2, 3):
            raise ValueError(f"Invalid curriculum stage: {stage}. Must be in [0, 1, 2, 3].")
        self.curriculum_stage = stage

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment for a new episode.
        Options dictionary supports:
          - 'start_pos': [x, y, z]
          - 'start_yaw': float
          - 'goal_pos': [gx, gy, gz]
          - 'curriculum_stage': int in [0, 3] (0: free space, 1: single obstacle, 2: corridor, 3: dense)
          - 'num_obstacles': int
          - 'obstacles': list of (cx, cy, r) tuples
        """
        super().reset(seed=seed)
        self.dynamics.rng = self.np_random

        opts = options or {}

        # 1. Spawn UAV position and orientation
        if "start_pos" in opts:
            start_pos = np.array(opts["start_pos"], dtype=np.float32)
            x0, y0, z0 = float(start_pos[0]), float(start_pos[1]), float(start_pos[2])
        else:
            x0 = float(self.np_random.uniform(self.config.start_pos_xy_range[0], self.config.start_pos_xy_range[1]))
            y0 = float(self.np_random.uniform(self.config.start_pos_xy_range[0], self.config.start_pos_xy_range[1]))
            z0 = float(self.config.start_altitude)

        if "start_yaw" in opts:
            yaw0 = float(opts["start_yaw"])
        elif self.config.randomize_start_yaw:
            yaw0 = float(self.np_random.uniform(-np.pi, np.pi))
        else:
            yaw0 = 0.0

        state = self.dynamics.reset(x0=x0, y0=y0, z0=z0, yaw0=yaw0)

        # 2. Spawn 3D Navigation Goal
        if "goal_pos" in opts:
            self.goal_world = np.array(opts["goal_pos"], dtype=np.float32)
        else:
            goal_dist = float(self.np_random.uniform(self.config.min_goal_dist, self.config.max_goal_dist))
            goal_azimuth = float(self.np_random.uniform(-np.pi, np.pi))
            gx = x0 + goal_dist * np.cos(goal_azimuth)
            gy = y0 + goal_dist * np.sin(goal_azimuth)

            margin = 1.0
            gx = float(np.clip(gx, self.config.arena_x[0] + margin, self.config.arena_x[1] - margin))
            gy = float(np.clip(gy, self.config.arena_y[0] + margin, self.config.arena_y[1] - margin))
            gz = float(self.np_random.uniform(self.config.goal_altitude_range[0], self.config.goal_altitude_range[1]))
            self.goal_world = np.array([gx, gy, gz], dtype=np.float32)

        # 3. Configure Procedural Obstacles
        if "obstacles" in opts:
            self.obstacles = [tuple(obs) for obs in opts["obstacles"]]  # type: ignore
        else:
            stage = opts.get("curriculum_stage", self.curriculum_stage)
            if "num_obstacles" in opts:
                n_obs = int(opts["num_obstacles"])
            elif stage == 0:
                n_obs = 0
            elif stage == 1:
                n_obs = 1
            elif stage == 2:
                n_obs = int(self.np_random.integers(3, 5))
            else:
                n_obs = int(self.np_random.integers(6, 9))

            adv_p = self.config.adversarial_prob if stage >= 1 else 0.0
            self.obstacles = generate_procedural_obstacles(
                rng=self.np_random,
                num_obstacles=n_obs,
                spawn_xy=np.array([x0, y0], dtype=np.float32),
                goal_xy=self.goal_world[:2],
                arena_x=self.config.arena_x,
                arena_y=self.config.arena_y,
                min_radius=self.config.min_obstacle_radius,
                max_radius=self.config.max_obstacle_radius,
                min_spawn_clearance=self.config.min_spawn_clearance,
                min_goal_clearance=self.config.min_goal_clearance,
                min_inter_dist=self.config.min_inter_obstacle_dist,
                adversarial_prob=adv_p
            )

        # Cache obstacle arrays for vectorized ray casting and collision checks
        if len(self.obstacles) > 0:
            self.obstacle_centers = np.array([obs[:2] for obs in self.obstacles], dtype=np.float32)
            self.obstacle_radii = np.array([obs[2] for obs in self.obstacles], dtype=np.float32)
        else:
            self.obstacle_centers = np.empty((0, 2), dtype=np.float32)
            self.obstacle_radii = np.empty(0, dtype=np.float32)

        # 4. Reset internal tracking
        self.step_count = 0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.last_pos = state.pos_world.copy()
        self.path_length = 0.0
        self.cumulative_reward = 0.0

        dist = float(np.linalg.norm(state.pos_world - self.goal_world))
        self.prev_dist = dist
        self.initial_dist = dist

        # 5. Execute Initial 72-Ray LiDAR Scan
        obs_state = self.dynamics.get_noisy_state() if self.config.enable_noise else state
        raw_ranges = fast_ray_cast_cylinders(
            uav_xy=obs_state.pos_world[:2],
            uav_yaw=obs_state.yaw,
            centers=self.obstacle_centers,
            radii=self.obstacle_radii,
            num_rays=self.config.lidar_num_rays,
            min_angle=self.config.lidar_min_angle_rad,
            angle_inc=self.config.lidar_angle_increment_rad,
            min_range=self.config.lidar_min_range_m,
            max_range=self.config.lidar_max_range_m
        )
        self.last_raw_ranges = raw_ranges.copy()

        # 6. Compute 26-D Unified Observation
        obs, meta = self.obs_handler.compute_observation_with_metadata(
            pos_world=obs_state.pos_world,
            vel_world=obs_state.vel_world,
            quat=obs_state.quat,
            rates_body=obs_state.body_rates,
            goal_world=self.goal_world,
            prev_action=self.prev_action,
            raw_scan=(raw_ranges, self.config.lidar_min_angle_rad, self.config.lidar_angle_increment_rad),
            ground_z=0.0
        )
        self.last_clearances = np.array(meta.sanitized_clearances, dtype=np.float32)
        self.last_meta = meta

        # 7. Check Initial Collision Status
        _, _, min_surf_dist = self.check_collision(
            state.pos_world[:2],
            clearances=self.last_clearances
        )

        info = {
            "step": 0,
            "pos_world": state.pos_world.copy(),
            "goal_world": self.goal_world.copy(),
            "dist_to_goal": dist,
            "obstacles": list(self.obstacles),
            "obstacle_count": len(self.obstacles),
            "clearances_m": self.last_clearances.copy(),
            "min_clearance_m": float(np.min(self.last_clearances)),
            "min_surface_dist_m": min_surf_dist,
            "sector_status": [s.value for s in meta.sector_status],
            "validity_mask": [bool(v) for v in meta.validity_mask],
        }

        return obs, info

    def check_collision(
        self,
        uav_pos_xy: np.ndarray,
        obstacles: Optional[Sequence[Tuple[float, float, float]]] = None,
        clearances: Optional[np.ndarray] = None
    ) -> Tuple[bool, str, float]:
        """
        Evaluates collision against cylindrical obstacles.
        Collision criteria:
          1. Physical bounding penetration: distance from UAV center to cylinder surface <= uav_radius (0.35m)
          2. Measured clearance proximity: any clearance sector <= uav_radius (0.35m)
        Returns:
          (is_collision, reason, min_surface_dist)
        """
        if hasattr(self, "obstacle_centers") and len(self.obstacle_centers) > 0:
            centers = self.obstacle_centers
            radii = self.obstacle_radii
        elif obstacles is not None and len(obstacles) > 0:
            centers = np.array([obs[:2] for obs in obstacles], dtype=np.float32)
            radii = np.array([obs[2] for obs in obstacles], dtype=np.float32)
        else:
            return False, "", float("inf")

        center_dists = np.linalg.norm(centers - uav_pos_xy[None, :2], axis=1)
        surface_dists = center_dists - radii
        min_surface_dist = float(np.min(surface_dists))

        if min_surface_dist <= (self.config.uav_radius + 1e-5):
            return True, "obstacle_surface_penetration", min_surface_dist

        if clearances is not None and len(clearances) > 0:
            min_clearance = float(np.min(clearances))
            if min_clearance <= self.config.uav_radius:
                return True, "obstacle_clearance_breach", min_surface_dist

        return False, "", min_surface_dist

    def compute_obstacle_reward(
        self,
        dist_prev: float,
        dist_curr: float,
        action_curr: np.ndarray,
        action_prev: np.ndarray,
        clearances: np.ndarray,
        is_success: bool,
        is_out_of_bounds: bool,
        is_ground_crash: bool,
        is_obstacle_collision: bool
    ) -> ObstacleRewardComponents:
        """
        Computes decomposed reward incorporating quadratic clearance penalty.
        R = R_prog + R_succ - R_clear - R_smooth - R_bound - R_time
        """
        # 1. Progress reward
        delta_d = dist_prev - dist_curr
        progress_reward = float(self.config.reward_progress_weight * delta_d)

        # 2. Goal success terminal bonus
        success_reward = float(self.config.reward_success if is_success else 0.0)

        # 3. Obstacle Clearance Continuous Quadratic Penalty:
        #    R_clear = -w_clear * sum(max(0, (d_safe - d_i) / d_safe)^2)
        d_safe = self.config.clearance_safe_dist
        clearance_deficit = np.maximum(0.0, (d_safe - clearances) / d_safe)
        clearance_penalty = float(self.config.reward_clearance_weight * np.sum(clearance_deficit**2))

        # 4. Action smoothness penalty: ||a_t - a_{t-1}||^2
        action_diff = action_curr - action_prev
        smoothness_penalty = float(self.config.reward_action_smoothness * np.sum(np.square(action_diff)))

        # 5. Boundary / Crash terminal penalty
        boundary_penalty = 0.0
        if is_obstacle_collision:
            boundary_penalty = float(self.config.reward_obstacle_collision_penalty)
        elif is_out_of_bounds or is_ground_crash:
            boundary_penalty = float(self.config.reward_collision_penalty)

        # 6. Step time penalty
        time_penalty = float(self.config.reward_time_penalty)

        # Total reward calculation
        total = progress_reward + success_reward - clearance_penalty - smoothness_penalty - boundary_penalty - time_penalty

        if not np.isfinite(total):
            total = 0.0

        return ObstacleRewardComponents(
            progress=progress_reward,
            success=success_reward,
            clearance_penalty=clearance_penalty,
            smoothness_penalty=smoothness_penalty,
            boundary_penalty=boundary_penalty,
            time_penalty=time_penalty,
            total=total
        )

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one policy control step (0.1s at 10 Hz).
        """
        self.step_count += 1

        # 1. Action Scaling & Clamping
        cmd = self.action_handler.scale_action(action)
        act_norm = np.clip(np.asarray(action, dtype=np.float32).flatten(), -1.0, 1.0)

        # 2. Advance Physics via 100 Hz Sub-stepping
        state = self.dynamics.step(cmd)
        step_disp = float(np.linalg.norm(state.pos_world - self.last_pos))
        self.path_length += step_disp
        self.last_pos = state.pos_world.copy()

        # 3. Distance & Navigation state
        dist_curr = float(np.linalg.norm(state.pos_world - self.goal_world))

        # 4. Execute 72-Ray Planar LiDAR Scan
        obs_state = self.dynamics.get_noisy_state() if self.config.enable_noise else state
        raw_ranges = fast_ray_cast_cylinders(
            uav_xy=obs_state.pos_world[:2],
            uav_yaw=obs_state.yaw,
            centers=self.obstacle_centers,
            radii=self.obstacle_radii,
            num_rays=self.config.lidar_num_rays,
            min_angle=self.config.lidar_min_angle_rad,
            angle_inc=self.config.lidar_angle_increment_rad,
            min_range=self.config.lidar_min_range_m,
            max_range=self.config.lidar_max_range_m
        )
        self.last_raw_ranges = raw_ranges.copy()

        # 5. Compute Unified 26-D Observation
        obs, meta = self.obs_handler.compute_observation_with_metadata(
            pos_world=obs_state.pos_world,
            vel_world=obs_state.vel_world,
            quat=obs_state.quat,
            rates_body=obs_state.body_rates,
            goal_world=self.goal_world,
            prev_action=act_norm,
            raw_scan=(raw_ranges, self.config.lidar_min_angle_rad, self.config.lidar_angle_increment_rad),
            ground_z=0.0
        )
        self.last_clearances = np.array(meta.sanitized_clearances, dtype=np.float32)
        self.last_meta = meta

        # 6. Collision Checks
        is_obs_collision, col_reason, min_surf_dist = self.check_collision(
            state.pos_world[:2],
            self.obstacles,
            clearances=self.last_clearances
        )

        # 7. Check Standard Terminations (Goal, Geofence, Ground, Tilt, Timeout)
        term_status: TerminationStatus = self.term_checker.check(
            state=state,
            dist_to_goal=dist_curr,
            step_count=self.step_count
        )

        # If obstacle collision occurred, override termination
        if is_obs_collision:
            term_status.terminated = True
            term_status.is_crashed = True
            term_status.reason = f"obstacle_collision:{col_reason}"

        # 8. Compute Reward
        reward_comp = self.compute_obstacle_reward(
            dist_prev=self.prev_dist,
            dist_curr=dist_curr,
            action_curr=act_norm,
            action_prev=self.prev_action,
            clearances=self.last_clearances,
            is_success=term_status.is_success,
            is_out_of_bounds=term_status.is_out_of_bounds,
            is_ground_crash=(term_status.is_crashed and not is_obs_collision),
            is_obstacle_collision=is_obs_collision
        )
        self.cumulative_reward += reward_comp.total

        # 9. Update History
        self.prev_action = act_norm.copy()
        self.prev_dist = dist_curr

        # 10. Diagnostic info dictionary
        info: Dict[str, Any] = {
            "step": self.step_count,
            "dist_to_goal": dist_curr,
            "pos_world": state.pos_world.copy(),
            "vel_world": state.vel_world.copy(),
            "goal_world": self.goal_world.copy(),
            "roll_deg": float(np.degrees(state.roll)),
            "pitch_deg": float(np.degrees(state.pitch)),
            "yaw_deg": float(np.degrees(state.yaw)),
            "obstacles": list(self.obstacles),
            "obstacle_count": len(self.obstacles),
            "clearances_m": self.last_clearances.copy(),
            "min_clearance_m": float(np.min(self.last_clearances)),
            "min_surface_dist_m": min_surf_dist,
            "sector_status": [s.value for s in meta.sector_status],
            "validity_mask": [bool(v) for v in meta.validity_mask],
            "is_success": term_status.is_success,
            "is_crashed": term_status.is_crashed,
            "is_obstacle_collision": is_obs_collision,
            "is_out_of_bounds": term_status.is_out_of_bounds,
            "termination_reason": term_status.reason,
            "path_length": self.path_length,
            "cumulative_reward": self.cumulative_reward,
            **reward_comp.to_dict()
        }

        if term_status.terminated or term_status.truncated:
            info["terminal_observation"] = obs.copy()
            info["final_distance"] = dist_curr

        return obs, reward_comp.total, term_status.terminated, term_status.truncated, info

    def render(self) -> Optional[str]:
        if self.render_mode == "ansi":
            state = self.dynamics.get_state()
            min_c = float(np.min(self.last_clearances))
            s = (
                f"[Step {self.step_count:3d}] "
                f"Pos: [{state.pos_world[0]:5.2f}, {state.pos_world[1]:5.2f}, {state.pos_world[2]:5.2f}] m | "
                f"Goal: [{self.goal_world[0]:5.2f}, {self.goal_world[1]:5.2f}, {self.goal_world[2]:5.2f}] m | "
                f"Dist: {self.prev_dist:5.2f} m | "
                f"Min Clear: {min_c:5.2f} m | "
                f"Obs Count: {len(self.obstacles)} | "
                f"Reward: {self.cumulative_reward:6.1f}"
            )
            return s
        return None

    def close(self):
        pass
