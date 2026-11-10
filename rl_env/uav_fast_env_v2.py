"""
uav_fast_env_v2.py
------------------
Generalization and Robust Autonomous Navigation Environment for Milestone 5 (Phase M5.4).

Integrates:
  - 26-D Unified Observation Space (strictly preserved from Milestone 4)
  - 4-D Continuous Action Space (strictly preserved from Milestone 4)
  - Smooth C^∞ Heading Alignment Reward and Smooth Speed Gating Penalty
  - Deterministic 2D BFS Route Feasibility & Clearance Validator
  - Procedural Multi-Stage Curriculum Navigation with Verified Traversability
  - Closed-loop multicopter dynamics and 360° Planar LiDAR Sensor Simulation

DISCLAIMER:
2D connectivity does not establish dynamic flight feasibility. While 2D BFS in inflated free-space
guarantees the geometric existence of a collision-free corridor of width >= 2*(r_uav + margin),
actual flight success depends on UAV acceleration limits, actuation lag, velocity saturation,
attitude tilt limits, and closed-loop control tracking error.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import gymnasium as gym
import numpy as np

from rl_env.action import ActionHandler, CommandedVelocities
from rl_env.config import UAVEnvConfig
from rl_env.dynamics import FastUAVDynamics, UAVState
from rl_env.geometry_validator import RouteGeometryValidator, ValidationResult
from rl_env.heading_reward import compute_smooth_heading_reward
from rl_env.observation_26d import (
    ObservationHandler26D,
    Observation26DMetadata,
    SectorStatus,
    InvalidClearancePolicy
)
from rl_env.termination import TerminationChecker, TerminationStatus
from rl_env.uav_fast_env_v1 import (
    UAVFastEnvV1,
    UAVObstacleEnvConfig,
    fast_ray_cast_cylinders
)


@dataclass
class UAVObstacleEnvConfigV2(UAVObstacleEnvConfig):
    """Configuration for UAVFastNav-v2 with smooth heading reward and geometry validation."""
    # --- Heading Alignment and Speed Gating Weights ---
    reward_align_weight: float = 0.10          # Smooth heading alignment reward multiplier
    reward_speed_gate_weight: float = 0.10     # Smooth speed penalty multiplier under misalignment
    reward_reverse_weight: float = 0.50        # Asymmetric reverse speed penalty multiplier under misalignment
    reward_action_divergence_weight: float = 0.10 # Smooth quadratic action divergence penalty multiplier

    # --- Geometry Validation & Route Feasibility Parameters ---
    min_path_clearance: float = 0.45           # Minimum clearance (m) required along validated route (0.45m)
    safety_margin: float = 0.10                # Safety margin for inflation: r_inflate = uav_radius + safety_margin = 0.45m
    grid_resolution: float = 0.10              # Grid cell size (m) for 2D BFS search
    max_resample_attempts: int = 50            # Maximum attempts to resample valid layout before exhaustion

    # --- Domain Randomization ---
    enable_domain_randomization: bool = False  # Enable physically grounded domain randomization on reset

    # --- Curriculum Stage Management ---
    curriculum_stage: int = 0                  # 0: Free space, 1: Sparse, 2: Cluttered, 3: Multi-Regime Stage 3

    # --- Regime-Aware Episode Horizon ---
    max_episode_steps_default: int = 200       # Standard episode horizon for Stage 0-2 and Stage 3 Regimes 1-2
    max_episode_steps_slalom: int = 300        # Extended horizon for Stage 3 Regime 3 (Sequential Slaloms)


@dataclass
class ObstacleRewardComponentsV2:
    progress: float
    success: float
    clearance_penalty: float
    smoothness_penalty: float
    boundary_penalty: float
    time_penalty: float
    heading_alignment: float
    speed_penalty: float
    action_divergence_penalty: float = 0.0
    stagnation_penalty: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "reward_progress": float(self.progress),
            "reward_success": float(self.success),
            "penalty_clearance": float(self.clearance_penalty),
            "penalty_smoothness": float(self.smoothness_penalty),
            "penalty_boundary": float(self.boundary_penalty),
            "penalty_time": float(self.time_penalty),
            "reward_heading_align": float(self.heading_alignment),
            "penalty_speed_gate": float(self.speed_penalty),
            "penalty_action_divergence": float(self.action_divergence_penalty),
            "penalty_stagnation": float(self.stagnation_penalty),
            "reward_total": float(self.total),
        }


class UAVFastEnvV2(UAVFastEnvV1):
    """
    UAVFastNav-v2: Advanced Generalization Environment for Milestone 5.
    
    DISCLAIMER:
    2D connectivity does not establish dynamic flight feasibility. While 2D BFS in inflated free-space
    guarantees the geometric existence of a collision-free corridor of width >= 2*(r_uav + margin),
    actual flight success depends on UAV acceleration limits, actuation lag, velocity saturation,
    attitude tilt limits, and closed-loop control tracking error.
    
    Contracts Preserved Exactly:
      Observation Space: Box(-inf, inf, shape=(26,), dtype=np.float32)
      Action Space: Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
    """

    metadata = {
        "render_modes": ["ansi"],
        "render_fps": 10
    }

    def __init__(
        self,
        config: Optional[UAVObstacleEnvConfigV2] = None,
        render_mode: Optional[str] = None
    ):
        v2_config = config if config is not None else UAVObstacleEnvConfigV2()
        super().__init__(config=v2_config, render_mode=render_mode)
        self.config: UAVObstacleEnvConfigV2 = v2_config

        # Initialize deterministic geometry validator
        self.validator = RouteGeometryValidator(
            arena_x=self.config.arena_x,
            arena_y=self.config.arena_y,
            uav_radius=self.config.uav_radius,
            safety_margin=getattr(self.config, "safety_margin", 0.10),
            grid_resolution=self.config.grid_resolution,
            min_path_clearance=self.config.min_path_clearance,
            max_resample_attempts=self.config.max_resample_attempts
        )

        # Route validation cache for current episode
        self.last_validation_result: Optional[ValidationResult] = None
        self.last_heading_error: float = 0.0
        self.curriculum_stage: int = int(self.config.curriculum_stage)

        # Observation caching and regime tracking for Milestone 5 Stage 3
        self.last_obs: Optional[np.ndarray] = None
        self.active_regime: int = 0
        self.forced_regime: Optional[int] = None
        self.active_lidar_dropout: float = 0.0
        self.active_lidar_noise: float = 0.0
        self._episode_max_steps: Optional[int] = None

    def set_curriculum_stage(self, stage: int) -> None:
        """Sets the active training curriculum stage (0, 1, 2, or 3)."""
        if stage not in (0, 1, 2, 3):
            raise ValueError(f"Invalid curriculum stage {stage}. Must be 0, 1, 2, or 3.")
        self.curriculum_stage = stage
        self.config.curriculum_stage = stage

    @property
    def current_max_episode_steps(self) -> int:
        """
        Returns the active episode step horizon based on curriculum stage and regime.
        If an episode is active, returns the immutable episode horizon locked at reset.
        """
        if getattr(self, "_episode_max_steps", None) is not None:
            return int(self._episode_max_steps)
        if self.curriculum_stage == 3 and self.active_regime == 3:
            return int(getattr(self.config, "max_episode_steps_slalom", 300))
        return int(getattr(self.config, "max_episode_steps_default", getattr(self.config, "max_episode_steps", 200)))

    def compute_obstacle_reward_v2(
        self,
        dist_prev: float,
        dist_curr: float,
        action_curr: np.ndarray,
        action_prev: np.ndarray,
        clearances: np.ndarray,
        goal_body: np.ndarray,
        vel_body: np.ndarray,
        is_success: bool,
        is_out_of_bounds: bool,
        is_ground_crash: bool,
        is_obstacle_collision: bool,
        cmd_body: Optional[np.ndarray] = None,
        raw_action: Optional[np.ndarray] = None,
        shielded_action: Optional[np.ndarray] = None
    ) -> ObstacleRewardComponentsV2:
        """
        Computes decomposed reward incorporating:
        1. Progress reward
        2. Success terminal bonus
        3. Quadratic continuous clearance penalty
        4. Action smoothness penalty
        5. Boundary / crash terminal penalty
        6. Step time penalty
        7. Smooth C^∞ heading alignment reward
        8. Smooth C^∞ speed gating penalty
        9. Smooth quadratic action divergence penalty: -w_div * ||a_raw - a_shield||^2
        """
        # 1. Progress reward
        delta_d = dist_prev - dist_curr
        progress_reward = float(self.config.reward_progress_weight * delta_d)

        # 2. Goal success terminal bonus
        success_reward = float(self.config.reward_success if is_success else 0.0)

        # 3. Obstacle Clearance Continuous Quadratic Penalty
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

        # 7 & 8. Smooth Heading Alignment Reward and Speed Gating Penalty
        heading_error = math.atan2(float(goal_body[1]), float(goal_body[0]))
        self.last_heading_error = heading_error
        cmd_vel_xy = cmd_body[:2] if cmd_body is not None else None
        r_align, p_speed, _ = compute_smooth_heading_reward(
            heading_error_rad=heading_error,
            vel_body_xy=vel_body[:2],
            cmd_vel_xy=cmd_vel_xy,
            weight_align=self.config.reward_align_weight,
            weight_speed=self.config.reward_speed_gate_weight,
            weight_reverse=self.config.reward_reverse_weight
        )

        # 9. Smooth Quadratic Action Divergence Penalty
        divergence_penalty = 0.0
        if raw_action is not None and shielded_action is not None:
            raw_a = np.asarray(raw_action, dtype=np.float32).flatten()
            shield_a = np.asarray(shielded_action, dtype=np.float32).flatten()
            divergence = float(np.sum(np.square(raw_a - shield_a)))
            divergence_penalty = float(self.config.reward_action_divergence_weight * divergence)

        # 10. Stagnation penalty for passive hovering far from goal
        speed_mag = float(np.linalg.norm(vel_body[:2]))
        stagnation_penalty = 0.0
        if speed_mag < 0.15 and dist_curr > 1.0:
            stagnation_penalty = 0.05

        # Total reward sum
        total = (progress_reward + success_reward + r_align + p_speed
                 - clearance_penalty - smoothness_penalty - boundary_penalty - time_penalty - divergence_penalty - stagnation_penalty)

        if not np.isfinite(total):
            total = 0.0

        return ObstacleRewardComponentsV2(
            progress=progress_reward,
            success=success_reward,
            clearance_penalty=clearance_penalty,
            smoothness_penalty=smoothness_penalty,
            boundary_penalty=boundary_penalty,
            time_penalty=time_penalty,
            heading_alignment=r_align,
            speed_penalty=p_speed,
            action_divergence_penalty=divergence_penalty,
            stagnation_penalty=stagnation_penalty,
            total=total
        )

    def apply_domain_randomization(self, rng: np.random.Generator) -> None:
        """
        Applies physically grounded domain randomization to dynamics and sensors
        according to Milestone 5 Stage 3 specification.
        """
        if not (self.config.enable_domain_randomization or self.curriculum_stage == 3):
            self.active_lidar_dropout = 0.0
            self.active_lidar_noise = 0.0
            return

        # 1. Actuation Dynamics: velocity-loop time constant, +/-25% around the PX4-fitted 0.60 s
        self.dynamics.tau_velocity_xy = float(rng.uniform(0.45, 0.75))

        # 2. Actuation Authority: horizontal acceleration cap in [4.0, 8.0] m/s^2 (PX4 tilt limit ~9.8)
        self.dynamics.max_horizontal_accel = float(rng.uniform(4.0, 8.0))

        # 3. Perception Sensing: LiDAR ray dropout in [0.00, 0.15] and noise in [0.01, 0.08] m
        self.active_lidar_dropout = float(rng.uniform(0.00, 0.15))
        self.active_lidar_noise = float(rng.uniform(0.01, 0.08))

        # 4. State Estimation Noise: pos_std in [0.01, 0.05]m, vel_std in [0.02, 0.08]m/s
        self.dynamics.noise_pos_std = float(rng.uniform(0.01, 0.05))
        self.dynamics.noise_vel_std = float(rng.uniform(0.02, 0.08))

        # 5. Environment Wind Disturbances in [0.05, 0.15] m/s^2
        self.dynamics.wind_disturbance_std = float(rng.uniform(0.05, 0.15))

    def _sample_legacy_obstacles(
        self,
        spawn_xy: np.ndarray,
        goal_xy: np.ndarray,
        num_obs: int,
        r_min: float,
        r_max: float,
        adv_prob: float
    ) -> List[Tuple[float, float, float]]:
        """Legacy obstacle sampler for Curriculum Stages 1 and 2."""
        obstacles: List[Tuple[float, float, float]] = []
        margin = 1.5
        ax_min, ax_max = self.config.arena_x[0] + margin, self.config.arena_x[1] - margin
        ay_min, ay_max = self.config.arena_y[0] + margin, self.config.arena_y[1] - margin

        # Adversarial placement along path
        if adv_prob > 0.0 and self.np_random.random() < adv_prob:
            path_vec = goal_xy - spawn_xy
            path_len = float(np.linalg.norm(path_vec))
            if path_len > (self.config.min_spawn_clearance + self.config.min_goal_clearance + 0.5):
                path_dir = path_vec / path_len
                perp_dir = np.array([-path_dir[1], path_dir[0]])
                frac = float(self.np_random.uniform(0.35, 0.65))
                lat_jitter = float(self.np_random.uniform(-0.4, 0.4))
                adv_pos = spawn_xy + frac * path_vec + lat_jitter * perp_dir
                adv_r = float(self.np_random.uniform(r_min, min(r_max, r_min + 0.15)))

                if ax_min <= adv_pos[0] <= ax_max and ay_min <= adv_pos[1] <= ay_max:
                    d_spawn = float(np.linalg.norm(adv_pos - spawn_xy))
                    d_goal = float(np.linalg.norm(adv_pos - goal_xy))
                    if d_spawn >= (self.config.min_spawn_clearance + adv_r) and d_goal >= (self.config.min_goal_clearance + adv_r):
                        obstacles.append((float(adv_pos[0]), float(adv_pos[1]), adv_r))

        # General placement
        max_attempts = 100
        attempts = 0
        min_inter_dist = 1.5
        while len(obstacles) < num_obs and attempts < max_attempts:
            attempts += 1
            cx = float(self.np_random.uniform(ax_min, ax_max))
            cy = float(self.np_random.uniform(ay_min, ay_max))
            r = float(self.np_random.uniform(r_min, r_max))
            c_pos = np.array([cx, cy])

            if np.linalg.norm(c_pos - spawn_xy) < (self.config.min_spawn_clearance + r):
                continue
            if np.linalg.norm(c_pos - goal_xy) < (self.config.min_goal_clearance + r):
                continue

            conflict = False
            for ox, oy, orad in obstacles:
                if np.linalg.norm(c_pos - np.array([ox, oy])) < (min_inter_dist + r + orad):
                    conflict = True
                    break
            if not conflict:
                obstacles.append((cx, cy, r))

        return obstacles

    def _generate_regime1_clutter(
        self,
        spawn_xy: np.ndarray,
        goal_xy: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """
        Regime 1: Dense Clutter Forest (35% sampling frequency).
        N_obs in [7, 10], radius in [0.20, 0.32]m, pairwise surface gap >= 1.10m.
        """
        num_obs = int(self.np_random.integers(7, 11))
        r_min, r_max = 0.20, 0.32
        d_gap_min = 1.10
        margin = 1.5
        ax_min, ax_max = self.config.arena_x[0] + margin, self.config.arena_x[1] - margin
        ay_min, ay_max = self.config.arena_y[0] + margin, self.config.arena_y[1] - margin

        obstacles: List[Tuple[float, float, float]] = []
        max_attempts = 150
        attempts = 0

        while len(obstacles) < num_obs and attempts < max_attempts:
            attempts += 1
            cx = float(self.np_random.uniform(ax_min, ax_max))
            cy = float(self.np_random.uniform(ay_min, ay_max))
            r = float(self.np_random.uniform(r_min, r_max))
            c_pos = np.array([cx, cy], dtype=np.float32)

            # Spawn and goal clearance check
            if np.linalg.norm(c_pos - spawn_xy) < (1.50 + r):
                continue
            if np.linalg.norm(c_pos - goal_xy) < (1.20 + r):
                continue

            # Pairwise surface gap >= 1.10m
            conflict = False
            for ox, oy, orad in obstacles:
                dist_centers = float(np.linalg.norm(c_pos - np.array([ox, oy])))
                surface_gap = dist_centers - r - orad
                if surface_gap < d_gap_min:
                    conflict = True
                    break
            if not conflict:
                obstacles.append((cx, cy, r))

        if len(obstacles) < 7:
            return self._fallback_regime1(spawn_xy, goal_xy)

        return obstacles

    def _generate_regime2_gates(
        self,
        spawn_xy: np.ndarray,
        goal_xy: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """
        Regime 2: Isolated Narrow Aperture Gates (35% sampling frequency).
        2 primary posts (r=0.25m), aperture width in [1.00, 1.30]m,
        tilt in [-25, +25] deg, 2 flanking guide cylinders.
        """
        r_post = 0.25
        path_vec = goal_xy - spawn_xy
        path_len = float(np.linalg.norm(path_vec))
        if path_len < 0.5:
            d_unit = np.array([1.0, 0.0], dtype=np.float32)
        else:
            d_unit = (path_vec / path_len).astype(np.float32)

        u_trans = np.array([-d_unit[1], d_unit[0]], dtype=np.float32)

        # Gate location along flight path
        d_gate = float(self.np_random.uniform(2.5, 3.8))
        d_gate = min(d_gate, max(1.8, path_len - 1.5))
        gate_center = spawn_xy + d_gate * d_unit

        # Aperture width and tilt
        w_aperture = float(self.np_random.uniform(1.00, 1.30))
        theta_tilt_deg = float(self.np_random.uniform(-15.0, 15.0))
        theta_tilt_rad = math.radians(theta_tilt_deg)

        # Rotated transverse gate axis
        u_gate = math.cos(theta_tilt_rad) * u_trans + math.sin(theta_tilt_rad) * d_unit
        u_gate = u_gate / np.linalg.norm(u_gate)

        half_span = (w_aperture / 2.0) + r_post
        c1 = gate_center + half_span * u_gate
        c2 = gate_center - half_span * u_gate

        obstacles: List[Tuple[float, float, float]] = [
            (float(c1[0]), float(c1[1]), r_post),
            (float(c2[0]), float(c2[1]), r_post)
        ]

        # Flanking guide cylinders (r = 0.30m) at lateral offset >= 2.2m from gate center
        r_flank = 0.30
        flank_dist = max(2.20, half_span + 1.20)
        flank1 = gate_center + flank_dist * u_gate
        flank2 = gate_center - flank_dist * u_gate

        ax_min, ax_max = self.config.arena_x[0] + 1.0, self.config.arena_x[1] - 1.0
        ay_min, ay_max = self.config.arena_y[0] + 1.0, self.config.arena_y[1] - 1.0

        for f_pos in [flank1, flank2]:
            if ax_min <= f_pos[0] <= ax_max and ay_min <= f_pos[1] <= ay_max:
                if (np.linalg.norm(f_pos - spawn_xy) >= (1.50 + r_flank) and
                    np.linalg.norm(f_pos - goal_xy) >= (1.20 + r_flank)):
                    obstacles.append((float(f_pos[0]), float(f_pos[1]), r_flank))

        return obstacles

    def _generate_regime3_slaloms(
        self,
        spawn_xy: np.ndarray,
        goal_xy: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """
        Regime 3: Sequential Multi-Stage Slalom Barriers (30% sampling frequency).
        2 or 3 barrier stages spaced longitudinally along flight axis,
        openings in [1.30, 1.60]m alternating left/right.
        """
        r_post = 0.25
        path_vec = goal_xy - spawn_xy
        path_len = float(np.linalg.norm(path_vec))
        if path_len < 0.5:
            d_unit = np.array([1.0, 0.0], dtype=np.float32)
        else:
            d_unit = (path_vec / path_len).astype(np.float32)

        u_trans = np.array([-d_unit[1], d_unit[0]], dtype=np.float32)

        if path_len >= 5.5:
            num_stages = int(self.np_random.choice([2, 3]))
        else:
            num_stages = 2

        if num_stages == 2:
            fracs = [float(self.np_random.uniform(0.30, 0.40)),
                     float(self.np_random.uniform(0.60, 0.70))]
        else:
            fracs = [float(self.np_random.uniform(0.25, 0.32)),
                     float(self.np_random.uniform(0.48, 0.55)),
                     float(self.np_random.uniform(0.68, 0.75))]

        obstacles: List[Tuple[float, float, float]] = []
        ax_min, ax_max = self.config.arena_x[0] + 1.0, self.config.arena_x[1] - 1.0
        ay_min, ay_max = self.config.arena_y[0] + 1.0, self.config.arena_y[1] - 1.0

        for s in range(num_stages):
            xs = fracs[s] * path_len
            w_win = float(self.np_random.uniform(1.30, 1.60))

            if s == 0:
                y_offset = float(self.np_random.uniform(1.0, 1.4))   # Left opening
            elif s == 1:
                y_offset = float(self.np_random.uniform(-1.4, -1.0)) # Right opening
            else:
                y_offset = float(self.np_random.uniform(-0.3, 0.3))   # Center opening

            win_center = spawn_xy + xs * d_unit + y_offset * u_trans
            half_win = (w_win / 2.0) + r_post

            post_left = win_center + half_win * u_trans
            post_right = win_center - half_win * u_trans

            for post in [post_left, post_right]:
                px = float(np.clip(post[0], ax_min, ax_max))
                py = float(np.clip(post[1], ay_min, ay_max))
                if (math.hypot(px - spawn_xy[0], py - spawn_xy[1]) >= (1.20 + r_post) and
                    math.hypot(px - goal_xy[0], py - goal_xy[1]) >= (1.20 + r_post)):
                    obstacles.append((px, py, r_post))

        if len(obstacles) < 4:
            return self._fallback_regime3(spawn_xy, goal_xy)

        return obstacles

    def _fallback_regime1(
        self,
        spawn_xy: np.ndarray,
        goal_xy: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """Certified deterministic fallback template for Regime 1."""
        path_vec = goal_xy - spawn_xy
        path_len = float(np.linalg.norm(path_vec))
        d_unit = path_vec / max(0.1, path_len)
        u_trans = np.array([-d_unit[1], d_unit[0]], dtype=np.float32)
        r = 0.25
        obstacles = []
        for frac, lat in [(0.25, 1.8), (0.40, -1.8), (0.55, 1.9), (0.70, -1.9),
                          (0.35, 3.2), (0.60, -3.2), (0.75, 2.0)]:
            pos = spawn_xy + (frac * path_len) * d_unit + lat * u_trans
            obstacles.append((float(pos[0]), float(pos[1]), r))
        return obstacles

    def _fallback_regime2(
        self,
        spawn_xy: np.ndarray,
        goal_xy: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """Certified deterministic fallback template for Regime 2."""
        path_vec = goal_xy - spawn_xy
        u_trans = np.array([-path_vec[1], path_vec[0]], dtype=np.float32)
        u_trans_norm = float(np.linalg.norm(u_trans))
        if u_trans_norm > 0.1:
            u_trans = u_trans / u_trans_norm
        else:
            u_trans = np.array([0.0, 1.0], dtype=np.float32)

        r_post = 0.25
        w_aperture = 1.20
        gate_center = spawn_xy + 0.50 * path_vec
        half_span = (w_aperture / 2.0) + r_post
        c1 = gate_center + half_span * u_trans
        c2 = gate_center - half_span * u_trans
        return [(float(c1[0]), float(c1[1]), r_post), (float(c2[0]), float(c2[1]), r_post)]

    def _fallback_regime3(
        self,
        spawn_xy: np.ndarray,
        goal_xy: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """Certified deterministic fallback template for Regime 3."""
        path_vec = goal_xy - spawn_xy
        u_trans = np.array([-path_vec[1], path_vec[0]], dtype=np.float32)
        u_trans_norm = float(np.linalg.norm(u_trans))
        if u_trans_norm > 0.1:
            u_trans = u_trans / u_trans_norm
        else:
            u_trans = np.array([0.0, 1.0], dtype=np.float32)

        r_post = 0.25
        w_win = 1.40
        half_win = (w_win / 2.0) + r_post

        win1 = spawn_xy + 0.35 * path_vec + 1.2 * u_trans
        win2 = spawn_xy + 0.65 * path_vec - 1.2 * u_trans

        obs = [
            (float(win1[0] + half_win * u_trans[0]), float(win1[1] + half_win * u_trans[1]), r_post),
            (float(win1[0] - half_win * u_trans[0]), float(win1[1] - half_win * u_trans[1]), r_post),
            (float(win2[0] + half_win * u_trans[0]), float(win2[1] + half_win * u_trans[1]), r_post),
            (float(win2[0] - half_win * u_trans[0]), float(win2[1] - half_win * u_trans[1]), r_post)
        ]
        return obs

    def _sample_curriculum_obstacles(
        self,
        spawn_xy: np.ndarray,
        goal_xy: np.ndarray
    ) -> List[Tuple[float, float, float]]:
        """
        Samples procedural cylinder obstacles according to the active curriculum stage:
          Stage 0: 0 obstacles (free space)
          Stage 1: 1-3 obstacles, radius 0.25-0.45m
          Stage 2: 3-5 obstacles, radius 0.25-0.60m
          Stage 3: Multi-Regime Procedural Generator:
                   - Regime 1: Dense Clutter Forest (35%)
                   - Regime 2: Isolated Narrow Aperture Gates (35%)
                   - Regime 3: Sequential Slalom Barriers (30%)
        """
        stage = self.curriculum_stage

        if stage == 0:
            self.active_regime = 0
            return []

        if stage == 1:
            self.active_regime = 0
            num_obs = int(self.np_random.integers(1, 4))
            r_min, r_max = 0.25, 0.45
            adv_prob = 0.4
            return self._sample_legacy_obstacles(spawn_xy, goal_xy, num_obs, r_min, r_max, adv_prob)
        elif stage == 2:
            self.active_regime = 0
            num_obs = int(self.np_random.integers(3, 6))
            r_min, r_max = 0.25, 0.60
            adv_prob = 0.6
            return self._sample_legacy_obstacles(spawn_xy, goal_xy, num_obs, r_min, r_max, adv_prob)
        else:  # stage == 3
            if self.forced_regime is not None and self.forced_regime in (1, 2, 3):
                regime = self.forced_regime
            else:
                u = float(self.np_random.uniform(0.0, 1.0))
                if u < 0.35:
                    regime = 1
                elif u < 0.70:
                    regime = 2
                else:
                    regime = 3

            self.active_regime = regime
            if regime == 1:
                return self._generate_regime1_clutter(spawn_xy, goal_xy)
            elif regime == 2:
                return self._generate_regime2_gates(spawn_xy, goal_xy)
            else:
                return self._generate_regime3_slaloms(spawn_xy, goal_xy)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment with deterministic route feasibility validation.
        """
        super(UAVFastEnvV1, self).reset(seed=seed)
        self.dynamics.rng = self.np_random

        opts = options or {}
        if "curriculum_stage" in opts:
            self.set_curriculum_stage(int(opts["curriculum_stage"]))
        if "enable_domain_randomization" in opts:
            self.config.enable_domain_randomization = bool(opts["enable_domain_randomization"])
        if "active_regime" in opts:
            self.forced_regime = int(opts["active_regime"])
            self.active_regime = int(opts["active_regime"])
        elif not hasattr(self, "forced_regime"):
            self.forced_regime = None

        # Apply domain randomization on episode reset
        self.apply_domain_randomization(self.np_random)

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

        # 3. Configure Procedural Obstacles with Deterministic Feasibility Validation
        spawn_xy = np.array([x0, y0], dtype=np.float32)
        goal_xy = self.goal_world[:2]

        if "obstacles" in opts:
            self.obstacles = [tuple(obs) for obs in opts["obstacles"]]
            val_res = self.validator.validate_route(spawn_xy, goal_xy, self.obstacles)
            self.last_validation_result = val_res
        else:
            # Resample until a verified traversable route exists
            def sample_fn():
                return self._sample_curriculum_obstacles(spawn_xy, goal_xy)

            obstacles, val_res = self.validator.validate_and_generate_obstacles(
                generator_fn=sample_fn,
                start_xy=spawn_xy,
                goal_xy=goal_xy,
                raise_on_exhaustion=False
            )
            self.obstacles = obstacles
            self.last_validation_result = val_res

        # Cache NumPy obstacle arrays for fast vectorized ray-casting
        if len(self.obstacles) > 0:
            self.obstacle_centers = np.array([obs[:2] for obs in self.obstacles], dtype=np.float32)
            self.obstacle_radii = np.array([obs[2] for obs in self.obstacles], dtype=np.float32)
        else:
            self.obstacle_centers = np.empty((0, 2), dtype=np.float32)
            self.obstacle_radii = np.empty((0,), dtype=np.float32)

        # 4. Reset Episode Counters and Metrics
        self.step_count = 0
        self._episode_max_steps = None
        if "max_episode_steps" in opts:
            active_limit = int(opts["max_episode_steps"])
        else:
            active_limit = self.current_max_episode_steps
        self._episode_max_steps = active_limit
        self.config.max_episode_steps = active_limit
        if hasattr(self, "term_checker") and hasattr(self.term_checker, "config"):
            self.term_checker.config.max_episode_steps = active_limit
        dist = float(np.linalg.norm(state.pos_world - self.goal_world))
        self.initial_dist = dist
        self.prev_dist = dist
        self.last_pos = state.pos_world.copy()
        self.path_length = 0.0
        self.cumulative_reward = 0.0
        self.prev_action = np.zeros(4, dtype=np.float32)

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
        # Apply active LiDAR noise and dropout from domain randomization
        if getattr(self, "active_lidar_noise", 0.0) > 0.0:
            noise = self.np_random.normal(0.0, self.active_lidar_noise, size=raw_ranges.shape).astype(np.float32)
            hit_mask = raw_ranges < self.config.lidar_max_range_m
            raw_ranges[hit_mask] = np.clip(
                raw_ranges[hit_mask] + noise[hit_mask],
                self.config.lidar_min_range_m,
                self.config.lidar_max_range_m
            )
        if getattr(self, "active_lidar_dropout", 0.0) > 0.0:
            drop_mask = self.np_random.random(size=raw_ranges.shape) < self.active_lidar_dropout
            raw_ranges[drop_mask] = self.config.lidar_max_range_m
        self.last_raw_ranges = raw_ranges.copy()

        # 6. Compute Unified 26-D Observation
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
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()

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
            "min_hull_clearance_m": float(min_surf_dist - self.config.uav_radius),
            "sector_status": [s.value for s in meta.sector_status],
            "validity_mask": [bool(v) for v in meta.validity_mask],
            "route_traversable": bool(val_res.is_traversable),
            "route_reason": str(val_res.reason),
            "route_min_clearance": float(val_res.min_route_clearance),
            "route_length": float(val_res.route_length),
            "route_attempts": int(val_res.attempts),
            "curriculum_stage": int(self.curriculum_stage),
            "active_regime": int(self.active_regime),
            "max_episode_steps": int(active_limit),
            "tau_velocity_xy": float(getattr(self.dynamics, "tau_velocity_xy", self.config.tau_velocity_xy)),
            "max_horizontal_accel": float(getattr(self.dynamics, "max_horizontal_accel", self.config.max_horizontal_accel)),
            "active_lidar_dropout": float(self.active_lidar_dropout),
            "active_lidar_noise": float(self.active_lidar_noise),
        }

        return obs, info

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one policy control step (0.1s at 10 Hz) with M5.4 reward components.
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
        if getattr(self, "active_lidar_noise", 0.0) > 0.0:
            noise = self.np_random.normal(0.0, self.active_lidar_noise, size=raw_ranges.shape).astype(np.float32)
            hit_mask = raw_ranges < self.config.lidar_max_range_m
            raw_ranges[hit_mask] = np.clip(
                raw_ranges[hit_mask] + noise[hit_mask],
                self.config.lidar_min_range_m,
                self.config.lidar_max_range_m
            )
        if getattr(self, "active_lidar_dropout", 0.0) > 0.0:
            drop_mask = self.np_random.random(size=raw_ranges.shape) < self.active_lidar_dropout
            raw_ranges[drop_mask] = self.config.lidar_max_range_m
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
        self.last_obs = np.asarray(obs, dtype=np.float32).copy()

        # 6. Collision Checks
        is_obs_collision, col_reason, min_surf_dist = self.check_collision(
            state.pos_world[:2],
            clearances=self.last_clearances
        )

        # 7. Check Standard Terminations
        term_status: TerminationStatus = self.term_checker.check(
            state=state,
            dist_to_goal=dist_curr,
            step_count=self.step_count,
            max_episode_steps=self.current_max_episode_steps
        )

        if is_obs_collision:
            term_status.terminated = True
            term_status.is_crashed = True
            term_status.reason = f"obstacle_collision:{col_reason}"

        # 8. Compute Upgraded M5.4 Reward Components
        cmd_body_vec = np.array([cmd.vx, cmd.vy, cmd.vz], dtype=np.float32)
        reward_comp: ObstacleRewardComponentsV2 = self.compute_obstacle_reward_v2(
            dist_prev=self.prev_dist,
            dist_curr=dist_curr,
            action_curr=act_norm,
            action_prev=self.prev_action,
            clearances=self.last_clearances,
            goal_body=obs[0:3],
            vel_body=obs[3:6],
            is_success=term_status.is_success,
            is_out_of_bounds=term_status.is_out_of_bounds,
            is_ground_crash=(term_status.is_crashed and not is_obs_collision),
            is_obstacle_collision=is_obs_collision,
            cmd_body=cmd_body_vec
        )
        self.cumulative_reward += reward_comp.total

        # 9. Update History
        self.prev_action = act_norm.copy()
        self.prev_dist = dist_curr

        # 10. Diagnostic info dictionary
        info: Dict[str, Any] = {
            "step": self.step_count,
            "max_episode_steps": int(self.current_max_episode_steps),
            "dist_to_goal": dist_curr,
            "path_length": self.path_length,
            "cumulative_reward": self.cumulative_reward,
            "reward_components": reward_comp.to_dict(),
            "min_clearance_m": float(np.min(self.last_clearances)),
            "min_surface_dist_m": min_surf_dist,
            "min_hull_clearance_m": float(min_surf_dist - self.config.uav_radius),
            "heading_error_rad": self.last_heading_error,
            "is_success": term_status.is_success,
            "is_crashed": term_status.is_crashed,
            "is_out_of_bounds": term_status.is_out_of_bounds,
            "termination_reason": term_status.reason,
            "sector_status": [s.value for s in meta.sector_status],
            "validity_mask": [bool(v) for v in meta.validity_mask],
            "active_regime": int(self.active_regime),
        }

        terminated = term_status.terminated
        truncated = term_status.truncated

        return obs, float(reward_comp.total), bool(terminated), bool(truncated), info
