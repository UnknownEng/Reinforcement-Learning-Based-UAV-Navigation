"""
config.py
---------
Configuration parameters for the Fast UAV Gymnasium Training Environment.
All values are physically grounded in PX4 multicopter (x500) parameters and SI units.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class UAVEnvConfig:
    # --- Timing & Frequencies ---
    policy_rate_hz: float = 10.0          # PPO decision rate (10 Hz, dt = 0.1s)
    physics_substep_hz: float = 100.0     # Internal physics sub-stepping (100 Hz, dt = 0.01s)
    max_episode_steps: int = 200          # Maximum episode duration = 20.0 seconds

    # --- Velocity Limits (Scaled from action [-1, 1]) ---
    max_horizontal_velocity: float = 3.0  # Max horizontal speed vx, vy (m/s)
    max_vertical_velocity: float = 1.5    # Max climb/descent speed vz (m/s)
    max_yaw_rate: float = 1.0             # Max yaw rate (rad/s, ~57.3 deg/s)

    # --- Acceleration & Dynamic Response Limits ---
    max_horizontal_accel: float = 10.0    # Cap on the horizontal acceleration setpoint (m/s^2); tilt limit binds first
    max_vertical_accel: float = 6.0       # Cap on the vertical acceleration setpoint (m/s^2)
    max_yaw_accel: float = 4.0            # Max yaw angular acceleration (rad/s^2)
    max_tilt_setpoint_rad: float = np.radians(45.0)  # PX4 MPC_TILTMAX_AIR
    thrust_to_weight: float = 2.0         # Max collective thrust / weight (x500)

    # --- PX4-style velocity cascade, fitted to PX4 v1.15 x500 SITL ground truth
    #     (uav_sitl/fit_dynamics.py on logs/px4_step_response.json) ---
    tau_velocity_xy: float = 0.602        # Horizontal velocity loop, Kp = 1/tau (fitted; PX4 MPC_XY_VEL_P_ACC = 1.8)
    tau_velocity_z: float = 0.166         # Vertical velocity loop, Kp = 1/tau (fitted)
    vel_i_xy: float = 0.4                 # Horizontal velocity integral gain (PX4 MPC_XY_VEL_I_ACC)
    vel_i_z: float = 2.0                  # Vertical velocity integral gain (PX4 MPC_Z_VEL_I_ACC)
    tau_attitude: float = 0.10            # Roll/pitch tracking lag (s, fitted)
    tau_thrust: float = 0.018             # Collective thrust (motor) lag (s, fitted)
    drag_coeff_xy: float = 0.177          # Linear horizontal drag (1/s, fitted)
    drag_coeff_z: float = 0.069           # Linear vertical drag (1/s, fitted)
    tau_yaw: float = 0.33                 # Yaw rate response time constant (s, fitted)

    # --- Latency & Transport Delay ---
    action_latency_s: float = 0.08        # Command delay (s, fitted to PX4 SITL)

    # --- Geofence Boundaries (ENU World Frame) ---
    arena_x: Tuple[float, float] = (-12.0, 12.0)  # East bounds (m)
    arena_y: Tuple[float, float] = (-12.0, 12.0)  # North bounds (m)
    arena_z: Tuple[float, float] = (0.0, 12.0)    # Altitude bounds (m, 0 = ground)
    max_tilt_angle_rad: float = np.radians(60.0)  # Max allowed roll/pitch before tilt failure
    ground_crash_speed: float = 1.0       # Touchdown faster than this downward speed is a crash (m/s)

    # --- Mission & Goal Parameters ---
    goal_tolerance_dist: float = 0.40     # Distance threshold for goal success (m)
    goal_tolerance_vel: float = 0.50      # Velocity magnitude threshold for stabilization (m/s)
    min_goal_dist: float = 2.5            # Minimum initial distance from start to goal (m)
    max_goal_dist: float = 8.0            # Maximum initial distance from start to goal (m)
    goal_altitude_range: Tuple[float, float] = (1.5, 4.5)  # Goal altitude bounds (m)

    # --- Randomization (Episode Reset) ---
    start_pos_xy_range: Tuple[float, float] = (-1.5, 1.5)  # Initial ground position range (m)
    start_altitude: float = 0.0           # Initial altitude (0.0 = landed on ground)
    randomize_start_yaw: bool = True      # Randomize initial orientation yaw [-pi, pi]

    # --- Disturbances & Measurement Noise ---
    enable_noise: bool = True             # Enable sensor noise and wind perturbations
    noise_pos_std: float = 0.01           # Position measurement standard deviation (m)
    noise_vel_std: float = 0.02           # Velocity measurement standard deviation (m/s)
    noise_rates_std: float = 0.01         # Angular rate measurement standard deviation (rad/s)
    wind_disturbance_std: float = 0.10    # Stochastic force/wind disturbance (m/s^2)

    # --- Reward Function Weights ---
    reward_progress_weight: float = 2.5       # Progress reward coefficient: k * (d_{t-1} - d_t)
    reward_success: float = 100.0             # Large terminal bonus on stable goal arrival
    reward_action_smoothness: float = 0.05    # Penalty coefficient on ||a_t - a_{t-1}||^2
    reward_collision_penalty: float = 50.0    # Penalty for crashing into ground or geofence
    reward_time_penalty: float = 0.02         # Small per-step penalty to encourage efficient paths

    @property
    def dt_policy(self) -> float:
        return 1.0 / self.policy_rate_hz

    @property
    def dt_substep(self) -> float:
        return 1.0 / self.physics_substep_hz

    @property
    def num_substeps(self) -> int:
        return int(round(self.dt_policy / self.dt_substep))
