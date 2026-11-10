"""
dynamics.py
-----------
Fast multicopter dynamics with a PX4-style control cascade (Milestone 1).

Causal chain, integrated at 100 Hz:
  velocity command (heading frame, delayed by action latency)
    -> PI velocity loop                       -> acceleration setpoint
    -> thrust-vector geometry (tilt limit)    -> roll/pitch/thrust setpoints
    -> first-order attitude and thrust lags   -> actual attitude and thrust
    -> thrust along the *actual* body z-axis, gravity, linear drag, wind
    -> translational acceleration             -> velocity, position

So the vehicle can only accelerate by tilting, and the tilt lags its setpoint,
as on a real quadrotor. Parameters default to values fitted to PX4 v1.15 x500
SITL velocity-step responses (see uav_sitl/fit_dynamics.py).

Conventions (World ENU, Body FLU, ZYX Euler angles):
  - Velocity commands are in the *heading* frame (body frame rotated by yaw
    only), matching PX4 offboard MAV_FRAME_BODY_NED velocity setpoints.
  - Accelerating forward pitches the nose down (pitch > 0); accelerating left
    rolls the right side down (roll < 0).
  - body_rates are true body angular rates [p, q, r].
"""

from collections import deque
from dataclasses import dataclass
import math
import numpy as np

from rl_env.action import CommandedVelocities
from rl_env.config import UAVEnvConfig
from rl_env.observation import euler_to_quaternion


def _clamp(x: float, lo: float, hi: float) -> float:
    """Scalar clip; much faster than np.clip on Python floats in the 100 Hz loop."""
    return lo if x < lo else hi if x > hi else x


try:
    from numba import njit as _njit
except ImportError:  # pragma: no cover
    def _njit(*a, **k):
        return (lambda f: f) if not a or not callable(a[0]) else a[0]

# Packed state:  0-2 pos, 3-5 vel, 6 roll, 7 pitch, 8 yaw, 9 yaw_rate, 10 thrust, 11-13 velocity integrator,
#                14 ground impact speed (max over the call), 15 roll_dot, 16 pitch_dot
# Packed params: 0 tau_xy, 1 tau_z, 2 i_xy, 3 i_z, 4 tau_att, 5 tau_thrust, 6 drag_xy, 7 drag_z, 8 max_h_acc,
#                9 max_v_acc, 10 max_tilt, 11 thrust_to_weight, 12 tau_yaw, 13 max_yaw_acc, 14 max_yaw_rate,
#                15-17 wind velocity, 18-20 gust acceleration
@_njit(cache=True)
def _cascade_kernel(s, prm, cmds, dt, g):
    for k in range(cmds.shape[0]):
        cvx, cvy, cvz, cyr = cmds[k, 0], cmds[k, 1], cmds[k, 2], cmds[k, 3]
        yaw = s[8]
        cy, sy = math.cos(yaw), math.sin(yaw)
        vx, vy, vz = s[3], s[4], s[5]
        # 1-2. heading-frame setpoint -> PI velocity loop -> acceleration setpoint
        ex = cy * cvx - sy * cvy - vx
        ey = sy * cvx + cy * cvy - vy
        ez = cvz - vz
        s[11] = min(max(s[11] + ex * dt, -2.0), 2.0)
        s[12] = min(max(s[12] + ey * dt, -2.0), 2.0)
        s[13] = min(max(s[13] + ez * dt, -2.0), 2.0)
        ax = ex / prm[0] + prm[2] * s[11]
        ay = ey / prm[0] + prm[2] * s[12]
        az = min(max(ez / prm[1] + prm[3] * s[13], -prm[9]), prm[9])
        a_h = math.sqrt(ax * ax + ay * ay)
        if a_h > prm[8]:
            sc = prm[8] / a_h
            ax *= sc
            ay *= sc
        # 3. thrust-vector setpoint, tilt limited
        a_fwd = cy * ax + sy * ay
        a_left = -sy * ax + cy * ay
        f_up = max(g + az, 0.1 * g)
        f_h = math.sqrt(a_fwd * a_fwd + a_left * a_left)
        f_h_max = f_up * math.tan(prm[10])
        if f_h > f_h_max:
            sc = f_h_max / f_h
            a_fwd *= sc
            a_left *= sc
            f_h = f_h_max
        pitch_sp = math.atan2(a_fwd, f_up)
        roll_sp = math.atan2(-a_left, math.sqrt(a_fwd * a_fwd + f_up * f_up))
        thrust_sp = min(max(math.sqrt(f_h * f_h + f_up * f_up), 0.05 * g), prm[11] * g)
        # 4. actuation lags
        roll_dot = (roll_sp - s[6]) / prm[4]
        pitch_dot = (pitch_sp - s[7]) / prm[4]
        s[6] += roll_dot * dt
        s[7] += pitch_dot * dt
        s[10] += (thrust_sp - s[10]) / prm[5] * dt
        yaw_acc = min(max((cyr - s[9]) / prm[12], -prm[13]), prm[13])
        s[9] = min(max(s[9] + yaw_acc * dt, -prm[14]), prm[14])
        s[8] = (s[8] + s[9] * dt + math.pi) % (2.0 * math.pi) - math.pi
        # 5. translational dynamics from the actual body z-axis
        cr, sr = math.cos(s[6]), math.sin(s[6])
        cp, sp = math.cos(s[7]), math.sin(s[7])
        cy, sy = math.cos(s[8]), math.sin(s[8])
        b3x = cy * sp * cr + sy * sr
        b3y = sy * sp * cr - cy * sr
        b3z = cp * cr
        vx += (s[10] * b3x - prm[6] * (vx - prm[15]) + prm[18]) * dt
        vy += (s[10] * b3y - prm[6] * (vy - prm[16]) + prm[19]) * dt
        vz += (s[10] * b3z - g - prm[7] * (vz - prm[17]) + prm[20]) * dt
        s[0] += vx * dt
        s[1] += vy * dt
        s[2] += vz * dt
        # 6. ground contact
        if s[2] <= 0.0:
            s[2] = 0.0
            if vz < 0.0:
                s[14] = max(s[14], -vz)
                vz = 0.0
                s[13] = max(s[13], 0.0)
            vx *= 0.8
            vy *= 0.8
        s[3], s[4], s[5] = vx, vy, vz
        s[15], s[16] = roll_dot, pitch_dot


@dataclass
class UAVState:
    pos_world: np.ndarray       # [x, y, z] in World ENU (m)
    vel_world: np.ndarray       # [vx, vy, vz] in World ENU (m/s)
    roll: float                 # Roll angle (rad)
    pitch: float                # Pitch angle (rad)
    yaw: float                  # Yaw angle (rad, [-pi, pi])
    body_rates: np.ndarray      # [p, q, r] in Body FLU (rad/s)
    quat: np.ndarray            # [qw, qx, qy, qz] World ENU
    ground_impact_speed: float = 0.0  # Max downward speed at ground contact during the last step (m/s)


class FastUAVDynamics:
    GRAVITY = 9.80665

    def __init__(self, config: UAVEnvConfig, np_random: np.random.Generator = None):
        self.config = config
        self.rng = np_random if np_random is not None else np.random.default_rng()

        # Per-instance parameters (domain randomization overrides these attributes)
        self.tau_velocity_xy = float(config.tau_velocity_xy)   # velocity-loop P: Kp = 1 / tau
        self.tau_velocity_z = float(config.tau_velocity_z)
        self.vel_i_xy = float(config.vel_i_xy)                 # velocity-loop I gain (1/s^2)
        self.vel_i_z = float(config.vel_i_z)
        self.tau_attitude = float(config.tau_attitude)
        self.tau_thrust = float(config.tau_thrust)
        self.drag_xy = float(config.drag_coeff_xy)             # linear drag (1/s)
        self.drag_z = float(config.drag_coeff_z)
        self.max_horizontal_accel = float(config.max_horizontal_accel)
        self.max_vertical_accel = float(config.max_vertical_accel)
        self.max_tilt = float(config.max_tilt_setpoint_rad)
        self.thrust_to_weight = float(config.thrust_to_weight)
        self.noise_pos_std = float(config.noise_pos_std)
        self.noise_vel_std = float(config.noise_vel_std)
        self.noise_rates_std = float(config.noise_rates_std)
        self.wind_disturbance_std = float(config.wind_disturbance_std)
        self.wind_velocity = np.zeros(3, dtype=np.float64)     # steady wind (m/s, ENU); M7 randomizes it

        # Command transport delay (substep FIFO)
        self.cmd_queue = deque()
        self.set_latency(config.action_latency_s)

        # Gust disturbance (Ornstein-Uhlenbeck acceleration)
        self.wind_state = np.zeros(3, dtype=np.float32)

        # Controller / actuator internal state
        self.yaw_rate = 0.0
        self._last_euler_rates = (0.0, 0.0)
        self.vel_int = np.zeros(3, dtype=np.float64)
        self.thrust = self.GRAVITY
        self.state = self._create_initial_state()

    # ------------------------------------------------------------------ setup
    def set_latency(self, latency_s: float) -> None:
        self.action_latency_s = float(latency_s)
        self.latency_substeps = max(1, int(round(self.action_latency_s / self.config.dt_substep)))

    def _create_initial_state(self, x0: float = 0.0, y0: float = 0.0, z0: float = 0.0,
                              yaw0: float = 0.0) -> UAVState:
        return UAVState(
            pos_world=np.array([x0, y0, z0], dtype=np.float32),
            vel_world=np.zeros(3, dtype=np.float32),
            roll=0.0, pitch=0.0, yaw=float(yaw0),
            body_rates=np.zeros(3, dtype=np.float32),
            quat=euler_to_quaternion(0.0, 0.0, yaw0),
        )

    def reset(self, x0: float = 0.0, y0: float = 0.0, z0: float = 0.0, yaw0: float = 0.0) -> UAVState:
        """Reset vehicle state, controller memory and flush the latency queue."""
        self.state = self._create_initial_state(x0, y0, z0, yaw0)
        self.yaw_rate = 0.0
        self.vel_int[:] = 0.0
        self.thrust = self.GRAVITY
        self.wind_state = np.zeros(3, dtype=np.float32)
        self.cmd_queue.clear()
        zero_cmd = CommandedVelocities(0.0, 0.0, 0.0, 0.0)
        for _ in range(self.latency_substeps):
            self.cmd_queue.append(zero_cmd)
        return self.get_state()

    def set_state(self, pos, vel, roll: float, pitch: float, yaw: float, yaw_rate: float = 0.0) -> None:
        """Place the vehicle in an arbitrary flying state (calibration / tests)."""
        self.state.pos_world = np.asarray(pos, dtype=np.float32).copy()
        self.state.vel_world = np.asarray(vel, dtype=np.float32).copy()
        self.state.roll, self.state.pitch, self.state.yaw = float(roll), float(pitch), float(yaw)
        self.state.quat = euler_to_quaternion(roll, pitch, yaw)
        self.yaw_rate = float(yaw_rate)
        self.vel_int[:] = 0.0
        self.thrust = self.GRAVITY / max(0.3, math.cos(roll) * math.cos(pitch))

    # ------------------------------------------------------------------- step
    def step(self, cmd_body: CommandedVelocities) -> UAVState:
        """Advance physics by one policy step (0.1 s) using 100 Hz sub-steps."""
        if self.config.enable_noise:
            theta_w = 0.1
            dt_p = self.config.dt_policy
            dw = self.rng.normal(0.0, np.sqrt(dt_p), size=3).astype(np.float32)
            self.wind_state += -theta_w * self.wind_state * dt_p + self.wind_disturbance_std * dw
            self.wind_state = np.clip(self.wind_state, -0.5, 0.5)

        while len(self.cmd_queue) > self.latency_substeps:   # latency was shortened
            self.cmd_queue.popleft()
        n = self.config.num_substeps
        cmds = np.empty((n, 4))
        zero = CommandedVelocities(0.0, 0.0, 0.0, 0.0)
        for k in range(n):
            self.cmd_queue.append(cmd_body)
            c = self.cmd_queue.popleft() if len(self.cmd_queue) > self.latency_substeps else zero
            cmds[k, 0], cmds[k, 1], cmds[k, 2], cmds[k, 3] = c.vx, c.vy, c.vz, c.yaw_rate
        self._run(cmds, self.config.dt_substep)
        self._update_outputs()
        return self.get_state()

    def _pack(self):
        st, cfg = self.state, self.config
        s = np.empty(17)
        s[0:3] = st.pos_world
        s[3:6] = st.vel_world
        s[6], s[7], s[8], s[9], s[10] = st.roll, st.pitch, st.yaw, self.yaw_rate, self.thrust
        s[11:14] = self.vel_int
        s[14] = 0.0
        s[15], s[16] = self._last_euler_rates
        prm = np.array([self.tau_velocity_xy, self.tau_velocity_z, self.vel_i_xy, self.vel_i_z, self.tau_attitude,
                        self.tau_thrust, self.drag_xy, self.drag_z, self.max_horizontal_accel,
                        self.max_vertical_accel, self.max_tilt, self.thrust_to_weight, cfg.tau_yaw,
                        cfg.max_yaw_accel, cfg.max_yaw_rate, self.wind_velocity[0], self.wind_velocity[1],
                        self.wind_velocity[2], self.wind_state[0], self.wind_state[1], self.wind_state[2]],
                       dtype=np.float64)
        return s, prm

    def _run(self, cmds: np.ndarray, dt: float) -> None:
        s, prm = self._pack()
        _cascade_kernel(s, prm, cmds, float(dt), self.GRAVITY)
        st = self.state
        st.pos_world = s[0:3].astype(np.float32)
        st.vel_world = s[3:6].astype(np.float32)
        st.roll, st.pitch, st.yaw = float(s[6]), float(s[7]), float(s[8])
        self.yaw_rate, self.thrust = float(s[9]), float(s[10])
        self.vel_int[:] = s[11:14]
        st.ground_impact_speed = float(s[14])
        self._last_euler_rates = (float(s[15]), float(s[16]))

    def _update_outputs(self) -> None:
        """Body rates and quaternion from the attitude state (only needed once per policy step)."""
        st = self.state
        sphi, cphi = math.sin(st.roll), math.cos(st.roll)
        sth, cth = math.sin(st.pitch), math.cos(st.pitch)
        roll_dot, pitch_dot = self._last_euler_rates
        p = roll_dot - sth * self.yaw_rate
        q = cphi * pitch_dot + sphi * cth * self.yaw_rate
        r = -sphi * pitch_dot + cphi * cth * self.yaw_rate
        st.body_rates = np.array([p, q, r], dtype=np.float32)
        st.quat = euler_to_quaternion(st.roll, st.pitch, st.yaw)

    def _substep(self, cmd: CommandedVelocities, dt: float, update_outputs: bool = True):
        """One physics sub-step with an already-delayed command (used by calibration and tests)."""
        self._run(np.array([[cmd.vx, cmd.vy, cmd.vz, cmd.yaw_rate]], dtype=np.float64), dt)
        if update_outputs:
            self._update_outputs()

    # ---------------------------------------------------------------- outputs
    def get_state(self) -> UAVState:
        """Return a copy of the current UAV physical state."""
        s = self.state
        return UAVState(pos_world=s.pos_world.copy(), vel_world=s.vel_world.copy(), roll=s.roll,
                        pitch=s.pitch, yaw=s.yaw, body_rates=s.body_rates.copy(), quat=s.quat.copy(),
                        ground_impact_speed=float(s.ground_impact_speed))

    def get_noisy_state(self) -> UAVState:
        """Return state with sensor measurement noise injected for observation computation."""
        s = self.get_state()
        if self.config.enable_noise:
            s.pos_world += self.rng.normal(0.0, self.noise_pos_std, size=3).astype(np.float32)
            s.vel_world += self.rng.normal(0.0, self.noise_vel_std, size=3).astype(np.float32)
            s.body_rates += self.rng.normal(0.0, self.noise_rates_std, size=3).astype(np.float32)
        return s
