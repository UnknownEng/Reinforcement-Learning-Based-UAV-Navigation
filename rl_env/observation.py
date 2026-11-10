"""
observation.py
--------------
Observation space definition, computation, and normalization for the UAV environment.
Strictly adheres to the 18-dimensional low-dimensional proprioceptive specification.
"""

from dataclasses import dataclass
import gymnasium as gym
import numpy as np

from rl_env.config import UAVEnvConfig


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """
    Convert unit quaternion [qw, qx, qy, qz] to 3x3 rotation matrix R (Body to World).
    """
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    # Normalize quaternion to prevent numerical drift
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm > 1e-8:
        qw /= norm
        qx /= norm
        qy /= norm
        qz /= norm
    else:
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qw * qz)
    r02 = 2.0 * (qx * qz + qw * qy)

    r10 = 2.0 * (qx * qy + qw * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qw * qx)

    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    return np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ], dtype=np.float32)


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw in radians, ZYX convention) to unit quaternion [qw, qx, qy, qz].
    """
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    q = np.array([qw, qx, qy, qz], dtype=np.float32)
    norm = np.linalg.norm(q)
    return q / norm if norm > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def canonical_unit_quaternion(quat: np.ndarray) -> np.ndarray:
    """
    Normalize [qw, qx, qy, qz] and pick the sign with qw >= 0.

    q and -q are the same rotation, but a policy sees them as different inputs.
    Simulator quaternions and quaternions from PX4/scipy (which may return qw < 0,
    e.g. for yaw between -90 and -180 deg) must map to the same observation.
    """
    q = np.asarray(quat, dtype=np.float32).reshape(-1)[:4].copy()
    q_norm = float(np.linalg.norm(q))
    if not np.isfinite(q_norm) or q_norm <= 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    q /= q_norm
    if q[0] < 0.0:
        q = -q
    return q.astype(np.float32)


class ObservationHandler:
    def __init__(self, config: UAVEnvConfig):
        self.config = config
        self.obs_dim = 18

        # Define observation space bounds
        low = np.array([
            -50.0, -50.0, -50.0,  # goal_body (x, y, z)
            -15.0, -15.0, -15.0,  # vel_body (vx, vy, vz)
            -10.0, -10.0, -10.0,  # rates_body (p, q, r)
            -1.0,  -1.0,  -1.0,  -1.0,  # quat (qw, qx, qy, qz)
            0.0,                  # relative_altitude
            -1.0,  -1.0,  -1.0,  -1.0   # prev_action (a0, a1, a2, a3)
        ], dtype=np.float32)

        high = np.array([
            50.0,  50.0,  50.0,   # goal_body (x, y, z)
            15.0,  15.0,  15.0,   # vel_body (vx, vy, vz)
            10.0,  10.0,  10.0,   # rates_body (p, q, r)
            1.0,   1.0,   1.0,   1.0,   # quat (qw, qx, qy, qz)
            50.0,                 # relative_altitude
            1.0,   1.0,   1.0,   1.0    # prev_action (a0, a1, a2, a3)
        ], dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            shape=(self.obs_dim,),
            dtype=np.float32
        )

    def compute_observation(
        self,
        pos_world: np.ndarray,
        vel_world: np.ndarray,
        quat: np.ndarray,
        rates_body: np.ndarray,
        goal_world: np.ndarray,
        prev_action: np.ndarray,
        ground_z: float = 0.0
    ) -> np.ndarray:
        """
        Construct the 18D observation vector from physical state variables.
        """
        # 1. Rotation matrix from body to world R_b^w
        R = quat_to_rotation_matrix(quat)
        R_trans = R.T  # World to Body

        # 2. Relative goal in body frame
        rel_pos_world = (goal_world - pos_world).astype(np.float32)
        goal_body = R_trans @ rel_pos_world

        # 3. Linear velocity in body frame
        vel_body = R_trans @ vel_world.astype(np.float32)

        # 4. Angular velocity in body frame
        rates = rates_body.astype(np.float32)

        # 5. Unit attitude quaternion
        q_unit = canonical_unit_quaternion(quat)

        # 6. Relative altitude above ground datum
        rel_alt = float(pos_world[2] - ground_z)

        # 7. Previous action
        act_prev = np.asarray(prev_action, dtype=np.float32).flatten()

        # Assemble observation vector
        obs = np.concatenate([
            goal_body,       # 3
            vel_body,        # 3
            rates,           # 3
            q_unit,          # 4
            [rel_alt],       # 1
            act_prev         # 4
        ]).astype(np.float32)

        # Sanitize against NaN / Inf
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)

        # Clip within observation space bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        return obs
