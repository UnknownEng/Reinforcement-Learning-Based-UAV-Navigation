"""
gz_link.py
----------
Direct Gazebo Harmonic transport access (no ROS needed):
  - ground-truth vehicle odometry  (/model/<name>/odometry_with_covariance)
  - LiDAR scans                    (LaserScan, or PointCloudPacked for 3-D LiDAR)
  - moving obstacles               (/world/<world>/set_pose service)

All values are returned in world ENU / body FLU, matching rl_env.
"""

import math
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from gz.msgs10.laserscan_pb2 import LaserScan
from gz.msgs10.pointcloud_packed_pb2 import PointCloudPacked
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.boolean_pb2 import Boolean
from gz.transport13 import Node


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_to_rot(qw, qx, qy, qz) -> np.ndarray:
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)],
    ])


@dataclass
class TruthState:
    stamp: float                 # sim time (s)
    wall: float                  # host monotonic time when received
    pos: np.ndarray              # world ENU (m)
    vel: np.ndarray              # world ENU (m/s)
    quat: np.ndarray             # [qw, qx, qy, qz] body FLU -> world ENU
    rates: np.ndarray            # body FLU (rad/s)


@dataclass
class Scan:
    stamp: float
    wall: float
    ranges: np.ndarray           # (rows, cols) ranges in m; inf/NaN = no return
    angle_min: float
    angle_step: float
    v_angle_min: float
    v_angle_step: float


class GzLink:
    def __init__(self, model: str = "x500_lidar_0", world: str = "default",
                 scan_topic: Optional[str] = "/uav/sensors/scan", cloud_topic: Optional[str] = None,
                 range_topic: Optional[str] = "/uav/sensors/range_down"):
        self.model, self.world = model, world
        self.node = Node()
        self.lock = threading.Lock()
        self.truth: Optional[TruthState] = None
        self.scan: Optional[Scan] = None
        self.range_down: Optional[float] = None       # downward rangefinder reading (m), None if no return
        if range_topic:
            self.node.subscribe(LaserScan, range_topic, self._on_range)
        self.truth_log = []
        self.log_truth = False
        self._prev_pose = None  # (stamp, pos, quat)
        # Ground truth: the model's world pose from the physics engine (~50 Hz).
        # Velocity and body rates are finite differences over sim time.
        self.node.subscribe(Pose_V, f"/world/{world}/dynamic_pose/info", self._on_poses)
        if scan_topic:
            self.node.subscribe(LaserScan, scan_topic, self._on_scan)
        if cloud_topic:
            self.node.subscribe(PointCloudPacked, cloud_topic, self._on_cloud)

    # ------------------------------------------------------------ callbacks
    def _on_poses(self, msg: Pose_V) -> None:
        pose = next((p for p in msg.pose if p.name == self.model), None)
        if pose is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9
        pos = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = np.array([pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z])
        q = q if q[0] >= 0 else -q
        vel, rates = np.zeros(3), np.zeros(3)
        prev = self._prev_pose
        if prev is not None and stamp > prev[0]:
            dt = stamp - prev[0]
            vel = (pos - prev[1]) / dt
            dq = quat_mul(np.array([prev[2][0], -prev[2][1], -prev[2][2], -prev[2][3]]), q)  # q_prev^-1 * q
            dq = dq if dq[0] >= 0 else -dq
            rates = 2.0 * dq[1:] / dt                                                          # body FLU
        elif prev is not None:
            return  # duplicate stamp
        self._prev_pose = (stamp, pos, q)
        st = TruthState(stamp=stamp, wall=time.monotonic(), pos=pos, vel=vel, quat=q, rates=rates)
        with self.lock:
            self.truth = st
            if self.log_truth:
                self.truth_log.append(st)

    def _on_scan(self, msg: LaserScan) -> None:
        ranges = np.asarray(msg.ranges, dtype=np.float32)
        rows = max(1, int(msg.vertical_count) or 1)
        scan = Scan(stamp=msg.header.stamp.sec + msg.header.stamp.nsec * 1e-9, wall=time.monotonic(),
                    ranges=ranges.reshape(rows, -1), angle_min=msg.angle_min, angle_step=msg.angle_step,
                    v_angle_min=msg.vertical_angle_min, v_angle_step=msg.vertical_angle_step)
        with self.lock:
            self.scan = scan

    def _on_cloud(self, msg: PointCloudPacked) -> None:  # kept for completeness; LaserScan carries the grid
        pass

    def _on_range(self, msg: LaserScan) -> None:
        r = float(msg.ranges[0]) if len(msg.ranges) else float("inf")
        with self.lock:
            self.range_down = r if math.isfinite(r) and msg.range_min <= r <= msg.range_max else None

    def height_above_ground(self, roll: float, pitch: float, sensor_offset: float = -0.18) -> Optional[float]:
        """Tilt-compensated height of the vehicle reference point above what the downward rangefinder sees.
        Offset measured in SITL: ground-truth model z = range - 0.18 m (on the ground and up to 5 m)."""
        with self.lock:
            r = self.range_down
        if r is None:
            return None
        return r * math.cos(roll) * math.cos(pitch) + sensor_offset

    # --------------------------------------------------------------- access
    def get_truth(self) -> Optional[TruthState]:
        with self.lock:
            return self.truth

    def get_scan(self) -> Optional[Scan]:
        with self.lock:
            return self.scan

    def wait_truth(self, timeout: float = 30.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.get_truth() is not None:
                return True
            time.sleep(0.05)
        return False

    def set_pose(self, name: str, xyz: Tuple[float, float, float], yaw: float = 0.0, timeout_ms: int = 200) -> bool:
        req = Pose()
        req.name = name
        req.position.x, req.position.y, req.position.z = map(float, xyz)
        req.orientation.w, req.orientation.z = math.cos(yaw / 2), math.sin(yaw / 2)
        ok, rep = self.node.request(f"/world/{self.world}/set_pose", req, Pose, Boolean, timeout_ms)
        return bool(ok and rep.data)