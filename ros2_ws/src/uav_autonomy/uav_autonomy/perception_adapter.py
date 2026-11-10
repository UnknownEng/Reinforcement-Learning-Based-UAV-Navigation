#!/usr/bin/env python3
"""
perception_adapter.py
---------------------
Milestone 4 Phase 4A Perception Adapter Node.
Consumes:
  - /uav/sensors/scan (sensor_msgs/msg/LaserScan)
  - /uav/sensors/range_down (sensor_msgs/msg/Range, optional)

Produces:
  - /uav/perception/clearance (std_msgs/msg/Float32MultiArray): 8 sector distances [d_F, d_FL, d_L, d_BL, d_B, d_BR, d_R, d_FR]
  - /uav/perception/diagnostics (std_msgs/msg/String): JSON metadata with per-sector ray statistics, latency, and ground clearance

Zero modification to the frozen PPO policy or 18-D observation vector.
"""

import math
import time
import json
from typing import Dict, Any, List, Tuple

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import LaserScan, Range
from std_msgs.msg import Float32MultiArray, String


class PerceptionSectorMapper:
    """
    Deterministic 8-sector partitioning and conservative clearance calculation.
    Sectors:
      0: Front (F)       [-22.5 deg, +22.5 deg)
      1: Front-Left (FL)  [+22.5 deg, +67.5 deg)
      2: Left (L)        [+67.5 deg, +112.5 deg)
      3: Back-Left (BL)   [+112.5 deg, +157.5 deg)
      4: Back (B)        [+157.5 deg, +180 deg] U [-180 deg, -157.5 deg)
      5: Back-Right (BR)  [-157.5 deg, -112.5 deg)
      6: Right (R)       [-112.5 deg, -67.5 deg)
      7: Front-Right (FR) [-67.5 deg, -22.5 deg)
    """

    SECTOR_NAMES = [
        "front",
        "front_left",
        "left",
        "back_left",
        "back",
        "back_right",
        "right",
        "front_right"
    ]

    def __init__(self, min_range: float = 0.10, max_range: float = 10.0, default_clearance: float = 10.0):
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.default_clearance = float(default_clearance)

    @staticmethod
    def angle_to_sector(angle_rad: float) -> int:
        """Map an angle in [-pi, pi] to sector index 0..7."""
        # Normalize to [-pi, pi]
        ang = math.atan2(math.sin(angle_rad), math.cos(angle_rad))
        deg = round(math.degrees(ang), 6)

        if -22.5 <= deg < 22.5:
            return 0  # Front
        elif 22.5 <= deg < 67.5:
            return 1  # Front-Left
        elif 67.5 <= deg < 112.5:
            return 2  # Left
        elif 112.5 <= deg < 157.5:
            return 3  # Back-Left
        elif deg >= 157.5 or deg < -157.5:
            return 4  # Back
        elif -157.5 <= deg < -112.5:
            return 5  # Back-Right
        elif -112.5 <= deg < -67.5:
            return 6  # Right
        elif -67.5 <= deg < -22.5:
            return 7  # Front-Right
        return 4

    def process_scan(self, ranges: List[float], angle_min: float, angle_increment: float) -> Tuple[List[float], Dict[str, Any]]:
        """
        Processes an array of ranges into 8 directional sector clearances.
        Returns:
          - clearances: list of 8 floats [d_0, ..., d_7]
          - sector_info: dictionary containing per-sector diagnostics
        """
        num_sectors = 8
        sector_rays: List[List[float]] = [[] for _ in range(num_sectors)]
        sector_valid_rays: List[List[float]] = [[] for _ in range(num_sectors)]

        num_rays = len(ranges)
        for i in range(num_rays):
            angle = angle_min + i * angle_increment
            sec_idx = self.angle_to_sector(angle)
            r = ranges[i]

            sector_rays[sec_idx].append(r)

            # Validation: filter non-finite, out-of-range, and zero/negative
            if math.isnan(r) or math.isinf(r):
                continue
            if r < self.min_range or r > self.max_range:
                continue
            if r <= 0.0:
                continue

            sector_valid_rays[sec_idx].append(r)

        clearances = [self.default_clearance] * num_sectors
        sector_diagnostics = {}

        for sec_idx in range(num_sectors):
            name = self.SECTOR_NAMES[sec_idx]
            total_rays = len(sector_rays[sec_idx])
            val_rays = sector_valid_rays[sec_idx]

            if len(val_rays) > 0:
                c = float(min(val_rays))
                is_valid = True
            else:
                c = float(self.default_clearance)
                is_valid = False

            clearances[sec_idx] = c
            sector_diagnostics[name] = {
                'sector_index': sec_idx,
                'clearance_m': round(c, 4),
                'valid': is_valid,
                'total_rays': total_rays,
                'valid_rays': len(val_rays)
            }

        return clearances, sector_diagnostics


class PerceptionAdapterNode(Node):
    def __init__(self):
        super().__init__('perception_adapter')

        # Declare parameters
        self.declare_parameter('min_range', 0.10)
        self.declare_parameter('max_range', 10.0)
        self.declare_parameter('default_clearance', 10.0)
        self.declare_parameter('publish_diagnostics', True)

        min_range = self.get_parameter('min_range').get_parameter_value().double_value
        max_range = self.get_parameter('max_range').get_parameter_value().double_value
        default_clearance = self.get_parameter('default_clearance').get_parameter_value().double_value
        self.publish_diag = self.get_parameter('publish_diagnostics').get_parameter_value().bool_value

        self.mapper = PerceptionSectorMapper(
            min_range=min_range,
            max_range=max_range,
            default_clearance=default_clearance
        )

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.latest_ground_clearance: float = 0.0

        # Subscriptions
        self.sub_scan = self.create_subscription(
            LaserScan, '/uav/sensors/scan', self._scan_callback, qos
        )
        self.sub_range_down = self.create_subscription(
            Range, '/uav/sensors/range_down', self._range_down_callback, qos
        )

        # Publishers
        self.pub_clearance = self.create_publisher(
            Float32MultiArray, '/uav/perception/clearance', qos_reliable
        )
        self.pub_diagnostics = self.create_publisher(
            String, '/uav/perception/diagnostics', qos_reliable
        )

        self.get_logger().info("Perception Adapter Node initialized successfully.")

    def _range_down_callback(self, msg: Range):
        r = float(msg.range)
        if not (math.isnan(r) or math.isinf(r)) and r > 0.0:
            self.latest_ground_clearance = r

    def _scan_callback(self, msg: LaserScan):
        t_start = time.perf_counter()

        ranges = [float(r) for r in msg.ranges]
        angle_min = float(msg.angle_min)
        angle_inc = float(msg.angle_increment)

        clearances, sector_diag = self.mapper.process_scan(ranges, angle_min, angle_inc)

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # Publish clearances
        arr_msg = Float32MultiArray()
        arr_msg.data = [float(c) for c in clearances]
        self.pub_clearance.publish(arr_msg)

        # Publish diagnostics
        if self.publish_diag:
            source_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            cur_time = self.get_clock().now().to_msg()
            cur_stamp = cur_time.sec + cur_time.nanosec * 1e-9

            # Find min clearance among valid sectors
            valid_clearances = [(name, d['clearance_m']) for name, d in sector_diag.items() if d['valid']]
            if valid_clearances:
                min_sec_name, min_dist = min(valid_clearances, key=lambda x: x[1])
            else:
                min_sec_name, min_dist = "none", float(self.mapper.default_clearance)

            diag = {
                'timestamp': round(cur_stamp, 4),
                'source_scan_timestamp': round(source_stamp, 4),
                'source_frame': msg.header.frame_id,
                'processing_latency_ms': round(t_elapsed_ms, 3),
                'ground_clearance_m': round(self.latest_ground_clearance, 4),
                'min_clearance_m': round(min_dist, 4),
                'min_clearance_sector': min_sec_name,
                'sectors': sector_diag
            }
            str_msg = String()
            str_msg.data = json.dumps(diag)
            self.pub_diagnostics.publish(str_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionAdapterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
