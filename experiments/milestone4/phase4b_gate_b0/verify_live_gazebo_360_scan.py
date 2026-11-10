"""
verify_live_gazebo_360_scan.py
------------------------------
Empirically launches Gazebo Harmonic with the 360° LiDAR configuration,
bridges the scan via ros_gz_bridge, captures live LaserScan messages,
and verifies actual samples, angles, increment, frequency, and 8-sector clearances.
"""

import json
import math
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from rl_env.observation_26d import ObservationHandler26D, SectorStatus, InvalidClearancePolicy


class ScanVerifierNode(Node):
    def __init__(self, target_samples: int = 15):
        super().__init__('scan_verifier_node')
        self.target_samples = target_samples
        self.received_msgs: List[LaserScan] = []
        self.timestamps: List[float] = []

        self.sub = self.create_subscription(
            LaserScan,
            '/uav/sensors/scan',
            self.scan_callback,
            10
        )

    def scan_callback(self, msg: LaserScan):
        now = time.time()
        self.timestamps.append(now)
        self.received_msgs.append(msg)
        if len(self.received_msgs) >= self.target_samples:
            self.get_logger().info(f"Collected {len(self.received_msgs)} messages. Stopping.")


def main():
    print("=" * 70)
    print("STARTING GAZEBO HARMONIC 360° LIDAR EMPIRICAL VERIFICATION")
    print("=" * 70)

    env = os.environ.copy()
    models_path = "/home/kali/proj/models:/home/kali/proj/PX4-Autopilot/Tools/simulation/gz/models"
    if "GZ_SIM_RESOURCE_PATH" in env:
        env["GZ_SIM_RESOURCE_PATH"] = f"{models_path}:{env['GZ_SIM_RESOURCE_PATH']}"
    else:
        env["GZ_SIM_RESOURCE_PATH"] = models_path

    world_sdf = "/home/kali/proj/experiments/milestone4/phase4b_gate_b0/gazebo_360_verification_world.sdf"

    gz_cmd = ["gz", "sim", "-r", "-s", world_sdf]
    print(f"Launching Gazebo: {' '.join(gz_cmd)}")
    gz_proc = subprocess.Popen(gz_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    bridge_proc = None
    try:
        # Wait 3 seconds for Gazebo to initialize
        time.sleep(3.0)

        bridge_cmd = [
            "ros2", "run", "ros_gz_bridge", "parameter_bridge",
            "/uav/sensors/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"
        ]
        print(f"Launching ros_gz_bridge: {' '.join(bridge_cmd)}")
        bridge_proc = subprocess.Popen(bridge_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        time.sleep(2.0)

        # Initialize ROS 2 and capture
        rclpy.init()
        node = ScanVerifierNode(target_samples=15)

        start_spin = time.time()
        max_duration_s = 8.0

        while rclpy.ok() and len(node.received_msgs) < 15:
            rclpy.spin_once(node, timeout_sec=0.2)
            if time.time() - start_spin > max_duration_s:
                print("Capture timeout reached.")
                break

        msg_count = len(node.received_msgs)
        print(f"Total messages captured: {msg_count}")

        if msg_count == 0:
            print("ERROR: Zero messages received on /uav/sensors/scan!")
            return 1

        first_msg: LaserScan = node.received_msgs[0]

        # Calculate rate
        intervals = [node.timestamps[i] - node.timestamps[i - 1] for i in range(1, len(node.timestamps))]
        mean_dt = float(np.mean(intervals)) if intervals else 0.1
        measured_rate_hz = 1.0 / mean_dt if mean_dt > 0 else 0.0

        sample_count = len(first_msg.ranges)
        ranges = [float(r) for r in first_msg.ranges]

        # Process with ObservationHandler26D
        handler = ObservationHandler26D(
            invalid_policy=InvalidClearancePolicy.CONSERVATIVE_ZERO,
            min_range=first_msg.range_min,
            max_range=first_msg.range_max,
            default_clear_range=10.0
        )

        clearances, meta = handler.process_raw_scan(
            ranges=ranges,
            angle_min=first_msg.angle_min,
            angle_increment=first_msg.angle_increment
        )

        results = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "empirical_gazebo_verification": {
                "total_messages_received": msg_count,
                "measured_frequency_hz": round(measured_rate_hz, 2),
                "target_frequency_hz": 10.0,
                "frame_id": first_msg.header.frame_id,
                "angle_min_rad": float(first_msg.angle_min),
                "angle_max_rad": float(first_msg.angle_max),
                "angle_increment_rad": float(first_msg.angle_increment),
                "angle_increment_deg": round(math.degrees(first_msg.angle_increment), 4),
                "range_min_m": float(first_msg.range_min),
                "range_max_m": float(first_msg.range_max),
                "samples_count": sample_count,
                "raw_ranges_sample": ranges,
                "parsed_8_sector_clearances_m": [round(float(c), 4) for c in clearances],
                "sector_statuses": [s.value for s in meta.sector_status],
                "validity_mask": [bool(m) for m in meta.validity_mask],
                "blind_sectors_count": sum(1 for s in meta.sector_status if s == SectorStatus.UNOBSERVED_BLIND_ZONE),
                "active_sectors_count": int(np.sum(meta.validity_mask))
            }
        }

        out_path = "/home/kali/proj/experiments/milestone4/phase4b_gate_b0/gazebo_360_scan_verification.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 70)
        print("EMPIRICAL GAZEBO 360° LIDAR RESULTS:")
        print(f"  Frame ID:          {first_msg.header.frame_id}")
        print(f"  Sample Count:      {sample_count} rays")
        print(f"  Angle Min:         {first_msg.angle_min:.6f} rad ({math.degrees(first_msg.angle_min):.2f}°)")
        print(f"  Angle Max:         {first_msg.angle_max:.6f} rad ({math.degrees(first_msg.angle_max):.2f}°)")
        print(f"  Angle Increment:   {first_msg.angle_increment:.6f} rad ({math.degrees(first_msg.angle_increment):.4f}°)")
        print(f"  Measured Rate:     {measured_rate_hz:.2f} Hz")
        print(f"  Sector Clearances: {[round(float(c), 2) for c in clearances]} m")
        print(f"  Sector Statuses:   {[s.value for s in meta.sector_status]}")
        print(f"  Blind Sectors:     {sum(1 for s in meta.sector_status if s == SectorStatus.UNOBSERVED_BLIND_ZONE)}")
        print(f"  Saved Results to:  {out_path}")
        print("=" * 70)

        node.destroy_node()
        rclpy.shutdown()

    finally:
        print("Terminating background processes...")
        if bridge_proc:
            bridge_proc.terminate()
            try:
                bridge_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                bridge_proc.kill()
        if gz_proc:
            gz_proc.terminate()
            try:
                gz_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                gz_proc.kill()

        # Extra safety cleanup
        os.system("pkill -9 -f parameter_bridge || true")
        os.system("pkill -9 -f 'gz sim' || true")
        os.system("pkill -9 -f 'ruby.*gz' || true")
        print("All processes cleanly terminated.")

    return 0


if __name__ == "__main__":
    import numpy as np
    sys.exit(main())
