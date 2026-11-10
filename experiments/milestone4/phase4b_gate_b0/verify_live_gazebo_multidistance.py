"""
verify_live_gazebo_multidistance.py
-----------------------------------
Gap 1: Live Gazebo Multi-Distance Verification

Empirically launches Gazebo Harmonic with the 360° LiDAR configuration
in an isolated world containing 8 obstacles at distinct known distances
and azimuths (front, lateral, rear, and diagonal).
Bridges `/uav/sensors/scan` via ros_gz_bridge, captures live LaserScan messages,
records exact geometry, ray count, angle increment, publication rate,
and calculates numerical errors against analytical expectations under a strict tolerance.
"""

import json
import math
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, List
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from rl_env.observation_26d import ObservationHandler26D, SectorStatus, InvalidClearancePolicy


class MultiDistanceScanVerifier(Node):
    def __init__(self, target_samples: int = 15):
        super().__init__('multidistance_scan_verifier')
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
    print("=" * 75)
    print("GAP 1: LIVE GAZEBO MULTI-DISTANCE & MULTI-AZIMUTH VERIFICATION")
    print("=" * 75)

    env = os.environ.copy()
    models_path = "/home/kali/proj/models:/home/kali/proj/PX4-Autopilot/Tools/simulation/gz/models"
    if "GZ_SIM_RESOURCE_PATH" in env:
        env["GZ_SIM_RESOURCE_PATH"] = f"{models_path}:{env['GZ_SIM_RESOURCE_PATH']}"
    else:
        env["GZ_SIM_RESOURCE_PATH"] = models_path

    world_sdf = "/home/kali/proj/experiments/milestone4/phase4b_gate_b0/gazebo_360_multidistance_world.sdf"

    # Expected Ground-Truth Clearances and Azimuths
    # In Option 1, rays are shifted by +2.5 deg (-177.5 deg to +177.5 deg, step 5.0 deg).
    # For an obstacle centered at azimuth theta_c with radius R=0.25m and center distance D_c:
    # The adjacent rays at theta_c +/- 2.5 deg intersect the cylinder at analytical distance:
    # t = D_c * cos(2.5 deg) - sqrt(R^2 - D_c^2 * sin^2(2.5 deg)).
    expected_obstacles = [
        {"sector": 0, "name": "front", "azimuth_deg": 0.0, "expected_clearance_m": 1.5103, "expected_apex_m": 1.500, "center_dist_m": 1.75, "radius_m": 0.25, "adjacent_rays": [35, 36]},
        {"sector": 1, "name": "front_left", "azimuth_deg": 45.0, "expected_clearance_m": 3.5574, "expected_apex_m": 3.500, "center_dist_m": 3.75, "radius_m": 0.25, "adjacent_rays": [44, 45]},
        {"sector": 2, "name": "left", "azimuth_deg": 90.0, "expected_clearance_m": 2.0179, "expected_apex_m": 2.000, "center_dist_m": 2.25, "radius_m": 0.25, "adjacent_rays": [53, 54]},
        {"sector": 3, "name": "back_left", "azimuth_deg": 135.0, "expected_clearance_m": 4.2882, "expected_apex_m": 4.200, "center_dist_m": 4.45, "radius_m": 0.25, "adjacent_rays": [62, 63]},
        {"sector": 4, "name": "back", "azimuth_deg": 180.0, "expected_clearance_m": 2.8354, "expected_apex_m": 2.800, "center_dist_m": 3.05, "radius_m": 0.25, "adjacent_rays": [0, 71]},
        {"sector": 5, "name": "back_right", "azimuth_deg": -135.0, "expected_clearance_m": 4.9270, "expected_apex_m": 4.800, "center_dist_m": 5.05, "radius_m": 0.25, "adjacent_rays": [8, 9]},
        {"sector": 6, "name": "right", "azimuth_deg": -90.0, "expected_clearance_m": 1.8146, "expected_apex_m": 1.800, "center_dist_m": 2.05, "radius_m": 0.25, "adjacent_rays": [17, 18]},
        {"sector": 7, "name": "front_right", "azimuth_deg": -45.0, "expected_clearance_m": 3.0410, "expected_apex_m": 3.000, "center_dist_m": 3.25, "radius_m": 0.25, "adjacent_rays": [26, 27]},
    ]

    gz_cmd = ["gz", "sim", "-r", "-s", world_sdf]
    print(f"Launching Gazebo: {' '.join(gz_cmd)}")
    gz_proc = subprocess.Popen(gz_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    bridge_proc = None
    try:
        # Allow Gazebo 3.5 seconds to load models and sensors
        time.sleep(3.5)

        bridge_cmd = [
            "ros2", "run", "ros_gz_bridge", "parameter_bridge",
            "/uav/sensors/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan"
        ]
        print(f"Launching ros_gz_bridge: {' '.join(bridge_cmd)}")
        bridge_proc = subprocess.Popen(bridge_cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        time.sleep(2.0)

        # Initialize ROS 2
        rclpy.init()
        node = MultiDistanceScanVerifier(target_samples=15)

        start_spin = time.time()
        max_duration_s = 10.0

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

        first_msg: LaserScan = node.received_msgs[-1]  # use latest stable message

        intervals = [node.timestamps[i] - node.timestamps[i - 1] for i in range(1, len(node.timestamps))]
        mean_dt = float(np.mean(intervals)) if intervals else 0.1
        measured_rate_hz = 1.0 / mean_dt if mean_dt > 0 else 0.0

        sample_count = len(first_msg.ranges)
        ranges = [float(r) for r in first_msg.ranges]

        # Parse with 26-D ObservationHandler
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

        # Numerical comparison against expected
        tolerance_m = 0.020  # 20 mm tolerance for simulation discretization & mesh geometry
        sector_results = []
        all_passed = True

        print("\n--- SECTOR-BY-SECTOR MULTI-DISTANCE COMPARISON ---")
        for obs in expected_obstacles:
            sec_idx = obs["sector"]
            name = obs["name"]
            az = obs["azimuth_deg"]
            exp_c = obs["expected_clearance_m"]
            act_c = float(clearances[sec_idx])
            err = abs(act_c - exp_c)
            status = meta.sector_status[sec_idx].value
            passed = (err <= tolerance_m) and (status == SectorStatus.VALID_HIT.value)

            if not passed:
                all_passed = False

            sector_results.append({
                "sector": sec_idx,
                "name": name,
                "azimuth_deg": az,
                "expected_clearance_m": exp_c,
                "actual_clearance_m": round(act_c, 4),
                "error_m": round(err, 4),
                "tolerance_m": tolerance_m,
                "status": status,
                "verdict": "PASS" if passed else "FAIL"
            })

            print(f"  Sector {sec_idx} ({name:>11}, {az:>6.1f}°): "
                  f"Expected={exp_c:.3f}m, Actual={act_c:.3f}m, "
                  f"Error={err*1000.0:.2f}mm, Status={status} -> {'PASS' if passed else 'FAIL'}")

        # Verification metrics
        sample_count_pass = (sample_count == 72)
        increment_pass = abs(math.degrees(first_msg.angle_increment) - 5.0) < 0.01
        frequency_pass = (8.0 <= measured_rate_hz <= 12.0)
        blind_sectors_pass = (sum(1 for s in meta.sector_status if s == SectorStatus.UNOBSERVED_BLIND_ZONE) == 0)

        overall_pass = all_passed and sample_count_pass and increment_pass and frequency_pass and blind_sectors_pass

        results = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gap_id": "Gap 1: Live Gazebo Multi-Distance Verification",
            "verdict": "PASS" if overall_pass else "FAIL",
            "metadata": {
                "world_sdf": world_sdf,
                "total_messages_received": msg_count,
                "measured_frequency_hz": round(measured_rate_hz, 2),
                "target_frequency_hz": 10.0,
                "frequency_in_spec": frequency_pass,
                "frame_id": first_msg.header.frame_id,
                "angle_min_rad": float(first_msg.angle_min),
                "angle_max_rad": float(first_msg.angle_max),
                "angle_increment_rad": float(first_msg.angle_increment),
                "angle_increment_deg": round(math.degrees(first_msg.angle_increment), 4),
                "range_min_m": float(first_msg.range_min),
                "range_max_m": float(first_msg.range_max),
                "samples_count": sample_count,
                "samples_count_in_spec": sample_count_pass,
                "blind_sectors_count": sum(1 for s in meta.sector_status if s == SectorStatus.UNOBSERVED_BLIND_ZONE),
                "active_sectors_count": int(np.sum(meta.validity_mask)),
                "raw_ranges_sample": ranges,
            },
            "sector_evaluations": sector_results
        }

        out_path = "/home/kali/proj/experiments/milestone4/phase4b_gate_b0/gazebo_multidistance_verification.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "=" * 75)
        print(f"GAP 1 OVERALL VERDICT: {'PASS' if overall_pass else 'FAIL'}")
        print(f"Results saved to: {out_path}")
        print("=" * 75)

        node.destroy_node()
        rclpy.shutdown()

    finally:
        print("\nCleaning up processes...")
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

        os.system("pkill -9 -f parameter_bridge || true")
        os.system("pkill -9 -f 'gz sim' || true")
        os.system("pkill -9 -f 'ruby.*gz' || true")
        print("All processes cleanly terminated.")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
