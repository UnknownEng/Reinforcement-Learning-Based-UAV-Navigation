#!/usr/bin/env python3
"""
gate_b5_evaluator.py
--------------------
Automated flight evaluator for Milestone 4 Phase 4B — Gate B5.
Executes autonomous corridor flight to goal [6.0, 0.0, 2.0] m through the
Gazebo Harmonic obstacle corridor.

Flight State Machine:
  INIT -> CALL_ARM -> WAIT_ARM_ACK -> CALL_TAKEOFF -> WAIT_TAKEOFF_ACK ->
  CLIMBING -> HOVER_STABILIZE -> CALL_OFFBOARD -> WAIT_OFFBOARD_ACK ->
  CALL_ENABLE_CTRL -> WAIT_ENABLE_CTRL_ACK -> NAVIGATING -> HOVER_AT_GOAL ->
  CALL_DISENGAGE -> CALL_LAND -> WAIT_TOUCHDOWN -> CALL_DISARM -> COMPLETE
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool, Trigger

# Ground truth cylindrical obstacles in obstacle_corridor.sdf (radius = 0.30m)
CORRIDOR_CYLINDERS = [
    {"name": "obs_ahead_3m", "pos": np.array([3.42, 0.0], dtype=np.float32), "radius": 0.30},
    {"name": "obs_boundary_3m", "pos": np.array([3.1688, 1.2629], dtype=np.float32), "radius": 0.30},
    {"name": "obs_front_left_2_5m", "pos": np.array([2.0999, 1.9799], dtype=np.float32), "radius": 0.30},
    {"name": "obs_far_5m", "pos": np.array([1.9327, 4.9804], dtype=np.float32), "radius": 0.30},
    {"name": "obs_close_1_5m", "pos": np.array([1.5945, -1.0324], dtype=np.float32), "radius": 0.30},
    {"name": "obs_front_right_2_5m", "pos": np.array([1.7260, -2.2936], dtype=np.float32), "radius": 0.30},
]
UAV_RADIUS = 0.35  # meters


class GateB5Evaluator(Node):
    def __init__(
        self,
        test_name: str = "gate_b5_trial_1",
        timeout_s: float = 60.0,
        out_dir: str = "/home/kali/proj/experiments/milestone4/phase4b_gate_b5",
        target_goal: Optional[List[float]] = None
    ):
        super().__init__('gate_b5_evaluator')
        self.test_name = test_name
        self.timeout_s = timeout_s
        self.out_dir = out_dir
        self.target_goal = np.array(target_goal if target_goal else [6.0, 0.0, 2.0], dtype=np.float32)

        # Service Clients
        self.cli_arm = self.create_client(SetBool, '/uav/arm')
        self.cli_takeoff = self.create_client(Trigger, '/uav/takeoff')
        self.cli_land = self.create_client(Trigger, '/uav/land')
        self.cli_offboard = self.create_client(SetBool, '/uav/set_offboard')
        self.cli_enable_ctrl = self.create_client(SetBool, '/uav/enable_ppo_control')

        # Publishers
        self.pub_goal = self.create_publisher(Point, '/uav/set_goal', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        # QoS
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        state_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )

        # Subscriptions
        self.sub_odom = self.create_subscription(Odometry, '/uav/odometry', self.odom_callback, sensor_qos)
        self.sub_clearance = self.create_subscription(Float32MultiArray, '/uav/perception/clearance', self.clearance_callback, sensor_qos)
        self.sub_status = self.create_subscription(String, '/uav/status', self.status_callback, state_qos)
        self.sub_diag = self.create_subscription(String, '/uav/adapter_diagnostics', self.diag_callback, 10)

        # Telemetry Cache
        self.latest_odom: Optional[Odometry] = None
        self.latest_clearances: Optional[np.ndarray] = None
        self.latest_status: Optional[dict] = None
        self.latest_diag: Optional[dict] = None

        # State Machine
        self.state = "INIT"
        self.state_start_time = time.time()
        self.state_start_sim_time = 0.0
        self.nav_start_sim_time = 0.0
        self.ground_z = 0.0

        # Flight metrics
        self.trajectory_history: List[Dict[str, Any]] = []
        self.collision_detected = False
        self.collision_reason = "NONE"
        self.min_ground_truth_clearance_m = 10.0
        self.min_measured_clearance_m = 10.0
        self.watchdog_tripped_count = 0
        self.goal_reached_flag = False
        self.final_dist_at_goal = 999.0
        self.final_pos_at_goal = [0.0, 0.0, 0.0]
        self.is_finished = False
        self.trial_result: Dict[str, Any] = {}
        self.active_future = None

        # 20 Hz loop
        self.timer = self.create_timer(0.05, self.state_machine_loop)
        self.get_logger().info(f"GateB5Evaluator initialized for '{self.test_name}' -> Goal: {list(self.target_goal)}")

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

    def clearance_callback(self, msg: Float32MultiArray):
        if len(msg.data) == 8:
            self.latest_clearances = np.array(msg.data, dtype=np.float32)

    def status_callback(self, msg: String):
        try:
            self.latest_status = json.loads(msg.data)
        except Exception:
            pass

    def diag_callback(self, msg: String):
        try:
            self.latest_diag = json.loads(msg.data)
        except Exception:
            pass

    def get_sim_time(self) -> float:
        if self.latest_odom is not None:
            return self.latest_odom.header.stamp.sec + self.latest_odom.header.stamp.nanosec * 1e-9
        return time.time()

    def set_state(self, new_state: str):
        prev = self.state
        self.state = new_state
        self.state_start_time = time.time()
        self.state_start_sim_time = self.get_sim_time()
        self.active_future = None
        self.get_logger().info(f"[STATE] {prev} -> {new_state}")

    def publish_goal(self):
        pt = Point(x=float(self.target_goal[0]), y=float(self.target_goal[1]), z=float(self.target_goal[2]))
        self.pub_goal.publish(pt)
        self.get_logger().info(f"Target goal published: [{self.target_goal[0]:.2f}, {self.target_goal[1]:.2f}, {self.target_goal[2]:.2f}]")

    def check_obstacle_proximity(self, pos_xy: np.ndarray) -> float:
        min_surf_dist = 100.0
        for obs in CORRIDOR_CYLINDERS:
            d_center = float(np.linalg.norm(pos_xy - obs["pos"]))
            d_surf = d_center - obs["radius"]
            if d_surf < min_surf_dist:
                min_surf_dist = d_surf

            if d_surf <= UAV_RADIUS:
                self.collision_detected = True
                self.collision_reason = f"COLLISION with {obs['name']} (surf_dist={d_surf:.3f}m <= {UAV_RADIUS:.2f}m)"

        if min_surf_dist < self.min_ground_truth_clearance_m:
            self.min_ground_truth_clearance_m = min_surf_dist
        return min_surf_dist

    def state_machine_loop(self):
        if self.is_finished:
            return

        now = time.time()
        elapsed_state = now - self.state_start_time

        # Track trajectory & obstacle proximity
        if self.state in ["NAVIGATING", "HOVER_AT_GOAL"] and self.latest_odom is not None:
            pos = self.latest_odom.pose.pose.position
            vel = self.latest_odom.twist.twist.linear
            pos_arr = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
            dist_to_goal = float(np.linalg.norm(self.target_goal - pos_arr))
            speed = float(np.sqrt(vel.x**2 + vel.y**2 + vel.z**2))

            surf_dist = self.check_obstacle_proximity(pos_arr[:2])

            if self.latest_clearances is not None:
                c_min = float(np.min(self.latest_clearances))
                if c_min < self.min_measured_clearance_m:
                    self.min_measured_clearance_m = c_min

            diag_d = self.latest_diag if self.latest_diag else {}
            if diag_d.get("watchdog_tripped", False):
                self.watchdog_tripped_count += 1

            self.trajectory_history.append({
                "time_sim": round(self.get_sim_time(), 3),
                "pos_enu": [round(float(pos.x), 3), round(float(pos.y), 3), round(float(pos.z), 3)],
                "vel_enu": [round(float(vel.x), 3), round(float(vel.y), 3), round(float(vel.z), 3)],
                "speed_m_s": round(speed, 3),
                "dist_to_goal_m": round(dist_to_goal, 3),
                "min_ground_truth_clearance_m": round(surf_dist, 3),
                "latency_ms": diag_d.get("latency_ms", 0.0),
                "watchdog_tripped": diag_d.get("watchdog_tripped", False),
                "cmd_sent": diag_d.get("cmd_sent", [])
            })

        # 1. INIT
        if self.state == "INIT":
            all_services_ready = (
                self.cli_arm.service_is_ready() and
                self.cli_takeoff.service_is_ready() and
                self.cli_land.service_is_ready() and
                self.cli_offboard.service_is_ready() and
                self.cli_enable_ctrl.service_is_ready()
            )
            if self.latest_status and self.latest_status.get("connected", False) and self.latest_odom is not None and all_services_ready:
                if elapsed_state >= 25.0:
                    self.ground_z = self.latest_odom.pose.pose.position.z
                    self.publish_goal()
                    self.set_state("CALL_ARM")
            elif elapsed_state > 60.0:
                self.abort_mission("SERVICES_OR_PX4_NOT_READY_TIMEOUT")

        # 2. CALL_ARM
        elif self.state == "CALL_ARM":
            if not self.cli_arm.service_is_ready():
                return
            req = SetBool.Request()
            req.data = True
            self.active_future = self.cli_arm.call_async(req)
            self.set_state("WAIT_ARM_ACK")

        # 3. WAIT_ARM_ACK
        elif self.state == "WAIT_ARM_ACK":
            if self.latest_status and self.latest_status.get("armed", False):
                if elapsed_state >= 2.0:
                    self.get_logger().info("Drone ARMED. Commanding TAKEOFF...")
                    self.set_state("CALL_TAKEOFF")
            elif self.active_future and self.active_future.done():
                res = self.active_future.result()
                if res and not res.success and elapsed_state > 2.0:
                    req = SetBool.Request()
                    req.data = True
                    self.active_future = self.cli_arm.call_async(req)
            elif elapsed_state > 20.0:
                self.abort_mission("WAIT_ARM_ACK_TIMEOUT")

        # 4. CALL_TAKEOFF
        elif self.state == "CALL_TAKEOFF":
            if not self.cli_takeoff.service_is_ready():
                return
            req = Trigger.Request()
            self.active_future = self.cli_takeoff.call_async(req)
            self.set_state("WAIT_TAKEOFF_ACK")

        # 5. WAIT_TAKEOFF_ACK
        elif self.state == "WAIT_TAKEOFF_ACK":
            mode = self.latest_status.get("flight_mode", "") if self.latest_status else ""
            if mode in ["AUTO_TAKEOFF", "AUTO_LOITER"]:
                self.set_state("CLIMBING")
            elif self.active_future and self.active_future.done():
                res = self.active_future.result()
                if res and not res.success and elapsed_state > 2.0:
                    req = Trigger.Request()
                    self.active_future = self.cli_takeoff.call_async(req)
            elif elapsed_state > 15.0:
                self.abort_mission("WAIT_TAKEOFF_ACK_TIMEOUT")

        # 6. CLIMBING
        elif self.state == "CLIMBING":
            if self.latest_status:
                mode = self.latest_status.get("flight_mode", "")
                armed = self.latest_status.get("armed", False)
                if not armed or "LAND" in mode:
                    self.abort_mission(f"PREFLIGHT_FAILSAFE_LAND (mode={mode}, armed={armed})")
                    return
            if self.latest_odom is not None:
                rel_alt = self.latest_odom.pose.pose.position.z - self.ground_z
                mode = self.latest_status.get("flight_mode", "") if self.latest_status else ""
                if (rel_alt >= 0.6 and mode == "AUTO_LOITER") or rel_alt >= 1.0:
                    self.get_logger().info(f"Climb altitude reached ({rel_alt:.2f}m). Hover stabilizing...")
                    self.set_state("HOVER_STABILIZE")
            if elapsed_state > 90.0:
                self.abort_mission("CLIMB_TIMEOUT")

        # 7. HOVER_STABILIZE (3.0s)
        elif self.state == "HOVER_STABILIZE":
            if elapsed_state >= 3.0:
                self.nav_start_sim_time = self.get_sim_time()
                self.set_state("CALL_OFFBOARD")

        # 8. CALL_OFFBOARD
        elif self.state == "CALL_OFFBOARD":
            if not self.cli_offboard.service_is_ready():
                return
            req = SetBool.Request()
            req.data = True
            self.active_future = self.cli_offboard.call_async(req)
            self.set_state("WAIT_OFFBOARD_ACK")

        # 9. WAIT_OFFBOARD_ACK
        elif self.state == "WAIT_OFFBOARD_ACK":
            mode = self.latest_status.get("flight_mode", "") if self.latest_status else ""
            if mode == "OFFBOARD":
                self.set_state("CALL_ENABLE_CTRL")
            elif elapsed_state > 30.0:
                self.abort_mission("WAIT_OFFBOARD_ACK_TIMEOUT")
            elif (now - getattr(self, '_last_offboard_retry_time', 0.0)) > 2.0:
                self._last_offboard_retry_time = now
                self.get_logger().info("Retrying OFFBOARD request...")
                req = SetBool.Request()
                req.data = True
                self.active_future = self.cli_offboard.call_async(req)

        # 10. CALL_ENABLE_CTRL
        elif self.state == "CALL_ENABLE_CTRL":
            if not self.cli_enable_ctrl.service_is_ready():
                return
            req = SetBool.Request()
            req.data = True
            self.active_future = self.cli_enable_ctrl.call_async(req)
            self.set_state("WAIT_ENABLE_CTRL_ACK")

        # 11. WAIT_ENABLE_CTRL_ACK
        elif self.state == "WAIT_ENABLE_CTRL_ACK":
            if self.latest_diag and self.latest_diag.get("control_enabled", False):
                self.get_logger().info(f"26-D PPO Authority Engaged. Navigating corridor to {list(self.target_goal)}...")
                self.set_state("NAVIGATING")
            elif self.active_future and self.active_future.done():
                res = self.active_future.result()
                if res and not res.success and elapsed_state > 2.0:
                    req = SetBool.Request()
                    req.data = True
                    self.active_future = self.cli_enable_ctrl.call_async(req)
            elif elapsed_state > 12.0:
                self.abort_mission("WAIT_ENABLE_CTRL_ACK_TIMEOUT")

        # 12. NAVIGATING
        elif self.state == "NAVIGATING":
            pos = self.latest_odom.pose.pose.position
            vel = self.latest_odom.twist.twist.linear
            current_pos = np.array([pos.x, pos.y, pos.z])
            dist = float(np.linalg.norm(self.target_goal - current_pos))
            speed = float(np.sqrt(vel.x**2 + vel.y**2 + vel.z**2))
            elapsed_sim = self.get_sim_time() - self.state_start_sim_time

            if (dist <= 0.45 and speed <= 0.50) or (current_pos[0] >= 5.80 and abs(current_pos[1]) <= 0.80 and speed <= 0.60) or (current_pos[0] >= 6.0 and abs(current_pos[1]) <= 1.0):
                self.get_logger().info(f"CORRIDOR GOAL REACHED! Final dist={dist:.3f}m, Speed={speed:.3f}m/s, Pos={list(current_pos)}.")
                self.goal_reached_flag = True
                self.final_dist_at_goal = float(dist)
                self.final_pos_at_goal = [float(current_pos[0]), float(current_pos[1]), float(current_pos[2])]
                self.set_state("HOVER_AT_GOAL")
            elif elapsed_sim > self.timeout_s or elapsed_state > (self.timeout_s * 2.0):
                self.get_logger().warn(f"Corridor navigation TIMED OUT (sim={elapsed_sim:.1f}s, wall={elapsed_state:.1f}s). Final dist: {dist:.3f}m.")
                self.set_state("HOVER_AT_GOAL")

        # 13. HOVER_AT_GOAL (2.0s)
        elif self.state == "HOVER_AT_GOAL":
            if elapsed_state >= 2.0:
                self.get_logger().info("Disengaging 26-D PPO flight control...")
                self.set_state("CALL_DISENGAGE")

        # 14. CALL_DISENGAGE
        elif self.state == "CALL_DISENGAGE":
            if not self.cli_enable_ctrl.service_is_ready():
                return
            req = SetBool.Request()
            req.data = False
            self.active_future = self.cli_enable_ctrl.call_async(req)
            self.set_state("WAIT_DISENGAGE_ACK")

        # 15. WAIT_DISENGAGE_ACK
        elif self.state == "WAIT_DISENGAGE_ACK":
            if self.latest_diag and not self.latest_diag.get("control_enabled", True):
                self.set_state("CALL_LAND")
            elif self.active_future and self.active_future.done():
                self.set_state("CALL_LAND")
            elif elapsed_state > 5.0:
                self.set_state("CALL_LAND")

        # 16. CALL_LAND
        elif self.state == "CALL_LAND":
            if not self.cli_land.service_is_ready():
                return
            req = Trigger.Request()
            self.active_future = self.cli_land.call_async(req)
            self.set_state("WAIT_TOUCHDOWN")

        # 17. WAIT_TOUCHDOWN
        elif self.state == "WAIT_TOUCHDOWN":
            mode = self.latest_status.get("flight_mode", "") if self.latest_status else ""
            landed = self.latest_status.get("landed_state", "") if self.latest_status else ""
            armed = self.latest_status.get("armed", False) if self.latest_status else False

            if not armed or landed == "ON_GROUND" or elapsed_state > 20.0:
                self.get_logger().info("Vehicle landed and disarmed. Mission COMPLETE.")
                self.set_state("COMPLETE")
                self.compile_results()
                self.is_finished = True

    def abort_mission(self, reason: str):
        self.get_logger().error(f"ABORTING CORRIDOR MISSION: {reason}")
        req_f = SetBool.Request()
        req_f.data = False
        self.cli_enable_ctrl.call_async(req_f)
        req_l = Trigger.Request()
        self.cli_land.call_async(req_l)
        self.set_state("ABORTED")
        self.compile_results(aborted_reason=reason)
        self.is_finished = True

    def compile_results(self, aborted_reason: Optional[str] = None):
        nav_pts = [p for p in self.trajectory_history]
        if len(nav_pts) > 1:
            pts_arr = np.array([p["pos_enu"] for p in nav_pts])
            total_path_len = float(np.sum(np.linalg.norm(np.diff(pts_arr, axis=0), axis=1)))
        else:
            total_path_len = 0.0

        all_latencies = [p["latency_ms"] for p in nav_pts if p["latency_ms"] > 0]
        steady_latencies = all_latencies[5:] if len(all_latencies) > 5 else all_latencies
        all_speeds = [p["speed_m_s"] for p in nav_pts]
        total_duration = (self.get_sim_time() - self.nav_start_sim_time) if self.nav_start_sim_time > 0 else 0.0

        if self.goal_reached_flag:
            is_goal_reached = True
            final_dist = self.final_dist_at_goal
            final_pos = self.final_pos_at_goal
        else:
            last_pt = nav_pts[-1] if nav_pts else {}
            final_dist = last_pt.get("dist_to_goal_m", 999.0)
            final_pos = last_pt.get("pos_enu", [0.0, 0.0, 0.0])
            is_goal_reached = (final_dist <= 0.45) or (final_pos[0] >= 5.80 and abs(final_pos[1]) <= 0.80) or (final_pos[0] >= 6.0 and abs(final_pos[1]) <= 1.0)
        has_zero_collisions = (not self.collision_detected)
        has_zero_watchdogs = (self.watchdog_tripped_count == 0)

        trial_success = (
            aborted_reason is None and
            is_goal_reached and
            has_zero_collisions and
            has_zero_watchdogs
        )

        self.trial_result = {
            "test_name": self.test_name,
            "overall_success": trial_success,
            "aborted_reason": aborted_reason,
            "goal_reached": is_goal_reached,
            "final_distance_to_goal_m": final_dist,
            "final_position_enu": final_pos,
            "collision_detected": self.collision_detected,
            "collision_reason": self.collision_reason,
            "min_ground_truth_clearance_m": round(self.min_ground_truth_clearance_m, 3),
            "min_measured_clearance_m": round(self.min_measured_clearance_m, 3),
            "watchdog_trips": self.watchdog_tripped_count,
            "total_nav_duration_sim_s": round(total_duration, 2),
            "total_path_length_m": round(total_path_len, 3),
            "max_speed_m_s": round(float(np.max(all_speeds)), 3) if all_speeds else 0.0,
            "mean_speed_m_s": round(float(np.mean(all_speeds)), 3) if all_speeds else 0.0,
            "mean_inference_latency_ms": round(float(np.mean(steady_latencies)), 3) if steady_latencies else 0.0,
            "p95_inference_latency_ms": round(float(np.percentile(steady_latencies, 95)), 3) if steady_latencies else 0.0,
            "trajectory_samples_count": len(self.trajectory_history),
            "trajectory_samples": self.trajectory_history
        }

        os.makedirs(self.out_dir, exist_ok=True)
        out_file = os.path.join(self.out_dir, f"{self.test_name}.json")
        with open(out_file, "w") as f:
            json.dump(self.trial_result, f, indent=2)

        self.get_logger().info("==================================================")
        self.get_logger().info(f"=== GATE B5 TRIAL '{self.test_name}' COMPLETE ===")
        self.get_logger().info(f"Overall Success: {trial_success}")
        self.get_logger().info(f"Goal Reached: {is_goal_reached} (Final Dist: {final_dist:.3f} m)")
        self.get_logger().info(f"Collisions: {self.collision_detected} (Min Clearance: {self.min_ground_truth_clearance_m:.3f} m)")
        self.get_logger().info(f"Watchdog Trips: {self.watchdog_tripped_count}")
        self.get_logger().info(f"Duration: {total_duration:.2f} s | Path: {total_path_len:.3f} m")
        self.get_logger().info(f"Saved to: {out_file}")
        self.get_logger().info("==================================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="gate_b5_trial_1")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--outdir", type=str, default="/home/kali/proj/experiments/milestone4/phase4b_gate_b5")
    args = parser.parse_args()

    rclpy.init()
    node = GateB5Evaluator(test_name=args.name, timeout_s=args.timeout, out_dir=args.outdir)
    try:
        while rclpy.ok() and not node.is_finished:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        node.abort_mission("KEYBOARD_INTERRUPT")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
