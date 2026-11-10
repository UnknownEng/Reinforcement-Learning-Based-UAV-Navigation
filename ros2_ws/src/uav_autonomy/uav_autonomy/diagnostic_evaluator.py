#!/usr/bin/env python3
"""
diagnostic_evaluator.py
-----------------------
Automated flight trial runner for Milestone 3B diagnostic experiments.
Executes standardized 2-waypoint sequence:
  WP 1: [3.0, 0.0, 2.5] (In-distribution forward climb)
  WP 2: [2.0, 2.0, 2.5] (Lateral/reverse transition where M3 stalled)
Records high-frequency telemetry including commanded/measured velocities,
yaw angles, yaw rates, setpoint frames, and tracking performance.
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
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

# Ensure project root and venv site-packages are in sys.path
proj_root = os.path.expanduser("~/proj")
venv_site = os.path.join(proj_root, "venv/lib/python3.12/site-packages")
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

WAYPOINT_COURSES = {
    "2wp": [
        [3.0, 0.0, 2.5],  # WP 1: Forward climb
        [2.0, 2.0, 2.5],  # WP 2: Lateral/reverse diagonal
    ],
    "4wp": [
        [3.0, 0.0, 2.5],   # WP 1: Forward climb
        [2.0, 2.0, 2.5],   # WP 2: Lateral/reverse diagonal
        [-2.0, 2.0, 2.2],  # WP 3: Reverse cross-field
        [0.0, 0.0, 2.0],   # WP 4: Return to home hover
    ]
}


class DiagnosticEvaluator(Node):
    def __init__(self, test_name: str = "diagnostic_trial", leg_timeout_s: float = 30.0, course: str = "2wp", out_dir: str = os.path.join(proj_root, "experiments/milestone3d/trials")):
        super().__init__('diagnostic_evaluator')
        self.test_name = test_name
        self.course_name = course
        self.out_dir = out_dir
        wp_list = WAYPOINT_COURSES.get(course, WAYPOINT_COURSES["2wp"])
        self.waypoints = [np.array(wp, dtype=np.float32) for wp in wp_list]
        self.current_wp_idx = 0
        self.leg_timeout_s = leg_timeout_s

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
        self.sub_status = self.create_subscription(String, '/uav/status', self.status_callback, state_qos)
        self.sub_diag = self.create_subscription(String, '/uav/adapter_diagnostics', self.diag_callback, 10)

        # Telemetry Cache
        self.latest_odom: Optional[Odometry] = None
        self.latest_status: Optional[dict] = None
        self.latest_diag: Optional[dict] = None

        # State Machine
        self.state = "INIT"
        self.state_start_time = time.time()
        self.nav_start_sim_time = 0.0
        self.leg_start_sim_time = 0.0
        self.leg_start_pos = None
        self.ground_z = 0.0

        self.trajectory_history: List[Dict[str, Any]] = []
        self.leg_results: List[Dict[str, Any]] = []
        self.leg_reached = False
        self.leg_arrival_speed = 0.0
        self.leg_arrival_dist = 0.0
        self.active_future = None
        self.trial_result: Dict[str, Any] = {}
        self.is_finished = False

        # 20 Hz Loop
        self.timer = self.create_timer(0.05, self.state_machine_loop)
        self.get_logger().info(f"DiagnosticEvaluator initialized: '{self.test_name}' with {len(self.waypoints)} waypoints.")

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg

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
        self.active_future = None
        self.get_logger().info(f"[STATE] {prev} -> {new_state}")

    def current_target_goal(self) -> np.ndarray:
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        return self.waypoints[-1]

    def publish_active_goal(self):
        target = self.current_target_goal()
        pt = Point(x=float(target[0]), y=float(target[1]), z=float(target[2]))
        self.pub_goal.publish(pt)
        self.get_logger().info(f"Target WP [{self.current_wp_idx + 1}/{len(self.waypoints)}] published: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")

    def state_machine_loop(self):
        if self.is_finished:
            return

        now = time.time()
        elapsed_state = now - self.state_start_time
        target_goal = self.current_target_goal()

        # High-frequency telemetry logging during navigation
        if self.state in ["NAVIGATING", "HOVER_AT_GOAL"] and self.latest_odom is not None:
            pos = self.latest_odom.pose.pose.position
            vel = self.latest_odom.twist.twist.linear
            dist = float(np.linalg.norm(target_goal - np.array([pos.x, pos.y, pos.z])))
            speed = float(np.sqrt(vel.x**2 + vel.y**2 + vel.z**2))

            diag_d = self.latest_diag if self.latest_diag else {}

            entry = {
                "time_sim": round(self.get_sim_time(), 3),
                "time_wall": round(now, 3),
                "state": self.state,
                "wp_index": self.current_wp_idx,
                "target_goal_enu": [round(float(v), 3) for v in target_goal],
                "pos_enu": [round(pos.x, 3), round(pos.y, 3), round(pos.z, 3)],
                "vel_enu": [round(vel.x, 3), round(vel.y, 3), round(vel.z, 3)],
                "vel_flu": diag_d.get("vel_flu", []),
                "speed_m_s": round(speed, 3),
                "yaw_enu_rad": diag_d.get("yaw_enu_rad", 0.0),
                "yaw_enu_deg": diag_d.get("yaw_enu_deg", 0.0),
                "rates_flu": diag_d.get("rates_flu", []),
                "dist_to_goal_m": round(dist, 3),
                "bearing_to_goal_rad": diag_d.get("bearing_to_goal_rad", 0.0),
                "heading_error_rad": diag_d.get("heading_error_rad", 0.0),
                "ppo_action_raw": diag_d.get("ppo_action_raw", []),
                "ppo_cmd_scaled": diag_d.get("ppo_cmd_scaled", []),
                "cmd_sent": diag_d.get("cmd_sent", []),
                "command_frame": diag_d.get("command_frame", "NONE"),
                "speed_scale": diag_d.get("speed_scale", 1.0),
                "latency_ms": diag_d.get("inference_latency_ms", 0.0),
                "watchdog_tripped": diag_d.get("watchdog_tripped", False),
                "flight_mode": self.latest_status.get("flight_mode", "UNKNOWN") if self.latest_status else "UNKNOWN"
            }
            self.trajectory_history.append(entry)

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
                if elapsed_state >= 8.0:
                    self.ground_z = self.latest_odom.pose.pose.position.z
                    self.publish_active_goal()
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
            if self.latest_odom is not None:
                rel_alt = self.latest_odom.pose.pose.position.z - self.ground_z
                mode = self.latest_status.get("flight_mode", "") if self.latest_status else ""
                if (rel_alt >= 0.6 and mode == "AUTO_LOITER") or rel_alt >= 1.0:
                    self.get_logger().info(f"Climb altitude reached ({rel_alt:.2f}m, mode={mode}). Hover stabilizing...")
                    self.set_state("HOVER_STABILIZE")
            if elapsed_state > 5.0 and self.latest_status and not self.latest_status.get("armed", True):
                self.abort_mission("PREFLIGHT_FAIL_DISARMED")
            elif elapsed_state > 90.0:
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
            elif self.active_future and self.active_future.done():
                res = self.active_future.result()
                if res and not res.success and elapsed_state > 2.0:
                    req = SetBool.Request()
                    req.data = True
                    self.active_future = self.cli_offboard.call_async(req)
            elif elapsed_state > 20.0:
                self.abort_mission("WAIT_OFFBOARD_ACK_TIMEOUT")

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
                self.get_logger().info(f"Diagnostic Flight Authority Engaged. Navigating to WP 1...")
                pos = self.latest_odom.pose.pose.position
                self.leg_start_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
                self.leg_start_sim_time = self.get_sim_time()
                self.leg_reached = False
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
            dist = float(np.linalg.norm(target_goal - current_pos))
            speed = float(np.sqrt(vel.x**2 + vel.y**2 + vel.z**2))

            # Success threshold: dist <= 0.40m and speed <= 0.50m/s
            if dist <= 0.40 and speed <= 0.50:
                self.leg_reached = True
                self.leg_arrival_speed = speed
                self.leg_arrival_dist = dist
                self.get_logger().info(f"WP [{self.current_wp_idx + 1}/{len(self.waypoints)}] REACHED! Dist={dist:.2f}m, Speed={speed:.2f}m/s.")
                self.set_state("HOVER_AT_GOAL")
            elif elapsed_state > self.leg_timeout_s:
                self.leg_reached = False
                self.leg_arrival_speed = speed
                self.leg_arrival_dist = dist
                self.get_logger().warn(f"WP [{self.current_wp_idx + 1}/{len(self.waypoints)}] TIMED OUT after {elapsed_state:.1f}s. Final dist: {dist:.2f}m.")
                self.set_state("HOVER_AT_GOAL")

        # 13. HOVER_AT_GOAL (3.0s)
        elif self.state == "HOVER_AT_GOAL":
            if elapsed_state >= 3.0:
                self.record_leg_result()
                self.current_wp_idx += 1
                if self.current_wp_idx < len(self.waypoints):
                    pos = self.latest_odom.pose.pose.position
                    self.leg_start_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
                    self.leg_start_sim_time = self.get_sim_time()
                    self.publish_active_goal()
                    self.leg_reached = False
                    self.set_state("NAVIGATING")
                else:
                    self.get_logger().info("All diagnostic waypoints completed! Disengaging control...")
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
            if self.active_future and self.active_future.done():
                self.set_state("CALL_DISENGAGE_OFFBOARD")
            elif elapsed_state > 3.0:
                self.set_state("CALL_DISENGAGE_OFFBOARD")

        # 16. CALL_DISENGAGE_OFFBOARD
        elif self.state == "CALL_DISENGAGE_OFFBOARD":
            if not self.cli_offboard.service_is_ready():
                return
            req = SetBool.Request()
            req.data = False
            self.active_future = self.cli_offboard.call_async(req)
            self.set_state("WAIT_DISENGAGE_OFFBOARD_ACK")

        # 17. WAIT_DISENGAGE_OFFBOARD_ACK
        elif self.state == "WAIT_DISENGAGE_OFFBOARD_ACK":
            if self.active_future and self.active_future.done():
                self.set_state("CALL_LAND")
            elif elapsed_state > 3.0:
                self.set_state("CALL_LAND")

        # 18. CALL_LAND
        elif self.state == "CALL_LAND":
            if not self.cli_land.service_is_ready():
                return
            req = Trigger.Request()
            self.active_future = self.cli_land.call_async(req)
            self.set_state("WAIT_TOUCHDOWN")

        # 19. WAIT_TOUCHDOWN
        elif self.state == "WAIT_TOUCHDOWN":
            z = self.latest_odom.pose.pose.position.z if self.latest_odom else 0.0
            rel_alt = z - self.ground_z
            landed = self.latest_status.get('landed_state', '') if self.latest_status else ''
            armed = self.latest_status.get('armed', False) if self.latest_status else False

            if (landed == 'ON_GROUND') or (rel_alt < 0.15 and elapsed_state > 5.0) or not armed:
                self.get_logger().info(f"Touchdown confirmed (rel_alt={rel_alt:.2f}m).")
                self.set_state("CALL_DISARM")
            elif elapsed_state > 25.0:
                self.get_logger().warn("Touchdown timeout. Forcing DISARM...")
                self.set_state("CALL_DISARM")

        # 20. CALL_DISARM
        elif self.state == "CALL_DISARM":
            if not self.cli_arm.service_is_ready():
                return
            req = SetBool.Request()
            req.data = False
            self.active_future = self.cli_arm.call_async(req)
            self.set_state("WAIT_DISARM_ACK")

        # 21. WAIT_DISARM_ACK
        elif self.state == "WAIT_DISARM_ACK":
            armed = self.latest_status.get('armed', True) if self.latest_status else True
            if not armed or elapsed_state > 6.0:
                self.get_logger().info("Disarm confirmed. Diagnostic mission COMPLETE.")
                self.set_state("COMPLETE")
                self.compile_mission_results()
                self.is_finished = True

    def record_leg_result(self):
        wp_idx = self.current_wp_idx
        target = self.current_target_goal()
        curr_sim_t = self.get_sim_time()
        leg_duration = curr_sim_t - self.leg_start_sim_time

        leg_pts = [p for p in self.trajectory_history if p["wp_index"] == wp_idx and p["state"] in ["NAVIGATING", "HOVER_AT_GOAL"]]
        if leg_pts:
            pts_arr = np.array([p["pos_enu"] for p in leg_pts])
            path_len = float(np.sum(np.linalg.norm(np.diff(pts_arr, axis=0), axis=1)))
            final_err = float(np.linalg.norm(target - np.array(leg_pts[-1]["pos_enu"])))
            direct_dist = float(np.linalg.norm(target - self.leg_start_pos)) if self.leg_start_pos is not None else path_len
            min_dist = float(np.min([p["dist_to_goal_m"] for p in leg_pts]))
            all_speeds = [p["speed_m_s"] for p in leg_pts]
            mean_speed = float(np.mean(all_speeds))
            max_speed = float(np.max(all_speeds))
            all_rates = [abs(p["rates_flu"][2]) for p in leg_pts if p.get("rates_flu") and len(p["rates_flu"]) > 2]
            mean_yaw_rate = float(np.mean(all_rates)) if all_rates else 0.0
            max_yaw_rate = float(np.max(all_rates)) if all_rates else 0.0
        else:
            path_len, final_err, direct_dist, min_dist = 0.0, 999.0, 0.0, 999.0
            mean_speed, max_speed, mean_yaw_rate, max_yaw_rate = 0.0, 0.0, 0.0, 0.0

        is_success = self.leg_reached
        efficiency = (path_len / direct_dist) if direct_dist > 0.05 else 1.0

        leg_res = {
            "wp_index": wp_idx,
            "target": [round(float(v), 3) for v in target],
            "start_pos": [round(float(v), 3) for v in self.leg_start_pos] if self.leg_start_pos is not None else [],
            "final_error_m": round(final_err, 3),
            "min_distance_m": round(min_dist, 3),
            "arrival_dist_m": round(self.leg_arrival_dist, 3),
            "arrival_speed_m_s": round(self.leg_arrival_speed, 3),
            "duration_sim_s": round(leg_duration, 2),
            "path_length_m": round(path_len, 3),
            "direct_distance_m": round(direct_dist, 3),
            "efficiency": round(efficiency, 3),
            "mean_speed_m_s": round(mean_speed, 3),
            "max_speed_m_s": round(max_speed, 3),
            "mean_yaw_rate_rad_s": round(mean_yaw_rate, 3),
            "max_yaw_rate_rad_s": round(max_yaw_rate, 3),
            "success": is_success
        }
        self.leg_results.append(leg_res)
        self.get_logger().info(
            f"=== WP {wp_idx + 1} RESULT: Success={is_success} | Final Err={final_err:.3f}m | Min Dist={min_dist:.3f}m | "
            f"Duration={leg_duration:.2f}s | Path={path_len:.3f}m | Speed={mean_speed:.2f}m/s | YawRate={mean_yaw_rate:.2f}rad/s ==="
        )

    def abort_mission(self, reason: str):
        self.get_logger().error(f"ABORTING MISSION: {reason}")
        req_f = SetBool.Request()
        req_f.data = False
        self.cli_enable_ctrl.call_async(req_f)
        req_l = Trigger.Request()
        self.cli_land.call_async(req_l)
        self.set_state("ABORTED")
        self.compile_mission_results(aborted_reason=reason)
        self.is_finished = True

    def compile_mission_results(self, aborted_reason: Optional[str] = None):
        nav_pts = [p for p in self.trajectory_history if p["state"] in ["NAVIGATING", "HOVER_AT_GOAL"]]
        if len(nav_pts) > 1:
            pts_arr = np.array([p["pos_enu"] for p in nav_pts])
            total_path_len = float(np.sum(np.linalg.norm(np.diff(pts_arr, axis=0), axis=1)))
        else:
            total_path_len = 0.0

        all_latencies = [p["latency_ms"] for p in nav_pts if p["latency_ms"] > 0]
        all_speeds = [p["speed_m_s"] for p in nav_pts]
        total_nav_duration = (self.get_sim_time() - self.nav_start_sim_time) if self.nav_start_sim_time > 0 else 0.0

        overall_success = (aborted_reason is None and len(self.leg_results) > 0 and all(leg["success"] for leg in self.leg_results))

        self.trial_result = {
            "test_name": self.test_name,
            "overall_success": overall_success,
            "aborted_reason": aborted_reason,
            "total_waypoints": len(self.waypoints),
            "completed_waypoints": len(self.leg_results),
            "successful_waypoints": sum(1 for leg in self.leg_results if leg["success"]),
            "success_rate_percent": round(100.0 * sum(1 for leg in self.leg_results if leg["success"]) / len(self.waypoints), 1) if self.waypoints else 0.0,
            "total_nav_duration_sim_s": round(total_nav_duration, 2),
            "total_path_length_m": round(total_path_len, 3),
            "max_speed_m_s": round(float(np.max(all_speeds)), 3) if all_speeds else 0.0,
            "mean_speed_m_s": round(float(np.mean(all_speeds)), 3) if all_speeds else 0.0,
            "mean_inference_latency_ms": round(float(np.mean(all_latencies)), 3) if all_latencies else 0.0,
            "legs": self.leg_results,
            "trajectory_samples_count": len(self.trajectory_history),
            "trajectory_samples": self.trajectory_history
        }

        out_dir = self.out_dir
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{self.test_name}.json")
        with open(out_file, "w") as f:
            json.dump(self.trial_result, f, indent=2)

        self.get_logger().info("==================================================")
        self.get_logger().info(f"=== DIAGNOSTIC TEST '{self.test_name}' COMPLETE ===")
        self.get_logger().info(f"Course: {self.course_name} ({len(self.waypoints)} waypoints)")
        self.get_logger().info(f"Overall Success: {overall_success} ({self.trial_result['successful_waypoints']}/{len(self.waypoints)} waypoints)")
        self.get_logger().info(f"Total Nav Duration: {total_nav_duration:.2f} s | Total Path Length: {total_path_len:.3f} m")
        self.get_logger().info(f"Saved to: {out_file}")
        self.get_logger().info("==================================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="diagnostic_trial")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--course", type=str, choices=["2wp", "4wp"], default="2wp")
    parser.add_argument("--outdir", type=str, default=os.path.join(proj_root, "experiments/milestone3d/trials"))
    args = parser.parse_args()

    rclpy.init()
    node = DiagnosticEvaluator(test_name=args.name, leg_timeout_s=args.timeout, course=args.course, out_dir=args.outdir)
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
