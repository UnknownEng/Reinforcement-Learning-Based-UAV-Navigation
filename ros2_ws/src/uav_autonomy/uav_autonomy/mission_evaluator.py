#!/usr/bin/env python3
"""
mission_evaluator.py
--------------------
Automated multi-waypoint flight trial runner for Stage D & E evaluation in PX4 SITL + Gazebo Harmonic.
Executes standardized non-blocking asynchronous flight sequence:
  INIT -> CALL_ARM -> WAIT_ARM -> CALL_TAKEOFF -> WAIT_TAKEOFF -> CLIMBING ->
  HOVER_STABILIZE -> CALL_OFFBOARD -> CALL_ENABLE_CTRL ->
  [NAVIGATING(wp_i) -> HOVER_AT_GOAL(wp_i)] for all waypoints ->
  CALL_DISENGAGE -> CALL_DISENGAGE_OFFBOARD -> CALL_LAND -> WAIT_TOUCHDOWN -> CALL_DISARM -> COMPLETE
Records high-fidelity time-series telemetry, per-waypoint metrics, latency, and tracking performance.
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


DEFAULT_EVAL_WAYPOINTS = [
    [3.0, 0.0, 2.5],   # WP 1: Forward + climb
    [2.0, 2.0, 2.5],   # WP 2: Diagonal transfer
    [-2.0, 2.0, 2.2],  # WP 3: Reverse-diagonal
    [0.0, 0.0, 2.0],   # WP 4: Return-to-home hover
]


class MissionEvaluator(Node):
    def __init__(self, controller: str = "PPO", waypoints: Optional[List[List[float]]] = None, mission_name: str = "eval_mission"):
        super().__init__('mission_evaluator')
        self.controller = controller.upper()
        self.waypoints = [np.array(wp, dtype=np.float32) for wp in (waypoints if waypoints else DEFAULT_EVAL_WAYPOINTS)]
        self.current_wp_idx = 0
        self.mission_name = mission_name

        # Service Clients
        self.cli_arm = self.create_client(SetBool, '/uav/arm')
        self.cli_takeoff = self.create_client(Trigger, '/uav/takeoff')
        self.cli_land = self.create_client(Trigger, '/uav/land')
        self.cli_offboard = self.create_client(SetBool, '/uav/set_offboard')
        self.cli_enable_ctrl = self.create_client(SetBool, '/uav/enable_ppo_control')

        # Publishers
        self.pub_goal = self.create_publisher(Point, '/uav/set_goal', 10)
        self.pub_select_ctrl = self.create_publisher(String, '/uav/select_controller', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        # Subscriptions
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

        self.sub_odom = self.create_subscription(Odometry, '/uav/odometry', self.odom_callback, sensor_qos)
        self.sub_status = self.create_subscription(String, '/uav/status', self.status_callback, state_qos)
        self.sub_diag = self.create_subscription(String, '/uav/adapter_diagnostics', self.diag_callback, 10)

        # Telemetry Cache
        self.latest_odom: Optional[Odometry] = None
        self.latest_status: Optional[dict] = None
        self.latest_diag: Optional[dict] = None

        # Telemetry & Metrics History
        self.trajectory_history: List[Dict[str, Any]] = []
        self.leg_results: List[Dict[str, Any]] = []
        self.state = "INIT"
        self.state_start_time = time.time()
        self.nav_start_sim_time = 0.0
        self.leg_start_sim_time = 0.0
        self.leg_start_pos = None
        self.ground_z = 0.0

        self.active_future = None
        self.mission_result: Dict[str, Any] = {}
        self.is_finished = False

        # 20 Hz State Machine Timer
        self.timer = self.create_timer(0.05, self.state_machine_loop)
        self.get_logger().info(
            f"MissionEvaluator initialized for [{self.controller}] with {len(self.waypoints)} waypoints. Mission: '{self.mission_name}'"
        )

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
        prev_state = self.state
        self.state = new_state
        self.state_start_time = time.time()
        self.active_future = None
        self.get_logger().info(f"[STATE] {prev_state} -> {new_state}")

    def current_target_goal(self) -> np.ndarray:
        if self.current_wp_idx < len(self.waypoints):
            return self.waypoints[self.current_wp_idx]
        return self.waypoints[-1]

    def publish_active_goal(self):
        target = self.current_target_goal()
        pt = Point(x=float(target[0]), y=float(target[1]), z=float(target[2]))
        self.pub_goal.publish(pt)
        self.get_logger().info(f"Target Waypoint [{self.current_wp_idx + 1}/{len(self.waypoints)}] published: [{target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}]")

    def state_machine_loop(self):
        if self.is_finished:
            return

        now = time.time()
        elapsed_state = now - self.state_start_time
        target_goal = self.current_target_goal()

        # Record high-frequency telemetry during navigation
        if self.state in ["NAVIGATING", "HOVER_AT_GOAL"] and self.latest_odom is not None:
            pos = self.latest_odom.pose.pose.position
            vel = self.latest_odom.twist.twist.linear
            dist = float(np.linalg.norm(target_goal - np.array([pos.x, pos.y, pos.z])))
            speed = float(np.hypot(np.hypot(vel.x, vel.y), vel.z))

            entry = {
                "time_sim": round(self.get_sim_time(), 3),
                "time_wall": round(now, 3),
                "state": self.state,
                "wp_index": self.current_wp_idx,
                "target_goal_enu": [round(float(v), 3) for v in target_goal],
                "pos_enu": [round(pos.x, 3), round(pos.y, 3), round(pos.z, 3)],
                "vel_enu": [round(vel.x, 3), round(vel.y, 3), round(vel.z, 3)],
                "dist_to_goal_m": round(dist, 3),
                "speed_m_s": round(speed, 3),
                "latency_ms": self.latest_diag.get("inference_latency_ms", 0.0) if self.latest_diag else 0.0,
                "ppo_action": self.latest_diag.get("ppo_action", []) if self.latest_diag else [],
                "ppo_cmd": self.latest_diag.get("ppo_cmd", []) if self.latest_diag else [],
                "watchdog_tripped": self.latest_diag.get("watchdog_tripped", False) if self.latest_diag else False
            }
            self.trajectory_history.append(entry)

        # 1. INIT -> Set parameters and wait for telemetry
        if self.state == "INIT":
            if self.latest_status and self.latest_status.get("connected", False) and self.latest_odom is not None:
                self.ground_z = self.latest_odom.pose.pose.position.z
                # Select active controller
                ctrl_msg = String()
                ctrl_msg.data = self.controller
                self.pub_select_ctrl.publish(ctrl_msg)
                # Publish initial goal
                self.publish_active_goal()
                self.set_state("CALL_ARM")
            elif elapsed_state > 15.0:
                self.abort_mission("PX4_NOT_CONNECTED_TIMEOUT")

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
                self.get_logger().info("Drone successfully ARMED. Commanding TAKEOFF...")
                self.set_state("CALL_TAKEOFF")
            elif self.active_future and self.active_future.done():
                res = self.active_future.result()
                if res and res.success:
                    self.get_logger().info("Arm service acknowledged success. Waiting for armed state...")
                elif elapsed_state > 2.0:
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

        # 6. CLIMBING to ~2.0m hover
        elif self.state == "CLIMBING":
            if self.latest_odom is not None:
                rel_alt = self.latest_odom.pose.pose.position.z - self.ground_z
                mode = self.latest_status.get("flight_mode", "") if self.latest_status else ""
                if (rel_alt >= 0.8 and mode == "AUTO_LOITER") or rel_alt >= 1.2:
                    self.get_logger().info(f"Climb altitude reached ({rel_alt:.2f}m above ground, mode={mode}). Hovering...")
                    self.set_state("HOVER_STABILIZE")
            if elapsed_state > 50.0:
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
            elif elapsed_state > 12.0:
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
                self.get_logger().info(f"[{self.controller}] Autonomous Flight Authority Engaged. Navigating to WP 1...")
                pos = self.latest_odom.pose.pose.position
                self.leg_start_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
                self.leg_start_sim_time = self.get_sim_time()
                self.set_state("NAVIGATING")
            elif self.active_future and self.active_future.done():
                res = self.active_future.result()
                if res and res.success:
                    self.get_logger().info("Enable control acknowledged. Waiting for confirmation...")
                elif res and not res.success and elapsed_state > 2.0:
                    self.get_logger().warn(f"Enable control failed: {res.message}. Retrying...")
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
            speed = float(np.hypot(np.hypot(vel.x, vel.y), vel.z))

            # Success threshold: distance <= 0.40m and speed <= 0.50m/s
            if dist <= 0.40 and speed <= 0.50:
                self.get_logger().info(f"WP [{self.current_wp_idx + 1}/{len(self.waypoints)}] reached! Dist={dist:.2f}m, Speed={speed:.2f}m/s. Stabilizing at waypoint...")
                self.set_state("HOVER_AT_GOAL")
            elif elapsed_state > 45.0:  # Timeout after 45s of navigation for this leg
                self.get_logger().warn(f"WP [{self.current_wp_idx + 1}/{len(self.waypoints)}] leg timed out after {elapsed_state:.1f}s. Final dist: {dist:.2f}m.")
                self.set_state("HOVER_AT_GOAL")

        # 13. HOVER_AT_GOAL (2.0s stabilization and transition)
        elif self.state == "HOVER_AT_GOAL":
            if elapsed_state >= 2.0:
                # Compile leg results
                self.record_leg_result()

                self.current_wp_idx += 1
                if self.current_wp_idx < len(self.waypoints):
                    # Proceed to next waypoint
                    next_target = self.current_target_goal()
                    pos = self.latest_odom.pose.pose.position
                    self.leg_start_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
                    self.leg_start_sim_time = self.get_sim_time()
                    self.publish_active_goal()
                    self.set_state("NAVIGATING")
                else:
                    self.get_logger().info("All mission waypoints completed! Disengaging autonomous flight authority...")
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
            if (rel_alt < 0.25 and elapsed_state > 3.0) or (landed == "ON_GROUND" and rel_alt < 0.35):
                self.get_logger().info(f"Touchdown confirmed (rel_alt={rel_alt:.3f}m, landed={landed}). Disarming...")
                self.set_state("CALL_DISARM")
            elif elapsed_state > 45.0:
                self.get_logger().warn(f"Landing timeout (rel_alt={rel_alt:.3f}m). Forcing disarm...")
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
            armed = self.latest_status.get("armed", False) if self.latest_status else True
            if not armed or (self.active_future and self.active_future.done()):
                self.get_logger().info("Disarm confirmed. Mission complete.")
                self.set_state("COMPLETE")
            elif elapsed_state > 12.0:
                self.set_state("COMPLETE")

        # 22. COMPLETE
        elif self.state == "COMPLETE":
            self.compile_mission_results()
            self.is_finished = True

    def record_leg_result(self):
        target = self.current_target_goal()
        wp_idx = self.current_wp_idx
        leg_pts = [p for p in self.trajectory_history if p["wp_index"] == wp_idx and p["state"] in ["NAVIGATING", "HOVER_AT_GOAL"]]

        if len(leg_pts) > 1:
            pts_arr = np.array([p["pos_enu"] for p in leg_pts])
            path_len = float(np.sum(np.linalg.norm(np.diff(pts_arr, axis=0), axis=1)))
            straight_line = float(np.linalg.norm(target - self.leg_start_pos)) if self.leg_start_pos is not None else 1.0
            efficiency = (path_len / straight_line) if straight_line > 0.1 else 1.0
            final_pos = pts_arr[-1]
            final_err = float(np.linalg.norm(target - final_pos))
        else:
            path_len = 0.0
            efficiency = 1.0
            pos = self.latest_odom.pose.pose.position if self.latest_odom else None
            final_pos = [pos.x, pos.y, pos.z] if pos else [0.0, 0.0, 0.0]
            final_err = float(np.linalg.norm(target - np.array(final_pos)))

        latencies = [p["latency_ms"] for p in leg_pts if p["latency_ms"] > 0]
        speeds = [p["speed_m_s"] for p in leg_pts]
        leg_duration = (self.get_sim_time() - self.leg_start_sim_time) if self.leg_start_sim_time > 0 else 0.0
        is_success = (final_err <= 0.40)

        leg_res = {
            "waypoint_index": wp_idx + 1,
            "target_goal_enu": [round(float(v), 3) for v in target],
            "start_pos_enu": [round(float(v), 3) for v in self.leg_start_pos] if self.leg_start_pos is not None else [],
            "final_pos_enu": [round(float(v), 3) for v in final_pos],
            "success": is_success,
            "final_goal_error_m": round(final_err, 3),
            "leg_duration_sim_s": round(leg_duration, 2),
            "trajectory_length_m": round(path_len, 3),
            "path_efficiency_ratio": round(efficiency, 3),
            "max_speed_m_s": round(float(np.max(speeds)), 3) if speeds else 0.0,
            "mean_speed_m_s": round(float(np.mean(speeds)), 3) if speeds else 0.0,
            "mean_inference_latency_ms": round(float(np.mean(latencies)), 3) if latencies else 0.0,
            "p95_inference_latency_ms": round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
            "samples_count": len(leg_pts)
        }
        self.leg_results.append(leg_res)
        self.get_logger().info(
            f"=== WP {wp_idx + 1} RESULT: Success={is_success} | Final Err={final_err:.3f}m | Duration={leg_duration:.2f}s | Path={path_len:.3f}m | Eff={efficiency:.3f} ==="
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

        self.mission_result = {
            "mission_name": self.mission_name,
            "controller": self.controller,
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
            "p95_inference_latency_ms": round(float(np.percentile(all_latencies, 95)), 3) if all_latencies else 0.0,
            "legs": self.leg_results,
            "trajectory_samples_count": len(self.trajectory_history),
            "trajectory_samples": self.trajectory_history
        }

        out_dir = os.path.expanduser("~/proj/experiments/milestone3/trials")
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{self.mission_name}.json")
        with open(out_file, "w") as f:
            json.dump(self.mission_result, f, indent=2)

        self.get_logger().info(f"==================================================")
        self.get_logger().info(f"=== MISSION {self.mission_name} COMPLETE ===")
        self.get_logger().info(f"Controller: {self.controller} | Overall Success: {overall_success}")
        self.get_logger().info(f"Success Rate: {self.mission_result['success_rate_percent']}% ({self.mission_result['successful_waypoints']}/{len(self.waypoints)})")
        self.get_logger().info(f"Total Nav Duration: {total_nav_duration:.2f} s | Total Path Length: {total_path_len:.3f} m")
        self.get_logger().info(f"Mean Latency: {self.mission_result['mean_inference_latency_ms']} ms (p95: {self.mission_result['p95_inference_latency_ms']} ms)")
        self.get_logger().info(f"Saved to: {out_file}")
        self.get_logger().info(f"==================================================")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=str, choices=["PPO", "CLASSICAL"], default="PPO")
    parser.add_argument("--name", type=str, default="eval_mission")
    args = parser.parse_args()

    rclpy.init()
    node = MissionEvaluator(
        controller=args.controller,
        waypoints=DEFAULT_EVAL_WAYPOINTS,
        mission_name=args.name
    )

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
