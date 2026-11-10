#!/usr/bin/env python3
"""
diagnostic_adapter.py
---------------------
PPO inference adapter for Milestone 3B diagnostic experiments.
Supports three strictly controlled test modes:
  - TEST_A: Baseline reproduction (Raw PPO actions in Body FLU -> MAV_FRAME_BODY_NED).
  - TEST_B: Yaw coupling isolation (PPO translational velocities preserved, yaw rate
            replaced by bounded heading alignment to target bearing -> MAV_FRAME_BODY_NED).
  - TEST_C: Coordinate-frame hypothesis (PPO translational velocities transformed to World ENU
            -> MAV_FRAME_LOCAL_NED, raw PPO yaw rate preserved).
"""

import json
import math
import os
import sys
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, Header, String
from std_srvs.srv import SetBool, Trigger

import os
import sys

# Ensure project root and venv site-packages are in sys.path
proj_root = os.path.expanduser("~/proj")
venv_site = os.path.join(proj_root, "venv/lib/python3.12/site-packages")
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

# RL environment components
from rl_env.action import ActionHandler, CommandedVelocities
from rl_env.config import UAVEnvConfig
from rl_env.observation import ObservationHandler, quat_to_rotation_matrix


def wrap_to_pi(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class DiagnosticAdapter(Node):
    VALID_TEST_MODES = ["TEST_A", "TEST_B", "TEST_C", "TEST_D"]

    def __init__(self):
        super().__init__('diagnostic_adapter')

        # Parameters
        self.declare_parameter('test_mode', 'TEST_A')
        self.declare_parameter('model_path', os.path.join(proj_root, 'experiments/milestone2/models/ppo_uav_final.zip'))
        self.declare_parameter('goal_x', 3.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('goal_z', 2.5)
        self.declare_parameter('ground_z', 0.0)
        self.declare_parameter('policy_rate_hz', 10.0)
        self.declare_parameter('max_velocity_clamp', 1.5)
        self.declare_parameter('telemetry_timeout_s', 0.25)

        self.test_mode = str(self.get_parameter('test_mode').get_parameter_value().string_value).upper()
        if self.test_mode not in self.VALID_TEST_MODES:
            raise ValueError(f"Invalid test_mode '{self.test_mode}'. Must be one of {self.VALID_TEST_MODES}")

        self.model_path = str(self.get_parameter('model_path').get_parameter_value().string_value)
        self.goal_world = np.array([
            self.get_parameter('goal_x').get_parameter_value().double_value,
            self.get_parameter('goal_y').get_parameter_value().double_value,
            self.get_parameter('goal_z').get_parameter_value().double_value
        ], dtype=np.float32)
        self.ground_z = float(self.get_parameter('ground_z').get_parameter_value().double_value)
        self.policy_rate = float(self.get_parameter('policy_rate_hz').get_parameter_value().double_value)
        self.max_velocity_clamp = float(self.get_parameter('max_velocity_clamp').get_parameter_value().double_value)
        self.telemetry_timeout_s = float(self.get_parameter('telemetry_timeout_s').get_parameter_value().double_value)

        # Initialize RL handlers
        self.config = UAVEnvConfig()
        self.obs_handler = ObservationHandler(self.config)
        self.action_handler = ActionHandler(self.config)

        # Load PPO model
        self.load_ppo_model()

        # Telemetry Cache
        self.latest_odom: Optional[Odometry] = None
        self.last_odom_receipt_time: float = 0.0
        self.latest_status: Optional[dict] = None
        self.prev_action = np.zeros(4, dtype=np.float32)

        # Safety & State
        self.control_enabled: bool = False
        self.watchdog_tripped: bool = False
        self.watchdog_reason: str = "NONE"
        self.total_inference_cycles: int = 0
        self.total_dropped_cycles: int = 0
        self.inference_latencies_ms = []

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
        self.sub_goal = self.create_subscription(Point, '/uav/set_goal', self.goal_callback, 10)

        # Publishers
        self.pub_obs = self.create_publisher(Float32MultiArray, '/uav/observation', 10)
        self.pub_diag = self.create_publisher(String, '/uav/adapter_diagnostics', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_cmd_world = self.create_publisher(Twist, '/cmd_vel_world', 10)

        # Services
        self.srv_enable_control = self.create_service(SetBool, '/uav/enable_ppo_control', self.handle_enable_control)
        self.srv_set_goal = self.create_service(Trigger, '/uav/reset_goal', self.handle_reset_goal)

        # 10 Hz Policy Timer
        dt_policy = 1.0 / self.policy_rate
        self.timer_policy = self.create_timer(dt_policy, self.policy_step_callback)

        self.get_logger().info(
            f"DiagnosticAdapter started in [{self.test_mode}] mode. "
            f"Goal: [{self.goal_world[0]:.2f}, {self.goal_world[1]:.2f}, {self.goal_world[2]:.2f}] ENU. "
            f"Policy rate: {self.policy_rate} Hz."
        )

    def load_ppo_model(self):
        try:
            from stable_baselines3 import PPO
            self.get_logger().info(f"Loading frozen PPO model: {self.model_path}")
            self.model = PPO.load(self.model_path, device="cpu")
            self.get_logger().info("PPO model loaded successfully on CPU.")
        except Exception as e:
            self.get_logger().error(f"Failed to load PPO model: {e}")
            raise e

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg
        self.last_odom_receipt_time = time.time()

    def status_callback(self, msg: String):
        try:
            self.latest_status = json.loads(msg.data)
        except Exception:
            pass

    def goal_callback(self, msg: Point):
        self.goal_world = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.get_logger().info(f"Goal updated to: [{msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}] ENU")

    def handle_enable_control(self, request, response):
        if request.data:
            if self.latest_odom is None:
                response.success = False
                response.message = "Cannot enable flight control: No odometry received yet."
                return response

            p = self.latest_odom.pose.pose.position
            pos_enu = np.array([p.x, p.y, p.z], dtype=np.float32)
            q_msg = self.latest_odom.pose.pose.orientation
            q_wxyz = np.array([q_msg.w, q_msg.x, q_msg.y, q_msg.z], dtype=np.float32)
            v = self.latest_odom.twist.twist.linear
            speed = float(np.sqrt(v.x**2 + v.y**2 + v.z**2))

            is_fault, reason = self.check_safety_watchdogs(pos_enu, q_wxyz, speed)
            if is_fault:
                response.success = False
                response.message = f"Cannot enable control: {reason}"
                return response

            self.watchdog_tripped = False
            self.watchdog_reason = "NONE"
            self.control_enabled = True
            response.success = True
            response.message = f"[{self.test_mode}] Diagnostic Flight Authority ENGAGED."
            self.get_logger().warn(response.message)
        else:
            self.control_enabled = False
            self.publish_zero_cmd()
            response.success = True
            response.message = "Flight Authority DISENGAGED. Neutral hover."
            self.get_logger().info(response.message)
        return response

    def handle_reset_goal(self, request, response):
        gx = self.get_parameter('goal_x').get_parameter_value().double_value
        gy = self.get_parameter('goal_y').get_parameter_value().double_value
        gz = self.get_parameter('goal_z').get_parameter_value().double_value
        self.goal_world = np.array([gx, gy, gz], dtype=np.float32)
        response.success = True
        response.message = f"Goal reset to [{gx:.2f}, {gy:.2f}, {gz:.2f}] ENU."
        return response

    def build_observation(self) -> Optional[np.ndarray]:
        if self.latest_odom is None:
            return None

        now = time.time()
        telemetry_age = now - self.last_odom_receipt_time
        if telemetry_age > self.telemetry_timeout_s:
            self.watchdog_tripped = True
            self.watchdog_reason = f"STALE_TELEMETRY (age={telemetry_age*1000:.1f}ms > {self.telemetry_timeout_s*1000:.1f}ms)"
            return None

        p = self.latest_odom.pose.pose.position
        pos_enu = np.array([p.x, p.y, p.z], dtype=np.float32)

        v = self.latest_odom.twist.twist.linear
        vel_enu = np.array([v.x, v.y, v.z], dtype=np.float32)

        q_msg = self.latest_odom.pose.pose.orientation
        q_wxyz = np.array([q_msg.w, q_msg.x, q_msg.y, q_msg.z], dtype=np.float32)

        w = self.latest_odom.twist.twist.angular
        rates_flu = np.array([w.x, w.y, w.z], dtype=np.float32)

        obs = self.obs_handler.compute_observation(
            pos_world=pos_enu,
            vel_world=vel_enu,
            quat=q_wxyz,
            rates_body=rates_flu,
            goal_world=self.goal_world,
            prev_action=self.prev_action,
            ground_z=self.ground_z
        )
        return obs

    def check_safety_watchdogs(self, pos_enu: np.ndarray, q_wxyz: np.ndarray, speed: float) -> Tuple[bool, str]:
        x, y, z = float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2])
        if abs(x) > 10.0 or abs(y) > 10.0 or z > 10.0:
            return True, f"GEOFENCE_VIOLATION (pos=[{x:.2f}, {y:.2f}, {z:.2f}])"

        if self.control_enabled and z < 0.3:
            return True, f"ALTITUDE_TOO_LOW (z={z:.2f}m < 0.3m)"

        qw, qx, qy, qz = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
        if abs(roll) > math.radians(40.0) or abs(pitch) > math.radians(40.0):
            return True, f"EXCESSIVE_TILT (roll={math.degrees(roll):.1f}deg, pitch={math.degrees(pitch):.1f}deg)"

        if speed > 4.5:
            return True, f"SPEED_LIMIT_EXCEEDED (speed={speed:.2f}m/s)"

        return False, "OK"

    def publish_zero_cmd(self):
        stop_twist = Twist()
        self.pub_cmd_vel.publish(stop_twist)
        self.pub_cmd_world.publish(stop_twist)

    def policy_step_callback(self):
        self.total_inference_cycles += 1

        obs = self.build_observation()
        if obs is None:
            self.total_dropped_cycles += 1
            if self.control_enabled:
                self.publish_zero_cmd()
            return

        obs_msg = Float32MultiArray()
        obs_msg.data = [float(val) for val in obs]
        self.pub_obs.publish(obs_msg)

        # State Extraction
        p = self.latest_odom.pose.pose.position
        pos_enu = np.array([p.x, p.y, p.z], dtype=np.float32)
        v = self.latest_odom.twist.twist.linear
        vel_enu = np.array([v.x, v.y, v.z], dtype=np.float32)
        speed = float(np.linalg.norm(vel_enu))

        q_msg = self.latest_odom.pose.pose.orientation
        q_wxyz = np.array([q_msg.w, q_msg.x, q_msg.y, q_msg.z], dtype=np.float32)
        w = self.latest_odom.twist.twist.angular
        rates_flu = np.array([w.x, w.y, w.z], dtype=np.float32)

        # Compute Euler yaw in ENU (rad)
        qw, qx, qy, qz = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
        yaw_enu = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))

        # Target bearing and heading error
        rel_target_world = self.goal_world - pos_enu
        dist_to_goal = float(np.linalg.norm(rel_target_world))
        bearing_to_goal = math.atan2(float(rel_target_world[1]), float(rel_target_world[0]))
        heading_error = wrap_to_pi(bearing_to_goal - yaw_enu)

        # Rotation matrix Body to World ENU
        R_body_to_world = quat_to_rotation_matrix(q_wxyz)
        vel_flu = R_body_to_world.T @ vel_enu

        # Watchdogs
        is_fault, reason = self.check_safety_watchdogs(pos_enu, q_wxyz, speed)
        if is_fault:
            self.watchdog_tripped = True
            self.watchdog_reason = reason
            if self.control_enabled:
                self.get_logger().error(f"EMERGENCY DISENGAGE: {reason}")
                self.control_enabled = False
                self.publish_zero_cmd()
        else:
            self.watchdog_tripped = False
            self.watchdog_reason = "NONE"

        # PPO Inference
        t0 = time.perf_counter()
        ppo_act_norm, _ = self.model.predict(obs, deterministic=True)
        t1 = time.perf_counter()
        lat_ms = (t1 - t0) * 1000.0
        self.inference_latencies_ms.append(lat_ms)
        if len(self.inference_latencies_ms) > 200:
            self.inference_latencies_ms.pop(0)

        ppo_cmd = self.action_handler.scale_action(ppo_act_norm)

        # Mode-Specific Action Processing
        cmd_sent = [0.0, 0.0, 0.0, 0.0]
        cmd_frame_name = "NONE"

        if self.control_enabled and not self.watchdog_tripped:
            if self.test_mode == "TEST_A":
                # Baseline: Body FLU clamped, raw policy yaw rate
                vx = float(np.clip(ppo_cmd.vx, -self.max_velocity_clamp, self.max_velocity_clamp))
                vy = float(np.clip(ppo_cmd.vy, -self.max_velocity_clamp, self.max_velocity_clamp))
                vz = float(np.clip(ppo_cmd.vz, -self.max_velocity_clamp, self.max_velocity_clamp))
                yaw_rate = float(ppo_cmd.yaw_rate)

                twist = Twist()
                twist.linear.x = vx
                twist.linear.y = vy
                twist.linear.z = vz
                twist.angular.z = yaw_rate
                self.pub_cmd_vel.publish(twist)

                cmd_sent = [vx, vy, vz, yaw_rate]
                cmd_frame_name = "MAV_FRAME_BODY_NED"
                self.prev_action = ppo_act_norm.copy()

            elif self.test_mode == "TEST_B":
                # Yaw Isolation: PPO vx, vy, vz preserved; yaw rate replaced with bounded heading alignment
                vx = float(np.clip(ppo_cmd.vx, -self.max_velocity_clamp, self.max_velocity_clamp))
                vy = float(np.clip(ppo_cmd.vy, -self.max_velocity_clamp, self.max_velocity_clamp))
                vz = float(np.clip(ppo_cmd.vz, -self.max_velocity_clamp, self.max_velocity_clamp))

                # Bounded proportional alignment to target bearing
                if abs(heading_error) < 0.08:  # Deadband: 4.5 degrees
                    yaw_rate_cmd = 0.0
                else:
                    yaw_rate_cmd = float(np.clip(1.0 * heading_error, -0.6, 0.6))

                twist = Twist()
                twist.linear.x = vx
                twist.linear.y = vy
                twist.linear.z = vz
                twist.angular.z = yaw_rate_cmd
                self.pub_cmd_vel.publish(twist)

                cmd_sent = [vx, vy, vz, yaw_rate_cmd]
                cmd_frame_name = "MAV_FRAME_BODY_NED"
                # Keep prev_action consistent with actual commanded yaw
                norm_yaw = float(np.clip(yaw_rate_cmd / self.config.max_yaw_rate, -1.0, 1.0))
                self.prev_action = np.array([ppo_act_norm[0], ppo_act_norm[1], ppo_act_norm[2], norm_yaw], dtype=np.float32)

            elif self.test_mode == "TEST_C":
                # World Frame: PPO body velocity transformed to World ENU
                v_body_flu = np.array([ppo_cmd.vx, ppo_cmd.vy, ppo_cmd.vz], dtype=np.float32)
                v_world_enu = R_body_to_world @ v_body_flu

                vx_w = float(np.clip(v_world_enu[0], -self.max_velocity_clamp, self.max_velocity_clamp))
                vy_w = float(np.clip(v_world_enu[1], -self.max_velocity_clamp, self.max_velocity_clamp))
                vz_w = float(np.clip(v_world_enu[2], -self.max_velocity_clamp, self.max_velocity_clamp))
                yaw_rate = float(ppo_cmd.yaw_rate)

                twist_w = Twist()
                twist_w.linear.x = vx_w
                twist_w.linear.y = vy_w
                twist_w.linear.z = vz_w
                twist_w.angular.z = yaw_rate
                self.pub_cmd_world.publish(twist_w)

                cmd_sent = [vx_w, vy_w, vz_w, yaw_rate]
                cmd_frame_name = "MAV_FRAME_LOCAL_NED"
                self.prev_action = ppo_act_norm.copy()

            elif self.test_mode == "TEST_D":
                # Configuration C: Yaw-governed PPO + Proximity Deceleration Governor
                # 1. Bounded proportional alignment to target bearing
                if abs(heading_error) < 0.08:  # Deadband: 4.58 degrees
                    yaw_rate_cmd = 0.0
                else:
                    yaw_rate_cmd = float(np.clip(1.0 * heading_error, -0.6, 0.6))

                # 2. Distance-dependent speed scaling factor and terminal capture governor
                d_brake = 2.0
                if dist_to_goal <= 0.40:
                    # Within arrival sphere (0.40m): Proportional terminal capture to center target
                    # Rotate relative target vector into body FLU frame
                    rel_target_body = R_body_to_world.T @ rel_target_world
                    kp_pos = 1.0
                    vx = float(np.clip(kp_pos * rel_target_body[0], -0.35, 0.35))
                    vy = float(np.clip(kp_pos * rel_target_body[1], -0.35, 0.35))
                    vz = float(np.clip(kp_pos * rel_target_body[2], -0.30, 0.30))
                else:
                    # Transit regime: Governed PPO velocity with continuous proximity braking
                    speed_scale = float(np.clip(dist_to_goal / d_brake, 0.20, 1.0))
                    vx = float(np.clip(ppo_cmd.vx * speed_scale, -self.max_velocity_clamp, self.max_velocity_clamp))
                    vy = float(np.clip(ppo_cmd.vy * speed_scale, -self.max_velocity_clamp, self.max_velocity_clamp))
                    vz = float(np.clip(ppo_cmd.vz * speed_scale, -self.max_velocity_clamp, self.max_velocity_clamp))

                twist = Twist()
                twist.linear.x = vx
                twist.linear.y = vy
                twist.linear.z = vz
                twist.angular.z = yaw_rate_cmd
                self.pub_cmd_vel.publish(twist)

                cmd_sent = [vx, vy, vz, yaw_rate_cmd]
                cmd_frame_name = "MAV_FRAME_BODY_NED"

                # Update prev_action to reflect actual executed normalized command
                self.prev_action = self.action_handler.unscale_action(
                    CommandedVelocities(vx=vx, vy=vy, vz=vz, yaw_rate=yaw_rate_cmd)
                )
        else:
            if not self.control_enabled:
                self.publish_zero_cmd()

        # Telemetry Diagnostics
        diag = {
            "test_mode": self.test_mode,
            "control_enabled": self.control_enabled,
            "command_frame": cmd_frame_name,
            "watchdog_tripped": self.watchdog_tripped,
            "watchdog_reason": self.watchdog_reason,
            "dist_to_goal_m": round(dist_to_goal, 3),
            "bearing_to_goal_rad": round(bearing_to_goal, 3),
            "heading_error_rad": round(heading_error, 3),
            "speed_scale": round(float(np.clip(dist_to_goal / 2.0, 0.20, 1.0)), 3) if self.test_mode == "TEST_D" else 1.0,
            "pos_enu": [round(float(p), 3) for p in pos_enu],
            "goal_enu": [round(float(g), 3) for g in self.goal_world],
            "vel_enu": [round(float(v), 3) for v in vel_enu],
            "vel_flu": [round(float(v), 3) for v in vel_flu],
            "speed_m_s": round(speed, 3),
            "yaw_enu_rad": round(yaw_enu, 3),
            "yaw_enu_deg": round(math.degrees(yaw_enu), 1),
            "rates_flu": [round(float(r), 3) for r in rates_flu],
            "inference_latency_ms": round(lat_ms, 3),
            "ppo_action_raw": [round(float(a), 3) for a in ppo_act_norm],
            "ppo_cmd_scaled": [round(ppo_cmd.vx, 3), round(ppo_cmd.vy, 3), round(ppo_cmd.vz, 3), round(ppo_cmd.yaw_rate, 3)],
            "cmd_sent": [round(c, 3) for c in cmd_sent]
        }

        diag_msg = String()
        diag_msg.data = json.dumps(diag)
        self.pub_diag.publish(diag_msg)


def main(args=None):
    rclpy.init(args=args)
    adapter = DiagnosticAdapter()
    try:
        rclpy.spin(adapter)
    except KeyboardInterrupt:
        pass
    finally:
        adapter.publish_zero_cmd()
        adapter.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
