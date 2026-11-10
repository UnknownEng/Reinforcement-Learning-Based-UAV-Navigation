#!/usr/bin/env python3
"""
ppo_inference_adapter.py
------------------------
ROS 2 Jazzy adapter bridging live PX4 SITL + Gazebo telemetry to the trained PPO policy.
Supports 3 strictly staged execution modes:
  1. OBSERVATION_ONLY (Stage B): Telemetry ingestion, 18D observation construction & logging. ZERO commands.
  2. SHADOW (Stage C): Live PPO inference & latency profiling, comparison against classical controller. ZERO commands.
  3. CONTROLLED_SIM (Stage D/E): Supervised policy execution with software enable switch, geofence watchdog,
                                 velocity clamps, and failsafe hover disengagement.
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

# RL environment components
from rl_env.action import ActionHandler, CommandedVelocities
from rl_env.baselines import ProportionalWaypointController
from rl_env.config import UAVEnvConfig
from rl_env.observation import ObservationHandler, quat_to_rotation_matrix


class PPOInferenceAdapter(Node):
    VALID_MODES = ["OBSERVATION_ONLY", "SHADOW", "CONTROLLED_SIM"]

    def __init__(self):
        super().__init__('ppo_inference_adapter')

        # Declare ROS Parameters
        self.declare_parameter('mode', 'OBSERVATION_ONLY')
        self.declare_parameter('controller', 'PPO')
        self.declare_parameter('model_path', os.path.expanduser('~/proj/experiments/milestone2/models/ppo_uav_final.zip'))
        self.declare_parameter('goal_x', 3.0)
        self.declare_parameter('goal_y', 2.0)
        self.declare_parameter('goal_z', 2.5)
        self.declare_parameter('ground_z', 0.0)
        self.declare_parameter('policy_rate_hz', 10.0)
        self.declare_parameter('max_velocity_clamp', 1.5)  # Conservative sim test clamp (m/s)
        self.declare_parameter('telemetry_timeout_s', 0.25) # Watchdog timeout (s)

        self.mode = str(self.get_parameter('mode').get_parameter_value().string_value).upper()
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{self.mode}'. Must be one of {self.VALID_MODES}")

        self.active_controller = str(self.get_parameter('controller').get_parameter_value().string_value).upper()

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
        self.classical_controller = ProportionalWaypointController(self.config)

        # PPO Model handle
        self.model = None
        if self.mode in ["SHADOW", "CONTROLLED_SIM"]:
            self.load_ppo_model()

        # Telemetry Cache
        self.latest_odom: Optional[Odometry] = None
        self.last_odom_receipt_time: float = 0.0
        self.latest_status: Optional[dict] = None
        self.prev_action = np.zeros(4, dtype=np.float32)

        # Safety & Control State
        self.control_enabled: bool = False
        self.watchdog_tripped: bool = False
        self.watchdog_reason: str = "NONE"
        self.total_inference_cycles: int = 0
        self.total_dropped_cycles: int = 0
        self.inference_latencies_ms = []

        # QoS Profiles
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
        self.sub_select_ctrl = self.create_subscription(String, '/uav/select_controller', self.select_ctrl_callback, 10)

        # Publishers
        self.pub_obs = self.create_publisher(Float32MultiArray, '/uav/observation', 10)
        self.pub_diag = self.create_publisher(String, '/uav/adapter_diagnostics', 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        # Services
        self.srv_enable_control = self.create_service(SetBool, '/uav/enable_ppo_control', self.handle_enable_control)
        self.srv_set_goal = self.create_service(Trigger, '/uav/reset_goal', self.handle_reset_goal)

        # 10 Hz Policy Execution Timer
        dt_policy = 1.0 / self.policy_rate
        self.timer_policy = self.create_timer(dt_policy, self.policy_step_callback)

        self.get_logger().info(
            f"PPO Inference Adapter started in [{self.mode}] mode. "
            f"Goal: [{self.goal_world[0]:.2f}, {self.goal_world[1]:.2f}, {self.goal_world[2]:.2f}] ENU. "
            f"Policy rate: {self.policy_rate} Hz."
        )

    def load_ppo_model(self):
        try:
            from stable_baselines3 import PPO
            self.get_logger().info(f"Loading PPO model checkpoint: {self.model_path}")
            self.model = PPO.load(self.model_path, device="cpu")
            self.get_logger().info("PPO model successfully loaded into memory on CPU.")
        except Exception as e:
            self.get_logger().error(f"Failed to load PPO model from '{self.model_path}': {e}")
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
        self.get_logger().info(f"Goal set via topic to: [{msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f}] ENU")

    def select_ctrl_callback(self, msg: String):
        val = msg.data.strip().upper()
        if val in ["PPO", "CLASSICAL"]:
            self.active_controller = val
            self.get_logger().info(f"Active controller switched to: {self.active_controller}")

    def handle_enable_control(self, request, response):
        """Operator software switch for enabling/disabling PPO flight authority."""
        if self.mode != "CONTROLLED_SIM":
            response.success = False
            response.message = f"Rejected: Adapter is running in '{self.mode}' mode. Must be 'CONTROLLED_SIM'."
            self.get_logger().warn(response.message)
            return response

        if request.data:
            if self.latest_odom is None:
                response.success = False
                response.message = "Cannot enable flight control: No odometry received yet."
                self.get_logger().error(response.message)
                return response

            p = self.latest_odom.pose.pose.position
            pos_enu = np.array([p.x, p.y, p.z], dtype=np.float32)
            q_msg = self.latest_odom.pose.pose.orientation
            q_wxyz = np.array([q_msg.w, q_msg.x, q_msg.y, q_msg.z], dtype=np.float32)
            vel_enu = np.array([
                self.latest_odom.twist.twist.linear.x,
                self.latest_odom.twist.twist.linear.y,
                self.latest_odom.twist.twist.linear.z
            ], dtype=np.float32)
            speed = float(np.linalg.norm(vel_enu))

            is_fault, reason = self.check_safety_watchdogs(pos_enu, q_wxyz, speed)
            if is_fault:
                response.success = False
                response.message = f"Cannot enable flight control: {reason}"
                self.get_logger().error(response.message)
                return response

            self.watchdog_tripped = False
            self.watchdog_reason = "NONE"
            self.control_enabled = True
            response.success = True
            response.message = f"[{self.active_controller}] Flight Control ENGAGED. Authority active."
            self.get_logger().warn(response.message)
        else:
            self.control_enabled = False
            self.publish_zero_cmd()
            response.success = True
            response.message = "Flight Control DISENGAGED. Neutral hover commanded."
            self.get_logger().info(response.message)

        return response

    def handle_reset_goal(self, request, response):
        """Re-read goal parameters from ROS node parameters."""
        gx = self.get_parameter('goal_x').get_parameter_value().double_value
        gy = self.get_parameter('goal_y').get_parameter_value().double_value
        gz = self.get_parameter('goal_z').get_parameter_value().double_value
        self.goal_world = np.array([gx, gy, gz], dtype=np.float32)
        response.success = True
        response.message = f"Goal updated to [{gx:.2f}, {gy:.2f}, {gz:.2f}] ENU."
        self.get_logger().info(response.message)
        return response

    def build_observation(self) -> Optional[np.ndarray]:
        """
        Construct the verified 18-dimensional observation vector from latest Odometry.
        Performs scalar-last to scalar-first quaternion reordering and coordinate frame mapping.
        """
        if self.latest_odom is None:
            return None

        now = time.time()
        telemetry_age = now - self.last_odom_receipt_time
        if telemetry_age > self.telemetry_timeout_s:
            self.watchdog_tripped = True
            self.watchdog_reason = f"STALE_TELEMETRY (age={telemetry_age*1000:.1f}ms > {self.telemetry_timeout_s*1000:.1f}ms)"
            return None

        # Extract World ENU Position
        p = self.latest_odom.pose.pose.position
        pos_enu = np.array([p.x, p.y, p.z], dtype=np.float32)

        # Extract World ENU Velocity
        v = self.latest_odom.twist.twist.linear
        vel_enu = np.array([v.x, v.y, v.z], dtype=np.float32)

        # Extract Orientation Quaternion: ROS 2 convention is [x, y, z, w]
        q_msg = self.latest_odom.pose.pose.orientation
        # Reorder to scalar-first [w, x, y, z] for training ObservationHandler!
        q_wxyz = np.array([q_msg.w, q_msg.x, q_msg.y, q_msg.z], dtype=np.float32)

        # Extract Angular Rates in Body FLU
        w = self.latest_odom.twist.twist.angular
        rates_flu = np.array([w.x, w.y, w.z], dtype=np.float32)

        # Compute observation using frozen ObservationHandler
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
        """Verify geofence, attitude tilt, and speed bounds."""
        # 1. Geofence bounds
        x, y, z = float(pos_enu[0]), float(pos_enu[1]), float(pos_enu[2])
        if abs(x) > 10.0 or abs(y) > 10.0 or z > 10.0:
            return True, f"GEOFENCE_VIOLATION (pos=[{x:.2f}, {y:.2f}, {z:.2f}])"

        # Altitude safety check (active during control)
        if self.control_enabled and z < 0.3:
            return True, f"ALTITUDE_TOO_LOW (z={z:.2f}m < 0.3m)"

        # 2. Maximum tilt check (roll/pitch < 40 deg)
        qw, qx, qy, qz = q_wxyz[0], q_wxyz[1], q_wxyz[2], q_wxyz[3]
        roll = math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
        pitch = math.asin(max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx))))
        if abs(roll) > math.radians(40.0) or abs(pitch) > math.radians(40.0):
            return True, f"EXCESSIVE_TILT (roll={math.degrees(roll):.1f}deg, pitch={math.degrees(pitch):.1f}deg)"

        # 3. Maximum speed check
        if speed > 4.5:
            return True, f"SPEED_LIMIT_EXCEEDED (speed={speed:.2f}m/s)"

        return False, "OK"

    def publish_zero_cmd(self):
        """Command safe neutral hover."""
        stop_twist = Twist()
        self.pub_cmd_vel.publish(stop_twist)

    def policy_step_callback(self):
        """Main 10 Hz policy loop."""
        self.total_inference_cycles += 1

        # 1. Ingest telemetry and construct observation vector
        obs = self.build_observation()
        if obs is None:
            self.total_dropped_cycles += 1
            if self.control_enabled:
                self.publish_zero_cmd()
            return

        # Publish observation for telemetry logging
        obs_msg = Float32MultiArray()
        obs_msg.data = [float(val) for val in obs]
        self.pub_obs.publish(obs_msg)

        # Extract current state metrics
        pos_enu = np.array([
            self.latest_odom.pose.pose.position.x,
            self.latest_odom.pose.pose.position.y,
            self.latest_odom.pose.pose.position.z
        ], dtype=np.float32)
        dist_to_goal = float(np.linalg.norm(self.goal_world - pos_enu))
        vel_enu = np.array([
            self.latest_odom.twist.twist.linear.x,
            self.latest_odom.twist.twist.linear.y,
            self.latest_odom.twist.twist.linear.z
        ], dtype=np.float32)
        speed = float(np.linalg.norm(vel_enu))

        q_msg = self.latest_odom.pose.pose.orientation
        q_wxyz = np.array([q_msg.w, q_msg.x, q_msg.y, q_msg.z], dtype=np.float32)

        # Run safety watchdog
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

        # 2. Compute Classical Baseline action for comparison
        classical_act_norm = self.classical_controller.predict(obs, deterministic=True)
        classical_cmd = self.action_handler.scale_action(classical_act_norm)

        # 3. Mode Branching
        ppo_act_norm = np.zeros(4, dtype=np.float32)
        ppo_cmd = CommandedVelocities(0.0, 0.0, 0.0, 0.0)
        lat_ms = 0.0

        if self.mode in ["SHADOW", "CONTROLLED_SIM"] and self.model is not None:
            t0 = time.perf_counter()
            ppo_act_norm, _ = self.model.predict(obs, deterministic=True)
            t1 = time.perf_counter()
            lat_ms = (t1 - t0) * 1000.0
            self.inference_latencies_ms.append(lat_ms)
            if len(self.inference_latencies_ms) > 200:
                self.inference_latencies_ms.pop(0)

            # Sanitize and scale action
            ppo_cmd = self.action_handler.scale_action(ppo_act_norm)

        # 4. Action Authority Handling
        if self.mode == "CONTROLLED_SIM" and self.control_enabled and not self.watchdog_tripped:
            # Select command according to active controller selection
            if self.active_controller == "PPO":
                active_cmd = ppo_cmd
                active_act_norm = ppo_act_norm
            else:
                active_cmd = classical_cmd
                active_act_norm = classical_act_norm

            # Conservative velocity clamp for simulation testing
            vx = float(np.clip(active_cmd.vx, -self.max_velocity_clamp, self.max_velocity_clamp))
            vy = float(np.clip(active_cmd.vy, -self.max_velocity_clamp, self.max_velocity_clamp))
            vz = float(np.clip(active_cmd.vz, -self.max_velocity_clamp, self.max_velocity_clamp))
            yaw_rate = float(active_cmd.yaw_rate)

            twist = Twist()
            twist.linear.x = vx
            twist.linear.y = vy
            twist.linear.z = vz
            twist.angular.z = yaw_rate
            self.pub_cmd_vel.publish(twist)

            # Update prev_action history
            self.prev_action = active_act_norm.copy()
        else:
            # OBSERVATION_ONLY or SHADOW mode: ZERO command sent
            if self.mode == "CONTROLLED_SIM" and not self.control_enabled:
                self.publish_zero_cmd()

        # 5. Diagnostic Telemetry Reporting
        diag = {
            "mode": self.mode,
            "active_controller": self.active_controller,
            "control_enabled": self.control_enabled,
            "watchdog_tripped": self.watchdog_tripped,
            "watchdog_reason": self.watchdog_reason,
            "dist_to_goal_m": round(dist_to_goal, 3),
            "pos_enu": [round(float(p), 3) for p in pos_enu],
            "goal_enu": [round(float(g), 3) for g in self.goal_world],
            "speed_m_s": round(speed, 3),
            "inference_latency_ms": round(lat_ms, 3),
            "mean_latency_ms": round(float(np.mean(self.inference_latencies_ms)), 3) if self.inference_latencies_ms else 0.0,
            "ppo_action": [round(float(a), 3) for a in ppo_act_norm],
            "ppo_cmd": [round(ppo_cmd.vx, 3), round(ppo_cmd.vy, 3), round(ppo_cmd.vz, 3), round(ppo_cmd.yaw_rate, 3)],
            "classical_action": [round(float(a), 3) for a in classical_act_norm],
            "classical_cmd": [round(classical_cmd.vx, 3), round(classical_cmd.vy, 3), round(classical_cmd.vz, 3), round(classical_cmd.yaw_rate, 3)],
            "action_diff_norm": round(float(np.linalg.norm(ppo_act_norm - classical_act_norm)), 3)
        }

        diag_msg = String()
        diag_msg.data = json.dumps(diag)
        self.pub_diag.publish(diag_msg)


def main(args=None):
    rclpy.init(args=args)
    adapter = PPOInferenceAdapter()
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
