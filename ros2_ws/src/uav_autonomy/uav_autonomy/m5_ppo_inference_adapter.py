#!/usr/bin/env python3
"""ROS 2 M5 26D PPO controller for supervised PX4 SITL flights.

Flight authority is disabled at startup. The node requires fresh odometry and
LiDAR before enabling, applies the same deterministic shield as the benchmark,
and publishes body-FLU velocity commands to the existing PX4 MAVLink bridge.
"""

import json
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool

from rl_env.config import UAVEnvConfig
from rl_env.observation_26d import ObservationHandler26D
from rl_env.safety_shield import DeterministicSafetyShield, SafetyShieldConfig


class M5PPOInferenceAdapter(Node):
    def __init__(self):
        super().__init__('m5_ppo_inference_adapter')
        self.declare_parameter('model_path', os.path.expanduser('~/proj/experiments/milestone5/models/ppo_m5_stage3_completion_run1.zip'))
        self.declare_parameter('goal_x', 4.0)
        self.declare_parameter('goal_y', 2.0)
        self.declare_parameter('goal_z', 2.5)
        self.declare_parameter('ground_z', 0.0)
        self.declare_parameter('safety_margin_m', 0.55)
        self.declare_parameter('policy_rate_hz', 10.0)
        self.declare_parameter('telemetry_timeout_s', 0.25)
        self.declare_parameter('geofence_limit_m', 8.0)

        self.goal = np.array([
            self.get_parameter('goal_x').value,
            self.get_parameter('goal_y').value,
            self.get_parameter('goal_z').value,
        ], dtype=np.float32)
        self.ground_z = float(self.get_parameter('ground_z').value)
        self.timeout_s = float(self.get_parameter('telemetry_timeout_s').value)
        self.geofence_limit = float(self.get_parameter('geofence_limit_m').value)
        self.config = UAVEnvConfig()
        self.observer = ObservationHandler26D(self.config)
        self.shield = DeterministicSafetyShield(SafetyShieldConfig(
            safety_margin=float(self.get_parameter('safety_margin_m').value),
        ))

        import torch
        from stable_baselines3 import PPO
        torch.set_num_threads(1)
        self.model = PPO.load(str(self.get_parameter('model_path').value), device='cpu')
        self.model.policy.eval()

        self.latest_odom = None
        self.latest_scan = None
        self.odom_rx = 0.0
        self.scan_rx = 0.0
        self.prev_action = np.zeros(4, dtype=np.float32)
        self.enabled = False
        self.fault = ''
        self.cycles = 0
        self.interventions = 0
        self.latencies_ms = []

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.create_subscription(Odometry, '/uav/odometry', self.on_odom, sensor_qos)
        self.create_subscription(LaserScan, '/uav/sensors/scan', self.on_scan, sensor_qos)
        self.create_subscription(Point, '/uav/set_goal', self.on_goal, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.diag_pub = self.create_publisher(String, '/uav/m5_diagnostics', 10)
        self.create_service(SetBool, '/uav/enable_m5_control', self.on_enable)
        period = 1.0 / float(self.get_parameter('policy_rate_hz').value)
        self.create_timer(period, self.on_policy_tick)
        self.get_logger().info(
            f'M5 PPO ready; flight control DISABLED; model={self.get_parameter("model_path").value}; '
            f'goal ENU={self.goal.tolist()}; shield d_base={self.shield.config.d_base:.2f}m'
        )

    def on_odom(self, msg):
        self.latest_odom = msg
        self.odom_rx = time.monotonic()

    def on_scan(self, msg):
        self.latest_scan = msg
        self.scan_rx = time.monotonic()

    def on_goal(self, msg):
        self.goal = np.array([msg.x, msg.y, msg.z], dtype=np.float32)
        self.get_logger().info(f'Updated M5 goal ENU to {self.goal.tolist()}')

    def publish_zero(self):
        self.cmd_pub.publish(Twist())

    def on_enable(self, request, response):
        if request.data:
            now = time.monotonic()
            if self.latest_odom is None or now - self.odom_rx > self.timeout_s:
                response.success = False
                response.message = 'Fresh odometry is required before enabling M5 control.'
                return response
            if self.latest_scan is None or now - self.scan_rx > self.timeout_s:
                response.success = False
                response.message = 'Fresh LiDAR is required before enabling M5 control.'
                return response
            self.fault = ''
            self.prev_action.fill(0.0)
            self.enabled = True
            response.success = True
            response.message = 'M5 control enabled. Ensure PX4 is armed and OFFBOARD first.'
        else:
            self.enabled = False
            self.fault = 'OPERATOR_DISABLED'
            self.publish_zero()
            response.success = True
            response.message = 'M5 control disabled; zero velocity published.'
        return response

    def on_policy_tick(self):
        if not self.enabled:
            return
        now_mono = time.monotonic()
        if (self.latest_odom is None or self.latest_scan is None or
                now_mono - self.odom_rx > self.timeout_s or
                now_mono - self.scan_rx > self.timeout_s):
            self.enabled = False
            self.fault = 'STALE_ODOMETRY_OR_LIDAR'
            self.publish_zero()
            self.publish_diagnostics({'fault': self.fault, 'control_enabled': False})
            self.get_logger().error(f'M5 watchdog: {self.fault}; control disabled')
            return

        odom, scan = self.latest_odom, self.latest_scan
        p = odom.pose.pose.position
        pos = np.array([p.x, p.y, p.z], dtype=np.float32)
        if (abs(float(pos[0])) >= self.geofence_limit or
                abs(float(pos[1])) >= self.geofence_limit or
                pos[2] <= 0.25 or pos[2] >= 6.0):
            self.enabled = False
            self.fault = 'GEOFENCE_LIMIT'
            self.publish_zero()
            self.publish_diagnostics({'fault': self.fault, 'position_enu': pos.tolist(), 'control_enabled': False})
            self.get_logger().error(f'M5 watchdog: {self.fault}; control disabled')
            return

        q = odom.pose.pose.orientation
        quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
        v = odom.twist.twist.linear
        vel_enu = np.array([v.x, v.y, v.z], dtype=np.float32)
        w = odom.twist.twist.angular
        rates_body = np.array([w.x, w.y, w.z], dtype=np.float32)
        ros_now = self.get_clock().now().nanoseconds * 1e-9
        sensor_time = float(scan.header.stamp.sec) + float(scan.header.stamp.nanosec) * 1e-9
        obs, meta_obj = self.observer.compute_observation_with_metadata(
            pos_world=pos,
            vel_world=vel_enu,
            quat=quat_wxyz,
            rates_body=rates_body,
            goal_world=self.goal,
            prev_action=self.prev_action,
            raw_scan=(scan.ranges, scan.angle_min, scan.angle_increment),
            ground_z=self.ground_z,
            sensor_time=sensor_time,
            current_time=ros_now,
        )
        meta = {
            'is_sensor_stale': bool(meta_obj.is_sensor_stale),
            'sensor_age_s': float(meta_obj.sensor_age_s),
            'position_world': pos,
        }
        t0 = time.perf_counter()
        try:
            action, _ = self.model.predict(obs, deterministic=True)
            action, diag = self.shield.filter_action(obs, action, perception_metadata=meta)
        except Exception as exc:
            self.enabled = False
            self.fault = f'INFERENCE_EXCEPTION:{type(exc).__name__}'
            self.publish_zero()
            self.get_logger().error(f'M5 inference watchdog: {exc}; control disabled')
            return
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not np.all(np.isfinite(action)):
            self.enabled = False
            self.fault = 'NONFINITE_ACTION'
            self.publish_zero()
            self.get_logger().error('M5 inference watchdog: non-finite action; control disabled')
            return

        # Convert normalized action to the shielded physical body-FLU command.
        cmd = diag.shielded_cmd
        msg = Twist()
        msg.linear.x = float(cmd.vx)
        msg.linear.y = float(cmd.vy)
        msg.linear.z = float(cmd.vz)
        msg.angular.z = float(cmd.yaw_rate)
        self.cmd_pub.publish(msg)
        self.prev_action = np.asarray(action, dtype=np.float32).copy()
        self.cycles += 1
        self.interventions += int(diag.is_intervened)
        self.latencies_ms.append(latency_ms)
        self.publish_diagnostics({
            'control_enabled': True,
            'position_enu': pos.tolist(),
            'goal_enu': self.goal.tolist(),
            'latency_ms': latency_ms,
            'shield': diag.to_dict(),
            'cycles': self.cycles,
            'shield_interventions': self.interventions,
        })

    def publish_diagnostics(self, data):
        msg = String()
        msg.data = json.dumps(data, separators=(',', ':'))
        self.diag_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = M5PPOInferenceAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.enabled = False
        node.publish_zero()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
