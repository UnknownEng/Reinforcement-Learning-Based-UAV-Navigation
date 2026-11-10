#!/usr/bin/env python3
"""
test_flight_baseline.py
-----------------------
Automated flight test verifying the Milestone 1 UAV baseline sequence:
  GROUND -> ARM -> TAKEOFF -> HOVER (2m) -> FORWARD (1m/s) -> STOP/HOVER -> LAND -> DISARM

Saves full time-series telemetry to:
  ~/proj/tests/telemetry/milestone1_flight_telemetry.json
"""

import json
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger


class FlightBaselineTester(Node):
    def __init__(self):
        super().__init__('test_flight_baseline')

        # Subscriptions
        self.sub_odom = self.create_subscription(Odometry, '/uav/odometry', self.odom_callback, qos_profile_sensor_data)
        self.sub_status = self.create_subscription(String, '/uav/status', self.status_callback, 10)
        self.sub_imu = self.create_subscription(Imu, '/uav/imu', self.imu_callback, qos_profile_sensor_data)

        # Publisher
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        # Service Clients
        self.cli_arm = self.create_client(SetBool, '/uav/arm')
        self.cli_takeoff = self.create_client(Trigger, '/uav/takeoff')
        self.cli_land = self.create_client(Trigger, '/uav/land')
        self.cli_offboard = self.create_client(SetBool, '/uav/set_offboard')

        # Telemetry storage
        self.latest_odom = None
        self.latest_status = None
        self.latest_imu = None
        self.telemetry_history = []

        # State machine tracking
        self.state = 'WAIT_BRIDGE'
        self.state_start_wall = time.time()
        self.state_start_sim = 0.0
        self.active_future = None
        self.test_success = False
        self.metrics = {}
        self.initial_z = None
        self.pos_before_forward = None
        # PX4 can briefly refuse arming right after boot ("Preflight Fail: vertical velocity unstable"
        # while EKF2 settles); retry like an operator would instead of failing the whole test.
        self.arm_attempts = 0
        self.max_arm_attempts = 10

        # 20 Hz control loop
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Flight Baseline Tester node initialized.')

    def get_current_sim_time(self):
        if self.latest_odom is not None:
            return self.latest_odom.header.stamp.sec + self.latest_odom.header.stamp.nanosec * 1e-9
        return time.time()

    def set_state(self, new_state):
        self.state = new_state
        self.state_start_wall = time.time()
        self.state_start_sim = self.get_current_sim_time()

    def odom_callback(self, msg: Odometry):
        self.latest_odom = msg
        if self.state not in ['WAIT_BRIDGE', 'FINISHED']:
            self.record_telemetry()

    def status_callback(self, msg: String):
        try:
            self.latest_status = json.loads(msg.data)
        except Exception:
            pass

    def imu_callback(self, msg: Imu):
        self.latest_imu = msg

    def record_telemetry(self):
        if self.latest_odom is None:
            return
        entry = {
            'time_wall': time.time(),
            'time_sim': self.get_current_sim_time(),
            'state': self.state,
            'pos_x': self.latest_odom.pose.pose.position.x,
            'pos_y': self.latest_odom.pose.pose.position.y,
            'pos_z': self.latest_odom.pose.pose.position.z,
            'vel_x': self.latest_odom.twist.twist.linear.x,
            'vel_y': self.latest_odom.twist.twist.linear.y,
            'vel_z': self.latest_odom.twist.twist.linear.z,
            'ang_vel_z': self.latest_odom.twist.twist.angular.z,
            'status': self.latest_status
        }
        self.telemetry_history.append(entry)

    def send_cmd(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        t = Twist()
        t.linear.x = float(vx)
        t.linear.y = float(vy)
        t.linear.z = float(vz)
        t.angular.z = float(yaw_rate)
        self.pub_cmd_vel.publish(t)

    def control_loop(self):
        now_wall = time.time()
        now_sim = self.get_current_sim_time()
        elapsed_wall = now_wall - self.state_start_wall
        elapsed_sim = now_sim - self.state_start_sim

        if self.state == 'WAIT_BRIDGE':
            if self.latest_status is not None and self.latest_odom is not None:
                if self.latest_status.get('connected', False):
                    self.get_logger().info('PX4 Bridge connected and telemetry active. Starting baseline test.')
                    self.initial_z = self.latest_odom.pose.pose.position.z
                    self.metrics['initial_z'] = self.initial_z
                    self.get_logger().info(f'Ground datum initial altitude: {self.initial_z:.3f} m')
                    self.set_state('CALL_ARM')

        elif self.state == 'CALL_ARM':
            if not self.cli_arm.service_is_ready():
                return
            self.arm_attempts += 1
            self.metrics['arm_attempts'] = self.arm_attempts
            self.get_logger().info(f'Action: Sending ARM command (attempt {self.arm_attempts})...')
            req = SetBool.Request()
            req.data = True
            self.active_future = self.cli_arm.call_async(req)
            self.set_state('WAIT_ARM_ACK')

        elif self.state == 'WAIT_ARM_ACK':
            if self.active_future.done():
                res = self.active_future.result()
                if res and res.success:
                    self.get_logger().info('Drone ARMED successfully. Initiating TAKEOFF...')
                    self.set_state('CALL_TAKEOFF')
                elif self.arm_attempts < self.max_arm_attempts:
                    self.get_logger().warn(f'ARM refused ({res}); PX4 preflight checks not clear yet, retrying in 3 s')
                    self.set_state('ARM_RETRY_WAIT')
                else:
                    self.get_logger().error(f'ARM failed after {self.arm_attempts} attempts: {res}')
                    self.set_state('FAILED')

        elif self.state == 'ARM_RETRY_WAIT':
            if elapsed_wall >= 3.0:
                self.set_state('CALL_ARM')

        elif self.state == 'CALL_TAKEOFF':
            if not self.cli_takeoff.service_is_ready():
                return
            self.get_logger().info('Action: Sending TAKEOFF command to 2.0 m...')
            req = Trigger.Request()
            self.active_future = self.cli_takeoff.call_async(req)
            self.set_state('WAIT_TAKEOFF_ACK')

        elif self.state == 'WAIT_TAKEOFF_ACK':
            if self.active_future.done():
                res = self.active_future.result()
                if res and res.success:
                    self.get_logger().info('Takeoff commanded. Monitoring climb to hover...')
                    self.set_state('CLIMB_HOVER')
                else:
                    self.get_logger().error(f'Takeoff failed: {res}')
                    self.set_state('FAILED')

        elif self.state == 'CLIMB_HOVER':
            z = self.latest_odom.pose.pose.position.z
            rel_alt = z - self.initial_z
            mode = self.latest_status.get('flight_mode', '') if self.latest_status else ''
            # Hover is reached when either rel_alt >= 0.6m AND mode is AUTO_LOITER, or rel_alt >= 1.0m
            if ((rel_alt >= 0.6 and mode == 'AUTO_LOITER') or rel_alt >= 1.0) and elapsed_sim > 2.0:
                self.get_logger().info(f'Hover altitude reached: {rel_alt:.2f} m above ground (abs z={z:.2f} m, mode={mode}). Stabilizing for 3s...')
                self.metrics['hover_altitude'] = rel_alt
                self.metrics['hover_abs_z'] = z
                self.set_state('STABILIZE_HOVER')
            elif elapsed_wall > 120.0:
                self.get_logger().warn(f'Climb timeout after {elapsed_wall:.1f}s wall time (rel_alt={rel_alt:.2f} m)')
                self.set_state('CALL_LAND')

        elif self.state == 'STABILIZE_HOVER':
            if elapsed_sim >= 3.0:
                self.get_logger().info('Stabilized hover verified. Switching to OFFBOARD velocity control...')
                self.set_state('CALL_OFFBOARD')

        elif self.state == 'CALL_OFFBOARD':
            if not self.cli_offboard.service_is_ready():
                return
            req = SetBool.Request()
            req.data = True
            self.active_future = self.cli_offboard.call_async(req)
            self.set_state('WAIT_OFFBOARD_ACK')

        elif self.state == 'WAIT_OFFBOARD_ACK':
            if self.active_future.done():
                res = self.active_future.result()
                if res and res.success:
                    self.get_logger().info('OFFBOARD mode enabled. Commencing FORWARD movement...')
                    self.pos_before_forward = (
                        self.latest_odom.pose.pose.position.x,
                        self.latest_odom.pose.pose.position.y
                    )
                    self.set_state('FORWARD')
                else:
                    self.get_logger().error('Failed to set OFFBOARD mode')
                    self.set_state('CALL_LAND')

        elif self.state == 'FORWARD':
            # Command forward 1.0 m/s in body FLU for 3.0 simulated seconds
            self.send_cmd(vx=1.0, vy=0.0, vz=0.0, yaw_rate=0.0)
            if elapsed_sim >= 3.0:
                self.get_logger().info('Forward movement completed. Stopping forward thrust...')
                self.send_cmd(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)
                px = self.latest_odom.pose.pose.position.x
                py = self.latest_odom.pose.pose.position.y
                dx = px - self.pos_before_forward[0]
                dy = py - self.pos_before_forward[1]
                disp = (dx**2 + dy**2)**0.5
                self.metrics['forward_displacement'] = disp
                self.get_logger().info(f'Forward displacement: {disp:.3f} m (dx={dx:.3f}, dy={dy:.3f})')
                self.set_state('STOP_HOVER')

        elif self.state == 'STOP_HOVER':
            self.send_cmd(vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0)
            if elapsed_sim >= 3.0:
                self.get_logger().info('Hover stop verified. Commanding LAND...')
                self.set_state('CALL_LAND')

        elif self.state == 'CALL_LAND':
            if not self.cli_land.service_is_ready():
                return
            req = Trigger.Request()
            self.active_future = self.cli_land.call_async(req)
            self.set_state('WAIT_LAND_ACK')

        elif self.state == 'WAIT_LAND_ACK':
            if self.active_future.done():
                res = self.active_future.result()
                if res and res.success:
                    self.get_logger().info('Land commanded. Monitoring touchdown...')
                    self.set_state('WAIT_TOUCHDOWN')
                else:
                    self.get_logger().error('Land command failed')
                    self.set_state('FAILED')

        elif self.state == 'WAIT_TOUCHDOWN':
            z = self.latest_odom.pose.pose.position.z
            rel_alt = z - self.initial_z
            landed = self.latest_status.get('landed_state', '') if self.latest_status else ''
            # Trust PX4's land detector, not the barometric altitude estimate (drifts ~1 m in SITL):
            # disarming on a drifting "rel_alt < 0.3" while still airborne is refused by PX4.
            if landed == 'ON_GROUND' and elapsed_sim > 2.0:
                self.get_logger().info(f'Touchdown detected (rel_alt={rel_alt:.3f} m, landed={landed}). Disarming...')
                self.metrics['touchdown_altitude'] = rel_alt
                self.metrics['touchdown_landed_state'] = landed
                self.set_state('CALL_DISARM')
            elif elapsed_wall > 120.0:
                self.get_logger().warn(f'Landing wait timeout (rel_alt={rel_alt:.3f} m). Proceeding to disarm.')
                self.set_state('CALL_DISARM')

        elif self.state == 'CALL_DISARM':
            if not self.cli_arm.service_is_ready():
                return
            req = SetBool.Request()
            req.data = False
            self.active_future = self.cli_arm.call_async(req)
            self.set_state('WAIT_DISARM_ACK')

        elif self.state == 'WAIT_DISARM_ACK':
            if self.active_future.done():
                res = self.active_future.result()
                # PASS only if every phase actually happened (previously any completed disarm
                # call counted as a pass, even after a climb timeout or failed OFFBOARD switch).
                checks = {
                    'disarm_acknowledged': bool(res and res.success),
                    'reached_hover': self.metrics.get('hover_altitude', 0.0) >= 1.0,
                    'flew_forward_2m': self.metrics.get('forward_displacement', 0.0) >= 2.0,
                    'touched_down': self.metrics.get('touchdown_landed_state') == 'ON_GROUND',
                }
                self.metrics['pass_checks'] = checks
                self.test_success = all(checks.values())
                self.get_logger().info(f'Baseline flight finished: {"PASS" if self.test_success else "FAIL"} {checks}')
                self.set_state('FINISHED')
                self.save_report()

        elif self.state == 'FAILED':
            self.get_logger().error('Flight test marked as FAILED.')
            self.test_success = False
            self.set_state('FINISHED')
            self.save_report()

    def save_report(self):
        out_dir = os.path.expanduser('~/proj/tests/telemetry')
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, 'milestone1_flight_telemetry.json')
        data = {
            'success': self.test_success,
            'metrics': self.metrics,
            'telemetry_count': len(self.telemetry_history),
            'telemetry': self.telemetry_history
        }
        with open(report_path, 'w') as f:
            json.dump(data, f, indent=2)
        self.get_logger().info(f'Flight report saved to {report_path}')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    tester = FlightBaselineTester()
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    finally:
        tester.destroy_node()


if __name__ == '__main__':
    main()
