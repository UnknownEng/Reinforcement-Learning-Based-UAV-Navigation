#!/usr/bin/env python3
"""
px4_mavlink_bridge.py
---------------------
ROS 2 Jazzy node that bridges PX4 SITL MAVLink telemetry and control to ROS 2.

Topics Published:
  - /uav/odometry (nav_msgs/msg/Odometry) - ENU world frame, FLU body frame
  - /uav/imu      (sensor_msgs/msg/Imu)      - FLU body frame
  - /uav/status   (std_msgs/msg/String)      - JSON armed/mode/landed status

Topics Subscribed:
  - /cmd_vel      (geometry_msgs/msg/Twist)  - Linear [vx, vy, vz] in FLU body frame (m/s)
                                              Angular [vz] = yaw rate CCW (rad/s)

Services:
  - /uav/arm          (std_srvs/srv/SetBool) - True to arm, False to disarm
  - /uav/takeoff      (std_srvs/srv/Trigger) - Command autonomous takeoff to 2m
  - /uav/land         (std_srvs/srv/Trigger) - Command autonomous land
  - /uav/set_offboard (std_srvs/srv/SetBool) - True to enter OFFBOARD mode

Coordinate Conventions:
  - PX4: Local frame is NED (North-East-Down), Body frame is FRD (Forward-Right-Down).
  - ROS 2: World frame is ENU (East-North-Up), Body frame is FLU (Forward-Left-Up).
  - Conversions:
      x_enu = y_ned, y_enu = x_ned, z_enu = -z_ned
      vx_enu = vy_ned, vy_enu = vx_ned, vz_enu = -vz_ned
      R_enu = R_ned2enu @ R_ned @ R_frd2flu
"""

import json
import math
import sys
import threading
import time

import numpy as np
from pymavlink import mavutil
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from geometry_msgs.msg import Point, Quaternion, Twist, Vector3
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Header, String
from std_srvs.srv import SetBool, Trigger
from scipy.spatial.transform import Rotation as R_scipy

# Transformation matrices for coordinate frame mapping
R_NED2ENU = np.array([
    [0.0, 1.0,  0.0],
    [1.0, 0.0,  0.0],
    [0.0, 0.0, -1.0]
], dtype=float)

R_FRD2FLU = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0]
], dtype=float)


class PX4MavlinkBridge(Node):
    def __init__(self):
        super().__init__('px4_mavlink_bridge')

        # Parameters
        self.declare_parameter('connection_url', 'udpin:0.0.0.0:14540')
        self.declare_parameter('target_system', 1)
        self.declare_parameter('target_component', 1)
        self.declare_parameter('setpoint_rate_hz', 20.0)
        self.declare_parameter('takeoff_altitude_m', 2.0)

        self.conn_url = self.get_parameter('connection_url').get_parameter_value().string_value
        self.target_system = self.get_parameter('target_system').get_parameter_value().integer_value
        self.target_component = self.get_parameter('target_component').get_parameter_value().integer_value
        self.setpoint_rate = self.get_parameter('setpoint_rate_hz').get_parameter_value().double_value
        self.default_takeoff_alt = self.get_parameter('takeoff_altitude_m').get_parameter_value().double_value

        self.get_logger().info(f'Connecting to PX4 SITL at {self.conn_url} ...')
        self.mav = mavutil.mavlink_connection(self.conn_url)
        self.get_logger().info('MAVLink connection initialized.')

        # Vehicle state cache
        self.lock = threading.Lock()
        self.running = True
        self.is_connected = False
        self.is_armed = False
        self.flight_mode = 'UNKNOWN'
        self.landed_state = 'UNKNOWN'
        self.last_heartbeat_time = 0.0

        # Telemetry cache (NED)
        self.sim_time_sec = 0.0
        self.imu_sim_time_sec = 0.0
        self.pos_ned = np.array([0.0, 0.0, 0.0])
        self.vel_ned = np.array([0.0, 0.0, 0.0])
        self.attitude_euler_ned = np.array([0.0, 0.0, 0.0])  # roll, pitch, yaw
        self.attitude_rates_ned = np.array([0.0, 0.0, 0.0])  # rollspeed, pitchspeed, yawspeed
        self.cached_q_enu = np.array([0.0, 0.0, 0.0, 1.0])
        self.last_imu_pub_time = 0.0
        self.last_odom_pub_time = 0.0
        self.imu_acc_frd = np.array([0.0, 0.0, 0.0])
        self.imu_gyro_frd = np.array([0.0, 0.0, 0.0])

        # Active commanded setpoints (in body FLU frame)
        self.cmd_vx_flu = 0.0
        self.cmd_vy_flu = 0.0
        self.cmd_vz_flu = 0.0
        self.cmd_yaw_rate_flu = 0.0
        self.last_cmd_vel_time = 0.0
        self.offboard_enabled = False

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

        # Publishers
        self.pub_odom = self.create_publisher(Odometry, '/uav/odometry', sensor_qos)
        self.pub_imu = self.create_publisher(Imu, '/uav/imu', sensor_qos)
        self.pub_status = self.create_publisher(String, '/uav/status', state_qos)

        # Subscriptions
        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )

        # Services
        self.srv_arm = self.create_service(SetBool, '/uav/arm', self.handle_arm)
        self.srv_takeoff = self.create_service(Trigger, '/uav/takeoff', self.handle_takeoff)
        self.srv_land = self.create_service(Trigger, '/uav/land', self.handle_land)
        self.srv_offboard = self.create_service(SetBool, '/uav/set_offboard', self.handle_set_offboard)

        # Background thread for continuous non-blocking MAVLink reception
        self.rx_thread = threading.Thread(target=self.mavlink_rx_loop, daemon=True)
        self.rx_thread.start()

        # Timers
        # 1. Companion heartbeat sender (1 Hz)
        self.timer_heartbeat = self.create_timer(1.0, self.send_heartbeat)
        # 2. Offboard setpoint streaming timer (20 Hz)
        self.timer_setpoint = self.create_timer(1.0 / self.setpoint_rate, self.stream_setpoints)
        # 3. Status publisher timer (10 Hz)
        self.timer_status = self.create_timer(0.1, self.publish_status)

        self.get_logger().info('PX4 MAVLink Bridge Node successfully initialized and active.')

    def mavlink_rx_loop(self):
        """Dedicated thread for low-latency MAVLink socket reads."""
        while self.running:
            try:
                msg = self.mav.recv_match(blocking=False)
                if msg is not None:
                    self.process_mavlink_msg(msg)
                else:
                    time.sleep(0.002)
            except Exception as e:
                self.get_logger().error(f'Error reading MAVLink packet: {e}')
                time.sleep(0.01)

    def send_heartbeat(self):
        """Send companion computer heartbeat to PX4."""
        try:
            self.mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0, 0, 0
            )
        except Exception as e:
            self.get_logger().warn(f'Heartbeat send error: {e}')

    def process_mavlink_msg(self, msg):
        msg_type = msg.get_type()
        now = time.time()

        if msg_type == 'HEARTBEAT':
            with self.lock:
                self.is_connected = True
                self.last_heartbeat_time = now
                self.is_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                custom_main_mode = (msg.custom_mode >> 16) & 0xFF
                custom_sub_mode = (msg.custom_mode >> 24) & 0xFF
                self.flight_mode = self.decode_flight_mode(custom_main_mode, custom_sub_mode)

        elif msg_type == 'EXTENDED_SYS_STATE':
            landed_map = {
                mavutil.mavlink.MAV_LANDED_STATE_UNDEFINED: 'UNDEFINED',
                mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND: 'ON_GROUND',
                mavutil.mavlink.MAV_LANDED_STATE_IN_AIR: 'IN_AIR',
                mavutil.mavlink.MAV_LANDED_STATE_TAKEOFF: 'TAKEOFF',
                mavutil.mavlink.MAV_LANDED_STATE_LANDING: 'LANDING',
            }
            with self.lock:
                self.landed_state = landed_map.get(msg.landed_state, f'UNKNOWN_{msg.landed_state}')

        elif msg_type == 'STATUSTEXT':
            self.get_logger().warn(f'PX4 STATUSTEXT: {msg.text}')

        elif msg_type == 'COMMAND_ACK':
            self.get_logger().info(f'PX4 COMMAND_ACK: command={msg.command}, result={msg.result}')

        elif msg_type == 'LOCAL_POSITION_NED':
            with self.lock:
                self.sim_time_sec = msg.time_boot_ms * 1e-3
                self.pos_ned = np.array([msg.x, msg.y, msg.z])
                self.vel_ned = np.array([msg.vx, msg.vy, msg.vz])
            self.publish_odometry()

        elif msg_type == 'ATTITUDE':
            with self.lock:
                self.attitude_euler_ned = np.array([msg.roll, msg.pitch, msg.yaw])
                self.attitude_rates_ned = np.array([msg.rollspeed, msg.pitchspeed, msg.yawspeed])
                try:
                    r_ned = R_scipy.from_euler('xyz', self.attitude_euler_ned).as_matrix()
                    r_enu_mat = R_NED2ENU @ r_ned @ R_FRD2FLU
                    self.cached_q_enu = R_scipy.from_matrix(r_enu_mat).as_quat()
                except Exception:
                    pass

        elif msg_type == 'HIGHRES_IMU':
            with self.lock:
                self.imu_sim_time_sec = msg.time_usec * 1e-6
                self.imu_acc_frd = np.array([msg.xacc, msg.yacc, msg.zacc])
                self.imu_gyro_frd = np.array([msg.xgyro, msg.ygyro, msg.zgyro])
            now_t = time.time()
            if (now_t - self.last_imu_pub_time) >= 0.02:  # 50 Hz throttle
                self.last_imu_pub_time = now_t
                self.publish_imu()

    def decode_flight_mode(self, main_mode, sub_mode):
        modes = {
            1: 'MANUAL',
            2: 'ALTCTL',
            3: 'POSCTL',
            4: {1: 'AUTO_READY', 2: 'AUTO_TAKEOFF', 3: 'AUTO_LOITER', 4: 'AUTO_MISSION', 5: 'AUTO_RTL', 6: 'AUTO_LAND'},
            5: 'ACRO',
            6: 'OFFBOARD',
            7: 'STABILIZED',
            8: 'RATTITUDE'
        }
        res = modes.get(main_mode, f'MODE_{main_mode}')
        if isinstance(res, dict):
            return res.get(sub_mode, f'AUTO_{sub_mode}')
        return res

    def publish_odometry(self):
        """Convert NED to ENU and publish Odometry."""
        with self.lock:
            pos_ned = self.pos_ned.copy()
            vel_ned = self.vel_ned.copy()
            q_enu = self.cached_q_enu.copy()
            rates_ned = self.attitude_rates_ned.copy()

        # Position: x_enu = y_ned (East), y_enu = x_ned (North), z_enu = -z_ned (Up)
        pos_enu = np.array([pos_ned[1], pos_ned[0], -pos_ned[2]])

        # Velocity: vx_enu = vy_ned, vy_enu = vx_ned, vz_enu = -vz_ned
        vel_enu = np.array([vel_ned[1], vel_ned[0], -vel_ned[2]])

        # Angular velocity in body FLU:
        ang_vel_flu = np.array([rates_ned[0], -rates_ned[1], -rates_ned[2]])

        odom = Odometry()
        with self.lock:
            sim_t = self.sim_time_sec
        sec = int(sim_t)
        nanosec = int((sim_t - sec) * 1e9)
        odom.header.stamp.sec = sec
        odom.header.stamp.nanosec = nanosec
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = float(pos_enu[0])
        odom.pose.pose.position.y = float(pos_enu[1])
        odom.pose.pose.position.z = float(pos_enu[2])

        odom.pose.pose.orientation.x = float(q_enu[0])
        odom.pose.pose.orientation.y = float(q_enu[1])
        odom.pose.pose.orientation.z = float(q_enu[2])
        odom.pose.pose.orientation.w = float(q_enu[3])

        odom.twist.twist.linear.x = float(vel_enu[0])
        odom.twist.twist.linear.y = float(vel_enu[1])
        odom.twist.twist.linear.z = float(vel_enu[2])

        odom.twist.twist.angular.x = float(ang_vel_flu[0])
        odom.twist.twist.angular.y = float(ang_vel_flu[1])
        odom.twist.twist.angular.z = float(ang_vel_flu[2])

        self.pub_odom.publish(odom)

    def publish_imu(self):
        """Convert FRD to FLU and publish Imu."""
        with self.lock:
            acc_frd = self.imu_acc_frd.copy()
            gyro_frd = self.imu_gyro_frd.copy()
            q_enu = self.cached_q_enu.copy()

        acc_flu = np.array([acc_frd[0], -acc_frd[1], -acc_frd[2]])
        gyro_flu = np.array([gyro_frd[0], -gyro_frd[1], -gyro_frd[2]])

        imu = Imu()
        with self.lock:
            imu_t = self.imu_sim_time_sec
        sec = int(imu_t)
        nanosec = int((imu_t - sec) * 1e9)
        imu.header.stamp.sec = sec
        imu.header.stamp.nanosec = nanosec
        imu.header.frame_id = 'base_link'

        imu.orientation.x = float(q_enu[0])
        imu.orientation.y = float(q_enu[1])
        imu.orientation.z = float(q_enu[2])
        imu.orientation.w = float(q_enu[3])

        imu.linear_acceleration.x = float(acc_flu[0])
        imu.linear_acceleration.y = float(acc_flu[1])
        imu.linear_acceleration.z = float(acc_flu[2])

        imu.angular_velocity.x = float(gyro_flu[0])
        imu.angular_velocity.y = float(gyro_flu[1])
        imu.angular_velocity.z = float(gyro_flu[2])

        self.pub_imu.publish(imu)

    def publish_status(self):
        """Publish JSON status string."""
        with self.lock:
            status = {
                'connected': self.is_connected,
                'armed': self.is_armed,
                'flight_mode': self.flight_mode,
                'landed_state': self.landed_state,
                'offboard_enabled': self.offboard_enabled,
                'pos_enu': [round(self.pos_ned[1], 3), round(self.pos_ned[0], 3), round(-self.pos_ned[2], 3)],
                'vel_enu': [round(self.vel_ned[1], 3), round(self.vel_ned[0], 3), round(-self.vel_ned[2], 3)],
            }
        msg = String()
        msg.data = json.dumps(status)
        self.pub_status.publish(msg)

    def cmd_vel_callback(self, msg: Twist):
        """Store incoming velocity commands in body FLU frame."""
        self.cmd_vx_flu = msg.linear.x
        self.cmd_vy_flu = msg.linear.y
        self.cmd_vz_flu = msg.linear.z
        self.cmd_yaw_rate_flu = msg.angular.z
        self.last_cmd_vel_time = time.time()

    def stream_setpoints(self):
        """
        Send SET_POSITION_TARGET_LOCAL_NED setpoints at 20 Hz.
        PX4 requires continuous setpoints in OFFBOARD mode.
        """
        if not self.offboard_enabled:
            return

        now = time.time()
        # Timeout safety: If no /cmd_vel received in last 0.5s, command 0 velocity (hover)
        if (now - self.last_cmd_vel_time) > 0.5:
            vx_flu = 0.0
            vy_flu = 0.0
            vz_flu = 0.0
            yaw_rate_flu = 0.0
        else:
            vx_flu = self.cmd_vx_flu
            vy_flu = self.cmd_vy_flu
            vz_flu = self.cmd_vz_flu
            yaw_rate_flu = self.cmd_yaw_rate_flu

        # Map Body FLU to Body FRD:
        vx_frd = vx_flu
        vy_frd = -vy_flu
        vz_frd = -vz_flu
        yaw_rate_frd = -yaw_rate_flu

        type_mask = 0x05C7  # Ignore position, acceleration, and yaw angle; enable vx, vy, vz, yaw_rate

        try:
            self.mav.mav.set_position_target_local_ned_send(
                int(time.time() * 1000) & 0xFFFFFFFF,
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_FRAME_BODY_NED,
                type_mask,
                0.0, 0.0, 0.0,
                vx_frd, vy_frd, vz_frd,
                0.0, 0.0, 0.0,
                0.0,
                yaw_rate_frd
            )
        except Exception as e:
            self.get_logger().debug(f'Error sending setpoint: {e}')

    # Service Handlers
    def handle_arm(self, request: SetBool.Request, response: SetBool.Response):
        arm_val = 1.0 if request.data else 0.0
        action = 'ARM' if request.data else 'DISARM'
        self.get_logger().info(f'Handling {action} request...')
        try:
            self.mav.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0,
                arm_val, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            )
            start = time.time()
            success = False
            while time.time() - start < 12.0:
                with self.lock:
                    if self.is_armed == request.data:
                        success = True
                        break
                time.sleep(0.05)

            response.success = success
            response.message = f'{action} succeeded' if success else f'{action} timed out'
        except Exception as e:
            response.success = False
            response.message = f'{action} failed: {e}'
        return response

    def handle_takeoff(self, request: Trigger.Request, response: Trigger.Response):
        self.get_logger().info(f'Sending TAKEOFF command (target altitude {self.default_takeoff_alt} m)...')
        try:
            # Switch to AUTO_TAKEOFF mode (main 4, sub 2)
            self.mav.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                4, 2, 0, 0, 0, 0
            )
            time.sleep(0.05)
            self.mav.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0,
                0.0, 0.0, 0.0, float('nan'), float('nan'), float('nan'),
                float(self.default_takeoff_alt)
            )
            response.success = True
            response.message = f'Takeoff to {self.default_takeoff_alt} m commanded'
        except Exception as e:
            response.success = False
            response.message = f'Takeoff command failed: {e}'
        return response

    def handle_land(self, request: Trigger.Request, response: Trigger.Response):
        self.get_logger().info('Sending LAND command...')
        try:
            self.mav.mav.command_long_send(
                self.target_system,
                self.target_component,
                mavutil.mavlink.MAV_CMD_NAV_LAND,
                0,
                0.0, 0.0, 0.0, float('nan'), float('nan'), float('nan'), 0.0
            )
            response.success = True
            response.message = 'Land commanded'
        except Exception as e:
            response.success = False
            response.message = f'Land command failed: {e}'
        return response

    def handle_set_offboard(self, request: SetBool.Request, response: SetBool.Response):
        if request.data:
            self.get_logger().info('Requesting PX4 switch to OFFBOARD mode...')
            try:
                # First enable streaming setpoints for at least 0.2s so PX4 accepts offboard
                self.offboard_enabled = True
                type_mask = 0x05C7
                for _ in range(5):
                    self.mav.mav.set_position_target_local_ned_send(
                        int(time.time() * 1000) & 0xFFFFFFFF,
                        self.target_system, self.target_component,
                        mavutil.mavlink.MAV_FRAME_BODY_NED,
                        type_mask,
                        0, 0, 0,
                        0.0, 0.0, 0.0,
                        0, 0, 0, 0, 0
                    )
                    time.sleep(0.04)

                # Send MAV_CMD_DO_SET_MODE with custom mode 6 (PX4_CUSTOM_MAIN_MODE_OFFBOARD)
                self.mav.mav.command_long_send(
                    self.target_system,
                    self.target_component,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    0,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    6, 0, 0, 0, 0, 0
                )
                response.success = True
                response.message = 'OFFBOARD mode requested'
            except Exception as e:
                self.offboard_enabled = False
                response.success = False
                response.message = f'Failed to set OFFBOARD mode: {e}'
        else:
            self.get_logger().info('Switching out of OFFBOARD to HOLD/LOITER...')
            try:
                # Custom mode 4 (AUTO), sub-mode 3 (LOITER)
                self.mav.mav.command_long_send(
                    self.target_system,
                    self.target_component,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                    0,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    4, 3, 0, 0, 0, 0
                )
                self.offboard_enabled = False
                response.success = True
                response.message = 'HOLD mode requested'
            except Exception as e:
                response.success = False
                response.message = f'Failed to exit OFFBOARD mode: {e}'
        return response


def main(args=None):
    rclpy.init(args=args)
    bridge = PX4MavlinkBridge()
    executor = MultiThreadedExecutor()
    executor.add_node(bridge)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.running = False
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
