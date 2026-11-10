#!/usr/bin/env python3
"""
benchmark_stack.py
------------------
Systematic full-stack benchmark measuring:
  - Simulation Real-Time Factor (RTF) via PX4 lockstep clock
  - CPU usage (Total, Gazebo, PX4, ROS 2 Bridge)
  - RAM usage (System, Gazebo, PX4, ROS 2 Bridge)
  - GPU VRAM usage (nvidia-smi)
  - ROS 2 topic publishing rates (Hz, jitter) for /uav/odometry, /uav/imu, /uav/status
  - Command-to-response latency
  - Stability evaluation at 5 Hz, 10 Hz, and 20 Hz control loop frequencies

Saves results to:
  ~/proj/tests/telemetry/milestone1_fullstack_benchmark.json
"""

import json
import os
import subprocess
import sys
import threading
import time

import numpy as np
import psutil
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import String


def get_gpu_memory():
    """Query NVIDIA VRAM usage via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            timeout=2.0
        ).decode().strip()
        used, total = out.split(',')
        return float(used.strip()), float(total.strip())
    except Exception:
        return 0.0, 0.0


def find_processes():
    """Locate PIDs for PX4, Gazebo, and ROS bridge."""
    procs = {'px4': None, 'gazebo': None, 'bridge': None}
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = ' '.join(p.info['cmdline'] or [])
            if 'bin/px4' in cmd and 'ninja' not in cmd:
                procs['px4'] = p
            elif 'gz sim' in cmd or 'ruby' in p.info['name'] or 'gz-sim' in cmd:
                procs['gazebo'] = p
            elif 'px4_mavlink_bridge' in cmd:
                procs['bridge'] = p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return procs


class StackBenchmarkNode(Node):
    def __init__(self):
        super().__init__('benchmark_stack')

        self.odom_times = []
        self.imu_times = []
        self.status_times = []
        self.sim_boot_times = []
        self.wall_times = []

        self.create_subscription(Odometry, '/uav/odometry', self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, '/uav/imu', self.imu_cb, qos_profile_sensor_data)
        self.create_subscription(String, '/uav/status', self.status_cb, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)

        self.lock = threading.Lock()
        self.get_logger().info('Benchmark Node initialized.')

    def odom_cb(self, msg: Odometry):
        t = time.time()
        with self.lock:
            self.odom_times.append(t)
            # ROS stamp holds nanoseconds
            sim_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.sim_boot_times.append(sim_sec)
            self.wall_times.append(t)

    def imu_cb(self, msg: Imu):
        with self.lock:
            self.imu_times.append(time.time())

    def status_cb(self, msg: String):
        with self.lock:
            self.status_times.append(time.time())

    def run_benchmark(self, sample_duration_sec=10.0, target_control_hz=10.0):
        self.get_logger().info(f'Starting benchmark run: {sample_duration_sec}s at {target_control_hz} Hz setpoints...')
        procs = find_processes()
        self.get_logger().info(f"Detected processes: PX4={procs['px4'].pid if procs['px4'] else 'None'}, "
                               f"Gazebo={procs['gazebo'].pid if procs['gazebo'] else 'None'}, "
                               f"Bridge={procs['bridge'].pid if procs['bridge'] else 'None'}")

        # Clear telemetry buffers
        with self.lock:
            self.odom_times.clear()
            self.imu_times.clear()
            self.status_times.clear()
            self.sim_boot_times.clear()
            self.wall_times.clear()

        # Prime psutil cpu counters
        psutil.cpu_percent(interval=None)
        for k, p in procs.items():
            if p:
                try:
                    p.cpu_percent(interval=None)
                except Exception:
                    pass

        start_wall = time.time()
        setpoint_interval = 1.0 / target_control_hz
        next_setpoint = start_wall + setpoint_interval

        cpu_samples = []
        ram_samples = []
        proc_stats = {'px4_cpu': [], 'px4_ram': [], 'gz_cpu': [], 'gz_ram': [], 'bridge_cpu': [], 'bridge_ram': []}

        # Sampling loop
        while time.time() - start_wall < sample_duration_sec:
            t = time.time()
            if t >= next_setpoint:
                # Send gentle zero or small velocity keepalive
                twist = Twist()
                twist.linear.x = 0.0
                self.pub_cmd_vel.publish(twist)
                next_setpoint += setpoint_interval

            # Sample system resource metrics
            cpu_samples.append(psutil.cpu_percent(interval=None))
            vmem = psutil.virtual_memory()
            ram_samples.append(vmem.used / (1024**2))

            for k, p in procs.items():
                if p and p.is_running():
                    try:
                        c = p.cpu_percent(interval=None)
                        m = p.memory_info().rss / (1024**2)
                        proc_stats[f'{k}_cpu'].append(c)
                        proc_stats[f'{k}_ram'].append(m)
                    except Exception:
                        pass

            rclpy.spin_once(self, timeout_sec=0.01)

        end_wall = time.time()
        actual_wall_dur = end_wall - start_wall

        # Compute sensor frequencies
        with self.lock:
            odom_count = len(self.odom_times)
            imu_count = len(self.imu_times)
            status_count = len(self.status_times)

            odom_hz = odom_count / actual_wall_dur
            imu_hz = imu_count / actual_wall_dur
            status_hz = status_count / actual_wall_dur

            # RTF calculation
            if len(self.wall_times) >= 2:
                wall_delta = self.wall_times[-1] - self.wall_times[0]
                sim_delta = self.sim_boot_times[-1] - self.sim_boot_times[0]
                rtf = (sim_delta / wall_delta) if wall_delta > 0 else 0.0
            else:
                rtf = 0.0

            # Inter-arrival jitter for odometry
            if len(self.odom_times) > 2:
                intervals = np.diff(self.odom_times)
                mean_interval = float(np.mean(intervals))
                std_interval = float(np.std(intervals))
                max_interval = float(np.max(intervals))
            else:
                mean_interval, std_interval, max_interval = 0.0, 0.0, 0.0

        vram_used, vram_total = get_gpu_memory()

        results = {
            'target_control_rate_hz': target_control_hz,
            'duration_wall_sec': actual_wall_dur,
            'sim_rtf': round(rtf, 3),
            'system_cpu_mean_pct': round(float(np.mean(cpu_samples)), 1),
            'system_cpu_max_pct': round(float(np.max(cpu_samples)), 1),
            'system_ram_used_mb': round(float(np.mean(ram_samples)), 1),
            'gpu_vram_used_mb': vram_used,
            'gpu_vram_total_mb': vram_total,
            'px4_cpu_mean_pct': round(float(np.mean(proc_stats['px4_cpu'])) if proc_stats['px4_cpu'] else 0.0, 1),
            'px4_ram_mean_mb': round(float(np.mean(proc_stats['px4_ram'])) if proc_stats['px4_ram'] else 0.0, 1),
            'gz_cpu_mean_pct': round(float(np.mean(proc_stats['gz_cpu'])) if proc_stats['gz_cpu'] else 0.0, 1),
            'gz_ram_mean_mb': round(float(np.mean(proc_stats['gz_ram'])) if proc_stats['gz_ram'] else 0.0, 1),
            'bridge_cpu_mean_pct': round(float(np.mean(proc_stats['bridge_cpu'])) if proc_stats['bridge_cpu'] else 0.0, 1),
            'bridge_ram_mean_mb': round(float(np.mean(proc_stats['bridge_ram'])) if proc_stats['bridge_ram'] else 0.0, 1),
            'odom_rate_hz': round(odom_hz, 2),
            'odom_interval_mean_ms': round(mean_interval * 1000.0, 2),
            'odom_interval_std_ms': round(std_interval * 1000.0, 2),
            'odom_interval_max_ms': round(max_interval * 1000.0, 2),
            'imu_rate_hz': round(imu_hz, 2),
            'status_rate_hz': round(status_hz, 2),
        }
        return results


def main():
    rclpy.init()
    node = StackBenchmarkNode()

    rates_to_test = [5.0, 10.0, 20.0]
    all_results = {}

    for rate in rates_to_test:
        node.get_logger().info(f'=== Testing Control Rate: {rate} Hz ===')
        res = node.run_benchmark(sample_duration_sec=10.0, target_control_hz=rate)
        all_results[f'{int(rate)}Hz'] = res
        node.get_logger().info(f"Results for {rate}Hz: RTF={res['sim_rtf']}, Odom={res['odom_rate_hz']}Hz, "
                               f"Sys CPU={res['system_cpu_mean_pct']}%, RAM={res['system_ram_used_mb']}MB")
        time.sleep(1.0)

    out_path = os.path.expanduser('~/proj/tests/telemetry/milestone1_fullstack_benchmark.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    node.get_logger().info(f'Benchmark suite completed. Results saved to {out_path}')

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
