"""
mission_manager.py
------------------
M10 ROS 2 node: runs the autonomous mission state machine (uav_sitl.mission.Mission)
and exposes it on ROS 2.

Publishes
  /uav/mission/state   std_msgs/String   current state (PREFLIGHT, ARM, TAKEOFF, NAVIGATE, HOVER,
                                         RETURN, LAND, DONE, ABORTED)
  /uav/mission/report  std_msgs/String   JSON mission report when the mission ends
Parameters
  model_path   learned navigation policy (50-D observation)
  waypoints    flat list [x1, y1, z1, x2, y2, z2, ...] in the local ENU frame (m)
  world        Gazebo world name (for LiDAR / ground-truth topics)
  vehicle      Gazebo model instance (default x500_lidar3d_0)
  use_shield   apply the rule-based safety shield (default true)

Run (WSL, after `source scripts/setup_wsl_env.sh` and a running SITL):
  ros2 run uav_autonomy mission_manager --ros-args -p model_path:=<zip> -p world:=<name> \
      -p waypoints:="[5.0, 3.0, 2.5, 10.0, -2.0, 2.0]"
"""

import json
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from training.train_world_policy import shield_config
from uav_sitl.gz_link import GzLink
from uav_sitl.mission import Mission, MissionConfig
from uav_sitl.px4_link import PX4Link


class MissionManager(Node):
    def __init__(self):
        super().__init__("mission_manager")
        self.declare_parameter("model_path", "")
        self.declare_parameter("waypoints", [0.0])
        self.declare_parameter("world", "default")
        self.declare_parameter("vehicle", "x500_lidar3d_0")
        self.declare_parameter("use_shield", True)
        self.state_pub = self.create_publisher(String, "/uav/mission/state", 10)
        self.report_pub = self.create_publisher(String, "/uav/mission/report", 10)
        flat = list(self.get_parameter("waypoints").value)
        if len(flat) < 3 or len(flat) % 3:
            raise ValueError("waypoints must be a flat list of x, y, z triples")
        self.waypoints = [flat[i:i + 3] for i in range(0, len(flat), 3)]
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _publish_state(self, state: str) -> None:
        self.state_pub.publish(String(data=state))
        self.get_logger().info(f"mission state -> {state}")

    def _run(self) -> None:
        px4 = PX4Link()
        try:
            px4.connect(60)
            gz = GzLink(model=self.get_parameter("vehicle").value, world=self.get_parameter("world").value,
                        scan_topic="/uav/sensors/scan3d")
            mission = Mission(px4, gz, self.get_parameter("model_path").value, "50d",
                              MissionConfig(waypoints=self.waypoints),
                              shield=shield_config() if self.get_parameter("use_shield").value else None,
                              on_state=self._publish_state)
            report = mission.run()
        except Exception as exc:  # report, don't crash silently
            report = {"final_state": "ABORTED", "error": f"{type(exc).__name__}: {exc}"}
        finally:
            px4.close()
        self.report_pub.publish(String(data=json.dumps(report)))
        self.get_logger().info(f"mission finished: {report.get('final_state')}, "
                               f"waypoints reached {report.get('waypoints_reached')}/{len(self.waypoints)}")


def main(args=None):
    rclpy.init(args=args)
    node = MissionManager()
    try:
        while rclpy.ok() and node.thread.is_alive():
            rclpy.spin_once(node, timeout_sec=0.2)
        rclpy.spin_once(node, timeout_sec=0.5)   # flush the final report
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
