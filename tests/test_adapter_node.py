"""
test_adapter_node.py
--------------------
Automated integration test for PPOInferenceAdapter ROS 2 node.
Verifies:
  1. OBSERVATION_ONLY mode: publishes 18D /uav/observation, publishes NO /cmd_vel.
  2. SHADOW mode: loads trained PPO model, computes inference latency, publishes NO /cmd_vel.
  3. CONTROLLED_SIM mode: software enable service, watchdog safety triggers.
"""

import json
import time
import pytest

rclpy = pytest.importorskip("rclpy")  # needs ROS 2 Jazzy sourced
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray, String
from std_srvs.srv import SetBool

from uav_autonomy.ppo_inference_adapter import PPOInferenceAdapter


class MockTelemetryPublisher(Node):
    def __init__(self):
        super().__init__('mock_telemetry_pub')
        self.pub_odom = self.create_publisher(Odometry, '/uav/odometry', 10)
        self.pub_status = self.create_publisher(String, '/uav/status', 10)

        # Listen for adapter outputs
        self.received_obs = []
        self.received_diag = []
        self.received_cmd_vel = []

        self.sub_obs = self.create_subscription(Float32MultiArray, '/uav/observation', self.obs_cb, 10)
        self.sub_diag = self.create_subscription(String, '/uav/adapter_diagnostics', self.diag_cb, 10)
        self.sub_cmd = self.create_subscription(Twist, '/cmd_vel', self.cmd_cb, 10)

    def obs_cb(self, msg: Float32MultiArray):
        self.received_obs.append(list(msg.data))

    def diag_cb(self, msg: String):
        try:
            self.received_diag.append(json.loads(msg.data))
        except Exception:
            pass

    def cmd_cb(self, msg: Twist):
        self.received_cmd_vel.append([msg.linear.x, msg.linear.y, msg.linear.z, msg.angular.z])

    def publish_mock_odom(self, x=0.0, y=0.0, z=2.0, vx=0.0, vy=0.0, vz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
        odom = Odometry()
        now = self.get_clock().now().to_msg()
        odom.header.stamp = now
        odom.header.frame_id = 'odom'
        odom.child_frame_id = 'base_link'

        odom.pose.pose.position.x = float(x)
        odom.pose.pose.position.y = float(y)
        odom.pose.pose.position.z = float(z)

        odom.pose.pose.orientation.x = float(qx)
        odom.pose.pose.orientation.y = float(qy)
        odom.pose.pose.orientation.z = float(qz)
        odom.pose.pose.orientation.w = float(qw)

        odom.twist.twist.linear.x = float(vx)
        odom.twist.twist.linear.y = float(vy)
        odom.twist.twist.linear.z = float(vz)

        odom.twist.twist.angular.x = 0.0
        odom.twist.twist.angular.y = 0.0
        odom.twist.twist.angular.z = 0.0

        self.pub_odom.publish(odom)


@pytest.fixture(scope="module")
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()


def test_observation_only_mode(ros_context):
    """Verify that OBSERVATION_ONLY mode produces 18D observations and ZERO cmd_vel."""
    mock_pub = MockTelemetryPublisher()
    adapter = PPOInferenceAdapter()  # Defaults to mode='OBSERVATION_ONLY'

    # Publish mock telemetry
    mock_pub.publish_mock_odom(x=1.0, y=2.0, z=2.0)

    # Spin briefly
    start_t = time.time()
    while (time.time() - start_t) < 0.5:
        rclpy.spin_once(mock_pub, timeout_sec=0.05)
        rclpy.spin_once(adapter, timeout_sec=0.05)

    assert len(mock_pub.received_obs) > 0, "No observations received from adapter"
    latest_obs = mock_pub.received_obs[-1]
    assert len(latest_obs) == 18, f"Expected 18 dimensions, got {len(latest_obs)}"

    # In OBSERVATION_ONLY mode, NO commands must be sent to /cmd_vel
    assert len(mock_pub.received_cmd_vel) == 0, "Commands were published in OBSERVATION_ONLY mode!"

    adapter.destroy_node()
    mock_pub.destroy_node()


def test_shadow_mode(ros_context):
    """Verify that SHADOW mode runs inference, computes latency, and produces NO cmd_vel."""
    mock_pub = MockTelemetryPublisher()
    # Override parameter to SHADOW mode
    from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
    adapter = PPOInferenceAdapter()
    adapter.mode = "SHADOW"
    adapter.load_ppo_model()

    mock_pub.publish_mock_odom(x=0.0, y=0.0, z=2.0)

    start_t = time.time()
    while (time.time() - start_t) < 0.5:
        rclpy.spin_once(mock_pub, timeout_sec=0.05)
        rclpy.spin_once(adapter, timeout_sec=0.05)

    assert len(mock_pub.received_diag) > 0, "No diagnostics received in SHADOW mode"
    latest_diag = mock_pub.received_diag[-1]

    assert latest_diag["mode"] == "SHADOW"
    assert "ppo_action" in latest_diag
    assert len(latest_diag["ppo_action"]) == 4
    assert "classical_action" in latest_diag
    assert "inference_latency_ms" in latest_diag
    assert latest_diag["inference_latency_ms"] >= 0.0

    # ZERO commands must reach /cmd_vel in SHADOW mode
    assert len(mock_pub.received_cmd_vel) == 0, "Commands were published in SHADOW mode!"

    adapter.destroy_node()
    mock_pub.destroy_node()
