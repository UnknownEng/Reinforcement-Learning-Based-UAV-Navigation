import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('uav_autonomy')
    config_file = os.path.join(pkg_share, 'config', 'sensor_bridge.yaml')

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_sensor_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/uav/sensors/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/uav/sensors/range_down@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'
        ],
        parameters=[{
            'use_sim_time': True
        }]
    )

    mavlink_bridge_node = Node(
        package='uav_autonomy',
        executable='px4_mavlink_bridge',
        name='px4_mavlink_bridge',
        output='screen',
        parameters=[{
            'use_sim_time': True
        }]
    )

    return LaunchDescription([
        bridge_node,
        mavlink_bridge_node
    ])
