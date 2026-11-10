import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_share = get_package_share_directory('uav_autonomy')
    params_file = os.path.join(pkg_share, 'config', 'perception_params.yaml')

    bridge_node = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_sensor_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            '/uav/sensors/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            '/uav/sensors/range_down@sensor_msgs/msg/Range[gz.msgs.LaserScan'
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

    perception_adapter_node = Node(
        package='uav_autonomy',
        executable='perception_adapter',
        name='perception_adapter',
        output='screen',
        parameters=[
            params_file,
            {'use_sim_time': True}
        ]
    )

    # Static transforms: base_link -> lidar_link (+0.12m X, 0.0m Y, +0.242m Z)
    tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_lidar',
        arguments=['--x', '0.12', '--y', '0.0', '--z', '0.242',
                   '--roll', '0.0', '--pitch', '0.0', '--yaw', '0.0',
                   '--frame-id', 'base_link', '--child-frame-id', 'lidar_link']
    )

    # Static transforms: base_link -> range_down_link (0.0m X, 0.0m Y, -0.05m Z, pitch -90 deg)
    tf_range = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='tf_base_to_range_down',
        arguments=['--x', '0.0', '--y', '0.0', '--z', '-0.05',
                   '--roll', '0.0', '--pitch', '-1.57079632679', '--yaw', '0.0',
                   '--frame-id', 'base_link', '--child-frame-id', 'range_down_link']
    )

    return LaunchDescription([
        bridge_node,
        mavlink_bridge_node,
        perception_adapter_node,
        tf_lidar,
        tf_range
    ])
