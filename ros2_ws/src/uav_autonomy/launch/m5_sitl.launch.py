import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('uav_autonomy')
    bridge_launch = os.path.join(pkg_share, 'launch', 'm4_sensor_bridge.launch.py')
    return LaunchDescription([
        DeclareLaunchArgument(
            'model_path',
            default_value=os.path.expanduser('~/proj/experiments/milestone5/models/ppo_m5_stage3_completion_run1.zip'),
        ),
        DeclareLaunchArgument('goal_x', default_value='4.0'),
        DeclareLaunchArgument('goal_y', default_value='2.0'),
        DeclareLaunchArgument('goal_z', default_value='2.5'),
        DeclareLaunchArgument('safety_margin_m', default_value='0.55'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(bridge_launch)),
        Node(
            package='uav_autonomy',
            executable='m5_ppo_inference_adapter',
            name='m5_ppo_inference_adapter',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'model_path': LaunchConfiguration('model_path'),
                'goal_x': LaunchConfiguration('goal_x'),
                'goal_y': LaunchConfiguration('goal_y'),
                'goal_z': LaunchConfiguration('goal_z'),
                'safety_margin_m': LaunchConfiguration('safety_margin_m'),
            }],
        ),
    ])
