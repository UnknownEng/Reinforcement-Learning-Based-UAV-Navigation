import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'uav_autonomy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mansoor Ahmed',
    maintainer_email='175215439+UnknownEng@users.noreply.github.com',
    description='ROS 2 interface and autonomy controllers for PX4 SITL UAV',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'px4_mavlink_bridge = uav_autonomy.px4_mavlink_bridge:main',
            'perception_adapter = uav_autonomy.perception_adapter:main',
            'diagnostic_adapter = uav_autonomy.diagnostic_adapter:main',
            'diagnostic_evaluator = uav_autonomy.diagnostic_evaluator:main',
            'test_flight_baseline = uav_autonomy.test_flight_baseline:main',
            'benchmark_stack = uav_autonomy.benchmark_stack:main',
            'ppo_inference_adapter = uav_autonomy.ppo_inference_adapter:main',
            'm5_ppo_inference_adapter = uav_autonomy.m5_ppo_inference_adapter:main',
            'mission_evaluator = uav_autonomy.mission_evaluator:main',
            'mission_manager = uav_autonomy.mission_manager:main',
        ],
    },
)
