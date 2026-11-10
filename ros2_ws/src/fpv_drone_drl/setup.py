from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'fpv_drone_drl'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*')),
        (os.path.join('share', package_name, 'models/fpv_drone'), glob('models/fpv_drone/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mansoor Ahmed',
    maintainer_email='175215439+UnknownEng@users.noreply.github.com',
    description='End-to-end Deep Reinforcement Learning pipeline for autonomous FPV quadcopter navigation.',
    license='BSD-3-Clause',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train_ppo = scripts.train_ppo:main',
        ],
    },
)
