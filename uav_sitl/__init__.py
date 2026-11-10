"""
uav_sitl
--------
Automation for PX4 SITL + Gazebo Harmonic experiments, used for honest
closed-loop validation of the learned policies (no scripted waypoints):

  px4_link      MAVLink client: telemetry, arming, offboard velocity control, landing
  gz_link       Gazebo transport: LiDAR scans, ground-truth poses, moving obstacles
  sim_launcher  start/stop PX4 SITL + Gazebo for a given world
"""
