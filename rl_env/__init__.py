"""
rl_env package
--------------
Reinforcement Learning Environment for Autonomous UAV Flight.
"""

from gymnasium.envs.registration import register

from rl_env.config import UAVEnvConfig
from rl_env.action import ActionHandler, CommandedVelocities
from rl_env.observation import ObservationHandler
from rl_env.dynamics import FastUAVDynamics, UAVState
from rl_env.reward import RewardCalculator, RewardComponents
from rl_env.termination import TerminationChecker, TerminationStatus
from rl_env.uav_fast_env import UAVFastEnv
from rl_env.uav_fast_env_v1 import UAVFastEnvV1, UAVObstacleEnvConfig
from rl_env.uav_fast_env_v2 import UAVFastEnvV2, UAVObstacleEnvConfigV2
from rl_env.safety_shield import (
    DeterministicSafetyShield,
    SafetyShieldConfig,
    SafetyShieldDiagnostic,
    SafetyShieldWrapper
)

# Register Fast UAV Navigation Environment with Gymnasium (Frozen 18-D Baseline)
register(
    id="UAVFastNav-v0",
    entry_point="rl_env.uav_fast_env:UAVFastEnv",
    max_episode_steps=200,
)

# Register Upgraded Obstacle Navigation Environment with Gymnasium (26-D Option 1)
register(
    id="UAVFastNav-v1",
    entry_point="rl_env.uav_fast_env_v1:UAVFastEnvV1",
    max_episode_steps=200,
)

# Register Advanced Generalization Environment with Gymnasium (26-D Milestone 5)
register(
    id="UAVFastNav-v2",
    entry_point="rl_env.uav_fast_env_v2:UAVFastEnvV2",
    max_episode_steps=300,
)

# M6-M10: 3-D forest / urban / warehouse worlds with a multi-channel LiDAR.
# The environment enforces its own horizon (config.max_episode_steps).
register(
    id="UAVWorldNav-v0",
    entry_point="rl_env.uav_world_env:UAVWorldEnv",
)

__all__ = [
    "UAVEnvConfig",
    "UAVObstacleEnvConfig",
    "UAVObstacleEnvConfigV2",
    "ActionHandler",
    "CommandedVelocities",
    "ObservationHandler",
    "FastUAVDynamics",
    "UAVState",
    "RewardCalculator",
    "RewardComponents",
    "TerminationChecker",
    "TerminationStatus",
    "UAVFastEnv",
    "UAVFastEnvV1",
    "UAVFastEnvV2",
    "DeterministicSafetyShield",
    "SafetyShieldConfig",
    "SafetyShieldDiagnostic",
    "SafetyShieldWrapper",
]
