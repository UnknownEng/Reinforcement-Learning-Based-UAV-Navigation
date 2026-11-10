"""
baselines
---------
Milestone-9 classical navigation baselines for UAVWorldNav-v0 (rl_env/uav_world_env.py).

  APFController       artificial potential field            onboard LiDAR + observation
  DWAController       dynamic window approach               onboard LiDAR + observation
  RRTStarController   (Informed) RRT* + pure pursuit        GLOBAL MAP (env.world) + LiDAR for replanning
  CBFSafetyFilter     control-barrier-function QP filter    onboard LiDAR + observation; wraps any action

All controllers expose `act(obs, env) -> action (4,) in [-1, 1]` and `reset(seed=None)`.
See baselines/runner.py for `make_controller` and `evaluate`.
"""

from baselines.apf import APFConfig, APFController
from baselines.cbf import CBFConfig, CBFSafetyFilter, CBFSafetyWrapper, solve_projection_qp
from baselines.dwa import DWAConfig, DWAController
from baselines.rrt_star import (FollowerConfig, PlanResult, PurePursuit, RRTStarConfig, RRTStarController,
                                RRTStarPlanner, path_length, plan_path)
from baselines.runner import CONTROLLER_NAMES, evaluate, make_controller, summarize

__all__ = [
    "APFConfig", "APFController",
    "CBFConfig", "CBFSafetyFilter", "CBFSafetyWrapper", "solve_projection_qp",
    "DWAConfig", "DWAController",
    "FollowerConfig", "PlanResult", "PurePursuit", "RRTStarConfig", "RRTStarController", "RRTStarPlanner",
    "path_length", "plan_path",
    "CONTROLLER_NAMES", "evaluate", "make_controller", "summarize",
]
