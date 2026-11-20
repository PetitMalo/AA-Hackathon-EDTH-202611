"""One iteration of the exploration loop, called at 30 Hz:

    observation
        -> state estimate (IMU + VIO)
        -> voxel map update (ray cast the depth image from the estimated camera pose)
        -> goal selection:  target bbox seen?  -> go to the target estimate
                            otherwise          -> nearest frontier (FrontierPolicy, RL later)
        -> waypoint controller
    -> joystick command (roll, pitch, yaw, throttle)
"""

from dataclasses import dataclass, field

import numpy as np

from .camera_model import CameraModel
from .config import ExplorerConfig
from .controller import WaypointController
from .frontiers import FrontierManager
from .state_estimation import StateEstimator
from .target import TargetTracker
from .voxel_map import VoxelMap


@dataclass
class StepInfo:
    """What happened in the last iteration, for logging / debugging / an RL wrapper."""
    mode: str = "init"                     # init | explore | target | hover
    goal: np.ndarray = None
    position: np.ndarray = None
    yaw: float = 0.
    target_estimate: np.ndarray = None
    n_frontiers: int = 0
    coverage: float = 0.
    obstacle_ahead: bool = False
    action: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))


class FrontierExplorer:
    def __init__(self, cfg: ExplorerConfig = None, frontier_policy=None):
        self.cfg = cfg or ExplorerConfig()
        c = self.cfg
        self.camera = CameraModel(c.camera)
        self.estimator = StateEstimator(dt=c.dt, g=c.g)
        self.map = VoxelMap(c.map)
        self.frontiers = FrontierManager(c.frontier, policy=frontier_policy, dt=c.dt)
        self.target = TargetTracker(c.target, c.camera, c.map.max_ray_m, dt=c.dt)
        self.controller = WaypointController(c.control, c.camera, g=c.g)
        self.info = StepInfo()

    def reset(self):
        self.estimator.reset()
        self.map.reset()
        self.frontiers.reset()
        self.target.reset()
        self.info = StepInfo()

    def step(self, obs):
        c = self.cfg
        # 1) state estimation
        self.estimator.update(obs)
        est = self.estimator
        if not est.ready:                          # first frames: neutral sticks, let the heading align
            self.info = StepInfo(mode="init", position=est.p.copy(), yaw=est.yaw)
            return self.info.action

        depth = np.asarray(obs["depth_m"])
        cam_pos, cam_R = self.camera.pose(est.R, est.p)

        # 2) map update from the new image
        self.map.integrate_depth(depth, cam_pos, cam_R, self.camera)

        # 3) target in view? it becomes the goal
        target = self.target.update(obs, depth, cam_pos, cam_R, self.camera)
        context = {"position": est.p, "yaw": est.yaw, "velocity": est.v, "map": self.map, "target": target}

        if self.target.active:
            mode, goal, vmax = "target", target.copy(), c.control.approach_speed_m_s
            # glide slope: come down toward the box while closing in, never climb for it
            dist_h = np.linalg.norm(goal[:2] - est.p[:2])
            goal[2] = min(est.p[2], goal[2] + 0.4 * dist_h)
            ignore_beyond = max(np.linalg.norm(target - cam_pos) - 3., 0.)
        else:
            # 4) otherwise: next frontier
            goal = self.frontiers.update(self.map, context)
            mode, vmax, ignore_beyond = "explore", c.control.cruise_speed_m_s, np.inf
            if goal is None:
                mode, goal = "hover", est.p.copy()
            goal = goal.copy()
            lim = c.frontier.xy_limit_m
            goal[:2] = np.clip(goal[:2], -lim, lim)
            goal[2] = np.clip(goal[2], *c.frontier.z_band_m)

        # 5) joystick command toward the goal
        action = self.controller.compute(est, goal, vmax, depth, (cam_pos, cam_R, self.camera), ignore_beyond)

        self.info = StepInfo(mode=mode, goal=goal, position=est.p.copy(), yaw=est.yaw,
                             target_estimate=None if target is None else target.copy(),
                             n_frontiers=self.frontiers.n_candidates, coverage=self.map.coverage(),
                             obstacle_ahead=self.controller.obstacle_ahead, action=action)
        return action
