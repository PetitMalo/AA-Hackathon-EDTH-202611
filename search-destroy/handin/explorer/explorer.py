"""One iteration of the exploration loop, called at 30 Hz:

    observation
        -> state estimate (IMU + VIO)
        -> voxel map update (ray cast the depth image from the estimated camera pose)
        -> goal selection:  target bbox seen?  -> go to the target estimate
                            otherwise          -> nearest frontier (FrontierPolicy, RL later)
        -> guidance:  use_mpc  -> safe flight corridor (free voxel boxes along drone -> goal) + MPC (QP)
                      otherwise -> waypoint controller
    -> joystick command (roll, pitch, yaw, throttle)
"""

import time
from dataclasses import dataclass, field

import numpy as np

from .camera_model import CameraModel
from .config import ExplorerConfig
from .controller import WaypointController
from .corridor import CorridorPlanner
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
    n_boxes: int = 0                       # corridor boxes (MPC mode)
    corridor_blocked: bool = False
    mpc_ms: float = 0.
    mpc_iters: int = 0
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
        self.planner = CorridorPlanner(c.corridor, c.mpc)
        self.info = StepInfo()

    def reset(self):
        self.estimator.reset()
        self.map.reset()
        self.frontiers.reset()
        self.target.reset()
        self.planner.reset()
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
            # glide slope: come down toward the box over a long shallow ramp, never climb for it
            dist_h = np.linalg.norm(goal[:2] - est.p[:2])
            goal[2] = min(est.p[2], goal[2] + c.control.approach_glide_slope * dist_h)
            dist = np.linalg.norm(target - est.p)
            ignore_beyond = max(np.linalg.norm(target - cam_pos) - 3., 0.)
            if c.use_attack and dist < c.attack.engage_dist_m:
                attack = self._attack_action(est, target, dist, depth)
                if attack is not None:
                    self.frontiers.update(self.map, context)   # keep the frontier bookkeeping alive
                    mode, action = "attack", attack
                    self._publish_phase(obs, mode)
                    self.info = StepInfo(mode=mode, goal=target.copy(), position=est.p.copy(), yaw=est.yaw,
                                         n_frontiers=self.frontiers.n_candidates, coverage=self.map.coverage(),
                                         target_estimate=target.copy(), action=action)
                    return action
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
        cam = (cam_pos, cam_R, self.camera)
        mpc_info = {}
        action = None
        if c.use_mpc:
            action, mpc_info = self._mpc_action(est, goal, vmax, mode, target, depth, cam, ignore_beyond)
        if action is None:
            action = self.controller.compute(est, goal, vmax, depth, cam, ignore_beyond)

        self._publish_phase(obs, mode)
        self.info = StepInfo(mode=mode, goal=goal, position=est.p.copy(), yaw=est.yaw,
                             target_estimate=None if target is None else target.copy(),
                             n_frontiers=self.frontiers.n_candidates, coverage=self.map.coverage(),
                             obstacle_ahead=self.controller.obstacle_ahead, action=action, **mpc_info)
        return action

    def _publish_phase(self, obs, mode):
        """Tell the evaluator our mission phase (it latches 'terminal' on the first target commit)."""
        try:
            obs["phase"] = self.cfg.phase_of_mode.get(mode)
        except Exception:
            pass

    def _path_clear_to(self, p, point):
        """The straight segment drone -> point (above the ground layer) only crosses known-FREE voxels.

        The bottom voxel layer [0, ground_layer_m) is the ground itself and is ignored: a low shallow
        dive onto a ground box legitimately crosses it next to the target. Walls and trees (>= 2 m)
        along the dive are still caught.
        """
        c = self.cfg.attack
        res = self.map.res
        n = max(2, int(np.ceil(np.linalg.norm(point - p) / (res / 2))) + 1)
        pts = p + np.linspace(0., 1., n)[:, None] * (point - p)
        idx, valid = self.map.world_to_index(pts)
        if not valid.all():
            return False
        keep = idx[:, 2] >= int(np.ceil((c.ground_layer_m - self.map.origin[2]) / res))
        if not keep.any():
            return True
        states = self.map.state(idx[keep])
        return bool((states == VoxelMap.FREE).sum() >= len(states) - c.clear_voxels_ahead)

    def _attack_action(self, est, target, dist, depth):
        """Close range, target freshly seen, drone low enough: skip the corridor/MPC and sprint straight
        at the box. Returns the sticks, or None when a gate fails (freshness/altitude/path) and the
        caller should fall back to the usual target guidance."""
        c = self.cfg.attack
        if self.target.time_since_seen_s > c.lost_timeout_s:
            return None                                      # no sprint on a stale estimate
        if est.p[2] > c.engage_max_alt_m:
            return None                                      # approach glide slope has not brought us down yet
        aim = target.copy()
        aim[2] = max(target[2], c.box_top_clamp_m) + c.aim_above_box_m
        if not self._path_clear_to(est.p, aim):
            return None                                      # wall / tree in the dive: keep the safe guidance
        e = target - est.p
        dist_h = np.linalg.norm(e[:2])
        e_z = aim[2] - est.p[2]
        yaw_stick, _ = self.controller.yaw_stick_toward(est.p, est.yaw, aim)
        a = np.zeros(3)
        # Dive authority limits how fast we can close: sprint horizontally only as fast as we can
        # drop onto the box, so the drone lands ON the box instead of skimming over it.
        dive_a = np.clip(self.cfg.control.alt_kp * e_z, -self.cfg.mpc.a_max_z_m_s2, 0.)
        a[2] = dive_a
        # Inside the box's reach: stop sprinting, kill the yaw (the nose is already on the box), commit the dive.
        if dist_h < c.settle_dist_m:
            yaw_stick = 0.0
            a[:2] = 0.3 * e[:2]                       # gentle brake toward the box center
        elif dist_h > 1e-3:
            # Match horizontal accel to dive: time to drop ~ sqrt(2*|e_z|/|dive_a|); reach box in that time.
            t_drop = np.sqrt(2. * max(abs(e_z), 0.5) / max(abs(dive_a), 1.0))
            v_h_need = dist_h / max(t_drop, 0.3)
            accel_h = min(c.accel_m_s2, max(v_h_need - np.linalg.norm(est.v[:2]), 0.) * 3.0)
            a[:2] = e[:2] / dist_h * accel_h
        return self.controller.accel_to_sticks(a, est.v, est.yaw, yaw_stick, tilt_deg=c.tilt_deg)

    def _mpc_action(self, est, goal, vmax, mode, target, depth, cam, ignore_beyond):
        """Corridor + MPC: acceleration from the QP, nose toward the goal, slower while the nose is off it.

        Returns (sticks, info), or (None, {}) when no corridor can be built (falls back to the waypoint controller).
        """
        c = self.cfg
        t0 = time.perf_counter()
        yaw_stick, yaw_err = self.controller.yaw_stick_toward(est.p, est.yaw, goal)
        speed = vmax * max(0., np.cos(yaw_err)) ** 2                    # fly mostly where the camera has looked
        clear = target if mode == "target" else None                  # the target box itself must be reachable
        a = self.planner.step(self.map, est.p, est.v, goal, speed, vmax, clear, terminal=(mode == "target"))
        if a is None:
            return None, {}
        self.controller.obstacle_ahead = False
        if c.corridor.reactive_safety and self.controller.check_obstacle_ahead(est.v, depth, cam, ignore_beyond):
            a = np.array([-c.control.vel_gain * est.v[0], -c.control.vel_gain * est.v[1],
                          c.control.alt_kd * (2. - est.v[2])])
        action = self.controller.accel_to_sticks(a, est.v, est.yaw, yaw_stick)
        cor = self.planner.corridor
        return action, {"n_boxes": len(cor), "corridor_blocked": cor.blocked,
                        "mpc_ms": 1e3 * (time.perf_counter() - t0), "mpc_iters": self.planner.mpc.iters}
