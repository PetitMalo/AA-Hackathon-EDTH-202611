"""One iteration of the exploration loop, called at 30 Hz:

    observation
        -> state estimate (IMU + VIO)
        -> voxel map update (ray cast the depth image from the estimated camera pose)
        -> goal selection:  target bbox seen?  -> go to the target estimate
                            stuck exploring?   -> graded escape: re-plan / yaw scan / climb (progress.py)
                            otherwise          -> nearest frontier (FrontierPolicy, RL later)
        -> guidance:  use_mpc  -> safe flight corridor (free voxel boxes along drone -> goal) + MPC (QP)
                      otherwise -> waypoint controller
    -> joystick command (roll, pitch, yaw, throttle)
"""

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .camera_model import CameraModel
from .config import ExplorerConfig
from .controller import WaypointController
from .corridor import CorridorPlanner
from .frontiers import FrontierManager
from .progress import EscapeManager, column_clear
from .state_estimation import StateEstimator
from .target import TargetTracker
from .voxel_map import VoxelMap


@dataclass
class StepInfo:
    """What happened in the last iteration, for logging / debugging / an RL wrapper."""
    mode: str = "init"                     # init | explore | scan | climb | hover | target | attack
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
    vmax: float = 0.                       # speed cap of the current mode
    n_clutter: int = 0                     # obstacles near the drone (target phase only)
    margin: float = 0.                     # corridor margin given to the MPC
    stuck_disp: float = float("nan")       # net horizontal displacement over the escape window (nan: not full)
    escape_level: str = ""                 # last escape level started: replan | scan | climb
    stuck_reason: str = ""                 # what triggered it: stuck | blocked | clutter
    n_escapes: int = 0
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
        self.escape = EscapeManager(c.escape, dt=c.dt)
        self.info = StepInfo()
        self.frame, self.n_clutter_explore = 0, 0

    def reset(self):
        self.estimator.reset()
        self.map.reset()
        self.frontiers.reset()
        self.target.reset()
        self.planner.reset()
        self.escape.reset()
        self.info = StepInfo()
        self.frame, self.n_clutter_explore = 0, 0

    def step(self, obs):
        c = self.cfg
        self.frame += 1
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

        n_clutter, margin = 0, c.corridor.margin_m
        if self.target.active:
            self.escape.suspend()                          # the target phase is never "stuck"
            mode, goal = "target", target.copy()
            # glide slope: come down toward the box over a long shallow ramp, never climb for it
            dist_h = np.linalg.norm(goal[:2] - est.p[:2])
            goal[2] = min(est.p[2], goal[2] + c.control.approach_glide_slope * dist_h)
            dist = np.linalg.norm(target - est.p)
            n_clutter = self._count_obstacles_near(est.p, target)
            vmax = self._approach_speed(dist, n_clutter)
            if dist < c.corridor.near_target_radius_m:
                margin = c.corridor.margin_near_m
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
            # 4a) stuck? graded escape (re-plan / yaw scan / climb), executed here
            escape = self._escape_step(est) if c.escape.enabled else None
            if escape is not None:
                mode, goal, action = escape
                self._publish_phase(obs, mode)
                self.info = StepInfo(mode=mode, goal=goal, position=est.p.copy(), yaw=est.yaw,
                                     target_estimate=None if target is None else target.copy(),
                                     n_frontiers=self.frontiers.n_candidates, coverage=self.map.coverage(),
                                     action=action, **self._escape_info())
                return action
            # 4b) otherwise: next frontier
            goal = self.frontiers.update(self.map, context)
            mode, vmax, ignore_beyond = "explore", c.control.cruise_speed_m_s, np.inf
            if goal is None:
                mode, goal = "hover", est.p.copy()
            goal = goal.copy()
            lim = c.frontier.xy_limit_m
            goal[:2] = np.clip(goal[:2], -lim, lim)
            goal[2] = np.clip(goal[2], *self.frontiers.z_band)

        # 5) joystick command toward the goal
        cam = (cam_pos, cam_R, self.camera)
        mpc_info = {}
        action = None
        if c.use_mpc:
            action, mpc_info = self._mpc_action(est, goal, vmax, mode, target, depth, cam, ignore_beyond, margin)
        if action is None:
            action = self.controller.compute(est, goal, vmax, depth, cam, ignore_beyond)

        self._publish_phase(obs, mode)
        self.info = StepInfo(mode=mode, goal=goal, position=est.p.copy(), yaw=est.yaw,
                             target_estimate=None if target is None else target.copy(),
                             n_frontiers=self.frontiers.n_candidates, coverage=self.map.coverage(),
                             obstacle_ahead=self.controller.obstacle_ahead, action=action,
                             vmax=vmax, n_clutter=n_clutter, margin=margin, **mpc_info, **self._escape_info())
        return action

    # ------------------------------------------------------------------ escape (exploration only)
    def _escape_step(self, est):
        """Stuck detection + the running escape level. Returns (mode, goal, sticks) while a scan or a climb
        flies the drone, None when the usual frontier guidance should run (possibly after a forced re-plan)."""
        e, ce = self.escape, self.cfg.escape
        if e.mode is None:
            cor = self.planner.corridor
            blocked = bool(cor is not None and cor.blocked)
            if self.frame % ce.clutter_every == 0:
                self.n_clutter_explore = self._count_obstacles_near(est.p, None, ce.clutter_radius_m)
            level = e.update(est.p, est.v, blocked, self.n_clutter_explore)
            if level is None:
                return None
            if level == "climb" and (est.p[2] >= ce.climb_alt_m - ce.climb_tol_m
                                     or not column_clear(self.map, est.p, ce.climb_alt_m, ce.column_half_width_m)):
                level = "replan"                           # already high, or a roof / canopy right above
            # every level abandons the goal we were stuck on, and the place we were stuck in
            self.frontiers.force_replan([self.frontiers.goal, est.p], ce.blacklist_radius_m, ce.blacklist_s)
            if level == "replan":
                e.monitor.clear(ce.cooldown_s)
                return None
            e.start(level, est.p)
        else:
            e.update(est.p, est.v, False)                   # escape clock only

        if e.mode == "scan" and not e.scan_done():
            z_ref = e.start_p[2]
            return "scan", e.start_p.copy(), self._hold_action(est, e.start_p, z_ref, ce.scan_yaw_stick, 2.)
        if e.mode == "climb" and not e.climb_done(est.p, est.v):
            goal = np.array([*e.start_p[:2], ce.climb_alt_m])
            return "climb", goal, self._hold_action(est, e.start_p, ce.climb_alt_m, 0., ce.climb_speed_m_s)
        # the scan / climb just ended: back to the frontiers, with a fresh plan
        if e.mode == "climb":
            self.frontiers.widen_z_band(ce.widen_z_band_m, ce.widen_s)
        e.finish()
        self.frontiers.force_replan()
        self.planner.reset()
        return None

    def _hold_action(self, est, p_hold, z_ref, yaw_stick, vz_max):
        """Hold the horizontal position p_hold (slow P-PD loop), go to altitude z_ref at <= vz_max."""
        c = self.cfg.control
        v_des = np.zeros(3)
        v_des[:2] = np.clip(0.5 * (p_hold[:2] - est.p[:2]), -2., 2.)
        v_des[2] = np.clip(c.alt_kp * (z_ref - est.p[2]), -vz_max, vz_max)
        a = np.zeros(3)
        a[:2] = c.vel_gain * (v_des[:2] - est.v[:2])
        a[2] = c.alt_kd * (v_des[2] - est.v[2])
        return self.controller.accel_to_sticks(a, est.v, est.yaw, yaw_stick)

    def _escape_info(self):
        e = self.escape
        disp = e.monitor.displacement
        return {"stuck_disp": float("nan") if disp is None else disp, "escape_level": e.level_name,
                "stuck_reason": e.reason or "", "n_escapes": e.n_escapes}

    def _approach_speed(self, dist, n_clutter):
        """Target-phase speed cap: terminal speed (5 m/s clear, down to 3 m/s in clutter) at
        terminal_radius_m from the box, reached through a constant-deceleration profile from the far speed."""
        c = self.cfg.control
        k = np.clip((n_clutter - c.clutter_free_count) / max(c.clutter_max_count - c.clutter_free_count, 1), 0., 1.)
        v_term = c.terminal_speed_m_s + k * (c.terminal_speed_min_m_s - c.terminal_speed_m_s)
        v_brake = np.sqrt(v_term ** 2 + 2. * c.terminal_brake_m_s2 * max(dist - c.terminal_radius_m, 0.))
        return float(min(c.approach_speed_m_s, v_brake))

    def _count_obstacles_near(self, p, target, radius_m=None):
        """Number of separate obstacles (26-connected groups of occupied voxels) within radius_m
        (default control.clutter_radius_m) of the drone, leaving out the ground layer and the box itself
        (target None: no box to leave out)."""
        c, m = self.cfg.control, self.map
        r = c.clutter_radius_m if radius_m is None else radius_m
        lo = np.clip(np.floor((p - r - m.origin) / m.res).astype(int), 0, m.shape)
        hi = np.clip(np.floor((p + r - m.origin) / m.res).astype(int) + 1, 0, m.shape)
        if np.any(hi <= lo):
            return 0
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        occ = m.observed[sl] & (m.logodds[sl] > 0)
        centres = [m.origin[i] + (np.arange(lo[i], hi[i]) + .5) * m.res for i in range(3)]
        cx, cy, cz = np.meshgrid(*centres, indexing="ij")
        pts = np.stack([cx, cy, cz], axis=-1)
        occ &= np.linalg.norm(pts - p, axis=-1) <= r
        occ &= cz >= c.clutter_ground_m
        if target is not None:
            occ &= np.linalg.norm(pts - target, axis=-1) > c.clutter_box_clear_m
        if not occ.any():
            return 0
        return int(ndimage.label(occ, structure=np.ones((3, 3, 3)))[1])

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

    def _mpc_action(self, est, goal, vmax, mode, target, depth, cam, ignore_beyond, margin=None):
        """Corridor + MPC: acceleration from the QP, nose toward the goal, slower while the nose is off it.

        Returns (sticks, info), or (None, {}) when no corridor can be built (falls back to the waypoint controller).
        """
        c = self.cfg
        t0 = time.perf_counter()
        yaw_stick, yaw_err = self.controller.yaw_stick_toward(est.p, est.yaw, goal)
        speed = vmax * max(0., np.cos(yaw_err)) ** 2                    # fly mostly where the camera has looked
        clear = target if mode == "target" else None                  # the target box itself must be reachable
        a = self.planner.step(self.map, est.p, est.v, goal, speed, vmax, clear, terminal=(mode == "target"),
                              margin=margin)
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
