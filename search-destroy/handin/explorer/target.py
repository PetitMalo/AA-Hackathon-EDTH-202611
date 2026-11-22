"""Target phase: world-frame estimate of the box (TargetTracker) and the camera-locked pursuit (TargetPursuit).

TargetTracker: range comes from the depth map inside the box when it is within depth range (< 40 m),
otherwise from intersecting the box-centre ray with the horizontal plane at the box centre height
(the target always rests on the ground). It also collects the BOX VOXELS: voxels hit by the depth pixels
inside the detection box, at the box's depth. They are the only voxels the corridor may treat as free
around the target (contact = success): every other obstacle near the box stays an obstacle.

TargetPursuit: the camera stays on the box the whole time and the drone flies the line of sight to its centre.
The camera is fixed on the airframe, tilted 25 deg UP, +-34.8 deg vertical field of view:
    C1 yaw     locked on the box azimuth measured IN THE IMAGE (bbox centre; the estimate when not seen),
               independent of the path: the corridor + MPC moves the drone, the yaw points the camera
    C2 LOS     goal altitude = box_z + d_h tan(los_deg): the drone stays high and only comes down with the
               distance, along a straight line of sight (10 deg -> 18 m at 100 m, 4.3 m at 20 m)
    C3 image   the line-of-sight angle measured in the image (attitude compensated) is held at los_deg by
               correcting the goal altitude: a fixed sight angle on a static box = a collision course
    C4 view    pitch floor theta >= delta - (34.8 - view_margin_deg) + 25 deg: never pitch the nose up so
               far that the box leaves the bottom of the image (only while the corridor reaches the box:
               an obstacle ahead has priority). At 10 deg the drag alone brings the speed to ~8 m/s.
    strike     last strike_dist_m, box freshly seen, depth image clear on the way: own controller, straight
               line of sight to the box centre at strike_speed_m_s (no corridor: the ground layer of the map
               is occupied, only the box voxels are free in it); steering frozen in the last commit_dist_m.
"""

from dataclasses import dataclass

import numpy as np


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class TargetTracker:
    def __init__(self, cfg, cam_cfg, max_ray_m, dt=1. / 30.):
        self.cfg, self.cam_cfg = cfg, cam_cfg
        self.max_ray = max_ray_m
        self.dt = dt
        self.reset()

    def reset(self):
        self.estimate = None
        self.time_since_seen_s = np.inf
        self.n_detections = 0
        self.box_voxels = set()                   # voxel indices (i, j, k) of the box surface
        self.near_unseen_s = 0.                   # time spent close to the estimate without seeing the box

    @property
    def box_voxel_idx(self):
        """(N, 3) voxel indices of the box (empty until the box has been seen within depth range)."""
        if not self.box_voxels:
            return np.zeros((0, 3), dtype=int)
        return np.array(sorted(self.box_voxels), dtype=int)

    @property
    def confident(self):
        """The box was localised with the depth (box voxels known) or detected many times."""
        return bool(self.box_voxels) or self.n_detections >= self.cfg.confident_detections

    @property
    def active(self):
        """A target estimate exists and is fresh enough to be the next goal: lost_timeout_s without a detection,
        lost_timeout_confident_s for a well localised box (its position is known, the camera is kept on it)."""
        timeout = self.cfg.lost_timeout_confident_s if self.confident else self.cfg.lost_timeout_s
        return self.estimate is not None and self.time_since_seen_s < timeout

    def check_presence(self, p):
        """The drone is close to the estimate and has not seen the box for dismiss_s: the estimate was wrong,
        forget it (back to exploration; a new detection starts afresh)."""
        if self.estimate is None:
            return
        close = np.linalg.norm(self.estimate[:2] - np.asarray(p)[:2]) < self.cfg.dismiss_dist_m
        self.near_unseen_s = self.near_unseen_s + self.dt if (close and self.time_since_seen_s > 0.) else 0.
        if self.near_unseen_s >= self.cfg.dismiss_s:
            self.reset()

    def update(self, obs, depth_m, cam_pos, cam_R, camera, vmap=None):
        self.time_since_seen_s += self.dt
        if not obs["detected"]:
            return self.estimate
        point = self.localise(obs["box"], depth_m, cam_pos, cam_R, camera)
        if point is None:
            return self.estimate
        self.n_detections += 1
        self.time_since_seen_s = 0.
        self.near_unseen_s = 0.
        a = self.cfg.ema
        if self.estimate is not None and np.linalg.norm(point[:2] - self.estimate[:2]) > self.cfg.box_reset_m:
            self.box_voxels = set()               # the estimate jumped: its box voxels belong to the old one
        self.estimate = point if self.estimate is None else (1 - a) * self.estimate + a * point
        if vmap is not None:
            self._mark_box_voxels(obs["box"], depth_m, cam_pos, cam_R, camera, vmap)
        return self.estimate

    def _mark_box_voxels(self, box_norm, depth_m, cam_pos, cam_R, camera, vmap):
        """Voxels hit by the depth pixels inside the bbox that lie at the box's depth (the background seen
        around the box is farther), below box_max_z_m and close to the estimate."""
        c, t = self.cam_cfg, self.cfg
        x1, y1, x2, y2 = camera.bbox_to_pixels(box_norm, c.depth_width, c.depth_height)
        r0, r1 = max(int(np.floor(y1)), 0), min(int(np.ceil(y2)) + 1, c.depth_height)
        c0, c1 = max(int(np.floor(x1)), 0), min(int(np.ceil(x2)) + 1, c.depth_width)
        if r1 <= r0 or c1 <= c0:
            return
        patch = np.asarray(depth_m)[r0:r1, c0:c1].astype(float)
        d_box = float(np.quantile(patch, 0.3))
        if d_box >= min(self.max_ray, t.box_voxel_max_range_m):
            return                                # too far: depth too noisy / no return
        rows, cols = np.nonzero(np.abs(patch - d_box) < t.box_depth_tol_m)
        if len(rows) == 0:
            return
        rays = camera.rays_cam(cols + c0 + .5, rows + r0 + .5, c.depth_width, c.depth_height)
        pts = cam_pos + patch[rows, cols][:, None] * (rays @ cam_R.T)
        keep = (pts[:, 2] < t.box_max_z_m) & (np.linalg.norm(pts[:, :2] - self.estimate[:2], axis=1)
                                                < t.box_max_dist_m)
        idx, valid = vmap.world_to_index(pts[keep])
        for v in map(tuple, idx[valid]):
            if len(self.box_voxels) >= t.max_box_voxels:
                break
            self.box_voxels.add(v)

    def localise(self, box_norm, depth_m, cam_pos, cam_R, camera):
        c = self.cam_cfg
        x1, y1, x2, y2 = camera.bbox_to_pixels(box_norm, c.depth_width, c.depth_height)
        uc, vc = (x1 + x2) / 2, (y1 + y2) / 2
        ray_world = cam_R @ camera.rays_cam(uc, vc, c.depth_width, c.depth_height)

        # 1) depth inside the box (near quantile: the box is in front of what is behind it)
        r0, r1 = int(np.floor(y1)), int(np.ceil(y2)) + 1
        c0, c1 = int(np.floor(x1)), int(np.ceil(x2)) + 1
        patch = np.asarray(depth_m)[max(r0, 0):r1, max(c0, 0):c1]
        if patch.size:
            d = float(np.quantile(patch, 0.3))
            if d < self.max_ray:
                return cam_pos + d * ray_world

        # 2) far away: ray / ground-plane intersection at the box centre height
        if ray_world[2] > -1e-3:
            return None                                   # ray does not go down: no range
        s = (self.cfg.box_center_z_m - cam_pos[2]) / ray_world[2]
        return cam_pos + s * ray_world if s > 0 else None


# ---------------------------------------------------------------------- camera-locked pursuit
@dataclass
class PursuitPlan:
    mode: str                              # target (corridor + MPC) | strike | brake (own controller)
    goal: np.ndarray
    vmax: float = 0.
    look: np.ndarray = None                # nose aimed at this point (the box direction)
    pitch_min: float = None                # rad, C4 floor for the sticks (None: no floor)
    action: np.ndarray = None              # strike: sticks


def path_obstacles(depth, cam_pos, cam_R, grid, p_from, p_to, box, cfg, vmap=None, box_voxels=None,
                   bbox_px=None, d_box=None):
    """Depth points (subsampled grid) within cfg.tube_m of the segment p_from -> p_to: what the drone would
    hit before the box. Excluded: the ground (< cfg.ground_m) and the box itself, i.e.
      - the pixels inside this frame's detection box (bbox_px, padded by cfg.bbox_pad_px) that are at the
        box's depth d_box or behind it (within cfg.box_depth_tol_m): close up, the box fills the view;
      - the points in the tracker's box voxels; until those are known, the points within cfg.box_clear_m
        (horizontally) of the estimate.
    A pixel in the detection box clearly NEARER than the box is something in front of it: an obstacle."""
    vv, uu, rays = grid
    d = np.asarray(depth)[vv, uu].astype(float)
    hit = d < cfg.max_depth_m
    if bbox_px is not None and d_box is not None:
        x1, y1, x2, y2 = bbox_px
        pad = cfg.bbox_pad_px
        inside = (uu >= x1 - pad) & (uu <= x2 + pad) & (vv >= y1 - pad) & (vv <= y2 + pad)
        hit &= ~(inside & (d >= d_box - cfg.box_depth_tol_m))
    if not hit.any():
        return 0
    pts = cam_pos + d[hit, None] * (rays[hit] @ cam_R.T)
    keep = pts[:, 2] > cfg.ground_m
    if vmap is not None and box_voxels:
        idx, valid = vmap.world_to_index(pts)
        keep &= ~np.array([valid[i] and tuple(idx[i]) in box_voxels for i in range(len(pts))], dtype=bool)
    else:
        keep &= np.linalg.norm(pts[:, :2] - box[:2], axis=1) > cfg.box_clear_m
    pts = pts[keep]
    if len(pts) == 0:
        return 0
    u = p_to - p_from
    t = np.clip((pts - p_from) @ u / max(u @ u, 1e-9), 0., 1.)
    return int((np.linalg.norm(pts - (p_from + t[:, None] * u), axis=1) < cfg.tube_m).sum())


class TargetPursuit:
    def __init__(self, cfg, cam_cfg, control_cfg, controller, dt=1. / 30.):
        self.cfg, self.cam_cfg, self.ctl = cfg, cam_cfg, control_cfg
        self.controller = controller
        self.dt = dt
        self.half_vfov = float(np.arctan(np.tan(np.radians(cam_cfg.hfov_deg) / 2)
                                         * cam_cfg.depth_height / cam_cfg.depth_width))
        self._grid = None
        self.reset()

    def reset(self):
        self.mode = "target"
        self.mode_s = 0.
        self.commit_az = None
        self.delta_deg = float("nan")          # line-of-sight angle below the horizon (image or estimate)
        self.obstacle_pts = 0
        self.obstacle_frames = 0               # consecutive frames with an obstacle on the strike line
        self.n_strikes = 0
        self.strike_abort = ""                 # last strike given up: obstacle | lost
        self.floor_on = False                  # C4 pitch floor applied at the last pursuit step
        self.reacquire = False                 # T3: backing off onto the line of sight to see the box again
        self._brake_z = None

    # ------------------------------------------------------------------ measurements
    def sight(self, obs, cam_R, camera, p, box):
        """(azimuth, depression) of the box centre, world frame (rad): from the bbox when detected this
        frame, from the estimate otherwise; and whether it came from the image."""
        if obs["detected"]:
            cc = self.cam_cfg
            x1, y1, x2, y2 = camera.bbox_to_pixels(obs["box"], cc.depth_width, cc.depth_height)
            ray = cam_R @ camera.rays_cam((x1 + x2) / 2, (y1 + y2) / 2, cc.depth_width, cc.depth_height)
            return float(np.arctan2(ray[1], ray[0])), float(np.arctan2(-ray[2], np.linalg.norm(ray[:2]))), True
        e = box - p
        return float(np.arctan2(e[1], e[0])), float(np.arctan2(-e[2], np.linalg.norm(e[:2]))), False

    def pitch_floor(self, delta):
        """C4: smallest pitch (rad, nose down > 0) keeping a box seen delta below the horizon inside the
        image, view_margin_deg above its bottom edge."""
        up = np.radians(self.cam_cfg.uptilt_deg)
        return float(delta - (self.half_vfov - np.radians(self.cfg.view_margin_deg)) + up)

    def _bbox_depth(self, obs, depth, camera):
        """(bbox in depth pixels, box depth) of this frame's detection, or (None, None)."""
        if not obs["detected"]:
            return None, None
        cc = self.cam_cfg
        x1, y1, x2, y2 = camera.bbox_to_pixels(obs["box"], cc.depth_width, cc.depth_height)
        r0, r1 = max(int(np.floor(y1)), 0), min(int(np.ceil(y2)) + 1, cc.depth_height)
        c0, c1 = max(int(np.floor(x1)), 0), min(int(np.ceil(x2)) + 1, cc.depth_width)
        if r1 <= r0 or c1 <= c0:
            return None, None
        return (x1, y1, x2, y2), float(np.quantile(np.asarray(depth)[r0:r1, c0:c1], 0.3))

    # ------------------------------------------------------------------ one step
    def update(self, est, tracker, obs, depth, cam_pos, cam_R, camera, vmap, path_clear=True):
        """path_clear: the corridor is not blocked (T1: the C4 floor is on unless an obstacle stops us)."""
        c = self.cfg
        self.mode_s += self.dt
        p = est.p
        box = np.asarray(tracker.estimate, float)
        az, delta, seen = self.sight(obs, cam_R, camera, p, box)
        self.delta_deg = float(np.degrees(delta))
        d_h = float(np.linalg.norm(box[:2] - p[:2]))
        aim = np.array([box[0], box[1], max(c.aim_z_m, c.min_aim_z_m)])          # S1: fixed aim height
        check = lambda: self._obstacles(obs, depth, cam_pos, cam_R, camera, p, aim, box, vmap,
                                        getattr(tracker, "box_voxels", None))

        if self.mode == "brake":                                                 # S3: strike given up
            if self.mode_s < c.abort_brake_s or np.linalg.norm(est.v[:2]) > 1.:
                return self._brake(est, az)
            self._enter("target")
        if self.mode == "strike":
            if d_h > c.commit_dist_m:                                            # still allowed to give up
                self.obstacle_pts = check()
                self.obstacle_frames = self.obstacle_frames + 1 if self.obstacle_pts >= c.min_obstacle_pts else 0
                lost = tracker.time_since_seen_s > c.lost_abort_s
                if lost or self.obstacle_frames >= c.obstacle_frames:
                    self.strike_abort = "lost" if lost else "obstacle"
                    self._enter("brake")
                    self._brake_z = p[2] + c.abort_climb_m
                    return self._brake(est, az)
                if d_h > c.strike_dist_m + 5.:
                    self._enter("target")
            if self.mode == "strike":
                return self._strike(est, az, aim, d_h)
        elif d_h <= c.strike_dist_m and tracker.time_since_seen_s <= c.strike_fresh_s:
            self.obstacle_pts = check()
            if self.obstacle_pts < c.min_obstacle_pts:
                self._enter("strike")
                self.n_strikes += 1
                return self._strike(est, az, aim, d_h)

        # pursuit: corridor + MPC toward the point of the line of sight above the box, camera on the box
        los = np.tan(np.radians(c.los_deg))
        unseen = tracker.time_since_seen_s > c.reacq_after_s
        self.reacquire = bool(unseen and delta > np.radians(c.los_deg + c.reacq_excess_deg))
        if self.reacquire:                                   # T3: too high / too close to see it: back onto the LOS
            u = (p[:2] - box[:2]) / max(d_h, 1e-6)
            d_req = (p[2] - box[2]) / los
            d_goal = float(np.clip(d_req, d_h, d_h + c.reacq_max_back_m))
            goal = np.array([*(box[:2] + u * d_goal), min(p[2], box[2] + d_goal * los)])
        else:
            z_los = box[2] + d_h * los
            if seen:                                                   # C3: hold the measured sight angle
                z_los -= c.los_gain * d_h * (np.tan(delta) - los)
            z_goal = float(np.clip(z_los, box[2] + c.min_height_m, max(p[2], box[2] + d_h * los)))
            goal = np.array([box[0], box[1], z_goal])
        vmax = float(min(self.ctl.approach_speed_m_s,
                         np.sqrt(c.strike_speed_m_s ** 2 + 2. * c.brake_m_s2 * max(d_h - c.strike_dist_m, 0.))))
        look = p + c.look_dist_m * np.array([np.cos(az), np.sin(az), 0.])          # C1
        self.floor_on = bool(c.pitch_floor and path_clear and not self.reacquire)   # T1, obstacles first
        pitch_min = self.pitch_floor(delta) if self.floor_on else None             # C4
        return PursuitPlan("target", goal, vmax, look=look, pitch_min=pitch_min)

    def _enter(self, mode):
        self.mode, self.mode_s = mode, 0.
        if mode == "strike":
            self.commit_az, self.obstacle_frames = None, 0

    def _obstacles(self, obs, depth, cam_pos, cam_R, camera, p, aim, box, vmap, box_voxels):
        if self._grid is None:
            vv, uu, rays = camera.depth_grid(self.cfg.depth_stride)
            self._grid = (vv.ravel(), uu.ravel(), rays.reshape(-1, 3))
        bbox_px, d_box = self._bbox_depth(obs, depth, camera)
        return path_obstacles(depth, cam_pos, cam_R, self._grid, p, aim, box, self.cfg, vmap, box_voxels,
                              bbox_px, d_box)

    def _strike(self, est, az, aim, d_h):
        """S1: velocity strike_speed_m_s toward the box: horizontal direction = the box azimuth measured in the
        image (immune to an offset estimate), slope down to the fixed aim height; nose on the box; pitch
        floor. Steering frozen in the last commit_dist_m."""
        c, k = self.cfg, self.ctl
        p, v = est.p, est.v
        if d_h < c.commit_dist_m and self.commit_az is not None:
            az, yaw_stick = self.commit_az, 0.
        else:
            self.commit_az = az
            yaw_stick = -k.yaw_gain * _wrap(az - est.yaw)
        slope = np.arctan2(p[2] - aim[2], max(d_h, 0.5))                         # down to the aim height
        direction = np.array([np.cos(az) * np.cos(slope), np.sin(az) * np.cos(slope), -np.sin(slope)])
        a = c.strike_gain * (c.strike_speed_m_s * direction - v)
        floor = self.pitch_floor(slope) if c.pitch_floor else None
        action = self.controller.accel_to_sticks(a, v, est.yaw, yaw_stick, pitch_min=floor)
        return PursuitPlan("strike", aim, c.strike_speed_m_s, action=action)

    def _brake(self, est, az):
        """S3: strike given up between commit_dist_m and strike_dist_m: brake along the line and climb a little
        (no swerve into what is around the box), nose kept on the box; then the corridor + MPC again."""
        c, k = self.cfg, self.ctl
        p, v = est.p, est.v
        a = np.zeros(3)
        a[:2] = k.vel_gain * (0. - v[:2])
        z_ref = self._brake_z if self._brake_z is not None else p[2]
        a[2] = k.alt_kd * (np.clip(k.alt_kp * (z_ref - p[2]), -1., 2.) - v[2])
        yaw_stick = -k.yaw_gain * _wrap(az - est.yaw)
        action = self.controller.accel_to_sticks(a, v, est.yaw, yaw_stick)
        return PursuitPlan("brake", np.array([p[0], p[1], z_ref]), 0., action=action)
