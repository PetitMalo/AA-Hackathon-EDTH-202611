"""World-frame target estimate from the detector bounding box.

Range comes from the depth map inside the box when it is within depth range (< 40 m),
otherwise from intersecting the box-centre ray with the horizontal plane at the box
centre height (the target always rests on the ground).
"""

import numpy as np


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

    @property
    def active(self):
        """A target estimate exists and is fresh enough to be the next goal."""
        return self.estimate is not None and self.time_since_seen_s < self.cfg.lost_timeout_s

    def update(self, obs, depth_m, cam_pos, cam_R, camera):
        self.time_since_seen_s += self.dt
        if not obs["detected"]:
            return self.estimate
        point = self.localise(obs["box"], depth_m, cam_pos, cam_R, camera)
        if point is None:
            return self.estimate
        self.n_detections += 1
        self.time_since_seen_s = 0.
        a = self.cfg.ema
        self.estimate = point if self.estimate is None else (1 - a) * self.estimate + a * point
        return self.estimate

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
