"""Pinhole model of the onboard cameras (depth 224x126 and detector 160x90 share the same pose and FOV).

Camera axes, expressed in the body frame (FLU): forward / left / up, the body x axis tilted up by `uptilt`.
PyBullet's linearised depth is the distance along the optical axis (z-depth), not along the ray.
"""

import numpy as np


class CameraModel:
    def __init__(self, cfg):
        self.cfg = cfg
        t = np.radians(cfg.uptilt_deg)
        # Columns: camera forward, left, up, in the body frame.
        self.tilt = np.array([[np.cos(t), 0., -np.sin(t)],
                              [0., 1., 0.],
                              [np.sin(t), 0., np.cos(t)]])
        self.mount = np.array(cfg.mount_body_m)
        self.tan_half_h = np.tan(np.radians(cfg.hfov_deg) / 2)

    def pose(self, R_body, p_body):
        """Camera position and camera-to-world rotation (columns forward, left, up)."""
        return p_body + R_body @ self.mount, R_body @ self.tilt

    def rays_cam(self, u, v, width, height):
        """Unit-forward ray directions in the camera frame for pixel coordinates (u right, v down).

        Returned rays have forward component 1, so `depth * ray` is the 3D point for a z-depth.
        """
        f = (width / 2) / self.tan_half_h                # square pixels: fx = fy
        u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)
        left = -(u - width / 2) / f
        up = -(v - height / 2) / f
        return np.stack([np.ones_like(left), left, up], axis=-1)

    def depth_grid(self, stride):
        """Subsampled pixel centres of the depth image: (rows, cols) index arrays and their camera rays."""
        c = self.cfg
        rows = np.arange(stride // 2, c.depth_height, stride)
        cols = np.arange(stride // 2, c.depth_width, stride)
        vv, uu = np.meshgrid(rows, cols, indexing="ij")
        rays = self.rays_cam(uu + .5, vv + .5, c.depth_width, c.depth_height)
        return vv, uu, rays

    def bbox_to_pixels(self, box_norm, width, height):
        """Normalised (x1, y1, x2, y2) -> pixel box in an image of the given size."""
        x1, y1, x2, y2 = np.asarray(box_norm, dtype=float)
        return x1 * width, y1 * height, x2 * width, y2 * height
