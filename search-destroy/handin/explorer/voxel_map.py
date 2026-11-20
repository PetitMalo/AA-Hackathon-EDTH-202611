"""Dense 3D voxel grid of the arena, updated by ray casting the depth image.

Each voxel is UNKNOWN (never observed), FREE or OCCUPIED. Occupancy is a small integer
log-odds so that depth noise and pose noise do not flip voxels on a single frame.
A frontier voxel is a FREE voxel with at least one UNKNOWN 6-neighbour; the frontier
grid is kept up to date incrementally, only in the region touched by the last frame.
"""

import numpy as np


class VoxelMap:
    UNKNOWN, FREE, OCCUPIED = 0, 1, 2

    def __init__(self, cfg):
        self.cfg = cfg
        self.res = cfg.voxel_m
        self.origin = np.array([cfg.x_range_m[0], cfg.y_range_m[0], cfg.z_range_m[0]])
        extent = np.array([cfg.x_range_m[1], cfg.y_range_m[1], cfg.z_range_m[1]]) - self.origin
        self.shape = tuple(np.ceil(extent / self.res).astype(int))
        self.reset()

    def reset(self):
        self.logodds = np.zeros(self.shape, dtype=np.int8)
        self.observed = np.zeros(self.shape, dtype=bool)
        self.frontier = np.zeros(self.shape, dtype=bool)
        self._rays = None

    # ------------------------------------------------------------------ indexing
    def world_to_index(self, pts):
        idx = np.floor((np.asarray(pts) - self.origin) / self.res).astype(int)
        valid = np.all((idx >= 0) & (idx < self.shape), axis=-1)
        return idx, valid

    def index_to_world(self, idx):
        return self.origin + (np.asarray(idx) + .5) * self.res

    def state(self, idx=None):
        """UNKNOWN / FREE / OCCUPIED for the whole grid, or for an (N, 3) index array."""
        if idx is None:
            obs, lo = self.observed, self.logodds
        else:
            obs, lo = self.observed[tuple(idx.T)], self.logodds[tuple(idx.T)]
        out = np.full(obs.shape, self.UNKNOWN, dtype=np.int8)
        out[obs & (lo <= 0)] = self.FREE
        out[obs & (lo > 0)] = self.OCCUPIED
        return out

    def is_free(self):
        return self.observed & (self.logodds <= 0)

    def is_occupied(self):
        return self.observed & (self.logodds > 0)

    # ------------------------------------------------------------------ update from one depth image
    def integrate_depth(self, depth_m, cam_pos, cam_R, camera):
        """Ray-cast a depth image from the camera pose. Returns the touched index bounding box (lo, hi) or None."""
        c = self.cfg
        if self._rays is None:
            vv, uu, rays = camera.depth_grid(c.depth_stride)
            self._pix = (vv.ravel(), uu.ravel())
            self._rays = rays.reshape(-1, 3)                       # camera frame, forward component = 1
            self._ray_len = np.linalg.norm(self._rays, axis=1)     # metres travelled per metre of z-depth
            self._n_samples = int(np.ceil(c.max_ray_m * self._ray_len.max() / (self.res / 2)))

        d = np.asarray(depth_m)[self._pix].astype(float)
        hit = d < c.max_ray_m
        zrange = np.minimum(d, c.max_ray_m)
        dirs = self._rays @ cam_R.T                                 # world-frame rays (N, 3)

        # Free space: samples along each ray, stopping half a voxel before a hit.
        frac = np.arange(self._n_samples) / self._n_samples
        s = zrange[:, None] * frac[None, :]                          # (N, K) z-depth of each sample
        keep = (s * self._ray_len[:, None]) < (zrange * self._ray_len - np.where(hit, self.res / 2, 0.))[:, None]
        free_pts = cam_pos + s[keep][:, None] * np.repeat(dirs, keep.sum(1), axis=0)
        hit_pts = cam_pos + zrange[hit][:, None] * dirs[hit]

        free_idx = self._unique_valid(free_pts)
        hit_idx = self._unique_valid(hit_pts)
        self._add_logodds(free_idx, c.miss_logodds)
        self._add_logodds(hit_idx, c.hit_logodds)

        touched = np.concatenate([free_idx, hit_idx])
        if len(touched) == 0:
            return None
        lo, hi = touched.min(0), touched.max(0) + 1
        self._update_frontier(lo, hi)
        return lo, hi

    def _unique_valid(self, pts):
        idx, valid = self.world_to_index(pts)
        idx = idx[valid]
        if len(idx) == 0:
            return idx.reshape(0, 3)
        flat = np.unique(np.ravel_multi_index(idx.T, self.shape))
        return np.stack(np.unravel_index(flat, self.shape), axis=1)

    def _add_logodds(self, idx, delta):
        if len(idx) == 0:
            return
        t = tuple(idx.T)
        lim = self.cfg.logodds_clamp
        self.logodds[t] = np.clip(self.logodds[t].astype(np.int16) + delta, -lim, lim).astype(np.int8)
        self.observed[t] = True

    # ------------------------------------------------------------------ frontiers
    def _update_frontier(self, lo, hi):
        """Recompute frontier flags in [lo, hi) (neighbours read one voxel beyond, out-of-grid = known)."""
        lo = np.maximum(lo - 1, 0)
        hi = np.minimum(hi + 1, self.shape)
        pad_lo, pad_hi = np.maximum(lo - 1, 0), np.minimum(hi + 1, self.shape)
        unknown = np.pad(~self.observed[pad_lo[0]:pad_hi[0], pad_lo[1]:pad_hi[1], pad_lo[2]:pad_hi[2]],
                         [(int(lo[i] == pad_lo[i]), int(hi[i] == pad_hi[i])) for i in range(3)],
                         constant_values=False)
        # `unknown` now covers [lo-1, hi+1) on every axis; its centre is [lo, hi).
        core = (slice(1, -1),) * 3
        any_unknown_nb = np.zeros(unknown[core].shape, dtype=bool)
        for axis in range(3):
            for shift in (-1, 1):
                sl = [slice(1, -1)] * 3
                sl[axis] = slice(1 + shift, unknown.shape[axis] - 1 + shift)
                any_unknown_nb |= unknown[tuple(sl)]
        region = (slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2]))
        free = self.observed[region] & (self.logodds[region] <= 0)
        self.frontier[region] = free & any_unknown_nb

    def frontier_points(self, z_band_m=None, xy_limit_m=None):
        """World centres (N, 3) of the frontier voxels, optionally restricted to an altitude band and a square."""
        grid = self.frontier
        z0 = 0
        if z_band_m is not None:
            z0 = max(0, int((z_band_m[0] - self.origin[2]) / self.res))
            z1 = min(self.shape[2], int(np.ceil((z_band_m[1] - self.origin[2]) / self.res)))
            grid = grid[:, :, z0:z1]
        idx = np.argwhere(grid)
        idx[:, 2] += z0
        pts = self.index_to_world(idx)
        if xy_limit_m is not None and len(pts):
            pts = pts[np.all(np.abs(pts[:, :2]) <= xy_limit_m, axis=1)]
        return pts

    # ------------------------------------------------------------------ stats / debug
    def coverage(self):
        return float(self.observed.mean())

    def top_down(self):
        """2D summary for plotting: 0 unknown, 1 free only, 2 occupied somewhere in the column."""
        img = np.where(self.observed.any(2), 1, 0).astype(np.uint8)
        img[self.is_occupied().any(2)] = 2
        return img
