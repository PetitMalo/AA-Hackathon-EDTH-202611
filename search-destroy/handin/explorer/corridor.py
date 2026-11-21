"""Safe flight corridor (SFC) along the path, and the MPC loop that flies inside it.

Path: for now the straight segment drone -> goal (goal = target estimate or nearest frontier;
an A* path will replace it later). Corridor: a chain of axis-aligned voxel boxes that contain only
FREE voxels (occupied OR unknown = forbidden), built greedily along the segment:

    box 0 = drone voxel, grown face by face while the new layer is free
    exit point of the segment out of box j  ->  seed of box j+1 = the 2 voxels straddling that point
    (overlap with box j guaranteed), grown the same way; stop at the goal, at a blocked voxel,
    or after max_boxes / max_length.

If the segment hits an obstacle, the corridor stops before it and the reference stops there too:
the drone brakes in seen-free space until the frontier logic picks another goal.
"""

import numpy as np

from .mpc import CorridorMPC


# ---------------------------------------------------------------------- occupancy helpers
def forbidden_mask(voxel_map, lo, hi, unknown_is_forbidden=True, clear_center=None, clear_radius=0.):
    """Boolean grid over voxel indices [lo, hi): occupied (and unknown), minus a sphere around clear_center."""
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    obs, lo_odds = voxel_map.observed[sl], voxel_map.logodds[sl]
    mask = ~(obs & (lo_odds <= 0)) if unknown_is_forbidden else (obs & (lo_odds > 0))
    if clear_center is not None and clear_radius > 0:
        axes = [voxel_map.origin[i] + (np.arange(lo[i], hi[i]) + .5) * voxel_map.res for i in range(3)]
        d2 = ((axes[0] - clear_center[0])[:, None, None] ** 2 + (axes[1] - clear_center[1])[None, :, None] ** 2
              + (axes[2] - clear_center[2])[None, None, :] ** 2)
        mask &= d2 > clear_radius ** 2
    return mask


class BoxGrower:
    """O(1) 'how many forbidden voxels in this box' queries on a local crop (summed-volume table)."""

    def __init__(self, forbidden, offset):
        self.offset = np.asarray(offset, int)
        self.size = np.array(forbidden.shape)
        s = np.zeros(tuple(self.size + 1), dtype=np.int32)
        s[1:, 1:, 1:] = forbidden.astype(np.int32).cumsum(0).cumsum(1).cumsum(2)
        self.sat = s

    def inside(self, lo, hi):
        return bool(np.all(lo >= self.offset) and np.all(hi <= self.offset + self.size))

    def count(self, lo, hi):
        """Forbidden voxels in the global index box [lo, hi) (must be inside the crop)."""
        a0, a1, a2 = lo - self.offset
        b0, b1, b2 = hi - self.offset
        s = self.sat
        return int(s[b0, b1, b2] - s[a0, b1, b2] - s[b0, a1, b2] - s[b0, b1, a2]
                   + s[a0, a1, b2] + s[a0, b1, a2] + s[b0, a1, a2] - s[a0, a1, a2])

    def is_free(self, lo, hi):
        return self.inside(lo, hi) and self.count(lo, hi) == 0

    def grow(self, lo, hi, max_size):
        """Grow a free box one voxel layer at a time, cycling over the 6 faces, until stuck or at max_size."""
        lo, hi = lo.copy(), hi.copy()
        grown = True
        while grown:
            grown = False
            for axis in range(3):
                for side in (-1, 1):
                    if hi[axis] - lo[axis] >= max_size:
                        break
                    s_lo, s_hi = lo.copy(), hi.copy()
                    if side < 0:
                        s_lo[axis], s_hi[axis] = lo[axis] - 1, lo[axis]
                    else:
                        s_lo[axis], s_hi[axis] = hi[axis], hi[axis] + 1
                    if self.is_free(s_lo, s_hi):
                        lo, hi = np.minimum(lo, s_lo), np.maximum(hi, s_hi)
                        grown = True
        return lo, hi


# ---------------------------------------------------------------------- corridor
class Corridor:
    """Chain of boxes covering the segment start + t u, t in [0, length]; box j covers t in [t_in[j], t_out[j]]."""

    def __init__(self, start, u, length, idx_lo, idx_hi, t_out, voxel_map, margin, goal, reached, blocked):
        self.start, self.u, self.length = start, u, length
        self.idx_lo, self.idx_hi = np.array(idx_lo), np.array(idx_hi)          # (M, 3) voxel boxes, half-open
        self.lo = voxel_map.origin + self.idx_lo * voxel_map.res                # (M, 3) world boxes
        self.hi = voxel_map.origin + self.idx_hi * voxel_map.res
        mid = (self.lo + self.hi) / 2                                            # MPC boxes, shrunk by the margin
        self.lo_mpc = np.minimum(self.lo + margin, mid)
        self.hi_mpc = np.maximum(self.hi - margin, mid)
        self.t_out = np.array(t_out)
        self.goal, self.reached, self.blocked = goal, reached, blocked

    def __len__(self):
        return len(self.idx_lo)

    def reference(self, p, speed, n, dt):
        """Reference points p_ref[1..n] along the segment from the projection of p, and their box index."""
        s0 = float(np.clip(np.dot(p - self.start, self.u), 0., self.length))
        s = np.minimum(s0 + speed * dt * np.arange(1, n + 1), self.length)
        refs = self.start + s[:, None] * self.u
        box = np.searchsorted(self.t_out, s - 1e-6)                             # first box with t_out >= s
        box = np.maximum.accumulate(np.minimum(box, len(self) - 1))
        return refs, box

    def contains(self, p, tol=0.):
        return bool(np.any(np.all((p >= self.lo - tol) & (p <= self.hi + tol), axis=1)))

    def is_blocked(self, voxel_map, clear_center=None, clear_radius=0.):
        """A voxel of a box became OCCUPIED (unknown voxels can only turn free or occupied later)."""
        for lo, hi in zip(self.idx_lo, self.idx_hi):
            if forbidden_mask(voxel_map, lo, hi, False, clear_center, clear_radius).any():
                return True
        return False


def _segment_exit(start, u, length, lo, hi, t_from):
    """Largest t <= length such that start + t u stays in [lo, hi] (start + t_from u being inside)."""
    t = length
    for ax in range(3):
        if u[ax] > 1e-9:
            t = min(t, (hi[ax] - start[ax]) / u[ax])
        elif u[ax] < -1e-9:
            t = min(t, (lo[ax] - start[ax]) / u[ax])
    return max(t, t_from)


def build_corridor(voxel_map, start, goal, cfg, clear_center=None):
    """Corridor from `start` toward `goal` (truncated to cfg.max_length_m). None if start is off the map."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    d = goal - start
    dist = float(np.linalg.norm(d))
    u = d / dist if dist > 1e-6 else np.array([1., 0., 0.])
    length = min(dist, cfg.max_length_m)
    end = start + length * u
    res = voxel_map.res
    max_vox = max(1, int(round(cfg.max_box_size_m / res)))

    seed, valid = voxel_map.world_to_index(start[None])
    if not valid[0]:
        return None
    seed = seed[0]

    # Local crop: the segment plus the largest box around it.
    shape = np.array(voxel_map.shape)
    c_lo = np.clip(np.floor((np.minimum(start, end) - cfg.max_box_size_m - voxel_map.origin) / res).astype(int), 0, shape)
    c_hi = np.clip(np.ceil((np.maximum(start, end) + cfg.max_box_size_m - voxel_map.origin) / res).astype(int) + 1, 0, shape)
    forb = forbidden_mask(voxel_map, c_lo, c_hi, True, clear_center, cfg.target_clear_radius_m)
    forb[tuple(seed - c_lo)] = False                                     # the drone is there: it is free
    grower = BoxGrower(forb, c_lo)

    to_world = lambda i: voxel_map.origin + i * res
    boxes_lo, boxes_hi, t_out = [], [], []
    lo, hi, t = seed.copy(), seed + 1, 0.
    blocked = False
    eps = 0.01 * res
    while True:
        lo, hi = grower.grow(lo, hi, max_vox)
        t_exit = _segment_exit(start, u, length, to_world(lo), to_world(hi), t)
        boxes_lo.append(lo), boxes_hi.append(hi), t_out.append(t_exit)
        t = t_exit
        if t_exit >= length - 1e-6 or len(boxes_lo) >= cfg.max_boxes:
            break
        idx, ok = voxel_map.world_to_index(np.stack([start + (t_exit - eps) * u, start + (t_exit + eps) * u]))
        s_lo, s_hi = idx.min(0), idx.max(0) + 1                          # the voxels straddling the exit point
        if not ok.all() or not grower.is_free(s_lo, s_hi):
            blocked = True
            break
        lo, hi = s_lo, s_hi

    return Corridor(start, u, t, boxes_lo, boxes_hi, t_out, voxel_map, cfg.margin_m, goal.copy(),
                    reached=(not blocked and t >= dist - 1e-6), blocked=blocked)


# ---------------------------------------------------------------------- corridor + MPC guidance
class CorridorPlanner:
    """Keeps the corridor up to date and returns the MPC acceleration for the current state."""

    def __init__(self, corridor_cfg, mpc_cfg):
        self.cfg = corridor_cfg
        self.mpc_cfg = mpc_cfg
        self.mpc = CorridorMPC(mpc_cfg)
        self.reset()

    def reset(self):
        self.mpc.reset()
        self.corridor = None
        self.age = 0
        self.n_replans = 0
        self.refs = None

    def _needs_replan(self, voxel_map, p, goal, clear_center):
        c, cor = self.cfg, self.corridor
        if cor is None or self.age >= c.replan_every:
            return True
        if np.linalg.norm(goal - cor.goal) > c.goal_moved_m:
            return True
        if not cor.contains(p, c.outside_tol_m):
            return True
        return cor.is_blocked(voxel_map, clear_center, c.target_clear_radius_m)

    def step(self, voxel_map, p, v, goal, speed, max_speed, clear_center=None):
        """Acceleration command (3,), or None when no corridor can be built (drone off the map)."""
        self.age += 1
        if self._needs_replan(voxel_map, p, goal, clear_center):
            self.corridor = build_corridor(voxel_map, p, goal, self.cfg, clear_center)
            self.age = 0
            self.n_replans += 1
        if self.corridor is None:
            return None
        m = self.mpc_cfg
        refs, box = self.corridor.reference(p, speed, m.horizon, m.dt)
        lo, hi = self.corridor.lo_mpc[box].copy(), self.corridor.hi_mpc[box]
        lo[:, 2] = np.maximum(lo[:, 2], self.cfg.z_min_m)
        self.refs = refs
        return self.mpc.solve(p, v, refs, lo, np.maximum(hi, lo), max_speed)
