"""Safe flight corridor (SFC) along the path, and the MPC loop that flies inside it.

Path: a polyline drone -> goal. Exploration gives the geodesic path (geodesic.py: around the
obstacles, through known-free space); otherwise (target phase, no path found) the straight segment.
Corridor: a chain of axis-aligned voxel boxes that contain only FREE voxels (occupied OR unknown =
forbidden), built greedily along the path:

    box 0 = drone voxel, grown face by face while the new layer is free
    exit point of the path out of box j  ->  seed of box j+1 = the 2 voxels straddling that point
    (overlap with box j guaranteed), grown the same way; stop at the goal, at a blocked voxel,
    or after max_boxes / max_length (arc length along the path).

If the segment hits an obstacle, the corridor stops before it and the reference stops there too:
the drone brakes in seen-free space until the frontier logic picks another goal.
"""

import numpy as np

from .mpc import CorridorMPC


# ---------------------------------------------------------------------- occupancy helpers
def forbidden_mask(voxel_map, lo, hi, unknown_is_forbidden=True, clear_idx=None):
    """Boolean grid over voxel indices [lo, hi): occupied (and unknown), except the voxels of clear_idx
    ((N, 3) global voxel indices: the target box's own voxels, contact = success). Nothing else is cleared:
    an obstacle next to the box stays an obstacle."""
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    obs, lo_odds = voxel_map.observed[sl], voxel_map.logodds[sl]
    mask = ~(obs & (lo_odds <= 0)) if unknown_is_forbidden else (obs & (lo_odds > 0))
    if clear_idx is not None and len(clear_idx):
        rel = np.asarray(clear_idx, int) - np.asarray(lo, int)
        ok = np.all((rel >= 0) & (rel < np.asarray(mask.shape)), axis=1)
        if ok.any():
            mask[tuple(rel[ok].T)] = False
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


# ---------------------------------------------------------------------- polyline helpers
def make_polyline(points, max_len):
    """Drop zero-length segments, cut at arc length max_len. Returns (poly (K >= 2, 3), cum (K,), full length)."""
    pts = [np.asarray(points[0], float)]
    for q in points[1:]:
        q = np.asarray(q, float)
        if np.linalg.norm(q - pts[-1]) > 1e-6:
            pts.append(q)
    if len(pts) == 1:
        return np.stack([pts[0], pts[0]]), np.zeros(2), 0.
    pts = np.array(pts)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0.], np.cumsum(seg)])
    full = float(cum[-1])
    if full > max_len:
        j = int(np.searchsorted(cum, max_len))               # first vertex at or beyond max_len
        end = pts[j - 1] + (pts[j] - pts[j - 1]) * (max_len - cum[j - 1]) / seg[j - 1]
        pts, cum = np.vstack([pts[:j], end[None]]), np.append(cum[:j], max_len)
    return pts, cum, full


def point_at(poly, cum, s):
    """Points at arc lengths s (clamped to the polyline)."""
    s = np.clip(np.asarray(s, float), 0., cum[-1])
    i = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(poly) - 2)
    seg = np.maximum(cum[i + 1] - cum[i], 1e-9)
    frac = np.clip((s - cum[i]) / seg, 0., 1.)
    return poly[i] + np.asarray(frac)[..., None] * (poly[i + 1] - poly[i])


def project(poly, cum, p):
    """Arc length of the point of the polyline closest to p."""
    a, d = poly[:-1], np.diff(poly, axis=0)
    l2 = np.maximum(np.sum(d * d, axis=1), 1e-12)
    t = np.clip(np.sum((p - a) * d, axis=1) / l2, 0., 1.)
    i = int(np.argmin(np.linalg.norm(a + t[:, None] * d - p, axis=1)))
    return float(cum[i] + t[i] * (cum[i + 1] - cum[i]))


# ---------------------------------------------------------------------- corridor
class Corridor:
    """Chain of boxes covering the path, arc length s in [0, length]; box j covers s in [t_in[j], t_out[j]]."""

    def __init__(self, poly, cum, length, idx_lo, idx_hi, t_out, voxel_map, margin, goal, reached, blocked):
        self.poly, self.cum, self.length = poly, cum, length                    # length: end of the corridor
        self.start = poly[0]
        d = poly[1] - poly[0]
        self.u = d / np.linalg.norm(d) if np.linalg.norm(d) > 1e-9 else np.array([1., 0., 0.])
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
        """Reference points p_ref[1..n] along the path from the projection of p, and their box index."""
        s0 = min(project(self.poly, self.cum, p), self.length)
        s = np.minimum(s0 + speed * dt * np.arange(1, n + 1), self.length)
        refs = point_at(self.poly, self.cum, s)
        box = np.searchsorted(self.t_out, s - 1e-6)                             # first box with t_out >= s
        box = np.maximum.accumulate(np.minimum(box, len(self) - 1))
        return refs, box

    def point_ahead(self, p, dist):
        """Point of the path `dist` metres (arc length) ahead of the projection of p, within the corridor."""
        s0 = min(project(self.poly, self.cum, p), self.length)
        return point_at(self.poly, self.cum, min(s0 + dist, self.length))

    def contains(self, p, tol=0.):
        return bool(np.any(np.all((p >= self.lo - tol) & (p <= self.hi + tol), axis=1)))

    def is_blocked(self, voxel_map, clear_idx=None):
        """A voxel of a box became OCCUPIED (unknown voxels can only turn free or occupied later)."""
        for lo, hi in zip(self.idx_lo, self.idx_hi):
            if forbidden_mask(voxel_map, lo, hi, False, clear_idx).any():
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


def _polyline_exit(poly, cum, lo, hi, t_from):
    """Largest arc length s >= t_from such that the path stays in [lo, hi] from t_from to s
    (the point at t_from being inside): walks the segments while the box contains their end vertex."""
    if cum[-1] <= 1e-9:
        return t_from
    seg = int(np.clip(np.searchsorted(cum, t_from, side="right") - 1, 0, len(poly) - 2))
    t = t_from
    while True:
        a, seg_len = poly[seg], cum[seg + 1] - cum[seg]
        u = (poly[seg + 1] - a) / seg_len
        t_loc = _segment_exit(a, u, seg_len, lo, hi, t - cum[seg])
        if t_loc < seg_len - 1e-6 or seg + 1 >= len(poly) - 1:
            return float(cum[seg] + t_loc)
        seg += 1
        t = cum[seg]


def build_corridor(voxel_map, start, goal, cfg, clear_idx=None, path=None):
    """Corridor from `start` toward `goal` along `path` ((K, 3) polyline start -> goal; None: straight
    segment), truncated to cfg.max_length_m of arc length. None if start is off the map."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    pts = [start, goal] if path is None or len(path) < 2 else [start, *np.asarray(path, float)[1:]]
    poly, cum, full = make_polyline(pts, cfg.max_length_m)
    length = float(cum[-1])
    res = voxel_map.res
    max_vox = max(1, int(round(cfg.max_box_size_m / res)))

    seed, valid = voxel_map.world_to_index(start[None])
    if not valid[0]:
        return None
    seed = seed[0]

    # Local crop: the path plus the largest box around it.
    shape = np.array(voxel_map.shape)
    c_lo = np.clip(np.floor((poly.min(0) - cfg.max_box_size_m - voxel_map.origin) / res).astype(int), 0, shape)
    c_hi = np.clip(np.ceil((poly.max(0) + cfg.max_box_size_m - voxel_map.origin) / res).astype(int) + 1, 0, shape)
    forb = forbidden_mask(voxel_map, c_lo, c_hi, True, clear_idx)
    forb[tuple(seed - c_lo)] = False                                     # the drone is there: it is free
    grower = BoxGrower(forb, c_lo)

    to_world = lambda i: voxel_map.origin + i * res
    boxes_lo, boxes_hi, t_out = [], [], []
    lo, hi, t = seed.copy(), seed + 1, 0.
    blocked = False
    eps = 0.01 * res
    while True:
        lo, hi = grower.grow(lo, hi, max_vox)
        t_exit = _polyline_exit(poly, cum, to_world(lo), to_world(hi), t)
        boxes_lo.append(lo), boxes_hi.append(hi), t_out.append(t_exit)
        t = t_exit
        if t_exit >= length - 1e-6 or len(boxes_lo) >= cfg.max_boxes:
            break
        idx, ok = voxel_map.world_to_index(point_at(poly, cum, [t_exit - eps, t_exit + eps]))
        s_lo, s_hi = idx.min(0), idx.max(0) + 1                          # the voxels straddling the exit point
        if not ok.all() or not grower.is_free(s_lo, s_hi):
            blocked = True
            break
        lo, hi = s_lo, s_hi

    return Corridor(poly, cum, t, boxes_lo, boxes_hi, t_out, voxel_map, cfg.margin_m, goal.copy(),
                    reached=(not blocked and t >= full - 1e-6), blocked=blocked)


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
        self.path = None                   # polyline given by path_fn at the last re-plan (None: straight)
        self.terminal_active = False       # terminal MPC cost actually used at the last step

    def _needs_replan(self, voxel_map, p, goal, clear_idx):
        c, cor = self.cfg, self.corridor
        if cor is None or self.age >= c.replan_every:
            return True
        if np.linalg.norm(goal - cor.goal) > c.goal_moved_m:
            return True
        if not cor.contains(p, c.outside_tol_m):
            return True
        return cor.is_blocked(voxel_map, clear_idx)

    def step(self, voxel_map, p, v, goal, speed, max_speed, clear_idx=None, terminal=False, margin=None,
             path_fn=None):
        """Acceleration command (3,), or None when no corridor can be built (drone off the map).

        terminal=True (target phase): the MPC trades the stop-at-horizon cost for arriving at speed, but ONLY
        while the corridor reaches the goal; a corridor blocked short of the box keeps the normal cost, so
        the drone stops inside it instead of pressing into the obstacle that blocks it.
        margin: how much the corridor boxes are shrunk for the MPC (default cfg.margin_m); applied at
        every step, so a change takes effect without re-planning the corridor. A margin smaller than
        cfg.margin_m (near the box) is also only used while the corridor reaches the goal.
        path_fn(p, goal) -> (K, 3) polyline or None: called on each re-plan only (exploration: geodesic path).
        """
        self.age += 1
        if self._needs_replan(voxel_map, p, goal, clear_idx):
            self.path = path_fn(p, goal) if path_fn is not None else None
            self.corridor = build_corridor(voxel_map, p, goal, self.cfg, clear_idx, self.path)
            self.age = 0
            self.n_replans += 1
        if self.corridor is None:
            self.terminal_active = False
            return None
        reaches = bool(self.corridor.reached)
        self.terminal_active = bool(terminal) and reaches
        self.mpc.set_terminal(self.terminal_active)
        m = self.mpc_cfg
        refs, box = self.corridor.reference(p, speed, m.horizon, m.dt)
        margin = self.cfg.margin_m if margin is None else margin
        if not reaches:
            margin = max(margin, self.cfg.margin_m)
        box_lo, box_hi = self.corridor.lo[box], self.corridor.hi[box]
        mid = (box_lo + box_hi) / 2
        lo, hi = np.minimum(box_lo + margin, mid), np.maximum(box_hi - margin, mid)
        lo[:, 2] = np.maximum(lo[:, 2], self.cfg.z_min_m)
        self.refs = refs
        return self.mpc.solve(p, v, refs, lo, np.maximum(hi, lo), max_speed)
