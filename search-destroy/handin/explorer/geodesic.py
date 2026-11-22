"""Geodesic (shortest flyable path) distance from the drone, on a coarse copy of the voxel map.

    field = GeodesicField(cfg).compute(voxel_map, p)
    field.distance(points)            # (N,) path length in m, inf = unreachable (or outside the window)
    field.first_direction(point, m)   # unit vector of the path's first `m` metres (heading term)
    field.path_to(point, voxel_map)   # (K, 3) simplified polyline p -> point, or None

Grid: a window of +-half_xy_m around the drone, z in z_range_m, in cells of cell_m (an integer number of
voxels). A cell is traversable only if ALL its voxels are known FREE: the same rule as the corridor
(unknown = forbidden), so every path found here can be flown inside a corridor.
Graph: 26-neighbour moves, no corner cutting (a diagonal move needs its axis-aligned intermediate cells
traversable), edge cost = length x (1 + clearance_weight / clearance), clearance = distance to the nearest
OCCUPIED cell in cells, ground layer excluded (keeps the paths in the middle of the streets). Dijkstra (scipy.sparse.csgraph)
from the drone cell.
A frontier voxel has unknown neighbours, so its coarse cell is usually not traversable: a point is reached
through the best traversable cell of its 3 x 3 x 3 neighbourhood (+ the straight bit to the point).
"""

import itertools
import time

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

OFFS13 = [o for o in itertools.product((-1, 0, 1), repeat=3) if o > (0, 0, 0)]   # half of the 26 moves
OFFS27 = np.array(list(itertools.product((-1, 0, 1), repeat=3)))


def segment_free(vmap, a, b, tube_m=0.):
    """The segment a -> b, thickened to a tube of half-width tube_m (box around each sample), only
    crosses known-FREE voxels. Sampled every half voxel. A tube keeps shortcuts from grazing a corner."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / (vmap.res / 2))) + 1)
    pts = a + np.linspace(0., 1., n)[:, None] * (b - a)
    if tube_m > 0:
        pts = (pts[:, None, :] + tube_m * OFFS27[None, :, :]).reshape(-1, 3)
    idx, valid = vmap.world_to_index(pts)
    if not valid.all():
        return False
    t = tuple(idx.T)
    return bool(np.all(vmap.observed[t] & (vmap.logodds[t] <= 0)))


class GeodesicField:
    def __init__(self, cfg):
        self.cfg = cfg
        self.ok = False
        self.frame = None
        self.p = None
        self.ms = 0.

    # ------------------------------------------------------------------ build + Dijkstra
    def compute(self, vmap, p):
        t0 = time.perf_counter()
        self.ok = False
        self.p = np.asarray(p, float).copy()
        try:
            self._compute(vmap, self.p)
        finally:
            self.ms = 1e3 * (time.perf_counter() - t0)
        return self

    def _compute(self, vmap, p):
        c = self.cfg
        k = int(round(c.cell_m / vmap.res))
        if k < 1 or abs(k * vmap.res - c.cell_m) > 1e-6:
            raise ValueError(f"cell_m={c.cell_m} must be a multiple of the voxel size {vmap.res}")
        self.cell = c.cell_m
        shape = np.array(vmap.shape)
        lo_w = np.array([p[0] - c.half_xy_m, p[1] - c.half_xy_m, c.z_range_m[0]])
        hi_w = np.array([p[0] + c.half_xy_m, p[1] + c.half_xy_m, c.z_range_m[1]])
        lo = np.maximum(np.floor((lo_w - vmap.origin) / vmap.res).astype(int) // k * k, 0)   # aligned on cells
        hi = np.minimum(np.ceil((hi_w - vmap.origin) / vmap.res).astype(int), shape)
        n = (hi - lo) // k
        if np.any(n <= 0):
            return
        hi = lo + n * k
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        obs, lod = vmap.observed[sl], vmap.logodds[sl]
        blocks = lambda a: a.reshape(n[0], k, n[1], k, n[2], k)
        trav = blocks(obs & (lod <= 0)).all(axis=(1, 3, 5))
        occ_f = obs & (lod > 0)
        z_w = vmap.origin[2] + (np.arange(lo[2], hi[2]) + .5) * vmap.res
        occ_f[:, :, z_w < c.ground_m] = False                # the ground is not an obstacle to keep off
        occ = blocks(occ_f).any(axis=(1, 3, 5))
        clear = ndimage.distance_transform_edt(~occ) if occ.any() else np.full(tuple(n), 1e3)
        cost = 1. + c.clearance_weight / np.maximum(clear, 1.)

        self.origin_w = vmap.origin + lo * vmap.res
        self.n = n
        self.ids = np.full(tuple(n), -1, dtype=np.int64)
        n_nodes = int(trav.sum())
        if n_nodes == 0:
            return
        self.ids[trav] = np.arange(n_nodes)
        self.node_cells = np.argwhere(trav)                  # same (C) order as the ids

        src = self._source(p, trav)
        if src is None:
            return
        rows, cols, wts = [], [], []
        for o in OFFS13:
            sa = tuple(slice(max(0, -oi), ni - max(0, oi)) for oi, ni in zip(o, n))
            sb = tuple(slice(max(0, oi), ni - max(0, -oi)) for oi, ni in zip(o, n))
            m = trav[sa] & trav[sb]
            nz = [i for i in range(3) if o[i]]
            for r in range(1, len(nz)):                       # no corner cutting
                for sub in itertools.combinations(nz, r):
                    so = [o[i] if i in sub else 0 for i in range(3)]
                    m &= trav[tuple(slice(s.start + d, s.stop + d) for s, d in zip(sa, so))]
            if not m.any():
                continue
            rows.append(self.ids[sa][m])
            cols.append(self.ids[sb][m])
            wts.append(np.linalg.norm(o) * self.cell * .5 * (cost[sa][m] + cost[sb][m]))
        if rows:
            g = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n_nodes, n_nodes))
        else:
            g = csr_matrix((n_nodes, n_nodes))
        dist, pred = dijkstra(g, directed=False, indices=src, return_predecessors=True)
        self.src = src
        self.dist = dist + np.linalg.norm(p - self._centre(self.node_cells[src]))
        self.pred = pred
        self.ok = True

    def _centre(self, cells):
        return self.origin_w + (np.asarray(cells) + .5) * self.cell

    def _cells_of(self, pts):
        ci = np.floor((np.atleast_2d(pts) - self.origin_w) / self.cell).astype(int)
        return ci, np.all((ci >= 0) & (ci < self.n), axis=-1)

    def _source(self, p, trav):
        """Drone cell, or the nearest traversable cell within start_search_cells (drone next to a wall)."""
        ci, inside = self._cells_of(p)
        ci = ci[0]
        if inside[0] and trav[tuple(ci)]:
            return int(self.ids[tuple(ci)])
        r = self.cfg.start_search_cells
        offs = np.array(list(itertools.product(range(-r, r + 1), repeat=3)))
        cand = ci + offs
        ok = np.all((cand >= 0) & (cand < self.n), axis=1)
        cand = cand[ok]
        cand = cand[trav[tuple(cand.T)]] if len(cand) else cand
        if len(cand) == 0:
            return None
        best = cand[np.argmin(np.linalg.norm(self._centre(cand) - p, axis=1))]
        return int(self.ids[tuple(best)])

    # ------------------------------------------------------------------ queries
    def _best_nodes(self, pts):
        """For each point: (geodesic distance, node reached through), via its 3 x 3 x 3 cell neighbourhood."""
        pts = np.atleast_2d(np.asarray(pts, float))
        best = np.full(len(pts), np.inf)
        node = np.full(len(pts), -1, dtype=np.int64)
        if not self.ok or len(pts) == 0:
            return best, node
        ci, _ = self._cells_of(pts)
        for off in OFFS27:
            nb = ci + off
            ok = np.all((nb >= 0) & (nb < self.n), axis=1)
            ids = np.full(len(pts), -1, dtype=np.int64)
            ids[ok] = self.ids[tuple(nb[ok].T)]
            ok &= ids >= 0
            if not ok.any():
                continue
            d = np.full(len(pts), np.inf)
            d[ok] = self.dist[ids[ok]] + np.linalg.norm(self._centre(nb[ok]) - pts[ok], axis=1)
            better = d < best
            best[better], node[better] = d[better], ids[better]
        return best, node

    def distance(self, pts):
        return self._best_nodes(pts)[0]

    def _path_cells(self, node):
        """Cell centres source -> node (inclusive)."""
        chain = []
        while node >= 0:
            chain.append(node)
            if node == self.src:
                break
            node = self.pred[node]
        return self._centre(self.node_cells[np.array(chain[::-1])])

    def first_direction(self, point, look_m):
        """Unit 3D direction from the drone toward the path's point ~look_m along it (straight line if no path)."""
        point = np.asarray(point, float)
        d, node = self._best_nodes(point[None])
        fallback = point - self.p
        if not np.isfinite(d[0]):
            return fallback / max(np.linalg.norm(fallback), 1e-9)
        cells = np.vstack([self._path_cells(node[0]), point[None]])
        far = np.linalg.norm(cells - self.p, axis=1) >= look_m
        aim = cells[np.argmax(far)] if far.any() else point
        v = aim - self.p
        return v / max(np.linalg.norm(v), 1e-9)

    def path_to(self, point, vmap, p=None):
        """Polyline from p (default: where the field was computed) to point through the cells, shortcut
        wherever a tube of half-width los_tube_m around the straight segment is known free. None if unreachable."""
        point = np.asarray(point, float)
        d, node = self._best_nodes(point[None])
        if not np.isfinite(d[0]):
            return None
        start = self.p if p is None else np.asarray(p, float)
        cells = self._path_cells(node[0])
        poly = np.vstack([start[None], cells[1:], point[None]])
        out, i = [poly[0]], 0
        while i < len(poly) - 1:                             # greedy line-of-sight shortcut
            j = len(poly) - 1
            while j > i + 1 and not segment_free(vmap, poly[i], poly[j], self.cfg.los_tube_m):
                j -= 1
            out.append(poly[j])
            i = j
        return np.array(out)
