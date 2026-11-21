"""Frontier selection: which frontier voxel to fly to next.

`FrontierPolicy` is the plug-in point for a learned (RL) policy later: it receives the
candidate frontiers and a context, and returns one goal. Two policies:

- NearestFrontierPolicy: nearest frontier (optionally biased toward the arena centre).
- GainCostFrontierPolicy: information gain (weighted by a prior on the target position)
  x exp(-lambda * distance) x heading term, on clustered frontiers, with hysteresis.

`FrontierManager` keeps the committed goal, decides when to re-plan, tells the policy why
(`context["replan_reason"]`) and blacklists goals abandoned on commit timeout (unreachable).
"""

import numpy as np


class FrontierPolicy:
    """Interface. Replace `select` by a learned scorer later (same inputs, same output)."""

    reevaluate_periodically = False      # True: the manager also calls `select` every recompute_every frames

    def select(self, frontiers, context):
        """frontiers: (N, 3) world points. context: dict (position, yaw, velocity, map, target,
        replan_reason). Returns a (3,) goal or None."""
        raise NotImplementedError

    def reset(self):
        pass


class NearestFrontierPolicy(FrontierPolicy):
    def __init__(self, min_dist_m=8., center_weight=0.):
        self.min_dist = min_dist_m
        self.center_weight = center_weight

    def select(self, frontiers, context):
        if len(frontiers) == 0:
            return None
        p = context["position"]
        d = np.linalg.norm(frontiers - p, axis=1)
        ok = d >= self.min_dist
        if not ok.any():
            return None
        cost = d + self.center_weight * np.linalg.norm(frontiers[:, :2], axis=1)
        cost[~ok] = np.inf
        return frontiers[int(np.argmin(cost))]


# ---------------------------------------------------------------------- helpers
def group_by_cell(points, cell):
    """Bin points by floor(points / cell): list of (Ni, 3) arrays, one per occupied cell."""
    if len(points) == 0:
        return []
    keys = np.floor(points / cell).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    splits = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(points[order], splits)


def fibonacci_sphere(n):
    """n unit directions spread uniformly over the sphere, (n, 3)."""
    i = np.arange(n) + .5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z ** 2)
    phi = np.pi * (1 + 5 ** .5) * i
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


def is_still_frontier(vmap, g):
    idx, valid = vmap.world_to_index(np.asarray(g)[None])
    return bool(valid[0] and vmap.frontier[tuple(idx[0])])


def unit(x):
    n = np.linalg.norm(x)
    return x / n if n > 1e-9 else np.zeros_like(x)


class GainCostFrontierPolicy(FrontierPolicy):
    """Score = information gain (weighted by the prior on the target) x exp(-lambda * distance) x heading.
    Same interface as NearestFrontierPolicy."""

    reevaluate_periodically = True       # periodic re-scoring, the hysteresis prevents flip-flopping

    def __init__(self, cfg):
        self.min_dist = cfg.min_dist_m
        self.cluster_m = cfg.cluster_m              # frontier binning cell
        self.min_cluster = cfg.min_cluster          # min frontier points per cluster (no crumbs)
        self.max_candidates = cfg.max_candidates    # clusters actually ray-cast
        self.max_range = cfg.gain_range_m           # ray length (m)
        self.lam = cfg.lam                          # distance weight (1/m)
        self.heading_w = cfg.heading_w              # U-turn penalty (0 = off)
        self.hysteresis = cfg.hysteresis            # switch only if +25 % better
        self.prior_sigma = cfg.prior_sigma_m        # width of the prior around the centre (m)
        self.ray_dirs = fibonacci_sphere(cfg.n_rays)
        self.reset()

    def reset(self):
        self.prev_goal = None
        self.last_scores = None                     # (candidates, utilities) of the last call, for debugging

    # ---------- entry point ----------
    def select(self, frontiers, context):
        if len(frontiers) == 0:
            return None
        p, vmap = np.asarray(context["position"], float), context["map"]

        # 1. Cluster the frontiers (coarse binning, simpler than connected components)
        clusters = group_by_cell(frontiers, self.cluster_m)
        clusters = [c for c in clusters if len(c) >= self.min_cluster] or [frontiers]

        # 2. One candidate per cluster: the frontier point closest to the centroid
        cands, sizes = [], []
        for c in clusters:
            g = c[np.argmin(np.linalg.norm(c - c.mean(0), axis=1))]
            if np.linalg.norm(g - p) >= self.min_dist:
                cands.append(g)
                sizes.append(len(c))
        if not cands:
            return None
        cands, sizes = np.array(cands), np.array(sizes, float)

        # Bound the ray-casting cost: keep the best clusters by size x distance discount
        if len(cands) > self.max_candidates:
            pre = sizes * np.exp(-self.lam * np.linalg.norm(cands - p, axis=1))
            keep = np.argsort(-pre)[:self.max_candidates]
            cands = cands[keep]

        # 3. Score every candidate
        scores = np.array([self.utility(g, p, context, vmap) for g in cands])
        best = int(np.argmax(scores))
        goal, u_best = cands[best], scores[best]
        self.last_scores = (cands, scores)

        # 4. Hysteresis: keep the previous goal if still valid, still far enough, not abandoned, and almost as good
        prev = self.prev_goal
        if (prev is not None and context.get("replan_reason") not in ("timeout", "reached")
                and np.linalg.norm(prev - p) >= self.min_dist and is_still_frontier(vmap, prev)
                and not context.get("is_blacklisted", lambda g: False)(prev)):
            u_prev = self.utility(prev, p, context, vmap)
            if u_best < (1 + self.hysteresis) * u_prev:
                goal = prev

        self.prev_goal = goal.copy()
        return goal

    # ---------- utility ----------
    def utility(self, g, p, context, vmap):
        gain = self.gain(g, vmap)                           # what would be seen from g
        d = np.linalg.norm(g - p)                           # v1: Euclidean (v2: Dijkstra)
        u = gain * np.exp(-self.lam * d)
        # U-turn penalty: favours goals along the motion / the heading
        v = np.asarray(context["velocity"], float)
        fwd = unit(v) if np.linalg.norm(v) > 0.3 else np.array([np.cos(context["yaw"]), np.sin(context["yaw"]), 0.])
        cos_a = float(np.dot(unit(g - p), fwd))             # 1 = ahead, -1 = behind
        return u * (1 - self.heading_w * (1 - cos_a) / 2)

    # ---------- information gain by ray casting ----------
    def gain(self, g, vmap):
        """Sum of the prior over the UNKNOWN voxels visible from g (occlusions included).

        All rays at once: samples every half voxel, a ray stops at its first occupied voxel
        or when it leaves the grid; each unknown voxel counts once.
        """
        res = vmap.res
        steps = np.arange(1, int(self.max_range / (res / 2)) + 1) * (res / 2)
        pts = g + steps[None, :, None] * self.ray_dirs[:, None, :]           # (rays, samples, 3)
        idx = np.floor((pts - vmap.origin) / res).astype(np.int64)
        shape = np.array(vmap.shape)
        valid = np.all((idx >= 0) & (idx < shape), axis=-1)
        idx = np.clip(idx, 0, shape - 1)
        t = (idx[..., 0], idx[..., 1], idx[..., 2])
        observed = vmap.observed[t]
        occupied = observed & (vmap.logodds[t] > 0)
        alive = np.cumsum(~valid | occupied, axis=1) == 0                     # before the first stop
        seen = alive & ~observed
        if not seen.any():
            return 0.
        flat = np.unique(np.ravel_multi_index((t[0][seen], t[1][seen], t[2][seen]), vmap.shape))
        return float(self.prior(np.stack(np.unravel_index(flat, vmap.shape), axis=1), vmap).sum())

    def prior(self, idx, vmap):
        """Prior on the target position: Gaussian around the arena centre. Summed over UNKNOWN voxels only,
        so areas already seen without the target stop counting: that is the belief update."""
        x = vmap.index_to_world(idx)
        return np.exp(-np.sum(x[:, :2] ** 2, axis=1) / (2 * self.prior_sigma ** 2))


def make_policy(cfg):
    if cfg.policy == "gain_cost":
        return GainCostFrontierPolicy(cfg)
    if cfg.policy == "nearest":
        return NearestFrontierPolicy(cfg.min_dist_m, cfg.center_weight)
    raise ValueError(f"unknown frontier policy {cfg.policy!r}")


class FrontierManager:
    """Keeps a committed frontier goal and decides when to re-plan."""

    def __init__(self, cfg, policy=None, dt=1. / 30.):
        self.cfg = cfg
        self.dt = dt
        self.policy = policy or make_policy(cfg)
        self.reset()

    def reset(self):
        self.goal = None
        self.goal_age_s = 0.
        self.frame = 0
        self.n_candidates = 0
        self.time_s = 0.
        self.blacklist = []                              # (point, expiry time)
        self.last_reason = None
        self.policy.reset()

    def is_blacklisted(self, g):
        r = self.cfg.blacklist_radius_m
        return any(np.linalg.norm(g - b) < r for b, _ in self.blacklist)

    def update(self, voxel_map, context):
        self.frame += 1
        self.goal_age_s += self.dt
        self.time_s += self.dt
        self.blacklist = [(b, t) for b, t in self.blacklist if t > self.time_s]
        reason = self._replan_reason(voxel_map, context["position"])
        if reason is not None:
            if reason == "timeout":                      # committed for commit_s without reaching it: unreachable
                self.blacklist.append((self.goal.copy(), self.time_s + self.cfg.blacklist_s))
            cands = voxel_map.frontier_points(self.cfg.z_band_m, self.cfg.xy_limit_m)
            if self.blacklist and len(cands):
                bl = np.array([b for b, _ in self.blacklist])
                d = np.linalg.norm(cands[:, None, :] - bl[None, :, :], axis=2).min(1)
                cands = cands[d >= self.cfg.blacklist_radius_m]
            self.n_candidates = len(cands)
            ctx = dict(context, replan_reason=reason, is_blacklisted=self.is_blacklisted)
            new_goal = self.policy.select(cands, ctx)
            if new_goal is None or self.goal is None or np.linalg.norm(new_goal - self.goal) > 1e-6:
                self.goal_age_s = 0.                     # commit timer restarts only on a new goal
            self.goal = new_goal
            self.last_reason = reason
        return self.goal

    def _replan_reason(self, voxel_map, position):
        if self.goal is None:
            return "none"
        if np.linalg.norm(self.goal - position) < self.cfg.reach_dist_m:
            return "reached"
        if self.goal_age_s > self.cfg.commit_s:
            return "timeout"
        if self.frame % self.cfg.recompute_every == 0:
            idx, valid = voxel_map.world_to_index(self.goal[None])
            if not valid[0] or not voxel_map.frontier[tuple(idx[0])]:
                return "explored"                        # the goal got explored (or became occupied)
            if self.policy.reevaluate_periodically:
                return "periodic"
        return None
