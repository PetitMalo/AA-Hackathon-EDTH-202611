"""Frontier selection: which frontier voxel to fly to next.

`FrontierPolicy` is the plug-in point for a learned (RL) policy later: it receives the
candidate frontiers and a context, and returns one goal. The naive baseline picks the
nearest frontier (optionally biased toward the arena centre, where the target was reported).
"""

import numpy as np


class FrontierPolicy:
    """Interface. Replace `select` by a learned scorer later (same inputs, same output)."""

    def select(self, frontiers, context):
        """frontiers: (N, 3) world points. context: dict (position, yaw, velocity, voxel map...).
        Returns a (3,) goal or None."""
        raise NotImplementedError


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


class FrontierManager:
    """Keeps a committed frontier goal and decides when to re-plan."""

    def __init__(self, cfg, policy=None, dt=1. / 30.):
        self.cfg = cfg
        self.dt = dt
        self.policy = policy or NearestFrontierPolicy(cfg.min_dist_m, cfg.center_weight)
        self.reset()

    def reset(self):
        self.goal = None
        self.goal_age_s = 0.
        self.frame = 0
        self.n_candidates = 0

    def update(self, voxel_map, context):
        self.frame += 1
        self.goal_age_s += self.dt
        if self._needs_replan(voxel_map, context["position"]):
            cands = voxel_map.frontier_points(self.cfg.z_band_m, self.cfg.xy_limit_m)
            self.n_candidates = len(cands)
            self.goal = self.policy.select(cands, context)
            self.goal_age_s = 0.
        return self.goal

    def _needs_replan(self, voxel_map, position):
        if self.goal is None:
            return True
        if np.linalg.norm(self.goal - position) < self.cfg.reach_dist_m:
            return True
        if self.goal_age_s > self.cfg.commit_s:
            return True
        if self.frame % self.cfg.recompute_every == 0:
            idx, valid = voxel_map.world_to_index(self.goal[None])
            if not valid[0] or not voxel_map.frontier[tuple(idx[0])]:
                return True                              # the goal got explored (or became occupied)
        return False
