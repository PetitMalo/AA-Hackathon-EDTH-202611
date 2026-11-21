"""Exploration only: is the drone making progress? If not, a graded escape.

ProgressMonitor (stuck detection), from the estimated state only:
    "stuck"    net horizontal displacement over the last window_s below min_disp_m
    "blocked"  corridor blocked and horizontal speed below blocked_speed_m_s for blocked_s
    "clutter"  many obstacles around (>= clutter_count within clutter_radius_m) and a displacement
               over the window below clutter_min_disp_m

EscapeManager: each trigger starts the next level, cycling, and starts again at level 1 when the
previous trigger is older than escalation_reset_s:
    1 "replan"  abandon the goal: wide blacklist around it and around the drone, re-plan now
    2 "scan"    hold position and turn the nose a full turn: the camera fills the unknown around,
                which is what lets the corridor (unknown = forbidden) grow again
    3 "climb"   hold x, y and climb above the rooftops, then widen the frontier altitude band
The explorer executes the levels (FrontierExplorer._escape_step); this module only keeps the state.
"""

from collections import deque

import numpy as np


class ProgressMonitor:
    def __init__(self, cfg, dt=1. / 30.):
        self.cfg, self.dt = cfg, dt
        self.hist = deque(maxlen=int(round(cfg.window_s / dt)) + 1)
        self.reset()

    def reset(self):
        self.clear()

    def clear(self, cooldown_s=0.):
        """Forget the history (after an escape step: the window must refill before a new 'stuck')."""
        self.hist.clear()
        self.slow_blocked_s = 0.
        self.cooldown_s = cooldown_s

    @property
    def displacement(self):
        """Net horizontal displacement over the full window (m), None while the window is not full."""
        if len(self.hist) < self.hist.maxlen:
            return None
        return float(np.linalg.norm(self.hist[-1] - self.hist[0]))

    def update(self, p, v, corridor_blocked, n_clutter=0):
        """One frame. Returns the trigger reason ('stuck' | 'blocked' | 'clutter') or None."""
        c = self.cfg
        self.hist.append(np.asarray(p, float)[:2].copy())
        if self.cooldown_s > 0:
            self.cooldown_s -= self.dt
            self.slow_blocked_s = 0.
            return None
        slow = np.linalg.norm(np.asarray(v, float)[:2]) < c.blocked_speed_m_s
        self.slow_blocked_s = self.slow_blocked_s + self.dt if (corridor_blocked and slow) else 0.
        if self.slow_blocked_s >= c.blocked_s:
            return "blocked"
        disp = self.displacement
        if disp is None:
            return None
        if disp < c.min_disp_m:
            return "stuck"
        if n_clutter >= c.clutter_count and disp < c.clutter_min_disp_m:
            return "clutter"
        return None


class EscapeManager:
    LEVELS = ("replan", "scan", "climb")

    def __init__(self, cfg, dt=1. / 30.):
        self.cfg, self.dt = cfg, dt
        self.monitor = ProgressMonitor(cfg, dt)
        self.reset()

    def reset(self):
        self.monitor.reset()
        self.time_s = 0.
        self.mode = None                 # None | "scan" | "climb"
        self.mode_s = 0.
        self.start_p = None
        self.climb_stall_s = 0.
        self.level = -1                  # index in LEVELS of the last level started
        self.last_trigger_s = -np.inf
        self.n_escapes = 0
        self.reason = None               # last trigger reason

    @property
    def level_name(self):
        return self.LEVELS[self.level] if self.level >= 0 else ""

    def suspend(self):
        """Target phase: no stuck detection, any running escape is dropped."""
        self.time_s += self.dt
        self.mode = None
        self.monitor.clear()

    def update(self, p, v, corridor_blocked, n_clutter=0):
        """One exploration frame. Returns the level to start now ('replan' | 'scan' | 'climb'), or None.
        While an escape mode runs, only its clock advances."""
        self.time_s += self.dt
        if self.mode is not None:
            self.mode_s += self.dt
            return None
        reason = self.monitor.update(p, v, corridor_blocked, n_clutter)
        if reason is None:
            return None
        if self.time_s - self.last_trigger_s > self.cfg.escalation_reset_s:
            self.level = -1
        self.level = (self.level + 1) % len(self.LEVELS)
        self.last_trigger_s = self.time_s
        self.n_escapes += 1
        self.reason = reason
        return self.LEVELS[self.level]

    def start(self, mode, p):
        self.mode, self.mode_s, self.climb_stall_s = mode, 0., 0.
        self.start_p = np.asarray(p, float).copy()

    def finish(self):
        self.mode = None
        self.monitor.clear(self.cfg.cooldown_s)

    def scan_done(self):
        return self.mode_s >= self.cfg.scan_s

    def climb_done(self, p, v):
        """Reached the climb altitude, timed out, or no longer climbing (something above)."""
        c = self.cfg
        if p[2] >= c.climb_alt_m - c.climb_tol_m or self.mode_s >= c.climb_timeout_s:
            return True
        if self.mode_s > 1. and v[2] < c.climb_stall_speed_m_s:
            self.climb_stall_s += self.dt
        else:
            self.climb_stall_s = 0.
        return self.climb_stall_s >= c.climb_stall_s


def column_clear(vmap, p, z_top, half_width_m):
    """No OCCUPIED voxel in the column [p.z + 1 m, z_top] x (p.xy +- half_width_m): safe to climb straight up.
    Unknown voxels are accepted (the camera, tilted up, rarely sees straight above)."""
    p = np.asarray(p, float)
    lo_w = np.array([p[0] - half_width_m, p[1] - half_width_m, p[2] + 1.])
    hi_w = np.array([p[0] + half_width_m, p[1] + half_width_m, z_top])
    lo = np.clip(np.floor((lo_w - vmap.origin) / vmap.res).astype(int), 0, vmap.shape)
    hi = np.clip(np.floor((hi_w - vmap.origin) / vmap.res).astype(int) + 1, 0, vmap.shape)
    if np.any(hi <= lo):
        return True
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    return not bool((vmap.observed[sl] & (vmap.logodds[sl] > 0)).any())
