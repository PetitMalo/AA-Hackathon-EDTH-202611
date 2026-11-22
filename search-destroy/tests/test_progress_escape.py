"""Stuck detection + graded escape of handin.explorer (progress.py), and the joint tilt limit of the controller.

Synthetic states and voxel maps only: no simulator.
"""

from types import SimpleNamespace

import numpy as np

from handin.explorer import FrontierExplorer
from handin.explorer.config import (CameraConfig, ControlConfig, EscapeConfig, ExplorerConfig, FrontierConfig,
                                    MapConfig)
from handin.explorer.controller import WaypointController
from handin.explorer.frontiers import FrontierManager
from handin.explorer.progress import EscapeManager, ProgressMonitor, column_clear
from tests.test_corridor_mpc import free_map, set_occupied
from tests.test_frontier_policy import ctx, half_known_map

DT = 1. / 30.


def run(monitor, n, p=(0., 0., 10.), v=(0., 0., 0.), blocked=False, n_clutter=0, step=(0., 0., 0.)):
    """n frames; returns the first trigger reason (or None) and the last position."""
    p = np.array(p, float)
    for _ in range(n):
        r = monitor.update(p, v, blocked, n_clutter)
        if r is not None:
            return r, p
        p = p + np.array(step)
    return None, p


# ---------------------------------------------------------------------- ProgressMonitor
def test_stationary_is_stuck_after_the_window_only():
    cfg = EscapeConfig()
    m = ProgressMonitor(cfg, DT)
    n = int(cfg.window_s / DT)
    assert run(m, n - 1)[0] is None                                   # window not full yet
    assert run(m, 5)[0] == "stuck"


def test_moving_drone_is_not_stuck():
    m = ProgressMonitor(EscapeConfig(), DT)
    assert run(m, 600, step=(3. * DT, 0., 0.))[0] is None             # 3 m/s -> 24 m per window


def test_blocked_and_slow_triggers_fast():
    cfg = EscapeConfig()
    m = ProgressMonitor(cfg, DT)
    r, _ = run(m, int(cfg.blocked_s / DT) + 2, blocked=True)
    assert r == "blocked"
    m.clear()
    assert run(m, int(cfg.blocked_s / DT) + 2, v=(3., 0., 0.), blocked=True)[0] is None   # blocked but moving


def test_clutter_needs_more_progress():
    cfg = EscapeConfig()
    step = (10. / (cfg.window_s / DT), 0., 0.)                        # ~10 m per window: ok in the open...
    n = int(cfg.window_s / DT) + 5
    assert run(ProgressMonitor(cfg, DT), n, step=step, n_clutter=0)[0] is None
    assert run(ProgressMonitor(cfg, DT), n, step=step, n_clutter=cfg.clutter_count)[0] == "clutter"   # ...not here


def test_cooldown_blocks_triggers():
    cfg = EscapeConfig()
    m = ProgressMonitor(cfg, DT)
    m.clear(cooldown_s=3.)
    assert run(m, int(2.5 / DT), blocked=True)[0] is None
    assert run(m, int(2.5 / DT), blocked=True)[0] == "blocked"         # 0.5 s of cooldown left + blocked_s


# ---------------------------------------------------------------------- EscapeManager
def trigger(e):
    """Feed a blocked, motionless drone until the next level starts."""
    for _ in range(2000):
        level = e.update(np.zeros(3), np.zeros(3), True)
        if level is not None:
            return level
    raise AssertionError("no trigger")


def test_levels_escalate_then_cycle():
    e = EscapeManager(EscapeConfig(), DT)
    assert trigger(e) == "replan"
    e.monitor.clear(e.cfg.cooldown_s)
    assert trigger(e) == "scan"
    e.start("scan", np.zeros(3))
    while not e.scan_done():
        assert e.update(np.zeros(3), np.zeros(3), True) is None        # no trigger while a mode runs
    e.finish()
    assert trigger(e) == "climb"
    assert e.n_escapes == 3 and e.reason == "blocked"


def test_escalation_resets_after_a_quiet_period():
    cfg = EscapeConfig()
    e = EscapeManager(cfg, DT)
    assert trigger(e) == "replan"
    e.monitor.clear()
    for _ in range(int((cfg.escalation_reset_s + 1) / DT)):           # progressing well for a long time
        e.update(np.zeros(3), np.array([5., 0., 0.]), False)
        e.monitor.hist.clear()
    assert trigger(e) == "replan"


def test_climb_done_on_altitude_and_on_stall():
    cfg = EscapeConfig()
    e = EscapeManager(cfg, DT)
    e.start("climb", np.array([0., 0., 8.]))
    assert not e.climb_done(np.array([0., 0., 15.]), np.array([0., 0., 4.]))
    assert e.climb_done(np.array([0., 0., cfg.climb_alt_m]), np.zeros(3))
    e.start("climb", np.array([0., 0., 8.]))
    done = False
    for _ in range(int((1. + cfg.climb_stall_s) / DT) + 5):           # under a roof: not climbing
        e.update(np.zeros(3), np.zeros(3), False)
        done = done or e.climb_done(np.array([0., 0., 8.]), np.zeros(3))
    assert done


def test_column_clear():
    m = free_map()
    p = np.array([0., 0., 8.])
    assert column_clear(m, p, 26., 2.)
    set_occupied(m, [-1., -1., 18.], [1., 1., 20.])                   # a roof above
    assert not column_clear(m, p, 26., 2.)
    assert column_clear(m, np.array([10., 10., 8.]), 26., 2.)         # not above this one


# ---------------------------------------------------------------------- FrontierManager hooks
def test_force_replan_blacklists_with_its_own_radius():
    m = half_known_map()
    man = FrontierManager(FrontierConfig(policy="nearest", center_weight=0.))
    p = np.array([-10., 0., 10.])
    g0 = man.update(m, ctx(m, p))
    man.force_replan([g0], radius_m=25.)
    g1 = man.update(m, ctx(m, p))
    assert man.last_reason == "escape"
    assert np.linalg.norm(g1 - g0) >= 25.


def test_widened_z_band_expires():
    cfg = FrontierConfig()
    man = FrontierManager(cfg)
    man.widen_z_band((3., 30.), duration_s=1.)
    assert man.z_band == (3., 30.)
    man.time_s += 1.5
    assert man.z_band == cfg.z_band_m


# ---------------------------------------------------------------------- explorer integration
def explorer_on(m):
    ex = FrontierExplorer(ExplorerConfig(map=MapConfig(x_range_m=(-50., 50.), y_range_m=(-50., 50.),
                                                       z_range_m=(0., 40.))))
    ex.map = m
    return ex


def test_escape_step_runs_scan_then_climb():
    ex = explorer_on(free_map())
    ce = ex.cfg.escape
    est = SimpleNamespace(p=np.array([0., 0., 8.]), v=np.zeros(3), yaw=0.)
    seen = []
    for _ in range(int(60 / DT)):                                     # motionless for a minute
        ex.frame += 1
        out = ex._escape_step(est)
        if out is not None:
            mode, goal, action = out
            if not seen or seen[-1] != mode:
                seen.append(mode)
            if mode == "scan":
                assert abs(action[2]) > 0.5                             # turning the nose
            if mode == "climb":
                assert action[3] > 0. and goal[2] == ce.climb_alt_m     # more thrust, straight up
                break
    assert seen[:2] == ["scan", "climb"]
    assert ex.escape.n_escapes >= 3                                   # replan, scan, climb


def test_climb_skipped_under_a_roof():
    m = free_map()
    set_occupied(m, [-3., -3., 18.], [3., 3., 20.])
    ex = explorer_on(m)
    ex.escape.level = 1                                               # next level would be the climb
    ex.escape.last_trigger_s = 0.
    est = SimpleNamespace(p=np.array([0., 0., 8.]), v=np.zeros(3), yaw=0.)
    for _ in range(int(12 / DT)):
        ex.frame += 1
        out = ex._escape_step(est)
        assert out is None or out[0] != "climb"
        if ex.escape.n_escapes:
            break
    assert ex.escape.n_escapes == 1 and ex.escape.mode is None        # turned into a re-plan


def test_count_obstacles_without_target():
    m = free_map()
    ex = explorer_on(m)
    set_occupied(m, [4., 0., 2.], [6., 2., 8.])
    set_occupied(m, [-6., 0., 2.], [-4., 2., 8.])
    assert ex._count_obstacles_near(np.array([0., 0., 4.]), None, 10.) == 2
    assert ex._count_obstacles_near(np.array([0., 0., 4.]), None, 2.) == 0


# ---------------------------------------------------------------------- B1: joint tilt limit
def test_joint_tilt_limit_keeps_direction():
    c = ControlConfig()
    ctl = WaypointController(c, CameraConfig())
    a = np.array([30., 30., 0.])                                      # far beyond the tilt limit, diagonal
    roll_s, pitch_s, _, _ = ctl.accel_to_sticks(a, np.zeros(3), 0., 0.)
    full = np.radians(c.max_stick_deg)
    roll, pitch = roll_s * full, pitch_s * full
    tilt = np.degrees(np.arccos(np.cos(roll) * np.cos(pitch)))
    assert tilt <= c.max_tilt_deg + 1e-6
    # direction kept: thrust direction in the yaw frame (sin p cos r, -sin r) ~ (1, 1)
    fwd, left = np.sin(pitch) * np.cos(roll), -np.sin(roll)
    assert np.isclose(np.arctan2(left, fwd), np.pi / 4, atol=1e-3)


def test_small_accelerations_unchanged():
    ctl = WaypointController(ControlConfig(), CameraConfig())
    a = np.array([2., -1., 0.5])
    roll_s, pitch_s, _, _ = ctl.accel_to_sticks(a, np.zeros(3), 0., 0.)
    full = np.radians(60.)
    a_up = 9.8 + 0.5
    assert np.isclose(pitch_s * full, np.arctan2(2., a_up))
    assert np.isclose(roll_s * full, -np.arcsin(-1. / np.sqrt(4. + 1. + a_up ** 2)))
