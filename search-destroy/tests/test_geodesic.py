"""Geodesic distance / path (geodesic.py), its use in the frontier choice, the polyline corridor and the
look-ahead heading of handin.explorer. Synthetic voxel maps only: no simulator."""

import time
from types import SimpleNamespace

import numpy as np

from handin.explorer import FrontierExplorer
from handin.explorer.config import (CorridorConfig, ExplorerConfig, FrontierConfig, GeodesicConfig, MapConfig)
from handin.explorer.corridor import build_corridor
from handin.explorer.frontiers import GainCostFrontierPolicy, NearestFrontierPolicy
from handin.explorer.geodesic import GeodesicField, segment_free
from handin.explorer.voxel_map import VoxelMap
from tests.test_corridor_mpc import assert_boxes_free, free_map, set_occupied

P0 = np.array([0., 0., 10.])


def wall_map():
    """Free map with a full-height wall x in [10, 14], y in [-50, 30]: the way round is the gap at y > 30."""
    m = free_map()
    set_occupied(m, [10., -50., 0.], [14., 30., 40.])
    return m


def field_on(m, p=P0, **kw):
    return GeodesicField(GeodesicConfig(**kw)).compute(m, p)


def set_unknown(m, lo_w, hi_w):
    lo = np.floor((np.array(lo_w) - m.origin) / m.res).astype(int)
    hi = np.ceil((np.array(hi_w) - m.origin) / m.res).astype(int)
    m.observed[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = False
    m.logodds[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 0
    m._update_frontier(np.zeros(3, int), np.array(m.shape))


# ---------------------------------------------------------------------- field
def test_free_space_geodesic_close_to_euclidean():
    f = field_on(free_map())
    q = np.array([[30., 0., 10.], [0., -40., 14.]])
    d = f.distance(q)
    e = np.linalg.norm(q - P0, axis=1)
    assert np.all(d >= e - 1e-6) and np.all(d <= 1.15 * e)        # 26-neighbour grid, no obstacle


def test_wall_makes_the_way_round_longer():
    f = field_on(wall_map())
    q = np.array([30., 0., 10.])
    d = f.distance(q[None])[0]
    assert np.isfinite(d)
    assert d > 2. * np.linalg.norm(q - P0)                         # round the end of the wall at y = 30


def test_enclosed_region_is_unreachable_and_outside_window_too():
    m = free_map()
    set_occupied(m, [24., 24., 2.], [40., 40., 40.])                # solid block ...
    m.logodds[32:34, 32:34, 4:6] = -5                               # ... with a free pocket inside
    f = field_on(m)
    assert not np.isfinite(f.distance(np.array([[35., 35., 9.]]))[0])
    assert not np.isfinite(field_on(free_map(), half_xy_m=20.).distance(np.array([[45., 0., 10.]]))[0])


def test_path_avoids_the_wall_and_is_flyable():
    m = wall_map()
    f = field_on(m)
    q = np.array([30., 0., 10.])
    path = f.path_to(q, m)
    assert path is not None and np.allclose(path[0], P0) and np.allclose(path[-1], q)
    assert path[:, 1].max() > 30.                                   # goes through the gap
    for a, b in zip(path[:-1], path[1:]):
        assert segment_free(m, a, b)
    assert len(path) <= 6                                           # line-of-sight shortcuts


def test_start_next_to_a_wall_uses_a_nearby_cell():
    m = wall_map()
    f = field_on(m, p=np.array([9.5, 0., 10.]))                     # 0.5 m from the wall
    assert f.ok and np.isfinite(f.distance(np.array([[-20., 0., 10.]]))[0])


# ---------------------------------------------------------------------- frontier choice
def two_pockets():
    """Wall map + 2 unknown pockets: A just behind the wall (Euclidean 20 m), B in the open (35 m)."""
    m = wall_map()
    set_unknown(m, [18., -4., 6.], [24., 4., 14.])                  # A
    set_unknown(m, [-41., -4., 6.], [-35., 4., 14.])                # B
    return m


def context(m, geodesic):
    ctx = {"position": P0, "yaw": 0., "velocity": np.zeros(3), "map": m, "target": None,
           "replan_reason": "none", "heading_look_m": 12.}
    if geodesic:
        ctx["geodesic"] = lambda: field_on(m)
    return ctx


def test_nearest_policy_prefers_the_open_side_with_geodesic():
    m = two_pockets()
    fr = m.frontier_points((3., 20.))
    pol = NearestFrontierPolicy(min_dist_m=8., center_weight=0.)
    assert pol.select(fr, context(m, False))[0] > 0.                # Euclidean: behind the wall
    assert pol.select(fr, context(m, True))[0] < 0.                 # geodesic: the open pocket


def test_gain_cost_policy_prefers_the_open_side_with_geodesic():
    m = two_pockets()
    fr = m.frontier_points((3., 20.))
    cfg = FrontierConfig(heading_w=0.)
    assert GainCostFrontierPolicy(cfg).select(fr, context(m, False))[0] > 0.
    pol = GainCostFrontierPolicy(cfg)
    assert pol.select(fr, context(m, True))[0] < 0.
    assert pol.geodesic_used


def test_policy_falls_back_to_euclidean_without_reachable_frontier():
    m = two_pockets()
    fr = m.frontier_points((3., 20.))
    ctx = context(m, False)
    ctx["geodesic"] = lambda: field_on(m, half_xy_m=4.)             # window too small: nothing reachable
    assert NearestFrontierPolicy(8., 0.).select(fr, ctx) is not None


# ---------------------------------------------------------------------- polyline corridor
def test_polyline_corridor_goes_round_where_the_straight_one_is_blocked():
    m = wall_map()
    goal = np.array([30., 0., 10.])
    cfg = CorridorConfig(max_length_m=200., max_boxes=20)
    straight = build_corridor(m, P0, goal, cfg)
    assert straight.blocked and straight.length <= 10. + 1e-6         # stops at the wall
    path = field_on(m).path_to(goal, m)
    cor = build_corridor(m, P0, goal, cfg, path=path)
    assert_boxes_free(m, cor)
    assert not cor.blocked and cor.reached
    refs, box = cor.reference(P0, 8., 20, 0.1)
    for r, b in zip(refs, box):
        assert np.all(r >= cor.lo[b] - 1e-6) and np.all(r <= cor.hi[b] + 1e-6)
    far = cor.point_ahead(P0, 1e3)
    assert np.allclose(far, goal)


def test_polyline_is_cut_at_max_length():
    m = wall_map()
    goal = np.array([30., 0., 10.])
    path = field_on(m).path_to(goal, m)
    cor = build_corridor(m, P0, goal, CorridorConfig(max_length_m=25.), path=path)
    assert cor.length <= 25. + 1e-6 and not cor.reached


# ---------------------------------------------------------------------- explorer: cache, look-ahead
def explorer_on(m):
    ex = FrontierExplorer(ExplorerConfig(map=MapConfig(x_range_m=(-50., 50.), y_range_m=(-50., 50.),
                                                       z_range_m=(0., 40.))))
    ex.map = m
    return ex


def test_field_is_cached_then_recomputed():
    ex = explorer_on(wall_map())
    f1 = ex._geodesic(P0)
    fr = f1.frame
    ex.frame += 2
    assert ex._geodesic(P0 + 1.).frame == fr                        # young and close: reused
    ex.frame += ex.cfg.geodesic.max_age_frames + 1
    assert ex._geodesic(P0).frame == ex.frame                       # too old: recomputed


def test_nose_looks_along_the_bent_corridor():
    m = wall_map()
    ex = explorer_on(m)
    goal = np.array([30., 0., 10.])
    ex.planner.corridor = build_corridor(m, P0, goal, ex.cfg.corridor, path=ex._path(P0, goal))
    est = SimpleNamespace(p=P0, v=np.zeros(3), yaw=0.)
    look = ex._look_point(est, goal, 8.)
    ang = np.degrees(np.arctan2(look[1] - P0[1], look[0] - P0[0]))
    assert ang > 30.                                                # toward the gap (y > 30), not the wall
    assert np.isfinite(ex.path_len) and ex.path_len > 60.


def test_look_point_is_the_goal_without_corridor():
    ex = explorer_on(free_map())
    est = SimpleNamespace(p=P0, v=np.zeros(3), yaw=0.)
    goal = np.array([30., 5., 10.])
    assert np.allclose(ex._look_point(est, goal, 8.), goal)


# ---------------------------------------------------------------------- cost
def test_compute_time_on_a_full_size_city_like_map():
    """Default 600 x 600 x 40 m map, mostly free, ~150 random buildings: one field must stay cheap
    (computed at most every max_age_frames, at each frontier / corridor re-plan)."""
    m = VoxelMap(MapConfig())
    m.observed[:] = True
    m.logodds[:] = -5
    m.logodds[:, :, 0] = 5
    rng = np.random.default_rng(0)
    for _ in range(150):
        x, y = rng.uniform(-100, 100, 2)
        w, d, h = rng.uniform(6, 20, 3)
        set_occupied(m, [x, y, 0.], [x + w, y + d, h])
    f = GeodesicField(GeodesicConfig())
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        f.compute(m, np.array([0.5, 0.5, 10.]))
        times.append(time.perf_counter() - t0)
    assert f.ok
    assert min(times) < 0.25                                        # generous bound (CI); see report for real timings
