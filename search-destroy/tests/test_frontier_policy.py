"""GainCostFrontierPolicy and FrontierManager of handin.explorer on synthetic voxel maps."""

import numpy as np

from handin.explorer.config import FrontierConfig, MapConfig
from handin.explorer.frontiers import (FrontierManager, GainCostFrontierPolicy, fibonacci_sphere, group_by_cell)
from handin.explorer.voxel_map import VoxelMap


def half_known_map():
    """x < 0 observed free, x >= 0 unknown, on a 100 x 100 x 40 m map."""
    m = VoxelMap(MapConfig(x_range_m=(-50., 50.), y_range_m=(-50., 50.), z_range_m=(0., 40.)))
    nx = m.shape[0] // 2
    m.observed[:nx] = True
    m.logodds[:nx] = -5
    m._update_frontier(np.zeros(3, int), np.array(m.shape))
    return m


def ctx(m, p, v=(0., 0., 0.), yaw=0.):
    return {"position": np.array(p, float), "velocity": np.array(v, float), "yaw": yaw, "map": m, "target": None}


def test_helpers():
    d = fibonacci_sphere(128)
    assert np.allclose(np.linalg.norm(d, axis=1), 1.)
    assert np.abs(d.mean(0)).max() < 0.02                       # spread over the sphere
    pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.2, 0.1], [12., 0., 0.], [12.5, 1., 1.]])
    groups = group_by_cell(pts, 10.)
    assert sorted(len(g) for g in groups) == [2, 2]


def test_gain_counts_unknown_and_stops_at_obstacles():
    m = half_known_map()
    pol = GainCostFrontierPolicy(FrontierConfig())
    g = np.array([-1., 0., 10.])                                 # last free voxel before the unknown half
    open_gain = pol.gain(g, m)
    assert open_gain > 0
    assert pol.gain(np.array([-30., 0., 10.]), m) == 0.          # unknown out of ray range
    ix = (m.shape[0] // 2) - 1                                   # wall of occupied voxels right at the frontier
    m.logodds[ix] = 5
    assert pol.gain(g + np.array([-2., 0., 0.]), m) < 0.2 * open_gain


def test_prior_favours_the_centre():
    pol = GainCostFrontierPolicy(FrontierConfig())
    m = half_known_map()
    near = pol.prior(m.world_to_index(np.array([[0., 0., 5.]]))[0], m)
    far = pol.prior(m.world_to_index(np.array([[45., 45., 5.]]))[0], m)
    assert near[0] > far[0] > 0


def test_select_returns_a_frontier_ahead():
    m = half_known_map()
    pol = GainCostFrontierPolicy(FrontierConfig())
    fr = m.frontier_points((3., 20.))
    goal = pol.select(fr, ctx(m, [-20., 0., 10.], v=[5., 0., 0.]))
    assert goal is not None
    idx, _ = m.world_to_index(goal[None])
    assert m.frontier[tuple(idx[0])]
    assert np.linalg.norm(goal - np.array([-20., 0., 10.])) >= FrontierConfig().min_dist_m


def test_manager_blacklists_goals_on_timeout():
    m = half_known_map()
    cfg = FrontierConfig(commit_s=1.)
    man = FrontierManager(cfg, dt=0.1)
    c = ctx(m, [-20., 0., 10.])
    g0 = man.update(m, c).copy()
    for _ in range(11):                                          # drone does not move: commit timeout
        g = man.update(m, c)
    assert len(man.blacklist) == 1
    assert np.linalg.norm(g - g0) >= cfg.blacklist_radius_m
    assert man.last_reason == "timeout"
