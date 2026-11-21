"""Sky ceiling, ground-weighted gain and goal stability of handin.explorer. Synthetic voxel maps only."""

import numpy as np

from handin.explorer.config import FrontierConfig, MapConfig
from handin.explorer.frontiers import FrontierManager, GainCostFrontierPolicy
from handin.explorer.voxel_map import VoxelMap


def known_map():
    """100 x 100 x 40 m map, everything observed free except an occupied ground layer (z < 2 m)."""
    m = VoxelMap(MapConfig(x_range_m=(-50., 50.), y_range_m=(-50., 50.), z_range_m=(0., 40.)))
    m.observed[:] = True
    m.logodds[:] = -5
    m.logodds[:, :, 0] = 5
    return m


def set_unknown(m, lo_w, hi_w):
    lo = np.floor((np.array(lo_w) - m.origin) / m.res).astype(int)
    hi = np.ceil((np.array(hi_w) - m.origin) / m.res).astype(int)
    m.observed[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = False
    m.logodds[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 0


def refresh(m):
    m._update_frontier(np.zeros(3, int), np.array(m.shape))


def ctx(m, p, reason="none"):
    return {"position": np.array(p, float), "yaw": 0., "velocity": np.zeros(3), "map": m, "target": None,
            "replan_reason": reason}


# ---------------------------------------------------------------------- sky ceiling
def test_unknown_sky_makes_no_frontier():
    m = known_map()
    set_unknown(m, [-20., -20., 24.], [20., 20., 40.])          # unknown sky, above the 24 m ceiling
    refresh(m)
    assert m.frontier.sum() == 0
    set_unknown(m, [-4., -4., 10.], [4., 4., 16.])              # unknown pocket below the ceiling
    refresh(m)
    fr = m.frontier_points()
    assert len(fr) > 0 and fr[:, 2].max() < m.cfg.sky_ceiling_m


def test_gain_ignores_the_sky():
    m = known_map()
    set_unknown(m, [-20., -20., 24.], [20., 20., 40.])
    refresh(m)
    pol = GainCostFrontierPolicy(FrontierConfig())
    assert pol.gain(np.array([0., 0., 18.]), m) == 0.           # only sky is unknown
    set_unknown(m, [6., -4., 2.], [12., 4., 8.])                # low pocket within ray range
    assert pol.gain(np.array([0., 0., 18.]), m) > 0.


def test_altitude_weight_profile():
    m = known_map()
    pol = GainCostFrontierPolicy(FrontierConfig(ground_full_m=4.))
    z = np.array([1., 3., 14., 23., 25., 39.])
    idx = m.world_to_index(np.column_stack([np.zeros_like(z), np.zeros_like(z), z]))[0]
    w = pol.altitude_weight(idx, m)
    assert np.allclose(w[:2], 1.) and 0. < w[2] < 1. and 0. <= w[3] < w[2] and np.all(w[4:] == 0.)


def test_ground_pocket_beats_air_pocket():
    m = known_map()
    set_unknown(m, [16., -4., 2.], [24., 4., 8.])               # low pocket (where a box can be), x > 0
    set_unknown(m, [-24., -4., 14.], [-16., 4., 20.])           # same size, in the air, x < 0
    refresh(m)
    pol = GainCostFrontierPolicy(FrontierConfig(heading_w=0., prior_sigma_m=1e4))
    goal = pol.select(m.frontier_points((3., 20.)), ctx(m, [0., 0., 11.]))
    assert goal[0] > 0.


# ---------------------------------------------------------------------- goal stability
def symmetric_pockets():
    m = known_map()
    set_unknown(m, [21., -4., 6.], [29., 4., 12.])
    set_unknown(m, [-29., -4., 6.], [-21., 4., 12.])
    refresh(m)
    return m


def test_explored_goal_keeps_the_same_side():
    m = symmetric_pockets()
    fr = m.frontier_points((3., 20.))
    for side in (1., -1.):
        pol = GainCostFrontierPolicy(FrontierConfig(heading_w=0.))
        pol.prev_goal = np.array([side * 25., 0., 9.])          # the old goal, inside the pocket: no longer a frontier
        goal = pol.select(fr, ctx(m, [0., 0., 9.], reason="explored"))
        assert np.sign(goal[0]) == side


def test_periodic_rescoring_every_periodic_every_frames():
    m = symmetric_pockets()
    cfg = FrontierConfig()
    man = FrontierManager(cfg, dt=1. / 30.)
    calls = []
    select = man.policy.select
    man.policy.select = lambda f, c: (calls.append(c["replan_reason"]), select(f, c))[1]
    for _ in range(2 * cfg.periodic_every):                     # 3 s, static map, drone still
        man.update(m, ctx(m, [0., 0., 9.]))
    assert calls == ["none", "periodic", "periodic"]


def test_uturn_and_switch_stats():
    man = FrontierManager(FrontierConfig())
    p = np.zeros(3)
    man._count_switch(np.array([10., 0., 5.]), np.array([10., 3., 5.]), p)     # small change
    man._count_switch(np.array([10., 0., 5.]), np.array([-10., 1., 5.]), p)    # other side
    assert man.n_switches == 2 and man.n_uturns == 1
    man.time_s = 30.
    assert np.isclose(man.switches_per_min, 2 * 60. / 30.)
