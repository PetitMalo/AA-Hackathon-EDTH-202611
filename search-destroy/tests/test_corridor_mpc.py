"""Safe flight corridor + MPC of handin.explorer on synthetic voxel maps."""

import numpy as np

from handin.explorer.config import CorridorConfig, MapConfig, MPCConfig
from handin.explorer.corridor import CorridorPlanner, build_corridor, forbidden_mask
from handin.explorer.mpc import CorridorMPC
from handin.explorer.voxel_map import VoxelMap


def free_map():
    """100 x 100 x 40 m map, everything observed free except a ground layer."""
    m = VoxelMap(MapConfig(x_range_m=(-50., 50.), y_range_m=(-50., 50.), z_range_m=(0., 40.)))
    m.observed[:] = True
    m.logodds[:] = -5
    m.logodds[:, :, 0] = 5
    return m


def set_occupied(m, lo_w, hi_w):
    lo = np.floor((np.array(lo_w) - m.origin) / m.res).astype(int)
    hi = np.ceil((np.array(hi_w) - m.origin) / m.res).astype(int)
    m.logodds[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 5


def assert_boxes_free(m, cor):
    for lo, hi in zip(cor.idx_lo, cor.idx_hi):
        assert not forbidden_mask(m, lo, hi).any()


def test_free_space_reaches_goal():
    m = free_map()
    cor = build_corridor(m, [0., 0., 10.], [30., 5., 12.], CorridorConfig())
    assert cor.reached and not cor.blocked
    assert_boxes_free(m, cor)
    assert cor.contains(np.array([30., 5., 12.]))


def test_wall_blocks_and_corridor_stops_before_it():
    m = free_map()
    set_occupied(m, [20., -50., 0.], [22., 50., 40.])            # full wall across the path
    cor = build_corridor(m, [0., 0., 10.], [40., 0., 10.], CorridorConfig())
    assert cor.blocked and not cor.reached
    assert_boxes_free(m, cor)
    assert cor.length <= 20. + 1e-6
    refs, box = cor.reference(np.array([0., 0., 10.]), 8., 20, 0.1)
    assert np.all(refs[:, 0] <= 20. + 1e-6)


def test_unknown_is_forbidden_and_chain_overlaps():
    m = free_map()
    m.observed[:, :, 10:] = False                                 # unknown above 20 m
    set_occupied(m, [10., 4., 0.], [14., 50., 40.])               # pillar next to the path
    cor = build_corridor(m, [0., 0., 10.], [40., 0., 10.], CorridorConfig(max_box_size_m=10.))
    assert_boxes_free(m, cor)
    assert np.all(cor.hi[:, 2] <= 20.)
    for j in range(len(cor) - 1):                                 # consecutive boxes overlap
        assert np.all(np.maximum(cor.lo[j], cor.lo[j + 1]) <= np.minimum(cor.hi[j], cor.hi[j + 1]))


def test_target_clearing_opens_the_goal():
    m = free_map()
    # final glide onto a box on the ground (the explorer's glide slope keeps the path above the ground
    # layer until ~3 m from the target, i.e. inside the cleared sphere)
    goal = np.array([5., 0., 1.])
    cfg = CorridorConfig()
    assert not build_corridor(m, [0., 0., 3.], goal, cfg).reached       # goal inside the occupied ground layer
    assert build_corridor(m, [0., 0., 3.], goal, cfg, clear_center=goal).reached


def test_mpc_stays_in_box_and_stops_at_the_end():
    m = free_map()
    set_occupied(m, [20., -50., 0.], [22., 50., 40.])
    planner = CorridorPlanner(CorridorConfig(margin_m=0.35), MPCConfig())
    p, v = np.array([0., 0., 10.]), np.array([6., 0., 0.])
    dt = 1. / 30.
    for _ in range(300):                                          # 10 s, perfect double-integrator plant
        a = planner.step(m, p, v, np.array([40., 0., 10.]), 8., 8.)
        p, v = p + v * dt + 0.5 * a * dt ** 2, v + a * dt
    assert p[0] < 20. - 0.3                                       # stopped before the wall, with the margin
    assert np.linalg.norm(v) < 0.5


def test_admm_matches_unconstrained_optimum():
    c = MPCConfig(admm_eps_abs=1e-7, admm_eps_rel=1e-7, admm_max_iter=5000)
    mpc = CorridorMPC(c)
    n = c.horizon
    p0, v0 = np.zeros(3), np.array([1., -1., 0.])
    ref = np.outer(np.arange(1, n + 1) * 0.3, [1., 0.5, 0.])
    big = np.full((n, 3), 1e3)
    a0 = mpc.solve(p0, v0, ref, -big, big, 8.)
    # inactive constraints: closed-form solution of the quadratic cost
    p_free = p0 + mpc.kdt[:, None] * v0
    q = 2 * (c.w_pos * mpc.Sp.T @ (p_free - ref) + c.w_term_vel * mpc.Sv[-1][:, None] * v0[None])
    a_star = np.linalg.solve(mpc.P[:n, :n], -q)
    assert np.allclose(a0, a_star[0], atol=1e-3)
