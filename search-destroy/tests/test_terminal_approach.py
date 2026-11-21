"""Terminal approach of handin.explorer: adaptive speed cap, obstacle count, distance-dependent corridor margin."""

import numpy as np

from handin.explorer import FrontierExplorer
from handin.explorer.config import CorridorConfig, ExplorerConfig, MapConfig, MPCConfig
from handin.explorer.corridor import CorridorPlanner
from tests.test_corridor_mpc import free_map, set_occupied


def explorer_on(m):
    ex = FrontierExplorer(ExplorerConfig(map=MapConfig(x_range_m=(-50., 50.), y_range_m=(-50., 50.),
                                                       z_range_m=(0., 40.))))
    ex.map = m
    return ex


def test_speed_profile_far_terminal_and_clutter():
    ex = explorer_on(free_map())
    c = ex.cfg.control
    assert ex._approach_speed(200., 0) == c.approach_speed_m_s                  # far: unchanged
    assert np.isclose(ex._approach_speed(c.terminal_radius_m, 0), 5.)           # clear terminal
    assert np.isclose(ex._approach_speed(2., 2), 5.)                            # 2 obstacles: still clear
    assert np.isclose(ex._approach_speed(2., 3), 4.)                            # in between
    assert np.isclose(ex._approach_speed(2., 10), 3.)                           # cluttered: minimum
    v = [ex._approach_speed(d, 0) for d in np.linspace(0., 60., 61)]
    assert np.all(np.diff(v) >= 0)                                              # never speeds up while closing in


def test_obstacle_count_ignores_ground_and_box():
    m = free_map()                                                               # ground layer z < 2 m occupied
    ex = explorer_on(m)
    p, box = np.array([0., 0., 3.]), np.array([3., 0., .78])
    assert ex._count_obstacles_near(p, box) == 0                                 # ground only
    set_occupied(m, [2., -1., 0.], [4., 1., 4.])                                 # the box voxels, next to the estimate
    assert ex._count_obstacles_near(p, box) == 0
    set_occupied(m, [-4., 0., 2.], [-2., 2., 6.])                                # obstacle 1
    set_occupied(m, [0., -4., 2.], [2., -2., 6.])                                # obstacle 2, not touching 1
    assert ex._count_obstacles_near(p, box) == 2
    set_occupied(m, [0., 2., 2.], [2., 4., 6.])                                  # obstacle 3
    assert ex._count_obstacles_near(p, box) == 3
    set_occupied(m, [20., 20., 2.], [22., 22., 6.])                              # far away: not counted
    assert ex._count_obstacles_near(p, box) == 3


def test_margin_shrinks_mpc_boxes():
    m = free_map()
    set_occupied(m, [20., -50., 0.], [22., 50., 40.])
    stops = {}
    for margin in (0.8, 0.2):
        planner = CorridorPlanner(CorridorConfig(), MPCConfig())
        p, v = np.array([0., 0., 10.]), np.array([6., 0., 0.])
        for _ in range(300):
            a = planner.step(m, p, v, np.array([40., 0., 10.]), 8., 8., margin=margin)
            p, v = p + v * 1. / 30. + 0.5 * a / 900., v + a / 30.
        stops[margin] = p[0]
    assert stops[0.8] < 20. - 0.7 and stops[0.2] < 20. - 0.1
    assert stops[0.8] < stops[0.2]                                               # bigger margin = stops farther from the wall
