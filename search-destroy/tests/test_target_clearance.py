"""Only the box's own voxels are freed near the target (obstacles stay obstacles), and F4: terminal MPC only
when the corridor reaches the box. Synthetic maps only."""

import numpy as np

from handin.explorer.config import CorridorConfig, MPCConfig
from handin.explorer.corridor import CorridorPlanner, forbidden_mask
from tests.test_corridor_mpc import free_map, set_occupied

BOX = np.array([20., 0., .78])


def box_idx(m):
    """Voxels of a 4 x 2 m box footprint at BOX, in the ground layer (what the tracker would collect)."""
    pts = np.array([[x, y, 1.] for x in (18.5, 20.5) for y in (-.5,)])
    return m.world_to_index(pts)[0]


def test_only_the_box_voxels_are_freed():
    m = free_map()
    set_occupied(m, [18., -2., 0.], [22., 2., 2.])                # the box, in the ground layer
    set_occupied(m, [18., -2., 4.], [22., 2., 6.])                # a canopy right above it
    set_occupied(m, [22., -2., 0.], [24., 2., 2.])                # a car touching it
    lo, hi = np.zeros(3, int), np.array(m.shape)
    mask = forbidden_mask(m, lo, hi, False, box_idx(m))
    at = lambda x, y, z: mask[tuple(m.world_to_index(np.array([[x, y, z]]))[0][0])]
    assert not at(18.5, -.5, 1.) and not at(20.5, -.5, 1.)        # the box: freed (contact = success)
    assert at(20., 0., 5.)                                        # canopy above it: obstacle
    assert at(23., 0., 1.)                                        # car next to it: obstacle


def test_blocked_corridor_keeps_the_normal_mpc_cost_and_margin():
    m = free_map()
    set_occupied(m, [10., -50., 0.], [12., 50., 40.])             # wall between the drone and the box
    planner = CorridorPlanner(CorridorConfig(), MPCConfig())
    a = planner.step(m, np.array([0., 0., 4.]), np.zeros(3), BOX, 8., 8., clear_idx=box_idx(m), terminal=True,
                     margin=0.2)
    assert a is not None and planner.corridor.blocked
    assert not planner.terminal_active


def test_open_corridor_to_the_box_uses_the_terminal_cost():
    m = free_map()
    planner = CorridorPlanner(CorridorConfig(), MPCConfig())
    goal = np.array([20.5, -.5, 1.])                              # inside a box voxel, approached from right above
    planner.step(m, np.array([20.5, -.5, 4.]), np.zeros(3), goal, 8., 8., clear_idx=box_idx(m), terminal=True)
    assert planner.corridor.reached and planner.terminal_active


def test_drone_stops_before_a_blocking_wall_in_target_mode():
    """The F4 failure: terminal cost + soft boxes made the drone press into the wall ending the corridor."""
    m = free_map()
    set_occupied(m, [10., -50., 0.], [12., 50., 40.])
    planner = CorridorPlanner(CorridorConfig(), MPCConfig())
    p, v, dt = np.array([0., 0., 4.]), np.array([6., 0., 0.]), 1. / 30.
    for _ in range(300):
        a = planner.step(m, p, v, BOX, 8., 8., clear_idx=box_idx(m), terminal=True, margin=0.2)
        p, v = p + v * dt + 0.5 * a * dt ** 2, v + a * dt
    assert p[0] < 10. - 0.5                                       # stopped short of the wall, with margin
    assert np.linalg.norm(v[:2]) < 0.5
