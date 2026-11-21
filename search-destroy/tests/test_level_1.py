import json
import math
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

from fpv_sim.utils.terrain import load_terrain, terrain_start_pose

TERRAINS = Path(__file__).parents[1] / 'fpv_sim/assets/terrains'
DISTANCES = (5., 10., 20., 40., 80., 160.)
PER = 10
# Camera constants mirrored from fpv_sim.utils.perception.Camera and the attack generator.
UPTILT_DEG, HFOV_DEG, VFOV_DEG, PITCH_LIMIT_DEG, SAFETY_DEG = 25., 102., 70., 60., 3.


def _spec():
    return json.loads((TERRAINS / 'level_1.json').read_text())


def test_schema_and_empty_map():
    spec = _spec()
    assert spec['name'] == 'level_1'
    assert len(spec['setups']) == 60
    # empty map: only the 500 x 500 ground surface, centered at the origin
    assert len(spec['objects']) == 1
    ground = spec['objects'][0]
    assert ground['type'] == 'surface'
    assert ground['size_m'][:2] == [500, 500]
    assert ground['position_m'][:2] == [0, 0]


@pytest.mark.parametrize('version', list(range(1, 61)))
def test_setup_geometry_keeps_target_in_fov(version):
    s = _spec()['setups'][version - 1]
    t, st = s['target'], s['start']
    assert len(t['position_m']) == 3 and t['position_m'][2] == 0        # target on the ground
    dx, dy, h = st['position_m']
    tx, ty, _ = t['position_m']
    d = math.hypot(dx - tx, dy - ty)
    expected = DISTANCES[(version - 1) // PER]
    assert d == pytest.approx(expected, abs=0.05)                       # the bucket distance
    assert 2. <= h <= 30.                                               # wide height range, bounded
    roll, pitch, yaw = st['rpy_rad']
    assert -math.radians(PITCH_LIMIT_DEG) <= pitch <= math.radians(PITCH_LIMIT_DEG)
    # vertical FOV: |uptilt + depression - pitch| (optical axis vs target) within VFOV/2 - safety
    depression = math.atan2(h, d)
    v_off = math.radians(UPTILT_DEG) + depression - pitch
    assert abs(v_off) <= math.radians(VFOV_DEG / 2 - SAFETY_DEG) + 1e-6
    # horizontal FOV: drone yaw points within HFOV/2 - safety of the target bearing
    bearing = math.atan2(ty - dy, tx - dx)
    h_off = (yaw - bearing + math.pi) % (2 * math.pi) - math.pi
    assert abs(h_off) <= math.radians(HFOV_DEG / 2 - SAFETY_DEG)
    # the target and drone sit inside the 500 m tile
    assert max(abs(tx), abs(ty)) < 250 and max(abs(dx), abs(dy)) < 250
    # each setup pins a drone start pose
    assert terrain_start_pose(f'level_1@v{version}') == st


@pytest.mark.parametrize('version', [1, 30, 60, 61])
def test_load_terrain_version_bounds(version):
    client = p.connect(p.DIRECT)
    try:
        if version == 61:
            with pytest.raises(ValueError):
                load_terrain(f'level_1@v{version}', client)
        else:
            ids = load_terrain(f'level_1@v{version}', client)
            spec = _spec()
            assert len(ids) == 1 + len(spec['objects'])              # box + ground
            pose = p.getBasePositionAndOrientation(ids[0], physicsClientId=client)
            assert pose[0] == pytest.approx(tuple(spec['setups'][version - 1]['target']['position_m']), abs=1e-6)
    finally:
        p.disconnect(client)


@pytest.mark.parametrize('terrain', ['level_1', 'level_1@1', 'village@v0'])
def test_bad_terrain_names_raise(terrain):
    client = p.connect(p.DIRECT)
    try:
        with pytest.raises(ValueError):
            load_terrain(terrain, client)
    finally:
        p.disconnect(client)


def test_classic_scenes_have_no_start_pose():
    assert terrain_start_pose('village@v1') is None