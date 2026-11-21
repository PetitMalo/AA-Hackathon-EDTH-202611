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
UPTILT_DEG, HFOV_DEG, VFOV_DEG, PITCH_LIMIT_DEG, SAFETY_DEG = 25., 102., 70., 60., 3.
BOX_Z = 1.55 / 2                # BOX_SIZE[2]


def _spec():
    return json.loads((TERRAINS / 'level_2.json').read_text())


def _forest_spec():
    return json.loads((TERRAINS / 'forest.json').read_text())


def test_schema_reuses_forest_objects():
    spec = _spec()
    assert spec['name'] == 'level_2'
    assert len(spec['setups']) == 60
    # objects are the forest scene's, copied verbatim
    assert spec['objects'] == _forest_spec()['objects']
    assert len(spec['objects']) == 377


@pytest.mark.parametrize('version', list(range(1, 61)))
def test_setup_geometry_keeps_target_in_fov(version):
    """Same FOV invariant as level_1: distance bucket, height, pitch/yaw windows, inside the 500 m tile."""
    s = _spec()['setups'][version - 1]
    t, st = s['target'], s['start']
    assert len(t['position_m']) == 3 and t['position_m'][2] == 0
    dx, dy, h = st['position_m']
    tx, ty, _ = t['position_m']
    d = math.hypot(dx - tx, dy - ty)
    expected = DISTANCES[(version - 1) // PER]
    assert d == pytest.approx(expected, abs=0.05)
    assert 2. <= h <= 30.
    roll, pitch, yaw = st['rpy_rad']
    assert -math.radians(PITCH_LIMIT_DEG) <= pitch <= math.radians(PITCH_LIMIT_DEG)
    depression = math.atan2(h, d)
    v_off = math.radians(UPTILT_DEG) + depression - pitch
    assert abs(v_off) <= math.radians(VFOV_DEG / 2 - SAFETY_DEG) + 1e-6
    bearing = math.atan2(ty - dy, tx - dx)
    h_off = (yaw - bearing + math.pi) % (2 * math.pi) - math.pi
    assert abs(h_off) <= math.radians(HFOV_DEG / 2 - SAFETY_DEG)
    assert max(abs(tx), abs(ty)) < 250 and max(abs(dx), abs(dy)) < 250
    assert terrain_start_pose(f'level_2@v{version}') == st


def _world_with_box(version):
    """Loads level_2@<version> into a DIRECT client; returns (client, box_id, solid_ids, objects)."""
    client = p.connect(p.DIRECT)
    ids = load_terrain(f'level_2@v{version}', client)
    box_id = ids[0]
    spec = _spec()
    objects = spec['objects']
    solid_ids = [b for o, b in zip(objects, ids[1:])
                 if o['type'] == 'urdf' and o['asset'] != 'bush.urdf']
    return client, box_id, solid_ids


@pytest.mark.parametrize('version', [1, 15, 30, 45, 60])
def test_target_drone_clear_and_clean_los(version):
    """The level_2 sampling invariants, verified at load time:
       - the box touches no solid prop (bushes allowed);
       - the drone (a 0.6 m sphere) touches no solid prop at its start pose;
       - rays from the camera eye to box centre and both long-axis ends hit the box first.
    """
    s = _spec()['setups'][version - 1]
    t, st = s['target'], s['start']
    client, box_id, solid_ids = _world_with_box(version)
    try:
        p.performCollisionDetection(physicsClientId=client)
        # box clearance
        assert not any(p.getClosestPoints(box_id, b, 0, physicsClientId=client) for b in solid_ids)
        # drone clearance: 0.6 m probe sphere
        probe = p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.6,
                                                                                  physicsClientId=client),
                                  baseVisualShapeIndex=-1, physicsClientId=client)
        p.resetBasePositionAndOrientation(probe, tuple(st['position_m']),
                                          p.getQuaternionFromEuler((0, 0, 0)), physicsClientId=client)
        p.performCollisionDetection(physicsClientId=client)
        assert not any(p.getClosestPoints(probe, b, 0, physicsClientId=client) for b in solid_ids)
        # clean LOS: single ray from camera eye to the box centre (inside the box volume).
        roll, pitch, yaw = st['rpy_rad']
        quat = p.getQuaternionFromEuler((roll, pitch, yaw))
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        eye = np.array(st['position_m']) + rot @ np.array([0.30, 0., 0.])
        tx, ty, _ = t['position_m']
        hit = p.rayTest(eye.tolist(), [tx, ty, BOX_Z], physicsClientId=client)[0]
        assert hit[0] == box_id, f'v{version}: ray to box hit body {hit[0]} not the box'
    finally:
        p.disconnect(client)


@pytest.mark.parametrize('version', [1, 30, 60, 61])
def test_load_terrain_version_bounds(version):
    client = p.connect(p.DIRECT)
    try:
        if version == 61:
            with pytest.raises(ValueError):
                load_terrain(f'level_2@v{version}', client)
        else:
            ids = load_terrain(f'level_2@v{version}', client)
            spec = _spec()
            assert len(ids) == 1 + len(spec['objects'])
            pose = p.getBasePositionAndOrientation(ids[0], physicsClientId=client)
            assert pose[0] == pytest.approx(tuple(spec['setups'][version - 1]['target']['position_m']), abs=1e-6)
    finally:
        p.disconnect(client)


@pytest.mark.parametrize('terrain', ['level_2', 'level_2@1', 'forest@v0'])
def test_bad_terrain_names_raise(terrain):
    client = p.connect(p.DIRECT)
    try:
        with pytest.raises(ValueError):
            load_terrain(terrain, client)
    finally:
        p.disconnect(client)


def test_classic_scenes_still_have_no_start_pose():
    assert terrain_start_pose('forest@v1') is None
    assert terrain_start_pose('village@v1') is None