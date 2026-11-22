import json
from importlib.resources import files

import pybullet as p

TERRAINS_DIR = files('fpv_sim') / 'assets' / 'terrains'
OBJECTS_DIR = files('fpv_sim') / 'assets' / 'objects'

def _parse_version(terrain):
    """Split '<name>@v<N>' into (name, index0), raising ValueError on a malformed or missing version."""
    name, sep, version = terrain.partition('@')
    if not sep or not version.startswith('v') or not version[1:].isdigit():
        raise ValueError("in load_terrain(), expected '<name>@v<N>', got '{:s}'".format(terrain))
    return name, int(version[1:]) - 1

def _spec(name):
    return json.loads((TERRAINS_DIR / (name + '.json')).read_text())

def _poses(spec):
    """The pose list a terrain selects a version from: `setups` for attack maps, else `targets`.

    Each entry is either a pose dict (position_m, rpy_rad, ...) for the classic scenes,
    or a setup dict whose `target` holds that pose, for the attack (`level_*`) maps.
    """
    if 'setups' in spec:
        return spec['setups']
    return spec['targets']

def _target_pose(entry):
    """The target box pose carried by an entry: `entry['target']` for setups, the entry itself otherwise."""
    return entry.get('target', entry) if isinstance(entry, dict) else entry

def load_terrain(terrain: str, client: int):
    """Builds a terrain in the PyBullet world, with the target box at one of its listed positions.

    Parameters
    ----------
    terrain : str
        "<name>@v<N>": the scene `assets/terrains/<name>.json`, with the box at the N-th entry of
        `targets` (the classic scenes) or `setups` (the attack `level_*` maps).
    client : int
        The PyBullet client id.

    Returns
    -------
    list[int]
        The PyBullet body ids: the target box first, then the scene objects in file order.

    """
    name, index = _parse_version(terrain)
    spec = _spec(name)
    poses = _poses(spec)
    if not 0 <= index < len(poses):
        raise ValueError("in load_terrain(), {:s} has {:s} v1-v{:d}, got '{:s}'".format(
            name, 'setups' if 'setups' in spec else 'targets', len(poses), terrain))
    box = {'type': 'urdf', 'asset': 'target_box.urdf', **_target_pose(poses[index])}
    ids = []
    for obj in [box] + spec['objects']:
        if obj['type'] == 'urdf':
            ids.append(p.loadURDF(str(OBJECTS_DIR / obj['asset']),
                                  basePosition=obj['position_m'],
                                  baseOrientation=p.getQuaternionFromEuler(obj['rpy_rad']),
                                  useFixedBase=True, physicsClientId=client))
        elif obj['type'] == 'surface':
            half = [s / 2 for s in obj['size_m']]
            ids.append(p.createMultiBody(baseMass=0,    # static
                                         baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=client),
                                         baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=obj['rgba'], physicsClientId=client),
                                         basePosition=obj['position_m'],
                                         baseOrientation=p.getQuaternionFromEuler(obj['rpy_rad']),
                                         physicsClientId=client))
        else:
            raise ValueError("in load_terrain(), unknown object type '{:s}' in {:s}".format(obj['type'], terrain))
    return ids

def terrain_start_pose(terrain: str):
    """The drone start pose (position_m, rpy_rad) a setup pins the drone to, or None.

    Classic scenes have none (the drone starts at (0, 0, 3)); attack `level_*` setups each carry a
    `start` pose so the target begins in the camera's sight at the given range.
    """
    try:
        name, index = _parse_version(terrain)
    except ValueError:
        return None
    try:
        spec = _spec(name)
    except FileNotFoundError:
        return None
    if 'setups' not in spec or not 0 <= index < len(spec['setups']):
        return None
    return spec['setups'][index].get('start')