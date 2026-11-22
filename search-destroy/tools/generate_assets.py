"""Generates the low-polygon object URDFs, the terrain JSONs and their previews.

Run from the repository root: `uv run python tools/generate_assets.py`.
Everything is seeded, so a rerun reproduces the committed files.
"""
import json
import math
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / 'fpv_sim' / 'assets'
OBJECTS, TERRAINS = ASSETS / 'objects', ASSETS / 'terrains'
BOX_SIZE = (4., 1.65, 1.55)
TARGETS_PER_SCENE = 40
BORDER_M = 150.               # props farther out cost render time and are rarely seen
CLEAR_FROM_BOX = 1.95       # parts starting above this height may overhang the target box (box top 1.55 m + margin)

################################################################################
# Geometry: every part is a list of planar faces, oriented outwards and flat shaded.

def _rot(rpy):
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx

def _normal(f):
    f = np.asarray(f, float)
    return np.cross(f, np.roll(f, -1, axis=0)).sum(axis=0)    # Newell's method

def tf(faces, xyz=(0, 0, 0), rpy=(0, 0, 0)):
    r = _rot(rpy)
    return [[r @ np.asarray(v, float) + xyz for v in f] for f in faces]

def orient(faces, center):
    """Flips faces of a convex part so their normals point away from its center."""
    return [f if np.dot(_normal(f), np.mean(f, axis=0) - center) >= 0 else f[::-1]
            for f in ([np.asarray(v, float) for v in f] for f in faces)]

def prism(poly, z0, z1, bottom=True):
    top, bot = [(x, y, z1) for x, y in poly], [(x, y, z0) for x, y in poly]
    n = len(poly)
    faces = [top] + ([bot] if bottom else []) + [[bot[i], bot[(i + 1) % n], top[(i + 1) % n], top[i]] for i in range(n)]
    return orient(faces, np.r_[np.mean(poly, axis=0), (z0 + z1) / 2])

def box(size, xyz=(0, 0, 0), rpy=(0, 0, 0), bottom=True):
    sx, sy, sz = (s / 2 for s in size)
    return tf(prism([(-sx, -sy), (sx, -sy), (sx, sy), (-sx, sy)], -sz, sz, bottom), xyz, rpy)

def frustum(n, r0, r1, z0, z1, xyz=(0, 0, 0), phase=0., bottom=False):
    ring = lambda r, z: [(r * math.cos(phase + 2 * math.pi * i / n), r * math.sin(phase + 2 * math.pi * i / n), z) for i in range(n)]
    lo = ring(r0, z0)
    faces = [lo] if bottom else []
    if r1 == 0:
        faces += [[lo[i], lo[(i + 1) % n], (0, 0, z1)] for i in range(n)]
    else:
        hi = ring(r1, z1)
        faces += [hi] + [[lo[i], lo[(i + 1) % n], hi[(i + 1) % n], hi[i]] for i in range(n)]
    return tf(orient(faces, np.array([0, 0, (z0 + z1) / 2])), xyz)

def extrude(profile, w0, w1, axis):
    """Prism of a 2D profile extruded along `axis`: 'x' maps (u, v) to (y, z), 'y' maps it to (x, z)."""
    pt = (lambda u, v, w: (w, u, v)) if axis == 'x' else (lambda u, v, w: (u, w, v))
    a, b = [pt(u, v, w0) for u, v in profile], [pt(u, v, w1) for u, v in profile]
    n = len(profile)
    faces = [a, b] + [[a[i], a[(i + 1) % n], b[(i + 1) % n], b[i]] for i in range(n)]
    return orient(faces, (np.mean(a, axis=0) + np.mean(b, axis=0)) / 2)

_T = (1 + 5 ** .5) / 2
_ICO_V = np.array([(-1, _T, 0), (1, _T, 0), (-1, -_T, 0), (1, -_T, 0), (0, -1, _T), (0, 1, _T),
                   (0, -1, -_T), (0, 1, -_T), (_T, 0, -1), (_T, 0, 1), (-_T, 0, -1), (-_T, 0, 1)], float)
_ICO_V /= np.linalg.norm(_ICO_V[0])
_ICO_F = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
          (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]

def blob(center, radii, rng, jitter=.12):
    """A 20-triangle icosahedron, scaled and jittered into a foliage clump or a boulder."""
    v = _ICO_V * (1 + rng.uniform(-jitter, jitter, (12, 1))) * radii + center
    return orient([[v[i] for i in f] for f in _ICO_F], np.asarray(center, float))

def panel(center, w, h, yaw):
    """A one-sided vertical rectangle facing `yaw` (0 faces +x): windows, doors, lights."""
    d, t = np.array([math.cos(yaw), math.sin(yaw), 0]), np.array([-math.sin(yaw), math.cos(yaw), 0])
    c, z = np.asarray(center, float), np.array([0, 0, 1])
    f = [c - t * w / 2 - z * h / 2, c + t * w / 2 - z * h / 2, c + t * w / 2 + z * h / 2, c - t * w / 2 + z * h / 2]
    return orient([f], c - d)

################################################################################
# Assets

class Asset:
    def __init__(self, name, seed=0):
        self.name, self.rng, self.vis, self.col = name, np.random.default_rng(seed), {}, []

    def add(self, rgb, faces, collide=False):
        self.vis.setdefault(tuple(rgb), []).extend(f for f in faces if max(v[2] for v in f) > .01)
        if collide:
            self.col.append(faces)

    def tris(self):
        return sum(len(f) - 2 for faces in self.vis.values() for f in faces)

def write_obj(path, faces):
    lines, vi = [], 1
    for n, f in enumerate(faces, 1):
        nrm = _normal(f)
        nrm /= np.linalg.norm(nrm)
        lines += ['v {:.4f} {:.4f} {:.4f}'.format(*v) for v in f] + ['vn {:.4f} {:.4f} {:.4f}'.format(*nrm)]
        lines += ['f {0}//{3} {1}//{3} {2}//{3}'.format(vi, vi + k, vi + k + 1, n) for k in range(1, len(f) - 1)]
        vi += len(f)
    path.write_text('\n'.join(lines) + '\n')

def write_asset(a):
    meshes = OBJECTS / 'meshes'
    meshes.mkdir(exist_ok=True)
    body = []
    for i, (rgb, faces) in enumerate(a.vis.items()):
        write_obj(meshes / f'{a.name}_v{i}.obj', faces)
        body.append(f'    <visual name="part_{i}">\n      <geometry><mesh filename="meshes/{a.name}_v{i}.obj" /></geometry>\n'
                    f'      <material name="{a.name}_{i}"><color rgba="{rgb[0]:g} {rgb[1]:g} {rgb[2]:g} 1" /></material>\n    </visual>')
    for i, faces in enumerate(a.col):
        write_obj(meshes / f'{a.name}_c{i}.obj', faces)
        body.append(f'    <collision name="hull_{i}">\n      <geometry><mesh filename="meshes/{a.name}_c{i}.obj" /></geometry>\n    </collision>')
    (OBJECTS / f'{a.name}.urdf').write_text(
        f'<?xml version="1.0" encoding="utf-8"?>\n<!-- Generated by tools/generate_assets.py -->\n<robot name="{a.name}">\n  <link name="base_link">\n'
        '    <inertial>\n      <mass value="0" />\n      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0" />\n    </inertial>\n'
        + '\n'.join(body) + '\n  </link>\n</robot>\n')

BARK, BARK_DARK = (.40, .29, .19), (.33, .24, .16)
LEAF = [(.18, .38, .17), (.26, .47, .2), (.36, .56, .24)]
GLASS, FRAME = (.2, .3, .38), (.9, .89, .85)

def tree(name, h_trunk, clumps, seed):
    a = Asset(name, seed)
    a.add(BARK, frustum(5, .24, .14, 0, h_trunk), collide=True)
    for i, (c, r) in enumerate(clumps):
        a.add(LEAF[i % 3], blob(c, r, a.rng), collide=True)
    return a

def pine(seed):
    a = Asset('pine', seed)
    a.add(BARK_DARK, frustum(4, .2, .1, 0, 1.2))
    tiers = ((1., 4.2, 2.), (2.8, 6.4, 1.55), (4.7, 8.8, 1.05))
    for i, (z0, z1, r) in enumerate(tiers):
        a.add(((.12, .29, .17), (.16, .35, .2))[i % 2], frustum(7, r, 0, z0, z1, phase=i * .45, bottom=True), collide=True)
    a.col.insert(0, frustum(5, .2, .1, 0, 3.))
    return a

def bush(seed):
    a = Asset('bush', seed)
    for i, (c, r) in enumerate((((-.35, 0, .5), (1.05, .85, .65)), ((.45, .15, .7), (.85, .8, .7)))):
        a.add(LEAF[(i + 1) % 3], blob(c, r, a.rng, .18), collide=True)
    return a

def rock(seed):
    a = Asset('rock', seed)
    a.add((.53, .52, .49), blob((0, 0, .3), (1.05, .8, .65), a.rng, .28), collide=True)
    a.add((.43, .43, .41), blob((.95, .45, .15), (.5, .45, .38), a.rng, .28), collide=True)
    return a

def hay_bale():
    a = Asset('hay_bale')
    ring = [(.75 * math.cos(2 * math.pi * i / 8), .75 + .75 * math.sin(2 * math.pi * i / 8)) for i in range(8)]
    faces = extrude(ring, -.6, .6, 'x')
    a.add((.79, .67, .36), faces[2:], collide=True)
    a.add((.68, .55, .27), faces[:2])
    return a

def shed():
    a = Asset('shed')
    a.add((.5, .36, .23), box((3., 2.4, 2.3), (0, 0, 1.15), bottom=False), collide=True)
    a.add((.29, .3, .32), box((3.4, 2.9, .1), (0, 0, 2.42), (-.2, 0, 0)), collide=True)
    a.add((.33, .23, .15), panel((.5, -1.22, 1.), 1., 1.95, -math.pi / 2) + panel((-.85, 1.22, 1.45), .7, .6, math.pi / 2))
    a.add(FRAME, panel((-1.52, 0, 1.4), .65, .55, math.pi))
    return a

def fence():
    a = Asset('fence')
    for x in (-2, 0, 2):
        a.add(BARK, box((.12, .12, 1.2), (x, 0, .6), bottom=False), collide=True)
    for z in (.42, .9):
        a.add((.56, .42, .28), box((4.1, .05, .11), (0, -.085, z)), collide=True)
    return a

def electric_pole():
    a = Asset('electric_pole')
    a.add((.36, .27, .18), frustum(6, .15, .1, 0, 8.6), collide=True)
    a.add((.36, .27, .18), box((2.4, .12, .12), (0, 0, 7.9)), collide=True)
    for x, z in ((-1.05, 7.96), (1.05, 7.96), (0, 8.6)):
        a.add((.78, .8, .82), frustum(4, .06, .05, z, z + .22, (x, 0, 0)))
    a.add((.55, .58, .6), box((.5, .5, .8), (0, .33, 6.4)), collide=True)
    return a

def house(name, L, W, H, pitch, wall, roof, floors, door_x, extra=None):
    a = Asset(name)
    rise = W / 2 * math.tan(pitch)
    a.add((.45, .43, .4), box((L + .1, W + .1, .45), (0, 0, .225), bottom=False))
    a.add(wall, extrude([(-W / 2, 0), (W / 2, 0), (W / 2, H), (0, H + rise), (-W / 2, H)], -L / 2, L / 2, 'x'), collide=True)
    over, t = .45, .14
    s = (W / 2 + over) / math.cos(pitch)
    for side in (-1, 1):
        cy, cz = side * (W / 2 + over) / 2, H + rise - (W / 2 + over) / 2 * math.tan(pitch) + t / 2 / math.cos(pitch)
        a.add(roof, box((L + .6, s, t), (0, cy, cz), (-side * pitch, 0, 0)), collide=True)
    a.add(tuple(c * .8 for c in roof), box((L + .62, .3, .12), (0, 0, H + rise + t / math.cos(pitch))))
    a.add((.55, .3, .24), box((.6, .6, rise + 1.), (L * .25, W * .2, H + (rise + 1.) / 2)), collide=True)
    trim, glass = [], []
    a.add((.3, .22, .15), panel((door_x, -W / 2 - .03, 1.05), .95, 2.05, -math.pi / 2))
    trim += panel((door_x, -W / 2 - .015, 1.1), 1.15, 2.2, -math.pi / 2)
    for k in range(floors):
        z = 1.55 + 2.6 * k
        for x in np.linspace(-L / 2 + 1.4, L / 2 - 1.4, max(2, round(L / 3.2))):
            for y, yaw in ((-W / 2, -math.pi / 2), (W / 2, math.pi / 2)):
                if yaw < 0 and k == 0 and abs(x - door_x) < 1.3:
                    continue
                trim += panel((x, y + math.copysign(.015, y), z), 1., 1.2, yaw)
                glass += panel((x, y + math.copysign(.03, y), z), .78, .98, yaw)
        for y in (-W / 4, W / 4) if W > 5.5 else (0,):
            for x, yaw in ((-L / 2, math.pi), (L / 2, 0)):
                trim += panel((x + math.copysign(.015, x), y, z), 1., 1.2, yaw)
                glass += panel((x + math.copysign(.03, x), y, z), .78, .98, yaw)
    a.add(FRAME, trim)
    a.add(GLASS, glass)
    if extra:
        extra(a, L, W)
    return a

def porch(a, L, W):
    a.add((.42, .32, .24), box((2.6, 1.6, .1), (-1.2, -W / 2 - .8, 2.55), (.18, 0, 0)), collide=True)
    for x in (-2.35, -.05):
        a.add(FRAME, box((.12, .12, 2.45), (x, -W / 2 - 1.45, 1.23), bottom=False), collide=True)

def garage(a, L, W):
    a.add((.72, .66, .56), box((3.4, W - 1., 2.7), (L / 2 + 1.7, .5, 1.35), bottom=False), collide=True)
    a.add((.32, .33, .35), box((3.6, W - .8, .15), (L / 2 + 1.7, .5, 2.75)), collide=True)
    a.add((.8, .8, .78), panel((L / 2 + 1.7, -W / 2 - .03, 1.1), 2.6, 2.1, -math.pi / 2))

def building(name, L, W, floors, fh, wall, base, glass_style, roof=(.3, .31, .32), extra=None, seed=0):
    a = Asset(name, seed)
    H = floors * fh
    a.add(wall, box((L, W, H), (0, 0, H / 2), bottom=False), collide=True)
    a.add(base, box((L + .12, W + .12, 1.), (0, 0, .5), bottom=False))
    for side in range(4):       # parapet
        along = side % 2 == 0
        a.add(base, box((L + .1 if along else .3, .3 if along else W - .5, .8),
                        (0, (W / 2 - .1) * (1 if side == 0 else -1), H + .4) if along else ((L / 2 - .1) * (1 if side == 1 else -1), 0, H + .4)))
    a.add(roof, [[(-L / 2 + .2, -W / 2 + .2, H + .02), (L / 2 - .2, -W / 2 + .2, H + .02), (L / 2 - .2, W / 2 - .2, H + .02), (-L / 2 + .2, W / 2 - .2, H + .02)]])
    a.add((.6, .62, .63), box((2.6, 2., 1.6), (L / 5, W / 6, H + .8), bottom=False), collide=True)
    glass = []
    for f in range(1, floors):
        z = f * fh + fh * .55
        for (cx, cy, yaw, span) in ((0, -W / 2 - .02, -math.pi / 2, L), (0, W / 2 + .02, math.pi / 2, L),
                                    (-L / 2 - .02, 0, math.pi, W), (L / 2 + .02, 0, 0, W)):
            t = np.array([-math.sin(yaw), math.cos(yaw)])
            if glass_style == 'ribbon':
                glass += panel((cx, cy, z), span - 1.2, fh * .45, yaw)
            else:
                for u in np.linspace(-span / 2 + 1.4, span / 2 - 1.4, max(2, round(span / 4))):
                    glass += panel((cx + t[0] * u, cy + t[1] * u, z), 1.1, fh * .48, yaw)
    a.add(GLASS, glass)
    a.add((.25, .2, .15), panel((0, -W / 2 - .04, 1.15), 1.6, 2.3, -math.pi / 2))
    if extra:
        extra(a, L, W, fh)
    return a

def canopy_entrance(a, L, W, fh):
    a.add((.25, .27, .3), box((4., 1.8, .18), (0, -W / 2 - .9, 2.9)), collide=True)

def shopfront(color):
    def f(a, L, W, fh):
        a.add(GLASS, panel((-L / 4 - .5, -W / 2 - .03, 1.7), L / 2 - 1.8, 2.1, -math.pi / 2) + panel((L / 4 + .5, -W / 2 - .03, 1.7), L / 2 - 1.8, 2.1, -math.pi / 2))
        a.add(color, box((L - .6, 1.5, .08), (0, -W / 2 - .7, 3.05), (.35, 0, 0)), collide=True)
    return f

def balconies(a, L, W, fh):
    floors = round(a.vis[GLASS][-1][0][2] / fh)    # top floor index from the highest window row
    for f in range(1, floors + 1, 2):
        for x in (-L / 4, L / 4):
            a.add((.7, .7, .68), box((2.8, 1.2, .15), (x, -W / 2 - .6, f * fh + .08)), collide=True)
            a.add((.36, .38, .4), panel((x, -W / 2 - 1.2, f * fh + .6), 2.8, .9, -math.pi / 2))

def car(name, color):
    a = Asset(name)
    body = [(-2.15, .32), (2.15, .32), (2.22, .62), (2.02, .92), (-2.05, .98), (-2.2, .7)]
    a.add(color, extrude(body, -.88, .88, 'y'), collide=True)
    a.add(GLASS, extrude([(-1.55, .97), (1.0, .93), (.32, 1.42), (-1.3, 1.44)], -.8, .8, 'y'), collide=True)
    a.add(color, [[(-1.28, -.8, 1.455), (.33, -.8, 1.435), (.33, .8, 1.435), (-1.28, .8, 1.455)]])
    tire = []
    for x in (-1.35, 1.35):
        for y0, y1 in ((-.95, .95),):
            tire += extrude([(x + .36 * math.cos(2 * math.pi * i / 6 + math.pi / 6), .36 + .36 * math.sin(2 * math.pi * i / 6 + math.pi / 6)) for i in range(6)], y0, y1, 'y')
    a.add((.09, .09, .1), tire)
    a.add((.95, .92, .7), panel((2.17, -.6, .75), .32, .14, 0) + panel((2.17, .6, .75), .32, .14, 0))
    a.add((.7, .08, .06), panel((-2.21, -.65, .8), .3, .14, math.pi) + panel((-2.21, .65, .8), .3, .14, math.pi))
    return a

def all_assets():
    return [
        tree('tree', 3.2, [((0, 0, 4.2), (2.1, 2.1, 1.6)), ((.5, .3, 5.2), (1.6, 1.5, 1.3))], 1),
        tree('tree_02', 4.2, [((0, 0, 6.), (3.2, 3., 2.3)), ((1.5, -.6, 7.), (2.1, 2., 1.7)), ((-1.2, .9, 7.6), (2., 1.9, 1.6))], 2),
        pine(3), bush(4), rock(5), hay_bale(), shed(), fence(), electric_pole(),
        house('house_01', 8., 6., 5.4, math.radians(36), (.88, .82, .68), (.58, .23, .18), 2, -1.2),
        house('house_02', 9., 6.5, 2.9, math.radians(28), (.5, .66, .76), (.33, .27, .22), 1, -1.2, porch),
        house('house_03', 8.5, 6., 3.2, math.radians(42), (.84, .76, .6), (.25, .35, .46), 1, 1.5, garage),
        building('building_01', 14., 10., 4, 3.4, (.6, .63, .66), (.38, .4, .43), 'ribbon', extra=canopy_entrance),
        building('building_02', 12., 9., 2, 3.6, (.8, .68, .52), (.48, .4, .32), 'grid', extra=shopfront((.66, .2, .18))),
        building('building_03', 12., 12., 6, 3., (.87, .85, .79), (.55, .55, .52), 'grid', extra=balconies),
        building('building_04', 16., 10., 3, 3.3, (.6, .34, .27), (.42, .27, .22), 'grid', extra=shopfront((.2, .35, .3))),
        car('car_01', (.2, .33, .55)), car('car_02', (.85, .86, .87)), car('car_03', (.45, .1, .12)),
    ]

################################################################################
# Footprints and placement

def footprints(a):
    """Local (cx, cy, hx, hy) of all collision hulls and of the hulls low enough to touch the target box,
    the top height, and the underside of a canopy that clears the box (0 if none)."""
    def rect(faces_list):
        pts = np.array([v for faces in faces_list for f in faces for v in f])
        lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
        return (*(lo + hi) / 2, *(hi - lo) / 2)
    low = [faces for faces in a.col if min(v[2] for f in faces for v in f) < CLEAR_FROM_BOX]
    zs = [min(v[2] for f in faces for v in f) for faces in a.col]
    canopy = min((z for z in zs if z >= CLEAR_FROM_BOX), default=0.) if a.name != 'bush' else 0.
    return rect(a.col), (rect(low) if low and a.name != 'bush' else None), max(v[2] for faces in a.col for f in faces for v in f), canopy

def obb(local, x, y, yaw):
    cx, cy, hx, hy = local
    return (x + cx * math.cos(yaw) - cy * math.sin(yaw), y + cx * math.sin(yaw) + cy * math.cos(yaw), hx, hy, yaw)

def overlap(a, b, m=0.):
    for yaw in (a[4], a[4] + math.pi / 2, b[4], b[4] + math.pi / 2):
        c, s = math.cos(yaw), math.sin(yaw)
        ext = lambda o: o[2] * abs(math.cos(yaw - o[4])) + o[3] * abs(math.sin(yaw - o[4]))
        if abs((a[0] - b[0]) * c + (a[1] - b[1]) * s) > ext(a) + ext(b) + m:
            return False
    return True

VEG = {'tree', 'tree_02', 'pine', 'bush'}
LAYER_TOP = {'patch': .015, 'track': .03, 'road': .045}

class World:
    def __init__(self, name, seed, ground):
        self.name, self.rng = name, np.random.default_rng(seed)
        self.objs, self.surfaces, self.targets = [], [], []
        self.solid, self.veg, self.trunks, self.roads = [], [], [], []
        self.cover = []         # (footprint, asset, canopy underside) of every prop, to rate hiding spots
        self.contexts = []      # one label per target
        self.surface(0, 0, 500, 500, 0, ground, None)

    def surface(self, x, y, sx, sy, yaw, rgb, layer='patch'):
        top = 0 if layer is None else LAYER_TOP[layer]
        th = .04 if layer is None else .02
        self.surfaces.append({'type': 'surface', 'size_m': [round(sx, 2), round(sy, 2), th], 'position_m': [round(x, 2), round(y, 2), round(top - th / 2, 3)],
                              'rpy_rad': [0, 0, round(yaw, 4)], 'rgba': [*rgb, 1]})

    def road(self, pts, width, rgb, layer='road', block=True):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            yaw, length = math.atan2(y1 - y0, x1 - x0), math.hypot(x1 - x0, y1 - y0)
            self.surface((x0 + x1) / 2, (y0 + y1) / 2, length + width * .8, width, yaw, rgb, layer)
            if block:
                self.roads.append(((x0 + x1) / 2, (y0 + y1) / 2, length / 2 + width * .4, width / 2, yaw))

    def target(self, x, y, yaw, context):
        self.targets.append((x, y, BOX_SIZE[0] / 2, BOX_SIZE[1] / 2, yaw))
        self.contexts.append(context)

    def place(self, name, x, y, yaw=0., margin=.5, on_road=False):
        fp_local, blk_local, top, canopy = FOOTPRINT[name]
        fp = obb(fp_local, x, y, yaw)
        blk = obb(blk_local, x, y, yaw) if blk_local else None
        if max(abs(x), abs(y)) > 247 or math.hypot(x, y) < 9:
            return False
        if blk and any(overlap(blk, t, .45) for t in self.targets):
            return False
        base = blk or fp
        if not on_road and any(overlap(base, r, .2) for r in self.roads):
            return False
        # A canopy may overhang a solid lower than its underside; trunks and solids never overlap.
        if name in VEG:
            if any(overlap(base, o, .1) or (o_top > canopy and overlap(fp, o, .1)) for o, o_top in self.solid) \
                    or (blk and any(overlap(blk, o, .7) for o in self.trunks)):
                return False
            self.veg.append((fp, canopy))
            if blk:
                self.trunks.append(blk)
        else:
            if any(overlap(fp, o, margin) for o, _ in self.solid) or any(overlap(fp, o, .1) for o in self.trunks) \
                    or any(top > o_canopy and overlap(fp, o, 0) for o, o_canopy in self.veg):
                return False
            self.solid.append((fp, top))
        self.cover.append((fp, name, canopy))
        self.objs.append({'type': 'urdf', 'asset': name + '.urdf', 'position_m': [round(x, 2), round(y, 2), 0], 'rpy_rad': [0, 0, round(yaw % (2 * math.pi), 4)]})
        return True

    def candidates(self, limit=135., tries=30000):
        """Box poses beside existing props that clear every solid and trunk (bushes may clip them), sit off
        roads, inside `limit` and away from the start, with some cover: a canopy over them, bushes against
        them, or walls, vehicles, rocks, or bales beside them. `pick_targets` measures how hidden they are."""
        props = [c for c in self.cover if c[1] != 'bush']
        out = []
        for _ in range(tries):
            fp, name, _ = props[self.rng.integers(len(props))]
            ang, d = self.rng.uniform(0, 2 * math.pi), self.rng.uniform(1.5, 3.5) + max(fp[2], fp[3]) * .6
            x, y = fp[0] + d * math.cos(ang), fp[1] + d * math.sin(ang)
            if max(abs(x), abs(y)) > limit or math.hypot(x, y) < 14:
                continue
            yaw = fp[4] + self.rng.choice([0, math.pi / 2]) + self.rng.uniform(-.15, .15)   # roughly aligned with its anchor
            box = (x, y, BOX_SIZE[0] / 2, BOX_SIZE[1] / 2, yaw)
            if any(overlap(box, o, .4) for o, _ in self.solid) or any(overlap(box, o, .45) for o in self.trunks) \
                    or any(overlap(box, o, .5) for o in self.roads):
                continue
            canopy = any(c > 0 and overlap(box, o, -.6) for o, _, c in self.cover)
            bushes = [o for o, n, _ in self.cover if n == 'bush' and overlap(box, o, 0)]
            if any(math.hypot(o[0] - x, o[1] - y) < 1.4 for o in bushes):
                continue                    # a bush may clip an end, not swallow the box
            near = [n for o, n, _ in self.cover if n not in VEG and overlap(box, o, 2.)]
            kind = next((k for k, names in (('wall', ('house', 'building', 'shed')), ('cars', ('car',)),
                                            ('rocks or bales', ('rock', 'hay'))) if any(n.startswith(names) for n in near)), None)
            context = ' + '.join(filter(None, ['under canopy' if canopy else None, kind, 'in bushes' if bushes else None]))
            if context and all(math.hypot(x - c[0], y - c[1]) > 5 for c in out[-200:]):
                out.append((x, y, yaw, context))
        return out

    def scatter(self, names, n, sample, tries=40, **kw):
        placed = 0
        for _ in range(n):
            for _ in range(tries):
                x, y = sample()
                if self.place(self.rng.choice(names), x, y, self.rng.uniform(0, 2 * math.pi), **kw):
                    placed += 1
                    break
        return placed

    def line(self, names, p0, p1, spacing, jitter=0., yaw_off=None, gap=0., **kw):
        (x0, y0), (x1, y1) = p0, p1
        length, yaw = math.hypot(x1 - x0, y1 - y0), math.atan2(y1 - y0, x1 - x0)
        for s in np.arange(spacing / 2, length, spacing):
            if gap and self.rng.random() < gap:
                continue
            j = self.rng.normal(0, jitter, 2) if jitter else (0, 0)
            ry = self.rng.uniform(0, 2 * math.pi) if yaw_off is None else yaw + yaw_off
            self.place(self.rng.choice(names), x0 + (x1 - x0) * s / length + j[0], y0 + (y1 - y0) * s / length + j[1], ry, **kw)

    def ring(self, names, x, y, r, n, **kw):
        for k in range(n):
            ang = 2 * math.pi * k / n + self.rng.uniform(-.3, .3)
            rr = r * self.rng.uniform(.85, 1.15)
            self.place(self.rng.choice(names), x + rr * math.cos(ang), y + rr * math.sin(ang), self.rng.uniform(0, 2 * math.pi), **kw)

    def near(self, cx, cy, r):
        return lambda: (cx + self.rng.uniform(-r, r), cy + self.rng.uniform(-r, r))

    def annulus(self, r0, r1=250):
        def f():
            while True:
                x, y = self.rng.uniform(-r1, r1, 2)
                if max(abs(x), abs(y)) >= r0:
                    return x, y
        return f

    def write(self):
        targets = [{'position_m': [round(x, 2), round(y, 2), 0], 'rpy_rad': [0, 0, round(yaw % (2 * math.pi), 4)], 'context': c}
                   for (x, y, _, _, yaw), c in zip(self.targets, self.contexts)]
        # The renderer draws every triangle, near or far: past BORDER_M only the props around a target remain.
        objs = [o for o in self.objs if max(map(abs, o['position_m'][:2])) < BORDER_M
                or any(math.hypot(o['position_m'][0] - x, o['position_m'][1] - y) < 20 for x, y, *_ in self.targets)]
        spec = {'name': self.name, 'targets': targets, 'objects': objs + self.surfaces}
        (TERRAINS / f'{self.name}.json').write_text(json.dumps(spec, indent=1) + '\n')

def hide(w, x, y, yaw, bushes=((-2.1, .5), (2.2, -.4)), tree=None):
    """Bushes clipping into the box (allowed) and an optional canopy tree beside it, on side +1 or -1."""
    for u, v in bushes:
        w.place('bush', *local(x, y, yaw, u, v), w.rng.uniform(0, 6))
    if tree:
        w.place('tree_02', *local(x, y, yaw, w.rng.uniform(-.8, .8), tree * 1.75), w.rng.uniform(0, 6))

def local(x, y, yaw, u, v):
    """World point at (u, v) in a frame at (x, y) rotated by yaw."""
    return x + u * math.cos(yaw) - v * math.sin(yaw), y + u * math.sin(yaw) + v * math.cos(yaw)

################################################################################
# Scenes. Targets are reserved first, so later props leave room for every version's box.

GRASS, MEADOW, WHEAT, PLOUGH, CROP, RAPE = (.5, .65, .33), (.56, .7, .37), (.8, .72, .43), (.47, .36, .25), (.42, .6, .26), (.83, .78, .33)
DIRT, GRAVEL, ASPHALT, PAVING = (.55, .46, .34), (.62, .6, .55), (.24, .25, .27), (.6, .6, .58)
TREES, VEGETATION = ['tree', 'tree', 'tree_02', 'pine'], ['tree', 'tree_02', 'bush', 'bush']

def field():
    w = World('field', 11, GRASS)
    xs, ys = [-250, -178, -104, -36, 30, 96, 168, 250], [-250, -170, -95, -28, 40, 112, 180, 250]
    for i in range(7):
        for j in range(7):
            if w.rng.random() < .8:
                x0, x1, y0, y1 = xs[i] + 4, xs[i + 1] - 4, ys[j] + 4, ys[j + 1] - 4
                w.surface((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0, 0, [WHEAT, PLOUGH, CROP, MEADOW, RAPE][w.rng.integers(5)])
    w.road([(30, -250), (30, 250)], 4, DIRT, 'track')
    w.road([(-250, 40), (250, 40)], 4, DIRT, 'track')
    w.road([(30, 40), (96, 70), (112, 80)], 3.5, GRAVEL, 'track')
    # Targets: hedgerow under a tree, among hay bales, behind a shed, paddock corner, rock pile, track junction.
    w.target(-60, -25.4, 0., 'hedgerow, under an oak')
    w.target(132, 98, .35, 'among hay bales')
    w.target(104, 103.6, 0., 'behind a shed')
    w.target(-150, -104, 1.3, 'paddock corner, under an oak')
    w.target(-33.5, 107.5, 2.4, 'rock pile')
    w.target(24.4, 49, math.pi / 2, 'track junction, under an oak')
    hide(w, -60, -25.4, 0., tree=1)
    for dx, dy in ((-3.4, -.8), (3.4, .9), (-2.4, 2.3), (.5, 2.4), (2.6, -2.1), (-.4, -2.2)):
        bx, by = local(132, 98, .35, dx, dy)
        w.place('hay_bale', bx, by, .35 + (math.pi / 2 if abs(dy) > 2 else 0), margin=.1)
    w.place('shed', 104, 100.2, 0.), hide(w, 104, 103.6, 0., ((-2.2, .2), (2.2, .4), (.6, 1.1)))
    hide(w, -150, -104, 1.3, ((-2., .4), (1.8, .6), (0, -1.)), tree=1)
    for dx in (-3, 0, 3):
        w.place('rock', -33.5 + dx + w.rng.uniform(-.5, .5), 110.6 + w.rng.uniform(0, .6), w.rng.uniform(0, 6), margin=.1)
    w.place('rock', -37.2, 105.5, 1.), w.place('bush', -30, 106.5), w.place('bush', -34, 104.4)
    hide(w, 24.4, 49, math.pi / 2, ((-2.1, .3), (2.1, -.3), (.3, .9)), tree=1)
    # Farmsteads
    for (fx, fy, yaw, kind) in ((112, 82, math.pi, 'house_01'), (-138, -118, .5, 'house_03')):
        w.place(kind, fx, fy, yaw)
        sx, sy = local(fx, fy, yaw, -9, 2)
        w.place('shed', sx, sy, yaw)
        w.ring(['tree', 'tree_02'], fx, fy, 14, 7)
        w.scatter(['bush'], 6, w.near(fx, fy, 15))
    px, py, pyaw = -150, -100, .5
    for k in range(4):     # paddock around the second target
        for s in range(-3, 4):
            if k == 2 and s == 0:
                continue
            u, v = ((s * 4.2, -12.6), (12.6, s * 4.2), (s * 4.2, 12.6), (-12.6, s * 4.2))[k]
            fx, fy = local(px, py, pyaw, u, v)
            w.place('fence', fx, fy, pyaw + (math.pi / 2 if k % 2 else 0), margin=-.15)
    w.scatter(['hay_bale'], 8, w.near(150, 140, 18), margin=.3)
    # Hedgerows along parcel edges, with gaps
    for x in xs[1:-1]:
        for j in range(7):
            if x != 30 and w.rng.random() < .45:
                w.line(['bush', 'bush', 'tree', 'tree_02', 'pine'], (x, ys[j] + 6), (x, ys[j + 1] - 6), 5.5, .6, gap=.3)
    for y in ys[1:-1]:
        for i in range(7):
            if y != 40 and w.rng.random() < .45:
                w.line(['bush', 'bush', 'tree', 'tree_02'], (xs[i] + 6, y), (xs[i + 1] - 6, y), 5.5, .6, gap=.3)
    w.line(['electric_pole'], (35, -245), (35, 245), 42, yaw_off=0.)
    w.line(['fence'], (34, -60), (34, 20), 4.2, yaw_off=0., margin=-.15)
    w.scatter(['tree', 'tree_02'], 14, w.annulus(0))
    w.scatter(['rock'], 12, w.annulus(0))
    return w

def forest():
    w = World('forest', 12, (.24, .36, .2))
    trail = [(-250, -60), (-150, -30), (-60, -10), (0, -14), (60, 10), (140, 40), (250, 30)]
    branch = [(60, 10), (80, 120), (60, 250)]
    clearings = [(0, 0, 13), (90, -70, 20), (-110, 90, 22), (150, 130, 16), (-170, -160, 18)]
    for cx, cy, r in clearings:
        for yaw in (0, math.pi / 4):
            w.surface(cx, cy, 1.7 * r, 1.7 * r, yaw, (.42, .55, .3))
    for _ in range(40):
        x, y = w.rng.uniform(-240, 240, 2)
        w.surface(x, y, *w.rng.uniform(10, 40, 2), w.rng.uniform(0, 3), [(.3, .4, .22), (.35, .3, .2), (.2, .32, .19)][w.rng.integers(3)])
    w.road(trail, 3, (.42, .35, .25), 'track')
    w.road(branch, 2.5, (.42, .35, .25), 'track')
    # Targets: broadleaf undergrowth, clearing edge, trailside bushes, boulders, behind a hut, dense pines.
    w.target(-70, 30, .4, 'undergrowth, under an oak')
    w.target(67, -70, math.pi / 2, 'clearing edge, under an oak')
    w.target(-100, -14.6, .22, 'trailside bushes')
    w.target(40, 95, 2., 'between boulders')
    w.target(-86, 93.6, 0., 'behind a hut')
    w.target(150, -40, 1., 'inside a pine stand')
    hide(w, -70, 30, .4, ((-2.1, .4), (2.1, -.5), (.5, 1.)), tree=1)
    hide(w, 67, -70, math.pi / 2, tree=-1)
    for dx in (-3.5, -1, 1.5, 4):
        bx, by = local(-100, -14.6, .22, dx, 2.4)
        w.place('bush', bx, by)
    for dx, dy in ((-3.6, 1.6), (0, 2.6), (3.5, 1.5), (-1.8, -2.7), (2.4, -2.6)):
        bx, by = local(40, 95, 2., dx, dy)
        w.place('rock', bx, by, w.rng.uniform(0, 6), margin=.05)
    hide(w, 40, 95, 2., tree=-1)
    w.place('shed', -86, 90.4, 0.), w.place('pine', -82, 97.5), w.place('bush', -89.5, 95)
    w.ring(['pine'], 150, -40, 4.2, 7)
    # Forest: pines dominate the east, broadleaves the west, denser near the centre.
    def keep(x, y):
        if any(math.hypot(x - cx, y - cy) < r for cx, cy, r in clearings):
            return False
        return True
    for _ in range(4000):
        r = 245 * w.rng.random() ** 1.5 if w.rng.random() < .85 else 245 * w.rng.random()
        x, y = local(0, 0, w.rng.uniform(0, 2 * math.pi), min(r, 245), 0)
        if keep(x, y):
            name = 'pine' if w.rng.random() < 1 / (1 + math.exp(-x / 60)) else w.rng.choice(['tree', 'tree', 'tree_02'])
            w.place(name, x, y, w.rng.uniform(0, 2 * math.pi))
        if len(w.objs) > 330:
            break
    w.scatter(['bush'], 60, w.near(0, 0, 140))
    for _ in range(9):
        cx, cy = w.rng.uniform(-230, 230, 2)
        w.scatter(['rock'], 3, w.near(cx, cy, 5), margin=.05)
    w.ring(['bush', 'tree'], 90, -70, 22, 10)
    return w

def village():
    w = World('village', 13, (.5, .64, .36))
    main = [(-250, -22), (-120, -9), (40, -4), (130, 8), (250, 30)]
    side_n, side_s = [(40, -4), (52, 120), (70, 250)], [(-60, -7), (-74, -120), (-90, -250)]
    for _ in range(14):     # outer fields
        x, y = w.annulus(150)()
        w.surface(x, y, *w.rng.uniform(30, 70, 2), w.rng.uniform(0, .4), [WHEAT, PLOUGH, CROP, MEADOW][w.rng.integers(4)])
    w.road(main, 7, (.4, .4, .4))
    w.road(side_n, 5, (.45, .44, .42))
    w.road(side_s, 5, (.45, .44, .42))
    w.surface(40, 6, 22, 14, .05, PAVING, 'track')     # village square
    lots = []
    for pts, rw in ((main, 7), (side_n, 5), (side_s, 5)):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            th, length = math.atan2(y1 - y0, x1 - x0), math.hypot(x1 - x0, y1 - y0)
            for s in np.arange(14, length - 10, 19.5):
                for side in (1, -1):
                    cx, cy = local(x0, y0, th, s + w.rng.uniform(-1.5, 1.5), side * (rw / 2 + 11))
                    if max(abs(cx), abs(cy)) < 150 and math.hypot(cx - 40, cy) > 22:
                        lots.append((cx, cy, th if side == 1 else th + math.pi))
    # Lot frame: u along the road, v away from it; the house centre is the origin.
    targets = {4: ('backyard', 2., 7.5, 0.), 9: ('alley', 9.7, .5, math.pi / 2), 15: ('driveway', 7.7, -2.6, math.pi / 2),
               22: ('garden tree', -6., 10., 0.), 27: ('rear corner', 5.5, 15.5, 0.)}
    lots = [l for l in lots if math.hypot(l[0], l[1]) > 15]
    for i, (kind, u, v, yaw) in sorted(targets.items()):
        x, y, th = lots[i]
        w.target(*local(x, y, th, u, v), th + yaw, kind)
    w.target(-95, 62, .7, 'copse outside the village')
    for i, (x, y, th) in enumerate(lots):
        house_name = ['house_01', 'house_02', 'house_03'][w.rng.integers(3)]
        if i in targets and targets[i][0] == 'garden tree':
            w.place('tree_02', *local(x, y, th, -6.2, 12.5), 0.)
        if i in targets and targets[i][0] == 'backyard':
            w.place('shed', *local(x, y, th, 2., 10.8), th)
        w.place(house_name, x, y, th)
        if w.rng.random() < .7 or i in targets:
            w.place(['car_01', 'car_02', 'car_03'][w.rng.integers(3)], *local(x, y, th, 7.7, -6.8), th + math.pi / 2)
        w.surface(*local(x, y, th, 7.7, -6.), 3.2, 7, th, GRAVEL, 'track')
        lot_fence = w.rng.random() < .3
        for u in np.arange(-7.8, 6.1, 4.2):      # front fence, gap at the driveway
            if lot_fence:
                w.place('fence', *local(x, y, th, u, -8.6), th, margin=-.15)
        if w.rng.random() < .2 or i in targets:
            w.line(['bush'], local(x, y, th, -9.5, -6), local(x, y, th, -9.5, 15), 4.5, .3)
        if i in targets and targets[i][0] == 'rear corner':
            w.line(['fence'], local(x, y, th, -9, 17.9), local(x, y, th, 10, 17.9), 4.2, yaw_off=0., margin=-.15)
        if w.rng.random() < .35:
            w.place('shed', *local(x, y, th, -5, 10.5), th + w.rng.uniform(-.2, .2))
        for _ in range(w.rng.integers(1, 4)):
            w.place(w.rng.choice(['tree', 'tree', 'tree_02', 'pine']), *local(x, y, th, w.rng.uniform(-8, 8), w.rng.uniform(8, 16)))
        if w.rng.random() < .3:
            w.place('bush', *local(x, y, th, w.rng.uniform(-4, 4), -5.5))
        if i in targets:
            tx, ty, _, _, tyaw = w.targets[sorted(targets).index(i)]
            hide(w, tx, ty, tyaw, tree=1 if targets[i][0] in ('backyard', 'rear corner') else None)
    hide(w, -95, 62, .7, ((-2.1, .4), (2.1, -.4), (0, 1.)), tree=-1)
    w.ring(['tree', 'tree_02', 'bush'], -95, 62, 5.5, 7)
    w.ring(['tree', 'tree_02', 'pine'], -95, 62, 10, 10)
    w.place('tree_02', 40, 6), w.ring(['bush'], 40, 6, 5, 6)
    for pts in (main,):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            th = math.atan2(y1 - y0, x1 - x0)
            w.line(['electric_pole'], local(x0, y0, th, 0, -5.5), local(x1, y1, th, 0, -5.5), 36, yaw_off=0.)
    for _ in range(8):      # outlying farms
        x, y = w.annulus(160)()
        yaw = w.rng.uniform(0, 2 * math.pi)
        if w.place(['house_01', 'house_03'][w.rng.integers(2)], x, y, yaw):
            w.place('shed', *local(x, y, yaw, -10, 3), yaw)
            w.ring(['tree', 'tree_02'], x, y, 13, 5)
    for _ in range(10):
        cx, cy = w.annulus(150)()
        w.scatter(TREES + ['bush'], 5, w.near(cx, cy, 14))
    w.scatter(TREES, 30, w.annulus(150))
    return w

def city():
    w = World('city', 14, (.5, .62, .36))
    w.surface(0, 0, 300, 300, 0, PAVING)
    lines = [-140, -84, -28, 28, 84, 140]
    for c in lines:
        w.road([(c, -250), (c, 250)], 8 if abs(c) < 100 else 7, ASPHALT)
        w.road([(-250, c), (250, c)], 8 if abs(c) < 100 else 7, ASPHALT)
    w.surface(56, -56, 46, 46, 0, (.33, .34, .35))
    for bx, by in ((0, 0), (-112, 56)):
        w.surface(bx, by, 46, 46, 0, (.45, .6, .32))
        w.surface(bx, by, 46, 3, 0, GRAVEL, 'track'), w.surface(bx, by, 3, 46, 0, GRAVEL, 'track')
    # Targets: courtyard, alley, parking slot, kerbside among parked cars, park undergrowth, suburban yard.
    w.target(-53, -58, .3, 'block courtyard')
    w.target(56, 33.4, math.pi / 2, 'alley between buildings')
    w.target(53.2, -60, math.pi / 2, 'car park slot')
    w.target(-60, -30.9, 0., 'kerbside among parked cars')
    w.target(-104, 50, 2.6, 'park undergrowth')
    w.target(40.5, 199, 0., 'suburban yard, by a shed')
    w.place('tree', -55, -55.2), w.place('tree_02', -48, -63)
    hide(w, -104, 50, 2.6, ((-2.1, .4), (2.1, -.4), (.4, -1.)), tree=1)
    w.place('shed', 40.5, 202.4, 0.), hide(w, 40.5, 199, 0., tree=-1)
    for dx in (-6.4, 6.4, -12.6, 12.6, -18.8):
        w.place(['car_01', 'car_02', 'car_03'][w.rng.integers(3)], -60 + dx, -30.9, 0., margin=.3, on_road=True)
    # Parking lot
    for v in (-15, -5, 5, 15):
        for u in np.arange(-19.6, 20, 2.8):
            if w.rng.random() < .55:
                w.place(['car_01', 'car_02', 'car_03'][w.rng.integers(3)], 56 + u, -56 + v, math.pi / 2 + (math.pi if v > 0 else 0), margin=.1)
    # Parks
    for bx, by in ((0, 0), (-112, 56)):
        for _ in range(26):
            u, v = w.rng.uniform(-21, 21, 2)
            if abs(u) > 3 and abs(v) > 3:
                w.place(w.rng.choice(VEGETATION), bx + u, by + v, w.rng.uniform(0, 6))
    # Kerbside parking, street trees and lamps
    for c in lines[1:-1]:
        half = 4
        for a in np.arange(-136, 137, 7.):
            if min(abs(a - l) for l in lines) < 8:
                continue
            for side in (-1, 1):
                if w.rng.random() < .12:
                    w.place(['car_01', 'car_02', 'car_03'][w.rng.integers(3)], c + side * (half - 1.1), a + w.rng.uniform(-.5, .5), math.pi / 2 + (0 if side > 0 else math.pi), margin=.4, on_road=True)
                if w.rng.random() < .12:
                    w.place(['car_01', 'car_02', 'car_03'][w.rng.integers(3)], a + w.rng.uniform(-.5, .5), c + side * (half - 1.1), 0 if side < 0 else math.pi, margin=.4, on_road=True)
        for a in np.arange(-130, 131, 24.):
            if min(abs(a - l) for l in lines) < 9 or abs(c) > 60:
                continue
            for side in (-1, 1):
                w.place('tree', c + side * (half + 2), a + w.rng.uniform(-1, 1), w.rng.uniform(0, 6))
                w.place('tree', a + w.rng.uniform(-1, 1), c + side * (half + 2), w.rng.uniform(0, 6))
        for a in np.arange(-125, 126, 30.):
            if abs(c) < 60:
                w.place('electric_pole', c + half + .8, a + 7, math.pi / 2)
                w.place('electric_pole', a + 7, c - half - .8, 0.)
    # Blocks: buildings line the perimeter, facing the street; a failed fit leaves an alley.
    for bx in (-112, -56, 0, 56, 112):
        for by in (-112, -56, 0, 56, 112):
            if (bx, by) in ((0, 0), (56, -56), (-112, 56)):
                continue
            outer = max(abs(bx), abs(by)) > 100
            names = ['building_02', 'building_04', 'building_02', 'building_01'] if outer else ['building_01', 'building_02', 'building_03', 'building_04']
            for side in range(4):
                n_yaw = side * math.pi / 2          # outward normal
                yaw = n_yaw - math.pi / 2           # building front (-y) faces outward
                u = -18.
                while u < 18:
                    name = names[w.rng.integers(len(names))]
                    L = FOOTPRINT[name][0][2] * 2
                    D = FOOTPRINT[name][0][3] * 2
                    cx, cy = local(bx, by, n_yaw, 24 - 5.5 - D / 2, -(u + L / 2))
                    if outer and w.rng.random() < .45:     # vacant lot
                        u += L + 4
                    elif u + L <= 22 and w.place(name, cx, cy, yaw, margin=.3):
                        u += L + w.rng.uniform(.5, 4.)
                    else:
                        u += 1.5
            w.scatter(['tree', 'bush', 'car_01'], 3 if not outer else 1, w.near(bx, by, 8))
    # Suburbs
    for c in (-28, 28, -84, 84):
        for a in np.arange(160, 245, 18.):
            for s in (-1, 1):
                for side in (-1, 1):
                    h = ['house_01', 'house_02', 'house_03'][w.rng.integers(3)]
                    if w.rng.random() < .15:
                        w.place(h, c + side * 11, s * a + w.rng.uniform(-2, 2), -side * math.pi / 2)
                        w.place(w.rng.choice(TREES), c + side * 19, s * a + w.rng.uniform(-6, 6))
                    if w.rng.random() < .1:
                        w.place(h, s * a + w.rng.uniform(-2, 2), c + side * 11, math.pi if side > 0 else 0.)
    w.scatter(TREES + ['bush'], 18, w.annulus(160))
    return w

################################################################################
# Previews

def previews(scenes, asset_list):
    import pybullet as p
    from PIL import Image, ImageDraw

    def render(client, eye, target, w, h, fov, up=(0, 0, 1), far=1500):
        view = p.computeViewMatrix(eye, target, up, physicsClientId=client)
        proj = p.computeProjectionMatrixFOV(fov, w / h, .1, far, physicsClientId=client)
        _, _, rgba, _, seg = p.getCameraImage(w, h, view, proj, lightDirection=[-.5, -1., 2.], renderer=p.ER_TINY_RENDERER, physicsClientId=client)
        return Image.fromarray(np.reshape(rgba, (h, w, 4))[:, :, :3].astype(np.uint8)), np.reshape(seg, (h, w))

    (OBJECTS / 'previews').mkdir(exist_ok=True)
    tiles = []
    for a in asset_list + ['target_box']:
        name = a if isinstance(a, str) else a.name
        client = p.connect(p.DIRECT)
        body = p.loadURDF(str(OBJECTS / f'{name}.urdf'), useFixedBase=True, physicsClientId=client)
        lo, hi = map(np.array, p.getAABB(body, physicsClientId=client))
        c, r = (lo + hi) / 2, np.linalg.norm(hi - lo) / 2
        img, _ = render(client, c + r * np.array([1.6, -2.3, 1.2]), c, 360, 360, 40)
        tri = 12 if isinstance(a, str) else a.tris()
        ImageDraw.Draw(img).text((8, 8), f'{name}  {tri} tris  {hi[0]-lo[0]:.1f}x{hi[1]-lo[1]:.1f}x{hi[2]-lo[2]:.1f} m', fill=(0, 0, 0))
        img.save(OBJECTS / 'previews' / f'{name}.png')
        tiles.append(img)
        p.disconnect(client)
    sheet = Image.new('RGB', (360 * 5, 360 * math.ceil(len(tiles) / 5)), 'white')
    for i, t in enumerate(tiles):
        sheet.paste(t, (360 * (i % 5), 360 * (i // 5)))
    sheet.save(OBJECTS / 'previews' / 'contact_sheet.png')

    from fpv_sim.utils.terrain import load_terrain
    stats, tops = {}, []
    for w in scenes:
        client = p.connect(p.DIRECT)
        ids = load_terrain(f'{w.name}@v1', client)
        top, _ = render(client, (0, 0, 690), (0, 0, 0), 600, 600, 40, up=(0, 1, 0))
        d = ImageDraw.Draw(top)
        thumbs = []
        for v, ((x, y, _, _, yaw), context) in enumerate(zip(w.targets, w.contexts), 1):
            p.resetBasePositionAndOrientation(ids[0], (x, y, 0), p.getQuaternionFromEuler((0, 0, yaw)), physicsClientId=client)
            px, py = 300 + x / 251 * 300, 300 - y / 251 * 300
            d.ellipse((px - 6, py - 6, px + 6, py + 6), outline=(255, 0, 0), width=2)
            d.text((px + 6, py - 6), str(v), fill=(255, 255, 255))
            above, seen = visibility(client, ids[0], x, y, p)
            stats[f'{w.name}@v{v}'] = (above, seen)
            ex, ey = x + 20 * math.cos(yaw + 2.2), y + 20 * math.sin(yaw + 2.2)
            close, _ = render(client, (ex, ey, 16), (x, y, .8), 300, 210, 50)
            ImageDraw.Draw(close).text((4, 4), f'v{v} {context}\nabove {above:.0%}, low views {seen}/8', fill=(255, 255, 255))
            thumbs.append(close)
        p.disconnect(client)
        top.save(TERRAINS / 'previews' / f'{w.name}.png')
        tops.append(top)
        sheet = Image.new('RGB', (300 * 6, 210 * math.ceil(len(thumbs) / 6)), 'white')
        for i, t in enumerate(thumbs):
            sheet.paste(t, (300 * (i % 6), 210 * (i // 6)))
        sheet.save(TERRAINS / 'previews' / f'{w.name}_targets.png')
    sheet = Image.new('RGB', (1200, 1200), 'white')
    for i, t in enumerate(tops):
        sheet.paste(t, (600 * (i % 2), 600 * (i // 2)))
    sheet.save(TERRAINS / 'previews' / 'contact_sheet.png')
    return stats

################################################################################

def visibility(client, box, x, y, p):
    """Share of the box seen from straight above, and how many of 8 low views 25 m away see it."""
    view = lambda eye, target, w, h, fov, up=(0, 0, 1): p.getCameraImage(
        w, h, p.computeViewMatrix(eye, target, up, physicsClientId=client),
        p.computeProjectionMatrixFOV(fov, w / h, .1, 1500, physicsClientId=client),
        renderer=p.ER_TINY_RENDERER, physicsClientId=client)[4]
    seg = np.reshape(view((x, y, 40), (x, y, 0), 100, 100, 12, up=(0, 1, 0)), -1)
    px_per_m = 100 / (2 * 38.45 * math.tan(math.radians(6)))
    above = min(1., (seg == box).sum() / (BOX_SIZE[0] * BOX_SIZE[1] * px_per_m ** 2))
    seen = sum((np.reshape(view((x + 25 * math.cos(a), y + 25 * math.sin(a), 6), (x, y, .8), 160, 90, 40), -1) == box).sum() >= 10
               for a in np.arange(8) * math.pi / 4)
    return above, seen

def pick_targets(w, total, spacing=12.):
    """Keeps the hand-placed targets, then adds the most hidden candidates, one context at a time, `spacing` apart.

    Every target is checked in PyBullet against the written scene: none may touch a prop other than a bush.
    """
    import pybullet as p
    from fpv_sim.utils.terrain import load_terrain
    candidates = w.candidates()
    w.write()
    spec = json.loads((TERRAINS / f'{w.name}.json').read_text())
    client = p.connect(p.DIRECT)
    ids = load_terrain(f'{w.name}@v1', client)
    solid = [b for o, b in zip(spec['objects'], ids[1:]) if o['type'] == 'urdf' and o['asset'] != 'bush.urdf']
    def clear(x, y, yaw):
        p.resetBasePositionAndOrientation(ids[0], (x, y, 0), p.getQuaternionFromEuler((0, 0, yaw)), physicsClientId=client)
        p.performCollisionDetection(physicsClientId=client)
        return not any(p.getClosestPoints(ids[0], b, 0, physicsClientId=client) for b in solid)
    for t, context in zip(w.targets, w.contexts):
        assert clear(t[0], t[1], t[4]), f'{w.name}: hand-placed target {context} clips a prop'
    queues = {}
    for x, y, yaw, context in candidates:
        if clear(x, y, yaw):
            above, seen = visibility(client, ids[0], x, y, p)
            hidden = (1 - above) + (8 - seen) / 8
            if hidden >= .3:
                queues.setdefault(context, []).append((hidden, x, y, yaw))
    p.disconnect(client)
    for q in queues.values():
        q.sort(reverse=True)
    while len(w.targets) < total and any(queues.values()):
        for context, q in sorted(queues.items()):
            while q and len(w.targets) < total:
                _, x, y, yaw = q.pop(0)
                if all(math.hypot(x - t[0], y - t[1]) >= spacing for t in w.targets):
                    w.target(x, y, yaw, context)
                    break

asset_list = all_assets()
FOOTPRINT = {a.name: footprints(a) for a in asset_list}

################################################################################
# Attack level 1: empty maps, target on the ground, always in the camera's sight.
#
# Sixty setups, ten per distance, distance doubling from 5 m to 160 m.
#   v1..v10 5 m, v11..v20 10 m, v21..v30 20 m, v31..v40 40 m, v41..v50 80 m, v51..v60 160 m.
# Each setup pairs a random ground target with a drone start at the given range, azimuth and height,
# pitched so that the target sits inside the camera's FOV (102 deg HFOV / ~70 deg VFOV, 25 deg uptilt).
# The map is empty apart from the 500 x 500 m ground surface, the same size as the search scenes.

LEVEL_1_DISTANCES = (5., 10., 20., 40., 80., 160.)
LEVEL_1_PER_DISTANCE = 10
# Camera fitted to the drone onboard camera (see perception.Camera); used to bound the start attitude.
CAM_UPTILT_DEG, CAM_HFOV_DEG, CAM_VFOV_DEG = 25., 102., 70.
PITCH_LIMIT_DEG = 60.          # the sim's TOP_SPEED_PITCH_DEG; never start past this
SAFETY_DEG = 3.                # keep the target off the FOV edge by this margin

GRASS = (0.5, 0.65, 0.33)

def _level_1_height_cap(d):
    """Max drone height at distance d so that a start pitch exists that keeps the target inside the vertical FOV."""
    # optical-axis elevation = UPTILT - pitch; the pitch is capped at PITCH_LIMIT_DEG (nose down).
    # Feasible iff (UPTILT + depression) - PITCH_LIMIT <= VFOV/2 - safety.
    max_dep = math.radians(CAM_VFOV_DEG / 2 - SAFETY_DEG + PITCH_LIMIT_DEG - CAM_UPTILT_DEG)
    return min(30., math.tan(max_dep) * d)

def _level_1_pitch_window(depression):
    """Inclusive [lo, hi] start-pitch range, in radians, that keeps the target inside the vertical FOV."""
    half_v = math.radians(CAM_VFOV_DEG / 2 - SAFETY_DEG)
    limit = math.radians(PITCH_LIMIT_DEG)
    uptilt = math.radians(CAM_UPTILT_DEG)
    # offset of the optical axis from the target = (uptilt - pitch) + depression; bound its absolute value by half_v.
    lo = max(-limit, uptilt + depression - half_v)
    hi = min(limit, uptilt + depression + half_v)
    return lo, hi

def build_level_1(seed=101):
    """Returns the level_1 terrain spec (name, setups, objects), seeded and reproducible."""
    rng = np.random.default_rng(seed)
    setups = []
    for d in LEVEL_1_DISTANCES:
        h_cap = max(2., _level_1_height_cap(d))
        target_half = max(50., min(220., 245. - d - 5.))     # keep the drone (at distance d) inside the 500 m tile
        for _ in range(LEVEL_1_PER_DISTANCE):
            tx, ty = rng.uniform(-target_half, target_half, 2)
            phi = rng.uniform(0, 2 * math.pi)
            dx, dy = tx + d * math.cos(phi), ty + d * math.sin(phi)
            h = rng.uniform(2., h_cap)
            depression = math.atan2(h, d)
            # The drone yaws toward the target, plus a small offset that keeps it inside the horizontal FOV.
            bearing = math.atan2(ty - dy, tx - dx)
            yaw_off = rng.uniform(-math.radians(CAM_HFOV_DEG / 2 - SAFETY_DEG - 10),
                                  math.radians(CAM_HFOV_DEG / 2 - SAFETY_DEG - 10))
            yaw = bearing + yaw_off
            lo, hi = _level_1_pitch_window(depression)
            pitch = rng.uniform(lo, hi)
            roll = rng.uniform(-math.radians(15.), math.radians(15.))
            target_yaw = rng.uniform(0, 2 * math.pi)
            setups.append({
                'target': {'position_m': [round(tx, 2), round(ty, 2), 0],
                           'rpy_rad': [0, 0, round(target_yaw % (2 * math.pi), 4)],
                           'context': 'attack, d={:.0f} m, h={:.1f} m'.format(d, h)},
                'start':  {'position_m': [round(dx, 2), round(dy, 2), round(h, 2)],
                           'rpy_rad': [round(roll, 4), round(pitch, 4), round(yaw, 4)]},
            })
    ground = {'type': 'surface', 'size_m': [500, 500, 0.04], 'position_m': [0, 0, -0.02],
              'rpy_rad': [0, 0, 0], 'rgba': [*GRASS, 1]}
    return {'name': 'level_1', 'setups': setups, 'objects': [ground]}

def write_level_1():
    spec = build_level_1()
    (TERRAINS / 'level_1.json').write_text(json.dumps(spec, indent=1) + '\n')
    return spec

def level_1_previews(spec):
    """Top-down and per-setup thumbnails for level_1, drawn as 2D schematics (the map is empty)."""
    from PIL import Image, ImageDraw
    PRE = TERRAINS / 'previews'
    PRE.mkdir(exist_ok=True)
    setups = spec['setups']

    def to_px(x, y, w=600, half=250.):
        return w / 2 + x / half * (w / 2 - 10), w / 2 - y / half * (w / 2 - 10)

    top = Image.new('RGB', (600, 600), (60, 78, 40))
    d = ImageDraw.Draw(top)
    for i, s in enumerate(setups, 1):
        tx, ty, _ = s['target']['position_m']
        dx, dy, _ = s['start']['position_m']
        px_t, py_t = to_px(tx, ty)
        px_d, py_d = to_px(dx, dy)
        d.line((px_t, py_t, px_d, py_d), fill=(120, 120, 120))
        d.ellipse((px_d - 3, py_d - 3, px_d + 3, py_d + 3), fill=(60, 140, 230))
        d.ellipse((px_t - 5, py_t - 5, px_t + 5, py_t + 5), outline=(255, 80, 60), width=2)
        if i % 5 == 1:
            d.text((px_t + 6, py_t - 6), 'v{:d}'.format(i), fill=(255, 255, 255))
    top.save(PRE / 'level_1.png')

    cols, thumb = 6, 200
    rows = math.ceil(len(setups) / cols)
    sheet = Image.new('RGB', (cols * thumb, rows * thumb), (40, 50, 30))
    for i, s in enumerate(setups, 1):
        t = Image.new('RGB', (thumb, thumb), (60, 78, 40))
        td = ImageDraw.Draw(t)
        tx, ty, _ = s['target']['position_m']
        dx, dy, h = s['start']['position_m']
        span = max(8., 2 * max(abs(tx - dx), abs(ty - dy)) + 6)
        cx, cy = (tx + dx) / 2, (ty + dy) / 2
        p_t = to_px(tx - cx, ty - cy, thumb, span)
        p_d = to_px(dx - cx, dy - cy, thumb, span)
        td.line((p_t, p_d), fill=(120, 120, 120))
        td.ellipse((p_d[0] - 4, p_d[1] - 4, p_d[0] + 4, p_d[1] + 4), fill=(60, 140, 230))
        td.ellipse((p_t[0] - 7, p_t[1] - 7, p_t[0] + 7, p_t[1] + 7), outline=(255, 80, 60), width=2)
        td.text((4, 4), 'v{:d}  d~{:.0f}m  h={:.1f}m'.format(i, math.hypot(dx - tx, dy - ty), h), fill=(255, 255, 255))
        sheet.paste(t, (thumb * ((i - 1) % cols), thumb * ((i - 1) // cols)))
    sheet.save(PRE / 'level_1_targets.png')

################################################################################
# Attack level 2: the forest map with level_1's attack setups.
#
# Same sixty setups as level_1 (distance doubling 5..160 m, ten per bucket, target
# on the ground, drone pitched/yawed so the box sits inside the camera FOV), but the
# 377 forest props are kept verbatim: small obstacles (trees, pines, bushes, rocks)
# stand between the drone and the target. Each setup is rejection-sampled on the
# forest world so that:
#   - the target box does not clip any solid prop (bushes may clip, as in level_1);
#   - the drone (a 0.6 m probe sphere) does not intersect any solid prop at its
#     start pose;
#   - rays from the camera eye to the box centre and both long-axis ends hit the
#     box first (clean line of sight at reset, matching level_1's invariant).

LEVEL_2_FOREST = 'forest'
LEVEL_2_DRONE_RADIUS = 0.6
LEVEL_2_MAX_ATTEMPTS = 60000          # per distance bucket; the forest is dense but the box is small
LEVEL_2_SEED = 201

def _load_world(client, objects, with_box=True):
    """Loads a terrain's `objects[]` into a PyBullet DIRECT client; returns (object_ids, box_id|None)."""
    import pybullet as p
    box_id = None
    ids = []
    for obj in objects:
        if obj['type'] == 'urdf':
            ids.append(p.loadURDF(str(OBJECTS / obj['asset']),
                                  basePosition=obj['position_m'],
                                  baseOrientation=p.getQuaternionFromEuler(obj['rpy_rad']),
                                  useFixedBase=True, physicsClientId=client))
        elif obj['type'] == 'surface':
            half = [s / 2 for s in obj['size_m']]
            ids.append(p.createMultiBody(baseMass=0,
                                         baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=half,
                                                                                         physicsClientId=client),
                                         baseVisualShapeIndex=-1,
                                         basePosition=obj['position_m'],
                                         baseOrientation=p.getQuaternionFromEuler(obj['rpy_rad']),
                                         physicsClientId=client))
        else:
            raise ValueError("unknown object type '{:s}'".format(obj['type']))
    if with_box:
        box_id = p.loadURDF(str(OBJECTS / 'target_box.urdf'), useFixedBase=True, physicsClientId=client)
    return ids, box_id

def build_level_2(seed=LEVEL_2_SEED):
    """Returns the level_2 terrain spec (name, setups, objects), seeded and reproducible.

    The objects are the forest scene's, copied verbatim. Each setup is sampled by
    level_1's geometry rules then accepted only if the box is clear of solid props,
    the drone is clear of solid props, and the box has clean line of sight from the
    camera eye at reset.
    """
    import pybullet as p
    from fpv_sim.utils.terrain import TERRAINS_DIR
    forest_spec = json.loads((TERRAINS_DIR / '{:s}.json'.format(LEVEL_2_FOREST)).read_text())
    objects = forest_spec['objects']

    rng = np.random.default_rng(seed)
    client = p.connect(p.DIRECT)
    try:
        forest_ids, box_id = _load_world(client, objects, with_box=True)
        solids = [b for o, b in zip(objects, forest_ids)
                  if o['type'] == 'urdf' and o['asset'] != 'bush.urdf']
        probe = p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE,
                                                                                  radius=LEVEL_2_DRONE_RADIUS,
                                                                                  physicsClientId=client),
                                  baseVisualShapeIndex=-1, physicsClientId=client)
        box_z = BOX_SIZE[2] / 2
        setups = []
        for d in LEVEL_1_DISTANCES:
            h_cap = max(2., _level_1_height_cap(d))
            target_half = max(50., min(220., 245. - d - 5.))
            done = 0
            for _ in range(LEVEL_2_MAX_ATTEMPTS):
                if done >= LEVEL_1_PER_DISTANCE:
                    break
                tx, ty = rng.uniform(-target_half, target_half, 2)
                box_yaw = rng.uniform(0, 2 * math.pi)
                p.resetBasePositionAndOrientation(box_id, (tx, ty, 0),
                                                  p.getQuaternionFromEuler((0, 0, box_yaw)),
                                                  physicsClientId=client)
                p.performCollisionDetection(physicsClientId=client)
                if any(p.getClosestPoints(box_id, b, 0, physicsClientId=client) for b in solids):
                    continue
                phi = rng.uniform(0, 2 * math.pi)
                dx, dy = tx + d * math.cos(phi), ty + d * math.sin(phi)
                if max(abs(dx), abs(dy)) >= 248:
                    continue
                h = rng.uniform(2., h_cap)
                p.resetBasePositionAndOrientation(probe, (dx, dy, h),
                                                  p.getQuaternionFromEuler((0, 0, 0)),
                                                  physicsClientId=client)
                p.performCollisionDetection(physicsClientId=client)
                if any(p.getClosestPoints(probe, b, 0, physicsClientId=client) for b in solids):
                    continue
                depression = math.atan2(h, d)
                bearing = math.atan2(ty - dy, tx - dx)
                yaw_max = math.radians(CAM_HFOV_DEG / 2 - SAFETY_DEG - 10)
                roll = rng.uniform(-math.radians(15.), math.radians(15.))
                pitch = rng.uniform(*_level_1_pitch_window(depression))
                yaw_att = bearing + rng.uniform(-yaw_max, yaw_max)
                quat = p.getQuaternionFromEuler((roll, pitch, yaw_att))
                rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
                eye = np.array([dx, dy, h]) + rot @ np.array([0.30, 0., 0.])
                # Clean LOS: one ray from the camera eye to the box centre (inside the box volume, so
                # PyBullet's rayTest reports the box's entry face along the ray). Combined with level_1's
                # FOV geometry invariant (box centre inside the camera's FOV cone), a single ray suffices.
                if p.rayTest(eye.tolist(), [tx, ty, box_z], physicsClientId=client)[0][0] != box_id:
                    continue
                setups.append({
                    'target': {'position_m': [round(tx, 2), round(ty, 2), 0],
                               'rpy_rad': [0, 0, round(box_yaw % (2 * math.pi), 4)],
                               'context': 'attack, d={:.0f} m, h={:.1f} m (forest)'.format(d, h)},
                    'start':  {'position_m': [round(dx, 2), round(dy, 2), round(h, 2)],
                               'rpy_rad': [round(roll, 4), round(pitch, 4), round(yaw_att, 4)]},
                })
                done += 1
            if done < LEVEL_1_PER_DISTANCE:
                raise RuntimeError('level_2: only {:d}/{:d} setups at d={:.0f} m'.format(
                    done, LEVEL_1_PER_DISTANCE, d))
        return {'name': 'level_2', 'setups': setups, 'objects': objects}
    finally:
        p.disconnect(client)

def write_level_2():
    spec = build_level_2()
    (TERRAINS / 'level_2.json').write_text(json.dumps(spec, indent=1) + '\n')
    return spec

def level_2_previews(spec):
    """Top-down forest map with all setups overlaid, and a per-setup thumbnail sheet."""
    from PIL import Image, ImageDraw
    PRE = TERRAINS / 'previews'
    PRE.mkdir(exist_ok=True)
    objects = spec['objects']
    setups = spec['setups']

    def to_px(x, y, w, half):
        return w / 2 + x / half * (w / 2 - 10), w / 2 - y / half * (w / 2 - 10)

    # Forest palette: trunks dark green, pines blue-green, bushes olive, rocks grey, shed brown.
    kind_colour = {'tree.urdf': (26, 70, 26), 'tree_02.urdf': (30, 80, 30),
                   'pine.urdf': (18, 50, 35), 'bush.urdf': (45, 70, 25),
                   'rock.urdf': (90, 88, 85), 'shed.urdf': (120, 60, 30)}
    top = Image.new('RGB', (600, 600), (24, 40, 22))
    d = ImageDraw.Draw(top)
    for o in objects:
        if o['type'] != 'urdf':
            continue
        px, py = to_px(o['position_m'][0], o['position_m'][1], 600, 251.)
        c = kind_colour.get(o['asset'], (60, 60, 60))
        r = 2 if o['asset'] in ('tree.urdf', 'tree_02.urdf', 'pine.urdf') else 1
        d.ellipse((px - r, py - r, px + r, py + r), fill=c)
    for i, s in enumerate(setups, 1):
        tx, ty, _ = s['target']['position_m']
        dx, dy, _ = s['start']['position_m']
        p_t = to_px(tx, ty, 600, 251.)
        p_d = to_px(dx, dy, 600, 251.)
        d.line((p_t, p_d), fill=(120, 120, 120))
        d.ellipse((p_d[0] - 3, p_d[1] - 3, p_d[0] + 3, p_d[1] + 3), fill=(60, 140, 230))
        d.ellipse((p_t[0] - 5, p_t[1] - 5, p_t[0] + 5, p_t[1] + 5), outline=(255, 80, 60), width=2)
        if i % 5 == 1:
            d.text((p_t[0] + 6, p_t[1] - 6), 'v{:d}'.format(i), fill=(255, 255, 255))
    top.save(PRE / 'level_2.png')

    cols, thumb = 6, 200
    rows = math.ceil(len(setups) / cols)
    sheet = Image.new('RGB', (cols * thumb, rows * thumb), (24, 40, 22))
    for i, s in enumerate(setups, 1):
        t = Image.new('RGB', (thumb, thumb), (24, 40, 22))
        td = ImageDraw.Draw(t)
        tx, ty, _ = s['target']['position_m']
        dx, dy, h = s['start']['position_m']
        # local window: include all props within `span` m of the setup midpoint
        cx, cy = (tx + dx) / 2, (ty + dy) / 2
        span = max(20., 2 * max(abs(tx - dx), abs(ty - dy)) + 12)
        for o in objects:
            if o['type'] != 'urdf':
                continue
            if max(abs(o['position_m'][0] - cx), abs(o['position_m'][1] - cy)) > span:
                continue
            px, py = to_px(o['position_m'][0] - cx, o['position_m'][1] - cy, thumb, span / 2)
            c = kind_colour.get(o['asset'], (60, 60, 60))
            r = 2 if o['asset'] in ('tree.urdf', 'tree_02.urdf', 'pine.urdf') else 1
            td.ellipse((px - r, py - r, px + r, py + r), fill=c)
        p_t = to_px(tx - cx, ty - cy, thumb, span / 2)
        p_d = to_px(dx - cx, dy - cy, thumb, span / 2)
        td.line((p_t, p_d), fill=(140, 140, 140))
        td.ellipse((p_d[0] - 4, p_d[1] - 4, p_d[0] + 4, p_d[1] + 4), fill=(60, 140, 230))
        td.ellipse((p_t[0] - 7, p_t[1] - 7, p_t[0] + 7, p_t[1] + 7), outline=(255, 80, 60), width=2)
        td.text((4, 4), 'v{:d}  d~{:.0f}m  h={:.1f}m'.format(i, math.hypot(dx - tx, dy - ty), h),
                fill=(255, 255, 255))
        sheet.paste(t, (thumb * ((i - 1) % cols), thumb * ((i - 1) // cols)))
    sheet.save(PRE / 'level_2_targets.png')

LEVEL_3_CITY = 'city'
LEVEL_3_DRONE_RADIUS = LEVEL_2_DRONE_RADIUS
LEVEL_3_MAX_ATTEMPTS = 60000          # per distance bucket; the city is dense but the box is small
LEVEL_3_SEED = 301

def build_level_3(seed=LEVEL_3_SEED):
    """Returns the level_3 terrain spec (name, setups, objects), seeded and reproducible.

    Same as level_2 but overlaying level_1's 60 attack setups on the city scene
    instead of the forest scene. The objects are the city scene's, copied verbatim.
    Each setup is sampled by level_1's geometry rules then accepted only if the box
    is clear of solid props (bushes excepted), the drone is clear of solid props,
    and the box has clean line of sight from the camera eye at reset.
    """
    import pybullet as p
    from fpv_sim.utils.terrain import TERRAINS_DIR
    city_spec = json.loads((TERRAINS_DIR / '{:s}.json'.format(LEVEL_3_CITY)).read_text())
    objects = city_spec['objects']

    rng = np.random.default_rng(seed)
    client = p.connect(p.DIRECT)
    try:
        city_ids, box_id = _load_world(client, objects, with_box=True)
        solids = [b for o, b in zip(objects, city_ids)
                  if o['type'] == 'urdf' and o['asset'] != 'bush.urdf']
        probe = p.createMultiBody(baseMass=0,
                                  baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE,
                                                                                  radius=LEVEL_3_DRONE_RADIUS,
                                                                                  physicsClientId=client),
                                  baseVisualShapeIndex=-1, physicsClientId=client)
        box_z = BOX_SIZE[2] / 2
        setups = []
        for d in LEVEL_1_DISTANCES:
            h_cap = max(2., _level_1_height_cap(d))
            target_half = max(50., min(220., 245. - d - 5.))
            done = 0
            for _ in range(LEVEL_3_MAX_ATTEMPTS):
                if done >= LEVEL_1_PER_DISTANCE:
                    break
                tx, ty = rng.uniform(-target_half, target_half, 2)
                box_yaw = rng.uniform(0, 2 * math.pi)
                p.resetBasePositionAndOrientation(box_id, (tx, ty, 0),
                                                  p.getQuaternionFromEuler((0, 0, box_yaw)),
                                                  physicsClientId=client)
                p.performCollisionDetection(physicsClientId=client)
                if any(p.getClosestPoints(box_id, b, 0, physicsClientId=client) for b in solids):
                    continue
                phi = rng.uniform(0, 2 * math.pi)
                dx, dy = tx + d * math.cos(phi), ty + d * math.sin(phi)
                if max(abs(dx), abs(dy)) >= 248:
                    continue
                h = rng.uniform(2., h_cap)
                p.resetBasePositionAndOrientation(probe, (dx, dy, h),
                                                  p.getQuaternionFromEuler((0, 0, 0)),
                                                  physicsClientId=client)
                p.performCollisionDetection(physicsClientId=client)
                if any(p.getClosestPoints(probe, b, 0, physicsClientId=client) for b in solids):
                    continue
                depression = math.atan2(h, d)
                bearing = math.atan2(ty - dy, tx - dx)
                yaw_max = math.radians(CAM_HFOV_DEG / 2 - SAFETY_DEG - 10)
                roll = rng.uniform(-math.radians(15.), math.radians(15.))
                pitch = rng.uniform(*_level_1_pitch_window(depression))
                yaw_att = bearing + rng.uniform(-yaw_max, yaw_max)
                quat = p.getQuaternionFromEuler((roll, pitch, yaw_att))
                rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
                eye = np.array([dx, dy, h]) + rot @ np.array([0.30, 0., 0.])
                # Clean LOS: one ray from the camera eye to the box centre (inside the box volume, so
                # PyBullet's rayTest reports the box's entry face along the ray). Combined with level_1's
                # FOV geometry invariant (box centre inside the camera's FOV cone), a single ray suffices.
                if p.rayTest(eye.tolist(), [tx, ty, box_z], physicsClientId=client)[0][0] != box_id:
                    continue
                setups.append({
                    'target': {'position_m': [round(tx, 2), round(ty, 2), 0],
                               'rpy_rad': [0, 0, round(box_yaw % (2 * math.pi), 4)],
                               'context': 'attack, d={:.0f} m, h={:.1f} m (city)'.format(d, h)},
                    'start':  {'position_m': [round(dx, 2), round(dy, 2), round(h, 2)],
                               'rpy_rad': [round(roll, 4), round(pitch, 4), round(yaw_att, 4)]},
                })
                done += 1
            if done < LEVEL_1_PER_DISTANCE:
                raise RuntimeError('level_3: only {:d}/{:d} setups at d={:.0f} m'.format(
                    done, LEVEL_1_PER_DISTANCE, d))
        return {'name': 'level_3', 'setups': setups, 'objects': objects}
    finally:
        p.disconnect(client)

def write_level_3():
    spec = build_level_3()
    (TERRAINS / 'level_3.json').write_text(json.dumps(spec, indent=1) + '\n')
    return spec

def level_3_previews(spec):
    """Top-down city map with all setups overlaid, and a per-setup thumbnail sheet."""
    from PIL import Image, ImageDraw
    PRE = TERRAINS / 'previews'
    PRE.mkdir(exist_ok=True)
    objects = spec['objects']
    setups = spec['setups']

    def to_px(x, y, w, half):
        return w / 2 + x / half * (w / 2 - 10), w / 2 - y / half * (w / 2 - 10)

    # City palette: buildings grey, cars reddish, electric poles dark grey, trees green, bushes olive, shed brown.
    kind_colour = {'building_01.urdf': (120, 120, 120), 'building_02.urdf': (110, 110, 115),
                   'building_03.urdf': (130, 130, 125), 'building_04.urdf': (115, 115, 120),
                   'car_01.urdf': (170, 60, 50), 'car_02.urdf': (180, 70, 55), 'car_03.urdf': (160, 55, 45),
                   'electric_pole.urdf': (70, 70, 70),
                   'tree.urdf': (26, 70, 26), 'tree_02.urdf': (30, 80, 30),
                   'bush.urdf': (45, 70, 25), 'shed.urdf': (120, 60, 30)}
    big = ('building_01.urdf', 'building_02.urdf', 'building_03.urdf', 'building_04.urdf')
    top = Image.new('RGB', (600, 600), (30, 32, 36))
    d = ImageDraw.Draw(top)
    for o in objects:
        if o['type'] != 'urdf':
            continue
        px, py = to_px(o['position_m'][0], o['position_m'][1], 600, 251.)
        c = kind_colour.get(o['asset'], (60, 60, 60))
        r = 3 if o['asset'] in big else (2 if o['asset'] in ('tree.urdf', 'tree_02.urdf') else 1)
        d.ellipse((px - r, py - r, px + r, py + r), fill=c)
    for i, s in enumerate(setups, 1):
        tx, ty, _ = s['target']['position_m']
        dx, dy, _ = s['start']['position_m']
        p_t = to_px(tx, ty, 600, 251.)
        p_d = to_px(dx, dy, 600, 251.)
        d.line((p_t, p_d), fill=(120, 120, 120))
        d.ellipse((p_d[0] - 3, p_d[1] - 3, p_d[0] + 3, p_d[1] + 3), fill=(60, 140, 230))
        d.ellipse((p_t[0] - 5, p_t[1] - 5, p_t[0] + 5, p_t[1] + 5), outline=(255, 80, 60), width=2)
        if i % 5 == 1:
            d.text((p_t[0] + 6, p_t[1] - 6), 'v{:d}'.format(i), fill=(255, 255, 255))
    top.save(PRE / 'level_3.png')

    cols, thumb = 6, 200
    rows = math.ceil(len(setups) / cols)
    sheet = Image.new('RGB', (cols * thumb, rows * thumb), (30, 32, 36))
    for i, s in enumerate(setups, 1):
        t = Image.new('RGB', (thumb, thumb), (30, 32, 36))
        td = ImageDraw.Draw(t)
        tx, ty, _ = s['target']['position_m']
        dx, dy, h = s['start']['position_m']
        # local window: include all props within `span` m of the setup midpoint
        cx, cy = (tx + dx) / 2, (ty + dy) / 2
        span = max(20., 2 * max(abs(tx - dx), abs(ty - dy)) + 12)
        for o in objects:
            if o['type'] != 'urdf':
                continue
            if max(abs(o['position_m'][0] - cx), abs(o['position_m'][1] - cy)) > span:
                continue
            px, py = to_px(o['position_m'][0] - cx, o['position_m'][1] - cy, thumb, span / 2)
            c = kind_colour.get(o['asset'], (60, 60, 60))
            r = 3 if o['asset'] in big else (2 if o['asset'] in ('tree.urdf', 'tree_02.urdf') else 1)
            td.ellipse((px - r, py - r, px + r, py + r), fill=c)
        p_t = to_px(tx - cx, ty - cy, thumb, span / 2)
        p_d = to_px(dx - cx, dy - cy, thumb, span / 2)
        td.line((p_t, p_d), fill=(140, 140, 140))
        td.ellipse((p_d[0] - 4, p_d[1] - 4, p_d[0] + 4, p_d[1] + 4), fill=(60, 140, 230))
        td.ellipse((p_t[0] - 7, p_t[1] - 7, p_t[0] + 7, p_t[1] + 7), outline=(255, 80, 60), width=2)
        td.text((4, 4), 'v{:d}  d~{:.0f}m  h={:.1f}m'.format(i, math.hypot(dx - tx, dy - ty), h),
                fill=(255, 255, 255))
        sheet.paste(t, (thumb * ((i - 1) % cols), thumb * ((i - 1) // cols)))
    sheet.save(PRE / 'level_3_targets.png')

if __name__ == '__main__':
    import sys
    if '--level-1' in sys.argv:
        spec = write_level_1()
        level_1_previews(spec)
        print('wrote level_1.json with {:d} setups and previews/level_1{{,_targets}}.png'.format(len(spec['setups'])))
        sys.exit(0)
    if '--level-2' in sys.argv:
        spec = write_level_2()
        level_2_previews(spec)
        print('wrote level_2.json with {:d} setups and previews/level_2{{,_targets}}.png'.format(len(spec['setups'])))
        sys.exit(0)
    if '--level-3' in sys.argv:
        spec = write_level_3()
        level_3_previews(spec)
        print('wrote level_3.json with {:d} setups and previews/level_3{{,_targets}}.png'.format(len(spec['setups'])))
        sys.exit(0)
    for a in asset_list:
        write_asset(a)
    scenes = [field(), forest(), village(), city()]
    for w in scenes:
        pick_targets(w, TARGETS_PER_SCENE)
        w.write()
    print('| Asset | Visible triangles | Collision hulls |\n| --- | ---: | ---: |')
    for a in asset_list:
        print(f'| `{a.name}` | {a.tris()} | {len(a.col)} |')
    stats = previews(scenes, asset_list)
    print('\n| Scene | Bodies |\n| --- | ---: |')
    for w in scenes:
        print(f'| `{w.name}` | {1 + len(json.loads((TERRAINS / f"{w.name}.json").read_text())["objects"])} |')
    print('\n| Version | Box | Context | Visible from above | Low views that see it |\n| --- | --- | --- | ---: | ---: |')
    for w in scenes:
        for v, (t, c) in enumerate(zip(w.targets, w.contexts), 1):
            above, seen = stats[f'{w.name}@v{v}']
            print(f'| `{w.name}@v{v}` | ({t[0]:.1f}, {t[1]:.1f}) | {c} | {above:.0%} | {seen}/8 |')
