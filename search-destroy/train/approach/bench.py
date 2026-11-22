"""Scènes « setups » (cartes level_N du camarade) : départs fixes + conversion au format fpv_sim.

Format d'entrée (fpv_sim/assets/terrains/level_1.json) :
    {"name", "objects", "setups": [{"target": {position_m, rpy_rad, context}, "start": {position_m, rpy_rad}}, ...]}

fpv_sim.utils.terrain.load_terrain attend une clé "targets" : on écrit à côté `<scene>_targets.json`
(targets = les cibles des setups, mêmes objets), régénéré si la source est plus récente. Le fichier
source n'est jamais modifié. Le setup i correspond à la pose de cible `<scene>_targets@v<i+1>`.
"""

import json

import numpy as np

from fpv_sim.utils.terrain import TERRAINS_DIR


def has_setups(scene):
    path = TERRAINS_DIR / f"{scene}.json"
    return path.is_file() and "setups" in json.loads(path.read_text())


def ensure_scene(scene):
    """Nom de la scène chargeable par load_terrain : `scene` lui-même s'il a une clé "targets"
    (cas de level_1.json, complété à la main), sinon `<scene>_targets` (généré)."""
    path = TERRAINS_DIR / f"{scene}.json"
    if not has_setups(scene) or "targets" in json.loads(path.read_text()):
        return scene
    src, dst = TERRAINS_DIR / f"{scene}.json", TERRAINS_DIR / f"{scene}_targets.json"
    if not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
        spec = json.loads(src.read_text())
        out = {"name": f"{scene}_targets", "targets": [s["target"] for s in spec["setups"]],
               "objects": spec["objects"]}
        dst.write_text(json.dumps(out, indent=1) + "\n")
    return f"{scene}_targets"


def load_setups(scene, stages):
    """Départs fixes du benchmark, chacun rattaché à l'étape du curriculum de distance la plus proche."""
    name = ensure_scene(scene)
    spec = json.loads((TERRAINS_DIR / f"{scene}.json").read_text())
    d_means = np.array([s[0] for s in stages], float)
    out = []
    for i, s in enumerate(spec["setups"]):
        t = np.array(s["target"]["position_m"], float)
        xyz = [float(v) for v in s["start"]["position_m"]]
        rpy = [float(v) for v in s["start"]["rpy_rad"]]
        dv = t[:2] - np.array(xyz[:2])
        d = float(np.hypot(*dv))
        psi = float((np.degrees(rpy[2] - np.arctan2(dv[1], dv[0])) + 180) % 360 - 180)
        out.append({"index": i, "terrain": f"{name}@v{i + 1}", "xyz": xyz, "rpy": rpy, "d": d, "psi_deg": psi,
                    "stage": int(np.argmin(np.abs(np.log(d_means / max(d, .1))))),
                    "context": s["target"].get("context", "")})
    return out
