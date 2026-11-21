"""Config de l'agent d'approche : séparé de train/config.py pour ne pas toucher aux runs existants.

Le chargement TOML et les surcharges --set restent ceux de train.config (load, save_copy) ;
seul le découpage entraînement / holdout change : le nombre de poses est lu dans le JSON de la
scène (level1, level2, ...) et le holdout est donné par scène dans [terrains.holdout].
"""

import json

from fpv_sim.utils.terrain import TERRAINS_DIR
from train.approach.bench import ensure_scene
from train.config import load, save_copy  # noqa: F401  (réexportés pour les scripts d'approche)


def scene_versions(scene):
    return len(json.loads((TERRAINS_DIR / f"{scene}.json").read_text())["targets"])


def terrain_split(cfg):
    """(train, holdout) : listes de "scène@vN". Les versions holdout ne sont jamais dans train."""
    t = cfg["terrains"]
    per_scene = t.get("holdout", {})
    train, holdout = [], []
    for src in t["scenes"]:
        scene = ensure_scene(src)                  # level_1 (setups) -> level_1_targets, généré au besoin
        n = scene_versions(scene)
        held = set(per_scene.get(src, []))
        bad = [v for v in held if not 1 <= v <= n]
        if bad:
            raise ValueError(f"{scene} n'a que {n} versions, holdout invalide : {bad}")
        for v in range(1, n + 1):
            (holdout if v in held else train).append(f"{scene}@v{v}")
    if not train:
        raise ValueError("aucun terrain d'entraînement")
    return train, holdout
