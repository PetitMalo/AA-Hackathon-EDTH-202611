"""Chargement de train/integral/config.toml avec surcharges "section.clé=valeur" (valeur au format TOML)."""

import shutil
import tomllib
from pathlib import Path

DEFAULT = Path(__file__).parent / "integral" / "config.toml"
SCENE_VERSIONS = {"field": 18, "forest": 40, "village": 40, "city": 40}


def load(path=None, overrides=()):
    cfg = tomllib.loads(Path(path or DEFAULT).read_text())
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"surcharge invalide '{item}', attendu section.clé=valeur")
        try:
            value = tomllib.loads(f"v = {raw}")["v"]
        except tomllib.TOMLDecodeError:
            value = raw                                   # texte brut sans guillemets
        *parents, leaf = key.split(".")
        node = cfg
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = value
    return cfg


def save_copy(out_dir, path=None, overrides=()):
    """Copie la config d'origine et les surcharges dans le dossier du run (reproductibilité)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(path or DEFAULT), out_dir / "config.toml")
    (out_dir / "overrides.txt").write_text("\n".join(overrides) + "\n")


def terrain_split(cfg):
    """(train, holdout) : listes de "scène@vN". Les versions holdout ne sont jamais dans train."""
    t = cfg["terrains"]
    only = set(t.get("train_versions") or [])     # optionnel : n'entraîner que sur ces versions (ex. [1, 2, 3, 4, 5, 6])
    train, holdout = [], []
    for scene in t["scenes"]:
        n = SCENE_VERSIONS[scene]
        held = set(t["holdout_field"] if scene == "field" else t["holdout_others"])
        bad = [v for v in held if not 1 <= v <= n]
        if bad:
            raise ValueError(f"{scene} n'a que {n} versions, holdout invalide : {bad}")
        for v in range(1, n + 1):
            if v in held:
                holdout.append(f"{scene}@v{v}")
            elif not only or v in only:
                train.append(f"{scene}@v{v}")
    if not train:
        raise ValueError("aucun terrain d'entraînement")
    return train, holdout
