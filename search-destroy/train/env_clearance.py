"""FPVTrainEnv + pénalité de proximité des obstacles (vérité terrain, entraînement uniquement).

Constat (évaluations de validation à 10M pas) : 65-80 % des crashs ont lieu à moins de 15 m de la boîte,
toujours contre un objet (maison, arbre, abri), jamais au sol. Le réseau ne « voit » un obstacle qu'au choc.
Ici, à chaque step, si un objet du décor (hors boîte cible) est à moins de `reward.clearance_m` du drone :
    r -= reward.clearance * (1 - d / clearance_m)        d = distance au plus proche
La policy ne reçoit toujours que la sortie de handin.features : la pénalité ne change que la reward.

Réglages (section [reward], via --set) :
    clearance    = 0.1     pénalité max par step (au contact)
    clearance_m  = 1.5     rayon de la zone pénalisée (m)
"""

import numpy as np
import pybullet as p

from train.env_wrapper import FPVTrainEnv


class ClearanceEnv(FPVTrainEnv):

    def reset(self, seed=None, options=None):
        x, info = super().reset(seed=seed, options=options)
        self.obstacles = set(self.sim.TERRAIN_IDS[1:])          # tout sauf la boîte cible
        self.near_steps = 0
        return x, info

    def _clearance(self, radius):
        """Distance du drone à l'objet du décor le plus proche, si < radius ; sinon None."""
        sim = self.sim
        pos = sim.pos[0]
        cand = p.getOverlappingObjects((pos - radius).tolist(), (pos + radius).tolist(), physicsClientId=sim.CLIENT)
        if not cand:
            return None
        drone, best = sim.DRONE_IDS[0], None
        for body in {b for b, _ in cand if b in self.obstacles}:
            for pt in p.getClosestPoints(drone, body, radius, physicsClientId=sim.CLIENT):
                d = max(pt[8], 0.)
                best = d if best is None else min(best, d)
        return best

    def step(self, action):
        x, r, terminated, truncated, info = super().step(action)
        k = float(self.rw.get("clearance", 0.1))
        radius = float(self.rw.get("clearance_m", 1.5))
        if k > 0:
            d = self._clearance(radius)
            if d is not None:
                r -= k * (1. - d / radius)
                self.near_steps += 1
        if info.get("outcome"):
            info["near_frac"] = self.near_steps / max(self.t, 1)
        return x, float(r), terminated, truncated, info


def make_env_fn(cfg, terrains, rank, stage):
    def _init():
        return ClearanceEnv(cfg, terrains, stage=stage, seed=cfg["run"]["seed"] * 1000 + rank)
    return _init
