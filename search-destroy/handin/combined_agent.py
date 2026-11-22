"""Agent combiné : l'explorateur cherche, l'agent d'approche fonce, et l'explorateur reprend si la cible est perdue.

    EXPLORE  --(cible détectée, Kalman cible initialisé, distance estimée <= HANDOFF_M)-->  APPROACH
    APPROACH --(cible invisible depuis plus de LOST_S secondes)----------------------->  EXPLORE

Un seul handin.features.Features, mis à jour à chaque pas quel que soit le mode : les cartes, l'estimateur
d'état et le Kalman cible sont déjà convergés au moment du relais (c'est l'hypothèse d'entraînement de
l'agent d'approche, voir train/approach/env.py). En repassant en EXPLORE, l'explorateur voit encore la
cible dans sa carte globale (canal blob) et sait où revenir.

Poids (variables d'environnement, sinon fichiers à côté de ce module) :
    FPV_EXPLORER   policy de l'explorateur (format handin.policy_net)       défaut handin/explorer.pt
    FPV_APPROACH   policy d'approche (format handin.approach_net)            défaut handin/approach.pt
    FPV_LOST_S     secondes sans détection avant de rendre la main (déf. 4)
    FPV_HANDOFF_M  distance estimée (m) en dessous de laquelle on passe la main (déf. inf : dès la détection)
Sans poids d'approche : explorateur seul. Sans explorateur : hover tant que la cible n'est pas vue.
Dépend uniquement de numpy, torch et des modules de handin/.
"""

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EXPLORER = Path(os.environ.get("FPV_EXPLORER") or HERE / "explorer.pt")
APPROACH = Path(os.environ.get("FPV_APPROACH") or HERE / "approach.pt")
LOST_STEPS = int(30 * float(os.environ.get("FPV_LOST_S", 4.)))
HANDOFF_M = float(os.environ.get("FPV_HANDOFF_M", "inf"))
EXPLORE, APPROACH_MODE = "explore", "approach"


class Agent:
    def __init__(self):
        self.feat = self.explorer = self.approach = self.torch = None
        self.errors = 0
        self.switches = []                       # (pas, nouveau mode), pour le debug
        self.reset()
        try:
            import torch
            from handin.approach_net import load_approach_actor
            from handin.features import Features
            from handin.policy_net import load_actor
            torch.set_num_threads(2)             # l'évaluation lance un processus par cœur
            self.torch = torch
            self.feat = Features()
            for name, path, loader in (("explorer", EXPLORER, load_actor), ("approach", APPROACH, load_approach_actor)):
                if path.exists():
                    setattr(self, name, loader(path))
                else:
                    print(f"[agent] {path} absent", file=sys.stderr)
            self.approach_local = self.approach is not None and self.approach.extractor.use_local
        except Exception as exc:
            print(f"[agent] initialisation impossible, hover : {exc!r}", file=sys.stderr)

    def reset(self):
        self.last = np.zeros(4, dtype=np.float32)
        self.mode, self.lost, self.t = EXPLORE, 0, 0
        self.switches = []
        try:
            if self.feat is not None:
                self.feat.reset()
        except Exception as exc:
            self._report(exc)

    # ------------------------------------------------------------ relais
    def _target_rel(self):
        """(dx, dy, dz) cible - drone en repère drone horizontal, ou None si le Kalman cible est vide."""
        from handin.features import BOX_HALF_H
        est, tf = self.feat.est, self.feat.target
        if tf.x is None:
            return None
        fwd, left = self.feat._heading()
        rel = tf.x - est.pos[:2]
        return float(rel @ fwd), float(rel @ left), float(BOX_HALF_H - est.pos[2])

    def _update_mode(self, detected, rel):
        if self.mode == EXPLORE:
            if detected and rel is not None and np.hypot(rel[0], rel[1]) <= HANDOFF_M and self.approach is not None:
                self.mode, self.lost = APPROACH_MODE, 0
                self.switches.append((self.t, self.mode))
        else:
            self.lost = 0 if detected else self.lost + 1
            if rel is None or self.lost > LOST_STEPS:
                self.mode = EXPLORE
                self.switches.append((self.t, self.mode))

    # ------------------------------------------------------------ observations des deux réseaux
    @staticmethod
    def _explorer_obs(x):
        return {k: np.nan_to_num(v, nan=0., posinf=1., neginf=0.).astype(np.float32) for k, v in x.items()}

    def _approach_obs(self, x, rel):
        """Identique à train.approach.env.ApproachEnv._obs (vec + position attendue de la cible)."""
        dx, dy, dz = rel
        extra = np.array([np.clip(dz / 50, -2, 2), np.arctan2(dy, dx) / np.pi,
                          np.arctan2(dz, np.hypot(dx, dy)) / (np.pi / 2)], np.float32)
        vec = np.concatenate([x["vec"], extra]).astype(np.float32)
        depth = np.clip(np.nan_to_num(x["depth"], nan=1., posinf=1., neginf=0.), 0., 1.)
        out = {"depth": depth.astype(np.float32), "vec": np.nan_to_num(vec, nan=0., posinf=3., neginf=-3.)}
        if self.approach_local:
            out["local"] = np.clip(np.nan_to_num(x["local"], nan=0.), 0., 1.).astype(np.float32)
        return out

    def _run(self, net, obs):
        with self.torch.no_grad():
            batch = {k: self.torch.from_numpy(v).unsqueeze(0) for k, v in obs.items()}
            return net(batch)[0].numpy()

    # ------------------------------------------------------------ API évaluateur
    def act(self, observation):
        action = np.zeros(4, dtype=np.float32)
        try:
            if self.feat is not None:
                x = self.feat(observation, self.last)
                rel = self._target_rel()
                self._update_mode(bool(observation["detected"]), rel)
                if self.mode == APPROACH_MODE:
                    out = self._run(self.approach, self._approach_obs(x, rel))
                elif self.explorer is not None:
                    out = self._run(self.explorer, self._explorer_obs(x))
                else:
                    out = action
                action = np.clip(np.nan_to_num(out, nan=0., posinf=1., neginf=-1.), -1., 1.).astype(np.float32)
        except Exception as exc:                 # un bug ponctuel ne doit pas coûter le vol entier
            self._report(exc)
            action = np.zeros(4, dtype=np.float32)
        self.last = action
        self.t += 1
        return action

    def _report(self, exc):
        self.errors += 1
        if self.errors <= 3:
            print(f"[agent] erreur {self.errors} : {exc!r}", file=sys.stderr)
