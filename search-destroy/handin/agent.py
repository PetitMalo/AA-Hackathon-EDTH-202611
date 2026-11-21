"""Agent soumis : features (handin.features) -> réseau (handin.policy_net) -> sticks RC.

Dépend uniquement de numpy, torch et des modules de handin/. Les poids sont lus dans
handin/policy.pt ; s'ils manquent, ou en cas d'erreur, l'agent renvoie un hover valide.
"""

import os
import sys
from pathlib import Path

import numpy as np

# FPV_WEIGHTS permet d'évaluer un autre checkpoint sans copier de fichier (utilisé par train/eval_holdout.py)
WEIGHTS = Path(os.environ.get("FPV_WEIGHTS") or Path(__file__).with_name("policy.pt"))


class Agent:
    def __init__(self):
        self.feat = self.net = self.torch = None
        self.errors = 0
        self.last = np.zeros(4, dtype=np.float32)
        try:
            import torch
            from handin.features import Features
            from handin.policy_net import load_actor
            torch.set_num_threads(2)             # l'évaluation lance un processus par cœur
            self.torch = torch
            self.feat = Features()
            if WEIGHTS.exists():
                self.net = load_actor(WEIGHTS)
            else:
                print(f"[agent] {WEIGHTS} absent : hover", file=sys.stderr)
        except Exception as exc:
            print(f"[agent] initialisation impossible, hover : {exc!r}", file=sys.stderr)

    def reset(self):
        self.last = np.zeros(4, dtype=np.float32)
        try:
            if self.feat is not None:
                self.feat.reset()
        except Exception as exc:
            self._report(exc)

    def act(self, observation):
        action = np.zeros(4, dtype=np.float32)
        try:
            if self.net is not None and self.feat is not None:
                x = self.feat(observation, self.last)
                with self.torch.no_grad():
                    batch = {k: self.torch.from_numpy(v).unsqueeze(0) for k, v in x.items()}
                    out = self.net(batch)[0].numpy()
                action = np.clip(np.nan_to_num(out, nan=0., posinf=1., neginf=-1.), -1., 1.).astype(np.float32)
        except Exception as exc:                 # un bug ponctuel ne doit pas coûter le vol entier
            self._report(exc)
            action = np.zeros(4, dtype=np.float32)
        self.last = action
        return action

    def _report(self, exc):
        self.errors += 1
        if self.errors <= 3:
            print(f"[agent] erreur {self.errors} : {exc!r}", file=sys.stderr)
