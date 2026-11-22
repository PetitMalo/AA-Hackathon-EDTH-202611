"""Réseau de l'agent d'approche, en PyTorch pur (aucune dépendance à stable-baselines3).

    ApproachNet    encodeur : {"depth": (B, 2, 32, 56), "vec": (B, vec_dim)[, "local": (B, 5, 64, 64)]} -> (B, 256)
    ApproachActor  encodeur + 2x256 (tanh) + 4 sticks (moyenne non bornée : l'appelant coupe à [-1, 1])

Même CNN depth et même MLP vec que handin.policy_net.FPVNet ; la branche "local" (carte d'occupation
égocentrée, même CNN que FPVNet.local) est optionnelle. Ordre de concaténation : depth, vec, local, pour
qu'un réseau sans local se prolonge en réseau avec local (colonnes de fusion ajoutées à la fin, à zéro).
L'export (train.sb3_policy.export_policy) écrit les clés extractor / pi / action, chargées par load_approach_actor.
"""

import torch
from torch import nn

from handin.features import SHAPES, VEC_DIM
from handin.policy_net import FEATURES_DIM, N_ACTIONS, PI_HIDDEN, _conv, make_pi

APPROACH_VEC_DIM = VEC_DIM + 3          # + position attendue (dz_rel, bearing, elev), voir train.approach.env


class ApproachNet(nn.Module):
    def __init__(self, vec_dim=APPROACH_VEC_DIM, use_local=False):
        super().__init__()
        self.use_local = use_local
        self.depth = nn.Sequential(*_conv(2, 16, 5, 2), *_conv(16, 32, 3, 2), *_conv(32, 32, 3, 2), nn.Flatten())
        self.vec = nn.Sequential(nn.Linear(vec_dim, 128), nn.ReLU())
        with torch.no_grad():
            self.n_base = self.depth(torch.zeros(1, *SHAPES["depth"])).shape[1] + 128
        n = self.n_base
        if use_local:
            self.local = nn.Sequential(*_conv(5, 16, 5, 2), *_conv(16, 32, 3, 2), *_conv(32, 64, 3, 2),
                                       *_conv(64, 64, 3, 2), nn.Flatten())
            with torch.no_grad():
                n += self.local(torch.zeros(1, *SHAPES["local"])).shape[1]
        self.fuse = nn.Sequential(nn.Linear(n, FEATURES_DIM), nn.ReLU())

    def forward(self, obs):
        parts = [self.depth(obs["depth"]), self.vec(obs["vec"])]
        if self.use_local:
            parts.append(self.local(obs["local"]))
        return self.fuse(torch.cat(parts, dim=1))


class ApproachActor(nn.Module):
    def __init__(self, vec_dim=APPROACH_VEC_DIM, use_local=False):
        super().__init__()
        self.extractor = ApproachNet(vec_dim, use_local)
        self.pi = make_pi()
        self.action = nn.Linear(PI_HIDDEN[-1], N_ACTIONS)

    def forward(self, obs):
        return self.action(self.pi(self.extractor(obs)))


def load_approach_actor(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    ext = state["extractor"]
    net = ApproachActor(ext["vec.0.weight"].shape[1], use_local=any(k.startswith("local.") for k in ext))
    for key in ("extractor", "pi", "action"):
        getattr(net, key).load_state_dict(state[key])
    return net.eval()
