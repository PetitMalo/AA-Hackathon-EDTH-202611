"""Bloc 2 : réseau de la policy, en PyTorch pur (aucune dépendance à stable-baselines3).

    FPVNet     encodeur : dict d'observations (handin.features.SHAPES, avec dimension batch) -> (B, 256)
    ActorNet   encodeur + 2x256 (tanh) + sortie 4 sticks (moyenne, non bornée : l'appelant coupe à [-1, 1])

Entraînement : train/train.py enveloppe FPVNet comme extracteur de features SB3 ; les couches
`pi` et `action` d'ActorNet ont exactement la structure de mlp_extractor.policy_net et action_net
de SB3, ce qui permet d'exporter les poids tels quels (voir export_keys).
Déploiement : handin/agent.py appelle load_actor("policy.pt").
"""

import torch
from torch import nn

from handin.features import SHAPES

FEATURES_DIM = 256
PI_HIDDEN = (256, 256)
N_ACTIONS = 4


def _conv(cin, cout, k, stride):
    return [nn.Conv2d(cin, cout, k, stride, k // 2), nn.ReLU()]


class FPVNet(nn.Module):
    """Un petit CNN par carte, un MLP pour le vecteur, puis fusion."""

    def __init__(self):
        super().__init__()
        self.depth = nn.Sequential(*_conv(2, 16, 5, 2), *_conv(16, 32, 3, 2), *_conv(32, 32, 3, 2), nn.Flatten())
        self.local = nn.Sequential(*_conv(5, 16, 5, 2), *_conv(16, 32, 3, 2), *_conv(32, 64, 3, 2),
                                   *_conv(64, 64, 3, 2), nn.Flatten())
        self.glob = nn.Sequential(*_conv(4, 16, 3, 1), *_conv(16, 32, 3, 2), *_conv(32, 32, 3, 2),
                                  *_conv(32, 32, 3, 2), nn.Flatten())
        self.vec = nn.Sequential(nn.Linear(SHAPES["vec"][0], 128), nn.ReLU())
        with torch.no_grad():                       # taille d'entrée de la fusion, déduite des formes
            n = sum(self._branch(k, torch.zeros(1, *SHAPES[k])).shape[1] for k in SHAPES)
        self.fuse = nn.Sequential(nn.Linear(n, FEATURES_DIM), nn.ReLU())

    def _branch(self, key, x):
        return getattr(self, key)(x)

    def forward(self, obs):
        return self.fuse(torch.cat([self._branch(k, obs[k]) for k in ("depth", "local", "glob", "vec")], dim=1))


def make_pi():
    layers, last = [], FEATURES_DIM
    for h in PI_HIDDEN:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
    return nn.Sequential(*layers)


class ActorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = FPVNet()
        self.pi = make_pi()
        self.action = nn.Linear(PI_HIDDEN[-1], N_ACTIONS)

    def forward(self, obs):
        return self.action(self.pi(self.extractor(obs)))


def export_keys():
    """Clés du fichier policy.pt : state_dict de chaque sous-réseau."""
    return ("extractor", "pi", "action")


def load_actor(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    net = ActorNet()
    for key in export_keys():
        getattr(net, key).load_state_dict(state[key])
    return net.eval()
