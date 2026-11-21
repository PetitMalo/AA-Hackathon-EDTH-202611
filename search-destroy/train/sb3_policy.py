"""Pont entre handin.policy_net (PyTorch pur) et stable-baselines3.

Ce module doit rester importable sous ce nom : SB3 en garde la référence dans ses checkpoints.
"""

from pathlib import Path

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from handin.policy_net import FEATURES_DIM, FPVNet


class SB3Extractor(BaseFeaturesExtractor):
    """Enveloppe FPVNet comme extracteur de features d'une MultiInputPolicy."""

    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=FEATURES_DIM)
        self.net = FPVNet()

    def forward(self, observations):
        return self.net(observations)


def export_policy(model, path):
    """Écrit les poids de l'acteur au format attendu par handin.policy_net.load_actor."""
    pol = model.policy
    cpu = lambda module: {k: v.detach().cpu() for k, v in module.state_dict().items()}
    state = {"extractor": cpu(pol.features_extractor.net),
             "pi": cpu(pol.mlp_extractor.policy_net),
             "action": cpu(pol.action_net)}
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)                       # écriture atomique : jamais de fichier à moitié écrit
