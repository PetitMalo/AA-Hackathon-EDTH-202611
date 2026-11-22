"""Pont entre handin.approach_net et stable-baselines3 (agent d'approche).

Ce module doit rester importable sous ce nom : SB3 en garde la référence dans ses checkpoints.
L'export de policy.pt réutilise train.sb3_policy.export_policy (mêmes clés extractor / pi / action).
"""

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from handin.approach_net import ApproachNet
from handin.policy_net import FEATURES_DIM
from train.sb3_policy import export_policy  # noqa: F401


class SB3ApproachExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict):
        super().__init__(observation_space, features_dim=FEATURES_DIM)
        self.net = ApproachNet(observation_space["vec"].shape[0], use_local="local" in observation_space.spaces)

    def forward(self, observations):
        return self.net(observations)
