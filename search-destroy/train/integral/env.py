"""Environnement de l'agent intégral : ApproachAviary + handin.features + curriculum + reward.

Le wrapper utilise la vérité terrain (distance à la boîte, position de la cible) UNIQUEMENT pour
placer le drone et calculer la reward. La policy ne reçoit que la sortie de handin.features.
"""

import numpy as np
from gymnasium import spaces

from fpv_sim.envs.approach_aviary import ApproachAviary
from handin.features import Features, SHAPES
from train.base_env import ARENA, FINAL_STAGE, BaseFPVEnv


class FPVTrainEnv(BaseFPVEnv):
    def __init__(self, cfg, terrains, stage=0, seed=0):
        super().__init__()
        self.cur, self.rw = cfg["curriculum"], cfg["reward"]
        self.terrains = list(terrains)
        self.stage = int(stage)
        self.rng = np.random.default_rng(seed)
        self._targets = {}
        self.sim = ApproachAviary(continuous=True, terrain=self.terrains[0])
        self.feat = Features()
        self.observation_space = spaces.Dict({k: spaces.Box(-np.inf, np.inf, shape=s, dtype=np.float32)
                                              for k, s in SHAPES.items()})
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype=np.float32)
        self.stage_used, self.t = self.stage, 0

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.terrain = self.terrains[int(self.rng.integers(len(self.terrains)))]
        self.target_xy = self._target_xy(self.terrain)
        self.stage_used = self._pick_stage()
        self.sim.TERRAIN = self.terrain
        obs = self._place_drone()
        timeout_s = self.cur["final_timeout_s"] if self.stage_used >= FINAL_STAGE \
            else self.cur["stages"][self.stage_used][3]
        self.timeout_steps = int(30 * timeout_s)
        self.feat.reset()
        self.t = 0
        self.last_a = np.zeros(4)
        self.ever_seen = bool(obs["detected"])
        self.d_prev = self.sim._distance()
        self.seen_cells = 0.
        return self.feat(obs, None), {"terrain": self.terrain, "stage": self.stage_used}

    def step(self, action):
        rw = self.rw
        a = np.clip(np.asarray(action, dtype=float), -1., 1.)
        obs, _, _, _, info = self.sim.step(a)
        self.t += 1
        x = self.feat(obs, a)

        det = bool(obs["detected"])
        self.ever_seen |= det
        d = info["distance_m"]
        unseen = self.stage_used >= 2 and not self.ever_seen
        r = rw["progress"] * (rw["progress_unseen_factor"] if unseen else 1.) * (self.d_prev - d)
        r += rw["detect"] * det - rw["time"] - rw["smooth"] * float(np.sum((a - self.last_a) ** 2))
        cells = float(self.feat.coarse_seen.sum())
        if self.stage_used >= rw["coverage_from_stage"]:
            r += rw["coverage"] * (cells - self.seen_cells) / 1000.
        self.seen_cells, self.d_prev, self.last_a = cells, d, a

        outcome = None
        if info["is_success"]:
            outcome = "success"
        elif info["collision"]:
            outcome = "collision"
        elif np.abs(self.sim.pos[0, :2]).max() >= ARENA:
            outcome = "out_of_arena"
        elif self.t >= self.timeout_steps:
            outcome = "timeout"
        if outcome:
            r += rw[outcome]
        out = {"outcome": outcome, "stage": self.stage_used, "terrain": self.terrain, "ep_len": self.t} if outcome else {}
        return x, float(r), outcome in ("success", "collision", "out_of_arena"), outcome == "timeout", out


def make_env_fn(cfg, terrains, rank, stage):
    """Fabrique pour SubprocVecEnv : chaque worker crée son propre simulateur PyBullet."""
    def _init():
        return FPVTrainEnv(cfg, terrains, stage=stage, seed=cfg["run"]["seed"] * 1000 + rank)
    return _init
