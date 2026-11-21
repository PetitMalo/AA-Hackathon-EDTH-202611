"""Environnement d'entraînement de l'explorateur : chercher la cible, sans la toucher.

Mêmes features (handin.features), même départ que l'évaluation officielle. La reward de couverture
est calculée sur les cartes que la policy observe elle-même (canaux glob `sol_vu` et `sol_vu_loin`) :

    couverture d'une cellule de 12,5 m  c = max(seen_frac, alpha * (1 - exp(-far / far_k)))
    couverture totale                   C = sum_c c * 156,25 m²   sur la zone |x|, |y| <= zone_half_m

    reward = w_cov * dC / 1000 - time - smooth * |a - a_prev|^2
             + detect_bonus à la 1re détection, + pénalités de fin (collision, sortie, timeout)

Succès : première détection de la cible (end_on_detect) et/ou couverture >= coverage_goal (si > 0).
La vérité terrain ne sert qu'à détecter collisions et sorties d'arène.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from handin.features import COARSE_HALF, COARSE_RES, FINE_RES, SHAPES, Features
from train.env_wrapper import ARENA, FINAL_STAGE, FPVTrainEnv

CELL_M2 = COARSE_RES ** 2                        # 156,25 m² par cellule grossière
FINE_PER_COARSE = (COARSE_RES / FINE_RES) ** 2   # cellules fines de 1 m par cellule grossière
TERMINAL = ("detected", "success", "collision", "out_of_arena", "coverage_done")


class ExploreEnv(FPVTrainEnv):
    def __init__(self, cfg, terrains, seed=0):
        gym.Env.__init__(self)
        self.ex = cfg["explore"]
        self.terrains = list(terrains)
        self.stage = self.stage_used = FINAL_STAGE
        self.rng = np.random.default_rng(seed)
        self._targets = {}
        from fpv_sim.envs.approach_aviary import ApproachAviary
        self.sim = ApproachAviary(continuous=True, terrain=self.terrains[0])
        self.feat = Features()
        self.observation_space = spaces.Dict({k: spaces.Box(-np.inf, np.inf, shape=s, dtype=np.float32)
                                              for k, s in SHAPES.items()})
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype=np.float32)
        h = int(round(self.ex["zone_half_m"] / COARSE_RES))
        self.zone = slice(COARSE_HALF - h, COARSE_HALF + h)
        self.zone_cells = (2 * h) ** 2
        self.t = 0

    def set_stage(self, stage):                  # compatibilité avec env_method("set_stage")
        return self.stage

    # ------------------------------------------------------------ couverture
    def _coverage(self):
        """(m² couverts, fraction de la zone), à partir des cartes de la policy."""
        z = self.zone
        seen = np.minimum(self.feat.coarse_seen[z, z] / FINE_PER_COARSE, 1.)
        far = 1. - np.exp(-self.feat.coarse_far[z, z] / self.ex["far_k"])
        c = np.maximum(seen, self.ex["alpha"] * far)
        s = float(c.sum())
        return s * CELL_M2, s / self.zone_cells

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.terrain = self.terrains[int(self.rng.integers(len(self.terrains)))]
        self.target_xy = self._target_xy(self.terrain)
        self.sim.TERRAIN = self.terrain
        for _ in range(5):                       # évite un départ où la cible est déjà détectée
            obs = self._place_drone()
            if not obs["detected"]:
                break
        self.timeout_steps = int(30 * self.ex["timeout_s"])
        self.feat.reset()
        x = self.feat(obs, None)
        self.t = 0
        self.last_a = np.zeros(4)
        self.ever_seen = bool(obs["detected"])
        self.t_detect = 0. if self.ever_seen else -1.
        self.cov_m2, self.cov_frac = self._coverage()
        return x, {"terrain": self.terrain, "stage": FINAL_STAGE}

    def step(self, action):
        ex = self.ex
        a = np.clip(np.asarray(action, dtype=float), -1., 1.)
        obs, _, _, _, info = self.sim.step(a)
        self.t += 1
        x = self.feat(obs, a)

        cov_m2, cov_frac = self._coverage()
        r = ex["w_cov"] * (cov_m2 - self.cov_m2) / 1000.
        r -= ex["time"] + ex["smooth"] * float(np.sum((a - self.last_a) ** 2))
        self.cov_m2, self.cov_frac, self.last_a = cov_m2, cov_frac, a

        det = bool(obs["detected"])
        first = det and not self.ever_seen
        if first:
            self.ever_seen, self.t_detect = True, self.t / 30.
            r += ex["detect_bonus"]

        outcome = None
        if info["collision"]:
            outcome = "collision"
        elif np.abs(self.sim.pos[0, :2]).max() >= ARENA:
            outcome = "out_of_arena"
        elif info["is_success"]:
            outcome = "success"
        elif ex["end_on_detect"] and first:
            outcome = "detected"
        elif ex["coverage_goal"] > 0 and cov_frac >= ex["coverage_goal"]:
            outcome = "coverage_done"
        elif self.t >= self.timeout_steps:
            outcome = "timeout"
        if outcome:
            r += ex.get(outcome, 0.)
        out = {}
        if outcome:
            out = {"outcome": outcome, "stage": FINAL_STAGE, "terrain": self.terrain, "ep_len": self.t,
                   "coverage": cov_frac, "coverage_m2": cov_m2, "detected": self.ever_seen,
                   "t_detect": self.t_detect}
        return x, float(r), outcome in TERMINAL, outcome == "timeout", out


def make_explore_env_fn(cfg, terrains, rank):
    """Fabrique pour SubprocVecEnv."""
    def _init():
        return ExploreEnv(cfg, terrains, seed=cfg["run"]["seed"] * 1000 + rank)
    return _init
