"""ApproachEnv + pénalité de vrille (le drone faisait 500 à 600° de lacet par approche).

[reward] spin            coût par degré de lacet réel au-delà de spin_free_dps
         spin_free_dps   vitesse de lacet gratuite (°/s) : les virages d'évitement ne coûtent rien
         spin_max        plafond du cumul par épisode, pour que se crasher ne soit jamais rentable
Le lacet est la vitesse angulaire verticale de la simu : en piqué, le stick (repère corps) la sous-estime de moitié.
"""

import numpy as np

from train.approach.env import ApproachEnv

DT = 1 / 30


class StableApproachEnv(ApproachEnv):
    def reset(self, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self.spin_paid = self.yaw_abs = self.yaw_cum = self.yaw_lo = self.yaw_hi = 0.
        return out

    def step(self, action):
        x, r, term, trunc, info = super().step(action)
        rate = float(np.degrees(self.sim.ang_v[0][2]))
        if not (term or trunc) and np.isfinite(rate):           # le rebond du choc final n'est pas une vrille
            self.yaw_abs += abs(rate) * DT
            self.yaw_cum += rate * DT
            self.yaw_lo, self.yaw_hi = min(self.yaw_lo, self.yaw_cum), max(self.yaw_hi, self.yaw_cum)
            rw = self.rw
            pen = rw["spin"] * max(0., abs(rate) - rw["spin_free_dps"]) * DT
            pen = min(pen, max(0., rw["spin_max"] - self.spin_paid))
            self.spin_paid += pen
            r -= pen
        if term or trunc:
            info.update(yaw_abs_deg=self.yaw_abs, spun=float(self.yaw_hi - self.yaw_lo >= 360.),
                        spin_penalty=self.spin_paid)
        return x, float(r), term, trunc, info


def make_stable_env_fn(cfg, terrains, rank, stage):
    def _init():
        return StableApproachEnv(cfg, terrains, stage=stage, seed=cfg["run"]["seed"] * 1000 + rank)
    return _init
