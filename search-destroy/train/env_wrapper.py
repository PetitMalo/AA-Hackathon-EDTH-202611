"""Environnement d'entraînement : ApproachAviary + handin.features + curriculum + reward.

Le wrapper utilise la vérité terrain (distance à la boîte, position de la cible) UNIQUEMENT pour
placer le drone et calculer la reward. La policy ne reçoit que la sortie de handin.features.
"""

import json

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from fpv_sim.envs.approach_aviary import ApproachAviary
from fpv_sim.utils.terrain import TERRAINS_DIR
from handin.features import Features, SHAPES

ARENA = 350.
CLEAR_M = 2.            # distance minimale du départ à tout objet (comme evaluate.random_start)
FINAL_STAGE = 3


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class FPVTrainEnv(gym.Env):
    metadata = {"render_modes": []}

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

    # ------------------------------------------------------------ curriculum (appelé par env_method)
    def set_stage(self, stage):
        self.stage = int(stage)
        return self.stage

    def _pick_stage(self):
        if self.stage > 0 and self.rng.random() < self.cur["replay_previous"]:
            return int(self.rng.integers(self.stage))
        return self.stage

    # ------------------------------------------------------------ départ
    def _target_xy(self, terrain):
        name, v = terrain.split("@v")
        if name not in self._targets:
            self._targets[name] = json.loads((TERRAINS_DIR / f"{name}.json").read_text())["targets"]
        return np.array(self._targets[name][int(v) - 1]["position_m"][:2], dtype=float)

    def _sample_start(self, stage):
        rng, target = self.rng, self.target_xy
        if stage >= FINAL_STAGE:                       # départ officiel de evaluate.py
            r, a = rng.uniform(225, 275), rng.uniform(-np.pi, np.pi)
            xy = r * np.array([np.cos(a), np.sin(a)])
            z = rng.uniform(5, 25)
        else:                                          # près de la cible, cible dans le cône avant
            lo, hi, cone, _ = self.cur["stages"][stage]
            cone = np.radians(cone)
            for _ in range(100):
                r, a = rng.uniform(lo, hi), rng.uniform(-np.pi, np.pi)
                xy = target + r * np.array([np.cos(a), np.sin(a)])
                if np.abs(xy).max() > ARENA - 20:
                    continue
                to_center = np.arctan2(-xy[1], -xy[0])
                to_target = np.arctan2(target[1] - xy[1], target[0] - xy[0])
                if abs(_wrap(to_target - to_center)) <= cone:
                    break
            z = rng.uniform(5, 20)
        yaw = np.arctan2(-xy[1], -xy[0]) + rng.uniform(-np.radians(20), np.radians(20))
        return [float(xy[0]), float(xy[1]), float(z)], float(yaw), float(rng.uniform(0, 10))

    def _place_drone(self):
        """Charge la carte une fois, puis déplace le drone jusqu'à un départ dégagé."""
        sim = self.sim
        sim.INIT_XYZS, sim.INIT_RPYS = np.array([[0., 0., 3.]]), np.zeros((1, 3))
        sim.reset(seed=int(self.rng.integers(2**31)))
        drone = sim.DRONE_IDS[0]
        for _ in range(200):
            xyz, yaw, speed = self._sample_start(self.stage_used)
            p.resetBasePositionAndOrientation(drone, xyz, p.getQuaternionFromEuler([0, 0, yaw]),
                                              physicsClientId=sim.CLIENT)
            if not any(p.getClosestPoints(drone, b, CLEAR_M, physicsClientId=sim.CLIENT) for b in sim.TERRAIN_IDS):
                break
        else:                                          # aucun départ dégagé : point de départ sûr du README
            xyz, yaw, speed = [0., 0., 3.], 0., 0.
            p.resetBasePositionAndOrientation(drone, xyz, [0, 0, 0, 1], physicsClientId=sim.CLIENT)
        p.resetBaseVelocity(drone, [speed * np.cos(yaw), speed * np.sin(yaw), 0], [0, 0, 0],
                            physicsClientId=sim.CLIENT)
        sim._updateAndStoreKinematicInformation()
        sim.vio.reset(sim._getDroneStateVector(0))
        sim.imu.reset()
        return sim._sense()

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

    def close(self):
        self.sim.close()


def make_env_fn(cfg, terrains, rank, stage):
    """Fabrique pour SubprocVecEnv : chaque worker crée son propre simulateur PyBullet."""
    def _init():
        return FPVTrainEnv(cfg, terrains, stage=stage, seed=cfg["run"]["seed"] * 1000 + rank)
    return _init
