"""Socle commun aux environnements des trois agents (integral, explore, approach) : terrain, départ, étape.

La vérité terrain sert UNIQUEMENT à placer le drone et calculer la reward.
Chaque sous-classe définit __init__, reset et step.
"""

import json

import gymnasium as gym
import numpy as np
import pybullet as p

from fpv_sim.utils.terrain import TERRAINS_DIR

ARENA = 350.
CLEAR_M = 2.            # distance minimale du départ à tout objet (comme evaluate.random_start)
FINAL_STAGE = 3


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class BaseFPVEnv(gym.Env):
    metadata = {"render_modes": []}

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

    def close(self):
        self.sim.close()
