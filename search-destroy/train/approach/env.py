"""Environnement d'approche : la cible a déjà été vue, il faut foncer dessus.

Départ (vérité terrain, uniquement pour placer le drone et calculer la reward) :
    distance horizontale au centre de la boîte  d ~ N(d_mean, (d_sig * d_mean)^2), coupée à ±2 sigma
    altitude                                    z ~ U(z_inf, z_sup)
    cap                                         direction de la cible + psi, psi ~ U(-psi_max, psi_max)
    vitesse                                     U(0, v_max) dans l'axe du cap
    roulis                                      U(-roll_max, roll_max)
    tangage                                     uniforme dans la plage où la cible reste dans l'image

Relais du chercheur : en usage réel l'agent d'approche prend la main après l'explorateur, dont
l'estimateur d'état a convergé. On simule ça en recalant l'estimateur sur la vraie attitude
(+ bruit de cap) au départ, puis on fait `warmup_steps` pas en tenant le tangage du départ pour que
le Kalman cible soit initialisé. Départ rejeté (et re-tiré) si la cible n'est pas détectée à la fin.

Observation : depth (2, 32, 56) et vec (VEC_DIM + 3,) = vec de handin.features + position attendue
de la cible en repère drone horizontal (dz_rel, bearing, elev). Les cartes local/glob ne sont pas
envoyées au réseau.

Reward par pas : progress * (d_prev - d) + detect * [vue] - lost * [pas vue] - center * |offset box|
                 - time - smooth * |a - a_prev|^2
Fin : success_base + success_speed * (1 - t / timeout), collision, out_of_arena, timeout.
Pas de fin « cible perdue » : un détour (arbre, level 2) ne doit pas être puni par une fin d'épisode.
"""

import json

import gymnasium as gym
import numpy as np
import pybullet as p
from gymnasium import spaces

from fpv_sim.envs.approach_aviary import ApproachAviary
from fpv_sim.utils.terrain import TERRAINS_DIR
from handin.features import BOX_HALF_H, HFOV, SHAPES, TILT, VEC_DIM, Features
from train.base_env import ARENA, CLEAR_M, BaseFPVEnv

VFOV_HALF = np.arctan(np.tan(HFOV / 2) * 90 / 160)          # ~34,8°
CAM_BOTTOM = VFOV_HALF - TILT                                # bord bas de l'image sous l'horizon, à plat (~9,8°)
MAX_ANGLE = np.radians(60)                                   # pleine course roll / pitch (AngleControl)
EXTRA_DIM = 3
OBS_SHAPES = {"depth": SHAPES["depth"], "vec": (VEC_DIM + EXTRA_DIM,)}


def _seg_dist(pts, a, b):
    """Distance de chaque point (N, 2) au segment [a, b]."""
    ab = b - a
    t = np.clip(((pts - a) @ ab) / max(ab @ ab, 1e-9), 0., 1.)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


class ApproachEnv(BaseFPVEnv):
    def __init__(self, cfg, terrains, stage=0, seed=0):
        gym.Env.__init__(self)
        self.cur, self.rw, self.ap = cfg["curriculum"], cfg["reward"], cfg["approach"]
        self.terrains = list(terrains)
        self.stage = int(stage)
        self.rng = np.random.default_rng(seed)
        self._targets = {}
        self.sim = ApproachAviary(continuous=True, terrain=self.terrains[0])
        self.feat = Features()
        self.use_local = bool(self.ap.get("use_local", False))   # + carte d'occupation égocentrée (5, 64, 64)
        shapes = dict(OBS_SHAPES, **({"local": SHAPES["local"]} if self.use_local else {}))
        self.observation_space = spaces.Dict({k: spaces.Box(-np.inf, np.inf, shape=s, dtype=np.float32)
                                              for k, s in shapes.items()})
        self._props = {}                                          # scène -> xy des props (départs difficiles)
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype=np.float32)
        self.final_stage = len(self.cur["stages"]) - 1
        self.stage_used, self.t = self.stage, 0
        self.n_starts = self.n_rejects = 0

    # ------------------------------------------------------------ départ
    def _sample_start(self, stage):
        rng, ap = self.rng, self.ap
        d_mean, d_sig, z_lo, z_hi, psi_max, v_max, _ = self.cur["stages"][stage]
        s = d_sig * d_mean
        d = float(np.clip(rng.normal(d_mean, s), d_mean - 2 * s, d_mean + 2 * s))
        z = float(rng.uniform(z_lo, z_hi))
        lim = ap["ground_half_m"] - 10.
        hard = rng.random() < ap.get("hard_frac", 0.)          # départ difficile : un prop près de la trajectoire
        props = self._scene_props() if hard else None
        for _ in range(300 if hard else 100):                  # reste au-dessus du sol (500 x 500 m)
            th = rng.uniform(-np.pi, np.pi)                    # direction drone -> cible
            xy = self.target_xy - d * np.array([np.cos(th), np.sin(th)])
            if np.abs(xy).max() > lim:
                continue
            if not hard or not len(props) or _seg_dist(props, xy, self.target_xy).min() < ap.get("hard_dist_m", 3.):
                break
        psi = rng.uniform(-psi_max, psi_max)
        yaw = th + np.radians(psi)
        self.spawn = {"d": d, "z": z, "psi_deg": float(psi)}
        # tangage : la cible doit rester dans l'image (bord bas et bord haut, avec une marge)
        # décalage vertical de la cible dans l'image = elev + pitch - TILT, à garder dans ±(VFOV_HALF - marge)
        e, m = np.arctan2(BOX_HALF_H - z, d), np.radians(ap["pitch_margin_deg"])   # e < 0 : sous l'horizon
        lo = max(0., TILT - VFOV_HALF + m - e)
        hi = min(np.radians(ap["pitch_max_deg"]), TILT + VFOV_HALF - m - e)
        pitch = float(rng.uniform(lo, max(lo, hi)))
        roll = float(np.radians(rng.uniform(-ap["roll_max_deg"], ap["roll_max_deg"])))
        return [float(xy[0]), float(xy[1]), z], [roll, pitch, float(yaw)], float(rng.uniform(0, v_max))

    def _place_drone(self, setup=None):
        """Départ tiré (setup=None) ou départ fixe d'un setup du benchmark (vitesse nulle)."""
        sim = self.sim
        sim.INIT_XYZS, sim.INIT_RPYS = np.array([[0., 0., 3.]]), np.zeros((1, 3))
        sim.reset(seed=int(self.rng.integers(2**31)))
        drone = sim.DRONE_IDS[0]
        for _ in range(1 if setup else 50):
            xyz, rpy, speed = (setup["xyz"], setup["rpy"], 0.) if setup else self._sample_start(self.stage_used)
            p.resetBasePositionAndOrientation(drone, xyz, p.getQuaternionFromEuler(rpy), physicsClientId=sim.CLIENT)
            if not any(p.getClosestPoints(drone, b, CLEAR_M, physicsClientId=sim.CLIENT) for b in sim.TERRAIN_IDS):
                break
        yaw = rpy[2]
        p.resetBaseVelocity(drone, [speed * np.cos(yaw), speed * np.sin(yaw), 0], [0, 0, 0],
                            physicsClientId=sim.CLIENT)
        sim._updateAndStoreKinematicInformation()
        sim.vio.reset(sim._getDroneStateVector(0))
        sim.imu.reset()
        return sim._sense(), rpy

    def _sync_estimator(self):
        """Relais du chercheur : attitude et vitesse vraies (+ bruit de cap) dans l'estimateur."""
        est, sim = self.feat.est, self.sim
        R = np.array(p.getMatrixFromQuaternion(sim.quat[0])).reshape(3, 3)
        est.R = _rz(np.radians(self.rng.normal(0, self.ap["yaw_noise_deg"]))) @ R
        est.vel = np.array(sim.vel[0], dtype=float)
        est.zx[1] = est.vel[2]
        est.rp_model = np.array([est.roll, est.pitch])

    def _proximity(self, rng_m):
        """max(0, 1 - dist / rng_m) au prop le plus proche (arbres, rochers... ; ni sol, ni boîte). Vérité
        terrain : sert uniquement à la reward. Seuls les props à moins de rng_m + 15 m (centre) sont testés."""
        sim = self.sim
        if self._obstacles is None:
            ids = [b for b in sim.TERRAIN_IDS[1:] if self._is_urdf(b)]
            xy = np.array([p.getBasePositionAndOrientation(b, physicsClientId=sim.CLIENT)[0][:2] for b in ids]) \
                if ids else np.zeros((0, 2))
            self._obstacles = (ids, xy)
        ids, xy = self._obstacles
        if not len(ids):
            return 0.
        near = np.nonzero(np.linalg.norm(xy - sim.pos[0, :2], axis=1) < rng_m + 15.)[0]
        dmin = rng_m
        for k in near:
            for c in p.getClosestPoints(sim.DRONE_IDS[0], ids[k], rng_m, physicsClientId=sim.CLIENT):
                dmin = min(dmin, max(c[8], 0.))
        return 1. - dmin / rng_m

    def _scene_props(self):
        """xy des props (objets urdf) de la scène courante, lus dans son JSON."""
        scene = self.terrain.split("@")[0]
        if scene not in self._props:
            objs = json.loads((TERRAINS_DIR / f"{scene}.json").read_text())["objects"]
            self._props[scene] = np.array([o["position_m"][:2] for o in objs if o["type"] == "urdf"]).reshape(-1, 2)
        return self._props[scene]

    def _is_urdf(self, body):
        """Props chargés par loadURDF : lien de base "base_link" ; surfaces (createMultiBody) : "link0"."""
        return p.getBodyInfo(body, physicsClientId=self.sim.CLIENT)[0] != b"link0"

    # ------------------------------------------------------------ observation
    def _expected(self):
        """Position attendue de la cible, repère drone horizontal : (dz_rel, bearing, elev)."""
        est, tf = self.feat.est, self.feat.target
        if tf.x is None:
            return np.zeros(EXTRA_DIM, np.float32)
        fwd, left = self.feat._heading()
        rel = tf.x - est.pos[:2]
        dx, dy = rel @ fwd, rel @ left
        dz = BOX_HALF_H - est.pos[2]
        return np.array([np.clip(dz / 50, -2, 2), np.arctan2(dy, dx) / np.pi,
                         np.arctan2(dz, np.hypot(dx, dy)) / (np.pi / 2)], np.float32)

    def _obs(self, x):
        vec = np.concatenate([x["vec"], self._expected()]).astype(np.float32)
        depth = np.clip(np.nan_to_num(x["depth"], nan=1., posinf=1., neginf=0.), 0., 1.)   # état physique explosé
        out = {"depth": depth.astype(np.float32), "vec": np.nan_to_num(vec, nan=0., posinf=3., neginf=-3.)}
        if self.use_local:
            out["local"] = np.clip(np.nan_to_num(x["local"], nan=0.), 0., 1.).astype(np.float32)
        return out

    # ------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        """options={"setup": dict} : départ fixe du benchmark (voir train.approach.bench), sans re-tirage."""
        gym.Env.reset(self, seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        ap = self.ap
        setup = (options or {}).get("setup")
        for _ in range(1 if setup else ap["max_tries"]):
            if setup:
                self.terrain, self.stage_used = setup["terrain"], setup["stage"]
                self.spawn = {"d": setup["d"], "z": setup["xyz"][2], "psi_deg": setup["psi_deg"]}
            else:
                self.terrain = self.terrains[int(self.rng.integers(len(self.terrains)))]
                self.stage_used = self._pick_stage()
            self.target_xy = self._target_xy(self.terrain)
            self.sim.TERRAIN = self.terrain
            obs, rpy = self._place_drone(setup)
            self.feat.reset()
            x = self.feat(obs, None)
            self._sync_estimator()
            hold = np.clip(np.array([rpy[0] / MAX_ANGLE, rpy[1] / MAX_ANGLE, 0., 0.]), -1, 1)
            ok = True
            for _ in range(ap["warmup_steps"]):
                obs, _, _, _, info = self.sim.step(hold)
                x = self.feat(obs, hold)
                if info["is_success"] or info["collision"]:
                    ok = False
                    break
            self.n_starts += 1
            if ok and obs["detected"] and self.feat.target.x is not None:
                break
            self.n_rejects += 1
        self.timeout_steps = int(30 * self.cur["stages"][self.stage_used][6])
        self.t = 0
        self.last_a = hold
        self.lost_steps = 0
        self.d_prev = self.d0 = self.sim._distance()
        self._obstacles = None                         # (ids, xy) des props, calculés au 1er besoin (proximity)
        self.prox_sum = 0.
        return self._obs(x), {"terrain": self.terrain, "stage": self.stage_used}

    def step(self, action):
        rw = self.rw
        a = np.clip(np.asarray(action, dtype=float), -1., 1.)
        obs, _, _, _, info = self.sim.step(a)
        self.t += 1
        x = self.feat(obs, a)

        det = bool(obs["detected"])
        self.lost_steps += not det
        d = info["distance_m"] if np.isfinite(info["distance_m"]) else self.d_prev
        r = rw["progress"] * (self.d_prev - d) + rw["detect"] * det - rw["lost"] * (not det) - rw["time"] \
            - rw["smooth"] * float(np.sum((a - self.last_a) ** 2))
        if det:
            box = np.asarray(obs["box"])
            r -= rw["center"] * float(np.hypot((box[0] + box[2]) / 2 - .5, (box[1] + box[3]) / 2 - .5))
        if rw.get("proximity", 0.) > 0:
            pen = rw["proximity"] * self._proximity(rw.get("proximity_range_m", 2.5))
            r -= pen
            self.prox_sum += pen
        self.d_prev, self.last_a = d, a

        outcome = None
        if info["is_success"]:
            outcome = "success"
            r += rw["success"] + rw["success_speed"] * max(0., 1. - self.t / self.timeout_steps)
        elif info["collision"] or not np.isfinite(self.sim.pos).all():   # PyBullet diverge (drone dans un arbre)
            outcome = "collision"
        elif np.abs(self.sim.pos[0, :2]).max() >= ARENA:
            outcome = "out_of_arena"
        elif self.t >= self.timeout_steps:
            outcome = "timeout"
        if outcome and outcome != "success":
            r += rw[outcome]
        out = {}
        if outcome:
            out = {"outcome": outcome, "stage": self.stage_used, "terrain": self.terrain, "ep_len": self.t,
                   "t_s": self.t / 30, "lost_frac": self.lost_steps / self.t, "d0": self.d0,
                   "reject_rate": self.n_rejects / max(self.n_starts, 1), **self.spawn}
        return self._obs(x), float(r), outcome in ("success", "collision", "out_of_arena"), outcome == "timeout", out


def make_approach_env_fn(cfg, terrains, rank, stage):
    """Fabrique pour SubprocVecEnv : chaque worker crée son propre simulateur PyBullet."""
    def _init():
        return ApproachEnv(cfg, terrains, stage=stage, seed=cfg["run"]["seed"] * 1000 + rank)
    return _init
