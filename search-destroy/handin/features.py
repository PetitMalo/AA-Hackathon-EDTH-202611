"""Blocs B-E : observation brute -> entrées du réseau (le "belief state").

    feats = Features(); feats.reset()
    x = feats(obs, last_action)     # dict de tableaux float32, formes dans SHAPES

Entrées produites :
    depth  (2, 32, 56)   depth réduite par min-pooling, 2 dernières frames, /40
    local  (5, 64, 64)   carte égocentrée 1 m (haut = devant le drone) :
                         sol_vu, occ_bas (0,3-2 m), occ_moyen (2-5 m), occ_haut (>5 m), cible
    glob   (4, 32, 32)   carte égocentrée 12,5 m (±200 m) : sol_vu, sol_vu_loin, cible, trace du drone
    vec    (VEC_DIM,)    détection, mémoire, cible estimée, état, IMU, dernière action, temps

Toutes les cartes sont stockées dans un repère monde fixe puis échantillonnées autour
du drone et tournées selon son cap à chaque step.
"""

import numpy as np

from handin.estimator import StateEstimator

# ---------------------------------------------------------------- caméra (copie de fpv_sim Camera)
HFOV = np.radians(102.)
TILT = np.radians(25.)
MOUNT = np.array([.30, 0., 0.])
CAM_IN_BODY = np.array([[np.cos(TILT), 0., -np.sin(TILT)],      # colonnes : avant, gauche, haut caméra
                        [0., 1., 0.],
                        [np.sin(TILT), 0., np.cos(TILT)]])
MAX_DEPTH = 40.
DETECT_RANGE = 250.
BOX_HALF_H = .775
BOX_MIN_DIM = 1.55                       # boîte 4 x 1,65 x 1,55 m (target_box.urdf)
DET_F_PX = 80. / np.tan(HFOV / 2)        # focale de la caméra de détection 160 x 90, en pixels


def _pixel_rays(w, h, cols, rows):
    """Directions (avant, gauche, haut) en repère caméra, avant = 1 (profondeur planaire).

    cols, rows : indices de pixel (fractionnaires acceptés). Convention de rendu PyBullet mesurée
    par raycast : le pixel (row, col) vise x = col, y = row + 1 (et non les centres col + .5, row + .5).
    Avec cette convention, depth -> 3D coïncide avec rayTestBatch à ~1 mm près (au lieu de ~0,2-0,6 m).
    """
    th = np.tan(HFOV / 2)
    tv = th * h / w
    x = 2 * np.asarray(cols, np.float64) / w - 1
    y = 1 - 2 * (np.asarray(rows, np.float64) + 1) / h
    return np.stack([np.ones_like(x * y), -x * th, y * tv], axis=-1)


# ---------------------------------------------------------------- depth : pooling 4x4
D_H, D_W = 126, 224
POOL = 4
P_H, P_W = 32, 56                       # 128/4, 224/4 (2 lignes ajoutées en bas par répétition)
_rr, _cc = np.meshgrid(np.arange(P_H) * POOL + 1.5, np.arange(P_W) * POOL + 1.5, indexing="ij")
RAYS_CAM = _pixel_rays(D_W, D_H, _cc, np.minimum(_rr, D_H - 1)).reshape(-1, 3)   # (1792, 3)

# ---------------------------------------------------------------- cartes
FINE_RES, FINE_HALF = 1., 360            # monde : 720 x 720 cellules de 1 m
COARSE_RES, COARSE_HALF = 12.5, 32       # monde : 64 x 64 cellules de 12,5 m (±400 m)
LOCAL_N, GLOBAL_N = 64, 32
Z_LAYERS = (.3, 2., 5.)                  # bornes sol / bas / moyen / haut
N_SAT = 2.                               # saturation des compteurs : 1 - exp(-n / N_SAT)

VEC_DIM = 38
SHAPES = {"depth": (2, P_H, P_W), "local": (5, LOCAL_N, LOCAL_N),
          "glob": (4, GLOBAL_N, GLOBAL_N), "vec": (VEC_DIM,)}


def _ego_offsets(n, res):
    """Offsets (avant, gauche) en mètres des cellules d'une carte égocentrée n x n, haut = devant."""
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    return ((n / 2 - .5 - i) * res).ravel(), ((n / 2 - .5 - j) * res).ravel()


LOCAL_OFF = _ego_offsets(LOCAL_N, FINE_RES)
GLOBAL_OFF = _ego_offsets(GLOBAL_N, COARSE_RES)


class TargetFilter:
    """Kalman 2D sur la position (x, y) de la cible, statique, mesures anisotropes."""

    ANG_ERR = .03          # rad : erreur angulaire globale (attitude + jitter de la box)
    GATE = 25.             # chi2 2 ddl
    MAX_REJECT = 10

    def reset(self):
        self.x, self.P, self.rejects, self.n = None, None, 0, 0

    def update(self, z, cov):
        if self.x is None:
            self.x, self.P, self.n = z.copy(), cov.copy(), 1
            return
        S = self.P + cov
        r = z - self.x
        if r @ np.linalg.solve(S, r) > self.GATE:
            self.rejects += 1
            if self.rejects >= self.MAX_REJECT:      # l'estimation était fausse : on repart
                self.x, self.P, self.rejects, self.n = z.copy(), cov.copy(), 0, 1
            return
        K = self.P @ np.linalg.inv(S)
        self.x = self.x + K @ r
        self.P = (np.eye(2) - K) @ self.P
        self.rejects, self.n = 0, self.n + 1

    @property
    def sigma(self):
        return float(np.sqrt(np.max(np.linalg.eigvalsh(self.P)))) if self.P is not None else 1e3


class Features:
    def __init__(self):
        self.est = StateEstimator()
        self.target = TargetFilter()

    def reset(self):
        self.est.reset()
        self.target.reset()
        nf, nc = 2 * FINE_HALF, 2 * COARSE_HALF
        self.fine = np.zeros((4, nf, nf), np.float32)          # compteurs : sol, bas, moyen, haut
        self.coarse_seen = np.zeros((nc, nc), np.float32)      # nb de cellules fines vues (sol)
        self.coarse_far = np.zeros((nc, nc), np.float32)       # compteur rayons lointains au sol
        self.coarse_trace = np.zeros((nc, nc), np.float32)     # passage du drone
        self.prev_depth = None
        self.t = 0
        self.last_box = np.zeros(4, np.float32)
        self.since_det = 1e3
        self.ever_det = 0.
        self.last_action = np.zeros(4, np.float32)

    # ================================================================ step
    def __call__(self, obs, last_action=None):
        if last_action is not None:
            self.last_action = np.asarray(last_action, np.float32)
        est = self.est.update(obs, last_action)
        R_cam = est.R @ CAM_IN_BODY
        eye = est.pos + est.R @ MOUNT

        depth_net, depth_map = self._pool_depth(np.asarray(obs["depth_m"], np.float32))
        self._map_depth(depth_map, R_cam, eye)
        self._detection(obs, R_cam, eye, np.asarray(obs["depth_m"], np.float32))
        (ci,), (cj,) = self._coarse_idx(est.pos[:2])
        if 0 <= ci < 2 * COARSE_HALF and 0 <= cj < 2 * COARSE_HALF:
            self.coarse_trace[ci, cj] += 1

        if self.prev_depth is None:
            self.prev_depth = depth_net
        out = {"depth": np.stack([depth_net, self.prev_depth]),
               "local": self._local_map(), "glob": self._global_map(), "vec": self._vector(obs)}
        for k in ("depth", "local", "glob"):     # un NaN (physique/estimateur divergé) ne doit jamais atteindre le réseau
            out[k] = np.nan_to_num(out[k], nan=0., posinf=1., neginf=0.)
        self.prev_depth = depth_net
        self.t += 1
        return out

    # ================================================================ B. depth
    def _pool_depth(self, d):
        d = np.concatenate([d, d[-2:]], axis=0)                            # 128 x 224
        blocks = d.reshape(P_H, POOL, P_W, POOL).transpose(0, 2, 1, 3).reshape(P_H, P_W, -1)
        return (blocks.min(-1) / MAX_DEPTH).astype(np.float32), np.median(blocks, -1)

    def _map_depth(self, depth, R_cam, eye):
        d = depth.ravel()
        dirs = RAYS_CAM @ R_cam.T                                          # monde, non normalisés
        hit = d < MAX_DEPTH - 1.5
        # --- impacts < 40 m : sol ou obstacle selon l'altitude du point
        pts = eye + dirs[hit] * d[hit, None]
        unit_z = dirs[hit, 2] / np.linalg.norm(dirs[hit], axis=1)
        tol = np.maximum(Z_LAYERS[0], 2 * .03 * d[hit] * np.abs(unit_z))   # bruit depth projeté sur z
        layer = np.where(pts[:, 2] < tol, 0, np.searchsorted(Z_LAYERS, pts[:, 2]))
        fi, fj = self._fine_idx(pts[:, :2])
        ok = (fi >= 0) & (fi < 2 * FINE_HALF) & (fj >= 0) & (fj < 2 * FINE_HALF)
        fi, fj, layer = fi[ok], fj[ok], layer[ok]
        ground = layer == 0
        newly = self.fine[0, fi[ground], fj[ground]] == 0
        np.add.at(self.fine, (layer, fi, fj), 1.)
        if newly.any():                                                    # couverture grossière
            cells = np.unique(np.stack([fi[ground][newly], fj[ground][newly]]), axis=1)
            ci, cj = self._coarse_idx(self._fine_center(*cells))
            np.add.at(self.coarse_seen, (ci, cj), 1.)
        # --- pas d'impact < 40 m : le rayon atteint peut-être le sol plus loin (canal "loin")
        far = ~hit & (dirs[:, 2] < -1e-3)
        if far.any() and eye[2] > 0:
            s = -eye[2] / dirs[far, 2]                                     # paramètre jusqu'au sol
            gp = eye[:2] + dirs[far, :2] * s[:, None]
            dist = s * np.linalg.norm(dirs[far], axis=1)
            keep = (dist > MAX_DEPTH) & (dist < DETECT_RANGE)
            if keep.any():
                ci, cj = self._coarse_idx(gp[keep])
                ok = (ci >= 0) & (ci < 2 * COARSE_HALF) & (cj >= 0) & (cj < 2 * COARSE_HALF)
                np.add.at(self.coarse_far, (ci[ok], cj[ok]), 1.)

    # ================================================================ C. cible
    def _detection(self, obs, R_cam, eye, depth):
        box = np.asarray(obs["box"], np.float32)
        self.since_det += 1 / 30
        if not obs["detected"]:
            return
        self.last_box, self.since_det, self.ever_det = box.copy(), 0., 1.
        if self.est.t <= 3:          # yaw initial = cap vers (0, 0) à ±20°, recalé seulement à t = 3 (estimator)
            return
        cx, cy = (box[0] + box[2]) / 2 * 160, (box[1] + box[3]) / 2 * 90
        ray = R_cam @ _pixel_rays(160, 90, cx, cy)                        # centre de la box (indices inclusifs)
        u = ray / np.linalg.norm(ray)
        # profondeur dans la box si la cible est proche
        r0, r1 = int(box[1] * D_H), int(np.ceil(box[3] * D_H)) + 1
        c0, c1 = int(box[0] * D_W), int(np.ceil(box[2] * D_W)) + 1
        patch = depth[r0:r1, c0:c1]
        dmed = float(np.median(patch)) if patch.size else MAX_DEPTH
        # la depth n'est crédible que si la taille apparente de la box est compatible avec dmed :
        # une box de 1-2 px (cible lointaine) dont le patch touche un obstacle proche donnerait sinon
        # une mesure fausse de ~150 m annoncée à sigma 1,5 m, qui fige le filtre.
        size_px = max((box[2] - box[0]) * 160 + 1, (box[3] - box[1]) * 90 + 1)
        if dmed < MAX_DEPTH - 2 and size_px * dmed / DET_F_PX >= .8 * BOX_MIN_DIM:
            p = eye + ray * dmed                                           # ray a avant = 1 : profondeur planaire
            sig = max(1.5, .05 * dmed)      # la depth touche la face de la boîte (4 m de long), pas son centre
            self.target.update(p[:2], np.eye(2) * sig**2)
            return
        # sinon : intersection du rayon avec le plan z = centre de la boîte
        if u[2] > -.02 or eye[2] <= BOX_HALF_H:
            return
        s = (BOX_HALF_H - eye[2]) / u[2]
        p = eye + u * s
        h = u[:2] / np.linalg.norm(u[:2])
        lat = np.array([-h[1], h[0]])
        rng_h = s * np.linalg.norm(u[:2])
        s_lat = max(1.5, rng_h * TargetFilter.ANG_ERR)
        s_al = min(200., max(1.5, s * TargetFilter.ANG_ERR / max(-u[2], 1e-3)))
        cov = s_lat**2 * np.outer(lat, lat) + s_al**2 * np.outer(h, h)
        self.target.update(p[:2], cov)

    # ================================================================ D. cartes égocentrées
    def _heading(self):
        y = self.est.yaw
        return np.array([np.cos(y), np.sin(y)]), np.array([-np.sin(y), np.cos(y)])

    def _sample(self, off, res, half):
        fwd, left = self._heading()
        xy = self.est.pos[:2] + off[0][:, None] * fwd + off[1][:, None] * left
        i = np.floor(xy[:, 0] / res).astype(int) + half
        j = np.floor(xy[:, 1] / res).astype(int) + half
        ok = (i >= 0) & (i < 2 * half) & (j >= 0) & (j < 2 * half)
        return xy, np.clip(i, 0, 2 * half - 1), np.clip(j, 0, 2 * half - 1), ok

    def _target_blob(self, xy):
        if self.target.x is None:
            return np.zeros(len(xy), np.float32)
        s = max(self.target.sigma, 1.)
        d2 = np.sum((xy - self.target.x)**2, axis=1)
        return np.exp(-.5 * d2 / s**2).astype(np.float32)

    def _local_map(self):
        xy, i, j, ok = self._sample(LOCAL_OFF, FINE_RES, FINE_HALF)
        occ = 1 - np.exp(-self.fine[:, i, j] / N_SAT)
        occ[:, ~ok] = 0
        out = np.concatenate([occ, self._target_blob(xy)[None]])
        return out.reshape(5, LOCAL_N, LOCAL_N).astype(np.float32)

    def _global_map(self):
        xy, i, j, ok = self._sample(GLOBAL_OFF, COARSE_RES, COARSE_HALF)
        seen = self.coarse_seen[i, j] / (COARSE_RES / FINE_RES)**2
        far = 1 - np.exp(-self.coarse_far[i, j] / 10.)
        trace = 1 - np.exp(-self.coarse_trace[i, j] / 30.)
        out = np.stack([seen, far, self._target_blob(xy), trace])
        out[:, ~ok] = 0
        return out.reshape(4, GLOBAL_N, GLOBAL_N).astype(np.float32)

    # ================================================================ E. vecteur
    def _vector(self, obs):
        est, box = self.est, np.asarray(obs["box"], np.float32)
        fwd, left = self._heading()
        to_ego = lambda v: np.array([v @ fwd, v @ left])
        det = float(obs["detected"])
        bx = np.array([(box[0] + box[2]) / 2 - .5, (box[1] + box[3]) / 2 - .5,
                       box[2] - box[0], box[3] - box[1]]) * det
        lb = self.last_box
        lbx = np.array([(lb[0] + lb[2]) / 2 - .5, (lb[1] + lb[3]) / 2 - .5, lb[2] - lb[0], lb[3] - lb[1]])
        if self.target.x is not None:
            rel = to_ego(self.target.x - est.pos[:2])
            dist = np.linalg.norm(rel)
            tgt = [1., *np.clip(rel / 100, -2, 2), min(dist / 100, 3), min(self.target.sigma / 20, 3)]
        else:
            tgt = [0., 0., 0., 0., 3.]
        v = to_ego(est.vel[:2])
        x, y = est.pos[:2]
        vec = np.concatenate([
            [det], bx,                                                     # 5
            lbx, [min(self.since_det / 5, 1.), self.ever_det],             # 6
            tgt,                                                           # 5
            v / 20, [est.vel[2] / 5, est.pos[2] / 20, est.roll, est.pitch],  # 6
            to_ego(-est.pos[:2]) / 350, [min(350 - abs(x), 350 - abs(y)) / 350],  # 3
            np.asarray(obs["gyro_rad_s"]) / 3, np.asarray(obs["accel_m_s2"]) / 9.8,  # 6
            self.last_action, [self.t / 3600],                             # 5
            [np.cos(est.yaw), np.sin(est.yaw)],                            # 2 (cap absolu)
        ]).astype(np.float32)
        assert vec.shape == (VEC_DIM,), vec.shape
        return np.nan_to_num(vec, nan=0., posinf=3., neginf=-3.)

    # ================================================================ indices
    @staticmethod
    def _fine_idx(xy):
        xy = np.atleast_2d(xy)
        return (np.floor(xy[:, 0] / FINE_RES).astype(int) + FINE_HALF,
                np.floor(xy[:, 1] / FINE_RES).astype(int) + FINE_HALF)

    @staticmethod
    def _fine_center(i, j):
        return np.stack([(i - FINE_HALF + .5) * FINE_RES, (j - FINE_HALF + .5) * FINE_RES], axis=1)

    @staticmethod
    def _coarse_idx(xy):
        xy = np.atleast_2d(xy)
        i = np.floor(xy[:, 0] / COARSE_RES).astype(int) + COARSE_HALF
        j = np.floor(xy[:, 1] / COARSE_RES).astype(int) + COARSE_HALF
        return i, j
