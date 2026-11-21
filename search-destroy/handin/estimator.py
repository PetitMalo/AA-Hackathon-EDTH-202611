"""Bloc A : estimateur d'état à partir des seules observations (gyro, accel, VIO).

Sorties (repère monde, z vers le haut ; repère corps FLU) :
    R      (3,3) rotation corps -> monde
    pos    (3,)  x, y (VIO brut), z filtré
    vel    (3,)  vitesse monde filtrée
    yaw, roll, pitch (rad)

Hypothèses issues du simulateur :
- départ à plat (roll = pitch = 0) ; seul le yaw est inconnu, à ±20° du cap vers (0, 0) ;
- le drone part en volant dans l'axe de son cap ;
- gyro sans biais : on intègre la rotation complète ;
- accéléromètre = force spécifique corps : a_monde = R f + g, sans approximation
  (la traînée est une force extérieure, elle apparaît dans f).
"""

import numpy as np

DT = 1 / 30
G = 9.8
GRAVITY = np.array([0., 0., -G])


def _skew(w):
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])


def _expm_so3(w):
    """Rotation de vecteur w (rad), formule de Rodrigues."""
    theta = np.linalg.norm(w)
    if theta < 1e-9:
        return np.eye(3) + _skew(w)
    k = _skew(w / theta)
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * k @ k


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class StateEstimator:
    # Recalage du yaw par comparaison accéléro (prédit) / VIO (mesuré), fenêtre glissante
    YAW_WINDOW = 30          # steps (1 s)
    YAW_MIN_DV = 1.5         # m/s : variation de vitesse minimale pour que l'angle soit fiable
    YAW_GAIN = 0.02          # part de l'erreur corrigée à chaque recalage
    VEL_GAIN = 0.15          # complémentaire vitesse : poids de la mesure VIO à chaque step
    VEL_LAG = 3              # steps : la vitesse VIO est mesurée sur 3 steps (bruit / 3)
    RP_GAIN = 0.0            # recalage roll/pitch vers le modèle : désactivé, il dégrade l estimation (testé)

    def reset(self):
        self.t = 0
        self.R = None
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.prev_gyro = None
        self.vio_hist = []       # positions VIO xy
        self.acc_hist = []       # accélérations monde horizontales prédites (R f + g)
        # Kalman altitude [z, vz]
        self.zx = None
        self.zP = np.diag([.05**2, 1.])

    # ------------------------------------------------------------------
    def update(self, obs, last_action=None):
        """last_action : sticks envoyés au step précédent (pour le modèle d'attitude), None = inconnu."""
        gyro = np.asarray(obs["gyro_rad_s"], dtype=float)
        accel = np.asarray(obs["accel_m_s2"], dtype=float)
        vio = np.asarray(obs["vio_xyz_m"], dtype=float)

        if self.R is None:
            self._init(vio)
        else:
            # 1) attitude : intégration trapèze du gyro (corps), R <- R exp(w dt)
            w = .5 * (gyro + self.prev_gyro)
            self.R = self.R @ _expm_so3(w * DT)
            # ré-orthonormalisation numérique
            u, _, vt = np.linalg.svd(self.R)
            self.R = u @ vt
            # 1b) roll/pitch recalés doucement sur le modèle du contrôleur (mode angle)
            if last_action is not None:
                self._attitude_model(last_action)
                if self.RP_GAIN > 0:
                    d_roll = _wrap(self.rp_model[0] - self.roll)
                    d_pitch = _wrap(self.rp_model[1] - self.pitch)
                    # petites corrections en repère corps : roll autour de x, pitch autour de y
                    self.R = self.R @ _expm_so3(self.RP_GAIN * np.array([d_roll, d_pitch, 0.]))
        self.prev_gyro = gyro

        acc_world = self.R @ accel + GRAVITY
        self.vio_hist.append(vio[:2].copy())
        self.acc_hist.append(acc_world[:2].copy())
        if len(self.vio_hist) > self.YAW_WINDOW + 2 * self.VEL_LAG + 2:
            self.vio_hist.pop(0)
            self.acc_hist.pop(0)

        # 2) yaw initial affiné par la vitesse de départ (le drone vole dans l'axe de son cap)
        if self.t == 3:
            d = self.vio_hist[-1] - self.vio_hist[0]
            if np.linalg.norm(d) / (3 * DT) > 1.:
                self._set_yaw(np.arctan2(d[1], d[0]))
                self.vel[:2] = d / (3 * DT)

        # 3) recalage continu du yaw : Δv mesurée par le VIO vs Δv prédite par l'accéléro
        self._correct_yaw()

        # 4) vitesse horizontale : prédiction accéléro + correction VIO
        self.vel[:2] += acc_world[:2] * DT
        if len(self.vio_hist) > self.VEL_LAG:
            v_meas = (self.vio_hist[-1] - self.vio_hist[-1 - self.VEL_LAG]) / (self.VEL_LAG * DT)
            # v_meas est centrée sur t - VEL_LAG/2 : on la ramène à t avec l'accélération récente
            v_meas = v_meas + np.mean(self.acc_hist[-self.VEL_LAG:], axis=0) * (self.VEL_LAG / 2 * DT)
            self.vel[:2] += self.VEL_GAIN * (v_meas - self.vel[:2])
        self.pos[:2] = vio[:2]

        # 5) altitude : Kalman [z, vz], entrée = accélération verticale
        self._altitude(vio[2], acc_world[2])
        self.t += 1
        return self

    # ------------------------------------------------------------------
    def _attitude_model(self, action):
        """Simule la cascade AngleControl à 240 Hz : rate_sp = 6 (cible - angle), d(rate)/dt = 30 (rate_sp - rate)."""
        target = np.radians(60) * np.clip(np.asarray(action[:2], dtype=float), -1, 1)
        h = DT / 8
        for _ in range(8):
            rate_sp = 6. * (target - self.rp_model)
            self.rp_rate += 30. * (rate_sp - self.rp_rate) * h
            self.rp_model += self.rp_rate * h

    def _init(self, vio):
        self.rp_model = np.zeros(2)
        self.rp_rate = np.zeros(2)
        self.R = _rz(np.arctan2(-vio[1], -vio[0]))
        self.pos = vio.copy()
        self.zx = np.array([vio[2], 0.])

    def _set_yaw(self, yaw):
        self.R = _rz(yaw - self.yaw) @ self.R

    def _correct_yaw(self):
        n = self.YAW_WINDOW
        lag = self.VEL_LAG
        if len(self.vio_hist) < n + 2 * lag + 1:
            return
        v = lambda i: (self.vio_hist[i] - self.vio_hist[i - lag]) / (lag * DT)
        dv_meas = v(-1) - v(-1 - n)
        # même fenêtre, décalée de lag/2 comme les vitesses VIO
        dv_pred = np.sum(self.acc_hist[-n - lag // 2 - 1:-lag // 2 - 1 or None], axis=0) * DT
        if min(np.linalg.norm(dv_meas), np.linalg.norm(dv_pred)) < self.YAW_MIN_DV:
            return
        err = _wrap(np.arctan2(dv_meas[1], dv_meas[0]) - np.arctan2(dv_pred[1], dv_pred[0]))
        self.R = _rz(self.YAW_GAIN * err) @ self.R

    def _altitude(self, z_meas, az):
        F = np.array([[1., DT], [0., 1.]])
        B = np.array([.5 * DT**2, DT])
        Q = np.diag([1e-4, 4e-2])
        self.zx = F @ self.zx + B * az
        self.zP = F @ self.zP @ F.T + Q
        H = np.array([1., 0.])
        S = H @ self.zP @ H + .05**2
        K = self.zP @ H / S
        self.zx = self.zx + K * (z_meas - self.zx[0])
        self.zP = (np.eye(2) - np.outer(K, H)) @ self.zP
        self.pos[2], self.vel[2] = self.zx

    # ------------------------------------------------------------------
    @property
    def yaw(self):
        return float(np.arctan2(self.R[1, 0], self.R[0, 0]))

    @property
    def roll(self):
        return float(np.arctan2(self.R[2, 1], self.R[2, 2]))

    @property
    def pitch(self):
        return float(np.arcsin(np.clip(-self.R[2, 0], -1, 1)))
