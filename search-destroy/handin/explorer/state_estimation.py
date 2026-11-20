"""Pose estimate from what the agent actually gets: IMU (gyro + specific force) and noisy VIO position.

- Attitude: gyro integration + Mahony-style tilt correction with the accelerometer.
  Yaw has no absolute reference: it starts pointing at the arena centre (the start
  heading is within 20 deg of it) and is re-aligned on the measured velocity heading
  during the first frames, when the drone still coasts along its initial heading.
- Position / velocity: alpha-beta filter on VIO, predicted with the accelerometer.
"""

import numpy as np


def skew(w):
    return np.array([[0., -w[2], w[1]], [w[2], 0., -w[0]], [-w[1], w[0], 0.]])


def rot_from_rotvec(w):
    """Rodrigues: rotation matrix of the rotation vector w."""
    angle = np.linalg.norm(w)
    if angle < 1e-9:
        return np.eye(3) + skew(w)
    k = skew(w / angle)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k


def rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def orthonormalize(r):
    u, _, vt = np.linalg.svd(r)
    return u @ vt


class StateEstimator:
    """Body-to-world rotation `R` (FLU body, ENU-like world as in PyBullet), position `p`, velocity `v`."""

    def __init__(self, dt=1. / 30., g=9.8, tilt_gain=0.5, pos_alpha=(0.5, 0.5, 0.2), pos_beta=(0.1, 0.1, 0.02),
                 align_frames=6, align_min_dist_m=0.6):
        self.dt, self.g = dt, g
        self.tilt_gain = tilt_gain
        self.alpha, self.beta = np.array(pos_alpha), np.array(pos_beta)
        self.align_frames, self.align_min_dist = align_frames, align_min_dist_m
        self.reset()

    def reset(self):
        self.R = np.eye(3)
        self.p = None
        self.v = np.zeros(3)
        self.frame = 0
        self.p0 = None
        self.aligned = False

    # ------------------------------------------------------------------ public API
    @property
    def yaw(self):
        return float(np.arctan2(self.R[1, 0], self.R[0, 0]))

    @property
    def ready(self):
        """True once the initial heading alignment is done; the map is not fed before that."""
        return self.aligned

    def update(self, obs):
        gyro = np.asarray(obs["gyro_rad_s"], dtype=float)
        accel = np.asarray(obs["accel_m_s2"], dtype=float)
        z = np.asarray(obs["vio_xyz_m"], dtype=float)

        if self.p is None:
            # Start is level, heading within 20 deg of the centre (README): best prior we have.
            self.R = rot_z(np.arctan2(-z[1], -z[0]))
            self.p, self.p0 = z.copy(), z.copy()

        self._update_attitude(gyro, accel)
        self._update_position(accel, z)
        self._align_heading()
        self.frame += 1

    # ------------------------------------------------------------------ internals
    def _update_attitude(self, gyro, accel):
        w = gyro.copy()
        norm = np.linalg.norm(accel)
        if abs(norm - self.g) < 0.2 * self.g:            # only trust the accel when not accelerating hard
            up_meas = accel / norm                        # specific force points "up" in steady flight
            up_est = self.R.T @ np.array([0., 0., 1.])
            w += self.tilt_gain * np.cross(up_meas, up_est)
        self.R = orthonormalize(self.R @ rot_from_rotvec(w * self.dt))

    def _update_position(self, accel, z):
        acc_world = self.R @ accel - np.array([0., 0., self.g])
        p_pred = self.p + self.v * self.dt + 0.5 * acc_world * self.dt ** 2
        v_pred = self.v + acc_world * self.dt
        r = z - p_pred
        self.p = p_pred + self.alpha * r
        self.v = v_pred + self.beta * r / self.dt

    def _align_heading(self):
        if self.aligned or self.frame < self.align_frames:
            return
        d = self.p[:2] - self.p0[:2]
        if np.linalg.norm(d) > self.align_min_dist:
            # The drone coasted along its initial heading with neutral sticks: velocity heading = yaw.
            delta = np.arctan2(d[1], d[0]) - self.yaw
            self.R = rot_z(delta) @ self.R
        self.aligned = True
