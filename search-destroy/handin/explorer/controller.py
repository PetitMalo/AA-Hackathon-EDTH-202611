"""Goal point -> RC sticks (roll, pitch, yaw, throttle) in [-1, 1], for ApproachAviary's angle mode.

Stick conventions (see ApproachAviary._preprocessAction and AngleControl):
- roll  +1 = +60 deg roll  -> accelerates to the right (-y body)
- pitch +1 = +60 deg pitch -> nose down, accelerates forward (+x body)
- yaw   +1 = -200 deg/s    -> turns clockwise seen from above
- throttle 0 = hover thrust, thrust = hover * (1 + throttle)

Cascade: position error -> desired velocity -> desired acceleration -> attitude + thrust.
The nose (and so the camera) is kept pointing at the goal, and the drone slows down while
it turns so that it flies mostly where the camera has looked.
A reactive layer brakes and climbs when the depth image shows an obstacle along the velocity.
"""

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class WaypointController:
    def __init__(self, cfg, cam_cfg, g=9.8):
        self.cfg, self.cam_cfg, self.g = cfg, cam_cfg, g
        c = cfg
        # Quadratic drag of TerrainAviary, expressed as an acceleration: k |v| v
        self.drag_k = g * np.tan(np.radians(c.top_speed_pitch_deg)) / c.top_speed_m_s ** 2
        self.obstacle_ahead = False

    def compute(self, est, goal, max_speed, depth_m=None, cam=None, ignore_obstacles_beyond=np.inf):
        """est: StateEstimator; goal: (3,) world point; cam: (cam_pos, cam_R, CameraModel) for the safety layer."""
        c = self.cfg
        p, v, yaw = est.p, est.v, est.yaw
        e = goal - p
        dist_h = np.linalg.norm(e[:2])

        # --- heading: look where we go
        des_yaw = np.arctan2(e[1], e[0]) if dist_h > 2. else yaw
        yaw_err = wrap(des_yaw - yaw)
        yaw_stick = -c.yaw_gain * yaw_err

        # --- horizontal velocity setpoint, reduced while the nose is not on the goal
        speed = min(max_speed, c.speed_gain * dist_h) * max(0., np.cos(yaw_err)) ** 2
        v_des = np.zeros(3)
        if dist_h > 1e-3:
            v_des[:2] = e[:2] / dist_h * speed
        v_des[2] = np.clip(c.alt_kp * e[2], -3., 3.)

        # --- reactive safety: obstacle along the velocity vector -> brake and climb
        if self.check_obstacle_ahead(v, depth_m, cam, ignore_obstacles_beyond):
            v_des[:2] = 0.
            v_des[2] = 2.

        # --- acceleration setpoint
        a = np.zeros(3)
        a[:2] = c.vel_gain * (v_des[:2] - v[:2])
        a[2] = c.alt_kd * (v_des[2] - v[2])
        return self.accel_to_sticks(a, v, yaw, yaw_stick)

    def yaw_stick_toward(self, p, yaw, point):
        """Yaw stick that turns the nose toward `point` (horizontal plane), and the heading error."""
        e = point[:2] - p[:2]
        des_yaw = np.arctan2(e[1], e[0]) if np.linalg.norm(e) > 2. else yaw
        yaw_err = wrap(des_yaw - yaw)
        return -self.cfg.yaw_gain * yaw_err, yaw_err

    def accel_to_sticks(self, a, v, yaw, yaw_stick):
        """Desired world acceleration (without drag) -> (roll, pitch, yaw, throttle) sticks, drag fed forward."""
        c = self.cfg
        a = np.array(a, dtype=float)
        a[:2] = a[:2] + self.drag_k * np.linalg.norm(v) * v[:2]

        # --- acceleration -> attitude + collective thrust (yaw frame)
        cy, sy = np.cos(yaw), np.sin(yaw)
        a_fwd = cy * a[0] + sy * a[1]
        a_left = -sy * a[0] + cy * a[1]
        a_up = max(self.g + a[2], 0.3 * self.g)
        thrust = np.sqrt(a_fwd ** 2 + a_left ** 2 + a_up ** 2)
        max_tilt = np.radians(c.max_tilt_deg)
        pitch = np.clip(np.arctan2(a_fwd, a_up), -max_tilt, max_tilt)
        roll = np.clip(-np.arcsin(np.clip(a_left / thrust, -1, 1)), -max_tilt, max_tilt)
        throttle = a_up / (np.cos(pitch) * np.cos(roll)) / self.g - 1.

        full = np.radians(c.max_stick_deg)
        sticks = np.array([roll / full, pitch / full, yaw_stick, throttle])
        return np.clip(sticks, -1., 1.).astype(np.float32)

    def check_obstacle_ahead(self, v, depth_m, cam, ignore_obstacles_beyond=np.inf):
        """Reactive layer: depth image shows an obstacle along the velocity, closer than the braking distance."""
        self.obstacle_ahead = False
        if depth_m is not None and cam is not None:
            d_obs = self._depth_along_velocity(depth_m, v, *cam)
            self.obstacle_ahead = d_obs < min(self.cfg.safety_depth_m + np.linalg.norm(v[:2]), ignore_obstacles_beyond)
        return self.obstacle_ahead

    def _depth_along_velocity(self, depth_m, v, cam_pos, cam_R, camera, half_win=10):
        """Robust min depth around the pixel where the velocity vector points (inf if out of view / slow)."""
        if np.linalg.norm(v) < 1.:
            return np.inf
        v_cam = cam_R.T @ v                                   # (forward, left, up)
        if v_cam[0] <= 0.1 * np.linalg.norm(v):
            return np.inf
        cc = self.cam_cfg
        f = (cc.depth_width / 2) / camera.tan_half_h
        u = int(cc.depth_width / 2 - f * v_cam[1] / v_cam[0])
        r = int(cc.depth_height / 2 - f * v_cam[2] / v_cam[0])
        if not (0 <= u < cc.depth_width and 0 <= r < cc.depth_height):
            return np.inf
        win = np.asarray(depth_m)[max(r - half_win, 0):r + half_win, max(u - half_win, 0):u + half_win]
        return float(np.quantile(win, 0.05)) if win.size else np.inf
