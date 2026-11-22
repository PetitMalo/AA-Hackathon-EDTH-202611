"""Start of every flight, before any planning: rise, then a nose-down dive toward the arena centre.

    rise   hold x, y (brakes the start speed), climb rise_m above the start altitude, turn the nose toward
           the arena centre. Ends at the altitude, on timeout, or when the drone stops climbing (something above).
    dive   fixed nose-down pitch (dive_pitch_deg): the camera, tilted 25 deg UP on the airframe, then looks
           dive_pitch - 25 deg BELOW the horizon, so the ground ahead is in view (the detector sees 250 m).
           Fly toward the centre (where the boxes are), descending at <= dive_descent_m_s, pitch eased off past
           dive_max_speed_m_s. Ends at dive_end_alt_m (above the rooftops, ~20 m max), near the centre, or on
           timeout. Then the frontier planning starts.

In angle mode a nose-down attitude means forward acceleration: the drone cannot hover while looking down,
hence a dive rather than a look in place. Above the voxel map (40 m) the depth (40 m range) sees no ground,
so nothing is planned up there. A target detection at any time takes over (the explorer ends this manoeuvre).
"""

import numpy as np


class StartManeuver:
    def __init__(self, cfg, control_cfg, controller, g=9.8, dt=1. / 30.):
        self.cfg, self.ctl = cfg, control_cfg
        self.controller = controller
        self.g, self.dt = g, dt
        self.reset()

    def reset(self):
        self.done = not self.cfg.enabled
        self.mode = None                   # None (not started) | "rise" | "dive"
        self.mode_s = 0.
        self.p0 = None
        self.stall_s = 0.

    def abort(self):
        """Target detected (or anything else that takes over): no more start manoeuvre this flight."""
        self.done, self.mode = True, None

    def step(self, est):
        """(mode, goal, sticks) while the manoeuvre runs, None once it is over (planning starts this frame)."""
        if self.done:
            return None
        c = self.cfg
        p, v = est.p, est.v
        if self.mode is None:
            self.mode, self.mode_s, self.p0 = "rise", 0., p.copy()
        self.mode_s += self.dt
        if self.mode == "rise":
            top = self.p0[2] + c.rise_m
            if self._rise_done(p, v, top):
                self.mode, self.mode_s = "dive", 0.
            else:
                return "rise", np.array([self.p0[0], self.p0[1], top]), self._rise_sticks(est, top)
        if self.mode == "dive":
            if (p[2] <= c.dive_end_alt_m or np.linalg.norm(p[:2]) <= c.dive_stop_radius_m
                    or self.mode_s >= c.dive_timeout_s):
                self.abort()
                return None
            return "dive", np.array([0., 0., c.dive_end_alt_m]), self._dive_sticks(est)
        return None

    # ------------------------------------------------------------------ rise
    def _rise_done(self, p, v, top):
        c = self.cfg
        if p[2] >= top - c.rise_tol_m or self.mode_s >= c.rise_timeout_s:
            return True
        if self.mode_s > 1.5 and v[2] < c.stall_speed_m_s:   # not climbing any more: roof / canopy above
            self.stall_s += self.dt
        else:
            self.stall_s = 0.
        return self.stall_s >= c.stall_s

    def _rise_sticks(self, est, top):
        c, k = self.cfg, self.ctl
        p, v = est.p, est.v
        v_des = np.zeros(3)
        v_des[:2] = np.clip(0.5 * (self.p0[:2] - p[:2]), -2., 2.)            # brake the start speed, hold x, y
        v_des[2] = np.clip(k.alt_kp * (top - p[2]), -2., c.rise_speed_m_s)
        a = np.zeros(3)
        a[:2] = k.vel_gain * (v_des[:2] - v[:2])
        a[2] = k.alt_kd * (v_des[2] - v[2])
        yaw_stick, _ = self.controller.yaw_stick_toward(p, est.yaw, np.array([0., 0., p[2]]))   # face the centre
        return self.controller.accel_to_sticks(a, v, est.yaw, yaw_stick)

    # ------------------------------------------------------------------ dive
    def _dive_sticks(self, est):
        """Attitude set directly: fixed nose-down pitch (eased past the speed cap), roll cancels the sideways
        speed, collective sets the descent rate, nose on the arena centre."""
        c, k = self.cfg, self.ctl
        p, v, yaw = est.p, est.v, est.yaw
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        speed = float(np.linalg.norm(v[:2]))
        pitch = np.radians(c.dive_pitch_deg) - c.dive_speed_gain * max(0., speed - c.dive_max_speed_m_s)
        pitch = float(np.clip(pitch, 0., np.radians(c.dive_pitch_deg)))
        vz_des = np.clip(k.alt_kp * (c.dive_end_alt_m - p[2]), -c.dive_descent_m_s, 0.)
        a_up = max(self.g + k.alt_kd * (vz_des - v[2]), 0.3 * self.g)
        a_left = -k.vel_gain * float(v[:2] @ left)
        roll = float(np.clip(-np.arctan2(a_left, a_up), -np.radians(c.max_roll_deg), np.radians(c.max_roll_deg)))
        throttle = a_up / (np.cos(pitch) * np.cos(roll)) / self.g - 1.
        yaw_stick, _ = self.controller.yaw_stick_toward(p, yaw, np.array([0., 0., p[2]]))
        full = np.radians(k.max_stick_deg)
        return np.clip(np.array([roll / full, pitch / full, yaw_stick, throttle]), -1., 1.).astype(np.float32)
