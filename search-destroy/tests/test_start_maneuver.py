"""Start of the flight (start.py): rise 50 m above the start, then a nose-down dive toward the centre."""

from types import SimpleNamespace

import numpy as np

from handin.explorer.config import CameraConfig, ControlConfig, StartConfig
from handin.explorer.controller import WaypointController
from handin.explorer.start import StartManeuver

DT = 1. / 30.


def make(cfg=None):
    ctl = ControlConfig()
    return StartManeuver(cfg or StartConfig(), ctl, WaypointController(ctl, CameraConfig()), dt=DT)


def est(p, v=(0., 0., 0.), yaw=0.):
    return SimpleNamespace(p=np.asarray(p, float), v=np.asarray(v, float), yaw=yaw)


def test_rise_holds_position_and_climbs_50_m_above_the_start():
    sm = make()
    mode, goal, sticks = sm.step(est([200., 0., 12.], v=(-5., 0., 0.), yaw=np.pi))
    assert mode == "rise" and np.isclose(goal[2], 12. + 50.)
    assert sticks[3] > 0.                                          # more thrust: climbing
    assert sticks[1] < 0.                                          # nose up: braking the start speed (along the heading)


def test_rise_then_dive_nose_down_toward_the_centre():
    sm = make()
    c = sm.cfg
    sm.step(est([200., 0., 12.], yaw=np.pi))
    mode, goal, sticks = sm.step(est([200., 0., 12. + c.rise_m], yaw=np.pi))   # at the top
    assert mode == "dive"
    full = np.radians(ControlConfig().max_stick_deg)
    assert np.isclose(sticks[1] * full, np.radians(c.dive_pitch_deg))          # nose down, camera ~20 deg below
    assert abs(sticks[2]) < 1e-6                                                # already facing the centre


def test_dive_eases_the_pitch_past_the_speed_cap():
    sm = make()
    c = sm.cfg
    sm.mode, sm.p0 = "dive", np.array([200., 0., 12.])
    slow = sm.step(est([150., 0., 50.], v=(-10., 0., -3.), yaw=np.pi))[2]
    fast = sm.step(est([150., 0., 50.], v=(-(c.dive_max_speed_m_s + 6.), 0., -3.), yaw=np.pi))[2]
    assert fast[1] < slow[1]


def test_dive_ends_above_the_rooftops_or_near_the_centre():
    sm = make()
    c = sm.cfg
    sm.mode, sm.p0 = "dive", np.array([200., 0., 12.])
    assert sm.step(est([150., 0., c.dive_end_alt_m - .5], yaw=np.pi)) is None and sm.done
    sm2 = make()
    sm2.mode, sm2.p0 = "dive", np.array([200., 0., 12.])
    assert sm2.step(est([10., 0., 50.], yaw=np.pi)) is None and sm2.done


def test_rise_stops_when_blocked_above():
    sm = make()
    c = sm.cfg
    out = None
    for _ in range(int((1.5 + c.stall_s) / DT) + 5):               # not climbing (roof / canopy)
        out = sm.step(est([200., 0., 12.], yaw=np.pi))
    assert out is not None and out[0] == "dive"


def test_abort_on_target():
    sm = make()
    sm.step(est([200., 0., 12.], yaw=np.pi))
    sm.abort()
    assert sm.done and sm.step(est([200., 0., 20.])) is None


def test_disabled():
    assert make(StartConfig(enabled=False)).step(est([200., 0., 12.])) is None
