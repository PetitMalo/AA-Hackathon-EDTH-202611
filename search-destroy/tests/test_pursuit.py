"""Camera-locked pursuit of the box (target.py TargetPursuit). Synthetic states only, no simulator."""

from types import SimpleNamespace

import numpy as np

from handin.explorer.camera_model import CameraModel
from handin.explorer.config import CameraConfig, ControlConfig, PursuitConfig, TargetConfig
from handin.explorer.controller import WaypointController
from handin.explorer.target import TargetPursuit, TargetTracker
from tests.test_corridor_mpc import free_map

BOX = np.array([0., 0., .78])


def make(cfg=None):
    cam, ctl = CameraConfig(), ControlConfig()
    return TargetPursuit(cfg or PursuitConfig(), cam, ctl, WaypointController(ctl, cam)), CameraModel(cam)


def level_cam(camera, p, yaw, pitch=0.):
    """Camera pose of the drone at p, heading yaw, nose-down pitch."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    cp, sp = np.cos(pitch), np.sin(pitch)
    Ry = np.array([[cp, 0., sp], [0., 1., 0.], [-sp, 0., cp]])      # nose down for pitch > 0
    return camera.pose(Rz @ Ry, np.asarray(p, float))


def detection(camera, cam_pos, cam_R, point, half=0.01):
    v = cam_R.T @ (np.asarray(point) - cam_pos)
    cc = CameraConfig()
    f = (cc.depth_width / 2) / camera.tan_half_h
    x = (cc.depth_width / 2 - f * v[1] / v[0]) / cc.depth_width
    y = (cc.depth_height / 2 - f * v[2] / v[0]) / cc.depth_height
    return {"detected": 1, "box": np.array([x - half, y - half, x + half, y + half], np.float32)}


def tracker(seen_ago=0.):
    return SimpleNamespace(estimate=BOX.copy(), time_since_seen_s=seen_ago, box_voxels=set())


def est(p, yaw, v=(0., 0., 0.)):
    return SimpleNamespace(p=np.asarray(p, float), v=np.asarray(v, float), yaw=yaw)


def far_depth():
    return np.full((126, 224), 40., np.float32)


def test_sight_from_the_image_matches_the_geometry():
    tp, camera = make()
    p = np.array([60., 20., 12.])
    cam_pos, cam_R = level_cam(camera, p, np.pi, pitch=np.radians(10.))
    az, delta, seen = tp.sight(detection(camera, cam_pos, cam_R, BOX), cam_R, camera, p, BOX)
    e = BOX - cam_pos
    assert seen
    assert abs(az - np.arctan2(e[1], e[0])) < np.radians(1.)
    assert abs(delta - np.arctan2(-e[2], np.linalg.norm(e[:2]))) < np.radians(1.)


def test_goal_follows_the_line_of_sight_and_never_low_far_away():
    tp, camera = make()
    p = np.array([100., 0., 30.])
    cam_pos, cam_R = level_cam(camera, p, np.pi)
    plan = tp.update(est(p, np.pi), tracker(), {"detected": 0, "box": np.zeros(4)}, far_depth(), cam_pos, cam_R,
                     camera, free_map())
    assert plan.mode == "target"
    assert np.isclose(plan.goal[2], BOX[2] + 100. * np.tan(np.radians(10.)), atol=.1)   # ~18.4 m, not 3 m
    assert np.allclose(plan.goal[:2], BOX[:2])


def test_image_correction_lowers_the_goal_when_the_box_looks_too_steep():
    tp, camera = make()
    p = np.array([50., 0., 20.])                                        # sight ~21 deg: above the 10 deg line
    cam_pos, cam_R = level_cam(camera, p, np.pi, pitch=np.radians(20.))
    plan = tp.update(est(p, np.pi), tracker(), detection(camera, cam_pos, cam_R, BOX), far_depth(), cam_pos,
                     cam_R, camera, free_map())
    assert plan.goal[2] < BOX[2] + 50. * np.tan(np.radians(10.))


def test_nose_locked_on_the_box_and_pitch_floor_keeps_it_in_view():
    tp, camera = make()
    p = np.array([40., 30., 8.])
    cam_pos, cam_R = level_cam(camera, p, 0.)
    plan = tp.update(est(p, 0.), tracker(), {"detected": 0, "box": np.zeros(4)}, far_depth(), cam_pos, cam_R,
                     camera, free_map())
    d = plan.look[:2] - p[:2]
    assert abs(np.arctan2(d[1], d[0]) - np.arctan2(-30., -40.)) < 1e-6          # look toward the box
    delta = np.arctan2(p[2] - BOX[2], 50.)
    # with the floor, the box sits view_margin_deg above the bottom edge of the image
    up, half = np.radians(25.), tp.half_vfov
    assert np.isclose(plan.pitch_min, delta - (half - np.radians(tp.cfg.view_margin_deg)) + up)
    assert plan.pitch_min > 0.                                                   # no nose-up braking allowed


def test_no_pitch_floor_only_when_the_corridor_is_blocked():
    tp, camera = make()
    p = np.array([40., 0., 8.])
    cam_pos, cam_R = level_cam(camera, p, np.pi)
    none = {"detected": 0, "box": np.zeros(4)}
    plan = tp.update(est(p, np.pi), tracker(), none, far_depth(), cam_pos, cam_R, camera, free_map(),
                     path_clear=False)
    assert plan.pitch_min is None and not tp.floor_on                            # obstacles first
    plan = tp.update(est(p, np.pi), tracker(), none, far_depth(), cam_pos, cam_R, camera, free_map(),
                     path_clear=True)                                            # T1: truncated corridor is fine
    assert plan.pitch_min is not None and tp.floor_on


def test_strike_on_the_line_of_sight_then_back_off_when_lost():
    tp, camera = make()
    c = tp.cfg
    p = np.array([10., 0., 2.5])
    cam_pos, cam_R = level_cam(camera, p, np.pi, pitch=np.radians(6.))
    plan = tp.update(est(p, np.pi, v=(-8., 0., -1.4)), tracker(), detection(camera, cam_pos, cam_R, BOX),
                     far_depth(), cam_pos, cam_R, camera, free_map())
    assert plan.mode == "strike" and plan.action is not None and tp.n_strikes == 1
    full = np.radians(ControlConfig().max_stick_deg)
    assert plan.action[1] * full >= tp.pitch_floor(np.arctan2(p[2] - BOX[2], 10.)) - 1e-6
    p2 = np.array([8., 0., 2.2])
    cam_pos, cam_R = level_cam(camera, p2, np.pi)
    plan = tp.update(est(p2, np.pi, v=(-8., 0., -1.4)), tracker(seen_ago=c.lost_abort_s + .1),
                     {"detected": 0, "box": np.zeros(4)}, far_depth(), cam_pos, cam_R, camera, free_map())
    assert plan.mode == "brake" and tp.strike_abort == "lost"                   # S3: brake along the line
    roll, pitch, _, throttle = plan.action
    assert pitch < 0. and abs(roll) < 0.05                                       # braking, no swerve


def test_committed_last_metres_keep_striking():
    tp, camera = make()
    tp.mode, tp.commit_az = "strike", np.pi
    p = np.array([2., 0., 1.1])
    cam_pos, cam_R = level_cam(camera, p, np.pi)
    plan = tp.update(est(p, np.pi, v=(-8., 0., -1.)), tracker(seen_ago=5.), {"detected": 0, "box": np.zeros(4)},
                     far_depth(), cam_pos, cam_R, camera, free_map())
    assert plan.mode == "strike"


def test_strike_steers_with_the_image_not_the_offset_estimate():
    """S1: the estimate is 2 m to the side of the real box; the strike follows the box seen in the image."""
    tp, camera = make()
    real = np.array([0., 2., .78])
    p = np.array([10., 2., 2.5])                                                 # lined up with the real box
    cam_pos, cam_R = level_cam(camera, p, np.pi, pitch=np.radians(6.))
    plan = tp.update(est(p, np.pi, v=(-8., 0., 0.)), tracker(), detection(camera, cam_pos, cam_R, real),
                     far_depth(), cam_pos, cam_R, camera, free_map())
    assert plan.mode == "strike"
    roll, _, yaw, _ = plan.action
    assert abs(yaw) < 0.05 and abs(roll) < 0.1                                   # straight on, not toward y = 0


def test_box_face_filling_the_view_is_not_an_obstacle():
    """S2: close up, the box face fills the depth image inside the detection box."""
    tp, camera = make()
    p = np.array([5., 0., 1.2])
    cam_pos, cam_R = level_cam(camera, p, np.pi)
    depth = far_depth()
    depth[40:126, 30:194] = 4.2                                                  # the face, 4.2 m ahead
    obs = {"detected": 1, "box": np.array([30 / 224, 40 / 126, 193 / 224, 125 / 126], np.float32)}
    aim = np.array([0., 0., .78])
    assert tp._obstacles(obs, depth, cam_pos, cam_R, camera, p, aim, BOX, None, None) == 0
    depth[60:90, 100:124] = 2.                                                   # something in front of it
    assert tp._obstacles(obs, depth, cam_pos, cam_R, camera, p, aim, BOX, None, None) >= tp.cfg.min_obstacle_pts


def test_reacquire_backs_off_onto_the_line_of_sight():
    """T3: box not seen, far too steep below (drone high above it): goal moved back on the axis."""
    tp, camera = make()
    p = np.array([10., 0., 20.])
    cam_pos, cam_R = level_cam(camera, p, np.pi)
    plan = tp.update(est(p, np.pi), tracker(seen_ago=2.), {"detected": 0, "box": np.zeros(4)}, far_depth(),
                     cam_pos, cam_R, camera, free_map())
    assert tp.reacquire and plan.pitch_min is None
    assert plan.goal[0] > p[0] and plan.goal[2] <= p[2]


def test_confident_target_kept_25_s_and_dismissed_when_absent():
    """T2."""
    cfg = TargetConfig()
    tr = TargetTracker(cfg, CameraConfig(), 38.)
    tr.estimate, tr.n_detections, tr.time_since_seen_s = BOX.copy(), 3, 10.
    assert not tr.active                                                         # few detections: 6 s
    tr.n_detections = cfg.confident_detections
    assert tr.confident and tr.active                                            # well localised: 25 s
    tr.time_since_seen_s = cfg.lost_timeout_confident_s + 1.
    assert not tr.active
    tr.time_since_seen_s = 5.
    for _ in range(int(cfg.dismiss_s * 30) + 2):                                 # right there, nothing seen
        tr.check_presence(np.array([3., 0., 5.]))
    assert tr.estimate is None and not tr.active
