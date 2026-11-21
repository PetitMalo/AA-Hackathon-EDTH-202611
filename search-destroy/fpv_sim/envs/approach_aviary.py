"""Synthetic tagged-object search with a target-contact goal."""

import numpy as np
import pybullet as p
from gymnasium import spaces
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from fpv_sim.control.angle_control import AngleControl
from fpv_sim.envs.terrain_aviary import TerrainAviary
from fpv_sim.utils.perception import Camera, Detector, DepthModel, IMU, VIO
from fpv_sim.utils.terrain import terrain_start_pose


class ApproachAviary(TerrainAviary):
    """30 Hz sensor and RC-stick interface"""

    def __init__(self, continuous=False, terrain="village@v1", start_pose=None):
        """Initialization of an approach aviary.

        Parameters
        ----------
        continuous : bool, optional
            Whether the episode continues after contact, instead of terminating.
        terrain : str, optional
            A versioned terrain, "<name>@v<N>" from `fpv_sim/assets/terrains`.
        start_pose : dict, optional
            An explicit drone start pose (`position_m`, `rpy_rad`); when None and the terrain is an
            attack `level_*` map, it is read from the setup, so the target starts in the camera's sight.
            Otherwise the drone starts at (0, 0, 3) level, as in the classic search scenes.

        """
        self.seen = False
        self.continuous = continuous
        pose = start_pose if start_pose is not None else terrain_start_pose(terrain)
        if pose is not None:
            initial_xyzs = np.atleast_2d(np.asarray(pose['position_m'], dtype=float))
            initial_rpys = np.atleast_2d(np.asarray(pose['rpy_rad'], dtype=float))
        else:
            initial_xyzs, initial_rpys = np.array([[0., 0., 3.]]), np.zeros((1, 3))
        super().__init__(drone_model=DroneModel.RACE, physics=Physics.PYB,
                         pyb_freq=240, ctrl_freq=240, terrain=terrain, gui=False,
                         user_debug_gui=False, initial_xyzs=initial_xyzs, initial_rpys=initial_rpys)
        self.ctrl = AngleControl(drone_model=DroneModel.RACE)
        self.camera = Camera(self.CLIENT, width=160, height=90)
        self.detector = Detector()
        self.depth_camera = Camera(self.CLIENT, width=224, height=126)
        self.depth_model = DepthModel(self.depth_camera)
        self.vio = VIO()
        self.imu = IMU()

    def reset(self, seed=None, options=None):
        self.seen = False
        super().reset(seed=seed, options=options)
        self.last_box = None
        self.vio.reset(self._getDroneStateVector(0))
        self.imu.reset()
        return self._sense(), self._computeInfo()

    def step(self, action):
        action = np.asarray(action, dtype=float)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("action must contain four finite RC-stick values")
        sticks = np.clip(action, -1., 1.)
        reward = 0.
        for _ in range(8):
            _, r, terminated, truncated, _ = super().step(sticks)
            reward += r
            if terminated or truncated or self._success():
                break
        obs = self._sense()
        success = self._success()
        return obs, 1. if success else reward, terminated or (success and not self.continuous), truncated, self._computeInfo()

    def _sense(self):
        state = self._getDroneStateVector(0)
        _, seg = self.camera.read(state, segmentation=True)
        box = self.detector.detect(seg, self.TERRAIN_IDS[0], self._distance(), self.np_random)
        self.last_box = box
        self.seen |= box is not None
        if box is None:
            norm_box = np.zeros(4, dtype=np.float32)
        else:
            norm_box = (np.array(box, dtype=np.float32) / [160, 90, 160, 90]).astype(np.float32)
        depth_m = self.depth_model.predict(self.depth_camera.read_depth_buffer(state), self.np_random)
        gyro, accel = self.imu.read(state, 1 / 30, self.np_random)
        return {"detected": int(box is not None), "box": norm_box,
                "depth_m": depth_m, "gyro_rad_s": gyro.astype(np.float32),
                "accel_m_s2": accel.astype(np.float32), "vio_xyz_m": self.vio.read(state, self.np_random)}

    def _distance(self):
        body = self.TERRAIN_IDS[0]
        center, quat = p.getBasePositionAndOrientation(body, physicsClientId=self.CLIENT)
        distances = []
        for shape in p.getCollisionShapeData(body, -1, physicsClientId=self.CLIENT):
            shape_center, shape_quat = p.multiplyTransforms(center, quat, shape[5], shape[6])
            half = np.asarray(shape[3]) / 2
            local = (self.pos[0] - shape_center) @ np.asarray(p.getMatrixFromQuaternion(shape_quat)).reshape(3, 3)
            distances.append(np.linalg.norm(np.maximum(np.abs(local) - half, 0)))
        return float(min(distances))

    def _actionSpace(self):
        return spaces.Box(-1, 1, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        return spaces.Dict({
            "detected": spaces.Discrete(2),
            "box": spaces.Box(0, 1, shape=(4,), dtype=np.float32),
            "depth_m": spaces.Box(0, 40, shape=(126, 224), dtype=np.float32),
            "gyro_rad_s": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "accel_m_s2": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            "vio_xyz_m": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
        })

    def _success(self):
        return bool(p.getContactPoints(self.DRONE_IDS[0], self.TERRAIN_IDS[0], physicsClientId=self.CLIENT)) and not self._collision()

    def _collision(self):
        return self.pos[0, 2] < .2 or any(p.getContactPoints(self.DRONE_IDS[0], body,
                                                              physicsClientId=self.CLIENT)
                                              for body in self.TERRAIN_IDS[1:])

    def _preprocessAction(self, action):
        roll, pitch, yaw, throttle = action
        rpm = self.ctrl.computeControlFromState(
            control_timestep=self.PYB_TIMESTEP, state=self._getDroneStateVector(0),
            throttle=self.ctrl.HOVER_THROTTLE * (1 + throttle),
            target_rp=np.radians(60) * np.array([roll, pitch]),
            target_yaw_rate=-np.radians(200) * yaw)
        return rpm.reshape(1, 4)

    def _computeObs(self):
        return None

    def _computeReward(self):
        return 1. if self._success() else 0.

    def _computeTerminated(self):
        return not self.continuous and (self._success() or self._collision() or
                                        np.any(np.abs(self.pos[0, :2]) >= 350))

    def _computeTruncated(self):
        return not self.continuous and self.step_counter >= 240 * 60

    def _computeInfo(self):
        return {"is_success": bool(self._success()), "collision": bool(self._collision()),
                "distance_m": self._distance()}
