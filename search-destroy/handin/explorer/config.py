"""All tunable parameters of the exploration stack, in one place."""

from dataclasses import dataclass, field


@dataclass
class CameraConfig:
    """Matches fpv_sim.utils.perception.Camera as used by ApproachAviary (do not read the sim, mirror it)."""
    depth_width: int = 224
    depth_height: int = 126
    det_width: int = 160
    det_height: int = 90
    hfov_deg: float = 102.
    uptilt_deg: float = 25.
    mount_body_m: tuple = (0.30, 0., 0.)
    max_depth_m: float = 40.           # depth is clipped here: a pixel at 40 m means "no return"


@dataclass
class MapConfig:
    voxel_m: float = 2.
    x_range_m: tuple = (-300., 300.)
    y_range_m: tuple = (-300., 300.)
    z_range_m: tuple = (0., 40.)
    depth_stride: int = 4              # use 1 pixel out of N in each direction for ray casting
    max_ray_m: float = 38.             # depth beyond this is treated as free space, no hit
    hit_logodds: int = 3
    miss_logodds: int = -1
    logodds_clamp: int = 20


@dataclass
class FrontierConfig:
    recompute_every: int = 15          # frames (0.5 s at 30 Hz)
    z_band_m: tuple = (3., 20.)        # only frontiers in this altitude band are candidates
    xy_limit_m: float = 240.           # stay well inside the 250 m floor / 350 m arena
    min_dist_m: float = 8.             # ignore frontiers right next to the drone
    reach_dist_m: float = 5.           # goal considered reached
    commit_s: float = 8.               # max time spent on one frontier before re-planning
    center_weight: float = 0.5         # cost = d(drone) + center_weight * d(centre); 0 -> pure nearest


@dataclass
class TargetConfig:
    box_center_z_m: float = 0.78       # the 1.55 m high box rests on the ground
    ema: float = 0.3                   # smoothing of successive target estimates
    lost_timeout_s: float = 6.         # keep chasing the last estimate this long without detection


@dataclass
class ControlConfig:
    cruise_speed_m_s: float = 8.
    approach_speed_m_s: float = 6.
    speed_gain: float = 0.5            # v_des = min(v_max, gain * distance)
    vel_gain: float = 1.5              # 1/s, velocity error -> acceleration
    max_tilt_deg: float = 35.
    max_stick_deg: float = 60.         # full stick = 60 deg (ApproachAviary)
    yaw_gain: float = 0.8              # stick per rad of heading error
    alt_kp: float = 1.2
    alt_kd: float = 1.0
    top_speed_m_s: float = 119.5 / 3.6 # TerrainAviary drag model: used as a feed-forward
    top_speed_pitch_deg: float = 60.
    safety_depth_m: float = 5.         # reactive layer: obstacle closer than this in front -> brake + climb


@dataclass
class ExplorerConfig:
    dt: float = 1. / 30.
    g: float = 9.8
    camera: CameraConfig = field(default_factory=CameraConfig)
    map: MapConfig = field(default_factory=MapConfig)
    frontier: FrontierConfig = field(default_factory=FrontierConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
