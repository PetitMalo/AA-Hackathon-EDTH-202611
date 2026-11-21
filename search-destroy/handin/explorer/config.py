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
class CorridorConfig:
    """Safe flight corridor: chain of free axis-aligned voxel boxes along the path (occupied OR unknown = not free)."""
    replan_every: int = 15             # frames (0.5 s at 30 Hz)
    goal_moved_m: float = 2.           # re-plan when the goal moves more than this
    outside_tol_m: float = 1.          # re-plan when the drone is this far outside every box
    max_boxes: int = 6
    max_length_m: float = 60.          # the path is truncated to this length
    max_box_size_m: float = 30.        # per axis
    margin_m: float = 0.35              # boxes shrunk by this for the MPC (no obstacle inflation for now)
    target_clear_radius_m: float = 4.  # voxels this close to the target estimate count as free (contact = success)
    z_min_m: float = 1.                # altitude floor in the MPC
    reactive_safety: bool = False      # also run the old depth-image brake-and-climb layer on top of the MPC


@dataclass
class MPCConfig:
    horizon: int = 20                  # steps
    dt: float = 0.1                    # s, MPC model step (the agent itself runs at 30 Hz)
    w_pos: float = 1.                  # position tracking
    w_acc: float = 0.02                # acceleration effort
    w_jerk: float = 0.3                # acceleration change (smooth attitude)
    w_term_vel: float = 2.             # terminal velocity -> 0: the plan can always stop inside the corridor
    w_slack_quad: float = 200.         # soft box constraints
    w_slack_lin: float = 200.
    a_max_xy_m_s2: float = 6.5         # ~ g tan(35 deg), the controller's tilt limit
    a_max_z_m_s2: float = 4.
    v_max_z_m_s: float = 3.
    v_max_margin: float = 1.2          # |v_xy| per axis <= margin * v_max of the current mode
    admm_rho: float = 0.5
    admm_sigma: float = 1e-6
    admm_alpha: float = 1.6
    admm_max_iter: int = 400
    admm_eps_abs: float = 1e-4
    admm_eps_rel: float = 1e-4


@dataclass
class ExplorerConfig:
    dt: float = 1. / 30.
    g: float = 9.8
    use_mpc: bool = True               # corridor + MPC guidance; False -> the original waypoint controller
    camera: CameraConfig = field(default_factory=CameraConfig)
    map: MapConfig = field(default_factory=MapConfig)
    frontier: FrontierConfig = field(default_factory=FrontierConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
