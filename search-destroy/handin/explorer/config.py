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
    policy: str = "gain_cost"          # "gain_cost" (GainCostFrontierPolicy) | "nearest" (NearestFrontierPolicy)
    blacklist_s: float = 30.           # a goal abandoned on commit timeout is excluded this long...
    blacklist_radius_m: float = 10.    # ... with every frontier this close to it
    # GainCostFrontierPolicy
    cluster_m: float = 10.             # frontier binning cell (several voxels per cell)
    min_cluster: int = 5               # frontier points per cluster (no crumbs)
    max_candidates: int = 40           # clusters ray-cast, pre-selected by size * exp(-lam d)
    n_rays: int = 128                  # rays per viewpoint (Fibonacci sphere)
    gain_range_m: float = 15.          # ray length for the information gain
    lam: float = 0.08                  # 1/m, distance discount exp(-lam d)
    heading_w: float = 0.5             # U-turn penalty (0 = off)
    hysteresis: float = 0.25           # switch goal only if the new one is 25 % better
    prior_sigma_m: float = 100.        # Gaussian prior on the target around the arena centre


@dataclass
class TargetConfig:
    box_center_z_m: float = 0.78       # the 1.55 m high box rests on the ground
    ema: float = 0.3                   # smoothing of successive target estimates
    lost_timeout_s: float = 6.         # keep chasing the last estimate this long without detection


@dataclass
class ControlConfig:
    cruise_speed_m_s: float = 8.
    approach_speed_m_s: float = 12.         # target phase, far from the box
    approach_glide_slope: float = 0.15      # descend toward the box over a long, shallow ramp (target_z + slope*dist)
    # Terminal approach: the speed cap follows a braking profile down to the terminal speed at
    # terminal_radius_m from the box estimate: v = sqrt(v_term^2 + 2 * brake * (d - terminal_radius)).
    terminal_radius_m: float = 15.
    terminal_brake_m_s2: float = 4.
    terminal_speed_m_s: float = 5.          # clear surroundings
    terminal_speed_min_m_s: float = 3.      # cluttered surroundings
    clutter_radius_m: float = 4.            # obstacles (connected groups of occupied voxels) counted this close to the drone
    clutter_free_count: int = 2             # up to this many obstacles: terminal_speed_m_s
    clutter_max_count: int = 4              # this many or more: terminal_speed_min_m_s (linear in between)
    clutter_ground_m: float = 2.            # voxels below this height are the ground, not obstacles
    clutter_box_clear_m: float = 3.         # voxels this close to the box estimate are the box itself, not obstacles
    speed_gain: float = 0.5            # v_des = min(v_max, gain * distance)
    vel_gain: float = 1.5              # 1/s, velocity error -> acceleration
    max_tilt_deg: float = 50.          # TOTAL tilt (roll and pitch combined), the horizontal acceleration is
                                       # scaled down along its own direction when it would need more
    max_stick_deg: float = 60.         # full stick = 60 deg (ApproachAviary)
    yaw_gain: float = 0.8              # stick per rad of heading error
    alt_kp: float = 1.2
    alt_kd: float = 1.0
    top_speed_m_s: float = 119.5 / 3.6 # TerrainAviary drag model: used as a feed-forward
    top_speed_pitch_deg: float = 60.
    safety_depth_m: float = 5.         # reactive layer: obstacle closer than this in front -> brake + climb


@dataclass
class AttackConfig:
    """Terminal run: once close to the target estimate and low enough, skip the corridor/MPC and fly a
    near-max-acceleration straight line at the box (contact at any speed = success)."""
    engage_dist_m: float = 12.0       # engage the attack below this range to the box estimate
    engage_max_alt_m: float = 6.0      # only sprint once the approach glide slope has brought us this low
    settle_dist_m: float = 6.0        # inside this: cut horizontal accel, nose down + slight dive onto the box
    lost_timeout_s: float = 1.         # require a sighting fresher than this (no sprint on stale memory)
    accel_m_s2: float = 12.            # horizontal acceleration toward the box (matches the tilt cap)
    tilt_deg: float = 50.              # mild tilt cap: keeps some margin to correct the aim late
    box_top_clamp_m: float = 1.5      # never aim below the box top (z ~= 1.55 m): hit the side, not the ground
    aim_above_box_m: float = 0.2      # aim this far above the box estimate
    ground_layer_m: float = 2.        # ignore the bottom voxel layer in the path-clear check (it is the ground)
    clear_voxels_ahead: int = 0        # 0 = none; allow this many non-free voxels on the straight segment


@dataclass
class CorridorConfig:
    """Safe flight corridor: chain of free axis-aligned voxel boxes along the path (occupied OR unknown = not free)."""
    replan_every: int = 15             # frames (0.5 s at 30 Hz)
    goal_moved_m: float = 2.           # re-plan when the goal moves more than this
    outside_tol_m: float = 1.          # re-plan when the drone is this far outside every box
    max_boxes: int = 6
    max_length_m: float = 60.          # the path is truncated to this length
    max_box_size_m: float = 30.        # per axis
    margin_m: float = 0.8               # boxes shrunk by this for the MPC, away from the target (and when exploring)
    margin_near_m: float = 0.2          # ... and by this once the drone is within near_target_radius_m of the box
    near_target_radius_m: float = 5.
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
    w_term_vel_terminal: float = 0.1   # target phase: arrive at the box at speed, not stopped
    w_slack_quad: float = 200.         # soft box constraints
    w_slack_lin: float = 200.
    a_max_xy_m_s2: float = 12.         # ~ g tan(50 deg), the controller's tilt limit
    a_max_z_m_s2: float = 6.
    v_max_z_m_s: float = 5.
    v_max_z_terminal_m_s: float = 6.   # faster dive onto the box in the target phase
    v_max_margin: float = 1.2          # |v_xy| per axis <= margin * v_max of the current mode
    admm_rho: float = 0.5
    admm_sigma: float = 1e-6
    admm_alpha: float = 1.6
    admm_max_iter: int = 400
    admm_eps_abs: float = 1e-4
    admm_eps_rel: float = 1e-4


@dataclass
class GeodesicConfig:
    """Exploration only: shortest flyable path (geodesic.py) for the frontier score and the corridor."""
    enabled: bool = True
    cell_m: float = 4.                 # coarse cell = 2 x 2 x 2 voxels, traversable only if all are known free
    half_xy_m: float = 100.            # window around the drone; frontiers outside it keep the Euclidean distance
    z_range_m: tuple = (0., 32.)       # covers the escape climb (26 m) and the widened frontier band (30 m)
    clearance_weight: float = 1.       # edge cost x (1 + w / clearance in cells): stay off the walls
    ground_m: float = 2.               # occupied voxels below this are the ground: not counted in the clearance
    start_search_cells: int = 2        # drone cell not traversable (next to a wall): nearest one this close
    los_tube_m: float = 1.             # path shortcut only if this half-width tube around it is known free
    max_age_frames: int = 8            # a field is reused this long ...
    max_move_m: float = 4.             # ... if the drone moved less than this
    heading_look_m: float = 12.        # frontier heading term: direction of the first metres of the path
    look_ahead_s: float = 1.           # nose (and speed-while-turning) aimed this far ahead along the corridor ...
    look_ahead_min_m: float = 8.       # ... and at least this far


@dataclass
class EscapeConfig:
    """Exploration only: detect that the drone is stuck (city: frontiers behind buildings, blocked corridor,
    hover) and escape with a graded response: 1) re-plan with a wide blacklist, 2) yaw scan in place,
    3) climb above the rooftops (~20 m max in the city) and widen the frontier altitude band for a while."""
    enabled: bool = True
    # --- stuck detection (ProgressMonitor)
    window_s: float = 8.               # displacement measured over this sliding window
    min_disp_m: float = 8.             # net horizontal displacement over the window below this -> stuck
    blocked_speed_m_s: float = 1.      # corridor blocked AND horizontal speed below this ...
    blocked_s: float = 1.5             # ... for this long -> stuck (no need to wait for the frontier commit timeout)
    clutter_radius_m: float = 10.      # obstacles (connected voxel groups) counted this close to the drone
    clutter_count: int = 6             # this many or more = cluttered ...
    clutter_min_disp_m: float = 15.    # ... and then a displacement below this over the window -> stuck
    clutter_every: int = 15            # frames between two obstacle counts (cost)
    cooldown_s: float = 4.             # no new trigger this long after an escape step ends
    escalation_reset_s: float = 30.    # a trigger this long after the previous one starts again at level 1
    # --- level 1: re-plan
    blacklist_radius_m: float = 20.    # around the abandoned goal and around where the drone was stuck
    blacklist_s: float = 45.
    # --- level 2: yaw scan in place (holds position, turns the nose: the camera fills the unknown around)
    scan_s: float = 2.5
    scan_yaw_stick: float = 0.8        # 0.8 -> 160 deg/s, a full turn in ~2.3 s
    # --- level 3: climb
    climb_alt_m: float = 26.           # above the highest building (~20 m), below the map ceiling (40 m)
    climb_speed_m_s: float = 4.
    climb_tol_m: float = 1.5
    climb_timeout_s: float = 10.
    climb_stall_s: float = 2.          # vertical speed below climb_stall_speed this long (after 1 s) -> abort
    climb_stall_speed_m_s: float = 0.3
    column_half_width_m: float = 2.    # the column above the drone must hold no OCCUPIED voxel (roof, canopy)
    widen_z_band_m: tuple = (3., 30.)  # after a climb, frontiers in this band for widen_s
    widen_s: float = 20.


@dataclass
class ExplorerConfig:
    dt: float = 1. / 30.
    g: float = 9.8
    use_mpc: bool = True               # corridor + MPC guidance; False -> the original waypoint controller
    use_attack: bool = False          # close-range full-acceleration run at the box when the path is clear (EXPERIMENTAL: still overshoots, see ATTACK_NOTES.md)
    camera: CameraConfig = field(default_factory=CameraConfig)
    map: MapConfig = field(default_factory=MapConfig)
    frontier: FrontierConfig = field(default_factory=FrontierConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    corridor: CorridorConfig = field(default_factory=CorridorConfig)
    mpc: MPCConfig = field(default_factory=MPCConfig)
    escape: EscapeConfig = field(default_factory=EscapeConfig)
    geodesic: GeodesicConfig = field(default_factory=GeodesicConfig)
    # Evaluator flight phases: the mission latches from "exploration" to "terminal" the first time
    # the policy reports its target mode (see fpv_sim.scripts.evaluate).
    phase_of_mode: dict = field(default_factory=lambda: {"explore": "exploration", "hover": "exploration",
                                                         "scan": "exploration", "climb": "exploration",
                                                         "target": "terminal", "attack": "terminal"})
