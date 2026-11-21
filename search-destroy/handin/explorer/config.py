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
    sky_ceiling_m: float = 24.         # exploration only: unknown voxels above count as already seen (no frontier,
                                       # no gain), the target is on the ground; the corridor still treats them unknown
    hit_logodds: int = 3
    miss_logodds: int = -1
    logodds_clamp: int = 20


@dataclass
class FrontierConfig:
    recompute_every: int = 15          # frames (0.5 s at 30 Hz): "goal explored?" check
    periodic_every: int = 45           # frames (1.5 s): periodic re-scoring of the goal (GainCostFrontierPolicy)
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
    heading_w: float = 0.8             # U-turn penalty (0 = off)
    hysteresis: float = 0.5            # switch goal only if the new one is 50 % better
    continuity_w: float = 0.5          # goal just explored: candidates near it get up to +50 % ...
    continuity_m: float = 15.          # ... decaying with the distance to it (exp(-d / continuity_m))
    ground_full_m: float = 4.          # gain: unknown voxels weigh 1 below this altitude (where the box can be),
                                       # then linearly less, 0 at map.sky_ceiling_m
    uturn_deg: float = 120.            # stats: a goal switch this far round from the previous one is a U-turn
    prior_sigma_m: float = 100.        # Gaussian prior on the target around the arena centre


@dataclass
class TargetConfig:
    box_center_z_m: float = 0.78       # the 1.55 m high box rests on the ground
    ema: float = 0.3                   # smoothing of successive target estimates
    lost_timeout_s: float = 6.         # keep chasing the last estimate this long without detection ...
    lost_timeout_confident_s: float = 25.  # ... this long for a well localised box (box voxels known, or ...
    confident_detections: int = 10     # ... detected this many times): the camera is kept on it meanwhile
    dismiss_dist_m: float = 8.         # drone this close (horizontally) to the estimate ...
    dismiss_s: float = 3.              # ... without seeing the box for this long: the estimate was wrong, forget it
    # box voxels: the only voxels the corridor treats as free near the target (everything else stays an obstacle)
    box_depth_tol_m: float = 1.5       # bbox pixels within this of the box depth are the box (background is farther)
    box_voxel_max_range_m: float = 30. # only from depth closer than this (noise 3 % of the range)
    box_max_z_m: float = 2.            # the box is 1.55 m high
    box_max_dist_m: float = 3.         # and within this (horizontally) of the estimate
    box_reset_m: float = 5.            # the estimate jumped this far: forget the box voxels
    max_box_voxels: int = 40


@dataclass
class PursuitConfig:
    """Camera-locked pursuit of the box (target.py TargetPursuit): nose on the box at all times, straight line
    of sight to its centre at los_deg below the horizon, pitch floor so the box never leaves the image."""
    los_deg: float = 10.               # line of sight: altitude = box_z + d_h tan(los): 18 m at 100 m, 4.3 m at 20 m
    los_gain: float = 0.5              # C3: goal altitude correction from the sight angle measured in the image
    min_height_m: float = 0.5          # goal never lower than this above the box centre
    pitch_floor: bool = False          # C4 pitch floor (target + strike). OFF: it forced the nose down, the MPC
                                       # could not brake, the drone reached the box at 10-15 m/s and overshot it
    view_margin_deg: float = 5.        # C4: keep the box this far above the bottom edge of the image
    look_dist_m: float = 10.           # C1: nose aimed at a point this far along the box azimuth
    strike_speed_m_s: float = 8.       # contact speed (the drag + C4 floor at 10 deg settle there)
    brake_m_s2: float = 1.             # speed cap sqrt(v_strike^2 + 2 brake (d - strike_dist)): gentle, mostly drag
    strike_dist_m: float = 12.         # strike: own controller on the line of sight from this horizontal distance ...
    strike_fresh_s: float = 0.5        # ... box detected less than this ago ...
    strike_gain: float = 2.            # 1/s, velocity error -> acceleration
    aim_z_m: float = 0.78              # S1: strike aimed at the box centre height, not the estimate's z ...
    min_aim_z_m: float = 0.6           # ... and never lower (the ground limit is 0.2 m)
    lost_abort_s: float = 1.           # strike given up when the box is not seen for this long ...
    obstacle_frames: int = 3           # ... or an obstacle is on the line this many frames in a row ...
    commit_dist_m: float = 5.          # ... except in the last metres: steering frozen, no giving up
    abort_brake_s: float = 1.5         # S3: giving up = brake along the line and climb abort_climb_m (no swerve),
    abort_climb_m: float = 2.          #     at least this long, then the corridor + MPC again
    # T3: box not seen and too steep below (too high / too close): back onto the line of sight
    reacq_after_s: float = 0.5
    reacq_excess_deg: float = 10.      # sight angle (from the estimate) above los_deg + this
    reacq_max_back_m: float = 30.
    # depth-image check of the strike line (low obstacles live in the map's ground layer)
    depth_stride: int = 3
    max_depth_m: float = 38.           # depth at or beyond this = no return
    tube_m: float = 0.8                # depth points this close to the drone -> box segment are obstacles ...
    ground_m: float = 0.35             # ... unless lower than this (ground, depth noise) ...
    box_clear_m: float = 1.5           # ... or the box itself: the tracker's box voxels; until they are known,
                                       # the points this close (horizontally) to the estimate ...
    bbox_pad_px: int = 3               # ... and (S2) the detection box pixels, padded by this, at the box depth ...
    box_depth_tol_m: float = 1.        # ... or behind it (minus this): close up, the box fills the view
    min_obstacle_pts: int = 3          # this many obstacle points (noise filter)


@dataclass
class ControlConfig:
    cruise_speed_m_s: float = 12.      # exploration speed cap; more speed = more pitch = camera lower
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
                                        # (only while the corridor reaches the box, see CorridorPlanner.step)
    near_target_radius_m: float = 5.
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
    w_term_vel_terminal: float = 0.1   # target phase: arrive at the box at speed, not stopped (only while the
                                       # corridor reaches the box, otherwise w_term_vel)
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
class StartConfig:
    """Start of every flight (start.py): rise above the start point, then a nose-down dive toward the centre,
    then the planning. A target detection at any time ends it."""
    enabled: bool = True
    rise_m: float = 50.                # climb this much above the start altitude (start 5-25 m -> 55-75 m)
    rise_speed_m_s: float = 8.
    rise_tol_m: float = 2.
    rise_timeout_s: float = 15.
    stall_speed_m_s: float = 0.5       # rise: vertical speed below this ...
    stall_s: float = 2.                # ... this long (after 1.5 s) -> something above, stop climbing
    dive_pitch_deg: float = 45.        # nose down: the camera (tilted 25 deg up) looks ~20 deg below the horizon
    dive_max_speed_m_s: float = 20.    # past this, the pitch is eased off (dive_speed_gain rad per m/s)
    dive_speed_gain: float = 0.05
    dive_descent_m_s: float = 6.
    dive_end_alt_m: float = 25.        # above the rooftops (~20 m max): planning starts here ...
    dive_stop_radius_m: float = 60.    # ... or this close to the arena centre ...
    dive_timeout_s: float = 15.        # ... or after this long
    max_roll_deg: float = 30.


@dataclass
class GeodesicConfig:
    """Exploration only: shortest flyable path (geodesic.py) for the frontier score and the corridor."""
    enabled: bool = True
    cell_m: float = 4.                 # coarse cell = 2 x 2 x 2 voxels, traversable only if all are known free
    half_xy_m: float = 100.            # window around the drone; frontiers outside it keep the Euclidean distance
    z_range_m: tuple = (0., 32.)       # covers the escape climb (26 m) and the widened frontier band (24 m)
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
    widen_z_band_m: tuple = (3., 24.)  # after a climb, frontiers in this band for widen_s (up to map.sky_ceiling_m)
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
    pursuit: PursuitConfig = field(default_factory=PursuitConfig)
    start: StartConfig = field(default_factory=StartConfig)
    # Evaluator flight phases: the mission latches from "exploration" to "terminal" the first time
    # the policy reports its target mode (see fpv_sim.scripts.evaluate).
    phase_of_mode: dict = field(default_factory=lambda: {"explore": "exploration", "hover": "exploration",
                                                         "scan": "exploration", "climb": "exploration",
                                                         "rise": "exploration", "dive": "exploration",
                                                         "target": "terminal", "attack": "terminal",
                                                         "strike": "terminal", "brake": "terminal"})
