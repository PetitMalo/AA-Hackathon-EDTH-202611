# Search & Destroy

>Russian armor hides and moves inside tree lines and ruins near the front. A Ukrainian ISR drone geolocates a BMP, then gets chased off by EW, so nobody is watching anymore. An FPV launches with stale, imprecise coordinates, and its video link dies to jamming on approach.
>
>The onboard guidance takes over. It treats the coordinates as a search zone, re-finds the BMP through gaps in the trees, flies around the ruins and under the canopy, and hits it with no pilot and no link.

![Trailer: FPV flight from high over the map down between buildings and trees, with depth-map cuts](trailer.gif)

To emulate this challenge, the simulated FPV arrives at the edge of an area where an object was last reported. It must **find the tagged object again and reach it**, without hitting anything else on the way. The object is an orange 4 × 1.65 × 1.55 m box hidden somewhere in a synthetic terrain: under trees, inside bushes, against walls, in alleys, or among similar-looking objects.

You write the flight policy. It sees what an onboard computer would see (a camera-based detector, a noisy depth map, an IMU and noisy odometry) and outputs RC stick commands at 30 Hz.

## The problem

- **Search area.** The object was last reported near the arena centre `(0, 0)`. That prior is always the same, so it is not passed to the agent. The search zone is the 700 × 700 m arena centred on it; the visible floor is 500 × 500 m, with ground at z = 0.
- **Start.** The drone enters 225–275 m from the centre, 5–25 m above ground, already flying at 0–10 m/s, heading within 20° of the centre.
- **Goal.** Touch the target box.
- **Failure.** Touching any other object, flying below 0.2 m altitude, leaving the arena (|x| or |y| ≥ 350 m), raising an exception, or running out of time (120 s of simulated flight).

## What you build

Implement `Agent.reset()` and `Agent.act(observation)` in [`handin/agent.py`](handin/agent.py). That file (plus any modules or weights it imports from `handin/`) is your submission.

- `reset()` is called before each flight. `act(observation)` is called once per 33 ms step and returns four RC stick values `(roll, pitch, yaw, throttle)` in `[-1, 1]`. The agent may keep state between calls.
- Use only the `observation` passed to `act`. Simulator internals, `info`, and the terrain files are for scoring and debugging, not policy input.
- Do not hard-code target positions or terrain layouts: the evaluation uses **terrains you have not seen** (see [Evaluation protocol](#evaluation-protocol)).
- Any method is welcome: classical guidance, mapping and planning, learning, or a mix.

The sample `Agent` hovers. It passes the API check but never succeeds.

## Hypotheses

What you can rely on, in the public terrains and in the hidden ones:

- The target is always the same orange 4 × 1.65 × 1.55 m box, resting on the ground, somewhere in the 700 × 700 m arena around `(0, 0)` (in practice on the 500 × 500 m floor).
- There is exactly one target. Terrains are static: nothing moves.
- The detector reports only the target box within 250 m of the drone, measured as 3D distance to the box surface. It never fires on other objects, but it misses a visible target 5% of the time and reports nothing when the target is out of view or fully occluded.
- Sensors, noise models, controller, start conditions, and rules are the same as in this repository.
- **Position comes from `vio_xyz_m` only.** The observation has no exact pose or attitude; use the IMU's body-frame angular rates and specific force to estimate motion.

## Limits

This is a simulator built for a weekend, not a flight-ready digital twin:

- **Perception is synthetic.** The detector reads the simulator's ground-truth segmentation and adds noise; it is not a neural network looking at pixels. The depth map is simulator depth plus Gaussian noise. Real cameras need different inputs and calibration.
- **Odometry is synthetic.** `VIO` adds drift to the true position; it does not process camera or IMU data.
- **Dynamics are approximate.** `AngleControl` is a PyBullet approximation, **not Betaflight**. Do not assume hardware-equivalent dynamics, tuning, or calibration. There is no wind, battery sag, motor vibration, or IMU bias drift.
- **Contact is the success criterion.** Speed at contact is not scored, and a hit at any speed counts.
- **Rendering is limited.** PyBullet's CPU renderer draws simple low-poly geometry. Props stop 150 m from the centre, except around box poses.
- **No geography.** Coordinates are local metres; there is no latitude/longitude.

## Evaluation protocol

Mentors run the same script you can run, [`fpv_sim/scripts/evaluate.py`](fpv_sim/scripts/evaluate.py), on **hidden terrains**: new layouts (possibly new terrain types) with new box placements, using the same box, sensors, controller, and rules. Scoring on the public terrains only tells you how you do on scenes you could tune for.

For each terrain version (one box placement), the script flies several runs:

1. A random clear start is drawn as described in [The problem](#the-problem). Starts are seeded, so a run is reproducible.
2. `Agent.reset()` is called, then `Agent.act(observation)` every step, with the agent in full command.
3. The flight ends on the first of:

   | Outcome | Condition |
   | --- | --- |
   | `success` | The drone touches the target box and nothing else |
   | `collision` | Contact with any other object, or altitude below 0.2 m |
   | `out_of_arena` | \|x\| or \|y\| ≥ 350 m |
   | `timeout` | 120 s of simulated flight |
   | `error` | `act` raised an exception or returned an invalid action |

The simulation waits for your agent, so a slow agent still gets every step. Its latency is measured and reported, and the script warns when the mean exceeds 33 ms (below 30 FPS). This is a warning, not a penalty, but a real drone does not wait.

**Judging.** There is no fixed formula. A jury ranks teams using the hidden-terrain metrics below, with success rate first, together with the robustness and soundness of the approach.

| Metric (`metrics.json`, per terrain type) | Meaning |
| --- | --- |
| `success_rate`, `collision_rate`, `timeout_rate`, `out_of_arena_rate`, `error_rate` | Share of flights per outcome |
| `exploration_time_s`, `terminal_time_s` | Mean time spent searching vs. committed to the target, all flights |
| `reach_time_s`, `reach_mean_speed_m_s` | Flight time and mean speed, successful flights only |
| `path_efficiency` | Start distance to the box ÷ path flown, successful flights only |
| `detection_rate`, `first_detection_s` | Share of flights in which the target was ever detected, and when |
| `failed_final_distance_m` | Distance to the box surface at the end of failed flights |
| `latency_mean_ms`, `agent_fps` | Mean time spent in `act`, and the matching frame rate |

`flights.csv` has one row per flight with the start state and outcome.

## Setup

From the repository root (Python 3.12, [`uv`](https://docs.astral.sh/uv/) and a CPU are sufficient):

```sh
git clone --recurse-submodules https://github.com/Alta-Ares/AA-Hackathon-EDTH-202611.git
cd AA-Hackathon-EDTH-202611/search-destroy
uv sync
uv run python -m handin.check_handin     # checks your Agent against the API
uv run --with pytest pytest -q tests
```

If you already cloned the repository, run `git submodule update --init --recursive` at its root before `uv sync`. On macOS, if PyBullet must build from source, run `CFLAGS="-Dfdopen=fdopen" uv sync`.

## Evaluate your agent

```sh
uv run python -m fpv_sim.scripts.evaluate                                # every terrain and box, 10 flights each
uv run python -m fpv_sim.scripts.evaluate --terrains field --runs 2      # quick check
```

Options: `--runs` (flights per box), `--terrains`, `--timeout` (s), `--workers` (parallel processes, default all CPUs), `--seed`, `--out`, and `--depth-every` (log one depth map in N frames; `1` logs every frame, about 130 MB a flight).

Results go to `out/<time>/`: `flights.csv`, `metrics.json`, and one `<terrain>@v<N>/runNN.mcap` per flight. MCAP files open in [Foxglove](https://foxglove.dev) and hold the observations, IMU, true state, and agent actions with their latency. They also carry two analysis topics:

| Topic | Content |
| --- | --- |
| `/mission/phase` | Phase events: `enter` (run start), `transition` (the flight's single commit from `exploration` to `terminal`), `policy_transition` (debug: every mode change the agent reports, including flickers), `final` (run end). Per-flight `exploration_time_s` / `terminal_time_s` in `flights.csv` split at the `transition` event. |
| `/analysis/target_relative` | Per-step target pose in the drone frame: `distance_m`, `bearing_deg` (left +), `elevation_deg` (up +), `lateral_offset_m`, `vertical_offset_m`. |

The phase switch is the policy's own: a flight latches into `terminal` the first time the agent reports its target mode; there is no distance threshold in the evaluator.

## Observation and action

```python
from fpv_sim.envs.approach_aviary import ApproachAviary

env = ApproachAviary(terrain="village@v1")
observation, info = env.reset(seed=1)
observation, reward, terminated, truncated, info = env.step([0, 0, 0, 0])
env.close()
```

`env.step` returns the standard Gymnasium five-tuple, at 30 Hz (8 physics steps at 240 Hz). The reward is 1 on success and 0 otherwise. `info` holds `is_success`, `collision`, and the true `distance_m` to the box surface, for debugging only. Used directly, the environment stops after 60 s; the evaluation script uses its own 120 s limit.

The observation is a dictionary:

| Key | Type and shape | Meaning |
| --- | --- | --- |
| `detected` | `0` or `1` | Whether the target was detected this frame. Detection stops beyond 250 m of 3D distance to the box surface; a visible target within range is missed independently 5% of the time. |
| `box` | float32 `(4,)` | `(x1, y1, x2, y2)` of the detector box, normalized to `[0, 1]` in the 160 × 90 detection image; zeros when absent. Each edge has Gaussian jitter of at least 0.5 pixel or 1% of the box size, whichever is larger. |
| `depth_m` | float32 `(126, 224)` | Forward camera depth in metres, clipped to `[0, 40]`. Each pixel has independent Gaussian error with standard deviation `max(0.03, 0.03d)` m at depth `d`: about 0.3 m at 10 m and 0.6 m at 20 m. Rows run top to bottom, columns left to right. |
| `gyro_rad_s` | float32 `(3,)` | Body-frame angular rates in rad/s, with 0.002 rad/s Gaussian noise per axis. |
| `accel_m_s2` | float32 `(3,)` | Body-frame specific force in m/s², with 0.03 m/s² Gaussian noise per axis; reads approximately `[0, 0, 9.8]` at rest or hover. |
| `vio_xyz_m` | float32 `(3,)` | Noisy `(x, y, altitude)` in metres relative to `(0, 0)`. Horizontal increments have Gaussian error of `0.003 + 0.01 × distance moved` m per axis and accumulate drift; altitude has 0.05 m Gaussian error per reading. It is anchored at the true start position on reset. |

The action is `(roll, pitch, yaw, throttle)`: four finite floats in `[-1, 1]`. Zero roll and pitch request level attitude, zero yaw requests zero yaw rate, and zero throttle requests hover. Full stick is a 60° roll/pitch command and a 200°/s yaw rate.

The IMU is sampled with each observation at 30 Hz. It uses white noise without bias drift or vibration.

## Terrains

Four public terrains: `field` (18 box placements), `forest`, `village`, and `city` (40 each). Select one with `terrain="<name>@v<N>"`, for example `forest@v4`. See the [terrain notes](fpv_sim/assets/terrains/README.md) for layouts and hiding spots.

## Fly with a USB RC transmitter

Flying by hand is the fastest way to get a feel for the dynamics and the sensors. Connect an EdgeTX/OpenTX radio in **USB Joystick** mode and check its axes first:

```sh
uv run python -m fpv_sim.utils.rc_input
uv run python -m fpv_sim.scripts.rc                       # village@v1
uv run python -m fpv_sim.scripts.rc --terrain forest@v4
```

The default mapping is AETR on joystick axes 0–3: roll, pitch, throttle, yaw. The reader returns `(roll, pitch, yaw, throttle)`; change `axes` or `invert` in `RCInput()` in `fpv_sim/scripts/rc.py` if your radio differs. Lower the throttle to arm, then raise it past the printed hover threshold to start flying. The simulation stays paused while you arm. The window shows the 480 × 270 onboard camera and the detection box. Press **F11** for fullscreen, **R** to reset and re-arm; close the window to quit. RC flight continues through collisions, success, and time limits. The radio controls only this local simulator.

## Hand-in

Submit your `handin/` folder.
