"""Headless evaluation of handin.agent.Agent on every terrain and target box, logged to MCAP."""

import argparse
import base64
import csv
import json
import os
import time
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pybullet as p
from mcap.writer import Writer

from fpv_sim.utils.terrain import TERRAINS_DIR, terrain_start_pose

TERRAINS = ("field", "forest", "village", "city")
ATTACK_TERRAINS = ("level_1",)
FPS = 30
BUDGET_MS = 1000 / FPS
ARENA = 350    # |x| or |y| beyond this leaves the arena
RAW_IMAGE = {"title": "foxglove.RawImage", "type": "object", "properties": {
    "timestamp": {"type": "object", "properties": {"sec": {"type": "integer"}, "nsec": {"type": "integer"}}},
    "frame_id": {"type": "string"}, "width": {"type": "integer"}, "height": {"type": "integer"},
    "encoding": {"type": "string"}, "step": {"type": "integer"},
    "data": {"type": "string", "contentEncoding": "base64"}}}

# Cheap per-step analysis topics (JSON): no extra rendering, no physics raycasts.
TARGET_RELATIVE_SCHEMA = {"title": "TargetRelative", "type": "object", "properties": {
    "distance_m": {"type": "number"}, "bearing_deg": {"type": "number"},
    "elevation_deg": {"type": "number"}, "lateral_offset_m": {"type": "number"},
    "vertical_offset_m": {"type": "number"}}}
MISSION_PHASE_SCHEMA = {"title": "MissionPhase", "type": "object", "properties": {
    "event": {"type": "string"}, "phase": {"type": "string"}, "from": {"type": "string"},
    "to": {"type": "string"}, "phase_time_s": {"type": "number"}}}


def versions(terrain):
    spec = json.loads((TERRAINS_DIR / f"{terrain}.json").read_text())
    return len(spec["setups"] if "setups" in spec else spec["targets"])


def target_center(env):
    """World position of the single collision box the target body holds."""
    body = env.TERRAIN_IDS[0]
    center, quat = p.getBasePositionAndOrientation(body, physicsClientId=env.CLIENT)
    shape = p.getCollisionShapeData(body, -1, physicsClientId=env.CLIENT)[0]
    local_pos, yaw = shape[5], p.getEulerFromQuaternion(quat)[2]
    c, s = np.cos(yaw), np.sin(yaw)
    offset = np.array([c * local_pos[0] - s * local_pos[1], s * local_pos[0] + c * local_pos[1], local_pos[2]])
    return np.array(center, dtype=float) + offset


def target_relative(drone_xyz, drone_yaw, box_xyz):
    """Target position in the drone frame: range, bearing (left +), elevation (up +), lateral, vertical."""
    rel = np.asarray(box_xyz, dtype=float) - np.asarray(drone_xyz, dtype=float)
    c, s = np.cos(-drone_yaw), np.sin(-drone_yaw)
    fwd, lat = c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]
    horiz = np.hypot(fwd, lat)
    return {"distance_m": float(np.linalg.norm(rel)),
            "bearing_deg": float(np.degrees(np.arctan2(lat, fwd))),
            "elevation_deg": float(np.degrees(np.arctan2(rel[2], horiz))),
            "lateral_offset_m": float(lat),
            "vertical_offset_m": float(rel[2])}


def random_start(env, rng):
    """A clear start 225-275 m from the centre, 5-25 m up, flying at 0-10 m/s within 20 deg of the centre."""
    for _ in range(100):
        r, angle = rng.uniform(225, 275), rng.uniform(-np.pi, np.pi)
        xyz = np.array([r * np.cos(angle), r * np.sin(angle), rng.uniform(5, 25)])
        yaw = np.arctan2(-xyz[1], -xyz[0]) + rng.uniform(-np.radians(20), np.radians(20))
        speed = rng.uniform(0, 10)
        env.INIT_XYZS, env.INIT_RPYS = xyz[None], np.array([[0, 0, yaw]])
        obs, _ = env.reset(seed=int(rng.integers(2**31)))
        if not any(p.getClosestPoints(env.DRONE_IDS[0], body, 2., physicsClientId=env.CLIENT) for body in env.TERRAIN_IDS):
            break
    p.resetBaseVelocity(env.DRONE_IDS[0], [speed * np.cos(yaw), speed * np.sin(yaw), 0], [0, 0, 0],
                        physicsClientId=env.CLIENT)
    env._updateAndStoreKinematicInformation()
    env.vio.reset(env._getDroneStateVector(0))
    env.imu.reset()
    return env._sense(), {"position_m": xyz.tolist(), "yaw_rad": float(yaw), "speed_m_s": float(speed)}


def mission_start(env, rng, terrain):
    """Start of a flight: the `level_*` setup pose when the terrain pins one, else a random search start.

    For attack maps (`level_*`), the drone starts at the setup's `start` pose with the target already in
    the camera's sight (ApproachAviary reads it at construction); each flight differs only in sensing
    noise. For the search scenes, falls back to `random_start` (a clear start 225-275 m from the centre).
    """
    pose = terrain_start_pose(terrain)
    if pose is None:
        return random_start(env, rng)
    obs, _ = env.reset(seed=int(rng.integers(2 ** 31)))
    env.vio.reset(env._getDroneStateVector(0))
    env.imu.reset()
    return obs, {"position_m": list(pose["position_m"]), "yaw_rad": float(pose["rpy_rad"][2]), "speed_m_s": 0.}


class Log:
    """One flight's MCAP file, JSON-encoded so Foxglove opens it as is."""

    def __init__(self, path):
        self.file = open(path, "wb")
        self.writer = Writer(self.file)
        self.writer.start(library="fpv_sim.scripts.evaluate")
        self.channels = {}

    def write(self, topic, t, message, schema=None):
        if topic not in self.channels:
            schema = schema or {"type": "object"}
            sid = self.writer.register_schema(name=schema.get("title", topic), encoding="jsonschema",
                                              data=json.dumps(schema).encode())
            self.channels[topic] = self.writer.register_channel(topic=topic, message_encoding="json", schema_id=sid)
        ns = int(t * 1e9)
        self.writer.add_message(self.channels[topic], log_time=ns, publish_time=ns, data=json.dumps(message).encode())

    def close(self, metadata):
        self.writer.add_metadata("flight", {k: str(v) for k, v in metadata.items()})
        self.writer.finish()
        self.file.close()


def log_sensors(log, env, obs, t, step, depth_every):
    state = env._getDroneStateVector(0)
    info = env._computeInfo()
    log.write("/observation/detection", t, {"detected": int(obs["detected"]), "box": obs["box"].tolist()})
    log.write("/observation/vio", t, dict(zip(("x", "y", "z"), obs["vio_xyz_m"].tolist())))
    log.write("/imu", t, {"gyro_rad_s": obs["gyro_rad_s"].tolist(), "accel_m_s2": obs["accel_m_s2"].tolist()})
    log.write("/truth", t, {"position_m": state[0:3].tolist(), "quaternion": state[3:7].tolist(),
                            "rpy_rad": state[7:10].tolist(), "velocity_m_s": state[10:13].tolist(),
                            "angular_velocity_rad_s": state[13:16].tolist(), **info})
    if step % depth_every == 0:
        depth = obs["depth_m"]
        log.write("/observation/depth", t, {
            "timestamp": {"sec": int(t), "nsec": int(t % 1 * 1e9)}, "frame_id": "camera",
            "width": depth.shape[1], "height": depth.shape[0], "encoding": "16UC1", "step": 2 * depth.shape[1],
            "data": base64.b64encode(np.round(depth * 1000).astype("<u2").tobytes()).decode()}, RAW_IMAGE)
    return state, info


def fly(env, agent, rng, path, timeout_s=120., depth_every=10, terrain=None):
    """Flies one run with the agent in full command; returns its summary."""
    obs, start = mission_start(env, rng, terrain if terrain is not None else getattr(env, "TERRAIN", None))
    agent.reset()
    log = Log(path)
    record = {"outcome": "timeout", "error": "", **{f"start_{k}": v for k, v in start.items()},
              "start_distance_m": env._distance()}
    box_xyz = target_center(env)
    latencies, speeds, seen_at, path_m = [], [], None, 0.
    # Flight phase mirrors the policy's own mode (Agent exposes "exploration" | "terminal"); there
    # is no separate distance threshold in the evaluator. The mission split latches on the FIRST
    # time the policy commits to the target: one exploration phase, then one terminal phase.
    phase, phase_start_t, stance = "exploration", 0., {"exploration": 0., "terminal": 0.}
    policy_phase, committed = "exploration", False
    t, step = 0., 0
    state, info = log_sensors(log, env, obs, t, step, depth_every)
    log.write("/mission/phase", t, {"event": "enter", "phase": phase, "phase_time_s": 0.}, MISSION_PHASE_SCHEMA)
    try:
        while True:
            if obs["detected"] and seen_at is None:
                seen_at = t
            tic = time.perf_counter()
            action = np.asarray(agent.act(obs), dtype=float)
            latencies.append(1000 * (time.perf_counter() - tic))
            # Policy mode transition (debugging): log every change; the mission split below latches
            # on the first commit to terminal only.
            new_policy_phase = obs.get("phase")
            if new_policy_phase is not None and new_policy_phase != policy_phase:
                log.write("/mission/phase", t, {"event": "policy_transition", "from": policy_phase,
                                                "to": new_policy_phase, "phase_time_s": t - phase_start_t},
                          MISSION_PHASE_SCHEMA)
                if not committed and new_policy_phase == "terminal":
                    committed = True
                    stance["exploration"] += t - phase_start_t
                    log.write("/mission/phase", t, {"event": "transition", "from": "exploration",
                                                    "to": "terminal", "phase_time_s": t - phase_start_t},
                              MISSION_PHASE_SCHEMA)
                    phase, phase_start_t = "terminal", t
                policy_phase = new_policy_phase
            log.write("/agent/action", t, {**dict(zip(("roll", "pitch", "yaw", "throttle"), action.tolist())),
                                           "latency_ms": latencies[-1]})
            prev = state[0:3]
            obs, _, _, _, _ = env.step(action)
            step += 1
            t = env.step_counter / env.PYB_FREQ
            state, info = log_sensors(log, env, obs, t, step, depth_every)
            path_m += np.linalg.norm(state[0:3] - prev)
            log.write("/analysis/target_relative", t,
                      target_relative(state[0:3], state[9], box_xyz), TARGET_RELATIVE_SCHEMA)
            speeds.append(np.linalg.norm(state[10:13]))
            if info["is_success"]:
                record["outcome"] = "success"
            elif info["collision"]:
                record["outcome"] = "collision"
            elif np.any(np.abs(state[0:2]) >= ARENA):
                record["outcome"] = "out_of_arena"
            if record["outcome"] != "timeout" or t >= timeout_s:
                break
    except Exception as exc:    # a bad action or a crashing agent fails this flight, not the whole evaluation
        record.update(outcome="error", error=repr(exc))
    stance[phase] += t - phase_start_t
    log.write("/mission/phase", t, {"event": "final", "phase": phase, "phase_time_s": t - phase_start_t},
              MISSION_PHASE_SCHEMA)
    lat = np.array(latencies or [np.nan])
    record.update(time_s=t, steps=step, path_m=path_m, mean_speed_m_s=path_m / t if t else 0.,
                  exploration_time_s=stance["exploration"], terminal_time_s=stance["terminal"],
                  max_speed_m_s=max(speeds, default=0.), final_speed_m_s=float(np.linalg.norm(state[10:13])),
                  final_distance_m=info["distance_m"], seen=seen_at is not None,
                  first_detection_s=seen_at if seen_at is not None else "",
                  latency_mean_ms=float(np.mean(lat)))
    log.close(record)
    return record


def run_target(job):
    """Every run on one terrain version, in its own process and PyBullet client."""
    terrain, runs, out, seed, timeout_s, depth_every = job
    from fpv_sim.envs.approach_aviary import ApproachAviary
    from handin.agent import Agent
    env = ApproachAviary(continuous=True, terrain=terrain)
    agent = Agent()
    records = []
    try:
        (out / terrain).mkdir(parents=True, exist_ok=True)
        for run in range(runs):
            run_seed = [seed, sum(map(ord, terrain)), run]
            rng = np.random.default_rng(run_seed)
            record = fly(env, agent, rng, out / terrain / f"run{run:02d}.mcap", timeout_s, depth_every, terrain)
            records.append({"terrain": terrain, "run": run, **record})
    finally:
        env.close()
    return records


def summarize(records):
    """Per-terrain metrics; reach time and speed average over successful flights only."""
    metrics = {}
    for name in sorted({r["terrain"].split("@")[0] for r in records}):
        rs = [r for r in records if r["terrain"].split("@")[0] == name]
        ok = [r for r in rs if r["outcome"] == "success"]
        fail = [r for r in rs if r["outcome"] != "success"]
        mean = lambda xs: float(np.mean(xs)) if len(xs) else float("nan")
        lat = [r["latency_mean_ms"] for r in rs if r["steps"]]
        metrics[name] = {
            "flights": len(rs),
            **{f"{o}_rate": mean([r["outcome"] == o for r in rs])
               for o in ("success", "collision", "timeout", "out_of_arena", "error")},
            "exploration_time_s": mean([r["exploration_time_s"] for r in rs]),
            "terminal_time_s": mean([r["terminal_time_s"] for r in rs]),
            "reach_time_s": mean([r["time_s"] for r in ok]),
            "reach_mean_speed_m_s": mean([r["mean_speed_m_s"] for r in ok]),
            "path_efficiency": mean([r["start_distance_m"] / r["path_m"]
                                     for r in ok if r["path_m"]]),
            "detection_rate": mean([r["seen"] for r in rs]),
            "first_detection_s": mean([r["first_detection_s"] for r in rs if r["seen"]]),
            "failed_final_distance_m": mean([r["final_distance_m"] for r in fail]),
            "latency_mean_ms": mean(lat),
            "agent_fps": 1000 / mean(lat) if lat else float("nan"),
        }
    return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=10, help="flights per target box")
    parser.add_argument("--terrains", nargs="+", default=TERRAINS, choices=TERRAINS + ATTACK_TERRAINS)
    parser.add_argument("--timeout", type=float, default=120., help="seconds of simulated flight")
    parser.add_argument("--depth-every", type=int, default=10, help="log 1 depth map in N frames (1 = all, ~130 MB a flight)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("out") / datetime.now().strftime("%Y%m%d-%H%M%S"))
    args = parser.parse_args(argv)

    jobs = [(f"{t}@v{v}", args.runs, args.out, args.seed, args.timeout, args.depth_every)
            for t in args.terrains for v in range(1, 1 + versions(t))]
    records = []
    with get_context("spawn").Pool(args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_target, jobs), 1):
            records += result
            print(f"[{i}/{len(jobs)}] {result[0]['terrain']}: {sum(r['outcome'] == 'success' for r in result)}/{len(result)} reached",
                  flush=True)

    records.sort(key=lambda r: (r["terrain"], r["run"]))
    with open(args.out / "flights.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = summarize(records)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    print(f"\n{'terrain':8} {'flights':>7} {'reach':>6} {'crash':>6} {'t/o':>5} {'out':>5} {'err':>5} "
          f"{'t_expl_s':>9} {'t_term_s':>9} {'t_reach_s':>9} {'v_reach_m/s':>11} {'seen':>5} {'lat_ms':>7} {'fps':>6}")
    for name, m in metrics.items():
        print(f"{name:8} {m['flights']:7d} {m['success_rate']:6.0%} {m['collision_rate']:6.0%} {m['timeout_rate']:5.0%} "
              f"{m['out_of_arena_rate']:5.0%} {m['error_rate']:5.0%} {m['exploration_time_s']:9.1f} {m['terminal_time_s']:9.1f} "
              f"{m['reach_time_s']:9.1f} {m['reach_mean_speed_m_s']:11.1f} "
              f"{m['detection_rate']:5.0%} {m['latency_mean_ms']:7.2f} {m['agent_fps']:6.0f}")
        if m["latency_mean_ms"] > BUDGET_MS:
            print(f"  WARNING: {name} agent averages {m['latency_mean_ms']:.1f} ms a step, below {FPS} FPS")
    print(f"\nMCAP logs, flights.csv and metrics.json in {args.out}")


if __name__ == "__main__":
    main()
