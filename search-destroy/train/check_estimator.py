"""Compare handin.estimator.StateEstimator à la vérité terrain sur des vols agressifs.

uv run python -m train.check_estimator --runs 4 --seconds 30
"""

import argparse
from multiprocessing import get_context

import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fly(seed, seconds, params=None):
    from fpv_sim.envs.approach_aviary import ApproachAviary
    from fpv_sim.scripts.evaluate import random_start
    from handin.estimator import StateEstimator

    rng = np.random.default_rng(seed)
    env = ApproachAviary(continuous=True, terrain="field@v1")
    obs, start = random_start(env, rng)
    est = StateEstimator()
    for k_, v_ in (params or {}).items():
        setattr(est, k_, v_)
    action = None
    est.reset()
    # manœuvres : sinus aléatoires sur roll/pitch, rafales de yaw, maintien d'altitude
    f = rng.uniform(.1, .5, 3)
    ph = rng.uniform(0, 2 * np.pi, 3)
    amp = rng.uniform(.2, .6, 3)
    err = {k: [] for k in ("roll", "pitch", "yaw", "v_xy", "z", "vz")}
    for k in range(int(seconds * 30)):
        est.update(obs, action)
        t = k / 30
        s = env._getDroneStateVector(0)
        if k > 90:      # après 3 s : hors transitoire de départ
            err["roll"].append(np.degrees(wrap(est.roll - s[7])))
            err["pitch"].append(np.degrees(wrap(est.pitch - s[8])))
            err["yaw"].append(np.degrees(wrap(est.yaw - s[9])))
            err["v_xy"].append(np.linalg.norm(est.vel[:2] - s[10:12]))
            err["z"].append(est.pos[2] - s[2])
            err["vz"].append(est.vel[2] - s[12])
        roll, pitch, yaw = amp * np.sin(2 * np.pi * f * t + ph)
        pitch = .3 + .5 * pitch if t > 2 else 0.     # 2 s sans commande, puis vol avant manœuvré
        roll = roll if t > 2 else 0.
        yaw = yaw if t > 2 else 0.
        throttle = np.clip(.8 * (15 - est.pos[2]) - .4 * est.vel[2], -.5, .5) / max(np.cos(.6 * abs(pitch) + .6 * abs(roll)), .5)
        action = np.clip([roll, pitch, yaw, throttle], -1, 1)
        obs, *_ = env.step(action)
    env.close()
    true_yaw0 = start["yaw_rad"]
    bearing = np.arctan2(-start["position_m"][1], -start["position_m"][0])
    return seed, start["speed_m_s"], np.degrees(wrap(true_yaw0 - bearing)), {k: np.array(v) for k, v in err.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--set", nargs="*", default=[], help="ex. YAW_GAIN=0.02 RP_GAIN=0")
    a = ap.parse_args()
    params = {k: float(v) if "." in v else int(v) for k, v in (x.split("=") for x in a.set)}
    with get_context("spawn").Pool(min(a.runs, 4)) as pool:
        res = pool.starmap(fly, [(s, a.seconds, params) for s in range(a.runs)])
    print(f"{'seed':>4} {'v0':>5} {'yaw0':>6} | {'roll':>11} {'pitch':>11} {'yaw':>11} | {'v_xy':>11} {'z':>11} {'vz':>11}")
    print(f"{'':>4} {'m/s':>5} {'deg':>6} | {'rms / max (deg)':^35} | {'rms / max (m, m/s)':^35}")
    tot = {k: np.concatenate([r[3][k] for r in res]) for k in res[0][3]}
    for seed, v0, yaw0, e in res:
        cells = [f"{np.sqrt(np.mean(e[k]**2)):5.2f}/{np.max(np.abs(e[k])):5.2f}" for k in e]
        print(f"{seed:4d} {v0:5.1f} {yaw0:6.1f} | " + " ".join(cells[:3]) + " | " + " ".join(cells[3:]))
    cells = [f"{np.sqrt(np.mean(tot[k]**2)):5.2f}/{np.max(np.abs(tot[k])):5.2f}" for k in tot]
    print(f"{'ALL':>4} {'':>5} {'':>6} | " + " ".join(cells[:3]) + " | " + " ".join(cells[3:]))


if __name__ == "__main__":
    main()
