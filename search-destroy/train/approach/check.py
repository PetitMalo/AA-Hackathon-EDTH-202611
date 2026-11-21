"""Vérifie train.approach.env.ApproachEnv : départs, position attendue de la cible, reward, issues.

    uv run python -m train.approach.check                    # toutes les étapes, 20 épisodes par politique
    uv run python -m train.approach.check --episodes 50 --stages 0 3 5
    uv run python -m train.approach.check --stages --bench          # seulement les départs fixes du benchmark

Pour chaque étape :
  - départs : taux de détection au départ, taux de re-tirage, distance et altitude réelles ;
  - position attendue (bearing, elev, dz_rel) comparée à la vérité terrain, cible vue / cible perdue ;
  - politiques scriptées hover et pursuit (suit bearing / elev du vec) : issues, durée, reward.
Les politiques lisent seulement l'observation, la vérité terrain sert uniquement aux erreurs affichées.
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from handin.features import BOX_HALF_H, VEC_DIM
from train.approach import config as cfgmod
from train.approach.env import ApproachEnv

CONFIG = str(Path(__file__).with_name("config_l1.toml"))
I_DZ, I_BEAR, I_ELEV = VEC_DIM, VEC_DIM + 1, VEC_DIM + 2


def hover(vec):
    return np.zeros(4)


AIM_ABOVE_CENTER = .45      # vise le haut de la boîte (~1,2 m) : toucher le sol en même temps = collision


def pursuit(vec):
    """Cap vers la cible attendue, piqué constant, pente de vol asservie vers le haut de la boîte."""
    bear = vec[I_BEAR] * np.pi
    dz, dist = vec[I_DZ] * 50 + AIM_ABOVE_CENTER, max(vec[14] * 100, .5)
    elev = np.arctan2(dz, dist)                        # direction visée
    vh, vz = np.hypot(vec[16], vec[17]) * 20, vec[18] * 5
    gamma = np.arctan2(vz, max(vh, 1.))                # pente de vol actuelle
    yaw = np.clip(-2. * bear, -1, 1)                   # stick yaw > 0 = tourner à droite
    throttle = np.clip(.12 + 2. * (elev - gamma), -.5, .6)
    pitch = .35 if dist > 6 else .2                    # on ralentit à l'approche
    return np.array([0., pitch, yaw, throttle])


POLICIES = {"hover": hover, "pursuit": pursuit}


def truth(env):
    """(dz_rel, bearing, elev) vrais, mêmes normalisations que ApproachEnv._expected."""
    pos, yaw = env.sim.pos[0], env.sim.rpy[0, 2]
    rel = env.target_xy - pos[:2]
    fwd, left = np.array([np.cos(yaw), np.sin(yaw)]), np.array([-np.sin(yaw), np.cos(yaw)])
    dx, dy = rel @ fwd, rel @ left
    dz = BOX_HALF_H - pos[2]
    return np.array([dz / 50, np.arctan2(dy, dx) / np.pi, np.arctan2(dz, np.hypot(dx, dy)) / (np.pi / 2)])


def run_stage(env, stage, episodes, seed):
    env.set_stage(stage)
    env.cur = {**env.cur, "replay_previous": 0.}
    starts = dict(det=0, d=[], z=[])
    err = {True: [], False: []}                         # détectée ? -> erreurs (deg, deg, m)
    res = {}
    for name, pol in POLICIES.items():
        outs, lens, rets = Counter(), [], []
        env.n_starts = env.n_rejects = 0
        for ep in range(episodes):
            x, _ = env.reset(seed=seed + 1000 * stage + ep)
            if name == "hover":
                starts["det"] += int(x["vec"][0] > 0)
                starts["d"].append(env.d0)
                starts["z"].append(env.sim.pos[0, 2])
            total, info = 0., {}
            while True:
                a = pol(x["vec"])
                x, r, term, trunc, info = env.step(a)
                total += r
                if x["vec"][11] > 0:                    # Kalman valide
                    e = x["vec"][I_DZ:I_DZ + 3] - truth(env)
                    e[1] = (e[1] + 1) % 2 - 1           # bearing : angle normalisé, on replie
                    err[bool(x["vec"][0] > 0)].append([abs(e[1]) * 180, abs(e[2]) * 90, abs(e[0]) * 50])
                if term or trunc:
                    break
            outs[info["outcome"]] += 1
            lens.append(info["t_s"])
            rets.append(total)
        res[name] = (outs, np.mean(lens), np.mean(rets), env.n_rejects / max(env.n_starts, 1))
    return starts, err, res


def run_bench(env, cfg, seed):
    """Départs fixes du benchmark : cible vue au départ ? issue de la politique pursuit."""
    from train.approach.bench import load_setups
    by_stage = {}
    for u in load_setups(cfg["terrains"]["bench"], cfg["curriculum"]["stages"]):
        x, _ = env.reset(seed=seed + u["index"], options={"setup": u})
        seen = bool(x["vec"][0] > 0)
        while True:
            x, _, term, trunc, info = env.step(pursuit(x["vec"]))
            if term or trunc:
                break
        by_stage.setdefault(u["stage"], []).append((seen, info["outcome"], u))
    print("\n=== benchmark", cfg["terrains"]["bench"])
    for s, rs in sorted(by_stage.items()):
        missed = [u["index"] for seen, _, u in rs if not seen]
        print(f"  étape {s} ({cfg['curriculum']['stages'][s][0]} m) : cible vue au départ {sum(r[0] for r in rs)}/{len(rs)}"
              f" | pursuit {dict(Counter(r[1] for r in rs))}" + (f" | non vue : setups {missed}" if missed else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--stages", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true", help="aussi passer les départs fixes du benchmark")
    a = ap.parse_args()
    cfg = cfgmod.load(a.config)
    train_terrains, _ = cfgmod.terrain_split(cfg)
    env = ApproachEnv(cfg, train_terrains, stage=0, seed=a.seed)
    stages = a.stages if a.stages is not None else range(len(cfg["curriculum"]["stages"]))
    assert env.observation_space.contains(env.reset(seed=0)[0])
    for s in stages:
        starts, err, res = run_stage(env, s, a.episodes, a.seed)
        n = a.episodes
        print(f"\n=== étape {s} : {cfg['curriculum']['stages'][s]}")
        print(f"  départs : cible vue {starts['det']}/{n} | distance surface {np.mean(starts['d']):.1f} m "
              f"[{np.min(starts['d']):.1f}, {np.max(starts['d']):.1f}] | z {np.mean(starts['z']):.1f} m")
        for det, e in err.items():
            if e:
                m = np.median(e, axis=0)
                print(f"  position attendue, cible {'vue   ' if det else 'perdue'} ({len(e):5d} pas) : "
                      f"erreur médiane bearing {m[0]:.1f}°, elev {m[1]:.1f}°, dz {m[2]:.2f} m")
        for name, (outs, t, ret, rej) in res.items():
            print(f"  {name:8s} issues {dict(outs)} | durée {t:.1f} s | retour {ret:+.1f} | re-tirage {rej:.0%}")
    if a.bench:
        run_bench(env, cfg, a.seed)
    rw, cur = cfg["reward"], cfg["curriculum"]
    worst = max(st[6] for st in cur["stages"]) * 30 * (rw["time"] + rw["lost"]) - rw["timeout"]
    ok = "OK" if worst < -rw["collision"] else "ATTENTION : se crasher devient rentable"
    print(f"\npire épisode sans succès ni crash (temps + cible perdue + timeout) : -{worst:.1f} "
          f"vs collision {rw['collision']} -> {ok}")


if __name__ == "__main__":
    main()
