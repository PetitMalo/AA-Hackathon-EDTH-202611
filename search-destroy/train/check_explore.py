"""Vérifie train.explore_env.ExploreEnv avec des politiques scriptées et affiche les termes de reward.

    uv run python -m train.check_explore                     # 3 politiques x 2 terrains, 30 s de vol chacune
    uv run python -m train.check_explore --seconds 120 --terrains city@v1 forest@v2

But : vérifier les ordres de grandeur (couverture vs détection vs pénalités), que la couverture est
monotone, et que tourner sur place en altitude ne rapporte pas « trop » par rapport à avancer.
Les politiques scriptées lisent la vraie altitude du simulateur : c'est un banc de test, pas une policy.
"""

import argparse
import copy
from pathlib import Path

import numpy as np

from train import config as cfgmod
from train.explore_env import ExploreEnv

CONFIG = str(Path(__file__).with_name("config_explore.toml"))


def hover(env, t):
    return [0., 0., 0., 0.]


def straight(env, t):                    # léger piqué avant (pitch > 0 = avancer), un peu de gaz
    return [0., .15, 0., .08]


def climb_spin(env, t):                  # monte vers 40 m, puis tourne sur place
    z = env.sim.pos[0, 2]
    return [0., 0., .3 if z > 30 else 0., float(np.clip((40 - z) / 20, -.5, .5))]


POLICIES = {"hover": hover, "straight": straight, "climb_spin": climb_spin}


def run(env, policy, seed):
    x, _ = env.reset(seed=seed)
    cov0, prev = env.cov_m2, env.cov_frac
    total, monotone, info = 0., True, {}
    for t in range(env.timeout_steps):
        x, r, term, trunc, info = env.step(policy(env, t))
        total += r
        monotone &= env.cov_frac >= prev - 1e-9
        prev = env.cov_frac
        assert env.observation_space.contains(x)
        if term or trunc:
            break
    ex = env.ex
    r_cov = ex["w_cov"] * (env.cov_m2 - cov0) / 1000.
    return dict(outcome=info.get("outcome", "?"), steps=env.t, cov=env.cov_frac, cov_m2=env.cov_m2,
                r_total=total, r_cov=r_cov, detected=env.ever_seen, t_det=env.t_detect, monotone=monotone)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--terrains", nargs="*", default=["village@v1", "forest@v4"])
    ap.add_argument("--seconds", type=float, default=30.)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = copy.deepcopy(cfgmod.load(a.config))
    cfg["explore"]["timeout_s"] = a.seconds
    env = ExploreEnv(cfg, a.terrains, seed=a.seed)
    print(f"zone {env.zone_cells} cellules de 12,5 m | couverture max = "
          f"{cfg['explore']['w_cov'] * env.zone_cells * 156.25 / 1000:.1f} | détection = {cfg['explore']['detect_bonus']}")
    print(f"{'terrain':12s} {'policy':11s} {'issue':13s} {'pas':>5s} {'couv %':>7s} {'m²':>8s} "
          f"{'R cov':>6s} {'R tot':>7s} {'détect':>7s} mono")
    for terrain in a.terrains:
        env.terrains = [terrain]
        for name, pol in POLICIES.items():
            r = run(env, pol, a.seed)
            det = f"{r['t_det']:.1f}s" if r["detected"] else "-"
            print(f"{terrain:12s} {name:11s} {r['outcome']:13s} {r['steps']:5d} {100 * r['cov']:7.1f} "
                  f"{r['cov_m2']:8.0f} {r['r_cov']:6.2f} {r['r_total']:7.2f} {det:>7s} {r['monotone']}")
            assert r["monotone"], "couverture non monotone"
    env.close()
    print("OK")


if __name__ == "__main__":
    main()
