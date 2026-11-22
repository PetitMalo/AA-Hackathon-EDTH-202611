"""Évalue une policy d'approche, moyenne déterministe de la policy.

    uv run python -m train.approach.eval --policy /tmp/fpv_runs/approach_<date>/policy.pt            # benchmark
    uv run python -m train.approach.eval --policy <...>/policy.pt --repeats 3 --workers 8
    uv run python -m train.approach.eval --policy <...>/policy.pt --random --episodes 20 --stages 3 4 5

Benchmark (défaut) : les départs fixes de [terrains].bench (level_1.json : 60 setups, 10 par distance),
vitesse nulle, `--repeats` passages par setup (le détecteur et les capteurs sont bruités).
--random : départs tirés comme à l'entraînement, `--episodes` par étape et par worker.
Sortie : tableau par distance, grille altitude x |psi|, CSV des épisodes à côté de la policy.
"""

import argparse
import csv
from multiprocessing import get_context
from pathlib import Path

import numpy as np

CONFIG = str(Path(__file__).with_name("config_l1.toml"))


def _worker(job):
    import torch
    from handin.approach_net import load_approach_actor
    from train.approach import config as cfgmod
    from train.approach.env import ApproachEnv

    torch.set_num_threads(1)
    (config, overrides), policy, terrains, stage, episodes = job       # episodes : liste de (seed, setup ou None)
    cfg = cfgmod.load(config, overrides)
    cfg["curriculum"]["replay_previous"] = 0.
    env = ApproachEnv(cfg, terrains, stage=stage, seed=episodes[0][0])
    net = load_approach_actor(policy)
    rows = []
    for seed, setup in episodes:
        x, _ = env.reset(seed=seed, options={"setup": setup} if setup else None)
        detected0 = bool(x["vec"][0] > 0)
        while True:
            with torch.no_grad():
                a = np.clip(net({k: torch.as_tensor(v)[None] for k, v in x.items()})[0].numpy(), -1, 1)
            x, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        rows.append({"setup": setup["index"] if setup else -1, "stage": info["stage"], "terrain": info["terrain"],
                     "seed": seed, "outcome": info["outcome"], "t_s": info["t_s"], "d": info["d"], "z": info["z"],
                     "psi_deg": info["psi_deg"], "lost_frac": info["lost_frac"], "seen_at_start": detected0})
    env.close()
    return rows


def _report(rows, stages_cfg):
    stages = sorted({r["stage"] for r in rows})
    print(f"{'étape':>5} {'d':>6} {'n':>4} {'succès':>7} {'collision':>9} {'timeout':>8} {'t succès':>9} "
          f"{'sans cible':>10} {'vue au départ':>13}")
    for s in stages:
        rs = [r for r in rows if r["stage"] == s]
        rate = lambda o: sum(r["outcome"] == o for r in rs) / len(rs)
        ts = [r["t_s"] for r in rs if r["outcome"] == "success"]
        print(f"{s:>5} {stages_cfg[s][0]:>5}m {len(rs):>4} {rate('success'):>7.0%} {rate('collision'):>9.0%} "
              f"{rate('timeout'):>8.0%} {np.mean(ts) if ts else float('nan'):>8.1f}s "
              f"{np.mean([r['lost_frac'] for r in rs]):>10.0%} {np.mean([r['seen_at_start'] for r in rs]):>13.0%}")
    allr = sum(r["outcome"] == "success" for r in rows) / len(rows)
    print(f"total : {allr:.0%} de succès sur {len(rows)} épisodes")

    print("\nsuccès par altitude (bas / haut de [z_inf, z_sup]) x |psi| (petit / grand, seuil psi_max / 2) :")
    for s in stages:
        _, _, z_lo, z_hi, psi_max, _, _ = stages_cfg[s]
        cells = []
        for z_half in (0, 1):
            for p_half in (0, 1):
                rs = [r for r in rows if r["stage"] == s
                      and (r["z"] >= (z_lo + z_hi) / 2) == z_half and (abs(r["psi_deg"]) >= psi_max / 2) == p_half]
                cells.append(f"{sum(r['outcome'] == 'success' for r in rs) / len(rs):4.0%} (n={len(rs):3d})"
                             if rs else "   -       ")
        print(f"  étape {s} : z bas  psi petit {cells[0]} | psi grand {cells[1]}\n"
              f"            z haut psi petit {cells[2]} | psi grand {cells[3]}")


def _run_config(policy, config, extra):
    """Config du run (config.toml + overrides.txt du dossier du run, policy.pt ou ckpt/policy_N.pt), + --set."""
    for d in (policy.parent, policy.parent.parent):
        if (d / "config.toml").exists():
            ov = (d / "overrides.txt").read_text().split() if (d / "overrides.txt").exists() else []
            return (config or str(d / "config.toml")), (ov if not config else []) + list(extra)
    return (config or CONFIG), list(extra)


def eval_one(policy, a):
    from train.approach import config as cfgmod
    from train.approach.bench import load_setups
    config, overrides = _run_config(policy, a.config, a.set)
    cfg = cfgmod.load(config, overrides)
    stages_cfg = cfg["curriculum"]["stages"]
    train_t, hold_t = cfgmod.terrain_split(cfg)
    keep = lambda s: a.stages is None or s in a.stages
    bench = a.bench or cfg["terrains"]["bench"]
    if a.random:
        terrains = hold_t or train_t
        jobs = [((config, overrides), str(policy), terrains, s,
                 [(100_000 * s + 1000 * w + e, None) for e in range(a.episodes)])
                for s in range(len(stages_cfg)) if keep(s) for w in range(a.workers)]
        label = f"départs tirés, {len(terrains)} terrains"
    else:
        setups = [u for u in load_setups(bench, stages_cfg) if keep(u["stage"])]
        eps = [(10_000 * k + u["index"], u) for k in range(a.repeats) for u in setups]
        terrains = sorted({u["terrain"] for u in setups})
        jobs = [((config, overrides), str(policy), terrains, 0, eps[w::a.workers])
                for w in range(a.workers) if eps[w::a.workers]]
        label = f"benchmark {bench} : {len(setups)} setups x {a.repeats}"
    with get_context("spawn").Pool(a.workers) as pool:
        rows = [r for chunk in pool.map(_worker, jobs) for r in chunk]
    print(f"\npolicy {policy}\nconfig {config} {' '.join(overrides)} | {label}")
    _report(rows, stages_cfg)
    out = policy.with_name(policy.stem + ("_eval_random.csv" if a.random else f"_eval_{bench}.csv"))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} épisodes -> {out}")
    return sum(r["outcome"] == "success" for r in rows) / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, nargs="+", help="une ou plusieurs policies (ex. plusieurs checkpoints)")
    ap.add_argument("--config", default=None, help="défaut : config.toml + overrides.txt du run, sinon config_l1.toml")
    ap.add_argument("--set", nargs="*", default=[], metavar="section.clé=valeur", help="surcharges en plus")
    ap.add_argument("--bench", default=None, help="scène de benchmark (défaut : [terrains].bench de la config)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3, help="benchmark : passages par setup")
    ap.add_argument("--random", action="store_true", help="départs tirés au lieu du benchmark")
    ap.add_argument("--episodes", type=int, default=10, help="--random : épisodes par étape et par worker")
    ap.add_argument("--stages", type=int, nargs="*", default=None, help="limiter à ces étapes")
    a = ap.parse_args()
    res = [(str(pol), eval_one(Path(pol), a)) for pol in a.policy]
    if len(res) > 1:
        print("\n=== synthèse")
        for pol, r in sorted(res, key=lambda t: -t[1]):
            print(f"  {r:5.1%}  {pol}")


if __name__ == "__main__":
    main()
