"""Évalue un policy.pt sur les terrains de VALIDATION d'un run, avec le protocole des mentors.

    uv run python -m train.eval_holdout --run /tmp/fpv_runs/<date> --runs 3 --workers 20
    uv run python -m train.eval_holdout --run /tmp/fpv_runs/<date> --weights /tmp/fpv_runs/<date>/ckpt/policy_5000000.pt

Réutilise run_target et summarize de fpv_sim.scripts.evaluate : même départ officiel (225-275 m du centre),
mêmes issues, mêmes métriques. Seuls les terrains diffèrent : ceux de <run>/holdout_terrains.txt.
Attention : ce sont de nouvelles POSITIONS de boîte sur des cartes déjà vues, pas de nouvelles cartes.
"""

import argparse
import csv
import json
import os
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path

from fpv_sim.scripts.evaluate import BUDGET_MS, run_target, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="dossier du run (contient holdout_terrains.txt)")
    ap.add_argument("--weights", type=Path, default=None, help="défaut : <run>/policy.pt")
    ap.add_argument("--runs", type=int, default=3, help="vols par position de boîte")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--timeout", type=float, default=120.)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    weights = (a.weights or a.run / "policy.pt").resolve()
    if not weights.exists():
        raise SystemExit(f"poids introuvables : {weights}")
    terrains = (a.run / "holdout_terrains.txt").read_text().split()
    out = a.out or a.run / "holdout" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    os.environ["FPV_WEIGHTS"] = str(weights)          # lu par handin.agent dans chaque worker (spawn hérite de l'env)
    print(f"[holdout] {len(terrains)} positions x {a.runs} vols, poids {weights}", flush=True)

    jobs = [(t, a.runs, out, a.seed, a.timeout, 10**9) for t in terrains]     # depth_every énorme : logs légers
    records = []
    with get_context("spawn").Pool(a.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_target, jobs), 1):
            records += result
            ok = sum(r["outcome"] == "success" for r in result)
            print(f"[{i}/{len(jobs)}] {result[0]['terrain']}: {ok}/{len(result)} reached", flush=True)

    records.sort(key=lambda r: (r["terrain"], r["run"]))
    with open(out / "flights.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    metrics = summarize(records)
    n = len(records)
    total = {o: sum(r["outcome"] == o for r in records) / n for o in ("success", "collision", "timeout", "out_of_arena", "error")}
    metrics["ALL"] = {"flights": n, **{f"{o}_rate": v for o, v in total.items()}}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    print(f"\n{'groupe':8} {'vols':>5} {'reach':>6} {'crash':>6} {'t/o':>5} {'out':>5} {'err':>5} {'seen':>5} {'lat_ms':>7}")
    for name, m in metrics.items():
        seen = f"{m['detection_rate']:5.0%}" if "detection_rate" in m else "    -"
        lat = f"{m['latency_mean_ms']:7.2f}" if "latency_mean_ms" in m else "      -"
        print(f"{name:8} {m['flights']:5d} {m['success_rate']:6.0%} {m['collision_rate']:6.0%} {m['timeout_rate']:5.0%} "
              f"{m['out_of_arena_rate']:5.0%} {m['error_rate']:5.0%} {seen} {lat}")
        if m.get("latency_mean_ms", 0) > BUDGET_MS:
            print(f"  WARNING: {name} dépasse {BUDGET_MS:.0f} ms par pas")
    print(f"\nflights.csv et metrics.json dans {out}")


if __name__ == "__main__":
    main()
