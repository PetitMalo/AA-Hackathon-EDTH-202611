"""Évalue handin.combined_agent avec le protocole officiel (run_target de fpv_sim.scripts.evaluate).

    uv run python -m train.eval_combined --explorer E.pt --approach A.pt --terrains-file holdout_terrains.txt
    uv run python -m train.eval_combined ... --lost-s 4 --handoff-m 60 --runs 3 --workers 20

Dans chaque worker, handin.agent est remplacé par handin.combined_agent avant d'appeler run_target :
aucun fichier partagé n'est modifié. Sortie : flights.csv + metrics.json dans --out.
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from multiprocessing import get_context
from pathlib import Path


def _job(job):
    import handin.combined_agent as combined
    sys.modules["handin.agent"] = combined            # run_target fait `from handin.agent import Agent`
    from fpv_sim.scripts.evaluate import run_target
    return run_target(job)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--explorer", type=Path, required=True)
    ap.add_argument("--approach", type=Path, required=True)
    ap.add_argument("--terrains-file", type=Path, required=True, help="un terrain@vN par ligne")
    ap.add_argument("--lost-s", type=float, default=4.)
    ap.add_argument("--handoff-m", type=float, default=float("inf"))
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--timeout", type=float, default=120.)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    os.environ.update(FPV_EXPLORER=str(a.explorer.resolve()), FPV_APPROACH=str(a.approach.resolve()),
                      FPV_LOST_S=str(a.lost_s), FPV_HANDOFF_M=str(a.handoff_m))   # hérités par les workers spawn
    terrains = a.terrains_file.read_text().split()
    out = a.out or Path("/tmp/fpv_eval_combined") / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[combined] {len(terrains)} terrains x {a.runs} vols, lost {a.lost_s}s, handoff {a.handoff_m} m", flush=True)

    jobs = [(t, a.runs, out, a.seed, a.timeout, 10**9) for t in terrains]
    records = []
    with get_context("spawn").Pool(a.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_job, jobs), 1):
            records += result
            print(f"[{i}/{len(jobs)}] {result[0]['terrain']}: "
                  f"{sum(r['outcome'] == 'success' for r in result)}/{len(result)}", flush=True)

    from fpv_sim.scripts.evaluate import summarize
    records.sort(key=lambda r: (r["terrain"], r["run"]))
    with open(out / "flights.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    metrics = summarize(records)
    n = len(records)
    metrics["ALL"] = {"flights": n, **{f"{o}_rate": sum(r["outcome"] == o for r in records) / n
                                       for o in ("success", "collision", "timeout", "out_of_arena", "error")},
                      "detection_rate": sum(r["seen"] for r in records) / n}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"\n{'groupe':8} {'vols':>5} {'reach':>6} {'crash':>6} {'t/o':>5} {'out':>5} {'err':>5} {'seen':>5}")
    for name, m in metrics.items():
        print(f"{name:8} {m['flights']:5d} {m['success_rate']:6.0%} {m['collision_rate']:6.0%} {m['timeout_rate']:5.0%} "
              f"{m['out_of_arena_rate']:5.0%} {m['error_rate']:5.0%} {m['detection_rate']:5.0%}")
    print(f"\nflights.csv et metrics.json dans {out}")


if __name__ == "__main__":
    main()
