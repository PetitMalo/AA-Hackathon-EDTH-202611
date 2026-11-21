#!/usr/bin/env bash
# Lance un entraînement en arrière-plan sur une machine distante (survit à la déconnexion).
#
#   train/launch.sh <machine> <nom> <config> [surcharges --set ...]
#   train/launch.sh indre explore_default train/config_explore.toml
#   train/launch.sh niva explore_pure train/config_explore.toml explore.end_on_detect=false explore.coverage_goal=0.8
#   RESUME=runs/explorer.zip train/launch.sh lada approach_from_explorer train/config.toml   # reprise d'un checkpoint
#
# Sorties sur la machine : /tmp/fpv_runs/<nom>/<date>/ (policy.pt, ckpt/, tb/) et log /tmp/fpv_logs/<nom>.log
# Suivi :  ssh <machine> tail -f /tmp/fpv_logs/<nom>.log
# Arrêt :  ssh <machine> pkill -INT -f "fpv_runs/<nom>"     (Ctrl-C propre : sauvegarde finale)
set -euo pipefail
host=$1 name=$2 config=$3; shift 3
repo=$(cd "$(dirname "$0")/.." && pwd)
sets="run.out_dir=/tmp/fpv_runs/$name $*"
resume=${RESUME:+--resume $RESUME}             # RESUME=<checkpoint .zip visible de la machine> train/launch.sh ...
ssh -o BatchMode=yes "$host" "mkdir -p /tmp/fpv_logs && cd '$repo' || exit 1; \
  setsid nohup .venv/bin/python -m train.train --config '$config' $resume --set $sets \
  > /tmp/fpv_logs/$name.log 2>&1 < /dev/null & echo \"[$host] $name lancé, pid \$!\""
