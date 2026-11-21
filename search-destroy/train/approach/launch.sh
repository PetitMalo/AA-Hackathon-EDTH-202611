#!/usr/bin/env bash
# Comme train/launch.sh, mais pour l'agent d'approche (train.approach.train). launch.sh reste inchangé.
#
#   train/approach/launch.sh <machine> <nom> [config] [surcharges --set ...]
#   train/approach/launch.sh indre approach_l1
#   train/approach/launch.sh niva approach_l1_fast train/approach/config_l1.toml reward.success_speed=100
#   RESUME=/tmp/fpv_runs/approach_l1/<date>/ckpt/model_5000000.zip train/approach/launch.sh indre approach_l1_b
#
# Sorties sur la machine : /tmp/fpv_runs/<nom>/approach_<date>/ et log /tmp/fpv_logs/<nom>.log
# Suivi :  ssh <machine> tail -f /tmp/fpv_logs/<nom>.log
# Arrêt :  ssh <machine> pkill -INT -f "fpv_runs/<nom>"
set -euo pipefail
host=$1 name=$2; shift 2
config=${1:-train/approach/config_l1.toml}; [ $# -gt 0 ] && shift
repo=$(cd "$(dirname "$0")/../.." && pwd)
sets="run.out_dir=/tmp/fpv_runs/$name $*"
resume=${RESUME:+--resume $RESUME}
ssh -o BatchMode=yes "$host" "mkdir -p /tmp/fpv_logs && cd '$repo' || exit 1; \
  setsid nohup .venv/bin/python -m train.approach.train --config '$config' $resume --set $sets \
  > /tmp/fpv_logs/$name.log 2>&1 < /dev/null & echo \"[$host] $name lancé, pid \$!\""
