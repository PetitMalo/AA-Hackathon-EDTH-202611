#!/bin/bash
# Agent d'approche (train.approach.train), exécuté SUR la machine cible :
#   launch_approach_remote.sh <nom_du_run> [surcharges --set ...]
# Session tmux "approach" (ne touche jamais la session "train" des autres runs).
# Sorties : /Data/arthur.fournier/fpv_runs/approach/<nom>/approach_<date>/, log /Data/arthur.fournier/fpv_runs/approach/<nom>.log
set -e
NAME=$1; shift
mkdir -p /Data/arthur.fournier/home_offload/{venvs/search-destroy,local_share/uv,cache} /Data/arthur.fournier/fpv_runs/approach
cd ~/AA-Hackathon-EDTH-202611/search-destroy
OUT=/Data/arthur.fournier/fpv_runs/approach
uv sync --frozen > $OUT/uv_sync_$NAME.log 2>&1
tmux kill-session -t approach 2>/dev/null || true
tmux new -d -s approach "cd $PWD && uv run --no-sync python -m train.approach.train --set run.out_dir='\"$OUT/$NAME\"' $* 2>&1 | tee $OUT/$NAME.log"
echo "$(hostname -s) lancé : $NAME"
