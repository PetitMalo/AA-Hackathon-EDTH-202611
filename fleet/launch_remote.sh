#!/bin/bash
# usage : launch_remote.sh <nom_du_run> [--set ...]   (exécuté sur la machine cible)
set -e
NAME=$1; shift
mkdir -p /Data/arthur.fournier/home_offload/{venvs/search-destroy,local_share/uv,cache} /Data/arthur.fournier/fpv_runs
cd ~/AA-Hackathon-EDTH-202611/search-destroy
LOG=/Data/arthur.fournier/fpv_runs/$NAME.log
uv sync > /Data/arthur.fournier/fpv_runs/uv_sync.log 2>&1
tmux kill-session -t train 2>/dev/null || true
tmux new -d -s train "cd $PWD && uv run fpv-train --set run.total_steps=40000000 run.out_dir='\"/Data/arthur.fournier/fpv_runs\"' $* 2>&1 | tee $LOG"
echo "$(hostname -s) lancé : $NAME"
