#!/bin/bash
# Arrête le run d'approche de cette machine (session tmux "approach", uv sync, python train.approach)
# puis relance launch_approach_remote.sh en détaché. Usage (sur la machine) : restart_approach.sh <nom> [--set ...]
NAME=$1
tmux kill-session -t approach 2>/dev/null
pkill -u "$USER" -f 'python -m train[.]approach[.]train'
pkill -u "$USER" -f '^uv sync --frozen'
pkill -u "$USER" -f 'bash .*/launch_approach_remote[.]sh'
sleep 2
mkdir -p /Data/arthur.fournier/fpv_runs/approach
setsid nohup bash "$(dirname "$0")/launch_approach_remote.sh" "$@" \
  > "/Data/arthur.fournier/fpv_runs/approach/launch_$NAME.log" 2>&1 < /dev/null &
echo "$(hostname -s) relancé : $*"
