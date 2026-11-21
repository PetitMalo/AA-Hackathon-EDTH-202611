#!/bin/bash
mkdir -p /Data/arthur.fournier/home_offload/{venvs/search-destroy,local_share/uv,cache} /Data/arthur.fournier/fpv_runs
cd ~/AA-Hackathon-EDTH-202611/search-destroy
uv sync > /Data/arthur.fournier/fpv_runs/uv_sync.log 2>&1
exec uv run fpv-train --resume /Data/arthur.fournier/fpv_base/manche_v1.zip --set run.total_steps=40000000 run.out_dir='"/Data/arthur.fournier/fpv_runs"' reward.smooth=0.0 reward.progress=2.0 reward.timeout=-20.0 reward.coverage=0.5 reward.coverage_from_stage=0 reward.time=0.0021 2>&1 | tee /Data/arthur.fournier/fpv_runs/manche_v1_t21.log
