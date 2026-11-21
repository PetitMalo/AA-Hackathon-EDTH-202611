#!/bin/bash
cd ~/AA-Hackathon-EDTH-202611/search-destroy
mkdir -p /Data/arthur.fournier/fpv_runs
exec uv run fpv-train --resume /Data/arthur.fournier/fpv_base/loire_2654k.zip --set run.total_steps=40000000 run.out_dir='"/Data/arthur.fournier/fpv_runs"' reward.smooth=0.001 reward.progress=1.0 reward.timeout=-15.0 2>&1 | tee /Data/arthur.fournier/fpv_runs/rw_p1_s001_to15.log
