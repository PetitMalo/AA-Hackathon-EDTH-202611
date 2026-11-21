#!/bin/bash
cd ~/AA-Hackathon-EDTH-202611/search-destroy
mkdir -p /Data/arthur.fournier/fpv_runs
exec uv run fpv-train --resume /Data/arthur.fournier/fpv_base/loire_2654k.zip --set run.total_steps=40000000 run.out_dir='"/Data/arthur.fournier/fpv_runs"' reward.smooth=0.0 reward.progress=2.0 reward.timeout=-20.0 2>&1 | tee /Data/arthur.fournier/fpv_runs/rw_p2_s0_to20.log
