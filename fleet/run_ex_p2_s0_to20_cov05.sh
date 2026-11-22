#!/bin/bash
cd ~/AA-Hackathon-EDTH-202611/search-destroy
exec uv run fpv-train --resume /Data/arthur.fournier/fpv_base/loire_2654k.zip --set run.total_steps=40000000 run.out_dir='"/Data/arthur.fournier/fpv_runs"' reward.smooth=0.0 reward.progress=2.0 reward.timeout=-20.0 reward.coverage=0.5 reward.coverage_from_stage=0 2>&1 | tee /Data/arthur.fournier/fpv_runs/ex_p2_s0_to20_cov05.log
