#!/bin/bash
cd ~/AA-Hackathon-EDTH-202611/search-destroy
exec uv run fpv-train --resume /Data/arthur.fournier/fpv_base/loire_2654k.zip --set run.total_steps=40000000 run.out_dir='"/Data/arthur.fournier/fpv_runs"' reward.smooth=0.001 reward.progress=1.0 reward.timeout=-15.0 curriculum.thresholds=[0.5,0.4,0.3] reward.coverage=0.5 reward.coverage_from_stage=0 2>&1 | tee /Data/arthur.fournier/fpv_runs/ex_p1_to15_fast_cov05.log
