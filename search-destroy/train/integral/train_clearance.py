"""train.train avec l'environnement à pénalité de proximité des obstacles (train/integral/env_clearance.py).

    uv run python -m train.integral.train_clearance --resume <ckpt>.zip --set reward.clearance=0.1 reward.clearance_m=1.5 ...

Mêmes options, sorties et checkpoints que train.train : seule la fabrique d'environnements change.
"""

import train.train as base
from train.integral.env_clearance import make_env_fn

base.make_env_fn = make_env_fn          # train.train.main() appelle make_env_fn depuis son propre espace de noms

if __name__ == "__main__":
    base.main()
