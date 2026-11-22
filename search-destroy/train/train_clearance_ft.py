"""Affinage : train.train_clearance + learning rate forcé à la reprise.

    FPV_FINETUNE_LR=1e-4 uv run python -m train.train_clearance_ft --resume <ckpt>.zip --set ...

En reprise, SB3 recharge le learning rate sauvegardé dans le checkpoint (3e-4) et ignore --set ppo.lr.
Ici, PPO.load reçoit custom_objects={learning_rate, lr_schedule} = FPV_FINETUNE_LR.
Sans FPV_FINETUNE_LR, identique à train.train_clearance.
"""

import os

import train.train as base
from train.env_clearance import make_env_fn

base.make_env_fn = make_env_fn

_lr = os.environ.get("FPV_FINETUNE_LR")
if _lr:
    _lr = float(_lr)
    _load = base.PPO.load

    def _load_with_lr(*args, **kwargs):
        custom = dict(kwargs.pop("custom_objects", None) or {})
        custom.update(learning_rate=_lr, lr_schedule=lambda _progress: _lr)
        return _load(*args, custom_objects=custom, **kwargs)

    base.PPO.load = _load_with_lr
    print(f"[affinage] learning rate forcé à {_lr:g}", flush=True)

if __name__ == "__main__":
    base.main()
