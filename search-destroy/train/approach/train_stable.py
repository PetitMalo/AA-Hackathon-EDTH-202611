"""train.approach.train avec la reward anti-vrille de env_stable.

    uv run python -m train.approach.train_stable --config train/approach/config_stable.toml \
        --init-from <model.zip> --set run.seed=1 reward.spin=0.05 ppo.init_log_std=-1.2

ppo.init_log_std : plancher du bruit d'exploration appliqué après --init-from, pour désapprendre la vrille.
En cas d'exception pendant learn, le modèle est sauvé AVANT venv.close(), qui peut bloquer indéfiniment.
"""

import os
import traceback
from pathlib import Path

from train.approach import train as base
from train.approach.env_stable import make_stable_env_fn


class SafePPO(base.PPO):
    def learn(self, *args, **kwargs):
        try:
            return super().learn(*args, **kwargs)
        except Exception:
            traceback.print_exc()
            out = Path(self.tensorboard_log).parent
            self.save(out / "ckpt" / "model_crash.zip")
            base.export_policy(self, out / "ckpt" / "policy_crash.pt")
            print(f"[crash] modèle sauvé dans {out / 'ckpt'}", flush=True)
            os._exit(1)


class StableCallback(base.ApproachCurriculumCallback):
    KEYS = ("yaw_abs_deg", "spun", "spin_penalty")

    def __init__(self, cfg, stage, out):
        super().__init__(cfg, stage, out)
        self.floor = cfg["ppo"].get("init_log_std")
        self.eps = base.deque(maxlen=300)

    def _on_training_start(self):
        super()._on_training_start()
        if self.floor is not None:
            self.model.policy.log_std.data.clamp_(min=float(self.floor))
            print(f"[stable] log_std >= {self.floor} au départ", flush=True)

    def _on_step(self):
        self.eps.extend(i for i, d in zip(self.locals["infos"], self.locals["dones"]) if d and "spun" in i)
        if self.eps and self.n_calls % 256 == 0:
            for k in self.KEYS:
                self.logger.record(f"stable/{k}", sum(i[k] for i in self.eps) / len(self.eps))
            self.logger.record("stable/success_all", sum(i["outcome"] == "success" for i in self.eps) / len(self.eps))
        return super()._on_step()


if __name__ == "__main__":
    base.PPO = SafePPO
    base.make_approach_env_fn = make_stable_env_fn
    base.ApproachCurriculumCallback = StableCallback
    base.main()
