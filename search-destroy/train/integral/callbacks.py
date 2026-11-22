"""Curriculum de l'agent intégral."""

from collections import deque
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from train.base_env import FINAL_STAGE
from train.sb3_policy import export_policy

OUTCOMES = ("success", "collision", "out_of_arena", "timeout")


class CurriculumCallback(BaseCallback):
    """Suit les issues des épisodes de l'étape courante et passe à l'étape suivante au seuil."""

    def __init__(self, cfg, stage, out):
        super().__init__()
        self.cur, self.stage, self.out = cfg["curriculum"], stage, Path(out)
        self.recent = deque(maxlen=self.cur["window"])

    def _on_training_start(self):
        self.training_env.env_method("set_stage", self.stage)

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and info.get("outcome") and info.get("stage") == self.stage:
                self.recent.append(info["outcome"])
        n = len(self.recent)
        rates = {o: (sum(1 for x in self.recent if x == o) / n if n else 0.) for o in OUTCOMES}
        if self.n_calls % 256 == 0:
            self.logger.record("curriculum/stage", self.stage)
            self.logger.record("curriculum/episodes_in_window", n)
            for o in OUTCOMES:
                self.logger.record(f"curriculum/{o}_rate", rates[o])
        if (self.stage < FINAL_STAGE and n >= self.cur["min_episodes"]
                and rates["success"] >= self.cur["thresholds"][self.stage]):
            print(f"[curriculum] étape {self.stage} réussie ({rates['success']:.0%} sur {n} épisodes, "
                  f"{self.num_timesteps} pas) -> étape {self.stage + 1}", flush=True)
            self.model.save(self.out / "ckpt" / f"stage{self.stage}_done.zip")
            export_policy(self.model, self.out / "ckpt" / f"policy_stage{self.stage}_done.pt")
            self.stage += 1
            self.recent.clear()
            self.training_env.env_method("set_stage", self.stage)
        return True
