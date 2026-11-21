"""Entraînement PPO de l'agent d'approche (cible déjà vue), séparé de train/train.py.

    uv run python -m train.approach.train                                         # config_l1.toml
    uv run python -m train.approach.train --set run.envs=4 run.total_steps=200000  # essai rapide
    uv run python -m train.approach.train --resume <run>/ckpt/model_5000000.zip --set curriculum.start_stage=2

Sorties dans <out_dir>/approach_<date>/ : config.toml, overrides.txt, holdout_terrains.txt, tb/, ckpt/, policy.pt.
TensorBoard : curriculum/* (étape, taux par issue) et approach/* (durée, part du temps sans cible, distance de départ).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch import nn

from handin.policy_net import PI_HIDDEN
from train.approach import config as cfgmod
from train.approach.env import make_approach_env_fn
from train.approach.sb3 import SB3ApproachExtractor, export_policy

CONFIG = str(Path(__file__).with_name("config_l1.toml"))
OUTCOMES = ("success", "collision", "out_of_arena", "timeout")


class ApproachCurriculumCallback(BaseCallback):
    """Taux par issue sur l'étape courante, passage à l'étape suivante au seuil, stats d'approche."""

    def __init__(self, cfg, stage, out):
        super().__init__()
        self.cur, self.stage, self.out = cfg["curriculum"], stage, Path(out)
        self.final = len(self.cur["stages"]) - 1
        self.recent = deque(maxlen=self.cur["window"])

    def _on_training_start(self):
        self.training_env.env_method("set_stage", self.stage)

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and info.get("outcome") and info.get("stage") == self.stage:
                self.recent.append(info)
        n = len(self.recent)
        rate = lambda o: sum(1 for i in self.recent if i["outcome"] == o) / n if n else 0.
        if self.n_calls % 256 == 0:
            self.logger.record("curriculum/stage", self.stage)
            self.logger.record("curriculum/episodes_in_window", n)
            for o in OUTCOMES:
                self.logger.record(f"curriculum/{o}_rate", rate(o))
            if n:
                succ = [i["t_s"] for i in self.recent if i["outcome"] == "success"]
                self.logger.record("approach/lost_frac", sum(i["lost_frac"] for i in self.recent) / n)
                self.logger.record("approach/d0_mean_m", sum(i["d0"] for i in self.recent) / n)
                self.logger.record("approach/reject_rate", self.recent[-1]["reject_rate"])
                if succ:
                    self.logger.record("approach/t_success_s", sum(succ) / len(succ))
        if (self.stage < self.final and n >= self.cur["min_episodes"]
                and rate("success") >= self.cur["thresholds"][self.stage]):
            print(f"[curriculum] étape {self.stage} réussie ({rate('success'):.0%} sur {n} épisodes, "
                  f"{self.num_timesteps} pas) -> étape {self.stage + 1}", flush=True)
            self.model.save(self.out / "ckpt" / f"stage{self.stage}_done.zip")
            export_policy(self.model, self.out / "ckpt" / f"policy_stage{self.stage}_done.pt")
            self.stage += 1
            self.recent.clear()
            self.training_env.env_method("set_stage", self.stage)
        return True


class CheckpointCallback(BaseCallback):
    """Checkpoint complet + export de policy.pt tous les `every` pas (copie de train.train)."""

    def __init__(self, every, out):
        super().__init__()
        self.every, self.out, self.next = every, Path(out), every

    def _save(self):
        n = self.num_timesteps
        self.model.save(self.out / "ckpt" / f"model_{n}.zip")
        export_policy(self.model, self.out / "ckpt" / f"policy_{n}.pt")
        export_policy(self.model, self.out / "policy.pt")
        print(f"[checkpoint] {n} pas -> {self.out / 'policy.pt'}", flush=True)

    def _on_training_start(self):
        self.next = (self.num_timesteps // self.every + 1) * self.every

    def _on_step(self):
        if self.num_timesteps >= self.next:
            self._save()
            self.next += self.every
        return True

    def _on_training_end(self):
        self._save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--set", nargs="*", default=[], metavar="section.clé=valeur")
    ap.add_argument("--resume", default=None, help="checkpoint .zip à reprendre")
    a = ap.parse_args()

    cfg = cfgmod.load(a.config, a.set)
    run, ppo = cfg["run"], cfg["ppo"]
    train_terrains, holdout = cfgmod.terrain_split(cfg)
    out = Path(run["out_dir"]) / datetime.now().strftime("approach_%Y%m%d-%H%M%S")
    (out / "ckpt").mkdir(parents=True)
    cfgmod.save_copy(out, a.config, a.set)
    (out / "holdout_terrains.txt").write_text("\n".join(holdout) + "\n")
    print(f"[run] {out} | {len(train_terrains)} terrains d'entraînement, {len(holdout)} de validation", flush=True)

    stage = int(cfg["curriculum"]["start_stage"])
    n_envs = run["envs"]
    if (ppo["n_steps"] * n_envs) % ppo["batch_size"]:
        print(f"[attention] rollout {ppo['n_steps'] * n_envs} pas n'est pas un multiple de batch_size", flush=True)
    torch.set_num_threads(4)

    venv = VecMonitor(SubprocVecEnv([make_approach_env_fn(cfg, train_terrains, i, stage) for i in range(n_envs)],
                                    start_method="spawn"))
    if a.resume:
        model = PPO.load(a.resume, env=venv, device="auto", tensorboard_log=str(out / "tb"))
    else:
        model = PPO(
            "MultiInputPolicy", venv,
            learning_rate=ppo["lr"], n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"], gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"],
            ent_coef=ppo["ent_coef"], clip_range=ppo["clip_range"], max_grad_norm=ppo["max_grad_norm"],
            policy_kwargs=dict(features_extractor_class=SB3ApproachExtractor,
                               net_arch=dict(pi=list(PI_HIDDEN), vf=list(PI_HIDDEN)),
                               activation_fn=nn.Tanh, log_std_init=ppo["log_std_init"]),
            tensorboard_log=str(out / "tb"), seed=run["seed"], device="auto", verbose=1)

    callbacks = CallbackList([ApproachCurriculumCallback(cfg, stage, out),
                              CheckpointCallback(run["save_every_steps"], out)])
    try:
        model.learn(total_timesteps=run["total_steps"], callback=callbacks, tb_log_name="ppo",
                    reset_num_timesteps=a.resume is None)
    except KeyboardInterrupt:
        print("[interrompu] sauvegarde finale", flush=True)
        model.save(out / "ckpt" / "model_interrupted.zip")
        export_policy(model, out / "policy.pt")
    finally:
        try:
            venv.close()
        except (EOFError, BrokenPipeError, ConnectionResetError):
            pass


if __name__ == "__main__":
    main()
