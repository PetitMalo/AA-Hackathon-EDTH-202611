"""Entraînement PPO de integral/ et explore/ (choix : run.task de la config). Approche : train.approach.train.

    uv run python -m train.train                                   # intégral : integral/config.toml
    uv run python -m train.train --config train/explore/config.toml    # explorateur
    uv run python -m train.train --set run.envs=4 run.total_steps=200000     # essai rapide
    uv run python -m train.train --resume /tmp/fpv_runs/<run>/ckpt/model_5000000.zip --set curriculum.start_stage=1

Sorties dans <out_dir>/<date>/ : config.toml, overrides.txt, tb/ (TensorBoard), ckpt/, policy.pt (dernière).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # un thread par worker, sinon les 24 envs se gênent

import argparse
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from torch import nn

from handin.policy_net import PI_HIDDEN
from train import config as cfgmod
from train.base_env import FINAL_STAGE
from train.explore.callbacks import ExploreStatsCallback
from train.explore.env import make_explore_env_fn
from train.integral.callbacks import CurriculumCallback
from train.integral.env import make_env_fn
from train.sb3_policy import SB3Extractor, export_policy


class CheckpointCallback(BaseCallback):
    """Checkpoint complet + export de policy.pt tous les `every` pas."""

    def __init__(self, every, out):
        super().__init__()
        self.every, self.out, self.next = every, Path(out), every

    def _save(self):
        n = self.num_timesteps
        self.model.save(self.out / "ckpt" / f"model_{n}.zip")
        export_policy(self.model, self.out / "ckpt" / f"policy_{n}.pt")
        export_policy(self.model, self.out / "policy.pt")
        print(f"[checkpoint] {n} pas -> {self.out / 'policy.pt'}", flush=True)

    def _on_training_start(self):           # --resume : prochain multiple de `every` après le compteur repris
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
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="section.clé=valeur")
    ap.add_argument("--resume", default=None, help="checkpoint .zip à reprendre")
    a = ap.parse_args()

    cfg = cfgmod.load(a.config, a.set)
    run, ppo = cfg["run"], cfg["ppo"]
    train_terrains, holdout = cfgmod.terrain_split(cfg)
    out = Path(run["out_dir"]) / datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / "ckpt").mkdir(parents=True)
    cfgmod.save_copy(out, a.config, a.set)
    (out / "holdout_terrains.txt").write_text("\n".join(holdout) + "\n")
    print(f"[run] {out} | {len(train_terrains)} terrains d'entraînement, {len(holdout)} de validation", flush=True)

    explore = run.get("task", "approach") == "explore"
    stage = FINAL_STAGE if explore else int(cfg["curriculum"]["start_stage"])
    n_envs = run["envs"]
    rollout = ppo["n_steps"] * n_envs
    if rollout % ppo["batch_size"]:
        print(f"[attention] rollout {rollout} pas n'est pas un multiple de batch_size {ppo['batch_size']}", flush=True)
    torch.set_num_threads(4)

    fns = [make_explore_env_fn(cfg, train_terrains, i) if explore else make_env_fn(cfg, train_terrains, i, stage)
           for i in range(n_envs)]
    venv = SubprocVecEnv(fns, start_method="spawn")
    venv = VecMonitor(venv)

    if a.resume:
        model = PPO.load(a.resume, env=venv, device="auto", tensorboard_log=str(out / "tb"))
    else:
        model = PPO(
            "MultiInputPolicy", venv,
            learning_rate=ppo["lr"], n_steps=ppo["n_steps"], batch_size=ppo["batch_size"],
            n_epochs=ppo["n_epochs"], gamma=ppo["gamma"], gae_lambda=ppo["gae_lambda"],
            ent_coef=ppo["ent_coef"], clip_range=ppo["clip_range"], max_grad_norm=ppo["max_grad_norm"],
            policy_kwargs=dict(features_extractor_class=SB3Extractor,
                               net_arch=dict(pi=list(PI_HIDDEN), vf=list(PI_HIDDEN)),
                               activation_fn=nn.Tanh, log_std_init=ppo["log_std_init"]),
            tensorboard_log=str(out / "tb"), seed=run["seed"], device="auto", verbose=1)

    first = ExploreStatsCallback() if explore else CurriculumCallback(cfg, stage, out)
    callbacks = CallbackList([first, CheckpointCallback(run["save_every_steps"], out)])
    try:
        model.learn(total_timesteps=run["total_steps"], callback=callbacks, tb_log_name="ppo",
                    reset_num_timesteps=a.resume is None)
    except KeyboardInterrupt:                       # Ctrl-C : on garde quand même l'état courant
        print("[interrompu] sauvegarde finale", flush=True)
        model.save(out / "ckpt" / "model_interrupted.zip")
        export_policy(model, out / "policy.pt")
    finally:
        try:
            venv.close()
        except (EOFError, BrokenPipeError, ConnectionResetError):   # workers déjà tués par le Ctrl-C
            pass


if __name__ == "__main__":
    main()
