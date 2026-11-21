"""Entraînement PPO.

    uv run python -m train.train                                   # config.toml par défaut
    uv run python -m train.train --config train/config_b.toml
    uv run python -m train.train --set run.envs=4 run.total_steps=200000     # essai rapide
    uv run python -m train.train --resume /tmp/fpv_runs/<run>/ckpt/model_5000000.zip --set curriculum.start_stage=1

Sorties dans <out_dir>/<date>/ : config.toml, overrides.txt, tb/ (TensorBoard), ckpt/, policy.pt (dernière).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # un thread par worker, sinon les 24 envs se gênent

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
from train import config as cfgmod
from train.env_wrapper import FINAL_STAGE, make_env_fn
from train.explore_env import make_explore_env_fn
from train.sb3_policy import SB3Extractor, export_policy

OUTCOMES = ("success", "collision", "out_of_arena", "timeout")
EXPLORE_OUTCOMES = ("detected", "success", "collision", "out_of_arena", "timeout", "coverage_done")


class ExploreStatsCallback(BaseCallback):
    """Explorateur : couverture finale, taux et temps de détection, issues, sur les derniers épisodes."""

    def __init__(self, window=300):
        super().__init__()
        self.recent = deque(maxlen=window)

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and info.get("outcome"):
                self.recent.append(info)
        if self.n_calls % 256 == 0 and self.recent:
            n = len(self.recent)
            det = [i["t_detect"] for i in self.recent if i["detected"]]
            self.logger.record("explore/episodes_in_window", n)
            self.logger.record("explore/coverage_mean", sum(i["coverage"] for i in self.recent) / n)
            self.logger.record("explore/detection_rate", len(det) / n)
            if det:
                self.logger.record("explore/t_detect_mean_s", sum(det) / len(det))
            for o in EXPLORE_OUTCOMES:
                self.logger.record(f"explore/{o}_rate", sum(1 for i in self.recent if i["outcome"] == o) / n)
        return True


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
