"""Commandes `uv run fpv-*` (déclarées dans pyproject.toml, [project.scripts]).

Chaque commande exécute le module correspondant exactement comme `python -m <module>`,
les arguments passés après la commande sont transmis tels quels :

    uv run fpv-train [--set ...] [--resume ckpt.zip]     train.train
    uv run fpv-smoke [--set ...]                          train.train, essai court sur 4 envs
    uv run fpv-eval --run <dossier>                       train.integral.eval
    uv run fpv-tb [--logdir ...]                          TensorBoard sur /tmp/fpv_runs
    uv run fpv-check                                      handin.check_handin
    uv run fpv-check-env / -estimator / -features         bancs de test de train/
    uv run fpv-check-explore                              train.explore.check (env d'exploration)
"""

import runpy
import sys

SMOKE = ["--set", "run.envs=4", "run.total_steps=40000", "run.save_every_steps=16384",
         "ppo.batch_size=1024", 'run.out_dir="/tmp/fpv_smoke"']


def _run(module, preset=()):
    sys.argv = [module, *preset, *sys.argv[1:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


def train():
    _run("train.train")


def smoke():
    # un --set ajouté par l'utilisateur remplace celui-ci en entier (argparse garde le dernier)
    _run("train.train", SMOKE)


def evaluate():
    _run("train.integral.eval")


def check_handin():
    _run("handin.check_handin")


def check_env():
    _run("train.integral.check")


def check_estimator():
    _run("train.check_estimator")


def check_features():
    _run("train.check_features")


def check_explore():
    _run("train.explore.check")


def tensorboard():
    from tensorboard.main import run_main
    preset = [] if any(a.startswith("--logdir") for a in sys.argv[1:]) else ["--logdir", "/tmp/fpv_runs"]
    sys.argv = ["tensorboard", *preset, *sys.argv[1:]]
    sys.exit(run_main())
