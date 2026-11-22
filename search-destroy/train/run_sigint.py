"""Lance train.train avec Ctrl-C (SIGINT) réactivé.

Un processus lancé en arrière-plan (`... &`) par un shell non interactif ignore SIGINT, et Python
hérite de ce réglage : `kill -INT` ne déclenche alors plus la sauvegarde finale de train.train.
Ce lanceur remet le gestionnaire Python par défaut puis exécute train.train à l'identique.

    python -m train.run_sigint --config ... --set ...      # mêmes arguments que train.train
"""

import runpy
import signal
import sys

signal.signal(signal.SIGINT, signal.default_int_handler)
sys.argv = ["train.train", *sys.argv[1:]]
runpy.run_module("train.train", run_name="__main__", alter_sys=True)
