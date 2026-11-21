# Vérification du pipeline RL sur la machine d'entraînement

À donner à Claude sur la machine distante. Objectif : s'assurer que tout le pipeline fonctionne
**là où il va tourner**, corriger les bugs, puis lancer l'entraînement.

## Contexte
Le projet `search-destroy` (hackathon EDTH) : un drone FPV simulé doit retrouver et toucher une boîte
orange cachée dans un terrain (voir `README.md`). La livraison est le dossier `handin/`. Un pipeline RL a
été écrit ; **rien n'a jamais été exécuté de bout en bout**. Certains morceaux ont été testés
séparément (voir section 3), d'autres jamais.

## Règles
- Ne pas modifier `fpv_sim/`, `tests/`, `fpv_sim/scripts/evaluate.py` (fournis par les organisateurs).
- `handin/` doit rester autonome : il ne doit rien importer depuis `train/` ni `fpv_sim/` (sauf ce qui existe déjà).
  Il peut utiliser numpy et torch, pas stable-baselines3.
- Le simulateur ne doit jamais fournir à la policy d'autre information que `observation` (README, « What you build »).
  `distance_m`, la position de la cible, etc. ne servent que dans `train/` (reward, placement du drone).
- Corriger les bugs de façon minimale, noter chaque correction, et la rapporter à l'utilisateur à la fin.
- Ne rien supprimer. Ne pas lancer l'entraînement long avant que les sections 1 à 5 soient vertes.

## Carte du pipeline
| Fichier | Rôle |
|---|---|
| `handin/estimator.py` | estimateur d'état : attitude (gyro), vitesse (accéléro + VIO), altitude (Kalman) |
| `handin/features.py` | depth réduite, cartes locale (1 m) et globale (12,5 m), filtre de cible, vecteur de 38 valeurs |
| `handin/policy_net.py` | réseau PyTorch pur (`FPVNet`, `ActorNet`, `load_actor`) |
| `handin/agent.py` | agent : features + réseau ; hover si poids absents ou erreur ; env `FPV_WEIGHTS` optionnelle |
| `train/config.toml`, `train/config.py` | tous les réglages, surcharges `--set section.clé=valeur` |
| `train/env_wrapper.py` | `FPVTrainEnv` : départs, curriculum, reward, fins d'épisode |
| `train/sb3_policy.py`, `train/train.py` | PPO (stable-baselines3), curriculum automatique, checkpoints, export `policy.pt` |
| `train/eval_holdout.py` | évalue un `policy.pt` sur les terrains de validation, protocole des mentors |
| `train/check_estimator.py`, `train/check_features.py` | bancs de test contre la vérité terrain |

## 1. Environnement de la machine
1. `nproc`, `free -h`, `nvidia-smi`, `df -h` : l'utilisateur a indiqué 28 threads (i7-14700K), 62 Go, RTX 4000 Ada.
2. Le `$HOME` est sur un **NFS** (lent). Mettre le repo, `.venv` et surtout `run.out_dir` sur un **disque local**
   (`df -h /tmp /scratch`). `out_dir` par défaut : `/tmp/fpv_runs`.
3. La machine est peut-être partagée : `who`, `htop`. Laisser 4 threads libres (`run.envs = 24` sur 28).
4. `uv sync`, puis `uv run python -m handin.check_handin` (doit afficher « Hand-in API OK » ; l'agent renvoie un hover
   tant qu'il n'y a pas de `handin/policy.pt`).
5. Vérifier que torch voit le GPU : `uv run python -c "import torch; print(torch.cuda.is_available())"`.
6. Vérifier les imports de `stable_baselines3`, `gymnasium`, `tomllib` (Python 3.12).
7. Benchmark du débit (sert à dimensionner) : 1, 8, 16, 24 workers, `ApproachAviary(terrain='city@v1')`,
   300 `env.step([0,0,0,0])` chacun. Mesure locale : 24 à 39 steps/s par cœur.

## 2. Vérifications statiques
- `uv run python -m py_compile handin/*.py train/*.py`
- Importer chaque module un par un ; corriger toute erreur d'import ou de syntaxe.

## 3. Vérifications par bloc

### A. Estimateur (`handin/estimator.py`) — testé en local, à reconfirmer
`uv run python -m train.check_estimator --runs 8 --seconds 30`
Résultats obtenus (erreur RMS, après les 3 premières secondes) : roll/pitch ~1°, yaw ~2,4° (max 9°),
vitesse horizontale 0,11 m/s, altitude 3 cm, vitesse verticale 0,19 m/s.
Si les chiffres sont très différents, investiguer avant de continuer.
Note : un recalage de roll/pitch sur un modèle de commande (`RP_GAIN`) a été essayé et **désactivé** (il dégradait l'estimation).

### B à E. Features (`handin/features.py`) — testé sur UN seul vol (village@v1)
`uv run python -m train.check_features --terrain village@v1 --out <dossier>` puis `forest@v4`, `city@v2`, `field@v1`.
Observé sur village@v1 : toutes les formes et valeurs correctes (aucun NaN), 3,2 ms par step en moyenne,
position de la cible à ~2 m de la vérité 3 s après la première estimation.
À vérifier :
- Les 4 terrains passent sans assertion, et le temps de calcul reste < 10 ms en moyenne.
- Regarder les images `maps_*.png` : la carte locale doit montrer les obstacles devant le drone, les zones de sol vues
  le long de la trajectoire, la cible estimée proche de la vraie.
- **Point non expliqué** : la reprojection depth → 3D comparée à des raycasts PyBullet donne une erreur médiane de
  0,58 m (p95 0,89 m). Un test séparé a trouvé un rapport profondeur / raycast planaire de 0,996 (échelle correcte), donc
  l'écart vient probablement des pixels de bord et des incidences rasantes. À confirmer ou à corriger (le décalage
  pourrait être un demi-pixel).
- Vérifier aussi : une `box` normalisée se convertit en pixels de la depth par `col = x * 224`, `row = y * 126`.

### Point 2 connu : correctif de σ du filtre de cible — NON APPLIQUÉ
Le filtre de cible annonce σ = 0,05 m pour une erreur réelle de 1,7 m (la depth touche la face de la boîte, qui fait
4 m de long, pas son centre). Dans `handin/features.py`, fonction `_detection` :
- remplacer `sig = max(.5, .05 * dmed)` par `sig = max(1.5, .05 * dmed)` ;
- remplacer `s_lat = max(.5, rng_h * TargetFilter.ANG_ERR)` par `s_lat = max(1.5, rng_h * TargetFilter.ANG_ERR)` ;
- remplacer `s_al = min(200., max(.5, s * ...` par `s_al = min(200., max(1.5, s * ...`.
Puis relancer `check_features` : σ annoncé et erreur réelle doivent être du même ordre.

### Environnement d'entraînement (`train/env_wrapper.py`) — jamais exécuté
Écrire un court script (hors `handin/`) qui, pour chaque scène (field, forest, village, city) et chaque étape 0 à 3 :
1. crée `FPVTrainEnv(cfg, train_terrains, stage=s)` ; `reset()` ; contrôle que l'observation respecte `observation_space`,
   que `set_stage` change l'étape, que `info['stage']` est cohérent ;
2. vérifie que le terrain chargé correspond bien à `sim.TERRAIN` : changer de terrain entre deux resets modifie
   `len(sim.TERRAIN_IDS)` et la position de la boîte égale celle du JSON ;
3. vérifie que le départ est dégagé (`p.getClosestPoints(drone, body, 2.)` vide pour tous les `TERRAIN_IDS`),
   que la distance de départ à la cible est dans la plage de l'étape, que le cap est à ±20° du centre ;
4. fait ~200 steps avec des actions aléatoires puis nulles : reward finie, issues possibles `success / collision /
   out_of_arena / timeout`, `timeout` correct (truncated) ;
5. contrôle que `train_terrains` et `holdout` sont disjoints (`train.config.terrain_split`).
Hypothèses non vérifiées : changer `sim.TERRAIN` avant `reset()` recharge bien la carte ; déplacer le drone après le
reset (`resetBasePositionAndOrientation` + `_updateAndStoreKinematicInformation`) équivaut à un reset avec cette position.
Pour une comparaison de référence, `fpv_sim/scripts/evaluate.py::random_start` fait le départ officiel.

### Réseau et export (`handin/policy_net.py`, `train/sb3_policy.py`)
1. Passe avant de `ActorNet` sur un batch aléatoire aux formes de `handin.features.SHAPES` : sortie `(B, 4)`.
2. Création d'un `PPO("MultiInputPolicy", ...)` minimal (voir `train/train.py`), `export_policy`, puis `load_actor` :
   les sorties doivent être **identiques** (à 1e-5) à `model.predict(obs, deterministic=True)` avant clipping.
3. Vérifier que SB3 ne traite pas les cartes `float32` à 3 dimensions comme des images (il faut `uint8` pour ça)
   et que l'extracteur reçoit bien les tenseurs aux formes d'origine, pas aplatis.

### Entraînement (`train/train.py`) — essai de fumée
`uv run python -m train.train --set run.envs=4 run.total_steps=200000 run.save_every_steps=50000 run.out_dir=<disque local>`
Vérifier :
- pas de crash au démarrage (le mode `spawn` des workers, l'import de `train.env_wrapper` par cloudpickle) ;
- le débit (steps/s, ligne `fps` de SB3) ;
- `tb/` est créé, TensorBoard montre `curriculum/stage` et les taux d'issue ;
- `ckpt/` contient des `model_*.zip` et `policy_*.pt`, et `policy.pt` existe à la racine du run ;
- `holdout_terrains.txt` ne contient aucun terrain d'entraînement ;
- relancer avec `--resume <ckpt>.zip` fonctionne ;
- Ctrl-C écrit `model_interrupted.zip` et `policy.pt`.

### Agent et validation
1. Copier le `policy.pt` du run dans `handin/` : `uv run python -m handin.check_handin` doit passer, sans message d'erreur sur stderr.
2. Mesurer la latence d'`act` (cible : bien sous 33 ms) : features ~3 ms + réseau.
3. `uv run python -m train.eval_holdout --run <dossier du run> --runs 1 --workers 4` : doit tourner, écrire
   `flights.csv` et `metrics.json`, et imprimer un tableau par groupe (field, forest, village, city, ALL).
   Un policy encore à l'étape 0 aura ~0 % de succès à ce test (le départ officiel est à 250 m) : c'est normal.
4. Vérifier que `FPV_WEIGHTS` est bien pris en compte (changer le chemin change le comportement).

## 4. Points ouverts, connus
- **Validation incomplète** : les terrains de validation sont de nouvelles *positions de boîte* sur les **mêmes 4 cartes**,
  pas de nouvelles cartes (l'évaluation officielle utilisera de nouvelles cartes). Les nouvelles cartes sont laissées de côté pour l'instant.
- **Risque de mémorisation** des 4 cartes par le réseau (la position absolue est dans le vecteur d'entrée).
- Le yaw estimé a ~2,4° d'erreur RMS : un obstacle vu à 40 m peut être décalé de 1 à 2 cellules dans la carte (1 m).
- Le bonus de nouvelle zone vue (`reward.coverage`, actif à partir de l'étape 2) n'a jamais été calibré.
- Les seuils de passage d'étape (`curriculum.thresholds = [0.6, 0.5, 0.4]`) sont des valeurs de départ, à ajuster.
- Pas de LSTM, critique partagé avec l'acteur (un critique asymétrique est une amélioration possible plus tard).

## 5. Lancement de nuit (une fois les sections 1 à 4 vertes)
```sh
tmux new -s train
uv run python -m train.train              # config.toml par défaut, 24 envs
# détacher : Ctrl-b d ; suivre : tensorboard --logdir <out_dir>/<date>/tb
```
Surveiller dans TensorBoard : `curriculum/success_rate` à l'étape 0 doit monter dans les premières heures.
S'il reste proche de 0, le problème vient de la reward ou des observations, pas du budget de calcul.

## 6. À rapporter à l'utilisateur
Pour chaque section : vert / rouge, les corrections faites (fichier et raison), les chiffres mesurés (débit, latence,
erreurs de l'estimateur), et tout comportement surprenant.
