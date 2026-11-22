# Poids de l'agent d'approche

Agents PPO qui foncent sur la boîte **une fois qu'elle a été vue**. Ils ne savent pas chercher : pour une
mission complète, les mettre derrière un explorateur avec `handin/combined_agent.py`.

> **Ne pas déployer tel quel sur les départs level 3 inclinés : 6 à 12 % de touches sans vérité terrain.**
> Les 98-99,7 % du banc d'approche supposent attitude et vitesse vraies au relais. Voir « Problème connu ».
> **Tous les poids mesurés ici tournent en vrille** (≈ 550° de lacet par approche). Voir « Vrille ».

- **Meilleur en approche level 3 : `diuris.pt`** (299/300, 97,2 % sur départs aléatoires, 89 % en mission complète).
- **À égalité en mission, moins de crashs : `saone_1M.pt`** (90 %, 5 % de crashs contre 9 % ; un seul passage de 102 vols).
- `maserati.pt` : meilleur poids entraîné sans level 3. `policy.pt` : son 12 M, gardé comme référence.

### Où sont les fichiers

Les noms sont ceux des machines d'entraînement (état au 22/11, 12:30, `arthur-work` mergé dans `main` à 12:08) :

- `*.pt` : les 11 poids level 3 finaux, `maserati.pt` et `policy.pt`, sur `main` et `arthur-work`.
- `saone_1M.pt` : checkpoint 1M de saone, non commité (98,7 % contre 91,0 % pour `saone.pt`).
- `stable/` : les 42 poids de l'essai anti-vrille, **à ne pas déployer** (voir `stable/README.md`).
- Les autres checkpoints 1M (belgique, hongrie, irlande, sternum, tibia) ne sont plus sur disque ici :
  `git show c01678b:./weights/approach/<machine>.pt`.

## Tous les poids level 3

Banc = `train.approach.eval`, **attitude et vitesse vraies données au relais**. Level 3 : 60 départs x 5 = 300 vols.
Level 2 : 60 x 3 = 180. Aléatoire : 504 départs level 3 tirés, mêmes réglages pour tous.
Mission : `train.eval_combined` + `femur.pt`, 102 vols, seed 1, sans vérité terrain. `-` = non mesuré.

| Poids | Run `l3_*` | Pas | Level 3 | Level 2 | Aléatoire | Mission |
|---|---|--:|--:|--:|--:|--:|
| **`diuris.pt`** | full_c | 866k | **99,7 %** | 98,3 % | 97,2 % | 89 % |
| **`saone_1M.pt`** | col150 | 1,00M | 98,7 % | 98,3 % | 95,6 % | **90 %** |
| `vanilla.pt` | full_a | 921k | 98,7 % | 97,8 % | 95,8 % | - |
| `quiscale.pt` | lr5e-5 | 858k | 98,3 % | 97,8 % | 95,2 % | - |
| `encyclia.pt` | only | 716k | 98,3 % | 97,8 % | 96,4 % | - |
| `maserati.pt` (level 2 seul) | v3_full_c | 14M | 98 % | - | 96,2 % | 86 % |
| `tibia.pt` de `c01678b` | nohardbase | 1,00M | 96,3 % | 95,0 % | - | - |
| `zeuxine.pt` | full_b | 907k | 93,7 % | 97,2 % | 96,8 % | - |
| `hongrie.pt` de `c01678b` | nohard | 1,00M | 93,7 % | 93,9 % | - | - |
| `hongrie.pt` | nohard | 1,34M | 93,7 % | 92,2 % | - | - |
| `irlande.pt` | prox1 | 1,40M | 93,3 % | 91,7 % | - | - |
| `policy.pt` (level 2 seul) | v3_full_c | 12M | 93,0 % | 94,4 % | - | - |
| `tibia.pt` | nohardbase | 1,60M | 92,3 % | 93,3 % | - | - |
| `sternum.pt` | p1base_b | 1,19M | 92,0 % | 93,9 % | - | - |
| `saone.pt` | col150 | 1,70M | 91,0 % | 95,0 % | - | - |
| `sternum.pt` de `c01678b` | p1base_b | 1,00M | 87,7 % | 90,0 % | - | - |
| `irlande.pt` de `c01678b` | prox1 | 1,00M | 86,0 % | 93,3 % | - | - |
| `belgique.pt` de `c01678b` | p1base_a | 1,00M | 85,0 % | 86,7 % | - | - |
| `belgique.pt` | p1base_a | 1,06M | 84,7 % | 88,3 % | - | - |

Sans mention de commit : le fichier du dossier. `l3_hard03` (perdrix, 762k pas) est perdu, mort sans checkpoint.
Les runs l3 partent de poids level 2 et s'entraînent sur levels 2 + 3, sauf `only` (level 3 seul).
Modèles complets (reprise possible) : `<machine>:/Data/arthur.fournier/fpv_runs/approach/l3_<run>/ram_dump/model.zip`.

### Détail par distance, banc level 3

| Distance | 5 m | 10 m | 20 m | 40 m | 80 m | 160 m |
|---|--:|--:|--:|--:|--:|--:|
| `diuris.pt` | 100 % | 100 % | 100 % | 98 % | 100 % | 100 % |
| `saone_1M.pt` | 100 % | 100 % | 100 % | 100 % | 100 % | 92 % |
| `vanilla.pt` | 100 % | 100 % | 100 % | 100 % | 100 % | 92 % |
| `quiscale.pt` | 100 % | 100 % | 100 % | 100 % | 100 % | 90 % |
| `encyclia.pt` | 100 % | 100 % | 100 % | 90 % | 100 % | 100 % |

### Sans le recalage sur la vérité terrain (60 départs level 3 x 2)

| Poids | Avec recalage (x 5) | Sans recalage |
|---|--:|--:|
| `diuris.pt` | 99,7 % | **12 %** |
| `maserati.pt` | 98 % | **7 %** |
| `saone_1M.pt` | 98,7 % | **6 %** |

## Vrille

Banc level 3, 120 vols par poids, rotation réelle lue dans la simu :

| Poids | Vols avec >= 360° de lacet | Lacet médian | Vitesse de lacet moyenne |
|---|--:|--:|--:|
| `diuris.pt` | 91 % | 568° | 133°/s |
| `saone_1M.pt` | 97 % | 545° | 132°/s |
| `vanilla.pt` | 95 % | 598° | 138°/s |
| `maserati.pt` | 89 % | 533° | 140°/s |

Correctif essayé le 22/11 : 42 runs `stable_*` partis de `diuris` (`train/approach/train_stable.py`),
**arrêtés vers 12:13 à 63k-172k pas, donc trop tôt : vrille non corrigée, poids non évalués au banc.**
En entraînement, 89 à 99 % des vols faisaient encore >= 360° de lacet. Ne pas les déployer à la place de `diuris`.
Sauvegardes : `stable/<machine>.pt` ici, et sur chaque machine `/Data/arthur.fournier/fpv_runs/approach/stable_*/approach_*/`
`ckpt/model_interrupted.zip` (reprise avec `train.approach.train_stable --resume`).

## Provenance

| Poids | Run | Checkpoint | md5 |
|---|---|---|---|
| `maserati.pt` | `v3_full_c` | `policy_14000016.pt` | `aca770562c0f8b66b8242da25b681c3d` |
| `policy.pt` | `v3_full_c` | `policy_12000000.pt` (mêmes tenseurs) | `48928dcd54ca382a491c200d01b26697` |

Source : `maserati:/Data/arthur.fournier/fpv_runs/approach/v3_full_c/approach_20261121-191316/`.
Entraînement sur `level_2` (forêt) uniquement, seed 303, `use_local=true`, `hard_frac=0.5`, `proximity=0.5`,
collision et sortie d'arène à -100, `lr=1e-4`.

## Performances

Succès = le drone touche la boîte. Politique déterministe. Mesures du 22/11.

### Level 3 : approche seule, attitude vraie donnée au départ (non déployable tel quel)

`train.approach.eval`, 60 départs fixes x 5 passages = 300 vols, cible en vue au départ.

| Distance de départ | 5 m | 10 m | 20 m | 40 m | 80 m | 160 m | **Total** | Crash |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **`maserati.pt`** | 100 % | 100 % | 92 % | 98 % | 98 % | 98 % | **98 %** | 2 % |
| `policy.pt` | 100 % | 90 % | 92 % | 84 % | 96 % | 88 % | **92 %** | 8 % |

### Level 3 : agent combiné, sans vérité terrain

`maserati.pt` + `weights/explore/femur.pt`, évaluateur officiel, 60 départs x 1 vol, seed 1, plafond 30 s :
**8 % de touches** (32 % de crashs, 47 % de timeouts, 13 % hors arène), alors que la cible est vue dans 100 % des vols.

### Mission complète : chercher puis toucher (cartes classiques)

`handin/combined_agent.py` avec l'explorateur `weights/explore/femur.pt`, départ officiel, timeout 120 s,
34 positions de validation x 3 vols x seeds 1, 2, 3 = 306 vols.

| Poids | city (90) | field (36) | forest (90) | village (90) | **Total** | Crash | Timeout |
|---|--:|--:|--:|--:|--:|--:|--:|
| **`maserati.pt`** | 72 % | 75 % | 98 % | 97 % | **87 %** | 8 % | 5 % |
| `policy.pt` | 68 % | 75 % | 97 % | 87 % | **83 %** | 15 % | 3 % |

Le 13 M du même run, non rangé ici, tombe à 66 % (102 vols, seed 1).

## Problème connu : attitude initiale

- Les départs `level_*` sont inclinés (pitch de 31 à 58° sur les 3 départs tracés).
- `handin/estimator.py` démarre à pitch = 0 et n'intègre ensuite que le gyro : l'erreur de départ reste.
- Le rayon vers la cible est donc faux : le Kalman cible ne s'initialise pas (pas de relais) ou vise à côté.
- `train.approach.eval` ne le voit pas : `ApproachEnv._sync_estimator` recale l'estimateur sur l'attitude vraie.
  **Tous les scores de ce banc sont concernés, quel que soit le poids.**
- Les 4 cartes classiques ne sont pas sur ces départs fixes, d'où les 87 %.

À corriger dans l'initialisation de l'estimateur, puis à revalider avec l'évaluateur officiel.

## Format

- `handin.approach_net.load_approach_actor(path)` : torch seul, acteur seul (pas de `--resume` possible).
- Entrées : `depth` (2, 32, 56), `local` (5, 64, 64), `vec` (41 = 38 de `handin.features` + position attendue de la cible).
- Sortie : 4 sticks (roll, pitch, yaw, throttle), à couper à [-1, 1].
- Les features doivent tourner depuis le début du vol, pas seulement à partir du relais.

## Utilisation

Depuis `search-destroy/` :

```sh
# mission complète (refaire avec --seed 2 et 3 pour le tableau)
uv run python -m train.eval_combined --explorer weights/explore/femur.pt \
    --approach weights/approach/maserati.pt \
    --terrains-file runs/combined/holdout_terrains.txt --runs 3 --seed 1

# level 3 : le config.toml et l'overrides.txt du run doivent être à côté du poids (sinon KeyError 'local')
uv run python -m train.approach.eval --policy <dossier>/policy.pt --bench level_3 --repeats 5
```

## Limites

- **Le 87 % est optimiste** : run, checkpoint et explorateur ont été choisis sur ces mêmes 34 positions,
  qui sont sur les 4 cartes d'entraînement de l'explorateur.
- **Level 3 : 300 vols = 60 départs répétés**, pas 300 situations différentes. Le meilleur est choisi sur ces mêmes
  départs, et les départs aléatoires sont tirés sur la carte d'entraînement : aucune carte level 3 inédite n'a été testée.
- **Checkpoints instables** : ne jamais prendre le dernier sans refaire l'éval.
- **Rien ici n'est soumis** : la soumission est le dossier `handin/` seul, et `handin/agent.py` charge
  `handin/policy.pt`. Pour soumettre le combiné : copier les poids en `handin/explorer.pt` et
  `handin/approach.pt`, puis brancher `handin/agent.py` sur `handin.combined_agent`.
