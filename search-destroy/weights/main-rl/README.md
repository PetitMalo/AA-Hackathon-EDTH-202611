# Poids de l'agent RL principal

Agent PPO de bout en bout (recherche + approche) : `handin/agent.py` = `handin/features.py` + `handin/policy_net.py`.
Chaque fichier porte le **nom de la machine** qui l'a entraîné et contient le meilleur checkpoint évalué de cette machine.

- **Meilleur agent : `venturi.pt`**
- **Agent de `handin/policy.pt` : `rotule.pt`** (branche `arthur-work`)

Succès = le drone touche la boîte cible, sans collision avec le sol ou un autre objet au même moment.

## Classement

Mêmes 68 vols pour tous : 34 positions de boîte de validation x 2 vols, départ officiel (225-275 m du centre),
timeout 120 s, politique déterministe, `--seed 0`.

| # | Poids | Pas | Succès | IC 95 % | Crash | city | field | forest | village |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | **`venturi.pt`** | 20,5M | **88 %** | 78-94 | 10 % | 80 % | 100 % | 85 % | 95 % |
| 2 | `sacrum.pt` | 20,5M | 82 % | 72-90 | 18 % | 65 % | 100 % | 90 % | 85 % |
| 3= | `ardennes.pt` | 20,5M | 79 % | 68-87 | 21 % | 50 % | 75 % | 95 % | 95 % |
| 3= | `manche.pt` | 20,5M | 79 % | 68-87 | 19 % | 65 % | 100 % | 80 % | 85 % |
| 3= | `rover.pt` | 20,5M | 79 % | 68-87 | 19 % | 60 % | 100 % | 85 % | 85 % |
| 6= | `rotule.pt` | 20M | 78 % | 67-86 | 19 % | 50 % | 88 % | 85 % | 95 % |
| 6= | `royce.pt` | 20,5M | 78 % | 67-86 | 22 % | 60 % | 100 % | 75 % | 90 % |
| 8 | `marne.pt` | 20,5M | 75 % | 64-84 | 25 % | 60 % | 88 % | 90 % | 70 % |
| 9 | `moselle.pt` | 20,5M | 72 % | 60-81 | 25 % | 55 % | 75 % | 85 % | 75 % |
| 10 | `loire.pt` | 20M | LOIRE_S0 |
| 11 | `essonne.pt` | 10M | 59 % | 47-70 | 34 % | 45 % | 75 % | 65 % | 60 % |

Sur 68 vols, les rangs 2 à 9 ne sont pas départageables.

## Confirmation des finalistes

**Validation, autre tirage** (34 positions x 3 vols, `--seed 1`) :

| Poids | Succès | Crash | city | field | forest | village |
|---|--:|--:|--:|--:|--:|--:|
| **`venturi.pt`** | **89 %** | 8 % | 87 % | 83 % | 97 % | 87 % |
| `rotule.pt` | 80 % | 18 % | 50 % | 100 % | 90 % | 93 % |
| `loire.pt` | 71 % | 28 % | 43 % | 75 % | 93 % | 73 % |

**Évaluation complète** (`fpv_sim.scripts.evaluate --runs 3` : 138 positions, 414 vols, entraînement + validation) :

| Poids | Succès | city | field | forest | village |
|---|--:|--:|--:|--:|--:|
| **`venturi.pt`** | VENTURI_FULL |
| `rotule.pt` | 72 % | 48 % | 83 % | 88 % | 75 % |

## Réglages de chaque poids

Lignée : `zéro -> 2,65M -> 6,55M -> 7,43M (étape 3) -> rover 10M -> rotule 20M -> affinages 20,5M`.
`essonne.pt` sort de la branche 7,43M -> 10M, `loire.pt` de rover 10M -> 20M.

Base de `rotule` : `progress=2`, `progress_unseen_factor=1`, `coverage=0.5` dès l'étape 0, `smooth=0`, `timeout=-30`,
`collision=-50`, `replay_previous=0.5`, pénalité de proximité des obstacles `clearance=0.1` sous `clearance_m=1.5`.

| Poids | Run | Parti de | Learning rate | Différence avec la base | SHA-1 |
|---|---|---|--:|---|---|
| `venturi.pt` | `ft_lr1e4` | rotule 20M | 1e-4 | - | `686839cf78dc` |
| `sacrum.pt` | `ft_lr5e5` | rotule 20M | 5e-5 | - | `26f1a054baa9` |
| `ardennes.pt` | `ft_lr1e4_t21` | rotule 20M | 1e-4 | `time=0.0021` | `936d04f2956f` |
| `manche.pt` | `ft_lr1e4_col70` | rotule 20M | 1e-4 | `collision=-70` | `f6391f48b6e5` |
| `rover.pt` | `ft_lr1e4_r20` | rotule 20M | 1e-4 | `clearance_m=2.0` | `f707efe27d52` |
| `rotule.pt` | `clr_k01_r15_unseen1` | rover 10M | 3e-4 | - | `96d26bda3edf` |
| `royce.pt` | `ft_lr1e4_rp06` | rotule 20M | 1e-4 | `replay_previous=0.6` | `5b92cf2d2ae9` |
| `marne.pt` | `ft_lr3e4_t19` | rotule 20M | 3e-4 | `time=0.0019` | `b5154cbbfa5c` |
| `moselle.pt` | `ft_lr1e4_clr015` | rotule 20M | 1e-4 | `clearance=0.15` | `02bfd1895ff3` |
| `loire.pt` | `clr_k01_r20_t21` | rover 10M | 3e-4 | `clearance_m=2.0`, `time=0.0021`, `progress_unseen_factor=0.3` | `8d91a91d62b4` |
| `essonne.pt` | `essonne_v1_base` | 7,43M | 3e-4 | sans proximité, `collision=-30`, `replay_previous=0.2`, `progress_unseen_factor=0.3` | `61d4bfc85d78` |

## Utilisation

Depuis `search-destroy/` :

```sh
# tester un poids sans toucher au rendu
FPV_WEIGHTS=$PWD/weights/main-rl/venturi.pt uv run python -m fpv_sim.scripts.evaluate --runs 3

# en faire l'agent de handin/
command cp -f weights/main-rl/venturi.pt handin/policy.pt
uv run fpv-check
```

`fpv-check` affiche « OK » même sans poids (l'agent fait alors du hover) : vérifier qu'il n'y a pas de ligne
`[agent] ... absent : hover` dans sa sortie.

Ces `.pt` ne contiennent que l'acteur. Ils ne permettent pas un `--resume` : il faut pour cela le `model_<pas>.zip` du run.
Ils supposent le `handin/features.py` actuel : le modifier peut invalider tous les poids.

## Limites

- **Cartes déjà vues.** La validation change les positions de boîte, pas les 4 cartes. Le réseau reçoit sa position et
  son cap absolus, il peut avoir mémorisé les cartes : attendre moins sur des cartes nouvelles (non mesuré).
- **Biais de sélection.** Ces checkpoints ont été choisis parmi une trentaine d'évaluations sur ces mêmes positions de
  validation. Le classement à 68 vols est optimiste, d'où les confirmations sur d'autres tirages.
- **Checkpoints instables.** Un même run varie de 15 à 20 points en 1M pas (rotule : 78 % à 20M, 62 % à 21M, mêmes vols).
  Toujours évaluer un checkpoint, le succès à l'entraînement ne le prédit pas.
- **Causes non isolées.** Les meilleurs runs cumulent plusieurs changements (proximité, `unseen`, learning rate) :
  ces chiffres ne prouvent pas l'effet de chacun.
- **Cartes `level_*` non évaluées**, et non utilisées à l'entraînement de ces poids.
