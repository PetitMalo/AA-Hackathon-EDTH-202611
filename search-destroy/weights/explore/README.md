# Poids des agents d'exploration

Agents PPO « explorateur » : chercher la cible, puis s'en approcher à 10 m (`train/explore_env.py`, `ExploreReachEnv`).
Ils ne finissent pas le vol seuls : `handin/combined_agent.py` passe la main à un agent d'approche dès la détection,
et la reprend si la cible est perdue plus de 4 s.

Chaque fichier porte le **nom de la machine** qui l'a entraîné.

- **Meilleur explorateur : `femur.pt`**
- `frontal.pt` et `indre.pt` sont **deux checkpoints du même entraînement**, pas deux seeds (voir « Limites »).
- Checkpoint retenu : le dernier enregistré le 22/11 à 11h06, sauf `femur.pt`, qui est le 7M, meilleur que le 20M sur validation.

## Performances en validation

Mêmes 102 vols pour tous : 34 positions de boîte de validation x 3 vols, `--seed 1`, départ officiel, timeout 120 s,
politique déterministe, relais dès la détection vers la **même approche** `runs/combined/approach_v3_full_c_12M.pt`.

Succès = le drone touche la boîte cible. Le score est celui du couple explorateur + approche.

| # | Poids | Pas | Succès | IC 95 % | Crash | Timeout | Hors arène | Cible vue | Détection médiane | Écart avec femur (p) |
|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | **`femur.pt`** | 7M | **83 %** | 75-89 | 15 % | 2 % | 0 % | **99 %** | 3,6 s | - |
| 2 | `niva.pt` | 20M | 75 % | 66-83 | 24 % | 1 % | 0 % | 91 % | 3,8 s | 0,17 |
| 3 | `frontal.pt` | 19M | 70 % | 60-78 | 30 % | 0 % | 0 % | 90 % | 2,8 s | 0,03 |
| 4= | `indre.pt` | 20M | 67 % | 57-75 | 31 % | 2 % | 0 % | 86 % | 2,6 s | 0,005 |
| 4= | `metacarpe.pt` | 18M | 67 % | 57-75 | 13 % | 1 % | 20 % | 79 % | 3,2 s | 0,003 |

p : test des signes apparié vol par vol. `femur` et `niva` ne sont pas départageables.
La détection médiane ne porte que sur les vols où la cible a été vue.

### Par scène : succès (cible vue)

| Poids | city (30 vols) | field (12) | forest (30) | village (30) |
|---|--:|--:|--:|--:|
| **`femur.pt`** | **73 % (97 %)** | 75 % (100 %) | 97 % (100 %) | 83 % (100 %) |
| `niva.pt` | 43 % (70 %) | 67 % (100 %) | 100 % (100 %) | 87 % (100 %) |
| `frontal.pt` | 43 % (67 %) | 42 % (100 %) | 90 % (100 %) | 87 % (100 %) |
| `indre.pt` | 13 % (57 %) | 75 % (100 %) | 93 % (100 %) | 90 % (97 %) |
| `metacarpe.pt` | 40 % (63 %) | 83 % (100 %) | 90 % (97 %) | 63 % (70 %) |

### Points forts et faibles

- **`femur.pt`** : le seul fiable en ville. Voit la cible dans 101 vols sur 102, donc ses échecs arrivent presque tous après la détection.
- **`niva.pt`** : 30/30 en forêt, mais ne voit pas la cible dans 30 % des vols en ville.
- **`frontal.pt`** : correct en forêt et au village, faible en ville et en champ.
- **`indre.pt`** : bon au village et en forêt, très mauvais en ville (4/30, 83 % de crash).
- **`metacarpe.pt`** : sort de l'arène dans 37 % des vols en ville et 27 % au village. À éviter.

## À l'entraînement, au même pas

Politique stochastique, succès = arriver à 10 m, moyenne sur ± 0,25M pas. **Ne prédit pas le score de validation.**

| Poids | Succès à 10 m | Crash | Détection |
|---|--:|--:|--:|
| `femur.pt` | 82 % | 16 % | 100 % |
| `niva.pt` | 91 % | 9 % | 100 % |
| `frontal.pt` | 82 % | 18 % | 99 % |
| `indre.pt` | 78 % | 21 % | 99 % |
| `metacarpe.pt` | 78 % | 19 % | 98 % |

Le 20M de femur (non inclus) illustre le piège : 93 % à l'entraînement, mais 77 % en validation (ville : 53 %).

## Réglages

Tous : `train/config_explore_reach.toml`, repris de `runs/femur_explorer.zip` (4M pas).
Base : `collision=-30`, `progress=0.5`, `cov_after_detect=true`, seed 0.

| Poids | Run | Checkpoint | Différence avec la base | SHA-1 |
|---|---|---|---|---|
| `femur.pt` | `reach_nocov_col60` | `policy_7000008.pt` | `cov_after_detect=false`, `collision=-60` | `f007fc956b3f` |
| `niva.pt` | `reach_nocov_prog1` | `policy_20000016.pt` | `cov_after_detect=false`, `progress=1.0` | `d999f91096ef` |
| `frontal.pt` | `reach_nocov_s1` | `policy_19000008.pt` | `cov_after_detect=false` (`run.seed=1` ignoré) | `32d755b7229b` |
| `indre.pt` | `reach_nocov` | `policy_20000016.pt` | `cov_after_detect=false` | `740e5f5c5f6d` |
| `metacarpe.pt` | `explore_reach10` | `policy_18000000.pt` | - | `316c2dd30fc7` |

## Utilisation

Depuis `search-destroy/` :

```sh
# refaire l'évaluation de ce tableau
uv run python -m train.eval_combined --explorer weights/explore/femur.pt \
    --approach runs/combined/approach_v3_full_c_12M.pt \
    --terrains-file runs/combined/holdout_terrains.txt --runs 3 --seed 1
```

`handin/agent.py` n'utilise pas ces poids. `handin/combined_agent.py` les lit via `FPV_EXPLORER`, ou dans
`handin/explorer.pt` par défaut (et l'approche via `FPV_APPROACH` ou `handin/approach.pt`). Sans fichier explorateur,
il fait du hover tant que la cible n'est pas vue.

Ces `.pt` ne contiennent que l'acteur. Pour un `--resume`, il faut le `model_<pas>.zip`, qui est dans
`/tmp/fpv_runs/<run>/<date>/ckpt/` sur la machine du run (perdu si `/tmp` est vidé).

## Limites

- **Cartes déjà vues.** La validation change les positions de boîte, pas les 4 cartes. Cartes nouvelles non mesurées.
- **Biais de sélection.** `femur.pt` et l'approche 12M ont été choisis sur ces mêmes positions : 83 % est optimiste.
- **Un seul tirage de départs** (`--seed 1`), et seulement 12 vols en champ.
- **Seed ignorée avec `--resume`.** `PPO.load` reprend la seed du zip (0). Les checkpoints 5M à 19M de frontal et d'indre
  sont identiques octet pour octet : l'écart entre `frontal.pt` et `indre.pt` vient uniquement de 1M pas de plus.
- **Checkpoints non balayés.** Seul femur a été évalué à plusieurs pas (7M : 83 %, 17M : 75 %, 20M : 77 %).
  Un checkpoint intermédiaire des autres runs peut être meilleur que celui rangé ici.
