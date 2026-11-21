# FrontierExplorer : méthode d'exploration

L'agent soumis ([`handin/agent.py`](../agent.py)) délègue chaque pas de simulation à `FrontierExplorer.step` ([`explorer.py`](explorer.py)). L'exploration est **entièrement algorithmique** : aucun réseau entraîné n'est utilisé. Elle n'utilise que les capteurs de l'agent (IMU, VIO bruité, profondeur et détecteur). La vérité terrain ne sert jamais.

En résumé :
1. Le drone monte, puis pique vers le centre en regardant le sol.
2. Il construit une carte 3D de voxels à partir de l'image de profondeur.
3. Il choisit la prochaine frontière, c'est-à-dire la limite entre le connu et l'inconnu, selon ce qu'il y verrait et la longueur du chemin pour y aller.
4. Il s'y rend dans un couloir de boîtes libres, piloté par un MPC.
5. Une détection de la cible met fin à l'exploration.

## La boucle à 30 Hz

```
observation (IMU, VIO, profondeur 224x126, détection 160x90)
  │
  ├─ 1. estimation d'état ............ state_estimation.py
  ├─ 2. mise à jour de la carte voxel  voxel_map.py
  ├─ 3. suivi de la cible ............ target.py (TargetTracker)
  │
  ├─ cible active ? ──► poursuite de la cible (target.py, hors de ce document)
  ├─ début du vol ? ──► montée + piqué (start.py)
  ├─ bloqué ? ───────► échappée graduée (progress.py)
  └─ sinon ──────────► 4. choix de la frontière (frontiers.py + geodesic.py)
                        5. couloir + MPC (corridor.py, mpc.py)
                        6. accélération → manches (controller.py)
```

| Fichier | Rôle |
|---|---|
| `explorer.py` | Boucle principale, choix du mode, journalisation (`StepInfo`) |
| `config.py` | Tous les réglages (dataclasses) |
| `state_estimation.py` | Attitude, position et vitesse à partir de l'IMU et du VIO |
| `camera_model.py` | Pose et rayons de la caméra (copie du modèle du simulateur) |
| `voxel_map.py` | Grille 3D de log-odds et frontières |
| `frontiers.py` | Score des frontières, engagement sur un but, liste noire |
| `geodesic.py` | Distance et chemin du plus court trajet volable |
| `corridor.py`, `mpc.py` | Couloir de boîtes libres et MPC (QP résolu par ADMM) |
| `controller.py` | Accélération voulue → roll, pitch, yaw, throttle |
| `progress.py` | Détection de blocage et échappée |
| `start.py` | Manœuvre de départ |

## 1. Départ : montée, puis piqué vers le centre (`start.py`)

- **`rise`** : le drone freine sa vitesse de départ, tient sa position horizontale, monte de **+50 m** au-dessus de son altitude de départ (soit 55 à 75 m) et tourne le nez vers le centre de l'arène.
- **`dive`** : il pique **nez bas à 45°** vers le centre, en descendant à 6 m/s au plus, jusqu'à **25 m** d'altitude (au-dessus des toits, qui culminent vers 20 m).
  - La caméra est inclinée de 25° vers le haut sur le drone. Ce piqué la fait regarder environ 20° sous l'horizon, donc le sol devant le drone est dans le champ du détecteur, qui porte à 250 m.
  - En mode angle, garder le nez bas fait accélérer : le drone ne peut pas regarder le sol en restant sur place. D'où un piqué plutôt qu'un regard sur place.
- **Fin de la manœuvre** : une détection de la cible l'interrompt à tout moment. Sinon, la planification démarre ensuite avec un plan et un couloir neufs.

Pourquoi ne pas planifier depuis 50 m : la profondeur ne porte qu'à 40 m, et la carte voxel s'arrête à 40 m. Depuis là-haut, la carte ne verrait pas le sol.

## 2. Estimation d'état (`state_estimation.py`)

- **Attitude** : le gyroscope est intégré, et l'inclinaison (roll et pitch) est recalée sur l'accéléromètre par une correction de type Mahony.
- **Cap** : il n'a pas de référence absolue. Au départ, on suppose le nez tourné vers le centre (le départ est à moins de 20° de ce cap), puis on le recale sur la direction de la vitesse mesurée pendant les premières images.
- **Position et vitesse** : un filtre alpha-bêta sur le VIO, avec une prédiction par l'accéléromètre.
- **Démarrage** : la carte n'est pas alimentée avant la fin de l'alignement du cap.

## 3. Carte voxel (`voxel_map.py`)

- **Grille** : 600 × 600 × 40 m, en voxels de 2 m, avec un log-odds entier par voxel (+3 pour un impact, −1 pour un passage, borné à ±20). Un voxel est soit inconnu, soit libre, soit occupé.
- **Mise à jour** : chaque image de profondeur, échantillonnée à 1 pixel sur 4, est projetée depuis la pose estimée de la caméra. Les voxels traversés deviennent libres et le voxel d'impact devient occupé. Un pixel sans retour (40 m) donne de l'espace libre jusqu'à 38 m.
- **Frontière** : un voxel **libre** qui touche au moins un voxel **inconnu**. Les frontières sont mises à jour seulement dans la zone touchée par l'image.
- **Plafond « ciel » à 24 m** : au-dessus, l'inconnu compte comme déjà vu. Il ne crée ni frontière ni gain d'information.
  - Pourquoi : la caméra, tournée vers le haut, découvre sans cesse du « ciel » nouveau autour du drone. Cela créait des frontières aériennes qui faisaient faire des allers-retours au drone, alors que la cible est toujours au sol.
  - Ce plafond ne vaut que pour l'exploration : pour le couloir de vol, l'inconnu reste interdit.

## 4. Choix de la frontière (`frontiers.py`)

**Candidats**
- Les frontières situées entre **3 et 20 m** d'altitude, à moins de 240 m du centre, et à plus de 8 m du drone.
- Elles sont regroupées dans des cases de 10 m (au moins 5 points par groupe). Le candidat d'un groupe est le point le plus proche de son centre.
- Au plus 40 candidats sont évalués, présélectionnés selon taille × e^(−λd).

**Score de `GainCostFrontierPolicy`**

```
score = gain × exp(−λ · d_géodésique) × (1 − w_cap · (1 − cos α) / 2)
gain  = Σ sur les voxels INCONNUS visibles depuis le candidat de  prior(x, y) × poids_altitude(z)
```

| Terme | Ce qu'il représente |
|---|---|
| `gain` | 128 rayons sur une sphère (répartition de Fibonacci), de 15 m de long, arrêtés par les obstacles. Chaque voxel inconnu vu compte une fois. |
| `prior(x, y)` | Gaussienne de 100 m de sigma centrée sur l'arène, là où se trouvent les boîtes. Comme on ne la somme que sur l'inconnu, une zone déjà vue sans cible cesse de compter. |
| `poids_altitude(z)` | 1 sous 4 m d'altitude (là où peut être la boîte), puis décroissance linéaire jusqu'à 0 au plafond de 24 m : on favorise les frontières d'où l'on découvre du sol. |
| `d_géodésique` | Longueur du chemin réel pour y aller (voir section 5). λ = 0,08 m⁻¹. |
| terme de cap | Pénalise les demi-tours (`w_cap` = 0,8). α est l'angle entre la direction des 12 premiers mètres du chemin et la vitesse du drone. |

**Engagement sur un but**
- **Hystérésis** : on ne change de but que si le nouveau est 50 % meilleur.
- **Réévaluation** : le score du but est recalculé toutes les 1,5 s, et on vérifie toutes les 0,5 s qu'il est encore une frontière.
- **Continuité** : quand le but vient d'être exploré, les candidats proches de lui reçoivent jusqu'à +50 %, pour continuer dans la même zone.
- **Abandon** : un but non atteint en 8 s est abandonné, et toute frontière à moins de 10 m de lui est mise sur liste noire pendant 30 s.
- **Compteurs** : le nombre de changements de but par minute et le nombre de demi-tours (plus de 120°) sont journalisés.

## 5. Distance géodésique (`geodesic.py`)

En ville, une frontière juste derrière un immeuble paraissait proche en ligne droite. On mesure donc la longueur du **plus court chemin volable** :

- **Grille** : une fenêtre de ±100 m autour du drone, de 0 à 32 m d'altitude, en cellules de **4 m**. Une cellule n'est praticable que si ses 8 voxels sont **libres connus**, la même règle que le couloir : tout chemin trouvé est donc volable.
- **Calcul** : Dijkstra (`scipy.sparse.csgraph`) avec 26 voisins, sans couper les coins. Le coût d'un pas vaut longueur × (1 + 1 / distance à l'obstacle le plus proche), pour garder les chemins au milieu des rues. Le sol n'est pas compté comme obstacle.
- **Rejoindre une frontière** : sa cellule contient de l'inconnu, donc on passe par la meilleure cellule voisine praticable.
- **Chemin final** : il est raccourci en ligne droite partout où un tube de 1 m autour du segment est libre.
- **Sert à deux choses** : au score (section 4) et au tracé du couloir (section 6). Les frontières inatteignables sont écartées. S'il n'y en a aucune d'atteignable, on revient à la distance euclidienne.
- **Coût** : environ 6 ms par calcul sur une carte synthétique pleine taille. Le résultat est réutilisé pendant 8 images tant que le drone a bougé de moins de 4 m.

## 6. Guidage : couloir, MPC, puis manches

**Couloir** (`corridor.py`)
- Une chaîne d'au plus 6 boîtes alignées sur les axes, d'au plus 30 m de côté, qui ne contiennent que des voxels **libres connus**. Elle est construite le long du **chemin géodésique**, sur au plus 60 m.
- Il est replanifié toutes les 0,5 s, et aussi si le but bouge, si le drone sort des boîtes, ou si un voxel du couloir devient occupé.
- Si le chemin rencontre un obstacle, le couloir s'arrête devant et le drone freine dans l'espace libre.

**MPC** (`mpc.py`)
- Horizon de 20 pas de 0,1 s, sur un modèle de double intégrateur. On pénalise l'écart à la référence, l'effort d'accélération et ses variations.
- Un coût impose une vitesse nulle en fin d'horizon : le plan doit toujours pouvoir s'arrêter dans le couloir.
- Les boîtes du couloir, réduites de 0,8 m, sont des contraintes souples. Le QP est résolu par ADMM, avec la matrice factorisée une seule fois.

**Vitesse et cap**
- La vitesse de croisière est de **12 m/s**. Plus de vitesse veut dire plus de piqué, donc une caméra qui regarde plus bas.
- Le nez vise un point situé environ **1 s plus loin sur le couloir**, et non le but final, pour que la caméra regarde là où le drone va passer.
- La vitesse est multipliée par cos²(erreur de cap) tant que le nez n'est pas aligné.

**Manches** (`controller.py`)
- L'accélération voulue, plus une compensation de la traînée (modèle quadratique du simulateur), est convertie en roll, pitch et throttle (décomposition ZYX exacte).
- L'inclinaison **totale** est limitée à 50° en réduisant l'accélération horizontale dans sa propre direction. Saturer roll et pitch séparément faussait la direction en virage freiné.

## 7. Anti-blocage (`progress.py`)

**Détection**, à partir de l'état estimé :
- déplacement horizontal net de moins de 8 m sur 8 s ;
- **ou** couloir bloqué et vitesse de moins de 1 m/s pendant 1,5 s ;
- **ou** au moins 6 obstacles dans un rayon de 10 m avec moins de 15 m de déplacement sur 8 s.

**Échappée graduée** : chaque déclenchement passe au niveau suivant, en boucle. Après 30 s sans déclenchement, on repart du niveau 1. Un délai minimal de 4 s sépare deux niveaux.
1. **replan** : liste noire de 20 m pendant 45 s autour du but et de l'endroit où le drone piétine, puis replanification immédiate.
2. **scan** : le drone tient sa position et fait un tour complet en yaw (environ 2,5 s) pour découvrir l'inconnu autour de lui.
3. **climb** : il monte à 26 m (au-dessus des toits) si la colonne au-dessus de lui ne contient pas de voxel occupé. Les frontières sont ensuite cherchées entre 3 et 24 m pendant 20 s.

## 8. Passage à la phase cible

- **Bascule** : dès que la cible est détectée, `TargetTracker` estime sa position (par la profondeur à moins de 38 m, sinon par l'intersection du rayon avec le sol) et la phase cible prend le relais.
- **Voxels de la boîte** : le suivi relève aussi les voxels qui appartiennent à la boîte. Ce sont les seuls que le couloir considère comme libres près de la cible ; tout autre obstacle reste un obstacle.
- **Retour à l'exploration** : il a lieu si la cible n'est plus vue depuis 6 s, ou depuis 25 s si elle est bien localisée. Il a aussi lieu si le drone arrive à moins de 8 m de l'estimation sans la voir pendant 3 s.

## Résultats sur les terrains publics

Évaluation rapide (`fpv_sim.scripts.evaluate --runs 1`) : **1 vol par position de boîte**, soit 138 vols, avec 120 s de vol et la graine 0. L'évaluation officielle en fait 10 par boîte, sur des terrains cachés. Ces chiffres sont donc indicatifs.

| Terrain | Boîte détectée | 1re détection, médiane (avant → maintenant) | Temps d'exploration, moyenne (avant → maintenant) | Succès |
|---|---|---|---|---|
| city | 92 % | 30,2 s → **15,1 s** | 48,5 s → **29,0 s** | 40 % |
| field | 100 % | 9,1 s → 11,4 s | 14,6 s → **9,2 s** | 61 % |
| forest | 100 % | 0,7 s → 2,5 s | 6,2 s → 7,3 s | 72 % |
| village | 98 % | 13,7 s → **7,9 s** | 22,7 s → **12,0 s** | 57 % |

- **Colonne « avant »** : run du même protocole réalisé avant la manœuvre de départ (`out/eval-quick-20261121-153815`).
- **Colonne « maintenant »** : `out/eval-quick-20261121-192054`.
- **Ce que la manœuvre de départ change** : la cible est trouvée deux fois plus vite en ville et au village. Sur `field` et en forêt, où la cible est vite visible de toute façon, elle coûte quelques secondes.
- **Latence** : environ 12 ms par pas, pour un budget de 33 ms.
- **Origine des échecs** : la plupart viennent maintenant de la **phase cible** (crashs à basse vitesse près de la boîte, ou drone bloqué à 8–33 m d'elle), plus de l'exploration. Seuls 4 vols sur 138 n'ont jamais vu la boîte (3 en ville, 1 au village).

## Limites connues

- **Couche du sol** : dans la carte, la couche 0–2 m est occupée partout dès que le sol a été vu. Les obstacles bas (voitures, buissons) s'y confondent avec le sol, et le couloir ne peut pas voler sous 2 m.
- **Caméra inclinée vers le haut** : à 8 m/s (environ 6° de piqué), le bas de l'image n'est qu'à environ 16° sous l'horizon. À 10 m d'altitude, le sol n'apparaît alors qu'au-delà d'environ 35 m, à la limite de la portée de la profondeur. À 12 m/s, il apparaît dès environ 24 m.
- **Géodésique** : la fenêtre de ±100 m et les cellules de 4 m peuvent faire paraître fermées des rues de moins de 8 m. Au-delà de la fenêtre, on revient à la distance euclidienne.
- **Manœuvre de départ** : le freinage initial se fait à basse altitude, sans couloir, et la fin du piqué arrive vers 20 m/s, freinée par le MPC.
- **Tests** : les tests unitaires des derniers modules (`start.py`, poursuite de la cible, zone dégagée réduite aux voxels de la boîte) sont écrits, mais n'ont pas tous été relancés.

## Réglages principaux (`config.py`)

| Réglage | Valeur | Effet |
|---|---|---|
| `StartConfig.rise_m`, `dive_pitch_deg`, `dive_end_alt_m` | 50 m, 45°, 25 m | Manœuvre de départ |
| `MapConfig.voxel_m`, `sky_ceiling_m` | 2 m, 24 m | Résolution, plafond du ciel |
| `FrontierConfig.z_band_m` | (3, 20) m | Altitude des frontières candidates |
| `FrontierConfig.lam`, `heading_w`, `hysteresis` | 0,08, 0,8, 0,5 | Coût de la distance, pénalité de demi-tour, stabilité du but |
| `FrontierConfig.ground_full_m`, `prior_sigma_m` | 4 m, 100 m | Gain tourné vers le sol, prior centré |
| `GeodesicConfig.cell_m`, `half_xy_m`, `clearance_weight` | 4 m, 100 m, 1 | Grille du plus court chemin |
| `ControlConfig.cruise_speed_m_s` | 12 m/s | Vitesse d'exploration |
| `CorridorConfig.margin_m`, `max_length_m` | 0,8 m, 60 m | Marge du couloir, portée |
| `EscapeConfig.window_s`, `min_disp_m`, `climb_alt_m` | 8 s, 8 m, 26 m | Anti-blocage |

Chaque module se désactive séparément, pour comparer avec et sans : `start.enabled`, `geodesic.enabled`, `escape.enabled`.

## Lancer

Depuis `search-destroy/` :

```sh
uv run python -m handin.check_handin                                      # l'agent respecte l'API
uv run python -m handin.view_explorer --terrain city@v12 --seed 0         # vue FPV + carte, en direct
uv run python -m handin.run_explorer --terrain village@v1                 # journal texte + carte PNG
uv run python -m fpv_sim.scripts.evaluate --runs 1                        # évaluation rapide (~20 min)
```
