# Approche : essai anti-vrille (22/11)

42 runs partis de `../diuris.pt`, reward de `train/approach/env_stable.py`, **arrêtés à 63k-172k pas**.

> **À ne pas déployer : vrille non corrigée, aucun de ces poids n'est évalué au banc ni en mission.**
> `<machine>.pt` est ici un autre modèle que `../<machine>.pt`.

Dernière fenêtre d'entraînement (300 vols, politique stochastique, attitude vraie au relais), triée par vrille.

| Poids | Réglage | Pas | Succès | Vols avec >= 360° | Overrides |
|---|---|--:|--:|--:|---|
| `peugeot.pt` | s1_lr3e-4 | 147,720 | 89.7 % | 88.7 % | `reward.spin=0.1 ppo.lr=3e-4` |
| `tarse.pt` | s05_lr3e-4 | 150,864 | 91.3 % | 89.0 % | `reward.spin=0.05 ppo.lr=3e-4` |
| `serapias.pt` | s1_lr3e-4 | 81,264 | 96.0 % | 90.0 % | `reward.spin=0.1 ppo.lr=3e-4` |
| `saone.pt` | s1 | 144,696 | 96.0 % | 90.3 % | `reward.spin=0.1` |
| `xylobium.pt` | s15_free45 | 81,552 | 97.0 % | 92.3 % | `reward.spin=0.15 reward.spin_free_dps=45 reward.spin_max=100` |
| `groenland.pt` | s1_explore | 134,180 | 95.7 % | 92.7 % | `reward.spin=0.1 ppo.init_log_std=-0.7` |
| `portugal.pt` | s1_lr3e-4 | 135,020 | 91.3 % | 92.7 % | `reward.spin=0.1 ppo.lr=3e-4` |
| `urabu.pt` | s1_free60 | 65,388 | 98.0 % | 93.3 % | `reward.spin=0.1 reward.spin_free_dps=60` |
| `autriche.pt` | s03 | 131,640 | 97.0 % | 93.3 % | `reward.spin=0.03` |
| `epipactis.pt` | s15_free45 | 82,032 | 96.0 % | 93.7 % | `reward.spin=0.15 reward.spin_free_dps=45 reward.spin_max=100` |
| `pologne.pt` | s05_l3 | 103,060 | 96.0 % | 93.7 % | `reward.spin=0.05 terrains.scenes=["level_3"]` |
| `zeuxine.pt` | s03 | 73,716 | 97.7 % | 94.3 % | `reward.spin=0.03` |
| `dordogne.pt` | s05_free15 | 106,152 | 98.7 % | 94.7 % | `reward.spin=0.05 reward.spin_free_dps=15` |
| `hongrie.pt` | s05_smooth05 | 131,080 | 98.0 % | 94.7 % | `reward.spin=0.05 reward.smooth=0.05` |
| `belgique.pt` | s1_free60 | 130,060 | 97.0 % | 94.7 % | `reward.spin=0.1 reward.spin_free_dps=60` |
| `orchis.pt` | s1 | 80,376 | 96.7 % | 94.7 % | `reward.spin=0.1` |
| `telipogon.pt` | s05_free15 | 81,024 | 96.7 % | 94.7 % | `reward.spin=0.05 reward.spin_free_dps=15` |
| `roumanie.pt` | s15_free45 | 132,340 | 96.0 % | 94.7 % | `reward.spin=0.15 reward.spin_free_dps=45 reward.spin_max=100` |
| `tibia.pt` | s05_explore | 152,376 | 95.0 % | 94.7 % | `reward.spin=0.05 ppo.init_log_std=-0.7` |
| `suede.pt` | s05_lr3e-4 | 121,920 | 94.7 % | 94.7 % | `reward.spin=0.05 ppo.lr=3e-4` |
| `finlande.pt` | s05 | 131,200 | 97.7 % | 95.0 % | `reward.spin=0.05` |
| `sternum.pt` | s1_l3 | 117,288 | 97.7 % | 95.0 % | `reward.spin=0.1 terrains.scenes=["level_3"]` |
| `vanda.pt` | s05_smooth05 | 82,884 | 97.0 % | 95.0 % | `reward.spin=0.05 reward.smooth=0.05` |
| `bentley.pt` | s05_l3 | 115,896 | 97.3 % | 95.3 % | `reward.spin=0.05 terrains.scenes=["level_3"]` |
| `sole.pt` | s05_lr3e-4 | 83,040 | 98.0 % | 95.7 % | `reward.spin=0.05 ppo.lr=3e-4` |
| `irlande.pt` | s05_center02 | 131,320 | 97.3 % | 95.7 % | `reward.spin=0.05 reward.center=0.02` |
| `pleione.pt` | s1_l3 | 66,720 | 97.3 % | 95.7 % | `reward.spin=0.1 terrains.scenes=["level_3"]` |
| `islande.pt` | s1 | 131,380 | 97.0 % | 95.7 % | `reward.spin=0.1` |
| `malte.pt` | s1_l3 | 104,820 | 97.3 % | 96.0 % | `reward.spin=0.1 terrains.scenes=["level_3"]` |
| `oncidium.pt` | s05 | 80,280 | 97.0 % | 96.0 % | `reward.spin=0.05` |
| `calanthe.pt` | s1_free60 | 71,940 | 96.0 % | 96.0 % | `reward.spin=0.1 reward.spin_free_dps=60` |
| `encyclia.pt` | s05_center02 | 81,864 | 97.3 % | 96.3 % | `reward.spin=0.05 reward.center=0.02` |
| `diuris.pt` | s05_smooth05 | 82,068 | 99.0 % | 97.0 % | `reward.spin=0.05 reward.smooth=0.05` |
| `aerides.pt` | s05_free15 | 73,716 | 98.3 % | 97.0 % | `reward.spin=0.05 reward.spin_free_dps=15` |
| `maserati.pt` | s05 | 172,056 | 97.3 % | 97.0 % | `reward.spin=0.05` |
| `quiscale.pt` | s05_l3 | 63,180 | 97.3 % | 97.0 % | `reward.spin=0.05 terrains.scenes=["level_3"]` |
| `perdrix.pt` | s05_explore | 67,356 | 95.0 % | 97.3 % | `reward.spin=0.05 ppo.init_log_std=-0.7` |
| `lituanie.pt` | s05_explore | 122,320 | 97.3 % | 97.7 % | `reward.spin=0.05 ppo.init_log_std=-0.7` |
| `vanilla.pt` | s05_center02 | 82,716 | 98.7 % | 98.0 % | `reward.spin=0.05 reward.center=0.02` |
| `neotinea.pt` | s03 | 82,884 | 96.0 % | 98.0 % | `reward.spin=0.03` |
| `hollande.pt` | s1_explore | 137,280 | 93.0 % | 98.0 % | `reward.spin=0.1 ppo.init_log_std=-0.7` |
| `ophrys.pt` | s1_explore | 83,208 | 97.0 % | 98.7 % | `reward.spin=0.1 ppo.init_log_std=-0.7` |

Modèles complets (reprise) : `<machine>:/Data/arthur.fournier/fpv_runs/approach/stable_<réglage>_<machine>/approach_*/ckpt/model_interrupted.zip`.
