# Explorateurs : poids de fin de run (22/11)

**Non évalués.** Les checkpoints choisis et mesurés restent dans `../` (le dernier pas n'est pas forcément le meilleur).

| Poids | Pas |
|---|--:|
| `femur.pt` | 21,568,656 |
| `frontal.pt` | 21,074,640 |
| `indre.pt` | 22,036,608 |
| `niva.pt` | 21,380,304 |
| `metacarpe.pt` | 19,000,008 |

`metacarpe.pt` n'est pas un poids de fin : le run s'est arrêté vers 12:26 sans sauvegarder, c'est son checkpoint de 11:39
(le log TensorBoard va jusqu'à 19802376 pas).
Modèles `.zip` (reprise), hors de /tmp : `venturi:/Data/arthur.fournier/fpv_explore_final/<machine>/`.
