"""Vérifie handin.features : géométrie caméra, estimation de cible, cartes, temps de calcul.

uv run python -m train.check_features --terrain village@v1 --out /tmp/feat
Le pilote de test utilise la vérité terrain (position de la cible) : c'est un banc de test,
pas une policy.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pybullet as p

from fpv_sim.envs.approach_aviary import ApproachAviary
from fpv_sim.scripts.evaluate import random_start
from fpv_sim.utils.terrain import TERRAINS_DIR
from handin import features as F


def true_pose(env):
    s = env._getDroneStateVector(0)
    return s[0:3], np.array(p.getMatrixFromQuaternion(s[3:7])).reshape(3, 3)


def check_geometry(env):
    """Profondeur reconstruite (pixels exacts, sans bruit) vs raycast PyBullet le long des mêmes rayons."""
    pos, R = true_pose(env)
    R_cam = R @ F.CAM_IN_BODY
    eye = pos + R @ F.MOUNT
    buf = env.depth_camera.read_depth_buffer(env._getDroneStateVector(0))
    cam = env.depth_camera
    depth = cam.FAR * cam.NEAR / (cam.FAR - (cam.FAR - cam.NEAR) * buf)
    rng = np.random.default_rng(0)
    rows, cols = rng.integers(0, F.D_H, 400), rng.integers(0, F.D_W, 400)
    d = depth[rows, cols]
    keep = d < 39
    dirs = F._pixel_rays(F.D_W, F.D_H, cols, rows) @ R_cam.T
    pts = eye + dirs * d[:, None]
    ends = eye + dirs * 60
    hits = p.rayTestBatch(eye[None].repeat(400, 0).tolist(), ends.tolist(), physicsClientId=env.CLIENT)
    hit_pts = np.array([h[3] for h in hits])
    hit_ok = np.array([h[0] >= 0 for h in hits])
    err = np.linalg.norm(pts - hit_pts, axis=1)[keep & hit_ok]
    print(f"[géométrie] {len(err)} pixels : erreur point 3D médiane {np.median(err):.3f} m, p95 {np.percentile(err, 95):.3f} m")
    return err


def pilot(est, goal, cruise_z=10.):
    """Pilote de test : cap vers goal, vitesse ~8 m/s, altitude constante."""
    d = goal - est.pos[:2]
    want = np.arctan2(d[1], d[0])
    e = (want - est.yaw + np.pi) % (2 * np.pi) - np.pi
    yaw = np.clip(-1.5 * e, -1, 1)                 # stick yaw + = tourne à droite
    fwd = np.array([np.cos(est.yaw), np.sin(est.yaw)])
    v_fwd = est.vel[:2] @ fwd
    v_goal = min(8., .3 * np.linalg.norm(d))
    pitch = np.clip(.25 + .05 * (v_goal - v_fwd), -.3, .6) if abs(e) < .6 else .05
    thr = np.clip(.6 * (cruise_z - est.pos[2]) - .4 * est.vel[2], -.5, .5) / np.cos(pitch * np.pi / 3)
    return np.clip([0., pitch, yaw, thr], -1, 1)


def save_maps(feat, out, true_box, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["sol_vu", "occ_bas", "occ_moyen", "occ_haut", "cible"]
    loc, glo = feat._local_map(), feat._global_map()
    fig, ax = plt.subplots(2, 5, figsize=(17, 7))
    for k in range(5):
        ax[0, k].imshow(loc[k], vmin=0, vmax=1, cmap="viridis")
        ax[0, k].set_title(f"local {names[k]}")
        ax[0, k].plot(LOC_C, LOC_C, "r^")
    for k, n in enumerate(["sol_vu", "sol_vu_loin", "cible", "trace"]):
        ax[1, k].imshow(glo[k], vmin=0, vmax=1, cmap="viridis")
        ax[1, k].set_title(f"global {n}")
        ax[1, k].plot(GLO_C, GLO_C, "r^")
    # vue monde fine autour de la cible : occupation + sol, avec vraie position
    h = 60
    ci, cj = F.Features._fine_idx(true_box[None])
    sl = (slice(ci[0] - h, ci[0] + h), slice(cj[0] - h, cj[0] + h))
    occ = 1 - np.exp(-feat.fine[1:, sl[0], sl[1]].sum(0) / F.N_SAT)
    ground = 1 - np.exp(-feat.fine[0, sl[0], sl[1]] / F.N_SAT)
    ax[1, 4].imshow(np.stack([occ, ground, np.zeros_like(occ)], -1).transpose(1, 0, 2), origin="lower",
                    extent=[true_box[0] - h, true_box[0] + h, true_box[1] - h, true_box[1] + h])
    ax[1, 4].plot(*true_box, "w*", ms=14, label="vraie cible")
    if feat.target.x is not None:
        ax[1, 4].plot(*feat.target.x, "c+", ms=14, mew=3, label="estimée")
    ax[1, 4].set_title("monde ±60 m (rouge=obstacle, vert=sol vu)")
    ax[1, 4].legend(fontsize=7)
    for a in ax.ravel()[:-1]:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    path = out / f"maps_{tag}.png"
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return path


LOC_C = F.LOCAL_N / 2 - .5
GLO_C = F.GLOBAL_N / 2 - .5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="village@v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--out", type=Path, default=Path("out/check_features"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    name, v = a.terrain.split("@")
    box = np.array(json.loads((TERRAINS_DIR / f"{name}.json").read_text())["targets"][int(v[1:]) - 1]["position_m"][:2])
    env = ApproachAviary(continuous=True, terrain=a.terrain)
    obs, start = random_start(env, np.random.default_rng(a.seed))
    feat = F.Features()
    feat.reset()
    action = None
    times, terr, first_det = [], [], None
    shots = {}
    for k in range(int(a.seconds * 30)):
        tic = time.perf_counter()
        x = feat(obs, action)
        times.append(1e3 * (time.perf_counter() - tic))
        for key, shape in F.SHAPES.items():
            assert x[key].shape == shape and x[key].dtype == np.float32 and np.isfinite(x[key]).all(), key
        if obs["detected"] and first_det is None:
            first_det = k / 30
        if feat.target.x is not None:
            terr.append((k / 30, np.linalg.norm(feat.target.x - box), feat.target.sigma))
        info = env._computeInfo()
        if k == 300 or (info["distance_m"] < 15 and "proche" not in shots):
            shots["proche" if k != 300 else "t10s"] = save_maps(feat, a.out, box, "proche" if k != 300 else "t10s")
        if k == 450:
            check_geometry(env)
        action = pilot(feat.est, box)
        obs, *_ = env.step(action)
        if info["distance_m"] < .3 or env._collision():
            print(f"[vol] fin à t={k/30:.1f}s : {'succès' if env._success() else 'collision/sol'} (dist {info['distance_m']:.1f} m)")
            break
    shots["fin"] = save_maps(feat, a.out, box, "fin")
    env.close()

    t = np.array(times)
    print(f"[temps] features : moyenne {t.mean():.2f} ms, p95 {np.percentile(t, 95):.2f} ms, max {t.max():.2f} ms")
    print(f"[détection] première détection à {first_det} s")
    if terr:
        terr = np.array(terr)
        for tt in (0, 1, 3, 10):
            sel = terr[terr[:, 0] >= terr[0, 0] + tt]
            if len(sel):
                print(f"[cible] {tt:>2} s après la 1re estimation : erreur {sel[0, 1]:.1f} m (sigma annoncé {sel[0, 2]:.1f} m)")
        print(f"[cible] erreur finale {terr[-1, 1]:.2f} m, sigma {terr[-1, 2]:.2f} m")
    print("[images]", *map(str, shots.values()))


if __name__ == "__main__":
    main()
