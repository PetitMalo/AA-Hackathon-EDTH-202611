"""Vérifie train.env_wrapper.FPVTrainEnv : départs, curriculum, terrain chargé, reward, fins d'épisode.

uv run python -m train.check_env            # 4 scènes x 4 étapes, ~2 min
"""

import argparse
import copy
import json
import time

import numpy as np
import pybullet as p

from fpv_sim.utils.terrain import TERRAINS_DIR
from train import config as cfgmod
from train.env_wrapper import ARENA, CLEAR_M, FINAL_STAGE, FPVTrainEnv, _wrap

OUTCOMES = {"success", "collision", "out_of_arena", "timeout"}


def box_json(terrain):
    name, v = terrain.split("@v")
    return np.array(json.loads((TERRAINS_DIR / f"{name}.json").read_text())["targets"][int(v) - 1]["position_m"])


def check_reset(env, stage, terrain):
    env.terrains = [terrain]
    assert env.set_stage(stage) == stage and env.stage == stage
    tic = time.perf_counter()
    x, info = env.reset()
    dt = time.perf_counter() - tic
    assert env.observation_space.contains(x), "observation hors observation_space"
    assert info["stage"] == stage == env.stage_used and info["terrain"] == terrain == env.sim.TERRAIN
    sim = env.sim
    # terrain effectivement chargé : boîte = JSON
    pos, _ = p.getBasePositionAndOrientation(sim.TERRAIN_IDS[0], physicsClientId=sim.CLIENT)
    assert np.allclose(pos[:2], box_json(terrain)[:2], atol=1e-3), (pos, box_json(terrain))
    # départ dégagé
    drone = sim.DRONE_IDS[0]
    clear = not any(p.getClosestPoints(drone, b, CLEAR_M, physicsClientId=sim.CLIENT) for b in sim.TERRAIN_IDS)
    xyz = sim.pos[0]
    fallback = np.allclose(xyz, [0, 0, 3], atol=1e-6)
    assert clear or fallback, "départ non dégagé"
    # distance et cap
    yaw = p.getEulerFromQuaternion(sim.quat[0])[2]
    d_head = abs(_wrap(yaw - np.arctan2(-xyz[1], -xyz[0])))
    r_c = np.linalg.norm(xyz[:2])
    r_t = np.linalg.norm(xyz[:2] - env.target_xy)
    if not fallback:
        assert d_head <= np.radians(20) + 1e-6, f"cap à {np.degrees(d_head):.1f}° du centre"
        if stage >= FINAL_STAGE:
            assert 225 <= r_c <= 275 and 5 <= xyz[2] <= 25, (r_c, xyz)
        else:
            lo, hi, cone, _ = env.cur["stages"][stage]
            assert lo - 1e-6 <= r_t <= hi + 1e-6 and 5 <= xyz[2] <= 20, (r_t, xyz)
            assert np.abs(xyz[:2]).max() <= ARENA - 20
    return dict(dt=dt, n_ids=len(sim.TERRAIN_IDS), r_t=r_t, d_head=np.degrees(d_head), fallback=fallback)


def run_episode(env, rng, n=200):
    """~n steps aléatoires puis nuls ; renvoie (somme reward, issue, nb steps)."""
    total, outcome = 0., None
    for k in range(n):
        a = rng.uniform(-1, 1, 4) if k < n // 2 else np.zeros(4)
        x, r, term, trunc, info = env.step(a)
        assert np.isfinite(r) and env.observation_space.contains(x)
        total += r
        if term or trunc:
            outcome = info["outcome"]
            assert outcome in OUTCOMES and info["ep_len"] == env.t
            assert trunc == (outcome == "timeout") and term == (outcome != "timeout")
            break
    return total, outcome, k + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = cfgmod.load()
    train, holdout = cfgmod.terrain_split(cfg)
    assert not set(train) & set(holdout), "train et holdout se recouvrent"
    print(f"[split] {len(train)} terrains d'entraînement, {len(holdout)} holdout, disjoints")

    cfg = copy.deepcopy(cfg)
    cfg["curriculum"]["replay_previous"] = 0.        # étape utilisée = étape demandée
    rng = np.random.default_rng(a.seed)
    env = FPVTrainEnv(cfg, train, stage=0, seed=a.seed)
    outcomes = {}
    for scene in cfg["terrains"]["scenes"]:
        ts = [t for t in train if t.startswith(scene + "@")]
        for stage in range(FINAL_STAGE + 1):
            terrain = ts[int(rng.integers(len(ts)))]
            r = check_reset(env, stage, terrain)
            tot, out, k = run_episode(env, rng)
            outcomes[out] = outcomes.get(out, 0) + 1
            print(f"[{scene:7s} étape {stage}] {terrain:12s} reset {r['dt']*1e3:5.0f} ms, {r['n_ids']:3d} corps, "
                  f"dist cible {r['r_t']:6.1f} m, cap {r['d_head']:4.1f}°{' (départ de secours)' if r['fallback'] else ''}"
                  f" | {k:3d} steps, reward {tot:+8.2f}, issue {out}")
    # changement de carte entre deux resets
    counts = {}
    for t in ("field@v1", "city@v1", "field@v2"):
        env.terrains = [t]
        env.reset()
        counts[t] = len(env.sim.TERRAIN_IDS)
    assert counts["field@v1"] != counts["city@v1"], counts
    print(f"[terrain] nb de corps après reset : {counts}")
    # timeout = truncated
    cfg_t = copy.deepcopy(cfg)
    cfg_t["curriculum"]["stages"][0][3] = 1.
    env.cur = cfg_t["curriculum"]
    env.terrains = ["field@v1"]
    env.set_stage(0)
    env.reset()
    for _ in range(31):
        _, r, term, trunc, info = env.step(np.zeros(4))
        if term or trunc:
            break
    print(f"[timeout] après {env.t} steps (attendu 30) : terminated={term}, truncated={trunc}, issue {info.get('outcome')}")
    assert trunc and not term and env.t == 30 or info.get("outcome") in ("collision", "success")
    env.close()
    print("[issues]", outcomes)
    print("Env OK")


if __name__ == "__main__":
    main()
