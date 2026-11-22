"""Debug run of the explorer on one flight: prints the loop state and saves a top-down map.

    uv run python -m handin.run_explorer --terrain field@v1 --seed 0 --time 60

Uses `info` / the true state for printing only, never for the policy.
"""

import argparse
import time

import numpy as np
import pybullet as p
from PIL import Image

from fpv_sim.envs.approach_aviary import ApproachAviary
from fpv_sim.scripts.evaluate import mission_start
from handin.agent import Agent


def map_rgb(voxel_map):
    """Top-down colour image of the voxel map, indexed (x, y): unknown / free / occupied / frontier."""
    top = voxel_map.top_down()
    rgb = np.zeros(top.shape + (3,), dtype=np.uint8)
    rgb[top == 0] = (40, 40, 40)
    rgb[top == 1] = (200, 200, 200)
    rgb[top == 2] = (30, 120, 30)
    rgb[voxel_map.frontier.any(2)] = (60, 140, 255)
    return rgb


def save_map(explorer, path, true_track, est_track, box_xy=None):
    m = explorer.map
    rgb = map_rgb(m)
    top = rgb[..., 0]

    def px(xy):
        i = ((np.asarray(xy) - m.origin[:2]) / m.res).astype(int)
        return np.clip(i, 0, np.array(top.shape) - 1)

    for p in true_track:
        rgb[tuple(px(p[:2]))] = (255, 0, 0)
    for p in est_track:
        rgb[tuple(px(p[:2]))] = (255, 255, 0)
    if box_xy is not None:
        i, j = px(box_xy)
        rgb[max(i - 2, 0):i + 3, max(j - 2, 0):j + 3] = (255, 140, 0)
    # array axes are (x, y): show x to the right, y up
    Image.fromarray(np.flipud(rgb.transpose(1, 0, 2))).resize((top.shape[0] * 2, top.shape[1] * 2),
                                                              Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--terrain", default="field@v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--time", type=float, default=120.)
    ap.add_argument("--out", default="out/explorer_map.png")
    args = ap.parse_args()

    env = ApproachAviary(terrain=args.terrain)
    agent = Agent()
    try:
        obs, start = mission_start(env, np.random.default_rng(args.seed), args.terrain)
        agent.reset()
        true_track, est_track, lat = [], [], []
        outcome = "timeout"
        for step in range(int(args.time * 30)):
            t0 = time.perf_counter()
            action = agent.act(obs)
            lat.append(time.perf_counter() - t0)
            obs, _, terminated, truncated, info = env.step(action)
            ex = agent.explorer.info
            true_track.append(env.pos[0].copy())
            if ex.position is not None:
                est_track.append(ex.position)
            if step % 30 == 0:
                goal = "-" if ex.goal is None else np.round(ex.goal, 1)
                print(f"t={step / 30:5.1f}s mode={ex.mode:8s} pos_est={np.round(ex.position, 1)} "
                      f"pos_true={np.round(env.pos[0], 1)} goal={goal} frontiers={ex.n_frontiers} "
                      f"obst={ex.obstacle_ahead} boxes={ex.n_boxes}{'(blk)' if ex.corridor_blocked else ''} "
                      f"mpc={ex.mpc_ms:.1f}ms/{ex.mpc_iters}it vmax={ex.vmax:.1f} clutter={ex.n_clutter} "
                      f"margin={ex.margin:.2f} path={ex.path_len:.0f}m geo={ex.geo_ms:.1f}ms unreach={ex.n_unreachable} "
                      f"sw/min={ex.switches_min:.0f} uturns={ex.n_uturns} disp8s={ex.stuck_disp:.1f} esc={ex.n_escapes}"
                      f"{'(' + ex.escape_level + '/' + ex.stuck_reason + ')' if ex.n_escapes else ''} "
                      f"sight={ex.delta_deg:.1f}deg lost={ex.lost_s:.1f}s floor={int(ex.floor_on)} reacq={int(ex.reacquire)} "
                      f"obst={ex.obstacle_pts} strikes={ex.n_strikes}"
                      f"{'(' + ex.strike_abort + ')' if ex.strike_abort else ''} "
                      f"box_dist={info['distance_m']:.1f} act={np.round(action, 2)}")
            if terminated or truncated:
                outcome = "success" if info["is_success"] else "collision" if info["collision"] else "terminated"
                break
        if agent.explorer.cfg.use_mpc:
            print(f"corridor replans={agent.explorer.planner.n_replans}")
        print(f"outcome={outcome} t={step / 30:.1f}s mean_latency={1e3 * np.mean(lat):.1f}ms "
              f"max_latency={1e3 * np.max(lat):.1f}ms")
        box_xy = np.array(p.getBasePositionAndOrientation(env.TERRAIN_IDS[0], physicsClientId=env.CLIENT)[0][:2])
        save_map(agent.explorer, args.out, true_track, est_track, box_xy)
        print("map saved to", args.out)
    finally:
        env.close()


if __name__ == "__main__":
    main()
