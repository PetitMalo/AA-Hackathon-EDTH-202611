"""Live viewer of the explorer: onboard FPV view (left) and the voxel map seen from above (right).

    uv run python -m handin.view_explorer --terrain village@v1 --seed 0
    uv run python -m handin.view_explorer --terrain city@v12 --eval-run 0   # replays run 0 of `evaluate` on city@v12

Keys: SPACE pause, N next flight (new seed, or next eval run), F fast mode (no 30 FPS limit), ESC / close to quit.
The true pose and box position are only drawn for debugging, never given to the agent.
"""

import argparse

import numpy as np
import pybullet as p
import pygame

from fpv_sim.envs.approach_aviary import ApproachAviary
from fpv_sim.scripts.evaluate import random_start
from fpv_sim.utils.perception import Camera, draw_box
from handin.agent import Agent
from handin.run_explorer import map_rgb

FPV_SIZE = (640, 360)
MAP_PX = 600
MODE_COLOURS = {"init": "gray", "explore": "deepskyblue", "target": "orange", "attack": "red", "hover": "yellow"}


class Viewer:
    def __init__(self, terrain, seed, timeout_s, eval_run=None):
        self.env = ApproachAviary(terrain=terrain)
        self.agent = Agent()
        self.fpv_cam = Camera(self.env.CLIENT, width=480, height=270)
        self.terrain, self.seed, self.timeout_s = terrain, seed, timeout_s
        self.eval_run = eval_run           # None: free seed; int: same start as fpv_sim.scripts.evaluate's run
        pygame.init()
        self.screen = pygame.display.set_mode((FPV_SIZE[0] + MAP_PX, MAP_PX))
        self.font = pygame.font.Font(None, 26)
        self.clock = pygame.time.Clock()
        self.new_flight()

    def new_flight(self):
        self.env.reset(seed=self.seed)
        if self.eval_run is None:
            rng, label = np.random.default_rng(self.seed), f"seed {self.seed}"
        else:                              # the exact rng of evaluate.run_target: same start, same sensor noise
            rng = np.random.default_rng([self.seed, sum(map(ord, self.terrain)), self.eval_run])
            label = f"eval run {self.eval_run} (--seed {self.seed})"
        self.obs, _ = random_start(self.env, rng)
        self.agent.reset()
        self.step, self.outcome = 0, None
        self.true_track, self.est_track = [], []
        self.box_xy = np.array(p.getBasePositionAndOrientation(self.env.TERRAIN_IDS[0],
                                                               physicsClientId=self.env.CLIENT)[0][:2])
        pygame.display.set_caption(f"Explorer — {self.terrain} {label}")

    def advance(self):
        action = self.agent.act(self.obs)
        self.obs, _, terminated, truncated, info = self.env.step(action)
        self.step += 1
        self.info = info
        self.true_track.append(self.env.pos[0].copy())
        ex = self.agent.explorer.info
        if ex.position is not None:
            self.est_track.append(ex.position.copy())
        if terminated or truncated:
            self.outcome = "SUCCESS" if info["is_success"] else "COLLISION" if info["collision"] else "OUT / TERMINATED"
        elif self.step / 30 >= self.timeout_s:
            self.outcome = "TIMEOUT"

    # ------------------------------------------------------------------ drawing
    def world_to_map(self, xy):
        m = self.agent.explorer.map
        scale = MAP_PX / m.shape[0]
        i, j = (np.asarray(xy)[:2] - m.origin[:2]) / m.res
        return FPV_SIZE[0] + int(i * scale), int((m.shape[1] - j) * scale)

    def draw(self):
        s = self.screen
        s.fill("black")

        # FPV view with the detector box (box from the observation, drawn in the 480x270 render)
        rgb = self.fpv_cam.read(self.env._getDroneStateVector(0))
        if self.obs["detected"]:
            b = self.obs["box"] * [480, 270, 480, 270]
            rgb = draw_box(rgb, tuple(int(v) for v in b))
        s.blit(pygame.transform.scale(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), FPV_SIZE), (0, 0))

        # Voxel map, x right / y up
        img = map_rgb(self.agent.explorer.map)[:, ::-1]
        s.blit(pygame.transform.scale(pygame.surfarray.make_surface(img), (MAP_PX, MAP_PX)), (FPV_SIZE[0], 0))
        for track, col, w in ((self.true_track, "red", 2), (self.est_track, "yellow", 1)):
            pts = [self.world_to_map(q) for q in track[::3]]
            if len(pts) > 1:
                pygame.draw.lines(s, col, False, pts, w)
        pygame.draw.circle(s, "orange", self.world_to_map(self.box_xy), 5, 2)
        ex = self.agent.explorer.info
        if ex.position is not None:
            pos = self.world_to_map(ex.position)
            pygame.draw.circle(s, "yellow", pos, 4)
            head = ex.position[:2] + 12 * np.array([np.cos(ex.yaw), np.sin(ex.yaw)])
            pygame.draw.line(s, "yellow", pos, self.world_to_map(head), 2)
            if ex.goal is not None:
                pygame.draw.line(s, "magenta", pos, self.world_to_map(ex.goal), 1)
                pygame.draw.circle(s, "magenta", self.world_to_map(ex.goal), 5)
        if ex.target_estimate is not None:
            pygame.draw.circle(s, "orange", self.world_to_map(ex.target_estimate), 3)

        # HUD
        v = np.linalg.norm(self.env.vel[0])
        lines = [
            (f"t = {self.step / 30:5.1f} s    mode = {ex.mode}", MODE_COLOURS.get(ex.mode, "white")),
            (f"alt {self.env.pos[0][2]:5.1f} m    speed {v:4.1f} m/s    box dist {self.info['distance_m']:6.1f} m"
             if hasattr(self, "info") else "", "white"),
            (f"frontiers {ex.n_frontiers}    coverage {100 * ex.coverage:4.1f} %    obstacle ahead {ex.obstacle_ahead}",
             "white"),
            (f"goal {None if ex.goal is None else np.round(ex.goal, 1)}", "magenta"),
            (f"sticks r/p/y/t {np.round(ex.action, 2)}", "white"),
            ("map: red true track, yellow estimate, magenta goal, blue frontiers, green obstacles", "gray"),
            ("SPACE pause   N next flight   F fast   ESC quit" + ("   [FAST]" if self.fast else ""), "gray"),
        ]
        for k, (txt, col) in enumerate(lines):
            s.blit(self.font.render(txt, True, col), (10, FPV_SIZE[1] + 10 + 30 * k))
        if self.outcome:
            s.blit(pygame.font.Font(None, 64).render(self.outcome, True, "white", "black"), (20, 20))
        if self.paused and not self.outcome:
            s.blit(pygame.font.Font(None, 64).render("PAUSE", True, "white", "black"), (20, 20))
        pygame.display.flip()

    # ------------------------------------------------------------------ main loop
    def run(self):
        self.paused, self.fast = False, False
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.paused = not self.paused
                        elif event.key == pygame.K_f:
                            self.fast = not self.fast
                        elif event.key == pygame.K_n:
                            if self.eval_run is None:
                                self.seed += 1
                            else:
                                self.eval_run += 1
                            self.new_flight()
                if not self.paused and not self.outcome:
                    self.advance()
                self.draw()
                if not self.fast:
                    self.clock.tick(30)
        finally:
            self.env.close()
            pygame.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain", default="field@v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=120.)
    ap.add_argument("--eval-run", type=int, default=None,
                    help="replay this run of fpv_sim.scripts.evaluate (the `run` column of flights.csv; "
                         "--seed is then evaluate's --seed)")
    args = ap.parse_args()
    Viewer(args.terrain, args.seed, args.timeout, args.eval_run).run()


if __name__ == "__main__":
    main()
