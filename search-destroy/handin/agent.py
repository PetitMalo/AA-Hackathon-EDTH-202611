"""Participant entry point: voxel-map frontier exploration, then go to the detected target."""

from handin.explorer import FrontierExplorer


class Agent:
    def __init__(self):
        self.explorer = FrontierExplorer()

    def reset(self):
        self.explorer.reset()

    def act(self, observation):
        # One iteration: update the voxel map, pick target / nearest frontier, return (roll, pitch, yaw, throttle).
        return self.explorer.step(observation)
