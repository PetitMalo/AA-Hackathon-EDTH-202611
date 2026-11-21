"""Participant entry point: voxel-map frontier exploration, then go to the detected target."""

from handin.explorer import FrontierExplorer


class Agent:
    def __init__(self):
        self.explorer = FrontierExplorer()

    def reset(self):
        self.explorer.reset()

    def act(self, observation):
        # One iteration: update the voxel map, pick target / nearest frontier, return (roll, pitch, yaw, throttle).
        action = self.explorer.step(observation)
        # Telemetry for the evaluation logger: which phase the policy is in (explore | target). This is
        # the only condition that switches the flight's exploration -> terminal phase; there is no
        # separate distance threshold in the policy.
        observation["phase"] = "terminal" if self.explorer.info.mode == "target" else "exploration"
        return action
