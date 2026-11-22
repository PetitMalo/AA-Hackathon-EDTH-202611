"""Suivi TensorBoard de l'explorateur."""

from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

EXPLORE_OUTCOMES = ("detected", "success", "collision", "out_of_arena", "timeout", "coverage_done")


class ExploreStatsCallback(BaseCallback):
    """Explorateur : couverture finale, taux et temps de détection, issues, sur les derniers épisodes."""

    def __init__(self, window=300):
        super().__init__()
        self.recent = deque(maxlen=window)

    def _on_step(self):
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done and info.get("outcome"):
                self.recent.append(info)
        if self.n_calls % 256 == 0 and self.recent:
            n = len(self.recent)
            det = [i["t_detect"] for i in self.recent if i["detected"]]
            self.logger.record("explore/episodes_in_window", n)
            self.logger.record("explore/coverage_mean", sum(i["coverage"] for i in self.recent) / n)
            self.logger.record("explore/detection_rate", len(det) / n)
            if det:
                self.logger.record("explore/t_detect_mean_s", sum(det) / len(det))
            for o in EXPLORE_OUTCOMES:
                self.logger.record(f"explore/{o}_rate", sum(1 for i in self.recent if i["outcome"] == o) / n)
        return True
