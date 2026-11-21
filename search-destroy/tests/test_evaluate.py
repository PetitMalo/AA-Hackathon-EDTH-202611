import json

import numpy as np
from mcap.reader import make_reader

from fpv_sim.envs.approach_aviary import ApproachAviary
from fpv_sim.scripts.evaluate import fly, summarize, target_center, target_relative


class Climb:
    def reset(self):
        self.calls = 0
        self.first_imu = None

    def act(self, observation):
        self.calls += 1
        assert "pose" not in observation
        if self.first_imu is None:
            self.first_imu = {key: observation[key].tolist() for key in ("gyro_rad_s", "accel_m_s2")}
        return np.array([0, 0, 0, .5])


def test_flight_is_logged_and_summarized(tmp_path):
    env = ApproachAviary(continuous=True, terrain="field@v1")
    agent = Climb()
    try:
        record = fly(env, agent, np.random.default_rng(0), tmp_path / "run.mcap", timeout_s=1., depth_every=5)
    finally:
        env.close()
    assert record["outcome"] == "timeout" and record["steps"] == 30
    # An agent that never reports a phase spends the whole flight in exploration.
    assert record["exploration_time_s"] == record["time_s"] and record["terminal_time_s"] == 0.
    with open(tmp_path / "run.mcap", "rb") as f:
        reader = make_reader(f)
        messages = [(channel.topic, message.data) for _, channel, message in reader.iter_messages()]
        topics = [topic for topic, _ in messages]
        assert reader.get_summary().statistics.metadata_count == 1
    assert topics.count("/agent/action") == 30 and topics.count("/truth") == 31
    assert topics.count("/observation/depth") == 7
    assert topics.count("/analysis/target_relative") == 30
    assert {"/observation/detection", "/observation/vio", "/imu", "/mission/phase"} <= set(topics)
    assert "/observation/pose" not in topics
    assert json.loads(next(data for topic, data in messages if topic == "/imu")) == agent.first_imu
    phases = [json.loads(data) for topic, data in messages if topic == "/mission/phase"]
    assert phases[0]["event"] == "enter" and phases[0]["phase"] == "exploration"
    assert phases[-1]["event"] == "final" and phases[-1]["phase"] == "exploration"
    rel = json.loads(next(data for topic, data in messages if topic == "/analysis/target_relative"))
    assert set(rel) == {"distance_m", "bearing_deg", "elevation_deg", "lateral_offset_m", "vertical_offset_m"}

    m = summarize([{"terrain": "field@v1", **record},
                   {"terrain": "field@v2", **record, "outcome": "success", "time_s": 10., "mean_speed_m_s": 4.}])["field"]
    assert m["success_rate"] == m["timeout_rate"] == .5
    assert m["reach_time_s"] == 10. and m["reach_mean_speed_m_s"] == 4.
    assert m["exploration_time_s"] == 1. and m["terminal_time_s"] == 0.
    assert m["latency_mean_ms"] < 33


class PhaseSwitch(Climb):
    """Reports terminal after 0.5 s, flickers back once; the mission split must latch."""

    def act(self, observation):
        action = super().act(observation)
        observation["phase"] = ("terminal" if self.calls >= 15 else "exploration") if self.calls != 20 else "exploration"
        return action


def test_phase_split_latches_on_first_commit(tmp_path):
    env = ApproachAviary(continuous=True, terrain="field@v1")
    agent = PhaseSwitch()
    try:
        record = fly(env, agent, np.random.default_rng(0), tmp_path / "run.mcap", timeout_s=1., depth_every=5)
    finally:
        env.close()
    # Commit at step 15 (t = 14/30 s); the flicker at step 20 stays a debug event only.
    assert np.isclose(record["exploration_time_s"], 14 / 30)
    assert np.isclose(record["terminal_time_s"], record["time_s"] - 14 / 30)
    with open(tmp_path / "run.mcap", "rb") as f:
        events = [json.loads(m.data) for _, c, m in make_reader(f).iter_messages(topics=["/mission/phase"])]
    kinds = [e["event"] for e in events]
    assert kinds.count("transition") == 1 and kinds.count("policy_transition") == 3
    commit = events[kinds.index("transition")]
    assert commit["from"] == "exploration" and commit["to"] == "terminal"


def test_target_relative_frame():
    # Drone at origin facing +x (yaw = 0): target ahead, left, below.
    rel = target_relative([0., 0., 1.55], 0., [10., 0., 1.55])
    assert rel["distance_m"] == 10. and rel["bearing_deg"] == 0. and rel["elevation_deg"] == 0.
    rel = target_relative([0., 0., 1.55], 0., [0., 10., 1.55])
    assert rel["bearing_deg"] == 90. and rel["lateral_offset_m"] == 10.
    rel = target_relative([0., 0., 1.55], np.pi / 2, [0., 10., 1.55])
    assert np.isclose(rel["bearing_deg"], 0.)                # facing +y, target in +y
    rel = target_relative([0., 0., 11.55], 0., [10., 0., 1.55])
    assert np.isclose(rel["elevation_deg"], -45.)
