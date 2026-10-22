"""Real-simulator V1/V2 equivalence: the app's replay must equal a direct run of
the research engine built with the research repo's own builders (no visualizer
logic in between). Skipped unless VISUALIZER_RESEARCH_DATA_ROOT is set.
"""

import os
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
from visualizer.backend.config import load_settings
from visualizer.backend.replay_engine import ReplayRequest, run_replay_episode

RESEARCH_ROOT = os.environ.get("VISUALIZER_RESEARCH_DATA_ROOT")
pytestmark = pytest.mark.skipif(
    not RESEARCH_ROOT,
    reason="Set VISUALIZER_RESEARCH_DATA_ROOT to run the real simulator semantics tests.",
)

SEED = 130000


@pytest.fixture(scope="module", autouse=True)
def research_on_path():
    root = str(Path(RESEARCH_ROOT).resolve())
    sys.path.insert(0, root)
    yield
    try:
        sys.path.remove(root)
    except ValueError:
        pass


def _replay(semantics, clutter="C0", defender="greedy", seed=SEED):
    request = ReplayRequest(
        defender_policy_id=defender, attacker_condition="A0", clutter_level=clutter,
        perception_mode="map_local", semantics=semantics, evaluation_seed=seed,
    )
    return run_replay_episode(request, load_settings()).episodes[0]


def _direct_run(builder, clutter="C0", defender="greedy", seed=SEED):
    """Reference: the research engine driven exactly as the research code does."""
    from uav_defend.envs.soldier_env import SoldierEnv
    from uav_defend.policies.registry import get_policy
    from uav_defend.policies.sanitize import build_policy_info

    config = builder(clutter, "measurement", awareness_mode="map_local")
    env = SoldierEnv(config)
    env.enemy_tactical_override = None
    policy = get_policy(defender) if defender in ("greedy", "kalman_greedy") else get_policy(defender, config=config)
    _obs, info = env.reset(seed=seed)
    policy.reset()
    states = [{k: np.asarray(info[k], dtype=np.float64) for k in ("defender_pos", "enemy_pos", "soldier_pos")}]
    rewards, infos = [], []
    done = False
    while not done:
        action = np.asarray(policy.act(env._get_obs(), build_policy_info(env._get_info(), "measurement")), dtype=np.float32)
        _obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        states.append({k: np.asarray(info[k], dtype=np.float64) for k in ("defender_pos", "enemy_pos", "soldier_pos")})
        rewards.append(float(reward))
        infos.append(info)
    env.close()
    return states, rewards, infos


def _assert_same_trajectory(replay, states, rewards, infos):
    assert len(replay.frames) == len(states) == len(rewards) + 1
    for frame, state in zip(replay.frames, states):
        for key, entity in (("defender_pos", frame.defender), ("enemy_pos", frame.attacker), ("soldier_pos", frame.asset)):
            assert np.array_equal(np.asarray(entity.position_m, dtype=np.float64), state[key])
    assert [frame.reward for frame in replay.frames[1:]] == rewards
    assert replay.frames[-1].outcome == infos[-1]["outcome"]


def test_v1_replay_equals_the_original_canonical_research_run():
    from experiments.journal_clutter_protocol import build_canonical_env_config

    replay = _replay("V1")
    assert replay.environment.semantics == "V1"
    assert replay.environment.builder == "experiments.journal_clutter_protocol.build_canonical_env_config"
    assert replay.environment.config["mission_contact_mode"] == "endpoint"
    assert replay.environment.config["enemy_nominal_guidance_mode"] == "legacy"
    _assert_same_trajectory(replay, *_direct_run(build_canonical_env_config))
    # Known legacy behavior: the weaving attacker circles for hundreds of steps.
    assert (replay.summary.outcome, replay.summary.steps) == ("intercepted", 402)


def test_v2_replay_equals_the_research_v2_builder_run_and_swept_contact_is_reported():
    from experiments.journal_v2_protocol import build_v2_env_config

    replay = _replay("V2")
    assert replay.environment.semantics == "V2"
    assert replay.environment.builder == "experiments.journal_v2_protocol.build_v2_env_config"
    assert replay.environment.config["mission_contact_mode"] == "swept"
    assert replay.environment.config["enemy_nominal_guidance_mode"] == "direct_obstacle_only"
    assert replay.environment.config["defender_standby_until_detection"] is True
    states, rewards, infos = _direct_run(build_v2_env_config)
    _assert_same_trajectory(replay, states, rewards, infos)
    assert (replay.summary.outcome, replay.summary.steps) == ("intercepted", 11)
    info = replay.frames[-1].privileged_diagnostics.simulator_info
    assert info["terminal_event"] == "interception"
    assert info["terminal_event_fraction"] == pytest.approx(0.4413194345555628)
    (interception,) = [event for event in replay.events if event.kind == "interception"]
    assert interception.contact_fraction == pytest.approx(0.4413194345555628)
    # Corrected A0: no weave, no evasion, always in the direct mission mode in clear space.
    assert {frame.environmental_state["enemy_behavior_mode"] for frame in replay.frames} == {"MISSION"}
    assert all(not frame.environmental_state["enemy_evasion_active"] for frame in replay.frames)


def test_selected_semantics_change_the_trajectory_and_are_deterministic():
    v1, v2 = _replay("V1"), _replay("V2")
    assert v1.environment.config_sha256 != v2.environment.config_sha256
    assert v1.summary.steps != v2.summary.steps
    assert _replay("V2") == v2


@pytest.mark.parametrize("clutter,seed,steps", [("C2", 130065, 101), ("C3", 130063, 72), ("C3", 130081, 46)])
def test_diagnostic_seeds_load_under_v2_with_recorded_avoidance_detail(clutter, seed, steps):
    from experiments.journal_v2_protocol import build_v2_env_config

    replay = _replay("V2", clutter=clutter, seed=seed)
    assert replay.environment.clutter == clutter and replay.simulation_seed == seed
    assert replay.summary.steps == steps and replay.summary.outcome == "soldier_caught"
    _assert_same_trajectory(replay, *_direct_run(build_v2_env_config, clutter=clutter, seed=seed))
    modes = {frame.environmental_state["enemy_behavior_mode"] for frame in replay.frames[1:]}
    assert modes == {"MISSION", "COLLISION_AVOID"}
    avoiding = [frame.environmental_state for frame in replay.frames[1:]
                if frame.environmental_state["enemy_behavior_mode"] == "COLLISION_AVOID"]
    assert all(state["enemy_obstacle_avoidance_mode"] in ("left", "right", "climb") for state in avoiding)
    assert all(state.get("enemy_avoidance_obstacle_id") is not None for state in avoiding)
    (reached,) = [event for event in replay.events if event.kind == "asset_reached"]
    assert reached.contact_fraction is not None and 0.0 <= reached.contact_fraction <= 1.0
