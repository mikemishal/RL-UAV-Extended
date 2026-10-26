"""Passive defender replay (V2 endurance inspection).

`passive` reproduces the research validation harness's parked defender
(`experiments.journal_v2_validation_core.record_episode(controller="passive")`): no policy object,
a zero action every step, `defender_standby_until_detection=False`, defender parked at
`PARKED_DEFENDER` at rest. The simulator-dependent tests are skipped unless
VISUALIZER_RESEARCH_DATA_ROOT is set; no training or learned policy is involved.
"""

import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

np = pytest.importorskip("numpy")
from visualizer.backend import replay_catalog, replay_cli
from visualizer.backend.replay_contract import EpisodeSummary, ReplayEvent
from visualizer.backend.replay_requests import POLICY_IDS, ReplayBatchRequest, ReplayRequest

RESEARCH_ROOT = os.environ.get("VISUALIZER_RESEARCH_DATA_ROOT")
needs_research = pytest.mark.skipif(
    not RESEARCH_ROOT, reason="Set VISUALIZER_RESEARCH_DATA_ROOT to run the real simulator tests.",
)
EXISTING_DEFENDERS = ("greedy", "kalman_greedy", "pn", "lead", "rmpc", "ppo", "ppo_kalman", "lr_ppo")


def _request(defender="passive", **kwargs):
    values = dict(
        defender_policy_id=defender, attacker_condition="A0", clutter_level="C3",
        perception_mode="map_local", semantics="V2", evaluation_seed=130017,
    )
    values.update(kwargs)
    return ReplayRequest(**values)


# ------------------------------------------------------------ request validation
def test_passive_is_a_valid_analytical_request_and_existing_defenders_are_unchanged():
    assert _request().defender_policy_id == "passive"
    assert POLICY_IDS[: len(EXISTING_DEFENDERS)] == EXISTING_DEFENDERS and POLICY_IDS[-1] == "passive"
    for defender in ("greedy", "kalman_greedy", "pn", "lead", "rmpc"):
        assert _request(defender).defender_policy_id == defender
    batch = ReplayBatchRequest(
        defender_policy_id="passive", attacker_condition="A0", clutter_level="C3", semantics="V2",
        seed_start=130017, episode_count=1,
    )
    assert batch.episode_request(130017).defender_policy_id == "passive"


@pytest.mark.parametrize("extra", [
    {"defender_generation": "d0"}, {"defender_training_seed": 71},
    {"defender_architecture": "passive"}, {"checkpoint_role": "d0_start"},
])
def test_passive_rejects_learned_defender_fields(extra):
    with pytest.raises(ValidationError):
        _request(**extra)


def test_unknown_defender_is_still_rejected_and_learned_defenders_still_need_a_seed():
    with pytest.raises(ValidationError):
        _request("parked")
    with pytest.raises(ValidationError):
        _request("lr_ppo")


def test_cli_and_catalog_expose_passive_without_changing_the_existing_analytical_order():
    parser = replay_cli._parser()
    assert parser.parse_args(["--defender", "passive", "--clutter", "C3", "--evaluation-seed", "1"]).defender == "passive"
    assert list(replay_catalog.ANALYTICAL_METADATA)[:5] == ["greedy", "kalman_greedy", "pn", "lead", "rmpc"]
    assert list(replay_catalog.ANALYTICAL_METADATA)[5:] == ["passive"]


# ------------------------------------------------------------ replay contract
def test_contract_accepts_the_endurance_outcome_event_and_metrics():
    event = ReplayEvent(kind="endurance_exhausted", step=54, time_s=27.0, source="info.outcome", entity="attacker",
                        contact_fraction=0.63)
    assert event.kind == "endurance_exhausted"
    summary = EpisodeSummary(
        outcome="enemy_endurance_exhausted", steps=54, duration_s=27.0, final_defender_attacker_distance_m=1.0,
        final_attacker_asset_distance_m=9.6, total_reward=-1.0,
        safety_metrics={"attacker_endurance_traveled_m": 285.6, "attacker_endurance_budget_m": 285.6},
    )
    assert summary.outcome == "enemy_endurance_exhausted"
    with pytest.raises(ValidationError):
        EpisodeSummary(outcome="battery_empty", steps=1, duration_s=0.5, final_defender_attacker_distance_m=1.0,
                       final_attacker_asset_distance_m=1.0, total_reward=0.0)


# ------------------------------------------------------------ real simulator
@pytest.fixture(scope="module")
def research_on_path():
    root = str(Path(RESEARCH_ROOT).resolve())
    sys.path.insert(0, root)
    yield root
    try:
        sys.path.remove(root)
    except ValueError:
        pass


def _replay(defender, clutter, seed, semantics="V2"):
    from visualizer.backend.config import load_settings
    from visualizer.backend.replay_engine import run_replay_episode

    request = _request(defender, clutter_level=clutter, evaluation_seed=seed, semantics=semantics)
    return run_replay_episode(request, load_settings())


@pytest.fixture(scope="module")
def passive_130017(research_on_path):
    return _replay("passive", "C3", 130017)


@needs_research
def test_passive_c3_130017_ends_by_endurance_exhaustion_with_the_validated_k5_values(passive_130017):
    bundle = passive_130017
    episode = bundle.episodes[0]
    assert episode.summary.outcome == "enemy_endurance_exhausted" and episode.summary.steps == 54
    assert episode.environment.config["enemy_endurance_enabled"] is True
    assert episode.environment.config["enemy_endurance_path_factor"] == 5.0
    metrics = episode.summary.safety_metrics
    assert metrics["attacker_endurance_enabled"] is True
    assert metrics["attacker_endurance_traveled_m"] == pytest.approx(285.6496, abs=1e-3)
    assert metrics["attacker_endurance_budget_m"] == pytest.approx(metrics["attacker_endurance_traveled_m"])
    assert metrics["attacker_endurance_remaining_m"] == 0.0
    assert metrics["attacker_path_length_m"] == pytest.approx(285.6496, abs=1e-3)
    assert metrics["attacker_collision_count"] == 0 and episode.summary.collision_type is None
    (event,) = [e for e in episode.events if e.kind == "endurance_exhausted"]
    assert event.step == 54 and 0.0 < event.contact_fraction < 1.0
    assert not any(e.kind in ("collision", "asset_reached", "interception") for e in episode.events)
    final = episode.frames[-1].privileged_diagnostics.simulator_info
    assert final["terminal_event"] == "endurance" and final["enemy_endurance_remaining_m"] == 0.0
    # Endurance is reported as a resource, never as a behavior mode.
    assert {f.environmental_state["enemy_behavior_mode"] for f in episode.frames[1:]} <= {"MISSION", "COLLISION_AVOID"}


@needs_research
def test_every_frame_carries_monotone_endurance_fields(passive_130017):
    frames = passive_130017.episodes[0].frames
    infos = [f.privileged_diagnostics.simulator_info for f in frames]
    assert all(i["enemy_endurance_enabled"] is True for i in infos)
    traveled = [i["enemy_endurance_traveled_m"] for i in infos]
    assert traveled[0] == 0.0 and all(b >= a for a, b in zip(traveled, traveled[1:]))
    assert all(i["enemy_endurance_traveled_m"] + i["enemy_endurance_remaining_m"] == pytest.approx(i["enemy_endurance_budget_m"])
               for i in infos)


@needs_research
def test_passive_replay_equals_the_research_passive_harness_and_the_no_endurance_prefix(passive_130017):
    import dataclasses

    from experiments.journal_v2_validation_core import PARKED_DEFENDER, record_episode

    episode = passive_130017.episodes[0]
    assert episode.environment.config["defender_standby_until_detection"] is False
    assert tuple(episode.frames[0].defender.position_m) == PARKED_DEFENDER
    assert all(tuple(f.defender.position_m) == PARKED_DEFENDER for f in episode.frames)
    attacker = np.array([f.attacker.position_m for f in episode.frames])
    harness = record_episode("C3", "V2", "passive", 130017)
    assert np.array_equal(attacker, np.array([r["attacker_pos"] for r in harness["rows"]]))
    baseline = record_episode(
        "C3", "V2", "passive", 130017,
        cfg_transform=lambda cfg: dataclasses.replace(cfg, enemy_endurance_enabled=False),
    )
    assert baseline["final_info"]["outcome"] == "soldier_caught"
    prefix = len(attacker) - 1  # the final frame is the interpolated exhaustion instant
    assert np.array_equal(attacker[:prefix], np.array([r["attacker_pos"] for r in baseline["rows"][:prefix]]))


@needs_research
def test_replay_reports_the_accepted_clean_research_source(passive_130017):
    from visualizer.backend.replay_engine import EXPECTED_RESEARCH_COMMIT

    provenance = passive_130017.provenance
    assert provenance.simulation_source_commit == EXPECTED_RESEARCH_COMMIT
    assert EXPECTED_RESEARCH_COMMIT == "bdbee45dc08a583a0da5ca882dc6b1269207a8cd"
    assert provenance.source_dirty is False


@needs_research
@pytest.mark.parametrize("defender", ["greedy", "lead"])
def test_existing_defenders_on_the_clear_space_case_are_unchanged(research_on_path, defender):
    from experiments.journal_v2_protocol import build_v2_env_config
    from uav_defend.envs.soldier_env import SoldierEnv
    from uav_defend.policies.registry import get_policy
    from uav_defend.policies.sanitize import build_policy_info

    config = build_v2_env_config("C0", "measurement", awareness_mode="map_local")
    env = SoldierEnv(config)
    env.enemy_tactical_override = None
    policy = get_policy(defender) if defender == "greedy" else get_policy(defender, config=config)
    _obs, info = env.reset(seed=130000)
    policy.reset()
    positions, done = [np.asarray(info["enemy_pos"], dtype=np.float64)], False
    while not done:
        action = np.asarray(policy.act(env._get_obs(), build_policy_info(env._get_info(), "measurement")), dtype=np.float32)
        _o, _r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        positions.append(np.asarray(info["enemy_pos"], dtype=np.float64))
    episode = _replay(defender, "C0", 130000).episodes[0]
    assert episode.summary.outcome == info["outcome"]
    assert np.array_equal(np.array([f.attacker.position_m for f in episode.frames]), np.array(positions))
