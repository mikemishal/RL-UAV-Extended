"""Simulation-semantics selector (V1/V2): request, contract, config wiring, CLI.

The visualizer only forwards the selector to the research repository's own
builders. These tests use fake builders, so they need no research checkout; the
real-simulator checks live in test_replay_equivalence.py.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

np = pytest.importorskip("numpy")
from visualizer.backend import replay_cli
from visualizer.backend import replay_engine as engine
from visualizer.backend.application_error import ApplicationError
from visualizer.backend.replay_contract import EnvironmentSpec
from visualizer.backend.replay_requests import ReplayBatchRequest, ReplayRequest

CANONICAL = "experiments.journal_clutter_protocol.build_canonical_env_config"
V2_BUILDER = "experiments.journal_v2_protocol.build_v2_env_config"
V1_SWITCHES = ("endpoint", "legacy")
V2_SWITCHES = ("swept", "direct_obstacle_only")


def _request(**overrides):
    values = dict(
        defender_policy_id="greedy", attacker_condition="A0", clutter_level="C0", evaluation_seed=130000,
    )
    values.update(overrides)
    return ReplayRequest(**values)


# ----------------------------------------------------------------- request
def test_semantics_defaults_to_v1_everywhere():
    assert _request().semantics == "V1"
    batch = ReplayBatchRequest(defender_policy_id="greedy", attacker_condition="A0", clutter_level="C0")
    assert batch.semantics == "V1"
    assert batch.episode_request(7).semantics == "V1"


def test_v2_is_accepted_and_propagates_to_every_batch_episode():
    assert _request(semantics="V2").semantics == "V2"
    batch = ReplayBatchRequest(
        defender_policy_id="greedy", attacker_condition="A0", clutter_level="C2", seed_start=130065,
        episode_count=3, semantics="V2",
    )
    assert [batch.episode_request(s).semantics for s in range(130065, 130068)] == ["V2"] * 3


@pytest.mark.parametrize("bad", ["v2", "V3", "", None, 2])
def test_only_exact_v1_or_v2_is_accepted(bad):
    with pytest.raises(ValidationError):
        _request(semantics=bad)
    with pytest.raises(ValidationError):
        ReplayBatchRequest(defender_policy_id="greedy", attacker_condition="A0", clutter_level="C0", semantics=bad)


# ---------------------------------------------------------------- contract
def _environment(**overrides):
    values = dict(
        clutter="C0", awareness="map_local", builder=CANONICAL, config={}, config_sha256="a" * 64, dt_s=0.5,
    )
    values.update(overrides)
    return EnvironmentSpec(**values)


def test_legacy_environment_without_semantics_fields_is_v1():
    env = _environment()
    assert env.semantics == "V1"


def test_v1_and_v2_environment_specs_validate():
    assert _environment(
        config={"mission_contact_mode": "endpoint", "enemy_nominal_guidance_mode": "legacy"},
    ).semantics == "V1"
    env = _environment(
        builder=V2_BUILDER, semantics="V2",
        config={"mission_contact_mode": "swept", "enemy_nominal_guidance_mode": "direct_obstacle_only"},
    )
    assert env.semantics == "V2"


@pytest.mark.parametrize("kwargs", [
    # label says V2 but the resolved config is legacy / endpoint
    dict(builder=V2_BUILDER, semantics="V2", config={"mission_contact_mode": "endpoint", "enemy_nominal_guidance_mode": "legacy"}),
    dict(builder=V2_BUILDER, semantics="V2", config={}),
    # only one of the two V2 switches
    dict(builder=V2_BUILDER, semantics="V2", config={"mission_contact_mode": "swept", "enemy_nominal_guidance_mode": "legacy"}),
    # label says V1 but the config is swept/direct
    dict(semantics="V1", config={"mission_contact_mode": "swept", "enemy_nominal_guidance_mode": "direct_obstacle_only"}),
    # builder disagrees with the label
    dict(builder=CANONICAL, semantics="V2", config={"mission_contact_mode": "swept", "enemy_nominal_guidance_mode": "direct_obstacle_only"}),
    dict(builder=V2_BUILDER, semantics="V1", config={}),
])
def test_semantics_label_can_never_disagree_with_the_resolved_config(kwargs):
    with pytest.raises(ValidationError):
        _environment(**kwargs)


# ------------------------------------------------------------ config wiring
def _fake_modules(canonical=V1_SWITCHES, v2=V2_SWITCHES):
    calls = []

    def builder(name, switches):
        def build(level, estimator, *, awareness_mode, **extra):
            assert not extra, "the visualizer must not pass extra overrides"
            calls.append((name, level, estimator, awareness_mode))
            return SimpleNamespace(mission_contact_mode=switches[0], enemy_nominal_guidance_mode=switches[1])
        return build

    modules = {
        "experiments.journal_clutter_protocol": SimpleNamespace(build_canonical_env_config=builder("canonical", canonical)),
        "experiments.journal_v2_protocol": SimpleNamespace(build_v2_env_config=builder("v2", v2)),
    }
    return modules, calls


def test_v1_uses_only_the_original_canonical_builder_with_the_original_arguments():
    modules, calls = _fake_modules()
    config, builder = engine.build_replay_config(modules, "V1", "C2", "kalman", "local")
    assert calls == [("canonical", "C2", "kalman", "local")]
    assert builder == CANONICAL
    assert (config.mission_contact_mode, config.enemy_nominal_guidance_mode) == V1_SWITCHES


def test_v2_uses_only_the_research_v2_builder_and_preserves_selection():
    modules, calls = _fake_modules()
    config, builder = engine.build_replay_config(modules, "V2", "C3", "measurement", "map_local")
    assert calls == [("v2", "C3", "measurement", "map_local")]
    assert builder == V2_BUILDER
    assert (config.mission_contact_mode, config.enemy_nominal_guidance_mode) == V2_SWITCHES


@pytest.mark.parametrize("semantics,kwargs", [
    ("V2", dict(v2=V1_SWITCHES)),               # V2 builder resolved to legacy
    ("V2", dict(v2=("swept", "legacy"))),       # half-applied
    ("V1", dict(canonical=V2_SWITCHES)),        # V1 builder drifted to V2
])
def test_a_config_that_does_not_resolve_to_the_requested_semantics_is_rejected(semantics, kwargs):
    modules, _calls = _fake_modules(**kwargs)
    with pytest.raises(ApplicationError) as error:
        engine.build_replay_config(modules, semantics, "C0", "measurement", "map_local")
    assert error.value.code == "SEMANTICS_CONFIG_MISMATCH"


def test_the_simulator_commit_pin_is_the_audited_v2_research_head():
    assert engine.EXPECTED_RESEARCH_COMMIT == "408c4a3f1e554c827a3e50fb9a40875c3fd1b7f2"


# ----------------------------------------------------------------- events
def test_swept_contact_fraction_is_recorded_only_when_the_simulator_reports_it():
    swept = {"mission_contact_mode": "swept", "terminal_event": "capture", "terminal_event_fraction": 0.42,
             "soldier_pos": (1.0, 2.0, 0.0), "enemy_pos": (1.0, 2.0, 0.0)}
    (event,) = engine._events_for_step(5, swept, "soldier_caught", 0.5)
    assert event.kind == "asset_reached" and event.contact_fraction == pytest.approx(0.42)
    assert "contact_fraction" not in event.unavailable_fields

    endpoint = {"mission_contact_mode": "endpoint", "soldier_pos": (1.0, 2.0, 0.0), "enemy_pos": (1.0, 2.0, 0.0)}
    (legacy,) = engine._events_for_step(5, endpoint, "soldier_caught", 0.5)
    assert legacy.contact_fraction is None and "contact_fraction" in legacy.unavailable_fields

    wrong_event = dict(swept, terminal_event="interception")
    (mismatch,) = engine._events_for_step(5, wrong_event, "soldier_caught", 0.5)
    assert mismatch.contact_fraction is None


# --------------------------------------------------------------------- CLI
def test_cli_defaults_to_v1_and_accepts_v2():
    parser = replay_cli._parser()
    base = ["--defender", "greedy", "--clutter", "C0", "--evaluation-seed", "130000"]
    assert parser.parse_args(base).semantics == "V1"
    assert parser.parse_args(base + ["--semantics", "V2"]).semantics == "V2"
    with pytest.raises(SystemExit):
        parser.parse_args(base + ["--semantics", "v3"])
