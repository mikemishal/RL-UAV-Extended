"""Canonical V2 training contract guard (config-only; no training, no stepping, no PPO.learn).

The guard lives in experiments/journal_v2_protocol.py and is opt-in on the journal training runners
via `require_canonical_v2=True`. See docs/journal_v2_canonical_benchmark.md.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uav_defend.envs.soldier_env import SoldierEnv

from experiments.journal_clutter_protocol import build_randomized_training_config
from experiments.journal_v2_protocol import (
    CANONICAL_V2_FROZEN_VALUES,
    CanonicalV2ConfigError,
    assert_canonical_v2_env_fn,
    assert_canonical_v2_training_config,
    build_randomized_v2_training_config,
    build_v1_env_config,
    build_v2_env_config,
    canonical_v2_config_sha256,
    canonical_v2_config_violations,
    find_soldier_env,
    guard_canonical_v2_config_fn,
)


# -------------------------------------------------------------- the config assertion
@pytest.mark.parametrize("level", ["C0", "C1", "C2", "C3"])
@pytest.mark.parametrize("estimator", ["measurement", "kalman"])
def test_every_canonical_v2_builder_config_passes(level, estimator):
    assert canonical_v2_config_violations(build_v2_env_config(level, estimator)) == []
    assert_canonical_v2_training_config(build_v2_env_config(level, estimator, awareness_mode="local"))


def test_randomized_v2_training_distribution_passes_and_only_changes_the_semantic_fields():
    for seed in range(8):
        v1 = build_randomized_training_config(np.random.default_rng(seed), "measurement")
        v2 = build_randomized_v2_training_config(np.random.default_rng(seed), "measurement")
        assert_canonical_v2_training_config(v2)
        differing = {f.name for f in dataclasses.fields(v1) if getattr(v1, f.name) != getattr(v2, f.name)}
        assert differing == {"mission_contact_mode", "enemy_nominal_guidance_mode",
                             "enemy_endurance_enabled"}  # k=5 is already the dataclass default


def test_the_frozen_values_are_the_current_defaults_not_a_second_source_of_truth():
    cfg = build_v2_env_config("C3", "measurement")
    for field, value in CANONICAL_V2_FROZEN_VALUES.items():
        assert getattr(cfg, field) == value


@pytest.mark.parametrize("change,needle", [
    (dict(mission_contact_mode="endpoint", enemy_endurance_enabled=False), "mission_contact_mode"),
    (dict(enemy_nominal_guidance_mode="legacy"), "enemy_nominal_guidance_mode"),
    (dict(enemy_endurance_enabled=False), "enemy_endurance_enabled"),
    (dict(enemy_endurance_path_factor=3.0), "enemy_endurance_path_factor"),
    (dict(enemy_endurance_path_factor=10.0), "enemy_endurance_path_factor"),
    (dict(asset_motion_model="gauss_markov_velocity"), "asset_motion_model"),
    (dict(v_e=15.0), "v_e"),
    (dict(enemy_max_turn_rate_deg=90.0), "enemy_max_turn_rate_deg"),
    (dict(threat_radius=3.0), "threat_radius"),
    (dict(intercept_radius=3.0), "intercept_radius"),
    (dict(dt=0.25), "dt"),
    (dict(defender_standby_until_detection=False), "defender_standby_until_detection"),
    (dict(enemy_route_planning_enabled=True), "enemy_route_planning_enabled"),
])
def test_every_departure_from_the_canonical_benchmark_is_reported(change, needle):
    cfg = dataclasses.replace(build_v2_env_config("C3", "measurement"), **change)
    problems = canonical_v2_config_violations(cfg)
    assert any(needle in p for p in problems), problems
    with pytest.raises(CanonicalV2ConfigError, match=needle):
        assert_canonical_v2_training_config(cfg)


def test_v1_config_is_rejected_with_all_reasons_listed():
    problems = canonical_v2_config_violations(build_v1_env_config("C3", "measurement"))
    assert any("mission_contact_mode" in p for p in problems)
    assert any("enemy_nominal_guidance_mode" in p for p in problems)
    assert any("enemy_endurance_enabled" in p for p in problems)


def test_legacy_waypoint_avoidance_switch_is_rejected(monkeypatch):
    cfg = build_v2_env_config("C3", "measurement")
    assert_canonical_v2_training_config(cfg)
    monkeypatch.setattr(SoldierEnv, "STATEFUL_ENEMY_AVOIDANCE", False)
    with pytest.raises(CanonicalV2ConfigError, match="STATEFUL_ENEMY_AVOIDANCE"):
        assert_canonical_v2_training_config(cfg)


def test_guard_is_config_only_and_sha_is_stable():
    cfg = build_v2_env_config("C3", "measurement", awareness_mode="map_local")
    before = canonical_v2_config_sha256(cfg)
    assert_canonical_v2_training_config(cfg)
    assert canonical_v2_config_sha256(cfg) == before == canonical_v2_config_sha256(
        build_v2_env_config("C3", "measurement", awareness_mode="map_local"))
    assert before != canonical_v2_config_sha256(build_v2_env_config("C2", "measurement", awareness_mode="map_local"))


def test_config_fn_guard_wraps_any_rng_factory():
    guarded_v2 = guard_canonical_v2_config_fn(lambda rng: build_randomized_v2_training_config(rng, "kalman"))
    assert_canonical_v2_training_config(guarded_v2(np.random.default_rng(1)))
    guarded_v1 = guard_canonical_v2_config_fn(lambda rng: build_randomized_training_config(rng, "measurement"))
    with pytest.raises(CanonicalV2ConfigError):
        guarded_v1(np.random.default_rng(1))


# -------------------------------------------------- existing training environments / entry points
def test_existing_defender_clutter_env_silently_resamples_v1_so_the_env_fn_guard_catches_it():
    """RandomizedClutterWrapper.reset() replaces the config with the V1 randomized one, so a V2
    config handed to the wrapper would be silently discarded. The env_fn guard resets first."""
    from experiments.clutter_training_env import build_ppo_clutter_env

    env = build_ppo_clutter_env("measurement", randomized=True)
    env.reset(seed=3)
    assert find_soldier_env(env).config.mission_contact_mode == "endpoint"
    env.close()
    with pytest.raises(CanonicalV2ConfigError):
        assert_canonical_v2_env_fn(lambda seed: build_ppo_clutter_env("measurement", randomized=True), 3)


def test_fixed_v2_config_passes_the_env_fn_guard_through_the_sector_wrapper():
    from experiments.clutter_training_env import build_ppo_clutter_env

    cfg = build_v2_env_config("C2", "measurement")
    assert_canonical_v2_env_fn(lambda seed: build_ppo_clutter_env("measurement", randomized=False, fixed_config=cfg), 5)
    legacy = build_v1_env_config("C2", "measurement")
    with pytest.raises(CanonicalV2ConfigError):
        assert_canonical_v2_env_fn(lambda seed: build_ppo_clutter_env("measurement", randomized=False, fixed_config=legacy), 5)


def _lead_only_pool():
    from experiments.adaptive_attacker_env import DefenderPoolMember
    from uav_defend.policies.registry import get_policy

    cfg = build_v2_env_config("C2", "measurement")
    return [DefenderPoolMember(family="Lead", seed=None, estimator="measurement", needs_sector_features=False,
                               is_lr_ppo=False, policy=get_policy("lead", config=cfg))]


def test_attacker_training_env_defaults_to_v1_and_is_caught_but_v2_config_fn_passes():
    from experiments.adaptive_attacker_env import AdaptiveAttackerTrainingEnv

    pool = _lead_only_pool()
    with pytest.raises(CanonicalV2ConfigError):
        assert_canonical_v2_env_fn(lambda seed: AdaptiveAttackerTrainingEnv(defender_pool=pool), 7)
    assert_canonical_v2_env_fn(
        lambda seed: AdaptiveAttackerTrainingEnv(
            defender_pool=pool, clutter_config_fn=lambda rng: build_randomized_v2_training_config(rng, "measurement")),
        7,
    )


@pytest.mark.parametrize("module_name,runner_name,kwargs", [
    ("experiments.adaptive_attacker_training_runner", "run_adaptive_attacker_training", {}),
    ("experiments.clutter_training_runner", "run_clutter_training", {"architecture": "ppo_clutter_v1"}),
    ("experiments.defender_adaptation_training_runner", "run_defender_adaptation_training",
     {"architecture": "ppo_clutter_v1", "generation": "d1", "starting_checkpoint_path": "unused.zip",
      "total_additional_timesteps": 1}),
])
def test_training_runners_with_require_canonical_v2_fail_before_any_ppo_or_output(
        module_name, runner_name, kwargs, tmp_path, monkeypatch):
    import importlib

    module = importlib.import_module(module_name)

    def forbidden(*args, **kw):  # a hard safety net: nothing in this test may construct or train a PPO
        raise AssertionError("PPO must not be constructed by the canonical-V2 guard tests")

    monkeypatch.setattr(module, "PPO", forbidden)
    from experiments.clutter_training_env import build_ppo_clutter_env

    out = tmp_path / "run"
    runner = getattr(module, runner_name)
    with pytest.raises(CanonicalV2ConfigError):
        runner(env_fn=lambda seed: build_ppo_clutter_env("measurement", randomized=True), seed=11,
               output_dir=out, total_timesteps=1, require_canonical_v2=True, **kwargs) \
            if runner_name != "run_defender_adaptation_training" else \
            runner(env_fn=lambda seed: build_ppo_clutter_env("measurement", randomized=True), seed=11,
                   output_dir=out, require_canonical_v2=True, **kwargs)
    assert not out.exists()


def test_handoff_manifest_matches_the_live_canonical_config():
    import json
    from pathlib import Path

    manifest = json.loads((Path(__file__).resolve().parents[1] / "docs" / "journal_v2_training_handoff.json").read_text())
    assert "MUST NOT be used as canonical V2 results" in manifest["statement"]
    config = manifest["canonical_config"]
    assert config["enemy_endurance_path_factor"] == 5.0 and config["asset_motion_model"] == "legacy_gaussian_random_walk"
    for key, expected in config["config_sha256_map_local"].items():
        level, estimator = key.split("/")
        live = build_v2_env_config(level, estimator, awareness_mode="map_local")
        assert canonical_v2_config_sha256(live) == expected, key
        assert live.enemy_endurance_path_factor == config["enemy_endurance_path_factor"]
    assert manifest["research"]["superseded_tag_unmoved"]["journal-corrected-attacker-semantics-v2"].startswith("db8c299")


def test_runners_default_remains_unguarded_so_v1_reproductions_are_unchanged():
    import inspect

    from experiments.adaptive_attacker_training_runner import run_adaptive_attacker_training
    from experiments.clutter_training_runner import run_clutter_training
    from experiments.defender_adaptation_training_runner import run_defender_adaptation_training

    for runner in (run_adaptive_attacker_training, run_clutter_training, run_defender_adaptation_training):
        assert inspect.signature(runner).parameters["require_canonical_v2"].default is False
