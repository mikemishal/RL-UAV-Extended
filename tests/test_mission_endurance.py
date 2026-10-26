"""Tests for finite mission endurance (uav_defend/config/env_config.py:enemy_endurance_*,
uav_defend/envs/soldier_env.py's swept-contact arbitration). See
docs/journal_v2_mission_endurance.md for the full investigation.

Scenario helpers are shared with tests/test_corrected_attacker_semantics_v2.py: entities are
placed directly and the attacker's commanded direction is forced via `enemy_tactical_override`.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv

from experiments.journal_v2_protocol import V1_SEMANTIC_FIELDS, V2_SEMANTIC_FIELDS, build_v2_env_config

V2 = dict(V2_SEMANTIC_FIELDS)
V1 = dict(V1_SEMANTIC_FIELDS)
FAR_DEFENDER = (45.0, 45.0, 25.0)


def _cfg(semantics: dict, **kw) -> EnvConfig:
    params = dict(defender_standby_until_detection=False)
    params.update(semantics)
    params.update(kw)
    return EnvConfig(**params)


def _scenario(cfg, enemy_pos, enemy_vel=(0.0, 0.0, 0.0), asset_pos=(0.0, 0.0, 0.0), defender_pos=FAR_DEFENDER,
              asset_step=None, direction=None, seed=0):
    env = SoldierEnv(cfg)
    env.reset(seed=seed)
    env._enemy_pos = np.asarray(enemy_pos, dtype=np.float32)
    env._enemy_vel = np.asarray(enemy_vel, dtype=np.float32)
    env._soldier_pos = np.asarray(asset_pos, dtype=np.float32)
    env._defender_pos = np.asarray(defender_pos, dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    step = np.zeros(3, dtype=np.float32) if asset_step is None else np.asarray(asset_step, dtype=np.float32)

    def _move_asset():
        env._soldier_pos = (env._soldier_pos + step).astype(np.float32)

    env._move_soldier = _move_asset
    if direction is not None:
        d = np.asarray(direction, dtype=np.float32)
        env.enemy_tactical_override = lambda: d
    return env


def _step(env):
    return env.step(np.zeros(3, dtype=np.float32))


_MEASURE_SEMANTICS = {"mission_contact_mode": "endpoint", "enemy_nominal_guidance_mode": "direct_obstacle_only"}


def _full_step_endpoint(enemy_pos, enemy_vel, direction, defender_pos=FAR_DEFENDER):
    """The TRUE (uninterpolated) one-step attacker end position for a given scenario, measured
    under endpoint contact mode (never clamps the position mid-step) with V2's guidance mode (so
    an unoverridden `direction=None` pursuit command matches what the V2 scenario under test will
    actually fly)."""
    env = _scenario(_cfg(_MEASURE_SEMANTICS), enemy_pos=enemy_pos, enemy_vel=enemy_vel, direction=direction,
                     defender_pos=defender_pos)
    _step(env)
    return env._enemy_pos.astype(np.float64).copy()


def _full_step_distance(enemy_pos, enemy_vel, direction, defender_pos=FAR_DEFENDER):
    """The TRUE (uninterpolated) one-step attacker travel distance for a given scenario, so it
    can be reused to design exact endurance budgets for the matching V2 scenario."""
    start = np.asarray(enemy_pos, dtype=np.float64)
    end = _full_step_endpoint(enemy_pos, enemy_vel, direction, defender_pos=defender_pos)
    return float(np.linalg.norm(end - start))


def _with_budget(env, budget):
    """Overrides the auto-computed (k * d0) budget from reset() with an exact test value."""
    env._enemy_endurance_budget = float(budget)
    env._enemy_endurance_traveled = 0.0
    return env


# --------------------------------------------------------------- 1. disabled is bit-identical
def test_disabled_endurance_is_bit_identical_to_a_config_without_the_field_touched():
    cfg_plain = _cfg(V2)
    cfg_explicit_disabled = _cfg(V2, enemy_endurance_enabled=False)
    env_a = _scenario(cfg_plain, enemy_pos=(-40.0, 20.0, 15.0), defender_pos=FAR_DEFENDER)
    env_b = _scenario(cfg_explicit_disabled, enemy_pos=(-40.0, 20.0, 15.0), defender_pos=FAR_DEFENDER)
    for _ in range(40):
        _, _, term_a, _, info_a = _step(env_a)
        _, _, term_b, _, info_b = _step(env_b)
        assert np.array_equal(env_a._enemy_pos, env_b._enemy_pos)
        assert info_a["outcome"] == info_b["outcome"]
        assert info_b["enemy_endurance_enabled"] is False
        assert info_b["enemy_endurance_budget_m"] is None
        if term_a or term_b:
            break


def test_enabled_with_a_never_binding_budget_does_not_change_the_trajectory_prefix():
    """A huge k never exhausts, so the attacker must fly IDENTICALLY to the disabled case for
    the whole episode (Phase 14's bit-identical-prefix requirement, degenerate full-episode
    case)."""
    env_disabled = _scenario(_cfg(V2), enemy_pos=(-40.0, 20.0, 15.0))
    env_enabled = _scenario(_cfg(V2, enemy_endurance_enabled=True, enemy_endurance_path_factor=1000.0),
                             enemy_pos=(-40.0, 20.0, 15.0))
    for _ in range(80):
        _, _, term_a, _, info_a = _step(env_disabled)
        _, _, term_b, _, info_b = _step(env_enabled)
        assert np.array_equal(env_disabled._enemy_pos, env_enabled._enemy_pos)
        assert info_a["outcome"] == (info_b["outcome"] if info_b["outcome"] != "ongoing" else info_a["outcome"])
        if term_a:
            assert term_b and info_a["outcome"] == info_b["outcome"]
            break


# --------------------------------------------------------------- 2. path accumulation
def test_traveled_distance_accumulates_the_actual_integrated_path_not_nominal_speed():
    env = _scenario(_cfg(V2, enemy_endurance_enabled=True, enemy_endurance_path_factor=1000.0),
                     enemy_pos=(-40.0, 20.0, 15.0))
    prev_pos = env._enemy_pos.copy().astype(np.float64)
    traveled = 0.0
    for _ in range(15):
        _, _, term, _, info = _step(env)
        traveled += float(np.linalg.norm(env._enemy_pos.astype(np.float64) - prev_pos))
        prev_pos = env._enemy_pos.copy().astype(np.float64)
        assert info["enemy_endurance_traveled_m"] == pytest.approx(traveled, abs=1e-4)
        if term:
            break


# --------------------------------------------------------------- 3. normalized budget init
def test_budget_equals_k_times_initial_attacker_asset_distance():
    for k in (1.0, 3.0, 5.0):
        cfg = _cfg(V2, enemy_endurance_enabled=True, enemy_endurance_path_factor=k)
        env = SoldierEnv(cfg)
        env.reset(seed=123)
        d0 = float(np.linalg.norm(env._soldier_pos.astype(np.float64) - env._enemy_pos.astype(np.float64)))
        assert env._enemy_endurance_budget == pytest.approx(k * d0, rel=1e-6)


def test_budget_is_none_when_disabled():
    env = SoldierEnv(_cfg(V2))
    env.reset(seed=123)
    assert env._enemy_endurance_budget is None
    assert env._enemy_endurance_traveled == 0.0


# --------------------------------------------------------------- 4. exact exhaustion
def test_exact_exhaustion_terminates_with_the_documented_outcome_and_reward():
    kwargs = dict(enemy_pos=(-40.0, 20.0, 15.0), enemy_vel=(0.0, 0.0, 0.0), direction=None)
    full_step = _full_step_distance(**kwargs)
    env = _scenario(_cfg(V2, enemy_endurance_enabled=True, enemy_endurance_path_factor=5.0), **kwargs)
    start = env._enemy_pos.astype(np.float64).copy()
    _with_budget(env, budget=full_step * 0.4)  # exhausts partway through this step
    _, reward, term, _, info = _step(env)
    assert term
    assert info["terminal_event"] == "endurance"
    assert info["outcome"] == "enemy_endurance_exhausted"
    assert reward == pytest.approx(env.config.reward_enemy_endurance_exhausted)
    assert info["enemy_endurance_remaining_m"] == pytest.approx(0.0, abs=1e-6)
    assert info["enemy_endurance_traveled_m"] == pytest.approx(full_step * 0.4, abs=1e-4)
    # Position is clamped to the exhaustion instant (40% of the way through this step's
    # displacement), not the full step.
    actual_full_step_end = _full_step_endpoint(**kwargs)
    expected_pos = start + 0.4 * (actual_full_step_end - start)
    assert np.allclose(env._enemy_pos.astype(np.float64), expected_pos, atol=1e-3)


def test_outcome_is_not_collision_capture_soldier_caught_or_timeout():
    kwargs = dict(enemy_pos=(-40.0, 20.0, 15.0),)
    env = _scenario(_cfg(V2, enemy_endurance_enabled=True, enemy_endurance_path_factor=5.0), **kwargs)
    full_step = _full_step_distance(enemy_pos=(-40.0, 20.0, 15.0), enemy_vel=(0.0, 0.0, 0.0), direction=None)
    _with_budget(env, budget=full_step * 0.5)
    _, _, term, _, info = _step(env)
    assert term and info["outcome"] == "enemy_endurance_exhausted"
    for other in ("soldier_caught", "timeout", "intercepted", "unsafe_intercept",
                  "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"):
        assert info["outcome"] != other


# --------------------------------------------------------------- 5. within-step tau
def test_endurance_tau_formula_matches_remaining_over_full_step_distance():
    kwargs = dict(enemy_pos=(-40.0, 20.0, 15.0), enemy_vel=(0.0, 0.0, 0.0), direction=None)
    full_step = _full_step_distance(**kwargs)
    for frac in (0.1, 0.5, 0.9):
        env = _scenario(_cfg(V2, enemy_endurance_enabled=True, enemy_endurance_path_factor=5.0), **kwargs)
        _with_budget(env, budget=full_step * frac)
        _, _, term, _, info = _step(env)
        assert term and info["terminal_event"] == "endurance"
        assert info["enemy_endurance_traveled_m"] == pytest.approx(full_step * frac, abs=1e-4)


def test_endurance_tau_is_bounded_to_unit_interval_even_with_a_negative_remaining_budget():
    kwargs = dict(enemy_pos=(-40.0, 20.0, 15.0),)
    env = _scenario(_cfg(V2, enemy_endurance_enabled=True, enemy_endurance_path_factor=5.0), **kwargs)
    _with_budget(env, budget=-5.0)  # already over-budget before the step starts
    _, _, term, _, info = _step(env)
    assert term and info["terminal_event"] == "endurance"
    assert info["enemy_endurance_traveled_m"] == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------- 6/7. capture vs exhaustion
def test_capture_before_exhaustion_wins():
    # Same geometry as test_corrected_attacker_semantics_v2's capture test: capture at
    # fraction 1/6 (distance 1.0 of a 6.0 m full step).
    kwargs = dict(enemy_pos=(-3.0, 0.0, 0.0), enemy_vel=(12.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    full_step = _full_step_distance(**kwargs)
    env = _scenario(_cfg(V2, enemy_endurance_enabled=True, enemy_endurance_path_factor=5.0), **kwargs)
    _with_budget(env, budget=full_step * 0.9)  # endurance_tau=0.9 > capture_tau~0.167
    _, _, term, _, info = _step(env)
    assert term and info["terminal_event"] == "capture" and info["outcome"] == "soldier_caught"


def test_exhaustion_before_capture_wins():
    kwargs = dict(enemy_pos=(-3.0, 0.0, 0.0), enemy_vel=(12.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    full_step = _full_step_distance(**kwargs)
    env = _scenario(_cfg(V2, enemy_endurance_enabled=True, enemy_endurance_path_factor=5.0), **kwargs)
    _with_budget(env, budget=full_step * 0.05)  # endurance_tau=0.05 < capture_tau~0.167
    _, _, term, _, info = _step(env)
    assert term and info["terminal_event"] == "endurance" and info["outcome"] == "enemy_endurance_exhausted"


# --------------------------------------------------------------- 8. interception vs exhaustion
def test_interception_before_exhaustion_wins():
    kwargs = dict(enemy_pos=(-20.0, 0.0, 10.0), enemy_vel=(12.0, 0.0, 0.0), defender_pos=(-17.0, 0.0, 10.0),
                  direction=(1.0, 0.0, 0.0))
    full_step = _full_step_distance(**kwargs)
    env = _scenario(_cfg(V2, enemy_endurance_enabled=True, enemy_endurance_path_factor=5.0), **kwargs)
    _with_budget(env, budget=full_step * 0.5)  # interception_tau ~0.083
    _, _, term, _, info = _step(env)
    assert term and info["terminal_event"] == "interception" and info["outcome"] == "intercepted"


def test_exhaustion_before_interception_wins():
    kwargs = dict(enemy_pos=(-20.0, 0.0, 10.0), enemy_vel=(12.0, 0.0, 0.0), defender_pos=(-17.0, 0.0, 10.0),
                  direction=(1.0, 0.0, 0.0))
    full_step = _full_step_distance(**kwargs)
    env = _scenario(_cfg(V2, enemy_endurance_enabled=True, enemy_endurance_path_factor=5.0), **kwargs)
    _with_budget(env, budget=full_step * 0.02)  # endurance_tau=0.02 < interception_tau~0.083
    _, _, term, _, info = _step(env)
    assert term and info["terminal_event"] == "endurance" and info["outcome"] == "enemy_endurance_exhausted"


# --------------------------------------------------------------- 9. collision vs exhaustion
def _wall_cfg(x_center, **kw):
    return _cfg(V2, obstacles_enabled=True, obstacle_layout_mode="fixed",
                obstacle_fixed_spec=((x_center, 0.0, 5.0, 0.3, 5.0, 5.0),), **kw)


def test_collision_before_exhaustion_wins():
    kwargs = dict(enemy_pos=(-3.6, 0.0, 0.0), enemy_vel=(12.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    full_step = _full_step_distance(**kwargs)
    env = _scenario(_wall_cfg(-2.5, enemy_endurance_enabled=True, enemy_endurance_path_factor=5.0), **kwargs)
    _with_budget(env, budget=full_step * 0.95)
    _, _, term, _, info = _step(env)
    assert term and info["terminal_event"] == "enemy_collision"
    assert info["outcome"] == "enemy_static_obstacle_collision"


def test_exhaustion_before_collision_wins():
    kwargs = dict(enemy_pos=(-3.6, 0.0, 0.0), enemy_vel=(12.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    full_step = _full_step_distance(**kwargs)
    env = _scenario(_wall_cfg(-2.5, enemy_endurance_enabled=True, enemy_endurance_path_factor=5.0), **kwargs)
    _with_budget(env, budget=full_step * 0.1)
    _, _, term, _, info = _step(env)
    assert term and info["terminal_event"] == "endurance" and info["outcome"] == "enemy_endurance_exhausted"
    assert info["enemy_obstacle_collision"] is False
    assert env._enemy_collision_count == 0  # the later collision must be rolled back, never counted


# --------------------------------------------------------------- 10. no guidance effect
def test_endurance_never_alters_guidance_before_exhaustion_on_a_clutter_episode():
    """A full randomized clutter episode: with endurance disabled vs enabled-but-never-binding
    (huge k), the attacker trajectory, behavior mode, and commands must be bit-identical every
    step (Phase 15: no guidance code reads remaining endurance)."""
    cfg_disabled = dataclasses.replace(
        build_v2_env_config("C3", "measurement", extra_overrides={"enemy_endurance_enabled": False}),
        defender_standby_until_detection=False)
    cfg_enabled = dataclasses.replace(
        cfg_disabled, enemy_endurance_enabled=True, enemy_endurance_path_factor=1000.0)
    env_a = SoldierEnv(cfg_disabled)
    env_b = SoldierEnv(cfg_enabled)
    env_a.reset(seed=130017)
    env_b.reset(seed=130017)
    assert np.array_equal(env_a._enemy_pos, env_b._enemy_pos)
    for _ in range(100):
        _, _, term_a, _, info_a = env_a.step(np.zeros(3, dtype=np.float32))
        _, _, term_b, _, info_b = env_b.step(np.zeros(3, dtype=np.float32))
        assert np.array_equal(env_a._enemy_pos, env_b._enemy_pos)
        assert np.array_equal(env_a._enemy_vel, env_b._enemy_vel)
        assert info_a["enemy_behavior_mode"] == info_b["enemy_behavior_mode"]
        assert np.allclose(info_a["enemy_final_command"], info_b["enemy_final_command"], atol=1e-7)
        if term_a or term_b:
            assert term_a and term_b and info_a["outcome"] == info_b["outcome"]
            break


# --------------------------------------------------------------- 11. config serialization
def test_config_serializes_and_round_trips():
    cfg = dataclasses.replace(EnvConfig(mission_contact_mode="swept"),
                               enemy_endurance_enabled=True, enemy_endurance_path_factor=4.0)
    payload = dataclasses.asdict(cfg)
    assert payload["enemy_endurance_enabled"] is True
    assert payload["enemy_endurance_path_factor"] == 4.0
    restored = EnvConfig(**payload)
    assert restored == cfg


def test_invalid_path_factor_is_rejected():
    with pytest.raises(ValueError):
        dataclasses.replace(EnvConfig(mission_contact_mode="swept"), enemy_endurance_enabled=True,
                             enemy_endurance_path_factor=0.0)
    with pytest.raises(ValueError):
        dataclasses.replace(EnvConfig(mission_contact_mode="swept"), enemy_endurance_enabled=True,
                             enemy_endurance_path_factor=-1.0)


def test_enabling_endurance_requires_swept_contact_mode():
    with pytest.raises(ValueError):
        dataclasses.replace(EnvConfig(mission_contact_mode="endpoint"), enemy_endurance_enabled=True)


# --------------------------------------------------------------- 12. legacy V1 unaffected
def test_default_config_has_endurance_disabled():
    assert EnvConfig().enemy_endurance_enabled is False


def test_v1_episodes_never_report_endurance_tracking():
    env = _scenario(_cfg(V1), enemy_pos=(-40.0, 20.0, 15.0))
    for _ in range(10):
        _, _, term, _, info = _step(env)
        assert info["enemy_endurance_enabled"] is False
        assert info["enemy_endurance_budget_m"] is None
        assert info["enemy_endurance_traveled_m"] is None
        assert info["enemy_endurance_remaining_m"] is None
        if term:
            break
