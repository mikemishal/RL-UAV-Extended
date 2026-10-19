"""Deterministic micro-tests for the journal V2 corrected semantics.

Scenarios place entities directly (as the existing obstacle tests do) so each
test isolates one behavior: swept contact, event ordering, corrected A0
guidance, obstacle-triggered deviation, and V1 compatibility.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv

from experiments.journal_v2_protocol import (
    CANONICAL_C0_OVERRIDES,
    V1_SEMANTIC_FIELDS,
    V2_SEMANTIC_FIELDS,
    assert_v2_semantics,
    build_v1_env_config,
    build_v2_env_config,
)
from experiments.journal_v2_trace import run_trace_episode, trace_row, write_trace_jsonl

FAR_DEFENDER = (45.0, 45.0, 25.0)
V2 = dict(V2_SEMANTIC_FIELDS)
V1 = dict(V1_SEMANTIC_FIELDS)


def _cfg(semantics: dict, **kw) -> EnvConfig:
    params = dict(defender_standby_until_detection=False)
    params.update(semantics)
    params.update(kw)
    return EnvConfig(**params)


def _scenario(cfg, enemy_pos, enemy_vel=(0.0, 0.0, 0.0), asset_pos=(0.0, 0.0, 0.0), defender_pos=FAR_DEFENDER,
              asset_step=None, direction=None, seed=0):
    """Reset, then place entities. `asset_step` (dx, dy, dz) makes the asset
    move deterministically; otherwise it is stationary. `direction` forces
    the attacker's commanded direction via the tactical override."""
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


# ---------------------------------------------------------------- Test A
def test_a_stationary_asset_tunneling_legacy_misses_swept_captures():
    kwargs = dict(enemy_pos=(-3.0, 0.0, 0.0), enemy_vel=(12.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    legacy = _scenario(_cfg(V1), **kwargs)
    _, _, term, _, info = _step(legacy)
    assert not term and info["outcome"] == "ongoing"
    assert info["enemy_pos"][0] == pytest.approx(3.0, abs=1e-4)
    assert info["enemy_soldier_dist"] > legacy.config.threat_radius  # endpoints both outside

    swept = _scenario(_cfg(V2), **kwargs)
    _, _, term, _, info = _step(swept)
    assert term and info["outcome"] == "soldier_caught"
    assert info["terminal_event"] == "capture"
    assert info["capture_contact_fraction"] == pytest.approx(1.0 / 6.0, abs=1e-6)
    assert np.allclose(info["capture_contact_position"], [-2.0, 0.0, 0.0], atol=1e-4)
    assert info["capture_contact_distance"] == pytest.approx(2.0, abs=1e-6)
    assert info["enemy_soldier_dist"] == pytest.approx(2.0, abs=1e-4)
    assert info["mission_contact_mode"] == "swept"


# ---------------------------------------------------------------- Test B
def test_b_moving_asset_crossing():
    kwargs = dict(enemy_pos=(-3.0, 2.5, 0.0), enemy_vel=(12.0, 0.0, 0.0), asset_step=(0.0, 1.5, 0.0),
                  direction=(1.0, 0.0, 0.0))
    legacy = _scenario(_cfg(V1), **kwargs)
    _, _, term, _, info = _step(legacy)
    assert not term and info["outcome"] == "ongoing"

    swept = _scenario(_cfg(V2), **kwargs)
    _, _, term, _, info = _step(swept)
    assert term and info["outcome"] == "soldier_caught" and info["terminal_event"] == "capture"
    assert 0.0 < info["capture_contact_fraction"] < 1.0
    assert info["capture_contact_distance"] == pytest.approx(2.0, abs=1e-6)


# ---------------------------------------------------------------- Test C
def test_c_defender_interception_tunneling():
    kwargs = dict(enemy_pos=(-20.0, 0.0, 10.0), enemy_vel=(12.0, 0.0, 0.0), defender_pos=(-17.0, 0.0, 10.0),
                  direction=(1.0, 0.0, 0.0))
    legacy = _scenario(_cfg(V1), **kwargs)
    _, _, term, _, info = _step(legacy)
    assert not term and info["outcome"] == "ongoing"
    assert info["defender_enemy_dist"] > legacy.config.intercept_radius

    swept = _scenario(_cfg(V2), **kwargs)
    _, _, term, _, info = _step(swept)
    assert term and info["outcome"] == "intercepted"
    assert info["terminal_event"] == "interception"
    assert info["interception_contact_fraction"] == pytest.approx(0.5 / 6.0, abs=1e-6)
    assert info["interception_contact_distance"] == pytest.approx(2.5, abs=1e-6)


# ---------------------------------------------------------------- Test D
def _competing(defender_x):
    return _scenario(
        _cfg(V2), enemy_pos=(-3.2, 0.0, 0.0), enemy_vel=(12.0, 0.0, 0.0),
        defender_pos=(defender_x, 0.0, 0.0), direction=(1.0, 0.0, 0.0),
    )


def test_d_defender_interception_earlier_than_capture_wins():
    # Interception sphere entered at x=-2.6 (tau=0.1); asset sphere at x=-2.0 (tau=0.2).
    _, _, term, _, info = _step(_competing(-0.1))
    assert term
    assert info["terminal_event"] == "interception"
    assert info["interception_contact_fraction"] < info["capture_contact_fraction"]
    assert info["outcome"] == "unsafe_intercept"  # still classified by the existing unsafe radius
    assert info["outcome"] != "soldier_caught"


def test_d_capture_earlier_than_interception_wins():
    # Interception sphere entered at x=-1.0 (tau~0.37), after capture at tau=0.2.
    _, _, term, _, info = _step(_competing(1.5))
    assert term
    assert info["terminal_event"] == "capture"
    assert info["outcome"] == "soldier_caught"
    assert info["capture_contact_fraction"] < info["interception_contact_fraction"]


def test_d_simultaneous_events_use_documented_tie_rule_capture_wins():
    # Both spheres first entered at x=-2.0.
    _, _, term, _, info = _step(_competing(0.5))
    assert term
    assert info["capture_contact_fraction"] == pytest.approx(info["interception_contact_fraction"], abs=1e-9)
    assert info["terminal_event"] == "capture" and info["outcome"] == "soldier_caught"


# ------------------------------------------------- Phase 8: collisions vs mission
def _wall_cfg(x_center):
    return _cfg(
        V2, obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((x_center, 0.0, 5.0, 0.3, 5.0, 5.0),),
    )


def test_capture_before_later_wall_collision_is_not_overridden():
    env = _scenario(_wall_cfg(2.5), enemy_pos=(-3.2, 0.0, 0.0), enemy_vel=(12.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    _, _, term, _, info = _step(env)
    assert term and info["outcome"] == "soldier_caught" and info["terminal_event"] == "capture"
    assert info["enemy_obstacle_collision"] is False
    assert info["enemy_collision_event_type"] is None
    assert env._enemy_collision_count == 0  # rolled back, never counted


def test_earlier_wall_collision_beats_later_capture():
    env = _scenario(_wall_cfg(-2.5), enemy_pos=(-3.6, 0.0, 0.0), enemy_vel=(12.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    _, _, term, _, info = _step(env)
    assert term and info["outcome"] == "enemy_static_obstacle_collision"
    assert info["terminal_event"] == "enemy_collision"
    assert env._enemy_collision_count == 1


def test_legacy_collision_precedence_is_unchanged_even_if_capture_was_earlier():
    env = _scenario(_cfg(V1, obstacles_enabled=True, obstacle_layout_mode="fixed",
                         obstacle_fixed_spec=((2.5, 0.0, 5.0, 0.3, 5.0, 5.0),)),
                    enemy_pos=(-3.2, 0.0, 0.0), enemy_vel=(12.0, 0.0, 0.0), direction=(1.0, 0.0, 0.0))
    _, _, term, _, info = _step(env)
    assert term and info["outcome"] == "enemy_static_obstacle_collision"


def test_interception_before_later_wall_collision_is_not_overridden():
    env = _scenario(
        _cfg(V2, obstacles_enabled=True, obstacle_layout_mode="fixed",
             obstacle_fixed_spec=((-18.5, 0.0, 10.0, 0.3, 5.0, 10.0),)),
        enemy_pos=(-20.0, 0.0, 10.0), enemy_vel=(12.0, 0.0, 0.0), defender_pos=(-17.0, 0.0, 10.0),
        direction=(1.0, 0.0, 0.0),
    )
    _, _, term, _, info = _step(env)
    # Interception sphere is entered at x=-19.5 (tau 0.083); the attacker would
    # only reach the wall face at x=-18.8 (tau 0.2). The earlier event wins and
    # the later collision is rolled back.
    assert term and info["terminal_event"] == "interception" and info["outcome"] == "intercepted"
    assert info["enemy_obstacle_collision"] is False
    assert env._enemy_collision_count == 0


# ---------------------------------------------------------------- Test E
def _clear_space_run(cfg, enemy_start, defender_pos=FAR_DEFENDER, seed=0, max_steps=400):
    env = _scenario(cfg, enemy_pos=enemy_start, defender_pos=defender_pos, seed=seed)
    infos = []
    for _ in range(max_steps):
        _, _, term, trunc, info = _step(env)
        infos.append(info)
        if term or trunc:
            break
    return infos


def test_e_clear_space_a0_pursues_directly_without_weave():
    infos = _clear_space_run(_cfg(V2), (-40.0, 20.0, 15.0))
    assert infos[-1]["outcome"] == "soldier_caught"
    assert infos[-1]["terminal_event"] == "capture"
    ranges = [i["enemy_soldier_dist"] for i in infos]
    assert all(b <= a + 1e-6 for a, b in zip(ranges, ranges[1:])), "range must close monotonically"
    for info in infos:
        assert info["weave_bias"] == 0.0
        assert info["enemy_behavior_mode"] == "MISSION"
        assert np.allclose(info["enemy_final_command"], info["enemy_mission_command"], atol=1e-6)
        assert np.allclose(info["enemy_evasion_command"], 0.0)
    assert infos[-1]["attacker_overshoot_count"] == 0
    assert infos[-1]["attacker_heading_change_total_deg"] < 5.0
    assert infos[-1]["attacker_path_efficiency"] > 0.97


def test_e_a0_ignores_the_defender_and_uses_no_randomness():
    base = _clear_space_run(_cfg(V2), (-40.0, 20.0, 15.0), seed=0)
    other_seed = _clear_space_run(_cfg(V2), (-40.0, 20.0, 15.0), seed=123)
    near_defender = _clear_space_run(_cfg(V2), (-40.0, 20.0, 15.0), defender_pos=(-20.0, 32.0, 15.0), seed=0)
    n = min(len(base), len(other_seed), len(near_defender)) - 1
    assert n >= 8
    for k in range(n):
        assert np.allclose(base[k]["enemy_pos"], other_seed[k]["enemy_pos"])
        assert np.allclose(base[k]["enemy_pos"], near_defender[k]["enemy_pos"])


def test_legacy_a0_still_weaves_and_depends_on_seed():
    a = _clear_space_run(_cfg(V1), (-40.0, 20.0, 15.0), seed=0, max_steps=30)
    b = _clear_space_run(_cfg(V1), (-40.0, 20.0, 15.0), seed=123, max_steps=30)
    assert any(abs(i["weave_bias"]) > 0.0 for i in a)
    assert not np.allclose(a[-1]["enemy_pos"], b[-1]["enemy_pos"])


# ------------------------------------------------------------- Tests F/G
def _obstacle_cfg(semantics, center, **kw):
    return _cfg(
        semantics, obstacles_enabled=True, obstacle_layout_mode="fixed",
        obstacle_fixed_spec=((center[0], center[1], 5.0, 4.0, 4.0, 5.0),),
        enemy_obstacle_avoidance_enabled=True, **kw,
    )


def _obstacle_run(cfg, max_steps=250):
    env = _scenario(cfg, enemy_pos=(-20.0, 0.0, 3.0), asset_pos=(45.0, 0.0, 0.0))
    infos = []
    for _ in range(max_steps):
        _, _, term, trunc, info = _step(env)
        infos.append(info)
        if term or trunc:
            break
    return infos


@pytest.mark.parametrize("route_planning", [False, True])
def test_f_single_blocking_obstacle_deviates_then_returns_to_direct(route_planning):
    infos = _obstacle_run(_obstacle_cfg(V2, (15.0, 0.0), enemy_route_planning_enabled=route_planning))
    assert infos[-1]["outcome"] == "soldier_caught", infos[-1]["outcome"]
    assert not any(i["enemy_obstacle_collision"] for i in infos)
    deviated = [i for i, info in enumerate(infos) if info["enemy_behavior_mode"] != "MISSION"]
    assert deviated, "the blocking obstacle must trigger a deviation"
    last_deviation = max(deviated)
    tail = infos[last_deviation + 1:]
    assert len(tail) >= 5, "A0 must resume direct pursuit well before reaching the asset"
    for info in tail:
        assert info["enemy_behavior_mode"] == "MISSION"
        assert info["enemy_obstacle_avoidance_active"] is False
        assert np.allclose(info["enemy_final_command"], info["enemy_mission_command"], atol=1e-6)
    # No chatter: a single contiguous deviation episode.
    assert deviated == list(range(deviated[0], deviated[-1] + 1))
    # Detour was real but bounded.
    lateral = max(abs(i["enemy_pos"][1]) for i in infos) + max(abs(i["enemy_pos"][2] - 3.0) for i in infos) * 0
    assert lateral > 0.5 or max(i["enemy_pos"][2] for i in infos) > 10.0


@pytest.mark.parametrize("route_planning", [False, True])
def test_g_nearby_obstacle_that_does_not_block_causes_no_deviation(route_planning):
    infos = _obstacle_run(_obstacle_cfg(V2, (15.0, 14.0), enemy_route_planning_enabled=route_planning))
    assert infos[-1]["outcome"] == "soldier_caught"
    for info in infos:
        assert info["enemy_behavior_mode"] == "MISSION"
        assert info["enemy_obstacle_avoidance_active"] is False
        assert info["enemy_route_active"] is False
        assert np.allclose(info["enemy_final_command"], info["enemy_mission_command"], atol=1e-6)
    assert max(abs(i["enemy_pos"][1]) for i in infos) < 1e-3  # straight line, no lateral drift


def test_route_planner_is_discarded_once_direct_path_is_clear():
    env = _scenario(_obstacle_cfg(V2, (15.0, 0.0), enemy_route_planning_enabled=True),
                    enemy_pos=(-20.0, 0.0, 3.0), asset_pos=(45.0, 0.0, 0.0))
    saw_route = False
    for _ in range(250):
        _, _, term, trunc, info = _step(env)
        saw_route = saw_route or info["enemy_route_active"]
        if term or trunc:
            break
    assert saw_route, "the planner is used while the direct path is obstructed"
    assert env._enemy_route_planner._route is None, "no stale route survives after clearing"


# ---------------------------------------------------------------- Test H
# Golden full-trajectory hashes recorded from the FROZEN V1 code
# (commit e72d421): sha256 over enemy/defender positions and rewards of every
# step of a greedy-defender episode at the canonical C0/C2/C3 settings.
V1_GOLDEN = {
    ("C0", 130000): ("intercepted", 402, "170db415bcedfffc"),
    ("C0", 130003): ("intercepted", 49, "70f48f9d4bf3dd45"),
    ("C0", 130007): ("soldier_caught", 473, "a75db9ae0a8c918e"),
    ("C2", 130000): ("intercepted", 237, "6058eae42f56c0f2"),
    ("C2", 130003): ("intercepted", 63, "e6963d273276f4c1"),
    ("C2", 130007): ("intercepted", 270, "f6615fa8f3e6efda"),
    ("C3", 130000): ("intercepted", 482, "0e490dfc6c822dcb"),
    ("C3", 130003): ("intercepted", 365, "d600ad9a37d55d3b"),
    ("C3", 130007): ("intercepted", 244, "a49f7c64e4efed3a"),
}


def _greedy_episode_hash(level, seed, build):
    import hashlib

    from uav_defend.policies.registry import get_policy
    from uav_defend.policies.sanitize import build_policy_info

    cfg = build(level)
    env = SoldierEnv(cfg)
    env.enemy_tactical_override = None
    policy = get_policy("greedy")
    env.reset(seed=seed)
    policy.reset()
    digest = hashlib.sha256()
    done, steps = False, 0
    while not done:
        action = policy.act(env._get_obs(), build_policy_info(env._get_info(), "measurement"))
        _, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        digest.update(np.asarray(info["enemy_pos"], dtype=np.float32).tobytes())
        digest.update(np.asarray(info["defender_pos"], dtype=np.float32).tobytes())
        digest.update(np.float64(reward).tobytes())
    return info["outcome"], steps, digest.hexdigest()[:16]


def _v1_default_builder(level):
    from experiments.journal_clutter_protocol import build_canonical_env_config

    return build_canonical_env_config(
        level, "measurement", extra_overrides=dict(CANONICAL_C0_OVERRIDES) if level == "C0" else None,
    )


@pytest.mark.parametrize("key", sorted(V1_GOLDEN))
def test_h_legacy_default_config_reproduces_frozen_v1_trajectories(key):
    level, seed = key
    assert _greedy_episode_hash(level, seed, _v1_default_builder) == V1_GOLDEN[key]


@pytest.mark.parametrize("key", [("C0", 130000), ("C2", 130003)])
def test_h_explicit_v1_builder_matches_frozen_v1(key):
    level, seed = key
    assert _greedy_episode_hash(level, seed, lambda lv: build_v1_env_config(lv, "measurement")) == V1_GOLDEN[key]


def test_default_envconfig_is_v1():
    cfg = EnvConfig()
    assert cfg.mission_contact_mode == "endpoint" and cfg.enemy_nominal_guidance_mode == "legacy"


# ------------------------------------------------- config / protocol / trace
def test_invalid_mode_values_rejected():
    with pytest.raises(ValueError):
        EnvConfig(mission_contact_mode="continuous")
    with pytest.raises(ValueError):
        EnvConfig(enemy_nominal_guidance_mode="weaving")


def test_v2_builder_is_explicit_and_c0_matches_canonical_overrides():
    from experiments.defender_adaptation_canonical_c0_evaluation import CANONICAL_EXTRA_OVERRIDES

    assert CANONICAL_C0_OVERRIDES == CANONICAL_EXTRA_OVERRIDES
    for level in ("C0", "C2", "C3"):
        cfg = build_v2_env_config(level, "kalman")
        assert cfg.mission_contact_mode == "swept"
        assert cfg.enemy_nominal_guidance_mode == "direct_obstacle_only"
        assert cfg.defender_standby_until_detection is True
    c0 = build_v2_env_config("C0", "measurement")
    assert c0.obstacles_enabled is False and c0.enemy_obstacle_avoidance_enabled is False
    with pytest.raises(AssertionError):
        assert_v2_semantics(EnvConfig())


def test_trace_rows_have_the_documented_schema_and_serialise(tmp_path):
    cfg = build_v2_env_config("C2", "measurement")
    env = SoldierEnv(cfg)
    rows = run_trace_episode(env, seed=130001, max_steps=60)
    assert rows[0]["step"] == 0 and len(rows) > 5
    required = {
        "step", "time_s", "asset_pos", "attacker_pos", "defender_pos", "attacker_vel",
        "attacker_mission_direction", "attacker_final_command", "behavior_mode", "obstacle_blocking",
        "nearest_obstacle_clearance", "detected", "capture_contact_fraction",
        "interception_contact_fraction", "terminal_event", "outcome",
    }
    assert required <= set(rows[0])
    assert {r["behavior_mode"] for r in rows} <= {"DIRECT", "OBSTACLE_AVOID", "ROUTE_BYPASS", "EVADE"}
    path = tmp_path / "trace.jsonl"
    write_trace_jsonl(rows, path)
    reloaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(reloaded) == len(rows) and reloaded[-1]["step"] == rows[-1]["step"]


def test_efficiency_info_fields_present_in_both_modes():
    for semantics in (V1, V2):
        env = _scenario(_cfg(semantics), enemy_pos=(-40.0, 20.0, 15.0))
        _, _, _, _, info = _step(env)
        for key in ("attacker_path_length", "attacker_flight_time_s", "attacker_heading_change_total_deg",
                    "attacker_initial_range", "attacker_overshoot_count", "terminal_event"):
            assert key in info
