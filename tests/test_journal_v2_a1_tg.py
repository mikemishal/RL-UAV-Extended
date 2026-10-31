"""Tests for the finite defender-evasion tactical geometry (`a1_v2_action_v5`, `a1_v2_obs_v2`). No PPO / learn() anywhere."""

from __future__ import annotations

import hashlib
import json
import math
import random

import numpy as np
import pytest

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_a1_tg as G
from experiments import journal_v2_a1_tg_sweep as S
from experiments import journal_v2_a1_tg_state as TS


# ----------------------------------------------------------------------------- geometry
def _angle(u, v):
    return math.degrees(math.acos(float(np.clip(np.dot(u, v), -1, 1))))


@pytest.mark.parametrize("theta", G.THETA_GRID_DEG)
def test_lateral_pulses_are_unit_symmetric_and_have_the_requested_angle(theta):
    m = np.array([0.6, 0.0, 0.8])
    left, right = G.pulse_direction("TACTICAL_LEFT", m, theta), G.pulse_direction("TACTICAL_RIGHT", m, theta)
    assert np.linalg.norm(left) == pytest.approx(1.0) and np.linalg.norm(right) == pytest.approx(1.0)
    assert np.dot(G.lateral_axis(m), m) == pytest.approx(0.0, abs=1e-12)   # l is orthogonal to m, so the deflection is exactly theta
    assert _angle(left, m) == pytest.approx(theta) and _angle(right, m) == pytest.approx(theta)


def test_left_is_counter_clockwise_of_the_horizontal_mission_projection():
    m = np.array([1.0, 0.0, 0.0])
    assert np.allclose(G.lateral_axis(m), [0.0, 1.0, 0.0])
    left, right = G.pulse_direction("TACTICAL_LEFT", m, 20.0), G.pulse_direction("TACTICAL_RIGHT", m, 20.0)
    assert left[1] > 0 > right[1] and left[0] == pytest.approx(right[0]) and left[2] == pytest.approx(0.0) == pytest.approx(right[2])
    assert _angle(left, m) == pytest.approx(20.0)  # purely horizontal geometry: exact angle
    assert np.dot(G.lateral_axis(m), m) == pytest.approx(0.0)


def test_lateral_axis_is_horizontal_and_orthogonal_for_arbitrary_directions():
    rng = np.random.default_rng(1)
    for _ in range(200):
        m = rng.normal(size=3)
        m /= np.linalg.norm(m)
        lateral = G.lateral_axis(m)
        if lateral is None:
            continue
        assert lateral[2] == 0.0 and np.linalg.norm(lateral) == pytest.approx(1.0)
        assert np.dot(lateral, [m[0], m[1], 0.0]) == pytest.approx(0.0, abs=1e-12)


def test_direction_is_recomputed_from_current_m_not_a_fixed_bearing():
    a, b = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    assert np.allclose(G.pulse_direction("TACTICAL_LEFT", b, 20.0), [-math.sin(math.radians(20.0)), math.cos(math.radians(20.0)), 0.0])
    assert not np.allclose(G.pulse_direction("TACTICAL_LEFT", a, 20.0), G.pulse_direction("TACTICAL_LEFT", b, 20.0))


def test_degenerate_horizontal_direction_falls_back_to_base():
    vertical = np.array([0.0, 0.0, 1.0])
    assert G.lateral_axis(vertical) is None
    assert G.pulse_direction("TACTICAL_LEFT", vertical, 20.0) is None and G.pulse_direction("TACTICAL_RIGHT", vertical, 20.0) is None
    assert G.pulse_direction("TACTICAL_LEFT", np.zeros(3), 20.0) is None
    almost = np.array([5e-4, 0.0, 1.0])
    assert G.lateral_axis(almost / np.linalg.norm(almost)) is None


@pytest.mark.parametrize("theta", G.THETA_GRID_DEG)
def test_climb_keeps_forward_mission_progress_and_is_never_pure_vertical(theta):
    for m in ([1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.6, 0.0, -0.8], [0.8, 0.0, 0.6]):
        m = np.asarray(m)
        u = G.pulse_direction("TACTICAL_CLIMB", m, theta)
        assert np.linalg.norm(u) == pytest.approx(1.0) and np.dot(u, m) > 0.8       # forward mission component preserved
        assert u[2] > m[2]
    assert abs(G.pulse_direction("TACTICAL_CLIMB", np.array([1.0, 0, 0]), theta)[2]) < 1.0


def test_ceiling_resolves_climb_to_base_and_headroom_boundary():
    assert G.climb_headroom_ok(10.0, 30.0, 5.0) and not G.climb_headroom_ok(29.0, 30.0, 5.0) and not G.climb_headroom_ok(30.0, 30.0, 5.0)
    assert G.climb_headroom_ok(27.5, 30.0, 5.0) and not G.climb_headroom_ok(27.6, 30.0, 5.0)
    m = G.TacticalMachine(pulse_n=2)
    decision = m.step(G.CLIMB_A, info_available=True, avoid_required=False, geometry_reason=lambda kind: "ceiling")
    assert decision.executed == "BASE" and decision.tactical_kind is None and decision.suppression == "ceiling" and m.mode == "BASE"
    assert m.counters["pulses_started"]["TACTICAL_CLIMB"] == 0   # a ceiling request never even begins a (hover) pulse


# ----------------------------------------------------------------------------- state machine
def _run(machine, requests, *, info=True, avoid=lambda t: False):
    out = []
    for t, request in enumerate(requests):
        d = machine.step(request, info_available=info, avoid_required=avoid(t))
        out.append(d.executed)
    return out


@pytest.mark.parametrize("n", (1, 2, 3, 4))
def test_pulse_is_finite_and_followed_by_exactly_two_rejoin_steps(n):
    modes = _run(G.TacticalMachine(pulse_n=n), [G.LEFT_A] * 20)
    assert modes[:n] == ["TACTICAL_LEFT"] * n and modes[n:n + 2] == ["REJOIN"] * 2 and modes[n + 2] == "TACTICAL_LEFT"
    assert G.check_mode_sequence(modes, n) == []
    run = max(len(list(g)) for k, g in __import__("itertools").groupby(modes) if k == "TACTICAL_LEFT")
    assert run == n


def test_repeated_requests_cannot_extend_or_restart_a_pulse_or_skip_rejoin():
    m = G.TacticalMachine(pulse_n=2)
    modes = _run(m, [1, 1, 1, 2, 3, 1, 2, 3, 1, 1])
    assert modes[:2] == ["TACTICAL_LEFT"] * 2 and modes[2:4] == ["REJOIN"] * 2
    assert m.counters["suppressed"]["rejoin"] >= 2 and "pulse_active" in m.counters["suppressed"]
    assert G.check_mode_sequence(modes, 2) == []


def test_base_requests_in_a_pulse_do_not_cancel_it():
    assert _run(G.TacticalMachine(pulse_n=3), [1, 0, 0, 0, 0, 0]) == ["TACTICAL_LEFT"] * 3 + ["REJOIN"] * 2 + ["BASE"]


def test_no_defender_information_forces_base():
    m = G.TacticalMachine(pulse_n=2)
    assert _run(m, [1, 2, 3], info=False) == ["BASE"] * 3 and m.counters["suppressed"]["no_defender_info"] == 3


def test_obstacle_override_cancels_immediately_then_full_rejoin_then_base():
    m = G.TacticalMachine(pulse_n=4)
    avoid = lambda t: t in (1, 2)
    modes = _run(m, [1] * 4 + [0] * 4, avoid=avoid)
    assert modes[:3] == ["TACTICAL_LEFT", "OBSTACLE_OVERRIDE", "OBSTACLE_OVERRIDE"]
    assert modes[3:5] == ["REJOIN", "REJOIN"] and modes[5] == "BASE"
    assert m.counters["pulses_cancelled_by_obstacle"] == 1
    assert G.check_mode_sequence(modes, 4) == []


def test_obstacle_override_when_required_even_in_base_and_requests_are_ignored():
    m = G.TacticalMachine(pulse_n=2)
    d = m.step(G.LEFT_A, info_available=True, avoid_required=True)
    assert d.executed == "OBSTACLE_OVERRIDE" and d.tactical_kind is None and d.suppression == "obstacle_authority"
    assert [m.step(1, info_available=True, avoid_required=False).executed for _ in range(2)] == ["REJOIN", "REJOIN"]


def test_override_during_rejoin_restarts_a_full_rejoin():
    m = G.TacticalMachine(pulse_n=1)
    modes = _run(m, [1, 1, 1, 1, 1, 1, 1], avoid=lambda t: t == 2)
    assert modes[:3] == ["TACTICAL_LEFT", "REJOIN", "OBSTACLE_OVERRIDE"] and modes[3:5] == ["REJOIN", "REJOIN"]
    assert G.check_mode_sequence(modes, 1) == []


def test_geometry_failure_mid_pulse_aborts_to_rejoin():
    m = G.TacticalMachine(pulse_n=4)
    assert m.step(1, info_available=True, avoid_required=False).executed == "TACTICAL_LEFT"
    d = m.step(1, info_available=True, avoid_required=False, geometry_reason=lambda kind: "degenerate_geometry")
    assert d.executed == "REJOIN" and d.tactical_kind is None and m.counters["pulses_aborted_geometry"] == 1


def test_transitions_are_deterministic_and_invariants_hold_on_random_request_and_obstacle_sequences():
    rng = random.Random(7)
    for n in (1, 2, 3, 4):
        for _ in range(300):
            reqs = [rng.randrange(4) for _ in range(40)]
            avoids = [rng.random() < 0.2 for _ in range(40)]
            infos = [rng.random() < 0.8 for _ in range(40)]
            runs = []
            for _rep in range(2):
                m = G.TacticalMachine(pulse_n=n)
                runs.append([m.step(r, info_available=i, avoid_required=a).executed for r, i, a in zip(reqs, infos, avoids)])
            assert runs[0] == runs[1]
            assert G.check_mode_sequence(runs[0], n) == [], (n, reqs, avoids, runs[0])
            assert not any(mm in G.TACTICAL_MODES and a for mm, a in zip(runs[0], avoids))  # obstacle ownership always beats tactical steering


@pytest.mark.parametrize("name,modes", [
    ("long pulse", ["TACTICAL_LEFT"] * 3 + ["REJOIN"] * 2), ("no rejoin", ["TACTICAL_LEFT", "TACTICAL_LEFT", "BASE"]),
    ("short rejoin", ["TACTICAL_LEFT", "TACTICAL_LEFT", "REJOIN", "TACTICAL_LEFT"]), ("tactical after override", ["OBSTACLE_OVERRIDE", "TACTICAL_LEFT"]),
    ("override to base", ["OBSTACLE_OVERRIDE", "BASE"]), ("rejoin extension", ["TACTICAL_LEFT", "TACTICAL_LEFT", "REJOIN", "REJOIN", "REJOIN", "REJOIN"]),
    ("kind switch", ["TACTICAL_LEFT", "TACTICAL_RIGHT"])])
def test_the_invariant_checker_detects_violations(name, modes):
    assert G.check_mode_sequence(modes, 2), name


# ----------------------------------------------------------------------------- observation v2
def test_observation_v2_schema_is_the_37_feature_prefix_plus_six_state_features():
    assert G.OBS_V2_DIM == 43 and len(G.OBS_V2_SCHEMA) == 43 and G.OBS_V2_SCHEMA[:37] == A.A1_V2_OBSERVATION_SCHEMA
    assert [n for n, _ in G.OBS_V2_SCHEMA[37:]] == ["tactical_mode_base", "tactical_mode_left", "tactical_mode_right", "tactical_mode_climb",
                                                   "tactical_mode_rejoin_or_obstacle_suppressed", "tactical_remaining_fraction"]
    assert G.OBSERVATION_VERSION == "a1_v2_obs_v2" and G.ACTION_VERSION == "a1_v2_action_v5" and G.TRAINING_PROTOCOL == "a1_v2_training_v3"


def test_machine_observation_is_one_hot_with_fraction_in_unit_interval_for_every_state():
    m = G.TacticalMachine(pulse_n=3)
    seen = set()
    for t in range(30):
        d = m.step(1 + t % 3, info_available=True, avoid_required=(t % 11 == 5))
        obs = m.observation()
        assert obs.shape == (6,) and obs[:5].sum() == 1.0 and set(obs[:5]) <= {0.0, 1.0} and 0.0 <= obs[5] <= 1.0
        seen.add(int(np.argmax(obs[:5])))
    assert seen == {0, 1, 2, 3, 4}   # BASE, LEFT, RIGHT, CLIMB and REJOIN/OBSTACLE-SUPPRESSED are all reachable


def test_every_machine_state_is_distinguishable_in_the_observation():
    m = G.TacticalMachine(pulse_n=3)
    keys = {"BASE": tuple(m.observation())}
    m.step(1, info_available=True, avoid_required=False)
    keys["LEFT"] = tuple(m.observation())
    m2 = G.TacticalMachine(pulse_n=3)
    m2.step(2, info_available=True, avoid_required=False)
    keys["RIGHT"] = tuple(m2.observation())
    m3 = G.TacticalMachine(pulse_n=3)
    m3.step(3, info_available=True, avoid_required=False)
    keys["CLIMB"] = tuple(m3.observation())
    m4 = G.TacticalMachine(pulse_n=3)
    m4.step(0, info_available=True, avoid_required=True)
    keys["OVERRIDE"] = tuple(m4.observation())
    assert len(set(keys.values())) == 5
    for _ in range(3):
        m.step(1, info_available=True, avoid_required=False)
    assert m.mode == "REJOIN" and m.observation()[5] == pytest.approx(0.5)


def test_pulse_remaining_fraction_counts_down_to_zero_pulse_progress():
    m = G.TacticalMachine(pulse_n=4)
    m.step(1, info_available=True, avoid_required=False)
    assert m.observation()[5] == pytest.approx(3 / 4)
    m.step(1, info_available=True, avoid_required=False)
    assert m.observation()[5] == pytest.approx(2 / 4)


# ----------------------------------------------------------------------------- environment
def _episode(env, scenario, policy):
    env.reset(options={"scenario": scenario})
    done, info, t = False, {}, 0
    total = 0.0
    while not done:
        obs, r, te, tr, info = env.step(policy(t))
        total += r
        done = te or tr
        t += 1
    return info["a1_episode"], total


@pytest.mark.parametrize("level", ("C0", "C2", "C3"))
def test_forced_base_is_bit_identical_to_canonical_a0(level):
    bank = A.A1V2DefenderBank()
    machine = G.A1V5TrainingEnv(0, 0, defender_bank=bank, theta_deg=20.0, pulse_s=1.0, record_trajectory=True)
    a0 = G.A1V5TrainingEnv(0, 0, defender_bank=bank, scripted_a0=True, theta_deg=20.0, pulse_s=1.0, record_trajectory=True)
    for family in ("greedy", "pn"):
        for seed in (145000, 145001, 145002):
            x, rx = _episode(a0, (level, family, seed), lambda t: 0)
            y, ry = _episode(machine, (level, family, seed), lambda t: 0)
            assert x["trajectory"] == y["trajectory"] and (x["outcome"], x["steps"]) == (y["outcome"], y["steps"])
            assert rx == pytest.approx(ry) and y["tactical_steps"] == 0 and y["state_machine_violations"] == 0


def test_spaces_observation_and_reward_version():
    env = G.A1V5TrainingEnv(0, 0, theta_deg=20.0, pulse_s=1.0)
    obs, info = env.reset(options={"scenario": ("C1", "greedy", 145003)})
    assert env.action_space.n == 4 and env.observation_space.shape == (43,) and obs.shape == (43,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    assert info["a1_reward_version"] == "a1_v2_reward_v2" == G.REWARD_VERSION == R.REWARD_VERSION
    assert info["a1_action_version"] == "a1_v2_action_v5" and info["a1_observation_version"] == "a1_v2_obs_v2"
    obs2, *_ = env.step(0)
    assert np.allclose(obs2[37:], [1, 0, 0, 0, 0, 0]) and np.all(np.abs(obs2) <= 1.0)


def test_first_37_observation_features_equal_the_v1_observation():
    env = G.A1V5TrainingEnv(0, 0, theta_deg=20.0, pulse_s=1.0)
    env.reset(options={"scenario": ("C2", "lead", 145004)})
    for _ in range(3):
        obs, *_ = env.step(0)
        v1 = A.build_a1_v2_observation(env.soldier_env, env._detection_range)
        assert np.array_equal(obs[:37], v1.astype(np.float32))


def test_reward_v2_is_unchanged_in_the_new_env():
    env = G.A1V5TrainingEnv(0, 0, theta_deg=20.0, pulse_s=1.0)
    ep, total = _episode(env, ("C0", "greedy", 145005), lambda t: 0)
    assert total == pytest.approx(ep["return_undiscounted"]) and set(ep["components"]) == {"terminal", "progress", "path", "step"}
    assert ep["components"]["step"] == pytest.approx(-0.1 * ep["steps"])
    assert R.step_reward(range_prev_m=10.0, range_next_m=9.0, initial_range_m=20.0, path_step_m=1.0, path_budget_m=100.0, outcome="timeout").total < 0


def test_tactical_pulses_start_only_with_defender_information_and_are_finite_in_the_real_env():
    env = G.A1V5TrainingEnv(0, 0, theta_deg=20.0, pulse_s=1.0, record_trajectory=True)
    ep, _ = _episode(env, ("C0", "greedy", 145006), lambda t: 1)
    assert ep["modes"][0] == "BASE"                                            # defender far away at t = 0: LEFT is suppressed
    first = next((i for i, m in enumerate(ep["modes"]) if m == "TACTICAL_LEFT"), None)
    assert first is None or first > 0
    assert ep["longest_tactical_run"] <= 2 and ep["state_machine_violations"] == 0


def test_forced_gate_mission_only_diagnostic_drives_pulses_and_rejoins():
    env = G.A1V5TrainingEnv(0, 0, theta_deg=30.0, pulse_s=1.0, gate_override=True, record_trajectory=True)
    ep, _ = _episode(env, ("C0", G.PASSIVE_DIAGNOSTIC, 145007), lambda t: 1)
    assert ep["modes"][:4] == ["TACTICAL_LEFT", "TACTICAL_LEFT", "REJOIN", "REJOIN"] and ep["state_machine_violations"] == 0
    base, _ = _episode(G.A1V5TrainingEnv(0, 0, theta_deg=30.0, pulse_s=1.0, gate_override=True, record_trajectory=True),
                       ("C0", G.PASSIVE_DIAGNOSTIC, 145007), lambda t: 0)
    assert S.divergence(ep["trajectory"], base["trajectory"]) > 0.5     # the lateral pulse really changes the trajectory


def test_env_is_deterministic_for_a_fixed_scenario_and_action_sequence():
    outs = []
    for _ in range(2):
        env = G.A1V5TrainingEnv(0, 0, theta_deg=20.0, pulse_s=1.5, gate_override=True, record_trajectory=True)
        ep, total = _episode(env, ("C3", G.PASSIVE_DIAGNOSTIC, 145008), lambda t: 1 + (t * 7) % 3)
        outs.append((ep["trajectory"], ep["modes"], ep["outcome"], total))
    assert outs[0] == outs[1]


def test_obstacle_override_beats_tactical_steering_in_clutter():
    env = G.A1V5TrainingEnv(0, 0, theta_deg=30.0, pulse_s=2.0, gate_override=True, record_trajectory=True)
    cancelled = 0
    for seed in range(145000, 145012):
        ep, _ = _episode(env, ("C3", G.PASSIVE_DIAGNOSTIC, seed), lambda t: 1 + t % 3)
        cancelled += ep["pulses_cancelled_by_obstacle"]
        assert ep["state_machine_violations"] == 0
    assert cancelled > 0


# ----------------------------------------------------------------------------- policies / specification / restart
def test_scripted_policies_cover_the_pre_registered_list_and_are_deterministic():
    rng = np.random.default_rng(0)
    assert S.POLICY_NAMES == ("BASE", "LEFT", "RIGHT", "CLIMB", "ALT_LR", "CYCLE_LCR", "RANDOM", "ADVERSARIAL")
    assert [S.policy_action("ALT_LR", t, rng) for t in range(4)] == [1, 2, 1, 2]
    assert [S.policy_action("CYCLE_LCR", t, rng) for t in range(6)] == [1, 3, 2, 1, 3, 2]
    assert {S.policy_action("ADVERSARIAL", t, rng) for t in range(60)} == {1, 2, 3}
    assert {S.policy_action("RANDOM", t, np.random.default_rng(t)) for t in range(60)} <= {0, 1, 2, 3}


def test_grid_is_the_pre_registered_one():
    assert G.THETA_GRID_DEG == (10.0, 15.0, 20.0, 30.0) and G.PULSE_GRID_S == (0.5, 1.0, 1.5, 2.0) and G.REJOIN_S == 1.0 and G.REJOIN_STEPS == 2
    assert [G.pulse_steps(t) for t in G.PULSE_GRID_S] == [1, 2, 3, 4] and len(S.all_tasks()) == 64
    assert S.SEEDS[0] == 145000 and S.SEEDS[-1] == 145099 and S.LEVELS == ("C0", "C1", "C2", "C3")


def test_frozen_specification_is_stable_and_matches_the_artifact_when_present():
    assert S.spec_sha256() == hashlib.sha256(json.dumps(S.frozen_specification(), sort_keys=True).encode()).hexdigest()
    path = TS.STATE_DIR / "maneuver_specification.json"
    if path.exists():
        assert json.loads(path.read_text())["spec_sha256"] == S.spec_sha256()


def test_sweep_group_is_atomic_verified_and_restart_skips_valid_groups(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "GROUP_DIR", tmp_path)
    monkeypatch.setattr(S, "SEEDS", (145000, 145001))
    monkeypatch.setattr(S, "expected_records", lambda: len(S.POLICY_NAMES) * len(S.CONTEXTS) * 2)
    task = (20.0, 0.5, "C0")
    first = S.run_group(task)
    assert first["episodes"] == len(S.POLICY_NAMES) * len(S.CONTEXTS) * 2 and first["determinism_ok"] and S.verify_group(S.group_path(*task))
    assert S.run_group(task) == {"task": task, "skipped": True}
    path = S.group_path(*task)
    payload = json.loads(path.read_text())
    payload["records"][0]["collision"] = not payload["records"][0]["collision"]
    path.write_text(json.dumps(payload))
    assert not S.verify_group(path)                                              # tampering is detected
    path.write_text("{truncated")
    assert not S.verify_group(path)                                              # partial file is detected
    assert "skipped" not in S.run_group(task) and S.verify_group(path)           # and the group is recomputed on restart
    assert not list(tmp_path.glob("*.tmp*"))                                      # atomic write leaves no temporary files behind


def test_analysis_gates_flag_a_gained_collision_and_select_by_the_registered_order():
    def rec(theta, pulse, policy, *, collision=False, context="greedy", level="C1", seed=1, div=3.0, differs=True, sep=1.0, path_excess=1.2, exhausted=False,
            timeout=False, violations=0, run=1):
        return {"theta": theta, "pulse_s": pulse, "level": level, "context": context, "policy": policy, "seed": seed, "outcome": "x", "steps": 10,
                "collision": collision, "exhausted": exhausted, "timeout": timeout, "captured": False, "intercepted": False, "path_excess": path_excess,
                "min_sep": 5.0, "progress_net": 0.5, "ceiling_hover": False, "sustained_hover": False, "orbit": 0.0, "overshoots": 0, "tactical_steps": 2,
                "longest_tactical_run": run, "pulses": 1, "pulses_cancelled": 0, "pulses_aborted": 0, "violations": violations, "suppressed": {},
                "transitions": {}, "steps_by_mode": {}, "div_m": 0.0 if policy == "BASE" else div, "base_outcome": "x",
                "outcome_differs": differs and policy != "BASE", "base_collision": False, "min_sep_diff": 0.0 if policy == "BASE" else sep, "steps_diff": 0,
                "base_steps": 10, "base_min_sep": 5.0, "mean_altitude_m": 10.0}

    records = []
    for theta, pulse, kw in ((10.0, 1.0, {}), (15.0, 0.5, {"path_excess": 1.5}), (20.0, 1.0, {"collision": True}), (30.0, 2.0, {"div": 0.1, "differs": False, "sep": 0.0})):
        for policy in S.POLICY_NAMES:
            shared = {k: v for k, v in kw.items() if k in ("div", "differs", "sep")}   # authority applies to every policy, the rest only to LEFT
            records.append(rec(theta, pulse, policy, **(kw if policy == "LEFT" else shared)))
    result = S.analyze(records)
    by = {(c["theta"], c["pulse_s"]): c for c in result["candidates"]}
    assert by[(20.0, 1.0)]["G1"] is False and by[(20.0, 1.0)]["gained_collisions_total"] == 1 and not by[(20.0, 1.0)]["passed"]
    assert by[(30.0, 2.0)]["G3"] is False and not by[(30.0, 2.0)]["passed"]
    assert by[(10.0, 1.0)]["passed"] and by[(15.0, 0.5)]["passed"]
    assert result["selected"] == {"theta_deg": 15.0, "pulse_s": 0.5}   # equal endurance excess and median path-excess -> the shorter pulse wins
