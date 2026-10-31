"""Tests for the defender-evasion gating interface (`a1_v2_action_v6`). No PPO / learn() anywhere."""

from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
import pytest

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_eg as E
from experiments import journal_v2_a1_eg_state as TS
from experiments import journal_v2_a1_eg_sweep as S
from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_a1_tg as G
from uav_defend.envs.soldier_env import SoldierEnv


# ----------------------------------------------------------------------------- primitive (existing, unmodified)
def _env():
    env = SoldierEnv(A.build_episode_config("C0", "greedy"))
    env.reset(seed=145000)
    return env


def test_primitive_matches_its_documented_formula_and_is_pure():
    env = _env()
    cfg = env.config
    defender = np.array([0.0, 0.0, 10.0])
    for d in (1.0, 5.0, 10.0, 19.0):
        enemy = defender + np.array([d, 0.0, 0.0])
        out = env._compute_enemy_evasion(defender, enemy)
        alpha = max(0.0, 1.0 - d / cfg.enemy_evasion_radius)
        assert out["activation"] == pytest.approx(alpha) and out["effective_weight"] == pytest.approx(cfg.enemy_evasion_gain * alpha)
        assert np.allclose(out["evasion_component"], cfg.enemy_evasion_gain * alpha * np.array([1.0, 0.0, 0.0]), atol=1e-6)
        again = env._compute_enemy_evasion(defender, enemy)
        assert np.array_equal(out["evasion_component"], again["evasion_component"])


def test_primitive_is_zero_beyond_its_radius_and_finite_at_zero_distance():
    env = _env()
    defender = np.array([0.0, 0.0, 10.0])
    far = env._compute_enemy_evasion(defender, defender + np.array([env.config.enemy_evasion_radius + 1.0, 0.0, 0.0]))
    assert far["activation"] == 0.0 and not np.any(far["evasion_component"])
    same = env._compute_enemy_evasion(defender, defender.copy())
    assert np.isfinite(same["evasion_component"]).all() and np.allclose(same["away_direction"], [0, 0, 1])
    assert same["activation"] == pytest.approx(1.0)


def test_evasion_activates_strictly_inside_the_a1_information_gate():
    cfg = A.build_episode_config("C0", "greedy")
    assert cfg.enemy_evasion_radius < cfg.enemy_defender_detection_range  # the 25 m gate never removes an active evasion


def test_v2_zeroes_the_evasion_contribution_in_the_canonical_command_path():
    cfg = A.build_episode_config("C0", "greedy")
    assert cfg.enemy_nominal_guidance_mode == "direct_obstacle_only"
    env = _env()
    env.step(np.zeros(3, dtype=np.float32))
    assert not np.any(env._enemy_evasion_command)  # canonical A0 never uses it, even when the defender is in range


# ----------------------------------------------------------------------------- blend mapping
def test_blend_zero_or_inactive_primitive_returns_none():
    m = np.array([1.0, 0.0, 0.0])
    assert E.blended_direction(m, np.array([0.0, 0.5, 0.0]), 0.0) is None
    assert E.blended_direction(m, np.zeros(3), 1.0) is None
    assert E.blended_direction(m, np.array([-1.0, 0.0, 0.0]), 1.0) is None  # exactly cancelling -> degenerate -> BASE


@pytest.mark.parametrize("blend", E.BLENDS[1:])
def test_blended_direction_is_unit_and_monotone_in_blend(blend):
    m = np.array([1.0, 0.0, 0.0])
    component = np.array([0.0, 0.5, 0.0])
    u = E.blended_direction(m, component, blend)
    assert np.linalg.norm(u) == pytest.approx(1.0)
    assert math.degrees(math.acos(u[0])) == pytest.approx(math.degrees(math.atan(0.5 * blend)))
    if blend > 0.25:
        assert math.acos(u[0]) > math.acos(E.blended_direction(m, component, 0.25)[0])


def test_blend_levels_validation_and_action_space():
    with pytest.raises(ValueError):
        E.A1V6TrainingEnv(0, 0, blend_levels=(0.25, 0.5))
    with pytest.raises(ValueError):
        E.A1V6TrainingEnv(0, 0, blend_levels=(0.0, 1.5))
    assert E.A1V6TrainingEnv(0, 0).action_space.n == 5 and E.A1V6TrainingEnv(0, 0, blend_levels=(0.0, 0.5, 1.0)).action_space.n == 3


# ----------------------------------------------------------------------------- environment
def _episode(env, scenario, policy):
    env.reset(options={"scenario": scenario})
    done, info, t, total = False, {}, 0, 0.0
    while not done:
        _o, r, te, tr, info = env.step(policy(env, t))
        total += r
        done, t = te or tr, t + 1
    return info["a1_episode"], total


def test_spaces_versions_and_observation_are_unchanged_37d():
    env = E.A1V6TrainingEnv(0, 0)
    obs, info = env.reset(options={"scenario": ("C1", "greedy", 145003)})
    assert obs.shape == (37,) == env.observation_space.shape and env.observation_space.contains(obs)
    assert (info["a1_action_version"], info["a1_observation_version"], info["a1_reward_version"]) == ("a1_v2_action_v6", "a1_v2_obs_v1", "a1_v2_reward_v2")
    assert np.array_equal(obs, A.build_a1_v2_observation(env.soldier_env, env._detection_range).astype(np.float32))


@pytest.mark.parametrize("level,family", [("C0", "greedy"), ("C2", "pn"), ("C3", "lead"), ("C1", "kalman_greedy"), ("C2", "rmpc")])
def test_blend_zero_is_bit_identical_to_canonical_a0(level, family):
    bank = A.A1V2DefenderBank()
    machine = E.A1V6TrainingEnv(0, 0, defender_bank=bank, record_trajectory=True)
    a0 = E.A1V6TrainingEnv(0, 0, defender_bank=bank, scripted_a0=True, record_trajectory=True)
    for seed in (145000, 145001):
        x, rx = _episode(a0, (level, family, seed), lambda e, t: 0)
        y, ry = _episode(machine, (level, family, seed), lambda e, t: 0)
        assert x["trajectory"] == y["trajectory"] and (x["outcome"], x["steps"]) == (y["outcome"], y["steps"]) and rx == pytest.approx(ry)
        assert y["evasion_steps"] == 0


def test_no_defender_information_forces_blend_zero_and_is_logged():
    env = E.A1V6TrainingEnv(0, 0, record_trajectory=True)
    ep, _ = _episode(env, ("C0", "greedy", 145000), lambda e, t: 4)
    assert ep["suppressed"].get("no_defender_info", 0) >= 1
    assert ep["executed_blends"][0] == 0.0  # the defender starts outside the 25 m gate
    base, _ = _episode(env, ("C0", "greedy", 145000), lambda e, t: 0)
    n = ep["suppressed"]["no_defender_info"]
    assert ep["trajectory"][: n + 1] == base["trajectory"][: n + 1]  # identical until information becomes available


def test_evasion_executes_inside_the_gate_and_never_while_the_planner_owns_steering():
    env = E.A1V6TrainingEnv(0, 0, record_trajectory=True)
    executed = owned_exec = 0
    for seed in range(145000, 145010):
        ep, _ = _episode(env, ("C3", "greedy", seed), lambda e, t: 4)
        executed += ep["evasion_steps"]
        owned_exec += ep["executed_while_owned"]
        assert ep["evasion_steps"] <= ep["steps_info_available"]
    assert executed > 0 and owned_exec == 0


def test_obstacle_ownership_suppresses_evasion_completely(monkeypatch):
    monkeypatch.setattr(G, "common_layer_required", lambda env: {"required": True, "layout_size": 1})
    env = E.A1V6TrainingEnv(0, 0, record_trajectory=True)
    ep, _ = _episode(env, ("C0", "greedy", 145000), lambda e, t: 4)
    base, _ = _episode(env, ("C0", "greedy", 145000), lambda e, t: 0)
    assert ep["evasion_steps"] == 0 and ep["suppressed"].get("obstacle_authority", 0) >= 1 and ep["trajectory"] == base["trajectory"]


def test_threat_gating_switching_equals_fixed_and_is_deterministic():
    bank = A.A1V2DefenderBank()
    env = E.A1V6TrainingEnv(0, 0, defender_bank=bank, record_trajectory=True)
    rng = np.random.default_rng(0)
    for sc in (("C1", "pn", 145002), ("C3", "lead", 145004)):
        fixed, rf = _episode(env, sc, lambda e, t: S.action_for("FIXED_0.75", t, e, rng, 5))
        sw, rs = _episode(env, sc, lambda e, t: S.action_for("SWITCH_0.75", t, e, rng, 5))
        again, _ = _episode(env, sc, lambda e, t: S.action_for("FIXED_0.75", t, e, rng, 5))
        assert fixed["trajectory"] == sw["trajectory"] == again["trajectory"] and rf == pytest.approx(rs)


@pytest.mark.parametrize("policy", S.STRESS_POLICIES)
def test_switching_stress_policies_are_finite_and_respect_ownership(policy):
    env = E.A1V6TrainingEnv(0, 0, blend_levels=(0.0, 0.5, 1.0), record_trajectory=True)
    for seed in (145001, 145005):
        ep, total = _episode(env, ("C3", "pn", seed), lambda e, t: S.action_for(policy, t, e, np.random.default_rng(seed), 3))
        assert np.isfinite(np.asarray(ep["trajectory"], dtype=float)).all() and np.isfinite(total) and ep["executed_while_owned"] == 0


def test_reward_v2_is_unchanged_in_the_new_env():
    env = E.A1V6TrainingEnv(0, 0)
    ep, total = _episode(env, ("C0", "greedy", 145005), lambda e, t: 0)
    assert total == pytest.approx(ep["return_undiscounted"]) and ep["components"]["step"] == pytest.approx(-0.1 * ep["steps"])
    assert E.REWARD_VERSION == R.REWARD_VERSION == "a1_v2_reward_v2"
    assert R.step_reward(range_prev_m=10.0, range_next_m=9.0, initial_range_m=20.0, path_step_m=1.0, path_budget_m=100.0, outcome="timeout").total < 0


# ----------------------------------------------------------------------------- specification / sweep restart / locks / state
def test_pre_registered_grid_and_specification_hash():
    assert E.BLENDS == (0.0, 0.25, 0.5, 0.75, 1.0) and S.SEEDS[0] == 145000 and S.SEEDS[-1] == 145099 and S.LEVELS == ("C0", "C1", "C2", "C3")
    assert S.FAMILIES == ("greedy", "kalman_greedy", "pn", "lead", "rmpc") and len(S.all_tasks()) == 20 and len(S.POLICIES) == 9
    assert S.spec_sha256() == hashlib.sha256(json.dumps(S.frozen_specification(), sort_keys=True).encode()).hexdigest()
    path = TS.STATE_DIR / "interface_specification.json"
    if path.exists():
        assert json.loads(path.read_text())["spec_sha256"] == S.spec_sha256()


def test_cell_is_atomic_verified_restart_skips_and_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "CELL_DIR", tmp_path)
    monkeypatch.setattr(S, "SEEDS", (145000, 145001))
    task = ("sweep", "C0", "greedy", E.BLENDS)
    first = S.run_cell(task)
    assert first["episodes"] == 18 and first["base_equals_a0"] and first["switch_equals_fixed"] and first["deterministic"] and S.cell_ok(task)
    assert S.run_cell(task) == {"task": task, "skipped": True}
    path = S.cell_for(task)
    payload = json.loads(path.read_text())
    payload["records"][1]["collision"] = not payload["records"][1]["collision"]
    path.write_text(json.dumps(payload))
    assert not S.cell_ok(task)
    path.write_text("{trunc")
    assert not S.cell_ok(task)
    assert "skipped" not in S.run_cell(task) and S.cell_ok(task)
    assert not list(tmp_path.glob(".*tmp*")) and not list(tmp_path.glob("*.lock"))


def test_cell_lock_prevents_duplicate_writers_and_takes_over_stale_locks(tmp_path):
    a, b = TS.CellLock(tmp_path / "cell.json"), TS.CellLock(tmp_path / "cell.json")
    assert a.acquire()
    assert not b.acquire()  # a live owner (this process) holds it: a duplicate launch must not take the cell
    a.release()
    (tmp_path / "cell.lock").write_text("999999")  # no such process -> stale
    assert b.acquire()
    b.release()
    assert not (tmp_path / "cell.lock").exists()


def test_run_cell_skips_a_cell_whose_lock_is_held(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "CELL_DIR", tmp_path)
    monkeypatch.setattr(S, "SEEDS", (145000,))
    task = ("sweep", "C0", "pn", E.BLENDS)
    holder = TS.CellLock(S.cell_for(task))
    assert holder.acquire()
    result = S.run_cell(task)
    holder.release()
    assert result.get("skipped") and not S.cell_for(task).exists()


def test_task_state_restart_skips_only_verified_stages_and_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(TS, "STATE_PATH", tmp_path / "task_state.json")
    rel = "docs/journal_v2_a1_evasion_gate_v1/_restart_probe.txt"
    probe = TS.PROJECT_ROOT / rel
    probe.parent.mkdir(parents=True, exist_ok=True)
    try:
        probe.write_text("v1")
        state = TS.load_state()
        assert TS.next_stage(state) == TS.STAGES[0]
        TS.begin(state, TS.STAGES[0])
        assert not TS.stage_is_verified(state, TS.STAGES[0])  # running stages are never trusted
        TS.complete(state, TS.STAGES[0], [rel], "ok")
        assert TS.next_stage(TS.load_state()) == TS.STAGES[1]
        probe.write_text("v2")  # artifact changed -> the stage must be redone
        assert TS.next_stage(TS.load_state()) == TS.STAGES[0]
        probe.write_text("v1")
        TS.halt(state, "05_unit_tests", "D", "test")
        later = TS.load_state()
        assert later["stages"]["06_deterministic_sweep"]["status"] == "not_applicable" and later["stages"]["07_analysis"]["status"] == "not_applicable"
        assert later["stages"]["10_publication_documentation"]["status"] == "pending"  # evidence stages stay runnable after a halt
    finally:
        if probe.exists():
            probe.unlink()
    (tmp_path / "task_state.json").write_text("{corrupt")
    assert TS.load_state()["stages"][TS.STAGES[0]]["status"] == "pending"


# ----------------------------------------------------------------------------- analysis gates (synthetic records)
def _rec(policy, level="C1", defender="greedy", seed=1, collision=False, base_collision=False, exhausted=False, div=2.0, conv=False, **kw):
    r = {"level": level, "defender": defender, "policy": policy, "seed": seed, "outcome": "soldier_caught" if conv else "intercepted", "steps": 10,
         "collision": collision, "exhausted": exhausted, "timeout": False, "captured": conv, "intercepted": not conv, "path_excess": 1.1, "path_length_m": 40.0,
         "min_sep": 3.0, "progress_net": 0.5, "ceiling_hover": False, "sustained_hover": False, "orbit": 0.0, "overshoots": 0, "evasion_steps": 2,
         "requested_steps": 5, "suppressed": {"no_defender_info": 3}, "steps_primitive_active": 2, "steps_info_available": 3, "steps_obstacle_owned": 0,
         "max_deflection_deg": 10.0, "evasion_in_last3": 0, "executed_while_owned": 0, "div_m": 0.0 if policy == "BASE" else div, "base_outcome": "intercepted",
         "outcome_differs": conv, "base_collision": base_collision, "min_sep_diff": 0.0, "steps_diff": 0, "path_diff_m": 0.0, "base_steps": 10}
    r.update(kw)
    return r


def _synthetic(bad_blend=None, weak=False):
    records = []
    for level in S.LEVELS:
        for seed in range(40):
            records.append(_rec("BASE", level=level, seed=seed))
            for b in S.NONZERO:
                for kind in ("FIXED", "SWITCH"):
                    name = f"{kind}_{b:.2f}"
                    row = _rec(name, level=level, seed=seed, collision=(bad_blend == b and level == "C2" and seed == 0), conv=(not weak and seed < 15 and b >= 0.5),
                               div=0.1 if weak else 2.0)
                    for other in S.NONZERO:
                        row[f"div_to_{other:.2f}"] = 0.0 if other == b else (2.0 if abs(other - b) >= 0.5 else 0.2)
                    records.append(row)
    return records


FLAGS = {"base_equals_a0": True, "switch_equals_fixed": True, "deterministic_replay": True, "finite": True}


def test_gates_flag_a_single_gained_collision_and_selection_skips_that_blend():
    result = S.analyze(_synthetic(bad_blend=0.5), FLAGS)
    p = result["policies"]
    assert p["FIXED_0.50"]["G1"] is False and p["FIXED_0.50"]["gained_collisions_total"] == 1 and p["FIXED_0.25"]["G1"] is True
    assert 0.5 not in result["selection"]["passing_blends"] and 0.75 in result["selection"]["passing_blends"]


def test_weak_authority_fails_g3_and_selects_nothing():
    result = S.analyze(_synthetic(weak=True), FLAGS)
    assert all(not p["G3"] for p in result["policies"].values()) and result["selection"]["final_blends"] is None


def test_final_set_is_base_low_high_when_distinct_and_has_no_redundant_strengths():
    result = S.analyze(_synthetic(), FLAGS)
    final = result["selection"]["final_blends"]
    assert final[0] == 0.0 and len(final) in (2, 3) and final == sorted(final)
    if len(final) == 3:
        assert result["selection"]["low_high_divergence_fraction"] >= S.DIVERGENCE_FRACTION


def test_g5_failure_blocks_selection():
    flags = dict(FLAGS, base_equals_a0=False)
    result = S.analyze(_synthetic(), flags)
    assert result["G5"] is False and result["selection"]["final_blends"] is None
