"""Tests for the route-level A1 interface (`a1_route_action_v1`, `a1_v2_obs_route_v1`). No PPO / learn() anywhere."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math

import numpy as np
import pytest

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_a1_route as RT
from experiments import journal_v2_a1_route_env as RE
from experiments import journal_v2_a1_route_sweep as S
from experiments import journal_v2_a1_rt_state as TS
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.perception import PerceivedObstacleSet, PerceivedStaticObstacle

PARAMS = RT.RouteParams(radius_m=7.5, replan_steps=2, clearance=2.0, max_altitude=30.0, planning_range=40.0, max_graph_nodes=150, waypoint_tolerance=2.0)
EMPTY = PerceivedObstacleSet()


def _obstacle(oid, lo, hi):
    return PerceivedStaticObstacle(id=oid, min_corner=np.array(lo, dtype=float), max_corner=np.array(hi, dtype=float), source="map")


A_POS, S_POS = np.array([0.0, 0.0, 10.0]), np.array([60.0, 0.0, 10.0])


# ----------------------------------------------------------------------------- geometry
def test_defender_projection_clips_to_the_mission_segment():
    g = RT.mission_geometry(A_POS, S_POS, [30.0, 5.0, 10.0])
    assert g["s"] == pytest.approx(30.0) and np.allclose(g["q"], [30.0, 0.0, 10.0]) and np.allclose(g["m"], [1, 0, 0])
    assert RT.mission_geometry(A_POS, S_POS, [-20.0, 5.0, 10.0])["s"] == 0.0          # defender behind the attacker
    assert RT.mission_geometry(A_POS, S_POS, [90.0, 5.0, 10.0])["s"] == pytest.approx(60.0)  # beyond the asset
    assert RT.mission_geometry(A_POS, A_POS, [1, 1, 1]) is None                       # attacker at the asset: degenerate


def test_lateral_and_above_subgoals_follow_the_specification():
    d = [30.0, 5.0, 10.0]
    left, _ = RT.tactical_subgoal("LEFT", A_POS, S_POS, d, PARAMS)
    right, _ = RT.tactical_subgoal("RIGHT", A_POS, S_POS, d, PARAMS)
    above, _ = RT.tactical_subgoal("ABOVE", A_POS, S_POS, d, PARAMS)
    assert np.allclose(left, [30.0, 7.5, 10.0]) and np.allclose(right, [30.0, -7.5, 10.0]) and np.allclose(above, [30.0, 0.0, 17.5])
    assert np.allclose(RT.lateral_axis(np.array([1.0, 0, 0])), [0, 1, 0])


def test_above_subgoal_is_ceiling_capped_and_rejected_when_no_room():
    high_a, high_s = np.array([0.0, 0.0, 26.0]), np.array([60.0, 0.0, 26.0])
    capped, reason = RT.tactical_subgoal("ABOVE", high_a, high_s, [30.0, 5.0, 26.0], PARAMS)
    assert capped is None and reason == "ceiling"                                       # only 2 m of room (< 0.5 R = 3.75 m)
    ok, _ = RT.tactical_subgoal("ABOVE", np.array([0, 0, 22.0]), np.array([60, 0, 22.0]), [30.0, 5.0, 22.0], PARAMS)
    assert ok[2] == pytest.approx(28.0)                                                 # 22 + 7.5 capped at ceiling 30 - clearance 2


def test_rejections_defender_behind_and_degenerate_geometry():
    assert RT.tactical_subgoal("LEFT", A_POS, S_POS, [-10.0, 3.0, 10.0], PARAMS) == (None, "defender_not_ahead")
    assert RT.tactical_subgoal("LEFT", A_POS, np.array([0.0, 0.0, 40.0]), [0.0, 0.0, 20.0], PARAMS)[1] == "degenerate_geometry"
    assert RT.tactical_subgoal("DIRECT", A_POS, S_POS, [30.0, 5.0, 10.0], PARAMS)[1] == "invalid_mode"
    assert RT.tactical_subgoal("LEFT", A_POS, S_POS, [np.nan, 0, 0], PARAMS)[1] == "degenerate_geometry"


# ----------------------------------------------------------------------------- route validation
def test_candidate_route_is_finite_and_passes_through_the_subgoal_then_the_asset():
    cand, reason = RT.plan_candidate("LEFT", A_POS, S_POS, [30.0, 5.0, 10.0], EMPTY, PARAMS)
    assert reason is None and np.allclose(cand.waypoints[cand.subgoal_index], [30.0, 7.5, 10.0]) and np.allclose(cand.waypoints[-1], S_POS)
    assert all(np.isfinite(w).all() for w in cand.waypoints) and cand.length > 60.0


def test_candidate_route_avoids_perceived_obstacles_and_defender_is_not_an_obstacle():
    wall = PerceivedObstacleSet(static=(_obstacle(7, [14, -6, 0], [18, 6, 30]),))
    cand, reason = RT.plan_candidate("LEFT", A_POS, S_POS, [30.0, 5.0, 10.0], wall, PARAMS)
    assert reason is None
    boxes = RT.inflated_boxes(wall, PARAMS.clearance)
    path = (A_POS,) + cand.waypoints
    assert not any(b.segment_intersects(path[i], path[i + 1]) for b in boxes for i in range(len(path) - 1))
    assert RT.inflated_boxes(wall, 2.0) and len(RT.inflated_boxes(EMPTY, 2.0)) == 0      # the defender never enters the obstacle set


def test_subgoal_inside_an_obstacle_or_unreachable_is_rejected_and_nan_is_never_committed():
    block = PerceivedObstacleSet(static=(_obstacle(1, [26, 3, 0], [34, 12, 30]),))
    assert RT.plan_candidate("LEFT", A_POS, S_POS, [30.0, 5.0, 10.0], block, PARAMS)[1] == "no_path"
    enclosed = PerceivedObstacleSet(static=(_obstacle(2, [-1, -40, 0], [1, 40, 30]),))  # start engulfed by clearance: no feasible edge
    assert RT.plan_candidate("LEFT", np.array([0.0, 0.0, 10.0]), S_POS, [30.0, 5.0, 10.0], enclosed, PARAMS)[0] is None


# ----------------------------------------------------------------------------- persistence / replanning
def _step(c, request, a=A_POS, s=S_POS, d=(30.0, 5.0, 10.0), info=True, perceived=EMPTY, owned=False):
    return c.step(request, a, s, np.array(d), info, perceived, owned)


def test_route_persists_between_cadence_ticks_and_only_switches_at_a_tick():
    c = RT.RouteController(PARAMS)
    d0 = _step(c, 1)
    assert d0.mode == "LEFT" and "start" in d0.events and d0.direction is not None
    d1 = _step(c, 2, a=np.array([6.0, 0.0, 10.0]))                                       # request changes mid-route: ignored
    assert d1.mode == "LEFT" and not d1.planned
    d2 = _step(c, 2, a=np.array([12.0, 0.0, 10.0]))                                      # cadence expired (2 steps): re-decision -> RIGHT
    assert d2.planned and d2.mode == "RIGHT" and "replan" in d2.events


def test_request_direct_at_a_tick_cancels_and_defender_info_loss_cancels_immediately():
    c = RT.RouteController(PARAMS)
    _step(c, 1)
    _step(c, 1, a=np.array([6.0, 0.0, 10.0]))
    assert _step(c, 0, a=np.array([12.0, 0.0, 10.0])).mode == "DIRECT"
    c = RT.RouteController(PARAMS)
    _step(c, 1)
    d = _step(c, 1, a=np.array([6.0, 0.0, 10.0]), info=False)
    assert d.mode == "DIRECT" and "cancel:defender_info_lost" in d.events and d.direction is None and not d.planned


def test_no_defender_information_collapses_every_request_to_direct_without_planning():
    c = RT.RouteController(PARAMS)
    for request in (1, 2, 3):
        d = _step(c, request, info=False)
        assert d.mode == "DIRECT" and d.direction is None and not d.planned
    assert c.counters["plans_attempted"] == 0


def test_completion_returns_to_direct_then_holds_one_replan_interval_before_rearming():
    c = RT.RouteController(PARAMS)
    _step(c, 1)
    past = np.array([40.0, 0.0, 10.0])                                                   # abeam/past the subgoal, direct line clear
    d = _step(c, 1, a=past, d=(60.0, 10.0, 10.0))
    assert d.mode == "DIRECT" and "complete" in d.events and not d.planned and "locked:post_completion_hold" in d.events
    held = [_step(c, 1, a=past + [i, 0, 0], d=(70.0, 8.0, 10.0)) for i in range(1, 3)]
    assert all(h.mode == "DIRECT" and "locked:post_completion_hold" in h.events and not h.planned for h in held)
    rearmed = _step(c, 1, a=np.array([45.0, 0.0, 10.0]), d=(70.0, 8.0, 10.0))
    assert rearmed.planned                                                               # hold over: a planner call is allowed again


def test_rearming_after_the_hold_is_possible_and_obstacle_ownership_suppresses_the_command_but_not_the_state():
    c = RT.RouteController(PARAMS)
    _step(c, 1)
    d = _step(c, 1, a=np.array([5.0, 0.0, 10.0]), owned=True)
    assert d.mode == "LEFT" and d.direction is None and "suppressed:obstacle_authority" in d.events and c.counters["suppressed_by_obstacle"] == 1
    again = _step(c, 1, a=np.array([6.0, 0.0, 10.0]))
    assert again.mode == "LEFT" and again.direction is not None


def test_new_perceived_obstacle_blocking_the_route_forces_an_immediate_replan():
    c = RT.RouteController(PARAMS)
    _step(c, 1)
    wall = PerceivedObstacleSet(static=(_obstacle(5, [12, -1, 0], [16, 12, 30]),))
    d = _step(c, 1, a=np.array([5.0, 0.0, 10.0]), perceived=wall)
    assert d.planned and ("replan" in d.events or any(e.startswith("cancel") for e in d.events))


def test_unbounded_loops_cannot_occur_route_run_is_finite_in_a_closed_loop_toy():
    c = RT.RouteController(PARAMS)
    pos = A_POS.copy()
    for t in range(60):
        d = c.step(1, pos, S_POS, np.array([30.0, 5.0, 10.0]), True, EMPTY, False)
        if d.direction is not None:
            pos = pos + 6.0 * d.direction
    assert c.longest_run <= RT.ROUTE_RUN_BOUND_STEPS


# ----------------------------------------------------------------------------- observation
def test_observation_schema_dimension_and_ranges():
    assert RE.OBS_ROUTE_DIM == 43 and RE.OBS_ROUTE_SCHEMA[:37] == A.A1_V2_OBSERVATION_SCHEMA and RE.OBSERVATION_VERSION == "a1_v2_obs_route_v1"
    c = RT.RouteController(PARAMS)
    assert np.allclose(c.observation(), [1, 0, 0, 0, 0, 0])
    seen = set()
    rng = np.random.default_rng(0)
    pos = A_POS.copy()
    for t in range(80):
        d = c.step(int(rng.integers(0, 4)), pos, S_POS, np.array([30.0, 5.0, 10.0]), bool(rng.random() < 0.85), EMPTY, bool(rng.random() < 0.1))
        if d.direction is not None:
            pos = pos + 3.0 * d.direction
        o = c.observation()
        assert o.shape == (6,) and o[:4].sum() == 1.0 and 0.0 <= o[4] <= 1.0 and 0.0 <= o[5] <= 1.0
        seen.add(int(np.argmax(o[:4])))
    assert {0, 1} <= seen


# ----------------------------------------------------------------------------- environment
def _episode(env, scenario, policy):
    env.reset(options={"scenario": scenario})
    done, info, t, total = False, {}, 0, 0.0
    while not done:
        _o, r, te, tr, info = env.step(policy(env, t))
        total += r
        done, t = te or tr, t + 1
    return info["a1_episode"], total


def test_spaces_versions_reward_and_first_37_features():
    env = RE.A1RouteTrainingEnv(0, 0, radius_m=7.5, replan_s=1.0)
    obs, info = env.reset(options={"scenario": ("C1", "greedy", 145003)})
    assert env.action_space.n == 4 and obs.shape == (43,) and env.observation_space.contains(obs)
    assert (info["a1_action_version"], info["a1_observation_version"], info["a1_reward_version"]) == ("a1_route_action_v1", "a1_v2_obs_route_v1", "a1_v2_reward_v2")
    assert np.array_equal(obs[:37], A.build_a1_v2_observation(env.soldier_env, env._detection_range).astype(np.float32))
    ep, total = _episode(env, ("C0", "greedy", 145005), lambda e, t: 0)
    assert total == pytest.approx(ep["return_undiscounted"]) and ep["components"]["step"] == pytest.approx(-0.1 * ep["steps"])
    assert R.step_reward(range_prev_m=10.0, range_next_m=9.0, initial_range_m=20.0, path_step_m=1.0, path_budget_m=100.0, outcome="timeout").total < 0


@pytest.mark.parametrize("radius,replan", S.CONFIGS)
@pytest.mark.parametrize("level,family", [("C0", "greedy"), ("C3", "lead")])
def test_direct_is_bit_identical_to_canonical_a0_for_every_configuration(radius, replan, level, family):
    bank = A.A1V2DefenderBank()
    wrapper = RE.A1RouteTrainingEnv(0, 0, defender_bank=bank, radius_m=radius, replan_s=replan, record_trajectory=True)
    a0 = RE.A1RouteTrainingEnv(0, 0, defender_bank=bank, scripted_a0=True, radius_m=radius, replan_s=replan, record_trajectory=True)
    x, rx = _episode(a0, (level, family, 145001), lambda e, t: 0)
    y, ry = _episode(wrapper, (level, family, 145001), lambda e, t: 0)
    assert x["trajectory"] == y["trajectory"] and (x["outcome"], x["steps"]) == (y["outcome"], y["steps"]) and rx == pytest.approx(ry)
    assert y["plans_attempted"] == 0 and y["route_steps"] == 0


def test_requests_before_the_defender_is_visible_do_not_plan_and_audit_counters_stay_clean():
    env = RE.A1RouteTrainingEnv(0, 0, radius_m=7.5, replan_s=1.0, record_trajectory=True)
    ep, _ = _episode(env, ("C2", "pn", 145003), lambda e, t: 1)
    assert ep["modes"][0] == "DIRECT" and ep["audit"]["plan_calls_without_info"] == 0 and ep["audit"]["ids_outside_perceived"] == 0
    assert ep["executed_while_owned"] == 0 and ep["finite"]


def test_environment_is_deterministic_and_tactical_routes_change_the_trajectory():
    outs = []
    for _ in range(2):
        env = RE.A1RouteTrainingEnv(0, 0, radius_m=10.0, replan_s=2.0, record_trajectory=True)
        ep, total = _episode(env, ("C1", "greedy", 145004), lambda e, t: 1 + (t * 3) % 3)
        outs.append((ep["trajectory"], ep["modes"], ep["outcome"], total))
    assert outs[0] == outs[1]
    base_env = RE.A1RouteTrainingEnv(0, 0, radius_m=10.0, replan_s=2.0, record_trajectory=True)
    base, _ = _episode(base_env, ("C1", "greedy", 145004), lambda e, t: 0)
    assert S.divergence(outs[0][0], base["trajectory"]) > 0.5


def test_candidate_separations_is_pure_and_includes_direct():
    env = RE.A1RouteTrainingEnv(0, 0, radius_m=7.5, replan_s=1.0)
    env.reset(options={"scenario": ("C0", "greedy", 145000)})
    for _ in range(6):
        env.step(0)
    before = (env.controller.counters["plans_attempted"], env.controller.mode)
    seps = env.candidate_separations()
    assert 0 in seps and all(np.isfinite(v) for v in seps.values()) and (env.controller.counters["plans_attempted"], env.controller.mode) == before


# ----------------------------------------------------------------------------- perception contract
def test_controller_module_has_no_reference_to_ground_truth_world_state():
    assert S.forbidden_references() == []


def test_planner_uses_only_the_perceived_subset_under_local_perception():
    config = dataclasses.replace(A.build_episode_config("C3", "greedy"), enemy_obstacle_awareness_mode="local")
    env = SoldierEnv(config)
    env.reset(seed=145002)
    env.step(np.zeros(3, dtype=np.float32))
    perceived = env._enemy_last_perceived
    truth = len(env._obstacle_layout)
    assert len(perceived.static) < truth                                                  # undiscovered obstacles really exist in this mode
    p_a, p_s = np.asarray(env._enemy_pos, float), np.asarray(env._soldier_pos, float)
    mid = (p_a + p_s) / 2
    cand, _ = RT.plan_candidate("LEFT", p_a, p_s, mid + [0, 3, 0], perceived, RT.RouteParams.from_config(config, 7.5, 1.0))
    if cand is not None:
        assert set(cand.obstacle_ids) <= {o.id for o in perceived.static} and len(cand.obstacle_ids) == len(perceived.static) < truth


def test_defender_position_is_not_used_when_information_is_unavailable():
    a, b = RT.RouteController(PARAMS), RT.RouteController(PARAMS)
    da = a.step(1, A_POS, S_POS, np.array([30.0, 5.0, 10.0]), False, EMPTY, False)
    db = b.step(1, A_POS, S_POS, np.array([-500.0, 900.0, 1.0]), False, EMPTY, False)
    assert da.direction is None and db.direction is None and not da.planned and not db.planned


# ----------------------------------------------------------------------------- specification / restart / analysis
def test_pre_registered_grid_and_specification_hash():
    assert RT.RADII_M == (5.0, 7.5, 10.0) and RT.REPLAN_S == (1.0, 2.0) and len(S.CONFIGS) == 6 and S.SEEDS[0] == 145000 and S.SEEDS[-1] == 145099
    assert S.POLICIES == ("DIRECT", "LEFT", "RIGHT", "ABOVE", "ALT_LR", "RANDOM", "MAX_SEP", "ADVERSARIAL") and len(S.all_tasks()) == 120
    assert S.spec_sha256() == hashlib.sha256(json.dumps(S.frozen_specification(), sort_keys=True).encode()).hexdigest()
    path = TS.STATE_DIR / "route_interface_specification.json"
    if path.exists():
        assert json.loads(path.read_text())["spec_sha256"] == S.spec_sha256()


def test_scripted_policies_are_deterministic_and_cover_all_intents():
    env = RE.A1RouteTrainingEnv(0, 0, radius_m=7.5, replan_s=1.0)
    env.reset(options={"scenario": ("C0", "greedy", 145000)})
    rng = np.random.default_rng(0)
    assert {S.action_for("ADVERSARIAL", t, env, rng) for t in range(40)} == {0, 1, 2, 3}
    assert [S.action_for("LEFT", t, env, rng) for t in range(3)] == [1, 1, 1] and S.action_for("DIRECT", 0, env, rng) == 0
    assert {S.action_for("RANDOM", t, env, np.random.default_rng(t)) for t in range(60)} == {0, 1, 2, 3}


def test_staged_task_selection_and_available_cell_coverage(tmp_path, monkeypatch):
    assert len(S.all_tasks(("greedy",), ("C0",))) == 6 and len(S.all_tasks(("greedy", "pn"), ("C1", "C2", "C3"))) == 36
    assert all(t[3] != "rmpc" for t in S.all_tasks(("greedy", "kalman_greedy", "pn", "lead")))
    monkeypatch.setattr(S, "CELL_DIR", tmp_path)
    monkeypatch.setattr(S, "SEEDS", (145000,))
    monkeypatch.setattr(S, "EQUIV_SEEDS_PER_CELL", 1)
    S.run_cell((5.0, 1.0, "C0", "greedy"))
    records, flags, coverage = S.load_available()
    assert coverage == {"C0": ["greedy"]} and len(records) == 8 and flags["direct_equals_a0"] and records[0]["radius_m"] == 5.0


def test_cell_is_atomic_verified_restart_skips_and_detects_tampering(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "CELL_DIR", tmp_path)
    monkeypatch.setattr(S, "SEEDS", (145000, 145001))
    monkeypatch.setattr(S, "EQUIV_SEEDS_PER_CELL", 2)
    task = (7.5, 1.0, "C0", "greedy")
    first = S.run_cell(task)
    assert first["episodes"] == 16 and first["direct_equals_a0"] and first["deterministic"] and S.verify_cell(S.cell_path(*task))
    assert S.run_cell(task) == {"task": task, "skipped": True}
    path = S.cell_path(*task)
    payload = json.loads(path.read_text())
    payload["records"][1]["collision"] = not payload["records"][1]["collision"]
    path.write_text(json.dumps(payload))
    assert not S.verify_cell(path)
    path.write_text("{trunc")
    assert not S.verify_cell(path)
    assert "skipped" not in S.run_cell(task) and S.verify_cell(path)
    assert not list(tmp_path.glob(".*tmp*")) and not list(tmp_path.glob("*.lock"))


def test_duplicate_launch_is_prevented_by_the_cell_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "CELL_DIR", tmp_path)
    monkeypatch.setattr(S, "SEEDS", (145000,))
    task = (5.0, 2.0, "C0", "pn")
    holder = TS.CellLock(S.cell_path(*task))
    assert holder.acquire()
    result = S.run_cell(task)
    holder.release()
    assert result.get("skipped") and not S.cell_path(*task).exists()


def test_task_state_has_twelve_stages_and_restart_logic(tmp_path, monkeypatch):
    assert len(TS.STAGES) == 12 and TS.STAGES[0] == "01_repository_verification" and TS.STAGES[-1] == "12_commit_tag_push"
    monkeypatch.setattr(TS, "STATE_PATH", tmp_path / "task_state.json")
    rel = "docs/journal_v2_a1_route_level_v1/_restart_probe.txt"
    probe = TS.PROJECT_ROOT / rel
    probe.parent.mkdir(parents=True, exist_ok=True)
    try:
        probe.write_text("v1")
        state = TS.load_state()
        TS.begin(state, TS.STAGES[0])
        assert not TS.stage_is_verified(state, TS.STAGES[0])
        TS.complete(state, TS.STAGES[0], [rel], "ok")
        assert TS.next_stage(TS.load_state()) == TS.STAGES[1]
        probe.write_text("v2")
        assert TS.next_stage(TS.load_state()) == TS.STAGES[0]
        probe.write_text("v1")
        TS.halt(state, "06_unit_tests", "D", "test")
        later = TS.load_state()
        assert later["stages"]["07_deterministic_route_sweep"]["status"] == "not_applicable" and later["stages"]["11_publication_documentation"]["status"] == "pending"
    finally:
        probe.unlink(missing_ok=True)


def _rec(policy, level="C1", defender="greedy", seed=1, collision=False, base_collision=False, exhausted=False, div=2.0, conv=False, **kw):
    r = {"level": level, "defender": defender, "policy": policy, "seed": seed, "outcome": "soldier_caught" if conv else "intercepted", "steps": 10, "collision": collision,
         "exhausted": exhausted, "timeout": False, "captured": conv, "intercepted": not conv, "path_excess": 1.1, "path_length_m": 40.0, "min_sep": 3.0, "ceiling_hover": False,
         "sustained_hover": False, "orbit": 0.0, "overshoots": 0, "finite": True, "route_steps": 4, "longest_route_run": 5, "plans_attempted": 3, "plans_committed": 2, "replans": 1,
         "completions": 1, "invalid": {"defender_not_ahead": 1}, "cancels": {}, "suppressed_by_obstacle": 0, "waypoint_mean": 3.0, "route_length_mean": 60.0, "max_plans_in_4_steps": 2,
         "executed_while_owned": 0, "steps_info_available": 4, "plan_calls": 3, "plan_calls_without_info": 0, "ids_outside_perceived": 0, "plan_calls_with_unperceived": 0,
         "div_m": 0.0 if policy == "DIRECT" else div, "base_outcome": "intercepted", "outcome_differs": conv, "base_collision": base_collision, "min_sep_diff": 0.0, "steps_diff": 0,
         "path_diff_m": 0.0, "base_steps": 10, "radius_m": 7.5, "replan_s": 1.0}
    r.update(kw)
    return r


def _synthetic(bad=False, weak=False, forbid=False):
    records = []
    for level in S.LEVELS:
        for seed in range(30):
            records.append(_rec("DIRECT", level=level, seed=seed))
            for p in S.TACTICAL:
                records.append(_rec(p, level=level, seed=seed, collision=(bad and p == "RIGHT" and level == "C2" and seed == 0), conv=(not weak and seed < 10),
                                    div=0.1 if weak else 2.0))
    return records


FLAGS = {"direct_equals_a0": True, "deterministic_replay": True}


def test_gates_flag_a_single_gained_collision_weak_authority_and_forbidden_references():
    good = S.analyze(_synthetic(), FLAGS, forbidden=[])["configs"][0]
    assert good["G1"] and good["G2"] and good["G3_positive"] and good["G4"] and good["G5"] and good["G6"] and good["passed"]
    bad = S.analyze(_synthetic(bad=True), FLAGS, forbidden=[])["configs"][0]
    assert bad["G1"] is False and bad["gained_collisions_total"] == 1 and not bad["passed"]
    weak = S.analyze(_synthetic(weak=True), FLAGS, forbidden=[])
    assert weak["configs"][0]["G3"] is False and weak["selected"] is None
    leak = S.analyze(_synthetic(), FLAGS, forbidden=["_obstacle_layout"])["configs"][0]
    assert leak["G5"] is False and not leak["passed"]
    no_a0 = S.analyze(_synthetic(), dict(FLAGS, direct_equals_a0=False), forbidden=[])["configs"][0]
    assert no_a0["G5"] is False


def test_g6_flags_unbounded_route_runs_and_rapid_replanning():
    long_run = [_rec("DIRECT")] + [_rec("LEFT", longest_route_run=41)]
    assert S.analyze(long_run, FLAGS, forbidden=[])["configs"][0]["G6"] is False
    rapid = [_rec("DIRECT")] + [_rec("LEFT", max_plans_in_4_steps=4)]
    assert S.analyze(rapid, FLAGS, forbidden=[])["configs"][0]["G6"] is False
