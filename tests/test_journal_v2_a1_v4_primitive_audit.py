"""Locks in the primitive-audit facts behind the verdict that the existing guidance primitives cannot support defender-driven
BASE/LEFT/RIGHT/CLIMB tactical maneuvers without new geometry (journal_v2_a1_discrete_v4). No PPO."""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

import experiments.journal_v2_a1_task_state as TS
import experiments.journal_v2_a1_v4_primitive_audit as P
from uav_defend.obstacles import stateful_avoidance as SA
from uav_defend.obstacles.geometry import AABBObstacle


def test_no_guidance_primitive_takes_a_defender_input():
    facts = P.signature_facts()
    assert all(not v["has_defender_input"] for v in facts.values())
    assert "mode" not in inspect.signature(SA.StatefulAvoidanceController.plan).parameters  # the mode cannot be requested


def test_in_clear_space_left_right_climb_are_exactly_the_mission_command():
    facts = P.clear_space_facts(500)
    assert facts["bit_identical_to_mission_command_without_blockers"] == {"left": 500, "right": 500, "climb": 500}


def test_climb_is_finite_and_the_laws_are_pure():
    facts = P.statelessness_facts()
    assert facts["deterministic_pure_functions"] and facts["climb_ascends_only_while_below_blocker_top_plus_clearance"]
    assert facts["climb_capped_at_ceiling_holds_altitude"]


def test_left_and_right_are_geometric_functions_of_the_blocker_and_ignore_everything_else():
    lim = P.limits()
    u, pos = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 6.0])
    blocker = AABBObstacle(min_corner=np.array([4.0, -3.0, 0.0]), max_corner=np.array([9.0, 3.0, 12.0]))
    left = SA.guidance_command("left", pos, u, [blocker], 2.0, 30.0, lim)
    right = SA.guidance_command("right", pos, u, [blocker], 2.0, 30.0, lim)
    assert left[1] > 0.0 > right[1]  # steer to the corners of the blocker, one on each side
    moved = AABBObstacle(min_corner=blocker.min_corner + [0, 20, 0], max_corner=blocker.max_corner + [0, 20, 0])
    assert not np.allclose(SA.guidance_command("left", pos, u, [moved], 2.0, 30.0, lim), left)  # depends on the obstacle, not a defender


def test_v2_attacker_has_no_defender_driven_tactical_component():
    from experiments.journal_v2_a1 import A1V2TrainingEnv

    env = A1V2TrainingEnv(0, 0, scripted_a0=True)
    env.reset(options={"scenario": ("C0", "greedy", 140001)})
    assert env.soldier_env.config.enemy_nominal_guidance_mode == "direct_obstacle_only"
    done = False
    while not done:
        _o, _r, te, tr, info = env.step(np.zeros(3, np.float32))
        assert not np.any(np.asarray(info["enemy_evasion_command"]))  # legacy evasion is zeroed in V2
        done = te or tr


def test_existing_threat_gate_is_the_25m_observation_gate():
    assert P.THREAT_RANGE_M == 25.0 and P.EVASION_RADIUS_M == 20.0


def test_audit_conclusion_and_threat_window_evidence_if_available():
    path = TS.STATE_DIR / "primitive_audit.json"
    if not path.exists():
        pytest.skip("audit not generated")
    audit = json.loads(path.read_text())
    assert audit["conclusion"]["verdict"] == "D" and not audit["conclusion"]["defender_driven_left_right_climb_primitive_exists"]
    window = audit["threat_window"]
    assert window["C0"]["avoidance_active_steps"] == 0 and window["C1"]["avoidance_active_steps"] == 0
    assert all(window[lv]["fraction_of_threat_steps_with_primitives_defined"] < 0.05 for lv in window)


def test_restart_state_skips_verified_stages_and_reruns_tampered_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(TS, "STATE_DIR", tmp_path)
    monkeypatch.setattr(TS, "STATE_PATH", tmp_path / "task_state.json")
    monkeypatch.setattr(TS, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(TS, "git", lambda *a: "deadbeef" if a[:1] == ("rev-parse",) else "")
    state = TS.load_state()
    TS.begin(state, "01_repository_verification")
    artifact = tmp_path / "a.json"
    TS.atomic_write_json(artifact, {"x": 1})
    TS.complete(state, "01_repository_verification", ["a.json"])
    assert TS.stage_is_verified(TS.load_state(), "01_repository_verification")
    assert TS.next_stage(TS.load_state()) == "02_primitive_audit"
    artifact.write_text('{"x": 2}')  # artifact changed after completion
    assert not TS.stage_is_verified(TS.load_state(), "01_repository_verification")
    assert TS.next_stage(TS.load_state()) == "01_repository_verification"
    # a half-written (running) stage is never treated as complete, and an unreadable state file is not trusted
    s = TS.load_state()
    TS.begin(s, "02_primitive_audit")
    assert not TS.stage_is_verified(TS.load_state(), "02_primitive_audit")
    (tmp_path / "task_state.json").write_text("{not json")
    assert TS.next_stage(TS.load_state()) == "01_repository_verification"


def test_halt_marks_later_stages_not_applicable_but_keeps_evidence_stages_runnable(tmp_path, monkeypatch):
    monkeypatch.setattr(TS, "STATE_DIR", tmp_path)
    monkeypatch.setattr(TS, "STATE_PATH", tmp_path / "task_state.json")
    monkeypatch.setattr(TS, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(TS, "git", lambda *a: "deadbeef")
    state = TS.load_state()
    TS.halt(state, "02_primitive_audit", "D", "no defender-driven primitive")
    s = TS.load_state()
    for stage in ("03_architecture_implementation", "04_unit_tests", "05_stress_study_generation", "06_stress_study_analysis",
                  "07_protocol_documentation"):
        assert s["stages"][stage]["status"] == "not_applicable" and TS.stage_is_verified(s, stage)
    assert s["halt"]["verdict"] == "D" and s["stages"]["08_non_training_regression"]["status"] == "pending"
