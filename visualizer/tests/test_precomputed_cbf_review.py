"""Tests for the A1-CBF defender-review precomputed trajectory path (journal_cbf_attacker, defender_cbf /
Prompt 4 manual review). SEPARATE from the existing strict ReplayBundle/CaseLibrary tests -- see
visualizer/backend/precomputed_cases.py's module docstring for why a parallel, lighter contract is used.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from visualizer.backend.config import Settings
from visualizer.backend.main import create_app
from visualizer.backend.precomputed_cases import (
    PrecomputedCaseProvider,
    RESEARCH_RELATIVE_DIR,
    _sanitize,
    _summarize,
)
from visualizer.backend.replay_engine import EXPECTED_RESEARCH_COMMIT

VALID_CASE = {
    "defender": "greedy",
    "seed": 600001,
    "defender_cbf_enabled": True,
    "outcome": "soldier_caught",
    "obstacles": [{"min": [1.0, 2.0, 0.0], "max": [3.0, 4.0, 5.0]}],
    "steps": [
        {
            "t": 0.0, "attacker_pos": [0.0, 0.0, 10.0], "defender_pos": [20.0, 0.0, 10.0], "asset_pos": [0.0, 0.0, 0.0],
            "defender_observable": False, "intervention_source": "none", "cbf_qp_status": "inactive",
            "defender_barrier_h": None, "min_inflated_obstacle_clearance": float("inf"),
        },
        {
            "t": 0.5, "attacker_pos": [1.0, 0.0, 10.0], "defender_pos": [18.0, 0.0, 10.0], "asset_pos": [0.1, 0.0, 0.0],
            "defender_observable": True, "intervention_source": "defender", "cbf_qp_status": "hard",
            "defender_barrier_h": 2.5, "min_inflated_obstacle_clearance": 7.3,
        },
    ],
    "what_to_look_for": "Test fixture: does the attacker react to the defender becoming observable?",
    "schema": "a1_cbf_defender_cbf_visualizer_trajectory/1",
}


@pytest.fixture
def research_root(tmp_path: Path) -> Path:
    root = tmp_path / "research"
    (root / "uav_defend" / "envs").mkdir(parents=True)
    (root / "uav_defend" / "envs" / "soldier_env.py").write_text("# fixture, not a simulation.\n")
    return root


@pytest.fixture
def populated_root(research_root: Path) -> Path:
    artifacts = research_root / RESEARCH_RELATIVE_DIR
    artifacts.mkdir(parents=True)
    for stem in (
        "01_representative_defender_evasion", "02_lateral_pass", "03_vertical_climb_avoidance",
        "04_speed_reduction_evasion", "05_joint_obstacle_defender_maneuver", "06_defender_interception_failure",
        "07_asset_capture_after_evasion", "08_hardest_successful_stress_case",
        "09_obstacle_only_matched_reference", "10_full_cbf_matched_case",
    ):
        payload = dict(VALID_CASE)
        (artifacts / f"{stem}.json").write_text(json.dumps(payload))
    return research_root


# ----------------------------------------------------------------------------- 1: precomputed trajectory loading
def test_collection_loads_all_ten_cases(populated_root: Path):
    provider = PrecomputedCaseProvider(Settings(research_data_root=populated_root))
    collection = provider.collection()
    assert collection["label"].endswith("NOT FROZEN")
    assert len(collection["cases"]) == 10
    assert collection["warnings"] == []


def test_case_detail_includes_steps_and_obstacles(populated_root: Path):
    provider = PrecomputedCaseProvider(Settings(research_data_root=populated_root))
    case = provider.case("01_representative_defender_evasion")
    assert case is not None
    assert len(case["steps"]) == 2
    assert len(case["obstacles"]) == 1
    assert case["steps"][1]["attacker_defender_distance_m_derived"] == pytest.approx(17.0)


def test_unknown_case_id_returns_none(populated_root: Path):
    provider = PrecomputedCaseProvider(Settings(research_data_root=populated_root))
    assert provider.case("does-not-exist") is None


def test_missing_research_root_yields_empty_collection():
    provider = PrecomputedCaseProvider(Settings())
    collection = provider.collection()
    assert collection["cases"] == []
    assert collection["opening_case_id"] is None


# ----------------------------------------------------------------------------- 2: malformed JSON rejection
def test_malformed_json_file_is_skipped_not_fatal(populated_root: Path):
    artifacts = populated_root / RESEARCH_RELATIVE_DIR
    (artifacts / "11_malformed.json").write_text("{not valid json")
    provider = PrecomputedCaseProvider(Settings(research_data_root=populated_root))
    collection = provider.collection()
    assert len(collection["cases"]) == 10  # the 10 valid files are still all present
    assert any("11_malformed.json" in w for w in collection["warnings"])


def test_json_missing_steps_array_is_skipped(populated_root: Path):
    artifacts = populated_root / RESEARCH_RELATIVE_DIR
    (artifacts / "12_no_steps.json").write_text(json.dumps({"defender": "greedy"}))
    provider = PrecomputedCaseProvider(Settings(research_data_root=populated_root))
    collection = provider.collection()
    assert len(collection["cases"]) == 10
    assert any("12_no_steps.json" in w for w in collection["warnings"])


def test_infinity_tokens_are_sanitized_to_null():
    sanitized = _sanitize({"a": float("inf"), "b": float("-inf"), "c": float("nan"), "d": 1.5, "e": [float("inf"), 2.0]})
    assert sanitized == {"a": None, "b": None, "c": None, "d": 1.5, "e": [None, 2.0]}


# ----------------------------------------------------------------------------- 3: collection registration
def test_collection_is_labeled_development_not_frozen(populated_root: Path):
    provider = PrecomputedCaseProvider(Settings(research_data_root=populated_root))
    collection = provider.collection()
    assert "DEVELOPMENT" in collection["label"] and "NOT FROZEN" in collection["label"]


def test_cases_are_ordered_by_documented_priority_rank(populated_root: Path):
    provider = PrecomputedCaseProvider(Settings(research_data_root=populated_root))
    ranks = [c["rank"] for c in provider.collection()["cases"]]
    assert ranks == sorted(ranks)
    assert ranks == list(range(1, 11))


def test_opening_case_matches_section_nine_fallback_rule(populated_root: Path):
    provider = PrecomputedCaseProvider(Settings(research_data_root=populated_root))
    collection = provider.collection()
    assert collection["opening_case_id"] == "06_defender_interception_failure"


# ----------------------------------------------------------------------------- 4: existing SB3 replay compatibility
def test_existing_replay_and_case_routes_are_unaffected():
    app = create_app(Settings())
    paths = {route.path for route in app.routes}
    assert "/api/replays" in paths
    assert "/api/replays/{replay_id}" in paths
    assert "/api/cases" in paths
    assert "/api/cbf-review/collection" in paths
    assert "/api/cbf-review/cases/{case_id}" in paths


def test_health_endpoint_still_returns_200_with_new_routes_present():
    with TestClient(create_app(Settings())) as client:
        response = client.get("/api/health")
        assert response.status_code == 200


# ----------------------------------------------------------------------------- 5: research-source pin
def test_research_pin_matches_the_cbf_attacker_commit():
    assert EXPECTED_RESEARCH_COMMIT == "a989217b38dc0316718d04e41b065943fec29ce7"


# ----------------------------------------------------------------------------- 6: outcome metadata
def test_summarize_derives_outcome_metadata_without_inventing_values():
    summary = _summarize("case_x", VALID_CASE)
    assert summary["outcome"] == "soldier_caught"
    assert summary["asset_captured"] is True
    assert summary["defender_intercepted"] is False
    assert summary["obstacle_collision"] is False
    assert summary["min_defender_separation_m"] == pytest.approx(17.0)
    assert summary["min_obstacle_clearance_m"] == pytest.approx(7.3)  # inf at step 0 excluded, never fabricated
    assert summary["episode_duration_s"] == pytest.approx(0.5)
    assert summary["step_count"] == 2


def test_summarize_handles_missing_outcome_without_fabricating_booleans():
    payload = dict(VALID_CASE)
    payload["outcome"] = None
    summary = _summarize("case_y", payload)
    assert summary["asset_captured"] is None
    assert summary["defender_intercepted"] is None
    assert summary["obstacle_collision"] is None


# ----------------------------------------------------------------------------- API-level integration
def test_api_collection_and_case_endpoints(populated_root: Path):
    with TestClient(create_app(Settings(research_data_root=populated_root))) as client:
        collection_response = client.get("/api/cbf-review/collection")
        assert collection_response.status_code == 200
        body = collection_response.json()
        assert len(body["cases"]) == 10
        opening = body["opening_case_id"]
        case_response = client.get(f"/api/cbf-review/cases/{opening}")
        assert case_response.status_code == 200
        assert "steps" in case_response.json()
        missing_response = client.get("/api/cbf-review/cases/not-a-real-case")
        assert missing_response.status_code == 404
