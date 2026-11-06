"""Tests for the simplified two-scenario episode backend (No Clutter / Clutter), brief section 19.

Fast tests (model validation, store/service logic, API wiring) use a synthetic `research_root` fixture and a
disposable temporary `EpisodeStore`, never the user's real episode library (brief section 18). The handful of
tests that actually RUN an episode (`needs_research`, mirroring the established pattern in
`test_passive_defender.py`) use the real `VISUALIZER_RESEARCH_DATA_ROOT`/`VISUALIZER_SIMULATION_PYTHON` from the
local `.env`, skipped gracefully if unset, and ALWAYS write into a temporary episode store.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from visualizer.backend.config import Settings, load_settings
from visualizer.backend.episode_models import EpisodeCreateRequest, SCENARIO_PRESETS, SUPPORTED_DEFENDERS
from visualizer.backend.episode_service import EpisodeService
from visualizer.backend.episode_store import EpisodeStore
from visualizer.backend.main import create_app

RESEARCH_ROOT = os.environ.get("VISUALIZER_RESEARCH_DATA_ROOT")
needs_research = pytest.mark.skipif(
    not RESEARCH_ROOT, reason="Set VISUALIZER_RESEARCH_DATA_ROOT to run the real simulator tests.",
)


@pytest.fixture
def research_root(tmp_path: Path) -> Path:
    root = tmp_path / "research"
    (root / "uav_defend" / "envs").mkdir(parents=True)
    (root / "uav_defend" / "envs" / "soldier_env.py").write_text("# fixture, not a simulation.\n", encoding="utf-8")
    return root


@pytest.fixture
def isolated_store(tmp_path: Path) -> EpisodeStore:
    return EpisodeStore(tmp_path / "episodes")


@pytest.fixture
def isolated_settings(research_root: Path) -> Settings:
    return Settings(research_data_root=research_root)


def _client(settings: Settings, store: EpisodeStore) -> TestClient:
    service = EpisodeService(settings, store)
    return TestClient(create_app(settings, episode_service=service))


# ----------------------------------------------------------------------------- 3: attacker cannot be overridden
def test_attacker_cannot_be_overridden_through_simplified_api():
    with pytest.raises(ValidationError):
        EpisodeCreateRequest(scenario="no_clutter", defender="greedy", attacker="A1-CBF")  # type: ignore[call-arg]


def test_scenario_presets_have_exactly_the_two_expected_mappings():
    assert set(SCENARIO_PRESETS) == {"no_clutter", "clutter"}
    assert SCENARIO_PRESETS["no_clutter"].resolved_attacker == "A0"
    assert SCENARIO_PRESETS["no_clutter"].clutter_level == "C0"
    assert SCENARIO_PRESETS["no_clutter"].viability_conditioned_init is False
    assert SCENARIO_PRESETS["clutter"].resolved_attacker == "A1-CBF"
    assert SCENARIO_PRESETS["clutter"].clutter_level == "C2"
    assert SCENARIO_PRESETS["clutter"].viability_conditioned_init is True
    assert SCENARIO_PRESETS["clutter"].defender_cbf_enabled is True


# ----------------------------------------------------------------------------- 4: defender selection validation
def test_defender_selection_validation(isolated_settings, isolated_store):
    with _client(isolated_settings, isolated_store) as client:
        response = client.post("/api/episodes", json={"scenario": "no_clutter", "defender": "not_a_defender"})
        assert response.status_code == 422
    for defender in SUPPORTED_DEFENDERS:
        EpisodeCreateRequest(scenario="no_clutter", defender=defender)  # must not raise


# ----------------------------------------------------------------------------- 8: atomic episode persistence
def test_atomic_episode_persistence_leaves_no_temp_files(isolated_store):
    from visualizer.backend.episode_models import EpisodeMetadata

    metadata = EpisodeMetadata(
        episode_id="a" * 32, created_at="2024-01-01T00:00:00+00:00", scenario="no_clutter",
        scenario_label="No Clutter", resolved_attacker="A0", defender="greedy", seed=1, status="completed",
        outcome="soldier_caught", duration=1.0, steps=2, asset_captured=True, defender_interception=False,
        obstacle_collision=False, replay_path="a" * 32, source_commit="deadbeef", controller_version="baseline",
        is_new=True,
    )
    isolated_store.create(metadata, {"schema": "app_episode_replay/1", "outcome": "soldier_caught", "obstacles": [], "steps": []})
    leftovers = list(isolated_store.metadata_dir.glob("*.tmp")) + list(isolated_store.replays_dir.glob("*.tmp"))
    assert leftovers == []
    assert (isolated_store.metadata_dir / f"{metadata.episode_id}.json").exists()
    assert (isolated_store.replays_dir / f"{metadata.episode_id}.json").exists()


def _seed_episode(store: EpisodeStore, *, scenario="no_clutter", defender="greedy", is_new=True) -> str:
    from uuid import uuid4

    from visualizer.backend.episode_models import EpisodeMetadata

    episode_id = uuid4().hex
    metadata = EpisodeMetadata(
        episode_id=episode_id, created_at="2024-01-01T00:00:00+00:00", scenario=scenario,
        scenario_label=SCENARIO_PRESETS[scenario].scenario_label, resolved_attacker=SCENARIO_PRESETS[scenario].resolved_attacker,
        defender=defender, seed=1, status="completed", outcome="soldier_caught", duration=1.0, steps=2,
        asset_captured=True, defender_interception=False, obstacle_collision=False, replay_path=episode_id,
        source_commit="deadbeef", controller_version="baseline", is_new=is_new,
    )
    store.create(metadata, {"schema": "app_episode_replay/1", "outcome": "soldier_caught", "obstacles": [], "steps": []})
    return episode_id


# ----------------------------------------------------------------------------- 9/10: new vs saved listing
def test_new_and_saved_episode_listing(isolated_settings, isolated_store):
    new_id = _seed_episode(isolated_store, is_new=True)
    saved_id = _seed_episode(isolated_store, is_new=False)
    with _client(isolated_settings, isolated_store) as client:
        new_resp = client.get("/api/episodes", params={"new": "true"}).json()
        saved_resp = client.get("/api/episodes", params={"new": "false"}).json()
        assert {e["episode_id"] for e in new_resp["episodes"]} == {new_id}
        assert {e["episode_id"] for e in saved_resp["episodes"]} == {saved_id}
        all_resp = client.get("/api/episodes").json()
        assert all_resp["new_count"] == 1 and all_resp["saved_count"] == 1


# ----------------------------------------------------------------------------- 11: opening marks viewed
def test_opening_an_episode_marks_it_viewed(isolated_settings, isolated_store):
    episode_id = _seed_episode(isolated_store, is_new=True)
    with _client(isolated_settings, isolated_store) as client:
        before = client.get("/api/episodes", params={"new": "true"}).json()
        assert episode_id in {e["episode_id"] for e in before["episodes"]}
        opened = client.get(f"/api/episodes/{episode_id}")
        assert opened.status_code == 200
        assert opened.json()["metadata"]["is_new"] is False
        after = client.get("/api/episodes", params={"new": "true"}).json()
        assert episode_id not in {e["episode_id"] for e in after["episodes"]}


# ----------------------------------------------------------------------------- 12: single delete
def test_single_delete(isolated_settings, isolated_store):
    episode_id = _seed_episode(isolated_store)
    with _client(isolated_settings, isolated_store) as client:
        response = client.delete(f"/api/episodes/{episode_id}")
        assert response.status_code == 200 and response.json()["deleted"] is True
        assert client.get(f"/api/episodes/{episode_id}").status_code == 404


# ----------------------------------------------------------------------------- 13: bulk delete
def test_bulk_delete(isolated_settings, isolated_store):
    ids = [_seed_episode(isolated_store) for _ in range(3)]
    with _client(isolated_settings, isolated_store) as client:
        response = client.post("/api/episodes/delete", json={"episode_ids": ids + ["b" * 32]})
        assert response.status_code == 200
        results = {r["episode_id"]: r["deleted"] for r in response.json()["results"]}
        assert all(results[i] for i in ids)
        assert results["b" * 32] is False
        assert client.get("/api/episodes").json()["total"] == 0


# ----------------------------------------------------------------------------- 14: missing episode delete
def test_missing_episode_delete_returns_not_found(isolated_settings, isolated_store):
    with _client(isolated_settings, isolated_store) as client:
        response = client.delete(f"/api/episodes/{'c' * 32}")
        assert response.status_code == 404


# ----------------------------------------------------------------------------- 15: path traversal protection
@pytest.mark.parametrize("bad_id", ["../../etc/passwd", "..%2f..%2fetc", "a" * 31, "A" * 32, "a/b" + "c" * 29])
def test_path_traversal_protection(isolated_store, bad_id):
    from visualizer.backend.application_error import ApplicationError

    with pytest.raises(ApplicationError):
        isolated_store.get(bad_id)
    with pytest.raises(ApplicationError):
        isolated_store.delete(bad_id)


# ----------------------------------------------------------------------------- 16: shared artifact protection
def test_delete_never_touches_other_files(isolated_store, tmp_path):
    keep_id = _seed_episode(isolated_store)
    victim_id = _seed_episode(isolated_store)
    sentinel = isolated_store.root.parent / "shared_checkpoint.zip"
    sentinel.write_bytes(b"not an episode file")
    isolated_store.delete(victim_id)
    assert sentinel.exists()
    assert (isolated_store.metadata_dir / f"{keep_id}.json").exists()
    sentinel.unlink()


# ----------------------------------------------------------------------------- 17: precomputed CBF replay loading (regression)
def test_precomputed_cbf_review_collection_endpoint_is_unaffected(isolated_settings, isolated_store):
    with _client(isolated_settings, isolated_store) as client:
        response = client.get("/api/cbf-review/collection")
        assert response.status_code == 200  # route still exists and responds; content depends on the research fixture


# ----------------------------------------------------------------------------- 19: existing SB3 replay compatibility (regression)
def test_existing_replay_and_case_routes_are_unaffected(isolated_settings, isolated_store):
    with _client(isolated_settings, isolated_store) as client:
        assert client.get("/api/replays").status_code == 200
        assert client.get("/api/cases").status_code == 200


@needs_research
def test_existing_scenarios_route_is_unaffected_by_the_new_episode_api(real_settings, real_store):
    with _client(real_settings, real_store) as client:
        assert client.get("/api/scenarios").status_code == 200
        assert client.get("/api/policies").status_code == 200


# ----------------------------------------------------------------------------- 21: backend health
def test_backend_health_reports_new_episode_fields(isolated_settings, isolated_store):
    _seed_episode(isolated_store, is_new=True)
    with _client(isolated_settings, isolated_store) as client:
        response = client.get("/api/health")
        body = response.json()
        assert response.status_code == 200
        for key in ("research_source_available", "a0_available", "a1_cbf_available", "episode_storage_writable", "new_episode_count", "saved_episode_count"):
            assert key in body
        assert body["new_episode_count"] == 1
        assert body["research_source_available"] is True  # synthetic fixture marker file is present
        assert body["a0_available"] is True  # soldier_env.py present
        assert body["a1_cbf_available"] is False  # no cbf_obstacle.py in the synthetic fixture


# ----------------------------------------------------------------------------- 22: test-store isolation
def test_store_isolation_never_touches_the_real_episode_directory(isolated_store):
    from visualizer.backend.config import EPISODE_STORE_ROOT

    assert isolated_store.root != EPISODE_STORE_ROOT.resolve()
    assert not str(isolated_store.root).startswith(str(EPISODE_STORE_ROOT.resolve()))


# ===================================================================================== real-simulator tests
@pytest.fixture
def real_settings() -> Settings:
    return load_settings()


@pytest.fixture
def real_store(tmp_path: Path) -> EpisodeStore:
    return EpisodeStore(tmp_path / "real_episodes")


# ----------------------------------------------------------------------------- 1/18: No Clutter -> C0 + A0
@needs_research
def test_no_clutter_preset_resolves_to_c0_and_a0_and_replay_loads(real_settings, real_store):
    with _client(real_settings, real_store) as client:
        response = client.post("/api/episodes", json={"scenario": "no_clutter", "defender": "passive", "seed": 1})
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["scenario"] == "no_clutter" and body["resolved_attacker"] == "A0"
        assert body["status"] == "completed"
        replay = client.get(f"/api/episodes/{body['episode_id']}").json()
        assert replay["replay"]["schema"] == "app_episode_replay/1"
        assert len(replay["replay"]["steps"]) >= 1


# ----------------------------------------------------------------------------- 2: Clutter -> C2 + A1-CBF
@needs_research
def test_clutter_preset_resolves_to_c2_and_a1_cbf(real_settings, real_store):
    with _client(real_settings, real_store) as client:
        response = client.post("/api/episodes", json={"scenario": "clutter", "defender": "greedy", "seed": 1})
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["scenario"] == "clutter" and body["resolved_attacker"] == "A1-CBF"
        assert body["status"] == "completed"


# ----------------------------------------------------------------------------- 5: seeded run determinism
@needs_research
def test_seeded_run_is_deterministic(real_settings, real_store):
    with _client(real_settings, real_store) as client:
        first = client.post("/api/episodes", json={"scenario": "no_clutter", "defender": "passive", "seed": 42}).json()
        second = client.post("/api/episodes", json={"scenario": "no_clutter", "defender": "passive", "seed": 42}).json()
        assert first["outcome"] == second["outcome"]
        assert first["steps"] == second["steps"]
        assert first["seed"] == second["seed"] == 42


# ----------------------------------------------------------------------------- 6: automatically generated seed
@needs_research
def test_automatically_generated_seed_when_omitted(real_settings, real_store):
    with _client(real_settings, real_store) as client:
        response = client.post("/api/episodes", json={"scenario": "no_clutter", "defender": "passive"})
        assert response.status_code == 201
        assert isinstance(response.json()["seed"], int)


# ----------------------------------------------------------------------------- 7: episode metadata creation
@needs_research
def test_episode_metadata_has_all_required_fields(real_settings, real_store):
    with _client(real_settings, real_store) as client:
        body = client.post("/api/episodes", json={"scenario": "no_clutter", "defender": "passive", "seed": 7}).json()
        required = (
            "episode_id", "created_at", "scenario", "scenario_label", "resolved_attacker", "defender", "seed",
            "status", "outcome", "duration", "steps", "asset_captured", "defender_interception",
            "obstacle_collision", "replay_path", "source_commit", "controller_version", "is_new",
        )
        for key in required:
            assert key in body, key


# ----------------------------------------------------------------------------- 20: source/tag metadata
@needs_research
def test_clutter_episode_records_the_frozen_a1_cbf_tag_and_commit(real_settings, real_store):
    from visualizer.backend.episode_models import A1_CBF_SOURCE_COMMIT, A1_CBF_TAG

    with _client(real_settings, real_store) as client:
        body = client.post("/api/episodes", json={"scenario": "clutter", "defender": "greedy", "seed": 7}).json()
        assert body["source_commit"] == A1_CBF_SOURCE_COMMIT
        assert body["controller_version"] == A1_CBF_TAG
