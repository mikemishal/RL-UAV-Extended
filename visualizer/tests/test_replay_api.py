from __future__ import annotations

import gzip
import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from visualizer.backend.application_error import ApplicationError
from visualizer.backend.config import Settings
from visualizer.backend.main import create_app
from visualizer.backend.replay_contract import (
    EntityState,
    EnvironmentSpec,
    EpisodeSummary,
    PolicySelection,
    PolicyTelemetry,
    PolicyVisible,
    PrivilegedDiagnostics,
    ReplayBundle,
    ReplayEpisode,
    ReplayFrame,
    ReplayProvenance,
)
from visualizer.backend.replay_jobs import (
    ReplayJob,
    ReplayJobManager,
    _EpisodeTimeout,
    _JobCancelled,
    _replay_id_for_path,
)
from visualizer.backend.replay_requests import ReplayBatchRequest


def _catalog():
    supported = [{"clutter": level, "perception_modes": ["local", "map_local"]} for level in ("C0", "C1", "C2", "C3")]
    return {
        "source_commit": "a" * 40,
        "registry_sha256": "b" * 64,
        "deterministic_inference": True,
        "defenders": [
            {"policy_id": "greedy", "classification": "analytical", "available": True, "supported_scenarios": supported},
            {
                "policy_id": "lr_ppo",
                "classification": "learned",
                "available": True,
                "supported_scenarios": supported,
                "generations": [
                    {"generation": "d1", "training_seeds": [{"training_seed": 77, "available": True}]},
                ],
            },
        ],
        "attackers": [
            {"attacker_id": "A0", "available": True},
            {"attacker_id": "A1-203", "available": False},
        ],
        "scenarios": [
            {"clutter": "C0", "display_name": "Unobstructed", "perceptions": [{"id": "local"}, {"id": "map_local"}]},
        ],
    }


class FakeCatalog:
    def get(self):
        return _catalog()


def _bundle(seed: int = 41) -> ReplayBundle:
    visible = PolicyVisible(
        estimator="measurement",
        observation=[0.0] * 16,
        soldier_pos_m=(0.0, 0.0, 0.0),
        defender_pos_m=(0.0, 0.0, 1.0),
        defender_vel_mps=(0.0, 0.0, 0.0),
        enemy_detected=False,
    )
    defender = PolicySelection(
        policy_kind="analytical",
        policy_id="greedy",
        architecture="greedy",
        estimator="measurement",
        parameters={},
    )
    attacker = PolicySelection(
        policy_kind="scripted",
        policy_id="A0",
        architecture="canonical_scripted",
        estimator="scripted_ground_truth",
        parameters={},
    )
    environment = EnvironmentSpec(
        clutter="C0",
        awareness="map_local",
        builder="experiments.journal_clutter_protocol.build_canonical_env_config",
        config={},
        config_sha256="c" * 64,
        dt_s=0.5,
    )
    frames = []
    for step in range(2):
        final = step == 1
        frames.append(ReplayFrame(
            step=step,
            time_s=step * 0.5,
            defender=EntityState(position_m=(0.0, 0.0, 1.0), velocity_mps=(0.0, 0.0, 0.0)),
            attacker=EntityState(position_m=(1.0, 0.0, 1.0), velocity_mps=(0.0, 0.0, 0.0)),
            asset=EntityState(position_m=(0.0, 0.0, 0.0), velocity_mps=(0.0, 0.0, 0.0)),
            moving_obstacles=[],
            policy_visible=visible,
            telemetry=None if step == 0 else PolicyTelemetry(
                decision_step=0,
                decision_policy_visible=visible,
                requested_action=(1.0, 0.0, 0.0),
                requested_action_norm=1.0,
                defender_policy_input=[0.0] * 16,
                controller_action_executed=True,
                defender_standby=False,
                avoidance_active=False,
                executed_velocity_mps=(0.0, 0.0, 0.0),
            ),
            privileged_diagnostics=PrivilegedDiagnostics(
                attacker_position_m=(1.0, 0.0, 1.0),
                attacker_velocity_mps=(0.0, 0.0, 0.0),
                defender_attacker_range_m=1.0,
                attacker_asset_range_m=1.0,
                simulator_info={},
            ),
            detection_radius_m=15.0,
            detection_time_step=None,
            defender_action_applied=final,
            cumulative_reward=1.0 if final else 0.0,
            environmental_state={},
            reward=1.0 if final else None,
            terminated=final,
            truncated=False,
            outcome="intercepted" if final else "ongoing",
        ))
    episode = ReplayEpisode(
        episode_id=f"{seed}-greedy-analytical-script-A0-C0",
        simulation_seed=seed,
        defender=defender,
        attacker=attacker,
        environment=environment,
        terminal_flags={"terminated": True, "truncated": False},
        static_obstacles=[],
        frames=frames,
        events=[],
        summary=EpisodeSummary(
            outcome="intercepted",
            steps=1,
            duration_s=0.5,
            final_defender_attacker_distance_m=1.0,
            final_attacker_asset_distance_m=1.0,
            total_reward=1.0,
        ),
    )
    return ReplayBundle(
        provenance=ReplayProvenance(
            simulation_source_commit="a" * 40,
            visualizer_source_commit="b" * 40,
            source_dirty=False,
            visualizer_dirty=False,
            python_version=sys.version.split()[0],
            package_versions={},
            model_device="cpu",
            registry_sha256="c" * 64,
        ),
        seed_start=seed,
        episodes=[episode],
    )


@pytest.fixture
def research_root(tmp_path):
    source = tmp_path / "research" / "uav_defend" / "envs" / "soldier_env.py"
    source.parent.mkdir(parents=True)
    source.write_text("# API test fixture\n", encoding="utf-8")
    return source.parents[2]


def _settings(research_root: Path) -> Settings:
    return Settings(research_data_root=research_root, simulation_python=Path(sys.executable))


def _request(**overrides) -> ReplayBatchRequest:
    payload = {
        "defender_policy_id": "greedy",
        "attacker_condition": "A0",
        "clutter_level": "C0",
        "perception_mode": "map_local",
        "seed_start": 0,
        "episode_count": 3,
    }
    payload.update(overrides)
    return ReplayBatchRequest(**payload)


def test_catalog_api_returns_verified_metadata_without_paths(research_root):
    settings = _settings(research_root)
    with TestClient(create_app(settings, catalog_provider=FakeCatalog())) as client:
        policies = client.get("/api/policies")
        scenarios = client.get("/api/scenarios")
        assert policies.status_code == 200
        assert {item["policy_id"] for item in policies.json()["defenders"]} == {"greedy", "lr_ppo"}
        assert policies.json()["attackers"][1]["available"] is False
        assert scenarios.json()["scenarios"][0]["clutter"] == "C0"
        assert str(research_root) not in policies.text + scenarios.text
        assert client.get("/api/health").json()["phase"] == 3


@pytest.mark.parametrize("payload", [
    {"defender_policy_id": "greedy", "attacker_condition": "A0", "clutter_level": "C0", "episode_count": 2},
    {"defender_policy_id": "greedy", "attacker_condition": "A0", "clutter_level": "C0", "seed_start": 2**32 - 1, "episode_count": 3},
    {"defender_policy_id": "greedy", "attacker_condition": "A0", "clutter_level": "C0", "seed_start": -1},
])
def test_batch_request_rejects_invalid_count_or_seed(research_root, payload):
    with TestClient(create_app(_settings(research_root), catalog_provider=FakeCatalog())) as client:
        response = client.post("/api/replay-jobs", json=payload)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    assert '"input":' not in response.text


def test_batch_request_defaults_to_five():
    request = ReplayBatchRequest(
        defender_policy_id="greedy", attacker_condition="A0", clutter_level="C0",
        perception_mode="local", seed_start=20,
    )
    assert request.episode_count == 5


@pytest.mark.parametrize("episode_count", [1, 3, 5, 10])
def test_allowed_batch_counts_expand_to_consecutive_seeds(episode_count):
    request = ReplayBatchRequest(
        defender_policy_id="greedy",
        attacker_condition="A0",
        clutter_level="C0",
        seed_start=20,
        episode_count=episode_count,
    )
    assert [request.seed_start + offset for offset in range(request.episode_count)] == list(
        range(20, 20 + episode_count)
    )


def test_unavailable_attacker_checkpoint_is_rejected(research_root):
    with TestClient(create_app(_settings(research_root), catalog_provider=FakeCatalog())) as client:
        response = client.post("/api/replay-jobs", json={
            "defender_policy_id": "greedy",
            "attacker_condition": "A1-203",
            "clutter_level": "C0",
            "episode_count": 1,
        })
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "ATTACKER_UNAVAILABLE"


def test_single_active_job_cancellation_and_busy_rejection(research_root, monkeypatch):
    import visualizer.backend.replay_jobs as jobs_module

    manager = ReplayJobManager(_settings(research_root), FakeCatalog())
    entered = threading.Event()

    def wait_until_cancelled(job_id: str, _seed: int, _replay_id: str) -> str:
        entered.set()
        assert manager._cancel[job_id].wait(5)
        raise _JobCancelled

    monkeypatch.setattr(manager, "_execute_episode", wait_until_cancelled)
    job = manager.start(_request())
    try:
        assert entered.wait(5)
        assert manager.get(job.job_id).state == "running"
        with pytest.raises(ApplicationError) as busy:
            manager.start(_request(seed_start=20))
        assert busy.value.code == "REPLAY_JOB_BUSY"
        cancelling = manager.cancel(job.job_id)
        assert cancelling.cancellation_requested
        deadline = time.monotonic() + 5
        while manager.get(job.job_id).state == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
        assert manager.get(job.job_id).state == "cancelled"
        assert manager.get(job.job_id).completed_count == 0
    finally:
        manager.shutdown()


def test_episode_timeout_is_failed_not_a_simulator_outcome(research_root, monkeypatch):
    manager = ReplayJobManager(_settings(research_root), FakeCatalog())
    monkeypatch.setattr(manager, "_execute_episode", lambda *_args: (_ for _ in ()).throw(_EpisodeTimeout("seed timed out")))
    job = manager.start(_request(episode_count=1))
    try:
        deadline = time.monotonic() + 5
        while manager.get(job.job_id).state in {"queued", "running"} and time.monotonic() < deadline:
            time.sleep(0.02)
        final = manager.get(job.job_id)
        assert final.state == "failed"
        assert final.error == "seed timed out"
        assert final.completed_count == 0
    finally:
        manager.shutdown()


def test_restart_marks_incomplete_job_interrupted(research_root, tmp_path, monkeypatch):
    import visualizer.backend.replay_jobs as jobs_module

    output = tmp_path / "local"
    output.mkdir()
    monkeypatch.setattr(jobs_module, "local_output_root", lambda: output)
    directory = output / "jobs"
    directory.mkdir()
    now = "2026-10-16T00:00:00+00:00"
    old = ReplayJob(
        job_id="1" * 32,
        request=_request(),
        state="running",
        requested_episode_count=3,
        completed_count=1,
        completed_replay_ids=["2" * 32],
        created_at=now,
        started_at=now,
        updated_at=now,
    )
    (directory / f"{old.job_id}.json").write_text(old.model_dump_json(), encoding="utf-8")
    manager = ReplayJobManager(_settings(research_root), FakeCatalog())
    try:
        recovered = manager.get(old.job_id)
        assert recovered.state == "interrupted"
        assert recovered.completed_count == 1
        assert recovered.completed_replay_ids == ["2" * 32]
    finally:
        manager.shutdown()


def test_job_metadata_persistence_retries_transient_windows_lock(research_root, tmp_path, monkeypatch):
    import visualizer.backend.replay_jobs as jobs_module

    output = tmp_path / "local"
    output.mkdir()
    monkeypatch.setattr(jobs_module, "local_output_root", lambda: output)
    manager = ReplayJobManager(_settings(research_root), FakeCatalog())
    original_replace = jobs_module.os.replace
    attempts = 0

    def transient_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("temporary Windows file lock")
        original_replace(source, destination)

    monkeypatch.setattr(jobs_module.os, "replace", transient_replace)
    job = ReplayJob(
        job_id="3" * 32,
        request=_request(),
        state="queued",
        requested_episode_count=3,
        created_at="2026-10-16T00:00:00+00:00",
        updated_at="2026-10-16T00:00:00+00:00",
    )
    try:
        manager._persist(job)
        assert attempts == 3
        assert (output / "jobs" / f"{job.job_id}.json").is_file()
    finally:
        manager.shutdown()


def test_replay_listing_lookup_and_summary_are_confined_to_local_output(research_root, tmp_path, monkeypatch):
    import visualizer.backend.replay_jobs as jobs_module

    output = tmp_path / "local"
    output.mkdir()
    monkeypatch.setattr(jobs_module, "local_output_root", lambda: output)
    path = output / f"{'4' * 32}.json.gz"
    with gzip.open(path, "wb") as target:
        target.write(_bundle().model_dump_json().encode("utf-8"))
    manager = ReplayJobManager(_settings(research_root), FakeCatalog())
    try:
        replay_id = _replay_id_for_path(path)
        catalog = manager.list_replays()
        assert len(catalog.replays) == 1
        assert catalog.replays[0].replay_id == replay_id
        assert catalog.replays[0].outcome == "intercepted"
        assert not catalog.warnings
        assert manager.get_replay(replay_id).episodes[0].simulation_seed == 41
        with pytest.raises(ApplicationError) as invalid:
            manager.get_replay("../outside")
        assert invalid.value.code == "INVALID_REPLAY_ID"
    finally:
        manager.shutdown()


def test_replay_listing_excludes_invalid_legacy_artifacts_with_warning(research_root, tmp_path, monkeypatch):
    import visualizer.backend.replay_jobs as jobs_module

    output = tmp_path / "local"
    output.mkdir()
    monkeypatch.setattr(jobs_module, "local_output_root", lambda: output)
    valid_path = output / f"{'5' * 32}.json.gz"
    with gzip.open(valid_path, "wb") as target:
        target.write(_bundle().model_dump_json().encode("utf-8"))
    (output / "legacy.json.gz").write_bytes(b"invalid gzip data")
    manager = ReplayJobManager(_settings(research_root), FakeCatalog())
    try:
        catalog = manager.list_replays()
        assert len(catalog.replays) == 1
        assert len(catalog.warnings) == 1
        assert "failed current schema or trace validation" in catalog.warnings[0]
    finally:
        manager.shutdown()


def test_cors_is_restricted_to_configured_local_frontend(research_root):
    settings = Settings(
        research_data_root=research_root,
        simulation_python=Path(sys.executable),
        frontend_origin="http://localhost:5173",
    )
    with TestClient(create_app(settings, catalog_provider=FakeCatalog())) as client:
        allowed = client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        denied = client.get("/api/health", headers={"Origin": "https://example.invalid"})
    assert allowed.headers.get("access-control-allow-origin") == "http://localhost:5173"
    assert denied.headers.get("access-control-allow-origin") is None
