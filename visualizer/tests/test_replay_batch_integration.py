from __future__ import annotations

import gzip
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator

from visualizer.backend.config import Settings, VISUALIZER_ROOT
from visualizer.backend.main import create_app
from visualizer.backend.replay_contract import ReplayBundle


RESEARCH_ROOT = os.environ.get("VISUALIZER_RESEARCH_DATA_ROOT")
SIMULATION_PYTHON = os.environ.get("VISUALIZER_SIMULATION_PYTHON", sys.executable)
pytestmark = pytest.mark.skipif(
    not RESEARCH_ROOT,
    reason="Set VISUALIZER_RESEARCH_DATA_ROOT and VISUALIZER_SIMULATION_PYTHON for bounded real replay integration tests.",
)


def _wait_for_job(client: TestClient, job_id: str) -> dict:
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        response = client.get(f"/api/replay-jobs/{job_id}")
        assert response.status_code == 200, response.text
        job = response.json()
        if job["state"] not in {"queued", "running"}:
            return job
        time.sleep(0.5)
    pytest.fail("Replay batch did not finish within the configured 900-second per-episode timeout window.")


def _settings() -> Settings:
    return Settings(
        research_data_root=Path(RESEARCH_ROOT),
        simulation_python=Path(SIMULATION_PYTHON),
    )


def _wait_for_replay(client: TestClient, job: dict) -> list[dict]:
    replay_bundles = []
    schema = ReplayBundle.model_json_schema()
    validator = Draft202012Validator(schema)
    for replay_id in job["completed_replay_ids"]:
        response = client.get(f"/api/replays/{replay_id}")
        assert response.status_code == 200, response.text
        validator.validate(response.json())
        replay_bundles.append(response.json())
    return replay_bundles


def test_real_policy_and_scenario_catalog_matches_registered_artifacts():
    with TestClient(create_app(_settings())) as client:
        policy_response = client.get("/api/policies")
        scenario_response = client.get("/api/scenarios")
    assert policy_response.status_code == scenario_response.status_code == 200
    policies = policy_response.json()
    defenders = {item["policy_id"]: item for item in policies["defenders"]}
    assert set(defenders) == {
        "greedy", "kalman_greedy", "pn", "lead", "rmpc", "ppo", "ppo_kalman", "lr_ppo",
    }
    assert all(item["available"] for item in defenders.values())
    expected_seeds = {"ppo": [71, 72, 73], "ppo_kalman": [74, 75, 76], "lr_ppo": [77, 78, 79]}
    expected_roles = {"d0": "d0_start", "d0_plus": "final", "d1": "final"}
    for policy_id, seeds in expected_seeds.items():
        generations = {item["generation"]: item for item in defenders[policy_id]["generations"]}
        assert generations["conference"]["available"] is False
        for generation, role in expected_roles.items():
            entry = generations[generation]
            assert entry["checkpoint_role"] == role
            assert [item["training_seed"] for item in entry["training_seeds"] if item["available"]] == seeds
    attackers = {item["attacker_id"]: item for item in policies["attackers"]}
    assert set(attackers) == {"A0", "A1-201", "A1-202", "A1-203"}
    assert all(item["available"] for item in attackers.values())
    assert attackers["A1-203"]["held_out"] is True
    scenarios = scenario_response.json()["scenarios"]
    assert [item["clutter"] for item in scenarios] == ["C0", "C1", "C2", "C3"]
    for scenario in scenarios:
        assert {item["id"] for item in scenario["perceptions"]} == {"local", "map_local"}
        for perception in scenario["perceptions"]:
            assert set(perception["configurations"]) == {"measurement", "kalman"}
    c0_map = next(item for item in scenarios[0]["perceptions"] if item["id"] == "map_local")
    assert c0_map["configurations"]["measurement"]["parameters"]["max_steps"] == 2000


def test_real_three_episode_greedy_a0_c0_batch_and_engine_equivalence():
    from visualizer.backend.replay_jobs import _replay_id_for_path

    output_root = VISUALIZER_ROOT / "data" / "local"
    output_root.mkdir(parents=True, exist_ok=True)
    seed_start = 71030000
    request = {
        "defender_policy_id": "greedy",
        "attacker_condition": "A0",
        "clutter_level": "C0",
        "perception_mode": "map_local",
        "seed_start": seed_start,
        "episode_count": 3,
        "deterministic_inference": True,
    }
    with TestClient(create_app(_settings())) as client:
        response = client.post("/api/replay-jobs", json=request)
        assert response.status_code == 202, response.text
        job = _wait_for_job(client, response.json()["job_id"])
        assert job["state"] == "completed"
        assert job["requested_episode_count"] == job["completed_count"] == 3
        assert len(job["completed_replay_ids"]) == 3
        bundles = _wait_for_replay(client, job)
        episodes = [bundle["episodes"][0] for bundle in bundles]
        assert [episode["simulation_seed"] for episode in episodes] == [seed_start, seed_start + 1, seed_start + 2]
        for episode in episodes:
            assert len(episode["frames"]) == episode["summary"]["steps"] + 1
            assert episode["frames"][0]["step"] == 0
            assert episode["frames"][-1]["outcome"] == episode["summary"]["outcome"]
            assert all(len(frame["defender"]["position_m"]) == 3 for frame in episode["frames"])
        listed = client.get("/api/replays", params={"limit": 20})
        assert listed.status_code == 200
        listed_ids = {item["replay_id"] for item in listed.json()["replays"]}
        assert set(job["completed_replay_ids"]).issubset(listed_ids)
        summary_response = client.get(f"/api/replays/{job['completed_replay_ids'][0]}/summary")
        assert summary_response.status_code == 200
        assert summary_response.json()["summary"] == episodes[0]["summary"]

        output_name = f"phase3-equivalence-{secrets.token_hex(8)}"
        env = os.environ.copy()
        env["VISUALIZER_RESEARCH_DATA_ROOT"] = RESEARCH_ROOT
        subprocess.run(
            [
                SIMULATION_PYTHON, "-m", "visualizer.backend.replay_cli",
                "--defender", "greedy", "--attacker", "A0", "--clutter", "C0",
                "--perception", "map_local", "--evaluation-seed", str(seed_start),
                "--output-name", output_name,
            ],
            cwd=VISUALIZER_ROOT.parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=True,
        )
        direct_path = output_root / f"{output_name}.json.gz"
        direct = ReplayBundle.model_validate_json(gzip.decompress(direct_path.read_bytes())).model_dump(mode="json")
        batch_episode = episodes[0]
        direct_episode = direct["episodes"][0]
        assert direct_episode["summary"] == batch_episode["summary"]
        assert direct_episode["events"] == batch_episode["events"]
        assert [frame["defender"]["position_m"] for frame in direct_episode["frames"]] == [
            frame["defender"]["position_m"] for frame in batch_episode["frames"]
        ]
        assert [frame["attacker"]["position_m"] for frame in direct_episode["frames"]] == [
            frame["attacker"]["position_m"] for frame in batch_episode["frames"]
        ]
        assert [frame["telemetry"]["requested_action"] if frame["telemetry"] else None for frame in direct_episode["frames"]] == [
            frame["telemetry"]["requested_action"] if frame["telemetry"] else None for frame in batch_episode["frames"]
        ]
        assert _replay_id_for_path(direct_path) not in job["completed_replay_ids"]


def test_real_learned_lrppo_request_against_held_out_a1_203():
    request = {
        "defender_policy_id": "lr_ppo",
        "defender_generation": "d1",
        "defender_training_seed": 77,
        "attacker_condition": "A1-203",
        "clutter_level": "C3",
        "perception_mode": "map_local",
        "seed_start": 71030010,
        "episode_count": 1,
        "deterministic_inference": True,
    }
    with TestClient(create_app(_settings())) as client:
        response = client.post("/api/replay-jobs", json=request)
        assert response.status_code == 202, response.text
        job = _wait_for_job(client, response.json()["job_id"])
        assert job["state"] == "completed"
        assert job["completed_count"] == 1
        episode = client.get(f"/api/replays/{job['completed_replay_ids'][0]}").json()["episodes"][0]
        assert episode["simulation_seed"] == request["seed_start"]
        assert episode["defender"]["generation"] == "d1"
        assert episode["defender"]["checkpoint"]["role"] == "final"
        assert episode["attacker"]["policy_id"] == "A1-203"
        assert episode["attacker"]["checkpoint"]["role"] == "attacker_best"
        assert episode["summary"]["outcome"] in {
            "intercepted", "soldier_caught", "unsafe_intercept", "timeout",
            "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision",
            "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision",
        }
