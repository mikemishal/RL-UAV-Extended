"""Simplified-API episode-generation orchestration service (brief sections 4, 15). Invokes `episode_cli.py` via
the configured simulation Python interpreter, synchronously (these are short scripted/optimization-based
episodes, not long RL training), and persists the result through `EpisodeStore`.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .application_error import ApplicationError
from .config import Settings
from .episode_models import (
    SCENARIO_PRESETS,
    SUPPORTED_DEFENDERS,
    A1_CBF_SOURCE_COMMIT,
    A1_CBF_TAG,
    EpisodeCreateRequest,
    EpisodeMetadata,
)
from .episode_store import EpisodeStore

VISUALIZER_ROOT = Path(__file__).resolve().parents[1]
# A0's provenance tag: the scripted legacy attacker is a frozen, un-versioned part of the research environment
# itself (no separate research tag exists for it, unlike A1-CBF's `journal-cbf-attacker-v1`) -- recorded
# honestly as such rather than inventing a tag that does not exist.
A0_PROVENANCE_LABEL = "A0 (scripted legacy attacker, research environment baseline)"

SAFE_WORKER_ERRORS = {
    "SIMULATION_DATA_ROOT_MISSING": "The simulation research data root is not configured.",
    "EPISODE_FAILED": "The episode failed to run; see backend logs for details.",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, encoding="utf-8",
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class EpisodeService:
    def __init__(self, settings: Settings, store: EpisodeStore):
        self.settings = settings
        self.store = store

    # ------------------------------------------------------------------------------- health
    def research_source_available(self) -> bool:
        return self.settings.research_data_root is not None and self.settings.research_data_root.is_dir()

    def a0_available(self) -> bool:
        if not self.research_source_available():
            return False
        return (self.settings.research_data_root / "uav_defend" / "envs" / "soldier_env.py").is_file()

    def a1_cbf_available(self) -> bool:
        if not self.research_source_available():
            return False
        return (self.settings.research_data_root / "uav_defend" / "attackers" / "cbf_obstacle.py").is_file()

    def storage_writable(self) -> bool:
        try:
            probe = self.store.root / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def counts(self) -> tuple[int, int]:
        episodes = self.store.list()
        new_count = sum(1 for e in episodes if e.is_new)
        return new_count, len(episodes) - new_count

    # ------------------------------------------------------------------------------- run
    def create_episode(self, request: EpisodeCreateRequest) -> EpisodeMetadata:
        if request.defender not in SUPPORTED_DEFENDERS:
            raise ApplicationError("UNSUPPORTED_DEFENDER", f"Defender {request.defender!r} is not supported.", 400)
        preset = SCENARIO_PRESETS.get(request.scenario)
        if preset is None:
            raise ApplicationError("UNSUPPORTED_SCENARIO", f"Scenario {request.scenario!r} is not supported.", 400)
        if self.settings.research_data_root is None:
            raise ApplicationError("SIMULATION_DATA_ROOT_MISSING", "The simulation research data root is not configured.", 503)

        seed = request.seed if request.seed is not None else int.from_bytes(os.urandom(4), "big")
        episode_id = uuid4().hex
        result = self._run_subprocess(request.scenario, request.defender, seed)

        source_commit, controller_version = (
            (A1_CBF_SOURCE_COMMIT, A1_CBF_TAG) if preset.resolved_attacker == "A1-CBF"
            else (_git_commit(self.settings.research_data_root) or "unknown", A0_PROVENANCE_LABEL)
        )

        if "error" in result:
            metadata = EpisodeMetadata(
                episode_id=episode_id, created_at=_now_iso(), scenario=request.scenario,
                scenario_label=preset.scenario_label, resolved_attacker=preset.resolved_attacker,
                defender=request.defender, seed=seed, status="failed", outcome="episode_failed",
                duration=0.0, steps=0, asset_captured=False, defender_interception=False, obstacle_collision=False,
                replay_path=episode_id, source_commit=source_commit, controller_version=controller_version,
                is_new=True, error_message=SAFE_WORKER_ERRORS.get(result["error"].get("code"), "The episode failed to run."),
            )
            return self.store.create(metadata, {"schema": "app_episode_replay/1", "outcome": "episode_failed", "obstacles": [], "steps": []})

        metadata = EpisodeMetadata(
            episode_id=episode_id, created_at=_now_iso(), scenario=request.scenario,
            scenario_label=preset.scenario_label, resolved_attacker=preset.resolved_attacker,
            defender=request.defender, seed=seed, status="completed", outcome=result["outcome"],
            duration=result["duration"], steps=result["steps"], asset_captured=result["asset_captured"],
            defender_interception=result["defender_interception"], obstacle_collision=result["obstacle_collision"],
            replay_path=episode_id, source_commit=source_commit, controller_version=controller_version, is_new=True,
            min_defender_separation_m=result.get("min_defender_separation_m"),
            min_obstacle_clearance_m=result.get("min_obstacle_clearance_m"), path_length_m=result.get("path_length_m"),
        )
        return self.store.create(metadata, result["replay"])

    def _run_subprocess(self, scenario: str, defender: str, seed: int) -> dict:
        command = [
            str(self.settings.simulation_python), "-m", "visualizer.backend.episode_cli",
            "--scenario", scenario, "--defender", defender, "--seed", str(seed),
        ]
        environment = os.environ.copy()
        environment["VISUALIZER_RESEARCH_DATA_ROOT"] = str(self.settings.research_data_root)
        environment["PYTHONPATH"] = str(VISUALIZER_ROOT.parent) + os.pathsep + environment.get("PYTHONPATH", "")
        try:
            process = subprocess.run(
                command, cwd=VISUALIZER_ROOT.parent, env=environment, capture_output=True, text=True,
                encoding="utf-8", timeout=self.settings.episode_timeout_seconds,
            )
        except OSError as exc:
            raise ApplicationError("SIMULATION_RUNTIME_UNAVAILABLE", "Cannot start the configured simulation Python runtime.", 503) from exc
        except subprocess.TimeoutExpired as exc:
            raise ApplicationError("EPISODE_TIMEOUT", f"Episode exceeded the {self.settings.episode_timeout_seconds}-second timeout.", 504) from exc
        try:
            return json.loads(process.stdout.strip().splitlines()[-1]) if process.stdout.strip() else {"error": {"code": "EPISODE_FAILED", "message": "no output"}}
        except (ValueError, IndexError):
            return {"error": {"code": "EPISODE_FAILED", "message": process.stderr[-2000:] if process.stderr else "unparseable worker output"}}

    # ------------------------------------------------------------------------------- read/list/delete
    def list_episodes(self, *, new: bool | None = None, scenario: str | None = None, defender: str | None = None) -> list[EpisodeMetadata]:
        episodes = self.store.list()
        if new is not None:
            episodes = [e for e in episodes if e.is_new == new]
        if scenario is not None:
            episodes = [e for e in episodes if e.scenario == scenario]
        if defender is not None:
            episodes = [e for e in episodes if e.defender == defender]
        return episodes

    def get_episode(self, episode_id: str) -> tuple[EpisodeMetadata, dict]:
        metadata, replay = self.store.get(episode_id)
        if metadata.status == "completed":
            metadata = self.store.mark_viewed(episode_id)
        return metadata, replay

    def delete_episode(self, episode_id: str) -> bool:
        return self.store.delete(episode_id)

    def delete_many(self, episode_ids: list[str]) -> dict[str, bool]:
        return self.store.delete_many(episode_ids)
