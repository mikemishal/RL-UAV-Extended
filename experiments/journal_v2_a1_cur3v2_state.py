"""Restartable task state for curriculum Stage3-v2 (`docs/journal_v2_a1_curriculum/stage3_v2/task_state.json`). Same guarantees as the Stage-2 task state: atomic writes, artifact hashes re-verified on
every restart, input commit must still be an ancestor of HEAD."""

from __future__ import annotations

import json
import subprocess

from experiments import journal_v2_a1_eg_state as _base

PROJECT_ROOT = _base.PROJECT_ROOT
STATE_DIR = PROJECT_ROOT / "docs" / "journal_v2_a1_curriculum" / "stage3_v2"
STATE_PATH = STATE_DIR / "task_state.json"
STAGES = (
    "01_repository_cleanup_verification", "02_stage3_v2_protocol_freeze", "03_hold_wrapper_implementation", "04_scenario_validation", "05_tests", "06_sb3_resume_validation",
    "07_seed_201_training", "08_seed_202_training", "09_seed_203_training", "10_heldout_evaluation", "11_visualizer_manual_recordings", "12_stage4_handoff", "13_manuscript_documentation", "14_commit_push",
)
sha256_file, atomic_write_bytes, atomic_write_json, git, CellLock, _now = (_base.sha256_file, _base.atomic_write_bytes, _base.atomic_write_json, _base.git, _base.CellLock, _base._now)


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"task": "journal_v2_a1_curriculum/stage3_v2", "created_utc": _now(), "stages": {s: {"status": "pending"} for s in STAGES}}


def save_state(state: dict) -> None:
    state["updated_utc"] = _now()
    atomic_write_json(STATE_PATH, state)


def stage_is_verified(state: dict, name: str, current_config_sha: str | None = None) -> bool:
    record = state["stages"].get(name, {})
    if record.get("status") == "not_applicable":
        return bool(record.get("reason"))
    if record.get("status") != "complete":
        return False
    if current_config_sha is not None and record.get("config_sha256") not in (None, current_config_sha):
        return False
    for rel, expected in record.get("artifact_sha256", {}).items():
        path = PROJECT_ROOT / rel
        if not path.is_file() or sha256_file(path) != expected:
            return False
    commit = record.get("input_commit")
    if commit:
        try:
            git("merge-base", "--is-ancestor", commit, "HEAD")
        except subprocess.CalledProcessError:
            return False
    return True


def begin(state: dict, name: str, config_sha256: str | None = None) -> None:
    state["stages"][name] = {"status": "running", "input_commit": git("rev-parse", "HEAD"), "config_sha256": config_sha256, "started_utc": _now()}
    save_state(state)


def complete(state: dict, name: str, artifacts: list[str], note: str = "", extra: dict | None = None) -> None:
    record = state["stages"][name]
    record.update(status="complete", output_artifacts=list(artifacts), note=note, completed_utc=_now(), artifact_sha256={rel: sha256_file(PROJECT_ROOT / rel) for rel in artifacts}, **(extra or {}))
    save_state(state)


def fail(state: dict, name: str, note: str, artifacts: list[str] | None = None) -> None:
    record = state["stages"][name]
    record.update(status="failed", note=note, completed_utc=_now(), output_artifacts=list(artifacts or []), artifact_sha256={rel: sha256_file(PROJECT_ROOT / rel) for rel in (artifacts or [])})
    save_state(state)


def skip(state: dict, name: str, reason: str) -> None:
    state["stages"][name] = {"status": "not_applicable", "reason": reason, "input_commit": git("rev-parse", "HEAD"), "completed_utc": _now()}
    save_state(state)


def next_stage(state: dict, current_config_sha: str | None = None) -> str | None:
    for name in STAGES:
        if not stage_is_verified(state, name, current_config_sha):
            return name
    return None
