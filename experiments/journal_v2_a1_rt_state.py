"""Restartable task state for the route-level A1 study (`a1_route_action_v1`). Same guarantees as `journal_v2_a1_eg_state`: atomic writes with retry; a stage is
skipped on restart only if its artifacts hash-verify and the recorded input commit is an ancestor of HEAD; `running`/`failed` stages and unreadable state are
never trusted; a single-writer lock prevents duplicate workers from writing the same sweep cell."""

from __future__ import annotations

from pathlib import Path

from experiments import journal_v2_a1_eg_state as _base

PROJECT_ROOT = _base.PROJECT_ROOT
STATE_DIR = PROJECT_ROOT / "docs" / "journal_v2_a1_route_level_v1"
STATE_PATH = STATE_DIR / "task_state.json"
STAGES = (
    "01_repository_verification", "02_route_planner_audit", "03_route_interface_specification", "04_implementation", "05_observation_state_implementation",
    "06_unit_tests", "07_deterministic_route_sweep", "08_safety_mission_analysis", "09_regression", "10_checkpoint_integrity", "11_publication_documentation",
    "12_commit_tag_push",
)
EVIDENCE_STAGES = ("09_regression", "10_checkpoint_integrity", "11_publication_documentation", "12_commit_tag_push")

sha256_file = _base.sha256_file
atomic_write_bytes = _base.atomic_write_bytes
atomic_write_json = _base.atomic_write_json
git = _base.git
CellLock = _base.CellLock
_now = _base._now


def load_state() -> dict:
    import json

    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"task": "journal_v2_a1_route_level_v1", "created_utc": _now(), "stages": {s: {"status": "pending"} for s in STAGES}}


def save_state(state: dict) -> None:
    state["updated_utc"] = _now()
    atomic_write_json(STATE_PATH, state)


def stage_is_verified(state: dict, name: str) -> bool:
    import subprocess

    record = state["stages"].get(name, {})
    if record.get("status") == "not_applicable":
        return state.get("halt") is not None
    if record.get("status") != "complete":
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


def begin(state: dict, name: str) -> None:
    state["stages"][name] = {"status": "running", "input_commit": git("rev-parse", "HEAD"), "started_utc": _now()}
    save_state(state)


def complete(state: dict, name: str, artifacts: list[str], note: str = "") -> None:
    record = state["stages"][name]
    record.update(status="complete", output_artifacts=list(artifacts), note=note, completed_utc=_now(),
                  artifact_sha256={rel: sha256_file(PROJECT_ROOT / rel) for rel in artifacts})
    save_state(state)


def fail(state: dict, name: str, note: str, artifacts: list[str] | None = None) -> None:
    record = state["stages"][name]
    record.update(status="failed", note=note, completed_utc=_now(), output_artifacts=list(artifacts or []),
                  artifact_sha256={rel: sha256_file(PROJECT_ROOT / rel) for rel in (artifacts or [])})
    save_state(state)


def halt(state: dict, stage: str, verdict: str, reason: str) -> None:
    state["halt"] = {"stage": stage, "verdict": verdict, "reason": reason, "utc": _now(), "input_commit": git("rev-parse", "HEAD")}
    for name in STAGES[STAGES.index(stage) + 1:]:
        if name not in EVIDENCE_STAGES:
            state["stages"][name] = {"status": "not_applicable", "halted_by": stage, "verdict": verdict, "completed_utc": _now()}
    save_state(state)


def next_stage(state: dict) -> str | None:
    for name in STAGES:
        if not stage_is_verified(state, name):
            return name
    return None
