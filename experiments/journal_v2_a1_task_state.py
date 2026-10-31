"""Restartable task-state tracking for the discrete-tactical-interface (action-v4) study.

State lives in `docs/journal_v2_a1_discrete_v4/task_state.json` (atomic temp-file -> rename). A stage is only ever skipped
on restart if (a) its recorded status is `complete`, (b) every recorded output artifact still exists and still has the recorded SHA-256,
and (c) the repository still resolves the recorded input commit as an ancestor of HEAD (code identity). Anything else re-runs.
A partially written artifact can never be mistaken for a complete one: the stage is marked `complete` only AFTER its artifacts are
written and hashed.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = PROJECT_ROOT / "docs" / "journal_v2_a1_discrete_v4"
STATE_PATH = STATE_DIR / "task_state.json"
STAGES = (
    "01_repository_verification", "02_primitive_audit", "03_architecture_implementation", "04_unit_tests",
    "05_stress_study_generation", "06_stress_study_analysis", "07_protocol_documentation", "08_non_training_regression",
    "09_checkpoint_integrity", "10_commit_tag_push",
)
STARTING_HEAD_PREFIX = "0ecf78e"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict, attempts: int = 8) -> None:
    """temp file -> fsync -> os.replace, retrying transient `PermissionError`s (OneDrive/AV file locks on the destination)."""
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=1, sort_keys=True, default=str))
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(attempts):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.5 * (attempt + 1))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass  # an unreadable state file is treated as absent: nothing is trusted
    return {"task": "journal_v2_a1_discrete_action_v4", "created_utc": _now(), "stages": {s: {"status": "pending"} for s in STAGES}}


def save_state(state: dict) -> None:
    state["updated_utc"] = _now()
    atomic_write_json(STATE_PATH, state)


def stage_is_verified(state: dict, name: str) -> bool:
    record = state["stages"].get(name, {})
    if record.get("status") == "not_applicable":  # halted by an earlier stage's verdict; nothing to verify
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


def halt(state: dict, stage: str, verdict: str, reason: str, not_applicable=(
        "03_architecture_implementation", "04_unit_tests", "05_stress_study_generation", "06_stress_study_analysis",
        "07_protocol_documentation")) -> None:
    """A critical decision stage ended the experimental pipeline (e.g. verdict D). The architecture/stress-study stages are marked
    `not_applicable` (a status beyond pending/running/complete/failed, deliberately distinct from `failed`); evidence-preservation
    stages (regression, integrity, commit) remain runnable. A restart will NOT silently re-open the halted pipeline."""
    state["halt"] = {"stage": stage, "verdict": verdict, "reason": reason, "utc": _now(), "input_commit": git("rev-parse", "HEAD")}
    for name in not_applicable:
        state["stages"][name] = {"status": "not_applicable", "halted_by": stage, "verdict": verdict, "completed_utc": _now()}
    save_state(state)


def next_stage(state: dict) -> str | None:
    """First stage that is not verified-complete (restart resumes here)."""
    for name in STAGES:
        if not stage_is_verified(state, name):
            return name
    return None


if __name__ == "__main__":
    s = load_state()
    print(json.dumps({"next_stage": next_stage(s), "statuses": {k: v["status"] for k, v in s["stages"].items()}}, indent=1))
