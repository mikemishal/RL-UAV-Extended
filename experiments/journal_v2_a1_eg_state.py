"""Restartable task state for the defender-evasion gating study (action-v6). Same guarantees as `journal_v2_a1_tg_state`: atomic writes with retry,
a stage is skipped on restart only if its artifacts still hash-verify and the recorded input commit is an ancestor of HEAD; `running`/`failed` stages and
unreadable state are never trusted. Adds a single-writer lock so duplicate launches cannot write the same sweep cell."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_DIR = PROJECT_ROOT / "docs" / "journal_v2_a1_evasion_gate_v1"
STATE_PATH = STATE_DIR / "task_state.json"
STAGES = (
    "01_repository_verification", "02_primitive_audit", "03_interface_specification", "04_implementation", "05_unit_tests",
    "06_deterministic_sweep", "07_analysis", "08_regression", "09_checkpoint_integrity", "10_publication_documentation", "11_commit_tag_push",
)
EVIDENCE_STAGES = ("08_regression", "09_checkpoint_integrity", "10_publication_documentation", "11_commit_tag_push")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes, attempts: int = 8) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp{os.getpid()}")
    with open(temporary, "wb") as handle:
        handle.write(data)
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


def atomic_write_json(path: Path, payload) -> None:
    atomic_write_bytes(path, json.dumps(payload, indent=1, sort_keys=True, default=str).encode("utf-8"))


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()


class CellLock:
    """Exclusive create-only lock file for one sweep cell. A stale lock (owner process gone) is taken over; a live owner makes `acquire` return False,
    so a duplicate launch skips the cell instead of writing it concurrently."""

    def __init__(self, path: Path):
        self.path = Path(path).with_suffix(".lock")
        self.held = False

    @staticmethod
    def _alive(pid: int) -> bool:
        if pid == os.getpid():
            return True
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
            return str(pid) in out
        except OSError:
            return True

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                self.held = True
                return True
            except FileExistsError:
                try:
                    owner = int(self.path.read_text().strip() or 0)
                except (OSError, ValueError):
                    owner = 0
                if owner and self._alive(owner):
                    return False
                try:
                    os.remove(self.path)
                except OSError:
                    return False
        return False

    def release(self) -> None:
        if self.held:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self.held = False


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"task": "journal_v2_a1_evasion_gate_v1", "created_utc": _now(), "stages": {s: {"status": "pending"} for s in STAGES}}


def save_state(state: dict) -> None:
    state["updated_utc"] = _now()
    atomic_write_json(STATE_PATH, state)


def stage_is_verified(state: dict, name: str) -> bool:
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
    """A gate stage ended the experimental pipeline: every later stage except the evidence stages becomes `not_applicable`."""
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
