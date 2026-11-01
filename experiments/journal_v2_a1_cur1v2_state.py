"""Restartable task state for curriculum Stage 1 V2 (`docs/journal_v2_a1_curriculum/stage1_v2/task_state.json`). Same guarantees as `journal_v2_a1_cur1_state`."""

from __future__ import annotations

from experiments import journal_v2_a1_cur1_state as _s

PROJECT_ROOT = _s.PROJECT_ROOT
STATE_DIR = PROJECT_ROOT / "docs" / "journal_v2_a1_curriculum" / "stage1_v2"
STATE_PATH = STATE_DIR / "task_state.json"
STAGES = (
    "01_verify_repository", "02_freeze_stage1_v2_protocol", "03_implement_clean_curriculum_wrapper", "04_generate_expert_dataset", "05_train_bc_seed_201", "06_train_bc_seed_202",
    "07_train_bc_seed_203", "08_fixed_expert_relabeling_round", "09_heldout_evaluation", "10_visualizer_preparation", "11_manuscript_documentation", "12_commit_push",
)
sha256_file, atomic_write_bytes, atomic_write_json, git, CellLock, _now = _s.sha256_file, _s.atomic_write_bytes, _s.atomic_write_json, _s.git, _s.CellLock, _s._now


def load_state() -> dict:
    import json

    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    return {"task": "journal_v2_a1_curriculum/stage1_v2", "created_utc": _now(), "stages": {s: {"status": "pending"} for s in STAGES}}


def save_state(state: dict) -> None:
    state["updated_utc"] = _now()
    atomic_write_json(STATE_PATH, state)


def stage_is_verified(state: dict, name: str, current_config_sha: str | None = None) -> bool:
    import subprocess

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


def complete(state: dict, name: str, artifacts: list[str], note: str = "") -> None:
    record = state["stages"][name]
    record.update(status="complete", output_artifacts=list(artifacts), note=note, completed_utc=_now(), artifact_sha256={rel: sha256_file(PROJECT_ROOT / rel) for rel in artifacts})
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
