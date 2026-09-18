"""
Centralized artifact-path resolver -- SINGLE SOURCE OF TRUTH for locating
result/checkpoint artifacts that are intentionally excluded from git (see
.gitignore: `results/`, `*.zip`, `*.pt`, ...).

This module does NOT change any experiment logic, hash, or protocol. It only
locates already-produced artifacts on disk so that path plumbing works
correctly regardless of which git worktree/clone is being used.

Why this exists
----------------
Large/generated artifacts (trained checkpoints, evaluation CSVs/JSONs, paper
figures) are never committed to git. When working from a fresh worktree that
has not (yet) locally regenerated a given artifact, an equivalent, byte-
identical copy already exists in the sibling conference repository
(`uav-defense-3d-resubmission`) that originally produced it.

Resolution precedence for a repo-relative path such as
``results/training_400k/direct/seed_42/best_model.zip``:

    1. An explicit ``artifact_root`` argument, or the
       ``UAV_DEFENSE_ARTIFACT_ROOT`` environment variable if set.
       ``<artifact_root>/<relative_path>`` is used if it exists.
    2. The local repository root (this repo). ``PROJECT_ROOT/<relative_path>``
       is used if it exists -- i.e. if the artifact has already been
       generated/copied locally, it is always preferred over any other repo.
    3. The sibling conference repository
       (``PROJECT_ROOT.parent / "uav-defense-3d-resubmission"``), if the path
       exists there.
    4. Otherwise: either raise ``FileNotFoundError`` (see `resolve_artifact`)
       or fall back to the local candidate path so that existing
       ``.exists()``-based error handling continues to fire unchanged (see
       `resolve_artifact_or_local`).

This module performs no globbing and no fuzzy matching: a given
`relative_path` always resolves to the SAME relative path under whichever
root is chosen, so it can never silently select an unrelated checkpoint or
result file.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SIBLING_REPO_NAME = "uav-defense-3d-resubmission"
ARTIFACT_ROOT_ENV_VAR = "UAV_DEFENSE_ARTIFACT_ROOT"


def _candidate_roots(artifact_root: str | Path | None = None) -> list[Path]:
    """Return the ordered list of candidate roots to search, per precedence."""
    roots: list[Path] = []
    explicit = artifact_root if artifact_root is not None else os.environ.get(ARTIFACT_ROOT_ENV_VAR)
    if explicit:
        roots.append(Path(explicit))
    roots.append(PROJECT_ROOT)
    roots.append(PROJECT_ROOT.parent / SIBLING_REPO_NAME)
    return roots


def candidate_paths(relative_path: str | Path, artifact_root: str | Path | None = None) -> list[Path]:
    """Return every path that would be tried, in resolution order (no I/O)."""
    relative_path = Path(relative_path)
    if relative_path.is_absolute():
        raise ValueError(f"relative_path must be repo-relative, got absolute path: {relative_path}")
    return [root / relative_path for root in _candidate_roots(artifact_root)]


def resolve_artifact(relative_path: str | Path, artifact_root: str | Path | None = None) -> Path:
    """Resolve `relative_path` to an existing artifact, or raise.

    Raises `FileNotFoundError` (listing every attempted location) if the
    artifact is not found under any candidate root.
    """
    attempted = candidate_paths(relative_path, artifact_root=artifact_root)
    for candidate in attempted:
        if candidate.exists():
            return candidate

    attempted_str = "\n".join(f"  - {p}" for p in attempted)
    raise FileNotFoundError(
        f"Artifact not found: {relative_path}\n"
        f"Attempted the following locations (in resolution order):\n{attempted_str}"
    )


def resolve_artifact_or_local(relative_path: str | Path, artifact_root: str | Path | None = None) -> Path:
    """Resolve `relative_path` like `resolve_artifact`, but never raise.

    If the artifact is not found under any candidate root, returns the local
    repository candidate path (``PROJECT_ROOT / relative_path``) unchanged so
    that callers relying on a later ``.exists()`` check (and the resulting
    error message) continue to behave exactly as before this resolver
    existed.
    """
    attempted = candidate_paths(relative_path, artifact_root=artifact_root)
    for candidate in attempted:
        if candidate.exists():
            return candidate
    return PROJECT_ROOT / Path(relative_path)
