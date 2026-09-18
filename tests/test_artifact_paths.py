"""
Tests for `experiments/artifact_paths.py` -- the centralized artifact-path
resolver. Covers: env-var root, explicit-root override, local-artifact-found
case, sibling-repo fallback, missing-artifact error, resolution-order
precedence, and deterministic/repeated-call behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experiments.artifact_paths import (
    PROJECT_ROOT,
    SIBLING_REPO_NAME,
    ARTIFACT_ROOT_ENV_VAR,
    candidate_paths,
    resolve_artifact,
    resolve_artifact_or_local,
)


def test_candidate_paths_order_is_explicit_local_sibling(tmp_path):
    relative = "results/some_artifact.csv"
    paths = candidate_paths(relative, artifact_root=tmp_path)
    assert paths == [
        tmp_path / relative,
        PROJECT_ROOT / relative,
        PROJECT_ROOT.parent / SIBLING_REPO_NAME / relative,
    ]


def test_candidate_paths_without_explicit_root_has_two_entries(monkeypatch):
    monkeypatch.delenv(ARTIFACT_ROOT_ENV_VAR, raising=False)
    relative = "results/some_artifact.csv"
    paths = candidate_paths(relative)
    assert paths == [
        PROJECT_ROOT / relative,
        PROJECT_ROOT.parent / SIBLING_REPO_NAME / relative,
    ]


def test_candidate_paths_rejects_absolute_path():
    with pytest.raises(ValueError):
        candidate_paths(Path(PROJECT_ROOT) / "results" / "x.csv")


def test_resolve_artifact_finds_local_file(tmp_path, monkeypatch):
    # Point PROJECT_ROOT-relative resolution at a temp explicit root that
    # actually contains the file, exercising the "found" path.
    target_dir = tmp_path / "results"
    target_dir.mkdir()
    target_file = target_dir / "present.csv"
    target_file.write_text("data")

    resolved = resolve_artifact("results/present.csv", artifact_root=tmp_path)
    assert resolved == target_file
    assert resolved.exists()


def test_resolve_artifact_env_var_root_used(tmp_path, monkeypatch):
    target_dir = tmp_path / "results"
    target_dir.mkdir()
    target_file = target_dir / "env_present.csv"
    target_file.write_text("data")

    monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, str(tmp_path))
    resolved = resolve_artifact("results/env_present.csv")
    assert resolved == target_file


def test_resolve_artifact_explicit_root_overrides_env_var(tmp_path, monkeypatch):
    env_root = tmp_path / "env_root"
    env_root.mkdir()
    explicit_root = tmp_path / "explicit_root"
    (explicit_root / "results").mkdir(parents=True)
    (explicit_root / "results" / "f.csv").write_text("explicit")

    monkeypatch.setenv(ARTIFACT_ROOT_ENV_VAR, str(env_root))
    resolved = resolve_artifact("results/f.csv", artifact_root=explicit_root)
    assert resolved == explicit_root / "results" / "f.csv"


def test_resolve_artifact_raises_with_all_attempted_paths_listed(monkeypatch, tmp_path):
    monkeypatch.delenv(ARTIFACT_ROOT_ENV_VAR, raising=False)
    missing_relative = "results/definitely_does_not_exist_anywhere_12345.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_artifact(missing_relative, artifact_root=tmp_path)
    message = str(exc_info.value)
    assert str(tmp_path / missing_relative) in message
    assert str(PROJECT_ROOT / missing_relative) in message
    assert str(PROJECT_ROOT.parent / SIBLING_REPO_NAME / missing_relative) in message


def test_resolve_artifact_or_local_never_raises_and_falls_back(monkeypatch, tmp_path):
    monkeypatch.delenv(ARTIFACT_ROOT_ENV_VAR, raising=False)
    missing_relative = "results/definitely_does_not_exist_anywhere_67890.csv"
    resolved = resolve_artifact_or_local(missing_relative, artifact_root=tmp_path)
    assert resolved == PROJECT_ROOT / missing_relative
    assert not resolved.exists()


def test_resolve_artifact_or_local_prefers_first_existing_root(tmp_path):
    # Local (PROJECT_ROOT) always wins over an explicit root that also has
    # the file, per precedence: explicit -> local -> sibling.
    relative = "requirements.txt"  # known to exist at PROJECT_ROOT
    (tmp_path / "requirements.txt").write_text("decoy")
    resolved = resolve_artifact_or_local(relative, artifact_root=tmp_path)
    assert resolved == tmp_path / relative  # explicit root takes precedence


def test_resolve_artifact_is_deterministic_across_repeated_calls(tmp_path):
    target_dir = tmp_path / "results"
    target_dir.mkdir()
    (target_dir / "x.csv").write_text("data")

    first = resolve_artifact("results/x.csv", artifact_root=tmp_path)
    second = resolve_artifact("results/x.csv", artifact_root=tmp_path)
    assert first == second
