import hashlib
import os
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from visualizer.backend.config import LOCAL_OUTPUT_ROOT, Settings, load_settings, local_output_root
from visualizer.backend.data_access import ResearchDataStore
from visualizer.backend.errors import ApplicationError
from visualizer.backend.main import create_app


@pytest.fixture
def research_root(tmp_path):
    root = tmp_path / "research"
    source = root / "uav_defend" / "envs" / "soldier_env.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Configuration fixture, not a simulation.\n")
    return root


def test_health_without_data_root():
    with TestClient(create_app(Settings())) as client:
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {
            "service": "counter-uas-visualizer", "phase": 1, "status": "degraded",
            "simulation_enabled": False, "data_root": "not_configured",
            "warnings": [{"code": "DATA_ROOT_NOT_CONFIGURED", "message": "Set VISUALIZER_RESEARCH_DATA_ROOT in the local backend configuration."}],
        }


def test_configured_health_and_no_path_endpoint(research_root):
    with TestClient(create_app(Settings(research_data_root=research_root))) as client:
        body = client.get("/api/health").json()
        assert body["status"] == "ok"
        assert body["warnings"] == []
        assert str(research_root) not in str(body)
        assert client.get("/api/files", params={"path": str(research_root)}).status_code == 404
        assert client.post("/api/episodes", json={"path": "C:\\private"}).status_code == 404
        assert client.get("/api/missing").json()["error"]["code"] == "HTTP_ERROR"


def test_settings_load_environment_over_dotenv(monkeypatch, tmp_path, research_root):
    env = tmp_path / ".env"
    env.write_text("VISUALIZER_RESEARCH_DATA_ROOT=does-not-exist\n")
    monkeypatch.setenv("VISUALIZER_RESEARCH_DATA_ROOT", str(research_root))
    assert load_settings(env).research_data_root == research_root.resolve()
    monkeypatch.delenv("VISUALIZER_RESEARCH_DATA_ROOT")
    with pytest.raises(ValidationError, match="not a directory"):
        load_settings(env)


def test_missing_env_is_explicit_unconfigured_state(monkeypatch, tmp_path):
    monkeypatch.delenv("VISUALIZER_RESEARCH_DATA_ROOT", raising=False)
    assert load_settings(tmp_path / "missing.env").research_data_root is None


def test_relative_root_uses_visualizer_directory(monkeypatch, research_root):
    import visualizer.backend.config as config
    monkeypatch.setattr(config, "VISUALIZER_ROOT", research_root.parent / "visualizer")
    assert Settings(research_data_root=Path("..") / "research").research_data_root == research_root.resolve()


def test_invalid_roots_and_overlap(research_root, tmp_path, monkeypatch):
    with pytest.raises(ValidationError, match="not a directory"):
        Settings(research_data_root=tmp_path / "absent")
    with pytest.raises(ValidationError, match="not a research worktree"):
        Settings(research_data_root=tmp_path)
    import visualizer.backend.config as config
    monkeypatch.setattr(config, "LOCAL_OUTPUT_ROOT", research_root / "results" / "trace")
    with pytest.raises(ValidationError, match="disjoint"):
        Settings(research_data_root=research_root)


def test_invalid_configuration_fails_startup(monkeypatch, tmp_path):
    monkeypatch.setenv("VISUALIZER_RESEARCH_DATA_ROOT", str(tmp_path / "missing"))
    with pytest.raises(ValidationError):
        with TestClient(create_app()):
            pass


@pytest.mark.parametrize("path", ["..\\secret", "../secret", "C:\\secret", "C:secret", "\\\\host\\share", "/secret", "", "file:stream"])
def test_artifact_path_rejects_escape(research_root, path):
    store = ResearchDataStore(Settings(research_data_root=research_root))
    with pytest.raises(ApplicationError) as error:
        store.resolve_file(path)
    assert error.value.code == "INVALID_ARTIFACT_PATH"


def test_unconfigured_and_missing_artifacts(research_root):
    with pytest.raises(ApplicationError, match="Configure"):
        ResearchDataStore(Settings()).resolve_file("checkpoint.zip")
    with pytest.raises(ApplicationError) as error:
        ResearchDataStore(Settings(research_data_root=research_root)).resolve_file("missing.zip")
    assert error.value.code == "ARTIFACT_NOT_FOUND"
    assert error.value.status_code == 404


def test_read_only_artifact_integrity(research_root):
    artifact = research_root / "checkpoint.zip"
    artifact.write_bytes(b"hash-test fixture, not a trained model")
    before = artifact.stat()
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    store = ResearchDataStore(Settings(research_data_root=research_root))
    assert store.read_bytes("checkpoint.zip", digest) == artifact.read_bytes()
    assert artifact.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ApplicationError) as error:
        store.read_bytes("checkpoint.zip", "0" * 64)
    assert error.value.code == "CHECKPOINT_HASH_MISMATCH"


def test_symlink_escape_is_rejected(research_root, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "artifact.txt").write_text("outside")
    link = research_root / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        if os.name != "nt":
            raise
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "$ErrorActionPreference = 'Stop'; New-Item -ItemType Junction "
             "-Path $env:VISUALIZER_TEST_LINK -Target $env:VISUALIZER_TEST_TARGET | Out-Null"],
            env={**os.environ, "VISUALIZER_TEST_LINK": str(link), "VISUALIZER_TEST_TARGET": str(outside)},
            capture_output=True, text=True, check=True,
        )
    with pytest.raises(ApplicationError) as error:
        ResearchDataStore(Settings(research_data_root=research_root)).read_bytes("link\\artifact.txt")
    assert error.value.code == "INVALID_ARTIFACT_PATH"


def test_output_is_separate_and_not_created():
    assert local_output_root() == LOCAL_OUTPUT_ROOT.resolve()


def test_structured_errors_and_logging(research_root, caplog):
    app = create_app(Settings(research_data_root=research_root))

    @app.get("/test/artifact")
    def missing():
        return ResearchDataStore(Settings(research_data_root=research_root)).read_bytes("missing.zip")

    @app.get("/test/validate")
    def validate(count: int):
        return count

    @app.get("/test/unexpected")
    def unexpected():
        raise RuntimeError("private error detail")

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/test/artifact").json()["error"]["code"] == "ARTIFACT_NOT_FOUND"
        assert client.get("/test/validate?count=no").json()["error"]["code"] == "INVALID_REQUEST"
        response = client.get("/test/unexpected")
        assert response.status_code == 500
        assert response.json()["error"]["code"] == "INTERNAL_ERROR"
        assert "private error detail" not in response.text
    assert "ARTIFACT_NOT_FOUND" in caplog.text
    assert "Unhandled API error" in caplog.text
