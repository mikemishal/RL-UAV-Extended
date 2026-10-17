import os
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, field_validator

VISUALIZER_ROOT = Path(__file__).resolve().parents[1]
LOCAL_OUTPUT_ROOT = VISUALIZER_ROOT / "data" / "local"


class ConfigurationError(ValueError):
    """Invalid explicit local configuration; abort startup rather than fall back."""


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    research_data_root: Path | None = None
    simulation_python: Path = Path(sys.executable)
    episode_timeout_seconds: int = 900
    frontend_origin: str = "http://127.0.0.1:5173"

    @field_validator("research_data_root")
    @classmethod
    def validate_root(cls, value: Path | None) -> Path | None:
        if value is None:
            return None
        candidate = value if value.is_absolute() else VISUALIZER_ROOT / value
        root = candidate.resolve()
        if not root.is_dir():
            raise ConfigurationError(f"Research data root is not a directory: {root}")
        if not (root / "uav_defend" / "envs" / "soldier_env.py").is_file():
            raise ConfigurationError(f"Research data root is not a research worktree: {root}")
        output = LOCAL_OUTPUT_ROOT.resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ConfigurationError("Research data root and visualization output must be disjoint.")
        return root

    @field_validator("simulation_python")
    @classmethod
    def validate_simulation_python(cls, value: Path) -> Path:
        executable = value if value.is_absolute() else Path(shutil.which(str(value)) or value)
        executable = executable.resolve()
        if not executable.is_file():
            raise ConfigurationError("VISUALIZER_SIMULATION_PYTHON must identify an existing Python executable.")
        return executable

    @field_validator("episode_timeout_seconds")
    @classmethod
    def validate_episode_timeout(cls, value: int) -> int:
        if not 1 <= value <= 3600:
            raise ConfigurationError("Episode timeout must be between 1 and 3600 seconds.")
        return value

    @field_validator("frontend_origin")
    @classmethod
    def validate_frontend_origin(cls, value: str) -> str:
        parsed = urlparse(value)
        if (
            parsed.scheme != "http"
            or parsed.hostname not in {"127.0.0.1", "localhost"}
            or not parsed.port
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ConfigurationError("Frontend origin must be an HTTP localhost origin with an explicit port.")
        return value.rstrip("/")


def load_settings(env_file: Path | None = None) -> Settings:
    path = env_file if env_file is not None else VISUALIZER_ROOT / ".env"
    values = dotenv_values(path) if path.is_file() else {}
    key = "VISUALIZER_RESEARCH_DATA_ROOT"
    raw = os.environ.get(key, values.get(key))
    python_raw = os.environ.get("VISUALIZER_SIMULATION_PYTHON", values.get("VISUALIZER_SIMULATION_PYTHON"))
    timeout_raw = os.environ.get("VISUALIZER_EPISODE_TIMEOUT_SECONDS", values.get("VISUALIZER_EPISODE_TIMEOUT_SECONDS"))
    origin_raw = os.environ.get("VISUALIZER_FRONTEND_ORIGIN", values.get("VISUALIZER_FRONTEND_ORIGIN"))
    try:
        timeout = int(timeout_raw) if timeout_raw else 900
    except ValueError as exc:
        raise ConfigurationError("VISUALIZER_EPISODE_TIMEOUT_SECONDS must be an integer.") from exc
    return Settings(
        research_data_root=Path(raw.strip()) if raw and raw.strip() else None,
        simulation_python=Path(python_raw.strip()) if python_raw and python_raw.strip() else Path(sys.executable),
        episode_timeout_seconds=timeout,
        frontend_origin=origin_raw.strip() if origin_raw and origin_raw.strip() else "http://127.0.0.1:5173",
    )


def local_output_root() -> Path:
    resolved = LOCAL_OUTPUT_ROOT.resolve()
    if not resolved.is_relative_to(VISUALIZER_ROOT.resolve()):
        raise ConfigurationError("Visualization output must remain inside the visualizer worktree.")
    return resolved
