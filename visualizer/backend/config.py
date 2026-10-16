import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, field_validator

VISUALIZER_ROOT = Path(__file__).resolve().parents[1]
LOCAL_OUTPUT_ROOT = VISUALIZER_ROOT / "data" / "local"


class ConfigurationError(ValueError):
    """Invalid explicit local configuration; abort startup rather than fall back."""


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    research_data_root: Path | None = None

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


def load_settings(env_file: Path | None = None) -> Settings:
    path = env_file if env_file is not None else VISUALIZER_ROOT / ".env"
    values = dotenv_values(path) if path.is_file() else {}
    key = "VISUALIZER_RESEARCH_DATA_ROOT"
    raw = os.environ.get(key, values.get(key))
    return Settings(research_data_root=Path(raw.strip()) if raw and raw.strip() else None)


def local_output_root() -> Path:
    resolved = LOCAL_OUTPUT_ROOT.resolve()
    if not resolved.is_relative_to(VISUALIZER_ROOT.resolve()):
        raise ConfigurationError("Visualization output must remain inside the visualizer worktree.")
    return resolved
