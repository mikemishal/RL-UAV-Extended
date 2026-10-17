import hashlib
from pathlib import Path, PureWindowsPath

from .config import Settings
from .application_error import ApplicationError


class ResearchDataStore:
    """Read-only internal access. Never accept these relative paths from HTTP."""

    def __init__(self, settings: Settings):
        self.root = settings.research_data_root

    def resolve_file(self, relative_path: str) -> Path:
        windows_path = PureWindowsPath(relative_path)
        path = Path(relative_path)
        if (
            not relative_path
            or windows_path.drive
            or windows_path.root
            or path.is_absolute()
            or ".." in windows_path.parts
            or ":" in relative_path
        ):
            raise ApplicationError("INVALID_ARTIFACT_PATH", "Artifact paths must be research-relative.", 400)
        if self.root is None:
            raise ApplicationError("DATA_ROOT_NOT_CONFIGURED", "Configure VISUALIZER_RESEARCH_DATA_ROOT locally.", 503)
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise ApplicationError("INVALID_ARTIFACT_PATH", "Artifact resolves outside the research data root.", 400)
        if not resolved.is_file():
            raise ApplicationError("ARTIFACT_NOT_FOUND", f"Missing research artifact: {relative_path}", 404)
        return resolved

    def read_bytes(self, relative_path: str, expected_sha256: str | None = None) -> bytes:
        path = self.resolve_file(relative_path)
        try:
            with path.open("rb") as source:
                data = source.read()
        except OSError as exc:
            raise ApplicationError("ARTIFACT_READ_FAILED", f"Cannot read research artifact: {relative_path}", 503) from exc
        if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
            raise ApplicationError("CHECKPOINT_HASH_MISMATCH", f"Hash mismatch for research artifact: {relative_path}", 409)
        return data
