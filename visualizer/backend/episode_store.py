"""Atomic, app-local episode storage (brief section 7). A SEPARATE, additive storage area from the existing
`data/local` replay-job/case-library mechanism -- that system is purpose-built for checkpoint-verified SB3
replays and the curated research case library; episodes created by the simplified two-scenario API have no
checkpoint provenance to verify and a much smaller, simpler schema, so a dedicated directory avoids entangling
the two. Every write is atomic (temporary file in the SAME directory, then `os.replace`), mirroring the
established pattern in `case_library.py`.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

from .application_error import ApplicationError
from .episode_models import EpisodeMetadata

EPISODE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _atomic_write_json(path: Path, payload: dict) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".episode-", suffix=".tmp", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ApplicationError("EPISODE_PERSISTENCE_FAILED", "Could not atomically save the episode.", 500) from exc


class EpisodeStore:
    """Metadata + replay JSON for app-generated episodes, under `root` (default `visualizer/data/episodes`).

    Layout: `root/metadata/{episode_id}.json`, `root/replays/{episode_id}.json`. `root` is injectable so tests
    can use a disposable temporary directory instead of the user's real episode library (brief section 18).
    """

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.metadata_dir = self.root / "metadata"
        self.replays_dir = self.root / "replays"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.replays_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _safe_path(self, directory: Path, episode_id: str) -> Path:
        if not EPISODE_ID_RE.fullmatch(episode_id):
            raise ApplicationError("INVALID_EPISODE_ID", "Episode ID must be a 32-character opaque identifier.", 400)
        path = directory / f"{episode_id}.json"
        if path.is_symlink():
            raise ApplicationError("INVALID_EPISODE_PATH", "Episode file cannot be a symbolic link.", 403)
        resolved_dir = directory.resolve()
        if not path.resolve().is_relative_to(resolved_dir):
            raise ApplicationError("INVALID_EPISODE_PATH", "Episode path must remain under episode storage.", 403)
        return path

    def create(self, metadata: EpisodeMetadata, replay: dict) -> EpisodeMetadata:
        with self._lock:
            meta_path = self._safe_path(self.metadata_dir, metadata.episode_id)
            replay_path = self._safe_path(self.replays_dir, metadata.episode_id)
            if meta_path.exists():
                raise ApplicationError("EPISODE_ALREADY_EXISTS", f"Episode {metadata.episode_id} already exists.", 409)
            _atomic_write_json(replay_path, replay)
            _atomic_write_json(meta_path, metadata.model_dump(mode="json"))
            return metadata

    def _read_metadata(self, path: Path) -> EpisodeMetadata | None:
        try:
            return EpisodeMetadata.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def list(self) -> list[EpisodeMetadata]:
        with self._lock:
            items: list[EpisodeMetadata] = []
            for path in self.metadata_dir.glob("*.json"):
                record = self._read_metadata(path)
                if record is not None:
                    items.append(record)
            items.sort(key=lambda e: e.created_at, reverse=True)
            return items

    def get(self, episode_id: str) -> tuple[EpisodeMetadata, dict]:
        with self._lock:
            meta_path = self._safe_path(self.metadata_dir, episode_id)
            replay_path = self._safe_path(self.replays_dir, episode_id)
            record = self._read_metadata(meta_path) if meta_path.exists() else None
            if record is None:
                raise ApplicationError("EPISODE_NOT_FOUND", f"No episode {episode_id!r} exists.", 404)
            try:
                replay = json.loads(replay_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ApplicationError("EPISODE_REPLAY_UNREADABLE", f"Episode {episode_id!r} has no readable replay.", 500) from exc
            return record, replay

    def mark_viewed(self, episode_id: str) -> EpisodeMetadata:
        """Marks an episode `is_new=False` on first open/replay (brief section 6). A no-op, not an error, when
        already viewed -- opening an episode repeatedly must never fail."""
        with self._lock:
            meta_path = self._safe_path(self.metadata_dir, episode_id)
            record = self._read_metadata(meta_path) if meta_path.exists() else None
            if record is None:
                raise ApplicationError("EPISODE_NOT_FOUND", f"No episode {episode_id!r} exists.", 404)
            if record.is_new:
                record = record.model_copy(update={"is_new": False})
                _atomic_write_json(meta_path, record.model_dump(mode="json"))
            return record

    def delete(self, episode_id: str) -> bool:
        """Deletes ONE episode's metadata and its own replay file. Never touches any other file -- shared
        checkpoints/models/research source artifacts are never under this store's directories at all (brief
        section 11)."""
        with self._lock:
            meta_path = self._safe_path(self.metadata_dir, episode_id)
            replay_path = self._safe_path(self.replays_dir, episode_id)
            existed = meta_path.exists()
            meta_path.unlink(missing_ok=True)
            replay_path.unlink(missing_ok=True)
            return existed

    def delete_many(self, episode_ids: list[str]) -> dict[str, bool]:
        return {episode_id: self.delete(episode_id) for episode_id in episode_ids}

    def delete_saved_older_than(self, cutoff_iso: str) -> list[str]:
        """Optional clear-old helper (brief section 13): deletes SAVED (is_new=False) episodes with
        `created_at < cutoff_iso` only. NEW episodes, and anything outside this store's own directories, are
        never touched."""
        deleted: list[str] = []
        with self._lock:
            for record in self.list():
                if not record.is_new and record.created_at < cutoff_iso:
                    if self.delete(record.episode_id):
                        deleted.append(record.episode_id)
        return deleted
