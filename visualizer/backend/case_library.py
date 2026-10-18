from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, ValidationError, field_validator, model_validator

from .application_error import ApplicationError
from .config import local_output_root
from .replay_contract import ReplayBundle, ReplayEpisode

CASE_SCHEMA_VERSION = 1
CASE_ID_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_CASES = 2000
MAX_BOOKMARKS = 1000
MAX_TAGS = 64
AnnotatedFinite = FiniteFloat


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class CaseBookmark(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    bookmark_id: str = Field(default_factory=lambda: uuid4().hex, pattern=r"^[0-9a-f]{32}$")
    title: str = Field(min_length=1, max_length=160)
    time_s: AnnotatedFinite = Field(ge=-1_000_000_000)
    frame_step: int = Field(ge=0)
    decision_step: int | None = Field(default=None, ge=0)
    sync_mode: Literal["absolute", "detection"] = "absolute"
    local_time_a_s: AnnotatedFinite | None = Field(default=None, ge=0)
    local_time_b_s: AnnotatedFinite | None = Field(default=None, ge=0)
    observation: str = Field(default="", max_length=4000)
    interpretation: str = Field(default="", max_length=4000)
    tags: list[str] = Field(default_factory=list, max_length=MAX_TAGS)
    camera: dict[str, object] = Field(default_factory=dict)

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        return _clean_tags(value)

    @field_validator("camera")
    @classmethod
    def bound_camera(cls, value: dict[str, object]) -> dict[str, object]:
        if len(json.dumps(value, allow_nan=False)) > 8000:
            raise ValueError("Camera and display settings are limited to 8 KiB.")
        return value


class ReplayReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replay_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class StoredResearchCase(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    schema_version: Literal[CASE_SCHEMA_VERSION]
    case_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    case_type: Literal["single", "comparison"]
    version: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(max_length=2000)
    tags: list[str] = Field(max_length=MAX_TAGS)
    notes: str = Field(max_length=12000)
    replay_references: list[ReplayReference] = Field(min_length=1, max_length=2)
    episodes: list[dict]
    comparison_validation: dict | None
    bookmarks: list[CaseBookmark] = Field(max_length=MAX_BOOKMARKS)
    created_at: str
    modified_at: str

    @model_validator(mode="after")
    def reference_count_matches_case_type(self):
        expected = 1 if self.case_type == "single" else 2
        if len(self.replay_references) != expected or len(self.episodes) != expected:
            raise ValueError("Stored replay and episode counts must match the case type.")
        return self





class CaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_type: Literal["single", "comparison"]
    replay_ids: list[str] = Field(min_length=1, max_length=2)
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)
    tags: list[str] = Field(default_factory=list, max_length=MAX_TAGS)
    notes: str = Field(default="", max_length=12000)
    bookmarks: list[CaseBookmark] = Field(default_factory=list, max_length=MAX_BOOKMARKS)

    @field_validator("replay_ids")
    @classmethod
    def validate_replay_ids(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value) or any(not CASE_ID_RE.fullmatch(item) for item in value):
            raise ValueError("Replay references must be distinct opaque replay IDs.")
        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str]) -> list[str]:
        return _clean_tags(value)

    @field_validator("title", "description", "notes")
    @classmethod
    def trim_text(cls, value: str) -> str:
        return value.strip()

class CasePatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    tags: list[str] | None = Field(default=None, max_length=MAX_TAGS)
    notes: str | None = Field(default=None, max_length=12000)

    @field_validator("title", "description", "notes")
    @classmethod
    def trim_optional_text(cls, value: str | None) -> str | None:
        return value.strip() if value is not None else value

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str] | None) -> list[str] | None:
        return _clean_tags(value) if value is not None else value


class BookmarkCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    bookmark: CaseBookmark


class BookmarkPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_version: int = Field(ge=1)
    title: str | None = Field(default=None, min_length=1, max_length=160)
    time_s: AnnotatedFinite | None = Field(default=None, ge=-1_000_000_000)
    frame_step: int | None = Field(default=None, ge=0)
    decision_step: int | None = Field(default=None, ge=0)
    sync_mode: Literal["absolute", "detection"] | None = None
    local_time_a_s: AnnotatedFinite | None = Field(default=None, ge=0)
    local_time_b_s: AnnotatedFinite | None = Field(default=None, ge=0)
    observation: str | None = Field(default=None, max_length=4000)
    interpretation: str | None = Field(default=None, max_length=4000)
    tags: list[str] | None = Field(default=None, max_length=MAX_TAGS)
    camera: dict[str, object] | None = None

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, value: list[str] | None) -> list[str] | None:
        return _clean_tags(value) if value is not None else value


def _clean_tags(value: list[str]) -> list[str]:
    cleaned = [tag.strip() for tag in value]
    if any(not tag or len(tag) > 64 for tag in cleaned) or len(set(map(str.casefold, cleaned))) != len(cleaned):
        raise ValueError("Tags must be non-empty, unique, and at most 64 characters.")
    return cleaned


def _case_root() -> Path:
    root = local_output_root().resolve()
    target = root / "cases"
    if target.is_symlink():
        raise ApplicationError("INVALID_CASE_ROOT", "Research case storage cannot be a symbolic link.", 500)
    target.mkdir(parents=True, exist_ok=True)
    resolved = target.resolve()
    if not resolved.is_relative_to(root):
        raise ApplicationError("INVALID_CASE_ROOT", "Research case storage must remain under local output.", 500)
    return resolved


def _atomic_json(path: Path, payload: dict) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".case-", suffix=".tmp", dir=path.parent, delete=False) as stream:
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
        raise ApplicationError("CASE_PERSISTENCE_FAILED", "Could not atomically save the research case.", 500) from exc


def _episode_metadata(episode: ReplayEpisode, provenance: dict) -> dict:
    return {
        "defender": episode.defender.model_dump(mode="json"),
        "attacker": episode.attacker.model_dump(mode="json"),
        "environment": episode.environment.model_dump(mode="json"),
        "simulation_seed": episode.simulation_seed,
        "outcome": episode.summary.outcome,
        "duration_s": episode.summary.duration_s,
        "detection_time_s": episode.summary.detection_time_s,
        "summary": episode.summary.model_dump(mode="json"),
        "provenance": provenance,
    }


def _pair_status(left: ReplayBundle, right: ReplayBundle) -> tuple[str, list[str]]:
    a, b = left.episodes[0], right.episodes[0]
    incompatible: list[str] = []
    exploratory: list[str] = []
    if a.simulation_seed != b.simulation_seed:
        incompatible.append("Evaluation seeds differ.")
    if a.attacker.policy_id != b.attacker.policy_id or a.attacker.checkpoint != b.attacker.checkpoint:
        incompatible.append("Attacker identities or checkpoint hashes differ.")
    if (a.environment.clutter, a.environment.awareness, a.environment.dt_s) != (
        b.environment.clutter, b.environment.awareness, b.environment.dt_s
    ):
        incompatible.append("Canonical scenario or timestep differs.")
    config_a = {key: value for key, value in a.environment.config.items() if key != "use_kalman_tracking"}
    config_b = {key: value for key, value in b.environment.config.items() if key != "use_kalman_tracking"}
    if config_a != config_b:
        incompatible.append("Resolved physical environment configuration differs.")
    initial_a, initial_b = a.frames[0], b.frames[0]
    world_a = [initial_a.defender, initial_a.attacker, initial_a.asset, initial_a.moving_obstacles, a.static_obstacles]
    world_b = [initial_b.defender, initial_b.attacker, initial_b.asset, initial_b.moving_obstacles, b.static_obstacles]
    if [item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in world_a] != [
        item.model_dump(mode="json") if hasattr(item, "model_dump") else item for item in world_b
    ]:
        incompatible.append("Initial world state or obstacle geometry differs.")
    if left.provenance.simulation_source_commit != right.provenance.simulation_source_commit:
        incompatible.append("Simulator source revisions differ.")
    if left.provenance.python_version != right.provenance.python_version or left.provenance.package_versions != right.provenance.package_versions:
        incompatible.append("Python or scientific package versions differ.")
    if left.provenance.deterministic_inference is not True or right.provenance.deterministic_inference is not True:
        incompatible.append("Deterministic inference is not recorded.")
    if a.defender.estimator != b.defender.estimator:
        exploratory.append("Controller estimator contracts differ.")
    if a.environment.config_sha256 != b.environment.config_sha256:
        exploratory.append("Full environment configuration fingerprints differ.")
    if left.provenance.source_dirty or right.provenance.source_dirty:
        exploratory.append("At least one trace has dirty simulator source provenance.")
    if left.provenance.visualizer_dirty or right.provenance.visualizer_dirty:
        exploratory.append("At least one trace has dirty visualizer source provenance.")
    if left.provenance.visualizer_source_commit != right.provenance.visualizer_source_commit:
        exploratory.append("Replay recorder revisions differ.")
    if incompatible:
        return "incompatible", incompatible
    if exploratory:
        return "exploratory", exploratory
    return "matched", ["Shared initial world state and clean compatible recorded provenance."]


class CaseLibrary:
    def __init__(self, replay_getter):
        self._replay_getter = replay_getter
        self._lock = threading.RLock()

    @staticmethod
    def _path(case_id: str) -> Path:
        if not CASE_ID_RE.fullmatch(case_id):
            raise ApplicationError("INVALID_CASE_ID", "Case ID must be a 32-character opaque identifier.", 400)
        root = _case_root()
        path = root / f"{case_id}.json"
        if path.is_symlink():
            raise ApplicationError("INVALID_CASE_PATH", "Research case sidecar cannot be a symbolic link.", 403)
        if not path.resolve().is_relative_to(root):
            raise ApplicationError("INVALID_CASE_PATH", "Research case must remain under local case storage.", 403)
        return path

    def _read(self, case_id: str) -> dict:
        path = self._path(case_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ApplicationError("CASE_NOT_FOUND", "Research case was not found.", 404) from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise ApplicationError("INVALID_CASE_FILE", "Research case sidecar is unreadable or invalid JSON.", 409) from exc
        try:
            validated = StoredResearchCase.model_validate(payload)
        except (ValidationError, TypeError) as exc:
            raise ApplicationError("INVALID_CASE_FILE", "Research case sidecar does not match the supported schema.", 409) from exc
        if validated.case_id != case_id:
            raise ApplicationError("INVALID_CASE_FILE", "Research case identifier does not match its sidecar filename.", 409)
        return validated.model_dump(mode="json")

    def _checked_replay(self, replay_id: str, expected_sha256: str | None = None) -> tuple[ReplayBundle, str]:
        digest = self._artifact_digest(replay_id, expected_sha256)
        bundle = self._replay_getter(replay_id)
        return bundle, digest

    @staticmethod
    def _artifact_digest(replay_id: str, expected_sha256: str | None = None) -> str:
        root = local_output_root().resolve()
        path = None
        for candidate in root.glob("*.json.gz"):
            from .replay_jobs import _replay_id_for_path, _safe_replay_path
            if _replay_id_for_path(candidate) == replay_id:
                if not _safe_replay_path(candidate):
                    raise ApplicationError("INVALID_REPLAY_PATH", "Replay artifact is outside local output.", 403)
                path = candidate
                break
        if path is None:
            raise ApplicationError("REPLAY_NOT_FOUND", "Referenced replay artifact is unavailable.", 404)
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ApplicationError("REPLAY_READ_FAILED", "Could not verify referenced replay integrity.", 503) from exc
        if expected_sha256 is not None and digest != expected_sha256:
            raise ApplicationError("REPLAY_HASH_MISMATCH", "Referenced replay contents changed since the case was saved.", 409)
        return digest

    def _integrity(self, payload: dict) -> dict:
        references = payload.get("replay_references", [])
        statuses = []
        for reference in references:
            try:
                self._artifact_digest(reference["replay_id"], reference["sha256"])
                statuses.append("valid")
            except ApplicationError as exc:
                statuses.append("missing" if exc.status_code == 404 else "changed")
        return {
            "status": "valid" if all(value == "valid" for value in statuses) else "unresolved",
            "replay_statuses": statuses,
        }

    def _public_case(self, payload: dict) -> dict:
        integrity = self._integrity(payload)
        validation = payload.get("comparison_validation")
        provenance = [entry.get("provenance", {}) for entry in payload["episodes"]]
        bookmarks = [{
            **bookmark,
            "label": bookmark["title"],
            "camera_pose": bookmark["camera"],
        } for bookmark in payload["bookmarks"]]
        return {
            **payload,
            "replay_ids": [item["replay_id"] for item in payload["replay_references"]],
            "replay_references": [{
                "replay_id": item["replay_id"],
                "artifact_sha256": item["sha256"],
                "integrity": integrity["replay_statuses"][index],
            } for index, item in enumerate(payload["replay_references"])],
            "integrity": integrity["status"],
            "comparison_status": validation["status"] if validation else None,
            "comparison_reasons": validation["reasons"] if validation else [],
            "metadata": {
                "defender_families": [entry["defender"]["policy_id"] for entry in payload["episodes"]],
                "generations": [entry["defender"]["generation"] for entry in payload["episodes"] if entry["defender"]["generation"]],
                "training_seeds": [entry["defender"]["training_seed"] for entry in payload["episodes"] if entry["defender"]["training_seed"] is not None],
                "attackers": [entry["attacker"]["policy_id"] for entry in payload["episodes"]],
                "clutter": [entry["environment"]["clutter"] for entry in payload["episodes"]],
                "perception": [entry["environment"]["awareness"] for entry in payload["episodes"]],
                "simulation_seeds": [entry["simulation_seed"] for entry in payload["episodes"]],
                "outcomes": [entry["outcome"] for entry in payload["episodes"]],
                "source_commits": sorted({entry.get("simulation_source_commit", "") for entry in provenance if entry.get("simulation_source_commit")}),
                "dirty_provenance": any(entry.get("source_dirty") or entry.get("visualizer_dirty") for entry in provenance),
                "created_at": payload["created_at"],
            },
            "bookmarks": bookmarks,
        }

    def create(self, request: CaseCreate) -> dict:
        if (request.case_type == "single" and len(request.replay_ids) != 1) or (
            request.case_type == "comparison" and len(request.replay_ids) != 2
        ):
            raise ApplicationError("INVALID_CASE", "Single cases require one replay; comparison cases require exactly two.", 422)
        with self._lock:
            if len(list(_case_root().glob("*.json"))) >= MAX_CASES:
                raise ApplicationError("CASE_LIMIT_REACHED", "Local case library has reached its configured limit.", 409)
        bundles: list[ReplayBundle] = []
        references = []
        for replay_id in request.replay_ids:
            bundle, digest = self._checked_replay(replay_id)
            bundles.append(bundle)
            references.append({"replay_id": replay_id, "sha256": digest})
        episodes = [bundle.episodes[0] for bundle in bundles]
        bookmarks = [item.model_dump(mode="json") for item in request.bookmarks]
        for bookmark in bookmarks:
            self._validate_bookmark(bookmark, episodes)
        case_id = uuid4().hex
        payload = {
            "schema_version": CASE_SCHEMA_VERSION,
            "case_id": case_id,
            "case_type": request.case_type,
            "version": 1,
            "title": request.title,
            "description": request.description,
            "tags": request.tags,
            "notes": request.notes,
            "replay_references": references,
            "episodes": [
                _episode_metadata(episode, bundle.provenance.model_dump(mode="json"))
                for episode, bundle in zip(episodes, bundles)
            ],
            "comparison_validation": None if len(bundles) == 1 else dict(zip(("status", "reasons"), _pair_status(*bundles))),
            "bookmarks": bookmarks,
            "created_at": _now(),
            "modified_at": _now(),
        }
        path = self._path(case_id)
        if path.exists():
            raise ApplicationError("CASE_ID_COLLISION", "Generated case identifier already exists; retry creation.", 409)
        with self._lock:
            if path.exists():
                raise ApplicationError("CASE_ID_COLLISION", "Generated case identifier already exists; retry creation.", 409)
            _atomic_json(path, payload)
        return self._public_case(payload)

    @staticmethod
    def _validate_bookmark(bookmark: dict, episodes: list[ReplayEpisode]) -> None:
        frame = bookmark["frame_step"]
        if frame >= len(episodes[0].frames):
            raise ApplicationError("INVALID_BOOKMARK_TIME", "Bookmark frame is outside the first recorded episode.", 422)
        local_a = bookmark.get("local_time_a_s")
        local_a = bookmark["time_s"] if local_a is None and bookmark.get("sync_mode") == "absolute" else local_a
        if local_a is None:
            raise ApplicationError("INVALID_BOOKMARK_TIME", "A comparison bookmark requires Policy A local time.", 422)
        if local_a > episodes[0].summary.duration_s:
            raise ApplicationError("INVALID_BOOKMARK_TIME", "Bookmark time is after the first recorded terminal state.", 422)
        if local_a < 0:
            raise ApplicationError("INVALID_BOOKMARK_TIME", "Local simulation time cannot be negative.", 422)
        episode_a = episodes[0]
        frame_a = episode_a.frames[frame]
        if local_a < frame_a.time_s - 1e-6 or (
            frame + 1 < len(episode_a.frames) and local_a > episode_a.frames[frame + 1].time_s + 1e-6
        ):
            raise ApplicationError("INVALID_BOOKMARK_TIME", "Policy A local time does not fall within the selected recorded frame interval.", 422)
        detection_a = episode_a.summary.detection_time_s
        offset_a = detection_a if bookmark["sync_mode"] == "detection" and detection_a is not None else 0.0
        expected_a = max(0.0, min(episode_a.summary.duration_s, bookmark["time_s"] + offset_a))
        if bookmark["sync_mode"] == "detection" and detection_a is None:
            raise ApplicationError("INVALID_BOOKMARK_TIME", "Detection-relative bookmark requires a recorded Policy A detection time.", 422)
        if abs(expected_a - local_a) > 1e-6:
            raise ApplicationError("INVALID_BOOKMARK_TIME", "Bookmark axis time does not map to the recorded Policy A local time.", 422)
        decision_step = bookmark["decision_step"]
        if decision_step is not None and (
            decision_step >= episode_a.summary.steps
            or frame_a.telemetry is None
            or frame_a.telemetry.decision_step != decision_step
        ):
            raise ApplicationError("INVALID_BOOKMARK_TIME", "Bookmark decision does not match the recorded decision interval at the selected frame.", 422)
        if bookmark.get("local_time_b_s") is not None and bookmark["local_time_b_s"] < 0:
            raise ApplicationError("INVALID_BOOKMARK_TIME", "Policy B local simulation time cannot be negative.", 422)
        if len(episodes) == 2 and bookmark.get("local_time_b_s") is not None:
            if bookmark["local_time_b_s"] > episodes[1].summary.duration_s:
                raise ApplicationError("INVALID_BOOKMARK_TIME", "Policy B bookmark time is after its terminal state.", 422)
            detection_b = episodes[1].summary.detection_time_s
            offset_b = detection_b if bookmark["sync_mode"] == "detection" and detection_b is not None else 0.0
            expected_b = max(0.0, min(episodes[1].summary.duration_s, bookmark["time_s"] + offset_b))
            if bookmark["sync_mode"] == "detection" and detection_b is None:
                raise ApplicationError("INVALID_BOOKMARK_TIME", "Detection-relative bookmark requires a recorded Policy B detection time.", 422)
            if abs(expected_b - bookmark["local_time_b_s"]) > 1e-6:
                raise ApplicationError("INVALID_BOOKMARK_TIME", "Bookmark axis time does not map to the recorded Policy B local time.", 422)

    def get(self, case_id: str) -> dict:
        payload = self._read(case_id)
        return self._public_case(payload)

    def load_for_export(self, case_id: str) -> tuple[dict, list[ReplayBundle]]:
        payload = self._read(case_id)
        bundles = [
            self._checked_replay(reference["replay_id"], reference["sha256"])[0]
            for reference in payload["replay_references"]
        ]
        return {**payload, **self._integrity(payload)}, bundles

    def list(self) -> list[dict]:
        root = _case_root()
        output = []
        for path in root.glob("*.json"):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                continue
            payload = self._read(path.stem)
            output.append(self._public_case(payload))
        return output

    def patch(self, case_id: str, request: CasePatch) -> dict:
        with self._lock:
            payload = self._read(case_id)
            self._check_version(payload, request.expected_version)
            for field in ("title", "description", "tags", "notes"):
                value = getattr(request, field)
                if value is not None:
                    payload[field] = value
            return self._save_modified(payload)

    @staticmethod
    def _check_version(payload: dict, expected: int) -> None:
        if payload["version"] != expected:
            raise ApplicationError("CASE_VERSION_CONFLICT", "Research case changed since it was loaded; refresh before saving.", 409)

    def delete(self, case_id: str, expected_version: int) -> None:
        with self._lock:
            payload = self._read(case_id)
            self._check_version(payload, expected_version)
            try:
                self._path(case_id).unlink()
            except OSError as exc:
                raise ApplicationError("CASE_DELETE_FAILED", "Could not delete the case sidecar; referenced replay files were not changed.", 500) from exc

    def add_bookmark(self, case_id: str, request: BookmarkCreate) -> dict:
        with self._lock:
            payload = self._read(case_id)
            self._check_version(payload, request.expected_version)
            if len(payload["bookmarks"]) >= MAX_BOOKMARKS:
                raise ApplicationError("BOOKMARK_LIMIT_REACHED", "Case has reached the bookmark limit.", 409)
            bundles = [self._checked_replay(item["replay_id"], item["sha256"])[0] for item in payload["replay_references"]]
            bookmark = request.bookmark.model_dump(mode="json")
            self._validate_bookmark(bookmark, [bundle.episodes[0] for bundle in bundles])
            if any(item["bookmark_id"] == bookmark["bookmark_id"] for item in payload["bookmarks"]):
                raise ApplicationError("DUPLICATE_BOOKMARK", "Bookmark identifier already exists in this case.", 409)
            payload["bookmarks"].append(bookmark)
            return self._save_modified(payload)

    def patch_bookmark(self, case_id: str, bookmark_id: str, request: BookmarkPatch) -> dict:
        with self._lock:
            payload = self._read(case_id)
            self._check_version(payload, request.expected_version)
            bookmark = next((item for item in payload["bookmarks"] if item["bookmark_id"] == bookmark_id), None)
            if bookmark is None:
                raise ApplicationError("BOOKMARK_NOT_FOUND", "Bookmark was not found in this case.", 404)
            for key, value in request.model_dump(exclude={"expected_version"}, exclude_unset=True).items():
                if value is not None:
                    bookmark[key] = value
            validated = CaseBookmark.model_validate(bookmark).model_dump(mode="json")
            bundles = [self._checked_replay(item["replay_id"], item["sha256"])[0] for item in payload["replay_references"]]
            self._validate_bookmark(validated, [bundle.episodes[0] for bundle in bundles])
            payload["bookmarks"] = [validated if item["bookmark_id"] == bookmark_id else item for item in payload["bookmarks"]]
            return self._save_modified(payload)

    def delete_bookmark(self, case_id: str, bookmark_id: str, expected_version: int) -> dict:
        with self._lock:
            payload = self._read(case_id)
            self._check_version(payload, expected_version)
            filtered = [item for item in payload["bookmarks"] if item["bookmark_id"] != bookmark_id]
            if len(filtered) == len(payload["bookmarks"]):
                raise ApplicationError("BOOKMARK_NOT_FOUND", "Bookmark was not found in this case.", 404)
            payload["bookmarks"] = filtered
            return self._save_modified(payload)

    def _save_modified(self, payload: dict) -> dict:
        payload["version"] += 1
        payload["modified_at"] = _now()
        _atomic_json(self._path(payload["case_id"]), payload)
        return self._public_case(payload)


def filter_cases(
    cases: list[dict],
    *,
    query: str = "",
    architecture: str | None = None,
    generation: str | None = None,
    training_seed: int | None = None,
    attacker: str | None = None,
    clutter: str | None = None,
    perception: str | None = None,
    evaluation_seed: int | None = None,
    outcome: str | None = None,
    case_type: str | None = None,
    tag: str | None = None,
    annotations: bool | None = None,
    bookmarked: bool | None = None,
    sort: Literal["modified", "created", "policy", "outcome", "duration"] = "modified",
) -> list[dict]:
    needle = query.casefold().strip()
    result = []
    for case in cases:
        text = " ".join([case.get("title", ""), case.get("description", ""), case.get("notes", "")]
            + case.get("tags", [])
            + [str(value) for entry in case.get("episodes", []) for value in (
                entry.get("defender", {}).get("policy_id", ""),
                entry.get("defender", {}).get("architecture", ""),
                entry.get("attacker", {}).get("policy_id", ""),
                entry.get("environment", {}).get("clutter", ""),
                entry.get("outcome", ""),
            )] + [
            " ".join([item.get("observation", ""), item.get("interpretation", "")])
            for item in case.get("bookmarks", [])
        ]).casefold()
        if needle and needle not in text:
            continue
        if case_type and case.get("case_type") != case_type:
            continue
        if tag and tag.casefold() not in {value.casefold() for value in case.get("tags", [])}:
            continue
        if annotations is not None and bool(case.get("bookmarks")) != annotations:
            continue
        if bookmarked is not None and bool(case.get("bookmarks")) != bookmarked:
            continue
        metadata = case.get("episodes", [])
        def matches(item: dict) -> bool:
            defender = item["defender"]
            env = item["environment"]
            attacker_info = item["attacker"]
            return (not architecture or defender.get("architecture") == architecture) \
                and (not generation or defender.get("generation") == generation) \
                and (training_seed is None or defender.get("training_seed") == training_seed) \
                and (not attacker or attacker_info.get("policy_id") == attacker) \
                and (not clutter or env.get("clutter") == clutter) \
                and (not perception or env.get("awareness") == perception) \
                and (evaluation_seed is None or item.get("simulation_seed") == evaluation_seed) \
                and (not outcome or item.get("outcome") == outcome)
        if any(not matches(item) for item in metadata):
            continue
        result.append(case)
    if sort == "created":
        key = lambda item: item.get("created_at", "")
    elif sort == "policy":
        key = lambda item: ",".join(entry["defender"]["policy_id"] for entry in item.get("episodes", []))
    elif sort == "outcome":
        key = lambda item: ",".join(entry["outcome"] for entry in item.get("episodes", []))
    elif sort == "duration":
        key = lambda item: max((entry["duration_s"] for entry in item.get("episodes", [])), default=0)
    else:
        key = lambda item: item.get("modified_at", "")
    return sorted(result, key=key, reverse=True)
