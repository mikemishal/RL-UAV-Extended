from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from .application_error import ApplicationError
from .catalog_service import CatalogProvider
from .config import Settings, VISUALIZER_ROOT, local_output_root
from .replay_contract import ReplayBundle
from .replay_requests import ReplayBatchRequest
from .replay_validation import validate_trace

logger = logging.getLogger(__name__)
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
REPLAY_ID_RE = re.compile(r"^[0-9a-f]{32}$")
MAX_REPLAY_UNCOMPRESSED_BYTES = 512 * 1024 * 1024
POLL_SECONDS = 0.2
CANCEL_GRACE_SECONDS = 5.0
MAX_EVALUATION_SEED = 2**32 - 1
SAFE_WORKER_ERRORS = {
    "SIMULATION_DEPENDENCY_MISSING": "Simulation dependencies are unavailable in the configured Python runtime.",
    "ARTIFACT_NOT_FOUND": "A registered checkpoint required by this selection is missing.",
    "CHECKPOINT_HASH_MISMATCH": "A selected checkpoint failed registry SHA-256 verification.",
    "CHECKPOINT_MUTATED": "A selected checkpoint changed during inference; the replay was rejected.",
    "INVALID_MODEL_REGISTRY": "The frozen model registry failed validation.",
    "CHECKPOINT_NOT_REGISTERED": "The selected policy has no registered checkpoint for this generation and seed.",
    "UNVERIFIED_SIMULATOR_COMMIT": "The configured simulator source revision does not match the verified research commit.",
    "DATA_ROOT_NOT_CONFIGURED": "Configure the canonical research data root before starting replay jobs.",
    "INCOMPATIBLE_POLICY": "The selected policy configuration is incompatible with its registered model.",
    "INCOMPATIBLE_CHECKPOINT": "The registered checkpoint does not match its required observation/action contract.",
    "INVALID_REQUEST": "The selected simulation configuration is invalid.",
    "SEMANTICS_CONFIG_MISMATCH": "The research config did not resolve to the requested V1/V2 semantics.",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReplayJob(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    request: ReplayBatchRequest
    state: Literal["queued", "running", "completed", "failed", "cancelled", "interrupted"]
    requested_episode_count: int
    completed_count: int = 0
    current_episode_seed: int | None = None
    completed_replay_ids: list[str] = []
    error: str | None = None
    cancellation_requested: bool = False
    created_at: str
    started_at: str | None = None
    updated_at: str
    completed_at: str | None = None


class ReplayListItem(BaseModel):
    replay_id: str
    simulation_seed: int
    defender: dict
    attacker: dict
    clutter: str
    perception: str
    semantics: str = "V1"
    outcome: str
    steps: int
    duration_s: float
    detection_time_s: float | None
    total_reward: float
    collision_type: str | None
    interception_time_s: float | None
    provenance: dict


class ReplayCatalog(BaseModel):
    replays: list[ReplayListItem]
    warnings: list[str] = []


def _replay_id_for_path(path: Path) -> str:
    return hashlib.sha256(f"counter-uas-replay-v1:{path.name}".encode("utf-8")).hexdigest()[:32]


def _job_directory() -> Path:
    root = local_output_root()
    root.mkdir(parents=True, exist_ok=True)
    directory = root / "jobs"
    if directory.is_symlink():
        raise ApplicationError("INVALID_OUTPUT_ROOT", "Replay job storage cannot be a symbolic link.", 500)
    directory.mkdir(exist_ok=True)
    resolved = directory.resolve()
    if not resolved.is_relative_to(root):
        raise ApplicationError("INVALID_OUTPUT_ROOT", "Replay job storage must remain under the local output directory.", 500)
    return resolved


def _load_bundle(path: Path) -> ReplayBundle:
    try:
        with gzip.open(path, "rb") as source:
            raw = source.read(MAX_REPLAY_UNCOMPRESSED_BYTES + 1)
        if len(raw) > MAX_REPLAY_UNCOMPRESSED_BYTES:
            raise ApplicationError("REPLAY_TOO_LARGE", "Replay exceeds the API's 512 MiB uncompressed limit.", 413)
        bundle = ReplayBundle.model_validate_json(raw)
        validate_trace(bundle)
        return bundle
    except ApplicationError:
        raise
    except (OSError, EOFError, UnicodeDecodeError, ValueError) as exc:
        raise ApplicationError("INVALID_REPLAY_FILE", "Replay artifact is not a valid compressed schema-v1 replay.", 409) from exc


def _safe_replay_path(path: Path) -> bool:
    root = local_output_root().resolve()
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    return resolved.is_relative_to(root) and resolved.is_file() and not path.is_symlink()


class ReplayJobManager:
    """Single active sequential batch, isolated simulation process per episode."""

    def __init__(self, settings: Settings, catalog: CatalogProvider):
        self.settings = settings
        self.catalog = catalog
        self._lock = threading.RLock()
        self._jobs: dict[str, ReplayJob] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay-batch")
        self._future: Future[None] | None = None
        self._load_persisted_jobs()

    def _persist(self, job: ReplayJob) -> None:
        job.updated_at = _now()
        directory = _job_directory()
        destination = directory / f"{job.job_id}.json"
        content = job.model_dump_json(indent=2).encode("utf-8")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(prefix=".job-", suffix=".tmp", dir=directory, delete=False) as target:
                temporary = Path(target.name)
                target.write(content)
                target.flush()
                os.fsync(target.fileno())
            for attempt in range(5):
                try:
                    os.replace(temporary, destination)
                    break
                except PermissionError:
                    if attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        except OSError as exc:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove temporary replay job metadata after a persistence failure.")
            raise ApplicationError("JOB_PERSISTENCE_FAILED", "Could not atomically persist replay job state.", 500) from exc

    def _load_persisted_jobs(self) -> None:
        directory = _job_directory()
        for path in directory.glob("*.json"):
            try:
                job = ReplayJob.model_validate_json(path.read_bytes())
            except (OSError, ValueError) as exc:
                logger.error("Ignoring unreadable replay job metadata %s: %s", path.name, exc)
                continue
            if not JOB_ID_RE.fullmatch(job.job_id) or path.name != f"{job.job_id}.json":
                logger.error("Ignoring replay job metadata with an invalid identifier: %s", path.name)
                continue
            if job.state in {"queued", "running"}:
                job.state = "interrupted"
                job.current_episode_seed = None
                job.cancellation_requested = False
                job.error = "Server restarted before this batch completed; no incomplete episode was reported as complete."
                job.completed_at = _now()
                self._persist(job)
            self._jobs[job.job_id] = job

    def _validate_available_selection(self, request: ReplayBatchRequest) -> None:
        catalog = self.catalog.get()
        defender = next((item for item in catalog["defenders"] if item["policy_id"] == request.defender_policy_id), None)
        if defender is None or not defender["available"]:
            raise ApplicationError("POLICY_UNAVAILABLE", "The selected defender is not available in the frozen policy catalog.", 409)
        if defender["classification"] == "learned":
            generation = next(
                (item for item in defender["generations"] if item["generation"] == request.defender_generation),
                None,
            )
            seed = next(
                (item for item in generation["training_seeds"] if item["training_seed"] == request.defender_training_seed),
                None,
            ) if generation else None
            if seed is None or not seed["available"]:
                raise ApplicationError(
                    "CHECKPOINT_UNAVAILABLE",
                    "The selected defender checkpoint is missing or failed its registry integrity check.",
                    409,
                )
        attacker = next((item for item in catalog["attackers"] if item["attacker_id"] == request.attacker_condition), None)
        if attacker is None or not attacker["available"]:
            raise ApplicationError("ATTACKER_UNAVAILABLE", "The selected attacker checkpoint is unavailable.", 409)
        if not any(
            item["clutter"] == request.clutter_level and request.perception_mode in item["perception_modes"]
            for item in defender["supported_scenarios"]
        ):
            raise ApplicationError("INCOMPATIBLE_SCENARIO", "The selected defender does not support this scenario.", 409)

    def start(self, request: ReplayBatchRequest) -> ReplayJob:
        self._validate_available_selection(request)
        with self._lock:
            if any(job.state in {"queued", "running"} for job in self._jobs.values()):
                raise ApplicationError("REPLAY_JOB_BUSY", "Another replay batch is queued or running; retry after it finishes.", 409)
            now = _now()
            job = ReplayJob(
                job_id=uuid4().hex,
                request=request,
                state="queued",
                requested_episode_count=request.episode_count,
                completed_count=0,
                current_episode_seed=None,
                completed_replay_ids=[],
                error=None,
                cancellation_requested=False,
                created_at=now,
                updated_at=now,
            )
            self._jobs[job.job_id] = job
            self._cancel[job.job_id] = threading.Event()
            self._persist(job)
            self._future = self._executor.submit(self._run, job.job_id)
            return job.model_copy(deep=True)

    def get(self, job_id: str) -> ReplayJob:
        self._validate_job_id(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ApplicationError("REPLAY_JOB_NOT_FOUND", "Replay job was not found.", 404)
            return job.model_copy(deep=True)

    def cancel(self, job_id: str) -> ReplayJob:
        self._validate_job_id(job_id)
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise ApplicationError("REPLAY_JOB_NOT_FOUND", "Replay job was not found.", 404)
            if job.state in {"completed", "failed", "cancelled", "interrupted"}:
                return job.model_copy(deep=True)
            cancel_event = self._cancel[job_id]
            cancel_event.set()
            process = self._processes.get(job_id)
            if process is None and job.state == "queued":
                job.state = "cancelled"
                job.current_episode_seed = None
                job.completed_at = _now()
                self._persist(job)
            else:
                job.cancellation_requested = True
                self._persist(job)
                if process is not None and process.poll() is None:
                    process.terminate()
            return job.model_copy(deep=True)

    def shutdown(self) -> None:
        with self._lock:
            active = next((job_id for job_id, job in self._jobs.items() if job.state in {"queued", "running"}), None)
        if active is not None:
            self.cancel(active)
        self._executor.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _validate_job_id(job_id: str) -> None:
        if not JOB_ID_RE.fullmatch(job_id):
            raise ApplicationError("INVALID_REPLAY_JOB_ID", "Replay job ID must be a 32-character opaque ID.", 400)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            cancel = self._cancel[job_id]
            if cancel.is_set():
                job.state = "cancelled"
                job.completed_at = _now()
                self._persist(job)
                return
            job.state = "running"
            job.started_at = _now()
            self._persist(job)
        try:
            for offset in range(job.request.episode_count):
                if cancel.is_set():
                    self._finish_cancelled(job_id)
                    return
                seed = job.request.seed_start + offset
                replay_id = uuid4().hex
                with self._lock:
                    job = self._jobs[job_id]
                    job.current_episode_seed = seed
                    self._persist(job)
                public_replay_id = self._execute_episode(job_id, seed, replay_id)
                with self._lock:
                    job = self._jobs[job_id]
                    job.completed_count += 1
                    job.completed_replay_ids.append(public_replay_id)
                    job.current_episode_seed = None
                    self._persist(job)
            with self._lock:
                job = self._jobs[job_id]
                job.state = "completed"
                job.completed_at = _now()
                self._persist(job)
        except _JobCancelled:
            self._finish_cancelled(job_id)
        except _EpisodeTimeout as exc:
            self._finish_failed(job_id, str(exc))
        except ApplicationError as exc:
            self._finish_failed(job_id, exc.message)
        except Exception:
            logger.exception("Unexpected replay worker failure for job %s", job_id)
            self._finish_failed(job_id, "Replay worker failed unexpectedly. Check backend logs for details.")
        finally:
            with self._lock:
                self._processes.pop(job_id, None)

    def _execute_episode(self, job_id: str, seed: int, replay_id: str) -> str:
        job = self._jobs[job_id]
        request = job.request
        command = [
            str(self.settings.simulation_python),
            "-m",
            "visualizer.backend.replay_cli",
            "--defender",
            request.defender_policy_id,
            "--attacker",
            request.attacker_condition,
            "--clutter",
            request.clutter_level,
            "--perception",
            request.perception_mode,
            "--semantics",
            request.semantics,
            "--evaluation-seed",
            str(seed),
            "--output-name",
            replay_id,
        ]
        if request.defender_generation is not None:
            command.extend(["--generation", request.defender_generation])
        if request.defender_training_seed is not None:
            command.extend(["--training-seed", str(request.defender_training_seed)])
        environment = os.environ.copy()
        environment["VISUALIZER_RESEARCH_DATA_ROOT"] = str(self.settings.research_data_root)
        environment["PYTHONPATH"] = str(VISUALIZER_ROOT.parent) + os.pathsep + environment.get("PYTHONPATH", "")
        try:
            process = subprocess.Popen(
                command,
                cwd=VISUALIZER_ROOT.parent,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
        except OSError as exc:
            raise ApplicationError("SIMULATION_RUNTIME_UNAVAILABLE", "Cannot start the configured simulation Python runtime.", 503) from exc
        with self._lock:
            self._processes[job_id] = process
            if self._cancel[job_id].is_set():
                process.terminate()
        deadline = time.monotonic() + self.settings.episode_timeout_seconds
        try:
            while True:
                if self._cancel[job_id].is_set():
                    self._terminate(process)
                    raise _JobCancelled
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._terminate(process)
                    raise _EpisodeTimeout(
                        f"Episode seed {seed} exceeded the {self.settings.episode_timeout_seconds}-second timeout; "
                        "it was terminated and is not a simulation outcome."
                    )
                try:
                    stdout, stderr = process.communicate(timeout=min(POLL_SECONDS, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
        finally:
            with self._lock:
                self._processes.pop(job_id, None)
        if process.returncode != 0:
            error_payload = self._parse_worker_output(stdout if "stdout" in locals() else "")
            if error_payload and isinstance(error_payload.get("error"), dict):
                error = error_payload["error"]
                code = str(error.get("code", "EPISODE_FAILED"))
                message = SAFE_WORKER_ERRORS.get(code, f"Simulation episode seed {seed} failed.")
                raise ApplicationError(code, message, 500)
            logger.error("Replay subprocess failed for seed %s (exit %s): %s", seed, process.returncode, stderr[-4000:])
            raise ApplicationError("EPISODE_FAILED", f"Simulation episode seed {seed} failed; see backend logs.", 500)
        payload = self._parse_worker_output(stdout)
        if not payload or payload.get("evaluation_seed") != seed:
            raise ApplicationError("INVALID_WORKER_RESPONSE", "Simulation worker did not return the expected structured result.", 500)
        expected_path = local_output_root().resolve() / f"{replay_id}.json.gz"
        reported_path = Path(str(payload.get("output_path", ""))).resolve()
        if reported_path != expected_path or not _safe_replay_path(expected_path):
            raise ApplicationError("INVALID_WORKER_RESPONSE", "Simulation worker saved the replay outside its assigned output.", 500)
        bundle = _load_bundle(expected_path)
        if len(bundle.episodes) != 1 or bundle.episodes[0].simulation_seed != seed:
            raise ApplicationError("INVALID_WORKER_RESPONSE", "The generated replay does not match the requested seed.", 500)
        episode = bundle.episodes[0]
        if (
            episode.defender.policy_id != request.defender_policy_id
            or episode.defender.generation != request.defender_generation
            or episode.defender.training_seed != request.defender_training_seed
            or episode.attacker.policy_id != request.attacker_condition
            or episode.environment.clutter != request.clutter_level
            or episode.environment.awareness != request.perception_mode
            or episode.environment.semantics != request.semantics
        ):
            raise ApplicationError("INVALID_WORKER_RESPONSE", "The generated replay metadata differs from the validated request.", 500)
        return _replay_id_for_path(expected_path)

    @staticmethod
    def _parse_worker_output(output: str) -> dict | None:
        try:
            payload = json.loads(output)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _terminate(process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=CANCEL_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()

    def _finish_cancelled(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.state = "cancelled"
            job.current_episode_seed = None
            job.cancellation_requested = False
            job.error = "Batch cancelled; only completed and validated replays were retained."
            job.completed_at = _now()
            self._persist(job)

    def _finish_failed(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            if self._cancel[job_id].is_set():
                job.state = "cancelled"
                job.error = "Batch cancelled; only completed and validated replays were retained."
            else:
                job.state = "failed"
                job.error = message
            job.current_episode_seed = None
            job.cancellation_requested = False
            job.completed_at = _now()
            self._persist(job)

    def list_replays(self, limit: int = 100) -> ReplayCatalog:
        root = local_output_root().resolve()
        if not root.exists():
            return ReplayCatalog(replays=[])
        files = sorted(root.glob("*.json.gz"), key=lambda item: item.stat().st_mtime_ns, reverse=True)
        results = []
        invalid_count = 0
        for path in files:
            if not _safe_replay_path(path):
                logger.error("Rejected replay path outside local output directory: %s", path.name)
                continue
            try:
                bundle = _load_bundle(path)
            except ApplicationError:
                invalid_count += 1
                logger.warning(
                    "Excluded replay artifact that failed schema or semantic validation (opaque replay id %s)",
                    _replay_id_for_path(path),
                )
                continue
            if len(bundle.episodes) != 1:
                invalid_count += 1
                logger.warning(
                    "Excluded replay artifact with an unsupported episode count (opaque replay id %s)",
                    _replay_id_for_path(path),
                )
                continue
            episode = bundle.episodes[0]
            summary = episode.summary
            results.append(ReplayListItem(
                replay_id=_replay_id_for_path(path),
                simulation_seed=episode.simulation_seed,
                defender=episode.defender.model_dump(mode="json"),
                attacker=episode.attacker.model_dump(mode="json"),
                clutter=episode.environment.clutter,
                perception=episode.environment.awareness,
                semantics=episode.environment.semantics,
                outcome=summary.outcome,
                steps=summary.steps,
                duration_s=summary.duration_s,
                detection_time_s=summary.detection_time_s,
                total_reward=summary.total_reward,
                collision_type=summary.collision_type,
                interception_time_s=summary.interception_time_s,
                provenance=bundle.provenance.model_dump(mode="json"),
            ))
            if len(results) >= limit:
                break
        warnings = (
            [f"{invalid_count} local replay artifact(s) failed current schema or trace validation and were excluded."]
            if invalid_count else []
        )
        return ReplayCatalog(replays=results, warnings=warnings)

    def get_replay(self, replay_id: str) -> ReplayBundle:
        if not REPLAY_ID_RE.fullmatch(replay_id):
            raise ApplicationError("INVALID_REPLAY_ID", "Replay ID must be a 32-character opaque ID.", 400)
        root = local_output_root().resolve()
        for path in root.glob("*.json.gz") if root.exists() else ():
            if _replay_id_for_path(path) == replay_id:
                if not _safe_replay_path(path):
                    raise ApplicationError("INVALID_REPLAY_PATH", "Replay artifact is outside the configured output root.", 403)
                return _load_bundle(path)
        raise ApplicationError("REPLAY_NOT_FOUND", "Replay was not found in the local output catalog.", 404)


class _JobCancelled(Exception):
    pass


class _EpisodeTimeout(Exception):
    pass
