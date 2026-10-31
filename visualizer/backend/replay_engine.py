"""Deterministic, single-episode recorder over the existing SoldierEnv."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import logging
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import FiniteFloat

from .config import Settings, local_output_root
from .data_access import ResearchDataStore
from .application_error import ApplicationError
from .replay_requests import ATTACKER_SEEDS, CLUTTER_ARCHITECTURES, CONFERENCE_ARCHITECTURES, V2_ATTACKER_SEEDS, ReplayRequest
from .replay_contract import (
    Checkpoint,
    EntityState,
    EnvironmentSpec,
    EpisodeSummary,
    LeadGuidance,
    LRPPOGuidance,
    MovingObstacle,
    PNGuidance,
    PolicySelection,
    PolicyTelemetry,
    PolicyVisible,
    PrivilegedDiagnostics,
    RMPCGuidance,
    ReplayBundle,
    ReplayEpisode,
    ReplayEvent,
    ReplayFrame,
    ReplayProvenance,
    StaticObstacle,
)
from .replay_validation import validate_trace as _validate_trace

logger = logging.getLogger(__name__)
EXPECTED_RESEARCH_COMMIT = "0ecf78e2d7e0727cdbdc1c6ca8e9e17b70699db5"  # research HEAD: V2 A1 campaign metadata on top of journal-v2-a1-training-protocol-v1 (A0 canonical: bdbee45)
REGISTRY_RELATIVE_PATH = "results/defender_adaptation_model_registry.json"
# Canonical-V2 A1 campaign registry written by the research repository after training (never inferred).
V2_A1_RESULTS_RELATIVE = "results/journal_v2_corrected_semantics/a1_training/a1_v2_training_v1"
V2_A1_REGISTRY_RELATIVE_PATH = f"{V2_A1_RESULTS_RELATIVE}/a1_v2_model_registry.json"
V2_A1_OBSERVATION_DIM = 37
V2_A1_PROTOCOL = "a1_v2_training_v1"
V2_A1_REWARD_VERSION = "a1_v2_reward_v1"
SECTOR_FEATURE_DIM = 20
LRPPO_MAX_ANGLE_DEG = 30.0
HEADING_REVERSAL_THRESHOLD_DEG = 120.0
_SAFE_OUTPUT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")
_COLLISION_OUTCOMES = frozenset(
    (
        "defender_static_obstacle_collision",
        "defender_dynamic_obstacle_collision",
        "enemy_static_obstacle_collision",
        "enemy_dynamic_obstacle_collision",
    )
)


@dataclasses.dataclass(frozen=True)
class _ResolvedCheckpoint:
    artifact_id: str
    relative_path: str
    sha256: str
    role: Literal["d0_start", "final", "secondary_best", "conference_best", "attacker_best", "attacker_v2_final"]
    training_seed: int


@dataclasses.dataclass
class _PolicyAdapter:
    policy_id: str
    architecture: str | None
    generation: str | None
    seed: int | None
    estimator: str
    policy: object
    checkpoint: _ResolvedCheckpoint | None
    model_input_shape: tuple[int, ...]
    needs_sector_features: bool
    is_lr_ppo: bool


class _PassivePolicy:
    """The research harness's passive defender: no policy object, a zero action every step.

    Mirrors `experiments.journal_v2_validation_core.record_episode` (`make_policy("passive")`
    returns None and the loop supplies `np.zeros(3)`); it contains no control equations.
    """

    def reset(self) -> None:
        return None

    def act(self, *_args: object, **_kwargs: object) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)


def _fail(code: str, message: str, status_code: int = 400) -> ApplicationError:
    return ApplicationError(code, message, status_code)


def _research_modules(root: Path) -> dict[str, object]:
    if not root.is_absolute() or not root.is_dir():
        raise _fail("INVALID_DATA_ROOT", "Research data root must resolve to an absolute directory.", 503)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module_names = (
        "numpy",
        "gymnasium",
        "stable_baselines3",
        "torch",
        "uav_defend.envs.soldier_env",
        "uav_defend.config.env_config",
        "uav_defend.policies.registry",
        "uav_defend.policies.sanitize",
        "uav_defend.policies.rl.ppo_policy_wrapper",
        "uav_defend.policies.rl_kalman.ppo_kalman_policy_wrapper",
        "uav_defend.policies.residual.lead_residual_composition",
        "uav_defend.policies.residual.lead_residual_ppo_policy_wrapper",
        "uav_defend.policies.residual.lead_residual_observation",
        "uav_defend.obstacles.sector_features",
        "uav_defend.policies.mpc.rmpc_policy",
        "experiments.journal_clutter_protocol",
        "experiments.journal_v2_protocol",
        "experiments.journal_v2_validation_core",
        "experiments.adaptive_attacker_env",
        "experiments.journal_v2_a1",
        "experiments.post_detection_diagnostic_protocol",
        "experiments.lead_residual_diagnostic_protocol",
        "experiments.clutter_benchmark_protocol",
    )
    try:
        return {name: importlib.import_module(name) for name in module_names}
    except ImportError as exc:
        missing = exc.name or "unknown dependency"
        raise _fail(
            "SIMULATION_DEPENDENCY_MISSING",
            f"The selected Python interpreter cannot import '{missing}'. "
            "Run the CLI with the research Python environment and install the existing research requirements.",
            503,
        ) from exc


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _git_dirty(root: Path) -> bool:
    return bool(_git_value(root, "status", "--porcelain=v1"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_registry(store: ResearchDataStore) -> tuple[dict, str, bytes]:
    raw = store.read_bytes(REGISTRY_RELATIVE_PATH)
    try:
        registry = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail("INVALID_MODEL_REGISTRY", "The frozen defender model registry is not valid JSON.", 409) from exc
    if not isinstance(registry, dict) or not isinstance(registry.get("defenders"), dict) or not isinstance(
        registry.get("attackers"), dict
    ):
        raise _fail("INVALID_MODEL_REGISTRY", "The frozen defender model registry has an unexpected structure.", 409)
    return registry, hashlib.sha256(raw).hexdigest(), raw


def _check_registry_path(entry: dict, root: Path, relative_path: str, identity: str) -> str:
    expected = (root / Path(relative_path)).resolve()
    raw_path = entry.get("path")
    if not isinstance(raw_path, str):
        raise _fail("INVALID_MODEL_REGISTRY", f"{identity} has no frozen checkpoint path.", 409)
    registered = Path(raw_path).resolve()
    if registered != expected:
        raise _fail(
            "INVALID_MODEL_REGISTRY",
            f"{identity} registry path does not match its allowlisted artifact location.",
            409,
        )
    return relative_path


def _verify_artifact(store: ResearchDataStore, item: _ResolvedCheckpoint) -> Path:
    path = store.resolve_file(item.relative_path)
    actual = _sha256(path)
    if actual != item.sha256:
        raise _fail(
            "CHECKPOINT_HASH_MISMATCH",
            f"{item.artifact_id} SHA-256 mismatch: expected {item.sha256}, found {actual}.",
            409,
        )
    return path


def _resolve_defender_checkpoint(
    request: ReplayRequest,
    store: ResearchDataStore,
    registry: dict,
    modules: dict[str, object],
) -> tuple[_ResolvedCheckpoint, str, tuple[int, ...], bool, bool]:
    policy_id = request.defender_policy_id
    seed = request.defender_training_seed
    generation = request.defender_generation
    if seed is None or generation is None:
        raise _fail("INVALID_REQUEST", "Learned defender generation and training seed are required.")

    if generation == "conference":
        family, seeds, estimator, input_shape = CONFERENCE_ARCHITECTURES[policy_id]
        if seed not in seeds:
            raise _fail("INCOMPATIBLE_POLICY", f"{policy_id} conference seeds are {seeds}, not {seed}.")
        protocol_name = (
            "experiments.lead_residual_diagnostic_protocol"
            if policy_id == "lr_ppo"
            else "experiments.post_detection_diagnostic_protocol"
        )
        frozen = modules[protocol_name].FROZEN_MODELS.get((family, seed))
        if not isinstance(frozen, dict):
            raise _fail("CHECKPOINT_NOT_REGISTERED", f"No frozen {policy_id} conference checkpoint exists for seed {seed}.", 404)
        relative = {
            "ppo": f"results/training_post_detection_400k/direct/seed_{seed}/best_model.zip",
            "ppo_kalman": f"results/training_post_detection_400k/kalman/seed_{seed}/best_model.zip",
            "lr_ppo": f"models/lead_residual_post_detection/seed_{seed}/best_model.zip",
        }[policy_id]
        frozen_path = frozen.get("path")
        if not isinstance(frozen_path, str) or Path(frozen_path).resolve() != (
            store.root / Path(relative)
        ).resolve():
            raise _fail(
                "INVALID_MODEL_REGISTRY",
                f"{policy_id} conference checkpoint path does not match its allowlisted location.",
                409,
            )
        return (
            _ResolvedCheckpoint(
                artifact_id=f"conference:{policy_id}:{seed}:best",
                relative_path=relative,
                sha256=frozen["sha256"],
                role="conference_best",
                training_seed=seed,
            ),
            family,
            (input_shape,),
            False,
            policy_id == "lr_ppo",
        )

    architecture, seeds, estimator, sectors, is_lr = CLUTTER_ARCHITECTURES[policy_id]
    if seed not in seeds:
        raise _fail("INCOMPATIBLE_POLICY", f"{policy_id} {generation} seeds are {seeds}, not {seed}.")
    key = f"{generation}|{architecture}|{seed}"
    entry = registry["defenders"].get(key)
    if not isinstance(entry, dict):
        raise _fail("CHECKPOINT_NOT_REGISTERED", f"No frozen checkpoint is registered for {key}.", 404)
    if entry.get("architecture") != architecture or entry.get("defender_seed") != seed or entry.get("generation") != generation:
        raise _fail("INVALID_MODEL_REGISTRY", f"Frozen registry identity mismatch for {key}.", 409)
    role = "d0_start" if generation == "d0" else "final"
    if entry.get("checkpoint_role") != role:
        raise _fail("INVALID_MODEL_REGISTRY", f"{key} has unexpected checkpoint role '{entry.get('checkpoint_role')}'.", 409)
    folder = {
        "d0": f"results/clutter_training_400k/{architecture}/seed_{seed}/best_model.zip",
        "d0_plus": f"results/defender_adaptation/d0_plus_scripted/{architecture}/seed_{seed}/final_model.zip",
        "d1": f"results/defender_adaptation/d1_adaptive_mixture/{architecture}/seed_{seed}/final_model.zip",
    }[generation]
    _check_registry_path(entry, store.root, folder, key)
    input_width = 16 + (SECTOR_FEATURE_DIM if sectors else 0) + (3 if is_lr else 0)
    return (
        _ResolvedCheckpoint(
            artifact_id=key,
            relative_path=folder,
            sha256=entry["sha256"],
            role=role,
            training_seed=seed,
        ),
        architecture,
        (input_width,),
        sectors,
        is_lr,
    )


def _resolve_v2_attacker_checkpoint(condition: str, store: ResearchDataStore) -> _ResolvedCheckpoint:
    """Canonical-V2 A1 model: the research campaign registry, the model's own sidecar manifest and the
    file itself must all agree on the SHA-256. Legacy V1 checkpoints are never mapped to a V2 identifier."""
    seed = V2_ATTACKER_SEEDS[condition]
    try:
        registry = json.loads(store.read_bytes(V2_A1_REGISTRY_RELATIVE_PATH))
    except ApplicationError as exc:
        raise _fail("CHECKPOINT_NOT_REGISTERED", f"The V2 A1 model registry is unavailable: {exc.message}", 404) from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail("INVALID_MODEL_REGISTRY", "The V2 A1 model registry is not valid JSON.", 409) from exc
    if registry.get("protocol") != V2_A1_PROTOCOL or registry.get("reward_version") != V2_A1_REWARD_VERSION:
        raise _fail("INVALID_MODEL_REGISTRY", "The V2 A1 registry protocol/reward version is not the frozen one.", 409)
    entry = (registry.get("attackers") or {}).get(condition)
    if not isinstance(entry, dict):
        raise _fail("CHECKPOINT_NOT_REGISTERED", f"No V2 A1 checkpoint is registered for {condition}.", 404)
    relative = f"{V2_A1_RESULTS_RELATIVE}/seed_{seed}/checkpoints/final_model.zip"
    if entry.get("path") != relative or entry.get("ppo_seed") != seed:
        raise _fail("INVALID_MODEL_REGISTRY", f"{condition} registry entry does not point at its allowlisted final model.", 409)
    if entry.get("observation_dim") != V2_A1_OBSERVATION_DIM or entry.get("action_dim") != 3:
        raise _fail("INCOMPATIBLE_CHECKPOINT", f"{condition} is not registered as a 37-D observation / 3-D action model.", 409)
    try:
        manifest = json.loads(store.read_bytes(relative.replace(".zip", ".manifest.json")))
    except (ApplicationError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _fail("CHECKPOINT_NOT_REGISTERED", f"{condition} has no readable sidecar manifest.", 404) from exc
    if (
        manifest.get("model_sha256") != entry.get("sha256")
        or manifest.get("ppo_seed") != seed
        or manifest.get("training_protocol") != V2_A1_PROTOCOL
        or manifest.get("reward_version") != V2_A1_REWARD_VERSION
        or manifest.get("observation_dim") != V2_A1_OBSERVATION_DIM
    ):
        raise _fail("CHECKPOINT_HASH_MISMATCH", f"{condition} registry entry and sidecar manifest disagree.", 409)
    return _ResolvedCheckpoint(
        artifact_id=f"a1_v2:{seed}:final",
        relative_path=relative,
        sha256=entry["sha256"],
        role="attacker_v2_final",
        training_seed=seed,
    )


def _resolve_attacker_checkpoint(
    condition: str, registry: dict, store: ResearchDataStore
) -> _ResolvedCheckpoint | None:
    if condition in V2_ATTACKER_SEEDS:
        return _resolve_v2_attacker_checkpoint(condition, store)
    seed = ATTACKER_SEEDS.get(condition)
    if seed is None:
        return None
    entry = registry["attackers"].get(str(seed))
    if not isinstance(entry, dict):
        raise _fail("CHECKPOINT_NOT_REGISTERED", f"No frozen checkpoint is registered for attacker seed {seed}.", 404)
    relative = f"results/adaptive_attacker_400k/seed_{seed}/best_model.zip"
    _check_registry_path(entry, store.root, relative, condition)
    return _ResolvedCheckpoint(
        artifact_id=f"adaptive_attacker_v1:{seed}:best",
        relative_path=relative,
        sha256=entry["sha256"],
        role="attacker_best",
        training_seed=seed,
    )


def _checkpoint_model_input(model: object, expected_shape: tuple[int, ...], artifact_id: str) -> None:
    observation_space = getattr(model, "observation_space", None)
    action_space = getattr(model, "action_space", None)
    if getattr(observation_space, "shape", None) != expected_shape:
        actual = getattr(observation_space, "shape", None)
        raise _fail(
            "INCOMPATIBLE_CHECKPOINT",
            f"{artifact_id} observation space {actual} does not match required {expected_shape}.",
            409,
        )
    if getattr(action_space, "shape", None) != (3,):
        actual = getattr(action_space, "shape", None)
        raise _fail("INCOMPATIBLE_CHECKPOINT", f"{artifact_id} action space {actual} does not match (3,).", 409)


def _load_defender(
    request: ReplayRequest,
    config: object,
    store: ResearchDataStore,
    registry: dict,
    modules: dict[str, object],
) -> _PolicyAdapter:
    policy_id = request.defender_policy_id
    if policy_id in CLUTTER_ARCHITECTURES:
        checkpoint, architecture, input_shape, sectors, is_lr = _resolve_defender_checkpoint(
            request, store, registry, modules
        )
        path = _verify_artifact(store, checkpoint)
        if request.defender_generation == "conference":
            sectors = False
        deterministic = request.deterministic_inference
        if policy_id == "lr_ppo":
            policy = modules["uav_defend.policies.residual.lead_residual_ppo_policy_wrapper"].LeadResidualPPOPolicyWrapper.load(
                path,
                residual_max_angle_deg=LRPPO_MAX_ANGLE_DEG,
                config=config,
                deterministic=deterministic,
                device="cpu",
            )
        elif policy_id == "ppo":
            policy = modules["uav_defend.policies.rl.ppo_policy_wrapper"].PPOPolicyWrapper.load(
                path, deterministic=deterministic, device="cpu"
            )
        else:
            policy = modules[
                "uav_defend.policies.rl_kalman.ppo_kalman_policy_wrapper"
            ].PPOKalmanPolicyWrapper.load(path, deterministic=deterministic, device="cpu")
        model = policy.model
        _checkpoint_model_input(model, input_shape, checkpoint.artifact_id)
        if getattr(policy, "deterministic", None) is not True:
            raise _fail("INCOMPATIBLE_CHECKPOINT", "Learned defender inference must be deterministic.", 409)
        if str(getattr(model, "device", "")) != "cpu":
            raise _fail("INCOMPATIBLE_CHECKPOINT", "Replay requires CPU model inference for recorded provenance.", 409)
        if _sha256(path) != checkpoint.sha256:
            raise _fail("CHECKPOINT_MUTATED", f"{checkpoint.artifact_id} changed during model loading.", 409)
        return _PolicyAdapter(
            policy_id=policy_id,
            architecture=architecture,
            generation=request.defender_generation,
            seed=request.defender_training_seed,
            estimator=CLUTTER_ARCHITECTURES[policy_id][2] if request.defender_generation != "conference"
            else CONFERENCE_ARCHITECTURES[policy_id][2],
            policy=policy,
            checkpoint=checkpoint,
            model_input_shape=input_shape,
            needs_sector_features=sectors,
            is_lr_ppo=is_lr,
        )

    estimator = "kalman" if policy_id == "kalman_greedy" else "measurement"
    if policy_id == "passive":
        return _PolicyAdapter(
            policy_id, "passive", None, None, estimator, _PassivePolicy(), None, (16,), False, False
        )
    if policy_id == "rmpc":
        frozen_config = modules["experiments.clutter_benchmark_protocol"]._frozen_rmpc_config()
        policy = modules["uav_defend.policies.mpc.rmpc_policy"].RMPCPolicy(
            config=config,
            rmpc_config=frozen_config,
        )
        return _PolicyAdapter(
            policy_id, "rmpc", None, None, estimator, policy, None, (16,), False, False
        )
    registry_factory = modules["uav_defend.policies.registry"].get_policy
    policy = (
        registry_factory(policy_id)
        if policy_id in {"greedy", "kalman_greedy"}
        else registry_factory(policy_id, config=config)
    )
    return _PolicyAdapter(
        policy_id, policy_id, None, None, estimator, policy, None, (16,), False, False
    )


def _load_attacker(
    condition: str,
    checkpoint: _ResolvedCheckpoint | None,
    store: ResearchDataStore,
    modules: dict[str, object],
):
    if checkpoint is None:
        return None
    path = _verify_artifact(store, checkpoint)
    model = modules["stable_baselines3"].PPO.load(str(path), device="cpu")
    _checkpoint_model_input(
        model, (V2_A1_OBSERVATION_DIM,) if condition in V2_ATTACKER_SEEDS else (36,), checkpoint.artifact_id
    )
    if _sha256(path) != checkpoint.sha256:
        raise _fail("CHECKPOINT_MUTATED", f"{condition} checkpoint changed during model loading.", 409)
    return model


def _build_attacker_observation(condition: str, modules: dict[str, object], env: object) -> np.ndarray:
    """Delegates to the research repository's own builders; the visualizer defines no observation features."""
    detection_range = env.config.enemy_defender_detection_range
    if condition in V2_ATTACKER_SEEDS:
        observation = modules["experiments.journal_v2_a1"].build_a1_v2_observation(env, detection_range)
        expected = V2_A1_OBSERVATION_DIM
    else:
        observation = modules["experiments.adaptive_attacker_env"].build_attacker_observation(env, detection_range)
        expected = 36
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (expected,):
        raise _fail("INCOMPATIBLE_CHECKPOINT", f"{condition} observation is {observation.shape}, expected ({expected},).", 500)
    return observation


def _plain(value: object, unavailable: dict[str, str], path: str = "$") -> object:
    if isinstance(value, np.ndarray):
        return [_plain(item, unavailable, f"{path}[{index}]") for index, item in enumerate(value.tolist())]
    if isinstance(value, np.generic):
        return _plain(value.item(), unavailable, path)
    if dataclasses.is_dataclass(value):
        return _plain(dataclasses.asdict(value), unavailable, path)
    if isinstance(value, dict):
        return {str(key): _plain(item, unavailable, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item, unavailable, f"{path}[{index}]") for index, item in enumerate(value)]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        unavailable[path] = "Simulator reported a non-finite diagnostic; no finite replacement was applied."
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if hasattr(value, "item"):
        return _plain(value.item(), unavailable, path)
    raise TypeError(f"Unsupported value at {path}: {type(value).__name__}")


def _vector(value: object | None) -> tuple[float, float, float] | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise _fail("INVALID_SIMULATION_STATE", f"Expected native 3D vector, got shape {arr.shape}.", 500)
    if not np.all(np.isfinite(arr)):
        return None
    return tuple(float(v) for v in arr)


def _heading(vector: object | None) -> float | None:
    native = _vector(vector)
    if native is None or math.hypot(native[0], native[1]) <= 1e-8:
        return None
    return float(math.degrees(math.atan2(native[1], native[0])))


def _direction(vector: object | None) -> tuple[float, float, float] | None:
    native = _vector(vector)
    if native is None:
        return None
    array = np.asarray(native, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-8:
        return None
    return tuple(float(v) for v in array / norm)


def _angle_deg(first: object | None, second: object | None) -> float | None:
    a, b = _vector(first), _vector(second)
    if a is None or b is None:
        return None
    aa, bb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = float(np.linalg.norm(aa)), float(np.linalg.norm(bb))
    if na <= 1e-8 or nb <= 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(aa, bb) / (na * nb), -1.0, 1.0))))


def _signed_horizontal_angle(first: object, second: object) -> float | None:
    a, b = _vector(first), _vector(second)
    if a is None or b is None:
        return None
    norm_a, norm_b = math.hypot(a[0], a[1]), math.hypot(b[0], b[1])
    if norm_a <= 1e-8 or norm_b <= 1e-8:
        return None
    cross_z = a[0] * b[1] - a[1] * b[0]
    dot = a[0] * b[0] + a[1] * b[1]
    return float(math.degrees(math.atan2(cross_z, dot)))


def _entity(position: object, velocity: object | None) -> EntityState:
    native_position = _vector(position)
    if native_position is None:
        raise _fail("INVALID_SIMULATION_STATE", "Simulator returned an unavailable entity position.", 500)
    native_velocity = _vector(velocity)
    if velocity is not None and native_velocity is None:
        raise _fail("INVALID_SIMULATION_STATE", "Simulator returned a non-finite entity velocity.", 500)
    speed = float(np.linalg.norm(native_velocity)) if native_velocity is not None else None
    return EntityState(
        position_m=native_position,
        velocity_mps=native_velocity,
        speed_mps=speed,
        horizontal_heading_deg=_heading(native_velocity),
        movement_direction=_direction(native_velocity),
    )


def _visible(
    observation: np.ndarray,
    info: dict,
    estimator: str,
    sectors: np.ndarray | None,
    unavailable: dict[str, str],
    sanitized: dict | None = None,
) -> PolicyVisible:
    if sanitized is None:
        sanitized = importlib.import_module("uav_defend.policies.sanitize").build_policy_info(info, estimator)
    absent = {}
    def optional_vector(key: str, field_name: str) -> tuple[float, float, float] | None:
        value = sanitized.get(key)
        vector = _vector(value)
        if value is not None and vector is None:
            absent[field_name] = "The policy sanitizer supplied a non-finite vector."
        return vector

    if sanitized.get("enemy_measurement") is None and estimator == "measurement":
        absent["enemy_measurement_m"] = "No valid direct measurement is available at this state."
    if sanitized.get("enemy_measurement_velocity") is None and estimator == "measurement":
        absent["enemy_measurement_velocity_mps"] = "No valid measured velocity is available at this state."
    if estimator == "kalman" and sanitized.get("e_hat") is None:
        absent["e_hat_m"] = "The Kalman tracker has no initialized estimate at this state."
    if estimator == "kalman" and sanitized.get("v_hat") is None:
        absent["v_hat_mps"] = "The Kalman tracker has no initialized velocity estimate at this state."
    if sectors is None:
        absent["sector_features"] = "This controller/checkpoint does not consume sector features."
    return PolicyVisible(
        estimator=estimator,
        observation=[float(v) for v in np.asarray(observation, dtype=np.float32).reshape(-1)],
        soldier_pos_m=_required_vector(sanitized["soldier_pos"], "policy-visible soldier position"),
        defender_pos_m=_required_vector(sanitized["defender_pos"], "policy-visible defender position"),
        defender_vel_mps=_required_vector(sanitized["defender_vel"], "policy-visible defender velocity"),
        enemy_detected=bool(sanitized["enemy_detected"]),
        enemy_measurement_m=optional_vector("enemy_measurement", "enemy_measurement_m"),
        enemy_measurement_velocity_mps=optional_vector(
            "enemy_measurement_velocity", "enemy_measurement_velocity_mps"
        ),
        enemy_measurement_velocity_valid=(
            bool(sanitized["enemy_measurement_velocity_valid"])
            if "enemy_measurement_velocity_valid" in sanitized else None
        ),
        e_hat_m=optional_vector("e_hat", "e_hat_m"),
        v_hat_mps=optional_vector("v_hat", "v_hat_mps"),
        sector_features=(
            [float(v) for v in np.asarray(sectors, dtype=np.float32).reshape(-1)]
            if sectors is not None else None
        ),
        unavailable_fields=absent,
    )


def _required_vector(value: object, label: str) -> tuple[float, float, float]:
    vector = _vector(value)
    if vector is None:
        raise _fail("INVALID_SIMULATION_STATE", f"Missing or non-finite {label}.", 500)
    return vector


def _static_obstacles(env: object) -> list[StaticObstacle]:
    return [
        StaticObstacle(
            obstacle_id=f"static-{index}",
            min_corner_m=_vector(obstacle.min_corner),
            max_corner_m=_vector(obstacle.max_corner),
            map_known=bool(obstacle.map_known),
            visual_kind="block",
        )
        for index, obstacle in enumerate(env._obstacle_layout.obstacles)
    ]


def _moving_obstacles(env: object) -> list[MovingObstacle]:
    return [
        MovingObstacle(
            obstacle_id=f"dynamic-{item.id}",
            kind=item.obstacle_type,
            position_m=_vector(item.position),
            velocity_mps=_vector(item.velocity),
            radius_m=float(item.radius),
        )
        for item in env._dynamic_obstacles
    ]


def _sector_features(env: object, adapter: _PolicyAdapter, modules: dict[str, object]) -> np.ndarray | None:
    if not adapter.needs_sector_features:
        return None
    sector_module = modules["uav_defend.obstacles.sector_features"]
    return sector_module.compute_sector_features(
        defender_pos=env._defender_pos,
        defender_vel=env._defender_vel,
        perceived=env._defender_last_perceived,
        sensor_range=sector_module.SECTOR_OBSERVATION_V1.sensor_range,
    )


def _environment_state(
    info: dict,
    unavailable: dict[str, str],
    prefix: str,
) -> dict:
    keys = (
        "step_count", "enemy_detected", "just_detected", "detected",
        "enemy_soldier_dist", "defender_enemy_dist", "unsafe_intercept",
        "defender_standby", "controller_action_executed",
        "defender_accel", "defender_turn_rate", "defender_accel_saturated",
        "defender_turn_saturated", "defender_climb_saturated",
        "enemy_accel", "enemy_turn_rate", "enemy_accel_saturated",
        "enemy_turn_saturated", "enemy_climb_saturated",
        "defender_obstacle_collision", "enemy_obstacle_collision",
        "defender_dynamic_obstacle_collision", "enemy_dynamic_obstacle_collision",
        "defender_collision_event_type", "defender_collision_event_point",
        "defender_collision_event_speed", "defender_collision_event_step",
        "enemy_collision_event_type", "enemy_collision_event_point",
        "enemy_collision_event_speed", "enemy_collision_event_step",
        "defender_collision_obstacle_id", "enemy_collision_obstacle_id",
        "defender_dynamic_collision_count", "enemy_dynamic_collision_count",
        "defender_min_obstacle_clearance", "enemy_min_obstacle_clearance",
        "defender_min_dynamic_obstacle_clearance", "enemy_min_dynamic_obstacle_clearance",
        "defender_avoidance_active", "defender_avoidance_obstacle_id",
        "defender_avoidance_obstacle_type", "defender_avoidance_angle_deg",
        "defender_desired_command", "defender_safe_command",
        "defender_predicted_min_clearance", "defender_actual_min_clearance",
        "enemy_obstacle_avoidance_active", "enemy_obstacle_avoidance_mode",
        "enemy_obstacle_avoidance_vector", "enemy_avoidance_obstacle_id",
        "enemy_avoidance_obstacle_type", "enemy_avoidance_angle_deg",
        "enemy_predicted_collision_step", "enemy_predicted_time_to_conflict",
        "enemy_actual_min_clearance", "enemy_mission_command", "enemy_evasion_command",
        "enemy_route_command", "enemy_reactive_avoidance_command", "enemy_final_command",
        "enemy_evasion_active", "enemy_pursuit_direction", "enemy_route_active",
        "enemy_route_current_waypoint", "enemy_route_replan_count",
        "enemy_route_replan_reason", "enemy_behavior_mode",
        "defender_awareness_mode", "enemy_awareness_mode",
        "defender_num_mapped_obstacles_known", "defender_num_locally_sensed_static",
        "defender_num_locally_sensed_dynamic", "defender_total_perceived_obstacles",
        "enemy_num_mapped_obstacles_known", "enemy_num_locally_sensed_static",
        "enemy_num_locally_sensed_dynamic", "enemy_total_perceived_obstacles",
        "enemy_measurement_velocity_valid",
    )
    record = {key: info[key] for key in keys if key in info}
    return _plain(record, unavailable, prefix)


def _event(
    kind: str,
    step: int,
    dt: float,
    source: str,
    *,
    entity: str | None = None,
    obstacle_id: str | None = None,
    position: object | None = None,
    contact_fraction: float | None = None,
) -> ReplayEvent:
    return ReplayEvent(
        kind=kind,
        step=step,
        time_s=step * dt,
        source=source,
        entity=entity,
        obstacle_id=obstacle_id,
        position_m=_vector(position),
        contact_fraction=contact_fraction,
        unavailable_fields=(
            {}
            if contact_fraction is not None
            else {"contact_fraction": "SoldierEnv reports the event state but does not expose substep contact timing."}
        ),
    )


def _swept_fraction(info: dict, *terminal_events: str) -> float | None:
    """Contact fraction of the step's terminal event, only when the simulator
    itself reported it (swept mission-contact semantics). Read-only."""
    if info.get("mission_contact_mode") != "swept" or info.get("terminal_event") not in terminal_events:
        return None
    fraction = info.get("terminal_event_fraction")
    return float(fraction) if fraction is not None else None


def _events_for_step(step: int, info: dict, outcome: str, dt: float) -> list[ReplayEvent]:
    events = []
    if bool(info.get("just_detected")):
        events.append(_event("detection", step, dt, "info.just_detected", entity="attacker", position=info.get("enemy_pos")))
    for entity, contact_key, id_key, event_key, swept_name in (
        ("defender", "defender_collision_event_point", "defender_collision_obstacle_id", "defender_collision_event_type", "defender_collision"),
        ("attacker", "enemy_collision_event_point", "enemy_collision_obstacle_id", "enemy_collision_event_type", "enemy_collision"),
    ):
        if info.get(event_key) is not None:
            events.append(_event(
                "collision", step, dt, f"info.{event_key}",
                entity=entity,
                obstacle_id=str(info[id_key]) if info.get(id_key) is not None else None,
                position=info.get(contact_key),
                contact_fraction=_swept_fraction(info, swept_name),
            ))
    if outcome == "intercepted":
        events.append(_event("interception", step, dt, "info.outcome", entity="attacker", position=info.get("enemy_pos"),
                             contact_fraction=_swept_fraction(info, "interception")))
    elif outcome == "unsafe_intercept":
        events.append(_event("unsafe_interception", step, dt, "info.outcome", entity="attacker", position=info.get("enemy_pos"),
                             contact_fraction=_swept_fraction(info, "interception")))
    elif outcome == "soldier_caught":
        events.append(_event("asset_reached", step, dt, "info.outcome", entity="attacker", position=info.get("soldier_pos"),
                             contact_fraction=_swept_fraction(info, "capture")))
    elif outcome == "timeout":
        events.append(_event("timeout", step, dt, "info.outcome"))
    elif outcome == "enemy_endurance_exhausted":
        events.append(_event("endurance_exhausted", step, dt, "info.outcome", entity="attacker", position=info.get("enemy_pos"),
                             contact_fraction=_swept_fraction(info, "endurance")))
    return events


def _telemetry(
    step: int,
    request: ReplayRequest,
    adapter: _PolicyAdapter,
    requested_action: np.ndarray,
    defender_policy_input: np.ndarray,
    defender_model_input: np.ndarray,
    decision_visible: PolicyVisible,
    previous_velocity: np.ndarray,
    attacker_observation: np.ndarray | None,
    attacker_command: np.ndarray | None,
    lead_vector: np.ndarray | None,
    info: dict,
) -> PolicyTelemetry:
    requested = np.asarray(requested_action, dtype=np.float32).reshape(3)
    velocity = np.asarray(info["defender_vel"], dtype=np.float64)
    executed_velocity = _vector(velocity)
    safe = _vector(info.get("defender_safe_command")) if bool(info.get("controller_action_executed")) else None
    desired = _vector(info.get("defender_desired_command")) if bool(info.get("controller_action_executed")) else None
    if not bool(info.get("controller_action_executed")):
        safe = None
        desired = None
    unavailable: dict[str, str] = {}
    lrppo = None
    lead = None
    pn = None
    rmpc = None

    if adapter.is_lr_ppo:
        native_angle = None
        signed_angle = None
        lead_direction = _vector(lead_vector)
        if lead_direction is None:
            raise _fail(
                "MISSING_LRPPO_DIAGNOSTICS",
                "The already-computed LR-PPO Lead guidance is not a finite native 3D vector.",
                500,
            )
        native_angle = _angle_deg(lead_direction, requested)
        signed_angle = _signed_horizontal_angle(lead_direction, requested)
        lrppo = LRPPOGuidance(
            lead_direction=lead_direction,
            residual_action=None,
            native_adjustment_deg=native_angle,
            signed_horizontal_adjustment_deg=signed_angle,
            max_adjustment_deg=LRPPO_MAX_ANGLE_DEG,
            residual_norm_raw=None,
            residual_perpendicular=None,
            residual_norm_clipped=None,
            near_bound=(native_angle is not None and native_angle >= LRPPO_MAX_ANGLE_DEG - 1.0),
            near_bound_margin_deg=1.0,
        )
        unavailable.update({
            "telemetry.lrppo.residual_action": "The existing wrapper returns only the composed direction; the raw PPO residual is not exposed.",
            "telemetry.lrppo.residual_norm_raw": "The existing wrapper does not expose the raw PPO residual norm.",
            "telemetry.lrppo.residual_perpendicular": "The existing wrapper does not expose its projected residual vector.",
            "telemetry.lrppo.residual_norm_clipped": "Norm clipping cannot be determined without the raw PPO output.",
        })

    policy = adapter.policy
    if hasattr(policy, "last_guidance_mode"):
        mode = getattr(policy, "last_guidance_mode")
        if adapter.policy_id in ("lead", "lead_kalman"):
            lead = LeadGuidance(
                guidance_mode=mode,
                target_position_m=_vector(getattr(policy, "last_target_position", None)),
                target_velocity_mps=_vector(getattr(policy, "last_target_velocity_estimate", None)),
                intercept_time_s=getattr(policy, "last_intercept_time", None),
                intercept_point_m=_vector(getattr(policy, "last_intercept_point", None)),
                discriminant=getattr(policy, "last_discriminant", None),
            )
        if adapter.policy_id in ("pn", "pn_kalman"):
            pn = PNGuidance(
                guidance_mode=mode,
                range_m=getattr(policy, "last_range", None),
                closing_speed_mps=getattr(policy, "last_closing_speed", None),
                los_unit=_vector(getattr(policy, "last_los_unit", None)),
                los_rate_radps=_vector(getattr(policy, "last_los_rate", None)),
                acceleration_mps2=_vector(getattr(policy, "last_pn_acceleration", None)),
            )
        if adapter.policy_id == "rmpc":
            rmpc = RMPCGuidance(
                guidance_mode=mode,
                initialization_mode=getattr(policy, "last_initialization_mode", None),
                scenario_scores=_plain(getattr(policy, "last_scenario_scores", None), unavailable, "$.rmpc.scores"),
                robust_score=getattr(policy, "last_robust_score", None),
                best_sequence=_plain(getattr(policy, "last_best_sequence", None), unavailable, "$.rmpc.sequence"),
                objective_evaluations=getattr(policy, "last_objective_evaluations", None),
                decision_time_s=getattr(policy, "last_decision_time_s", None),
            )

    if not bool(info.get("controller_action_executed")):
        unavailable["telemetry.safe_command"] = "The environment was in pre-detection standby; the requested command was not applied."
    if desired is None:
        unavailable["telemetry.desired_command"] = "The environment does not expose a current executed command for this transition."
    if velocity is not None and executed_velocity is None:
        raise _fail("INVALID_SIMULATION_STATE", "The simulator returned a non-finite defender velocity.", 500)
    if attacker_command is None and request.attacker_condition == "A0":
        unavailable["telemetry.attacker_tactical_command"] = "A0 is the existing scripted attacker; no learned tactical override is injected."

    current_heading = _heading(velocity)
    previous_heading = _heading(previous_velocity)
    heading_change = None
    heading_reversal = None
    if previous_heading is not None and current_heading is not None:
        heading_change = (current_heading - previous_heading + 180.0) % 360.0 - 180.0
        heading_reversal = abs(heading_change) >= HEADING_REVERSAL_THRESHOLD_DEG
    else:
        unavailable["telemetry.heading_change_deg"] = (
            "Horizontal heading is undefined for a zero horizontal velocity."
        )
        unavailable["telemetry.executed_horizontal_heading_deg"] = (
            "Horizontal heading is undefined for a zero horizontal velocity."
        )
    requested_velocity_angle = _angle_deg(requested, velocity)
    if requested_velocity_angle is None:
        unavailable["telemetry.requested_to_velocity_angle_deg"] = (
            "The request or resulting velocity has zero magnitude, so their 3D angle is undefined."
        )
    if lrppo is not None and lrppo.native_adjustment_deg is None:
        unavailable["telemetry.lrppo.native_adjustment_deg"] = "Lead or composed requested direction is degenerate."

    requested_native = _vector(requested)
    requested_to_desired = None
    requested_to_desired_angle = None
    if requested_native is not None and desired is not None:
        requested_to_desired = tuple(float(b - a) for a, b in zip(requested_native, desired))
        requested_to_desired_angle = _angle_deg(requested_native, desired)
    desired_to_safe = None
    desired_to_safe_angle = None
    if desired is not None and safe is not None:
        desired_to_safe = tuple(float(b - a) for a, b in zip(desired, safe))
        desired_to_safe_angle = _angle_deg(desired, safe)
    return PolicyTelemetry(
        decision_step=step - 1,
        decision_policy_visible=decision_visible,
        requested_action=_vector(requested),
        requested_action_norm=float(np.linalg.norm(requested)),
        defender_policy_input=[float(value) for value in defender_policy_input.reshape(-1)],
        defender_model_input=(
            [float(value) for value in defender_model_input.reshape(-1)]
            if defender_model_input.size else None
        ),
        attacker_observation=(
            [float(value) for value in np.asarray(attacker_observation).reshape(-1)]
            if attacker_observation is not None else None
        ),
        attacker_tactical_command=_vector(attacker_command),
        attacker_command_source="a1_learned" if attacker_command is not None else (
            "a0_scripted" if request.attacker_condition == "A0" else None
        ),
        attacker_mission_command=_vector(info.get("enemy_mission_command")),
        attacker_final_command=_vector(info.get("enemy_final_command")),
        desired_command=desired,
        safe_command=safe,
        requested_to_desired_delta=requested_to_desired,
        requested_to_desired_angle_deg=requested_to_desired_angle,
        desired_to_safe_delta=desired_to_safe,
        desired_to_safe_angle_deg=desired_to_safe_angle,
        requested_to_velocity_angle_deg=requested_velocity_angle,
        controller_action_executed=bool(info.get("controller_action_executed", False)),
        defender_standby=bool(info.get("defender_standby", False)),
        avoidance_active=bool(info.get("defender_avoidance_active", False)),
        avoidance_angle_deg=(
            float(info["defender_avoidance_angle_deg"])
            if info.get("defender_avoidance_angle_deg") is not None else None
        ),
        executed_velocity_mps=executed_velocity,
        executed_acceleration_mps2=(
            float(info["defender_accel"]) if info.get("defender_accel") is not None else None
        ),
        executed_turn_rate_degps=(
            float(info["defender_turn_rate"]) if info.get("defender_turn_rate") is not None else None
        ),
        acceleration_saturated=(
            bool(info["defender_accel_saturated"]) if "defender_accel_saturated" in info else None
        ),
        turn_saturated=bool(info["defender_turn_saturated"]) if "defender_turn_saturated" in info else None,
        climb_saturated=bool(info["defender_climb_saturated"]) if "defender_climb_saturated" in info else None,
        requested_horizontal_heading_deg=_heading(requested),
        executed_horizontal_heading_deg=current_heading,
        heading_change_deg=heading_change,
        heading_reversal=heading_reversal,
        lrppo=lrppo,
        lead=lead,
        pn=pn,
        rmpc=rmpc,
        unavailable_fields=unavailable,
    )


def _frame(
    step: int,
    dt: float,
    obs: np.ndarray,
    info: dict,
    env: object,
    estimator: str,
    sectors: np.ndarray | None,
    telemetry: PolicyTelemetry | None,
    reward: float | None,
    cumulative_reward: float | None,
    terminated: bool,
    truncated: bool,
    unavailable: dict[str, str],
    detection_step: int | None,
) -> ReplayFrame:
    visible = _visible(obs, info, estimator, sectors, unavailable)
    enemy_position = _vector(info.get("enemy_pos"))
    defender_position = _vector(info.get("defender_pos"))
    asset_position = _vector(info.get("soldier_pos"))
    if enemy_position is None or defender_position is None or asset_position is None:
        raise _fail("INVALID_SIMULATION_STATE", f"Missing native 3D state at frame {step}.", 500)
    defender_distance = float(np.linalg.norm(np.asarray(defender_position) - enemy_position))
    asset_distance = float(np.linalg.norm(np.asarray(asset_position) - enemy_position))
    tracking_error = info.get("tracking_error")
    privileged = PrivilegedDiagnostics(
        attacker_position_m=enemy_position,
        attacker_velocity_mps=_vector(info.get("enemy_vel")),
        tracking_error_m=(
            float(tracking_error)
            if tracking_error is not None and math.isfinite(float(tracking_error)) else None
        ),
        defender_attacker_range_m=defender_distance,
        attacker_asset_range_m=asset_distance,
        closing_speed_mps=(
            float(np.dot(
                np.asarray(info["defender_vel"], dtype=np.float64)
                - np.asarray(info["enemy_vel"], dtype=np.float64),
                (np.asarray(enemy_position) - np.asarray(defender_position)) / defender_distance,
            ))
            if defender_distance > 1e-8 else None
        ),
        simulator_info=_plain(info, unavailable, f"$.frames[{step}].simulator_info"),
        unavailable_fields={
            **(
                {}
                if defender_distance > 1e-8
                else {"closing_speed_mps": "Entity range is zero; radial closing speed is undefined."}
            ),
            **(
                {}
                if tracking_error is not None
                else {"tracking_error_m": "No hostile estimate exists before detection."}
            ),
            "asset.velocity_mps": "SoldierEnv exposes asset positions but no asset velocity vector.",
            "asset.movement_direction": "SoldierEnv exposes asset positions but no asset velocity vector.",
        },
    )
    if tracking_error is None or not math.isfinite(float(tracking_error)):
        unavailable[f"frames[{step}].privileged_diagnostics.tracking_error_m"] = (
            "No hostile estimate exists before detection."
            if tracking_error is None
            else "The simulator tracking diagnostic was non-finite."
        )
    if step == 0:
        unavailable[f"frames[{step}].reward"] = "The reset frame has no preceding transition reward."
    return ReplayFrame(
        step=step,
        time_s=step * dt,
        defender=_entity(info["defender_pos"], info.get("defender_vel")),
        attacker=_entity(info["enemy_pos"], info.get("enemy_vel")),
        asset=_entity(info["soldier_pos"], None),
        moving_obstacles=_moving_obstacles(env),
        policy_visible=visible,
        telemetry=telemetry,
        privileged_diagnostics=privileged,
        detection_radius_m=float(env.config.detection_radius),
        detection_time_step=detection_step,
        defender_action_applied=(
            bool(info.get("controller_action_executed"))
            if "controller_action_executed" in info else None
        ),
        reward=reward,
        cumulative_reward=cumulative_reward,
        terminated=terminated,
        truncated=truncated,
        outcome=info["outcome"],
        environmental_state=_environment_state(info, unavailable, f"$.frames[{step}].environmental_state"),
        unavailable_fields=unavailable,
    )


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "gymnasium", "stable-baselines3", "torch", "pydantic"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def build_replay_config(
    modules: dict[str, object],
    semantics: str,
    clutter_level: str,
    estimator: str,
    perception_mode: str,
) -> tuple[object, str]:
    """Resolve the research EnvConfig for the requested simulation semantics.

    The visualizer implements no simulation semantics itself: both branches call
    the research repository's own config builders.

    V1 keeps the exact call the visualizer always made (`build_canonical_env_config`),
    so V1 replays are unchanged. V2 uses the research repo's explicit V2 builder
    (swept mission contact + direct obstacle-only A0). After building, the
    resolved switches are asserted so the result never depends on a default.
    """
    expected = {
        "V1": ("endpoint", "legacy"),
        "V2": ("swept", "direct_obstacle_only"),
    }[semantics]
    if semantics == "V2":
        builder_name = "experiments.journal_v2_protocol.build_v2_env_config"
        config = modules["experiments.journal_v2_protocol"].build_v2_env_config(
            clutter_level, estimator, awareness_mode=perception_mode,
        )
    else:
        builder_name = "experiments.journal_clutter_protocol.build_canonical_env_config"
        config = modules["experiments.journal_clutter_protocol"].build_canonical_env_config(
            clutter_level, estimator, awareness_mode=perception_mode,
        )
    resolved = (config.mission_contact_mode, config.enemy_nominal_guidance_mode)
    if resolved != expected:
        raise _fail(
            "SEMANTICS_CONFIG_MISMATCH",
            f"Semantics {semantics} must resolve to {expected}, but the research config resolved to {resolved}.",
            500,
        )
    return config, builder_name


def run_replay_episode(request: ReplayRequest, settings: Settings | None = None) -> ReplayBundle:
    """Run one real, deterministic SoldierEnv episode and return its validated trace."""
    if request.deterministic_inference is not True:
        raise _fail("INVALID_REQUEST", "Phase 2 only supports deterministic model inference.")
    selected_settings = settings if settings is not None else __import__(
        "visualizer.backend.config", fromlist=["load_settings"]
    ).load_settings()
    store = ResearchDataStore(selected_settings)
    if store.root is None:
        raise _fail("DATA_ROOT_NOT_CONFIGURED", "Configure VISUALIZER_RESEARCH_DATA_ROOT before replay.", 503)
    research_root = store.root.resolve()
    source_commit = _git_value(research_root, "rev-parse", "HEAD")
    if source_commit != EXPECTED_RESEARCH_COMMIT:
        raise _fail(
            "UNVERIFIED_SIMULATOR_COMMIT",
            f"Expected research source {EXPECTED_RESEARCH_COMMIT}, found {source_commit}. "
            "Update the audited source identity before replay.",
            409,
        )
    registry, registry_sha256, _registry_bytes = _read_registry(store)
    modules = _research_modules(research_root)
    estimator = (
        CLUTTER_ARCHITECTURES[request.defender_policy_id][2]
        if request.defender_policy_id in CLUTTER_ARCHITECTURES
        and request.defender_generation != "conference"
        else CONFERENCE_ARCHITECTURES[request.defender_policy_id][2]
        if request.defender_policy_id in CONFERENCE_ARCHITECTURES
        else "kalman" if request.defender_policy_id == "kalman_greedy" else "measurement"
    )
    config, builder_name = build_replay_config(
        modules, request.semantics, request.clutter_level, estimator, request.perception_mode,
    )
    if request.defender_policy_id == "passive":
        # Same override as experiments.journal_v2_validation_core.build_config("passive"): the
        # parked defender is not synchronised onto the asset, so it can never pre-empt capture.
        config = dataclasses.replace(config, defender_standby_until_detection=False)
    adapter = _load_defender(request, config, store, registry, modules)
    attacker_checkpoint = _resolve_attacker_checkpoint(request.attacker_condition, registry, store)
    attacker_model = _load_attacker(request.attacker_condition, attacker_checkpoint, store, modules)
    expected_attacker_model = request.attacker_condition != "A0"
    if (attacker_model is not None) != expected_attacker_model:
        raise _fail("INVALID_ATTACKER_CONFIGURATION", "Attacker selection did not resolve to its expected policy.", 409)
    env = modules["uav_defend.envs.soldier_env"].SoldierEnv(config)

    checkpoint_hashes = {
        item.artifact_id: item.sha256
        for item in (adapter.checkpoint, attacker_checkpoint)
        if item is not None
    }
    static_snapshot: list[StaticObstacle] = []
    frames: list[ReplayFrame] = []
    events: list[ReplayEvent] = []
    unavailable_initial: dict[str, str] = {}
    total_reward = 0.0
    detection_step = None
    avoidance_steps = 0
    min_interception_distance = math.inf
    min_attacker_asset_distance = math.inf
    defender_path_length = 0.0
    attacker_path_length = 0.0
    previous_defender_position = None
    previous_attacker_position = None
    latest_info: dict | None = None
    completed_steps = 0
    try:
        if request.attacker_condition == "A0":
            env.enemy_tactical_override = None
        else:
            pending = {"action": np.zeros(3, dtype=np.float32)}
            env.enemy_tactical_override = lambda: pending["action"]

        observation, info = env.reset(seed=request.evaluation_seed)
        if request.defender_policy_id == "passive":
            # Same placement as record_episode(controller="passive"): parked far away, at rest.
            env._defender_pos = np.asarray(modules["experiments.journal_v2_validation_core"].PARKED_DEFENDER, dtype=np.float32)
            env._defender_vel = np.zeros(3, dtype=np.float32)
            observation = env._get_obs()
            info = env._get_info()
        if bool(info.get("enemy_detected")):
            detection_step = 0
        adapter.policy.reset()
        static_snapshot = _static_obstacles(env)
        sector = _sector_features(env, adapter, modules)
        initial_frame = _frame(
            0, config.dt, observation, info, env, adapter.estimator, sector,
            None, None, 0.0, False, False, unavailable_initial, detection_step,
        )
        frames.append(initial_frame)
        previous_defender_position = np.asarray(info["defender_pos"], dtype=np.float64).copy()
        previous_attacker_position = np.asarray(info["enemy_pos"], dtype=np.float64).copy()
        min_interception_distance = float(info["defender_enemy_dist"])
        min_attacker_asset_distance = float(info["enemy_soldier_dist"])
        latest_info = info
        done = False
        while not done:
            attacker_observation = None
            attacker_command = None
            if attacker_model is not None:
                attacker_observation = _build_attacker_observation(request.attacker_condition, modules, env)
                command, _state = attacker_model.predict(attacker_observation, deterministic=True)
                attacker_command = np.asarray(command, dtype=np.float32).reshape(3)
                if not np.all(np.isfinite(attacker_command)):
                    raise _fail("INVALID_POLICY_ACTION", "The selected attacker returned a non-finite action.", 500)
                pending["action"] = attacker_command

            defender_observation = env._get_obs()
            full_policy_info = env._get_info()
            policy_info = modules["uav_defend.policies.sanitize"].build_policy_info(
                full_policy_info,
                adapter.estimator,
            )
            decision_unavailable: dict[str, str] = {}
            decision_visible = _visible(
                defender_observation,
                full_policy_info,
                adapter.estimator,
                sector,
                decision_unavailable,
                sanitized=policy_info,
            )

            model_input: np.ndarray | None = None
            lead_vector = None
            if adapter.is_lr_ppo:
                requested_action = adapter.policy.act(
                    defender_observation,
                    policy_info,
                    extra_features=sector,
                )
                lead_vector = getattr(adapter.policy._lead_policy, "last_action", None)
                if lead_vector is None:
                    raise _fail(
                        "MISSING_LRPPO_DIAGNOSTICS",
                        "The LR-PPO wrapper did not expose its already-computed Lead action.",
                        500,
                    )
                residual_input = modules[
                    "uav_defend.policies.residual.lead_residual_observation"
                ].build_residual_observation(defender_observation, lead_vector)
                model_input = (
                    np.concatenate([residual_input, np.asarray(sector, dtype=np.float32)])
                    if sector is not None else residual_input
                ).astype(np.float32)
                defender_policy_input = defender_observation
            elif adapter.needs_sector_features:
                if sector is None:
                    raise _fail("INCOMPATIBLE_POLICY", "The selected clutter model requires sector features.", 409)
                defender_policy_input = np.concatenate(
                    [defender_observation, np.asarray(sector, dtype=np.float32)]
                ).astype(np.float32)
                model_input = defender_policy_input
                requested_action = adapter.policy.act(defender_policy_input, policy_info)
            else:
                defender_policy_input = np.asarray(defender_observation, dtype=np.float32)
                model_input = defender_policy_input
                requested_action = adapter.policy.act(defender_policy_input, policy_info)

            requested_action = np.asarray(requested_action, dtype=np.float32).reshape(3)
            if not np.all(np.isfinite(requested_action)):
                raise _fail("INVALID_POLICY_ACTION", "The selected defender returned a non-finite action.", 500)
            previous_velocity = np.asarray(env._defender_vel, dtype=np.float64).copy()
            next_observation, reward, terminated, truncated, step_info = env.step(requested_action)
            completed_steps += 1
            done = bool(terminated or truncated)
            if not done and step_info["outcome"] != "ongoing":
                raise _fail(
                    "INCONSISTENT_TERMINAL_STATE",
                    "The simulator reported a terminal outcome without its terminal/truncated flag.",
                    500,
                )
            if done and step_info["outcome"] == "ongoing":
                raise _fail(
                    "INCONSISTENT_TERMINAL_STATE",
                    "The simulator terminated without reporting a native terminal outcome.",
                    500,
                )

            # No extra simulation step, stateful policy call, or RNG draw is
            # performed for recording; only returned state and deterministic
            # diagnostic fields are read after the authoritative transition.
            total_reward += float(reward)
            if detection_step is None and bool(step_info.get("enemy_detected")):
                detection_step = completed_steps
            if bool(step_info.get("defender_avoidance_active")):
                avoidance_steps += 1
            next_sector = _sector_features(env, adapter, modules)
            defender_position = np.asarray(step_info["defender_pos"], dtype=np.float64)
            attacker_position = np.asarray(step_info["enemy_pos"], dtype=np.float64)
            asset_position = np.asarray(step_info["soldier_pos"], dtype=np.float64)
            defender_path_length += float(np.linalg.norm(defender_position - previous_defender_position))
            attacker_path_length += float(np.linalg.norm(attacker_position - previous_attacker_position))
            previous_defender_position = defender_position.copy()
            previous_attacker_position = attacker_position.copy()
            min_interception_distance = min(
                min_interception_distance,
                float(np.linalg.norm(defender_position - attacker_position)),
            )
            min_attacker_asset_distance = min(
                min_attacker_asset_distance,
                float(np.linalg.norm(asset_position - attacker_position)),
            )
            telemetry_unavailable: dict[str, str] = {}
            telemetry_unavailable.update(decision_unavailable)
            telemetry = _telemetry(
                completed_steps,
                request,
                adapter,
                requested_action,
                np.asarray(defender_policy_input, dtype=np.float32),
                np.asarray(model_input, dtype=np.float32) if model_input is not None else np.empty(0),
                decision_visible,
                previous_velocity,
                attacker_observation,
                attacker_command,
                np.asarray(lead_vector) if lead_vector is not None else None,
                step_info,
            )

            frames.append(_frame(
                completed_steps,
                config.dt,
                next_observation,
                step_info,
                env,
                adapter.estimator,
                next_sector,
                telemetry,
                float(reward),
                total_reward,
                bool(terminated),
                bool(truncated),
                telemetry_unavailable,
                detection_step,
            ))
            events.extend(_events_for_step(completed_steps, step_info, step_info["outcome"], config.dt))
            latest_info = step_info
            sector = next_sector

        if latest_info is None:
            raise _fail("INVALID_REPLAY", "SoldierEnv returned no final frame information.", 500)
        outcome = latest_info["outcome"]
        if outcome not in {
            "intercepted", "soldier_caught", "unsafe_intercept", "timeout",
            "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision",
            "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision",
            "enemy_endurance_exhausted",
        }:
            raise _fail("INVALID_SIMULATION_OUTCOME", f"Unknown native simulator outcome: {outcome!r}.", 500)
        collision_step = None
        collision_type = outcome if outcome in _COLLISION_OUTCOMES else None
        for event in reversed(events):
            if event.kind == "collision":
                collision_step = event.step
                break
        terminal_step = completed_steps
        detector = adapter.estimator
        visible_shape = (
            (16 + SECTOR_FEATURE_DIM,) if adapter.needs_sector_features and not adapter.is_lr_ppo
            else adapter.model_input_shape
        )
        defender_selection = PolicySelection(
            policy_kind="learned" if adapter.checkpoint else "analytical",
            policy_id=adapter.policy_id,
            architecture=adapter.architecture,
            generation=adapter.generation,
            training_seed=adapter.seed,
            checkpoint=(
                Checkpoint(
                    artifact_id=adapter.checkpoint.artifact_id,
                    research_relative_path=adapter.checkpoint.relative_path,
                    sha256=adapter.checkpoint.sha256,
                    role=adapter.checkpoint.role,
                )
                if adapter.checkpoint else None
            ),
            observation_shape=visible_shape,
            estimator=detector,
            parameters={
                "deterministic_inference": request.deterministic_inference,
                "sector_feature_dim": SECTOR_FEATURE_DIM if adapter.needs_sector_features else 0,
                "sector_observation_spec": (
                    modules["uav_defend.obstacles.sector_features"].SECTOR_OBSERVATION_V1.to_dict()
                    if adapter.needs_sector_features else None
                ),
                "residual_max_angle_deg": LRPPO_MAX_ANGLE_DEG if adapter.is_lr_ppo else None,
                "rmpc_config": (
                    dataclasses.asdict(adapter.policy.rmpc_config)
                    if adapter.policy_id == "rmpc" else None
                ),
                "heading_reversal_threshold_deg": HEADING_REVERSAL_THRESHOLD_DEG,
            },
        )
        is_v2_attacker = request.attacker_condition in V2_ATTACKER_SEEDS
        attacker_selection = PolicySelection(
            policy_kind="scripted" if attacker_model is None else "learned",
            policy_id=request.attacker_condition,
            architecture=(
                "scripted_a0" if attacker_model is None
                else "a1_v2_ppo" if is_v2_attacker else "adaptive_attacker_v1"
            ),
            generation=None,
            training_seed=attacker_checkpoint.training_seed if attacker_checkpoint else None,
            checkpoint=(
                Checkpoint(
                    artifact_id=attacker_checkpoint.artifact_id,
                    research_relative_path=attacker_checkpoint.relative_path,
                    sha256=attacker_checkpoint.sha256,
                    role=attacker_checkpoint.role,
                )
                if attacker_checkpoint else None
            ),
            observation_shape=(
                None if attacker_model is None else (V2_A1_OBSERVATION_DIM,) if is_v2_attacker else (36,)
            ),
            estimator="attacker_local" if attacker_model is not None else "scripted_ground_truth",
            parameters={
                "deterministic_inference": True if attacker_model is not None else None,
                "tactical_override": attacker_model is not None,
                "held_out_from_defender_adaptation": request.attacker_condition in ("A1-203", "A1-203-V2"),
                "canonical_v2_attacker": is_v2_attacker,
                "display_label": (
                    f"Attacker: {request.attacker_condition.replace('-V2', ' V2')}" if attacker_model is not None
                    else "Attacker: A0"
                ),
                "training_protocol": V2_A1_PROTOCOL if is_v2_attacker else None,
                "reward_version": V2_A1_REWARD_VERSION if is_v2_attacker else None,
            },
        )
        config_dict = _plain(dataclasses.asdict(config), {}, "$.config")
        config_json = json.dumps(config_dict, sort_keys=True, separators=(",", ":"), allow_nan=False)
        env_spec = EnvironmentSpec(
            clutter=request.clutter_level,
            awareness=request.perception_mode,
            builder=builder_name,
            semantics=request.semantics,
            config=config_dict,
            config_sha256=hashlib.sha256(config_json.encode("utf-8")).hexdigest(),
            dt_s=float(config.dt),
        )
        final_defender_distance = float(latest_info["defender_enemy_dist"])
        final_attacker_asset_distance = float(latest_info["enemy_soldier_dist"])
        summary = EpisodeSummary(
            outcome=outcome,
            steps=completed_steps,
            duration_s=completed_steps * float(config.dt),
            detection_time_step=detection_step,
            detection_time_s=detection_step * float(config.dt) if detection_step is not None else None,
            interception_step=terminal_step if outcome == "intercepted" else None,
            interception_time_s=terminal_step * float(config.dt) if outcome == "intercepted" else None,
            collision_type=collision_type,
            collision_step=collision_step,
            collision_time_s=collision_step * float(config.dt) if collision_step is not None else None,
            final_defender_attacker_distance_m=final_defender_distance,
            final_attacker_asset_distance_m=final_attacker_asset_distance,
            total_reward=total_reward,
            safety_metrics={
                "defender_avoidance_steps": avoidance_steps,
                "defender_avoidance_fraction": avoidance_steps / completed_steps,
                "minimum_defender_attacker_distance_m": min_interception_distance,
                "minimum_attacker_asset_distance_m": min_attacker_asset_distance,
                "defender_path_length_m": defender_path_length,
                "attacker_path_length_m": attacker_path_length,
                "attacker_endurance_enabled": bool(latest_info.get("enemy_endurance_enabled", False)),
                "attacker_endurance_budget_m": latest_info.get("enemy_endurance_budget_m"),
                "attacker_endurance_traveled_m": latest_info.get("enemy_endurance_traveled_m"),
                "attacker_endurance_remaining_m": latest_info.get("enemy_endurance_remaining_m"),
                "defender_collision_count": int(latest_info.get("defender_collision_count", 0)),
                "attacker_collision_count": int(latest_info.get("enemy_collision_count", 0)),
                "defender_dynamic_collision_count": int(latest_info.get("defender_dynamic_collision_count", 0)),
                "attacker_dynamic_collision_count": int(latest_info.get("enemy_dynamic_collision_count", 0)),
            },
        )
        simulation_dirty = _git_dirty(research_root)
        visualizer_root = Path(__file__).resolve().parents[2]
        visualizer_commit = _git_value(visualizer_root, "rev-parse", "HEAD")
        provenance = ReplayProvenance(
            simulation_source_commit=source_commit,
            visualizer_source_commit=visualizer_commit,
            source_dirty=simulation_dirty,
            visualizer_dirty=_git_dirty(visualizer_root),
            deterministic_inference=True,
            python_version=sys.version.split()[0],
            package_versions=_package_versions(),
            model_device="cpu",
            registry_sha256=registry_sha256,
        )
        episode = ReplayEpisode(
            episode_id=(
                f"{request.evaluation_seed}-{adapter.policy_id}-{adapter.generation or 'analytical'}-"
                f"{adapter.seed if adapter.seed is not None else 'script'}-{request.attacker_condition}-{request.clutter_level}"
            ),
            simulation_seed=request.evaluation_seed,
            defender=defender_selection,
            attacker=attacker_selection,
            environment=env_spec,
            terminal_flags={
                "terminated": frames[-1].terminated,
                "truncated": frames[-1].truncated,
            },
            static_obstacles=static_snapshot,
            frames=frames,
            events=events,
            summary=summary,
        )
        bundle = ReplayBundle(
            provenance=provenance,
            episode_count=1,
            seed_start=request.evaluation_seed,
            episodes=[episode],
        )
        _validate_trace(bundle)
        for item in (adapter.checkpoint, attacker_checkpoint):
            if item is not None and _sha256(store.resolve_file(item.relative_path)) != item.sha256:
                raise _fail("CHECKPOINT_MUTATED", f"{item.artifact_id} changed during replay.", 409)
        return bundle
    finally:
        env.close()


def save_replay_bundle(bundle: ReplayBundle, output_name: str | None = None) -> Path:
    """Atomically persist a single replay below the gitignored local output root."""
    _validate_trace(bundle)
    episode = bundle.episodes[0]
    default_name = (
        f"replay-{episode.simulation_seed}-{episode.defender.policy_id}-"
        f"{episode.defender.generation or 'analytical'}-{episode.attacker.policy_id}-{episode.environment.clutter}"
    )
    stem = output_name if output_name is not None else default_name
    if not _SAFE_OUTPUT_NAME.fullmatch(stem) or stem in {".", ".."}:
        raise _fail("INVALID_OUTPUT_NAME", "Output name must be a simple filename stem (letters, digits, dot, underscore, hyphen).")
    root = local_output_root()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{stem}.json.gz"
    if destination.exists():
        raise _fail("OUTPUT_ALREADY_EXISTS", f"Refusing to overwrite an existing replay: {destination}", 409)
    raw_json = json.dumps(
        bundle.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".replay-", suffix=".tmp", dir=root, delete=False) as temp:
            temporary = Path(temp.name)
            import gzip

            with gzip.GzipFile(fileobj=temp, mode="wb", filename="", mtime=0) as compressed:
                compressed.write(raw_json)
            temp.flush()
            os.fsync(temp.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise _fail("OUTPUT_ALREADY_EXISTS", f"Refusing to overwrite an existing replay: {destination}", 409) from exc
    except OSError as exc:
        raise _fail("REPLAY_WRITE_FAILED", f"Cannot atomically save replay under {root}: {exc}", 500) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def load_replay_bundle(path: Path) -> ReplayBundle:
    """Read and fully validate a compressed replay file."""
    import gzip

    try:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            bundle = ReplayBundle.model_validate_json(source.read())
    except (OSError, EOFError, UnicodeDecodeError, ValueError) as exc:
        raise _fail("INVALID_REPLAY_FILE", f"Cannot read a valid compressed replay at {path}.", 400) from exc
    _validate_trace(bundle)
    return bundle


def current_replay_source_provenance(settings: Settings) -> dict[str, object]:
    """Read-only utility for CLI/test reporting; does not load a model or env."""
    store = ResearchDataStore(settings)
    if store.root is None:
        raise _fail("DATA_ROOT_NOT_CONFIGURED", "Configure VISUALIZER_RESEARCH_DATA_ROOT before replay.", 503)
    source = _git_value(store.root, "rev-parse", "HEAD")
    if source != EXPECTED_RESEARCH_COMMIT:
        raise _fail("UNVERIFIED_SIMULATOR_COMMIT", f"Unverified simulator commit {source}.", 409)
    _registry, registry_sha, _raw = _read_registry(store)
    return {
        "research_root": str(store.root),
        "simulation_source_commit": source,
        "source_dirty": _git_dirty(store.root),
        "registry_sha256": registry_sha,
    }
