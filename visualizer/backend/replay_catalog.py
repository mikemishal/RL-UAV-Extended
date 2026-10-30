from __future__ import annotations

import hashlib
import json
import dataclasses

from .application_error import ApplicationError
from .config import Settings
from .data_access import ResearchDataStore
from .replay_engine import (
    ATTACKER_SEEDS,
    CLUTTER_ARCHITECTURES,
    CONFERENCE_ARCHITECTURES,
    EXPECTED_RESEARCH_COMMIT,
    REGISTRY_RELATIVE_PATH,
    SECTOR_FEATURE_DIM,
    V2_ATTACKER_SEEDS,
    _git_value,
    _load_defender,
    _read_registry,
    _resolve_attacker_checkpoint,
    _resolve_defender_checkpoint,
    _research_modules,
    _verify_artifact,
)
from .replay_requests import ReplayRequest

SCENARIO_LEVELS = (
    ("C0", "Unobstructed"),
    ("C1", "Low clutter"),
    ("C2", "Medium clutter"),
    ("C3", "High clutter"),
)
PERCEPTIONS = (
    ("local", "Obstacle information from the supported local sensing model."),
    ("map_local", "Known static obstacle map plus applicable local sensing."),
)
SUPPORTED_SCENARIOS = [
    {"clutter": clutter, "perception_modes": ["local", "map_local"]}
    for clutter, _label in SCENARIO_LEVELS
]
ANALYTICAL_METADATA = {
    "greedy": ("Greedy", "measurement", ["requested_action", "desired_command", "safe_command", "executed_velocity_mps"]),
    "kalman_greedy": ("Kalman Greedy", "kalman", ["requested_action", "e_hat_m", "v_hat_mps", "executed_velocity_mps"]),
    "pn": ("Proportional Navigation", "measurement", ["range_m", "closing_speed_mps", "los_unit", "los_rate_radps", "acceleration_mps2"]),
    "lead": ("Lead", "measurement", ["target_position_m", "target_velocity_mps", "intercept_time_s", "intercept_point_m"]),
    "rmpc": ("RMPC", "measurement", ["scenario_scores", "robust_score", "best_sequence", "objective_evaluations", "decision_time_s"]),
    "passive": ("Passive (parked, no action)", "measurement", ["executed_velocity_mps"]),
}
LEARNED_METADATA = {
    "ppo": ("PPO", ["policy_visible.observation", "requested_action", "executed_velocity_mps"]),
    "ppo_kalman": ("PPO-Kalman", ["policy_visible.e_hat_m", "policy_visible.v_hat_mps", "tracking_error_m", "requested_action", "executed_velocity_mps"]),
    "lr_ppo": ("LR-PPO", ["lead_direction", "native_adjustment_deg", "signed_horizontal_adjustment_deg", "requested_action", "safe_command"]),
}
GENERATION_LABELS = {
    "conference": "Conference",
    "d0": "D0 — original clutter-trained",
    "d0_plus": "D0+ — scripted continuation",
    "d1": "D1 — adversarial-mixture continuation",
}
GENERATION_ROLES = {
    "conference": "conference_best",
    "d0": "d0_start",
    "d0_plus": "final",
    "d1": "final",
}


def _checkpoint_reason(exc: ApplicationError) -> str:
    if exc.code == "ARTIFACT_NOT_FOUND":
        return "Registered checkpoint is missing."
    return "Registered checkpoint failed path or SHA-256 integrity verification."


def build_replay_catalog(settings: Settings) -> dict[str, object]:
    """Build catalog metadata from actual implementations, canonical builder, and frozen registries."""
    store = ResearchDataStore(settings)
    if store.root is None:
        raise ApplicationError("DATA_ROOT_NOT_CONFIGURED", "Configure VISUALIZER_RESEARCH_DATA_ROOT before listing policies.", 503)
    source_commit = _git_value(store.root, "rev-parse", "HEAD")
    if source_commit != EXPECTED_RESEARCH_COMMIT:
        raise ApplicationError("UNVERIFIED_SIMULATOR_COMMIT", f"Unverified simulator commit {source_commit}.", 409)
    modules = _research_modules(store.root)
    registry, registry_sha, _raw_registry = _read_registry(store)
    builder = modules["experiments.journal_clutter_protocol"].build_canonical_env_config

    scenarios: list[dict[str, object]] = []
    for clutter, label in SCENARIO_LEVELS:
        perceptions = []
        for mode, description in PERCEPTIONS:
            configurations = {}
            for estimator in ("measurement", "kalman"):
                config = builder(clutter, estimator, awareness_mode=mode)
                parameters = dataclasses.asdict(config)
                canonical = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
                configurations[estimator] = {
                    "config_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                    "parameters": parameters,
                }
            perceptions.append({"id": mode, "description": description, "configurations": configurations})
        scenarios.append({"clutter": clutter, "display_name": label, "perceptions": perceptions})

    defenders: list[dict[str, object]] = []
    for policy_id, (display, estimator, diagnostics) in ANALYTICAL_METADATA.items():
        request = ReplayRequest(
            defender_policy_id=policy_id, attacker_condition="A0", clutter_level="C0",
            perception_mode="map_local", evaluation_seed=0,
        )
        try:
            config = builder("C0", estimator, awareness_mode="map_local")
            _load_defender(request, config, store, registry, modules)
            available, reason = True, None
        except ApplicationError:
            available, reason = False, "The registered analytical controller could not be initialized."
        defenders.append({
            "policy_id": policy_id,
            "display_name": display,
            "classification": "analytical",
            "architecture": policy_id,
            "available": available,
            "unavailability_reason": reason,
            "estimator": estimator,
            "observation_shape": [16],
            "generations": [],
            "training_seeds": [],
            "checkpoint_required": False,
            "checkpoint_role_rule": None,
            "supported_scenarios": SUPPORTED_SCENARIOS,
            "configurable_parameters": [],
            "diagnostic_fields": diagnostics,
        })

    for policy_id, (display, diagnostics) in LEARNED_METADATA.items():
        is_clutter_policy = policy_id in CLUTTER_ARCHITECTURES
        generations = []
        all_seeds = []
        for generation in ("conference", "d0", "d0_plus", "d1"):
            if generation == "conference":
                architecture, seeds, estimator, width = CONFERENCE_ARCHITECTURES[policy_id]
                sector_features = False
                residual_input = 0
            else:
                architecture, seeds, estimator, sector_features, is_lr = CLUTTER_ARCHITECTURES[policy_id]
                residual_input = 3 if is_lr else 0
                width = 16 + (SECTOR_FEATURE_DIM if sector_features else 0) + residual_input
            generation_seeds = []
            for seed in seeds:
                request = ReplayRequest(
                    defender_policy_id=policy_id,
                    defender_generation=generation,
                    defender_training_seed=seed,
                    attacker_condition="A0",
                    clutter_level="C0",
                    perception_mode="map_local",
                    evaluation_seed=0,
                )
                checkpoint = None
                try:
                    checkpoint, architecture, shape, _sectors, _is_lr = _resolve_defender_checkpoint(
                        request, store, registry, modules
                    )
                    _verify_artifact(store, checkpoint)
                    available, reason = True, None
                    width = shape[0]
                except ApplicationError as exc:
                    available, reason = False, _checkpoint_reason(exc)
                seed_entry = {
                    "training_seed": seed,
                    "available": available,
                    "checkpoint_role": checkpoint.role if checkpoint else GENERATION_ROLES[generation],
                    "checkpoint_sha256": checkpoint.sha256 if checkpoint else None,
                    "unavailability_reason": reason,
                    "observation_shape": [width],
                    "estimator": estimator,
                }
                generation_seeds.append(seed_entry)
                all_seeds.append({"generation": generation, **seed_entry})
            generations.append({
                "generation": generation,
                "display_name": GENERATION_LABELS[generation],
                "architecture": architecture,
                "checkpoint_role": GENERATION_ROLES[generation],
                "available": any(item["available"] for item in generation_seeds),
                "training_seeds": generation_seeds,
                "estimator": estimator,
                "observation_shape": [width],
                "sector_features": sector_features,
                "configurable_parameters": [],
                "diagnostic_fields": diagnostics,
            })
        available = any(item["available"] for item in all_seeds)
        defenders.append({
            "policy_id": policy_id,
            "display_name": display + (" (clutter trained)" if is_clutter_policy else ""),
            "classification": "learned",
            "architecture": None,
            "available": available,
            "unavailability_reason": None if available else "No registered checkpoints passed availability and integrity checks.",
            "estimator": None,
            "observation_shape": None,
            "generations": generations,
            "training_seeds": all_seeds,
            "checkpoint_required": True,
            "checkpoint_role_rule": "D0=d0_start; D0+/D1=primary final; secondary best is not selectable.",
            "supported_scenarios": SUPPORTED_SCENARIOS,
            "configurable_parameters": [],
            "diagnostic_fields": diagnostics,
        })

    attackers: list[dict[str, object]] = [{
        "attacker_id": "A0",
        "display_name": "A0 — Scripted attacker",
        "classification": "scripted",
        "architecture": "canonical_scripted",
        "training_seed": None,
        "held_out": False,
        "available": True,
        "checkpoint_sha256": None,
        "unavailability_reason": None,
        "observation_shape": None,
        "supported_scenarios": SUPPORTED_SCENARIOS,
        "diagnostic_fields": ["scripted_mission_command", "final_movement_command"],
    }]
    for attacker_id, seed in ATTACKER_SEEDS.items():
        checkpoint = None
        try:
            checkpoint = _resolve_attacker_checkpoint(attacker_id, registry, store)
            if checkpoint is None:
                raise ApplicationError("CHECKPOINT_NOT_REGISTERED", "Attacker checkpoint is not registered.", 404)
            _verify_artifact(store, checkpoint)
            available, reason = True, None
        except ApplicationError as exc:
            available, reason = False, _checkpoint_reason(exc)
        attackers.append({
            "attacker_id": attacker_id,
            "display_name": f"{attacker_id} (legacy V1-trained) — " + ("Held-out learned attacker" if seed == 203 else "Learned attacker"),
            "classification": "learned",
            "architecture": "adaptive_attacker_v1",
            "training_seed": seed,
            "held_out": seed == 203,
            "available": available,
            "checkpoint_sha256": checkpoint.sha256 if checkpoint else None,
            "unavailability_reason": reason,
            "observation_shape": [36],
            "supported_scenarios": SUPPORTED_SCENARIOS,
            "diagnostic_fields": ["attacker_observation", "tactical_command", "mission_command", "final_movement_command"],
        })

    for attacker_id, seed in V2_ATTACKER_SEEDS.items():
        checkpoint = None
        try:
            checkpoint = _resolve_attacker_checkpoint(attacker_id, registry, store)
            if checkpoint is None:
                raise ApplicationError("CHECKPOINT_NOT_REGISTERED", "Attacker checkpoint is not registered.", 404)
            _verify_artifact(store, checkpoint)
            available, reason = True, None
        except ApplicationError as exc:
            available, reason = False, _checkpoint_reason(exc)
        attackers.append({
            "attacker_id": attacker_id,
            "display_name": f"Attacker: {attacker_id.replace('-V2', ' V2')}"
            + (" — held out from defender adaptation" if seed == 203 else " — canonical V2 learned attacker"),
            "classification": "learned",
            "architecture": "a1_v2_ppo",
            "training_seed": seed,
            "held_out": seed == 203,
            "available": available,
            "checkpoint_sha256": checkpoint.sha256 if checkpoint else None,
            "unavailability_reason": reason,
            "observation_shape": [37],
            "supported_scenarios": [item for item in SUPPORTED_SCENARIOS],
            "diagnostic_fields": ["attacker_observation", "tactical_command", "mission_command", "final_movement_command"],
        })

    return {
        "source_commit": source_commit,
        "registry_sha256": registry_sha,
        "deterministic_inference": True,
        "defenders": defenders,
        "attackers": attackers,
        "scenarios": scenarios,
    }
