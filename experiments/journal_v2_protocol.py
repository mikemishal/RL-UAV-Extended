"""Journal V2 (corrected attacker semantics) canonical configuration.

V2 = swept mission contacts + corrected scripted-A0 guidance (direct pursuit,
obstacle-triggered deviation only). V1 = the original endpoint contact and
weave/evasion A0 behavior that produced every dataset frozen at
`journal-robustness-assurance-v1`.

Publication experiments must build their environments through
`build_v2_env_config`, which sets BOTH semantic switches explicitly and
asserts them; they never depend on an `EnvConfig` default. V1 behavior stays
reproducible through `build_v1_env_config` (explicit legacy fields).

See docs/corrected_attacker_semantics_v2.md and
docs/journal_v2_results_provenance.md.
"""

from __future__ import annotations

from pathlib import Path

from uav_defend.config.env_config import EnvConfig

from experiments.journal_clutter_protocol import (
    CANONICAL_AWARENESS_MODE,
    CANONICAL_ENEMY_ROUTE_PLANNING_ENABLED,
    build_canonical_env_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# All future corrected-semantics results live here; V1 result directories
# are never written to again.
V2_RESULTS_DIR = PROJECT_ROOT / "results" / "journal_v2_corrected_semantics"

V1_SEMANTIC_FIELDS: dict = {
    "mission_contact_mode": "endpoint",
    "enemy_nominal_guidance_mode": "legacy",
}
V2_SEMANTIC_FIELDS: dict = {
    "mission_contact_mode": "swept",
    "enemy_nominal_guidance_mode": "direct_obstacle_only",
}

# Literal "no obstacle behavior" C0 state used by the canonical LR-PPO-
# compatible evaluation (mirrors
# experiments.defender_adaptation_canonical_c0_evaluation.CANONICAL_EXTRA_OVERRIDES;
# a test asserts the two stay identical).
CANONICAL_C0_OVERRIDES: dict = {
    "enemy_obstacle_avoidance_enabled": False,
    "defender_obstacle_avoidance_enabled": False,
    "enemy_obstacle_awareness_mode": "disabled",
    "defender_obstacle_awareness_mode": "disabled",
}

# Finite mission endurance (journal extension; see docs/journal_v2_mission_endurance.md): the
# canonical corrected-V2 attacker now flies under a finite path budget (k=5, the largest of the
# tested k in {3, 4, 5} -- a 400-episode-per-factor C0-C3 sweep found every factor changes only
# the single known pathological episode (C3 seed 130017) and zero others, so the LEAST
# restrictive factor is preferred). V1 defaults are untouched: `build_v1_env_config` never
# applies this override, and `EnvConfig`'s own default (`enemy_endurance_enabled=False`) is
# unaffected for any caller that builds an `EnvConfig` directly.
V2_ENDURANCE_OVERRIDES: dict = {
    "enemy_endurance_enabled": True,
    "enemy_endurance_path_factor": 5.0,
}


def _build(
    level: str,
    estimator: str,
    semantic_fields: dict,
    *,
    awareness_mode: str,
    enemy_route_planning_enabled: bool,
    extra_overrides: dict | None,
) -> EnvConfig:
    overrides: dict = {}
    if level == "C0":
        overrides.update(CANONICAL_C0_OVERRIDES)
    overrides.update(semantic_fields)
    if extra_overrides:
        overrides.update(extra_overrides)
    return build_canonical_env_config(
        level,
        estimator,
        awareness_mode=awareness_mode,
        enemy_route_planning_enabled=enemy_route_planning_enabled,
        extra_overrides=overrides,
    )


def build_v2_env_config(
    level: str,
    estimator: str,
    *,
    awareness_mode: str = CANONICAL_AWARENESS_MODE,
    enemy_route_planning_enabled: bool = CANONICAL_ENEMY_ROUTE_PLANNING_ENABLED,
    extra_overrides: dict | None = None,
) -> EnvConfig:
    """Canonical V2 EnvConfig: swept mission contact + direct/obstacle-only A0, finite mission
    endurance (k=5; see V2_ENDURANCE_OVERRIDES). Pass
    `extra_overrides={"enemy_endurance_enabled": False}` to reproduce the pre-endurance V2
    config exactly (e.g. for endurance-vs-disabled comparisons)."""
    merged_overrides = dict(V2_ENDURANCE_OVERRIDES)
    if extra_overrides:
        merged_overrides.update(extra_overrides)
    cfg = _build(
        level, estimator, V2_SEMANTIC_FIELDS,
        awareness_mode=awareness_mode,
        enemy_route_planning_enabled=enemy_route_planning_enabled,
        extra_overrides=merged_overrides,
    )
    assert_v2_semantics(cfg)
    return cfg


def build_v1_env_config(
    level: str,
    estimator: str,
    *,
    awareness_mode: str = CANONICAL_AWARENESS_MODE,
    enemy_route_planning_enabled: bool = CANONICAL_ENEMY_ROUTE_PLANNING_ENABLED,
    extra_overrides: dict | None = None,
) -> EnvConfig:
    """Explicit V1 compatibility config (endpoint contact, legacy A0)."""
    cfg = _build(
        level, estimator, V1_SEMANTIC_FIELDS,
        awareness_mode=awareness_mode,
        enemy_route_planning_enabled=enemy_route_planning_enabled,
        extra_overrides=extra_overrides,
    )
    assert_v1_semantics(cfg)
    return cfg


def assert_v2_semantics(cfg: EnvConfig) -> None:
    for field, expected in V2_SEMANTIC_FIELDS.items():
        actual = getattr(cfg, field)
        if actual != expected:
            raise AssertionError(f"V2 config requires {field}={expected!r}, got {actual!r}")
    if not cfg.defender_standby_until_detection:
        raise AssertionError("V2 canonical config requires defender_standby_until_detection=True")


def assert_v1_semantics(cfg: EnvConfig) -> None:
    for field, expected in V1_SEMANTIC_FIELDS.items():
        actual = getattr(cfg, field)
        if actual != expected:
            raise AssertionError(f"V1 config requires {field}={expected!r}, got {actual!r}")


# ---------------------------------------------------------------------------
# Canonical V2 training contract (frozen A0 benchmark; docs/journal_v2_canonical_benchmark.md)
#
# CONFIG-ONLY validation: nothing here changes dynamics, guidance, contact semantics or
# endurance. Every future journal V2 A1 / defender training campaign must pass
# `assert_canonical_v2_training_config` (directly, or through the opt-in
# `require_canonical_v2=True` argument of the journal training runners) so it fails
# immediately instead of silently training on V1, on a config with endurance disabled, on the
# legacy waypoint avoidance, or on a different asset-motion model. A named counterfactual
# experiment must say so explicitly by not using this guard.
# ---------------------------------------------------------------------------
CANONICAL_V2_ENDURANCE_PATH_FACTOR = V2_ENDURANCE_OVERRIDES["enemy_endurance_path_factor"]
CANONICAL_V2_ASSET_MOTION_MODEL = "legacy_gaussian_random_walk"

# Attacker dynamics, asset speed, timestep and capture / interception radii frozen with A0 V2.
CANONICAL_V2_FROZEN_VALUES: dict = {
    "asset_motion_model": CANONICAL_V2_ASSET_MOTION_MODEL,
    "enemy_endurance_enabled": True,
    "enemy_endurance_path_factor": CANONICAL_V2_ENDURANCE_PATH_FACTOR,
    "v_e": 12.0,
    "enemy_max_accel": 6.0,
    "enemy_max_turn_rate_deg": 75.0,
    "enemy_max_climb_rate": 5.0,
    "enemy_max_descent_rate": 5.0,
    "v_s": 1.5,
    "dt": 0.5,
    "threat_radius": 2.0,
    "intercept_radius": 2.5,
    "unsafe_intercept_radius": 3.5,
}


class CanonicalV2ConfigError(AssertionError):
    """The environment is not the frozen canonical V2 A0 benchmark."""


def canonical_v2_config_violations(cfg: EnvConfig) -> list[str]:
    """Every way `cfg` (and the stateful-avoidance switch it will run under) departs from the
    canonical V2 benchmark. Empty list == canonical."""
    from uav_defend.envs.soldier_env import SoldierEnv

    problems: list[str] = []
    for field, expected in {**V2_SEMANTIC_FIELDS, **CANONICAL_V2_FROZEN_VALUES}.items():
        actual = getattr(cfg, field)
        if actual != expected:
            problems.append(f"{field}={actual!r} (canonical V2 requires {expected!r})")
    if not cfg.defender_standby_until_detection:
        problems.append("defender_standby_until_detection=False (canonical V2 requires True)")
    if cfg.enemy_route_planning_enabled != CANONICAL_ENEMY_ROUTE_PLANNING_ENABLED:
        problems.append(
            f"enemy_route_planning_enabled={cfg.enemy_route_planning_enabled!r} "
            f"(canonical requires {CANONICAL_ENEMY_ROUTE_PLANNING_ENABLED!r})"
        )
    if SoldierEnv.STATEFUL_ENEMY_AVOIDANCE is not True:
        problems.append("SoldierEnv.STATEFUL_ENEMY_AVOIDANCE is not True (legacy waypoint avoidance would run)")
    return problems


def assert_canonical_v2_training_config(cfg: EnvConfig) -> None:
    """Fail fast unless `cfg` is the frozen canonical V2 A0 environment. Configuration only."""
    problems = canonical_v2_config_violations(cfg)
    if problems:
        raise CanonicalV2ConfigError(
            "Not the canonical V2 A0 benchmark (journal-v2-a0-canonical-v1): " + "; ".join(problems)
        )


def canonical_v2_config_sha256(cfg: EnvConfig) -> str:
    """Fingerprint of the full resolved config (same recipe as the visualizer's replay provenance)."""
    import dataclasses
    import hashlib
    import json

    canonical = json.dumps(dataclasses.asdict(cfg), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def guard_canonical_v2_config_fn(config_fn):
    """Wrap any `rng -> EnvConfig` factory so every config it yields is asserted canonical."""

    def guarded(rng):
        cfg = config_fn(rng)
        assert_canonical_v2_training_config(cfg)
        return cfg

    return guarded


def build_randomized_v2_training_config(
    rng,
    estimator: str,
    *,
    awareness_mode: str = CANONICAL_AWARENESS_MODE,
    enemy_route_planning_enabled: bool = CANONICAL_ENEMY_ROUTE_PLANNING_ENABLED,
) -> EnvConfig:
    """The existing randomized C1-C3 training-clutter distribution with the canonical V2 semantics
    and endurance applied (only those fields differ from `build_randomized_training_config`)."""
    import dataclasses

    from experiments.journal_clutter_protocol import build_randomized_training_config

    cfg = dataclasses.replace(
        build_randomized_training_config(
            rng, estimator, awareness_mode=awareness_mode,
            enemy_route_planning_enabled=enemy_route_planning_enabled,
        ),
        **V2_SEMANTIC_FIELDS, **V2_ENDURANCE_OVERRIDES,
    )
    assert_canonical_v2_training_config(cfg)
    return cfg


_SOLDIER_ENV_PATHS = ("unwrapped", "env", "_env", "_inner", "_soldier_env")


def find_soldier_env(env, _depth: int = 0):
    """The base `SoldierEnv` inside any journal training wrapper chain (gym wrappers, the
    attacker training env, the nested LR-PPO composition)."""
    from uav_defend.envs.soldier_env import SoldierEnv

    if isinstance(env, SoldierEnv):
        return env
    if _depth > 6:
        return None
    for name in _SOLDIER_ENV_PATHS:
        child = getattr(env, name, None)
        if child is not None and child is not env:
            found = find_soldier_env(child, _depth + 1)
            if found is not None:
                return found
    return None


def assert_canonical_v2_env_fn(env_fn, seed: int) -> None:
    """Build ONE environment from a training `env_fn`, reset it (randomized-clutter wrappers
    re-sample their config on reset) and assert its resolved config is canonical V2. Runs a
    single reset; no stepping, no learning."""
    env = env_fn(seed)
    try:
        env.reset(seed=seed)
        soldier = find_soldier_env(env)
        if soldier is None:
            raise CanonicalV2ConfigError("Cannot locate the SoldierEnv inside the training environment to verify it.")
        assert_canonical_v2_training_config(soldier.config)
    finally:
        close = getattr(env, "close", None)
        if close is not None:
            close()
