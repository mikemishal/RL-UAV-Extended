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
