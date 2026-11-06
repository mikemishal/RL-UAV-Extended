"""Read-only precomputed-trajectory review path (journal_cbf_attacker, defender_cbf / Prompt 4 manual review).

A SEPARATE, additive replay path for ALREADY-RECORDED, controller-neutral trajectory JSON files produced by the
research repository's `experiments.journal_cbf_attacker_defender_visualizer_export` script
(docs/journal_cbf_attacker/defender_cbf/visualizer_artifacts/*.json in the pinned research worktree). This module
does NOT reconstruct the CBF controller, does NOT re-run any episode, and does NOT touch the existing strict
`ReplayBundle` / `CaseLibrary` / SB3 replay-job machinery at all -- those contracts have MANDATORY fields (e.g.
`PolicyVisible.observation`, a 16-float vector) that a CBF-only trajectory cannot honestly provide without
inventing data. Every value served here is either read verbatim from the recorded JSON or a clearly-labeled,
deterministic derivation from recorded positions (e.g. speed from consecutive positions); nothing is fabricated,
and fields absent from a given file are simply omitted (returned as `null`), never defaulted to a plausible-looking
value.

Research worktree is read-only to this module -- only `*.json` files already committed by the research repository
are read; nothing is written back.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from .config import Settings

logger = logging.getLogger(__name__)

COLLECTION_ID = "a1-cbf-defender-review"
COLLECTION_LABEL = "A1-CBF Defender Review \u2014 DEVELOPMENT \u2014 NOT FROZEN"
COLLECTION_DESCRIPTION = (
    "Manual-review collection for the A1-CBF defender-avoidance stage (journal_cbf_attacker, defender_cbf / "
    "Prompt 4). The controller is NOT frozen: the primary defender-avoidance effect gate and the 200-case joint "
    "stress set's zero-collision requirement did not both pass (see the research repository's "
    "docs/journal_cbf_attacker/defender_cbf/README.md). These are precomputed, controller-neutral trajectory "
    "recordings for human visual inspection only -- the browser never re-runs or reconstructs the controller."
)
# Research-relative directory containing the exported trajectory JSONs (read-only).
RESEARCH_RELATIVE_DIR = "docs/journal_cbf_attacker/defender_cbf/visualizer_artifacts"
# Research commit this review collection was generated against (informational only -- this module does not
# enforce the pin the way the SB3 replay engine does, since precomputed JSON has no executable dependency on the
# research Python environment).
RESEARCH_PIN_COMMIT = "9f2a103650767d57c611da733bfbe3f3737f54c6"

DEFENDER_SAFETY_RADIUS_M = 3.5
DEFENDER_INTERCEPT_RADIUS_M = 2.5
STEP_DT_S = 0.5

INTERCEPTION_OUTCOMES = {"intercepted", "unsafe_intercept"}
OBSTACLE_COLLISION_OUTCOMES = {"enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"}

# Manual-review priority ordering (brief section 4). Each entry documents HONESTLY whether the requested category
# has an exact match among the 10 exported files, or whether the closest available substitute is shown instead --
# never silently relabeling a file as something it is not. See docs/journal_cbf_attacker/defender_cbf/ (research
# repo) for the underlying Monte Carlo campaigns these files were drawn from.
CASE_ORDER: tuple[tuple[str, int, str, str, str], ...] = (
    # (filename stem, rank, requested_category, match_quality, rationale)
    (
        "06_defender_interception_failure", 1,
        "Greedy \u2014 successful defender evasion / asset capture", "substitute",
        "No Greedy asset-capture case exists among the 10 exported files (Greedy's full-CBF capture rate was "
        "only 4.9% in the primary 1,000-episode campaign). This organic (non-stress) Greedy full-CBF encounter "
        "is the best available representative case; it ends in interception, not capture.",
    ),
    (
        "08_hardest_successful_stress_case", 2,
        "Greedy \u2014 defender interception failure", "exact",
        "Greedy, full A1-CBF, outcome=intercepted. Also the closest available 'stress case with strong joint "
        "constraints' (see rank 8's cross-reference) -- it was selected in the original export by joint-constraint "
        "fraction among the 200-case stress set.",
    ),
    (
        "09_obstacle_only_matched_reference", 3,
        "Matched Greedy obstacle-only reference", "substitute",
        "No Greedy matched obstacle-only/full pair was exported. This RMPC obstacle-only reference is the only "
        "matched reference pair among the 10 files; its full-CBF counterpart (same seed) is rank 8 below.",
    ),
    (
        "05_joint_obstacle_defender_maneuver", 4,
        "Joint obstacle + defender CBF maneuver", "exact",
        "Deterministically constructed (attacker <10 m from an inflated obstacle AND <25 m from a closing "
        "defender, simultaneously, by construction) \u2014 see build_stress_scenario in the research repository.",
    ),
    (
        "02_lateral_pass", 5,
        "PN case showing large CBF benefit", "exact",
        "PN showed a -33 percentage-point interception reduction in the generalization diagnostic (95% -> 62%), "
        "the second-largest effect found. This is a PN full-CBF successful lateral-evasion/capture case.",
    ),
    (
        "01_representative_defender_evasion", 6,
        "RMPC case showing large CBF benefit", "exact",
        "RMPC showed the LARGEST interception reduction in the generalization diagnostic (75% -> 37%, -38 "
        "percentage points). This is a representative RMPC full-CBF successful evasion/capture case.",
    ),
    (
        "03_vertical_climb_avoidance", 7,
        "Lead case", "substitute",
        "No Lead-defender case exists among the 10 exported files (the Lead generalization diagnostic showed only "
        "a small -2 pp effect and was not separately recorded as a trajectory). This additional PN case (vertical/"
        "climb evasion) is shown instead.",
    ),
    (
        "10_full_cbf_matched_case", 8,
        "Stress case with strong joint constraints", "substitute",
        "Not a stress-constructed scenario -- shown here as the full-CBF counterpart (same seed) of rank 3's "
        "matched obstacle-only reference, for direct side-by-side comparison. Rank 2's file is the closest true "
        "'strong joint constraint' stress case among the 10 exported files.",
    ),
    (
        "04_speed_reduction_evasion", 9,
        "Stress case associated with shared-slack conflict", "substitute",
        "No exported file represents the shared-slack-conflict obstacle-collision failure mode: that failure "
        "mode was identified only in the aggregate 200-case stress-set statistics (3/200 episodes, "
        "docs/journal_cbf_attacker/defender_cbf/stress_summary.json in the research repository) and none of those "
        "specific episodes was individually exported as a trajectory JSON. This RMPC speed-reduction evasion is "
        "shown instead as the remaining distinct file.",
    ),
    (
        "07_asset_capture_after_evasion", 10,
        "Remaining representative case", "exact",
        "PN full-CBF case where sustained defender-CBF activity precedes asset capture.",
    ),
)
CASE_RANK = {stem: rank for stem, rank, *_ in CASE_ORDER}
CASE_META = {stem: (category, quality, rationale) for stem, _rank, category, quality, rationale in CASE_ORDER}
# section 9's explicit opening-case rule: "Greedy + full A1-CBF + successful asset capture; if none exists, use
# the best available Greedy full-CBF case." No Greedy capture case exists among the 10 files (see rank 1 above),
# so the opening case is the best-available Greedy full-CBF encounter -- the SAME file as rank 1.
OPENING_CASE_STEM = "06_defender_interception_failure"


def _sanitize(value: Any) -> Any:
    """Recursively replaces non-finite floats (Infinity/-Infinity/NaN) with `None`. The research export script
    writes `float('inf')` for diagnostics not yet computed at the very first recorded frame (e.g.
    `min_inflated_obstacle_clearance` before the first CBF solve) -- Python's `json` module serializes these as
    the bare tokens `Infinity`/`-Infinity`/`NaN`, which are NOT valid JSON per the spec and would make the
    browser's `JSON.parse` throw. Treating 'not yet computed' as `null` (never a fabricated number) is the same
    convention the research repository's own strict Pydantic contracts use (`allow_inf_nan=False`)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    return value


def _distance(a: list[float] | None, b: list[float] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def _speed(prev: list[float] | None, cur: list[float] | None, dt_s: float) -> float | None:
    if prev is None or cur is None or dt_s <= 0:
        return None
    return _distance(prev, cur) / dt_s


def _augment_steps(steps: list[dict]) -> list[dict]:
    """Adds clearly-labeled DERIVED playback fields (brief section 6) computed only from recorded positions:
    attacker speed, attacker-asset distance, attacker-defender distance. Never overwrites a recorded field; never
    fabricates a value when the inputs are themselves missing."""
    augmented = []
    prev_attacker_pos = None
    for step in steps:
        attacker_pos = step.get("attacker_pos")
        defender_pos = step.get("defender_pos")
        asset_pos = step.get("asset_pos")
        intervention = step.get("intervention_source")
        enriched = dict(step)
        enriched["attacker_speed_mps_derived"] = _speed(prev_attacker_pos, attacker_pos, STEP_DT_S)
        enriched["attacker_asset_distance_m_derived"] = _distance(attacker_pos, asset_pos)
        enriched["attacker_defender_distance_m_derived"] = _distance(attacker_pos, defender_pos)
        enriched["obstacle_cbf_active_derived"] = (intervention in ("obstacle", "joint")) if intervention is not None else None
        enriched["defender_cbf_active_derived"] = (intervention in ("defender", "joint")) if intervention is not None else None
        augmented.append(enriched)
        prev_attacker_pos = attacker_pos
    return augmented


def _summarize(stem: str, raw: dict) -> dict:
    """Builds the brief-section-3 case summary fields. Every value is either read directly from the recorded JSON
    or computed via an explicit, documented formula over recorded positions -- `min_defender_separation_m` and
    `min_obstacle_clearance_m` are aggregated across the SAME per-step fields the player itself uses."""
    steps = raw.get("steps") or []
    outcome = raw.get("outcome")
    seps = [
        d for d in (_distance(s.get("attacker_pos"), s.get("defender_pos")) for s in steps) if d is not None
    ]
    clearances = [
        c for c in (s.get("min_inflated_obstacle_clearance") for s in steps)
        if isinstance(c, (int, float)) and math.isfinite(c)
    ]
    last_t = steps[-1]["t"] if steps and "t" in steps[-1] else None
    category, match_quality, rationale = CASE_META.get(stem, (None, None, None))
    return {
        "case_id": stem,
        "rank": CASE_RANK.get(stem),
        "requested_category": category,
        "category_match": match_quality,
        "category_rationale": rationale,
        "case_name": stem,
        "attacker_variant": "A1-CBF-full" if raw.get("defender_cbf_enabled") else "A1-CBF-obstacle-only",
        "defender_method": raw.get("defender"),
        "seed": raw.get("seed"),
        "outcome": outcome,
        "asset_captured": (outcome == "soldier_caught") if outcome is not None else None,
        "defender_intercepted": (outcome in INTERCEPTION_OUTCOMES) if outcome is not None else None,
        "obstacle_collision": (outcome in OBSTACLE_COLLISION_OUTCOMES) if outcome is not None else None,
        "min_defender_separation_m": min(seps) if seps else None,
        "min_obstacle_clearance_m": min(clearances) if clearances else None,
        "episode_duration_s": last_t,
        "step_count": len(steps),
        "what_to_look_for": raw.get("what_to_look_for"),
    }


class PrecomputedCaseProvider:
    """Loads and caches the A1-CBF defender-review collection from the pinned research worktree. Read-only;
    reloads from disk on each `collection()`/`case()` call (the files are small and static for this
    development-review collection, so no cache-invalidation complexity is needed)."""

    def __init__(self, settings: Settings):
        self._settings = settings

    def _directory(self) -> Path | None:
        root = self._settings.research_data_root
        if root is None:
            return None
        candidate = root / RESEARCH_RELATIVE_DIR
        return candidate if candidate.is_dir() else None

    def _load_all(self) -> tuple[dict[str, dict], list[str]]:
        """Returns `(stem -> sanitized raw dict, warnings)`. A malformed file is SKIPPED (never crashes the whole
        collection) and reported as a warning string; the collection remains usable for every other valid file."""
        directory = self._directory()
        cases: dict[str, dict] = {}
        warnings: list[str] = []
        if directory is None:
            return cases, warnings
        for path in sorted(directory.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                warnings.append(f"{path.name}: could not be parsed as JSON ({exc.__class__.__name__}); skipped.")
                continue
            if not isinstance(raw, dict) or "steps" not in raw or not isinstance(raw.get("steps"), list):
                warnings.append(f"{path.name}: missing required 'steps' array; skipped.")
                continue
            cases[path.stem] = _sanitize(raw)
        return cases, warnings

    def collection(self) -> dict:
        cases, warnings = self._load_all()
        summaries = []
        for stem, raw in cases.items():
            summary = _summarize(stem, raw)
            summaries.append(summary)
        # Known cases first (brief-requested priority order), then any unrecognized files (defensive -- the
        # collection is meant to be exactly the 10 files documented in CASE_ORDER, but an unexpected extra file
        # must not disappear silently).
        summaries.sort(key=lambda s: (s["rank"] is None, s["rank"] if s["rank"] is not None else 0, s["case_id"]))
        for extra in cases.keys() - set(CASE_RANK):
            warnings.append(f"{extra}.json: not in the documented review priority order (CASE_ORDER); listed last.")
        return {
            "collection_id": COLLECTION_ID,
            "label": COLLECTION_LABEL,
            "description": COLLECTION_DESCRIPTION,
            "research_pin_commit": RESEARCH_PIN_COMMIT,
            "research_relative_dir": RESEARCH_RELATIVE_DIR,
            "opening_case_id": OPENING_CASE_STEM if OPENING_CASE_STEM in cases else (summaries[0]["case_id"] if summaries else None),
            "defender_safety_radius_m": DEFENDER_SAFETY_RADIUS_M,
            "defender_intercept_radius_m": DEFENDER_INTERCEPT_RADIUS_M,
            "step_dt_s": STEP_DT_S,
            "cases": summaries,
            "warnings": warnings,
        }

    def case(self, case_id: str) -> dict | None:
        cases, _warnings = self._load_all()
        raw = cases.get(case_id)
        if raw is None:
            return None
        summary = _summarize(case_id, raw)
        return {
            **summary,
            "obstacles": raw.get("obstacles") or [],
            "steps": _augment_steps(raw.get("steps") or []),
        }
