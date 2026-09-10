"""
Central, programmatic seed registry (Phase 45).

Records every historical seed range discovered in this repository's
experiment protocols, plus newly reserved ranges for RA-LAPG development.
See `docs/experiment_seed_registry.md` for the full narrative and sources.

`assert_no_new_range_overlaps_reserved_final()` is the single guard that
MUST be called before any RA-LAPG development/evaluation run touches a
seed range, to guarantee the RESERVED final journal evaluation range is
never accidentally used during development.
"""

from __future__ import annotations

# --- Historical ranges (inclusive), name -> (start, end) ---------------
HISTORICAL_RANGES: dict[str, tuple[int, int]] = {
    "conference_training": (42, 44),
    "conference_validation": (10000, 10099),
    "conference_validation_diagnostic": (12000, 12199),
    "conference_evaluation_final_nominal": (20000, 24999),
    "robustness_acquisition_ablation": (25000, 25999),
    "robustness_detection_radius": (30000, 30999),
    "robustness_measurement_noise": (31000, 31999),
    "robustness_mobility_1d": (32000, 32999),
    "robustness_hostile_evasion": (33000, 33999),
    "robustness_maneuverability": (34000, 34999),
    "robustness_mobility_grid": (35000, 35499),
    "conference_validation_post_detection": (40000, 40099),
    "conference_validation_diagnostic_post_detection": (40100, 40299),
    "conference_evaluation_post_detection_final": (41000, 45999),
    "conference_validation_lead_residual": (60000, 60099),
    "conference_validation_diagnostic_lead_residual": (60100, 60299),
    "learning_benefit_pilot": (60000, 60199),  # KNOWN pre-existing overlap with the two rows above; documented, not corrected retroactively
    "conference_evaluation_lead_residual_expanded_final": (61000, 65999),
    "robustness_expanded_hostile_evasion": (66000, 66999),
    "robustness_expanded_maneuverability": (67000, 67999),
    "robustness_expanded_measurement_noise": (68000, 68999),
    "robustness_expanded_mobility": (69000, 69999),
    "robustness_expanded_detection_radius": (70000, 70999),
    "conference_evaluation_rmpc_final_nominal": (72000, 76999),
    "robustness_rmpc_targeted": (77000, 77999),
    "reserved_robustness_headroom": (71000, 99999),  # broad declaration; overlaps the concrete rows above by construction
}

# --- This project's own already-executed ranges (Phases 0-28) ----------
PROJECT_RANGES: dict[str, tuple[int, int]] = {
    "predictor_training": (70000, 70019),  # CONFIRMED overlap with robustness_expanded_detection_radius
    "predictor_validation": (71000, 71007),  # no concrete collision found
    "predictor_engineering_test": (72000, 72007),  # CONFIRMED overlap with conference_evaluation_rmpc_final_nominal
}

# Ranges known (Phase 45) to genuinely overlap a historical range. These
# are NOT deleted/invalidated -- only reclassified as engineering/
# foundation-validation results, per Phase 45 policy.
CONFIRMED_OVERLAPPING_PROJECT_RANGES = {"predictor_training", "predictor_engineering_test"}

# --- Newly reserved ranges (Phase 45), verified clear of everything above
RESERVED_RANGES: dict[str, tuple[int, int]] = {
    "ra_lapg_development": (100000, 100499),
    "ra_lapg_validation": (101000, 101499),
    "fresh_predictor_test_bank": (102000, 102199),
    "final_journal_evaluation": (110000, 114999),
}


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def assert_no_new_range_overlaps_reserved_final() -> None:
    """Guard: none of the newly reserved DEVELOPMENT ranges may ever
    overlap the RESERVED final journal evaluation range."""
    final_range = RESERVED_RANGES["final_journal_evaluation"]
    for name, rng in RESERVED_RANGES.items():
        if name == "final_journal_evaluation":
            continue
        if _overlaps(rng, final_range):
            raise AssertionError(f"Reserved range {name}={rng} overlaps final_journal_evaluation={final_range}")


def assert_reserved_ranges_disjoint_from_historical() -> None:
    """Guard: every newly reserved range (Phase 45) must be fully clear of
    every historical range this repository has ever used."""
    for reserved_name, reserved_rng in RESERVED_RANGES.items():
        for hist_name, hist_rng in HISTORICAL_RANGES.items():
            if _overlaps(reserved_rng, hist_rng):
                raise AssertionError(
                    f"Reserved range {reserved_name}={reserved_rng} overlaps historical range {hist_name}={hist_rng}"
                )


def assert_reserved_ranges_pairwise_disjoint() -> None:
    names = list(RESERVED_RANGES)
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            if _overlaps(RESERVED_RANGES[a_name], RESERVED_RANGES[b_name]):
                raise AssertionError(f"Reserved ranges {a_name} and {b_name} overlap")


def find_overlaps(range_to_check: tuple[int, int]) -> list[str]:
    """Return the names of every historical range overlapping
    `range_to_check` (used to classify/audit a candidate seed range before
    using it, per Phase 45)."""
    return [name for name, rng in HISTORICAL_RANGES.items() if _overlaps(range_to_check, rng)]
