# Experiment Seed Registry

Full inventory of every seed range discovered in this repository's
experiment protocols (`experiments/*.py`), classified by purpose, plus the
newly reserved ranges for RA-LAPG development going forward. Built by
direct inspection of seed constants in the codebase (Phase 45 audit).

## Historical ranges (pre-existing, do not reuse)

| range | category | source |
|---|---|---|
| 42, 43, 44 | conference training | `training_protocol.py::TRAINING_SEEDS` |
| 10000-10099 | conference validation | `training_protocol.py::VALIDATION_SEED_OFFSET` |
| 12000-12199 | conference validation (diagnostic) | `training_protocol.py::DIAGNOSTIC_SEED_OFFSET` |
| 20000-24999 | conference evaluation (final nominal) | `final_experiment_protocol.py::FINAL_NOMINAL_SEED_OFFSET` |
| 25000-25999 | robustness evaluation (acquisition ablation) | `acquisition_ablation_protocol.py` |
| 30000-30999 | robustness evaluation (detection radius) | `robustness_detection_protocol.py` |
| 31000-31999 | robustness evaluation (measurement noise) | `robustness_measurement_noise_protocol.py` |
| 32000-32999 | robustness evaluation (1-D mobility) | `robustness_mobility_protocol.py` |
| 33000-33999 | robustness evaluation (hostile evasion) | `robustness_hostile_evasion_protocol.py` |
| 34000-34999 | robustness evaluation (maneuverability) | `robustness_maneuverability_protocol.py` |
| 35000-35499 | robustness evaluation (mobility grid) | `robustness_mobility_grid_protocol.py` |
| 40000-40099 | conference validation (post-detection) | `post_detection_training_protocol.py::VALIDATION_SEED_OFFSET` |
| 40100-40299 | conference validation/diagnostic (post-detection) | `post_detection_diagnostic_protocol.py` |
| 41000-45999 | conference evaluation (post-detection final test) | `post_detection_training_protocol.py::FINAL_TEST_SEED_START` |
| 60000-60099 | conference validation (lead-residual) | `lead_residual_training_protocol.py::RESIDUAL_VALIDATION_SEED_START` |
| 60100-60299 | conference validation/diagnostic (lead-residual) | `lead_residual_training_protocol.py::RESIDUAL_DIAGNOSTIC_SEED_START` |
| 60000-60199 | **learning-benefit pilot** (this project, Phases 0-22) | `experiments/learning_benefit/run_diagnostic_battery.py::PILOT_SEED_START` -- **overlaps the row above**; both were already executed/committed, neither is retroactively invalidated (Phase 45 policy: classify, don't delete). The learning-benefit pilot is explicitly documented as exploratory/pilot, not a pristine evaluation bank, so this pre-existing overlap does not corrupt any journal claim. |
| 61000-65999 | conference evaluation (lead-residual expanded final) | `lead_residual_training_protocol.py::EXPANDED_FINAL_SEED_START` |
| 66000-66999 | robustness evaluation (expanded hostile evasion) | `expanded_robustness_protocol.py::HOSTILE_EVASION_SEED_START` |
| 67000-67999 | robustness evaluation (expanded maneuverability) | `expanded_robustness_protocol.py::MANEUVERABILITY_SEED_START` |
| 68000-68999 | robustness evaluation (expanded measurement noise) | `expanded_robustness_protocol.py::MEASUREMENT_NOISE_SEED_START` |
| 69000-69999 | robustness evaluation (expanded mobility) | `expanded_robustness_protocol.py::MOBILITY_SEED_START` |
| 70000-70999 | robustness evaluation (expanded detection radius) | `expanded_robustness_protocol.py::DETECTION_RADIUS_SEED_START` |
| 72000-76999 | **conference evaluation** (RMPC final nominal) | `run_rmpc_final_nominal.py::FINAL_SEED_START` |
| 77000-77999 | robustness evaluation (RMPC targeted robustness) | `run_rmpc_targeted_robustness.py::ROBUSTNESS_SEED_START` |
| 71000-99999 | reserved/mostly-unused robustness headroom | `expanded_robustness_protocol.py::FUTURE_UNUSED_SEED_START` / `lead_residual_training_protocol.py::ROBUSTNESS_SEED_START..99_999` (broad declaration; only the concrete sub-ranges above are actually consumed within it) |

## FA-LAPG foundation predictor ranges (this project, Phases 27-28) -- CONFIRMED OVERLAP

| range | intended category | ACTUAL overlap found |
|---|---|---|
| 70000-70019 | predictor training | overlaps `70000-70999` (expanded detection-radius robustness evaluation) |
| 71000-71007 | predictor validation | falls in the broad reserved-but-unconsumed headroom; no concrete collision found |
| 72000-72007 | predictor engineering test | **overlaps `72000-76999`** (RMPC final-nominal conference evaluation) -- confirmed per Phase 45 |

Per Phase 45 policy: **these results are NOT deleted or invalidated.** They
are reclassified as **engineering / foundation validation** results (as
already labeled throughout `docs/fa_lapg_foundation_study.md`), not a
pristine, non-overlapping journal test bank. All CV/CA/GRU comparisons
reported from this data remain valid AS DEVELOPMENT EVIDENCE; they must not
be cited as a clean held-out test in the final journal submission.

## Newly reserved ranges (Phase 45, going forward -- all fully clear of every range above)

| range | category |
|---|---|
| 100000-100499 | RA-LAPG controller development (500 episodes) |
| 101000-101499 | RA-LAPG controller validation (500 episodes) |
| 102000-102199 | fresh predictor test bank (Phase 51, 200 episodes) |
| **110000-114999** | **RESERVED final journal evaluation (5000 episodes) -- MUST NOT be used during any development, tuning, or controller-ablation work** |

All four are verified programmatically disjoint from every historical range
and from each other -- see `experiments/fa_lapg_foundation/seed_registry.py`
and `tests/test_seed_registry.py`.
