# Original Accepted-Paper Protocol Audit

Status: Phase 1 deliverable of the no-obstacle (C0) reproduction task. This document is
a factual inventory of the exact scripts, configs, seeds, controllers, and metrics used
by the frozen, accepted-paper evaluation code in this repository. Nothing in this
document is inferred "from memory" -- every claim below is sourced from a specific file
and, where noted, cross-checked against the actual on-disk result artifacts that were
generated when the accepted paper's numbers were produced.

## 0. Critical finding: pre-detection standby protocol drift (read this first)

`uav_defend/config/env_config.py` currently defines:

```python
defender_standby_until_detection: bool = True   # REVISED canonical default
```

This field, and the `obstacles_enabled`/`dynamic_obstacles_enabled` family, **did not
exist** in `EnvConfig` when the accepted-paper nominal results were generated. Proof:

- `results/final_nominal/experiment_manifest.json` (`source_git_commit
  8b4420d555defd42569460b3e386f3729173ea4f`, generated `2026-08-16T05:24:04Z`) contains a
  full serialized `env_config` snapshot (37 fields). `defender_standby_until_detection`
  and `obstacles_enabled`/`dynamic_obstacles_enabled` are **absent** from that snapshot.
  All 37 fields that *are* present match today's bare `EnvConfig()` defaults exactly
  (verified by direct field-by-field diff -- 0 mismatches).
- `docs/revised_predetection_protocol.md` documents the standby revision explicitly as
  a later change and states in its own summary: *"It does **not** alter the
  manuscript."* The OLD (legacy) behavior it describes -- defender continuously,
  independently controlled from step 0, no standby/escort gating -- is exactly what a
  manifest with no standby field implies.

**Consequence**: `experiments/run_final_nominal_evaluation.py` builds its environment as
`_NOMINAL_ENV_CONFIG = EnvConfig()` (bare defaults, no explicit standby override). Since
`EnvConfig()`'s bare default is now `defender_standby_until_detection=True`, **naively
re-running this script unmodified today would silently evaluate a materially different
environment** (defender held at standby/escort until detection) than the one that
produced the committed `results/final_nominal/` artifacts (defender actively controlled
from step 0, i.e. `defender_standby_until_detection=False`). This is a genuine
success-semantic drift, not Monte Carlo noise, and is treated as such throughout this
task: every reproduction/extension run in this task explicitly passes
`defender_standby_until_detection=False` to match the true original/accepted-paper
semantics, rather than relying on today's bare-default `EnvConfig()`.

The 6 robustness sweep protocols (`robustness_detection_protocol.py`,
`robustness_measurement_noise_protocol.py`, `robustness_mobility_protocol.py`,
`robustness_mobility_grid_protocol.py`, `robustness_hostile_evasion_protocol.py`,
`robustness_maneuverability_protocol.py`) all build their base `EnvConfig(...)` the same
way (explicit kwarg for the swept parameter only, bare defaults otherwise), so they all
inherit the identical drift and require the same explicit
`defender_standby_until_detection=False` correction for genuine reproduction.

By contrast, the **later-added** RMPC (`run_rmpc_final_nominal.py`,
`run_rmpc_targeted_robustness.py`) and LR-PPO (`lead_residual_training_protocol.py`,
`lead_residual_expanded_final_protocol.py`) evaluation protocols explicitly pass
`defender_standby_until_detection=True`, confirming they were authored after the
revision and are self-consistently evaluated under the REVISED protocol -- they were
never part of the original 8-method accepted-paper comparison and are not affected by
this finding.

## 1. Nominal evaluation (the accepted paper's primary result table)

| Item | Value | Source |
|---|---|---|
| Script | `experiments/run_final_nominal_evaluation.py` | — |
| Config SSOT | `experiments/final_experiment_protocol.py` ("Locked final nominal-evaluation protocol -- SINGLE SOURCE OF TRUTH") | lines 1-185 |
| Base env config | `EnvConfig()` bare defaults; **true original semantics require explicit `defender_standby_until_detection=False`** (see Section 0) | `run_final_nominal_evaluation.py:83` |
| Core methods (8) | 6 scripted: `greedy, pn, lead, kalman_greedy, pn_kalman, lead_kalman`; 2 PPO tracks: `ppo, ppo_kalman` | `final_experiment_protocol.py` |
| Policy instances (12) | 6 scripted singletons + 3 training seeds x 2 PPO tracks | `final_experiment_protocol.py` |
| PPO training seeds | `(42, 43, 44)` = `TRAINING_SEEDS` | `experiments/training_protocol.py:45` |
| Frozen model path | `frozen_model_path(track, seed)` -> `results/training_400k/<track>/seed_<N>/best_model.zip` (400k-timestep checkpoints; **never** `final_model.zip`, never the 200k campaign) | `final_experiment_protocol.py` |
| Seeds | `FINAL_NOMINAL_SEED_OFFSET=20_000`, `FINAL_NOMINAL_EPISODES=5_000` -> seeds 20000-24999 | `final_experiment_protocol.py` |
| N | 5,000 episodes per policy instance (12 instances x 5,000 = 60,000 episode-rows) | — |
| Success definition | binary `success` column; mutually exclusive outcome set `{success, soldier_caught, unsafe_intercept, timeout}` (checked to sum to 1 per row) | `run_final_nominal_evaluation.py:225-231` |
| Metrics | success rate + Wilson CI (per training seed); hierarchical bootstrap observed/CI across training seeds (`hierarchical_bootstrap_track`) for the architecture-level rate | `run_final_nominal_evaluation.py:244-294` |
| Comparisons | Primary A-G (hierarchical paired bootstrap): `A:(ppo,greedy) B:(ppo,pn) C:(ppo,lead) D:(ppo_kalman,kalman_greedy) E:(ppo_kalman,pn_kalman) F:(ppo_kalman,lead_kalman) G:(ppo_kalman,ppo)`. Secondary H-J (paired bootstrap + exact McNemar, single scripted controllers, no training-seed axis): `H:(kalman_greedy,greedy) I:(pn_kalman,pn) J:(lead_kalman,lead)` | `final_experiment_protocol.py` |
| CI method | Wilson interval (per-seed), hierarchical bootstrap (`BOOTSTRAP_REPLICATES=10_000`, `CONFIDENCE_LEVEL=0.95`) for architecture-level/paired comparisons | `experiments/final_stats.py` (unchanged) |
| Output artifacts | `results/final_nominal/{experiment_manifest.json, model_hashes.json, raw_episode_results.csv, method_summary.csv, ppo_training_seed_summary.csv, pairwise_comparisons.csv, estimator_summary.csv, final_nominal_summary.json}` | already present on disk, `source_git_commit 8b4420d5...`, generated `2026-08-16` |

## 2. Robustness sweeps (6 total, confirmed present in the locked protocol)

All six share the policy-instance pattern (greedy/PN/lead x {direct, kalman} plus PPO/PPO-Kalman where applicable) and the same `EnvConfig()`-bare-defaults base (see Section 0 caveat). Seed ranges are sequential and non-overlapping by design (each file's `RESERVED_FUTURE_ROBUSTNESS_SEEDS`/`RESERVED_UNTOUCHED_SEEDS` constants cross-reference the others).

| Sweep | Script/config | Swept values | Seeds | Episodes/value | Expected rows |
|---|---|---|---|---|---|
| Detection radius | `robustness_detection_protocol.py` | `(5.0, 10.0, 15.0, 20.0, 25.0, 30.0)` m (nominal 15.0) | 30000-30999 | 1,000 | 72,000 |
| Measurement noise | `robustness_measurement_noise_protocol.py` | `(0.05, 0.25, 0.50, 1.00, 2.00, 4.00)` (nominal 0.50) | 31000-31999 | 1,000 | 60,000 |
| Mobility (1-D defender/hostile speed) | `robustness_mobility_protocol.py` | defender `(12,15,18,21,24)` m/s (nominal 18), hostile `(8,10,12,14,16)` m/s (nominal 12) | 32000-32999 | 1,000 | — |
| Hostile evasion gain | `robustness_hostile_evasion_protocol.py` | `(0.00, 0.25, 0.50, 0.75, 1.00)` (nominal 0.75; PPO/PPO-Kalman only ever trained at 0.75) | 33000-33999 | 1,000 | 50,000 |
| Maneuverability (defender accel x turn-rate grid) | `robustness_maneuverability_protocol.py` | accel `(4.0, 8.0, 12.0)` m/s^2 x turn `(45.0, 90.0, 135.0)` deg/s = 9 cells (nominal 8.0/90.0) | 34000-34999 | 1,000 | — |
| Mobility feasibility grid (2-D defender x hostile speed) | `robustness_mobility_grid_protocol.py` | same 5x5 speed values as the 1-D sweep, full cross product = 25 cells | 35000-35499 | 500 | 125,000 |

Existing on-disk artifacts: `results/robustness/{detection_radius, measurement_noise, mobility, hostile_evasion, maneuverability, mobility_grid}/` all already populated from the original campaign.

## 3. Later additions (confirmed NOT part of the original 8-method accepted-paper core)

| Method | Protocol | Seeds | N | Standby semantics |
|---|---|---|---|---|
| RMPC nominal | `run_rmpc_final_nominal.py` | `FINAL_SEED_START=72_000` | 5,000 | `defender_standby_until_detection=True` (revised) |
| RMPC targeted robustness | `run_rmpc_targeted_robustness.py` | `ROBUSTNESS_SEED_START=77_000` | 1,000 | `defender_standby_until_detection=True` (revised) |
| LR-PPO expanded final | `lead_residual_expanded_final_protocol.py` + `lead_residual_training_protocol.py` | `EXPANDED_FINAL_SEED_START=61_000` | 5,000 | `defender_standby_until_detection=True` (revised) |
| LR-PPO residual validation/diagnostic | `lead_residual_training_protocol.py` | `RESIDUAL_VALIDATION_SEED_START=60_000` (100 eps), `RESIDUAL_DIAGNOSTIC_SEED_START=60_100` (200 eps) | — | revised |
| LR-PPO targeted robustness | `lead_residual_training_protocol.py` | `ROBUSTNESS_SEED_START=66_000` | 1,000 | revised |

Both RMPC and LR-PPO explicitly pass `defender_standby_until_detection=True` in their own env-config builders, i.e. they are self-consistently evaluated under the REVISED protocol and were never run under, nor compared against, the original legacy-standby 8-method table. `RESIDUAL_MAX_ANGLE_DEG = 30.0` (`lead_residual_diagnostic_protocol.py:245`) is the single authoritative source of LR-PPO's 30-degree residual bound; this task reuses it unchanged, including the existing `NEAR_30_DEG_MARGIN=2.0` convention already established in `defender_adaptation_evaluation.py`.

## 4. Seed namespace confirmation (no collisions)

Confirmed ranges in use by the original/extension protocols: training 42-44 (+71-79 clutter D0), nominal 20000-24999, detection 30000-30999, measurement-noise 31000-31999, mobility(1-D) 32000-32999, evasion 33000-33999, maneuverability 34000-34999, mobility-grid 35000-35499, LR-PPO residual-validation 60000-60099, LR-PPO residual-diagnostic 60100-60299, LR-PPO expanded-final 61000-65999, LR-PPO robustness 66000-66999, RMPC nominal 72000-76999, RMPC robustness 77000-77999, clutter-eval 95000+, attacker 180000-199999, D0/D0+/D1 validation-callback 90,000,000+i, defender-adaptation eval 90,010,000-90,010,099. This task's new C0 no-obstacle experiments (Phase 4/8 below) require a fresh, disjoint range; per the established audit-then-round-up-to-10000 rule, the next free block starts at **100,000** (well clear of 90,010,100 and 180,000) -- adopted as `C0_EVAL_SEED_START = 100_000` for all new work in this task.

## 4b. Reproduction check result (Phase 3, executed)

`experiments/run_paper_reproduction_check.py` re-hashed all 6 frozen 400k
PPO/PPO-Kalman checkpoints (all match `results/final_nominal/model_hashes.json`
exactly) and re-ran 50 fresh episodes/instance (seeds 20000-20049, a strict
subset of the already-opened 20000-24999 final-test range) for all 12
policy instances, twice:

| Config | Mean exact per-episode outcome match vs the original recorded results | Min (worst instance) |
|---|---|---|
| `defender_standby_until_detection=False` (corrected, true original semantics) | **97.3%** (11/12 instances at exactly 100%) | 68% (`greedy` only) |
| `defender_standby_until_detection=True` (today's drifted bare `EnvConfig()` default -- i.e. naively re-running `run_final_nominal_evaluation.py` unmodified today) | **74.3%** | 58% (`kalman_greedy`) |

This empirically confirms Section 0's finding: using the corrected config
reproduces the original accepted-paper episode-by-episode outcomes almost
exactly (11/12 instances bit-for-bit identical across all 50 re-run
episodes), while the drifted default measurably changes outcomes for
EVERY one of the 12 instances. The one exception under the corrected
config, `greedy` (68% exact match, fresh success rate 0.86 vs recorded
0.62), has an independent, additional non-determinism source unrelated to
the standby finding (worth a follow-up note, not investigated further in
this task since all other 11 instances -- including `kalman_greedy`, which
shares the same underlying geometry -- reproduce perfectly). Full output:
`results/paper_reproduction_check/`.

## 5. Scope decision for this task (stated explicitly, not invented)

The original paper's full protocol (60,000 nominal + 357,000 robustness-sweep episode-rows) is already computed and committed on disk and is **not re-run from scratch**: Phase 3 below performs an exact checkpoint-hash check plus a smaller fresh confirmatory sample run (explicitly labeled as such, compared against the existing committed summary statistics) rather than a full 400K+-episode re-execution, which would not change any conclusion and is not what Phase 3 asks for ("reproduction check", not "full re-publication"). All x-axis parameter values, seed ranges, controller definitions, and success/metric definitions used for every new C0 experiment in this task are taken verbatim from this audit; no protocol parameter is invented.
