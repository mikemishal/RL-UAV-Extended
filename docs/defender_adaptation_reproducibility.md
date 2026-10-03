# Defender-Adaptation (D1) Reproducibility Artifact

All paths below are repository-relative (never machine-specific absolute paths). Repository:
`RL-UAV-Extended-lrppo-clutter`, branch `research/lr-ppo-clutter`.

## Git commits / tags

| Stage | Commit SHA | Tag | Tag object SHA |
|---|---|---|---|
| Defender-adaptation training pipeline | `69596d8af8dc27bb080fa1eee37a38ea5090b6ef` | `journal-defender-adaptation-training-v1` | `952be4fd482f6f686122ff38f6d2782c0302ba7e` |
| Held-out evaluation harness | `1abd7c57ab7a78d43752cfb21d3e254e2af0a548` | `journal-defender-adaptation-eval-v1` | `f2fbe7f143a70a0345bd950af65bffa422f4a6e8` |
| Publication analysis code | `3d3e1597455c90b3424fa7837af57856b9bcd681` | `journal-defender-adaptation-results-v1` | `5b39777877dc085aecff98c093f0968e6c0eebc5` |

Every episode record in `results/defender_adaptation_evaluation/episodes.jsonl` carries an
`eval_code_commit` field stamped at write time via `git rev-parse HEAD`; every one of the 18
`training_manifest.json` files under `results/defender_adaptation/` carries a `git_commit` field
stamped the same way.

## Model provenance

- Frozen model registry (paths + SHA-256 for all 9 D0 + 18 D0+/D1-final + 18 D0+/D1-best + 3 A1 checkpoints):
  `results/defender_adaptation_model_registry.json`.
- D0 starting-checkpoint SHA-256 baseline (established before any continuation training):
  `results/clutter_training_400k/model_hashes.json`.
- A1 (Adaptive Attacker v1) checkpoint SHA-256 baseline:
  `results/adaptive_attacker_400k/model_hashes.json`.
- **Primary D0+/D1 checkpoint = `final_model.zip`** (NOT `best_model.zip`) — see module docstring of
  `experiments/defender_adaptation_evaluation.py` for the documented rationale (D0+ and D1 were validated
  during training against DIFFERENT opponent distributions, making their respective `best_model.zip`
  selections not directly comparable).

## Seeds

| Namespace | Seeds | Purpose |
|---|---|---|
| Defender training seeds | PPO-C: 71,72,73; PPO-Kalman-C: 74,75,76; LR-PPO-C: 77,78,79 | D0 original training; D0+/D1 continue from these exact checkpoints |
| Adaptive Attacker v1 training seeds | 201, 202, 203 | 201/202 = "A1-seen" (D1 training pool); 203 = "A1-unseen" (held out entirely from training/validation) |
| D1 in-training validation (shared with D0/D0+) | `90,000,000`..`90,000,019` (20 seeds) | periodic in-training checkpoint selection, same-distribution sanity eval |
| **Held-out evaluation seeds (this phase)** | **`90,010,000`..`90,010,099` (100 seeds)** | final N=100 matched held-out evaluation; disjoint from every other seed namespace in the repo (derivation documented in `experiments/defender_adaptation_evaluation.py` module docstring) |

## Actual transition counts (not approximated)

Every one of the 18 D0+/D1 continuation runs added exactly **425,984** additional environment
transitions (= one full rollout past the requested 400,000, given `num_envs=16`, `n_steps=2,048` →
rollout width 32,768; `ceil(400,000/32,768) × 32,768 = 425,984`, independent of each run's starting
timestep offset). Source: `actual_additional_timesteps` field in every
`results/defender_adaptation/*/*/seed_*/training_manifest.json`.

## Observation dimensions / architecture (unchanged, frozen)

- PPO-C / PPO-Kalman-C: 36-D observation (16-D base + 20-D obstacle-sector features).
- LR-PPO-C: 39-D observation (19-D Lead+residual base + 20-D obstacle-sector features).
- LR-PPO-C residual bound: 30° (`DEFAULT_RESIDUAL_MAX_ANGLE_DEG`), unchanged throughout training and
  evaluation.

## Environment configuration

- Canonical clutter protocol: `experiments/journal_clutter_protocol.py::build_canonical_env_config`
  (C1/C2/C3, Level-C "map_local" perception/awareness, reactive-only obstacle avoidance, route planning
  disabled) — unchanged throughout training and evaluation.
- Physical collision semantics: unchanged (`journal-physical-collision-semantics-v1`).
- Common-random-numbers protocol: for a fixed (clutter level, evaluation seed) pair,
  `SoldierEnv.reset(seed=eval_seed)` reproduces an identical spawn state / protected-asset walk /
  static-obstacle layout / dynamic-obstacle trajectories / measurement-noise stream regardless of which
  defender/attacker condition is evaluated, since `build_canonical_env_config(level, estimator)` is a
  pure deterministic function of its arguments (no RNG consumed at config-build time). Only the
  attacker's own tactical decisions (and the defender's resulting reactive trajectory) differ between
  conditions sharing the same (level, eval_seed).

## Result file locations

- Raw episode data: `results/defender_adaptation_evaluation/episodes.jsonl`,
  `results/defender_adaptation_evaluation/episodes.csv` (32,400 rows).
- Immutable evaluation manifest: `results/defender_adaptation_evaluation/evaluation_manifest.json`.
- Integrity/audit: `results/defender_adaptation_evaluation/analysis_input_integrity.json`,
  `results/defender_adaptation_evaluation/factorial_design_audit.csv`,
  `results/defender_adaptation_evaluation/outcome_integrity.json`.
- Statistics: `results/defender_adaptation_evaluation/statistics.json`.
- Supporting analysis CSVs: `defender_seed_summary.csv`, `robustness_scores.csv`,
  `safety_layer_dependence.csv`, `interception_efficiency.csv`, `lr_ppo_residual_analysis.csv`,
  `outcome_decomposition.csv` (all under `results/defender_adaptation_evaluation/`).
- Publication tables: `results/defender_adaptation_evaluation/tables/table{1,2,3,4}_*.csv`.
- Figures: `results/defender_adaptation_evaluation/figures/fig{1-10}_*.png` (fig9 = 5 matched-trajectory
  case files).

## Table/figure generation commands

```
python -m experiments.defender_adaptation_analysis_integrity   # Phase 1-3 integrity/audit
python -m experiments.defender_adaptation_tables                # Tables 1-4
python -m experiments.plot_defender_adaptation_figures           # Figures 1-8, 10
python -m experiments.defender_adaptation_trajectory_analysis     # Figure 9 (5 matched-trajectory cases)
```

Each command is idempotent and reads only from the frozen `episodes.csv` / model registry — none of
them execute any new episodes or modify any model.
