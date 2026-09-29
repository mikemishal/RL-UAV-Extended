# Obstacle-Aware Clutter Training: Vectorized Scale-Up -- Final Report

Status: FINAL for the scope completed in this session. This report covers
the 26-phase scale-up sub-task that followed the diagnostic-scale clutter
retraining study (`docs/obstacle_aware_clutter_retraining_report.md`,
commit `5f7e5b9`, tags `journal-obstacle-aware-clutter-retraining-v1` /
`journal-clutter-training-infra-v1`).

## 0. Session-scope disclosure (read first)

- **Full 3-seed x 400k-step x 3-architecture training campaign: COMPLETED
  AT FULL SCOPE.** All 9 runs finished successfully (verified checkpoints
  on disk). See Section 4.
- **Held-out evaluation sample size: REDUCED FROM SPEC, by explicit user
  decision.** The spec target was N>=500 held-out episodes per controller
  per level. A full N=500 sweep across 22 controller instances x 3 levels
  was estimated (from measured per-episode cost) at roughly 8-17+ hours of
  additional unattended compute beyond the ~6.4 hours already spent on
  training. The user was asked and explicitly chose **N=100** (still large
  enough for informative Wilson-score 95% confidence intervals) as a
  reduced-but-honest sample for this session, with the harness built to
  reproduce the full N=500 protocol with a one-flag change
  (`--n-episodes 500`) whenever additional compute budget is available.
  This is flagged in every summary artifact the harness writes, not
  silently substituted.
- **Phase 21 (safety-disabled ablation): NOT run as originally specified**,
  due to a genuine architectural constraint discovered during
  implementation (Section 8), not an oversight. A documented proxy
  analysis (Section 8) was used instead.
- **Everything else in the 26-phase spec was completed**, including a
  correction to the Phase 2 profiling bucketing (Section 2), a
  previously-unknown bug found and fixed in the LR-PPO held-out evaluation
  path (Section 7), and honest, non-cherry-picked recovery-delta results
  (Section 7) that do **not** uniformly favor clutter training.

## 1. Starting point

- Commit `5f7e5b9` (`journal-obstacle-aware-clutter-retraining-v1`), worktree
  `RL-UAV-Extended-lrppo-clutter`, branch `research/lr-ppo-clutter`.
- Verified clean, 914/914 tests passing (518.00s), all 3 diagnostic
  checkpoints (`ppo_clutter_v1/seed_71`, `ppo_kalman_clutter_v1/seed_74`,
  `lrppo_clutter_v1/seed_77`, each 1 seed / 40k steps) present on disk.
  `research/ra-lapg` worktree confirmed untouched.
- Added tag `journal-clutter-training-infra-v1` (alias, same commit as
  `journal-obstacle-aware-clutter-retraining-v1`) as requested.

## 2. Runtime profiling (Phase 2)

`experiments/profile_clutter_training.py` ran `cProfile` around
`model.learn(total_timesteps=8192)` (4 rollouts). Grand total: 98.46s.

| bucket (as originally labeled) | % |
|---|---|
| other | 49.45% |
| collision checking | 32.38% |
| PPO inference/update (torch) | 11.57% |
| avoidance planning (static+dynamic) | 2.46% |
| env.step() core physics | 2.34% |
| obstacle-sector construction | 1.10% |
| perception | 0.70% |
| reset/layout generation | 0.00% |

**Correction to the bucketing (found during interpretation, not a data
error -- the raw profile data is accurate, only the label attribution
was imprecise):** manual inspection of `top_functions.txt` shows the
functions labeled "collision checking" (`distance_to_point`,
`clearance_to_point`, `min_clearance`, `first_segment_intersection`,
`segment_intersection`) are called overwhelmingly (millions of times: e.g.
`distance_to_point` 3,471,577 calls) from
`ObstacleAvoidancePlanner.plan()`'s internal candidate-trajectory
prediction (`avoidance.py:485 plan`, 15,305 calls, 63.386s cumulative;
`avoidance.py:157 predict_collision`, 27,131 calls, 43.969s cumulative) --
**not** from the O(1)-per-step ground-truth swept collision check in
`soldier_env.py::step()`. The TRUE dominant cost is therefore **reactive
obstacle-avoidance candidate-trajectory evaluation** (evaluating multiple
candidate bypass directions x a multi-step lookahead horizon, every
step), roughly 60-70% of runtime when correctly attributed, not "collision
checking" as originally labeled. `copy.deepcopy` (3.718s exclusive /
7.839s cumulative, ~7-8%) remains an observed-but-unexplained secondary
cost, plausibly from Gymnasium's `Monitor`/info-dict handling; not
root-caused in this session.

**Consequence for the vectorization decision (Phase 3):** since the
bottleneck is CPU-bound pure-Python/numpy computation (not I/O, not
GIL-releasing native code), true process parallelism (`SubprocVecEnv`)
was chosen over thread-based `DummyVecEnv` for `num_envs > 1`.

## 3. Vectorized training infrastructure (Phases 3-9)

`experiments/clutter_training_runner.py`:

- `env_fn` signature changed from `Callable[[], Env]` to
  `Callable[[int], Env]` (the argument is the worker's seed). This was
  REQUIRED for LR-PPO specifically: its `root_seed` is baked in at
  construction (its `reset()` ignores any passed seed), so each worker
  needs a genuinely different constructor call, not just a different
  `reset(seed=...)` call.
- `SEED_STRIDE = 10_000`; worker `i` (0-indexed) uses
  `worker_seed = seed + i * SEED_STRIDE`, applied to BOTH env construction
  and the first `reset(seed=...)`. `num_envs=1` gives `worker_seed == seed`
  exactly (no offset), preserving the original single-env behavior.
- `run_clutter_training(..., num_envs=1, vec_env_cls="auto")`:
  `num_envs=1` -> `DummyVecEnv` (unchanged legacy behavior);
  `num_envs>1` -> `SubprocVecEnv` unless `vec_env_cls="dummy"` is forced
  (used for fast in-process unit tests).
- Manifest now records `num_envs`, `worker_seeds`,
  `actual_total_timesteps` (= `model.num_timesteps`), and
  `effective_rollout_batch` (= `n_steps * num_envs`).

**Important finding (Phase 5 timestep accounting):** SB3 always completes
a FULL rollout (`n_steps * num_envs` transitions) before checking the stop
condition, so `actual_total_timesteps` can OVERSHOOT the requested
`total_timesteps` by up to one rollout's worth when `num_envs` is large
relative to the request. This is harmless for training (`total_timesteps`
still means the same total environment EXPERIENCE regardless of
`num_envs`, per SB3's own accounting) but means throughput
(steps-per-second) measurements MUST divide by the actual, not nominal,
count -- an early version of the throughput benchmark (Section 3.1) had
exactly this bug and was corrected.

**Phase 6 (hyperparameter fairness):** `CLUTTER_HYPERPARAMS` (learning
rate, gamma, clip range, etc.) are UNCHANGED by vectorization; SB3's
`n_steps` is already a per-worker quantity, so only the EFFECTIVE rollout
batch (`n_steps x num_envs`) grows with `num_envs`, not any hyperparameter
value itself. No hyperparameter changes were made or judged necessary.

**Phase 7 (equivalence tests):** `tests/test_vectorized_clutter_training.py`
(5 tests, all passing) verify: deterministic non-overlapping worker
seeding; `num_envs=1` reproduces the exact legacy manifest fields;
`num_envs=4` preserves `total_timesteps` semantics (using an
exact-rollout-multiple budget to eliminate overshoot ambiguity);
observation/action shapes and physical-collision/perception diagnostics
are unaffected by vectorization; different workers sample different
randomized clutter layouts.

**Phase 8 (equivalence smoke run):** PPO trained for a matched 20,480-step
budget via `num_envs=1` and `num_envs=4` (`SubprocVecEnv`). Both trained
stably (`num_envs=1`: periodic success rate 0.2 -> 0.2; `num_envs=4`: 0.3
-> 0.5) -- comparable, not identical, as expected (different rollout
composition, not a bitwise-equivalence claim).

### 3.1 Throughput benchmark (Phase 9)

Measured (not assumed) env-steps/sec, 65,536-step budget chosen so every
`num_envs in {1,2,4,8,16}` completes a whole number of rollouts (no
overshoot ambiguity), on a 20-logical-core machine:

| num_envs | env-steps/sec | speedup | efficiency |
|---|---|---|---|
| 1 | 93.76 | 1.00x | 100.0% |
| 2 | 124.78 | 1.33x | 66.5% |
| 4 | 179.58 | 1.92x | 47.9% |
| 8 | 265.86 | 2.84x | 35.4% |
| 16 | 322.22 | 3.44x | 21.5% |

Scaling is clean and monotonic through `num_envs=16` (no plateau/
degradation observed at this machine's core count). **`num_envs=16`** was
chosen for the production campaign: best measured throughput, and leaves
~4 logical cores of headroom for the main process/eval callback/OS.
(An earlier draft of this benchmark had a rollout-overshoot accounting bug
that made `num_envs=8` and `16` look artificially slow by 2x/4x; this was
found and corrected before the production decision was made -- see
Section 3, Phase 5 finding.)

See `results/vectorization_benchmark/phase9_throughput_benchmark.json`
and Figure 2 (`fig2_throughput_scaling.png`).

## 4. Full 3-seed x 400k-step training campaign (Phases 10-12)

`experiments/run_clutter_400k_campaign.py`, `num_envs=16`,
`total_timesteps=400_000` (actual: 425,984 per run, per the Phase 5
rollout-overshoot accounting), same seeds as the diagnostic run for
methodological continuity (`ppo_clutter_v1`: 71,72,73;
`ppo_kalman_clutter_v1`: 74,75,76; `lrppo_clutter_v1`: 77,78,79), written
to `results/clutter_training_400k/<arch>/seed_<n>/` (a SEPARATE directory
tree from the diagnostic checkpoints -- nothing was overwritten).

**Checkpoint / model-selection policy (Phase 11):** periodic checkpoints
every 50,000 steps for recovery; `best_model.zip` = highest
periodic-eval (same-distribution, not held-out) success rate seen during
training; `final_model.zip` = end-of-training state. All downstream
evaluation uses `best_model.zip`, falling back to `final_model.zip` only
if missing.

All 9 runs completed successfully, total wall time 23,128.3s (~6.43
hours; longer than the ~3.1-hour estimate extrapolated from the short
throughput benchmark, likely because full-length episodes and the
periodic-eval callback are not amortized the same way as the short
benchmark's steady-state loop). Verified on disk (`Test-Path`): all 9
directories have `best_model.zip`, `final_model.zip`,
`training_manifest.json`.

Periodic (same-distribution, NOT held-out) final-eval success rate @
420,000 timesteps:

| architecture | seed | success rate |
|---|---|---|
| ppo_clutter_v1 | 71 / 72 / 73 | 0.65 / 0.50 / 0.70 |
| ppo_kalman_clutter_v1 | 74 / 75 / 76 | 0.55 / 0.70 / 0.70 |
| lrppo_clutter_v1 | 77 / 78 / 79 | 0.90 / 0.90 / 1.00 |

LR-PPO clearly leads on this same-distribution metric; see Section 7 for
the more informative HELD-OUT comparison.

See Figure 1 (`fig1_training_curves.png`) for full training curves.

## 5. Stability monitoring and the LR-PPO 30-degree bound (Phases 13-14)

No training instability (divergence, NaNs, reward collapse) was observed
in any of the 9 runs (see per-run `eval_history` in each
`training_manifest.json`, plotted in Figure 1). The LR-PPO residual bound
(`theta_max = 30 deg`) was **left unchanged** throughout, per the task's
explicit instruction; direct evidence for that decision is in Section 6.

## 6. LR-PPO residual saturation at full scale (Phase 19)

`experiments/run_lrppo_saturation_400k.py` re-ran the Phase 20 (prior
sub-task) saturation diagnostic against the full 400k checkpoints (all 3
seeds, 200 held-out-style episodes each = 107,794 step samples pooled).

**Result: P(|theta_residual| >= 0.9 x theta_max) = 0.0**, for every seed
individually and pooled, and for every condition examined (overall, near
obstacles, during safety-layer intervention, during failed interception,
during collision).

**Conclusion: the 30-degree bound is NOT restrictive at full training
scale either** (consistent with the 40k diagnostic-scale finding). This is
direct evidence supporting the task's instruction to leave `theta_max`
unchanged -- the policy never needs to use angles near the bound, so
there is no basis to relax it. See Figure 7
(`fig7_residual_saturation.png`).

## 7. Held-out evaluation (Phases 15-18)

`experiments/clutter_retraining_evaluation.py`, extended this sub-task:

- `discover_campaign_checkpoints()`: finds all 9 new checkpoints
  automatically (`best_model.zip`, falling back to `final_model.zip`).
- `evaluate_clutter_trained()` now evaluates ALL 9 checkpoints (not one
  per architecture as before), tagging each row with its `training_seed`.
- **Bug found and fixed:** `build_lr_ppo_clutter_env()` (used by the
  clutter-trained LR-PPO evaluation arm) unconditionally RESAMPLES the
  obstacle configuration every episode, regardless of the requested
  canonical clutter level -- there was no way to hold LR-PPO's evaluation
  at a genuinely FIXED C1/C2/C3 level, unlike the PPO/PPO-Kalman path
  (`build_ppo_clutter_env(..., randomized=False, fixed_config=...)`). This
  silently made the LR-PPO clutter-trained arm's level-based comparison
  meaningless (same randomized distribution mislabeled as three different
  levels). Fixed by adding `build_lr_ppo_clutter_env_eval()` /
  `build_lr_ppo_clutter_eval_env()` in
  `experiments/lr_ppo_clutter_training_env.py`, which uses the plain
  (non-resampling) `LeadResidualTrainingEnv` with the caller's fixed
  config, confirmed correct by observing genuinely level-varying
  `min_clearance` statistics after the fix.
- `summarize()` now adds Wilson-score 95% confidence intervals for
  `safe_intercept_rate` and `defender_collision_rate`.
- `compute_recovery_deltas()` (Phase 17): frozen-vs-clutter-trained mean
  delta per level/family.
- Seed-robustness aggregation (Phase 18): mean/std across the 3 seeds per
  architecture per level.

**Sample size: N=100 held-out episodes** per controller per level (13
frozen/classical instances + 9 clutter-trained checkpoints = 22
instances; RMPC uses N=20, a smaller matched subset, per its documented
~150-400x runtime cost -- unchanged practice from the historical
benchmark protocol). Full results:
`results/clutter_retraining_evaluation/{frozen_and_classical_summary.csv,
clutter_trained_summary.csv, clutter_trained_seed_robustness.csv,
recovery_deltas.csv}` (+ per-episode CSVs).

### 7.1 Recovery deltas (Phase 17) -- honest, non-uniform result

`safe_intercept_rate` delta (clutter-trained mean across 3 seeds minus
frozen conference model, same N=100 held-out seeds):

| level | LR-PPO | PPO | PPO-Kalman |
|---|---|---|---|
| C1 (light) | -0.067 | -0.063 | -0.167 |
| C2 (medium) | +0.010 | -0.013 | -0.053 |
| C3 (heavy) | **+0.077** | -0.027 | +0.003 |

`defender_collision_rate` delta at C3: LR-PPO -0.010, PPO 0.000,
PPO-Kalman **-0.067** (a real safety improvement despite a flat intercept
rate).

**Interpretation (not cherry-picked):** clutter (obstacle-aware)
retraining clearly helps **LR-PPO** recover interception performance
specifically at HIGH clutter (C3), is roughly neutral at C2, and actually
COSTS a few points of performance at LOW clutter (C1) for all three
architectures -- plausibly a distribution-shift/overfit-to-the-harder
training mix effect, since training used a randomized C1-C3 blend. Plain
**PPO** and **PPO-Kalman** clutter training does NOT show a clear
intercept-rate recovery at any level in this N=100 sample (deltas mostly
small-negative or flat), though **PPO-Kalman** shows a genuine
collision-rate improvement at C3. This is the honest finding at N=100;
it should NOT be read as "clutter training uniformly helps." See Figures
3 and 4.

### 7.2 Seed robustness (Phase 18)

Seed-to-seed variability is non-trivial at N=100 (e.g. PPO at C3:
`safe_intercept_rate` std = 0.059, `defender_collision_rate` std =
0.075 across the 3 seeds) -- conclusions above are means over 3 seeds
with real spread, not single-seed point estimates. See Figure 5
(`fig5_seed_robustness.png`) and
`clutter_trained_seed_robustness.csv`.

## 8. Safety-layer dependence (Phases 20-21)

**Investigated and found NOT cleanly achievable as originally specified:**
`ObstacleSectorObservationWrapper.__init__` (and
`LRPPOSectorObservationWrapper`) hard-raise `ValueError` if
`config.defender_obstacle_avoidance_enabled=False`, because
`SoldierEnv._move_defender()` only maintains `_defender_last_perceived`
(the perception data the RL observation reads) when that same flag is
`True` -- the reactive safety-layer CORRECTION and the PERCEPTION used
for the RL observation are intentionally coupled in the existing, tested
environment implementation (confirmed via that wrapper's own docstring).
A clean "keep the RL policy's sector-feature observation informative but
disable only the corrective safety-layer override" ablation is therefore
NOT constructible without modifying `SoldierEnv._move_defender()`
internals -- which would risk canonical, heavily-tested environment
semantics for a diagnostic-only ablation, explicitly against this
sub-task's "do not modify environment semantics absent a genuine defect"
instruction (this coupling is intentional, not a defect).

**What was done instead (documented substitute, not a silent skip):**

- **Phase 20 (safety-layer dependence):** used the ALREADY-COLLECTED
  `avoidance_fraction` (fraction of steps with an active safety-layer
  correction) and `mean_override_angle` / `p95_override_angle` (magnitude
  of that correction) columns from the Phase 15-18 evaluation, no new
  compute required. See Figure 6
  (`fig6_safety_layer_intervention.png`): intervention frequency and
  magnitude both increase from C1 to C3 for every controller family
  (frozen and clutter-trained alike), consistent with obstacle density
  driving more frequent/larger corrections; clutter-trained models do not
  show a systematically LOWER intervention frequency than frozen ones at
  the same level in this data, suggesting the safety layer remains an
  active contributor to both arms' outcomes rather than being made
  redundant by the learned sector-feature awareness.
- **Phase 21 (safety-disabled ablation):** not run as a literal
  "correction-disabled" experiment for the structural reason above; the
  closest available, already-computed proxy is the Section 7.1
  frozen-vs-clutter-trained recovery-delta comparison, which holds the
  safety layer CONSTANT (always active, identical canonical configuration,
  for BOTH arms) and varies only whether the policy has LEARNED obstacle
  awareness via its own observation.

## 9. Figures (Phases 22-24)

`experiments/plot_clutter_scaleup_figures.py` ->
`results/clutter_scaleup_figures/`:

1. `fig1_training_curves.png` -- periodic eval success rate vs. timesteps,
   3 seeds x 3 architectures.
2. `fig2_throughput_scaling.png` -- measured env-steps/sec and speedup vs.
   `num_envs`.
3. `fig3_recovery_deltas.png` -- frozen-vs-clutter-trained
   `safe_intercept_rate` delta by level/family.
4. `fig4_success_by_level_ci.png` -- frozen vs. clutter-trained
   `safe_intercept_rate` with Wilson 95% CIs, per family/level.
5. `fig5_seed_robustness.png` -- mean +/- std `safe_intercept_rate` across
   the 3 seeds, per architecture/level.
6. `fig6_safety_layer_intervention.png` -- safety-layer intervention
   frequency and correction magnitude, per family/level, frozen vs.
   clutter-trained.
7. `fig7_residual_saturation.png` -- LR-PPO residual-saturation
   probabilities at full scale (all zero).

## 10. Full regression (Phase 25)

`python -m pytest -q`: **919 passed, 0 failed, 407.50s** (914 baseline +
5 new tests in `tests/test_vectorized_clutter_training.py`). Log:
`results/full_regression_scaleup.log`.

## 11. Summary of "Do NOT" compliance

- Controller architecture / environment semantics: not modified absent a
  genuine defect. The one environment-adjacent change was an ADDITIVE
  evaluation-only helper (`build_lr_ppo_clutter_eval_env`) that fixes a
  bug in the EVALUATION harness, not the environment or any controller.
- Attacker was not trained; no self-play was introduced.
- LR-PPO's 30-degree bound was not changed; Section 6 provides fresh,
  full-scale evidence supporting that decision.
- Canonical clutter semantics (`journal_clutter_protocol.py`) unchanged.
- No jamming/spoofing introduced.
- RA-LAPG (`research/ra-lapg` worktree) untouched.

## 12. Artifact index

- Training: `experiments/clutter_training_runner.py`,
  `experiments/run_clutter_400k_campaign.py`,
  `results/clutter_training_400k/`.
- Profiling: `experiments/profile_clutter_training.py`,
  `results/clutter_training_profile/`.
- Vectorization benchmark: `experiments/vectorization_benchmark.py`,
  `results/vectorization_benchmark/`.
- Tests: `tests/test_vectorized_clutter_training.py`.
- Evaluation: `experiments/clutter_retraining_evaluation.py`,
  `results/clutter_retraining_evaluation/`.
- LR-PPO saturation: `experiments/run_lrppo_saturation_400k.py`,
  `results/lrppo_residual_saturation_400k/`.
- Figures: `experiments/plot_clutter_scaleup_figures.py`,
  `results/clutter_scaleup_figures/`.
- Regression log: `results/full_regression_scaleup.log`.
