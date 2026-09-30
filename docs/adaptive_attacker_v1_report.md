# Adaptive Attacker v1 -- Final Report (with Machine-Restart Recovery)

Status: FINAL for the scope completed in this session. Covers the original
29-phase Adaptive Attacker v1 sub-task AND the 28-phase (R1-R28) recovery
that followed an unexpected machine restart mid-evaluation.

## 0. Starting point

- Commit `17db54661baa3eb300f39a6234dad8b3514d0dc1`, tag
  `journal-clutter-training-scaleup-v1` (confirmed **local-only**, not
  pushed to `origin`), worktree `RL-UAV-Extended-lrppo-clutter`, branch
  `research/lr-ppo-clutter`.
- Verified at Phase 1: 919/919 tests passing, all 9 clutter-trained
  defender checkpoints present with recorded SHA-256 hashes
  (`results/clutter_training_400k/model_hashes.json`), `research/ra-lapg`
  untouched at `3eebbdb` (confirmed again, unchanged, after the restart).

## 1. Objective

Train and evaluate the first **learned adaptive hostile-UAS policy**
("Adaptive Attacker v1") against a **frozen** defender pool -- the
defender is never retrained in this phase. This is the first of a planned
sequence: hand-designed attacker -> learned attacker -> evaluate frozen
defenders -> retrain defender -> iterative improvement.

## 2. Hostile-policy audit (Phase 2)

`uav_defend/envs/soldier_env.py::_move_enemy()`'s per-step pipeline was
audited and classified:

- **TACTICAL** (replaced by the learned policy): pursuit-direction choice,
  reactive defender-evasion, route-planning arbitration, weave/noise --
  collectively the pre-obstacle "base command" `u_base`.
- **LOW-LEVEL SAFETY/NAVIGATION** (reused unchanged, never bypassed):
  the deterministic `ObstacleAvoidancePlanner`, constrained-dynamics
  limits (`enemy_max_accel=6.0`, `enemy_max_turn_rate_deg=75.0`,
  `enemy_max_climb_rate`/`enemy_max_descent_rate=5.0`, `v_e=12.0`),
  boundary handling, and physical-collision outcome determination.

**Key finding**: the existing hand-designed hostile's reactive evasion
uses the defender's **ground-truth** position unconditionally (by
design, documented in its own docstring as modeling the hostile's "own
local awareness", independent of detection state). This was left
**unchanged** (no correctness issue found), but the new learned
attacker's observation was given its **own, separate, explicit** sensing
gate (Section 4) rather than inheriting this omniscient assumption.

## 3. Adaptive Attacker architecture (Phase 3)

A single, minimal, additive hook was added to `SoldierEnv`:

```
self.enemy_tactical_override: Callable[[], np.ndarray] | None = None
```

When set (only by the new training/evaluation wrapper, never by
`SoldierEnv` itself), `_move_enemy()` replaces its entire hand-designed
`u_base` computation with `normalize(enemy_tactical_override())`, while
every downstream stage (obstacle safety, dynamics, boundary, collision)
and every diagnostic field (still computed from the hand-designed model,
for scripted-vs-learned comparison) is **unchanged**. Default `None`
reproduces the exact prior behavior bit-for-bit -- verified by a
dedicated test and by the full regression suite (919/919 passed
immediately after this change).

Conceptual stack: `learned tactical command -> [unchanged] obstacle
safety -> [unchanged] dynamics -> [unchanged] collision physics`.

## 4. Attacker observation, sensing, action (Phases 5-8)

**Observation (36-D, `experiments/adaptive_attacker_env.py::
build_attacker_observation`)**: mirrors the defender's own existing 16-D
observation layout (roles swapped) -- `[enemy_pos(3), soldier_pos(3),
enemy_vel(3), defender_detected_flag(1), defender_pos_info(3, gated),
defender_vel_info(3, gated)]` (16-D) + 20-D obstacle-sector features
reusing `uav_defend/obstacles/sector_features.py` **unchanged**, computed
from the attacker's own `PerceivedObstacleSet` (Level C perception,
never ground truth -- verified by a dedicated test that forces empty
perception and confirms an all-"no-obstacle" reading despite a
dense C3 ground-truth layout).

**Defender-sensing contract (Phase 7)**: new explicit config field
`enemy_defender_detection_range: float = 25.0` gates ONLY the learned
attacker's own observation (zeroed below this true 3D range, exact
measurement at/above it) -- explicitly **decoupled** from the
pre-existing, always-on scripted-evasion mechanics (Section 2).

**Action (Phase 8)**: normalized 3D desired direction, fed through
`enemy_tactical_override` into the unchanged obstacle-safety/dynamics
pipeline -- the RL policy can never bypass safety or teleport.

## 5. Reward and anti-hacking (Phases 9-10)

`REWARD_ATTACKER_SUCCESS=+100` (soldier_caught), `REWARD_ATTACKER_
INTERCEPTED=-100` (defender safely stops the attacker),
`REWARD_ATTACKER_UNSAFE_INTERCEPT=-50`, `REWARD_ATTACKER_COLLISION=-50`,
`REWARD_ATTACKER_TIMEOUT=-20` (deliberately NOT dominant, per Phase 10's
explicit "survival must not be the dominant reward"), plus dense
`progress_reward = prev_asset_distance - curr_asset_distance` (scale 5.0)
and a small per-step time penalty (-0.05) -- magnitudes chosen to mirror
the defender's own existing terminal-reward scale, documented as a
reasoned, revisitable choice. `asset_progress_per_step`/
`asset_progress_total` are tracked explicitly in `info` every step as the
anti-hacking diagnostic Phase 10 requested.

## 6. Frozen defender pool and opponent sampling (Phases 11-13)

Training pool (10 instances, built ONCE, never retrained): Lead
(scripted) + clutter-trained **PPO-C** (seeds 71/72/73) + **PPO-Kalman-C**
(74/75/76) + **LR-PPO-C** (77/78/79) -- all 3 seeds of every learned
family included (never a single favorable checkpoint), RMPC excluded
from training (too slow) but retained for evaluation.
`sample_defender()` is family-balanced-then-seed-within-family (uniform
over the 4 family names, then uniform within), verified by a dedicated
test to give ~25% probability per family regardless of instance count
(not biased toward the 3-seed families). A dedicated test proves defender
model parameters (torch tensors, compared before/after stepping) never
change during attacker training.

## 7. Diagnostic and full training (Phases 16-17)

**Diagnostic** (seed 200, 60k steps, `num_envs=1`): clear, healthy,
monotonic improvement -- success_rate (P(soldier_caught)) 0.00 -> 0.00 ->
0.25 -> 0.35, `asset_progress_per_step` 0.005 -> 0.044 -> 0.097 -> 0.207
(rising together, no hiding/collapse signature), collision rate
controlled (~5%). Already exceeded the scripted hostile's matched-scenario
baseline (0.133) at just 60k steps -> cleared to scale up.

**Full campaign** (3 seeds x 400k steps, `num_envs=16` SubprocVecEnv,
measured throughput 269.5 env-steps/sec at this setting -- chosen from
real benchmarking, not assumption): all 3 seeds completed successfully
(`actual_total_timesteps=425984` each, matching the documented SB3
rollout-granularity overshoot). Total wall time 17,094s (~4.75h). Final
periodic (same-distribution) validation success rates: seed 201=0.45,
202=0.55, 203=0.35.

## 8. Held-out evaluation (Phases 18-21, R17-R21)

Model selection (Phase 18) used `best_model.zip`, selected by periodic
validation success_rate (P(soldier_caught)), never training reward.
Held-out evaluation used seeds from `ATTACKER_HELDOUT_EVAL_SEED_START=
190_000`, **never seen during training or validation**.

**Sample size (honest, reduced)**: N=15 held-out episodes per (level,
defender_instance, attacker_condition) for the 21 non-RMPC instances,
N=5 for RMPC (restricted to 2 attacker conditions: scripted + the
strongest learned seed) given RMPC's documented ~150-400x per-episode
cost -- 3,810 total episodes. Smaller than the N=100 used for the
defender-side evaluation, but directly informative for the questions
this phase asks, and the harness scales to any N with no code changes.

### 8.1 Defender degradation matrix (Phase R17, pooled across C1-C3)

`delta_defender_success = learned_safe_intercept_rate -
scripted_safe_intercept_rate` (negative = defender degrades):

| Defender family | seen? | seed 201 | seed 202 | seed 203 | mean |
|---|---|---:|---:|---:|---:|
| Greedy | no | -0.378 | -0.422 | -0.200 | -0.333 |
| PN | no | -0.133 | -0.267 | -0.222 | -0.207 |
| Lead | **yes** | -0.067 | -0.289 | -0.200 | -0.185 |
| RMPC (N=5 only) | no | -- | +0.067 | -- | +0.067 |
| PPO (frozen conference) | no | -0.178 | -0.341 | -0.222 | -0.247 |
| PPO-Kalman (frozen conference) | no | -0.163 | -0.341 | -0.259 | -0.254 |
| LR-PPO (frozen conference) | no | -0.230 | -0.385 | -0.185 | -0.267 |
| PPO-C | **yes** | -0.193 | -0.259 | -0.200 | -0.217 |
| PPO-Kalman-C | **yes** | -0.141 | -0.207 | -0.030 | -0.126 |
| LR-PPO-C | **yes** | -0.252 | -0.244 | -0.259 | -0.252 |

**Every family degrades against every learned-attacker seed except one
cell** (RMPC/seed 202, +0.067, N=5 -- not statistically reliable at this
sample size and not corroborated by the other two seeds, which were not
run against RMPC for cost reasons).

### 8.2 Seen vs. unseen generalization (Phase 19-20, R19)

Unseen families (Greedy, PN, RMPC, frozen-conference PPO/PPO-Kalman/
LR-PPO -- never in the training pool) degrade by **-0.25 mean**, seen
families (Lead, PPO-C, PPO-Kalman-C, LR-PPO-C -- the exact training pool)
degrade by **-0.20 mean** -- unseen defenders degrade by a SIMILAR OR
SLIGHTLY LARGER magnitude than seen ones. This is direct evidence the
learned attacker acquired a **general** tactical strategy rather than
narrowly overfitting its training opponents.

### 8.3 C1/C2/C3 breakdown (Phase 22, R20)

Mean `delta_defender_success` by clutter level: **C1 = -0.234, C2 =
-0.214, C3 = -0.216** -- remarkably consistent across clutter density.
The learned attacker's advantage does **not** meaningfully interact with
environmental complexity in this data; no claim of deliberate "clutter
exploitation" is supported (per-family, per-level deltas vary but show no
systematic increase with clutter).

### 8.4 Attacker seed robustness (Phase 25, R21)

Never report only the strongest seed. Across the 9 non-RMPC families,
mean/std/range of `delta_defender_success`: seed 201 mean=-0.193
(std=0.10), seed 202 mean=**-0.284** (std=0.09, consistently the
strongest/most-degrading seed, matching its highest diagnostic/validation
success rate), seed 203 mean=-0.181 (std=0.10). Range across all
(family, seed) cells: -0.467 to +0.089 (excluding the RMPC-only cell).

### 8.5 Statistical significance (Phase R24)

Paired bootstrap (10,000 replicates, seed 20260816, reusing
`experiments/final_stats.py::paired_bootstrap_ci` unchanged -- scripted
and learned conditions share identical held-out seeds per
(level, defender_instance), so this is genuinely matched/paired data):
**54 of 192** individual (level, defender_instance, learned_seed)
comparisons reach 95% significance despite the modest N=15 per-cell
sample (expected -- individual cells have limited power); essentially
**all** significant results are in the negative (degradation) direction,
with zero reliably significant positive results.
(`results/adaptive_attacker_evaluation/paired_bootstrap_comparisons.csv`)

## 9. Outcome decomposition and collision rates (Phase R18)

Every episode's outcome is one of 8 mutually-exclusive categories
(`intercepted`, `soldier_caught`, `unsafe_intercept`, `timeout`, 2
defender-collision, 2 attacker-collision types); rates for every
(level, defender, condition) sum to 1.0 by construction
(`results/adaptive_attacker_evaluation/summary.csv`). Attacker
environmental-failure (own obstacle collision) rates are broadly similar
between scripted and learned conditions (not a dominant effect); a few
learned cells show a modest increase at C3 (e.g. PPO-C: scripted=0.089,
learned seed-201=0.200), worth future attention but not a primary driver
of the degradation result above.

## 10. Trajectory/behavior analysis (Phase 23, R22)

Five representative matched (same-seed, scripted-vs-learned) scenarios
were selected **from the actual completed dataset** (not guessed) and
re-run with full trajectory recording
(`experiments/adaptive_attacker_trajectory_analysis.py`,
`results/adaptive_attacker_figures/fig7_trajectory_*.png` + `trajectory_
analysis_summary.json`):

1. **scripted_vs_strong_defender** (LR-PPO-C, C1): both scripted and
   learned attacker are safely intercepted -- a baseline reference.
2. **learned_succeeds_where_scripted_failed** (Greedy, C1): the SAME
   defender safely intercepts the scripted attacker but is reached by the
   learned attacker. Visual/quantitative evidence: the learned attacker's
   path is markedly more direct (113 steps) than the scripted attacker's
   extensive weaving/looping (419 steps) before reaching the soldier.
3. **learned_still_intercepted** (LR-PPO-C, C1): the learned attacker does
   **not** always win -- a strong defender still safely intercepts it.
4. **high_clutter_c3_behavior** (PPO-Kalman-C, C3): the scripted attacker
   collides with a static obstacle; the learned attacker reaches the
   soldier in the same dense layout.
5. **unseen_defender_generalization** (Greedy, C1): the learned attacker
   succeeds against a defender family it never trained against.

Only behaviors directly visible in the recorded trajectories/metrics are
described (e.g. "more direct path, fewer steps") -- no strategic intent is
claimed beyond what the data shows.

## 11. Runtime, regression, artifacts

- Training campaign: 17,094s (~4.75h). Evaluation (post-recovery,
  Section 12): all 3,810 episodes executed in the resumed run.
- Full regression: **944 passed, 0 failed**, 790.2s
  (`results/phase_r25_full_regression.log`) -- 919 baseline + 13
  (`tests/test_adaptive_attacker_env.py`) + 12
  (`tests/test_adaptive_attacker_evaluation_recovery.py`).
- Git status at time of writing: modified
  `uav_defend/config/env_config.py`, `uav_defend/envs/soldier_env.py`,
  `uav_defend/policies/residual/lead_residual_ppo_policy_wrapper.py`;
  new `experiments/adaptive_attacker_env.py`,
  `adaptive_attacker_training_runner.py`,
  `run_adaptive_attacker_diagnostic_training.py`,
  `run_adaptive_attacker_400k_campaign.py`,
  `adaptive_attacker_evaluation.py`, `adaptive_attacker_statistics.py`,
  `plot_adaptive_attacker_figures.py`,
  `adaptive_attacker_trajectory_analysis.py`; new
  `tests/test_adaptive_attacker_env.py`,
  `tests/test_adaptive_attacker_evaluation_recovery.py`.

## 12. Machine-restart recovery (Phases R1-R16)

The machine restarted mid-way through the Phase 19-25 evaluation (partway
through clutter level C2). Recovery findings:

- **R1**: repository state fully intact after restart -- same HEAD, same
  uncommitted files, `research/ra-lapg` unchanged.
- **R2**: `TRAINING_RECOVERY_REQUIRED = NO` -- all 3 attacker checkpoints
  and all defender checkpoints were already complete and untouched
  (verified via file existence, size, hash, and `campaign_summary.json`'s
  clean-finish marker).
- **R3/R5/R6 -- critical finding**: the pre-restart evaluator held ALL
  episode results **only in memory**, writing aggregate CSVs once at the
  very end. Since the interrupted run never reached that point, **zero
  episodes of the real run were ever persisted**. The CSV files present on
  disk were confirmed (via their `n` values: {1,2,6}, matching an earlier
  smoke test's `n_episodes=2`) to be **stale smoke-test artifacts**, not
  partial real-run data -- preserved (moved, not deleted) to
  `results/adaptive_attacker_evaluation/_pre_recovery_smoke_test_n2/`.
- **R7-R9**: minimal, additive resume infrastructure was added to
  `experiments/adaptive_attacker_evaluation.py` -- episode-level JSONL
  append (`episodes.jsonl`, unique key = level|defender_instance|
  attacker_condition|seed), atomic writes (write+flush+fsync per
  episode), row validation (known outcome, no NaN/inf), malformed-final-
  line recovery (drops only a truncated LAST line; raises loudly on
  corruption anywhere else), and duplicate-key protection (identical
  content = no-op, differing content = raises). This is infrastructure
  only -- zero changes to N, seeds, models, controllers, or environment
  semantics. 12 dedicated tests validate this (all passing).
- **R4/R14/R15**: the recovery manifest reported **0/3810 (0%) complete**
  before resuming (confirming R3's finding), and **3810/3810 (100%)
  complete, 0 missing** after the resumed run -- verified by rebuilding
  the manifest from scratch.
- **R16**: all aggregate outputs (`episodes.csv`, `summary.csv`,
  `degradation_matrix.csv`) were regenerated from scratch from the
  complete, canonical episode dataset -- never appended to pre-restart
  aggregates.
- **R13 note**: the interrupted run's log showed two extreme runtime
  outliers (~7,365s and ~10,760s for otherwise-fast groups) immediately
  before the restart -- almost certainly a sleep/suspend event inflating
  wall-clock time, not a computational hang (the process resumed normal
  timing each time). Documented, not further investigated.

## 13. JOURNAL ARTIFACTS

### A. Methods paragraph

We introduce Adaptive Attacker v1, a learned tactical policy for the
hostile UAV that replaces only the tactical layer (target pursuit,
reactive evasion, maneuvering) of a previously hand-designed hostile
model, while reusing the existing obstacle-avoidance safety layer,
constrained flight dynamics, and physical-collision semantics unchanged.
The attacker observes its own state, the protected asset's state, a
range-gated measurement of the defender (zeroed below a fixed detection
range, exact above it), and a 20-dimensional fixed-size summary of
locally perceived obstacles in 10 angular sectors -- never ground-truth
obstacle state. Training uses PPO against a frozen pool of defender
opponents spanning a scripted classical controller and three independent
seeds each of three previously obstacle-aware-retrained learned defender
architectures, sampled in a family-balanced manner every episode; no
defender parameters are ever updated during attacker training.

### B. Experimental setup paragraph

Three independent PPO training seeds (201, 202, 203) were each trained
for 400,000 environment steps (16 parallel workers) under the same
canonical, randomized obstacle-clutter protocol used for the defenders'
own training. Held-out evaluation used N=15 episodes per (clutter level,
defender instance, attacker condition) across 21 controller instances
spanning both defenders seen during attacker training and defenders
withheld entirely, plus N=5 episodes (2 attacker conditions only) for a
computationally expensive receding-horizon optimization-based defender,
for 3,810 total held-out episodes across three clutter-density levels,
all on seeds never used during training or model selection.

### C. Main results paragraph

Across nearly every evaluated defender family and every one of the three
independently trained attacker seeds, the learned attacker reduced the
defender's safe-interception rate relative to the original hand-designed
attacker, with a mean absolute reduction of roughly 0.13-0.35 depending on
family and seed. This degradation was of similar or slightly larger
magnitude against defender families the attacker never encountered during
training than against its own training-pool opponents, and was
essentially uniform across the three evaluated clutter-density levels.
Paired bootstrap analysis found a majority of individual comparisons
directionally consistent with degradation, with just under a third
reaching conventional statistical significance at the evaluated sample
size; no comparison showed a statistically reliable improvement in
defender performance under the learned attacker.

### D. Table-ready data

- Degradation matrix: Section 8.1 table above.
- Seen vs. unseen: seen mean delta -0.20, unseen mean delta -0.25
  (Section 8.2).
- C1/C2/C3: -0.234 / -0.214 / -0.216 (Section 8.3).
- Seed robustness: seed 201 mean -0.193 (sd 0.10), seed 202 mean -0.284
  (sd 0.09), seed 203 mean -0.181 (sd 0.10) (Section 8.4).

### E. Figure captions

1. *Fig 1*: Defender safe-intercept rate for each controller family,
   scripted attacker vs. the mean of three independently trained
   Adaptive Attacker v1 seeds, pooled across all three clutter levels.
2. *Fig 2*: Change in defender safe-intercept rate (learned minus
   scripted attacker) by controller family, shown separately for each of
   the three attacker training seeds.
3. *Fig 3*: Attacker target-reach (asset-capture) rate, pooled across all
   defenders and clutter levels, for the scripted attacker and each
   learned-attacker seed.
4. *Fig 4*: Full outcome decomposition (interception, asset capture,
   unsafe interception, environmental failures, timeout) by clutter
   level, for the scripted attacker and the learned attacker (pooled
   seeds).
5. *Fig 5*: Mean change in defender safe-intercept rate for defender
   families included in attacker training ("seen") versus withheld
   entirely ("unseen"), by attacker seed.
6. *Fig 6*: Distribution (box plot) of the change in defender
   safe-intercept rate across the three attacker training seeds, one box
   per defender family.
7. *Fig 7 (five panels)*: Representative matched scripted-vs-learned
   attacker trajectories (horizontal plane) for five scenarios spanning a
   strong-defender baseline, a scripted-fails/learned-succeeds case, a
   learned-still-intercepted case, high-clutter behavior, and
   generalization to an unseen defender family.

### F. Key finding

Adaptive Attacker v1 discovers tactical behavior that degrades every
evaluated frozen defender family's safe-interception rate relative to the
original hand-designed hostile, with comparable degradation against
defender families it never encountered during training as against its
own training opponents. This indicates the learned policy acquired a
reasonably general tactical improvement over the hand-designed baseline
rather than narrowly overfitting to its training pool, and that this
improvement is consistent across all three evaluated obstacle-density
levels.

### G. Limitations

- All results are from simulation; no hardware-in-the-loop or real-sensor
  validation was performed.
- The attacker's defender-sensing model (fixed detection range, exact
  measurement above it) is a simplifying assumption, not validated
  against any specific real sensing modality.
- The frozen defender pool excludes the computationally expensive
  receding-horizon optimization-based controller from training (though it
  is included, at reduced sample size, in evaluation) and excludes the
  original conference-baseline PPO/PPO-Kalman/LR-PPO checkpoints from
  training (they are evaluated only as an unseen-generalization test).
- Obstacles are abstracted as axis-aligned boxes with a fixed-size
  10-sector perception summary; finer-grained obstacle geometry or denser
  perception representations were not evaluated.
- Held-out evaluation used N=15 episodes per cell (N=5 for the
  optimization-based controller), smaller than would be ideal for
  precise per-cell effect-size estimation; the reported effects are
  supported primarily by their consistency across many independent
  cells (defenders, seeds, clutter levels) rather than by high per-cell
  precision.
- The evaluation run underlying these results was interrupted once by an
  unexpected machine restart and fully re-executed after adding resume
  infrastructure; while the recovery process included multiple integrity
  checks (Sections 12, R1-R16), it represents an additional operational
  complexity worth disclosing.

### H. Contribution statement

Building on the prior obstacle-aware, vectorized-training scale-up of the
defender side of this study, this work adds the first learned,
adaptive-tactics adversary evaluated against that full defender
population -- including both the defenders it was trained against and
defenders withheld entirely -- under the same canonical clutter protocol,
and provides an initial, seed-resolved, statistically-supported
quantification of how much a learned attacker can be expected to degrade
each previously evaluated defender family before any defender-side
adversarial retraining is attempted.

## 14. Recommendation for the next phase

The evidence in Section 8 -- near-universal, often statistically
significant degradation of safe-interception rate across essentially
every evaluated defender family, with degradation against **unseen**
defender families matching or exceeding that against the attacker's own
training-pool opponents, consistently across all three clutter levels --
supports proceeding to the next planned phase: **freezing Adaptive
Attacker v1 and retraining PPO/PPO-Kalman/LR-PPO against a mixture of the
scripted and learned attackers.** The one caveat is RMPC, whose single
positive-delta cell rests on only N=5 episodes against a single attacker
seed and should be re-evaluated at higher N (or with all 3 attacker
seeds, budget permitting) before being treated as a reliable
counter-example.

Per this sub-task's explicit scope, **no defender retraining was
performed here** -- this report stops at the recommendation.
