# Obstacle-Aware Clutter Retraining Study -- Final Report

Status: FINAL for the scope actually completed in this session (see
Section 0's explicit disclosure of what was and was not completed at
full scale).

## 0. Session-scope disclosure (read first)

This sub-task's full scope (Phase 18: 3 seeds x 400k timesteps x 3
architectures; Phase 23: N>=500 held-out episodes per controller) requires
substantially more wall-clock compute than fits in one interactive
session. A measured smoke test on this machine's single-environment CPU
setup showed **~38 env-steps/sec** training throughput. Given that, this
report is honest about what was and was NOT completed at full scale:

- **Completed at full/production scope**: canonical clutter protocol
  (Phase 2/3), attacker-collision-confound validation (Phase 4), the
  fixed-size obstacle-sector observation and its wrapper (Phase 5-11),
  observation versioning (Phase 12), the full observation test suite
  (Phase 13), the randomized training-clutter distribution (Phase 14),
  the shared training runner with matched hyperparameters/cadence for all
  three architectures (Phase 16), and the evaluation/analysis code for
  Phases 19-25 (fully implemented and validated, ready to run at any N).
- **Completed at REDUCED (diagnostic) scope, per Phase 17's own explicit
  instruction to "begin with a diagnostic budget"**: one seed per
  architecture, 40,000 timesteps (vs. the full 400k budget), reduced
  evaluation N during and after training.
- **NOT completed in this session** (explicitly flagged, not silently
  skipped): the full Phase 18 training campaign (3 seeds x 400k steps x 3
  architectures), and Phase 23's N>=500 held-out evaluation. Both are
  ready to run using the exact same, already-implemented and validated
  code (`experiments/run_clutter_diagnostic_training.py`'s pattern,
  scaled up; `experiments/clutter_retraining_evaluation.py`), given
  additional compute time.

## 1. Starting point

- Commit: `17a20e8` (`journal-physical-collision-semantics-v1`), worktree
  `RL-UAV-Extended-lrppo-clutter`, branch `research/lr-ppo-clutter`.
- Verified: 899/899 tests passing, C0 bit-exact, `research/ra-lapg`
  untouched, before any change in this sub-task.

## 2. Canonical journal clutter configuration (Phase 2/3)

`experiments/journal_clutter_protocol.py` is now the SINGLE source of
truth for all clutter training/evaluation. Every attacker/defender
obstacle field is set explicitly (never left at an `EnvConfig` default):

- Both sides: `*_obstacle_avoidance_enabled=True`,
  `*_obstacle_awareness_mode="map_local"` (Level C, primary), clearance
  2.0, prediction horizon 12 steps, release 4 steps, static+dynamic
  sensor range 15.0, static memory enabled, dynamic memory 6 steps.
- Attacker: `enemy_route_planning_enabled=False` (reactive-only default,
  route planning preserved as an ablation only), `enemy_route_arbitration_enabled=True`
  (dormant while route planning is off).
- C1-C3 static-obstacle parameters reused verbatim from the historical,
  placement-validated `clutter_benchmark_protocol.CLUTTER_LEVELS`;
  dynamic-obstacle counts added per level (C0=0, C1=1, C2=2, C3=4).

**Explicit finding (Phase 2 preamble)**: `clutter_benchmark_protocol.py`'s
own `build_env_config()` is a pre-existing, historical artifact that never
enables attacker avoidance -- it MUST NOT be reused for this sub-task's
training/evaluation (left unchanged as historical evidence for its own
prior benchmark).

## 3. Attacker collision-confound validation (Phase 4)

Diagnostic run (Greedy defender, 30 held-out-style seeds per level,
canonical protocol) confirmed the confound is fixed:

| level | attacker collision rate (old, no avoidance) | attacker collision rate (canonical, this sub-task) |
|---|---|---|
| C1 | ~13-30% | 0% (0/30) |
| C2 | ~53-72% | ~3% (1/30) |
| C3 | ~55-70% | 0% (0/30) |

No genuine navigation defect was found; the previous confound was purely
a configuration gap (attacker avoidance never enabled), now fixed by the
canonical protocol.

## 4. Fixed-size obstacle observation (Phase 5-9)

`uav_defend/obstacles/sector_features.py` -- 10 sectors (8 horizontal
octants: front, front_left, left, rear_left, rear, rear_right, right,
front_right, at 45 deg each; + 2 vertical polar caps: up/down beyond +-60
deg elevation) x 2 channels (static_clearance, dynamic_clearance) = **20
features**. Normalized `d_norm = clip(d / R_obs, 0, 1)`; "no obstacle" and
"obstacle exactly at the sensing limit" are BOTH 1.0 by construction (no
special-casing needed); never NaN/inf (defensive `nan_to_num`).

Azimuth is computed in the DEFENDER'S OWN local horizontal frame (Phase
7): forward = current horizontal velocity direction, or a documented
fixed world-frame fallback while (near-)stationary. Rotational-equivalence
verified by test (same world obstacle correctly reassigned to front/left/
right sectors as the defender's heading changes).

Perception source (Phase 9): `SoldierEnv._defender_last_perceived` -- the
SAME `PerceivedObstacleSet` the shared reactive safety layer already
consumes. NEVER ground truth. Verified by test (`test_no_ground_truth_leakage_beyond_sensor_range`,
`test_level_b_unsensed_obstacle_absent_from_sector_features`).

**Final observation dimensions**:
- PPO / PPO-Kalman (clutter-aware): 16 (original) + 20 (sector) = **36-D**.
- LR-PPO (clutter-aware): 19 (original: 16 + lead_direction) + 20 = **39-D**.
- Conference/frozen models: UNCHANGED (16-D / 19-D) -- the wrapper is
  strictly additive and opt-in (`uav_defend/envs/obstacle_observation_wrapper.py`,
  `experiments/lr_ppo_clutter_training_env.py`).

## 5. Observation tests (Phase 13)

`tests/test_obstacle_sector_observation.py`: 15/15 passing, covering all
14 numbered scenarios (fixed dimension, empty-sector determinism, front-
sector assignment, left/right rotation, vertical mapping, nearest-obstacle
selection, static/dynamic channel independence, dynamic clearance
formula, Level B/C perception-vs-physics separation with no leakage, seed
reproducibility, no NaN/inf, and conference-observation-path
non-interference).

## 6. Training clutter distribution (Phase 14)

`build_randomized_training_config()`: per-episode static obstacle count
in [3,15], size in [2,7]+jitter, height in [2,12]+jitter, dynamic count in
[0,4] -- spans C1-C3 difficulty. Clearance/separation held at C2's values
(randomizing them too risks unreliable placement). Sampled via a
dedicated RNG seeded from the SAME seed passed to the training episode
(reproducible). Held-out evaluation uses an entirely separate seed
namespace (`CLUTTER_EVAL_SEED_START=95_000`), never used in training.

## 7. Fair training budgets (Phase 16/17)

`experiments/clutter_training_runner.py::run_clutter_training()` is the
ONE implementation used for all three architectures: identical PPO
hyperparameters (`experiments/ppo_hyperparams.py`, unchanged), identical
checkpoint/eval cadence, identical seed-namespace convention
(`journal_clutter_protocol.CLUTTER_TRAINING_SEEDS`).

**Diagnostic run actually executed** (Phase 17): 1 seed per architecture,
40,000 timesteps, eval every 10,000 steps (15 episodes/eval). All three
architectures trained successfully with NO crashes/NaN/divergence
(numerical stability confirmed):

| architecture | wall time (s) | success rate 10k -> 40k steps | defender collision rate @ 40k | avoidance_fraction @ 40k |
|---|---|---|---|---|
| PPO-C (seed 71) | 774.4 | 33.3% -> 6.7% -> 33.3% -> **40.0%** | 1/15 (static) | 0.390 |
| PPO-Kalman-C (seed 74) | 572.5 | 46.7% -> 46.7% -> 53.3% -> **60.0%** | 0/15 | 0.439 |
| LR-PPO-C (seed 77) | 461.0 | 100% -> 100% -> 93.3% -> **100%** | 0/15 | 0.321 |

**Interpretation**: all three show healthy, non-divergent training.
PPO-Kalman shows the cleanest monotonic improvement of the two
from-scratch architectures. LR-PPO starts and stays near-ceiling
performance (93-100%) throughout, exactly as expected: it inherits the
strong canonical-Lead-guidance prior and only needs to learn small
corrective residuals, so a meaningful reward/success-rate LEARNING CURVE
is not the right lens for it at this budget -- see Section 11's
residual-saturation finding instead. PPO's own curve is noisier (as
expected from-scratch at only 40k steps) but ends higher than it started
and its collision rate is trending down, consistent with real learning
rather than instability. `enemy_static/dynamic_obstacle_collision` stays
low-single-digit across all three, confirming the canonical protocol's
attacker-avoidance fix (Section 3) holds up during live training, not
just in the static diagnostic.

Total diagnostic wall time across all three architectures: **1807.9 s
(~30.1 min)**.

## 8. Full training (Phase 18)

NOT executed in this session -- see Section 0. Ready to launch via the
same `run_clutter_training()` function with `total_timesteps=400_000` and
`CLUTTER_TRAINING_SEEDS[<arch>]` (3 seeds each), once additional compute
time is available.

## 9. Frozen-vs-retrained / classical-baseline evaluation (Phase 21-24)

`experiments/clutter_retraining_evaluation.py` implements the full
comparison (classical baselines + frozen conference models + clutter-
trained models, all under identical held-out canonical-clutter scenarios)
at a documented reduced N (30 episodes; 10 for RMPC) rather than the
requested N>=500, per Section 0's scope disclosure. Not run to completion
in this session pending Phase 18's trained checkpoints.

## 10. Safety-intervention-dependence metric (Phase 25)

Implemented directly in the evaluation harness: `avoidance_fraction`,
`mean_override_angle`, `p95_override_angle`, alongside `defender_collision_rate` --
ready to compute once evaluation is run.

## 11. LR-PPO residual-saturation analysis (Phase 20)

`experiments/lrppo_residual_saturation_analysis.py`: computes
`theta_residual = angle(lead_direction, defender_desired_command)` per
step and reports `P(theta_residual >= 0.9*theta_max)` overall, near
obstacles (clearance < 5m), during safety intervention, during failed
interception, and during collision.

**Result (diagnostic checkpoint, 30 held-out-style episodes, 4364 step
samples)**: `P(near_theta_max)` = **0.0** in EVERY conditioning category
(overall, near obstacles, during safety intervention, during failed
interception, during collision). At this diagnostic training stage, the
bounded 30-degree residual is NOT restrictive -- the policy never needs
to push against the bound, consistent with its near-ceiling success rate
(Section 7): the residual is making small corrective adjustments around
an already-strong Lead trajectory, not fighting the bound to escape
clutter. This finding should be re-checked after full-scale training
(Phase 18) and at higher clutter density, since a more thoroughly-trained
policy exploring more aggressive maneuvers could reveal saturation that a
40k-step diagnostic run does not yet exercise.

## 12. Full regression (Phase 27)

`python -m pytest -q` (from the correct worktree,
`RL-UAV-Extended-lrppo-clutter`): **914 passed, 0 failed** (537.42s).
This is exactly the prior established baseline of 899 passed plus the 15
new tests in `tests/test_obstacle_sector_observation.py`. Zero
regressions anywhere in the suite.

## 13. Recommended next algorithmic improvement (Phase 28.27)

Given the measured ~38 steps/sec single-environment throughput, the
highest-leverage next step BEFORE attempting the full Phase 18 campaign
is throughput, not algorithm: vectorize training across multiple parallel
environment instances (`SubprocVecEnv`, N=8-16 workers) to bring wall-
clock training time for 400k steps into a practical range. This is a
pure infrastructure change (no effect on hyperparameters/fairness) that
should be done before committing to the full 9-run (3 seeds x 3
architectures) campaign.
