# LR-PPO Paper vs Journal Extension: Environment Continuity Audit

Scope: a publication-critical, read-only audit comparing the frozen LR-PPO paper
reference (`origin/research/ra-lapg` @ `3eebbdbeaf829b937570e789b3af54b66d510132`)
against the current journal-development branch (`origin/research/lr-ppo-clutter` @
`ab037cbcfe9e211077cea76b91726691474649f0`). Nothing in either branch was modified,
checked out destructively, retrained, or re-evaluated. All claims below are backed by
`git show`/`git diff` against the two frozen SHAs, plus one small (20-seed, 960-step)
runtime-equivalence diagnostic described in Phase 10.

**This document supersedes `docs/original_paper_protocol_audit.md` /
`journal-paper-reproduction-v1` as the canonical LR-PPO baseline reference.** That
earlier work compared against the pre-LR-PPO conference-baseline protocol
(`results/final_nominal/`, generated from commit `8b4420d5...`), not the actual LR-PPO
paper branch -- see Phase 13 below for the reclassification.

## Branch topology (discovered, not assumed)

```
git merge-base 3eebbdbe ab037cbc  =>  93474b98a6a9ff909b0bc808b2117afb2d601958 (tag: clutter-env-v1)
```

`clutter-env-v1` is the last common ancestor. At that commit, `env_config.py` and
`soldier_env.py` ALREADY contain: the full LR-PPO paper protocol infrastructure
(`experiments/final_experiment_protocol.py`, lead-residual composition/diagnostic
protocols, RMPC baseline, theoretical-proof reports, conference-baseline freeze) AND
basic 3-D static-obstacle geometry with a reactive, controller-agnostic, RNG-free
avoidance layer for the hostile only (`uav_defend/obstacles/{geometry,layout,avoidance}.py`,
OFF by default). From `clutter-env-v1`, the two branches diverge:

- `research/ra-lapg` adds its own unique commits: a separate "FA-LAPG" exploratory
  research line (`experiments/fa_lapg_foundation/`, `experiments/learning_benefit/`,
  `uav_defend/guidance/{dynamic_feasibility,reachability_selection}.py`,
  `uav_defend/policies/analysis/*`) -- reachability-aware/horizon-conditioned
  predictors, failure-mode partitioning, factorial ablation studies. Verified this
  tooling is NOT imported by `uav_defend/policies/residual/lead_residual_ppo_policy_wrapper.py`
  or any core environment file -- it is self-contained analysis infrastructure, not
  part of SoldierEnv/EnvConfig.
- `research/lr-ppo-clutter` (journal) adds the obstacle-perception/clutter/attacker-
  intelligence/defender-adaptation work (`journal-clutter-baseline-v1` through
  `journal-paper-reproduction-v1`, 14 tags, see the branch-push report from the prior
  task).

The journal branch does NOT carry forward ra-lapg's FA-LAPG files (168 files differ
total; FA-LAPG accounts for ~35 of the 42 pure deletions). This is out-of-scope
removal of an unrelated research thread, not environment drift.

## Critical Question 1 verdict: defender/asset standby

**`defender_standby_until_detection: bool = True` already exists, byte-for-byte
identical (including its docstring and `docs/revised_predetection_protocol.md`), at
the merge-base `clutter-env-v1`.** It is part of the LR-PPO paper's own protocol, not
a journal addition, and the journal inherits it completely unchanged (zero lines of
`git diff` touch this field, `SoldierEnv.step()`'s standby state machine, or
`_move_defender()`'s core velocity computation).

Verified exact semantics (from `ra-lapg`'s own docstring, confirmed identical on
journal HEAD via `git show`):
1. `detected_at_step_start = self._enemy_detected` (captured BEFORE this step's mutations).
2. A. Soldier moves (existing stochastic random walk, unchanged by either branch).
3. B. If `standby_mode and not detected_at_step_start`: defender position :=
   soldier position (post-move), defender velocity := zero; supplied action has no effect.
4. C. Hostile moves (pursuit + weave + evasion); evasion reads the CURRENT (possibly
   just-synced) defender position -- i.e. effectively asset-centered during standby.
5. D. `_update_detection()` runs using the resulting defender-enemy distance (unchanged, see below).
6. `just_detected = (not detected_at_step_start) and self._enemy_detected`.
7. E. Action is executed through constrained dynamics ONLY if `detected_at_step_start`
   was True; the detection-transition step itself never executes the action -- first
   executed action is on the FOLLOWING step.
8. `standby_mode=False` (legacy) applies the action every step from the start, exactly
   as in the original pre-revision environment -- retained ONLY for historical
   diagnostics (`experiments/run_paper_reproduction_check.py`), never for any
   publication-relevant journal campaign (see Phase 9 table).

**Verdict: IDENTICAL.**

## Phase 3 -- structural diff categorization

168 changed files total between the two SHAs (`git diff --name-status`). Every
production-code change falls into exactly one category:

**A -- Intended journal extension**: `uav_defend/obstacles/{dynamic,perception,route_planner,sector_features}.py`
(new), `uav_defend/envs/obstacle_observation_wrapper.py` (new), all new fields in
`uav_defend/config/env_config.py` (dynamic obstacles, awareness modes, route
planning, obstacle-collision rewards, `enemy_defender_detection_range`,
`obstacle_unmapped_fraction`), the physical-collision-termination precedence block
and dynamic-obstacle advance in `uav_defend/envs/soldier_env.py`, `enemy_tactical_override`
hook (Adaptive Attacker v1 injection point), `uav_defend/policies/residual/lead_residual_ppo_policy_wrapper.py`'s
optional `extra_features` parameter, `experiments/adaptive_attacker_*.py`,
`experiments/defender_adaptation_*.py`, `experiments/journal_clutter_protocol.py`.

**B -- Required support for intended extension**: `defender_obstacle_avoidance_enabled`
field family (mirrors the pre-existing hostile avoidance layer for the defender,
default False), `AABBObstacle.map_known` metadata field (default True, no geometry/physics
effect), dedicated `_rng_dynamic_obstacles`/obstacle RNG streams (spawned strictly
AFTER the pre-existing spawn/soldier/hostile-motion/sensor streams via `seed_seq.spawn(1)`,
never perturbing them), `uav_defend/obstacles/{__init__,geometry,layout}.py` additive
exports/fields.

**C -- Experiment/training/evaluation infrastructure only**: all `experiments/run_*.py`,
`experiments/plot_*.py`, `experiments/*_statistics.py`, `experiments/*_tables.py`,
`clutter_training_env.py`/`clutter_training_runner.py`, `experiments/artifact_paths.py`
(portability wrapper -- see below), model/checkpoint registries, the DELETED
`experiments/fa_lapg_foundation/*` and `experiments/learning_benefit/*` (ra-lapg's own
exploratory tooling, out of journal scope), DELETED `uav_defend/policies/analysis/*`
and `uav_defend/guidance/*` (unused by the core environment or by
`LeadInterceptPolicy`/`LeadResidualPPOPolicyWrapper`; confirmed via import trace).
`uav_defend/policies/baseline/lead_math.py` DELETION + its logic INLINED into
`uav_defend/policies/baseline/lead_intercept_policy.py` -- verified line-for-line
identical algorithm (`solve_quadratic_intercept_time`, `pure_pursuit_direction`,
`get_target_state_estimate`, `solve_cv_lead`'s `a/b/c` quadratic coefficients all
match exactly); a pure refactor (the shared module existed only to serve the
now-deleted analysis controllers), not a behavior change.
`experiments/{final_experiment_protocol,lead_residual_diagnostic_protocol,post_detection_diagnostic_protocol}.py`'s
only changes wrap existing hardcoded paths in `resolve_artifact_or_local()` for
cross-worktree portability -- zero sha256/step/success-rate values changed.

**D -- Documentation/tests only**: all `docs/*.md` additions, `tests/test_enemy_obstacle_avoidance.py`
(+20 lines) and `tests/test_obstacle_env_integration.py` (+40/-10 lines) extend
existing test coverage; corresponding test deletions (`test_lead_math_regression.py`,
`test_ra_lapg_policy.py`, `test_*_predictor*.py`, etc.) track the Category-C deletions above.

**E -- Potential unintended environment drift**: none identified in production code
(see Phase 12).

## Phase 4 -- core semantic comparison table

| Item | Verdict |
|---|---|
| Soldier/asset initial position, defender initial position/velocity, hostile spawn geometry/velocity | IDENTICAL |
| Soldier random walk (`_move_soldier`) | IDENTICAL (zero diff) |
| Defender standby/co-location, velocity, action execution | IDENTICAL (see Critical Question 1) |
| Hostile movement base command (pursuit + weave + heading noise), reactive evasion | IDENTICAL math; evasion call relocated earlier in `_move_enemy()` for arbitration bookkeeping, same inputs/outputs when route-planning/avoidance disabled |
| Detection update ordering, `_update_detection()` | IDENTICAL (zero diff -- function untouched) |
| First post-detection controller action timing | IDENTICAL |
| Detection equation / detection radius | IDENTICAL (zero diff) |
| Measurement generation, measurement noise, Direct-track finite-difference velocity | IDENTICAL (zero diff) |
| Kalman initialization/update, ground-truth leakage prevention | IDENTICAL (zero diff -- `uav_defend/tracking/` untouched) |
| Defender dynamics (`_move_defender`): normalization, max speed/accel/turn/climb/descent, dt, integration order | IDENTICAL when `defender_obstacle_avoidance_enabled=False` (default) -- `u_safe = u_desired` exactly, `desired_velocity = v_d * u_desired` bit-for-bit |
| Original hostile A0 (route planning + obstacle avoidance disabled, no tactical override) | IDENTICAL -- `u_base` computed by the unchanged pursuit+weave+noise+evasion formula; avoidance/route/override all no-op by default |
| Reward components (progress, time penalty, proximity warning, intercept/caught/unsafe/timeout) | IDENTICAL -- original logic preserved verbatim, now nested inside an `else` branch only reachable when no obstacle-collision event occurred this step |
| Mission termination priority (soldier_caught > unsafe_intercept > intercepted > timeout > ongoing) | IDENTICAL, unchanged internal order; obstacle-collision outcomes checked FIRST but are structurally unreachable (`_*_collision_event_type` stays `None`) when obstacles are disabled |
| RNG streams: spawn, soldier, hostile-motion, sensor | IDENTICAL draw sequence and order -- new obstacle/dynamic-obstacle streams are spawned strictly AFTER via additional `seed_seq.spawn(1)` calls, consumed only when the corresponding feature is enabled |

## Phase 5 -- "C0 + A0 + LR-PPO semantics" compatibility configuration

**This is simply `EnvConfig()` with bare defaults** (equivalently
`journal_clutter_protocol.build_canonical_env_config("C0", estimator)`, which sets
`obstacles_enabled=False, dynamic_obstacles_enabled=False` and otherwise does not
touch any LR-PPO-paper field): `obstacles_enabled=False`, `dynamic_obstacles_enabled=False`,
`enemy_obstacle_avoidance_enabled=False`, `defender_obstacle_avoidance_enabled=False`,
`enemy_route_planning_enabled=False`, `enemy_obstacle_awareness_mode="disabled"`,
`defender_obstacle_awareness_mode="disabled"`, `enemy_tactical_override=None` (A0
scripted), `defender_standby_until_detection=True` (bare default, matching LR-PPO).
No new implementation was required -- this configuration already exists and is
exactly what Phase 10's runtime check exercised.

## Phase 6 -- observation-contract comparison

| Model | LR-PPO paper dim | Journal (D0/D0+/D1) dim | Added channels |
|---|---|---|---|
| PPO (Direct) | base Direct observation (unchanged) | base + 20 obstacle-sector features (`SECTOR_OBSERVATION_V1`) when `needs_sector_features=True` | obstacle-sector features only, via `extra_features` concatenation at the POLICY call site, never inside `SoldierEnv._get_obs()` |
| PPO-Kalman | base Kalman observation (unchanged) | base + 20 sector features | same as above |
| LR-PPO | 19-D residual observation (unchanged) | 19 + 20 = 39-D for clutter-trained LRPPO-C checkpoints; frozen conference LR-PPO checkpoints still receive exactly 19-D (`extra_features=None` default) | `lead_residual_ppo_policy_wrapper.act()`'s new optional parameter, opt-in only |

This is classified as **intended model-input extension**, not environment drift:
`SoldierEnv._get_obs()` itself (the actual environment observation contract) is
UNCHANGED (zero diff); the sector features are computed by a separate module
(`uav_defend/obstacles/sector_features.py`) and concatenated by the EXPERIMENT/POLICY
layer before calling `model.predict()`, never by the environment. Environment
dynamics change vs. policy observation change are cleanly separated by construction.

## Phase 7 -- attacker-intelligence hierarchy

- **A0**: `enemy_tactical_override=None` (the default). Verified via Phase 10's
  runtime check: bit-for-bit identical hostile trajectories to the frozen LR-PPO
  reference across 20 seeds / 960 steps when obstacles/route-planning/perception are
  all at their (also-identical) defaults. Semantically equivalent to LR-PPO A0.
- **A0 + obstacle intelligence**: adds perception (`enemy_obstacle_awareness_mode`),
  reactive avoidance (`enemy_obstacle_avoidance_enabled`), and optional route
  planning/arbitration (`enemy_route_planning_enabled`/`enemy_route_arbitration_enabled`).
  These act on `u_base` (the unchanged hand-designed tactical command) or substitute
  `r_hat` (pursuit direction) with a locally-planned waypoint direction -- they change
  ONLY obstacle navigation, never the mission-pursuit/evasion tactical logic itself
  (confirmed: `_compute_enemy_evasion()` call is unchanged; route planning only
  engages `enemy_obstacle_avoidance_enabled and enemy_route_planning_enabled`, and is
  explicitly suspended, not blended, during defender-evasion per the arbitration
  priority order).
- **A1 (Adaptive Attacker v1)**: `enemy_tactical_override` is set to a callable
  wrapping a frozen, inference-only PPO model's `.predict()`. This REPLACES the
  entire hand-designed tactical command (`u_base`) with the learned direction; the
  diagnostic fields for the hand-designed model are still computed in parallel (for
  comparison), but never fed back into control. Obstacle-safety (reactive avoidance),
  vehicle dynamics (`_move_enemy`'s desired-velocity -> `_advance_velocity` pipeline),
  and collision semantics are completely UNCHANGED and apply identically regardless
  of which `u_base` is in effect (override or hand-designed).

  **Verified claim**: "The learned attacker changes tactical maneuver selection,
  while low-level vehicle dynamics and obstacle-safety mechanisms remain shared." --
  TRUE, confirmed directly in code: the override substitution happens strictly
  between `u_base` computation and the (shared, unmodified) obstacle-avoidance
  `plan()` call and `_advance_velocity()` dynamics integration.

## Phase 8 -- defender-intelligence comparison

- **Original LR-PPO**: PPO, PPO-Kalman, LR-PPO trained/evaluated with
  `defender_obstacle_avoidance_enabled` not present (field doesn't exist pre-journal).
- **Journal D0**: obstacle-aware retraining using the SAME `_move_defender()`
  pipeline, SAME detection logic, SAME terminal interception geometry
  (`intercept_radius`/`threat_radius`/`unsafe_intercept_radius` unchanged) -- the ONLY
  addition is the optional, controller-agnostic obstacle-safety layer inserted between
  the policy's desired direction and the dynamics integration (exact no-op when
  disabled, confirmed above).
- **D0+**: equal-budget (425,984 transitions) continuation against the scripted
  attacker only; uses the identical D0 environment/config pipeline.
- **D1**: equal-budget continuation against the A0/A1-seen adversarial mixture; uses
  the identical environment/config pipeline, differing only in which
  `enemy_tactical_override` is wired per episode (`OpponentMixtureWrapper`).

**Verified**: D0/D0+/D1 do NOT alter physical defender dynamics, detection logic, or
terminal interception geometry -- confirmed via the same zero-diff functions
(`_update_detection`, `_move_defender`'s core velocity law, the terminal-condition
`else` branch) traced above.

## Phase 9 -- configuration provenance audit (standby)

Full source-level call-chain trace (not relying on isolated default inspection):

| Campaign | Config builder actually called | `defender_standby_until_detection` | Evidence |
|---|---|---:|---|
| Clutter-training scale-up (D0, PPO-C/PPO-Kalman-C/LR-PPO-C, 400k x3 seeds x3 archs) | `journal_clutter_protocol.build_randomized_training_config` (via `clutter_training_env.py`) | `True` | Function never passes this kwarg; traced to bare `EnvConfig()` default. Directly re-invoked `build_canonical_env_config('C1','measurement')` live: `cfg.defender_standby_until_detection == True`. `training_manifest.json` stores only a `config_hash`, not a full dict -- noted as a limitation. |
| Adaptive Attacker v1 (A1 training, 400k x3 seeds) | `journal_clutter_protocol.build_canonical_env_config("C2", estimator)` | `True` | Same function/trace as above; `adaptive_attacker_env.py:181-182` call sites confirmed directly. |
| Defender-adaptation training (D0+/D1, 18 runs) | `clutter_training_env.build_ppo_clutter_env`/equivalent LR-PPO env builder (imported transitively from `defender_adaptation_env.py` -> `experiments.clutter_training_env`) | `True` | Same `build_randomized_training_config` call chain, confirmed via import trace through `defender_adaptation_env.py::build_ppo_d1_env`. |
| Defender-adaptation evaluation (32,400 episodes, C1-C3) + C0 extension (10,800 episodes) | `journal_clutter_protocol.build_canonical_env_config(level, estimator)` | `True` | Direct call sites in `experiments/defender_adaptation_evaluation.py` and `experiments/defender_adaptation_c0_evaluation.py`; neither ever passes a standby kwarg. |
| `run_paper_reproduction_check.py` (prior sub-task, legacy-protocol diagnostic ONLY) | explicit `EnvConfig(..., defender_standby_until_detection=False)` for the "corrected original semantics" arm | `False` (deliberately, diagnostic-only) | The ONLY place in the entire `experiments/` tree where `False` is ever passed (`grep` confirmed). Not a publication training/evaluation campaign -- see Phase 13. |

**No publication-relevant journal campaign used `False`.** This is NOT a
CRITICAL COMPARABILITY FAILURE.

## Phase 10 -- runtime semantic verification

Diagnostic harness: 20 seeds (900001-900020, disjoint diagnostic-only range), bare
`EnvConfig(use_kalman_tracking=False)` on both sides (no obstacles, no route
planning, no tactical override, standby=True), identical externally-supplied
constant action `[1,0,0]`, run via a temporary detached worktree of
`3eebbdbeaf829b937570e789b3af54b66d510132` (removed after use; `research/ra-lapg`
branch ref itself never checked out or modified) and the current journal checkout.

Compared per-step: soldier position, defender position/velocity, enemy
position/velocity, detection flag, reward, outcome, across all 20 episodes (960 total
steps).

**Result: 0 mismatches, max absolute difference = 0.0 (bit-for-bit identical).**

No tolerance was needed -- exact equivalence, not merely within-tolerance agreement.
This directly confirms the static diff analysis above: with all journal extensions at
their default (disabled) settings, the environment is mathematically identical to the
frozen LR-PPO reference.

## Phase 11 -- publication difference matrix

| Component | LR-PPO paper | Journal | Classification | Scientific rationale |
|---|---|---|---|---|
| 3-D dynamics (`_advance_velocity`, accel/turn/climb/descent limits) | baseline | unchanged | IDENTICAL | n/a |
| Defender standby (`defender_standby_until_detection`) | `True` (paper's own revised protocol) | `True` (inherited unchanged) | IDENTICAL | n/a |
| Asset/soldier movement | stochastic random walk | unchanged | IDENTICAL | n/a |
| Detection (`_update_detection`) | baseline | unchanged | IDENTICAL | n/a |
| Measurement model (noise, finite-difference velocity) | baseline | unchanged | IDENTICAL | n/a |
| Kalman filtering | baseline | unchanged | IDENTICAL | n/a |
| Base A0 attacker (pursuit+weave+evasion) | baseline | unchanged (verified bit-for-bit, Phase 10) | IDENTICAL | n/a |
| Static obstacles | present (geometry only, from `clutter-env-v1`), OFF by default | present, extended with perception/map-knowledge metadata, OFF by default | INTENDED EXTENSION | obstacle-rich operation (journal objective 2) |
| Dynamic obstacles | absent | present, OFF by default | INTENDED EXTENSION | obstacle-rich operation |
| Obstacle perception | absent | present (`disabled`/`local`/`map_local`), default `disabled` (= full ground truth, prior behavior) | INTENDED EXTENSION | realistic sensing-constrained safety |
| Attacker obstacle avoidance | present (reactive only), OFF by default | present, extended with perception + optional route planning, OFF by default | INTENDED EXTENSION | obstacle-rich operation |
| Defender obstacle avoidance | absent | present, OFF by default | INTENDED EXTENSION | obstacle-rich operation (journal objective 2) |
| Physical collisions | diagnostic-only | terminal + reward consequence, gated behind obstacles being enabled | INTENDED EXTENSION | obstacle-rich operation |
| Route planning (hostile) | absent | present, OFF by default, with behavior-arbitration vs. evasion | INTENDED EXTENSION | obstacle-rich operation |
| Obstacle-sector observations | absent | present, opt-in via `extra_features` | INTENDED EXTENSION (policy-input, not environment) | model capacity for obstacle-aware control |
| LR-PPO residual composition | baseline | unchanged core algorithm; optional extra-features concatenation opt-in | IDENTICAL (core) / INTENDED EXTENSION (optional input) | preserve paper's architecture; add obstacle awareness |
| Adaptive attacker (A1) | absent | present via `enemy_tactical_override` hook, shares dynamics/safety/collision with A0 | INTENDED EXTENSION | increased attacker intelligence (journal objective 3) |
| Defender adaptation (D0/D0+/D1) | absent | present, reuses unchanged defender dynamics/detection/terminal geometry | INTENDED EXTENSION | defender robustness/adaptation (journal objective 4) |

## Phase 12 -- unintended drift search

Searched specifically for: default-value changes, execution-order changes, dynamics
equation changes, reward-coefficient changes, terminal-ordering changes, detection-
semantics changes, hostile-base-behavior changes, action-scaling changes,
observation-normalization changes, seed-generation changes.

**No candidate drift found in production code.** Every one of the 10 modified
(not purely-added) production files was individually reviewed:
`env_config.py`, `soldier_env.py`, `lead_intercept_policy.py`,
`lead_residual_ppo_policy_wrapper.py`, `final_experiment_protocol.py`,
`lead_residual_diagnostic_protocol.py`, `post_detection_diagnostic_protocol.py`,
`obstacles/{geometry,layout,__init__}.py`. Every change is either (a) a new field/
function with a default that reproduces prior behavior exactly, (b) a path-portability
wrapper with no value change, or (c) a verified-identical inlining refactor. Severity:
**none** for all examined items.

## Phase 13 -- reassessment of `journal-paper-reproduction-v1` / the prior C0 study

`docs/original_paper_protocol_audit.md` and the 10,800-episode C0 evaluation
(commit `ab037cb`, tag `journal-paper-reproduction-v1`) compared journal checkpoints
against `results/final_nominal/` -- an artifact generated from commit
`8b4420d555defd42569460b3e386f3729173ea4f`, a PRE-LR-PPO conference-baseline state
that predates the standby-protocol revision entirely (confirmed: its stored
`env_config` snapshot has NO `defender_standby_until_detection` field at all, i.e.
legacy/implicit standby=False behavior).

Given this audit's finding that the standby=True revision is part of the LR-PPO
paper's OWN protocol (present at `clutter-env-v1`, the common ancestor, and verified
bit-for-bit shared with the frozen `research/ra-lapg` branch), the prior sub-task's
"drift" finding was a genuine, correctly-diagnosed artifact of comparing against the
WRONG (pre-LR-PPO) baseline -- not a journal-vs-LR-PPO difference.

Reclassification:
- **Historical legacy-protocol evidence (valid as such)**: the checkpoint-hash
  verification (all 6 frozen 400k PPO/PPO-Kalman checkpoints unchanged) and the
  concrete standby=True-vs-False behavioral-drift demonstration (97.3% vs 74.3%
  exact-match) remain scientifically useful artifacts documenting that (a) checkpoints
  are uncorrupted and (b) the standby field genuinely changes outcomes -- a fact worth
  keeping on record.
- **NOT valid as a direct LR-PPO-paper-to-journal comparison**: the 10,800-episode C0
  retention/generalization results (PPO-Kalman-C significant gain, PPO-C concerning
  negative trend, LR-PPO-C residual-usage shift) were computed under
  `defender_standby_until_detection=True` throughout (both D0/D0+/D1 sides), so they
  are internally self-consistent, but the DOCUMENT's framing of this as "reproducing
  the accepted paper" conflated the pre-LR-PPO conference baseline with the LR-PPO
  paper itself. The underlying 10,800 episodes of data are NOT invalidated (they
  remain a valid D0-vs-D0+-vs-D1 comparison under consistent standby semantics) --
  only the framing/baseline label needs correction in any future publication text.
- Per the explicit instruction: `ab037cb` and `journal-paper-reproduction-v1` are
  NOT altered, amended, retagged, or deleted. No data is discarded.

## Phase 14 -- publication wording

**The audit supports the target conceptual wording.** Methods paragraph:

> The journal study preserves the engagement, sensing, vehicle-dynamics, and
> pre-detection standby semantics of the LR-PPO paper while extending the environment
> with obstacle-rich operation (static and dynamic obstacles, perception-constrained
> attacker/defender avoidance, physical collision consequences, and optional
> obstacle-aware observation features) and progressively more capable hostile
> behavior (a learned adaptive attacker, Adaptive Attacker v1, sharing the base
> vehicle dynamics and obstacle-safety layer with the original scripted hostile).
> When every journal extension is disabled, the environment is verified bit-for-bit
> identical to the frozen LR-PPO reference implementation across a 20-seed,
> 960-step runtime equivalence check.

**Reproducibility note**:

> An earlier reproducibility investigation (commit `ab037cb`, tag
> `journal-paper-reproduction-v1`) compared journal checkpoints against the
> pre-LR-PPO conference-baseline protocol (`results/final_nominal/`, generated under
> legacy `defender_standby_until_detection=False` semantics that predate the LR-PPO
> paper's own standby revision). That investigation remains valid evidence of
> checkpoint integrity and of the standby field's behavioral significance, but is NOT
> the canonical LR-PPO-paper comparison. The canonical comparison is documented here,
> against `research/ra-lapg` directly, under identical `defender_standby_until_detection=True`
> semantics confirmed shared by both the LR-PPO paper and every publication-relevant
> journal campaign (Phase 9).
