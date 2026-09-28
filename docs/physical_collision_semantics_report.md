# Physical Obstacle-Collision Semantics -- Final Report

Status: FINAL. Benchmark and full regression suite both complete; numbers
below are the actual measured results.

## 0. Starting point

- Commit: `e51fef3` (`journal-defender-obstacle-awareness-v1`), worktree
  `RL-UAV-Extended-lrppo-clutter`, branch `research/lr-ppo-clutter`.
- Verified baseline reproduced: `python -m pytest -q` -> 884 passed, 0
  failed, before any change in this sub-task.
- `research/ra-lapg` worktree confirmed untouched (still separate, no
  cross-contamination).

## 1. Files changed

- `uav_defend/config/env_config.py` -- 4 new reward fields:
  `reward_defender_static_obstacle_collision`,
  `reward_defender_dynamic_obstacle_collision` (default -150.0, matching
  the existing worst-case `reward_unsafe_intercept` severity),
  `reward_enemy_static_obstacle_collision`,
  `reward_enemy_dynamic_obstacle_collision` (default 0.0, NEUTRAL --see
  Section 5).
- `uav_defend/obstacles/dynamic.py` -- new
  `swept_sphere_vs_sphere_first_contact()` helper: returns the earliest
  contact time `t in [0,1]` (not merely the closest-approach distance) for
  two linearly-moving spheres, needed to clamp position at first contact.
- `uav_defend/obstacles/__init__.py` -- exports the new helper.
- `uav_defend/envs/soldier_env.py` -- `step()`'s collision-diagnostic
  block extended to also determine each mover's first-contact event
  (static vs dynamic, whichever is earlier), clamp position/zero velocity
  at that point, and a new outcome-precedence block that checks physical
  collision BEFORE the four original mission outcomes.
- `tests/test_physical_collision_semantics.py` -- new, 15 tests (Phase 7/11).
- `tests/test_obstacle_env_integration.py`,
  `tests/test_defender_obstacle_avoidance.py`,
  `tests/test_enemy_obstacle_avoidance.py`,
  `tests/test_clutter_benchmark_protocol.py` -- 6 pre-existing tests
  updated to reflect the new (intended) behavior; see Section 2.
- `experiments/collision_semantics_benchmark.py` -- new, Phase 12-14
  frozen A/B/C benchmark (N=20 seeds) across all 13 controller instances.
- `docs/physical_collision_semantics_report.md` -- this file.

No changes were made to: any controller's core equations/architecture,
`LeadResidualPPOPolicyWrapper`'s composition logic or its 30-degree
residual bound, any RL observation vector or its dimensionality, obstacle
PERCEPTION code (Level B/C semantics, sensor ranges, map-known logic,
memory) in `uav_defend/obstacles/perception.py` or the reactive
`ObstacleAvoidancePlanner`/`avoidance.py`, RMPC's
`selected_rmpc_config.json`, any PPO/PPO-Kalman/LR-PPO checkpoint, or
`research/ra-lapg`.

## 2. Old vs. new collision semantics (Phase 2 audit + Phase 3-6 change)

### Old (pre-this-sub-task) semantics, confirmed by inspection

For all four combinations (defender/attacker x static/dynamic), a
collision was detected via swept-motion geometry (`first_segment_intersection`
for static AABBs, `swept_sphere_vs_sphere_min_distance` for dynamic
spheres) computed EVERY step, but was **purely diagnostic**:
- motion was NEVER blocked -- the mover's position after the step was
  exactly the unconstrained physics result, even if that segment passed
  through an obstacle;
- `terminated` was computed ONLY from `dist_de`/`dist_es`/step-count;
- `reward` never referenced any collision flag;
- `info["outcome"]` was always one of `intercepted` / `soldier_caught` /
  `unsafe_intercept` / `timeout` / `ongoing` -- never a collision string.

This was confirmed both by direct code inspection (Phase 21 of the prior
sub-task) and by two existing tests that explicitly asserted this
invariant: `test_obstacle_env_integration.py::test_collision_does_not_terminate_episode_or_alter_reward`
and `::test_collision_does_not_modify_trajectory` (both now intentionally
rewritten -- see below).

### New semantics (this sub-task)

- **Detection**: unchanged swept-motion geometry, but the static case now
  also records `t_enter` (already available from `first_segment_intersection`)
  and the dynamic case uses a NEW `swept_sphere_vs_sphere_first_contact()`
  helper that returns the earliest contact time `t in [0,1]` (previously
  only a closest-approach DISTANCE was computed, not a contact TIME).
- **Per-mover first-event selection**: for each mover (defender, enemy)
  independently, the static contact time and dynamic contact time (if
  both fired this step) are compared; the EARLIER one is that mover's
  "collision event" for the step (Phase 6/7: first-event-wins).
- **Position clamping (Phase 3/7 "must not pass through")**: the
  colliding mover's position is set to EXACTLY its first-contact point
  (not the unconstrained end-of-step position), and its velocity is
  zeroed.
- **Termination + outcome + reward precedence (Phase 6)**: checked, in
  order, BEFORE the four original mission outcomes:
  1. `defender_static_obstacle_collision`
  2. `defender_dynamic_obstacle_collision`
  3. `enemy_static_obstacle_collision`
  4. `enemy_dynamic_obstacle_collision`
  5. (unchanged) `soldier_caught` > `unsafe_intercept` > `intercepted` > `timeout` > `ongoing`
  A collision this step ALWAYS wins over a mission outcome that would
  otherwise also be true this same step (e.g. clipping a wall exactly at
  the moment of reaching intercept range is a crash, not an interception).
  Ties between the defender's own event and the enemy's are broken in
  favor of the defender's event -- an explicit, documented convention.
- **Diagnostics recorded** (new `info` keys): `defender_collision_event_type`
  / `enemy_collision_event_type` (`"static"`/`"dynamic"`/`None`),
  `..._collision_event_point`, `..._collision_event_speed` (speed at the
  instant of impact, before zeroing), `..._collision_event_step`, plus the
  pre-existing `..._collision_obstacle_index`/`..._collision_obstacle_id`
  (now populated symmetrically for the defender's dynamic-obstacle id too
  -- previously only tracked for the enemy).

### Tests intentionally updated (not regressions -- see each diff)

1. `test_obstacle_env_integration.py::test_collision_does_not_terminate_episode_or_alter_reward`
   -> renamed `test_collision_terminates_episode_with_dedicated_outcome_and_penalty`.
2. `test_obstacle_env_integration.py::test_collision_does_not_modify_trajectory`
   -> renamed `test_collision_clamps_position_instead_of_passing_through`.
3. `test_defender_obstacle_avoidance.py::test_obstacle_collisions_remain_diagnostic_only_never_terminal`
   -> renamed `test_obstacle_collision_terminates_immediately_with_dedicated_outcome`.
4. `test_enemy_obstacle_avoidance.py::test_avoidance_disabled_matches_no_obstacle_hostile_trajectory`
   -> the two trajectories now legitimately diverge starting at the
   collision step; updated to assert equality only up to that point, and
   correct classification from there.
5. `test_clutter_benchmark_protocol.py::test_obstacle_collision_is_diagnostic_not_terminal`
   -> outcome set widened to include the 4 new collision outcomes (still
   asserts collision is never MISCLASSIFIED as one of the 4 original
   outcomes).
6. Two `test_defender_obstacle_avoidance.py` tests
   (`test_obstacle_ahead_triggers_collision_free_modification`,
   `test_rmpc_also_uses_the_same_safety_layer`) needed the enemy to also
   have avoidance enabled (or a deterministic close-range placement for
   RMPC), because the enemy could now physically collide with the SAME
   shared test obstacle before the defender's own scenario played out --
   an interaction effect of Phase 4, not a defect in Phase 3's defender
   logic.

## 3. Event precedence rules (Phase 6)

See Section 2's "Termination + outcome + reward precedence" above; also
documented as an inline code comment at the exact decision point in
`SoldierEnv.step()`.

## 4. Reward treatment (Phase 5)

- **Defender collision**: `-150.0` by default (matches the existing
  `reward_unsafe_intercept` worst-case magnitude) -- strongly negative,
  as instructed, since this is the defender's own vehicle-loss failure.
- **Attacker (hostile) collision**: `0.0` by default -- NEUTRAL. Per
  instruction, the defender must NOT be automatically rewarded as if it
  had intercepted the attacker merely because the environment destroyed
  it. Mission-outcome accounting (`info["outcome"]`) and RL reward
  accounting are kept independently configurable (four separate reward
  fields), so a future study COULD assign a small positive reward to this
  case deliberately, but the default is neutral.

## 5. Swept-collision / tunneling implementation (Phase 7)

- Static: unchanged `ObstacleLayout.first_segment_intersection()`
  (already swept, already returns `t_enter`/`point`).
- Dynamic: NEW `swept_sphere_vs_sphere_first_contact()` -- solves the
  quadratic `|| (p0_a - p0_b) + t*((p1_a-p0_a)-(p1_b-p0_b)) || = radius_sum`
  for the smallest `t in [0,1]`, distinct from the pre-existing
  `swept_sphere_vs_sphere_min_distance()` (closest-APPROACH distance,
  which can occur strictly after first contact).
- Tests added for: a thin (1-unit-thick) wall crossed at 40 m/s in one
  `dt`, a fast-crossing bird timed to intersect the defender's path
  mid-step (not at either segment's endpoint), and a high-closing-speed
  head-on dynamic obstacle. All three correctly register a collision (no
  tunneling).

## 6. Deterministic collision-test results (Phase 11)

`tests/test_physical_collision_semantics.py`, 15/15 passing, covering all
13 numbered scenarios from the task list (several numbered items are
covered by a single test each; a few share a test where natural,
documented inline). All pass; see file for exact assertions.

## 7. C0 regression result (Phase 10)

`test_c0_exact_regression_preserved` (in the new file) plus the pre-existing
C0 tests in `test_obstacle_env_integration.py`
(`test_obstacles_disabled_reproduces_conference_baseline_multi_seed`) all
pass: with `obstacles_enabled=False`/`dynamic_obstacles_enabled=False`,
observations/rewards/terminated/truncated/outcome are bit-for-bit
identical to the plain `EnvConfig()` baseline across multiple seeds --
the new collision code paths are structurally unreachable in C0 (no
obstacles means `first_segment_intersection`/dynamic checks trivially
return no intersection every step).

## 8. Frozen-controller benchmark (Phase 12-14)

Benchmark: `experiments/collision_semantics_benchmark.py`, conditions
A_disabled / B_level_b / C_level_c, levels C1-C3, all 13 controller
instances, N=20 seeds/cell (780 episodes total, seeds 80000-80019).
Total wall time: **1073.0 s (~17.9 min)**. Full results:
`results/physical_collision_semantics/phase12_14_benchmark/{summary.csv,episode_metrics.csv}`.

### 8.1 Key finding: the pre-existing clutter-benchmark config never gave
the ATTACKER obstacle avoidance

`experiments/clutter_benchmark_protocol.py::build_env_config()` (reused
unmodified from the prior sub-task) never sets
`enemy_obstacle_avoidance_enabled` -- it stays at its default, `False`.
Under the OLD diagnostic-only collision semantics this was harmless (the
hostile could fly straight through an obstacle with no consequence).
Under the NEW physical semantics, this means the hostile now crashes into
obstacles very frequently at C2/C3, REGARDLESS of the defender's
condition:

| level | condition | `enemy_static_collision_rate` (mean across families) |
|---|---|---|
| C1 | A_disabled | ~0.13-0.30 |
| C1 | B_level_b / C_level_c | ~0.20-0.45 |
| C2 | A_disabled | ~0.53-0.72 |
| C2 | B_level_b / C_level_c | ~0.70-0.97 |
| C3 | A_disabled | ~0.55-0.70 |
| C3 | B_level_b / C_level_c | ~0.75-0.97 |

This is **not a bug** -- it is the direct, correct consequence of Phase 4
(attacker physical collision termination) applied to a benchmark
configuration that was never designed to give the attacker a way to
avoid a real collision. It is an important, honest finding: **the
existing clutter-benchmark protocol's default configuration is no longer
representative once collisions are physically consequential** --
`intercept_rate` at C2/C3 is now low in absolute terms mostly because the
attacker crashes before an interception/soldier-caught outcome can even
occur, not primarily because of defender performance. Any FUTURE re-run
of this benchmark intended to isolate defender-safety-layer effectiveness
should also enable `enemy_obstacle_avoidance_enabled=True` (with a
Level B/C mode) so the attacker's own environmental-failure rate does not
dominate the outcome distribution. This is flagged as a recommendation,
not fixed in this sub-task (changing the clutter-benchmark protocol's
long-standing default config is out of scope here).

### 8.2 Defender physical-collision rate: the safety layer works as designed

Despite the confound above, the DEFENDER-side result is clean and
positive. `defender_static_collision_rate`, averaged per family:

| level | condition | defender_static_collision_rate |
|---|---|---|
| C1 | A_disabled | 0.15 - 0.30 |
| C1 | B_level_b | **0.00** (every family) |
| C1 | C_level_c | **0.00** (every family) |
| C2 | A_disabled | 0.15 - 0.32 |
| C2 | B_level_b | 0.00 - 0.10 |
| C2 | C_level_c | 0.00 - 0.02 |
| C3 | A_disabled | 0.18 - 0.35 |
| C3 | B_level_b | 0.00 - 0.10 |
| C3 | C_level_c | 0.00 - 0.02 |

Enabling the shared safety layer (Level B or Level C) reduces the
defender's own physical-collision rate from ~15-35% down to 0-10%
(frequently exactly 0%) uniformly across every controller family, at
every clutter level -- confirming the safety layer built in the prior
sub-task remains fully effective now that collisions carry real physical
consequences. Level C performs at least as well as Level B everywhere
observed (both reach 0% at C1; Level C is strictly better or equal at
C2/C3). `defender_dynamic_collision_rate` and `enemy_dynamic_collision_rate`
are 0.0 in every cell -- the C1-C3 clutter levels (reused unmodified from
the prior sub-task) define only static obstacles, no dynamic ones.

### 8.3 Safety-layer intervention statistics

`mean_avoidance_fraction` and `mean_override_angle` rise with clutter
density (roughly 0.05-0.14 at C1 up to 0.10-0.28 at C3), consistent with
the prior sub-task's diagnostic-only-collision benchmark, and similar in
magnitude across all 7 controller families -- the shared safety layer
continues to apply uniformly regardless of which controller it sits
downstream of.

## 9. Recommended common obstacle RL observation (Phase 15, NOT implemented)

No RL observation was modified in this sub-task. This section is a
design RECOMMENDATION only, for a future, separately-reviewed change.

### Current observation (unchanged)
16-D, normalized to `[-1, 1]`: `[soldier(3), defender(3), defender_vel(3),
detected_flag(1), hostile_pos_est(3), hostile_vel_est(3)]`. Fixed-size,
estimator-track-agnostic (Direct vs Kalman differ only in how the hostile
fields are populated).

### Candidate designs considered

1. **Directional range rays / clearance sectors** (e.g. 8-16 fixed
   azimuth sectors x {min static clearance, min dynamic clearance,
   closing-rate} per sector). Fixed-size regardless of obstacle count;
   naturally generalizes to unseen layouts (a sector's value is a
   scalar summary, not a per-obstacle slot); cheap to compute from the
   SAME ground-truth/perceived-obstacle-set machinery already used by
   the reactive safety layer (`EnemyObstaclePerception`/
   `ObstacleAvoidancePlanner`). Downside: coarser directional
   resolution near the sector boundaries; a thin obstacle exactly on a
   sector edge is not sharply resolved.

2. **Nearest-K obstacle relative states** (K nearest static + K nearest
   dynamic obstacles' relative position/velocity/half-extents,
   zero-padded if fewer than K exist). Fixed-size; can represent an
   individual obstacle's exact geometry (not just a coarse summary).
   Downside: an arbitrary, somewhat brittle sort-and-truncate at
   boundary cases (an obstacle at rank K+1 that is actually more
   relevant, e.g. fast-closing but slightly farther, is invisible);
   requires a fixed K decision that may not generalize well between
   sparse (C1) and dense (C3) clutter.

3. **Hybrid local occupancy/range representation** (e.g. a small fixed
   grid or range histogram in body-relative coordinates, encoding
   local free-space directly rather than per-obstacle features).
   Best generalization to unseen layouts and variable density;
   naturally unifies static+dynamic (dynamic obstacles contribute to
   the same instantaneous occupancy/range read as static ones, which
   already matches the existing swept-AABB bridge in
   `dynamic_obstacle_swept_aabb()`). Downside: highest implementation
   complexity of the three, and requires deciding a grid/ray
   resolution and update rate that stays cheap enough for RL rollout
   throughput.

### Recommendation

**Directional range/clearance sectors** (option 1), for three reasons:
(a) it is the closest fit to information the environment ALREADY computes
for the reactive safety layer (`min_clearance`, per-obstacle distances),
so it requires the least new geometry code and the least risk of a
silent perception/observation mismatch; (b) it is fixed-size and
naturally handles variable obstacle count/density (C1 through C3) without
a hand-picked K; (c) it cleanly unifies static and dynamic obstacles (a
sector's minimum clearance is simply the min over both kinds present in
that direction this step), consistent with the existing
`_combined_avoidance_layout()` design already used by the reactive layer.
A reasonable starting point: 12 azimuthal sectors x 2 elevation bands x
{min_clearance, closing_rate} = 48 scalars, appended to the existing 16-D
vector (giving a 64-D observation) -- small enough to keep PPO/PPO-Kalman/
LR-PPO training tractable, and structurally IDENTICAL for every future
learned controller (fairness preserved by construction, matching this
sub-task's precedent of a single shared, controller-agnostic feed).

This design is a RECOMMENDATION for the next phase; implementing it, and
any resulting retraining, is explicitly OUT OF SCOPE for this sub-task.

## 10. Full regression (Phase 16)

`python -m pytest -q` (from the correct worktree,
`RL-UAV-Extended-lrppo-clutter`): **899 passed, 0 failed** (546.68s).
This is exactly the prior established baseline of 884 passed plus the 15
new tests in `tests/test_physical_collision_semantics.py`. Zero
regressions anywhere in the suite.

## 11. Readiness recommendation for PPO/PPO-Kalman/LR-PPO clutter retraining

Per the established staged plan (also stated in the prior sub-task's
report): (1) this sub-task -- physical collision consequences -- is now
DONE; (2) a shared, fairness-preserving obstacle observation (Section 9)
should be designed and implemented as its own reviewed change; (3) ONLY
THEN should PPO / PPO-Kalman / LR-PPO retraining under clutter be
considered. No retraining was performed in this sub-task, per instruction.
