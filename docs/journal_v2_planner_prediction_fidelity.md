# V2 attacker: obstacle-planner prediction fidelity

Scope: the scripted attacker A0 under V2 semantics (`enemy_nominal_guidance_mode =
"direct_obstacle_only"`). V1 and every defender planner are unchanged and still use the
original fixed-direction predictor (`ObstacleAvoidancePlanner.executed_prediction = False`).
No training, no learned policy, no change to mission guidance, perception, collision or
swept-contact semantics, candidate geometry, or the candidate ordering rule.

## Why

The planner scores each bypass candidate with a predicted trajectory and persists the winning
candidate's waypoint. Candidate scoring and the motion that is subsequently flown must be the
same motion, otherwise the planner can rank candidates by trajectories it will never fly.

## Execution (what `SoldierEnv._move_enemy` does every step)

```
u_base   = unit(target - p)                       # corrected A0: no weave / noise / evasion
plan():
  base   = predict_constrained_trajectory(p, v, u_base)           # fixed direction, <= 12 steps,
                                                                  # cut at the capture sphere
  if active:
     if base is collision-free AND segment(p -> target) touches no obstacle:   # V2 release
         reset();   command = u_base                              # back to DIRECT, immediately
     else:
         command = unit(waypoint - p)                             # SAME persisted waypoint
         if prediction of that persisted flight collides: reselect (new candidate + waypoint)
  elif base collides: activate = select among {left, right, climb}
  else: command = u_base
v  <- advance_velocity(v, v_e * command)   # turn-rate, accel, speed, climb/descent limits
p  <- p + v*dt ; boundary clip ; p, v stored as float32
```

A climb waypoint is the point at the current x/y at obstacle-top + clearance; a lateral waypoint is
a tangent corner of the inflated footprint. A waypoint persists until release or reselection.

## Predictor before this change

```
direction = unit(candidate.waypoint - p0)     # computed ONCE
trajectory = advance_velocity(v, v_e * direction) repeated for the whole horizon, float64
```

Differences from execution: (1) the direction is held instead of re-derived toward the persisted
waypoint each step; (2) the release-to-DIRECT rule is ignored; (3) float64 states, whereas execution
stores float32; (4) the same fixed-direction model was used for the persisted-waypoint feasibility
check.

## Predictor now (`predict_executed_trajectory`)

For each predicted step: if not yet released and step >= 1, evaluate the planner's own release rule
(same `predict_constrained_trajectory` base prediction, same obstacle layout, same
`segment_intersects` test); then command `unit(waypoint - p)` or, once released, the mission command
formed exactly as the environment forms it; advance with the shared `advance_velocity` /
`apply_boundary` primitives and store the state as float32 exactly as `_move_enemy` does
(`_executed_step`). Scoring keys, candidate order and weights are unchanged. The persisted-waypoint
feasibility check uses the same predictor.

`predict_waypoint_following_trajectory` is the release-free building block (persist on the waypoint
for the whole horizon). It is exact while a waypoint is persisted, but it is not sufficient as a
candidate score: it assumed the waypoint is flown for the full horizon even when execution releases
to DIRECT after a step or two. As the sole model it failed the safety gate (see below).

### Why float32 matters

A climb waypoint sits directly above the activation point. One step later the desired horizontal
heading is almost exactly opposite the current heading, and the sign of the turn-rate-limited
rotation is decided by sub-micrometre rounding. A float64 replica flies the opposite way
(C3 / 130006, step 67: 34.7 m error over 7 steps; `tests/test_obstacle_prediction_fidelity.py`).

## Measurements

Method: `experiments/journal_v2_planner_fidelity.py`. At every selection the prediction is compared
with the positions actually flown until the next selection, the 12-step horizon, or the end of the
episode (the final interpolated contact step is excluded). Errors are positions after k steps.
Passive A0 (defender parked), seeds 130000-130099, C1-C3, map_local perception.

Planner's own predictor, before (181f466) vs after:

| Category | Mean position error | Max | Mean heading error |
|---|---|---|---|
| Climb | 11.78 m -> 0.97 m | 49.6 -> 16.8 m | 49.4 -> 5.5 deg |
| Lateral | 4.92 m -> 1.68 m | 34.1 -> 37.2 m | 16.5 -> 6.4 deg |
| Dynamic obstacle | 13.03 m -> 3.45 m | 53.1 -> 23.1 m | 38.2 -> 20.1 deg |
| All | 8.71 m -> 1.50 m | 53.1 -> 37.2 m | 33.7 -> 7.2 deg |

The residual error in the full benchmark comes from changes the predictor cannot know, all of them
documented assumptions of the planner (static target and static layout over the horizon): the asset
random walk, newly perceived obstacles, the threatening dynamic-obstacle set, and a re-selection or
re-activation inside the interval. Under controlled conditions (asset frozen, full awareness, no
moving obstacles; C2/C3 seeds 130000-130199, 96 selections: 40 climb, 56 lateral) the executed-policy
predictor reproduces the flown trajectory to below 1 mm in every event and predicts the release step
in 96/96, whereas the original fixed-direction model is off by up to 56.5 m (92/96 events above 1 mm).

## Effect on paths (paired, 100 seeds per level, passive A0)

No collision was gained or lost in any paired seed and no level has a collision (0/300 both). C1 is
identical. C2 improves (max path-excess 9.08 -> 1.19, overshoots 5 -> 0, no case above 1.3x).
C3 is mixed: the previously pathological seeds 130063 (4.47 -> 0.99) and 130081 (3.06 -> 0.98) are
resolved, and 130065 improves (2.09 -> 1.32), but five other seeds gain long detours (e.g. 130006,
0.97 -> 12.98), raising the C3 mean path-excess from 1.14 to 1.21. With an honest predictor the
climb candidate (a point above the activation position that the vehicle immediately leaves) is
correctly seen to be a poor motion primitive, and in dense clutter the alternatives are often
margin-deficient. The remaining inefficiency is a waypoint-geometry problem, not a prediction
problem.

## Rejected variants

**Pure persisted-waypoint predictor (no release modeling).** Bit-exact while a waypoint persisted
but ended C3 seed 130001 in an attacker-obstacle collision, because it scored candidates as if they
would be flown for the whole horizon. Rejected by the safety gate and replaced by the release-aware
predictor above.

**Clearance-aware release (2 m release buffer).** A release rule that also required the mission flight
to keep `min(enemy_obstacle_clearance, distance(asset, obstacle) - capture_radius)` from every obstacle
was implemented, shared between `plan()` and the predictor, and measured on the same 300 paired
seeds. It produced no collisions but substantially worsened path efficiency and orbit behavior: C2 mean
path-excess 1.05 -> 1.21 (predictor-only: 0.98), C3 1.14 -> 1.50 (predictor-only: 1.21), C3 seeds above
1.3x 6 -> 12, new orbit events, mean avoidance duration 3.8 -> 8.5 steps, and C3 seed 130064 became a
single 79-step, 414 m climb segment. It did not improve clearance either (the C3 minimum fell to
0.055 m). Whenever release is withheld the vehicle orbits the persisted climb waypoint, so a clearance
rule on release cannot work until the waypoint geometry is redesigned. It is not part of this change;
clearance robustness will be revisited together with the waypoint geometry rather than patched
independently.

Per-episode minimum obstacle clearance (100 seeds per level), baseline / predictor-only / 2 m release
buffer: C1 identical; C2 min 0.51 / 0.51 / 0.49 m; C3 min 0.67 / 0.28 / 0.06 m, with 3 / 3 / 3 episodes
below 1 m.

## Known limitations (unresolved)

**Obstacle clearance is not a validated safety margin.** The V2 activation trigger and release rule use
physical geometry only; `enemy_obstacle_clearance` (2 m) constrains only the selection of a bypass
candidate. Consequently DIRECT flight can pass arbitrarily close to an obstacle without any avoidance
being triggered, and this predates the predictor change (baseline per-episode minimum clearance is
already 0.3-0.7 m in some seeds).

The clearest example is the staggered-obstacle scenario in
`tests/test_journal_v2_validation.py::test_staggered_obstacles_are_negotiated_and_return_to_direct`
(two pillars on the straight line to the asset). With the faithful predictor the attacker chooses a
single climb over both pillars, rather than the baseline's two lateral detours (38 steps and 183.6 m
versus 39 steps and 191.3 m). The trajectory is collision-free, returns to DIRECT and captures the
asset, but the final DIRECT approach passes **about 0.02 m** from the second pillar's corner. That is
not an operationally acceptable margin and must not be read as one. It follows directly from the
physical-only trigger and release semantics.

The attempted 2 m release buffer (above) was rejected, so the issue remains open. Both the clearance
semantics and the climb waypoint (a point above the activation position that the vehicle immediately
leaves, which also causes the remaining C3 path inefficiencies) are to be redesigned together.

## Test definitions

`test_two_staggered_obstacles_give_two_returned_excursions` asserted at least two excursions, which
encoded a historical lateral maneuver rather than an invariant. It is replaced by two tests:

* `test_staggered_obstacles_are_negotiated_and_return_to_direct` -- same geometry. Requires at least one
  collision-necessary excursion, no obstacle collision and strictly positive clearance (non-penetration
  only), mission completion, return to DIRECT, no unassociated or non-returning avoidance, no orbiting,
  and that both obstacles lie on the straight line and are perceived. The excursion count is not
  asserted. The minimum clearance is recorded as a diagnostic (`record_property`) and is deliberately
  not asserted against any margin.
* `test_separated_obstacles_require_two_distinct_returned_excursions` -- a geometry that genuinely needs
  two encounters. Two full-height pillars (ceiling-high, so a climb is excluded) with a deterministic
  sideways drift of the asset, so the line to it swings into the second pillar only after the first was
  cleared and the attacker is flying DIRECT. Requires AVOID -> DIRECT -> AVOID -> DIRECT: two
  collision-necessary excursions on distinct obstacles, at least five DIRECT steps between them, no
  collision, capture and no orbiting. Current result: excursions at steps 1-3 and 16-17, 12 DIRECT
  steps between, minimum clearance 1.47 m.

## Tooling

* `experiments/journal_v2_avoidance_recorder.py` -- observational recorder of planner inputs, outputs and
  selected-candidate evaluations for passive-A0 V2 episodes, plus per-level aggregation.
* `experiments/journal_v2_planner_fidelity.py` -- command-line driver (usage in its docstring):
  `cases`, `seeds`, `analyze`, `compare`, `controlled`. Results go to the gitignored
  `results/journal_v2_planner_fidelity/<label>/` and existing labels are never overwritten.
* `tests/test_obstacle_prediction_fidelity.py` -- the exactness tests against the real environment.
