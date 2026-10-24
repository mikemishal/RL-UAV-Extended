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
  selected-candidate evaluations for passive-A0 V2 episodes, plus per-level aggregation. It also records the
  stateful controller's per-decision record and diagnostics counters.
* `experiments/journal_v2_planner_fidelity.py` -- command-line driver (usage in its docstring):
  `cases`, `seeds`, `analyze`, `compare`, `controlled`. Results go to the gitignored
  `results/journal_v2_planner_fidelity/<label>/` and existing labels are never overwritten. Setting
  `JOURNAL_V2_LEGACY_AVOIDANCE=1` records with the legacy waypoint planner (reproduces `3c61845` exactly:
  0 changed trajectories over the 300 seeds).
* `tests/test_obstacle_prediction_fidelity.py` -- the exactness tests against the real environment (these run
  the legacy waypoint planner, `SoldierEnv.STATEFUL_ENEMY_AVOIDANCE = False`, which V1 and the defender
  planners still use).
* `tests/test_stateful_avoidance.py` -- the controlled scenarios for the stateful controller (below).

## Stateful receding-horizon avoidance (V2 attacker only)

`uav_defend/obstacles/stateful_avoidance.py`, selected by `SoldierEnv.STATEFUL_ENEMY_AVOIDANCE` (default
`True`; read at construction) and only for `enemy_nominal_guidance_mode == "direct_obstacle_only"`. V1, the
defender planners, swept contact, mission-event ordering, collision radii, perception, A1 and the asset are
untouched. A0 still has no defender evasion, no weave and no tactical intelligence.

### Old state machine (3c61845, still used by V1 and the defender planners)

```
DIRECT: u = u_base
  fixed-direction prediction along u_base (12 steps, cut at capture) collides?  -> ACTIVATE
ACTIVATE: candidates = {left tangent corner, right tangent corner, point above the activation position}
  select (SAFE = collision-free & clearance>=2 m  <  margin-deficient  <  colliding;
          then -progress, angular deviation, -clearance, order)
  planner.waypoint, planner.mode, planner.obstacle_index  <- selection        (persisted)
AVOID each step:
  release if fixed-direction prediction is collision-free AND straight position->target line is physically clear
        -> DIRECT                          (uninflated geometry, no turn dynamics)
  else u = unit(waypoint - position)       (re-aimed at the HISTORICAL point every step)
        if the executed rollout toward the waypoint collides -> reselect for the same obstacle
```
`AVOID -> DIRECT -> AVOID` arises because trigger and release use a fixed-direction prediction while real
DIRECT flight re-aims every step, the release line ignores the turn-rate limit, and the persisted waypoint
sits where the vehicle already was.

### New state machine

```
every step, from the CURRENT float32 state:
  R_d = DIRECT rollout        (mission command re-derived each step through the executed dynamics,
                               cut at the capture sphere or, if the asset is not reached, at the closest approach)
  if R_d is collision-free:   DIRECT  (release any avoidance; same test as the trigger)
  else:
    B = obstacles R_d hits or passes within `clearance` of           (active blocker set)
    for law in {LEFT, RIGHT, CLIMB}:
        rollout = fly guidance(law, state, B) until DIRECT's rollout is collision-free, then fly DIRECT
    B <- B U obstacles threatened by the chosen rollout; re-evaluate once
    choose lexicographically (below); command = guidance(chosen law, state, B)
```
There is no stored x/y/z target. The only state is `{active, mode, obstacle_index (primary blocker),
blockers, entry_step}` plus diagnostics counters; the mode is the hysteresis input.

* **CLIMB** (`climb_guidance`): horizontal direction = the current direct-to-asset horizontal direction
  (never toward the activation point). Vertical: climb at the vehicle's climb rate while below
  `z_req = min(max(top of B) + clearance, ceiling)`, then hold level.
* **LEFT / RIGHT** (`lateral_guidance`): head horizontally for the visible clearance-inflated corner of the
  nearest blocker on that side, recomputed from the current position every step (the existing
  tangent-corner rule, no historical point); if that points behind the mission direction, sidestep
  perpendicular to it instead. **Changed from the old law:** the vertical component stays the mission's own,
  so a diving attacker is not made to hold altitude and fly over the asset. Untouched: the corner geometry
  and the inflated footprint (`_horizontal_bypass_candidate`, `_visible_corner_waypoint`).
* **DIRECT rejoin**: eligible iff the actual turn-limited DIRECT rollout from the current float32 state is
  collision-free under the existing physical geometry. No clearance buffer is added. Straight-line
  clearance is neither required nor sufficient (`test_a_clear_straight_line_does_not_release_...`).
* **Why the closest-approach cut**: a turn-limited pursuer that misses the asset flies a wide loop; scoring
  the loop's far side against obstacles triggered avoidance for events many steps away that the attacker
  would not reach before re-planning. The cut keeps the engagement to the approach, and the immediate next
  step is always kept. A candidate that is collision-free but predicted to pass the asset outside the capture
  sphere and move away (`flyby`) ranks below one that captures.
* **Candidate order** (`select_candidate`, no weights): (1) collision-free beats colliding; (2) not predicted
  to fly past the asset; (3) meets the required clearance over the avoided part (a margin-deficient
  candidate then ranks by clearance first, as before); (4) positive progress; (5) lower predicted
  mission-path cost (path minus progress, bucketed by one step's travel `v*dt`); (6) less steering
  (cumulative heading change); (7) larger clearance; (8) order left, right, climb.
* **Anti-chatter**: no minimum-duration timer. The current mode is kept when it is collision-free, in the
  same feasibility class as the best candidate, still makes progress (or the best does not), and its path
  cost is within one step's travel of the best (effectively equivalent). A clearly better candidate still
  replaces it.
* **Emergency (all candidates collide)**: the previous rule picked the roomiest candidate, which ties at
  zero when the vehicle is already inside a dynamic swept box and degenerated to the fixed order. It is now:
  latest first contact, then fewest conflicted rollout segments (leave the conflict region soonest), then
  roomiest, then order. No hysteresis inside an emergency. Adding this rule changed exactly one of the 300
  recorded trajectories (C3 130047: 11 -> 10 steps, same outcome) and no tail or collision; it is covered by
  unit tests.

### Results (passive A0, seeds 130000-130099, paired with the legacy planner)

| | C1 | C2 | C3 |
|---|---|---|---|
| Collisions gained / lost | 0 / 0 | 0 / 0 | 0 / 0 |
| Capture rate | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Path-excess mean (old -> new) | 0.970 -> 0.970 | 0.977 -> 0.971 | 1.209 -> 1.235 |
| Path-excess max | 1.02 -> 1.02 | 1.19 -> 1.02 | 12.98 -> 27.06 |
| > 1.3x / > 2x / > 5x | 0/0/0 -> 0/0/0 | 0/0/0 -> 0/0/0 | 7/5/1 -> 1/1/1 |
| Avoidance activations | 1 -> 1 | 10 -> 11 | 64 -> 25 |
| Short AVOID-DIRECT-AVOID cycles (<= 2 DIRECT steps) | 0 -> 0 | 3 -> 4 | 15 -> 2 |
| Mode switches | 0 -> 0 | 1 -> 1 | 7 -> 1 |
| All-candidates-collide selections | 1 -> 1 | 1 -> 1 | 14 -> 4 |
| Overshoots / orbit events | 0/0 -> 0/0 | 0/0 -> 0/0 | 14/0 -> 18/0 |
| Min clearance min / p1 / p5 / median (m) | 0.30/1.10/2.43/16.2 -> 0.30/1.10/2.39/16.2 | 0.51/0.55/1.77/7.27 -> 0.23/0.50/1.03/6.90 | 0.27/0.66/1.14/4.13 -> 0.38/0.41/0.67/3.39 |
| Episodes with min clearance < 0.25 / 0.5 / 1.0 m | 0/1/1 -> 0/1/1 | 0/0/4 -> 1/1/5 | 0/1/3 -> 0/2/7 |

Selections (new): C2 left 25 / right 8 / climb 2; C3 left 37 / right 12 / climb 25. DIRECT rejoin accepted /
rejected: C2 11 / 23, C3 22 / 47. Unassociated and non-returning avoidance: 0. Tactical evasion: none.
Only C2 and C3 trajectories changed (7 and 26 of 100 each); C0, C1 and all V1 trajectories are bit-identical.

**Residual: pure-pursuit orbit trap (C3 130017, 27x).** The single remaining C3 tail has 2 avoidance steps in
347. After a brief avoidance the attacker passes the asset by ~3-5 m, inside its turning circle, and the
unchanged DIRECT mission command then circles it on a near-closed orbit that never enters the 2 m capture
sphere. That is the mission guidance, not obstacle avoidance (the foundation's 130006 tail had 69 avoidance
steps in 167). Excluding this episode from both sets the C3 mean is 0.974 (foundation 1.212). It is the same mechanism that
made earlier rejected designs fail near the asset, and it needs a separate decision on the mission command's
speed profile near the asset; it is deliberately not changed here.

The staggered-obstacle diagnostic improves from 0.0241 m to 0.905 m minimum clearance (14 steps instead of
39) without any clearance buffer; the separated-obstacle case is unchanged (1.47 -> 1.48 m, two excursions).
The 0.9 m is a by-product, not a validated operational margin.

## Rejected: forward far-side climb waypoint

Tried on top of the committed predictor (release rule unchanged). Original climb waypoint:
`(x_att, y_att, obstacle_top + clearance)` -- straight above the mover's position at activation, omitted
only if that height exceeds the ceiling. Replacement (V2 attacker only): horizontal point on the nominal
mission ray (horizontal part of `u_base`, from the current position) at the exit of the clearance-inflated
footprint plus one more `clearance` (the footprint's far support distance if the ray misses it); altitude
`z0 + (top + clearance - z0) * t_waypoint / t_enter` so the straight climb already clears the obstacle at
footprint entry (a mover already above `top + clearance` keeps that altitude), either omitted (variant 1) or
clamped to the ceiling (variant 2) when that is not reachable. Scoring and the faithful predictor unchanged.

It removed the large circling arcs: C3 seed 130006 went from 12.98x to 1.00x and C2/130065 and C3/130063
stayed clean. It failed the population gates (300 paired passive-A0 seeds against the committed predictor):

| | foundation | variant 1 | variant 2 |
|---|---:|---:|---:|
| Collisions gained (C3) | - | 4 (130017, 130030, 130048, 130096) | 1 (130030) |
| Episodes above 5x (C2+C3) | 1 | 8 | 10 |
| C2 mean / max path-excess | 0.98 / 1.19 | 1.34 / 27.9 | 1.34 / 27.9 |
| C3 mean / max path-excess | 1.21 / 12.98 | 2.00 / 28.4 | 2.37 / 31.5 |
| Avoidance activations C2 / C3 | 10 / 64 | 94 / 236 | 94 / 309 |
| Overshoots C2 / C3 | 0 / 14 | 24 / 64 | 24 / 93 |
| Orbit events / non-returning (C3) | 0 / 0 | 1 / 4 | 2 / 1 |

C3 seeds 130081 (0.98x -> 18.8x) and 130064 (1.20x -> 11.8x) show the new failure: the attacker circles
the asset at about 20 m radius, interrupted again and again by short avoidance segments from an obstacle
cluster around the asset (C2 130054: 11 steps -> 346 steps of repeating climb/lateral segments). The new
collisions are all-candidates-collide emergencies at a high descent rate, where the original vertical climb
candidate had been the escape. The root cause therefore also includes the avoid/release/re-activate cycle
of a turn-limited vehicle near obstacle clusters, not only the climb waypoint. Rejected; production code
reverted. Evidence is in the gitignored `results/journal_v2_planner_fidelity/{foundation,forward,forward2}_*`
(figure: `forward_waypoint_before_after.png`).
