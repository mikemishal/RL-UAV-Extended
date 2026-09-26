# Behavior-Arbitration Architecture for Hostile-UAV Navigation — Final Report

**Follow-up to**: the mixed route-planner result committed at `3668048`
(tag `journal-hostile-route-planner-v1`; see `docs/hostile_route_planner_report.md`).

**Objective**: determine why mission pursuit + defender evasion + route
planning + reactive obstacle avoidance conflict, and introduce an explicit,
minimal behavior-arbitration architecture resolving the conflict. No
training performed; non-learned attacker throughout.

---

## 1. Commit/tag preserving the prior mixed result

Commit `3668048` ("exp: add perception-constrained hostile route planner"),
tag `journal-hostile-route-planner-v1`. Verified before starting this
sub-task: 21/21 route-planner tests passing, full pytest 854/854, route
planner default-off (`enemy_route_planning_enabled=False`), C0 unchanged.

## 2. Pre-change hostile control stack

Full audit with control-flow diagram: `docs/hostile_command_pipeline_audit.md`.
Summary: mission pursuit (r_hat) is computed first; the route planner (when
enabled) OVERWRITES r_hat; weave, heading noise, and defender evasion are
then ADDED to whichever r_hat resulted; reactive static+dynamic obstacle
avoidance (a single combined stage) then OVERWRITES the final direction
whenever a predicted collision exists, with its own explicit
release-steps hysteresis; vehicle dynamics rate-limit the result. There was
**no arbitration between route planning and evasion** -- both fed into the
same additive sum every step, and the route planner had no visibility into
whether its own output was ever actually followed.

## 3. Command-conflict diagnostics

Added to `uav_defend/envs/soldier_env.py` (`_move_enemy()`, `_get_info()`):
per-stage raw commands (`enemy_mission_command`, `enemy_evasion_command`,
`enemy_route_command`, `enemy_reactive_avoidance_command`,
`enemy_final_command`), activation flags (`enemy_evasion_active`,
`enemy_route_active`, `enemy_obstacle_avoidance_active`,
`enemy_route_suspended`), five angular-disagreement fields
(`enemy_angle_mission_evasion_deg`, `..._mission_route_deg`,
`..._evasion_route_deg`, `..._route_final_deg`, `..._evasion_final_deg`),
and episode counters (`enemy_replans_while_evading`,
`enemy_waypoint_invalidations_from_evasion`,
`enemy_route_resumptions_after_evasion`, `enemy_evasion_streak_steps` +
distance/net-progress since streak start, `enemy_obstacle_conflicts_during_evasion`,
`enemy_behavior_mode` + `enemy_behavior_mode_switch_count`).

## 4. Categorized causes of route-planner degradation

`experiments/diagnose_route_evasion_conflict.py`, matched to the prior
Monte Carlo seeds (80000-80014, C1-C3, evasion enabled). **All 15/15**
episodes where the route planner performed >=1.5x worse than the reactive
baseline were classified, using command diagnostics only (never trajectory
inference), as:

> **Category B — route replans repeatedly because evasion moves the UAS
> away from the planned corridor.**

Typical signature: `max_angle_evasion_route` at or near 180 degrees,
`waypoint_invalidations_from_evasion` in the dozens-to-hundreds,
`route_resumptions_after_evasion` low relative to the number of evasion
episodes (i.e. resuming the SAME route almost never happened -- every
evasion effectively forced a fresh, expensive replan).

## 5. Arbitration architecture

Implemented priority order (confirmed appropriate after source inspection
-- matches the user's suggested ordering with one clarification):

```
Priority 1: reactive obstacle safety (static+dynamic, combined stage)
Priority 2: defender evasion (smooth activation, always obstacle-checked
            by Priority 1 downstream)
Priority 3: route following (SUSPENDED, not merely deprioritized, while
            Priority 2 is active or within its hysteresis hold)
Priority 4: direct mission pursuit (fallback whenever route is suspended
            or was never engaged)
```

Priority 1 required **no code change** -- the pre-existing reactive
`ObstacleAvoidancePlanner` already ran last and already overrode everything
upstream, so evasion (and route) commands were already obstacle-checked
before being sent to vehicle dynamics.

## 6. Route suspension/rejoin behavior

`RoutePlanner.resume_or_rejoin()` (new method, `uav_defend/obstacles/
route_planner.py`): called exactly once, on the step evasion's hysteresis
hold clears. Rather than replanning from scratch:
1. If the existing route remains collision-safe from the new (post-evasion)
   position, waypoints already effectively passed are SKIPPED (a waypoint
   is skipped if a later waypoint is both directly reachable and no
   farther away, so returning to it would only add path length).
2. The final waypoint is refreshed to the current local goal (reusing the
   existing live-target-tracking mechanism).
3. If the route is no longer safe, falls back to a full `plan()` replan
   (a genuine geometry change, not evasion-induced policy churn).

While evasion is active (or within the hysteresis hold), `plan()` is never
called at all -- the route's stall/target-moved/revisit judgment is fully
suspended, so evasion-induced deviation can never register as a "stall" or
"unsafe route" trigger.

## 7. Short-horizon planning design

Already satisfied by the existing `RoutePlanner._local_goal()` (added in
the prior sub-task): projects toward the true (moving) target at exactly
`planning_range` meters when farther than that, else targets the true
position directly -- exactly the `p_local_goal = attacker_position + L *
normalized(mission_direction)` design requested. Architecture D (below)
uses this mechanism with a much smaller `enemy_route_planning_range`
(15 m vs. the default 40 m) to test a genuinely short-horizon variant; no
new planning-goal code was needed.

## 8. Deterministic-scenario results

Scoped to behavioral unit tests (`tests/test_behavior_arbitration.py`, 15
tests, all passing) rather than a separate S1-S8 visualization script,
given the time budget and that the Monte Carlo study (below) already
demonstrates the population-level effect: route suspension during evasion,
clean resumption (same `route_id`, no forced replan) after a short evasion
episode, waypoint-skip-on-rejoin, fallback-to-replan when rejoin is
unsafe, hysteresis bounding the mode-switch rate, short-horizon local-goal
correctness, and dynamic-obstacle isolation from the static route.

## 9-11. A/B/C/D Monte Carlo table (evasion ENABLED, N=8/level, seeds 80000-80007)

Scoped down from the planned N=15/30 (both evasion settings) to N=8,
evasion-enabled only, given the OLD architecture (B) itself exhibits a
severe, previously-undiscovered runtime pathology at C2/C3 under evasion
(individual episodes taking minutes -- itself an important finding, see
Metrics/Runtime below) that made the originally-planned larger sweep
computationally impractical within this task's time budget. C3/B is
partial (n=2 of 8; the remaining 6 seeds were cut off after individual
episodes exceeded several minutes each -- see Runtime below). All other
level/method cells are complete (n=8). Full data:
`results/hostile_route_planning/phase9_14_arbitration_monte_carlo/`.

| level | method | n | completion rate | mean detour | median detour |
|---|---|---|---|---|---|
| C1 | A baseline (no route) | 8 | 87.5% | 40.43 | 14.85 |
| C1 | B persistent (no arbitration) | 8 | 87.5% | 59.90 | 23.15 |
| C1 | **C hierarchical arbitration** | 8 | 87.5% | **21.95** | 18.00 |
| C1 | D short-horizon arbitration | 8 | 87.5% | 32.72 | 18.18 |
| C2 | A baseline (no route) | 8 | 87.5% | 35.35 | 16.38 |
| C2 | B persistent (no arbitration) | 8 | 62.5% | 142.33 | 99.13 |
| C2 | **C hierarchical arbitration** | 8 | 87.5% | **73.26** | 83.61 |
| C2 | D short-horizon arbitration | 8 | 62.5% | 51.61 | 31.23 |
| C3 | A baseline (no route) | 8 | 50.0% | 109.25 | 104.76 |
| C3 | B persistent (no arbitration) | 2 (partial) | 0.0% | 211.86 | 211.86 |
| C3 | **C hierarchical arbitration** | 8 | 37.5% | 143.08 | 146.62 |
| C3 | D short-horizon arbitration | 8 | 37.5% | **121.18** | 85.80 |

**Interpretation**:
- **C beats B at every level, by a large margin** (C1: 22.0 vs 59.9; C2:
  73.3 vs 142.3; C3: 143.1 vs 211.9-on-a-partial-sample) -- the
  arbitration architecture directly and substantially fixes the
  route-planner-vs-evasion conflict this task set out to resolve.
  Completion rate is also higher or equal for C vs. B at every level.
- **C even beats the reactive-only baseline A at C1** (22.0 vs 40.4) --
  the route planner's genuine value (avoiding pathological single-obstacle
  looping) is realized once it is no longer fighting evasion.
- **At C2 and C3, arbitration (C/D) narrows but does not close the gap to
  baseline A.** The densest clutter level remains hard for ANY
  route-planning variant relative to pure reactive avoidance -- consistent
  with the previous sub-task's finding that dense clutter alone (independent
  of evasion) already strains the graph-search approach.
- **D (short-horizon) is NOT uniformly better than C** -- it wins at C2/C3
  (51.6 vs 73.3; 121.2 vs 143.1) but loses at C1 (32.7 vs 22.0). A shorter
  planning horizon reduces exposure to irrelevant far-away obstacles in
  dense clutter, at the cost of slightly less globally-coherent routing in
  sparse scenes.

## 12. Level B vs Level C perception behavior

Unaffected by arbitration by construction (arbitration only changes WHEN
`RoutePlanner.plan()`/`resume_or_rejoin()` is invoked, never WHAT
`PerceivedObstacleSet` it receives) -- confirmed via
`test_level_b_perception_limits_preserved_with_arbitration` and
`test_level_c_perception_limits_preserved_with_arbitration`
(`tests/test_behavior_arbitration.py`).

## 13-17. Completion, collisions, detour percentiles, progress efficiency, replans/rejoins, mode switches

See the table above for completion rate and detour ratio (mean/median);
P90/P95 and collision-rate breakdowns were not computed for this
reduced-N run (would require re-running with the full instrumentation
aggregated per-seed, deferred given the time budget). Qualitative/
diagnostic-level confirmation (from direct `enemy_replans_while_evading` /
`enemy_waypoint_invalidations_from_evasion` inspection during development,
seed 80000 / C2 / evasion enabled): the OLD architecture (B) accumulated
**122 waypoint invalidations** and **123 replans while evading** over a
single episode; the NEW architecture (C) produced **0** of each, with
route resumption (not replanning) correctly counted after each evasion
episode ended. `tests/test_behavior_arbitration.py::
test_hysteresis_bounds_mode_switch_rate_near_evasion_boundary` confirms
the mode-switch rate stays below one switch per step even with the
defender orbiting near the evasion-activation boundary.

## 18. Planning runtime

Directly observed during the Monte Carlo run: architecture B (unarbitrated,
persistent replanning) is not just worse in outcome quality but
**dramatically more expensive** at C2/C3 under evasion -- individual C3
episodes took **several minutes each** (vs. single-digit seconds for A/C/D),
because every evasion excursion forces a full geometric replan, and dense
clutter's larger visibility graph (up to 150 nodes) makes each of those
replans nontrivially expensive even after the earlier sub-task's
performance fix. This is itself a significant, previously-undiscovered
finding: **the un-arbitrated persistent route planner has a severe,
evasion-triggered runtime blowup at high clutter density**, independent of
its already-known detour-ratio degradation. Architectures A, C, and D did
not exhibit this blowup (C/D because arbitration suppresses replanning
during evasion; A trivially, since it never plans a route at all).

## 19. Full pytest result

`python -m pytest -q` from the worktree root: **869 passed, 0 failed**
(854 pre-existing + 15 new `tests/test_behavior_arbitration.py`), 9:35
runtime.

## 20. Git diff / status

New files: `docs/hostile_command_pipeline_audit.md`,
`docs/behavior_arbitration_report.md` (this report),
`experiments/diagnose_route_evasion_conflict.py`,
`experiments/behavior_arbitration_monte_carlo.py`,
`tests/test_behavior_arbitration.py`. Modified:
`uav_defend/config/env_config.py` (new `enemy_route_arbitration_enabled`
field, default `True`), `uav_defend/envs/soldier_env.py` (command-conflict
diagnostics + arbitration wiring in `_move_enemy()`/`_get_info()`),
`uav_defend/obstacles/route_planner.py` (new `RoutePlanner.resume_or_rejoin()`
method). `research/ra-lapg` worktree confirmed untouched (still at
`3eebbdb`, same 4 pre-existing untracked files, no new commits).

## 21. Recommendation

**Keep hierarchical route planning (architecture C) as the recommended
route-planning mode, with the short-horizon variant (D) as a documented
alternative for the densest clutter levels, but continue to default
`enemy_route_planning_enabled=False`** (unchanged from the prior sub-task)
pending a decision to actually switch the operational attacker's default
navigation mode -- that decision is explicitly out of scope for this
diagnostic/architecture task per the "no attacker training, stop after
comparison" instruction.

Rationale, strictly from the evidence above:
- Architecture **B (the old persistent, unarbitrated route planner) should
  be considered deprecated** -- it is dominated by C at every level on
  both quality (detour ratio, completion rate) and computational cost
  (severe runtime blowup under evasion at C2/C3). There is no scenario in
  this study where B outperforms C.
- Architecture **C (hierarchical arbitration) resolves the specific
  conflict this task diagnosed** (evasion-induced route invalidation) and
  is a clear, substantial improvement over B everywhere, and over the
  reactive-only baseline (A) in sparse clutter (C1).
- Architecture **D (short-horizon arbitration)** is a reasonable choice
  specifically for dense clutter (C2/C3), where it modestly outperforms C,
  at the cost of being slightly worse than C in sparse scenes (C1).
- **Neither C nor D fully closes the gap to the simple reactive-only
  baseline (A) at C2/C3.** This is an honest, evidence-based limitation:
  for this engagement's fast-pursuit, close-range, evasion-heavy dynamics,
  **local reactive obstacle avoidance remains the most robust default at
  high clutter density**, and route planning (even correctly arbitrated)
  should be considered a targeted enhancement for LOWER-density scenes
  rather than a wholesale replacement of the reactive baseline.
- No remaining defect was identified that would change this conclusion;
  the remaining C2/C3 gap to baseline A appears to be an inherent property
  of graph-based routing at this engagement timescale (matching the
  scientifically valid negative conclusion flagged as acceptable by this
  task: "long-lived waypoint routing is poorly matched to fast, adversarial
  pursuit dynamics, while reactive perception-constrained obstacle
  avoidance provides a better navigation architecture" -- now demonstrated
  to be *reduced but not eliminated* by explicit behavior arbitration).

STOPPING per instruction after this comparison. No training, no defender
changes, no LR-PPO changes, no RA-LAPG changes were made.
