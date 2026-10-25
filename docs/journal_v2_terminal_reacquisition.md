# V2 attacker: reactive terminal reacquisition (rejected)

Starting research HEAD for this task: `607b28a5c04223170e015d3f0c01ee80dba09cd4` (diagnostics/
tests/docs only beyond the accepted production commit `873026ea15d0794bc45bf767af2df3aaa45e49fa`).
Production code at the end of this task is again byte-identical to `873026ea`
(`git diff 873026ea -- uav_defend` is empty); only this document and a diagnostic module/tests
were kept.

## Phase 1: characterize the observed fly-by (accepted production, no changes)

Reproducing C3 seed `130017`: the last obstacle-avoidance exit is at step 19; the first failed
close pass occurs at **step 31**:

| quantity | value |
|---|---|
| range | 2.90 m (missed the 2.0 m capture sphere by 0.9 m) |
| closing velocity | previous step +6.95 m/s (closing) -> this step -1.35 m/s (opening) |
| attacker speed | 10.56 m/s |
| LOS bearing / attacker heading | -62.4 deg / -159.7 deg -> heading error 97.3 deg (already pointing away) |
| `R_turn = speed / max_turn_rate_rad` | 8.07 m |
| `capture_radius + R_turn` (relevance threshold) | 10.07 m |
| nearest obstacle clearance | 3.95 m (not obstacle-constrained) |
| obstacle mode | DIRECT throughout |

The closing-to-opening transition is observable in real time (it is computed from the CURRENT,
already-post-turn velocity, not a future one), so a causal detector can fire at the same step
rather than one step late. The miss (2.90 m) is well inside the relevance threshold (10.07 m),
confirming this is exactly the kind of "close enough that the turning circle matters" fly-by the
detector is meant to catch.

## Design tried: reactive terminal reacquisition

A new `TERMINAL_REACQUIRE` mode, activating ONLY after the detector above fires (never
preemptively, never from a prediction):

* **Detector** (`flyby_detected`): `previous_closing > 0`, `closing_now <= 0`, no capture, and
  `range <= capture_radius + R_turn`.
* **Direction while active**: unchanged pure current-position pursuit
  (`normalize(asset_position - attacker_position)`) -- no lead point, no waypoint.
* **Speed while active -- Variant A (instantaneous curvature law)**:
  `v_cmd = min(v_e, max_turn_rate_rad * max(range - capture_radius, 0))`, the highest speed whose
  own turning radius fits the remaining range. Actual speed reaches this only through the existing
  acceleration limit.
* **Speed while active -- Variant B (deceleration-aware)**: accounts for the extra distance
  consumed while decelerating from the current speed to the target before the turn can begin, by
  solving `v_target^2 - (2*accel/omega_max)*v_target + 2*accel*C >= 0` (`C = (range -
  capture_radius) - current_speed^2/(2*accel)`) for the highest feasible `v_target`.
* **Local obstacle-safety entry gate**: never activates on a step where the currently perceived
  nearest-obstacle clearance is already below the existing `enemy_obstacle_clearance` margin (the
  same physical quantity the accepted avoidance controller treats as the comfortable-clearance
  boundary -- no new threshold invented).
* **Exit** (`reacquisition_achieved`): recomputed every step from current geometry only -- closing
  again (`closing_now > 0`) with heading error `<= 90` degrees (broadly facing the asset again).
  Obstacle avoidance overrides this controller unconditionally and immediately; it does not
  auto-resume afterwards, only re-triggers on a genuine new fly-by.

Both variants were tried against C3 130017 directly (Phase 10/11's explicit instruction: stop
immediately and do not touch production further if the diagnostic fails on this seed, with at
most one additional variant).

## Result: both variants fail on the one seed they target

| | episode length | path-excess | outcome |
|---|---:|---:|---|
| accepted foundation (no recovery) | 346 steps | 27.06x | `soldier_caught` |
| Variant A (instantaneous) | 59 steps | 5.31x | **`enemy_static_obstacle_collision`** |
| Variant B (deceleration-aware) | 114 steps | 10.25x | **`enemy_static_obstacle_collision`** |

Both variants dramatically shorten the episode and reduce path-excess (Variant A gets to within
sight of the `<5x` target), but **both introduce a new collision in the same episode** -- and
Variant B is strictly worse on every other axis too (slower convergence, larger excess), so there
is no reason to prefer it. The collision in the Variant A run occurs at step 59 while obstacle
avoidance IS active (`behavior_mode` other than DIRECT), in the same class of event as the two
previously rejected designs: the perturbed trajectory revisits the stateful avoidance controller's
"all-candidates-collide" emergency path in a place the original (unperturbed) 130017 trajectory
never visited.

This is now the **third independently-designed terminal-guidance mechanism** -- a speed-only
governor (predictive), a lead-intercept controller (predictive), and this purely reactive
curvature-speed recovery (no prediction at all) -- that converges on the identical failure mode
when applied to this one seed: altering the attacker's approach trajectory in ANY way, regardless
of the guidance law used to do it, risks exposing the one known pre-existing gap in the
(unmodifiable, per this task's hard restriction) obstacle-avoidance controller's emergency
handling. The convergence across three structurally different designs is strong evidence that the
problem is not a property of any one guidance law's math, but of the chaotic sensitivity of this
specific seed's pursuit-avoidance interaction to any trajectory perturbation at all, combined with
a latent (rare, previously unexercised) weakness in the avoidance controller's emergency fallback.

Per the task's explicit stopping rule ("If it does not solve 130017 safely: do NOT modify
production. Return verdict D" and "Maximum experimental variants in this task: Variant A; Variant
B. No grid search."), production integration, named-case regression, the 400-seed population
evaluation, and Greedy/Lead invariance were not run -- the diagnostic result on the primary target
seed already fails the stop condition for both permitted variants.

Production code has been reverted to `873026ea` (bit-identical). The diagnostic module
(`experiments/terminal_reacquisition_diagnostic.py`, the pure detector/curvature-law geometry,
never wired into `uav_defend/`) and its unit tests
(`tests/test_terminal_reacquisition_diagnostic.py`) are kept, since the geometry itself is correct.

## Open problem (unchanged across all three rejected designs)

C3 130017 remains the single unresolved terminal-pursuit case in the accepted population. Three
different local, receding-horizon terminal-guidance mechanisms -- predictive-speed, predictive-
lead, and purely reactive-speed -- have each been tried and each exposes the same pre-existing
avoidance-controller emergency-handling gap once the trajectory is perturbed enough to meaningfully
shorten the fly-by. A durable fix likely requires either (a) hardening the avoidance controller's
"all candidates collide" emergency rule itself (explicitly out of scope for all three of these
tasks), or (b) a route-level/global planning approach that reasons about the whole remaining
trajectory rather than one receding-horizon decision at a time, as this task's final verdict
option anticipates.
