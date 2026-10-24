# V2 attacker: terminal capture speed governor (rejected)

Starting research HEAD for this task: `873026ea15d0794bc45bf767af2df3aaa45e49fa` (the accepted
stateful receding-horizon obstacle-avoidance controller). Production code at the end of this task
is byte-identical to that commit (`git diff 873026ea -- uav_defend` is empty); only this document
and a new diagnostic (below) were kept.

## Root cause of the remaining C3 130006 tail -> actually C3 130017

After the stateful avoidance controller was accepted, the only remaining >5x path-excess case in
the 300-seed population was C3 seed `130017` (27.06x, 347 steps, only ~2 of which are obstacle
avoidance). Detailed per-step diagnostics (position, velocity, speed, heading, range,
line-of-sight bearing error, closing velocity, turn used) confirmed this is a TERMINAL GUIDANCE
problem, not an avoidance-loop problem:

1. the attacker executes a short, correct avoidance maneuver and approaches the asset;
2. it passes the asset by about 2.9-3 m -- already inside the turn radius implied by its speed
   and maximum turn rate (`speed / max_turn_rate_rad`, about 8 m at full 12 m/s cruise speed, once
   the turn-vs-acceleration coupling settles the sustained-turn speed down to roughly 10.5 m/s);
3. unchanged full-speed DIRECT pursuit (mission command re-aimed at the asset every step, speed
   always `v_e`) cannot realign quickly enough and sustains a wide, repeating fly-by: distance to
   the asset oscillates between about 3 m and 35 m, with the attacker's bearing about the asset
   sweeping through many revolutions over hundreds of steps, never entering the 2 m capture
   sphere until step 346;
4. `orbit_revolutions` (the existing orbit metric, fixed 20 m radius, resets its cumulative-bearing
   accumulator every time the attacker leaves that radius) reports **zero** for this episode,
   because the attacker repeatedly swings out past 20 m between near misses -- it never completes
   one continuous revolution within a single sub-20-m excursion, even though it is clearly
   loitering around the asset. This diagnostic gap is fixed separately (see below).

## Attacker speed/dynamics audit (Phase 2)

* Nominal/maximum attacker speed: `v_e = 12.0 m/s` (`EnvConfig.v_e`), always commanded as
  `desired_velocity = v_e * u` in `SoldierEnv._move_enemy` -- there is no existing notion of a
  commanded speed other than full speed.
* No minimum speed or hover constraint exists for the attacker anywhere in
  `uav_defend/config/env_config.py` or `uav_defend/dynamics/constrained_point_mass.py`; the point
  -mass dynamics model (`advance_velocity`) already accepts any `desired_velocity` magnitude,
  including zero, so a speed governor's search range is legitimately `[0, v_e]`.
* `enemy_max_accel = 6.0 m/s^2`, `enemy_max_turn_rate_deg = 75 deg/s`, `enemy_max_climb_rate =
  enemy_max_descent_rate = 5.0 m/s`, `dt = 0.5 s`. Speed is never commanded directly by any
  existing layer; it only emerges from `advance_velocity`'s acceleration-limited blend toward
  whatever `desired_velocity` is requested, exactly as a generic point-mass speed command would.
* Climb/descent do not interact with achievable horizontal speed in this model (vertical rate is
  clipped independently of the horizontal turn-rate/acceleration blend).

## Design tried: predictive terminal speed governor

Hierarchy: physical safety > obstacle avoidance > terminal speed governor > direct pursuit. The
governor never changed the commanded DIRECTION (still `current attacker -> current asset`,
`_mission_direction`, re-derived every step); it only scaled the commanded SPEED, and only while
the stateful avoidance controller was not active (DIRECT mode).

Shared, execution-faithful rollout architecture (no duplicated equations): a new
`_executed_step_at_speed` generalized `_executed_step` with an explicit commanded speed decoupled
from the dynamics' `max_speed` bound (`_executed_step` became the `speed == max_speed` case,
bit-identical to before). A new `uav_defend/obstacles/terminal_governor.py` module used this plus
`_mission_direction` to roll out candidate speeds and predict swept capture against the (for the
rollout's duration, position-fixed) asset using the same `first_contact` primitive the real
simulation uses for attacker -> asset contact.

* **Activation**: every DIRECT step, roll out the nominal full-speed trajectory. If it predicts
  capture, or if its closest approach to the asset is not strictly inside the horizon (nothing
  useful to decide this step), the governor stays inactive and commands full speed -- bit-
  identical to no governor. Otherwise (a predicted fly-by) it searches for a better speed.
* **Speed search**: a deterministic, bounded 1 m/s grid from `v_e` down to `0`, each rolled out and
  checked for (a) swept capture and (b) no collision -- and no reduced clearance margin below the
  existing `enemy_obstacle_clearance` (2 m) -- against the currently perceived obstacle layout
  (physical safety is never overridden by the governor; a slower candidate that the avoidance
  controller's own full-speed trigger never evaluated could otherwise newly clip an obstacle).
  Among capturing-and-safe candidates, the SOONEST predicted capture step was preferred over the
  highest speed (ties broken by the higher of those) -- not literal highest-speed-that-captures --
  because a capture predicted many steps into a rollout that assumes a motionless asset is the
  least trustworthy one (the asset in fact keeps moving at up to `v_s = 1.5 m/s`); bisecting to an
  exact highest-speed boundary was tried first and made this worse, not better (below).
* **No-feasible-speed fallback**: lexicographic (safety first: never prefer a colliding candidate;
  then smallest predicted closest-approach distance; then closing rather than opening at the end
  of the rollout; then smaller heading error at the closest-approach point; then the highest
  speed).
* **Persistence**: no stored waypoint or timer; recomputed from the current state every step. The
  only retained state was the previous target speed, used solely to break an exact tie in favour
  of not changing it.

## Why it was rejected

Four independent variants were tried (highest-capturing-speed with bisection; earliest-capture
preferred, no bisection; the same plus a minimum 2-step look-ahead before activating; the same
plus a clearance-margin safety requirement instead of bare non-collision). Against the accepted
foundation (`873026ea`), **every variant fixed C3 130017's path-excess** (27.06x -> 2.25x-3.88x
depending on the variant) **without changing any other trajectory in the 300-seed population**
(C0/C1 identical; C2 identical; only `C3_130017.json` changed in every variant). But in **every
variant**, 130017 **gained a new collision** (`enemy_static_obstacle_collision`) later in the same
episode, at a different position each time depending on exactly which step the governor first
intervened.

Investigation traced the mechanism precisely: the governor's one-step speed change alters the
attacker's subsequent state enough that the (otherwise unmodified) pursuit dynamics are sensitive
to it -- a materially different trajectory results many steps later, independent of how large or
small the intervention is (an 11 m/s, barely-below-full-speed intervention produced the same
outcome as a 3 m/s one). That different trajectory later revisits the stateful avoidance
controller's documented "all candidates collide" emergency path (`fallback_used`), which this time
resolves into an actual physical collision -- confirmed directly in the trace (`stateful.fallback`
is set on the colliding step). The ORIGINAL (ungoverned) 130017 trajectory also repeatedly hits
this same emergency path during its long orbit without ever colliding; the governed trajectories
hit it in a different place and are not so fortunate. This is not a bug introduced by the governor
in the sense of an incorrect single-step command -- every per-step rollout the governor acted on
was itself verified collision-free and within the required clearance margin at the moment it was
chosen -- it is a genuine sensitivity of this long, chaotic pursuit-and-avoidance interaction to
any change in the approach trajectory, exposing a pre-existing edge case in the ACCEPTED (and, per
this task's hard restriction, unmodifiable) obstacle-avoidance controller's emergency handling
that the original 300-seed population never happened to trigger into an actual collision.

Given the hard gate ("reject if any paired foundation-safe episode gains a collision") is written
as a bright line and 130017 is the one case where it is violated in **every** tested variant, with
clean negative evidence (no variant does better), this design is rejected. Production code has
been reverted to `873026ea` (bit-identical).

## Terminal orbit/fly-by diagnostic (kept)

`experiments/journal_v2_validation_core.terminal_flyby_metrics(rec)` is a diagnostic-only addition
(never used to steer the vehicle) that detects a sustained post-avoidance fly-by/orbit which the
existing fixed-20-m-radius `orbit_revolutions` metric can miss (see root-cause section above).
Computed over the TERMINAL WINDOW (from the end of the last avoidance segment, or episode start if
none, to the end of the episode), independent of any fixed radius:

* `fly_by_count`: local-minima-of-range events (closest approaches) that do not reach the capture
  sphere;
* `angular_travel_revolutions`: cumulative unsigned bearing change about the asset over the whole
  terminal window, in revolutions (never reset by a fixed radius);
* `closing_sign_changes`: sign changes of the range rate (supporting evidence only);
* `detected`: `fly_by_count >= 3` AND `angular_travel_revolutions >= 1.5` (both signals, chosen
  comfortably above what a single normal curved intercept produces).

Calibration: `detected` is `True` for C3 130017 at `873026ea` (`fly_by_count = 22`,
`angular_travel_revolutions` far above the threshold) and `False` for every other recorded named
case (C2 130065, C3 130063/130081/130064, C2 130054, C0 130000) -- see
`tests/test_journal_v2_validation.py::test_terminal_flyby_detects_the_known_sustained_fly_by_case`
and `::test_terminal_flyby_does_not_flag_normal_captures`.

## Open problem

C3 130017 (and, by the mechanism above, potentially other seeds under a sufficiently different
approach trajectory) remains an unresolved terminal pursuit-geometry failure. Fixing it safely
appears to require either (a) a genuine terminal GUIDANCE (not merely speed) correction near the
asset -- explicitly out of scope for this task, which was restricted to a speed-only governor --
or (b) hardening the obstacle-avoidance controller's "all candidates collide" emergency handling,
which this task was explicitly restricted from redesigning. Both are legitimate directions for a
future task; this document and the `terminal_flyby_metrics` diagnostic are the evidence base for
it.
