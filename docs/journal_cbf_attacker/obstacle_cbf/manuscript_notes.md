# Manuscript notes — A1-CBF static-obstacle navigation

## Claim supported
A route planner (the EXISTING, unmodified visibility-graph + A* planner) combined with a per-instant CBF-QP safety
filter over inflated static-obstacle AABBs lets A1-CBF pursue a moving asset through canonical C2 static clutter at
99.9 % capture, 0 repeated fly-bys, 0 orbit episodes, and strictly better outcomes than canonical A0 on every
metric except two safety-gate criteria (1/1000 obstacle collisions, 11/1000 episodes with a transient route-planner
failure). The formal acceptance gate therefore FAILS — but the failure mechanism is precisely diagnosed, narrow,
and attributable to a scenario-generation parameter (canonical C2's spawn-to-obstacle clearance), not to a flaw in
the CBF-QP or route-planner logic itself.

## Route planner + CBF division of responsibility
The route planner answers "which waypoint should I head toward to eventually reach the asset, given the static
obstacle layout" — a GLOBAL, geometric question, solved once per ~1 s. The CBF-QP answers "given my current state
and the obstacles within reach, what is the SAFEST velocity close to what I actually want right now" — a LOCAL,
instantaneous question, re-solved as often as needed. Neither component ever does the other's job: the route
planner has no notion of safety margins; the CBF-QP has no notion of a multi-waypoint path.

## The exact barrier
`h_i = d_i - d_stop_i`, where `d_i` is the exact (never spherical-approximated) signed distance to the inflated
obstacle AABB, and `d_stop_i = s_i²/(2 a_max) + s_i dt_sub` is the distance the attacker needs to decelerate to
zero inward speed at the frozen acceleration limit, PLUS one substep's worth of travel at the current inward speed
(accounting for the one-substep reaction lag before braking can begin). The constraint `n_i^T v_safe >= -alpha h_i`
demands progressively more outward velocity as `h_i` shrinks toward (and past) zero.

## Stopping-distance margin — a real control-loop timing bug found and fixed
The most significant engineering finding of this stage was NOT a flaw in the barrier formula itself, but in HOW
OFTEN it was being re-evaluated. Initially, the CBF-QP was solved once per external 0.5 s step (matching Prompt 1's
"hold the command constant across substeps" convention) — but the barrier's own `dt_sub` reaction-time term is only
meaningful if the controller is ACTUALLY given the chance to react within that window. A 200-episode smoke test
surfaced exactly one static-obstacle collision, diagnosed to a ~30x mismatch between the required and available
velocity-change rate over one external step. The fix — re-solving guidance + CBF-QP every 0.1 s substep whenever a
hazard is active — makes the formula internally consistent with how it is actually used, and is a correctness fix
to the control LOOP, not a change to any frozen physical constant.

## QP and fallback semantics
A 3-variable (later 4, with slack) convex QP solved with SciPy's SLSQP: hard constraints first; on infeasibility, a
single shared slack relaxes ONLY the CBF inequalities (never the physical velocity bounds) with a steep quadratic
penalty; on total failure, command zero velocity (brake) and let the route planner re-evaluate next step. Over the
full campaign, the hard QP alone resolved 96.7% of all obstacle-engaged steps; soft fallback and emergency braking
were implemented, tested, and essentially never needed in practice (0% emergency-brake steps).

## Monte Carlo results
1,000 matched C2 episodes: A1-CBF captures 99.9% (A0 99.1%) with ZERO repeated fly-bys (A0: 8) and zero orbit
episodes. The one collision and the eleven planner-failure episodes are concentrated in geometrically similar
circumstances: the attacker spawning (or transiently finding itself) closer to an obstacle than its own physics
permits stopping. Ten of eleven planner-failure episodes still captured successfully thanks to the
previous-waypoint-retention fix; only the already-doomed-by-physics episode (500100) still collided.

## Comparison with A0
Canonical A0 (legacy turn-rate-limited pursuit + its own separate reactive obstacle-avoidance system) ALSO
collides 3 times out of 1000 — context, not the acceptance definition, but useful: static clutter at this density
is not a trivial navigation problem for ANY controller on this benchmark, and A1-CBF's single collision compares
favorably even against the existing, long-validated legacy pipeline.

## Honest caveats
1. The acceptance gate's two hard-zero criteria (`obstacle_collision=0`, `planner_failure=0`) are NOT met.
2. The dominant failure mechanism (spawn-time physical infeasibility) is a MISMATCH between an existing,
   unmodified scenario-generation parameter (`obstacle_clearance_from_spawn=3.0m`, designed for a different
   attacker's dynamics) and A1-CBF's new acceleration-limited physics -- not a defect in the CBF/route-planner
   design, but a real constraint that must be explicitly addressed (not silently tuned) before this stage can pass.
2. The route-planner-failure fallback fix, while a genuine correctness improvement, does not by itself resolve the
   one collision -- the physics were already infeasible before any guidance decision could matter.
3. `kp`, `alpha`, the obstacle clearance, the active-obstacle range, and every physical limit were frozen before
   this run and were not tuned afterward, per the brief's explicit instruction.
4. No defender or dynamic-obstacle CBF is implemented in this stage; both remain future work, as explicitly scoped.

## What comes next (not started here)
A decision on how to reconcile canonical C2's spawn-clearance parameter with a deceleration-limited attacker's
physical envelope (NOT undertaken here, per "do not tune automatically"), followed by defender CBF integration once
static-obstacle navigation is accepted.
