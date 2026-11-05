# Manuscript notes — A1-CBF viability-conditioned clutter benchmark

## Claim supported
> The initial C2 evaluation exposed rare states outside the viability region of the acceleration-limited attacker.
> Rather than tuning the controller or changing obstacle geometry, the revised protocol conditions only attacker
> initialization on braking feasibility while leaving canonical C2 clutter unchanged.

This single change — resampling only the attacker's spawn/heading, never the obstacle field, never any frozen
controller parameter — eliminates the one static-obstacle collision found in the Prompt-2 campaign, while a
companion fix to route-status handling (hover/brake when no safe target is available, rather than aiming at a
known-blocked one) safely resolves the 11 Prompt-2 planner-failure episodes. The frozen acceptance gate now passes
on all 13 criteria.

## Prompt-2 failure, restated
Prompt 2's 1/1000 collision (seed 500100) was traced to a precise, reproducible mechanism: the attacker's spawn
velocity pointed almost directly at a static obstacle whose inflated-boundary clearance was only ~7.96 m, while the
theoretical minimum stopping distance at the attacker's spawn speed (12 m/s) under the frozen deceleration limit
(6 m/s²) is ~12–13 m. `h_i` was already negative at the very first simulated instant — before the CBF-QP had any
chance to react. No velocity-command controller bound by the same acceleration limit can avoid this; it is a
mismatch between canonical C2's `obstacle_clearance_from_spawn=3.0 m` (designed for a different attacker's
dynamics) and this attacker's actual physical envelope.

## Viability equation
`d_i >= s_i²/(2·a_max) + s_i·dt_sub` for every static obstacle, `s_i = max(0, -n_i·v_0)`. This is deliberately THE
SAME formula and the SAME hazard-construction code (`build_static_hazards`) used by the live, in-flight CBF-QP —
not a new, independently-derived criterion — so the initialization gate and the real-time safety filter can never
disagree about what counts as "feasible." No additional buffer was added, per the brief's explicit instruction.

## Distribution-bias audit
Across 10,000 canonical-C2 scenarios, only 3.0% of attacker spawns required resampling (mean 1.05 resamples among
those, max 2; 0 generation failures). The obstacle field and asset trajectory are bit-identical between the
conditioned and unconditioned runs for every single seed — confirmed by direct measurement, not merely by code
inspection. The accepted attacker-state distributions (range, speed, heading, nearest-obstacle distance) shift by
under 2% in every statistic reported, confirming the conditioning removes a narrow, rare slice of initial states
rather than reshaping the benchmark.

## Route-failure audit
All 32 planner-failure events from the 11 Prompt-2 episodes were replayed and classified. Zero were implementation
bugs (no exceptions, no malformed output, no contract violations) — every one was a `start_inside_inflated_aabb` or
`goal_inside_inflated_aabb` (moving-goal transient) case, exactly matching the EXISTING, unmodified route planner's
own documented behavior for a geometric search failure. This grounded the decision NOT to touch
`build_and_search_route` itself, and to instead fix only how the calling code (`CBFRouteNavigator`) responds to
such an event.

## Route-status semantics and the one intentional behavioral change
Three explicit states — `route_ok`, `route_temporarily_unavailable`, `route_error` — replace the previous
undifferentiated "planner_failed" label. The acceptance gate requires `route_error=0` (achieved: 0/1,000). The ONE
behavioral change from Prompt 2 is what happens when a route is temporarily unavailable AND no previous waypoint
can be retained: Prompt 2 fell back to the (still-blocked) direct asset target; this stage commands hover/brake
(`v_nom=0`) instead. This is explicitly sanctioned by the brief as an intended difference, and is isolated
precisely to the handful of seeds where it applies (confirmed directly: exactly the Prompt-2 planner-failure seeds
diverge from the cached Prompt-2 trajectory when `viability_conditioned_init=False`, and nowhere else).

## Monte Carlo results
1,000 fresh matched episodes: 100.0% capture, 0 obstacle collisions (down from 8 even for canonical A0's own
reactive avoidance on this exact seed block), 1 repeated fly-by (down from 9), 0 route errors, 0 scenario-generation
failures. The CBF's own barrier was never actually violated in flight (minimum observed `h` was 0.058, always
positive) — the controller operated with genuine safety margin throughout, not merely "by luck."

## Near-viability-boundary stress test
A dedicated, separate diagnostic (120 synthetic states, 101 landing in the required 0–0.5 m margin window,
constructed from real C2 layouts) confirms the controller is not merely being handed easy starting conditions by
the viability rule: even states engineered right at the edge of feasibility are navigated with zero collisions and
zero numerical failures.

## Honest caveats
1. `scenario_generation_failure` is implemented and tested but never observed in 11,000+ combined scenarios —
   canonical C2's geometry gives ample heading diversity at the spawn perimeter.
2. The fix is intentionally narrow: it does not alter obstacle placement or any frozen CBF/route parameter, so it
   cannot and does not address denser obstacle configurations than canonical C2 presents.
3. No defender avoidance and no dynamic-obstacle CBF: out of scope, as explicitly directed.
4. Visualizer: data recordings only, not app integration (see README section 13 for the direct architectural
   justification).

## What comes next (not started here)
Defender CBF integration, now that static-obstacle navigation is accepted (verdict A). No canonical tag is created
for A1-CBF at this stage.
