# Hostile Command-Generation Pipeline Audit (Phase 2)

Traced directly from `uav_defend/envs/soldier_env.py::_move_enemy()` (and its
helpers `_compute_enemy_evasion()`, `uav_defend/obstacles/route_planner.py::
RoutePlanner.plan()`, `uav_defend/obstacles/avoidance.py::
ObstacleAvoidancePlanner.plan()`) as of commit `3668048` (tag
`journal-hostile-route-planner-v1`). No behavior was changed to produce this
document.

## Control-flow diagram

```
                         ┌─────────────────────────────┐
                         │ 1. Mission pursuit (r_hat)   │  r = soldier_pos - enemy_pos
                         │   ALWAYS computed first       │  r_hat = r / ||r||
                         └───────────────┬───────────────┘
                                         │ (overwrite candidate)
                                         v
                  ┌───────────────────────────────────────────┐
                  │ 4. Route planner (RoutePlanner.plan)        │  if enabled:
                  │   OVERWRITES r_hat with route direction     │  r_hat <- route waypoint dir
                  │   (own persistent RouteState + replan       │  (else r_hat unchanged = mission)
                  │    triggers = its own hysteresis)           │
                  └───────────────┬───────────────────────────┘
                                  │ r_hat (mission OR route)
                                  v
        ┌──────────────────────────────────────────────────────────┐
        │ 2+3+ evasion: ADDITIVE combination (same instant, one line)│
        │   weave_component  = weave_bias * amplitude * lateral(r_hat)│
        │   noise_component  = sigma_e * N(0,I)                       │
        │   evasion_component= evasion_gain * alpha(d) * away_dir     │
        │                      (defender_pos, enemy_pos; SMOOTH,      │
        │                       no hard on/off -- own analog "hysteresis")│
        │   u_base_raw = r_hat + weave + noise + evasion              │
        │   u_base = normalize(u_base_raw)                            │
        └───────────────────────────────┬────────────────────────────┘
                                        │ u_base (single combined command)
                                        v
        ┌──────────────────────────────────────────────────────────┐
        │ 5+6. Reactive obstacle avoidance (ObstacleAvoidancePlanner) │
        │   STATIC + DYNAMIC merged into ONE avoidance_layout list    │
        │   (NOT two sequential stages -- one call handles both)      │
        │   IF a collision against u_base is predicted within         │
        │   horizon_steps: OVERWRITES u_base with a fixed world-frame │
        │   bypass waypoint direction, held via its OWN hysteresis    │
        │   (release_steps consecutive clear steps before releasing)  │
        │   ELSE: passes u_base through unchanged (bit-for-bit)       │
        └───────────────────────────────┬────────────────────────────┘
                                        │ u (final commanded unit direction)
                                        v
        ┌──────────────────────────────────────────────────────────┐
        │ 8. Vehicle dynamics (_advance_velocity)                     │
        │   desired_velocity = v_e * u                                │
        │   next_vel <- rate-limited toward desired_velocity          │
        │   (accel / turn-rate / climb / descent limits)              │
        │   position += next_vel * dt; boundary reflection             │
        └──────────────────────────────────────────────────────────┘
```

## Per-stage table

| # | Stage | Input | Output | Overwrite vs. modify | Activation | Persistence | Reference target | Changes | Own hysteresis? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | Mission pursuit | `enemy_pos`, `soldier_pos` | `r_hat` (unit vector) | N/A (first stage; candidate for stage 4 to overwrite) | Always | None -- recomputed fresh every step (zero-lag) | `soldier_pos` (moving, live) | Heading only | No |
| 2 | Stochastic weaving | `r_hat` (lateral axis), `weave_bias` (AR(1) state), RNG | `weave_component` (added, not overwrite) | Additive | Always active | `weave_bias` persists across steps via AR(1): `a <- rho*a + sigma_a*eta` | None (perturbation) | Heading (small lateral bias) | No (has memory, not hysteresis) |
| 3 | Defender-reactive evasion | `defender_pos`, `enemy_pos` | `evasion_component` (added, not overwrite) | Additive | Smooth, continuous: `alpha(d)=clip(1-d/R_e,0,1)` -- no hard on/off | None -- recomputed fresh every step from current distance | `defender_pos` (away from, live) | Heading (magnitude scales with proximity) | No explicit switch, but smooth activation avoids hard toggling |
| 4 | Route-planner waypoint pursuit | `enemy_pos`, `soldier_pos` (as `true_target`), `PerceivedObstacleSet` | Replaces `r_hat` entirely (if route direction norm > eps) | **Overwrites** `r_hat` (stages 2/3 then combine with the ROUTE direction, not raw mission) | Gated on `enemy_obstacle_avoidance_enabled and enemy_route_planning_enabled` | Strong: `RouteState` persists across many steps; replanned only on explicit triggers (`no_active_route`, `route_unsafe_or_new_obstacle`, `target_moved`, `progress_stall`, `route_complete`) | `local_goal` (planning-horizon-clamped projection of `soldier_pos`, NOT always identical to live target) | Heading, waypoint, and altitude (vertical bypass) | **Yes** -- its own full state machine (persistent route + stall detector + widened-clearance loop suppression) |
| 5/6 | Static + dynamic obstacle avoidance | `u_base` (fully combined mission+weave+noise+evasion), `position`, `velocity`, merged `avoidance_layout` | `plan["direction"]` | **Overwrites** `u_base` entirely when active; passes through unchanged (bit-for-bit) otherwise | Predicted collision against `u_base`'s trajectory within `horizon_steps` | Fixed world-frame bypass waypoint, held via `release_steps` consecutive clear-step counter | A chosen bypass waypoint (world-frame, fixed until released) | Heading and altitude (3D bypass waypoint) | **Yes** -- explicit `_clear_streak`/`release_steps` hysteresis (documented in `ObstacleAvoidancePlanner`'s own docstring) |
| 7 | (No separate top-level hysteresis stage) | -- | -- | -- | -- | Hysteresis lives INSIDE stages 4 and 5/6 individually, not as a separate arbitration layer | -- | -- | See above |
| 8 | Vehicle dynamics constraints | Final `u`, current velocity | `next_vel`, new position | Converts direction -> physically achievable velocity | Always | Velocity itself is the persistent state being advanced | None (kinematic limiter only) | Converts "desired direction" into rate-limited actual motion | No (rate limits, not mode hysteresis) |

## Key structural observation (root of the Phase-4/Finding-5 conflict)

Stages 5/6 (reactive obstacle avoidance) can **silently override** the final
commanded direction on any given step, but stage 4 (the route planner) has
**no visibility into this override** -- `RoutePlanner.plan()` only ever sees
`(position, true_target, perceived, awareness_mode)` and decides whether to
replan based on **actual position progress**, never on whether its own
`direction` output was actually followed. Concretely:

- If reactive avoidance repeatedly overrides the route's chosen direction
  (e.g. because evasion pushed the attacker toward the SAME obstacle the
  route was trying to route around), the attacker's real trajectory
  diverges from the route's plan.
- The route planner then detects this via `progress_stall` (goal distance
  not shrinking) or `route_unsafe_or_new_obstacle` (its stored waypoints no
  longer look safe from the new, evasion-perturbed position) and replans --
  but the NEW route is built from a position that itself resulted from an
  evasion-driven detour, so the new route may again point back toward
  where evasion just pushed the attacker away from, repeating the cycle.
- Evasion (stage 3) is **always additive and always active near the
  defender**, so it never "gets out of the way" for the route planner the
  way the route planner's own replanning logic implicitly assumes a stable
  environment between replans.

This matches the empirically observed replan-reason distribution from the
prior Monte Carlo debugging (`progress_stall` and `route_unsafe_or_new_
obstacle` together accounted for >95% of replans in the worst-case seeds),
and explains why disabling evasion closed most (not all) of the
route-planner-vs-baseline gap.
