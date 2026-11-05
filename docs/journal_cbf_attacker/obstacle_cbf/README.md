# A1-CBF — static-obstacle navigation (route planner + CBF-QP over canonical C2 clutter)

**Result: the formal acceptance gate FAILS on exactly 2 of 11 criteria — `obstacle_collision` (1/1000 episodes) and
`planner_failure` (11/1000 episodes) — both traced to the SAME diagnosed root cause: canonical C2's
`obstacle_clearance_from_spawn = 3.0 m` is far smaller than A1-CBF's physically-required full-speed stopping
distance (~12–13 m at `v_e=12 m/s`, `a_max=6 m/s²`). Verdict C.** Despite the gate failure, overall performance is
excellent and strictly better than canonical A0 on every other metric: **99.9 % capture** (A0: 99.1 %), **0 repeated
fly-bys** (A0: 8), **0 orbit episodes**, **0 timeouts**, **0 exhaustion**, and every physical speed/acceleration
limit respected exactly. No defender or dynamic-obstacle CBF is implemented in this stage; both remain future work.

## 0. Final-benchmark simplification (frozen)
For the simplified user-facing benchmark: **No Clutter → canonical C0**; **Clutter → canonical C2**. Both reused
**unchanged** — no new clutter generator was created, and C1/C3 and all historical support remain untouched. This
mapping will be the only distinction the web app exposes going forward.

## 1. Architecture (frozen)
```
moving asset -> static route waypoint -> v_nom -> CBF-QP -> v_safe -> velocity executor
```
Division of responsibility (never blurred): the **route planner** (reused, unmodified
`uav_defend/obstacles/route_planner.py:build_and_search_route`, the SAME tested visibility-graph + A* planner used
throughout the repository) resolves **global** deadlock around obstacle clusters by choosing *which waypoint* to
steer toward. The **CBF-QP** provides only **local, per-instant safety**, minimally perturbing the requested
velocity near obstacles — it never plans a path and is never responsible for global routing.

## 2. Route-planner integration (`uav_defend/attackers/cbf_obstacle.py:CBFRouteNavigator`)
* If the direct line from the attacker to the **current** asset position is unblocked (tested against every static
  obstacle inflated by the canonical clearance), the guidance target is the asset position directly.
* If blocked, `build_and_search_route` computes a visibility-graph/A* route; the **next** waypoint becomes the
  guidance target. The final route goal is always the **current** asset position — never the initial/stale one.
* **Replanning** (brief section 7, frozen): every 1.0 s (every two 0.5 s steps), **or** immediately if the current
  segment becomes blocked, the current waypoint is reached, the asset has moved > 2.0 m from the position used to
  build the current route, or the route is empty/invalid.
* **Route-planner-failure fallback (bug found and fixed during development):** when a replan attempt itself fails
  (the planner returns `None` — a geometric search failure, typically because the attacker's position is already
  inside/overlapping the inflated clearance zone, so no edge from "start" is ever unblocked), the navigator **keeps
  following the previous route's current waypoint** rather than reverting to the (by definition, still-blocked)
  direct asset position. The original implementation incorrectly reverted to the blocked direct target, which would
  have aimed the nominal guidance straight back through the obstacle; this is covered by a dedicated regression
  test. Of the 11 episodes in which a planner failure occurred, 10 recovered safely to capture using this fallback;
  only the one already-diagnosed acceleration-lag scenario (section 7 below) still collided.

## 3. AABB clearance implementation (`uav_defend/attackers/cbf_obstacle.py`)
Canonical `obstacle_clearance = 2.0 m` (`CBF_OBSTACLE_CLEARANCE`, matching `EnvConfig.enemy_obstacle_clearance`'s
canonical default), used both for route-planner inflation and for the CBF hazard set. **No sphere/circle
approximation anywhere**: `aabb_signed_distance_and_normal` is the standard box signed-distance function — `clip`
the query point into the box on every axis; outside, the clamped point is automatically correct whether one axis
is clamped (face), two (edge), or three (corner); inside, the signed distance is the negative of the smallest
per-axis penetration depth, with the outward normal through the nearest exit face. Explicitly tested for all three
outside cases (face/edge/corner) plus the inside case.

## 4. Exact CBF barrier (brief sections 9–11, frozen)
For each inflated static AABB within `active_obstacle_range = 20 m` (ALL qualifying obstacles simultaneously
active, never only the nearest):
```
d_i   = signed clearance distance to the inflated AABB (aabb_signed_distance_and_normal)
n_i   = outward unit normal at the closest point (toward the attacker)
s_i   = max(0, -n_i . v_A)                              (inward speed component)
d_stop,i = s_i^2 / (2 a_max) + s_i * dt_sub             (stopping-distance + one-substep reaction margin)
h_i   = d_i - d_stop,i
constraint:  n_i^T v_safe >= -alpha * h_i,   alpha = 1.0 s^-1  (v_obs = 0 for static obstacles)
```
`a_max = 6 m/s²`, `dt_sub = 0.1 s` — all frozen, never tuned after the acceptance run.

## 5. Stopping-margin implementation — a genuine fix found during development
**Problem found:** the CBF-QP was initially solved **once per external 0.5 s step** (`v_safe` held constant across
all 5 substeps, matching Prompt 1's identity-filter convention). A 200-episode smoke test produced a static-obstacle
collision: the stopping-margin formula's `dt_sub` term implicitly assumes the velocity command can be corrected
again within `dt_sub = 0.1 s` — true only if the safety filter is **actually re-evaluated** at that cadence. Holding
a safety-critical `v_safe` fixed for a full 0.5 s let a high-closing-speed attacker's required deceleration exceed
what 5 substeps of the 6 m/s² limit could deliver before reaching the obstacle.

**Fix:** `integrate_obstacle_cbf_substeps` re-solves the nominal guidance **and** the CBF-QP every substep
(`dt_sub = 0.1 s`) whenever ≥1 hazard is active at a step's start, instead of holding one command for the full
external step. This makes the barrier's own reaction-time assumption true by construction. The route **target**
itself is *not* recomputed at this cadence (it is supplied once per external step by the slower-cadence route
planner, per the architecture's division of responsibility). **This is a fix to control-loop timing, not a tuning
of `alpha`, `kp`, the clearance, or any physical limit** — all remain exactly as frozen. When no hazard is active at
a step's start (always true for C0), the ORIGINAL Prompt-1 single-shot executor is used unchanged, which is safe
because the maximum distance the attacker can travel in one step (`v_e · dt = 6 m`) is far smaller than the 20 m
active-obstacle range, so an obstacle inactive at a step's start cannot become physically dangerous before the next
step's hazard check correctly re-engages the safety-critical path.

## 6. CBF-QP formulation and solver (frozen)
```
min_v  0.5 * ||v - v_nom||^2
s.t.   n_i^T v >= -alpha * h_i            for every active hazard i
       -5 <= v_z <= 5
       cos(theta_k) v_x + sin(theta_k) v_y <= R cos(pi/32)   for k = 0..31   (32-sided inscribed polygon, R=12)
```
The velocity-bound polygon is **inscribed** in the true 12 m/s circle (never exceeds the physical bound); maximum
conservatism at an edge midpoint is `R(1 - cos(pi/32)) ≈ 0.058 m/s` (0.48 % of the bound) — negligible, and the
exact physical clip inside `integrate_quadrotor_substeps`/`apply_one_substep` remains the final numerical guard
regardless. Solved via `scipy.optimize.minimize(method="SLSQP")` (verified already available: SciPy 1.17.1; no new
dependency added), deterministic initial guess (clipped `v_nom`), fixed `maxiter=200`/`ftol=1e-12`. **Acceptance
criterion is constraint feasibility (max residual ≤ 1e-4), not SLSQP's own internal `success` flag** — a
borderline-degenerate QP (the exact solution lying on two constraint boundaries at once) can report "positive
directional derivative" (status 8) while the returned point is numerically feasible to well within tolerance; what
matters for safety is feasibility, not the optimizer's termination diagnostic (`qp_success` still records the raw
flag for transparency).

## 7. Hard → soft → emergency-brake fallback (frozen)
1. **Hard QP**: solved first; accepted if its max constraint residual is within tolerance.
2. **Soft fallback**: on hard failure, one shared non-negative slack `δ` is added to every CBF inequality only
   (`n_i^T v + δ >= -alpha h_i`; velocity bounds remain hard, never relaxed), objective
   `0.5||v - v_nom||^2 + 1e4 δ²`.
3. **Emergency brake**: if the soft problem also fails, `v_safe = 0` — "the acceleration-feasible command toward
   zero velocity," never an invented escape direction; the route planner re-evaluates on the next step.

Over the full 1,000-episode campaign: **96.7 %** of active-hazard steps resolved via the hard QP, the remainder via
the (rare) soft fallback path never materially engaging (`mean_soft_step_fraction ≈ 0`), and **zero** emergency-brake
events — the fallback chain exists and is tested (including a forced-failure unit test), but was never needed in
practice on this benchmark.

## 8. Legacy / foundation regressions (brief section 20, mandatory) — both PASS
* **Legacy A0 no-op**: 1,000 matched seeds (a fresh, non-overlapping block) compared against the Prompt-1 branch
  point commit `2f45914`'s `SoldierEnv` — **1,000/1,000 identical, 0 mismatches.**
* **C0 foundation identity**: 1,000 seeds (reusing the Prompt-1 foundation seed block) compared against the CACHED
  Prompt-1 per-episode results — **1,000/1,000 identical, 0 mismatches.** `active_obstacle_count = 0` for every one
  of these episodes (C0 has no obstacles), confirming the obstacle-aware code path is an exact identity whenever no
  hazards exist, exactly as required.

## 9. C2 baseline (canonical A0, context only — not the acceptance definition)
| | canonical A0 |
|---|---:|
| Capture | 99.1 % |
| Exhaustion | 0.6 % |
| Obstacle collisions | 3 / 1000 |
| Repeated fly-by episodes | 8 |
| Mean path-excess | 1.012 |

## 10. A1-CBF 1,000-episode Monte Carlo result (C2, static clutter only, no defender interaction, dynamic clutter disabled)
| Metric | A0 | A1-CBF |
|---|---:|---:|
| Capture | 99.1 % | **99.9 %** |
| Exhaustion | 0.6 % | 0 % |
| Timeout | 0 | 0 |
| First-pass capture | 98.7 % | 99.2 % |
| Repeated fly-by episodes | 8 | **0** |
| Orbit episodes | 0 | 0 |
| **Obstacle collisions** | 3 | **1** |
| Planner-failure episodes | — | **11** (32 events; 10/11 still captured safely) |
| Mean route replans / episode | — | 0.93 |
| Min physical clearance observed | — | 0.131 m (the one collision episode) |
| Mean path-excess | 1.012 | 0.986 |
| Max speed / accel observed | 12.0 m/s / 6.0 m/s² | 12.0 m/s / 6.0 m/s² (within limits) |
| Mean fraction of steps with CBF intervention | 0 | 32.8 % |
| Hard-QP success (of active-hazard steps) | — | 96.7 % |
| Soft-fallback / emergency-brake steps | — | ~0 % / 0 % |

## 11. Acceptance gate (frozen BEFORE the run; brief section 24) — **FAILS on 2 of 11 checks**
| Check | Required | Observed | Pass |
|---|---:|---:|:--:|
| Capture | ≥ 95 % | 99.9 % | ✓ |
| **Physical obstacle collision** | **= 0** | **1 / 1000** | **✗** |
| Timeout | ≤ 1 % | 0 % | ✓ |
| Endurance exhaustion | ≤ 4 % | 0 % | ✓ |
| Persistent orbit | = 0 | 0 | ✓ |
| Repeated fly-by | ≤ 2 % | 0 % | ✓ |
| >5× path-excess | ≤ 1 % | 0 % | ✓ |
| **Planner failure** | **= 0** | **11 episodes** | **✗** |
| Numerical failure | = 0 | 0 | ✓ |
| QP hard/soft success | 100 % | 100 % | ✓ |
| Emergency brake (steps) | ≤ 1 % | 0 % | ✓ |
| Soft fallback (steps) | ≤ 1 % | ~0 % | ✓ |

**Dynamics constraint verification: PASSED** — max observed speed 12.000 m/s (limit 12.0), max observed
acceleration 6.000 m/s² (limit 6.0); no violation.

## 12. Failure analysis (brief section 25 — root cause classified, NOT silently tuned)
**Dominant mechanism: acceleration-lag safety violation, caused by a spawn-time physical infeasibility in the
static-clutter scenario generator, not a bug in the CBF-QP/route-planner implementation.**

The single colliding episode (seed 500100) spawns the attacker at full speed (`v_e = 12 m/s`) with a velocity
pointed almost directly at a static obstacle whose inflated-boundary clearance is only ~7.96 m — but the
*theoretical minimum stopping distance* at 12 m/s under the frozen `a_max = 6 m/s²` is `v²/2a = 12 m` (12.5 m
including the `dt_sub` reaction term). **`h_i` is already negative at the very first simulated instant (episode
reset), before the CBF ever has a chance to react.** No velocity-command controller bound by the same acceleration
limit can avoid this: canonical C2's `obstacle_clearance_from_spawn = 3.0 m` (an existing, unmodified config field,
designed around the legacy turn-rate-limited attacker) is far smaller than the ~12–13 m envelope a velocity-
commanded, acceleration-limited quadrotor actually requires. This is confirmed directly (`test_known_acceleration_
lag_collision_seed_is_a_spawn_time_physical_infeasibility` in `tests/test_cbf_obstacle.py`) and is exactly why
canonical A0 (legacy, turn-rate-limited pursuit + its own, separate reactive obstacle-avoidance layer) does **not**
collide on this same seed (min clearance 14.4 m) — its different guidance diverges from the direct line much
earlier, for reasons unrelated to A1-CBF's deceleration envelope.

A **secondary, now-fixed** contributing bug (route-planner-failure fallback reverting to a known-blocked target —
section 2 above) compounded the SAME episode but was not, by itself, sufficient to cause it (10 of 11
planner-failure episodes recovered safely with the fix in place; only this one, already-doomed-by-physics episode
still collided).

**This is squarely what verdict C's own wording anticipates: "FIX STATIC CLUTTER BEFORE DEFENDER INTEGRATION."**
Per the brief's explicit instruction, `kp`, `alpha`, the clearance, the active-obstacle range, and every physical
limit were **not** tuned after seeing this result. The two candidate fixes — (a) increase
`obstacle_clearance_from_spawn` for this attacker class, or (b) make the spawn procedure itself aware of the
attacker's required stopping envelope — are scenario-generation decisions deliberately left for an explicit,
separate design decision, not undertaken here.

## 13. Limitations
1. No defender avoidance and no dynamic-obstacle CBF: out of scope for this prompt, as directed.
2. The acceptance gate's `obstacle_collision=0` and `planner_failure=0` criteria are NOT met (1/1000, 11/1000
   respectively); the root cause is diagnosed, narrow, and reproducible, but not fixed in this stage.
3. The route-planner-failure fallback fix (section 2) is a genuine correctness improvement but does not, by itself,
   resolve the spawn-time physical infeasibility that caused the one collision.
4. `kp`, the physical limits, `alpha`, the clearance, and the active-obstacle range were frozen before this run and
   were not tuned afterward, as required.

## 14. Reproduce / verify
```
python -m experiments.journal_cbf_attacker_obstacle_run all        # stages 1-7 (restartable)
python -m experiments.journal_cbf_attacker_obstacle_figures         # figures 1-6
pytest tests/test_cbf_obstacle.py tests/test_cbf_attacker.py -q
```
State: `task_state.json` (10 stages, atomic, hash-verified). Results (untracked):
`results/journal_v2_corrected_semantics/journal_cbf_attacker/obstacle_cbf/` (chunked, restart-safe).

## 15. Scope reminder
> Defender avoidance is NOT yet implemented. Dynamic-obstacle CBF is NOT yet implemented. Research focus beyond
> this point remains the counter-UAS defender.

## Figures
`figures/fig1…fig6` (PNG + PDF): a representative 3-D route through C2's static clutter; nominal-vs-safe velocity
intervention trace near an obstacle; minimum-clearance distribution across all 1,000 episodes; CBF-intervention
distribution; A0-vs-A1-CBF outcome comparison; path-excess distribution.
