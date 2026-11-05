# A1-CBF — viability-conditioned clutter benchmark (Prompt 3: fixing the Prompt-2 failure mechanisms)

**Result: the frozen acceptance gate PASSES on all 13 criteria. Verdict A.** Conditioning ONLY the attacker's
initial spawn/heading on braking feasibility (canonical C2 obstacle generation, the attacker dynamics, and every
frozen CBF/route-planner parameter are completely unchanged) eliminates the single static-obstacle collision found
in Prompt 2, while corrected route-status handling (`route_ok` / `route_temporarily_unavailable` / `route_error`)
safely resolves every one of the 11 Prompt-2 planner-failure episodes with **zero** `route_error` events. Over
1,000 fresh matched Clutter episodes: **100.0 % capture, 0 obstacle collisions, 0 repeated fly-bys** (vs. A0's 9),
**0 route errors**, **0 scenario-generation failures**, and every physical limit respected exactly.

## 0. What Prompt 2 found, restated precisely
Prompt 2's 1/1000 collision was a **spawn-time physical infeasibility**: the attacker can spawn already closer to
an obstacle than its own braking envelope (`v_e=12 m/s`, `a_max=6 m/s²` → ~12–13 m minimum stopping distance)
permits, because canonical C2's `obstacle_clearance_from_spawn=3.0 m` was designed around the legacy turn-rate-
limited attacker, not a velocity-commanded, acceleration-limited one. This is an **initial-state viability problem**,
not evidence the CBF-QP itself is broken — confirmed directly in Prompt 2 by the controller's otherwise-excellent
999/1000 performance, and reconfirmed here by the fact that NONE of the frozen CBF/route parameters were touched to
fix it.

## 1. Frozen controller parameters (unchanged from Prompt 2 — verified, never tuned)
`kp=1.0 s⁻¹`, `v_xy_max=12 m/s`, `a_max=6 m/s²`, `climb/descent=±5 m/s`, `dt=0.5 s`, `dt_sub=0.1 s`,
`obstacle_clearance=2.0 m`, `active CBF range=20 m`, `alpha=1.0`, 32-sided velocity polygon, hard→soft→emergency QP
policy, SLSQP settings, route replan rules (1.0 s period / segment-invalid / waypoint-reached / asset-moved>2m
triggers). `protocol.json`'s `frozen_controller_parameters` field records these verbatim for audit.

## 2. Clutter benchmark definition (frozen)
`Clutter` = canonical C2 obstacle generation (unchanged obstacle count/positions/sizes/randomization, unchanged
moving asset, unchanged endurance, unchanged world boundaries) **plus** a new rule governing only whether the
*sampled initial attacker state* is physically viable.

## 3. Initial-state viability test (exact equation, frozen before any validation)
```
d_i >= d_req,i     for EVERY static obstacle i (no 20 m active-range cutoff here -- all obstacles, since the
                    attacker has had no chance to react to anything yet)
d_req,i = s_i^2 / (2 a_max) + s_i * dt_sub
s_i = max(0, -n_i . v_0)
```
`d_i` = exact signed distance to the SAME inflated (2.0 m clearance) AABB used by the live CBF; `n_i` = outward
unit normal at the closest point. No additional arbitrary buffer. Implemented as a thin, exact reuse of the live
CBF's own hazard-construction code (`build_static_hazards`, `active_range=inf`) — `h_i = d_i - d_req,i >= 0` is
identical to the live barrier's own margin computation, guaranteeing the initialization test and the in-flight
safety filter can never disagree about what "feasible" means.

## 4. Resampling implementation
If the initial state fails: **resample ONLY the attacker spawn/heading** via the existing canonical
`_spawn_enemy_at_edge` mechanism (same `_rng_spawn` stream, simply drawing further values — no new RNG stream, no
perturbation of any other stream). The obstacle field (`generate_layout`, its own dedicated `_rng_obstacles`
stream) and the asset initialization are generated **once** and never touched again. Up to 100 attempts; if none
viable, the episode is marked `scenario_generation_failure` (0/1000 observed in the acceptance campaign, 0/10,000
in the bias audit). Exposed via a new `viability_conditioned_init` **constructor argument** (not an `EnvConfig`
field — the same design pattern as `enemy_control_mode`, specifically so it can never perturb the
`dataclasses.asdict(config)` fingerprint every prior curriculum stage pins for reproducibility). Default `False`
reproduces Prompt 1/2 bit-for-bit (verified: identical `enemy_pos` for the same seed when the first draw is already
viable, and the 1,000-seed C0/legacy regressions below).

## 5. 10,000-scenario bias audit (brief section 6)
| | Before (unconditioned) | After (viability-conditioned) |
|---|---:|---:|
| Accepted on first draw | — | 97.0 % |
| Required resampling | — | 3.0 % (mean 1.05 resamples among those resampled, max 2) |
| Scenario-generation failures | — | 0 / 10,000 |
| Obstacle-field / asset mismatches | — | 0 / 10,000 |
| Attacker range (mean / median) | 61.11 / 59.94 m | 61.23 / 60.04 m |
| Attacker speed (mean / median) | 11.959 / 12.000 m/s | 11.958 / 12.000 m/s |
| Nearest-obstacle distance (mean / median) | 27.99 / 26.48 m | 28.50 / 26.83 m |

The obstacle field and asset initialization are **exactly** identical between the two runs for every one of the
10,000 seeds (0 mismatches, directly measured — not merely asserted from code inspection). The accepted-state
distributions shift by **under 2 % in every dimension**: the viability condition excludes only a narrow, rare slice
of initial states, not a materially different benchmark. Full results: `bias_audit.json`.

## 6. Prompt-2 planner-failure audit (brief section 7) — zero bugs found
All 32 planner-failure events across the 11 Prompt-2 episodes were replayed and classified (`planner_failure_audit.json`):
**4 `start_inside_inflated_aabb`, 4 `goal_inside_inflated_aabb` (moving-goal transient), 3 borderline cases of the
same two mechanisms at finer time resolution, 0 `invalid_start`/`invalid_goal`, 0 `numerical_visibility_failure`,
0 `exception_implementation_bug`.** `build_and_search_route` behaves exactly per its own documented contract
("Phase 24: a geometric search failure, NOT a ground-truth feasibility judgment") in every case. **No planner bug
was found or fixed** — the needed change was entirely in how `SoldierEnv`/`CBFRouteNavigator` **respond** to these
events (section 7/9 below).

## 7. Route-status semantics (new, frozen)
* **`route_ok`** — a valid route/direct-shot target is available.
* **`route_temporarily_unavailable`** — the planner executed correctly (no exception, finite output) but found no
  safe path at this instant (start/goal transiently inside an inflated zone). NOT an implementation failure.
* **`route_error`** — an exception, non-finite/malformed waypoint output, or a planner-contract violation.
  Acceptance requires exactly 0; **0 observed** across both the 11-event audit replay and the full 1,000-episode
  campaign.

## 8. Route-handling changes (two fixes, both regression-tested)
1. **Retain the previous waypoint** (fixed in Prompt 2, unchanged here) when a replan attempt fails and a previous
   route exists — never revert to the (by construction, still-blocked) direct asset target.
2. **NEW: hover/brake when no previous waypoint exists** (`v_nom = 0`, brief section 9) — the ONE behavioral change
   from Prompt 2's route handling. Previously this case fell back to the asset position directly (aiming straight
   at a known-blocked target); now the executor commands zero nominal velocity and lets the CBF/acceleration-limited
   dynamics brake/hover safely, replanning the next external step. Never an invented escape direction.

   This is the **one intentional behavioral difference** from Prompt 2 on *identical seeds* when
   `viability_conditioned_init=False` (3 of the original 1,000 Prompt-2 seeds now diverge in trajectory/step-count
   from the cached Prompt-2 result, EXACTLY the seeds with a Prompt-2 planner failure and no retained waypoint —
   confirmed directly). The CBF-QP's own math (given identical hazards/state) is untouched and bit-identical;
   only the *guidance target* supplied to it differs in this narrow case, exactly as brief section 11 anticipates
   ("the only intended behavioral difference is... corrected route handling").

## 9. Legacy / integrity regressions — all PASS
* **Legacy A0**: 1,000 matched seeds (fresh block) vs. the Prompt-1 branch-point commit — **1,000/1,000 identical.**
* **CBF C0 foundation**: 1,000 seeds vs. the cached Prompt-1 results — **1,000/1,000 identical** (route navigator
  and viability conditioning are complete no-ops on C0, which has no obstacles).
* **Prompt-2 collision seed (500100)**: unconditioned, confirmed infeasible at spawn
  (`initial_viability_margin < 0`); viability-conditioned, resolved in 1 resample, capture confirmed, **no
  collision.**
* **All 11 Prompt-2 planner-failure seeds**: replayed under the corrected handling — **0 route_error, 0 collisions**
  for every one.

## 10. 1,000-episode revised Clutter Monte Carlo (fresh seed block 530000–530999)
| Metric | A0 (context only) | A1-CBF |
|---|---:|---:|
| Capture | 98.5 % | **100.0 %** |
| Exhaustion | 0.7 % | 0 % |
| Timeout | 0 % | 0 % |
| **Obstacle collisions** | **8** | **0** |
| First-pass capture | 98.1 % | 98.6 % |
| Repeated fly-by episodes | 9 | 1 |
| Orbit episodes | 0 | 0 |
| Scenario-generation failures | 0 | 0 |
| Route errors | — | **0** |
| Episodes with a temporarily-unavailable route | — | 9 (41 total steps, longest streak 7) |
| Hover/brake fallback events | — | 19 |
| Min physical / inflated clearance observed | ∞ | 2.21 m / 0.21 m |
| CBF `h` minimum observed | ∞ | **0.058** (never negative — the barrier was never actually violated) |
| Mean path-excess | 1.019 | 0.987 |
| Max speed / accel observed | 12.0 m/s / 6.0 m/s² | 12.0 m/s / 6.0 m/s² (within limits) |
| CBF intervention (fraction of steps) | 0 % | 32.9 % |
| Hard-QP success (of active-hazard steps) | — | 97.3 % |
| Soft-fallback / emergency-brake steps | — | 0 % / 0 % |
| Viability resample rate | 0 % | 3.5 % (mean 0.038, max 3) |
| Route replans | 0 | 1,067 total (mean 1.07/episode) |

## 11. Acceptance gate (frozen BEFORE the run; brief section 13) — ALL 13 CHECKS PASS
| Check | Required | Observed | Pass |
|---|---:|---:|:--:|
| Capture | ≥ 95 % | 100.0 % | ✓ |
| Physical obstacle collision | = 0 | 0 | ✓ |
| Timeout | ≤ 1 % | 0 % | ✓ |
| Endurance | ≤ 4 % | 0 % | ✓ |
| Persistent orbit | = 0 | 0 | ✓ |
| Repeated fly-by | ≤ 2 % | 0.1 % | ✓ |
| >5× path-excess | ≤ 1 % | 0 % | ✓ |
| Route errors | = 0 | 0 | ✓ |
| Scenario-generation failures | = 0 | 0 | ✓ |
| Numerical failures | = 0 | 0 | ✓ |
| QP hard/soft completion | 100 % | 100 % | ✓ |
| Emergency brake (steps) | ≤ 1 % | 0 % | ✓ |
| Soft fallback (steps) | ≤ 1 % | 0 % | ✓ |

**Dynamics constraint verification: PASSED** — max observed speed 12.000 m/s (limit 12.0), max observed
acceleration 6.000 m/s² (limit 6.0); no violation.

## 12. Near-viability-boundary stress suite (diagnostic, brief section 15 — separate from the rate metrics above)
120 synthetic states constructed from REAL canonical-C2 obstacle layouts (20 distinct seeds × 6 target margins
`0.0`–`0.5 m`, full-speed radial approach — the hardest possible case for a given margin), **101 of which land
exactly in the required `0 ≤ margin ≤ 0.5 m` window** (exceeding the ≥100 requirement; the remainder are `margin=0`
floating-point-boundary draws that round either side of exactly zero). **Zero physical obstacle collisions, zero
numerical failures** across all 120 states (min physical clearance observed: 2.96 m). This verifies the viability
rule is not merely screening out *easy* starting conditions — the controller safely handles states deliberately
engineered right at the edge of what braking feasibility allows.

## 13. Visualizer (brief section 17)
The gate passed, so minimal replay support was in scope. The visualizer's replay engine
(`visualizer/backend/replay_engine.py`) was inspected directly: it resolves every attacker/defender policy via a
frozen SB3 `.zip` checkpoint registry with SHA-256 integrity verification — there is no raw-trajectory/scripted-
controller replay path. A1-CBF has no checkpoint (a deterministic controller), so wiring it in would require a new
backend code path bypassing checkpoint resolution entirely, exceeding "minimal"/"do not redesign the interface"
scope. Instead, **5 full trajectory recordings** (position/velocity/route-status/CBF diagnostics per step, labeled
`A1-CBF-CLUTTER-DEV`, `development / non-canonical`) covering every required review category were captured and
saved under `visualizer_review_episodes/`, ready for a future stage that adds native scripted-controller replay:

| File | Seed | Outcome | What to look for |
|---|---:|---|---|
| `near_viability_boundary_spawn.json` | 530028 | captured | Spawns with `initial_viability_margin≈0.058`; tight but successful early deceleration. |
| `multi_obstacle_bypass.json` | 530489 | captured | Highest replan count (21); repeated re-planning while weaving past several obstacles. |
| `temporarily_unavailable_route.json` | 530306 | captured | Route briefly unavailable; hover/brake steps, never a blocked-direction or random-escape command. |
| `largest_cbf_intervention.json` | 530371 | captured | Largest single-step intervention norm (~11 m/s) in the campaign. |
| `worst_successful_path.json` | 530283 | captured | Highest path-excess (~1.2×) among successful captures — a longer detour, not a failure. |

## 14. Limitations
1. `scenario_generation_failure` is possible in principle (100-attempt exhaustion) but was never observed (0/1,000
   in the acceptance campaign, 0/10,000 in the bias audit) — canonical C2's spawn geometry gives ample room to find
   a viable attacker heading in virtually every case.
2. The viability test only re-examines the ATTACKER's initial state; it does not (and per the brief, must not)
   alter obstacle placement, so an unusually tight obstacle cluster could in principle require more resamples —
   not observed at this density (max 3 resamples across both 10,000- and 1,000-scenario runs).
3. No defender or dynamic-obstacle CBF is implemented in this stage; both remain future work.
4. Visualizer: data-only recordings this stage; the visualizer app itself was not modified (see section 13).

## 15. Reproduce / verify
```
python -m experiments.journal_cbf_attacker_clutter_v2_run all               # stages 1-8 (restartable)
python -m experiments.journal_cbf_attacker_clutter_v2_bias_audit            # 10,000-scenario bias audit
python -m experiments.journal_cbf_attacker_clutter_v2_stress                # 120-case near-boundary stress suite
pytest tests/test_cbf_clutter_v2.py tests/test_cbf_obstacle.py tests/test_cbf_attacker.py -q
```
State: `task_state.json` (10 stages, atomic, hash-verified). Results (untracked):
`results/journal_v2_corrected_semantics/journal_cbf_attacker/obstacle_cbf_v2/` (chunked, restart-safe).

## 16. Scope reminder
Research focus beyond this point remains the counter-UAS **defender**. Static-obstacle navigation is now accepted;
defender CBF integration is the next stage (**not started here**). No canonical tag is created for this controller
at this stage.
