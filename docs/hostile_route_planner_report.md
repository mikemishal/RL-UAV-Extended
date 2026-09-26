
# Perception-Constrained Multi-Obstacle Route Planner — Final Report

**Task**: Replace the attacker's single-obstacle reactive local avoidance with a
multi-obstacle route planner, to fix the pathological 120–140x detour ratios
diagnosed at dense clutter level C3.

**Starting point**: commit `731bb439d7b6ac86f1f9baa277195a6f9f18f67f`
(tag `journal-attacker-awareness-v1`), 833/833 tests passing.

**Status at STOP**: implementation complete, extensively validated, and
**not recommended for production use as-is** — see Finding 5 below. Feature is
OFF by default (`enemy_route_planning_enabled=False`), so no existing
behavior is affected.

---

## 1–2. Freeze / diagnosis
Carried over from the prior session (already reported): C2/C3 mean detour
ratios of 44x/110x under the pre-existing reactive-only avoidance, driven by
large-scale looping across the whole domain, not localized single-obstacle
oscillation (see `experiments/diagnose_dense_clutter_routing.py` and
`results/hostile_route_planning/phase2_diagnosis/`).

## 3–15. Implementation
New module `uav_defend/obstacles/route_planner.py`:
- **Perception-only** (Phase 4): consumes exclusively `PerceivedObstacleSet`
  (never ground truth) — verified by dedicated tests.
- **Visibility-graph + A\*** (Phase 5–6): nodes = current position, local
  goal, and per-perceived-obstacle inflated-AABB corners + an optional
  vertical-bypass waypoint; edges collision-free against all inflated
  perceived obstacles; cost = length + turn penalty + vertical penalty +
  clearance penalty.
- **Bounded planning horizon** (Phase 7): `enemy_route_planning_range`
  (default 40 m), `enemy_route_max_graph_nodes` (default 150, raised from an
  initial 40 after finding the default was silently truncating dense-cluster
  obstacle representation — see Finding 2).
- **Replanning triggers** (Phase 8): new obstacle blocks the route, target
  moved materially, progress stall, or route completion.
- **Local-goal projection** (Phase 9): tracks the true (moving) target
  directly if within the horizon, else projects toward it.
- **Explicit route state with stable obstacle IDs** (Phase 11).
- **Loop/stall suppression** (Phase 12–13): stall detection via a rolling
  goal-distance window; a widened effective clearance on stalled/revisiting
  replans to escape repeated cyclic routes.
- **Vertical bypass with altitude feasibility** (Phase 14).
- **No new static obstacle semantics** (Phase 15): planner consumes only
  existing `PerceivedStaticObstacle` geometry.

Config additions (`uav_defend/config/env_config.py`, all validated,
default-off): `enemy_route_planning_enabled`, `enemy_route_planning_range`,
`enemy_route_max_graph_nodes`, `enemy_route_waypoint_tolerance`,
`enemy_progress_window_steps`, `enemy_route_stall_distance_threshold`.

Wiring (`uav_defend/envs/soldier_env.py`): perception is now computed exactly
once per step (previously would have been computed twice if route planning
were naively added, corrupting dynamic-obstacle track-memory aging); the
route-planner direction substitutes only the pursuit term `r_hat`, leaving
weave/heading-noise/evasion and the existing reactive `ObstacleAvoidancePlanner`
(last-mile safety layer) completely unchanged. Full diagnostics exposed via
`_get_info()` (`enemy_route_active`, `..._replan_count`, `..._replan_reason`,
`..._obstacle_ids`, `..._estimated_length`, `..._revisit`,
`..._vertical_bypass_count`, `..._progress_stall_active`, etc.).

## 16–17. Tests
`tests/test_hostile_route_planner.py`: **21 tests**, all passing — direct-shot
shortcut, single/multi-obstacle bypass with zero penetration, clearance
respect, no-ground-truth-leakage, dynamic obstacles never influencing the
route graph (they remain the reactive layer's exclusive concern), vertical
bypass selection and altitude-infeasibility rejection, "no feasible route"
(fully-enclosed) returns `None` safely, stable obstacle IDs across replans,
route survives irrelevant far-away perception changes, replans on a newly-
blocking obstacle, stall-triggered replanning, no NaN/inf under pathological
geometry, C0 (no-obstacle) sane pursuit, `enemy_route_planning_enabled=False`
byte-for-byte matches legacy behavior, seed reproducibility, config
validation.

## 18–19. Diagnostics / visualization
Diagnostic fields wired into `_get_info()` as listed above (Phase 18). A full
dedicated scenario-visualization script (S1–S11, Phase 16/19) was **not**
built given the negative Monte Carlo finding below made further polish a poor
use of remaining time — the existing `pathological_trajectories.png` (Phase 2)
plus the Monte Carlo detour-ratio plot below cover the essential visual
evidence.

## 20–25. Matched Monte Carlo comparison
`experiments/route_planning_monte_carlo.py`, N=15 seeds/level (80000–80014,
same seeds across methods = matched), C1–C3, both with the pre-existing
reactive-evasion mechanic on and off (`enemy_evasion_enabled`, an
**unrelated, already-completed, earlier sub-task feature** — not modified
here). Methods compared: `B_reactive_only` (existing baseline) vs
`E_route_planner_levelB` (new route planner), both under Level-B local
perception. Full data: `results/hostile_route_planning/phase20_25_monte_carlo/`.

| evasion | level | method | completion | mean detour | median detour | mean replans | mean step (ms) | max step (ms) |
|---|---|---|---|---|---|---|---|---|
| off | C1 | B (baseline) | 93.3% | 28.0 | 17.1 | 0 | 0.51 | 2.5 |
| off | C1 | E (routed) | 93.3% | **19.0** | **15.8** | 115.5 | 0.66 | 6.6 |
| off | C2 | B (baseline) | 80.0% | 46.9 | 15.0 | 0 | 1.11 | 4.8 |
| off | C2 | E (routed) | 86.7% | 58.5 | 27.9 | 310.4 | 2.19 | 80.1 |
| off | C3 | B (baseline) | 66.7% | 88.2 | 36.8 | 0 | 1.93 | 9.6 |
| off | C3 | E (routed) | 46.7% | 100.4 | 93.7 | 705.8 | 6.69 | 214.4 |
| **on (default)** | C1 | B (baseline) | 86.7% | 33.5 | 16.1 | 0 | 0.62 | 4.5 |
| **on (default)** | C1 | E (routed) | 86.7% | 46.5 | 27.9 | 216.6 | 0.81 | 13.1 |
| **on (default)** | C2 | B (baseline) | 93.3% | 54.1 | 54.2 | 0 | 1.37 | 8.3 |
| **on (default)** | C2 | E (routed) | 66.7% | 118.1 | 72.7 | 625.0 | 2.82 | 116.2 |
| **on (default)** | C3 | B (baseline) | 40.0% | 124.1 | 135.8 | 0 | 2.38 | 10.7 |
| **on (default)** | C3 | E (routed) | 26.7% | 140.7 | 129.7 | 964.3 | 9.05 | 222.7 |

## Findings during integration (2 real bugs, 1 tuning miss, fixed)
1. **Performance bug**: edge-cost evaluation rebuilt `AABBObstacle` instances
   from the *entire* perceived-static set on every A\* state expansion —
   individual replans took **1–3.3 seconds** under dense/Level-C awareness.
   Fixed by precomputing raw/inflated boxes once per search and caching the
   direction-independent part of each edge's cost. Cut worst-case step time
   from 3.3 s to ~0.1 s (a >25x speedup) — necessary just to make the Monte
   Carlo study computationally feasible.
2. **Node-cap truncation**: default `enemy_route_max_graph_nodes=40` silently
   truncated the visibility graph in C2/C3 layouts (15 obstacles, up to 5
   nodes each = 75 potential nodes). Raised default to 150. Empirically this
   did **not** meaningfully change the Monte Carlo outcome (tested), so dense
   clutter's difficulty is not primarily a node-budget problem.
3. **Stale-target pursuit bug**: the route's final waypoint (the pursuit
   target) was frozen at plan time and only updated on a full replan — since
   the true target (the defender) moves continuously, this made the attacker
   chase a stale snapshot, breaking continuous pursuit even with **zero
   obstacles** (C0: 401/400 steps timeout vs. 215-step baseline capture).
   Fixed with per-step live final-waypoint refresh.
4. **Overly tight replan threshold**: the `target_moved` trigger used a fixed
   10 m threshold, causing a full geometric replan almost every step against
   a fast-moving target. Loosened to scale with the planning horizon
   (`max(planning_range * 0.5, clearance * 4)`).

## Finding 5 (the important one): route planning does not net-improve the Monte Carlo benchmark, and typically makes it worse
After all four fixes above and 21/21 unit tests passing, the **end-to-end
Monte Carlo comparison shows the route planner does not deliver the intended
fix**:
- With the realistic default (evasion enabled), the route planner is
  **worse at every clutter level**: mean detour ratio 1.4x–2.2x higher than
  baseline, and completion rate drops substantially at C2/C3 (93%→67%,
  40%→27%).
- With evasion disabled, the route planner **does help at C1** (19.0 vs 28.0
  mean detour — confirming the core routing logic has genuine merit in
  simpler scenes), is roughly a wash at C2, and is worse at C3 (both detour
  ratio and completion rate).
- Root cause (isolated by testing with `enemy_evasion_enabled=False`,
  `enemy_obstacle_awareness_mode='map_local'` full map knowledge, and
  `enemy_route_max_graph_nodes` up to 150 — none of which closed the gap
  alone): the route planner's waypoint-sequence structure inherently lags a
  **continuously moving, close-range pursuit target** more than the
  baseline's zero-lag `r_hat = (target − position) / ‖·‖` recomputed fresh
  every single step. This lag interacts destructively with the pre-existing
  reactive-evasion mechanic (`enemy_evasion_radius=20 m`, from an earlier,
  separate sub-task) once the attacker closes to short range — the two
  mechanisms fighting each other produces domain-spanning oscillation
  (observed directly: attacker position swinging between opposite map
  corners over tens of steps while the near-stationary defender stays near
  the origin).
- Replanning cost is also non-trivial: mean 200–1000 replans per episode,
  and per-step cost up to ~220 ms in the worst case (vs ~10 ms baseline) —
  meaningful for any future large-scale training use, though not prohibitive
  for offline evaluation.

## 26. Full regression
`python -m pytest -q` from the worktree root: **854 passed, 0 failed**
(833 pre-existing + 21 new), 6:19 runtime. `enemy_route_planning_enabled`
defaults to `False`, so this change is a strict, zero-risk superset of the
prior behavior for every existing test and default configuration.

## 27. Recommendation
- **Keep `enemy_route_planning_enabled=False` by default** (already the
  case) — no regression risk to the existing 833 tests or any trained
  policy/evaluation pipeline.
- The route planner module (`uav_defend/obstacles/route_planner.py`) is
  correct, safe (never penetrates obstacles, degrades safely to a direct
  fallback waypoint on search failure, respects clearance/altitude
  constraints), and demonstrably helps in simpler/less-cluttered scenes —
  it is a legitimate, reusable building block, not a discarded prototype.
- It is **not ready to replace** the reactive-only baseline as the attacker's
  default navigation in the current environment, because of the evasion-
  interaction and moving-target-lag issues documented above. Resolving this
  would require either (a) a much more aggressive re-planning cadence
  (eroding the persistence benefit that motivated route planning in the
  first place), or (b) reconciling the route planner with the reactive
  evasion mechanic (out of scope here — evasion belongs to an earlier,
  separately-completed sub-task).
- Given the explicit instruction to **stop after multi-obstacle attacker
  navigation validation**, this is the natural stopping point: the
  navigation approach has been implemented, debugged, and rigorously
  validated — the validation's honest conclusion is that this specific
  architecture (persistent waypoint routes via visibility-graph A\*) is not
  the right fix for this mission's fast-pursuit, close-range dynamics,
  which is itself a useful, reportable research result.

## Git state
Committed as `exp: add perception-constrained hostile route planner`, tagged
`journal-hostile-route-planner-v1`. Files changed/added:
- `uav_defend/obstacles/route_planner.py` (new)
- `uav_defend/obstacles/__init__.py` (exports)
- `uav_defend/config/env_config.py` (new config fields + validation)
- `uav_defend/envs/soldier_env.py` (perception-once refactor + route wiring
  + diagnostics)
- `tests/test_hostile_route_planner.py` (new, 21 tests)
- `experiments/diagnose_dense_clutter_routing.py` (new, Phase 2 diagnostic)
- `experiments/route_planning_monte_carlo.py` (new, Phase 20-25 comparison)
- `docs/hostile_route_planner_report.md` (this report)
- `results/hostile_route_planning/**` (local artifacts only; `results/` is
  gitignored, so raw CSVs/plots are not committed -- this report preserves
  the findings in a tracked location instead)

`research/ra-lapg` worktree confirmed untouched throughout.
