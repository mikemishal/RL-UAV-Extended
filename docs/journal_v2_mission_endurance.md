# V2 journal benchmark: finite mission endurance (attacker path budget)

Starting research HEAD for this task: `917f8b5ef8e2a1e8e8d34f8f84b6cea2abda907b` (accepted
asset-motion model study: legacy `legacy_gaussian_random_walk` kept canonical; correlated motion
remains config-gated, non-default). This document covers the next, accepted step: a finite
mission-endurance / attacker path-budget mechanism that resolves the single remaining pathology
identified across every prior round -- C3 seed `130017`'s terminal fly-by (a 27.06x path-excess
episode where the attacker eventually captures the asset, but only after ~346 steps of a
recurring approach/overshoot/re-approach cycle; see `docs/journal_v2_terminal_speed_governor.md`,
`docs/journal_v2_terminal_lead_intercept.md`, `docs/journal_v2_terminal_reacquisition.md`, and
`docs/journal_v2_asset_motion_model_study.md` for the three independently rejected guidance/motion
interventions that all either failed to resolve it or introduced a new collision elsewhere).

## Why endurance, not a guidance fix

Every prior intervention (terminal speed governor, lead-intercept, reactive reacquisition, and the
asset-motion model study) perturbed the attacker's trajectory or the asset's own motion near
130017 closely enough to reroute the attacker through a rare, pre-existing weak spot in the
accepted stateful avoidance controller's "all candidates collide" emergency fallback, producing a
NEW collision that did not exist under the unmodified baseline. The avoidance controller itself is
explicitly off-limits for modification in every round of this investigation.

A finite mission-endurance budget sidesteps this entirely: it never alters *how* the attacker
flies. It only ever *terminates* an episode once a traveled-distance budget is exhausted. Because
it changes nothing about guidance before that point, it cannot reroute the attacker through the
avoidance controller's weak spot -- the accepted trajectory prefix is bit-identical to the
unmodified baseline right up until the budget runs out (Phase 14; verified by dedicated tests, see
below).

## Terminology

This is deliberately called **finite mission endurance** or **attacker mission path budget**, never
a "calibrated battery model." There is no empirical UAV energy-consumption model backing the
numbers below. The mechanism represents the operational fact that a hostile UAV cannot fly
indefinitely after repeatedly failing to reach the asset -- it is a normalized, scenario-scale-
independent travel-distance allowance, not a claim about real flight time or battery duration.

## Formulation

At `reset()`, record the initial 3-D attacker-to-asset distance `d0` (reusing the existing
`pursuit_norm` value already computed to initialize the attacker's launch velocity -- no duplicate
computation). The traveled-distance budget is fixed once, at reset, and never recomputed:

```
B_path = k * d0            # k = enemy_endurance_path_factor
```

Cumulative traveled distance uses the ACTUAL integrated attacker trajectory, never commanded or
nominal speed:

```
L_{t+1} = L_t + ||p_{t+1} - p_t||
```

`k = 1` corresponds to roughly one initial-range equivalent of travel. Larger `k` provides
allowance for target motion, obstacle detours, maneuvering, and imperfect pursuit, and avoids
choosing an absolute path budget that depends on scenario scale.

## New terminal outcome

A distinct outcome, `enemy_endurance_exhausted`, is added -- never classified as `timeout`,
`soldier_caught` (a.k.a. "capture"), or `interception`. It means: *the hostile UAV failed to reach
the protected asset within its available mission endurance.* It is exposed in `info`
(`enemy_endurance_enabled`, `enemy_endurance_budget_m`, `enemy_endurance_traveled_m`,
`enemy_endurance_remaining_m`), in `experiments/journal_v2_trace.py`'s per-step trace schema
(`endurance_enabled`, `endurance_budget_m`, `endurance_traveled_m`, `endurance_remaining_m`), and
in `experiments/journal_v2_validation_core.py`'s `FINAL_INFO_KEYS`.

Training reward is minimal terminal bookkeeping only: `reward_enemy_endurance_exhausted` reuses
`reward_timeout`'s value (-100.0) as a neutral placeholder. Reward design for endurance-aware
training is deferred to the A1 retraining task.

## Within-step event precedence (Phases 5-6)

Endurance is integrated into the SAME first-physical-event-wins swept-contact resolver used for
collisions, capture, and interception (`uav_defend/envs/soldier_env.py`'s `step()`), as the
LOWEST-priority candidate:

| priority | event | fraction |
|---|---|---|
| 0 | `defender_collision` | static/dynamic obstacle contact fraction |
| 1 | `enemy_collision` | static/dynamic obstacle contact fraction |
| 2 | `capture` | swept-sphere contact fraction |
| 3 | `interception` | swept-sphere contact fraction |
| 4 | `endurance` | `tau_E` (below) |

If the remaining budget would be exhausted strictly within this step, the within-step exhaustion
fraction is:

```
tau_E = clip(B_remaining / ||p_{t+1} - p_t||, 0, 1)
```

using the FULL (unclamped) distance the attacker's own guidance/dynamics produced this step --
captured immediately after `_move_enemy()` runs and before any event arbitration, so it can never
be influenced by (or influence) which event wins. The earliest candidate among those present wins;
ties within `EVENT_TIE_TOLERANCE` are broken by the existing fixed priority order above (an
administrative "ran out of budget" event never preempts a genuine physical event at an earlier or
tied fraction). This is the project's existing, documented tie-break convention -- unchanged by
this feature. Traveled distance only ever advances by what the attacker ACTUALLY flew before the
winning event this step (the full step if nothing else won sooner, or the fraction up to an
earlier-winning event otherwise).

## Default behavior / gating

`enemy_endurance_enabled: bool = False` by default; enabling it requires
`mission_contact_mode == "swept"` (the within-step resolver it integrates with only exists in that
mode), enforced in `EnvConfig.__post_init__`. Historical V1 behavior (`mission_contact_mode ==
"endpoint"`) is unaffected and cannot accidentally acquire endurance tracking.

## Sensitivity sweep (Phases 7-13): k in {3, 4, 5}

Passive-A0 (defender parked, no learned policy, no training, canonical legacy asset motion), 400
episodes per condition: C0-C3 x seeds 130000-130099, for endurance disabled and each of k=3, 4, 5.

Results (see `experiments/mission_endurance_population_sweep.py` /
`experiments/mission_endurance_population_analysis.py`; raw per-episode records under
`results/mission_endurance_study/<label>/`):

| condition | C0/C1/C2 outcomes | C3 outcomes | C3 mean path (m) | C3 max path (m) | C3 max excess |
|---|---|---|---|---|---|
| disabled | 100 soldier_caught each | 100 soldier_caught | 77.2 | 1828.5 | 27.058 |
| k=3 | unchanged | 99 soldier_caught, 1 enemy_endurance_exhausted | 60.6 | 171.4 | 3.409 |
| k=4 | unchanged | 99 soldier_caught, 1 enemy_endurance_exhausted | 61.2 | 228.5 | 4.539 |
| k=5 | unchanged | 99 soldier_caught, 1 enemy_endurance_exhausted | 61.8 | 285.6 | 5.407 |

**Every one of the three factors changes exactly one seed out of 400: C3 seed 130017**, converting
it from a 346-step, 1828.5 m terminal fly-by capture into a clean `enemy_endurance_exhausted`
termination. Zero other seeds change at any clutter level, for any factor. Zero collisions occur in
any of the 1600 episodes across all four conditions (disabled + k=3/4/5).

### C3 seed 130017 detail (Phase 10)

| k | exhaustion step | elapsed (s) | traveled (m) | budget (m) | range at exhaustion (m) | mode | collision |
|---|---|---|---|---|---|---|---|
| 3 | 33 | 16.00 | 171.39 | 171.39 (=3*d0) | 5.92 | DIRECT | none |
| 4 | 43 | 21.41 | 228.52 | 228.52 (=4*d0) | 31.47 | DIRECT | none |
| 5 | 54 | 26.81 | 285.65 | 285.65 (=5*d0) | 9.61 | DIRECT | none |

`d0 = 57.13 m` for this seed. At every factor, exhaustion occurs while the attacker is already
inside its characteristic approach/overshoot oscillation (ranges cycling between ~3 m and ~37 m
within the first 30 steps of the baseline trajectory) -- i.e. the attacker has already entered the
known fly-by pattern before the budget runs out, consistent with the intended interpretation: the
attacker terminates as `enemy_endurance_exhausted` instead of continuing to fly in the pathological
loop indefinitely.

### Acceptance criteria (Phase 11-13)

- **Population impact**: identical across k=3/4/5 -- only the one known pathological episode is
  affected, for every clutter level. No normal episode is terminated early by any factor.
- **No new collisions**: confirmed 0/1600 across disabled + all three factors.
- **No guidance perturbation**: dedicated bit-identical-prefix tests (Phase 14, below) and the
  population sweep's zero-impact-on-normal-episodes result together confirm the mechanism never
  alters trajectory, only termination timing.

Since all three factors perform identically on the population-impact axis, the task's explicit
preference rule applies: **prefer the largest tested factor that reliably resolves the pathology**
(the least restrictive choice). **k=5 is selected as the canonical V2 factor.**

## Bit-identical prefix verification (Phase 14)

For every paired episode (same seed, endurance disabled vs. enabled at any tested k), the
trajectory is required to be bit-identical up to the exact exhaustion instant. Verified:

- Directly on C3 130017 at k=3/4/5 (`np.array_equal` over the full attacker-position prefix,
  excluding only the final interpolated exhaustion row) -- identical for all three.
- Via `tests/test_mission_endurance.py`'s dedicated Phase 14 tests: an exact-fraction exhaustion
  test checks the clamped position matches a linear interpolation of the undisturbed full step; a
  100-step randomized C3 clutter episode (endurance disabled vs. enabled-but-never-binding, k=1000)
  asserts position, velocity, behavior mode, and final command are identical every step.
- A full endurance-disabled trajectory digest check (reusing the asset-motion-study's 64-scenario
  digest harness) shows 0/64 digests changed relative to the `917f8b5e` baseline.

## Greedy/Lead invariance (Phase 16)

Re-running the existing Greedy/Lead attacker-independence check (76 (level, seed, controller)
pairs across C0/C2/C3, plus the project's named historical seeds) under the k=5 canonical config:
**max attacker-position delta vs. passive across all pairs: 0.0 m** (attacker trajectory remains
fully independent of defender behavior; A0 still exhibits no tactical defender evasion). Under an
active defender (greedy/lead), C3 130017 is intercepted well before its budget could ever bind
(`intercepted`, not `enemy_endurance_exhausted`) -- the pathology, and therefore the budget
mechanism's effect, is specific to the passive-A0 harness, as expected.

## Implementation

- `uav_defend/config/env_config.py`: `enemy_endurance_enabled` (default `False`),
  `enemy_endurance_path_factor` (default `5.0`), `reward_enemy_endurance_exhausted` (default
  `-100.0`, mirrors `reward_timeout`), with `__post_init__` validation (requires
  `mission_contact_mode == "swept"`; factor must be `> 0`).
- `uav_defend/envs/soldier_env.py`: `_enemy_endurance_budget`/`_enemy_endurance_traveled` state;
  budget initialization in `reset()`; full per-step travel-distance capture and the `endurance`
  swept-contact candidate, clamping, and bookkeeping in `step()`; the `enemy_endurance_exhausted`
  outcome branch; four new `info` fields in `_get_info()`.
- `experiments/journal_v2_protocol.py`: `V2_ENDURANCE_OVERRIDES` (`enemy_endurance_enabled=True,
  enemy_endurance_path_factor=5.0`), merged as the canonical V2 default inside
  `build_v2_env_config` (overridable per-call via `extra_overrides={"enemy_endurance_enabled":
  False}` for diagnostics/counterfactuals that need the pre-endurance V2 config, e.g. the
  endpoint-semantics counterfactual branch in `journal_v2_validation_core.py` and the legacy
  waypoint-planner fidelity harness in `tests/test_obstacle_prediction_fidelity.py`, both of which
  are orthogonal to endurance and explicitly disable it). **V1 defaults are untouched.**
- `experiments/journal_v2_trace.py`, `experiments/journal_v2_validation_core.py`: endurance fields
  added to the per-step trace schema and `FINAL_INFO_KEYS`.
- `tests/test_mission_endurance.py`: 21 tests covering all 12 items in Phase 18's checklist
  (disabled bit-identical, path accumulation, budget initialization, exact exhaustion, within-step
  tau, capture/interception/collision precedence in both directions, no-guidance-effect, config
  serialization, V1 unaffected).

## Notes on an unrelated discovery

While validating this feature's interaction with the existing legacy-waypoint-planner fidelity
tests (`tests/test_obstacle_prediction_fidelity.py`, which force
`SoldierEnv.STATEFUL_ENEMY_AVOIDANCE = False` via an autouse fixture to test the pre-stateful-
controller planner, unrelated to canonical V2's accepted avoidance controller), it was discovered
that the deprecated legacy planner produces a substantially longer, more circuitous path for C3
seed 130006 than the accepted stateful avoidance controller does -- long enough that, under the
canonical k=5 budget, it too would be truncated by `enemy_endurance_exhausted`. This is expected
and desired behavior for that (deprecated, non-canonical) planner; it does not affect the accepted
production stateful avoidance controller at all (confirmed: 130006 captures normally in 10 steps
under the stateful controller, with or without endurance). The fidelity test's helper now
explicitly disables endurance (orthogonal to its scope, consistent with the other diagnostic call
sites listed above).
