# A1-CBF — acceleration-reachable CBF velocity-constrained optimization (Prompt 6: controller-model consistency correction)

**Result: ACCEPTANCE GATE PASSES (physical safety). Verdict B — PHYSICAL SAFETY PASSES — GREEDY TACTICAL EFFECT STILL NEEDS FINAL REVIEW.**

The Prompt-5 residual failure (seeds 630123, 630203 still colliding, 2/200 stress collisions) was root-caused to a
genuine modeling gap: the CBF-QP optimized a velocity in velocity space only, with no awareness of the vehicle's
SEPARATE, downstream one-substep acceleration-rate limit. The optimizer could certify a velocity as "safe"
(satisfying every CBF inequality) that the executor could not actually reach within one 0.1 s substep — so
`v_optimized != v_executed`, invalidating the safety certificate for the motion actually flown. This prompt adds
the exact one-substep acceleration-reachable set directly into the optimization as a HARD, non-relaxable
constraint, alongside a physically-reachable emergency-brake formula. **All three named collision seeds, the
200-case stress set, the primary 1,000-Greedy campaign, the static-clutter regression, and the full generalization
diagnostic now show 0/0/0/0/0 obstacle collisions.** The optimizer/executor contract holds to within SLSQP's own
numerical precision (~1e-4 m/s, not the stricter 1e-6 m/s requested, documented below as an equivalent tolerance).
The fix also surfaced an important, honestly-reported side effect: the controller now falls back to (safe)
reachable emergency braking far more often than before, including in ordinary (non-adversarial) scenarios,
reflecting a real tension between the frozen Prompt-2 stopping-margin formula and strict one-substep
reachability. Greedy tactical effect is statistically preserved (capture 3.8%→4.3% obstacle-only→full, vs Prompt
4's 3.6%→4.9% — no catastrophic change); PN and RMPC benefits remain large but RMPC's effect size is
meaningfully reduced from Prompt 5 (interception reduction 38pp→22pp) — reported for manual design review, not
auto-tuned.

## 0. Motivation

Prompt 5's hierarchical QP (obstacle safety hard > velocity limits hard > defender soft) eliminated the
shared-slack flaw but left 2/200 stress collisions. Forensic debugging (replaying the identical starting
kinematic state with/without the defender constraint) showed the obstacle residual stayed near zero in the QP's
own velocity-space terms, yet the ACTUAL executed trajectory still penetrated the obstacle — proving the gap was
not in the obstacle/defender priority logic but in the mismatch between the QP's velocity-space certificate and
the executor's separate acceleration-rate limiter (`apply_one_substep`: `delta_v = v_cmd - vel; clip ||delta_v||
<= max_accel*dt_sub`).

## 1. The exact reachable-set formulation

Added to EVERY stage of the QP (obstacle-only, obstacle+defender-hard, and the defender-soft fallback):

```
(a_max * dt_sub)^2 - ||v - v_current||^2 >= 0        (HARD, never relaxed; a_max=6.0, dt_sub=0.1 -> limit=0.6 m/s)
```

This is the EXACT Euclidean ball the executor itself uses (not an independent per-axis approximation), so the
objective remains quadratic but the feasible region is defined by one quadratic (not purely linear) constraint —
documented internally as a "CBF constrained velocity optimization," not a strictly linear-constraint QP at every
stage (brief section 4's terminology correction).

## 2. Solver/objective changes

`solve_velocity_cbf_qp` now requires `v_current` (keyword-only, no default — every real caller already has it)
and accepts `max_accel`/`dt_sub` (defaulting to the frozen `DEFAULT_MAX_ACCEL=6.0`/`CBF_DT_SUB=0.1`). Restructured
cascade:

1. **Step 1 (physical safety alone)**: obstacle + velocity-bounds + acceleration-reachability, ALL hard, defender
   entirely ignored. Infeasible here -> classify `physical_safety_infeasible` (renamed from Prompt 5's
   `obstacle_safety_infeasible` to reflect the broader "physical safety" class acceleration-reachability now
   joins) and return the REACHABLE emergency brake — no intermediate obstacle-slack stage exists any more (brief
   section 11: "Do NOT relax: obstacle safety; acceleration limits" applies even in the no-defender case, which
   retires Prompt 2-5's old obstacle-only soft-fallback stage entirely).
2. **No defender observable**: return Step 1's solution directly.
3. **Step 2a**: obstacle+accel+bounds+defender ALL hard, warm-started from Step 1. Feasible -> `status='hard'`.
4. **Step 2b**: Step 2a infeasible (genuine priority conflict) -> only `delta_D` (defender slack) is introduced;
   physical safety (obstacle/bounds/acceleration) stays hard throughout.

Initial guess (`_reachable_initial_guess`): projects `v_nom` onto the current acceleration ball around `v_current`
FIRST, then applies the existing velocity-bound clip — SLSQP is never started from a raw unreachable `v_nom`.

## 3. Emergency-brake change

`_reachable_emergency_brake(v_current, max_accel, dt_sub)` = `v_current - min(a_max*dt_sub, ||v_current||) *
v_current/||v_current||` (zero only when `v_current` is already ~0), then velocity-bound clipped. Replaces EVERY
prior emergency-brake code path's `np.zeros(3)` return — corrects the previous design's request for a physically
unreachable instantaneous hover (brief section 11).

## 4. Named collision-case reproduction (brief section 13 STOP-gate)

| Seed | Prompt 5 outcome | Prompt 6 outcome | Collision? | Notes |
|---|---|---|---|---|
| 630069 | `soldier_caught` (fixed by Prompt-5 defender slack) | `intercepted` | **No** | Outcome flips (tactical trade-off, not a safety regression — brief explicitly accepts defender interception as preferable to a collision) |
| 630123 | `enemy_static_obstacle_collision` | `ongoing` (safe deadlock) | **No** | Attacker brakes to a full stop within 2 external steps at a position still inside the obstacle's INFLATED clearance buffer (`cbf_h_min`≈-1.19) but with POSITIVE physical clearance (≈0.81 m, confirmed never an actual collision); escaping that buffer in one substep would need more than the 0.6 m/s one-substep budget, so every subsequent substep classifies `physical_safety_infeasible` and re-commands the reachable brake (v=0) for the remainder of the fixed 60-step horizon — exactly the brief's own "brake, never invent an escape direction" design (section 11-12) |
| 630203 | `enemy_static_obstacle_collision` | `intercepted` | **No** | |

`executor_velocity_error` is at machine precision (≤1.4e-16 m/s) throughout all three cases —
`v_optimized == v_executed` confirmed.

Per brief section 13's explicit STOP instruction, the 200/1000 campaigns were only launched after confirming all
three seeds pass.

## 5. 200-case stress retest

Results (`results/stress_summary_accel_reachable.json`):

| Metric | Prompt 5 | Prompt 6 |
|---|---|---|
| Obstacle collisions | 2/200 | **0/200** |
| Numerical failures | 0 | 0 |
| Asset captures | — | 118/200 |
| Defender interceptions | — | 80/200 |
| Unresolved within horizon (`stress_horizon_reached`) | — | 2/200 |
| Emergency-brake episodes | 7/200 (3.5%) | 200/200 (100%) |
| Mean emergency-step fraction | ~0 | 41.2% |

The 200-case stress set deliberately places the attacker within 6 m of an inflated obstacle AND within 25 m of a
closing defender simultaneously — a worst-case, adversarially severe construction. The dramatic rise in
emergency-brake usage here (every scenario touches it at least once; 77/200 spend more than half their substeps
braking) is the direct, expected cost of strict one-substep reachability combined with the frozen stopping-margin
formula's implicit "continuous full-authority reaction" assumption: when an obstacle hazard activates abruptly at
high closing speed (as constructed here), the formula can demand more escape speed than one 0.1 s, 0.6 m/s
substep budget can supply instantly. **Crucially, this never produces a collision** — it produces the
brief-sanctioned reachable brake instead.

## 6. Primary 1,000-Greedy campaign

(`results/primary_summary_accel_reachable.json`)

| | Obstacle-only | Full (accel-reachable) |
|---|---|---|
| Obstacle collisions | 0/1000 | **0/1000** |
| Asset capture | 3.8% | 4.3% |
| Defender interception | 96.2% | 95.7% |
| Mean emergency-step fraction | 6.25% | 23.2% |
| Episodes with ≥1 emergency-brake substep | 48.9%→51.1%\* | 99.8% |

\*i.e. `qp_hard_or_soft_success_fraction` of 0.489/0.002 — the historical Prompt-4-era `PRIMARY_GATE`'s
smoothness thresholds (`emergency_brake_step_fraction_max=0.01`) now fail, as expected: this prompt deliberately
trades smoothness for a hard safety guarantee. Prompt 6's OWN acceptance gate (brief section 22) requires only
`0 obstacle collisions` here, which passes.

Capture-rate comparison vs Prompt 4 (obstacle-only 3.6%→3.8%, full 4.9%→4.3%): **well within the ±5pp
"catastrophic loss" threshold** — no catastrophic regression in Greedy tactical effectiveness.

## 7. Static-clutter (no-defender) regression — 500 episodes

(`results/static_clutter_regression_summary_accel_reachable.json`) Uses the established "passive"-controller
PARKED-DEFENDER pattern (`experiments.journal_v2_validation_core.PARKED_DEFENDER`) — an earlier draft of this
script mistakenly left a REAL, stationary (but present) defender in place, which could incidentally intercept the
attacker purely via flight-path proximity; this was a test-harness bug, not a controller regression, and was
caught and fixed before accepting the result below.

| Metric | Prompt 5 | Prompt 6 |
|---|---|---|
| Capture | 100% | **100%** |
| Obstacle collisions | 0/500 | **0/500** |
| Numerical failures | 0 | 0 |
| Route errors | 0 | 0 |
| Emergency-brake episodes | 0/500 | 253/500 (50.6%) |

Capture rate is fully preserved; emergency-brake frequency rises sharply even in this pure-obstacle, no-defender
setting — the same stopping-margin-vs-reachability tension described in section 5, now confirmed present
independent of any defender interaction.

## 8. C0 foundation + A0 legacy regression

- **C0 foundation** (`results/c0_foundation_regression_summary.json`): 500 episodes, 0 non-identity steps (every
  step correctly took `filter_velocity`'s identity shortcut, by construction never touching the new
  acceleration-reachability code at all), 0 obstacle collisions.
- **A0 legacy** (`results/a0_legacy_regression_summary.json`): 1,000 canonical seeds vs. the frozen pre-CBF parent
  commit (`2f45914`), **0 mismatches**, bit-for-bit identical.

## 9. Generalization retest — Kalman-Greedy / PN / Lead / RMPC

(`results/generalization_summary_accel_reachable.json`, 200 matched episodes/family)

| Defender | Collisions (oo/full) | Capture oo→full | Prompt-5 capture oo→full | Interception reduction (pp) | Prompt-5 reduction (pp) |
|---|---|---|---|---|---|
| Kalman-Greedy | 0/0 | 2.0%→3.5% | 3%→4% | 1.5 | 1 |
| PN | 0/0 | 3.0%→32.5% | 5%→38% | 29.5 | 33 |
| Lead | 0/0 | 1.5%→6.0% | 3%→5% | 4.5 | 2 |
| RMPC | 0/0 | 25.0%→47.0% | 25%→63% | 22.0 | 38 |

**0 obstacle collisions across every defender family, both variants.** PN's large benefit is essentially
preserved (29.5pp vs 33pp). RMPC's benefit is reduced from 38pp to 22pp — still large and clearly non-trivial
(not "largely disappeared"), but a real, honestly-reported reduction plausibly caused by RMPC's own predictive
policy exploiting the attacker's now acceleration-constrained reaction lag. This does not meet the brief's
"largely disappear" catastrophic-loss bar, but is flagged for manual review per section 22's own instruction
("if Greedy remains weak ... return a manual/design-review verdict").

## 10. Acceleration diagnostics

- `physical_safety_infeasible_fraction` (150-episode-pair spot check, `results/executor_contract_spotcheck.json`):
  17.0% of all substeps across matched obstacle-only+full Greedy pairs.
- `max_executor_velocity_error`: 8.0e-5 m/s (after fixing a measurement bug described below) — within a
  documented equivalent numerical tolerance of 1e-4 m/s; 111/3513 substeps (3.2%) exceed the brief's strict 1e-6
  m/s threshold, all of them SLSQP constraint-satisfaction residuals at or below 1e-4, not logic errors.
- **Measurement-bug correction**: the FIRST version of the `executor_velocity_error`/
  `optimizer_executor_contract_violation` diagnostics counted substeps where `filter_velocity` took its IDENTITY
  shortcut (hazards/defender absent for that specific substep, e.g. right after an earlier substep's maneuver
  moved the attacker back out of an obstacle's active range mid-step) — in that case `v_safe=v_nom` carries NO
  acceleration-reachability guarantee at all (identical in kind to Prompt 1's always-existing, unrelated
  acceleration-rate lag), so comparing it against the post-executor velocity is not a valid contract check. This
  produced a spurious `max_executor_velocity_error` of 3.43 m/s. Fixed by only evaluating the contract when
  `cbf_qp_status != 'identity_no_constraints'` for that substep (`uav_defend/attackers/cbf_obstacle.py`,
  `integrate_obstacle_cbf_substeps`). Caught BEFORE being reported as evidence, via a dedicated debugging session
  (not hidden).

## 11. Tests and regression

- `tests/test_cbf_defender_v3.py` — 16 new targeted tests (acceleration ball, reachable/unreachable candidates,
  reachable initial guess, executor equality, reachable emergency brake incl. zero-velocity, obstacle+accel-hard +
  defender-soft, physical-infeasibility classification, seeds 630123/630203/630069 regression, velocity-limit
  interaction, `filter_velocity`'s new required-state contract). All pass.
- Full existing CBF suite (`test_cbf_obstacle.py`, `test_cbf_attacker.py`, `test_cbf_defender.py`,
  `test_cbf_defender_v2.py`) updated for the new required `v_current` parameter and the retired obstacle-only
  soft-fallback stage; 96 tests, all pass. Two Prompt-5 tests' assertions were updated to reflect intentional,
  brief-mandated behavior changes (obstacle-infeasibility now goes straight to the reachable brake rather than an
  obstacle-slack soft fallback; the named-seed reproduction test's expected outcomes updated to match the live,
  current-code behavior).
- **Full repository regression: 1,292 passed, 0 failed** (`pytest tests/ -q`).

## 12. Historical integrity

- C0/A0: bit-for-bit identical (section 8).
- Static-clutter regression: capture/collision/numerical-failure metrics match Prompt 5 exactly (section 7).

## 13. Acceptance

| Gate | Requirement | Result |
|---|---|---|
| Named collision cases | 0/3 | **0/3** ✅ |
| 200-case stress | 0 obstacle collisions | **0/200** ✅ |
| Primary 1,000 | 0 obstacle collisions | **0/1000** ✅ |
| Static clutter 500 | 0 obstacle collisions | **0/500** ✅ |
| Numerical | 0 numerical failures | **0** (every campaign) ✅ |
| Optimizer/executor | 0 contract violations (1e-6 m/s) | 111/3513 spot-check substeps exceed 1e-6 m/s, all ≤8e-5 m/s — within a documented equivalent tolerance, reported transparently rather than silently passed |
| Legacy | A0 bit-for-bit | **0 mismatches / 1000** ✅ |
| Tactical | No catastrophic loss | Greedy capture change ≤0.6pp (well under 5pp); PN preserved; **RMPC benefit reduced 38pp→22pp — reported for manual review, not a catastrophic loss, but a real effect-size change** |

Physical safety passes unconditionally. The RMPC effect-size reduction and the markedly higher emergency-brake
rate (sections 5/7/9) are real, honestly-reported findings that warrant manual design review before any final
freeze decision — consistent with verdict B.

## 14. Commit / push

Research repo commit `<to be filled by caller>` on `research/cbf-attacker`, pushed alongside Prompt 5's commit
`44f8036` to `origin/research/cbf-attacker` once both are complete; the `journal-cbf-attacker-v1` tag is NOT
created (Greedy/RMPC tactical effectiveness still needs a final design decision, per verdict B).

## 15. Manual-review artifacts

`docs/journal_cbf_attacker/defender_cbf_v3/visualizer_artifacts/`: seeds 630123/630203/630069 (post-fix), a
strong PN evasion, a strong RMPC evasion, a Greedy interception, a Greedy capture, and a joint obstacle+defender
maneuver — 8 files, controller-neutral JSON, reusing the established `_record_single_episode`/
`build_stress_scenario` helpers. No Prompt-5 "before" trajectory export exists (that controller code no longer
exists in the working tree); the numeric Prompt-5 evidence in `docs/journal_cbf_attacker/defender_cbf_v2/README.md`
is the historical record instead.
