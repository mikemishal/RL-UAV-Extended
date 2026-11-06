# A1-CBF — hierarchical obstacle-vs-defender CBF priority (Prompt 5: safety-priority correction)

**Result: ACCEPTANCE GATE FAILS. Verdict C — HIERARCHICAL CBF FAILS — DO NOT FREEZE ATTACKER.**

The shared-slack design flaw identified in Prompt 4 (obstacle safety trading against defender avoidance under
severe joint conflict) is CONFIRMED FIXED: the new hierarchical QP never relaxes obstacle-safety constraints to
help the defender, verified via a dedicated regression test suite and reproduced directly on the exact collision
seed this bug caused (630069, now resolved with 0 collisions). However, Monte Carlo evidence revealed a SEPARATE,
pre-existing limitation exposed by this restructuring: 2 of the 3 known Prompt-4 stress-collision seeds (630123,
630203) still collide, and the full 200-case stress retest confirms 2/200 obstacle collisions (down from 3/200,
but not the required 0/200). Root-caused to a gap in the frozen, velocity-space-only CBF formulation (not a
priority-hierarchy bug) — documented in detail below. Tactical effect against Greedy, Kalman-Greedy, PN, Lead, and
RMPC is statistically IDENTICAL to Prompt 4 in every case (the large PN/RMPC benefits are fully preserved).

## 0. Motivation

Prompt 4's joint CBF-QP solved `min ||v-v_nom||^2 + 1e4*delta^2` with a SINGLE shared slack `delta` relaxing BOTH
obstacle and defender constraints simultaneously under hard-QP infeasibility. The 200-case joint stress set found
3/200 (1.5%) obstacle collisions, traced to this shared slack allowing the solver to "spend" relaxation on the
obstacle constraint when satisfying the defender constraint was cheaper to achieve that way. A building/obstacle
collision is a physical safety failure; defender interception is a tactical mission failure -- these are NOT
equivalent, and the frozen priority should be: **obstacle safety > velocity limits > defender avoidance**.

## 1. The exact QP change

**Old (Prompt 4):** `min ||v-v_nom||^2 + 1e4*delta^2` s.t. obstacle(v)>=0, defender(v)>=0 -- hard first; on
infeasibility, BOTH relaxed via the SAME `delta` in a shared soft QP.

**New (Prompt 5):**
1. **Step 1** (brief section 5): solve obstacle+bounds ALONE (defender entirely ignored). If infeasible
   (`priority_classification='obstacle_safety_infeasible'`), fall back to the EXISTING Prompt-2/3 obstacle-only
   hard->soft->emergency chain -- the defender is NEVER consulted to help satisfy obstacle safety, even here.
2. **Step 2a**: if Step 1 is feasible, try obstacle+defender+bounds ALL hard (warm-started from Step 1's
   solution). If feasible, `status='hard'` -- no slack used at all.
3. **Step 2b**: if Step 2a is infeasible (a genuine priority conflict), solve
   `min 0.5*||v-v_nom||^2 + 1e4*delta_D^2` s.t. obstacle+bounds HARD, `n_D^T(v-v_D) >= -alpha_D*h_D - delta_D`,
   `delta_D>=0`. `status='soft_fallback'`, obstacle residual verified ~0 at every solve.

`cbf_qp_status` stays in its established 3-value vocabulary (`hard`/`soft_fallback`/`emergency_brake`) for
diagnostic/harness compatibility. New diagnostics: `defender_slack`,
`defender_constraint_satisfied_without_slack`, `defender_constraint_violation_with_slack`,
`obstacle_constraints_hard`, `obstacle_min_residual`, `defender_min_residual`, `priority_conflict`,
`priority_classification` (`None` / `obstacle_safety_infeasible`).

## 2. Three known collision-case reproduction (brief section 7)

| Seed | Prompt-4 (shared slack) | Prompt-5 (hierarchical) | Diagnosis |
|---|---|---|---|
| 630069 | `enemy_static_obstacle_collision` | **`soldier_caught` (FIXED)** | Genuine priority-conflict case: `priority_conflict=True`, `defender_slack=13.29` observed at the conflict step, `obstacle_constraints_hard=True` throughout. The hierarchy correctly absorbed the conflict entirely via the defender's slack. |
| 630123 | `enemy_static_obstacle_collision` | **`enemy_static_obstacle_collision` (STILL COLLIDES)** | NOT a priority-hierarchy bug -- `obstacle_constraints_hard=True` and obstacle residual <=1e-4 at every substep, confirmed. Root cause below. |
| 630203 | `enemy_static_obstacle_collision` | **`enemy_static_obstacle_collision` (STILL COLLIDES)** | Same mechanism as 630123. |

### Root cause of the 2 persisting collisions

At the colliding substep for seed 630123, the active obstacle hazard already has `h = -2.345` (deeply negative --
the stopping-margin formula's own "corrigible within one `dt_sub`" assumption is already violated when this
obstacle first entered the 20 m active range, because the attacker was closing on it fast). Solving the hierarchical
QP with the defender's hard constraint present produces `v_safe = [11.71, 2.35, -5.0]` (an extreme corner solution,
near max horizontal speed and max descent rate) vs. `v_safe = [3.96, 2.35, -0.32]` with the SAME obstacle hazard and
NO defender constraint at all (confirmed: `defender_cbf_enabled=False` on the IDENTICAL starting kinematic state
survives safely). Both solutions satisfy the obstacle's y-axis requirement (`vy=2.345`) EXACTLY and have
obstacle residual ~0 -- but the "with defender" solution points substantially more in x/z (components the
obstacle constraint does not pin down, driven by the QP's own 1e4-penalty trade-off trying to minimize
`delta_D`). Since the EXECUTOR applies ONE acceleration-limited step (0.6 m/s per 0.1 s substep) TOWARD the FULL
3-D target, a target pointing further off-axis "wastes" more of that budget on components irrelevant to the
obstacle escape, leaving less of it available for the y-axis correction that is actually urgent. This is a
genuine gap in the frozen, velocity-space-only CBF-QP formulation: it has no direct knowledge of the executor's
SEPARATE acceleration-rate limit (`max_accel` only enters the stopping-margin FORMULA, never as a QP constraint on
`v` itself) -- a pre-existing property of the architecture since Prompt 1/2, now exposed by the mere PRESENCE of a
defender hard-constraint (regardless of whether it ends up binding). **This is not authorized to be fixed in this
prompt** (would require adding an acceleration-reachability term to the QP itself, altering the frozen CBF
formulation) -- see section 7 below.

A warm-start experiment (seeding Step 2a/2b from Step 1's own obstacle-only-optimal point, rather than the raw
clipped `v_nom`) was tried and had **zero effect** on these 2 cases -- SLSQP converges to the identical point
regardless of initial guess here, ruling out a cold-start/local-optimum explanation. The warm-start is kept in the
final implementation as a harmless, principled consistency improvement (it DID fix an unrelated numerical
reliability issue caught by a dedicated unit test: introducing the slack variable into every solve, even when zero
slack is optimal, made SLSQP less reliable at finding `delta_D=0` from a cold start in the common no-conflict
case).

## 3. 200-case joint stress retest (brief section 8)

| metric | Prompt 4 (shared slack) | Prompt 5 (hierarchical) |
|---|---:|---:|
| Obstacle collisions | 3/200 (1.5%) | **2/200 (1.0%)** |
| Numerical failures | 0 | 0 |
| Simultaneous-constraint rate | 100% | 100% |
| Soft-QP episodes | 50 (25%) | 49 (24.5%) |
| Emergency-brake episodes | 3 (1.5%) | 7 (3.5%) |
| Defender interceptions | 61 | 65 |
| Asset captures | 133 | 131 |

**FAILS the brief's explicit section-8/11 requirement of 0/200 obstacle collisions.** The 2 remaining collisions
are exactly seeds 630123 and 630203 (the SAME 2 unresolved from the 3-case reproduction above). Emergency-brake
episodes roughly doubled (3->7): the stricter obstacle-hard priority makes the controller choose the safe
(zero-velocity) fallback more often when a genuine conflict cannot be resolved via defender slack alone, which is
the CORRECT, intended behavior (brief: "It is acceptable for the defender to intercept the attacker... preferable
to physically driving through an obstacle") -- it is simply insufficient to eliminate the 2 pre-existing
acceleration-reachability edge cases.

## 4. Primary 1,000-Greedy re-evaluation (brief section 9)

| | obstacle-only | Prompt-4 (shared slack) | Prompt-5 (hierarchical) |
|---|---:|---:|---:|
| Defender interception | 96.4% | 95.1% | 95.1% |
| Asset capture | 3.6% | 4.9% | 4.9% |
| Obstacle collisions | 0/1000 | 0/1000 | **0/1000** |
| Numerical failures | 0 | 0 | 0 |
| Route errors | 0 | 0 | 0 |
| Emergency-brake episodes | -- | 0 | 1 (seed 600006, 1 step, outcome still `soldier_caught`) |
| QP hard/soft/emergency step fraction | -- | 100%/0%/0% | 99.9962%/0.14%/0.004% |

**Capture and interception are statistically IDENTICAL between Prompt 4 and Prompt 5** against Greedy (common
random numbers, same 1,000 seeds). Physical safety is preserved (0 obstacle collisions, matching both prior
variants) at the cost of one single additional emergency-brake step across the entire 1,000-episode campaign (a
benign, momentary hover that did not change that episode's successful outcome). **This directly answers brief
section 9's question: making the system physically safe PRESERVES tactical effect against Greedy exactly --
neither improves nor degrades it.**

## 5. Generalization retest (brief section 10)

| Defender | Interception (oo -> Prompt-4) | Interception (oo -> Prompt-5) | Obstacle collisions (Prompt-5) |
|---|---|---|---:|
| Kalman-Greedy | 97% -> 96% (-1 pp) | 97% -> 96% (-1 pp) | 0/200 |
| **PN** | **95% -> 62% (-33 pp)** | **95% -> 62% (-33 pp)** | 0/200 |
| Lead | 97% -> 95% (-2 pp) | 97% -> 95% (-2 pp) | 0/200 |
| **RMPC** | **75% -> 37% (-38 pp)** | **75% -> 37% (-38 pp)** | 0/200 |

**Every single metric is statistically identical between Prompt 4 and Prompt 5, across all 4 additional defender
families (800 episodes).** Zero obstacle collisions across the entire generalization diagnostic. This directly
and unambiguously answers brief section 10's key question: **yes, the large PN/RMPC tactical gains are fully
preserved when obstacle safety is made strictly hard.**

## 6. Historical regressions

- **C0 + scripted A0**: 1,000 seeds, **0 mismatches** vs. the historical parent-commit baseline -- bit-for-bit
  identical (the `enemy_control_mode=="legacy"` path and the `defender_hazard is None` path both reuse the
  UNCHANGED Prompt-2/3 obstacle-only chain verbatim).
- **Static-clutter (no-defender) regression**: 500 episodes, **0 obstacle collisions**, 0 scenario-generation
  failures, 0 route errors -- gate PASSES.

## 7. Acceptance gate (brief section 11)

| Gate | Requirement | Result | Status |
|---|---|---|---|
| Primary 1,000 Greedy | obstacle collisions=0, numerical failures=0, route errors=0 | 0 / 0 / 0 | **PASS** |
| 200 stress | obstacle collisions=0, numerical failures=0 | **2** / 0 | **FAIL** |
| Historical | C0/A0 bit-identical; static-clutter regression passes | identical; passes | **PASS** |
| Tactical | no catastrophic loss (Greedy capture worsens <=5pp, OR PN/RMPC benefits largely disappear) | 0.0 pp change; PN/RMPC benefits fully preserved | **PASS** |
| **Overall** | **ALL must pass** | **200-stress gate fails** | **FAIL** |

Per the brief's own section-11 framing, a 200-stress-set failure on the explicit physical-safety criterion means
this is NOT a "physical safety passes, tactical behavior weak" (verdict B) situation -- physical safety itself
does not fully pass. The correct, honest verdict is **C**.

## 8. Targeted tests

`tests/test_cbf_defender_v2.py`, 10 tests covering: obstacle constraints cannot receive slack (severe conflict),
defender-only slack, positive-slack/`priority_conflict` flag correctness, obstacle-hard priority at every
feasible solve, `obstacle_safety_infeasible` classification + defender-dropping behavior, deterministic
reproduction of the exact 3 known collision seeds (documenting the HONEST outcome: 1 fixed, 2 still collide --
see section 2 above), hierarchical-vs-shared-slack regression (obstacle residual independent of conflict
severity), full diagnostic-field type/presence checks, bit-for-bit degeneration to Prompt 2/3 when no defender is
observable, and `filter_velocity` diagnostic pass-through. All 10 pass. Full project regression (`pytest tests -q
-p no:cacheprovider --ignore=...`) run with 0 new failures relative to the Prompt-4 baseline.

## 9. Limitations and next steps (not authorized in this prompt)

The 2 residual collision mechanisms stem from the CBF-QP operating entirely in velocity space with no direct
knowledge of the executor's separate, per-substep acceleration-rate limit. Candidate fixes for a FUTURE prompt
(explicitly NOT authorized here, per brief section 12's framing):

1. **Acceleration-reachability-aware CBF-QP**: add a hard constraint `||v - v_current|| <= max_accel * dt_sub`
   directly into the QP (not just the stopping-margin formula), so the solver cannot select a velocity target that
   is mathematically CBF-feasible but physically unreachable within the substep.
2. **Collision-cone CBF reformulation** (brief's own suggestion).
3. **Longer-horizon defender prediction** (brief's own suggestion) -- unrelated to this specific failure mode but
   listed by the brief as a parallel future option.

None of these were implemented here; doing so would require touching the frozen stopping-margin CBF formula /
`active_range` architecture, outside this prompt's authorized scope ("Do NOT tune defender radius or alpha yet").

## 10. Reproduction

```
pytest tests/test_cbf_defender_v2.py -q
python -c "import experiments.journal_cbf_attacker_defender_foundation as F; F.run_stress_campaign(n_scenarios=200, seed_base=F.STRESS_SEED_BASE)"
```

Campaign results are cached (content+protocol-hash verified) under
`docs/journal_cbf_attacker/defender_cbf_v2/results/*.json` with the `_hierarchical` tag suffix, kept entirely
separate from Prompt 4's `docs/journal_cbf_attacker/defender_cbf/results/` cache to avoid any stale-cache
collision between the two controller versions.
