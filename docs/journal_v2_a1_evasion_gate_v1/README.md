# Journal V2 A1 — defender-evasion gating study (`a1_v2_action_v6`) — NEGATIVE RESULT

**Verdict D: existing tactical control options are exhausted; A1 must move to route-level / planning-level learning.** No nonzero blend passes the pre-registered gates, so no action set is selected, no protocol tag is created, `a1_v2_training_v4` is not prepared, and no PPO or defender training was run.
Starting point research `ac0f585` (clean); A0 tag `journal-v2-a0-canonical-v1` = `bdbee45`; `uav_defend` byte-identical to it; reward `a1_v2_reward_v2` unchanged.

## Question
Can A1 gain useful tactical intelligence by learning WHEN and HOW STRONGLY to activate the existing deterministic defender-evasion primitive `SoldierEnv._compute_enemy_evasion`, while canonical obstacle avoidance retains safety authority?

## Primitive audit (`primitive_audit.json`, from the live code)
- Pure function of the two true 3-D positions (no `self` writes, repeat calls identical): `evasion_component = gain(0.75) * alpha * u_away`, `alpha = clip(1 − d/20 m, 0, 1)`; direction only, no speed term; finite at d = 0 (fallback +z).
- Canonical V2 (`direct_obstacle_only`) zeroes it in the command path; legacy use was `u_raw = r_hat + weave + noise + evasion`.
- Order inside `_move_enemy`: primitive → (A1 override replaces `u_base`) → frozen obstacle planner, which has the last word. STOP rule (primitive applied *after* the planner) **not triggered**.
- Caveat verified in `stateful_avoidance.py`: the planner returns `u_base` unchanged whenever its direct *mission* rollout is predicted safe, so it vetoes a blended command only in states where the direct line is itself unsafe. This was therefore tested empirically (G1/G6), not assumed.
- Stateless ⇒ observation `a1_v2_obs_v1` (37-D) retained; the 43-D pulse-state observation is not used. Evasion (d < 20 m) lies strictly inside the 25 m information gate, so the gate is structurally redundant.

## Interface and protocol (pre-registered, `interface_specification.json`)
`u(b) = normalize(m + b·evasion_component)`, b ∈ {0, 0.25, 0.5, 0.75, 1.0}; b = 0 returns the zero override (env's own `u_base`). Forced to 0 without defender information; suppressed (and logged) whenever the frozen common layer owns steering. Seeds 145000–145099, C0–C3, greedy / kalman_greedy / pn / lead / rmpc, matched BASE; FIXED_b and gated SWITCH_b. 18,000 episodes, 20 cells saved atomically with per-cell SHA-256, spec hash and a single-writer lock.

## Results (`sweep_results.md`, `sweep_summary.json`)
| blend | gained collisions (C0/C1/C2/C3) | exhaustion (BASE 0) | path >5× | captures (BASE 232) | interception→capture / reversals | G1 | G2 | G3 | G4 | G6 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 0/1/3/7 | 23 | 0 | 233 | 38 / 3 | FAIL | FAIL | FAIL (3.0% diverge) | PASS | PASS |
| 0.50 | 0/0/2/2 | 36 | 6 | 234 | 47 / 4 | FAIL | FAIL | PASS | PASS | FAIL |
| 0.75 | 0/0/11/20 | 49 | 18 | 208 | 47 / 2 | FAIL | FAIL | PASS | FAIL | FAIL |
| 1.00 | 0/1/14/23 | 48 | 15 | 219 | 62 / 6 | FAIL | FAIL | PASS | FAIL | FAIL |

G5 (blend 0 ≡ canonical A0): **PASS**, 2,000/2,000 bit-identical (max deviation 0 m); deterministic replay and finiteness pass; SWITCH_b ≡ FIXED_b trajectory-identical in all cells. Orbits 0, timeouts 0, ceiling/sustained hover 0.

## Mechanisms
- **Collisions:** 83 of 84 gained collisions occur while the frozen layer already owns steering, after evasion ended (typically > 3 steps earlier): the evasion displacement leaves a state the reactive planner cannot recover — the same closed-loop failure as designs 2, 3 and 5. Only 1 occurs on an evasion step; evasion was never executed while the planner owned steering (0).
- **Exhaustion / path:** every exhaustion (23–49) contains ≥ 1 fly-by (mean ≈ 2–2.6 overshoots); all had been BASE captures (11–12 steps) or interceptions, and become ≈ 56-step failures. The >5× cases are exhaustions plus 2 collisions and 1 interception.
- **Authority is real but cancelled:** up to 62 interception→capture conversions, but captures lost to exhaustion offset them (net +1, +2, −24, −13 captures).
- 0.50 is the nearest miss (4 gained collisions of 2,000, 36 exhaustions); the strict criterion was pre-registered and not relaxed.

## What was not done / limits
No switching-stress test of a final action set and no training-runner preparation (no set selected). RMPC included in the grid; fixed defenders only; the legacy weave/heading-noise terms are excluded (canonical V2 forbids RNG). Not tested: look-ahead admission of the evasion command, a different evasion geometry, or planner changes — all outside the "existing primitive, minimum divergence" scope.

## Files
`primitive_audit.json`, `interface_specification.json`, `repository_verification.json`, `sweep_summary.json`, `sweep_results.md`, `ablation_record_v3.md`, `manuscript_notes.md`, `unit_test_result.json`, `stage08_regression_summary.json`, `stage09_checkpoint_integrity.json`, `task_state.json`. Raw cells (SHA-256 per cell) remain in the gitignored `results/.../a1_v2_training_v4/evasion_gate_sweep/`.
Code: `experiments/journal_v2_a1_eg.py` (env), `_eg_sweep.py`, `_eg_report.py`, `_eg_audit.py`, `_eg_state.py`; tests `tests/test_journal_v2_a1_eg.py`.
