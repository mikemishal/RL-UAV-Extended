# Manuscript notes — defender-evasion gating as the A1 tactical interface (rejected)

## Motivation
Four earlier A1 interfaces failed (unrestricted continuous steering, bounded residual, filtered residual, obstacle-primitive reuse) and a fifth (finite LEFT/RIGHT/CLIMB pulses with mandatory rejoin) was rejected on safety. Before proposing a route-level learner we tested the minimum-divergence alternative: reuse the simulator's own, deterministic defender-evasion primitive and let the learner choose only its strength.

## Hypothesis (pre-registered)
A learned attacker can obtain useful tactical intelligence by learning when and how strongly to activate the existing evasion primitive while canonical obstacle avoidance retains safety authority. Accepted only if G1 (zero gained obstacle collisions at every clutter level), G2 (no mission pathology), G3 (meaningful authority), G4 (path/endurance), G5 (blend 0 ≡ A0) and G6 (planner authority) all hold for at least one nonzero strength.

## Protocol
Action `a1_v2_action_v6`: b ∈ {0, .25, .5, .75, 1}; `u = normalize(m + b·evasion_component)` with the unmodified primitive; forced to b = 0 without defender information; suppressed when the frozen planner owns steering. Observation `a1_v2_obs_v1` (37-D; the primitive is stateless), reward `a1_v2_reward_v2` unchanged. Deterministic sweep, no PPO: 4 clutter levels × 5 defenders × 100 matched seeds (145000–145099) × {BASE, 4 fixed, 4 gated-switching}; 18,000 episodes; thresholds frozen and hashed before running. Primitive audited from live code.

## Quantitative findings
- BASE ≡ canonical A0: 2,000/2,000 bit-identical; gated switching ≡ fixed blend (the 25 m gate is redundant with the primitive's 20 m radius).
- Gained collisions out of 2,000 matched episodes (BASE: 0): 11 / 4 / 31 / 38 for b = .25 / .5 / .75 / 1.0, all in C1–C3 (none at C0). 83 of 84 occur while the frozen layer owns steering after evasion has ended.
- Endurance exhaustion (BASE: 0): 23 / 36 / 49 / 48, each containing fly-bys; timeouts 0; orbits 0. Path excess > 5×: 0 / 6 / 18 / 15 (BASE: 0).
- Authority: 3.0 / 6.3 / 8.8 / 11.3% of episodes diverge ≥ 1 m from BASE; 38 / 47 / 47 / 62 interceptions become captures (3 / 4 / 2 / 6 reversals). Net captures vs BASE (232): +1 / +2 / −24 / −13.

## Interpretation
The primitive is a legitimate defender-driven signal, and the planner ordering is correct (evasion is computed before, and never executed over, the planner). Yet a perturbation of the *current heading* has two unavoidable costs in this simulator: it can hand the reactive planner a state it cannot recover (collision tail in clutter), and the small deflection that rescues some interceptions turns near-captures into fly-bys that exhaust the finite endurance budget. The two effects cancel the benefit. The same boundary caused the failure of residual, filtered-residual and pulse interfaces despite very different parameterizations, which supports the conclusion that heading-level tactical formulations are exhausted and A1 requires route-/plan-level learning with look-ahead safety.

## Limitations
Fixed (non-learning) defenders only; scripted policies, not learned ones, were evaluated (a learned gate might exploit state-dependence, but the gate is already redundant and the unsafe tail exists at every strength > 0); seeds 145000–145099 are a diagnostic set; pooled gates use 500 episodes per level and defender family, so 4 collisions (b = 0.5) is a small count — the strict pre-registered criterion was nevertheless kept; the legacy weave/noise terms are excluded; no look-ahead admission or modified planner was tested (out of scope).

## Decision
**Architecture rejected** (verdict D). Retained: reward v2; the strict gated-comparison methodology; the finding that heading-level modifiers cannot be made safe on the frozen reactive layer. Planned next: route-level / planning-level learned architecture, pre-registered as a new action version.
