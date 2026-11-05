# Manuscript notes — A1-CBF pursuit/dynamics foundation

## Claim supported
A deterministic, literature-motivated attacker control law — proportional guidance on the current relative target
position, an (identity, for now) CBF/QP safety-filter interface, and an acceleration-limited quadrotor velocity
executor — solves the canonical moving-target benchmark that defeated four stages of PPO/behaviour-cloning
development (Stage 4, 4A, 4A-1 of the A1 curriculum): 100.0 % capture, 0 repeated fly-bys, 0 orbit episodes, 0
numerical failures over 1,000 matched scenarios, with **no RL training**. This is an engineering/foundation result
(it establishes that the right CONTROL ARCHITECTURE, not more learning, fixes the recurring failure mode), not a
claim that A1-CBF is a research contribution in itself — the ongoing research direction is the counter-UAS
defender.

## Why the PPO/BC attacker direction was stopped
Stages 4/4A/4A-1 diagnosed, in increasing detail, why a neural-network attacker trained on this benchmark kept
producing repeated near-miss fly-bys: missing sensitivity to the asset-position input (Stage 4A), and then — after
that representation deficiency was directly fixed by randomized-goal imitation (Stage 4A-1) — a textbook
behaviour-cloning covariate-shift failure (open-loop cosine ≈0.98, closed-loop cosine collapsing to ≈0.03–0.48).
Each stage would have required a further, separate learning intervention (DAgger-style relabeling, then presumably
more curriculum stages) with no guarantee of eventually reaching reliable capture, for a sub-problem (a scripted
adversary) that is not the paper's actual contribution. A1-CBF sidesteps this entirely: moving-target pursuit under
a velocity-controlled, acceleration-limited dynamics model is a solved problem in guidance/control (proportional
navigation and its relatives), and implementing it directly removes an entire axis of experimentation that was not
advancing the defender research.

## Relative-target formulation (frozen)
`v_nom = min(12, 1.0 * ||r||) * r/||r||`, `r = p_asset - p_attacker` computed at the CURRENT state, once per
external 0.5 s step. No target-velocity feedforward (the asset's motion has no persistent velocity to predict —
see `docs/journal_v2_a1_curriculum/stage4/README.md`), no learned prediction, no minimum forward speed (hover is
always reachable). This is deliberately the simplest guidance law that solves the benchmark; it is NOT proportional
navigation or lead pursuit (both deferred, and arguably unnecessary against a target with no persistent velocity).

## Velocity-command multirotor abstraction
The key design decision, beyond the guidance law itself, is dropping the legacy fixed-wing-style heading/turn-rate
limit for this attacker and replacing it with a pure acceleration-limited 3-D velocity update — the physically
correct abstraction for a multirotor, which can decelerate, hover, and reverse without maintaining forward
airspeed. The legacy model's inability to do this (it must maintain a minimum turning radius at speed) is a
plausible *additional* contributor to the PPO/BC curriculum's fly-by problem, beyond the representation/covariate-
shift issues already diagnosed — though this was not isolated as a separate ablation here (both the guidance law
and the dynamics model changed together); a future ablation could test the hand-designed guidance law under the
NEW quadrotor dynamics, or vice versa, to attribute the improvement more precisely.

## Physical limits
Kept identical to the canonical hostile-UAV config (12 m/s horizontal, 6 m/s² acceleration, 5 m/s climb, 5 m/s
descent) so the comparison against A0 isolates the GUIDANCE LAW + DYNAMICS MODEL change, not a change in the
vehicle's physical envelope.

## Low-level substepping
5 internal 0.1 s substeps per external 0.5 s step, command held constant across them (computed once per external
step, matching how the legacy model's guidance is also computed once per step). The existing swept mission-contact
test is completely unchanged (one linear segment over the full external step) — verified both by direct code
inspection (no new contact-test code was added) and empirically (all captures terminate within the unchanged 2.0 m
radius).

## Results to quote
1,000 matched scenarios: A1-CBF captures 100.0 % (A0 99.3 %) with 0 repeated fly-bys (A0 9) and 0 orbit episodes
(both). Median miss distance among episodes with at least one near-miss approach dropped from 4.37 m (A0) to 2.59 m
(A1-CBF) — A1-CBF's misses are much closer to actually capturing. All 7 pre-registered acceptance-gate checks pass.
Dynamics constraint verification: max observed horizontal speed 12.000 m/s (limit 12.0), max observed acceleration
6.000 m/s² (limit 6.0) — both within floating-point tolerance of their configured limits, confirming no violation.

## Comparison with A0
A1-CBF is not required to, and does not need to, outperform A0 by a wide margin (A0 itself has a near-optimal
scripted guidance law on this exact benchmark) — the result that matters is that A1-CBF reaches effectively
A0-level performance **without any of the fly-by pathology that the learned attackers exhibited**, confirming the
hypothesis that the failure was architectural (guidance + dynamics model), not a training/data problem.

## Honest caveats
1. C0 only, no active defender interaction — says nothing about obstacle/defender CBF behaviour (not implemented).
2. `kp` and the physical limits were frozen before the acceptance run and not tuned afterward, as required; a
   different `kp` might do even better, but this was deliberately not explored.
3. The guidance-law change and the dynamics-model change were not ablated separately; the ~9-fly-by to 0 improvement
   is attributed to "the architecture" as a whole, not isolated to one component.
4. A1-CBF's hover/near-zero-speed events are rare (1 total over 1,000 episodes) simply because this benchmark
   virtually always ends in capture well before the attacker would need to hover for long; this is not a dedicated
   hover-robustness test.
5. Legacy no-op verification compares against one specific "greedy"-track config; other controller tracks were not
   separately re-verified (though they share the same `_move_enemy()` code path and are covered by the broader
   regression suite).

## What comes next (not started here)
Obstacle CBF (replacing the identity safety filter with a real QP over obstacle hazards) and, later, defender CBF.
No canonical tag is created for A1-CBF at this stage; it remains a development/foundation artifact pending those
extensions.
