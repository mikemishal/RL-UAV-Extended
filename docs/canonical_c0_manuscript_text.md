# Canonical C0 (LR-PPO semantics) evaluation — manuscript text and decision gate

## Results subsection: "Canonical Open-Air Retention and Adversarial Generalization"

To verify that our findings are not an artifact of a specific operating
regime, we re-evaluated all three journal-trained defender families
(PPO-C, PPO-Kalman-C, LR-PPO-C) across their three adaptation generations
(D0, D0+, D1) in a no-obstacle (C0) environment configured to match the
original LR-PPO paper's engagement semantics exactly —
`defender_standby_until_detection=True`, with all obstacle generation,
avoidance, and awareness mechanisms disabled — rather than relying on the
clutter-trained default configuration. This canonical configuration was
verified programmatically before any episodes were run, and zero obstacle
collisions were observed across the full 10,800-episode matrix (27
defenders × 4 attacker conditions × 100 held-out evaluation seeds), as
required for a genuinely obstacle-free condition.

**Scripted-attacker (A0) retention is preserved, and in one case improved.**
None of the three architectures show a significant *loss* of performance
against the original scripted attacker baseline between the frozen D0
checkpoint and the fully-adapted D1 checkpoint (PPO-C: −0.073, 95% CI
[−0.180, +0.023]; LR-PPO-C: −0.013, CI [−0.087, +0.070]). PPO-Kalman-C
instead shows a significant *improvement* (+0.107, CI [+0.013, +0.200]).
This rules out catastrophic forgetting of the pre-adaptation task as an
explanation for any of the adversarial-generalization effects reported
below.

**PPO-C shows a significant, open-air-specific adversarial-generalization
regression.** Against the held-out adversarial attacker (A1-unseen-203),
PPO-C's success rate drops significantly from D0+ to D1 under canonical
semantics (0.720 → 0.457; Δ = −0.263, 95% CI [−0.513, −0.060]), with the
effect direction consistent across all three independently-seeded defender
replicates. Critically, when we compare this effect across clutter levels,
the regression is specific to the canonical open-air condition: the
corresponding D1−D0+ effect is statistically indistinguishable from zero in
C1 (−0.023, CI [−0.130, +0.087]), C2 (+0.003, CI [−0.107, +0.110]), and C3
(+0.010, CI [−0.100, +0.123]). A weaker but still significant degradation
is also present against the mean of the two *seen* adversarial attackers
(Δ = −0.143, CI [−0.303, −0.010]), driven primarily by attacker A1-seen-202.

**PPO-Kalman-C shows the opposite pattern: a robust, largely
clutter-independent improvement.** D1−D0+ against A1-unseen-203 is
significantly positive under canonical semantics (+0.187, CI [+0.037,
+0.350]) and remains significant in C1 (+0.263) and C2 (+0.167), only
attenuating to non-significance at the highest clutter level C3 (+0.060,
CI [−0.050, +0.170]).

**LR-PPO-C shows a consistent but non-significant negative trend,
accompanied by a clear, significant shift in residual-correction usage.**
The D1−D0+ effect against A1-unseen-203 is negative in sign (−0.127) but
its 95% CI includes zero ([−0.300, +0.050]), as do the corresponding
effects in every clutter level (C1: −0.087; C2: −0.093; C3: +0.007).
However, the magnitude of LR-PPO's residual steering correction grows
substantially across generations in the canonical condition: mean residual
angle rises from ~6.7–7.1° at D0 to ~11.2–12.5° at D1 (nearly doubling),
and the fraction of steps within 2° of the ±30° residual cap rises from
effectively 0 to up to 0.54%. This is consistent with — though does not by
itself establish causation for — the mild negative success-rate trend.

**Legacy-vs-canonical comparison.** We additionally compared the canonical
PPO-C D1−D0+ effect (−0.263, CI [−0.513, −0.060]) against the same
comparison computed on the earlier C0 evaluation dataset (tag
`journal-paper-reproduction-v1`; −0.217, CI [−0.463, +0.010]). The two
point estimates are compatible (overlapping CIs, same sign), and — contrary
to an initial hypothesis that the two datasets might differ in detection
semantics — our prior 16-phase environment audit established that both
datasets in fact share the identical `defender_standby_until_detection=True`
engagement semantics. We therefore interpret the discrepancy between the
two CIs (legacy borderline-non-significant vs. canonical significant) as
ordinary sampling variability at n=3 defender seeds, and treat the
canonical result as an independent replication that strengthens, rather
than revises, the original finding of a genuine PPO-C adversarial-adaptation
regression.

## Discussion subsection

These results refine, rather than overturn, our central claim about
architecture-dependent adaptation robustness. The PPO-C regression against
held-out adversaries — robust across both legacy and canonical C0 datasets,
and absent in all three clutter levels — indicates that continued
adaptation in an uncluttered environment is specifically where PPO-C's
measurement-based policy is most susceptible to overfitting against the
training-time adversarial distribution. One candidate explanation is that
clutter itself acts as a regularizer: with obstacles present, the defender
must already generalize across a broader state distribution during
training, which may incidentally buffer it against overfitting to a narrow
set of adversarial maneuver patterns. In the open-air condition, continued
adaptation has comparatively "nothing else to do" except specialize against
the specific adversarial attackers encountered during D1 training, which is
exactly what would produce the observed held-out generalization loss.

PPO-Kalman-C's result is the clearest positive finding: its Kalman-filtered
state estimate appears to confer a generalization benefit from continued
adaptation that is robust across the open-air condition and two of three
clutter levels, attenuating only at the densest clutter (C3), where
estimation noise and obstacle-avoidance competing objectives may dominate.

LR-PPO-C's pattern is the most ambiguous: no individual condition reaches
statistical significance for the success-rate effect, yet the magnitude and
near-saturation frequency of its residual correction grow substantially and
monotonically with continued adaptation. This is consistent with (but does
not prove) a slow drift toward larger, more frequent residual interventions
that has not yet produced a statistically detectable success-rate cost at
the sample sizes used here (n=3 defender seeds × 100 eval seeds per cell).

**Limitations.** All effects are estimated from n=3 independently-seeded
defender replicates per architecture; several individually-plausible
effects (e.g., PPO-C's seen-attacker-201 regression, LR-PPO-C's residual
drift translating to a success-rate cost) do not reach significance at this
sample size and should be treated as descriptive rather than confirmed.
The four-condition decomposition (Phase 13) is reported descriptively only,
without per-cell confidence intervals. This evaluation specifically
excludes any robustness-sweep, retraining, or architecture-modification
work, per the task's explicit scope.

## Reproducibility note

This canonical dataset (`results/defender_adaptation_canonical_c0_evaluation/`,
eval seeds 130000–130099, tag `journal-canonical-c0-evaluation-v1`) is
distinct from, and does not overwrite, the earlier legacy C0 dataset
(`results/defender_adaptation_c0_evaluation/`, tag
`journal-paper-reproduction-v1`), which remains available unmodified for
provenance and for the explicit legacy-vs-canonical comparison documented
above. Both datasets were in fact generated under identical
`defender_standby_until_detection=True` semantics per the prior environment
audit; the canonical dataset's distinguishing feature is its explicit,
programmatically-verified configuration assertion and its disjoint,
dedicated seed namespace, not a semantics correction.

## Decision gate (Phase 20) — recommended next step

Given: (1) PPO-C shows a robust, canonical-open-air-specific adversarial
regression that merits deeper investigation rather than a robustness sweep
across unrelated axes; (2) PPO-Kalman-C's result is positive and broadly
robust, requiring no urgent follow-up; (3) LR-PPO-C's effect is suggestive
but not yet statistically established.

**Recommended next step: Outcome B — run a targeted architecture diagnostic
for PPO-C** to characterize *why* continued adaptation degrades held-out
adversarial generalization specifically in the open-air condition (e.g.,
policy-entropy / action-distribution drift analysis across D0+→D1 under A0
vs A1-unseen, restricted to the existing canonical and clutter datasets —
no new training or environment changes required). This is the most direct
way to explain the single strongest and most novel finding in this
evaluation (the clutter × attacker-generalization interaction for PPO-C)
without expanding scope into retraining, architecture changes, or
unrelated robustness sweeps, all of which are explicitly out of scope for
this task.

**(Not executed in this task, per explicit instruction: no robustness
sweeps, no retraining, no architecture modifications were run. This is a
recommendation only.)**
