# Architecture-mechanism diagnostic report

**Task**: explain the architecture-dependent adaptive-attacker results from
the canonical-C0 and clutter (C1-C3) evaluations. This is a MECHANISM
DIAGNOSTIC: no controller was retrained, no architecture or environment
physics were changed, no checkpoints were altered. All findings below come
from (a) re-reading already-frozen episode-level evaluation datasets, and
(b) a modest set of NEW, additive, step-level-instrumented diagnostic
rollouts (1,200 Phase 3/6/7 episodes + 2,400 Phase 4 authority-sweep
episodes = 3,600 total new episodes) that reuse the FROZEN D0/D0+/D1
checkpoints and matched-seed subsets of the already-published canonical-C0
and clutter evaluation seed ranges. See
[docs/architecture_mechanism_claims_ledger.md](architecture_mechanism_claims_ledger.md)
for the full evidence-classified ledger; this document focuses on the
decisions and manuscript-ready text.

## Data reused (no new rollouts)

- `results/defender_adaptation_canonical_c0_evaluation/` (10,800 episodes,
  canonical C0, `defender_standby_until_detection=True`) — success rates,
  CIs, and the `d1_minus_d0plus_by_condition.csv` / `c0_vs_clutter_table.csv`
  summary tables that anchor every quantitative claim about PPO-C's C0
  regression and PPO-Kalman's/LR-PPO's C0-C3 patterns.
- `results/defender_adaptation_evaluation/` (32,400 episodes, C1-C3) — the
  clutter-level comparison data and `lr_ppo_residual_analysis.csv`.
- `results/clutter_retraining_evaluation/recovery_deltas.csv` — LR-PPO's
  +7.7pp C3 obstacle-aware-retraining recovery (already-established,
  unchanged).
- `uav_defend/policies/residual/lead_residual_composition.py` and
  `uav_defend/policies/baseline/lead_intercept_policy.py` — the EXISTING,
  UNCHANGED Lead+residual math and Lead-guidance diagnostics
  (`last_intercept_point`, `last_intercept_time`, `last_target_position`,
  etc.), reused verbatim to instrument (not modify) policy behavior.

## New diagnostic episodes

| Script | Scope | N episodes |
|---|---|---|
| `experiments/mechanism_diagnostic.py` (Phase 3) | LR-PPO, D0/D0+/D1, A0+A1-unseen-203, C0+C3, N=20 seeds | 720 |
| `experiments/mechanism_diagnostic.py` (Phase 6) | PPO-Kalman-C, D0+/D1, A1-unseen-203, C0+C3, N=20 | 240 |
| `experiments/mechanism_diagnostic.py` (Phase 7) | PPO-C, D0+/D1, A1-unseen-203, C0+C3, N=20 | 240 |
| `experiments/mechanism_diagnostic_authority.py` (Phase 4) | LR-PPO D1, 15/30/45/60°, A1-unseen-203, C0+C3, N=100 | 2,400 |
| **Total** | | **3,600** |

All seeds used are exact subsets of the already-published, disjoint
`eval_seed` ranges (canonical C0: 130000-130099; clutter: 90010000-90010099)
— no new seed namespace was allocated. Step-level instrumentation (defender/
attacker kinematics, Lead-guidance internals, residual angle) was added
without changing any policy's `act()` output.

## Phase 3 — Lead-prediction error: A0 vs A1

`Lead error(A1) > Lead error(A0)` is **refuted**: A1-unseen-203 produces
*lower* mean angular Lead error than A0 in all 6 (level × generation) cells
(5 of 6 statistically significant; see ledger §1). Lead error IS higher for
failure episodes than success episodes in all 12 cells tested (ledger §2,
descriptive).

## Phase 3/5 — Residual-angle relationship

D1's larger residual corrections track increasing Lead error (22.99°→26.37°
mean error across residual bins 0-5°→20-25°) while attacker maneuver
intensity stays flat across bins — consistent with corrections compensating
for degraded Lead estimates rather than reacting to attacker maneuvering
per se (ledger §3, correlational).

## Phase 4 — 15/30/45/60° authority diagnostic

Wider authority does not help: 45°/60° are flat-to-significantly-worse than
the canonical 30° (60° in C3: −18.0pp, CI [−28.3,−7.3]); saturation at the
30° bound is vanishingly rare (0.007-0.09% of steps) (ledger §4).

## Phase 12 — 30° authority decision

### NOT AUTHORITY-LIMITED

Widening the inference-time residual authority from the trained 30° bound to
45° or 60° provides no benefit and is significantly *worse* at 60° in C3;
saturation against the 30° bound occurs in well under 0.1% of steps across
3,600 diagnostic episodes. A narrower 15° bound shows a suggestive (CI lower
bound exactly 0.0, not conclusively significant) improvement in C0 only.
There is no evidence that 30° materially constrains LR-PPO's performance; a
retrained 45°/60° ablation is **not justified** by this diagnostic.

## Phase 13 — LR-PPO mechanism decision

### MIXED (LEAD-ERROR-ASSOCIATED, NOT AUTHORITY-DOMINATED)

The evidence does not cleanly fit a single category as defined: Lead error
is strongly associated with failure and with larger residual corrections
(favoring "LEAD-ERROR DOMINATED"), but the authority diagnostic (Phase 4)
clearly shows correction *authority* is not the limiting factor (ruling out
"RESIDUAL-AUTHORITY DOMINATED" as the primary story), and we did not test a
temporal/observation-history manipulation, so "TEMPORAL/OBSERVATION LIMITED"
cannot be confirmed or ruled out. Given the strong, well-powered Lead-error
findings (§1-3) combined with the Phase-4 result that ALSO points away from
authority being the bottleneck, the single BEST-SUPPORTED primary driver
among the four candidate mechanisms is **Lead-error association** — but we
report this as **MIXED** rather than a clean "LEAD-ERROR DOMINATED" call,
because no causal manipulation of Lead error itself was performed (only
observational/correlational evidence), per the task's explicit instruction
to "not imply causality from correlation."

## Phase 14 — PPO-Kalman mechanism decision

D1 is distinguished from D0+ by a **significant reduction in action
reversals** (smoother pursuit, both C0 and C3) and **significantly improved
closing-velocity progress** (C0). The Kalman filter itself is identical
between D0+ and D1 — the measurable learned behavioral change enabled by
adversarial continuation is that **D1 uses the (unchanged) filtered state to
commit to smoother, more decisive closing maneuvers** rather than the more
reversal-heavy, oscillatory pursuit seen in D0+ (visually confirmed in the
matched D0+-failure/D1-success trajectory figure).

## Phase 15 — PPO-C mechanism decision

The strongest *quantitatively* supported effect is a small but significant
increase in mean action magnitude in C0 (+0.034, CI [+0.001,+0.072]); the
directionally-consistent-but-non-significant trends toward reduced
closing-velocity progress and more action reversals, together with a vivid
representative "oscillatory/looping pursuit" failure trajectory, are
**suggestive but not individually statistically defensible** at this
diagnostic sample size (N=20×3 seeds). Per the task's explicit instruction,
we do **not** invent a definitive mechanism: the most honest characterization
is an **empirical architecture×environment interaction** — PPO-C's
continued adaptation measurably degrades held-out generalization
specifically in the open-air (C0) condition, with behavioral evidence
(action-magnitude increase, qualitative oscillatory-pursuit trajectories)
pointing toward, but not conclusively proving, excessive/oscillatory
maneuvering as the proximate cause.

---

## Manuscript-ready subsections

### Architecture-Specific Response to Adaptive Threats

Across three architecturally distinct defenders adapted against a mixture
of scripted and learned adversarial attackers, we observe three qualitatively
different responses to continued adversarial training. PPO-Kalman-C
generalizes successfully to the held-out learned attacker, PPO-C regresses
specifically in the open-air (C0) condition, and LR-PPO-C shows negligible
success-rate change despite a substantial internal control-behavior shift.
Mechanism diagnostics targeting each pattern converge on behaviorally
distinct explanations rather than a single shared cause, indicating that
"adversarial adaptation" interacts with each architecture's representation
and control structure in materially different ways.

### Why LR-PPO Does Not Benefit from Adversarial Continuation

LR-PPO-C's Lead-guidance-anchored structure means its residual-correction
policy can only ever nudge, not replace, the analytical Lead solution. We
find that (1) the learned adaptive attacker is, if anything, *easier* to
lead-predict than the original scripted attacker — refuting the natural
hypothesis that adversarial training would need to compensate for worse
Lead predictions against A1 — yet (2) Lead-prediction error remains a
strong correlate of episode failure for both attacker types, and (3) D1's
residual corrections grow substantially in magnitude (mean ~6.9°→~11.9°)
specifically where Lead error is already elevated. An inference-time sweep
of the residual-correction authority bound (15°/30°/45°/60°, N=300 matched
episodes per condition) shows that widening authority beyond the trained
30° bound provides no benefit and is significantly worse at the extreme
(60°, C3); saturation against the current bound is observed in under 0.1%
of steps. We therefore conclude LR-PPO's lack of adversarial-adaptation
benefit is **not** explained by insufficient residual authority, and is
most plausibly linked to the quality of the underlying Lead estimate under
the adaptive attacker's (still imperfectly predictable) maneuvering, rather
than to any structural ceiling on the residual-correction mechanism itself.

### PPO-Kalman Adaptation Mechanism

Because PPO-Kalman-C's Kalman state estimator is byte-for-byte identical
between D0+ and D1, any behavioral difference must originate in the learned
policy's *use* of that filtered state. Matched-seed instrumented rollouts
show D1 commits to significantly smoother pursuit trajectories (fewer
large-angle action reversals per episode, both in open air and in dense
clutter) and significantly faster closing-distance progress in the open-air
condition. Representative matched trajectories illustrate this concretely:
a D0+ failure exhibits a repetitive, oscillatory looping pattern near the
protected asset, while the matched D1 rollout on the identical seed
intercepts the attacker via a direct, decisive maneuver almost immediately
after detection. We interpret PPO-Kalman's adversarial-adaptation benefit
as a learned refinement in how the filtered velocity/position estimate is
translated into committed pursuit behavior, not as an artifact of the
(unchanged) estimator itself.

### PPO-C Open-Air Regression

PPO-C's continued adversarial adaptation produces a significant, open-air-
specific regression against the held-out learned attacker (canonical C0:
D1−D0+ = −26.3pp, 95% CI [−51.3,−6.0]; absent in C1-C3). Matched-seed
behavioral diagnostics find a small but significant increase in mean action
magnitude in C0 under D1, alongside directionally consistent (but
individually non-significant at this diagnostic sample size) trends toward
reduced closing-velocity progress and more frequent large-angle action
reversals. A representative matched trajectory pair (identical seed, D0+
success vs. D1 failure) shows the D0+ policy completing a direct intercept
while the D1 policy enters a repetitive, non-converging looping pattern
near the protected asset — the same qualitative failure signature observed,
to varying degrees, in PPO-Kalman's D0+ failures and in LR-PPO's
high-Lead-error D1 failures, suggesting a shared open-air pursuit
instability that D1 training resolves for PPO-Kalman but does not resolve
(and may mildly exacerbate) for PPO-C. We report this as a real,
reproducible architecture×environment interaction; the statistical power of
this diagnostic's N=20×3-seed sample is insufficient to establish a single
dominant causal mechanism, and we do not claim one.

---

## Statistical limitations

- All architecture-level comparisons rest on exactly **3 independently
  trained seeds per architecture** — the hierarchical paired bootstrap
  (`experiments/final_stats.py`, unchanged) resamples both the seed axis and
  the episode axis to avoid pseudo-replication, consistent with the rest of
  this journal's statistics.
- The Phase 3/6/7 diagnostic rollouts use **N=20 matched eval seeds per
  cell** (a deliberately modest, targeted sample — not the N=100 used by
  the primary canonical/clutter evaluations) to keep this a bounded
  diagnostic rather than a full campaign; several PPO-C behavioral metrics
  are directionally consistent with the hypothesized mechanism but do not
  reach significance at this sample size, and are reported as such.
- The Phase 4 authority sweep uses **N=100 matched eval seeds per cell**
  (explicit task instruction), giving it comparatively higher power; its
  conclusions (not authority-limited) are correspondingly more firmly
  supported than the Phase 3/6/7 behavioral mechanism findings.
- Residual-magnitude bins at the extremes (0-5°, 20-30°) contain very few
  episodes (n=1-6) and their success-rate associations are explicitly
  flagged as descriptive-only, not statistically defensible.
- Lead-prediction error is computed retrospectively using ground-truth
  attacker trajectories recorded ONLY for this offline diagnostic analysis
  — never fed to any policy's action computation (verified: every
  `act()`/`predict()` call in the diagnostic harness uses the same
  legitimate, ground-truth-independent `info` fields as the original
  evaluation harnesses).

## Unsupported / speculative claims (explicitly flagged, not asserted)

- That the shared "oscillatory looping" trajectory pattern across PPO-C,
  PPO-Kalman (D0+), and LR-PPO (D1) represents a single, common underlying
  dynamical cause — this is a descriptive, cross-cutting *observation*, not
  a tested or statistically established claim.
- That PPO-C's C0 regression is CAUSED by excessive action magnitude or
  oscillatory pursuit — these are the best-supported *correlates* found,
  not a proven causal mechanism (explicitly reported as such in Phase 15).
- That narrowing LR-PPO's residual authority to 15° would reliably improve
  performance — the C0 CI touches exactly 0; this is suggestive, not
  conclusive.

## Next-step recommendation (Phase 17)

### D — More diagnostic evidence needed

Mechanism evidence for PPO-Kalman (Phase 14) and the 30° authority question
(Phase 12) is strong enough to close those two threads without further
work. However, the PPO-C open-air regression (Phase 15) and the LR-PPO
mechanism classification (Phase 13, "MIXED") are not yet resolved to a
single defensible mechanism at publication-grade confidence. Per the task's
explicit constraint ("do not propose broad redesign"), we recommend
**additional diagnostic evidence** — specifically, a larger-N (not
architecture-change) repeat of the Phase 7 behavioral diagnostic for PPO-C
(to push the currently-suggestive closing-velocity/reversal trends to
statistical significance or refute them) and, if resources allow, a direct
causal test of Lead-error's role for LR-PPO (e.g., an inference-time
perturbation of the Lead estimate itself, holding the trained policy fixed)
— rather than retraining any architecture at this stage.

**(Not executed in this task, per explicit instruction: no retraining, no
architecture changes, no full robustness sweep. This is a recommendation
only.)**
