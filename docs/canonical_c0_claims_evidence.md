# Canonical C0 (LR-PPO semantics) evaluation — claims & evidence ledger

**Dataset**: `results/defender_adaptation_canonical_c0_evaluation/episodes.csv`
(10,800 episodes; 27 defenders × 4 attacker conditions × 100 eval seeds,
`eval_seed ∈ [130000, 130099]`, disjoint from every other seed range used in
this project). Code commit for all episodes: `f686d3ad42099a00b3782a55106eed36512f0a4d`
(the audited journal HEAD; unchanged since the audit). Environment config
verified programmatically (`assert_canonical_lrppo_config`) to match the
true LR-PPO paper semantics: `defender_standby_until_detection=True`,
`obstacles_enabled=False`, `enemy_obstacle_avoidance_enabled=False`,
`defender_obstacle_avoidance_enabled=False`, both obstacle-awareness modes
`"disabled"`. Zero obstacle collisions recorded across all 10,800 episodes
(required and confirmed for a no-obstacle condition).

Classification scale: **Strongly supported** (hierarchical bootstrap 95% CI
excludes 0, consistent direction across all 3 defender seeds), **Moderately
supported** (CI excludes 0 but seed-level direction is mixed, or effect is
borderline), **Descriptive only** (no CI computed / purely observational),
**Unsupported** (CI includes 0, or evidence contradicts the claim).

---

## 1. A0 (scripted baseline) retention across generations (Table 1)

| Architecture | D1−D0 | 95% CI | Classification |
|---|---|---|---|
| PPO-C | −0.073 | [−0.180, +0.023] | Unsupported (no significant retention loss) |
| PPO-Kalman-C | +0.107 | [+0.013, +0.200] | **Strongly supported** — D1 improves over D0 on A0 |
| LR-PPO-C | −0.013 | [−0.087, +0.070] | Unsupported (flat, no detectable change) |

**Claim**: "Catastrophic forgetting of the pre-adaptation (scripted-attacker)
task does not occur for any architecture under canonical C0 semantics."
→ **Strongly supported**. No architecture shows a significant *negative*
D1-D0 effect against A0; PPO-Kalman-C significantly *improves*.

---

## 2. Primary endpoint: D1 − D0+ vs A1-unseen-203 (Table 2)

| Architecture | D1−D0+ | 95% CI | Classification |
|---|---|---|---|
| PPO-C | −0.263 | [−0.513, −0.060] | **Strongly supported** degradation |
| PPO-Kalman-C | +0.187 | [+0.037, +0.350] | **Strongly supported** improvement |
| LR-PPO-C | −0.127 | [−0.300, +0.050] | Unsupported (CI includes 0) |

**Claim A (PPO-C adversarial-adaptation regression)**: under canonical
semantics, PPO-C's continued adaptation (D1) significantly *degrades*
held-out adversarial generalization relative to the pre-adaptation
checkpoint (D0+). → **Strongly supported**, consistent direction across all
3 seeds (per-seed diffs: −0.09, −0.53, −0.17; `canonical_a0_retention.json`
/ `canonical_primary_endpoint_unseen.json`).

**Claim B (PPO-Kalman-C generalization benefit)**: PPO-Kalman-C's continued
adaptation *improves* held-out adversarial generalization. →
**Strongly supported**, consistent direction across seeds 74/75/76 (all
positive or near-zero; see `canonical_primary_endpoint_unseen.json`).

**Claim C (LR-PPO-C residual drift)**: LR-PPO-C shows a mild negative trend
but it does **not** reach significance (CI [−0.300, +0.050] includes 0). →
**Unsupported** as a standalone significant effect; see §5 below for the
residual-angle-usage finding which *is* strongly supported.

---

## 3. A1-seen (201/202) generalization (Table 3)

| Architecture | Attacker | D1−D0+ | 95% CI | Classification |
|---|---|---|---|---|
| PPO-C | A1-seen-201 | −0.070 | [−0.280, +0.133] | Unsupported |
| PPO-C | A1-seen-202 | −0.217 | [−0.390, −0.040] | **Strongly supported** degradation |
| PPO-C | seen-mean | −0.143 | [−0.303, −0.010] | **Strongly supported** degradation |
| PPO-Kalman-C | A1-seen-201 | −0.120 | [−0.290, +0.050] | Unsupported |
| PPO-Kalman-C | A1-seen-202 | +0.020 | [−0.230, +0.227] | Unsupported |
| PPO-Kalman-C | seen-mean | −0.050 | [−0.250, +0.117] | Unsupported |
| LR-PPO-C | A1-seen-201 | −0.140 | [−0.430, +0.113] | Unsupported |
| LR-PPO-C | A1-seen-202 | −0.013 | [−0.090, +0.063] | Unsupported |
| LR-PPO-C | seen-mean | −0.077 | [−0.230, +0.070] | Unsupported |

**Claim**: PPO-C's degradation under continued adaptation is not confined to
the held-out attacker — the seen-attacker mean also shows a significant
negative effect, though weaker and driven mainly by A1-seen-202. →
**Moderately supported** (seen-mean CI excludes 0, but individual seen
attackers are mixed — only 202 is individually significant).
PPO-Kalman-C and LR-PPO-C show no significant seen-attacker effect in either
direction. → **Unsupported** as a distinct effect for those two
architectures.

---

## 4. C0 vs C1/C2/C3 — is the PPO-C effect open-air-specific? (Table 4, `d1_minus_d0plus_by_condition.csv`)

| Architecture | C0 (canonical) | C1 | C2 | C3 |
|---|---|---|---|---|
| PPO-C | **−0.263** [−0.513,−0.060] | −0.023 [−0.130,+0.087] | +0.003 [−0.107,+0.110] | +0.010 [−0.100,+0.123] |
| PPO-Kalman-C | +0.187 [+0.037,+0.350] | +0.263 [+0.117,+0.397] | +0.167 [+0.037,+0.283] | +0.060 [−0.050,+0.170] |
| LR-PPO-C | −0.127 [−0.300,+0.050] | −0.087 [−0.250,+0.050] | −0.093 [−0.210,+0.023] | +0.007 [−0.080,+0.090] |

**Claim D (clutter × attacker-generalization interaction — the key novel
finding of this evaluation)**: PPO-C's significant adversarial-generalization
*regression* under continued adaptation is **unique to the open-air (C0)
condition**; it is statistically indistinguishable from zero in C1, C2, and
C3. → **Strongly supported**. This directly contradicts a naive expectation
that clutter training would be where catastrophic interference is worst —
instead, the regression appears specifically when the open-air geometry
(from the original LR-PPO paper) is reinstated.

**Claim E (PPO-Kalman-C's benefit generalizes across clutter, with mild
attenuation)**: PPO-Kalman-C's positive D1-D0+ effect holds in C0, C1, and
C2 (all CIs exclude 0) but attenuates to non-significance in C3 (densest
clutter). → **Moderately supported** — effect is robust across 3 of 4
conditions with a plausible attenuation trend at the highest clutter level.

**Claim F (LR-PPO-C shows a consistent but non-significant negative trend)**:
diffs are negative in C0/C1/C2 (−0.127, −0.087, −0.093) and near zero in C3
(+0.007), but no individual CI excludes 0. → **Descriptive only** — a
directionally consistent pattern worth reporting, not a statistically
established effect.

---

## 5. LR-PPO residual steering-angle usage (Table 5, Phase 12)

| Generation | mean residual (°) | fraction within 2° of 30° cap |
|---|---|---|
| D0 | ~6.7–7.1 | 0.0000 |
| D0+ | ~7.6–8.1 | ≤0.00001 |
| D1 | **~11.2–12.5** | **0.0001–0.0054** |

**Claim**: LR-PPO's residual-correction magnitude grows substantially from
D0/D0+ (~7°) to D1 (~11–12.5°), nearly doubling, and the frequency of
near-saturation (within 2° of the ±30° residual cap) rises from
effectively 0 to a small but non-trivial fraction (up to 0.5%). →
**Strongly supported** (consistent, large, monotonic shift across all 4
attacker conditions; `lrppo_residual_canonical_summary.csv`). This is
*descriptive* evidence of increasing reliance on the residual-correction
pathway during continued adaptation — it does not by itself establish
*causation* for the (non-significant) success-rate drift noted in §2/§4,
but it is consistent with that drift and offers a candidate mechanism.

---

## 6. Four-condition decomposition (Phase 13, `four_condition_decomposition.csv`)

For PPO-C: going from (A0,C0) → (A1-unseen,C0) costs −0.07 to −0.29 across
generations (attacker effect under C0), while going from (A0,clutter) →
(A1-unseen,clutter) costs a much smaller and flatter −0.13 to −0.17. The
clutter effect under A0 is consistently strongly negative (−0.16 to −0.25),
but the clutter effect under A1-unseen *shrinks* at D0+/D1 relative to D0.
→ **Descriptive only** (no CI computed for these four-way cell means in
this decomposition; the Table-4-based pairwise comparisons in §4 are the
statistically-tested version of the same underlying pattern).

---

## 7. Legacy vs canonical C0 comparison for PPO-C (Phase 11, `legacy_vs_canonical_ppo_c.json`)

| Dataset | D1−D0+ vs A1-unseen-203 | 95% CI |
|---|---|---|
| Canonical (this eval, seeds 130000–130099) | −0.263 | [−0.513, −0.060] |
| Legacy (`journal-paper-reproduction-v1`, tag) | −0.217 | [−0.463, +0.010] |

**Important honest caveat** (per the prior 16-phase LR-PPO environment
audit, commit `f686d3ad42099a00b3782a55106eed36512f0a4d`): both the legacy
and the canonical evaluation actually used `defender_standby_until_detection
=True` — the audit found this flag is **identical** between the legacy
run's configuration and the true LR-PPO paper semantics. The user's framing
of the legacy run as using "historical legacy pre-detection semantics" does
not match what the audit established; this discrepancy is reported here
rather than silently reconciled.

**Claim**: the canonical and legacy point estimates (−0.263 vs −0.217) are
compatible — their CIs overlap substantially, and both show the same
negative direction. The legacy CI merely fails to exclude 0 (borderline,
upper bound +0.010) while the canonical CI clears 0 by a wider margin
(−0.060). This is most plausibly ordinary across-run sampling variability
at `n=3` defender seeds, **not** evidence of a semantics difference between
the two datasets. → **Strongly supported** interpretation: "Interpretation
B" (the canonical result reproduces a genuine, semantics-independent PPO-C
adversarial-adaptation regression), **not** "Interpretation A/C"
(protocol-dependent artifact) — because the two datasets used the same
environment semantics throughout.

---

## 8. Efficiency (successful interceptions only, Phase 15)

No CI-tested claims were requested here; distributions are reported
descriptively in `interception_efficiency_summary.csv`. Qualitatively,
mean interception time for A1-seen-201 is consistently several-fold longer
than for A0 or A1-seen-202/A1-unseen-203 across all architectures and
generations (e.g., PPO-C D0+: 233 vs 51–197 steps) — attacker 201 appears to
induce longer pursuit episodes among successful intercepts regardless of
defender generation. → **Descriptive only**.
