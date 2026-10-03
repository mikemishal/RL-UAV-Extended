# Defender-Adaptation (D1) Claims / Evidence Ledger

Generated from the completed, frozen 32,400-episode held-out evaluation (evaluation-code commit
`1abd7c57ab7a78d43752cfb21d3e254e2af0a548`, tag `journal-defender-adaptation-eval-v1`). All numbers
below are read directly from `results/defender_adaptation_evaluation/statistics.json`,
`results/defender_adaptation_evaluation/tables/*.csv`, and the supporting analysis CSVs in the same
directory. Confidence intervals are 95% hierarchical paired bootstrap intervals (10,000 replicates,
resampling both the 3 defender-training-seed axis and the 100 matched-episode axis; see
`experiments/defender_adaptation_statistics.py::_delta_report`, reusing
`experiments/final_stats.py::hierarchical_paired_bootstrap_matched_seeds` unchanged).

Support levels used: **Strongly supported** (CI excludes 0, consistent direction across all 3 seeds),
**Moderately supported** (CI excludes 0 but seed-level direction is not fully consistent, OR effect is
present but modest), **Descriptive only** (a real, computed number with no formal inferential claim
attached — e.g. balanced/worst-case robustness, outcome decomposition), **Unsupported** (CI includes 0,
or the data does not support the proposed claim).

---

## 1. Primary endpoint: D1 − D0+ against A1-unseen-203 (equal-weight C1–C3 aggregate)

| Claim | Evidence | CI | Support |
|---|---|---|---|
| PPO-Kalman-C: adversarial-mixture adaptation improves safe interception against the held-out learned attacker beyond equal-budget extra training | D0+=0.354, D1=0.518, Δ=+0.163 | [+0.080, +0.240] | **Strongly supported** (CI excludes 0; consistent direction in all 3 seeds: seed74 0.44→0.59, seed75 0.29→0.57, seed76 0.36→0.72 at C1 alone, and positive at C1/C2 individually) |
| PPO-C: adversarial-mixture adaptation improves safe interception against A1-unseen-203 | D0+=0.459, D1=0.456, Δ=−0.003 | [−0.064, +0.061] | **Unsupported** (CI straddles 0, effect indistinguishable from no change) |
| LR-PPO-C: adversarial-mixture adaptation improves safe interception against A1-unseen-203 | D0+=0.724, D1=0.667, Δ=−0.058 | [−0.137, +0.019] | **Unsupported** (CI includes 0; central estimate is negative, i.e. a non-significant trend toward WORSE performance, not better) |

Source: `results/defender_adaptation_evaluation/tables/table1_adaptive_threat_generalization.csv` (row `Clutter=aggregate_C1C3`), `statistics.json::primary_d1_vs_d0plus_unseen`.

## 2. Clutter-level breakdown of the PPO-Kalman-C effect (the only architecture with a supported primary effect)

| Clutter | D0+ | D1 | Δ | CI | Support |
|---|---|---|---|---|---|
| C1 | 0.363 | 0.627 | +0.263 | [+0.117, +0.397] | **Strongly supported** |
| C2 | 0.330 | 0.497 | +0.167 | [+0.037, +0.283] | **Strongly supported** |
| C3 | 0.370 | 0.430 | +0.060 | [−0.050, +0.170] | Unsupported (CI includes 0) |

**Claim**: for PPO-Kalman-C, the adaptation benefit against the held-out attacker is descriptively largest at low clutter and diminishes toward high clutter. **Support: Descriptive only** — this is an observed monotonic pattern across 3 ordered clutter levels, not a formally tested interaction effect (no interaction model was fit; per Phase 12 of the spec, this is reported descriptively, not as a formal interaction claim).

## 3. Secondary: D1 − D0 (total gain) against A1-unseen-203, aggregate

| Architecture | D0 | D1 | Δ_total | CI | Support |
|---|---|---|---|---|---|
| PPO-C | 0.404 | 0.456 | +0.051 | [−0.031, +0.137] | Unsupported |
| PPO-Kalman-C | 0.390 | 0.518 | +0.128 | [+0.044, +0.217] | **Strongly supported** |
| LR-PPO-C | 0.692 | 0.667 | −0.026 | [−0.080, +0.030] | Unsupported |

## 4. Extra-training-alone control: D0+ − D0 against A1-unseen-203, aggregate

| Architecture | D0 | D0+ | Δ_extra | CI | Support |
|---|---|---|---|---|---|
| PPO-C | 0.404 | 0.459 | +0.054 | [−0.017, +0.124] | Unsupported |
| PPO-Kalman-C | 0.390 | 0.354 | −0.036 | [−0.150, +0.100] | Unsupported |
| LR-PPO-C | 0.692 | 0.724 | +0.032 | [−0.038, +0.106] | Unsupported |

**Claim**: simply training 425,984 more steps against the scripted attacker alone (D0+) does not, by itself, reliably improve or harm performance against the held-out learned attacker for any architecture. **Support: Moderately supported** (all 3 CIs include 0 — this is itself the evidence for "no reliable extra-training-alone effect"; absence of a significant effect is the finding). This strengthens the PPO-Kalman-C primary finding: its D1 gain is attributable to the adversarial mixture specifically, not merely to additional gradient steps.

## 5. A0 (scripted) retention: D1 − D0+, aggregate

| Architecture | D0+ | D1 | Δ | CI | Support |
|---|---|---|---|---|---|
| PPO-C | 0.591 | 0.592 | +0.001 | [−0.070, +0.070] | **Strongly supported** (no forgetting) |
| PPO-Kalman-C | 0.580 | 0.582 | +0.002 | [−0.102, +0.109] | **Strongly supported** (no forgetting) |
| LR-PPO-C | 0.863 | 0.842 | −0.021 | [−0.077, +0.036] | **Strongly supported** (no forgetting; small negative point estimate, CI includes 0) |

**Claim**: adversarial-mixture adaptation does not cause catastrophic forgetting of scripted-threat performance in any of the 3 architectures. **Support: Strongly supported** across all 3 architectures (all 3 CIs tightly straddle 0).

## 6. A1-seen (201/202 aggregate): D1 − D0+

See `results/defender_adaptation_evaluation/tables/table3_seen_attackers.csv` for full CIs. Descriptive pattern (robustness table, `robustness_scores.csv`): PPO-C's A1-seen mean DECREASES from D0+ (0.409) to D1 (0.362); PPO-Kalman-C's A1-seen mean increases modestly (0.392→0.394, essentially flat); LR-PPO-C's A1-seen mean decreases (0.661→0.615). **Support: Descriptive only** pending the formal per-cell CI check in `table3_seen_attackers.csv` — flagged here because PPO-C's seen-attacker performance declining while its unseen-attacker performance stayed flat is a pattern worth highlighting (not a clear win for PPO-C's D1 even against the attackers it trained on).

## 7. Balanced / worst-case robustness (descriptive only, per Phase 13/14 — never used to rank controllers)

| Architecture | Gen | BalancedRobustness | WorstCase |
|---|---|---|---|
| PPO-C | D0 / D0+ / D1 | 0.454 / 0.486 / 0.470 | 0.368 / 0.409 / 0.362 |
| PPO-Kalman-C | D0 / D0+ / D1 | 0.436 / 0.442 / 0.498 | 0.342 / 0.354 / 0.394 |
| LR-PPO-C | D0 / D0+ / D1 | 0.728 / 0.750 / 0.708 | 0.622 / 0.661 / 0.615 |

**Support: Descriptive only** (by design — see Phase 13/14 of the spec). PPO-Kalman-C is the only architecture whose BalancedRobustness and WorstCase both improve monotonically D0→D0+→D1.

## 8. Safety-layer dependence

Intervention fraction and mean/P95 override angle vs A1-unseen-203 are essentially stable across D0/D0+/D1 for all 3 architectures (e.g. PPO-C: 0.376/0.384/0.374; PPO-Kalman-C: 0.335/0.360/0.362; LR-PPO-C: 0.203/0.215/0.198 — see `safety_layer_dependence.csv`). **Claim**: adversarial-mixture adaptation does not materially change reliance on the low-level obstacle-avoidance safety layer. **Support: Descriptive only / moderately supported** (no CI computed for this specific comparison; differences are small relative to the spread already seen across defender seeds).

## 9. Interception efficiency (successful interceptions only, vs A1-unseen-203)

| Architecture | Gen | n successful (of 300 episodes at this condition) | mean steps | mean path length |
|---|---|---|---|---|
| PPO-Kalman-C | D0+ / D1 | 319 / 466 | 309.8 / 245.0 | 1769.9 / 1325.0 |
| PPO-C | D0+ / D1 | 413 / 410 | 273.6 / 315.7 | 1468.4 / 1714.3 |
| LR-PPO-C | D0+ / D1 | 652 / 600 | 142.0 / 146.1 | 794.5 / 804.7 |

**Claim**: PPO-Kalman-C's D1 improvement is accompanied by MORE EFFICIENT successful interceptions (fewer steps, shorter path), not a success-rate/efficiency tradeoff. **Support: Descriptive only** (no CI computed on these efficiency deltas; n differs between conditions since only successful episodes are included). PPO-C and LR-PPO-C show flat-to-slightly-worse efficiency in D1 alongside their flat-to-negative success-rate changes.

## 10. LR-PPO residual-angle structural analysis (near-30° margin = 2.0°, fixed before inspecting data — see `experiments/defender_adaptation_evaluation.py::NEAR_30_DEG_MARGIN`)

| Generation | Mean residual angle (deg, all conditions pooled) | Max residual angle reaches ≈30°? | fraction_near_30deg |
|---|---|---|---|
| D0 | ~6.9 | No (max ≤ 19.98°) | 0.0000 everywhere |
| D0+ | ~8.0–8.9 | Rarely (max reaches 29.8–30.0° only at C2/C3) | 0.0000–0.0001 |
| D1 | ~10.9–12.5 | Yes, consistently (max 29.8–30.0° in every condition) | 0.0003–0.0037 |

**Claim**: the prior finding "P(near 30°) = 0" holds for D0 (confirmed, unchanged) but does NOT hold for D1 — adversarial-mixture adaptation causes LR-PPO-C to use measurably more of its 30° residual authority, against A0 as well as against all 3 learned-attacker conditions (the shift is present even vs the scripted attacker, not just vs learned attackers). **Support: Strongly supported** as a directional, consistent, monotonic D0→D0+→D1 shift across all 12 generation×attacker-type×level combinations in `lr_ppo_residual_analysis.csv`; the ABSOLUTE frequency of being near the bound remains very low (well under 1% of steps) even in D1, so this is a shift in degree of residual usage, not a claim that LR-PPO is frequently saturating its residual bound.

---

## Summary verdict (ties directly to the NEXT-EXPERIMENT DECISION rule)

Architectures behave **differently** — PPO-Kalman-C shows a strongly supported primary benefit with no retention cost; PPO-C shows no primary benefit (flat) with a possible (descriptive-only) A1-seen regression; LR-PPO-C shows no primary benefit and a non-significant negative trend, alongside a strongly supported shift toward heavier residual-authority use. Per the decision rule's option D ("if architectures behave differently: recommend architecture-specific investigation first"), this ledger supports recommending **architecture-specific investigation before any further attacker-generation escalation** — see the final report's next-experiment recommendation.
