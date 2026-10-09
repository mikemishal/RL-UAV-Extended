# Meeting figures — 2026-10 professor meeting

Consolidated, publication-style summary figures for the current journal
story (clutter adaptation, canonical C0 re-evaluation, and the
PPO-C / PPO-Kalman-C / LR-PPO-C comparison). All figures are **pure
post-processing of already-generated results** — no training or
evaluation was re-run to produce them. See `generate_figures.py` for the
exact code, and each `fig*_source.csv` for the lightweight data backing
each plot.

Architecture colors are consistent across all figures: **PPO-C** = blue,
**PPO-Kalman-C** = green, **LR-PPO-C** = red.

---

## fig1_adaptation_effect_c0_c3

**Title:** Adversarial-adaptation effect vs held-out attacker (A1-unseen-203), across C0–C3

- Plots **D1−D0+ safe-interception rate** (percentage points, with 95% bootstrap CIs) against the held-out learned attacker, for all three architectures, across clutter levels C0–C3.
- Input: `results/defender_adaptation_canonical_c0_evaluation/d1_minus_d0plus_by_condition.csv`

**Takeaway:** PPO-Kalman-C benefits from continued adversarial adaptation across C0–C2 (weakening at C3); PPO-C is significantly *harmed* by adaptation, but only in the open-air (C0) condition; LR-PPO-C shows a consistent mild negative trend that never reaches significance.

---

## fig2_a0_retention_vs_a1_generalization

**Title:** A0 retention vs A1-unseen generalization across adaptation generations

- Two panels (left: original scripted attacker A0; right: held-out learned attacker A1-unseen-203), each showing D0 → D0+ → D1 safe-interception rate per architecture (canonical C0, error bars = std across 3 defender seeds).
- Input: `results/defender_adaptation_canonical_c0_evaluation/c0_vs_clutter_table.csv` (means) and `episodes.csv` (per-seed error bars via groupby aggregation).

**Takeaway:** All three architectures retain their A0 (scripted-attacker) performance through D1; PPO-Kalman-C simultaneously improves against the held-out learned attacker, PPO-C gets markedly worse against it, and LR-PPO-C changes little in success rate despite retraining.

---

## fig3_lrppo_clutter_vs_adaptation

**Title:** LR-PPO-C: clutter-navigation strength vs. adversarial-adaptation limitation

- **Panel A:** obstacle-aware retraining recovery for LR-PPO (trained − frozen safe-interception rate) across C1–C3, highlighting the **+7.7 pp** recovery at C3.
- **Panel B:** D1−D0+ adaptation effect vs A1-unseen-203 for LR-PPO-C across C0–C3 (same metric as Fig. 1, LR-PPO-C only).
- **Panel C:** mean residual steering-correction angle for D0/D0+/D1 (canonical C0, averaged over the 4 attacker conditions), showing growth from **~6.9°** to **~11.9°**.
- Inputs: `results/clutter_retraining_evaluation/recovery_deltas.csv` (Panel A); `results/defender_adaptation_canonical_c0_evaluation/d1_minus_d0plus_by_condition.csv` (Panel B); `results/defender_adaptation_canonical_c0_evaluation/lrppo_residual_canonical_summary.csv` (Panel C).

**Takeaway:** LR-PPO remains strong for clutter navigation — obstacle-aware retraining meaningfully recovers performance in the densest clutter — but adversarial retraining shifts its internal residual-correction behavior substantially without producing a corresponding, statistically detectable success-rate gain.

---

## fig4_ppoc_c0_regression

**Title:** PPO-C: adversarial-adaptation effect is significantly negative only in open-air (C0)

- Bar chart of PPO-C's D1−D0+ effect vs A1-unseen-203 across C0–C3, with 95% CIs; the canonical C0 value (**−26.3 pp**, 95% CI **[−51.3, −6.0]**) is annotated directly.
- Input: `results/defender_adaptation_canonical_c0_evaluation/d1_minus_d0plus_by_condition.csv`

**Takeaway:** PPO-C's adversarial-adaptation regression is statistically significant only in the open-air (C0) condition and is absent (CIs include zero) in C1–C3 — an architecture-specific, environment-dependent finding that motivates the next PPO-C diagnostic.

---

## fig5_four_condition_decomposition *(optional, included)*

**Title:** D1 four-condition decomposition: attacker type × environment

- Grouped bar chart of D1 safe-interception rate under the four combinations of {A0, A1-unseen-203} × {C0, clutter (mean of C1–C3)}, for all three architectures.
- Input: `results/defender_adaptation_canonical_c0_evaluation/four_condition_decomposition.csv`

**Takeaway:** The decomposition makes visible that PPO-C is the only architecture whose relative ranking drops when moving from A0 to A1-unseen in both environments, while LR-PPO-C and PPO-Kalman-C retain comparatively higher rates under the held-out attacker.

---

## Notes

- All PNGs are saved at 200 DPI; PDF copies are included for paper drafts.
- `fig*_source.csv` files are small, pre-aggregated summary tables (not full episode datasets) — safe to track in git.
- Every plotted number was checked against its underlying summary CSV/JSON before finalizing these figures (see commit message / task log for the verification pass).
