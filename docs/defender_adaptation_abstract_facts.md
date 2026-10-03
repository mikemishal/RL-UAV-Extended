# Defender-Adaptation (D1) Abstract-Ready Quantitative Facts

Publication-safe, conservative facts only. Every fact below is traceable to a specific table/figure/CSV
under `results/defender_adaptation_evaluation/`. Do not state a stronger version of any fact than listed
here in the abstract, Results, or conclusion — see `docs/defender_adaptation_claims_evidence.md` for the
full support-level ledger these facts are drawn from.

1. **Evaluation scale**: 32,400 held-out episodes (27 defender instances × 4 attacker conditions × 3
   clutter levels × 100 matched evaluation seeds), 100% complete, 0 missing, 0 duplicate/conflicting
   records. *Source: `results/defender_adaptation_evaluation/analysis_input_integrity.json`.*

2. **Training budget**: every one of the 18 D0+/D1 continuation runs received exactly 425,984 additional
   environment transitions (one rollout's worth beyond the requested 400,000, from
   `num_envs=16 × n_steps=2,048` vectorized rollout arithmetic — identical for every run). *Source:
   `results/defender_adaptation/{d0_plus_scripted,d1_adaptive_mixture}/*/seed_*/training_manifest.json`.*

3. **Primary finding (PPO-Kalman-C)**: safe-interception rate against the held-out learned attacker
   (A1-unseen, seed 203), equal-weight C1–C3 aggregate, increased from 35.4% (D0+) to 51.8% (D1), an
   absolute gain of +16.3 percentage points (95% hierarchical-bootstrap CI: [+8.0, +24.0] pp), with no
   loss of scripted-attacker (A0) performance (58.0% → 58.2%, CI [−10.2, +10.9] pp). *Source: Table 1 and
   Table 2, `results/defender_adaptation_evaluation/tables/`.*

4. **PPO-C and LR-PPO-C showed no statistically supported primary gain**: PPO-C's D1−D0+ aggregate delta
   against A1-unseen was −0.3 pp (CI [−6.4, +6.1] pp); LR-PPO-C's was −5.8 pp (CI [−13.7, +1.9] pp). Both
   CIs include zero. *Source: Table 1.*

5. **Extra training alone does not explain the PPO-Kalman-C gain**: the equal-budget D0+-vs-D0 control
   (same 425,984 extra transitions, scripted attacker only, no mixture) showed a non-significant −3.6 pp
   change against A1-unseen for PPO-Kalman-C (CI [−15.0, +10.0] pp) — the opposite sign from D1's gain —
   isolating the +16.3 pp primary effect as attributable to the adversarial opponent mixture specifically.
   *Source: `statistics.json::extra_training_d0plus_vs_d0_unseen`.*

6. **A0 retention held for all three architectures**: D1−D0+ against the original scripted attacker was
   within ±2.1 pp for every architecture, with every CI straddling zero — no evidence of catastrophic
   forgetting. *Source: Table 2.*

7. **Clutter dependence (PPO-Kalman-C, descriptive)**: the adaptation gain against A1-unseen was largest
   at low clutter (C1: +26.3 pp, CI [+11.7, +39.7]), smaller at medium clutter (C2: +16.7 pp, CI [+3.7,
   +28.3]), and not statistically distinguishable from zero at high clutter (C3: +6.0 pp, CI [−5.0,
   +17.0]). *Source: Table 1, per-level rows.*

8. **Defender-seed consistency (PPO-Kalman-C)**: the C1 adaptation gain was directionally consistent
   across all 3 independently-trained defender seeds (59%→72% range in D1 vs 29%→44% range in D0+, each
   seed individually higher in D1). *Source: `results/defender_adaptation_evaluation/defender_seed_summary.csv`.*

9. **LR-PPO residual-angle usage increased under adversarial adaptation**: mean residual angle rose from
   approximately 6.9° (D0) to 8.0–8.9° (D0+) to 10.9–12.5° (D1) across attacker conditions; the fraction
   of steps within 2° of the 30° bound rose from 0.0000 (D0, matching prior reported evidence) to at most
   0.0001 (D0+) to 0.0003–0.0037 (D1) — still a small fraction of steps, but a consistent, monotonic
   increase present even against the original scripted attacker. *Source:
   `results/defender_adaptation_evaluation/lr_ppo_residual_analysis.csv`.*

10. **Safety-layer dependence was essentially unchanged**: obstacle-avoidance intervention fraction varied
    by at most ~2 percentage points across D0/D0+/D1 for any architecture against any attacker condition.
    *Source: `results/defender_adaptation_evaluation/safety_layer_dependence.csv`.*

11. **Full regression**: 982 tests passed, 0 failures, after both the training-pipeline commit
    (`69596d8af8dc27bb080fa1eee37a38ea5090b6ef`) and the evaluation-harness commit
    (`1abd7c57ab7a78d43752cfb21d3e254e2af0a548`). *Source: `results/phase36_full_regression.log`.*
