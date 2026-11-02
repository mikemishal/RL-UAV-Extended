# Manuscript notes — curriculum Stage 3 (moving scripted defender)

## Claim supported
After expert initialization (Stage 1) and static-threat evasion (Stage 2), a PPO attacker transfers its evasive behaviour to a **moving** scripted interceptor (canonical Greedy), improving capture by +7.5…+14.5 pp over its own Stage-2 parent (paired 95 % CIs exclude 0; held-out capture 82.5–86.5 % vs 0 % for the scripted attacker), while retaining pursuit (Stage-1 anchor 1.00) and static-threat evasion (0.90–0.95). The claim is limited to a **curriculum scaffold in which the defender is held until canonical detection**.

## Stage-2 manual curriculum-transfer decision (state it plainly)
Stage2-v2 (16 rollouts) did **not** meet its frozen quantitative mastery rule (no seed had two consecutive 90 % validations; held-out capture 90.0 / 88.5 / 89.0 %). The rule was not relaxed or rewritten. After manual visual inspection the rollout-16 policies were judged qualitatively acceptable for curriculum transfer and were used only as *provisional curriculum initialization checkpoints* (`parent_status = provisional_curriculum_transfer`); they are not accepted A1 models. Report Stage 2 as "not mastered by rule; used as provisional parents after manual review" and never as a pass. Why the rule was not rewritten: a pre-registered gate that is changed after seeing the data stops being a test; the manual decision is a separate, documented, reversible curriculum choice.

## Design correction that must not be hidden (Stage3-v1 → v2)
The first Stage-3 design (canonical Greedy from step 0) was stopped by its own pre-registered difficulty check: A0 was intercepted in 0/1000 scenarios, because Greedy escorts the asset until detection and a defender placed between the attacker and asset accelerates away from it. The correction (hold until canonical detection, then unchanged Greedy) is a deliberate curriculum scaffold, introduced *before any PPO training*, with the stop evidence (`stage3/scenario_validation.json`), the side-by-side baseline (`stage3_v2/baseline_comparison.json`) and the tightened A0 difficulty bound (≥ 70 %) recorded. The A0-baseline-only examination used to define "immediate contact" is disclosed.

## Hypothesis (pre-registered)
> A policy that has learned static-threat evasion can transfer that behavior to a moving scripted interceptor while preserving its expert-initialized pursuit capability.

Stage 3 isolates moving-adversary complexity; moving asset, obstacles and defender-policy diversity are excluded.

## Protocol (what a reviewer needs)
* Environment: C0, stationary asset, canonical Greedy defender (18 m/s, 8 m/s², 90 °/s, swept interception 2.5 m), held fixed (zero velocity/acceleration) until canonical detection (15 m), released the next step, never re-held. Placement: nominal-A0-path method (λ ~ U(0.35, 0.65), r ~ U(0, 1.5 m)), unchanged.
* Observation 37-D (`a1_v2_obs_v1`), action `a1_cur1_action_v1`, reward `a1_v2_reward_v2`, PPO hyper-parameters frozen; release state is environment-side only (audit: no harmful hidden state because the defender is invisible/stationary until inside the 25 m gate and release onset is visible in the velocity feature).
* Initialization: Stage2-v2 rollout-16 policy (actor + critic + log_std), **fresh optimizer**, counter reset. 3 seeds × ≤ 16 rollouts × 32,768 steps.
* Gate (frozen): two consecutive validation passes: capture ≥ 0.80, ≥ +20 pp over matched A0, interception ≤ 0.20, timeout/exhaustion ≤ 0.02, median path-excess ≤ 1.80, >3× ≤ 0.05, >5× = 0, no orbit, Stage-1 anchor ≥ 0.98, Stage-2 static anchor ≥ 0.80, no numerical failure. Retention stop: two consecutive anchor failures.
* Fresh held-out blocks (the suggested 149000–149199 conflicted with Stage 2's own placement-validation block and was replaced).

## Results to quote
Mastery at rollouts 3 / 5 / 7 (98,304 / 163,840 / 229,376 Stage-3 steps). Held-out (200 matched scenarios): capture 83.5 / 82.5 / 86.5 %, interception 16.5 / 17.5 / 13.5 %, median path-excess 1.05, no >3×, median minimum defender separation 5.5–5.7 m (A0: 2.5 m). Paired A1 − A0: +83.5 / +82.5 / +86.5 pp; A1 − parent: +10.5 [6.5, 15.0] / +7.5 [4.0, 11.5] / +14.5 [9.5, 20.0] pp. Retention: Stage-1 anchor 1.00; static anchor 0.90 / 0.94 / 0.95 (parents 0.83 / 0.91 / 0.89).

## Behaviour (descriptive; never rewards)
No deviation before the defender enters the 25 m gate; a > 1 m displacement from the same policy's no-defender path in 86 % of captures, median 0.5 s after detection; median max displacement ≈ 2 m (lateral ≈ 1.2–1.6 m, vertical ≈ 0.8–1.1 m); closest approach ≈ 5.6–6.0 m; post-encounter cosine to the asset 0.95–0.97. Classes: mixed and lateral dominate; timing/turn ~8–16 %; vertical ≤ 7 %; unclear ~14 %.

## Generalization (diagnostic only)
Against Kalman-Greedy / PN / Lead under the same hold semantics: capture 69–87 %, +6…+26 pp over the parents, within a few points of Greedy. Interpreted as evidence the behaviour is not Greedy-specific; no controller other than Greedy was trained against.

## Limitations to state
1. Hold-until-detection scaffold (defender speed ≤ 8 m/s, ≤ 1 s of motion before the encounter) — a hard-start moving threat, not sustained fast pursuit; must be removed in later stages.
2. Provisional parents; Stage-2 not mastered by rule.
3. Mastery gate reached marginally; the 200-episode paired evaluation is the confirmation. One scripted defender family, one geometry, 3 seeds.
4. Small evasions (≈ 2 m displacement): policies buy just enough clearance; 13.5–17.5 % interceptions remain.
5. No manual inspection yet at the time of writing (Stage 3 recordings registered for review).

## Figures
`figures/fig1…fig6` (PNG + PDF): capture vs timesteps, interception vs timesteps, pursuit retention, static-evasion retention, path-excess, representative A0 vs A1 moving-defender trajectory (seed 203).
