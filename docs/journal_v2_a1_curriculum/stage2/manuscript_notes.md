# Manuscript notes — curriculum Stage 2 (static-defender evasion)

## What this stage shows
Starting from the A0-expert BC initializations (Stage 1 V2: 200/200 held-out captures), unmodified PPO (frozen hyperparameters, `a1_v2_reward_v2`, 262,144 steps) learned to evade a stationary, fully active interception threat that A0 *always* flies into: matched held-out capture rose from A0's **0%** to **67.0 / 58.5 / 55.5%** (paired differences +67.0 / +58.5 / +55.5 pp, 95% CIs [60.5, 73.5] / [51.5, 65.5] / [48.5, 62.5]), interception fell correspondingly, Stage-1 pursuit was retained perfectly (anchor capture 1.00, no-defender path-excess 1.03 vs BC 1.02), and detours were small and efficient (median path-excess 1.02–1.03, median minimum defender distance 3.0–3.7 m vs 2.5 m for A0). **However, no seed reached the pre-registered mastery gate (≥ 90% capture on two consecutive validations) within 8 rollouts, so Stage 2 is reported as FAILED**; validation capture was still increasing at the budget limit.

## Narrative / rationale
1. **Stage-1 BC success** (Stage 1 V2): the expert-initialized policies already solve basic pursuit, so Stage-2 learning can be attributed to the single new skill — avoidance — with a measurable pursuit-retention anchor.
2. **Why a stationary defender.** Isolating one factor: fixed threat, fixed asset, no obstacles, canonical interception and information gate (25 m), unchanged reward. A stationary defender removes anticipation and pursuit dynamics; the policy must only learn *where to deviate* from what it observes within the gate.
3. **Placement geometry (a transparent protocol amendment).** The first design (defender on the initial attacker→asset chord) failed the pre-registered scenario rule (A0 interception 51.5% < 70%) because the Stage-1 random initial heading bends A0 off the chord (A0 interception falls from 100% at < 15° to 1% at > 75° of initial offset). Training had not started; the stop was reported, the user approved placement relative to A0's noise-free nominal path, and the gate was then met with 100% A0 interception. Both designs and the diagnostic are archived. Report the amendment in the dissertation; it demonstrates that the stop rules operated as intended.
4. **PPO fine-tuning protocol.** Matching BC actor + `log_std = −2`, PPO-seed-fresh critic and fresh optimizer; frozen PPO configuration; 16 × 2048 steps per rollout; A/B alternating held-out validation with matched A0 and a no-defender anchor after every rollout; two-consecutive-pass mastery; retention-failure rule (never triggered). Power-loss-safe resume was validated with real SB3 before the campaign.
5. **Result.** 0/3 mastered (final validation capture 0.73 / 0.65 / 0.59). Capture is monotone-ish in timesteps (Fig. 1); failures are partial or misdirected dodges rather than a failure to react (intercepted episodes show larger displacement from the no-defender trajectory than captured ones).
6. **Learned-evasion evidence (causal).** Replaying the same policy without the defender: trajectories coincide until gate entry (observations identical) and then diverge in 74–88% of episodes (median displacement ≈ 1.7 m) — a defender-triggered, not a blind, detour. Pursuit → evasion → pursuit: post-encounter heading re-aligns with the asset (mean cosine 0.97–0.99), detours use both horizontal and vertical directions, and path-excess remains ≈ 1.03.
7. **Forgetting analysis.** No-defender capture 100%, action cosine to the BC initializer 0.989–0.993 (≈ 5–8°), path-excess +1–2%: adversarial fine-tuning barely perturbs the pursuit skill.

## Quantitative claims safe to cite
* A0 vs A1 on 200 matched scenarios: 0% → 55.5–67.0% capture; paired CI excludes 0 for all seeds; A1 never lost a scenario A0 won (A0 wins none).
* Anchor (Stage-1 pursuit) capture 1.00 (all rollouts ≥ 0.99).
* 0 fly-bys, 0 orbits, 0 timeouts/exhaustion, 0 numerical failures.

## Claims NOT supported
* "Stage 2 is passed / evasion is mastered": the 90% gate was not met by any seed.
* "A1 is near-optimal": A0's 0% capture is a property of the placement (by design), not a strong baseline.
* "Budget is the cause": consistent with the curves but untested; a longer budget would be a new registered version.
* Generalization beyond: a fixed threat on A0's own nominal path, C0 only, one fixed 25 m gate, three seeds.

## Limitations
* The defender is placed on A0's path by construction (adversarial to A0); an A1-specific path distribution may be easier or harder.
* Three seeds, 200 evaluation scenarios (binomial s.e. ≈ 3.4 pp at 60%).
* Validation sets are 100 episodes (s.e. ≈ 5 pp), so individual rollout curves are noisy.
* Detours are evaluated against the nominal A0 path and the initial chord; other evasion metrics are possible.
* The strict pursuit→evasion→pursuit pattern requires a pre-encounter pursuit phase and so under-counts scenarios that begin inside the gate (≈ 24%).

## Reproduction
`python -m experiments.journal_v2_a1_cur2_run all` (restart-safe), then `python -m experiments.journal_v2_a1_cur2_final all`. Frozen protocol SHA: `protocol.json`; parents: `stage2_handoff.json` of Stage 1 V2 (tag `journal-v2-a1-curriculum-bc-stage1-v1`).
