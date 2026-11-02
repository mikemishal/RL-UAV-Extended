# Manuscript notes — Stage2-v2 (training-budget ablation of static-defender evasion)

## Framing
Stage2-v1 (8 rollouts, 262,144 steps) failed the frozen mastery rule although all seeds were learning. Stage2-v2 is a **preregistered, budget-only continuation** (16 rollouts, 524,288 steps) with every other definition unchanged (protocol diff = `ppo.max_rollouts`, `ppo.max_timesteps`). Report both stages, including the v1 failure; this is a training-budget ablation, not a rescue.

## Result in one paragraph
Doubling the budget raised held-out capture on the same 200 scenarios from 67.0 / 58.5 / 55.5% (v1) to **90.0 / 88.5 / 89.0%** (v2) — paired +23.0 / +30.0 / +33.5 pp, 95% CIs [17.0, 29.0] / [24.0, 36.5] / [27.0, 40.0] — against canonical A0's 0% (paired A1 − A0 +90.0 / +88.5 / +89.0 pp). Interception fell from 33–45% to 10–11.5%; median path-excess stayed 1.04; Stage-1 pursuit was fully retained (anchor 1.00 at every rollout; no-defender capture 100%); no fly-bys, orbits, timeouts or numerical failures. **Nevertheless no seed mastered Stage 2**: the pre-registered rule requires two consecutive validation passes (≥ 90% capture on alternating 100-episode sets); seed 201 passed rollouts 14 and 16 but scored 0.86 at rollout 15, seed 203 passed only rollout 16, seed 202 peaked at 0.88.

## Answer to the research question
"Was static-defender evasion limited primarily by learning budget?" — **Mostly, yes**: the same policies, same protocol, twice the steps → +23 to +34 pp, with all structural diagnostics already healthy at 8 rollouts. But the preregistered prediction (that extension would reach mastery) was **not confirmed**: policies converge to the threshold (≈ 0.86–0.93 per validation set) without a stable two-consecutive pass inside the budget. Curves were still rising (descriptively), so a further budget increase is plausible but would be a new scientific decision and a new registered version; it was not made.

## Points to state carefully
* The final 200-scenario capture of seed 201 (90.0%) satisfies every individual gate check; this is **not** a mastery pass (mastery is defined on consecutive validation passes). Do not present it as passing.
* The validation noise near the threshold (s.e. ≈ 3 pp per 100 episodes) makes consecutive-pass criteria demanding; the gate was not relaxed after seeing the results.
* Behavior: detours became wider and safer (median closest defender distance 5.2–5.4 m vs 4.0–4.5 m at v1 and 2.5 m for A0), lateral 6.6–7.3 m, vertical 3.7–4.2 m, both horizontal and vertical sides used; reactive (identical to the no-defender trajectory until the 25 m gate, 75–86% displaced > 1 m afterwards). Post-encounter cosine toward the asset 0.94–0.98; resume criterion met in 43 / 70 / 93% of captures; the strict three-phase pattern fell to 14 / 20 / 29% (from 37 / 69 / 55%) because longer-trained policies take larger detours — a property of the fixed 10°-tolerance definition, reported as is.
* Forgetting: action cosine to the BC initializer 0.971–0.979 (v1: 0.989–0.993), +4–6% duration in no-defender C0, capture 100%: pursuit preserved, drift larger with more training.
* The v1→v2 comparison uses checkpoints from the same training trajectories (rollout 8 vs rollout 16), not independent runs.

## Learning-curve statements (descriptive, no curve fit)
Rollouts 9–16: validation capture rising on all seeds; interception falling; mean return rising (≈ −9.5 at rollout 1 → +8.0…+9.0 at rollout 16); median path-excess and anchor capture plateaued (≈ 1.03–1.04 and 1.00).

## Limitations
Three seeds; 100-episode validation sets; defender on A0's nominal path by construction; C0 only; fixed 25 m gate; checkpoints are curriculum initializations, not final A1 models; none accepted.

## Reproduction
`python -m experiments.journal_v2_a1_cur2v2_run all` (restart-safe, resumes from verified rollout 8/k), then `... final`. Protocol SHA `ff679fa4…`; parents hashes in `parent_checkpoints.json`.
