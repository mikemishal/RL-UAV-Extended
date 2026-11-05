# Manuscript notes — curriculum Stage 4A-1 (goal-conditioned translated-target imitation)

## Claim supported (a carefully qualified positive-and-negative result)
Randomizing the stationary asset's position during supervised imitation of canonical A0 **does** teach the policy to depend strongly and correctly on the asset-position observation slots
(asset/attacker input-sensitivity ratio rises from ~2.2% in every Stage3-v2 parent to 69–114% after fine-tuning — a 27–52× gain, far exceeding the pre-registered 2× representation-gate
requirement), confirming that the Stage-4 moving-target failure was in large part a representation problem, not a hard-ceiling task difficulty. However, the specific learning method used here
(single-round, open-loop behaviour cloning with no closed-loop correction) does **not** convert that representation change into a reliable closed-loop policy: held-out translated capture
reaches only 55–60.5% (mastery requires 99%), and all three seeds triggered the pre-registered `RETENTION_FAILURE` stop within 2 epochs because the SAME instability also wrecked the unrelated,
previously-mastered fixed-origin/static-defender/moving-defender anchors. Verdict **C**.

## Pre-registered hypothesis
> The Stage-4 moving-target failure was caused in part by inadequate dependence on asset-position inputs created by fixed-origin curriculum training. Randomizing the stationary goal location
> during expert imitation should produce translation-aware pursuit while preserving previously learned defender-evasion behavior.

The first half is **supported** (§ below); the second half ("while preserving previously learned ... behavior") is **falsified** — the opposite happened, and severely, within 2 epochs.

## Design (what a reviewer needs)
* Unchanged Stage-1-v2 no-defender mechanism (parked/masked/non-intercepting defender), C0, `a1_v2_obs_v1`, `a1_cur1_action_v1`, `a1_v2_reward_v2` (unused by BC). Only change: both the
  attacker's canonical edge-spawn position and the asset's origin are translated by the same horizontal δ (dedicated stream, independent of all canonical streams), exactly preserving relative
  geometry and the endurance budget; a 5 m margin keeps both bodies off the wall; exact per-scenario feasible box, no clipping (infeasible draws raise an exception and a different scenario is
  tried). The asset is never moved **during** an episode (`v_s = 0`), consistent with the task constraint; only its INITIAL position is randomized episode-to-episode.
* Expert dataset: 6,000 translated A0 episodes, 67,245 pairs (target 60k–80k), quadrant/diagonal/central/outer coverage all present, 80/20 episode split.
* Retention (rehearsal) dataset per seed: ~11,000–11,244 states, driven by the SEED's OWN Stage3-v2 parent (not A0) on reserved scenarios, labelled with the parent's own action (never a future
  outcome), classified post-hoc into not_visible/newly_visible/active_evasion/post_pass_pursuit for reporting only.
* Objective (frozen before training): `L = MSE(a, a_A0) + (1 − cos(a, a_A0)) + 0.25 · [MSE(a, a_parent) + (1 − cos(a, a_parent))]`, actor-only (verified: critic and `log_std` bit-identical
  before/after training), `log_std` forced to −2.0 at initialisation.
* Gate (frozen): two consecutive epoch passes with capture ≥ 0.99, exhaustion/timeout = 0, median path-excess ≤ 1.20, >2× ≤ 1%, >5× = 0, mean cosine ≥ 0.995, min cosine ≥ 0.95, three anchors
  ≥ 0.99/0.80/0.75, no numerical failure; retention stop = two consecutive epochs below ANY anchor minimum. Representation gate: POST/PRE sensitivity ratio ≥ 2×, no translation-generalization
  region < 95% capture.

## Results to quote
* PRE (parent) on translated held-out: capture 4.0–6.0% (A0 on the same scenarios: 100%); asset-input sensitivity ratio 2.14–2.52% — confirms the Stage-4A diagnostic at full statistical power
  and with a near-worst-case task (the parent essentially ignores the goal location it was never trained to attend to).
* POST (Stage4A-1, accepted = epoch 2, the last completed epoch before the stop): held-out capture 55.0–60.5% (+50 to +56.5 pp over the parent on identical scenarios); asset-input sensitivity
  69.3–114.4% (27.5–51.6× PRE); anchors 0.41–0.59 / 0.23–0.44 / 0.25–0.43 (required ≥ 0.99 / 0.80 / 0.75) — **every** anchor fails by 40–90 pp, including the exact fixed-origin task
  (Anchor 1) the parent already solves at ~1.00.
* Open-loop vs closed-loop cosine: validation-set (open-loop, states A0 itself visited) cosine is 0.97–0.99 at every epoch; closed-loop (states the POLICY ITSELF visits) cosine on the final
  200-scenario evaluation is only 0.03–0.48, and one traced episode (seed 201, scenario 178002) timed out at 2,000 steps with **negative** mean closed-loop cosine (−0.18) from the SAME
  checkpoint that scores ~0.99 on the open-loop set. This open/closed-loop gap is the direct, measured signature of behaviour-cloning covariate shift (Ross & Bagnell): small per-step errors push
  the policy into states the expert-only training set never covered, and it has no corrective data to recover.
* Generalization regions: capture 25–85% across the 8 quadrant×magnitude strata, no systematic quadrant or inner/outer pattern (ruling out "forgot one region" in favour of general closed-loop
  instability).
* Qualitative (fig. 6): the fine-tuned policy visibly turns toward and orbits the CORRECT, translated asset location in all four spot-checked targets, while the untrained parent frequently flies
  toward an unrelated point — visual confirmation that the representation changed as intended, even though the orbit rarely collapses into a capture.

## Why report this as a genuine finding, not a bug
Checked explicitly before accepting the result: (1) actor-only training and frozen critic/log_std were verified bit-identical pre/post in unit tests; (2) the retention dataset is driven by the
PARENT's own policy (verified byte-identical recomputation), not a mislabeled target; (3) translation preserves relative geometry and the endurance budget to < 1e-4 m (tested); (4) deterministic
replay holds; (5) the open-loop cosine is genuinely excellent (ruling out a broken loss/label), while the closed-loop cosine and the direct 2,000-step divergent trajectory are independent,
reproducible evidence of real closed-loop instability. The combination (huge representation gain + huge open-loop fit + closed-loop collapse) is the textbook covariate-shift signature, not an
artifact of the evaluation code.

## Limitations to state
1. Both seeds' training stopped at the pre-registered retention floor after 2 epochs; longer training under the SAME protocol was not attempted (the frozen rule was not relaxed to find out).
2. Per-epoch checks used 40/200 held-out scenarios for speed; only the final (200-scenario) numbers are authoritative.
3. No closed-loop (DAgger-style) relabeling round was included, unlike the original Stage-1-v2 BC protocol — this is the most likely, concrete fix and is proposed as the next learning step, not
   evaluated here.
4. Parents are provisional curriculum checkpoints (no manual-review acceptance of Stage3-v2 recorded).
5. Three seeds, with a consistent and severe retention collapse in all three — informative precisely because it is NOT an outlier.

## What this changes in the curriculum story
Stage 4 (moving target + moving defender, PPO) failed with weak asset sensitivity. Stage 4A (isolated moving-target pursuit, PPO) failed, with a measured representation deficiency as the
probable cause. Stage 4A-1 (isolated STATIONARY randomized-goal imitation) directly tests and **confirms** that representation deficiency, and fixes it — but exposes a second, independent
obstacle: naive open-loop imitation is unstable in closed loop at this scale of state-space expansion. The next learning step is therefore not "add clutter" or "move to Stage 4B," but a
closed-loop correction round (DAgger-style relabeling) applied to this SAME translated-goal task, before any moving-asset imitation is attempted.

## Figures
`figures/fig1…fig6` (PNG + PDF): translated-target capture vs epoch (40-scenario check); mean A0 cosine (open-loop, validation set) vs epoch; PRE vs POST asset/attacker sensitivity ratio; the
three retention anchors vs epoch; capture by translation-generalization region; representative translated trajectories (seed 202, four quadrant targets, Stage4A-1 vs Stage3-v2 parent).
