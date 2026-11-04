# Manuscript notes — curriculum Stage 4A (isolated moving-target pursuit)

## Claim supported (negative result with a mechanism — state it plainly)
After Stage 4 (moving canonical asset + moving defender) failed, the task was decomposed. With the defender removed from the interaction (parked, masked, no interception) and the canonical moving asset kept, **PPO fine-tuning of the
Stage3-v2 curriculum policies did not master moving-target pursuit** (0/3 seeds mastered in 12 rollouts = 393,216 steps; held-out capture 81.0 / 74.0 / 83.5 % vs 100 % for scripted A0 and 70.5 / 71.5 / 78.0 % for the parents; paired
Stage4A − parent +10.5 [4.5, 16.5] / +2.5 [−2.5, 7.5] / +5.5 [−0.5, 11.5] pp). The pre-registered hypothesis — *the Stage-4 failure is primarily missing moving-target pursuit competence, not loss of evasion* — is **supported**: with no defender at all
the parents already capture only 67.8–71.0 % of moving-asset scenarios (A0 98.7 %), and the failed Stage-4 checkpoints are no better on this task (66.5 / 74.5 / 80.0 %). The claim that isolating the skill makes it learnable within a modest PPO budget is **not** supported.

## Mechanism (the finding worth reporting)
* Behaviour: repeated fly-bys. 16.5–26 % of episodes exhaust endurance after ≈ 58 steps with 3–4 closest approaches at median miss distances 4.0–4.8 m (capture radius 2 m); first-pass capture is only 60–66 % (A0 99 %).
* Tracking diagnostic: restricted to decisions where the current and the initial asset locations are ≥ 3° apart, only 31–35 % of decisions (parents 19–30 %) are better aligned with the current asset than with the obsolete initial location.
* **Asset-input diagnostic:** in Stages 1–3 the asset sat at the origin, so the asset-position observation slots were constant zero and untrained. In the Stage3-v2 parents the first-layer weight columns of those slots have norm exactly √2 (orthogonal-initialisation value) vs 8–14
  for the attacker-position columns; the sensitivity of the executed direction to the asset position is 2.1–2.5 % of that to the attacker position (a relative tracker ≈ 100 %). After Stage 4A it is 9.4–10.4 % with the correct (relative) sign structure (alignment 0.81–0.90), after
  the 16 Stage-4 rollouts 13–14 %. Implication for the manuscript: *a pretrained policy can carry an observation channel that was never exercised by the curriculum; its absence is invisible in the stationary stages and is what makes the first moving-target stage hard.*
  This is an observational diagnostic of checkpoints (weights/Jacobians), not an intervention; do not state it as a proven cause.

## Design facts a reviewer needs
* Environment: the established Stage-1-v2 clean curriculum mechanism (defender parked and masked, interception impossible) with exactly one change — the asset's canonical motion is restored (`legacy_gaussian_random_walk`, v_s = 1.5 m/s, σ = 0.75 m per axis per step, reflecting walls, dedicated RNG
  stream). Config differs from canonical C0 V2 only in `intercept_radius` and `defender_standby_until_detection`. Obs `a1_v2_obs_v1`, action `a1_cur1_action_v1`, reward `a1_v2_reward_v2` and PPO unchanged; no moving-target shaping.
* Initialisation from the accepted Stage3-v2 checkpoints (actor + critic + log_std, fresh optimizer) — **not** from the failed Stage-4 models (they mix failed adaptation to both target motion and the defender).
* Common random numbers: A0, parent and Stage4A see the identical asset random walk on every evaluation seed (max difference 0.0 m on completed steps; terminal contact state excluded); asset realisation equals the Stage-4 environment's.
* Baseline (1,000 scenarios, configuration validity only): A0 98.7 % capture; parents 69.2 / 67.8 / 71.0 % with 29–32 % endurance exhaustion and 52.8–56.2 % first-pass capture.
* Gate (frozen before training): two consecutive validations with capture ≥ 0.95, exhaustion ≤ 0.02, timeout ≤ 0.01, median path-excess ≤ 1.5, >2× ≤ 0.05, >3× ≤ 0.02, >5× = 0, no orbit, repeated fly-bys ≤ 5 %, anchors ≥ 0.98 / 0.80 / 0.75, no numerical failure.
  The gate failed at all 36 rollout validations. Retention stop never triggered; final anchors 1.00 / 0.85–0.93 / 0.79–0.82 (parents 1.00 / 0.93–0.97 / 0.89–0.92).
* Validation seeds: the suggested 166000–166199 was occupied by the Stage-4 anchors, so 169000–169199 were used (documented before training).
* Failure rules were fixed in advance (`decide()` in `experiments/journal_v2_a1_cur4a_final.py`); verdict C was produced by them, not by judgement.

## Honest caveats
1. Retention eroded mildly under moving-target training (Anchor 3: 0.91 → 0.79 in seed 201; −8…−13 pp vs the parents at the end); no anchor fell below its minimum.
2. Short episodes and a small drift (median 2.9 m) — the task probes end-game precision more than long-range tracking.
3. Single frozen PPO configuration and budget; the flat-ish learning curves (validation capture 0.68–0.89, noisy) do not prove unlearnability under other settings.
4. Seed 201 shows a significant gain over its parent; seeds 202 and 203 do not.
5. Parents are provisional curriculum checkpoints (no manual-review acceptance of Stage3-v2 recorded). No manual review for Stage 4A (not passed).

## What this changes in the curriculum story
Stage 4 failure ⇒ decomposition ⇒ Stage 4A failure with a located cause. The next stage must change the learning step, not the difficulty: e.g. give the asset slots real signal before the asset moves (stationary asset at randomised positions) or imitate A0 on
moving-asset scenarios (the mechanism that made Stage 1-v2 work), keeping the reward unchanged. Stage 4B (moving asset + moving defender) is deferred until moving-target pursuit is mastered.

## Figures
`figures/fig1…fig5` (PNG + PDF): capture vs Stage-4A timesteps; repeated fly-by rate vs timesteps; median path-excess vs timesteps; the three retention anchors; A0 / Stage3-parent / Stage4A trajectories on the common moving-asset path (seed 203, scenario 169013).
