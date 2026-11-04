# Curriculum Stage 4A — isolated moving-target pursuit (canonical moving asset, no active defender, C0)

**Result: Stage 4A FAILS the pre-registered rule — 0/3 seeds mastered in 12 rollouts (393,216 Stage-4A timesteps each); verdict C ("moving-target pursuit requires a different learning step").** Held-out capture is
**81.0 / 74.0 / 83.5 %** (canonical A0 on the same scenarios: 100 %; each seed's own Stage3-v2 parent: 70.5 / 71.5 / 78.0 %). PPO fine-tuning with the frozen reward moved the policies in the right direction but slowly
(paired Stage4A − parent capture **+10.5 [4.5, 16.5], +2.5 [−2.5, 7.5], +5.5 [−0.5, 11.5] pp**; repeated fly-by episodes fall by 3.5–10 pp), and it eroded the retention anchors only mildly (all stay above their minimums).
The decisive new evidence is *why*: the Stage3-v2 parents essentially **ignore the asset-position input** (it was constant zero in Stages 1–3), and Stage 4A raises their sensitivity to it only from ~2 % to ~10 % of the sensitivity to
the attacker's own position (§8). The pre-registered hypothesis is supported — the Stage-4 failure is missing moving-target pursuit competence, not loss of evasion — but isolating the task did not by itself make it learnable within the budget.

## 1. Rationale: decomposing Stage 4
Stage 4 (research commit `15faece`; `docs/journal_v2_a1_curriculum/stage4/`) added the canonical moving asset to the Stage3-v2 task (moving Greedy defender, hold-until-detection scaffold). 0/3 seeds mastered, held-out capture
52.5 / 60.5 / 59.5 %, with defender-evasion behaviour retained but **repeated fly-bys at ≈ 3.4 m miss distance (capture radius 2 m) and endurance exhaustion in 17–23 % of episodes**. Stage 4 mixed two skills; Stage 4A isolates the
moving-target one. **Pre-registered hypothesis:** *the Stage-4 failure is primarily caused by insufficient moving-target pursuit competence rather than loss of defender-evasion skill; isolating canonical moving-target pursuit should
allow the curriculum policy to acquire tracking competence before the moving defender is reintroduced.* If Stage 4A had passed, Stage 4B (moving asset + moving defender + C0) would follow; it was not started.
The failed Stage-4 checkpoints and results are preserved untouched as non-canonical evidence and were **not** used for initialisation.

## 2. Environment (exactly one change relative to the Stage-1-v2 curriculum mode)
* **Mechanism reused, not invented:** Stage-1-v2 `curriculum_basic_pursuit_no_defender_interaction` (`journal_v2_a1_cur1v2.Stage1V2Env`): the defender is structurally present, stationary and parked at (−48, −48, 0), never acts; its observation slots
  (defender flag / position / velocity, slots 9–15) are **masked to exactly 0 in every state**; `intercept_radius = 1e-6 m` makes interception impossible; canonical perimeter spawn, altitude U(10, 30) m, initial heading offset U(0, 90°).
* **The one change:** the asset keeps its canonical V2 motion (`v_s = 1.5 m/s` instead of Stage-1's `v_s = 0`). Live canonical parameters (checked against `A.build_episode_config("C0","greedy")`, identical): model
  `legacy_gaussian_random_walk`, `dt = 0.5 s`, i.i.d. horizontal Gaussian displacement with σ = v_s·dt = **0.75 m per axis per step**, z = 0, reflecting boundaries at ±50 m, start at the origin, no velocity persistence, dedicated `_rng_soldier`
  stream (`SeedSequence(seed).spawn(4)[1]`). No simplification. The config differs from the canonical C0 V2 config only in `intercept_radius` and `defender_standby_until_detection`.
* C0 (no obstacles). Observation `a1_v2_obs_v1` (37-D; contains the CURRENT asset position, nothing about the future), action `a1_cur1_action_v1`, reward `a1_v2_reward_v2` — all unchanged; no moving-target shaping, no prediction bonus, no velocity-matching term, no A0 fallback.
* Initialisation: the accepted Stage3-v2 checkpoints (201 `c60ae85d…8e78`, 202 `3dfabf5c…a485`, 203 `818338a5…b52c`; full hashes in `protocol.json`; status "quantitatively accepted curriculum parent"): actor, critic and `log_std` copied, fresh optimizer, counter 0.
  Frozen PPO (MlpPolicy [64,64] tanh, lr 3e-4, γ 0.99, λ 0.95, n_steps 2048, batch 64, 10 epochs, clip 0.2, ent 0.01, vf 0.5, grad-norm 0.5, 16 envs, 2 torch threads), ≤ 12 rollouts × 32,768 steps.
* Protocol SHA-256: `402849afcdd6767bf0ffb78adbf6ed20f6feafc554597bae83779adcb58856a7` (`protocol.json`); manifests of any other protocol are rejected.

## 3. Seed namespaces
The suggested block `166000–166199` is **not free** (it holds the Stage-4 retention anchors `166000–166299`), so the next clean contiguous blocks were used (audited disjoint from every earlier reservation and from the training stream):
validation A `169000–169099`, B `169100–169199`; anchors 1 / 2 / 3 `169200–169299` / `169300–169399` / `169400–169499`; pre-training baseline `171000–171999`. Training scenarios come from the stream `0xC4A1` (seeds ≥ 1,000,000).

## 4. Common random numbers
Same evaluation seed ⇒ same initial attacker state, same asset state and the **same asset random walk** for A0, the Stage3-v2 parent and the Stage-4A policy (tests; baseline report; final evaluation). Verified: maximum difference
**0.0 m** on every completed step (1,752 positions over 50 baseline scenarios; 4,447 / 4,530 / 4,489 positions over the 200 held-out scenarios per seed); the asset realisation also equals the failed Stage-4 environment's on the same seed. Only the
terminal swept-contact event state of an episode differs (≤ 1.7 m) and is excluded from the exact comparison. The observation contains no future asset information.

## 5. Pre-training baseline (1,000 matched scenarios; configuration validity only)
All checks passed (asset moves; canonical motion; defender masked in 2,222 decisions, max |defender slot| = 0; no interception possible; no non-finite state; CRN identical; deterministic replay). The environment is never modified because of difficulty.

| | capture | exhaustion | timeout | median path-excess | >2× / >3× / >5× | fly-by episodes (repeated) | mean closest approaches | first-pass capture |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical A0 | **98.7 %** | 1.3 % | 0 | 1.01 | 1.5 / 1.5 / 0.5 % | 15 (14) | 1.04 | 98.5 % |
| Stage3-v2 parent 201 | 69.2 % | 30.8 % | 0 | 1.11 | 43.3 / 37.4 / 14.3 % | 438 (341) | 2.21 | 56.2 % |
| parent 202 | 67.8 % | 32.2 % | 0 | 1.19 | 47.0 / 40.3 / 17.1 % | 472 (365) | 2.26 | 52.8 % |
| parent 203 | 71.0 % | 29.0 % | 0 | 1.18 | 46.4 / 39.8 / 12.4 % | 464 (341) | 2.11 | 53.6 % |

Even with **no defender at all**, the parents — which capture ~83 % on a *stationary* asset with a moving defender and ≥ 98 % on the stationary no-defender task — capture only ~70 % of moving-asset episodes against A0's 98.7 %; the missing
competence is moving-target pursuit itself (gap to the 95 % gate 24–27 pp).

## 6. Training, mastery and retention (per-rollout tables: `rollout_results.md`)
| Seed | status | rollouts / steps | best validation capture | final (rollout-12) checkpoint SHA-256 (not accepted) |
|---|---|---|---:|---|
| 201 | FAIL | 12 / 393,216 | 0.85 | `4b94fe90d05ecdfa8f894894447251b7c102f3a840a07b249e9ffdac304dde95` |
| 202 | FAIL | 12 / 393,216 | 0.78 | `26ed0b49b462b52ddc62509de7eca5f25441a4e1d7f1b4f5a9b488899904e343` |
| 203 | FAIL | 12 / 393,216 | 0.89 | `bf2f5d97a90b6959e8e551f252a9c5a6f7dc92548d06a3a0492bc19a2bc77938` |

The gate failed at all 36 rollout validations (capture, exhaustion, >2×, >3×, >5×, repeated fly-bys). Validation capture is noisy (0.68–0.89) with a small upward drift (mean of the first / last six rollouts 0.76 → 0.80, 0.71 → 0.75, 0.80 → 0.83);
training-episode capture rose 0.65 → 0.70, 0.63 → 0.64, 0.66 → 0.68. **No retention stop**: anchors at the final checkpoints (Anchor 1 stationary pursuit / Anchor 2 static defender / Anchor 3 moving defender):
201 1.00 / 0.85 / 0.79, 202 1.00 / 0.93 / 0.80, 203 1.00 / 0.90 / 0.82 (parents 1.00 / 0.95 / 0.92, 1.00 / 0.97 / 0.89, 1.00 / 0.93 / 0.90; minimums 0.98 / 0.80 / 0.75). Anchors 2 and 3 erode slowly during
moving-target training (seed 201: 0.94 → 0.85 and 0.91 → 0.79), the first sign of a retention cost.

## 7. Held-out evaluation (`169000–169199`, 200 matched episodes, identical asset trajectories)
| | Stage4A 201 | Stage4A 202 | Stage4A 203 | A0 |
|---|---:|---:|---:|---:|
| Capture | 81.0 % | 74.0 % | 83.5 % | 100 % |
| Endurance exhaustion | 19.0 % | 26.0 % | 16.5 % | 0 |
| Timeout | 0 | 0 | 0 | 0 |
| Mean duration / path length | 12.4 s / 129 m | 13.6 s / 142 m | 11.8 s / 125 m | 5.7 s / 62 m |
| Median path-excess; >2× / >3× / >5× | 1.06; 35 / 28 / 10.5 % | 1.08; 39.5 / 33.5 / 10.5 % | 1.06; 34 / 26.5 / 7.5 % | 1.01; 1 / 0.5 / 0 % |
| Fly-by episodes (repeated, ≥ 2) | 70 (45) | 79 (56) | 68 (39) | 2 (1) |
| Orbit episodes | 0 | 1 | 0 | 0 |
| Mean closest approaches; median miss distance | 1.72; 4.2 m | 1.96; 4.8 m | 1.67; 4.0 m | 1.02; 2.2 m |
| First-pass capture (all episodes) | 65.0 % | 60.5 % | 66.0 % | 99 % |
| Stage3-v2 parent: capture / exhaustion | 70.5 / 29.5 % | 71.5 / 28.5 % | 78.0 / 22.0 % | |
| Parent: repeated fly-by episodes; first-pass; closest approaches | 65; 60 %; 2.17 | 63; 52.5 %; 2.10 | 49; 58 %; 1.89 | |
| Failed Stage-4 checkpoint on this task (context only) | 66.5 % | 74.5 % | 80.0 % | |

**Paired statistics** (percentile bootstrap, 10,000 resamples):
| | Stage4A − parent capture | − parent repeated fly-bys | − parent exhaustion | Stage4A − A0 capture | Stage4A − failed Stage 4 (context) |
|---|---:|---:|---:|---:|---:|
| 201 | **+10.5 pp [4.5, 16.5]** | −10.0 [−16.0, −4.0] | −10.5 [−16.5, −4.5] | −19.0 [−24.5, −13.5] | +14.5 [7.0, 22.5] |
| 202 | +2.5 [−2.5, 7.5] | −3.5 [−9.0, 2.0] | −2.5 [−7.5, 2.5] | −26.0 [−32.5, −20.0] | −0.5 [−6.5, 5.5] |
| 203 | +5.5 [−0.5, 11.5] | −5.0 [−11.5, 1.5] | −5.5 [−11.5, 0.5] | −16.5 [−21.5, −11.5] | +3.5 [−2.5, 9.5] |

(A1 is not required to beat A0 here; the A0 differences quantify the remaining expert gap.) The 200-episode gate also fails on capture, exhaustion, >2×, >3×, >5× and repeated fly-bys for every seed. Only seed 201 satisfies the pre-registered
"tracking improved" criterion (CI lower bound > 0 *and* fewer repeated fly-bys than the parent); no seed mastered, so the criterion could not contribute to a pass.

## 8. Moving-target tracking and the asset-input finding (`moving_target_analysis.json`, `asset_input_analysis.json`)
**Tracking.** At every decision we compare the executed direction with the direction to the asset's CURRENT position and to its INITIAL position (an obsolete target). Averaged over all decisions the cosines are close
(current 0.93 / 0.93 / 0.94, initial 0.95 / 0.95 / 0.96) because the asset moves only 2–3 m (median final displacement 2.9 m; 7–8 m in exhaustion episodes). Restricted to the **informative decisions** (≥ 3° apart as seen from the attacker),
the policies aim *closer to the obsolete initial location than to the current asset*: cos(current) − cos(initial) = −0.058 / −0.053 / −0.062, and only **30.8 / 33.8 / 35.4 %** of informative decisions are better aligned with the current
asset (parents 19.4 / 29.7 / 23.5 %; cosine gap −0.067 / −0.071 / −0.077). Training therefore moved tracking toward the current asset (consistent with the fly-by and capture gains) but the policies are still biased toward the origin.
Closing speed 7.4 / 7.0 / 7.5 m/s over all episodes (parents 6.7 / 6.3 / 6.8; 8.7–8.9 m/s in captures), median final-approach heading error to the current asset 19.0 / 18.3 / 14.5° (parents 19.6 / 18.3 / 14.8°).
**Failure mode:** exhaustion episodes (16.5–26 %) last ≈ 58 steps (captures: 12) with a median of 3–4 closest approaches; the asset has drifted a median 7.2–7.7 m (captures: 2.3–2.5 m) and the attacker ends 16–18 m away, having
looped around the target with median miss distances of 4.0–4.8 m (A0: 2.2 m). Figure 5 (scenario 169013): A0 captures on the first pass; the parent makes 3 fly-bys and exhausts its endurance at step 69; Stage4A captures after 2 fly-bys at step 61.

**Why: the parents never learned to use the asset-position input.** In Stages 1–3 the asset was always at the origin, so observation slots 3–5 were constant 0 and received no gradient. Verified in the checkpoints: the first-layer weight
columns of the three asset slots in the Stage3-v2 parents have norm **1.414 = √2, exactly the orthogonal-initialisation value** (never trained), versus 8–14 for the attacker-position columns. The Jacobian of the executed unit direction with
respect to the asset xy slots is only **2.2 / 2.5 / 2.1 %** of that with respect to the attacker xy slots in the parents (a perfect relative tracker would be ≈ 100 % with the opposite sign; an absolute-coordinate aimer at the origin ≈ 0 %).
After Stage 4A the ratio is **10.4 / 9.4 / 9.9 %** (alignment with the relative-tracking direction 0.81 / 0.86 / 0.90; the x/y asset columns grew from 1.41 to 1.7–2.3 while z, always 0, stays at 1.41); after the 16 failed Stage-4 rollouts 14.2 / 14.0 / 13.1 %.
The policies are thus slowly *learning* to track — in the right direction — but 393k steps of on-policy PPO with a sparse 2 m capture signal on a ±3 m effect buy only a fifth to a tenth of the needed sensitivity. (This is a diagnostic of the checkpoints; the causal claim —
that this is why PPO is slow here — is a strongly supported hypothesis, not an intervention result.)

## 9. Retention of evasion
Anchor 3 (the Stage3-v2 moving-defender task, stationary asset) stays above the 0.75 minimum at every rollout (final 0.79 / 0.80 / 0.82 vs parents 0.92 / 0.89 / 0.90) and Anchor 2 above 0.80 (0.85 / 0.93 / 0.90), so moving-target training
does not erase the evasion curriculum, but it costs 8–13 pp on Anchor 3 within 12 rollouts. The full evasion-pattern analysis is a Stage 4B question and was not repeated here (no defender is active in Stage 4A).

## 10. Verdict and implications
Overall rule (≥ 2/3 mastered, moving-target tracking improves, anchors retained, no shared pathology): **not met (0/3)**; the pre-registered "improves, needs manual review" rule (≥ 1 mastered, or ≥ 2 seeds with capture ≥ 0.90 and a paired-CI lower bound
above the parent) is not met either ⇒ **verdict C**. Per the stage rule: **STOP, do not reintroduce the moving defender.** No review IDs (`A1-CUR4A-*`), recordings, `stage4b_handoff.json` or tag were created (required only for a pass / verdict B).
What the evidence says about the next learning step (a new registered version, **not** started or decided here): the observation already carries the information, but the pretrained policies do not use it. Candidates that follow directly from §8 are
(i) imitation of A0 on moving-asset scenarios (the Stage-1-v2 expert-initialisation mechanism, which gives the asset slots non-trivial inputs and an exact relative-pursuit label) or (ii) a stationary-asset stage with *randomised asset positions* so that the asset
slots are trained before the asset moves; both leave the reward unchanged. They are listed so the decision can be taken explicitly, not applied.

## 11. Limitations
1. Short episodes and a small asset drift (median 2.9 m): the task rewards end-game precision rather than long-range tracking; the tracking diagnostics are limited to informative decisions (≥ 3° separation).
2. One frozen PPO configuration and a 12-rollout budget; flat-ish curves are not proof that no PPO setting could solve it.
3. Seed 201's significant gain and seed 202's null gain are within seed-to-seed variation; three seeds.
4. The asset-input finding is observational (weights and Jacobians), not an intervention.
5. Parents are provisional curriculum checkpoints (no manual-review acceptance of Stage3-v2 recorded); Stage 4A needed no manual review because it did not pass.

## 12. Reproduce / verify
```
python -m experiments.journal_v2_a1_cur4a_run resume-test    # stage 6 evidence (real SB3, injected crash)
python -m experiments.journal_v2_a1_cur4a_run all           # stages 1-9 (restartable)
python -m experiments.journal_v2_a1_cur4a_final all         # stages 10-11 (held-out, tracking, retention, decision, figures 1-5)
python -m experiments.journal_v2_a1_cur4a_final asset-input # §8 asset-input analysis
pytest tests/test_journal_v2_a1_cur4a.py -q
```
State: `task_state.json` (15 stages, atomic, hash-verified); scratch `C:\Users\Public\journal_v2_a1_curriculum\stage4a\`; results (untracked) `results/journal_v2_corrected_semantics/a1_curriculum/stage4a/`.
Figures (`figures/`, PNG + PDF): 1 capture vs timesteps, 2 repeated fly-by rate vs timesteps, 3 median path-excess vs timesteps, 4 retention anchors, 5 A0 / Stage3 parent / Stage4A trajectory on the moving asset (seed 203).
