# Curriculum Stage 4 — moving canonical asset + hold-until-detection Greedy defender, C0

**Result: Stage 4 FAILS the pre-registered rule — 0/3 seeds mastered in 16 rollouts (524,288 Stage-4 steps each); verdict C (do not add clutter).** Held-out capture is **52.5 / 60.5 / 59.5 %** (matched canonical A0: 0 %; each seed's own Stage3-v2 parent on the same scenarios: 50.5 / 46.0 / 54.5 %). The curriculum retained everything it had learned (anchors: pursuit 0.99–1.00, static defender 0.94–0.97, moving defender 0.87–0.93; no pathology shared by two seeds), and the Stage-4 gain over the parent is small: **+2 [−6, 10], +14.5 [7.5, 21.5], +5 [−2, 12] pp** (only seed 202 excludes 0). The binding failure is **endurance exhaustion after repeated near-miss fly-bys (17–23 %)**, which Stage-4 training did not reduce.

## 1. Provenance
* Stage3-v2 passed quantitatively (3/3, held-out 83.5 / 82.5 / 86.5 % on a **stationary** asset). No manual-review acceptance is recorded in the repository, so Stage3-v2 is used as a *quantitatively accepted curriculum parent* (`parent_status = quantitatively_accepted_curriculum_parent`); earlier mastery results are untouched and the checkpoints remain curriculum models, not final A1.
* Parents (full SHA-256, verified against the pre-registered values, the Stage3-v2 seed manifests, every Stage3-v2 rollout file and the committed docs manifests): 201 `c60ae85d0c0b84c2b52dd4d0c280ccede9f6ecae6166448bd3a9de8218988e78`; 202 `3dfabf5cf245bc07aa02ac67840792c9a422724a7784be2efe07d196389ba485`; 203 `818338a5579e1e63259dff8fb94a61f2518a2e61db9901e158703a177951b52c`.

## 2. The one change: canonical V2 asset motion
Stage 4 = Stage3-v2 with the stationary asset replaced by the canonical asset-motion model (config `asset_motion_model = "legacy_gaussian_random_walk"`, `v_s = 1.5 m/s`; read from live code `SoldierEnv._move_soldier_legacy`; **only `defender_standby_until_detection` differs from the canonical V2 config**, as in Stage3-v2):
* each step the asset draws an i.i.d. horizontal Gaussian displacement, σ = v_s·dt = 1.5 × 0.5 = **0.75 m per axis**, z = 0, reflecting boundaries at [−50, 50]², start at the origin; **no velocity persistence**; obstacle blocking inert in C0; the draw comes from the dedicated `_rng_soldier` stream (`SeedSequence(seed).spawn(4)[1]`);
* no correlated motion, no retuned speed/noise, no easier curriculum walk; the asset moves from step 1 under normal canonical timing.
Kept unchanged: C0, hold-until-detection Greedy (fixed/zero velocity until `env._enemy_detected` at 15 m, released the next step, never re-held — **scaffold not removed**), canonical attacker dynamics, `a1_v2_obs_v1` (37-D), `a1_cur1_action_v1`, `a1_v2_reward_v2`, frozen PPO (2 torch threads).

### Observation audit
The 37-D observation contains the **current** asset position (verified by test: obs slots 3–5 = asset position / L at every step). The canonical walk is memoryless, so the current position is the complete Markov state: an asset-velocity feature would carry no predictive information, and no future information is exposed. No augmentation was needed; the stop-clause was not triggered.

## 3. Placement and common random numbers
* Placement is the unchanged Stage3-v2 / Stage-2 procedure computed from the **initial** state with a **stationary-asset** nominal-A0 path (λ ~ U(0.35, 0.65), r ~ U(0, 1.5 m)); it is identical to Stage3-v2's placement for the same scenario seed (test), so it cannot depend on future asset motion.
* **CRN:** the asset stream is independent of the attacker, defender and sensor streams, so on the same evaluation seed A0, the Stage3 parent and the Stage-4 policy see the same initial attacker state, asset state, defender placement and realized asset trajectory. Verified (tests and reports): maximum difference **0.0 m** over all completed steps (792 asset positions in the 1,000-scenario check; 2,046–2,048 per seed in the final evaluation). Only the final element of an episode's trace — the simulator's swept-contact event state of the terminal step — legitimately differs (≤ 1.7 m) and is excluded from the exact comparison and reported separately.

## 4. Pre-training validation (1,000 scenarios `168000–168999`)
| | capture | interception | exhaustion | timeout | median path | asset max displacement (median) | median min defender separation |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical A0 | 0 % | 100 % | 0 % | 0 % | 24.7 m | 2.07 m | 2.5 m |
| Stage3-v2 parent 201 | 53.8 % | 24.6 % | 21.6 % | 0 % | 68.7 m | 3.72 m | 5.02 m |
| parent 202 | 54.6 % | 25.7 % | 19.7 % | 0 % | 69.7 m | 3.79 m | 5.06 m |
| parent 203 | 58.3 % | 19.5 % | 22.2 % | 0 % | 69.7 m | 3.87 m | 5.17 m |

Rules passed: A0 interception ≥ 50 % (100 %); at least one parent captures > 20 % (54–58 %); hold/release semantics, no degeneracy, asset moves, CRN identical, deterministic replay. **Already visible before training: a ~2–4 m asset drift cut the parents' capture from 83 % to ~55 % and created 20 % endurance exhaustion that does not exist with a stationary asset.**

## 5. Training (frozen PPO, fresh optimizer, parent actor + critic + log_std)
All three seeds ran the full 16 rollouts without a retention stop. Validation capture (alternating sets of 100): 201 mean 0.49 (rollouts 1–8) → 0.52 (9–16), best 0.59; 202 0.485 → 0.564, best 0.67; 203 0.544 → 0.574, best 0.63 — never two consecutive passes (every rollout fails capture ≥ 0.70, exhaustion ≤ 0.03 and the >3× / >5× path-excess limits; interception ≤ 0.30 fails intermittently for 202). Training-episode capture moved only from 0.49–0.52 (rollouts 1–3) to 0.54–0.56 (14–16). Per-rollout tables: `rollout_results.md`.

| Seed | status | final (rollout 16) checkpoint SHA-256 (not accepted) |
|---|---|---|
| 201 | FAIL (16 rollouts) | `e9f6f0c0c940101fee118ed8de83fcafaab9dc9e14025e706a0d57850f5d5a44` |
| 202 | FAIL (16 rollouts) | `e768197ab908c39f16b92ac740a9ac57a1d1f868b40b89be5c246fb3cd550bb1` |
| 203 | FAIL (16 rollouts) | `133302d45593c73040f4c821f10cb2c69f3b99e2a94facdb43cd1c1d77a5ffab` |

## 6. Held-out evaluation (`165000–165199`, 200 matched episodes, identical asset trajectories)
| | Stage4 201 | Stage4 202 | Stage4 203 | A0 |
|---|---:|---:|---:|---:|
| Capture | 52.5 % | 60.5 % | 59.5 % | 0 % |
| Interception | 24.0 % | 22.5 % | 18.0 % | 100 % |
| Endurance exhaustion | 23.5 % | 17.0 % | 22.5 % | 0 % |
| Timeout | 0 | 0 | 0 | 0 |
| Median path-excess / >3× / >5× | 1.11 / 30 % / 11.5 % | 1.10 / 27 % / 8.5 % | 1.13 / 31 % / 12 % | 1.02 |
| Mean duration / path length | 13.4 s / 130 m | 12.1 s / 121 m | 13.4 s / 138 m | 2.6 s / 25 m |
| Median min defender separation | 4.96 m | 5.22 m | 5.45 m | 2.5 m |
| Fly-by / non-closing / orbit episodes | 57 / 68 / 2 | 40 / 65 / 0 | 54 / 77 / 0 | – |
| Mean max asset displacement | 5.6 m | 5.4 m | 5.7 m | – |
| Anchors 1 / 2 / 3 | 0.99 / 0.94 / 0.87 | 1.00 / 0.96 / 0.93 | 1.00 / 0.97 / 0.92 | |
| Stage3 parent: capture / interception / exhaustion | 50.5 / 26.0 / 23.5 % | 46.0 / 36.5 / 17.5 % | 54.5 / 22.5 / 23.0 % | |
| Parent anchors 1 / 2 / 3 | 1.00 / 0.92 / 0.87 | 1.00 / 0.94 / 0.84 | 1.00 / 0.96 / 0.87 | |

**Paired statistics** (percentile bootstrap, 10,000 resamples, seed 20260201):

| | Stage4 − A0 capture | Stage4 − Stage3 parent capture | − parent interception | − parent exhaustion |
|---|---:|---:|---:|---:|
| 201 | +52.5 pp [45.5, 59.5] | +2.0 [−6.0, 10.0] (34 / 30 discordant) | −2.0 [−8.0, 4.0] | 0.0 [−6.5, 7.0] |
| 202 | +60.5 pp [53.5, 67.5] | **+14.5 [7.5, 21.5]** (41 / 12) | **−14.0 [−20.0, −8.0]** | −0.5 [−6.5, 5.5] |
| 203 | +59.5 pp [52.5, 66.5] | +5.0 [−2.0, 12.0] (31 / 21) | −4.5 [−11.0, 2.0] | −0.5 [−7.0, 6.0] |

So the moving-target curriculum bought a measurable improvement only for seed 202 (mostly fewer interceptions); it did not reduce exhaustion for any seed.

## 7. Failure analysis
* Failure modes of the Stage-4 policies: interception 18–24 %, **exhaustion 17–23 %**. Exhaustion episodes last ~60 steps (captures ~14), end 17–19 m from the asset, with median **3 overshoots** and the asset having drifted a median 8–10 m (long episodes accumulate random walk; captures: 2.7 m). Tracing the parent shows the attacker repeatedly passes the asset at a miss distance of ≈ 3.4 m (> 2 m capture radius) and re-approaches until the endurance budget (k = 5) is gone: an **end-game aiming-precision failure**, not a gross loss of pursuit (closing speed 9.1–9.3 m/s in captures; anchors intact).
* The learned policies were optimised for an asset at exactly the origin; a 0.75 m/axis/step jitter (≈ 1–4 m over an episode, i.e. 1–4° of aim at 50 m) pushes the last-pass miss distance outside the 2 m radius in about a fifth of the episodes, and 16 rollouts of PPO under the frozen settings did not fix it (capture ~0.5 → ~0.55).

## 8. Moving-target pursuit analysis (captures; `moving_target_analysis.json`)
Successful episodes (105 / 121 / 119): median asset displacement 2.5–2.7 m; **first-pass capture 84 % / 74 % / 77 %** (parents 79 / 73 / 76 %); mean closing speed ≈ 9.1–9.3 m/s. The question "is A1 tracking the current asset or an obsolete location?" **cannot be answered decisively**: over the last three decisions the executed direction is closer to the current asset than to the initial asset location in only 46 / 31 / 30 % of captures (parents 14 / 23 / 24 %), mean last-approach heading errors are 21–31°, and — decisively for the interpretation — the asset moves a median 2.7 m, i.e. the "current" and "obsolete" aim directions differ by only ~3° at 50 m, below the resolution of this diagnostic. We therefore report **no evidence of explicit tracking** and no evidence against it. Target-motion direction vs evasion: cosine between the asset's drift and the evasive deviation ≈ 0 (0.02 / 0.00 / 0.06; 48–54 % same side): evasion is independent of target motion.

## 9. Defender-evasion retention (captures vs the same policy flown without the defender, same asset path)
The pattern **pursuit → detection → evasion → reacquisition → capture** persists: no deviation before the defender enters the 25 m gate (0 %); median 0.5 s from detection to a > 1 m deviation; closest defender separation median 5.7 / 5.8 / 5.9 m (A0: 2.5 m); post-encounter cosine toward the (moving) asset 0.88 / 0.92 / 0.94; reacquisition median 0.5 / 0.5 / 1.5 s. Classes: lateral / mixed / timing-turn dominate (e.g. 202: lateral 30, mixed 53, timing 28, vertical 2, unclear 8).

## 10. Generalization diagnostic (not a gate; moving asset, same hold scaffold; `167000–167099`)
Capture of the Stage-4 policy (Stage3 parent) — matched A0 is 0 % everywhere:

| Seed | Greedy | Kalman-Greedy | PN | Lead |
|---|---:|---:|---:|---:|
| 201 | 66 (53) | 65 (52) | 57 (45) | 55 (49) |
| 202 | 64 (52) | 61 (53) | 62 (51) | 56 (49) |
| 203 | 70 (62) | 72 (63) | 68 (55) | 60 (50) |

Stage-4 beats its parent against every controller by 6–13 pp on this block (a slightly larger gain than on the Greedy held-out block, partly sampling noise at n = 100), and the other controllers are no harder than Greedy: the limitation is the moving target, not the interceptor type.

## 11. Verdict and what it means
Overall rule (≥ 2/3 mastered, anchors retained, no shared pathology): **not met (0/3)**. The pre-registered "promising" rule for verdict B (≥ 1 mastered, or ≥ 2 seeds with capture ≥ 0.60 and a paired-CI lower bound above the parent) is also not met (only seed 202). Per the stage rule: **STOP, do not add clutter**, and no review IDs / handoff were created (the visualizer package is required only for a pass or verdict B). Not started and not decided here: any change to budget, asset motion, placement, objective or the scaffold — each would be a new registered version.

## 12. Limitations
1. Canonical asset motion is small over these short episodes (median 2–4 m drift); the stage mostly probes **end-game precision under small target jitter**, not long-range target tracking, and the tracking diagnostic is weak for the same reason.
2. The hold-until-detection scaffold is retained (defender speed ≤ 8 m/s at the encounter); results are not about sustained fast pursuit.
3. One frozen PPO configuration and a 16-rollout budget; flat curves do not prove the task is unlearnable under other settings.
4. Parents are provisional curriculum checkpoints (no manual review recorded).
5. Held-out CIs are paired but based on 200 scenarios × 3 seeds; seed-to-seed differences (e.g. 202) are within what a different training budget could change.

## 13. Reproduce / verify
```
python -m experiments.journal_v2_a1_cur4_run resume-test    # stage 6 evidence (real SB3, injected crash)
python -m experiments.journal_v2_a1_cur4_run all            # stages 1-9 (restartable)
python -m experiments.journal_v2_a1_cur4_final all          # stages 10-11
pytest tests/test_journal_v2_a1_cur4.py -q
```
State: `task_state.json` (15 stages, atomic, hash-verified); scratch `C:\Users\Public\journal_v2_a1_curriculum\stage4\`; results (untracked) `results/journal_v2_corrected_semantics/a1_curriculum/stage4/`. Protocol SHA-256 in `protocol.json`: `5a1a9254a350a9204ea02a9a3e6aa6d0db98ed07e8bdd2fd20732cf740cfaa26`.
