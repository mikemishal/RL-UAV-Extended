# Curriculum Stage 2 — evade a STATIONARY, ACTIVE defender while reaching a stationary asset (C0)

**Result: Stage 2 FAILS the pre-registered mastery rule (0/3 seeds mastered within the 8-rollout / 262,144-step budget).** The evidence is nevertheless unambiguous that the BC-initialized policies *learned* fixed-threat evasion: capture rose from ≈ 1–2% to 55–67% on held-out scenarios where canonical A0 captures **0%**, with perfect retention of Stage-1 pursuit and no pathologies, and the learning curves were still rising when the budget ended. No parameters were tuned, the budget was not extended, and Stage 3 was not started. These are curriculum checkpoints, not final A1 models, and none of the three is accepted as a Stage-2 checkpoint.

## Pre-registered hypothesis

> After expert initialization establishes reliable goal-directed pursuit, PPO can learn evasive deviations around a stationary interception threat without destroying the learned pursuit skill.

Tests *pursuit preservation + elementary adversarial avoidance*. Outcome: **pursuit preservation confirmed; elementary avoidance learned but not to the pre-registered 90% mastery level within budget.**

## Environment (`experiments/journal_v2_a1_cur2.py`)

| Item | Definition |
|---|---|
| Scenario | C0, canonical V2 semantics (swept contact, endurance k=5), no static/dynamic obstacles; configuration identical to Stage-1-v1 (`v_s = 0`, passive-harness defender without standby synchronisation) |
| Asset | stationary at the canonical origin; initial attacker geometry/heading = successful Stage-1 distribution |
| Defender | **stationary, zero velocity, never acts, fully ACTIVE**: canonical swept interception with the unchanged `intercept_radius = 2.5 m`; contact terminates the episode |
| Observation | `a1_v2_obs_v1` (37-D); canonical defender gate restored: defender flag/position/velocity are non-zero only if the TRUE 3-D attacker–defender range ≤ `enemy_defender_detection_range` = 25 m (verified in `build_attacker_observation`) |
| Action | `a1_cur1_action_v1` (continuous desired direction, `Box(-1,1,(3,))`), no A0 fallback, no residual/route/pulse wrapper |
| Reward | `a1_v2_reward_v2`, unchanged, no Stage-2 terms |

### Defender placement (user-approved amendment, documented)

The specification placed the defender on the initial attacker→asset segment at fraction λ ~ U(0.35, 0.65) with perpendicular offset r ~ U(0, 1.5 m). Scenario validation of that **chord** design (1,000 scenarios) gave A0 interception **51.5% < 70%** (stop rule). The placement code was reviewed and is correct; the cause is the Stage-1 random initial heading (up to 90° off the line of sight): A0 curves away from the initial chord, so A0 intercepts 100% / 98% / 75% / 27% / 9% / 1% of scenarios for heading offsets 0–15 / 15–30 / 30–45 / 45–60 / 60–75 / 75–90°. Evidence: `scenario_validation_initial_STOP.json` (no training had started). After the stop, the user approved **Option A**: place the defender relative to A0's own **noise-free nominal path** (simulated without interception from the same initial state), keeping λ ~ U(0.35, 0.65) (now along the path's arc length), r ~ U(0, 1.5 m), uniform perpendicular orientation about the local tangent, the arena/altitude validity rule (re-sample, never clip) and the interception radius. Placement is a deterministic function of the scenario seed (`SeedSequence([0xC2A3, seed])`). The superseded design is kept (`place_defender_on_chord`) and tested only to reproduce the failure.

### A0 before training (1,000 scenarios 149000–149999, `scenario_validation.json`)

* **A0 intercepted 100.0%** (1000/1000), captures 0%, no resampled placement (max 1 draw); deterministic replay verified.
* Attacker–asset distance 51.2–76.6 m (median 60.4); λ 0.350–0.650 (median 0.500); perpendicular offset 0.0–1.50 m (median 0.81); defender altitude 2.5–24.9 m (median 10.6); defender within the 25 m gate at episode start in 24.3% of scenarios.
* Time between the defender entering the 25 m information gate and nominal interception: **median 2.0 s (1.5–3.5 s, i.e. 3–7 decisions)** — physically feasible (a course change of a few degrees suffices to clear 2.5 m) but demanding.

## Initialization (`journal_v2_a1_cur2_training.initialize_model`)

For PPO seed *s*: build `PPO(seed=s)` (random actor **and** critic from the PPO seed), then overwrite **only** `mlp_extractor.policy_net`, `action_net` and `log_std` (= −2.0, σ = 0.135) from the matching Stage-1 BC checkpoint (A1-BC1-201/202/203, hashes verified against `stage2_handoff.json`). The critic keeps its PPO-seed initialization, the optimizer is the fresh Adam from PPO construction (no BC optimizer state), nothing from failed A1-v1 or Stage-1-v1 PPO models is loaded. Recorded per seed in `init_record.json` (actor / critic hashes).

## PPO protocol (frozen, `protocol.json`)

MlpPolicy [64,64] tanh; lr 3e-4, γ 0.99, GAE 0.95, n_steps 2048, batch 64, 10 epochs, clip 0.2, ent 0.01, vf 0.5, grad-norm 0.5, 16 envs; one rollout = 32,768 steps, max 8 rollouts (262,144). Validation after every rollout, alternating set A (147000–147099, odd rollouts) and B (147100–147199, even rollouts), deterministic policy means, **plus** matched canonical A0 on the same set and the Stage-1 no-defender pursuit **anchor** (148000–148099, never used for BC). Mastery = two consecutive validation passes of: capture ≥ 90%, ≥ +20 pp over matched A0, timeout ≤ 2%, exhaustion ≤ 2%, median path-excess ≤ 1.60×, >3× ≤ 2%, >5× = 0, orbit 0, anchor capture ≥ 98%, no numerical failures. Pursuit-retention failure = anchor < 98% on two consecutive rollouts. Power-loss safety: per-rollout atomic commit (model + optimizer, RNG, episode log, manifest renamed LAST), local scratch `C:\Users\Public\journal_v2_a1_curriculum\stage2\`, OneDrive mirror, per-seed lock.

### Real-SB3 resume validation (`resume_validation.json`)

Production code path with BC initialization, SubprocVecEnv, injected abrupt death between model write and manifest, process-level destruction and reload from disk: policy+optimizer state hashes and episode logs of all 3 rollouts were **identical** to the uninterrupted run; the dead-PID lock was taken over. Passed before the campaign started.

## Training results (`rollout_results.md`, `seed_*_manifest.json`, `figures/`)

| Seed | rollouts | timesteps | final validation capture | final val interception | anchor capture (all rollouts) | status |
|---|---|---|---|---|---|---|
| 201 | 8 | 262,144 | 0.73 (set B) | 0.27 | 1.00 | FAIL (no mastery) |
| 202 | 8 | 262,144 | 0.65 (set B) | 0.35 | 0.99–1.00 | FAIL (no mastery) |
| 203 | 8 | 262,144 | 0.59 (set B) | 0.41 | 1.00 | FAIL (no mastery) |

Capture rose monotonically (noisily) from 0.01–0.02 at rollout 1 to 0.59–0.73 at rollout 8 (Fig. 1) with interception falling correspondingly (Fig. 2). The only gate check that failed on the final policies is **capture ≥ 90%**; the improvement over A0 (+55…+67 pp), path-excess (≈ 1.02–1.03), exhaustion/timeout (0), orbit and anchor checks all pass. No seed ever passed a validation, so no two-consecutive mastery was reached.

## Final evaluation — 200 matched scenarios (147000–147199), accepted = rollout-8 policy (`final_evaluation.json`)

| | Capture | Interception | Exhaustion/timeout | Mean duration | Mean path length | Median / mean path-excess | Median min. defender separation | Defender-visible time |
|---|---|---|---|---|---|---|---|---|
| **Canonical A0** | 0.0% | **100%** | 0 | 2.78 s | 28.5 m | 1.019 / 1.035 | 2.50 m | 2.42 s |
| **A1-CUR2-201** | **67.0%** | 33.0% | 0 | 4.81 s | 51.1 m | 1.029 / 1.044 | 3.66 m | 3.80 s |
| **A1-CUR2-202** | **58.5%** | 41.5% | 0 | 4.52 s | 48.1 m | 1.023 / 1.040 | 3.20 m | 3.62 s |
| **A1-CUR2-203** | **55.5%** | 44.5% | 0 | 4.45 s | 47.7 m | 1.026 / 1.040 | 2.96 m | 3.54 s |

All three: 0 fly-by episodes, 0 repeated fly-bys, 0 orbits, 0 non-closing episodes, 0 numerical failures, >3× path 0%. (Path length/duration of A1 are larger than A0's only because A0 terminates early by interception; A1 episodes that are captured fly the whole mission.)

### Paired A1 − A0 (matched scenarios, paired bootstrap 95% CI, 10,000 resamples)

| Seed | Capture difference | 95% CI | Interception difference | 95% CI | A1-only / A0-only successes |
|---|---|---|---|---|---|
| 201 | **+67.0 pp** | [+60.5, +73.5] | −67.0 pp | [−73.5, −60.5] | 134 / 0 |
| 202 | **+58.5 pp** | [+51.5, +65.5] | −58.5 pp | [−65.5, −51.5] | 117 / 0 |
| 203 | **+55.5 pp** | [+48.5, +62.5] | −55.5 pp | [−62.5, −48.5] | 111 / 0 |

This is the first curriculum stage in which A1 beats A0, and A1 never loses a scenario A0 wins (A0 wins none here). Because A0 captures 0% by construction of the placement, the A0 comparison establishes that the defender is a real obstacle, not that A1 is near-optimal.

## Learned-evasion analysis (`evasion_analysis.json`)

Successful A1 episodes (n = 134 / 117 / 111):

* **Causal reactivity.** Replaying the *same* final policy on the *same* 200 scenarios without the defender (Stage-1 masked env), the paired trajectories are **identical until the defender enters the 25 m gate in 100% of episodes** (0% diverge earlier — the observation is identical before gate entry), and diverge afterwards: 74% / 88% / 83% of episodes are displaced by > 1 m (median max displacement 1.7 / 1.7 / 1.8 m; mean 3.3 m in episodes that still got intercepted vs ≈ 1.5 m in captured ones: the typical failure is a reaction that is present but insufficient or misdirected).
* **Geometry.** Max lateral deviation from the initial chord: median 5.4 / 4.9 / 5.5 m (p95 11.7 / 13.6 / 10.0 m); vertical deviation median 3.0 / 3.0 / 2.5 m; minimum defender distance median 4.5 / 4.0 / 4.2 m (A0: exactly 2.5 m by contact). Side chosen at closest approach (left/right/over/under): 201: 23/56/33/22, 202: 18/47/38/14, 203: 36/52/17/6 — a mix of horizontal and vertical detours with a consistent right-hand horizontal preference.
* **Pursuit → evasion → pursuit.** After the closest approach the executed direction re-aligns with the line of sight (mean cosine 0.972 / 0.987 / 0.989); the resume criterion (mean cosine ≥ 0.95) holds in 87% / 100% / 100% of captured episodes, and the strict three-phase pattern (≥ 1 pursuit step before, deviating steps while the defender is visible, pursuit afterwards, capture) in 37% / 69% / 55% (not decomposed further; for the ≈ 24% of scenarios that begin inside the gate there is no pre-encounter pursuit phase by construction). Path-excess of captured episodes stays ≈ 1.03 (median 1.02–1.03, max 1.18–1.27): detours are small and efficient, not wandering.
* **Caveat on "first divergence from the BC direction".** 65–85% of captured episodes exceed a 10° departure from the BC direction at some step, but only 42% / 69% / 60% of those first departures occur while the defender is visible; the rest are the small general drift (cosine ≈ 0.99) PPO adds to the BC policy even without a defender (see below). The with/without-defender replay above is the clean measure of the reaction to the defender.

## BC-fidelity / forgetting (`bc_fidelity.json`; no-defender C0, Stage-1 held-out 146000–146199)

| Seed | BC capture → final | BC median/mean path-excess → final | BC → final mean duration | action cosine final-vs-BC (mean / median / p5 / min) | cosine to A0 label (mean / min) |
|---|---|---|---|---|---|
| 201 | 100% → 100% | 1.024/1.043 → 1.032/1.052 | 5.647 → 5.775 s | 0.989 / 0.990 / 0.971 / 0.941 | 1.000/0.990 → 0.986/0.858 |
| 202 | 100% → 100% | 1.024/1.043 → 1.029/1.050 | 5.645 → 5.772 s | 0.992 / 0.993 / 0.980 / 0.956 | 1.000/0.997 → 0.991/0.932 |
| 203 | 100% → 100% | 1.024/1.043 → 1.028/1.049 | 5.645 → 5.718 s | 0.993 / 0.995 / 0.979 / 0.952 | 1.000/0.995 → 0.991/0.840 |

Pursuit is **fully retained** (anchor capture 1.00 at the final rollout for all seeds; ≥ 0.99 at every rollout) with a ≈ 1–2% longer path in the no-defender task. Fine-tuning moved the actions by ≈ 5–8° on average, i.e. adversarial learning did not destroy pursuit. No retention failure occurred.

## Verdict

* Mastered seeds: **0/3**; Stage-2 rule (≥ 2/3 seeds master, retain pursuit, no shared pathology): **FAIL**.
* Retention: all three final policies retain Stage-1 pursuit (anchor 1.00); no shared structural pathology.
* Interpretation (hypothesis, not tested here): the training budget, rather than the observation or action interface, appears to be limiting — validation capture was still increasing at 262,144 steps (Fig. 1) and failures are reactions that are insufficient or misdirected rather than absent. A longer pre-registered budget would be a **new registered Stage-2 version**; nothing was changed in this run.

## Not started / not done (per instructions)

Stage 3 (moving defender), obstacles, moving asset, defender training, any change to canonical A0, any PPO tuning or budget extension, visualizer registration of `A1-CUR2-*` (required only if ≥ 2 seeds master) and a Stage-3 handoff. The research-only tag `journal-v2-a1-curriculum-stage2-v1` is **not** created.

## Files

`protocol.json`, `scenario_validation.json` (final design), `scenario_validation_initial_STOP.json` (superseded chord design), `resume_validation.json`, `repository_verification.json`, `checkpoint_integrity.json`, `seed_{201,202,203}_manifest.json`, `rollout_results.md`, `final_evaluation.json`, `evasion_analysis.json`, `bc_fidelity.json`, `stage2_decision.json`, `unit_test_result.json`, `regression_summary.json`, `task_state.json`, `figures/` (PNG + PDF), `manuscript_notes.md`, `ablation_record_v7.md`.

| Seed | accepted/final model (`a1_curriculum/stage2/seed_*/checkpoints/accepted_model.zip`, rollout 8) SHA-256 |
|---|---|
| 201 | `be8b85e4d9c30b3808215e5346e03718726d883c22aab6518ea5581aa676fbeb` |
| 202 | `25e67e794a44b392c0170f774096216ae44d0f84c9fd9e29d547a8d595001361` |
| 203 | `6472fbee620226379588871bdfb33c1b85edae82213170e68b208b16b4e472fa` |

(Models live under `results/` which is not tracked by git; hashes are recorded here and in the seed manifests. These are NOT accepted checkpoints.)
