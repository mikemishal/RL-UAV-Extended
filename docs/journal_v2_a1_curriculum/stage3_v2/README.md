# Curriculum Stage3-v2 — evade a MOVING Greedy defender (held until detection), stationary asset, C0

**Result: Stage3-v2 PASSES the pre-registered rule — 3/3 seeds mastered (rollouts 3 / 5 / 7), all gates also hold on the 200 held-out scenarios.** Held-out capture is **83.5 / 82.5 / 86.5 %** (matched canonical A0: 0 %), i.e. **+10.5 / +7.5 / +14.5 pp over each seed's own Stage2-v2 parent** on the same scenarios (paired 95 % CIs exclude 0). Both retention anchors hold (Stage-1 pursuit 1.00, Stage-2 static defender 0.90 / 0.94 / 0.95). No pathologies. The checkpoints remain **provisional curriculum checkpoints, not final A1 models**; manual inspection is the next step, then Stage 4.

This is a **curriculum scaffold**, not the final defender benchmark: the defender is held stationary until canonical detection (see §3 and §11).

## 1. Provenance — what is carried over, what is not rewritten

* **Stage2-v2 did NOT satisfy its frozen mastery rule (0/3, verdict B).** That result is preserved exactly (`docs/journal_v2_a1_curriculum/stage2_v2/`). After manual inspection the rollout-16 checkpoints were judged qualitatively acceptable for curriculum transfer ("approaches the asset, reacts to the defender, deviates to avoid interception, resumes mission pursuit") and are used **only** as `parent_status = provisional_curriculum_transfer` initialisation checkpoints. They are not accepted A1 models and Stage2-v2 is not rewritten as a pass.
* Parents (full SHA-256, verified against the Stage2-v2 seed manifests, the rollout-16 entries and the committed docs manifests): 201 `d70b85a6a4d068646da1319f3f44b3057dce1e5fedd1f66b28c85b4cbd55f90c`; 202 `97a934bc2d115c4eec1a3b838fd754fd2dba93f57d91d84203aed9d7efb4d174`; 203 `06993dca67b59876c65afa956bcdace905482745b58cf39c9da87d713a578cac`.

## 2. Stage3-v1 design failure (kept as evidence, never mixed)

Stage3-v1 used the canonical Greedy controller from step 0 on the Stage-2 nominal-A0-path placement. Greedy **escorts the asset until it detects the attacker (15 m)**; the defender placed between the attacker and the asset therefore accelerated **away** from the approaching attacker and carried 8–12 m/s of wrong-direction velocity at detection. Canonical A0 was intercepted in **0/1000** scenarios (pre-registered stop rule: ≥ 60 %), so the task could not teach evasion. It was **not a stepping bug** (traced episodes show Greedy pursuing the measurement correctly after detection). Evidence: `docs/journal_v2_a1_curriculum/stage3/scenario_validation.json`, `.../stage3/task_state.json` (marked superseded); code kept verbatim in `experiments/journal_v2_a1_cur3.py`. No PPO was trained under Stage3-v1.

## 3. Stage3-v2: the one conceptual change

> Defender motion is suppressed before canonical detection; after detection the existing Greedy controller is released unchanged.

* `experiments/journal_v2_a1_cur3v2.py::Stage3V2Env` (curriculum wrapper; `uav_defend`, Greedy and A0 untouched, verified by `git diff` against the A0 tag and the Stage2-v2 commit).
* Before release: defender position exactly fixed, velocity = acceleration = 0, the Greedy action is **not applied** (`_move_defender` is gated).
* Detection = the canonical environment criterion `env._enemy_detected` (true 3-D attacker–defender distance ≤ `config.detection_radius` = **15.0 m**, verified from live code `SoldierEnv._update_detection` and the config; sticky). The step in which detection becomes true is still held (the canonical standby ordering: first executed controller action = next step); from then on the unmodified Greedy action drives the unmodified constrained dynamics (18 m/s, 8 m/s², 90 °/s) and **the defender never returns to hold**. No detection delay, gains or limits changed.
* Placement: the unchanged Stage2-v2 nominal-A0-path method (λ ~ U(0.35, 0.65), r ~ U(0, 1.5 m)); never resampled on outcome. The defender starts at rest.
* Everything else frozen: C0, stationary asset, `a1_v2_obs_v1` (37-D), `a1_cur1_action_v1`, `a1_v2_reward_v2`, PPO hyper-parameters (MlpPolicy [64,64] tanh, lr 3e-4, γ 0.99, λ 0.95, 2048 × 16 envs, batch 64, 10 epochs, clip 0.2, ent 0.01, vf 0.5, grad-norm 0.5).

### Observation audit (hidden state)
The release flag, release time, λ and the nominal path are **not** observed (37-D unchanged). This creates no harmful hidden state: the defender is invisible (zero features) outside the 25 m gate and stationary until release; release occurs at 15 m, i.e. while the defender is already inside the gate, and its onset is then inferable from the observed defender-velocity feature. The only unobserved quantity is the pre-release dwell, which carries no information the attacker could exploit. Decision: keep 37-D.

## 4. Initialisation (pre-resolved)
Per seed: fresh `PPO(seed)`, then the **whole policy state_dict** of the matching Stage2-v2 rollout-16 model (actor, **critic** and `log_std`; observation/reward/value semantics are unchanged) is loaded; the **optimizer is reset** (fresh Adam, empty state) at the curriculum boundary and the timestep counter is 0 (Stage-3 rollouts 1..16, max 16 × 32,768 = 524,288 steps). Verified by test and `init_record.json`.

## 5. Namespaces
Validation A `154000–154099`, B `154100–154199` (alternating, odd rollout → A), Stage-1 pursuit anchor `155000–155099`, Stage-2 static-defender anchor `155100–155199`, generalization diagnostic `156000–156099`, scenario validation `157000–157999`. **The suggested 149000–149199 was rejected after the reservation audit**: it overlaps Stage 2's own placement-validation block (149000–149999). The fresh blocks overlap no reserved range, no Stage-3-v1 block and no training stream (`namespace_overlaps()` in the protocol). Training scenario seeds come from a new stream tag (`0xC3B1`, [1,000,000, 2³¹−1)).

## 6. Pre-training scenario validation (1,000 scenarios, canonical A0 vs held-until-detection Greedy)
A0 interception **100 %** (required 70–100 %), A0 capture 0 %, timeout 0 %; episodes 1.5–4.5 s; detection time median 1.5 s; release → interception median 1.0 s; defender speed at release **0** (max over all episodes), displacement before release **0.0 m**, Greedy moved the defender after release in 1000/1000 episodes; no non-finite state, no episode ending within 1 step of reset, ≥ 3 policy decisions with the defender inside the 25 m gate before every interception; interception 100 % in all six heading bins; replay deterministic (first 50). The immediate-contact criterion was fixed before PPO and examined only on the A0 baseline (interceptions ≤ 1 step after release occur in 15.8 % — inherent to a defender already on the path — and are reported informationally).

| Metric (A0, 1000 scenarios `157000–157999`) | Stage3-v1 Escort | Stage3-v2 Hold |
|---|---:|---:|
| A0 interception | 0 % | 100 % |
| A0 capture | 100 % | 0 % |
| Defender speed at detection (median / p95) | 8.0 / 12.0 m/s | 0 / 0 m/s |
| Defender velocity toward attacker at detection (median) | −8.0 m/s (away) | 0 |
| Median min defender separation | 24.9 m | 2.5 m (contact) |
| Median episode duration | 5.5 s | 2.5 s |

(`baseline_comparison.json`; both columns re-run through the same code path on the same scenarios.)

## 7. Training (provisional parents → Stage3-v2), all seeds mastered
| Seed | Mastery rollout | Stage-3 timesteps | Accepted checkpoint SHA-256 |
|---|---:|---:|---|
| 201 | 3 | 98,304 | `c60ae85d0c0b84c2b52dd4d0c280ccede9f6ecae6166448bd3a9de8218988e78` |
| 202 | 5 | 163,840 | `3dfabf5cf245bc07aa02ac67840792c9a422724a7784be2efe07d196389ba485` |
| 203 | 7 | 229,376 | `818338a5579e1e63259dff8fb94a61f2518a2e61db9901e158703a177951b52c` |

Per-rollout validation (capture / interception / Stage-1 anchor / static anchor / gate) is in `rollout_results.md` and the seed manifests. Retention never triggered (Stage-1 anchor 1.00 at every rollout; static anchor ≥ 0.84 everywhere, ≥ 0.90 at mastery). Mastery is marginal-by-design at the 0.80 gate (e.g. 201: 0.82, 0.81), which is why the 200-episode confirmation matters.

## 8. Held-out evaluation (`154000–154199`, 200 matched episodes)
| | Seed 201 | Seed 202 | Seed 203 | A0 (matched) |
|---|---:|---:|---:|---:|
| Capture | **83.5 %** | **82.5 %** | **86.5 %** | 0 % |
| Interception | 16.5 % | 17.5 % | 13.5 % | 100 % |
| Timeout / endurance | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| Median / >3× / >5× path-excess | 1.05 / 0 / 0 | 1.055 / 0 / 0 | 1.051 / 0 / 0 | 1.023 |
| Median min defender separation | 5.67 m | 5.64 m | 5.52 m | 2.5 m |
| Mean duration / path length | 5.62 s / 57.1 m | 5.63 s / 57.0 m | 5.56 s / 58.4 m | 2.70 s / 26.5 m |
| Visible-threat time (mean, 25 m gate) | 3.41 s | 3.51 s | 3.40 s | 2.27 s |
| Fly-by / orbit / non-closing episodes | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | – |
| Stage-1 anchor / static anchor | 1.00 / 0.90 | 1.00 / 0.94 | 1.00 / 0.95 | – |
| Gate on the 200 episodes | PASS | PASS | PASS | |

Detection → closest approach: median 1.0 s for both A1 (captured and intercepted) and A0 (the closest approach is within two decisions of detection by construction).

**Paired statistics** (percentile bootstrap, 10,000 resamples, seed 20260201):

| | A1 − A0 capture | A1 − own Stage2-v2 parent capture | A1 − parent interception |
|---|---:|---:|---:|
| 201 | +83.5 pp [78.0, 88.5] | **+10.5 pp [6.5, 15.0]** (21 A1-only / 0 parent-only wins) | −10.5 pp [−15.0, −6.5] |
| 202 | +82.5 pp [77.0, 87.5] | **+7.5 pp [4.0, 11.5]** (15 / 0) | −7.5 pp [−11.5, −4.0] |
| 203 | +86.5 pp [81.5, 91.0] | **+14.5 pp [9.5, 20.0]** (30 / 1) | −14.5 pp [−20.0, −9.5] |

Parents on the same scenarios: capture 73 / 75 / 72 % — they already transferred substantially from static to moving-defender evasion; Stage 3 adds the +7.5…+14.5 pp.

## 9. Moving-defender evasion analysis (successful episodes; `behavior_analysis.json`)
Measured against the **same policy flown without the defender** on the identical scenario (so “deviation” is causal), for 167 / 165 / 173 captures.
* **Pursuit → detection → evasion → resume → capture**: no episode deviates before the defender enters the 25 m gate (0 % early deviation); a > 1 m displacement from the no-defender path appears in 86 % of captures, median **0.5 s after detection** (1.5 s after gate entry).
* Magnitude: median maximum displacement 1.8–2.2 m (lateral 1.2–1.6 m, vertical 0.8–1.1 m); closest approach median 5.6–6.0 m (A0: 2.5 m = contact). Post-encounter mean cosine to the asset 0.95–0.97; median reacquisition 0.5–1.0 s.
* Defender at the closest approach: median speed 7.2–7.4 m/s; the attacker passes in front of the defender's velocity vector in 51–61 % of captures (behind in the rest). Closing speed at first visibility median ≈ 10.5 m/s.
* Classes (descriptive, thresholds in `definitions`): 201 — lateral 40, mixed 66, timing/turn 26, vertical 12, unclear 23; 202 — lateral 36, mixed 87, timing/turn 15, vertical 3, unclear 24; 203 — lateral 57, mixed 72, timing/turn 14, vertical 5, unclear 25. Labels are never used for reward.

## 10. Generalization diagnostic (not a gate; `generalization_diagnostic.json`, scenarios `156000–156099`, same hold-until-detection semantics)
Capture of the Stage3-v2 policy (Stage2-v2 parent) against other scripted interceptors; matched A0 is 0 % (100 % intercepted) everywhere:

| Seed | Greedy | Kalman-Greedy | PN | Lead |
|---|---:|---:|---:|---:|
| 201 | 76 (68) | 76 (68) | 72 (59) | 69 (57) |
| 202 | 84 (78) | 85 (78) | 86 (77) | 83 (74) |
| 203 | 87 (66) | 87 (67) | 84 (60) | 84 (58) |

The learned behaviour is therefore **not Greedy-specific**: it transfers to Kalman-Greedy, PN and Lead with capture within a few points of Greedy and above the parent in every cell (the gains over the parent are largest for the controllers the policy never saw). Not trained against, not used for mastery.

## 11. Limitations (read before citing)
1. **Hold-until-detection is a curriculum simplification.** The defender accelerates from rest at 15 m; its speed at the closest approach is ≤ 8 m/s (median ≈ 7.3 m/s of the 18 m/s limit) and release → interception is 0.5–1 s. Stage 3 is therefore a *short, hard-start* moving-interceptor task, not sustained pursuit by a fast defender. Later stages must remove it (pre-detection escort or patrol, fully moving defender).
2. Mastery was reached at the edge of the 0.80 gate with ± 5 pp validation noise; the 200-episode confirmation (82.5–86.5 %, CI vs parent excludes 0) is the stronger statement, but one defender family at one geometry is evidence of transfer, not of robustness.
3. Evasions are small (median displacement ≈ 2 m, closest approach ≈ 5.6 m): the policy buys just enough clearance for the 2.5 m interception radius; the remaining 13.5–17.5 % are interceptions.
4. Stage-2 (static) anchor capture is computed on a fresh block and is not paired to the parent beyond the figures reported above.
5. Mastery passes were obtained with 2 torch threads in the trainer process (numerics differ at the 1e-6 level from other thread counts); resume equivalence was validated under that setting.
6. All Stage3-v2 checkpoints are provisional curriculum checkpoints derived from provisional parents; none is a final A1.

## 12. Reproduce / verify
```
python -m experiments.journal_v2_a1_cur3v2_run resume-test    # stage 6 evidence (real SB3, injected crash)
python -m experiments.journal_v2_a1_cur3v2_run all            # stages 1-9 (restartable)
python -m experiments.journal_v2_a1_cur3v2_final all          # stage 10
pytest tests/test_journal_v2_a1_cur3v2.py -q
```
State: `task_state.json` (14 stages, atomic, hash-verified); scratch `C:\Users\Public\journal_v2_a1_curriculum\stage3_v2\`; results (untracked) `results/journal_v2_corrected_semantics/a1_curriculum/stage3_v2/`. Protocol SHA-256 in `protocol.json`.
