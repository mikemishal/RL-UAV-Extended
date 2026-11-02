# Curriculum Stage2-v2 — Stage 2 training-budget extension (8 → 16 rollouts)

**Result: Stage2-v2 does NOT meet the frozen mastery rule (0/3 seeds mastered by rollout 16), but the budget extension produced a large, consistent improvement.** Final held-out capture rose from 67.0 / 58.5 / 55.5% (Stage2-v1, rollout 8) to **90.0 / 88.5 / 89.0%** (rollout 16) on the same 200 scenarios (paired v2 − v1: **+23.0 / +30.0 / +33.5 pp**, CIs exclude 0), against matched A0 at 0%. Pursuit retention stayed perfect and no pathologies appeared. The mastery rule (two consecutive validation passes) was not met: seed 201 passed rollouts 14 and 16 but not 15; seed 203 passed only rollout 16; seed 202 never passed (best 0.88). No gate, parameter or definition was changed, the budget was not extended again, and Stage 3 was not started. These remain curriculum checkpoints, not final A1 models; none is accepted.

## What changed (and only this)

> Stage2-v2 is a preregistered continuation of Stage2-v1. It changes only the maximum PPO budget from 8 to 16 complete rollouts. All environment, reward, initialization, PPO, validation, and mastery definitions are unchanged.

`protocol.json` (SHA-256 `ff679fa4e65ac808ab0b9a5180e496bcf2ae19bfef12ba50806fbb2e56e2f8fa`) records the diff against Stage2-v1: **exactly two leaf fields** differ, `ppo.max_rollouts` (8 → 16) and `ppo.max_timesteps` (262,144 → 524,288); the Stage2-v1 protocol hash (`07c9fcf5…`) was re-computed from the unchanged definitions before starting. Everything else — C0, stationary asset, stationary active defender on A0's nominal path (λ ~ U(0.35, 0.65), r ~ U(0, 1.5 m), canonical 2.5 m interception, 25 m observation gate), `a1_v2_obs_v1`, `a1_cur1_action_v1`, `a1_v2_reward_v2`, PPO hyperparameters, validation sets A/B (147000–147199, alternating odd→A / even→B continuing from v1), the Stage-1 anchor (148000–148099), all mastery gates and the retention rule — is unchanged.

## Continuation mechanics

Each seed's verified Stage2-v1 rollout files 1–8 (policy + critic + optimizer, timestep counter, RNG state, episode logs, validations, manifests) were hash-verified against the v1 seed manifests and the pre-registered rollout-8 prefixes (**be8b85e4… / 25e67e79… / 6472fbee…**), copied (read-only on v1) into the Stage2-v2 namespace, and the unchanged Stage-2 runner was called with `max_rollouts = 16`. It restores rollout 8 exactly as after an interruption (no BC re-initialisation, no restart from zero) and trains rollouts 9–16 only as needed. A real-SB3 test confirms that extending a finished run with a larger budget is bit-identical (policy+optimizer state hashes, episode logs) to the uninterrupted run. Stage2-v1 results are never written (output namespace `a1_curriculum/stage2_v2/`, separate manifests/hashes).

## Training results (rollouts 9–16; rollouts 1–8 are Stage2-v1; full table in `rollout_results.md`)

Validation capture by rollout (sets alternate A/B; 100 episodes each; matched A0 captures 0 on every set):

| Rollout (set) | 8 (B) | 9 (A) | 10 (B) | 11 (A) | 12 (B) | 13 (A) | 14 (B) | 15 (A) | 16 (B) |
|---|---|---|---|---|---|---|---|---|---|
| Seed 201 | 0.73 | 0.59 | 0.75 | 0.71 | 0.81 | 0.79 | **0.93** | 0.86 | **0.93** |
| Seed 202 | 0.65 | 0.61 | 0.78 | 0.73 | 0.83 | 0.83 | 0.87 | 0.88 | 0.88 |
| Seed 203 | 0.59 | 0.60 | 0.70 | 0.69 | 0.79 | 0.75 | 0.86 | 0.83 | **0.90** |

(Bold = full gate passed.) The *only* gate check that ever failed in rollouts 9–16 is **capture ≥ 90%**; improvement over A0 (+59…+93 pp), timeout/exhaustion (0), median path-excess (1.03–1.04), >3× (0), orbit (0), anchor capture (1.00) and numerical checks pass at every rollout. Mastery requires two consecutive passes, so seed 201 (pass, fail at 0.86 on set A, pass) and seed 203 (single pass at rollout 16, the last) are not mastered; the run stops at the budget limit (no automatic further extension).

| Seed | rollouts | timesteps | passing rollouts | first / second passing rollout | status |
|---|---|---|---|---|---|
| 201 | 16 | 524,288 | 14, 16 | — / — (no two consecutive) | not mastered |
| 202 | 16 | 524,288 | none | — | not mastered |
| 203 | 16 | 524,288 | 16 | — | not mastered |

### Learning-curve description (descriptive only; no model fitted; second-half mean vs first-half mean of rollouts 9–16, ±0.05 threshold)

| Seed | validation capture | interception | mean return | median path-excess | anchor capture | training capture (stochastic) |
|---|---|---|---|---|---|---|
| 201 | rising | falling | rising | plateau (≈1.03–1.04) | plateau (1.00) | rising |
| 202 | rising | falling | rising | plateau | plateau | rising |
| 203 | rising | falling | rising | plateau | plateau | rising |

(`learning_curves.json` labels falling interception as "regressing" because the label is direction-of-metric; here it is an improvement.) Return rose from ≈ −9.5 at rollout 1 to +8.0…+9.0 at rollout 16. Capture was still increasing at the end of the extended budget (Fig. 1), with ≈ ±5 pp validation noise per 100-episode set.

## Final held-out evaluation — 200 matched scenarios (147000–147199), final = rollout-16 policy (`final_evaluation.json`)

| | Capture | Interception | Exhaustion / timeout | Mean path length | Median / mean path-excess | Median min. defender separation | Defender-visible time |
|---|---|---|---|---|---|---|---|
| **Canonical A0** | 0.0% | 100% | 0 | 28.5 m | 1.019 / 1.035 | 2.50 m | 2.42 s |
| **A1-CUR2-201** | **90.0%** | 10.0% | 0 | 59.3 m | 1.041 / 1.055 | 5.24 m | 4.41 s |
| **A1-CUR2-202** | **88.5%** | 11.5% | 0 | 58.9 m | 1.042 / 1.058 | 5.13 m | 4.30 s |
| **A1-CUR2-203** | **89.0%** | 11.0% | 0 | 58.9 m | 1.042 / 1.055 | 4.93 m | 4.28 s |

0 fly-bys, 0 repeated fly-bys, 0 orbits, 0 non-closing episodes, 0 numerical failures, >3× path 0%. On the combined 200 scenarios seed 201 satisfies every individual gate check (capture 90.0%); seeds 202/203 miss only capture (88.5 / 89.0%). **This combined-200 result is informational: mastery is defined on two consecutive alternating validation passes and was not met.**

### Paired statistics (matched scenarios, paired bootstrap 95% CI, 10,000 resamples)

| Seed | A1 − A0 capture | 95% CI | A1 − A0 interception | v2 − v1 capture (same scenarios) | 95% CI | v2-only / v1-only successes |
|---|---|---|---|---|---|---|
| 201 | **+90.0 pp** | [+86.0, +94.0] | −90.0 pp | **+23.0 pp** (67.0 → 90.0) | [+17.0, +29.0] | 47 / 1 |
| 202 | **+88.5 pp** | [+84.0, +92.5] | −88.5 pp | **+30.0 pp** (58.5 → 88.5) | [+24.0, +36.5] | 61 / 1 |
| 203 | **+89.0 pp** | [+84.5, +93.0] | −89.0 pp | **+33.5 pp** (55.5 → 89.0) | [+27.0, +40.0] | 67 / 0 |

(Counts are exact, from `budget_sensitivity.json`: scenarios captured by the rollout-16 policy but not the rollout-8 policy / the reverse.) This is the **training-budget sensitivity** result: doubling the budget under the identical protocol raised held-out capture by 23–34 pp on every seed.

## Behavior of the final policies (`evasion_analysis.json`; successful episodes n = 180 / 177 / 178)

* **Still reactive:** replaying the same policy without the defender, trajectories are identical until the 25 m gate in 100% of episodes; after it 82% / 86% / 75% of episodes are displaced > 1 m (median max displacement 1.9 / 1.6 / 1.6 m).
* **Wider, safer detours than v1:** median max lateral deviation from the initial chord 6.6 / 7.3 / 7.0 m (v1: 5.4 / 4.9 / 5.5), vertical 4.2 / 4.2 / 3.7 m, median closest defender distance **5.4 / 5.4 / 5.2 m** (v1: 4.5 / 4.0 / 4.2; A0: 2.5). Side at closest approach (left/right/over/under): 201: 27/56/65/32, 202: 24/73/67/13, 203: 59/54/48/17 — horizontal and vertical detours both used.
* **Return to pursuit:** mean cosine toward the asset after the closest approach 0.937 / 0.958 / 0.975; fraction of captured episodes meeting the pre-registered resume criterion (mean cosine ≥ 0.95) 43% / 70% / 93%; strict pursuit→evasion→pursuit structure (≥ 1 pursuit step before, deviating steps while the defender is visible, pursuit after) 14% / 20% / 29% (v1: 37 / 69 / 55%). The strict criterion uses a 10° line-of-sight tolerance and a pre-encounter pursuit phase, so the larger detours of the longer-trained policies (and scenarios beginning inside the gate, ≈ 24%) lower it; the criteria were **not** altered. Path-excess stays ≈ 1.04.
* **Pursuit retention / forgetting (no-defender C0, 146000–146199):** capture 100% (all seeds); median path-excess 1.040 / 1.043 / 1.040 vs BC 1.024; mean duration 5.98 / 5.98 / 5.88 s vs BC 5.645 s; action cosine to the BC initializer 0.971 / 0.976 / 0.979 (Stage2-v1: 0.989–0.993) — somewhat larger drift than at 8 rollouts but pursuit remains intact; anchor capture 1.00 at every rollout of every seed.

## Answer to the scientific question

> Was static-defender evasion limited primarily by learning budget?

**Largely yes, but not sufficiently to pass the frozen rule.** Doubling the budget improved held-out capture by 23–34 pp on every seed, brought the final policies to 88.5–90.0% (A0: 0%), and all structural indicators were already healthy at 8 rollouts (no pathologies, retention intact). The frozen two-consecutive-pass rule on 100-episode alternating validation sets, however, was not satisfied by any seed by rollout 16: the policies sit at the threshold (≈ 0.86–0.93 per validation set with ≈ ±5 pp sampling noise), and curves were still rising. The result supports "compute-limited, not structurally flawed" as the primary explanation but **does not confirm the preregistered hypothesis that extension would reach mastery**; a further budget increase would be a new scientific decision and was not made.

## Limitations

* Three seeds; 100-episode validation sets (s.e. ≈ 3 pp near 90%) make the pass/fail decision near the threshold noisy — e.g. seed 201's 0.86 at rollout 15 versus 0.93 at rollouts 14/16 on different sets.
* The mastery rule is deliberately strict (two consecutive passes); this report neither relaxes it nor reinterprets near-misses as passes.
* Defender is on A0's nominal path by construction; results are for a fixed threat in C0 with one gate range.
* The budget-sensitivity comparison is between the rollout-8 and rollout-16 (final, not selected) checkpoints of the same trajectories, not independent re-trainings.

## Not done (per instructions)

No Stage 3, no obstacles, no moving asset/defender, no defender training, no change to A0, reward, environment or PPO hyperparameters, no further budget extension, no `A1-CUR2-*` visualizer registration (only if ≥ 2 seeds master), no Stage-3 handoff, no tag.

## Files

`protocol.json`, `repository_verification.json`, `parent_checkpoints.json`, `seed_{201,202,203}_manifest.json`, `rollout_results.md`, `learning_curves.json`, `final_evaluation.json`, `budget_sensitivity.json`, `evasion_analysis.json`, `bc_fidelity.json`, `stage2_decision.json`, `checkpoint_integrity.json`, `unit_test_result.json`, `regression_summary.json`, `task_state.json`, `figures/` (PNG + PDF), `manuscript_notes.md`, `ablation_record_v8.md`.

| Seed | Stage2-v1 parent (rollout 8, not modified) | Stage2-v2 final (rollout 16) SHA-256 |
|---|---|---|
| 201 | `be8b85e4d9c30b3808215e5346e03718726d883c22aab6518ea5581aa676fbeb` | `d70b85a6a4d068646da1319f3f44b3057dce1e5fedd1f66b28c85b4cbd55f90c` |
| 202 | `25e67e794a44b392c0170f774096216ae44d0f84c9fd9e29d547a8d595001361` | `97a934bc2d115c4eec1a3b838fd754fd2dba93f57d91d84203aed9d7efb4d174` |
| 203 | `6472fbee620226379588871bdfb33c1b85edae82213170e68b208b16b4e472fa` | `06993dca67b59876c65afa956bcdace905482745b58cf39c9da87d713a578cac` |

(Models under `results/` are not tracked by git; hashes are recorded here and in the seed manifests. The rollout-16 models are NOT accepted checkpoints.)
