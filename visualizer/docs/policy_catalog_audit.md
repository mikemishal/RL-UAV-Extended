# Policy and checkpoint audit

Audited 2026-10-16 against local source commit
`a3bc6c4acf1d0b601945159d682f9f14bc702860` and read-only artifacts in the
original research worktree. No model was loaded, no episode was run, and
no training/evaluation campaign was launched. Imports are not model loads.

## Sources inspected

- `uav_defend/envs/soldier_env.py`, `config/env_config.py`
- `experiments/journal_clutter_protocol.py`, `clutter_benchmark_protocol.py`
- `experiments/defender_adaptation_evaluation.py`, `defender_adaptation_env.py`
- `experiments/adaptive_attacker_env.py`, `adaptive_attacker_evaluation.py`
- `experiments/policy_wrappers.py` (historical escort wrapper; not canonical)
- `uav_defend/policies/registry.py`, `sanitize.py`, baseline controllers,
  PPO/PPO-Kalman wrappers, residual observation/composition/wrapper and RMPC
- obstacle geometry, dynamic obstacles, sector specification and perception
  interfaces used by the canonical harness
- `experiments/eval_utils.py` and both adaptation/attacker trajectory utilities
- frozen model tables in `post_detection_diagnostic_protocol.py` and
  `lead_residual_diagnostic_protocol.py`, artifact resolver, model registry,
  representative D0/D0+/D1/A1 training manifests and frozen RMPC selection.

Paths below are relative to the configured **research data root**, not the
new worktree. The scientific registry currently contains absolute paths;
those must not become browser inputs.

## Analytical controllers

| Display | Factory / estimator | Parameters and diagnostics |
|---|---|---|
| Greedy | `get_policy("greedy")`, measurement | `eps=1e-8`; pursuit of legitimate noisy measurement; no checkpoint; action is available, no rich `last_*` trace |
| Kalman Greedy | `get_policy("kalman_greedy")`, Kalman | `eps=1e-8`; requires `use_kalman_tracking=True`, pursues `e_hat`; no learned checkpoint |
| PN | `get_policy("pn", config=cfg)`, measurement; `pn_kalman` separately exists | Navigation gain 3.0, minimum speed fraction 0.10, cfg dt/speed/epsilon; `last_range`, LOS unit/rate, closing speed, target velocity, PN acceleration, guidance/fallback mode |
| Lead | `get_policy("lead", config=cfg)`, measurement; `lead_kalman` separately exists | cfg `dt=0.5`, `v_d=18`, `eps=1e-8`; exposes target/relative state, quadratic coefficients/discriminant, intercept time/point, last action and guidance/fallback mode |
| RMPC | Direct `RMPCPolicy`, measurement; **not in registry** | Frozen `C1_FAST`: horizon 4, population 64, elite fraction 0.2, 3 CEM iterations, controller seed 0, initial std 1, min std .05, sample clip 3; best sequence, scenario/robust scores, initialization/guidance modes, objective evaluations, optional timings |

Registry also exposes `random`, `pn_kalman`, `lead_kalman`, and aliases; their
presence does not make them primary journal methods. Kalman Greedy is implemented
but explicitly excluded from the frozen primary paper/attacker-evaluation set.
It can be a separately labeled supported analytical baseline, not a fabricated
paper result. RMPC class defaults (6/128/5) are **not** the frozen selected
configuration. Its frozen helper disables timing for canonical evaluation even
though the selection JSON records timing enabled. Record any timing choice.
RMPC is materially slower; do not weaken its optimizer for interactive latency.

## Learned inventory and exact input requirements

All current policies issue a three-component desired-direction action.
Base SoldierEnv observation is normalized float32 `(16,)`:
soldier position 3, defender position 3, defender velocity 3, detection flag 1,
legitimate hostile position estimate 3, legitimate velocity estimate 3.
Direct and Kalman tracks have matching structure but different estimators.

| Family | Conference training seeds | Clutter architecture / training seeds | Model input / estimator |
|---|---|---|---|
| PPO | 45,46,47 | `ppo_clutter_v1` / 71,72,73 | conference 16; clutter 36 = 16 + 20 sectors; measurement |
| PPO-Kalman | 45,46,47 | `ppo_kalman_clutter_v1` / 74,75,76 | conference 16; clutter 36; Kalman must be enabled |
| LR-PPO | 55,56,57 | `lrppo_clutter_v1` / 77,78,79 | conference 19 = 16 + Lead direction 3; clutter 39 = 19 + 20 sectors; measurement |

LR-PPO is **not** a `get_policy` registry name. Use its existing wrapper.
Never send a 16-D observation directly to a 36/39-D model or append sectors
twice. The wrapper receives 16-D plus separate `extra_features`, builds Lead
and residual observation, then calls the model. PPO-Kalman's wrapper is a
semantic marker; it does not change environment estimator configuration itself.
All learned models require existing SB3 PPO/PyTorch and deterministic inference.
Actual checkpoint architecture/space metadata still needs runtime validation
in Phase 2; this audit does not claim deserialization validation.

`sector_obs_v1`: 20 features, 10 sectors x 2 channels, sensor range 15 m.
Sector order: front, front-left, left, rear-left, rear, rear-right, right,
front-right, up, down. Channel order: static clearance, dynamic clearance.
Vertical cap threshold: 60 degrees. Use the existing feature builder.

### D0 / D0+ / D1 roles (each architecture's same three seeds)

| Generation | Scientific meaning | Primary artifact |
|---|---|---|
| D0 | Original randomized-clutter defender | `results/clutter_training_400k/{architecture}/seed_{seed}/best_model.zip`, role `d0_start` |
| D0+ (`d0_plus`) | Continuation against scripted A0 only | `results/defender_adaptation/d0_plus_scripted/{architecture}/seed_{seed}/final_model.zip`, role `final` |
| D1 (`d1`) | Continuation against 50% A0 / 25% A1-201 / 25% A1-202 | `results/defender_adaptation/d1_adaptive_mixture/{architecture}/seed_{seed}/final_model.zip`, role `final` |

Authoritative registry: `results/defender_adaptation_model_registry.json`,
generated 2026-10-16T00:19:38Z, training source
`69596d8af8dc27bb080fa1eee37a38ea5090b6ef`.
27 primary defender entries = 3 architectures x 3 seeds x 3 generations.
There are 18 additional `best_model` references for D0+/D1, **secondary only**:
their callbacks used different opponent distributions, so silently replacing
final models with best would break the primary comparison.

All **48 registry references** (27 primary defenders + 18 secondary best +
3 attackers) existed and were SHA-256 verified during Phase 1. All nine
conference best checkpoints also existed and matched the source hash tables
(57 reference checks total). These are artifact checks, not new evaluations.
Hash verification must be repeated before future inference; an audit is not
a permanent assertion about mutable local storage.

Conference locations:

- PPO: `results/training_post_detection_400k/direct/seed_{45,46,47}/best_model.zip`
- PPO-Kalman: `results/training_post_detection_400k/kalman/seed_{45,46,47}/best_model.zip`
- LR-PPO: `models/lead_residual_post_detection/seed_{55,56,57}/best_model.zip`

Earlier pre-revision PPO seeds 42/43/44 exist as legacy protocol identities,
not replacements for the post-detection conference models or D0.

### Recorded training defaults and seed roles

Representative D0/D0+/D1/A1 manifests agree on PPO hyperparameters:
learning rate .0003, gamma .99, batch size 64, rollout n_steps 2048,
n_epochs 10, clip .2, entropy .01, value coefficient .5, max grad norm .5,
GAE lambda .95. These are audit metadata, not an invitation to retrain.

D0 and A1 campaigns target 400,000 transitions (actual vectorized totals
must come from each manifest, not be rounded to the target). D0 source:
`5f7e5b934384c0bb3ee5089562a71e81031d21e1`; A1 source:
`17db54661baa3eb300f39a6234dad8b3514d0dc1`.
D0+/D1 each add exactly 425,984 transitions from the selected D0 best,
not necessarily the D0 final model. Representative seed 71 and LR seed 77
start at 320,000 and end at 745,984; do not assume all seeds start at that
same counter. Worker seeds are separately recorded in manifests.

Training seeds are **not** simulation episode seeds:

| Namespace | Existing values |
|---|---|
| Legacy PPO training | 42,43,44 |
| Revised conference PPO/PPO-Kalman | 45,46,47 |
| Conference LR-PPO | 55,56,57 |
| Clutter defender training | PPO 71-73 / PPO-Kalman 74-76 / LR-PPO 77-79 |
| A1 training | 201,202,203; diagnostic 200 |
| Journal clutter held-out evaluation | starts 95,000 |
| A1 validation / held-out | starts 180,000 / 190,000 |
| Shared clutter/adaptation in-training validation | starts 90,000,000 (20 episodes) |
| Defender adaptation matched evaluation | 90,010,000 through 90,010,099 |

The viewer will use explicit simulation seed metadata and consecutive batches;
it must not invoke training validation or rewrite existing evaluation datasets.

## Attackers

| UI identity | Existing semantics | Primary checkpoint / requirements |
|---|---|---|
| A0 | Original scripted pursuit/weave/evasion and canonical obstacle handling | none; leave `enemy_tactical_override=None` |
| A1-201 | Seen by D1 adaptation | `results/adaptive_attacker_400k/seed_201/best_model.zip` |
| A1-202 | Seen by D1 adaptation | `results/adaptive_attacker_400k/seed_202/best_model.zip` |
| A1-203 | Held out from D1 training/model selection | `results/adaptive_attacker_400k/seed_203/best_model.zip` |

A1 is `adaptive_attacker_v1`: 36-D observation = role-swapped 16-D base +
20 obstacle sectors; three-component tactical desired direction. Defender-relative
features are zeroed beyond the separate 25 m attacker sensing range. It uses
its own perceived obstacles, not omniscient obstacle inputs.
203 is held out **from defender adaptation**, not an untrained attacker.
All A1 were trained against Lead plus all nine D0 clutter defenders, not RMPC.

The existing path helper permits best->final fallback; the viewer should instead
honor the frozen registry-selected best/hash and report a missing primary model,
not silently adopt fallback selection.

## Scenario and compatibility matrix

Canonical builder: `build_canonical_env_config`, not the historical
`clutter_benchmark_protocol.build_env_config` (which leaves attacker avoidance
off and has different physical-collision consequences).

| Scenario | Static AABBs | Static width/depth range (m) | Height range (m) | Moving obstacles |
|---|---:|---|---|---:|
| C0 | disabled | n/a | n/a | 0 |
| C1 | 3 | 2-4 | 2-8 | 1 |
| C2 | 8 | 3-6 | 3-10 | 2 |
| C3 | 15 | 3-7 | 4-12 | 4 |

Level B = `local`; Level C = `map_local` (canonical default), configured for
both roles. C gives prior mapped static obstacles plus local sensing of
unmapped static/dynamic obstacles. B relies on local sensing/memory.
Both use clearance 2 m, 12-step prediction, 4-step release,
static/dynamic sensor range 15 m, static memory enabled, dynamic memory 6 steps.
Route planning is off canonically; arbitration enabled but dormant.
Do not invent a building/tree class from map knowledge.

| Combination | Evidence / interpretation |
|---|---|
| Greedy / PN / Lead / RMPC, measurement, canonical C0-C3, A0 | Implemented inputs; journal canonical environment applies unchanged safety to requested commands |
| Kalman Greedy, Kalman, canonical C0-C3 | Factory and estimator contract exist; not in primary frozen paper/attacker evaluation inventory; do not claim measured parity |
| Conference learned models, matching estimator, C0-C3 | Existing historical clutter / attacker evaluators support frozen obstacle-unaware models; no sector features. Out-of-training-distribution in clutter, not equivalent to clutter-trained generation |
| Clutter D0/D0+/D1, matching estimator and sectors, C1-C3, A0/A1-201/202/203 | Explicit full matched adaptation evaluation inventory |
| Clutter models, C0 | Builder and same observation dimensions support it; separate C0/canonical-C0 result directories exist. Canonical C0 safety/awareness configuration is not automatically identical to every conference C0 protocol |
| Level B vs C | Builder/safety/perception support both; C is training default and B an ablation. Do not label every generation/opponent/B combination as previously evaluated |
| Analytical/Conference models vs A1 | Implemented in adaptive attacker evaluator; RMPC historical budget was 5 episodes and selected attacker 202 for cost, not evidence of a full symmetric campaign |

Compatibility is structural, not a promise that every Cartesian-product cell
has been scientifically evaluated. Wrong estimator, missing sector features,
incorrect generation/seed, missing frozen RMPC config, or wrong checkpoint role
must be explicit incompatibility errors in future catalog/API work.

## Environment defaults to preserve

Native **3D**, not the prompt's assumed 2D. Domain x,y in [-50,50] m;
z in [0,30] m; dt .5 s; max 2000 steps; soldier speed 1.5 m/s,
attacker max 12 m/s, defender max 18 m/s. Detection radius 15 m,
interception radius 2.5 m, unsafe radius 3.5 m. Defender starts in canonical
standby until detection; requested action may not be executed.
Acceleration/turn/climb/descent constraints remain scientific source-owned.
Record full dataclass configuration rather than reconstructing it from these
summary values.

Static obstacles: `AABBObstacle(min_corner,max_corner,map_known)`.
Dynamic obstacles: sphere geometry, IDs, type (`bird`/`airborne_clutter`),
position, velocity and radius; stochastic horizontal headings at fixed native
altitude. Record actual generated geometry and per-frame moving obstacle state,
not a regenerated browser layout.

Outcomes retain exact native names: `intercepted`, `soldier_caught`,
`unsafe_intercept`, `timeout`, defender static/dynamic obstacle collision,
enemy static/dynamic obstacle collision. Attacker environmental failure is
**not** a defender interception. The recorder must not reclassify outcomes.

## Honest diagnostics availability

- SoldierEnv exposes actual positions/velocities, estimates, `tracking_error`,
  `defender_desired_command`, `defender_safe_command`, avoidance activity/angle,
  standby, `controller_action_executed`, constrained-motion saturation and
  collision event points/steps. Truth is recorder-only.
- Lead, PN and RMPC expose named `last_*` fields; copy after the **single**
  actual decision, before next decision overwrites them.
- PPO/PPO-Kalman expose chosen action, not explanations or attention weights.
  Heading/reversals/closing speed are explicitly derived diagnostics.
- LR-PPO wrapper returns only composed guidance. It does **not** currently expose
  raw PPO residual, norm clipping, perpendicular component or saturation.
  A requested-vs-Lead angle can be measured, but raw residual cannot be uniquely
  recovered from final action. Keep those fields null with a reason until
  Phase 5 implements a verified observational adapter without modifying science.
- Keep `build_policy_info` sanitizer authoritative. `enemy_pos`, `enemy_vel`,
  tracking error and full obstacle/collision geometry must never leak into
  defender `act` inputs. Display ground truth under a privileged-diagnostics label.
