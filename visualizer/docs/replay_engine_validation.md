# Phase 2 replay-engine validation

## Scope and preserved source

The replay engine runs only from the visualization worktree and pins the
research source to `a3bc6c4acf1d0b601945159d682f9f14bc702860`. It loads the
allowlisted local artifacts read-only and does not train, write to the research
tree, start a batch, or add a replay endpoint. `A1-203` is evaluation-only.
The known pre-existing research documentation edit was left unchanged.

The backend's API/test virtual environment does not contain the research
Gymnasium/SB3/Torch stack. Run the CLI/tests that load the simulator using the
existing research Python environment; no second Torch installation or changes
to the scientific requirements are needed. `application_error.py` is isolated
from FastAPI so the standalone CLI can import the Phase 1 data resolver in that
environment.

## Real simulator semantics preserved

- SoldierEnv's native axes are `(x,y,z)` in meters, with positive `z` upward.
  The protected soldier remains on `z=0`; UAV altitude is part of the actual
  state and all recorded positions, velocities, commands and geometry remain
  three-component values. `dt=0.5 s` in the frozen canonical configuration.
- The selected model's `predict` / wrapper `.act` output is recorded as the
  requested float32 direction. SoldierEnv normalizes nonzero defender input to
  unit direction (using the configured epsilon); zero requests its existing
  deceleration-to-stop behavior. The visualizer does not clip, normalize,
  rescale or rewrite a policy action. The environment records its normalized
  `defender_desired_command`, post-obstacle `defender_safe_command`, and
  resulting constrained velocity separately.
- Static safety geometry is the existing AABB layout; moving obstacles use
  their existing native sphere IDs, types, positions, velocities and radii.
  The replay records current generated geometry, not browser-regenerated
  obstacles. Current simulator collision/event fields are retained verbatim in
  privileged frame info. The viewer's horizontal projection is never used for
  collision or outcome calculations.
- At each environment step, dynamic obstacles advance, the soldier moves,
  standby synchronizes the defender while undetected at step start, the
  scripted/learned hostile moves, detection/tracking updates, and then the
  supplied defender action is executed only if detection was already active at
  step start. Thus the transition in which detection first becomes true is
  still standby; the first applied defender action is the following transition.
  Existing obstacle safety runs on the normalized desired command before the
  constrained defender point-mass update. The source owns acceleration, turn,
  climb/descent, bounds, collision and termination semantics.
- Physical collision checks are the source's swept-motion checks: a mover's
  segment is intersected against static AABBs; moving-obstacle collision uses
  swept sphere-versus-sphere contact with the moving obstacle's configured
  radius. The first static or dynamic contact for each UAV is selected,
  position is clamped at contact and velocity is zeroed before native outcome
  precedence. No browser collision radius or 2D projection is introduced.
  Standby synchronization is bookkeeping, so the simulator does not sweep the
  defender across the soldier's pre-detection displacement.
- A frame at index zero is captured after reset; frame `k` is the returned
  state after transition `k`. Each transition's telemetry identifies its
  prior decision input and keeps the requested action, normalized desired
  direction, safety-modified direction and resulting velocity distinct.

## Resolver, adapters and model rules

`run_replay_episode(ReplayRequest, Settings) -> ReplayBundle` implements one
real episode. It validates a fixed policy catalog, source commit, checkpoint
role/path/hash and SB3 observation/action shapes, builds the canonical
`EnvConfig`, constructs `SoldierEnv`, installs the exact A1 override hook when
selected, resets with the requested episode seed, executes one deterministic
decision per policy per pre-step state, records reset plus every transition,
validates the trace, and rechecks checkpoint hashes. It closes the environment
in `finally`.

The engine reuses:

- `experiments.journal_clutter_protocol.build_canonical_env_config`
- `uav_defend.envs.soldier_env.SoldierEnv`
- `uav_defend.policies.registry.get_policy` for analytical families
- Existing PPO, PPO-Kalman and `LeadResidualPPOPolicyWrapper` load/act methods
- `experiments.defender_adaptation_evaluation`'s frozen model registry roles
  and exact `_run_matched_episode` ordering
- `experiments.adaptive_attacker_env.build_attacker_observation`
- `uav_defend.policies.sanitize.build_policy_info`
- `uav_defend.obstacles.sector_features.compute_sector_features` and its
  `sector_obs_v1` metadata
- `experiments.clutter_benchmark_protocol._frozen_rmpc_config`
- Existing Lead/PN/RMPC `last_*` diagnostics, environment info and native
  obstacle models

It does not call an evaluator writer, policy trainer, plotter or bulk runner.
The extra recorder reads are observational only; equivalence below verifies
they do not perturb the environment's random streams or policy outputs.

Supported learned defender catalog:

| Policy | Conference family/seeds/input | Clutter family/seeds/input | Estimator |
|---|---|---|---|
| PPO | `direct`, 45-47, 16 | `ppo_clutter_v1`, 71-73, 36 | measurement |
| PPO-Kalman | `kalman`, 45-47, 16 | `ppo_kalman_clutter_v1`, 74-76, 36 | Kalman |
| LR-PPO | `lr_ppo`, 55-57, 19 | `lrppo_clutter_v1`, 77-79, 39 | measurement |

For every clutter architecture, D0 resolves only to the registry's
`d0_start`/`best_model.zip`; D0+ and D1 resolve only to their `final`/
`final_model.zip`. Secondary D0+/D1 `best_model.zip` files are rejected.
Conference checkpoints use the existing frozen protocol tables. All learned
models are loaded on CPU in deterministic mode. PPO action shape must be 3;
the exact model observation width is verified before rollout.

Analytical support: Greedy and Kalman Greedy from the registry; PN and Lead
from the registry with canonical config; RMPC using the frozen selected config
(not RMPC class defaults). Analytical policies are checkpoint-free.

Attacker catalog: A0 leaves `enemy_tactical_override=None`; A1-201/202/203
load each frozen adaptive-attacker best checkpoint with the exact 36-D
observation and deterministic prediction. A1 is called once per pre-step state
and is supplied using the original pending-action callback. No A1 output is
recomputed for telemetry.

Canonical C0-C3 and Level B (`local`) / Level C (`map_local`) are constructed
through the journal builder. Clutter checkpoints use their cataloged
measurement/Kalman estimator and existing sector features. Structural support
does not imply every combination has a prior published result; the replay
metadata and explicit generation/seed preserve that distinction.

## Telemetry and missing values

`frame.policy_visible` represents the current state's sanctioned 16-D base
observation, measurement or Kalman fields and (when consumed) current perceived
20-D sector features. `telemetry.decision_policy_visible` is the exact
pre-transition sanitized input state for that policy decision. Privileged
ground truth is isolated in `privileged_diagnostics`, which also captures the
full returned simulator info. It is never passed to `act`.

PPO telemetry stores the exact chosen request, effective model input, actual
movement velocity and reward. PPO-Kalman stores its policy-visible `e_hat`,
`v_hat`, actual attacker state and simulator-provided tracking error separately.
Analytical Lead, PN and RMPC copy exposed `last_*` values from their one actual
decision. No unavailable model internals are synthesized.

LR-PPO captures the wrapper's already-computed Lead action and final requested
direction from the single wrapper call. It records native 3D angular separation
and a separately labeled horizontal signed angle; the native 3D separation
uses the canonical 30-degree definition, not a yaw-only limit. The existing
wrapper does not expose raw PPO residual output, raw/projected norm, or residual
clipping. Those fields remain null with per-field reasons. A zero/undefined
pre-detection command also has null angle with a reason. Safety command and
actual post-dynamics velocity are independent arrows/telemetry.

The protected asset's velocity is not exposed by SoldierEnv and remains null
with a reason; it is not estimated from consecutive positions. Missing optional
diagnostics remain null with field-path reasons. Non-finite values in JSON-safe
simulator diagnostics become explicit null plus a reason; invalid authoritative
positions/actions/velocities fail the replay rather than being fabricated.

## Validation run

The following four required bounded smoke episodes were run with deterministic
evaluation seed `0`, and each compressed JSON replay was reopened and validated
against the generated JSON Schema and Pydantic contract. Frame counts equal
steps plus one; initial and terminal frames are present; all entity coordinates
remain finite native 3D. Static geometry and dynamic obstacles were recorded
from the real environment.

| Defender | Generation / seed | Attacker | Environment | Frames / steps | Native outcome | Defender SHA-256 | Attacker SHA-256 |
|---|---|---|---|---:|---|---|---|
| Greedy | analytical | A0 | C0 / map_local | 282 / 281 | `intercepted` | n/a | n/a |
| LR-PPO | D0 / 77 | A0 | C2 / map_local | 170 / 169 | `soldier_caught` | `a62c71c738655d111c75575eea6ac84f5f9b5b0c46f8703a61cf0d05915c922c` | n/a |
| PPO-Kalman | D0 / 74 | A1-203 | C3 / map_local | 330 / 329 | `enemy_static_obstacle_collision` | `2bce9f825b12da9da3572550ebcee47c8922fb3425e388ebbfacbeff5d2a7e16` | `27f083a1f10c9cfc0a2cb379371845fd4b3a310d2ec88c98165b247d9c97bcf8` |
| LR-PPO | D1 / 77 | A1-203 | C3 / map_local | 974 / 973 | `intercepted` | `b57b551545d4bdf8b9990759edf09c70d2e44a709d6d779346654e1b24c25574` | `27f083a1f10c9cfc0a2cb379371845fd4b3a310d2ec88c98165b247d9c97bcf8` |

The original `_run_matched_episode` comparison used LR-PPO D1 seed 77, A1-203,
C3/map_local, evaluation seed 0. It compared every requested defender action,
initial and per-step defender/attacker/asset positions, complete returned
simulator info, moving obstacle IDs/positions, each reward, termination flags,
detection time, terminal outcome, step count and total reward. Result:
**exact equality** for all compared actions, states, obstacle positions,
rewards, flags, detection time, outcome and step count. Total rewards were
bit-for-bit equal in this run. A separate deterministic repeat also validates
the serialized bundle equality. No extra policy decision or RNG/physics step
was used for recording.

Three additional replays used seeds that exactly match frozen rows in
`results/defender_adaptation_evaluation/episodes.jsonl`. Terminal outcome,
episode length, and detection step agreed in every case:

| Defender / attacker / clutter | Evaluation seed | Frozen outcome / steps / detection | Replay outcome / steps / detection |
|---|---:|---|---|
| LR-PPO D0 seed 77 / A0 / C2 | 90010000 | `intercepted` / 112 / 14 | `intercepted` / 112 / 14 |
| PPO-Kalman D0 seed 74 / A1-203 / C3 | 90010000 | `intercepted` / 593 / 11 | `intercepted` / 593 / 11 |
| LR-PPO D1 seed 77 / A1-203 / C3 | 90010000 | `intercepted` / 13 / 11 | `intercepted` / 13 / 11 |

The Greedy/A0/C0 smoke selection has no corresponding exact row in that
defender-adaptation frozen dataset, so its result is not presented as a
frozen-record match.

PPO-Kalman smoke traces include actual native truth and policy-visible Kalman
estimates in their separate channels. LR-PPO traces include finite measured
native adjustment angles after detection; the captured values remained within
the 30-degree canonical bound. C2/C3 runs recorded all 8/15 static AABBs and
moving obstacles in every frame where present; C0 correctly recorded none.
No checkpoint substitution was made. These seeds were not training runs.

## CLI and output

From the visualization repository root in the existing scientific Python
environment, set the original research root and run one command:

```powershell
$env:VISUALIZER_RESEARCH_DATA_ROOT = 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-lrppo-clutter'
python -m visualizer.backend.replay_cli --defender lr_ppo --generation d1 --training-seed 77 --attacker A1-203 --clutter C3 --perception map_local --evaluation-seed 0
```

Other verified selections are:

```powershell
python -m visualizer.backend.replay_cli --defender greedy --attacker A0 --clutter C0 --perception map_local --evaluation-seed 0
python -m visualizer.backend.replay_cli --defender lr_ppo --generation d0 --training-seed 77 --attacker A0 --clutter C2 --perception map_local --evaluation-seed 0
python -m visualizer.backend.replay_cli --defender ppo_kalman --generation d0 --training-seed 74 --attacker A1-203 --clutter C3 --perception map_local --evaluation-seed 0
```

The CLI reports selected checkpoint IDs/roles/hashes, canonical scenario hash,
episode seed, frame count, native outcome, total reward and output path.
Generated files are under `visualizer/data/local/*.json.gz`; the directory is
gitignored. The writer uses stable key ordering, `allow_nan=False`, gzip
`mtime=0`, a same-directory temporary file and atomic create-without-overwrite.

## Tests and limitations

The focused replay unit suite covers policy/config validation, checkpoint hash
verification, model shape checks, native 3D frame validation, non-finite JSON
normalization, terminal/order semantics and gzip round-trip/non-overwrite.
The optional real integration test is enabled by
`VISUALIZER_RESEARCH_DATA_ROOT`; it performs deterministic repeat and original
runner equivalence using a single frozen selection. Phase 1 API and browser
tests remain unchanged and are run with the API virtualenv.

Completed checks:

- Replay unit + schema contract tests: 15 passed.
- Real deterministic-repeat/original-runner integration test: 1 passed.
- Phase 1 backend suite under the API virtualenv: 24 passed, 2 simulator tests
  skipped because that venv intentionally has no scientific NumPy/SB3 stack.
- Focused existing scientific regressions (adaptation environment/evaluation,
  obstacle sector observation, physical collision, soldier obstacle motion,
  frozen RMPC config): 77 passed.
- Frontend TypeScript typecheck and production build passed; browser tests: 4
  passed, including WebGL scene, responsive controls and outage behavior.
- JSON Schema Draft 2020-12 validation and gzip/Pydantic round-trip passed for
  all four required smoke traces and the three exact frozen-seed comparisons.

The frontend build retains the existing non-blocking large-bundle warning.
Browser tests intentionally exercise a backend outage, so Vite logs expected
`ECONNREFUSED` proxy messages while those tests pass. The API test client emits
the existing upstream AnyIO deprecation warning.

There is no REST execution surface, batch support, timeline, React playback or
policy comparison in this phase. A1 observation is retained per decision;
there is no learned-policy explanation/attention map. The environment info
retains native diagnostics, but the schema does not invent substep collision
fractions. The replay records current `source_dirty` and `visualizer_dirty`
flags alongside commits; a local episode's output remains untracked.
