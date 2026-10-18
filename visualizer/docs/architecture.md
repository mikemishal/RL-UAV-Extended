# Architecture / Phase 7

## Isolation and ownership

Research branch: `research/lr-ppo-clutter`. Verified local HEAD:
`a3bc6c4acf1d0b601945159d682f9f14bc702860`. The new worktree was created from
this exact commit on `tooling/journal-visualizer`. The research worktree had
one pre-existing unstaged documentation edit; it was neither copied into this
branch nor modified. No remote base, merge, reset, rebase, push or tag operation
is part of this setup.

Only `visualizer` additions are in scope. Do not change SoldierEnv, physics,
hostile/defender behavior, observation construction, reward, collision priorities,
policy classes, canonical residual authority, training/evaluation configuration,
checkpoint selection, or existing data. Do not invoke experiment campaign
entry points merely to produce browser data.

```text
browser (localhost:5173)
  React dashboard / R3F Canvas / Drei OrbitControls
  GET /api/health
       |
       | Vite development API proxy
       v
FastAPI (127.0.0.1:8000)
  validated local configuration / structured error handlers
  GET policy/scenario catalogs; bounded replay jobs; validated replay metadata
  one sequential worker -> existing Phase 2 CLI subprocess per episode
       |
       +-- original research worktree: read frozen artifacts only
       +-- visualizer/data/local: CLI-generated replay files (gitignored)

catalog inspection / replay execution (configured research Python)
  python -m visualizer.backend.replay_cli
       |
       v
  replay_engine.run_replay_episode
       |
       +-- canonical build_canonical_env_config + SoldierEnv
       +-- existing policy wrappers and frozen artifact catalogs
       +-- one validated EpisodeTrace -> atomic .json.gz
```

There is no database, cloud service, auth service, external control interface,
WebSocket stream, distributed queue, or browser-side simulation. A bounded
single-worker executor permits one active batch; episodes run sequentially in
fresh configured-Python child processes. During playback, the browser reads
only an already-validated full replay and never calls policy inference.

## Backend surfaces

- [main.py](../backend/main.py): app factory and lifespan; `GET /api/health`.
  Only localhost hosts are accepted; startup scripts bind IPv4 loopback.
  Health distinguishes API liveness from data-root configuration and never
  asserts checkpoint readiness.
- [config.py](../backend/config.py): frozen Pydantic settings; dotenv at an
  explicit location; process env takes precedence. Data root must contain
  `uav_defend/envs/soldier_env.py`. Output and input roots cannot contain each
  other. Output cannot resolve outside the visualization worktree.
- [data_access.py](../backend/data_access.py): internal relative-path resolver,
  traversal/drive/UNC/alternate-stream rejection, resolved containment including
  symlinks, read-only bytes and optional expected SHA-256 verification.
  No HTTP route exposes this helper; HTTP selections use allowlisted catalog
  IDs, not manifest paths supplied by a browser.
- [errors.py](../backend/errors.py): `{error: {code, message}}`, status codes,
  explicit validation/application failures and logged unexpected exceptions
  without leaking private stack traces in responses.
- [replay_contract.py](../backend/replay_contract.py): authoritative Pydantic
  v1 replay definitions; the single-episode JSON Schema is generated from these
  definitions and validated on read/write.
- [replay_engine.py](../backend/replay_engine.py): source-commit-pinned adapter,
  allowlisted checkpoint resolution/hash checks, deterministic inference,
  native-state/frame/event recording, semantic trace validation and atomic
  compressed output. It is a thin recorder, not new simulation dynamics.
- [replay_cli.py](../backend/replay_cli.py): standalone one-episode command.
  It has no arbitrary checkpoint or output-path input; output is a stem under
  the visualization application's ignored `data/local` directory.
- [replay_catalog.py](../backend/replay_catalog.py): reads frozen registries and
  hashes through the audited allowlist, and resolves canonical measurement and
  Kalman scenario configuration values through the actual scenario builder.
  Unavailable checkpoints remain visible with a reason; checkpoint paths are
  never returned by HTTP.
- [replay_jobs.py](../backend/replay_jobs.py): accepts validated bounded batches,
  rejects a second active batch, invokes the Phase 2 CLI sequentially, verifies
  each completed trace, and atomically persists opaque job metadata below
  `data/local/jobs`. Restarted queued/running jobs become `interrupted`.
- [replay_validation.py](../backend/replay_validation.py): full trace invariants
  shared by the Phase 2 writer/reader and Phase 3 API lookup/listing.

The Phase 3 API surfaces `GET /api/policies`, `GET /api/scenarios`,
`POST /api/replay-jobs`, job GET/cancel, replay list, full replay, and compact
summary endpoints. See [replay_api.md](replay_api.md) for exact payloads and
error behavior. Historical local artifacts that do not satisfy the current
schema/trace contract are excluded from listing with an explicit warning; they
are never silently presented as current valid replays or modified in place.
- [application_error.py](../backend/application_error.py): shared expected
  application failures, importable without FastAPI so the CLI can run in the
  existing research Python environment. API handlers retain the same errors.

Local dotenv and generated output are ignored. Never call the original
`build_model_registry()` or campaign writers from the API: they write into
scientific `results`. Existing manifests contain machine-specific absolute
paths; the adapter derives approved research-relative paths from audited
catalog identities, enforces containment, and compares hashes. No fallback to
an unrelated sibling repository, unregistered final checkpoint, or newly
generated artifact.

## Frontend scene and playback

[Battlefield.tsx](../frontend/src/Battlefield.tsx) owns the React Three Fiber
scene. With no selection it renders the ground plane, world grid, origin axes,
lighting, a resize-aware perspective camera, Drei orbit/pan/zoom controls, and
the explicit **No episode loaded** overlay. With a selected episode it renders
the recorded defender, attacker, asset, exact static AABBs, per-frame moving
obstacle spheres, 3D detection sphere, event markers and efficient trajectory
lines. Low-poly UAV/asset models are decorative skins around the recorded
position; no rotor, bank, pitch, or vehicle dynamics are simulated.

[ReplayPlayback.tsx](../frontend/src/ReplayPlayback.tsx) owns the single
simulated-time clock, exact-frame seeking, presentation-only interpolation,
play/pause/loop/speed controls, trail visibility and camera mode. Free and
follow cameras are perspective views; top-down switches to an orthographic
camera. Follow motion uses the same interpolated current timestamp as the
entities. Camera reset returns to an auto-fit view in each mode. Episode traces
are fetched lazily from the existing validated replay endpoint and are cleared
before a new episode becomes visible.

[App.tsx](../frontend/src/App.tsx) retains catalog-driven experiment setup,
job polling/cancellation, results, summary and provenance. `Ground Truth`
shows privileged ranges and outcomes; `Policy-Visible` masks them from the
scene, event overlays, chart markers, inspector, and result summaries. It
retains experiment-selection controls and identifies the scene as a partial
reconstruction because exact local sensing geometry is absent from the replay.
Interpolated positions are not reported as extra observations. Missing nullable
metadata remains unavailable rather than being reconstructed. WebGL loss,
trace validation, and backend failures are surfaced and do not hide other
selectable replays. No external assets or render services are used.

## Phase 5 policy diagnostics

[PolicyDiagnostics.tsx](../frontend/src/PolicyDiagnostics.tsx) displays the
selected episode's controller metadata, pre-step decision telemetry, action
disposition, recorded safety arbitration, and controller-specific diagnostics.
The current simulated time and the inspector/chart cursor share the playback
clock. Recorded command data on frame `k+1` is labeled as the command selected
at step `k` and applied during transition `k → k+1`; the terminal state has no
upcoming decision. Smooth position interpolation does not invent additional
policy decisions.

[diagnostics.ts](../frontend/src/diagnostics.ts) performs pure derivations over
the immutable episode when it is selected. `App.tsx` memoizes these series once
per episode and shares them with the inspector and chart panel. The 974-frame
LR-PPO trace therefore produces one set of decision-time diagnostic arrays,
not a set of React/Three.js objects per animation frame. The scene updates
mutable Three.js vectors in `useFrame`, memoizes geometry/material resources,
and disposes line and cone resources when their replay components unmount.

The diagnostics separate three quantities: the requested policy vector, the
environment's desired/safety-adjusted command, and actual post-transition
velocity. An observed requested-to-velocity angle is not attributed solely to
obstacle safety because acceleration/turn/climb constraints also affect motion.
Only the recorded `avoidance_active` flag is presented as a verified safety
override. See [policy diagnostics](policy_diagnostics.md) for the telemetry
coverage matrix, formulas, derivations, limitations, and validation evidence.

See [3D playback](3d_playback.md) for coordinate/time conventions, camera,
event, interpolation and long-trace behavior.

## Phase 6 synchronized comparison

[ComparisonExplorer.tsx](../frontend/src/ComparisonExplorer.tsx) is lazy-loaded
only after the user selects Compare Policies. It reads the same live policy,
scenario, replay-index, full-trace, and job APIs as the single-episode explorer.
There is no comparison-specific simulator, inference code, backend process
manager, persistence format, or HTTP endpoint. Existing exact checkpoint-role
and SHA-256 selections are resolved against the policy catalog; replay matching
also requires the frozen attacker identity/hash, scenario, perception, and
evaluation seed.

Both independent replay requests must yield one schema-v1 episode. The pair
validator requires equivalent simulator source revision, Python and scientific
package versions, model device, coordinate/presentation contract, timestep,
canonical scenario and physical parameters. The only allowed physical-config
hash difference is the declared `use_kalman_tracking` estimator setting; its
presence makes the pair exploratory and is surfaced to the researcher.
Checkpoint roles/hashes are exact, deterministic inference must be recorded,
and initial entity plus obstacle geometry is compared from the first recorded
state. Runtime, scenario, attacker, checkpoint, initial-state, or schema
mismatches reject the pair. Dirty simulator or visualizer source provenance
downgrades a compatible pair to exploratory rather than claiming a controlled
replication.

The shared comparison clock maps its axis independently into each episode's
recorded timestamps. Absolute mode uses simulation seconds from each reset;
detection mode subtracts each trace's recorded first-detection time and reports
both local times. Sampling uses the shared Phase 4 display-only interpolator,
clamped to each trace's terminal duration. A shorter episode remains frozen at
its own final state while the clock continues to the longer terminal time.
Events, trajectories and chart samples come from each pane's own bundle;
neither pane receives the other's attacker path or observations.

Two `Battlefield` instances reuse the same renderer with a common union bound.
Linked free/top-down cameras publish a world-space pose from one OrbitControls
instance to the other with change suppression; independent cameras and entity-
local follow modes remain available. Each pane receives its own readout,
telemetry series, policy inspector and display-information mode. Ground Truth /
Policy-Visible is one shared display-policy toggle; it masks privileged ranges,
outcomes, collisions and attacker truth in both panes.

Comparison jobs are front-end orchestration over existing single-episode job
requests. The UI first reuses indexed valid traces, then starts missing A/B
rollouts sequentially only after explicit confirmation. The API's existing
single-active-job guard, timeout, cancellation, atomic artifact persistence and
interrupted-job recovery remain authoritative. A failure leaves prior replay
artifacts indexable for later pairing; incomplete rows are not called matched.

See [policy comparison](policy_comparison.md) for pair-status semantics,
attacker/asset coupling limitations, chart formulas, detection alignment,
camera synchronization, divergence events, live test results and evidence.

## Phase 7 case persistence and exports

`case_library.py` stores versioned JSON sidecars below the ignored local data
root. Each sidecar references validated replay IDs and compressed-artifact
SHA-256 values; it does not copy or mutate replay bundles. Case deletion only
removes the sidecar. Every export/capture reloads the referenced bundle and
checks its recorded hash. Bookmarks preserve shared-axis and replay-local time,
frame/decision identity, synchronization mode and researcher-entered notes.

`scientific_exports.py` produces case reports, recorded-value CSVs, SVG charts,
ZIP packages, and validated capture artifacts. The browser rasterizes exported
SVGs; the API checks source hashes, dimensions, aspect ratio, PNG checksums and
300-DPI metadata before persisting chart PNGs. A native scene screenshot comes
from an on-demand WebGL render target and is accepted only for a replay/hash
referenced by the case. Export files remain in the ignored local output root.

The React case library is separate from the simulator configuration flow.
While the case page is displayed, the single-episode workspace remains mounted
but hidden so Drei `Html` portal roots are not torn down on each case/capture
view switch. Its render loop is paused while hidden and resumes when the single
workspace is active. Capture restores an actual saved replay into that view,
pauses it, and saves the PNG with time/camera/projection metadata. The Three.js
renderer's global drawing-buffer behavior is unchanged.

Phase 7 documentation and evidence: [case library](research_case_library.md),
[scientific exports](scientific_exports.md). Three saved exploratory cases
reuse four distinct Phase 6 replay artifacts; Phase 7 generated no episodes.

## Coordinate discrepancy and approved resolution

The prompt assumed 2D, but the verified source explicitly defines a **native
3D constrained point-mass** environment. Actions and positions are `(3,)`;
altitude influences constraints, detection, interceptions and collisions.
Existing trajectory comments mentioning `T x 2` and an LR wrapper docstring
claiming no model has been trained are historical/stale, not authoritative.

Phase 4 requirement: **preserve and display the native 3D state.** The viewer
does not flatten altitude or apply fixed-height substitutions.

- Recording coordinate system: `[x, y, z]`, meters, simulation z-up.
- Central transform: Three.js `[X,Y,Z] = [x,z,-y]`, a right-handed rotation.
  It is applied to positions, vectors, AABB bounds, dynamic obstacle
  coordinates, events, trails and camera targets. Distances and vector angles
  are preserved; a simulator ground altitude `z=0` maps to scene `Y=0`.
- The replay-v1 `presentation: horizontal_projection_fixed_heights` marker is
  retained for schema compatibility. The Phase 4 viewer uses the native
  `position_m` values and does not replace recorded altitude with fixed heights.
- Static obstacles use the recorded AABB bounds and dimensions. Moving
  obstacles use each recorded position and radius. Low-poly appearance changes
  only the visual skin, never collision geometry.
- Horizontal heading uses recorded heading when present, otherwise recorded
  velocity/movement direction. If neither exists, orientation is derived from
  adjacent recorded positions and explicitly labeled display-only.
- Top-down mode uses an orthographic camera to project the native 3D replay
  onto the horizontal plane. It does not change replay data or simulator
  collision semantics.

No altitude control, pitch state, 3D collision code or new dynamics is introduced
by this application. Those already in the scientific source remain unchanged.

## Exact Phase 2 integration seams

| Task | Reused entry point / replay behavior |
|---|---|
| Canonical scenario | `experiments.journal_clutter_protocol.build_canonical_env_config(level, estimator, awareness_mode=...)`; replay instantiates the actual `SoldierEnv(config=cfg)` |
| Environment lifetime | `reset(seed=simulation_seed)`, policy `reset()`, one deterministic inference per selected policy at each pre-step state, `step(np.float32(action))`, stop on native `terminated or truncated`, `close()` in `finally` |
| Analytical policies | `uav_defend.policies.registry.get_policy`; Greedy/Kalman Greedy omit unsupported `config` constructor kwargs; PN/Lead use canonical `config`; RMPC uses frozen config |
| RMPC | `RMPCPolicy(config=cfg, rmpc_config=...)`; frozen selected `C1_FAST` parameters from `clutter_benchmark_protocol._frozen_rmpc_config()`, never class defaults |
| Learned policies | Exact training seed/generation/role and relative path are derived from the frozen registry or conference `FROZEN_MODELS`; registry path and SHA-256 are verified before/after inference; model obs/action shape is checked |
| Policy information | `build_policy_info(info, estimator)` is the actual policy sanitizer; full environment info stays in recorder-only `privileged_diagnostics.simulator_info` |
| Sector features | Existing `compute_sector_features(..., perceived=env._defender_last_perceived, sensor_range=SECTOR_OBSERVATION_V1.sensor_range)`; recorded separately as decision-time controller-visible data |
| PPO / PPO-Kalman | Existing wrapper `.load(path, deterministic=True)` and `.act(obs, sanitized_info)`; 20 existing sector features are concatenated only for clutter checkpoints, yielding 36-D model input |
| LR-PPO | Existing `LeadResidualPPOPolicyWrapper.load(..., residual_max_angle_deg=30.0, config=cfg, deterministic=True)`; `.act(base_16_obs, sanitized_info, extra_features=sector)` constructs the actual 19/39-D input |
| A0 | `env.enemy_tactical_override = None`; no learned tactical callback is injected |
| A1 | Existing `build_attacker_observation` yields 36-D; frozen `PPO.load` and `predict(..., deterministic=True)` once per pre-step state; pending action is passed through the existing override hook before `env.step` |
| A1 hook ordering | Same as `defender_adaptation_evaluation._run_matched_episode`: pending-action callback, reset, A1 decision, sanitized defender decision, then step |
| State | Full returned `reset/step` info includes authoritative positions/velocities, detection, tracking, obstacle and safety diagnostics; initial plus each transitioned state is recorded |
| Geometry | Static AABBs from `env._obstacle_layout.obstacles`; dynamic obstacle spheres from current `env._dynamic_obstacles`; geometry is read-only and never regenerated in the browser |

Private scientific access is already used in the journal harness; document that
coupling and test against source commit rather than changing SoldierEnv to make
the viewer easier. The generic trajectory utility is not a complete A1/clutter
telemetry recorder and its NPZ loader uses pickle; do not expose arbitrary NPZ
uploads. Existing plot-generating entry points write scientific directories and
must not be called from the app.

The adapter imports private simulator state only at this isolated boundary. It
does not change the research source. Tests include a real exact-seed comparison
against `_run_matched_episode`, deterministic repeat, and the source's native
per-step states/commands/obstacles. Training, large evaluation, checkpoint
reselection and data mutation are not performed.
