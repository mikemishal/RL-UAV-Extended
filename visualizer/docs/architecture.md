# Architecture / Phase 2

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
  read-only internal research artifact access (no public file endpoint)
  health only; no replay endpoint in Phase 2
       |
       +-- original research worktree: read frozen artifacts only
       +-- visualizer/data/local: CLI-generated replay files (gitignored)

manual local invocation (research Python)
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
WebSocket stream, worker queue, REST replay endpoint, or browser-side simulation.
The React application remains the Phase 1 empty 3D scene.

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
  No HTTP route exposes this helper. Future APIs must accept allowlisted catalog
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
- [application_error.py](../backend/application_error.py): shared expected
  application failures, importable without FastAPI so the CLI can run in the
  existing research Python environment. API handlers retain the same errors.

Local dotenv and generated output are ignored. Never call the original
`build_model_registry()` or campaign writers from the API: they write into
scientific `results`. Existing manifests contain machine-specific absolute
paths; Phase 2 must derive approved research-relative paths from the audited
catalog identity, enforce containment, and compare hashes. No fallback to an
unrelated sibling repository, final checkpoint, or newly generated artifact.

## Frontend scene

[Battlefield.tsx](../frontend/src/Battlefield.tsx) contains a real R3F Canvas:
100 m square ground, 5 m coordinate grid, origin axes, ambient/directional light,
perspective camera, demand-driven rendering, resize-aware canvas, orbit/zoom/pan,
and camera reset through Drei controls. Geometry is static infrastructure only.
Orbit inertia is disabled so reset returns immediately to the saved camera,
without pending damping deltas moving it again.
The empty-state overlay ignores pointer events so camera interaction remains
available. A WebGL fallback/error boundary and context-loss notice avoid a
success-looking blank viewport. No models, episode arrays or moving placeholders
exist. No external assets, fonts or network rendering dependencies are needed.

[App.tsx](../frontend/src/App.tsx) owns the responsive research layout and health
status. Configuration, playback and inspector areas are clearly marked as
future functionality; forms are disabled. Default planned batch is five; choices
are 1/3/5/10. A backend failure is visible and never replaced with fake outcomes.

Planned camera modes: free orbit (implemented), defender follow, attacker
follow, and orthographic top-down (Phase 4). Reset is functional now; follow
and top-down controls are not falsely offered as working.

## Coordinate discrepancy and approved resolution

The prompt assumed 2D, but the verified source explicitly defines a **native
3D constrained point-mass** environment. Actions and positions are `(3,)`;
altitude influences constraints, detection, interceptions and collisions.
Existing trajectory comments mentioning `T x 2` and an LR wrapper docstring
claiming no model has been trained are historical/stale, not authoritative.

Researcher decision in this session: **preserve native 3D data; offer a clearly
labeled horizontal projection with fixed display heights.**

- Recording coordinate system: `[x, y, z]`, meters, simulation z-up.
- Projection: Three.js `[X,Y,Z] = [x, h_visual, -y]`.
  Simulation +x maps to scene +X and simulation +y maps to scene -Z;
  this preserves orientation when viewed from above in a y-up Three.js scene.
- Future fixed visual entity centers: defender/attacker `h_visual=1.5 m`,
  asset `h_visual=0.5 m`. These are display offsets, not recorded altitudes.
- Native alternative, if explicitly enabled later: `[x,z,-y]`; never rewrite
  source state. Ground entity remains simulation z=0.
- Native obstacle bounds/radii and all collision evidence are preserved.
  Buildings/trees are low-poly skins, not alternative collision models.
  Static simulator geometry is AABB; `map_known` is metadata, not a proven
  building/tree taxonomy. Any tree skin must be labeled decorative.
- Heading in the horizontal plane is `atan2(v_y,v_x)` with CCW-positive
  simulation convention; zero horizontal speed yields undefined/null heading.
- Projected overlap is not collision, and projected angular change is not the
  bounded native LR-PPO angular deviation.

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
