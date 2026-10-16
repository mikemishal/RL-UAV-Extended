# Architecture / Phase 1

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
  replay contract definitions only
       |
       +-- original research worktree: read frozen artifacts only
       +-- visualizer/data/local: future generated recordings (gitignored)
```

There is no database, cloud service, auth service, external control interface,
WebSocket stream, worker queue or running simulation in Phase 1.

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
- [replay_contract.py](../backend/replay_contract.py): Pydantic definitions,
  not a generator. Checked-in schema is generated from these definitions.

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

## Exact integration seams for Phase 2 (not implemented)

| Task | Existing entry point / required ordering |
|---|---|
| Canonical scenario | `experiments.journal_clutter_protocol.build_canonical_env_config(level, estimator, awareness_mode=...)`; instantiate `SoldierEnv(config=cfg, render_mode=None)` |
| Environment lifetime | `reset(seed=simulation_seed)`, policy `reset()`, deterministic inference, `step(np.float32(action))`, stop on `terminated or truncated`, `close()` in `finally` |
| Simple policies | `uav_defend.policies.registry.get_policy`; RMPC and LR-PPO are not registry entries |
| RMPC | `RMPCPolicy(config=cfg, rmpc_config=...)`; read frozen `results/rmpc_tuning/selected_rmpc_config.json`, require frozen marker, use selected values rather than class defaults |
| Clutter learned defenders | `defender_adaptation_evaluation.build_defender_instances` demonstrates exact estimator/feature requirements and registry roles; do not call its writer or bulk `evaluate_all` |
| Policy information | `build_policy_info(info, estimator)` from sanitizer; full `info` stays recorder-only |
| Sector features | `compute_sector_features(..., perceived=env._defender_last_perceived, sensor_range=SECTOR_OBSERVATION_V1.sensor_range)`; never feed ground-truth obstacle layout to policy |
| PPO / PPO-Kalman | Wrapper `.load(path, deterministic=True)` then `.act(obs, sanitized_info)`; concatenate 20 sector features for 36-D clutter checkpoints |
| LR-PPO | `LeadResidualPPOPolicyWrapper.load(..., residual_max_angle_deg=30.0, config=cfg, deterministic=True)`; `.act(base_16_obs, sanitized_info, extra_features=sector)` internally constructs 39-D clutter input; no external duplicate concatenation |
| A0 | `env.enemy_tactical_override = None`, not an alternate browser/Python attacker |
| A1 | `build_attacker_observation(env, cfg.enemy_defender_detection_range)` yields 36-D; frozen `PPO.load` and `predict(..., deterministic=True)` once per decision; set pending action through `env.enemy_tactical_override` before `step` |
| A1 hook ordering | Mirror `defender_adaptation_evaluation._run_matched_episode`: install callback referencing pending action, reset, compute A1 at current state, compute defender at same pre-step state, then step. Do not substitute the training mixture wrapper for an explicitly selected attacker |
| State | `reset/step` full `info`: `soldier_pos`, `defender_pos`, `enemy_pos`, velocities, detection/tracking and safety fields; `_get_obs()` / `_get_info()` available where established harness uses them |
| Static / dynamic geometry | `env._obstacle_layout.obstacles` and `env._dynamic_obstacles`; private access should be confined to one visualizer adapter, not scattered through HTTP handlers |
| Existing trajectory precedent | `experiments.eval_utils.Trajectory`, `run_episode_with_trajectory`; A1-aware `defender_adaptation_trajectory_analysis._run_episode_with_trajectory` and `adaptive_attacker_trajectory_analysis._run_episode_with_trajectory` |

Private scientific access is already used in the journal harness; document that
coupling and test against source commit rather than changing SoldierEnv to make
the viewer easier. The generic trajectory utility is not a complete A1/clutter
telemetry recorder and its NPZ loader uses pickle; do not expose arbitrary NPZ
uploads. Existing plot-generating entry points write scientific directories and
must not be called from the app.

Phase 2 should retain one isolated recording adapter with no scientific edits.
Read-only source imports and small targeted regression tests are permitted;
training, massive evaluation, checkpoint reselection and data mutation are not.
