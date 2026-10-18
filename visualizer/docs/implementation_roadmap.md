# Eight-phase implementation roadmap

Phases 1–6 are implemented and verified, including live API/Chromium validation
of synchronized comparisons using real replay traces. Phase 7 case-library,
figure-export and research-report features have not started.

## 1. Setup and architecture

- Isolate visualization worktree from verified latest local research commit.
- Audit policies, wrappers, canonical config, checkpoints and actual coordinates.
- Establish localhost FastAPI health/configuration/error and read-only data boundary.
- Create React/TypeScript/Vite/R3F/Drei empty battlefield with camera controls.
- Define replay v1 and policy-specific telemetry without generated episodes.
- Document setup, compatibility and scientific preservation; test and commit.

## 2. Actual Python replay engine

- Implement a small dedicated adapter over existing SoldierEnv and policy wrappers.
- Read registry-selected checkpoints from the research data root, verify hashes,
  spaces, dependencies, source identity and model requirements before inference.
- Use canonical builder and sanitized policy info; A0 None / deterministic A1 hook.
- Record reset through terminal frame, native 3D state, perceived/privileged data,
  static/dynamic geometry, exact outcome and provenance under local output only.
- Test determinism and no scientific data mutation with a minimal small-episode
  verification, not a training/evaluation campaign.
- Do not call scientific campaign writers or silently fall back checkpoint roles.

## 3. Policy catalog and multi-episode API

- Serve allowlisted policy/checkpoint IDs and validated compatibility metadata.
- Reject arbitrary paths, incompatible estimators/shapes/generations and missing data.
- Add bounded 1/3/5/10 episode jobs, default 5 consecutive simulation seeds,
  explicit parameters/provenance, progress/failure reporting and local output IDs.
- Preserve held-out A1-203 labeling and primary vs secondary checkpoint roles.

## 4. Interactive 3D animation and playback

- Render real Python frames with blue defender, red attacker, green asset,
  native obstacle geometry, decorative low-poly skins and distinct moving clutter.
- Preserve and render native 3D recordings. Offer an explicit orthographic
  top-down camera without altering or flattening the episode data.
- Add trajectories, detection and recorded event markers, no browser dynamics.
- Add play/pause/step/scrub/speed/replay/episode navigation and final summaries.
- Free orbit, defender follow, attacker follow, accurate top-down geometry;
  projected overlap is not evidence of native collision.

## 5. Policy diagnostics and LR-PPO angular visualization — complete

- Inspector separates policy-visible estimates/measurements from privileged
  evaluator truth and follows the recorded decision-to-frame interval.
- Shows available Lead/PN/RMPC telemetry and common requested/safe/executed
  command records without another policy inference.
- Kalman estimates and velocities are shown only after detection; derived
  estimate-to-truth errors are researcher-only in Ground Truth mode.
- LR-PPO shows Lead, final requested direction, actual post-transition velocity,
  native 3D angle, the source-verified 30° cone, and separate signed azimuth.
  Raw PPO residual and internal clipping remain explicitly unavailable.
- Ground Truth and partial Policy-Visible modes mask privileged data
  consistently. Synchronized charts seek to recorded timestamps only.
- Numerical, timing, masking, chart-seeking, state-reset, immutability and
  local real-trace browser checks are implemented. No scientific behavior or
  evaluation data changed.

## 6. Side-by-side policy comparison

- Resolve available policies, generations, seeds, and checkpoint roles/hashes
  from the live catalog. Share and validate attacker, canonical scenario,
  perception, and matched evaluation seeds.
- Reuse exact replay pairs; only after explicit user action generate missing A/B
  traces sequentially through the existing single-active-job API.
- Reject incompatible source/runtime, scenario/physical configuration,
  attacker/checkpoint, schema, or initial-world state. Downgrade dirty or
  estimator-different evidence to exploratory, never claim coupled future
  attacker/asset trajectories from a common seed.
- Render two independent authoritative replay states with one absolute or
  detection-relative playback axis, terminal clamping, linked/independent
  cameras, per-pane diagnostics, charts, outcome summaries, and event jumps.
- Keep the browser read-only with respect to simulator behavior and model
  inference. Phase 7 case libraries and export/report generation remain out of
  scope.

## 7. Episode research tools and exports

- Local annotations, bookmarks, trace inspection and versioned evidence exports.
- Include source commits, hashes, seed/config identities and diagnostic availability.
- Safe names/catalog IDs only; outputs remain outside frozen experiment directories.

## 8. Comprehensive testing and final local packaging

- Integration, numerical/coordinate, provenance, path-boundary and failure tests.
- Browser rendering, resize, keyboard/control, playback timing and camera regression.
- Minimal verified real-checkpoint sample without new scientific campaigns.
- Reproducible local Windows dependency/startup package, troubleshooting and final audit.
- No cloud deployment, external control service or scientific source modifications.

## Phase 2 completion / scientific boundary

The source and artifact entry points were audited read-only; the Phase 2 engine
uses the configured existing research Python runtime, resolves frozen manifest
identities under the configured data root, verifies checkpoint hashes, and
validates recording semantics beyond JSON shape. The API venv remains free of
heavy scientific dependencies. Native 3D/horizontal-projection discrepancy is
resolved by the researcher's explicit decision, not by changing the simulator.

## Phase 3 completion / stop boundary

- Registry-backed policy and scenario catalogs expose verified availability,
  generation roles, training seeds, observation requirements, and canonical
  scenario configurations without checkpoint paths.
- A Pydantic request validates policy/generation/seed combinations, A0/A1
  selections, C0–C3, perception mode, allowed counts (1/3/5/10), deterministic
  inference, and consecutive evaluation seeds in the simulator's accepted range.
- A single sequential worker invokes the Phase 2 CLI in the configured research
  Python subprocess. One active batch is allowed; status and health remain
  available while simulation runs. Timeout, cancellation, and restart recovery
  do not become simulation outcomes.
- Job state is atomically persisted under ignored `visualizer/data/local/jobs`.
  Replay lookup is opaque-ID-only and reuses full schema and semantic validation.
  Invalid historical artifacts remain unchanged and are excluded from the
  current replay listing with an explicit warning.
- The browser provides functional selectors, progress, cancellation, result
  selection, summaries, and provenance. The Three.js scene has no episode
  entities or fabricated movement; animation and diagnostic overlays are not
  implemented.
- Verified real batch: Greedy / A0 / C0 / `map_local`, seeds 71030000–71030002:
  outcomes `intercepted`, `soldier_caught`, `intercepted`, with 117, 655, and
  26 steps. Seed 71030000 matched a direct Phase 2 CLI run for summary, events,
  native positions, and telemetry actions.
- Verified learned request: LR-PPO D1 seed 77 / held-out A1-203 / C3 /
  `map_local`, evaluation seed 71030010; selected defender role `final` and
  attacker role `attacker_best`; one validated episode completed.
- The eight terminal outcomes preserved from SoldierEnv are `intercepted`,
  `soldier_caught`, `unsafe_intercept`, `timeout`,
  `defender_static_obstacle_collision`,
  `defender_dynamic_obstacle_collision`, `enemy_static_obstacle_collision`,
  and `enemy_dynamic_obstacle_collision`. `ongoing` is used only for non-final
  frames.
- Verified validation: 41 backend/API unit tests passed (5 skipped without
  opt-in scientific runtime configuration), all 3 real catalog/replay
  integration tests passed, all 6 browser tests passed, the production frontend
  build passed, and 48 focused scientific regression tests passed. Descriptive
  batch statistics are exploratory and not publication-grade estimates.

## Phase 4 completion / stop boundary

- The selected replay is fetched lazily from the validated full-trace endpoint.
  The frontend reads recorded frames only and has no inference or simulation
  API calls during playback.
- One simulated-time clock drives all entities and moving obstacles. Play,
  pause, restart, exact frame stepping, timeline scrubbing, jump-to-start/end,
  and 0.25×–8× simulated-time speeds are available. Playback stops at the
  recorded terminal time unless the user explicitly enables looping.
- Interpolation is display-only, clamped to adjacent frames in the selected
  episode, and never extrapolates beyond its terminal frame. Paused frame
  stepping and the information panel use authoritative recorded samples.
- The native right-handed mapping `(x,y,z) → (x,z,-y)` applies to entity and
  obstacle positions, vectors, AABB geometry, event positions, trails and
  camera targets. Distances and vector angles are tested.
- The scene renders low-poly UAV/asset models at native recorded positions,
  exact static AABBs, recorded dynamic obstacle spheres, a 3D detection sphere,
  event markers, and trajectories up to the current time. Future positions are
  hidden unless full-trajectory viewing is enabled.
- Free perspective orbit, defender/attacker follow, orthographic top-down,
  reset camera, obstacle display options and per-trail visibility are available.
  The attacker is labeled as privileged researcher visualization before
  recorded detection.
- Verified real browser data: Greedy/A0/C0 (282 frames),
  LR-PPO D0/A0/C2 (170 frames), PPO-Kalman D0/A1-203/C3 (330 frames), and
  LR-PPO D1 seed 77/A1-203/C3 (974 frames). Browser checks compare displayed
  frame-zero coordinates to the API and scrub the long trace to an interpolated
  midpoint and terminal frame. No episodes were regenerated.
- Phase 4's real-data Chromium screenshots are captured under ignored
  `visualizer/frontend/test-results/`; the automated pure helper tests cover
  coordinate/distance/angle preservation, AABB transforms, camera bounds,
  exact frame selection, interpolation and immutable replay frames.

See [3D playback documentation](3d_playback.md) for implementation details and
known visualization limitations.

## Phase 5 completion / stop boundary

- Added a collapsible inspector showing controller/generation/seed/role,
  current recorded state/time, detection, policy commands, safety-adjusted
  command, actual post-transition velocity, reward and available parameters.
- Telemetry on frame `k+1` represents the command chosen at step `k` and applied
  during `k → k+1`. Standby is not reported as applied, and terminal time has
  no fabricated next command.
- LR-PPO arrows identify Lead, final requested direction and actual velocity
  separately. Their native 3D angle is calculated from recorded vectors; the
  cone is the verified 30° spherical correction allowance. Raw residual and
  clipping internals are unavailable in the existing wrapper.
- PPO-Kalman/measurement channels are detection-conditioned and separate from
  privileged truth. Position/velocity tracking errors are explicitly derived.
- Safety markers use recorded arbitration flags; requested-to-velocity
  deviation is not attributed only to safety, since constrained dynamics also
  affect movement.
- Ground Truth is the default. Policy-Visible hides privileged attacker
  trajectories, events, ranges, outcomes, reward and provenance, and is
  explicitly labeled a partial reconstruction because full local sensing
  geometry is not in the replay. Both modes change display only.
- Chart cursor/seek follows the playback clock; arrays are precomputed once
  per loaded episode and samples remain at actual recorded frame/decision times.
- Existing replay artifacts were browser-checked: Greedy/A0/C0, LR-PPO D0/A0/C2,
  PPO-Kalman D0/A1-203/C3, and LR-PPO D1/A1-203/C3 (974 frames). Six Phase 5
  screenshots are captured without generating new episodes.
- Coverage and validation details: [policy diagnostics](policy_diagnostics.md).
- Phase 5 completion remains the single-episode boundary documented above.

## Phase 6 completion / stop boundary

- The Compare Policies mode independently selects Policy A/B from the live
  catalog, including generation, training seed, exact checkpoint role/hash,
  architecture and compatibility metadata. Shared attacker checkpoint,
  clutter/perception, evaluation seed and batch size use the canonical API
  contracts; the default batch is one matched seed.
- Replay pairing indexes the existing catalog by policy/checkpoint identity,
  attacker identity/hash, scenario, perception, seed and provenance. Full
  replay schema/integrity and initial entity/obstacle geometry are checked
  before a pair is shown as matched or exploratory. Python/scientific package
  versions and inference device must also agree. Estimator-only differences
  are disclosed; dirty sources are exploratory.
- The browser exposes an explicit Generate Missing Replays action and
  sequentially reuses the existing one-active-job API. It does not duplicate
  the process manager or run missing rollouts on dropdown change. Existing
  successful artifacts remain available if a later requested rollout fails.
- Both native-3D panels reuse `Battlefield`; each reads its own authoritative
  replay. Absolute simulation time is the default; optional first-detection
  alignment uses separate local times. Shorter episodes clamp at the recorded
  terminal frame while the shared clock continues to the later terminal time.
  Linked world-space free/top-down cameras suppress echo updates; entity
  follow modes remain pane-local.
- Phase 5 inspectors/diagnostic overlays and comparative charts are reused.
  Policy-Visible mode masks privileged data in both panes. Path length is the
  sum of Euclidean distances over recorded coordinates; divergence is tested
  only at common recorded timestamps and is descriptive, not causal. Missing
  metrics remain unavailable, and exploratory batch counts are not inferential.
- Live FastAPI/Vite and Chromium tests loaded both real traces from the API,
  verified frame-zero coordinates, shared stepping/playback, scrub seeking,
  terminal freeze, detection-relative mapping, camera link/unlink, and
  policy-visible masking. A separate bounded generation flow produced the
  three missing requested seed-0 traces through the existing job API.
- Pair A — LR-PPO D0 seed 77 vs D1 seed 77, A1-203/C3/map-local/seed 0:
  D0 used the `d0_start` checkpoint and terminated `soldier_caught` at
  760.0 s; D1 used `final` and was `intercepted` at 486.5 s.
- Pair B — LR-PPO D1 seed 77 vs PPO-Kalman D1 seed 74, same attacker,
  C3/map-local/seed 0: both were intercepted (486.5 s vs 442.5 s). The
  Kalman estimator configuration differs by its declared tracking flag.
- Pair C — Lead vs LR-PPO D1 seed 77, same attacker, C3/map-local/seed 0:
  Lead terminated `soldier_caught` at 27.0 s; LR-PPO was `intercepted` at
  486.5 s.
- Each requested pair reused the existing LR-PPO D1 replay; exactly three
  missing defender replays were generated. Every inspected trace reports dirty
  source provenance, so these cases are explicitly exploratory, not clean
  controlled comparisons. Matching reset conditions do not couple later
  learned-attacker, asset, or random trajectories.
- Eight screenshots from real validated data are listed in
  [policy comparison](policy_comparison.md#live-verification-and-evidence).
- The single-episode explorer remains available. Phase 7 case library,
  figure exports and automated research-report generation have not begun.
