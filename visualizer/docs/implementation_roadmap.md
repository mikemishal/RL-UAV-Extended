# Eight-phase implementation roadmap

Phases 1–4 are implemented and verified. Phase 4 provides native-3D playback
from validated stored traces. Stop at this boundary; Phase 5 policy diagnostics
are intentionally not implemented.

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

## 5. Policy diagnostics and LR-PPO angular visualization

- Inspector separates legitimate observations from privileged evaluator truth.
- Capture available Lead/PN/RMPC diagnostics without duplicate stateful decisions.
- Show PPO requested/safe/executed actions, Kalman estimates/errors and heading changes.
- Verify observational LR residual capture against canonical composition;
  three guidance arrows, bounded native 30-degree cone, separately labeled signed
  horizontal angle, residual clipping/saturation vs safety/dynamics distinction.
- Null unavailable quantities with explicit reasons; never fabricate explanations.

## 6. Side-by-side policy comparison

- Replay matched seeds/environment snapshots and explicit attacker/checkpoint roles.
- Synchronized views and trajectories with independent camera options.
- Clearly separate native outcomes from projected geometry and existing journal claims.

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
- Six real-data Chromium screenshots are captured under ignored
  `visualizer/frontend/test-results/`; the automated pure helper tests cover
  coordinate/distance/angle preservation, AABB transforms, camera bounds,
  exact frame selection, interpolation and immutable replay frames.
- Phase 5 is not started: no policy-specific arrows, LR-PPO vectors, Kalman
  estimates, residual-angle displays, safety-intervention explanations, or
  other diagnostic overlays were added.

See [3D playback documentation](3d_playback.md) for implementation details and
known visualization limitations.
