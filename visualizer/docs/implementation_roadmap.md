# Eight-phase implementation roadmap

Each phase requires an explicit request. Complete, test and commit Phase 1,
then stop. Later phases below are design boundaries, not implemented features.

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
- Preserve native 3D recordings; default to approved labeled horizontal projection.
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

## Phase 2 readiness

Source and artifact entry points are identified; 48 journal registry references
and 9 conference references were checked read-only. Phase 1 skeleton does not
deserialize models and intentionally keeps the API venv free of heavy scientific
dependencies. Phase 2 must select/verify a compatible research Python environment,
resolve frozen manifest identities under the configured data root, and validate
recording semantics beyond JSON shape. Native 3D/horizontal-projection discrepancy
is resolved by the researcher's explicit decision, not by changing the simulator.
