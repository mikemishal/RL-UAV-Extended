# Counter-UAS 3D Simulation Explorer

An isolated local research visualizer with validated replay jobs,
interactive native-3D playback, policy-specific diagnostics, synchronized
side-by-side comparison, a persistent research-case library, and scientific
exports derived from recorded episodes.
Comparison eligibility is checked against checkpoint hashes, attacker identity,
resolved physical configuration, initial world state, simulator/runtime
provenance, and estimator contracts. The browser never simulates, trains,
performs policy inference, or changes recorded outcomes.

## Scientific integrity and coordinate decision

Based on local research commit `8e2d59115423d329156d5bc468bf594019f4dc03`
(branch `research/journal-corrected-attacker-v2`),
not a remote branch. All additions are confined to this `visualizer` directory.
Scientific Python, experiment configuration, results, weights and tags are unchanged.
The engine refuses to run against any other research commit
(`EXPECTED_RESEARCH_COMMIT` in `backend/replay_engine.py`); update that pin only
after auditing the new research source.

### Simulation semantics: V1 and V2

The research repository contains two simulation semantics, selectable per replay:

| | V1 (default) | V2 |
|---|---|---|
| Mission contact (attacker→asset capture, defender→attacker interception) | endpoint only (tested at each step's final positions) | swept: first contact over the whole step |
| Scripted attacker A0 | pursuit + weave + noise + defender evasion | direct pursuit; deviates only while an obstacle blocks the direct path |
| Research builder | `build_canonical_env_config` | `experiments.journal_v2_protocol.build_v2_env_config` |

The visualizer **implements neither**: it only passes the choice to the research
repository's own builders, then asserts that the resolved config really has the
expected switches (`mission_contact_mode`, `enemy_nominal_guidance_mode`) so the
label can never disagree with the simulator that ran. V1 is the default and is
unchanged (frame-for-frame identical to the previous visualizer).

* UI: **Simulation semantics** dropdown in the run form; the playback status bar
  always shows a `Semantics: V1/V2` badge, the recorded-episode list is prefixed
  `[V1]`/`[V2]`, and the episode/provenance cards repeat it.
* Links: `http://127.0.0.1:5173/?semantics=V2&clutter=C2&seed=130065` pre-fills
  the form (invalid values fall back to V1 / C0 / seed 0).
* CLI: `python -m visualizer.backend.replay_cli ... --semantics V2`.
* API: optional `"semantics": "V1" | "V2"` on `POST /api/replay-jobs` (default `V1`).
* Replays recorded before the selector existed load as V1.
* In Ground Truth mode the status bar also shows the recorded attacker mode
  (`DIRECT`, `OBSTACLE_AVOID (climb, static #7)`, …), attacker–asset range, and
  for V2 the swept contact (`capture at τ = 0.42 of the step`). These are read from
  recorded simulator state only.
* The Comparison Explorer works with V1 replays only.

**The simulator and viewer preserve native 3D.** `SoldierEnv` records altitude,
three-component actions, climb/descent limits, and 3D collision/detection.
Coordinates map from simulator `(x, y, z)` to Three.js `(x, z, -y)` without
flattening altitude. The top-down camera is an orthographic projection for
accurate horizontal geometry; it does not alter the underlying replay.
See [architecture](docs/architecture.md) and [3D playback](docs/3d_playback.md).

## Windows setup

Prerequisites: Python 3.11+, Node 22.12+, npm, and a WebGL2-capable browser.
The default workspace is the **Simulation Viewer**. First-time setup prompts
for the absolute path to the existing scientific Python runtime:

```powershell
.\visualizer\setup-visualizer.ps1
```

If required frontend/API packages are not installed, install them once:

```powershell
.\visualizer\setup-visualizer.ps1 -InstallApiDependencies -InstallFrontendDependencies
```

Normal start and safe shutdown:

```powershell
.\visualizer\start-visualizer.ps1
.\visualizer\stop-visualizer.ps1
```

The launcher verifies the exact configured interpreter, data root, model
registry, frontend dependencies, and both required ports before opening the
browser. It never terminates a pre-existing listener or silently falls back to
another Python. Its PID record is used by `stop-visualizer.ps1`; inspect
`visualizer\data\local\logs` for server output. The Vite proxy forwards `/api`
to FastAPI at `http://127.0.0.1:8000`; both services bind only to localhost.
For the episode and camera controls see [QUICK_START](QUICK_START.md).

If local execution policy blocks scripts, invoke them under the execution
policy approved for your machine; no script changes global execution policy.

### Health and data configuration

`VISUALIZER_RESEARCH_DATA_ROOT` and `VISUALIZER_SIMULATION_PYTHON` are read from
the process environment, then `visualizer\.env`. Relative data roots are
anchored to the `visualizer` directory. Explicit nonexistent, invalid, or
overlapping roots fail startup with a configuration error. An unset data root
permits catalog-independent health only (`status: degraded`,
`simulation_enabled: false`); policy/scenario and replay endpoints require a
configured research root. `status: ok` means the configured worktree is valid,
not that every model is available. The catalog performs registry/path/SHA-256
checks and marks missing or mismatched checkpoints unavailable.

`VISUALIZER_EPISODE_TIMEOUT_SECONDS` defaults to 900 (allowed 1–3600).
`VISUALIZER_FRONTEND_ORIGIN` defaults to `http://127.0.0.1:5173`. Both are
local configuration values; CORS remains restricted to the configured
localhost frontend origin.

Frozen artifacts are opened only as `rb` through internal `ResearchDataStore`.
Missing artifacts, read failures, and hash mismatches have explicit errors.
No browser endpoint accepts filenames, checkpoint paths, output paths or uploads.
The CLI accepts policy/scenario selections and an optional filename stem only;
it never accepts a checkpoint path or output path. Replay output is written to
gitignored `visualizer\data\local` only when the CLI is explicitly invoked.
Neither API startup nor the frontend writes recordings. The application does
not change filesystem ACLs; read-only behavior is enforced by the code
boundary, not a claim of OS-level isolation. An optional read-only OS
account/ACL adds defense in depth without changing research files.

## Tests and build

From the visualization worktree root:

```powershell
.\visualizer\.venv\Scripts\python.exe -m pytest -q --noconftest .\visualizer\tests
Set-Location -LiteralPath '.\visualizer\frontend'
npm ci
npm run typecheck
npm run build
npx playwright install chromium
npm run test:browser
```

Run the bounded real integration checks with the existing scientific Python
runtime selected explicitly (three Greedy/A0/C0 episodes plus one learned
LR-PPO D1/A1-203 episode):

```powershell
Set-Location -LiteralPath '.\visualizer'
$env:PYTHONPATH = '..'
$env:VISUALIZER_RESEARCH_DATA_ROOT = 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-lrppo-clutter'
$env:VISUALIZER_SIMULATION_PYTHON = (Get-Command python).Source
.\.venv\Scripts\python.exe -m pytest tests\test_replay_batch_integration.py -q
```

The browser suite starts Vite itself if needed, uses a real WebGL context
(software rendering is permitted), tests orbit/reset, policy/scenario selectors,
job progress and error states, responsive layout, and writes screenshots to
gitignored `frontend\test-results`. Run the backend separately for real API
integration. Vite production output is a build artifact, not a cloud deployment
or packaged API server.

The API venv deliberately installs only the API/test stack. Replay catalog
inspection and execution run as bounded subprocesses with
`VISUALIZER_SIMULATION_PYTHON`; that interpreter must already contain the
frozen scientific Gymnasium / SB3 / Torch dependencies. The API health endpoint
does not load models. The policy catalog verifies registry artifacts when read;
an episode loads only the selected frozen checkpoints.

### Generate one real replay directly (Phase 2 engine)

Use the existing research Python with the local data root configured. The
following is a single episode, no batch mode; the output name is a stem, not a
path, and files are never overwritten:

```powershell
Set-Location -LiteralPath 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-visualizer'
$env:VISUALIZER_RESEARCH_DATA_ROOT = 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-lrppo-clutter'
python -m visualizer.backend.replay_cli --defender lr_ppo --generation d1 --training-seed 77 --attacker A1-203 --clutter C3 --perception map_local --evaluation-seed 0
```

Learned defenders require an explicit generation and training seed. Supported
attacker IDs set their own fixed model seed; A0 has none. The engine verifies
the source commit, allowlisted checkpoint location, frozen SHA-256, model
observation/action shapes and selected generation role before execution. Replay
files are gzip JSON under `visualizer\data\local` (gitignored). See
[replay-engine validation](docs/replay_engine_validation.md) for actual
compatibility, smoke, equivalence, and known-limit reports.

### Use the replay API and Phase 6 browser

Open <http://127.0.0.1:5173>, select an available defender and attacker, choose
C0–C3 and `local` or `map_local`, then request 1, 3, 5, or 10 episodes. Jobs
execute sequentially, with consecutive evaluation seeds and deterministic
inference. Select an existing result to lazily load its complete validated trace.
Play, pause, restart, step, scrub, change simulated-time speed, follow either UAV,
or switch to a top-down projection. Trajectories show only recorded history up
to the current time unless **Show full trajectory** is enabled. Obstacle
geometry, orientation, and event markers come from the trace; interpolation
between saved frames is display-only.

See [Replay API](docs/replay_api.md) for every endpoint, request/response
contracts, checkpoint-generation rules, actual successful local examples,
scenario parameter interpretation, timeout/cancellation behavior, and
reproducibility limitations.

Choose **Compare policies** to independently select two available controllers
and generations, then choose the shared attacker, clutter/perception scenario,
and matched evaluation seeds. The app reuses exact-hash replay artifacts first;
only the explicit **Generate missing replays** action invokes sequential
single-episode jobs through the existing one-active-job API. A mismatched
physical configuration, initial world state, frozen attacker, schema, runtime,
or simulator provenance is rejected; estimator-only differences are disclosed
and dirty-source traces are downgraded to exploratory. See
[policy comparison](docs/policy_comparison.md) for matching rules, coupling
limitations, timeline/camera behavior, metrics, and research interpretation.

### Phase 5 validation evidence

The Chromium suite includes numerical diagnostics tests, fixture-based timing,
view-mode privacy, Kalman estimate/truth separation, chart seeking, episode
state reset, and an opt-in run against the already-recorded local
Greedy/A0/C0, LR-PPO D0/A0/C2, PPO-Kalman D0/A1-203/C3, and LR-PPO D1 seed
77/A1-203/C3 traces. The latter contains 974 frames / 973 transitions and is
scrubbed through recorded decision intervals and the terminal frame. No
episodes were generated for Phase 5.

Run the live replay browser check only while the configured Phase 3 API is
running:

```powershell
Set-Location -LiteralPath 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-visualizer\visualizer\frontend'
$env:VISUALIZER_ENABLE_LOCAL_REPLAY_SCREENSHOTS = '1'
npm run test:browser
Remove-Item Env:\VISUALIZER_ENABLE_LOCAL_REPLAY_SCREENSHOTS
```

This browser test loads the four existing compressed replay artifacts without
running new simulation episodes. Six screenshots are written to the ignored
`visualizer\frontend\test-results` directory: LR-PPO's three guidance vectors
and cone, the LR-PPO angle chart, PPO-Kalman estimate versus truth, a paused
inspector, Policy-Visible mode, and a Greedy view without LR-PPO controls. The
Chromium software renderer does not require a display server. The separate
`VISUALIZER_ENABLE_REAL_REPLAY_TESTS=1` check uses the configured Phase 3 API.

### Phase 6 live comparison validation

With the configured FastAPI service and Vite server already running, the
opt-in live browser check loads existing validated replay pairs from the API,
checks both recorded canvases, synchronized playback/scrubbing, terminal
freezing, camera linking, detection-relative time, and policy-visible masking:

```powershell
$env:VISUALIZER_ENABLE_LIVE_COMPARISON_TESTS = '1'
npm run test:browser
Remove-Item Env:\VISUALIZER_ENABLE_LIVE_COMPARISON_TESTS
```

The read-only screenshot check captures the requested comparison evidence from
the three seed-0 replay pairs generated for Phase 6; it does not start jobs:

```powershell
$env:VISUALIZER_ENABLE_COMPARISON_SCREENSHOTS = '1'
npm run test:browser
Remove-Item Env:\VISUALIZER_ENABLE_COMPARISON_SCREENSHOTS
```

The separate `VISUALIZER_ENABLE_LIVE_COMPARISON_GENERATION=1` test is
stateful: it intentionally creates the minimum missing real replay artifacts
through the API. It must be opted into explicitly and run only when bounded
episode generation is appropriate.

See [comparison evidence](docs/policy_comparison.md#live-verification-and-evidence)
for recorded pair identities, exploratory provenance status, generated/reused
replay counts, and screenshot paths.

### Phase 7 research cases and exports

Research cases are local JSON sidecars referencing existing validated replay
IDs and compressed-artifact hashes. They support versioned metadata edits,
timestamped bookmarks, pair-status explanations, report/CSV/SVG/ZIP exports,
300-DPI chart rasterization, and on-demand high-resolution WebGL scene
captures. The implementation does not generate replacement traces. Three
documented Phase 6 comparison cases reuse four distinct existing episodes;
zero new episodes were generated for Phase 7.

The live Chromium workflow saved a 2560 × 1440 WebGL scene capture from the
recorded LR-PPO D0 / A1-203 / C3 `map_local` trace. Its ignored API output is
under `visualizer\data\local\exports\395eba85acb041a59bad313e4b56ab36\`.
The ignored browser screenshot is
`visualizer\frontend\test-results\phase7-recorded-battlefield.png`.
Chromium also created the local case report, six 300-DPI chart PNGs and ZIP
package (`104896408c1648eb9454f94638732ad8`) through the live API. Case-page
navigation and scene capture completed without browser page errors.

Focused validation: 22 backend API tests passed; frontend TypeScript checking
and production build passed. Vite reports its existing large-bundle warning.
Live replay listing, case management, chart rasterization, ZIP packaging and
scene capture worked, but the configured simulation interpreter lacks the
scientific dependencies needed by policy/scenario catalog inspection; those
catalog calls returned 503. See
[case library](docs/research_case_library.md) and
[scientific exports](docs/scientific_exports.md) for full details and
reproducibility limits.

### Phase 1 verification record (2026-10-16)

- API/configuration/data-boundary/contract tests: 24 passed, including a real
  Windows junction escape test (no symlink administrator privilege required).
- Existing scientific tests: 44 passed (Greedy measurement contract,
  experiment fairness, canonical LR-PPO composition); no scientific edits.
- Twelve actual scientific integration modules imported successfully using
  the existing research Python installation, without inference or episodes.
- 48 frozen journal registry references plus nine conference references
  checked read-only against SHA-256 metadata; all present and matching.
- Live FastAPI and proxied Vite health: HTTP 200, root configured,
  `simulation_enabled=false`; listeners verified on `127.0.0.1` only.
- Strict TypeScript and Vite production build passed; four Chromium browser
  tests passed, including actual WebGL orbit/reset, mobile layout and context
  loss/restoration. Desktop/mobile screenshots are local test artifacts.
- Nonblocking warnings: Vite reports the Three.js-inclusive bundle exceeding
  500 kB; Starlette's test client reports an upstream AnyIO deprecation.
  No failed checks remain. Heavy scene code splitting and dependency refresh
  can be addressed when packaging is in scope, without suppressing warnings.

## Documentation

- [Architecture and exact Python integration seams](docs/architecture.md)
- [Actual policy/checkpoint catalog audit](docs/policy_catalog_audit.md)
- [Replay v1 contract, units and telemetry](docs/replay_schema_v1.md)
- [Machine-readable JSON Schema](docs/replay_v1.schema.json)
- [Eight-phase implementation roadmap](docs/implementation_roadmap.md)
- [Phase 3 Replay API](docs/replay_api.md)
- [Phase 4 native-3D playback details](docs/3d_playback.md)
- [Phase 5 policy diagnostics](docs/policy_diagnostics.md)
- [Phase 6 matched policy comparison](docs/policy_comparison.md)
- [Phase 7 research case library](docs/research_case_library.md)
- [Phase 7 scientific exports](docs/scientific_exports.md)

Phases 1–7 are complete through local case management, annotations, and
provenance-checked exports. Phase 8 comprehensive packaging and final audit
remain unimplemented.
