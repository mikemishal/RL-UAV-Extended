# Counter-UAS 3D Simulation Explorer

Phase 4: an isolated local research visualizer with a live policy/scenario
catalog, validated multi-episode replay jobs, and interactive playback of
existing authoritative native-3D traces. React Three Fiber renders the recorded
defender, attacker, asset, obstacles, trajectories, detection, and events; one
simulated-time clock controls playback and scrubbing. The browser does not
simulate, train, infer policies, or change recorded outcomes.

## Scientific integrity and coordinate decision

Based on local research commit `a3bc6c4acf1d0b601945159d682f9f14bc702860`,
not a remote branch. All additions are confined to this `visualizer` directory.
Scientific Python, experiment configuration, results, weights and tags are unchanged.

**The simulator and viewer preserve native 3D.** `SoldierEnv` records altitude,
three-component actions, climb/descent limits, and 3D collision/detection.
Coordinates map from simulator `(x, y, z)` to Three.js `(x, z, -y)` without
flattening altitude. The top-down camera is an orthographic projection for
accurate horizontal geometry; it does not alter the underlying replay.
See [architecture](docs/architecture.md) and [3D playback](docs/3d_playback.md).

## Windows setup

Prerequisites: Python 3.11+ (verified with 3.13.14), Node 22.12+ (verified with
24.16.0), npm, and a WebGL2-capable browser. Use two PowerShell terminals.
Commands below use the requested worktree location; replace it if relocated.
Scripts resolve their own paths and work independently of the calling directory.

Terminal 1:

```powershell
Set-Location -LiteralPath 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-visualizer\visualizer'
Copy-Item -LiteralPath '.env.example' -Destination '.env'
.\start-backend.ps1 -Install
```

Review `.env` before starting: the example points to the original sibling
research worktree using a relative path, not a personal absolute path or secret.
Set the simulation interpreter to the existing research Python environment
that has Gymnasium, Stable-Baselines3, PyTorch, and the frozen project's
dependencies. The lightweight API venv is not the simulation runtime:

```powershell
$env:VISUALIZER_SIMULATION_PYTHON = (Get-Command python).Source
```

Alternatively, put its absolute executable path in the ignored `visualizer\.env`
as `VISUALIZER_SIMULATION_PYTHON=...`. Do not install or update scientific
dependencies as part of the visualizer setup. Do not overwrite an existing
`.env`. On subsequent starts:

```powershell
.\start-backend.ps1
```

Terminal 2:

```powershell
Set-Location -LiteralPath 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-visualizer\visualizer'
.\start-frontend.ps1 -Install
```

On subsequent starts use `.\start-frontend.ps1`. Open
<http://127.0.0.1:5173>. Backend: <http://127.0.0.1:8000/api/health>;
local API reference: <http://127.0.0.1:8000/docs>. Stop each server with Ctrl+C.
The frontend's Vite proxy sends `/api` to `127.0.0.1:8000`. No CORS wildcard
or external binding is needed. Ports do not silently change. Never bind
these development servers to `0.0.0.0`.

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

### Use the Phase 4 API and browser

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

### Phase 4 validation evidence

The Chromium suite includes deterministic replay fixtures and an opt-in run
against the already-recorded local Greedy/A0/C0, LR-PPO D0/A0/C2,
PPO-Kalman D0/A1-203/C3, and LR-PPO D1 seed 77/A1-203/C3 traces. The latter
contains 974 frames and was scrubbed at its midpoint and terminal frame. No
episodes were generated for Phase 4.

Run the live replay browser check only while the configured Phase 3 API is
running:

```powershell
Set-Location -LiteralPath 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-visualizer\visualizer\frontend'
$env:VISUALIZER_ENABLE_REAL_REPLAY_TESTS = '1'
npm run test:browser
Remove-Item Env:\VISUALIZER_ENABLE_REAL_REPLAY_TESTS
```

The six full-page screenshots from the real traces are written to the ignored
`visualizer\frontend\test-results` directory: beginning, mid-episode trails,
cluttered obstacles, top-down, terminal outcome, and selected multi-episode
list. The Chromium software renderer does not require a display server.

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

Phase 4 is complete at validated native-3D episode playback. Policy-specific
diagnostic overlays are intentionally reserved for Phase 5.
