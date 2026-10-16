# Counter-UAS 3D Simulation Explorer

Phase 1 only: isolated local API, real Three.js empty scene, data-root boundary,
policy audit, and versioned replay/telemetry contract. No simulation, inference,
episode endpoint, training, animation, or policy comparison is implemented.
Disabled controls explicitly show their future phase.

## Scientific integrity and coordinate decision

Based on local research commit `a3bc6c4acf1d0b601945159d682f9f14bc702860`,
not a remote branch. All additions are confined to this `visualizer` directory.
Scientific Python, experiment configuration, results, weights and tags are unchanged.

**The audited source is native 3D, not 2D.** `SoldierEnv` already has altitude,
three-component actions, climb/descent limits, and 3D collision/detection.
The researcher approved preserving native 3D recordings while offering a
**labeled horizontal projection with fixed visual heights**. No source dynamics
will be flattened or extended. Projection is presentation only; overlapping
projected paths do not establish a collision. See [architecture](docs/architecture.md).

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
Do not overwrite an existing `.env`. On subsequent starts:

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

`VISUALIZER_RESEARCH_DATA_ROOT` is read from the process environment, then
`visualizer\.env`. Relative values are anchored to the `visualizer` directory,
not the current shell directory. Explicit nonexistent, invalid, or overlapping
roots fail startup with a configuration error. Unset/blank configuration permits
an empty-scene-only startup, with health `status: degraded`, a structured
`DATA_ROOT_NOT_CONFIGURED` warning, and `simulation_enabled: false`.
`status: ok` means the configured worktree is valid, **not** that every model
has been checked or that replay is available.

Frozen artifacts are opened only as `rb` through internal `ResearchDataStore`.
Missing artifacts, read failures, and hash mismatches have explicit errors.
No browser endpoint accepts filenames, checkpoint paths, output paths or uploads.
New output is reserved for gitignored `visualizer\data\local`; Phase 1 does not
write recordings or create this directory on startup. The application does not
change filesystem ACLs; read-only behavior is enforced by the code boundary,
not a claim of OS-level isolation. An optional read-only OS account/ACL adds
defense in depth without changing the research files themselves.

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

The browser suite starts Vite itself if needed, uses a real WebGL context
(software rendering is permitted), tests orbit/reset, disabled controls,
responsive layout, backend outage and missing configuration, and writes
screenshots to gitignored `frontend\test-results`. Run the backend separately
to test a live proxy/health connection. Vite production output is a build
artifact, not a cloud deployment or packaged API server.

The Phase 1 venv deliberately installs only the API/test stack. The scientific
Gymnasium / SB3 / Torch dependencies remain in the existing research environment;
they are required for Phase 2, not API health or the empty scene. No checkpoint
is loaded by importing the backend. The existing root scientific requirements
are left unchanged.

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

Phase 1 ends here. Implement the real replay engine only after Phase 2 is
explicitly requested.
