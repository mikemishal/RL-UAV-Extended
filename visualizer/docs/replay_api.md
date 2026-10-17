# Phase 3 Replay API

The API is a local-only interface to the existing simulation and Phase 2
replay engine. Start FastAPI on `127.0.0.1:8000`; Vite on `127.0.0.1:5173`
proxies `/api` requests to it. The API does not accept checkpoint paths, output
paths, Python expressions, arbitrary policy IDs, or uploads.

All response examples below are from bounded local tests using research source
commit `a3bc6c4acf1d0b601945159d682f9f14bc702860`. Replay IDs and timestamps
are real local examples, not stable IDs across independent runs.

## Local configuration

The API uses a small Python venv. Catalog inspection and episode execution run
in a separate interpreter that must already have the existing scientific
Gymnasium, Stable-Baselines3, PyTorch, and project dependencies:

```powershell
Set-Location -LiteralPath 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-visualizer\visualizer'
$env:VISUALIZER_RESEARCH_DATA_ROOT = 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-lrppo-clutter'
$env:VISUALIZER_SIMULATION_PYTHON = 'C:\path\to\existing\research\python.exe'
.\start-backend.ps1
```

The same settings may be stored in the ignored `visualizer\.env`. Do not
install dependencies into or change the research worktree as part of setup.
Defaults: episode timeout 900 seconds, count 5, first evaluation seed 0,
deterministic inference, and `http://127.0.0.1:5173` as the allowed frontend
origin. The timeout can be set to 1–3600 seconds with
`VISUALIZER_EPISODE_TIMEOUT_SECONDS`.

## Endpoints

### `GET /api/health`

Returns local API liveness and research-root configuration; it does not
deserialize a model.

```json
{
  "service": "counter-uas-visualizer",
  "phase": 3,
  "status": "ok",
  "simulation_enabled": true,
  "data_root": "configured",
  "warnings": []
}
```

Without a configured data root, health is degraded and catalog/job endpoints
return actionable configuration errors. A healthy response does not guarantee
that every checkpoint is available.

### `GET /api/policies`

Returns `source_commit`, frozen `registry_sha256`, `deterministic_inference`,
`defenders`, and `attackers`. Entries are built by importing the audited
implementation and registry in the configured scientific runtime, resolving
only allowlisted artifacts, and verifying registered SHA-256 values.
Checkpoint paths are never returned.

Defender fields include stable `policy_id`, `display_name`, classification,
architecture, estimator and observation shape when applicable, availability
and reason, generation/seed availability and checkpoint roles, supported
clutter/perception scenarios, configurable parameters, and available
diagnostics. Attacker entries include their stable ID, architecture, training
seed, held-out flag, observation shape, checkpoint hash/availability, and
diagnostic fields.

Verified current catalog:

| Defender family | Type / estimator | Checkpoint generations and seeds |
|---|---|---|
| Greedy | Analytical / measurement | No checkpoint |
| Kalman Greedy | Analytical / Kalman | No checkpoint |
| Proportional Navigation | Analytical / measurement | No checkpoint |
| Lead | Analytical / measurement | No checkpoint |
| RMPC | Analytical / measurement | No checkpoint |
| PPO | Learned / measurement | Conference seeds 45–47 unavailable; D0, D0+, D1 seeds 71–73 available |
| PPO-Kalman | Learned / Kalman | Conference seeds 45–47 unavailable; D0, D0+, D1 seeds 74–76 available |
| LR-PPO | Learned / measurement | Conference seeds 55–57 unavailable; D0, D0+, D1 seeds 77–79 available |

The unavailability of the current conference checkpoints is reported from the
actual registry/artifact check; it is not a policy fallback. For clutter-trained
models, D0 selects role `d0_start`; D0+ and D1 select the registered primary
`final` checkpoint. Secondary best checkpoints are not selectable. Observation
widths are 16 for analytical policies and conference PPO/PPO-Kalman, 19 for
conference LR-PPO, 36 for clutter PPO/PPO-Kalman, and 39 for clutter LR-PPO.

Verified attacker catalog:

| ID | Type | Seed | Status |
|---|---|---:|---|
| A0 | Scripted | — | Available |
| A1-201 | Learned | 201 | Available |
| A1-202 | Learned | 202 | Available |
| A1-203 | Learned | 203 | Available; held out |

Catalog availability is rechecked against the registry and checkpoint hashes.
An absent or mismatched model is returned unavailable with a reason; the API
never substitutes a different seed, generation, role, or model.

### `GET /api/scenarios`

Returns C0–C3 built using the canonical `build_canonical_env_config` scenario
builder. Each clutter level has `local` and `map_local` perception entries.
`local` means obstacle data from the supported local sensing model;
`map_local` means known static obstacle map plus applicable local sensing.
For both perception modes, the response contains canonical measurement and
Kalman configurations, their SHA-256 identities, and all resolved builder
parameters. The UI renders these read-only.

Representative C0 `map_local` values (actual builder names and units):

```json
{
  "v_d": 18.0,
  "v_e": 12.0,
  "v_s": 1.5,
  "detection_radius": 15.0,
  "intercept_radius": 2.5,
  "threat_radius": 2.0,
  "unsafe_intercept_radius": 3.5,
  "dt": 0.5,
  "max_steps": 2000,
  "max_altitude": 30.0,
  "defender_max_accel": 8.0,
  "defender_max_climb_rate": 6.0,
  "defender_max_descent_rate": 6.0,
  "obstacle_count": 0,
  "dynamic_obstacle_count": 0
}
```

Distances are meters, speeds meters/second, acceleration meters/second², and
`dt` seconds. The full response includes further obstacle sensing, avoidance,
memory, route, reward, noise, and physical constraint settings. Phase 3 does
not permit modifying any physical parameter.

The current audited policy catalog reports all four clutter levels and both
perception modes as supported for the available policies. Learned model
estimators still matter: PPO-Kalman and Kalman Greedy use their Kalman
configuration; other currently catalogued defenders use measurement
configuration. The API validates every selection against catalog support.

### `POST /api/replay-jobs`

Creates one asynchronous batch and returns HTTP 202 with a `ReplayJob`.

Request fields:

| Field | Type | Rules |
|---|---|---|
| `defender_policy_id` | string | One available catalog ID |
| `defender_generation` | string, optional | `conference`, `d0`, `d0_plus`, or `d1`; required for learned policies |
| `defender_training_seed` | integer, optional | Must be listed as available for the selected generation |
| `attacker_condition` | string | `A0`, `A1-201`, `A1-202`, or `A1-203` |
| `clutter_level` | string | `C0`, `C1`, `C2`, or `C3` |
| `perception_mode` | string | `local` or `map_local`; defaults to `map_local` |
| `seed_start` | integer | 0 through 4,294,967,295; defaults to 0 |
| `episode_count` | integer | Exactly 1, 3, 5, or 10; defaults to 5 |
| `deterministic_inference` | boolean | Must be `true` |

The last requested seed must also fit the simulator's unsigned 32-bit seed
range. Episode seeds are consecutive:
`seed_start + offset` for `offset = 0..episode_count-1`. Each episode invokes
the established single-episode engine independently. The API rejects
unavailable/incompatible catalog combinations and does not silently choose a
checkpoint.

Request used by the successful three-episode REST integration:

```json
{
  "defender_policy_id": "greedy",
  "attacker_condition": "A0",
  "clutter_level": "C0",
  "perception_mode": "map_local",
  "seed_start": 71030000,
  "episode_count": 3,
  "deterministic_inference": true
}
```

Final `GET /api/replay-jobs/{job_id}` response from that run (timestamps
shortened here for readability):

```json
{
  "job_id": "7b7494c0deb84ff5a46e501bb0c6f38d",
  "state": "completed",
  "requested_episode_count": 3,
  "completed_count": 3,
  "current_episode_seed": null,
  "completed_replay_ids": [
    "bed82b3d65d08eb0b9e74e3791fd5d9e",
    "0b117096888025fcccc618a709e28914",
    "9e4f104ff7b3700a8738c7df4ad31ab0"
  ],
  "error": null,
  "created_at": "2026-10-17T04:13:13Z",
  "started_at": "2026-10-17T04:13:13Z",
  "updated_at": "2026-10-17T04:13:30Z",
  "completed_at": "2026-10-17T04:13:30Z"
}
```

The three real results had seeds/outcomes/steps/durations:

| Seed | Recorded outcome | Steps | Simulated duration |
|---:|---|---:|---:|
| 71030000 | `intercepted` | 117 | 58.5 s |
| 71030001 | `soldier_caught` | 655 | 327.5 s |
| 71030002 | `intercepted` | 26 | 13.0 s |

### `GET /api/replay-jobs/{job_id}`

Returns the full job state, validated request, requested/completed counts,
currently executing seed, completed opaque replay IDs, timestamps, and safe
error message. While running, for example, the browser formats the returned
fields as `Episode 3 of 5 running — 2 completed / seed 42`. No timestep
percentage is invented.

States: `queued`, `running`, `completed`, `failed`, `cancelled`, `interrupted`.
Job JSON is atomically persisted under the ignored `visualizer/data/local/jobs`.
On restart, any persisted queued or running job is marked interrupted; completed
episode IDs remain available.

### `POST /api/replay-jobs/{job_id}/cancel`

Cancels a queued batch immediately or marks an active batch for cancellation
and terminates its current subprocess. The worker waits for termination and
performs process cleanup. The terminal state is `cancelled`; completed,
validated replay IDs are retained, and an in-progress episode is not added to
the completed list. Cancelling an already terminal job returns its existing
state.

### `GET /api/replays?limit=100`

Returns an object with `replays` (validated compact metadata) and `warnings`.
`limit` is 1–500. Each replay listing item contains the opaque ID, seed,
defender and attacker selection, clutter/perception, native outcome, steps,
duration, detection time, total reward, collision/interception event, and
provenance.

Artifacts that fail current Pydantic schema or semantic trace validation are
excluded and counted in `warnings`; they are not changed or returned as valid
results. This also keeps older incompatible local traces from blocking the
listing of valid current-schema replays. The detail and summary endpoints
return explicit errors when a requested opaque ID cannot be validated.

### `GET /api/replays/{replay_id}`

Returns the full validated `ReplayBundle` (schema version, provenance, one
episode's metadata, frames, static obstacles, events, and summary). The
response is intended for the Phase 4 playback engine; the Phase 3 UI fetches
compact summaries only. Compressed inputs are bounded to 512 MiB uncompressed.
The route accepts only 32-character opaque IDs and resolves files under the
configured local output root.

### `GET /api/replays/{replay_id}/summary`

Returns the selected episode's metadata and summary without sending frame
arrays: `replay_id`, schema version, evaluation seed, defender, attacker,
environment, outcome/step/duration/detection/reward/collision/interception
summary, and provenance.

Successful learned-policy request tested against the held-out A1-203 artifact:

```json
{
  "defender_policy_id": "lr_ppo",
  "defender_generation": "d1",
  "defender_training_seed": 77,
  "attacker_condition": "A1-203",
  "clutter_level": "C3",
  "perception_mode": "map_local",
  "seed_start": 71030010,
  "episode_count": 1,
  "deterministic_inference": true
}
```

It completed with a valid native trace: 757 steps, outcome
`defender_static_obstacle_collision`, defender generation D1 / seed 77 / role
`final`, and attacker A1-203 / role `attacker_best`. This is an exploratory
single-run result, not a scientific estimate.

The terminal outcome strings preserved from SoldierEnv are
`intercepted`, `soldier_caught`, `unsafe_intercept`, `timeout`,
`defender_static_obstacle_collision`, `defender_dynamic_obstacle_collision`,
`enemy_static_obstacle_collision`, and `enemy_dynamic_obstacle_collision`.
`ongoing` is reserved for non-final frames; terminal labels are not collapsed
into a generic failure class.

## Errors

Errors use a safe envelope:

```json
{"error":{"code":"REPLAY_JOB_BUSY","message":"Another replay batch is queued or running; retry after it finishes."}}
```

Malformed Pydantic input is HTTP 422 (`INVALID_REQUEST`); unavailable policy,
checkpoint, attacker, or scenario is a specific HTTP 409; a busy worker is 409;
unknown job/replay IDs are 404; invalid opaque IDs are 400; invalid/oversize
replay artifacts are rejected. Raw stack traces and local research paths are
not returned.

## Worker and reproducibility guarantees

- Exactly one active batch; additional create requests fail with
  `REPLAY_JOB_BUSY` rather than being queued.
- Episodes within a batch run sequentially as the existing
  `python -m visualizer.backend.replay_cli` subprocess, under the configured
  simulation interpreter. The API server remains available for health and job
  polling.
- Per-episode timeout defaults to 900 seconds. A timeout fails the batch with
  an explicit timeout message and is never represented as a simulation
  `timeout` outcome.
- Each subprocess uses one requested seed and the same Phase 2 engine, frozen
  checkpoint resolver, deterministic inference, canonical scenario builder,
  recorder, and full trace validator. No additional policy inference occurs for
  UI or telemetry.
- Per-episode output is atomically finalized by the engine. A failed or
  cancelled in-progress episode is not marked complete; previously completed
  validated episodes are preserved.
- Provenance includes simulator and visualizer source commits/dirty flags,
  checkpoint hashes and roles, scenario config hash, registry hash, package
  versions, evaluation seed, and replay schema version. A changed/unverified
  simulator source or artifact identity is not treated as the old trace.
- The successful sample traces retain `source_dirty: true`, reflecting the
  known documentation-only edit in the research worktree at generation time.
  This flag is preserved as provenance; it does not relabel or alter the
  scientific configuration. The visualizer was also dirty at capture time,
  before the Phase 3 commit.
- Replay IDs are opaque hashes derived from generated local filenames. HTTP
  callers cannot supply or observe arbitrary filesystem paths.
- No training, checkpoint mutation, canonical-physics edits, cloud queue,
  database, or browser-side simulation is performed.

## Current boundary and limitations

The viewport remains a static, interactive Three.js battlefield. Selecting an
episode loads actual validated summary/provenance and states that trajectory
playback starts in Phase 4. There are no moving UAVs, interpolation, follow
cameras, top-down playback controls, diagnostic overlays, model parameter
editing, or publication-grade inference from small batches. Batch statistics
are descriptive only. Current local historical artifacts that do not match
the current replay contract are explicitly excluded from listings and remain
untouched on disk.
