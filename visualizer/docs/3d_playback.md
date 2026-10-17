# Native-3D episode playback

Phase 4 renders only complete, validated replay bundles produced by the Python
simulator. The browser does not run policies, estimate dynamics, generate
obstacles, or mutate the recorded episode. The selected trace is loaded from
`GET /api/replays/{replay_id}`; the list and summary continue to use the Phase 3
`GET /api/replays` and `GET /api/replays/{replay_id}/summary` endpoints.

## Windows startup

Open two PowerShell terminals. Start the configured local API from the
visualization worktree:

```powershell
Set-Location -LiteralPath 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-visualizer\visualizer'
.\start-backend.ps1
```

The `.env` file must already point to the research data root and the existing
research Python environment; see [the setup instructions](../README.md#windows-setup).
For a first setup, create the ignored `.env` from `.env.example` and use
`.\start-backend.ps1 -Install` only if the lightweight API environment is not
installed. Do not overwrite an existing `.env` or install/update research
dependencies.

Start Vite in the second terminal:

```powershell
Set-Location -LiteralPath 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-visualizer\visualizer'
.\start-frontend.ps1
```

For first-time frontend dependencies, use `.\start-frontend.ps1 -Install`.
Open <http://127.0.0.1:5173>. The frontend binds to `127.0.0.1:5173` and Vite
proxies `/api` to `127.0.0.1:8000`. Stop both processes with Ctrl+C. Neither
server should be exposed on `0.0.0.0`.

## Coordinate and geometry conventions

- Replay positions are simulator `(x, y, z)` in meters, with simulator `z` up.
- A single right-handed rotation maps them to Three.js `(x, z, -y)`. The
  same helper transforms velocities, movement directions, event positions,
  dynamic obstacles and the endpoints of static-obstacle AABBs.
- One Three.js unit is one meter. The simulator ground plane at `z = 0` becomes
  Three.js `y = 0`. Recorded altitude is preserved; there is no visual
  vertical-scale factor or fixed-altitude substitution.
- An AABB with simulator corners `min_corner_m` and `max_corner_m` is rendered
  as a box with the transformed center and exact dimensions. The sign reversal
  of the simulator `y` axis also reverses the corresponding transformed bounds.
- A moving obstacle uses the recorded `position_m` and `radius_m` at the
  current recorded time. When adjacent frames both contain the obstacle, only
  its displayed position is interpolated. No frontend movement equation is
  used.
- Low-poly vehicle and asset shapes are decorative. Their origins are placed at
  the recorded entity positions. They do not represent modelled dimensions or
  collision hulls. The static AABB and recorded moving-obstacle spheres are the
  authoritative physical geometry.

The schema-v1 `presentation` field remains `horizontal_projection_fixed_heights`
for compatibility. Playback does not use that hint to overwrite entity
altitudes; the viewer displays the native values in each `position_m`.

## Camera modes

- **Free 3D** is the default perspective camera with OrbitControls rotation,
  panning and zoom. Camera distance is fitted from the selected episode's
  recorded entity paths and obstacle bounds.
- **Follow defender** and **Follow attacker** keep the selected recorded entity
  as the camera target. The camera follows the same playback time as the scene;
  it has no independent animation clock.
- **Top-down** uses an orthographic camera looking down the Three.js vertical
  axis. Horizontal distances and relative positions are not changed by
  perspective scaling. It is a view of the original 3D episode, not a
  replacement or a 2D simulation. Rotation is disabled in this preset; pan and
  zoom remain available.
- **Reset camera** restores the current mode's fitted view. Returning from
  top-down restores the previous 3D camera mode.

## Playback clock and interpolation

`ReplayPlayback.tsx` stores one current simulated timestamp in seconds. It reads
`environment.dt_s` and frame `time_s` values from the selected replay. The
speed multiplier scales elapsed wall time: at `1×`, one simulated second takes
one wall-clock second; at `2×`, it takes half a second. Available speeds are
`0.25×`, `0.5×`, `1×`, `2×`, `4×`, and `8×`. Playback stops at the recorded
terminal time. Looping is explicitly user-enabled and is off by default.

The current timestamp is sampled against adjacent frames in the selected
episode. Positions are linearly interpolated only for smooth display; discrete
events, detection, outcomes, metrics and frame-by-frame stepping remain
authoritative recorded values. At exact frame timestamps the scene uses the
exact recorded position. The sampler clamps to the first and terminal frames
and never extrapolates. Episode selection clears the prior trace and resets the
clock before loading the new trace.

Interpolation is not a new simulation transition, sensor observation, policy
decision, or physical state. The information panel reports its metrics from the
latest authoritative recorded frame and labels when the visual position is
between recorded samples.

## Trajectories and obstacles

Defender, attacker and asset trails use recorded frame positions and are shown
only through the selected time by default. Their display toggles are independent;
**All trajectories** controls the three entity trails. Moving-obstacle trails
are separately optional. **Show full trajectory** is an explicit retrospective
mode that may reveal future positions.

Trajectory geometry is memoized per replay and updated through a bounded line
buffer rather than a React component per historical frame. Static obstacles are
created once for the selected trace and are not rebuilt on playback ticks.
Wireframe and transparency options change only obstacle appearance.

If a moving obstacle is absent between recorded intervals, its trail remains
segmented at the gap. The viewer does not bridge missing data or extrapolate a
new location. If a replay lacks required geometry, it is not synthesized.

## Detection, events and outcome

- Detection state comes from `policy_visible.enemy_detected` at the current
  recorded frame. Detection time comes from the recorded episode summary.
- Before detection, the attacker remains visible to the researcher in a clearly
  labeled privileged view. This does not imply that the defender has an
  attacker measurement or ground-truth access.
- The optional detection volume is a 3D sphere centered on the recorded
  defender position and scaled by that frame's recorded `detection_radius_m`.
- Recorded detection, interception, unsafe interception, asset-reached,
  collision and timeout events are shown only once their recorded event time is
  reached. Event positions use `position_m` where present. Otherwise the
  corresponding entity's recorded position at the event step is used and
  identified as the entity position, not an inferred contact point.
- Collision labels retain the actual terminal taxonomy (defender/enemy and
  static/dynamic obstacle). At the terminal frame, playback stops and the
  actual outcome and simulated duration are highlighted.
- The complete episode outcome is known to the researcher from the saved trace;
  before terminal playback the panel explicitly labels it as a known final
  result, not an event that has already occurred.

## Episode information and missing values

The information panel uses the replay's policy IDs, defender generation and
training seed, checkpoint role when present, attacker, clutter, awareness,
evaluation seed, recorded detection, privileged range metrics, and terminal
outcome. If an optional field is null or has an explicit unavailable reason, it
is shown as unavailable or not recorded. The viewer does not infer hidden
controller state.

Recorded velocity/heading is used for display orientation when available. If
orientation data are absent, the viewer may derive an orientation from adjacent
recorded positions; that label explicitly states that the orientation is
visualization-derived. It is never treated as an additional physical state.

## Long traces and verification

The browser test covers the existing 974-frame LR-PPO D1/A1-203/C3 trace. It
checks frame-zero positions against the full API payload, scrubs to an
interpolated midpoint and the exact terminal frame, and verifies the outcome.
Transforms and paths are memoized, dynamic obstacles are keyed by recorded ID,
and per-frame trails update draw ranges instead of creating a React node for
every historical step. Only the currently selected full trace is loaded.

Run deterministic browser tests with Vite started by Playwright:

```powershell
Set-Location -LiteralPath 'C:\Users\mimishal\OneDrive\Documents\UIC\PhD\Research\RL-UAV-Extended-visualizer\visualizer\frontend'
npm run test:browser
```

Run the additional real-replay browser check only while the configured local
Phase 3 API is running:

```powershell
$env:VISUALIZER_ENABLE_REAL_REPLAY_TESTS = '1'
npm run test:browser
Remove-Item Env:\VISUALIZER_ENABLE_REAL_REPLAY_TESTS
```

Screenshots from this real-trace check are written to ignored
`frontend\test-results`. No episode is regenerated by the test.

## Scientific limitations and next phase

This is a visualization of saved data, not a simulator front end. Display
interpolation is not physically integrated. Model dimensions, visual heading,
and low-poly appearance must not be interpreted as measured airframe geometry
or unrecorded attitude dynamics. Orthographic top-down overlap is not by itself
evidence of a native 3D collision. No collision contact point is invented.

Policy-specific diagnostic overlays, including LR-PPO guidance vectors,
Kalman-state explanations, residual angles and safety interventions, are not
implemented in Phase 4 and remain Phase 5 work.
