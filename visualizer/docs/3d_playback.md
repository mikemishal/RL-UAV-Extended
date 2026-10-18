# Native-3D episode playback and Phase 5 diagnostics

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
- In Policy-Visible mode, Follow attacker is disabled and camera fitting uses
  the configured engagement bounds rather than privileged entity trajectories.
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
  reached in Ground Truth mode. Policy-Visible mode hides privileged event
  markers. Event positions use `position_m` where present. Otherwise the
  corresponding entity's recorded position at the event step is used and
  identified as the entity position, not an inferred contact point.
- Collision labels retain the actual terminal taxonomy (defender/enemy and
  static/dynamic obstacle). At the terminal frame, playback stops and the
  actual outcome and simulated duration are highlighted.
- In Ground Truth mode the complete episode outcome is known to the researcher from the saved trace;
  before terminal playback the panel explicitly labels it as a known final
  result, not an event that has already occurred.

## Episode information and missing values

The information panel uses the replay's policy IDs, defender generation and
training seed, checkpoint role when present, attacker, clutter, awareness,
evaluation seed, recorded detection, privileged range metrics, and terminal
outcome in Ground Truth mode. Policy-Visible mode masks privileged ranges,
truth markers, collision events, outcomes and provenance. If an optional field
is null or has an explicit unavailable reason, it is shown as unavailable or
not recorded. The viewer does not infer hidden controller state.

Recorded velocity/heading is used for display orientation when available. If
orientation data are absent, the viewer may derive an orientation from adjacent
recorded positions; that label explicitly states that the orientation is
visualization-derived. It is never treated as an additional physical state.

## Decision timing and diagnostic overlays

The replay recorder attaches the telemetry for pre-transition decision `k` to
the resulting frame `k+1`, with `decision_step=k`. The command applies during
the transition `k → k+1`. `decisionIntervalAtTime` selects that interval from
the lower/current frame; telemetry is not moved to the end frame's displayed
state as a new decision. At the terminal time there is no upcoming command, so
the inspector and scene omit decision arrows and show the relevant fields as
unavailable. If `controller_action_executed` is false, the requested command is
identified as recorded but not applied during simulator standby.

The LR-PPO scene draws equal-length direction-only arrows at the defender:
orange for analytical Lead, purple for the final requested LR-PPO direction,
and cyan for actual post-transition velocity. The cyan arrow is actual
movement, not a safety command. The translucent cone is aligned with Lead and
has a 30-degree half-angle. Simulator vectors use the same `(x, y, z) → (x, z,
-y)` mapping as positions. Its cone geometry follows the existing residual
composition's native 3D angular bound; no altitude-control or 3D dynamics are
added. A zero or unavailable vector is not drawn.

The inspector calculates native angular separation as
`acos(clamp(dot(normalize(lead), normalize(requested)), -1, 1))`, in degrees.
This is a nonnegative 3D angle. Signed horizontal azimuth is calculated
separately with a wrap to `[-180°, 180°)` and is unavailable when either
horizontal projection is near zero. Elevation difference is separately
calculated from the two simulator-space vectors. The 30° maximum is a
reference bound, not evidence that the actor saturated or clipped.

Requested policy action, environment desired command, safety-adjusted command,
and actual velocity are distinct recorded quantities. Obstacle arbitration is
called a safety intervention only when the replay records `avoidance_active`.
Requested-to-velocity deviation can also reflect acceleration, turn and climb
constraints; the UI does not attribute that deviation solely to safety.

## Synchronized charts and information modes

Diagnostic series use actual frame or decision timestamps in seconds. They
contain no samples at interpolated display times. The cursor follows the same
playback clock; pointer click/drag maps the chart plot-area coordinate back to
recorded episode time. Series arrays are derived once per selected episode.
Common Ground Truth charts show recorded defender-attacker and
attacker-asset separation, transition and cumulative reward, and recorded event
markers. LR-PPO charts show native Lead-to-requested angle with the 30° bound,
signed azimuth, and the separate desired-to-safe/requested-to-velocity angles.
Kalman-enabled charts show derived estimate-to-truth position and velocity
errors plus decision-to-decision action turn angle. Each metric's source,
units and availability are described in
[policy diagnostics](policy_diagnostics.md).

**Ground Truth** is the default research view and identifies privileged
attacker state, trajectories and evaluator metrics. **Policy-Visible** displays
the recorded defender, protected asset, known static-map obstacles, and
detection-conditioned policy estimate/measurement. It hides the attacker and
its trail, moving/unknown obstacles, future paths, events, privileged
separations, reward and final outcome. Attacker-follow camera is disabled, and
camera bounds use only scenario engagement limits. This remains a partial
reconstruction: the replay does not preserve complete local sensing geometry
or every occlusion/measurement volume. It is not a reconstruction of the exact
policy observation tensor.

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

Phase 5 adds these read-only policy diagnostics. Raw PPO residual values,
internal residual clipping details, and an exact complete perception scene
remain unavailable; see [the coverage matrix and limitations](policy_diagnostics.md).

## Phase 6 synchronized comparison

Compare Policies mounts two independent `Battlefield` scenes using the same
renderer, world-coordinate mapping, and shared union bounds. Each scene owns
its replay, frame sample, moving-obstacle state, trails, event markers and
diagnostics. A single shared clock maps axis time through each trace's recorded
timestamps. Absolute mode starts both at local simulation time zero;
detection-relative mode maps axis zero to each trace's own recorded detection
time and always displays the two local simulation times. It changes no replay
state or action.

When one trace reaches its recorded terminal duration, its selected state is
clamped to the final authoritative frame and the pane explicitly says
"Episode ended · frozen at ...". The shared timeline continues only for the
longer recorded episode. There is no post-terminal interpolation, action,
attacker path substitution, or simulated movement.

Free and top-down camera poses may be linked in world space; pose updates are
suppressed when they are echoes from the paired controls to avoid a feedback
loop. Follow-defender and follow-attacker remain pane-local, so each view
follows its own entity. Both viewports share world bounds and projection
settings; disabling camera linking allows independent inspection.

Comparison overlays and charts reuse the native Phase 5 diagnostic series.
Policy-specific arrows appear only when that episode records the relevant
telemetry. Matching evaluation seeds establish a shared initial scenario only
when the replay validator confirms the initial world; they do not imply that
later learned-attacker or protected-asset paths are coupled.
See [policy comparison](policy_comparison.md) for the comparability criteria
and analysis definitions.

## Phase 7 scene capture

From a saved research case, selecting a replay capture restores that real
bundle into the single-episode view, pauses playback, and renders the current
scene into an on-demand 2560 × 1440 WebGL render target. The temporary target
is read back and vertically corrected as PNG; camera projection and renderer
target/viewport/scissor state are restored in `finally`. Global
`preserveDrawingBuffer` is not enabled. The API verifies the case's replay ID
and compressed-file hash before storing the screenshot and metadata.

Only WebGL scene geometry is included. Drei `Html` labels and application
chrome are DOM overlays and are not part of the PNG. This is a visualization
export of the currently recorded time and camera, not a new episode or
scientific measurement. See [scientific exports](scientific_exports.md).
