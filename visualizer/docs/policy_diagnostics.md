# Phase 5 policy diagnostics

This document describes the read-only telemetry inspector, 3D overlays,
synchronized charts, and research display modes added to the visualizer. The
replay schema and recorder remain authoritative:
[replay-v1 contract](replay_schema_v1.md),
[replay schema](replay_v1.schema.json),
[replay recorder](../backend/replay_engine.py), and
[frontend telemetry types](../frontend/src/api.ts).

No simulator, policy, checkpoint, experiment configuration, or evaluation
semantics were changed. The browser does not call policy inference and derives
all diagnostics from the already-recorded episode.

## Telemetry coverage

Availability is per recorded decision, not guaranteed for every controller,
scenario, or historical trace. `null` fields and `unavailable_fields` remain
unavailable; the inspector does not substitute a guessed value.

| Controller | Recorded policy-specific data | Visualization-derived quantities | Not exposed / limitations |
|---|---|---|---|
| Greedy | Common requested/desired/safe commands, execution disposition, post-transition velocity, environment and safety telemetry; policy-visible measurement channel when detected | Requested-action change between recorded decisions; common ranges and reward in Ground Truth | Greedy's internal target scoring and unrecorded guidance intermediates |
| Kalman Greedy | Common command/safety data; `e_hat_m` / `v_hat_mps` from the Kalman channel when detected | Euclidean estimate-to-truth position/velocity errors in Ground Truth; action turn | Internal filter covariance/update details unless explicitly recorded |
| Proportional Navigation (PN) | `pn.guidance_mode`, range, closing speed, LOS unit vector/rate, and acceleration command when emitted; common command/safety data | 3D action-change angle; common recorded distances/reward in Ground Truth | Any internal values omitted by that decision's telemetry |
| Lead | `lead.guidance_mode`, target position/velocity, intercept time/point, and discriminant when emitted; common command/safety data | 3D action-change angle; common recorded distances/reward in Ground Truth | Values not present in the decision record |
| RMPC | `rmpc.guidance_mode`, initialization mode, scenario scores, robust score, best sequence, objective evaluations, and decision time when emitted; common command/safety data | 3D action-change angle; common recorded distances/reward in Ground Truth | Optimizer internals not serialized by the replay |
| PPO | Recorded policy/model input, requested action and common execution/safety data; policy-visible observation/measurement | Consecutive requested-action turn angle; common recorded distances/reward in Ground Truth | Latent activations, alternate actions, or a second model inference |
| PPO-Kalman | PPO data plus detection-conditioned Kalman position/velocity estimate and common execution/safety data | Derived position/velocity estimate error against separate privileged truth; action turn | Ground truth is not an input to the policy; unrecorded filter internals remain unavailable |
| LR-PPO | Recorded analytical Lead direction, final requested direction, recorded native/horizontal adjustment metadata, and common execution/safety data | Native 3D Lead-to-requested angle, signed horizontal azimuth, elevation difference, desired-to-safe and requested-to-velocity angles | Raw PPO residual vector/output and internal projection/clipping state are not exposed by the current wrapper |

The common per-decision record includes the decision step, policy-visible
pre-step data, requested action and norm, the sanitized defender input and
optional model input, environment desired and safety-adjusted commands where
recorded, command deltas/angles, `controller_action_executed`, standby and
avoidance flags, actual post-transition velocity, and recorded execution
saturation/heading fields. Frame records separately contain privileged
attacker state, state distances, reward, environmental clearances/collision
events, and replay-level provenance. These truth/evaluator fields are not
treated as policy observations.

Controller identity, generation, training seed, checkpoint role, action and
observation shapes, and policy parameters come from the replay's recorded
selection metadata. They are not reconstructed from the action stream.
Compatibility and availability remain catalog/manifest decisions; this
telemetry table does not imply every policy supports every environment
configuration.

## LR-PPO native-3D interpretation

The recorded `lrppo.lead_direction` and `requested_action` are compared as
native simulator vectors. For finite, nonzero vectors `a` and `b`, the
inspector calculates:

```text
angle_deg = acos(clamp(dot(normalize(a), normalize(b)), -1, 1)) * 180 / pi
```

This is a nonnegative three-dimensional separation, not a signed yaw. The
visualizer uses the canonical simulator-to-Three.js vector transform
`(x, y, z) → (x, z, -y)` for both vectors. This is a rotation and preserves
their angle.

The controller implementation's residual composition is a native 3D angular
composition: it normalizes the component perpendicular to Lead and combines
it with Lead using the tangent of the bounded angle. The source sets the
canonical maximum to 30 degrees; the transparent cone therefore represents a
30-degree half-angle around the Lead direction. The cone is a visual reference,
not evidence of clipping. A correction close to the boundary does not establish
that an internal PPO action was clipped.

Signed horizontal azimuth is a separate secondary diagnostic. It uses the
simulator x/y horizontal projection and wraps the requested-minus-Lead
difference into `[-180°, 180°)`. It is unavailable when either horizontal
projection is near zero. Elevation difference is calculated separately from
the simulator z-up vectors. Zero, missing, or non-finite vectors produce
Unavailable.

The scene shows equal-length normalized direction indicators, not physical
movement distance or command magnitude:

- **Orange:** analytical Lead direction.
- **Purple:** final LR-PPO requested direction after correction.
- **Cyan:** actual post-transition velocity direction (movement), not the
  requested or safety-adjusted command.
- **Translucent orange cone:** canonical 30-degree native angular allowance.

Each arrow and the cone has an independent visibility toggle. The inspector
reports the recorded requested vector, desired/safe command, actual velocity,
the derived native angle, and separate azimuth/elevation diagnostics.

### Raw PPO residual limitation

The current policy wrapper does not serialize the raw PPO residual output,
projected residual, or internal clipping status. Those fields remain
**Unavailable** with the wrapper limitation stated. The displayed net
Lead-to-final-requested correction is observable from recorded vectors, but it
does not uniquely identify the raw network output or prove clipping.

The visualizer does not run inference a second time, invert the final vector to
claim a unique raw action, or modify policy code. A future research-approved
one-pass instrumentation could copy the already-computed PPO output and
projection/clipping diagnostics into the recorder's existing per-step record,
then validate the additional fields against the existing action and trace.
That instrumentation is not required for Phase 5 and is intentionally not
implemented.

## Requested command, safety command, and actual movement

The inspector distinguishes:

1. `requested_action`: the controller's recorded request.
2. `desired_command`: the environment-side desired direction, when recorded.
3. `safe_command`: the post-arbitration/safety direction, when recorded.
4. `executed_velocity_mps`: actual velocity after the simulator transition.

`avoidance_active` is the verified recorded safety-override flag. A
requested-to-velocity angular deviation alone is **not** labeled a safety
intervention: constrained acceleration, turn rate, and climb/descent can also
make actual movement differ from the requested vector. The requested-to-safe
and desired-to-safe fields are shown separately when recorded.

Clearances and collision events are read from the resulting recorded frame in
Ground Truth mode. They are not synthesized from a difference in position.
Policy-Visible mode hides privileged all-scene clearances, obstacle collision
events, and other evaluator-only state.

## Decision-to-frame timing

The recorder stores telemetry for the decision made from pre-transition frame
`k` on the resulting frame `k+1`, with `decision_step = k`. Therefore, the
decision command is labeled **“Command applies during step k → k+1”** and is
selected against the current lower frame/time. During fractional position
playback, the command remains attached to that same interval; interpolation
does not create decisions or samples.

At the initial time, a command is shown only if the replay records the
`0 → 1` decision. If `controller_action_executed` is false, the UI says the
decision was recorded but not applied (for example, during simulator standby).
At the terminal frame there is no next transition and no command or arrow is
invented. Safety fields for an interval use the recorded resulting frame; the
terminal state uses its own frame.

## PPO-Kalman estimates and truth

The policy-visible channel records either the applicable measurement or
Kalman estimate. Estimate position and velocity are displayed only when the
decision-time `enemy_detected` flag is true and the corresponding values are
present. Before detection, no estimate marker or velocity arrow is rendered.

Ground Truth mode can show privileged attacker position/velocity from the
decision's corresponding start frame, separately labeled and never described
as policy input. When an estimate and truth are both recorded, the visualizer
derives:

```text
tracking_error_m = norm(estimated_position_m - privileged_position_m)
velocity_error_mps = norm(estimated_velocity_mps - privileged_velocity_mps)
```

These are explicitly labeled derived Euclidean errors. Estimate markers,
privileged truth markers, the estimate-to-truth line, and estimate velocity
arrow have visibility controls. In Policy-Visible mode, truth markers, error
vectors, truth values, and estimate-to-truth errors are hidden.

## Charts, units, and seeking

Diagnostic arrays are computed once for a selected episode and shared by the
inspector and chart panel. Every plotted point uses an actual frame time or
recorded decision interval start time in seconds; no interpolated policy
samples are generated. The vertical cursor follows playback, and clicking or
dragging the plot seeks the common playback clock.

| Chart | Source / derivation | Units |
|---|---|---|
| Defender → attacker separation | Recorded privileged frame diagnostic; Ground Truth only | m |
| Attacker → asset separation | Recorded privileged frame diagnostic; Ground Truth only | m |
| Transition reward / cumulative reward | Recorded frame values; Ground Truth only | reward |
| LR-PPO Lead → requested | Derived native 3D angle from recorded vectors | degrees |
| LR-PPO horizontal azimuth | Derived wrapped horizontal projection difference | signed degrees |
| Desired → safe direction | Recorded arbitration angle | degrees |
| Requested → actual velocity | Recorded requested-to-velocity comparison; not safety-only | degrees |
| Kalman position error | Derived Euclidean estimate-to-truth distance; Ground Truth only | m |
| Kalman velocity error | Derived Euclidean estimate-to-truth distance; Ground Truth only | m/s |
| Action change | Derived 3D angle between consecutive recorded requested actions | degrees |
| Event markers | Recorded event kind and simulation time; Ground Truth only | s |
| 30° reference | Canonical LR-PPO source limit | degrees |

When a controller does not emit a field or the trace lacks a sample, the chart
has a gap or an explicit unavailable message; it does not fill missing values
with zero. On terminal time the cursor may move to the terminal timestamp,
while decision-only series correctly end at the last actual decision interval.

## Ground Truth and Policy-Visible modes

**Ground Truth** is the default researcher-analysis view. It shows recorded
attacker/defender state and paths, moving obstacles, events, privileged
separations, evaluator outcomes, and estimate/truth diagnostics where
available. Privileged state is identified as researcher-only.

**Policy-Visible** is a visualization-only partial reconstruction. It keeps
the defender and protected asset, recorded policy-visible estimate/measurement
after detection, and static obstacles marked known in the selected map. It
hides the attacker and attacker trajectory, moving and unknown obstacles,
future paths, event markers, privileged ranges, truth-derived errors, reward,
terminal outcome, and provenance. Attacker-follow camera is unavailable.
Configured engagement bounds are used for framing so hidden trajectories do
not influence camera fit.

The replay does not preserve enough information to reconstruct the exact local
sensing scene, including every sensing geometry, occlusion, or complete
observation tensor at rendered intermediate positions. The view is therefore
marked partial; it is not represented as the controller's exact observation.
Experiment selection controls remain researcher controls. Neither mode changes
the replay, policy input, or scientific results.

## Validation evidence and limits

Pure and browser tests cover native 3D angles, 30-degree interpretation,
signed azimuth conventions, zero/non-finite vectors, vector coordinate
transform, requested-versus-executed distinction, decision/frame alignment,
terminal no-command behavior, unavailable PPO residuals, Kalman
estimate/truth separation, derived errors, pre-detection masking, information
modes, chart cursor/seek, episode-switch reset, and replay immutability.

The Chromium screenshot run loaded existing local recordings for:

- Greedy / A0 / C0 — 282 frames.
- LR-PPO D0 / A0 / C2 — 170 frames.
- PPO-Kalman D0 / A1-203 / C3 — 330 frames.
- LR-PPO D1 / A1-203 / C3 — 974 frames / 973 transitions.

The test reads the existing gzip bundles and does not start the simulator or
generate an episode. Screenshots are ignored local artifacts under
`frontend/test-results`. Browser-level live Phase 3 API testing is a separate
opt-in check and requires the configured research Python runtime. See the
README for exact test commands and current validation status.

Known limitations: raw LR-PPO residual/clipping internals are unavailable;
not every policy emits every optional diagnostic; Policy-Visible is partial;
and requested-to-actual velocity deviation can have multiple physical causes.
These limitations are surfaced rather than filled with inferred values.

## Phase 6 side-by-side diagnostics

Each comparison pane receives only its own replay bundle and its own
`deriveDiagnosticSeries` result. The shared display toggle applies the same
privacy rule to both panes without copying an estimate, observation or truth
state between controllers. In Policy-Visible mode the comparison also hides
privileged outcomes, event times, collisions, ranges and aggregate outcome
counts; it retains the explicit partial-reconstruction warning.

Common comparative charts use the pair's shared axis but retain separate A/B
recorded samples:

| Comparison view | Definition / source | Interpretation |
|---|---|---|
| Defender → attacker range | Recorded privileged diagnostic | Comparable only in Ground Truth; state paths may diverge after reset |
| Attacker → asset range | Recorded privileged diagnostic | Each episode's own attacker and asset; not a coupled trajectory |
| Defender path length | Sum of Euclidean distances between consecutive recorded native positions | Derived metric; no interpolated display samples |
| Requested turn angle | Recorded requested-action vectors and adjacent recorded actions | Omitted where unavailable; architecture-specific meaning is retained |
| Safety intervention | Recorded `avoidance_active` decision telemetry | Missing telemetry is unavailable, not zero; decision frames are counted once |
| Detection / terminal markers | Recorded event and terminal times | Shared cursor is display alignment only |
| Policy-specific charts | Existing LR-PPO, PPO-Kalman or analytical telemetry | Separate per-pane scales/labels; missing metrics are not zero-filled |

Clicking a comparative chart seeks the shared axis; absolute mode maps that
time to both local episode times, while detection-relative mode maps through
each trace's own first detection. Samples stop at each trace's terminal state.
The first-position-divergence helper uses recorded defender coordinates only
at exact timestamps available in both traces; it does not compare an
interpolated coordinate or infer a cause. Its explicit threshold is a
display-only exploratory navigation aid, not a causal test.

Per-episode summaries identify native outcomes and the canonical
`intercepted` success label, duration, detection/interception times, derived
path length, recorded collisions, available safety telemetry, and final
privileged ranges. Policy-Visible mode masks this block. A/B duration and path
differences, and loaded-pair outcome counts, are descriptive for those
recordings only; small exploratory batches do not yield publication-grade
confidence intervals or controller rankings.

For complete eligibility and exact matching rules, see
[policy comparison](policy_comparison.md).
