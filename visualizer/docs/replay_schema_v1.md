# Replay JSON v1 / telemetry design

Version: `1.0.0`. This is the typed contract used by the single-episode replay
engine. The frontend remains an empty scene; no fabricated frames are included.

Canonical definitions: [replay_contract.py](../backend/replay_contract.py).
Machine schema: [replay_v1.schema.json](replay_v1.schema.json), generated with
Pydantic `ReplayBundle.model_json_schema()`. Models forbid unknown keys and
non-finite numeric fields; JSON must never contain NaN/Infinity. The engine
normalizes non-finite values in optional diagnostics to `null` and records a
field-path reason. It does not replace infinite clearance with a fictitious
large finite distance; authoritative non-finite entity/action state fails.

## Bundle and provenance

| Field | Meaning |
|---|---|
| `schema_version` | literal `1.0.0`; incompatible layouts require a new version |
| `coordinate_system` | `simulation_xyz_z_up_m`; preserve native three-component source data |
| `presentation` | `horizontal_projection_fixed_heights`; not a change to data or dynamics |
| `provenance` | 40-hex simulation/visualizer commits, both dirty flags, deterministic inference, Python/package versions, device, original registry SHA-256 |
| `episode_count` | literal `1`; Phase 2 is single-episode only |
| `seed_start` | nonnegative explicit simulation seed (not a model training seed) |
| `episodes` | exact count, consecutive seeds in recording order (validated in Pydantic) |

Each episode stores `episode_id`, `simulation_seed`, defender and attacker
selection snapshots, complete environment snapshot/hash, static geometry,
all real frames, events and final summary. Selection includes catalog ID,
generation, training seed, observation/action shapes, estimator and exact
parameters, including the versioned `sector_obs_v1` spec when consumed.
Checkpoint includes stable artifact ID, research-relative path,
SHA-256 and role. Analytical/A0 checkpoint fields are null; never synthesize
a checkpoint hash for them. Paths here are provenance, not a future file API.

`EnvironmentSpec.config` is the full JSON-serialized `EnvConfig` dataclass,
with a stable sorted JSON hash; it includes dt,
dimensions, thresholds, sensing, avoidance, collision/reward defaults and
every builder override. `builder` identifies the research config function:
`build_canonical_env_config` (V1) or `build_v2_env_config` (V2). `semantics`
(`V1` default, `V2`) is validated against the resolved config
(`mission_contact_mode`, `enemy_nominal_guidance_mode`) and the builder, so the
label cannot disagree with the simulator that ran. Event `contact_fraction` is
populated only for V2 mission/collision events, from the simulator's own report.
The engine records the frozen selected RMPC parameters and residual limit 30.

## Frame timing and units

Record frame 0 immediately after reset and frame k after completed transition
k (k >= 1), including terminal state. T transitions require T+1 frames.
Frame times are simulation seconds: `time_s = step * dt_s`, not wall-clock
rendering time. An initial frame has null telemetry and reward.

Every frame stores current defender/attacker/asset position, available velocity,
all moving obstacle states, **current** policy-visible observation/estimator
channels, privileged diagnostic state, reward, termination/truncation and native
outcome. The protected asset velocity is not exposed by SoldierEnv and remains
null with a reason. `PolicyVisible.observation` is the base 16-D environment observation;
20 sector features are a separate optional array. Model input shape lives in
selection metadata: 16/19/36/39, not an ambiguous flattened state.
An A1 decision's exact 36-D observation is stored in that transition's
`telemetry.attacker_observation`.

For k >= 1, telemetry describes the transition **ending at frame k**:
`telemetry.decision_step = k-1`. `telemetry.decision_policy_visible` is the
exact pre-step observation and sanitized channels supplied to that action.
`frame.policy_visible` describes the state at frame k; it is not substituted
for the earlier decision input. Safe/desired commands and executed velocity
come from the resulting `step` info. `requested_to_velocity_angle_deg` is the
angle from the request to actual post-step velocity; it is not an "executed
action." Requested-to-desired delta/angle describe normalization;
desired-to-safe delta/angle isolate obstacle safety. No stateful policy is
called twice to reconstruct past decisions.

- Distances/positions/radii/obstacle bounds: meters.
- Velocities: m/s; accelerations: m/s^2; native timestep: seconds.
- Desired directions/residual action/observations: dimensionless.
- Headings, heading changes, residual/avoidance angles: degrees.
- PN LOS rate: rad/s (native cross-product rate); do not silently label degrees.
- Wall-clock RMPC decision time: seconds; not replay simulation time.
- Native position has z-up; Three.js projection `[x,h_visual,-y]` uses fixed
  visibility heights only. Keep original altitudes available in the inspector.
- Heading is undefined at near-zero horizontal norm: null, not zero degrees.

Static geometry stores ID, native min/max corners, map knowledge and optional
decorative skin. `visual_kind="tree"` is not an inferred native physics type.
Dynamic geometry stores stable ID, native type, position/velocity/radius in each
frame. Browser playback must not integrate moving obstacles itself.

Pydantic validates shapes, finite values and enums. The recorder additionally
validates reset/terminal ordering, contiguous step numbers, T+1 length,
timestamps, final outcome, model observation/action shape and checkpoint
integrity. Tests compare the full native states/actions/rewards/dynamic
obstacles against `defender_adaptation_evaluation._run_matched_episode` for an
identical frozen configuration and seed.

## Telemetry and access separation

`policy_visible` contains only sanctioned own/asset state, detection and
measurement OR Kalman estimator fields. The typed model rejects mixing those
channels and rejects ground-truth extra fields. Use the existing sanitizer for
actual control; decision telemetry serializes that same sanitizer output and
the already-used observation. The test verifies that privileged attacker
position, velocity, full simulator info and obstacle geometry never enter the
defender input. Obstacle features come from the existing perceived set, never
full ground-truth geometry.

`privileged_diagnostics` contains native actual attacker position/velocity,
tracking error, ranges, raw returned simulator info, and radial closing speed
derived from defender/attacker positions and velocities (positive means closing).
It is recorder-only
and labeled privileged in the future inspector, never fed to the defender.
The whole browser scene is an evaluator view, not a claim about what the
controller knew. Optional `unavailable_fields` gives field-path -> reason.

Common policy telemetry:

- Single requested three-component action and decision step; normalized desired
  direction, post-safety safe command, controller-action-executed flag, standby,
  avoidance activity/angle.
- Separate requested-to-desired normalization and desired-to-safe intervention
  vectors/angles.
- Actual constrained executed velocity (do **not** equate safe heading to
  realized velocity; acceleration, turning and climb constraints still act).
- Requested horizontal heading, post-step velocity heading and signed
  horizontal heading change. The replay metadata records the 120-degree
  heading-reversal diagnostic threshold; it is a visual diagnostic, not a
  simulator or policy threshold.
- PPO/PPO-Kalman use these fields. Kalman e_hat/v_hat are current policy-visible
  channels; current actual attacker state and tracking error are privileged.
- Analytical Lead: guidance mode, target state, intercept time/point,
  discriminant from `last_*`. PN: LOS/range/rate/closing speed/acceleration and
  mode. RMPC: modes, best sequence, robust/per-scenario scores, evaluation count
  and timing when enabled. Greedy has no fabricated optimization internals.

## LR-PPO angular diagnostics

The existing composition is authoritative:
norm-clip residual, remove its Lead-parallel component, form
`u_lead + tan(30 deg) * residual_perpendicular`, then normalize.
Keep the canonical native angular deviation <= 30 deg (small numeric tolerance).
Do not alter the function or model to make a nicer visualization.

Plan three distinguishable arrows from the same defender:

1. Analytical Lead direction.
2. Final LR-PPO requested guidance (before environment safety).
3. Actual executed movement after standby, safety and motion constraints.

Safe command can be an additional dashed arrow. Privileged display arrows
are not extra policy inputs. Before detection, Lead may be escort/fallback;
zero vectors must be shown as undefined guidance, not arbitrary arrows.

Optional `lrppo` fields support Lead direction, raw PPO residual,
raw norm, perpendicular component, norm-clipping flag, unsigned native
`native_adjustment_deg`, signed horizontal adjustment, fixed max 30,
and near-bound flag with explicit margin.
Raw residual/clipping cannot be reverse-engineered from final heading:
the engine records Lead direction and actual final requested guidance from the
single wrapper call, derives the native 3D angle and signed horizontal angle,
and verifies the native angle against the 30-degree cap. Raw residual, projected
residual and clipping remain null with reasons because the wrapper does not
expose them. No second policy call or scientific source edit is used.

Signed horizontal adjustment:
`atan2(cross2(lead_xy,requested_xy), dot(lead_xy,requested_xy))` in degrees,
CCW-positive, null if either horizontal component is degenerate.
This projected angle can exceed +/-30 when native guidance is steep.
**The 30 deg bound is the native 3D angle, not projected azimuth.**
The original prompt's +/-30 signed angular visualization is therefore shown
as a native bounded cone plus a separately labeled horizontal signed arc.
Safety-induced or dynamics-induced executed movement can lie outside the
residual cone and must never be falsely labeled a residual violation.
Near-bound residual angle, residual-norm clipping, SB3 action-space clipping,
and environment motion saturation are different phenomena; do not collapse
them into one "clipped" indicator.

## Outcomes and events

Preserve all eight native terminal outcome labels. Environmental attacker
collisions are not interceptions. Events record kind, source native field,
step/time, entity, optional obstacle ID/contact point, optional fractional
contact timing **only when actually available**.
Use native collision event steps/points; time at a completed step is an interval
boundary, not a claim of exact substep interception time. Do not infer a more
precise event time from visual overlap. Summary stores native outcome,
step count/duration, total reward and nullable interception/collision times.
Keep `terminated` separate from `truncated`.

## Future playback

Play/pause, forward/back step, scrub, speed, replay, previous/next episode and
outcome summary use immutable recorded Python frames. Scrubbing seeks an index;
it never calls policy inference or modifies simulator state. No browser physics,
interception checks, outcome inference or obstacle regeneration is allowed.
Playback interpolation, if later added, is visual only and must not change
recorded event/state evidence. Phase 1 implements none of these controls.
