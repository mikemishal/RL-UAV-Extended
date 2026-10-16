# Replay JSON v1 / telemetry design

Version: `1.0.0`. This is a recording contract, **not** an implementation of
episode execution. No example in the UI is backed by invented frames.

Canonical definitions: [replay_contract.py](../backend/replay_contract.py).
Machine schema: [replay_v1.schema.json](replay_v1.schema.json), generated with
Pydantic `ReplayBundle.model_json_schema()`. Models forbid unknown keys and
non-finite numeric fields; JSON must never contain NaN/Infinity.
Unavailable infinite clearance values from the scientific info dict must
be normalized to null only in explicitly optional fields, with an unavailable
reason, not replaced with a fictitious large finite distance.

## Bundle and provenance

| Field | Meaning |
|---|---|
| `schema_version` | literal `1.0.0`; incompatible layouts require a new version |
| `coordinate_system` | `simulation_xyz_z_up_m`; preserve native three-component source data |
| `presentation` | `horizontal_projection_fixed_heights`; not a change to data or dynamics |
| `provenance` | 40-hex simulation/visualizer commits, source dirty flag, deterministic inference, Python/package versions, device, original registry SHA-256 |
| `episode_count` | 1,3,5,10; planned default 5 |
| `seed_start` | nonnegative explicit simulation seed (not a model training seed) |
| `episodes` | exact count, consecutive seeds in recording order (validated in Pydantic) |

Each episode stores `episode_id`, `simulation_seed`, defender and attacker
selection snapshots, complete environment snapshot/hash, static geometry,
all real frames, events and final summary. Selection includes catalog ID,
generation, training seed, observation/action shapes, estimator and exact
parameters. Checkpoint includes stable artifact ID, research-relative path,
SHA-256 and role. Analytical/A0 checkpoint fields are null; never synthesize
a checkpoint hash for them. Paths here are provenance, not a future file API.

`EnvironmentSpec.config` is the full JSON-serialized `EnvConfig` dataclass,
with an explicitly specified stable sorted JSON hash in Phase 2; include dt,
dimensions, thresholds, sensing, avoidance, collision/reward defaults and
every builder override. `builder` identifies the canonical journal function.
Capture the versioned sector specification in policy parameters, not merely
the number 20. Record frozen selected RMPC parameters and residual limit 30.

## Frame timing and units

Record frame 0 immediately after reset and frame k after completed transition
k (k >= 1), including terminal state. T transitions require T+1 frames.
Frame times are simulation seconds: `time_s = step * dt_s`, not wall-clock
rendering time. An initial frame has null telemetry and reward.

Every frame stores current defender/attacker/asset position, available velocity,
all moving obstacle states, **current** policy-visible observation/estimator
channels, privileged diagnostic state, reward, termination/truncation and native
outcome. `PolicyVisible.observation` is the base 16-D environment observation;
20 sector features are a separate optional array. Model input shape lives in
selection metadata: 16/19/36/39, not an ambiguous flattened state.
An A1 current observation, when captured, is a separate optional 36-D array.

For k >= 1, telemetry describes the transition **ending at frame k**:
`telemetry.decision_step = k-1`. Its requested action was based on the
policy-visible data in frame k-1. Safe command and executed velocity come
from the resulting `step` / frame k info. This avoids comparing a request
to next-step sensor data. For Kalman overlays, show current frame estimates
against current frame truth; for decision explanation use the explicitly
identified prior frame. Do not call a stateful policy twice to reconstruct
past decisions.

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

Pydantic validates shapes, finite values, enums, count and seed order. The
Phase 2 recorder/tests must additionally verify reset/terminal ordering,
continuous step numbering, T+1 length, timestamps, full config/hash agreement,
source/checkpoint integrity, policy/observation compatibility, event consistency
and that no state was mutated during capture. These semantic rules are not all
expressible by JSON Schema and must not be mistaken for already implemented
rollout validation.

## Telemetry and access separation

`policy_visible` contains only sanctioned own/asset state, detection and
measurement OR Kalman estimator fields. The typed model rejects mixing those
channels and rejects ground-truth extra fields. Use the existing sanitizer for
actual control; this schema is not a replacement sanitizer.
Obstacle features must come from the appropriate perceived set, never full
ground-truth geometry.

`privileged_diagnostics` contains native actual attacker position/velocity,
tracking error, ranges and optional derived closing speed. It is recorder-only
and labeled privileged in the future inspector, never fed to the defender.
The whole browser scene is an evaluator view, not a claim about what the
controller knew. Optional `unavailable_fields` gives field-path -> reason.

Common policy telemetry:

- Single requested three-component action and decision step.
- Safe command (post obstacle safety), controller-action-executed flag,
  standby, avoidance activity/angle.
- Actual constrained executed velocity (do **not** equate safe heading to
  realized velocity; acceleration, turning and climb constraints still act).
- Optional requested/executed horizontal headings, signed heading change,
  reversal. Reversal threshold must be explicitly recorded in policy parameters
  when implemented, not guessed in Phase 1.
- PPO/PPO-Kalman use these fields. Kalman e_hat/v_hat are current policy-visible
  channels; current actual attacker state and tracking error are privileged.
- Analytical Lead: guidance mode, target state, intercept time/point,
  discriminant from `last_*`. PN: LOS/range/rate/closing speed/acceleration and
  mode. RMPC: modes, best sequence, robust/per-scenario scores, evaluation count
  and timing when enabled. Greedy has no fabricated optimization internals.

## LR-PPO angular diagnostics (Phase 5 plan)

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
currently unavailable fields stay null with a reason. Future observational
capture must use the same **one** deterministic prediction and canonical
composition, with equivalence tests and no policy architecture edits.

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
