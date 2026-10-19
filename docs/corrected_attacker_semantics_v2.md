# Corrected attacker semantics (journal V2)

## Problem

Two modeling issues were found while visually inspecting episodes:

1. **Endpoint tunneling.** Attacker→asset capture and defender→attacker
   interception were tested only at the final positions of each 0.5 s step.
   The attacker moves up to `v_e·dt = 6 m` per step while the capture sphere
   (`threat_radius = 2.0 m`) is 4 m across, so the attacker could fly straight
   through the asset between two samples and keep going. The 2.5 m
   interception sphere has the same weakness.
2. **Scripted A0 weaves in clear space.** A0 added an AR(1) lateral weave
   (amplitude 1.5), heading noise, and defender-reactive evasion to its
   pursuit command. Combined with the attacker's turn-rate limit (75°/s at
   12 m/s, ≈ 9 m turn radius) this produced long trajectories and visible
   orbiting around the asset even with nothing in the way.

## V1 behavior (unchanged, still reproducible)

* `mission_contact_mode = "endpoint"` — contact tested at step endpoints only.
* `enemy_nominal_guidance_mode = "legacy"` — pursuit + weave + noise +
  evasion; persistent-release avoidance planner.
* These are the `EnvConfig` defaults. Every dataset frozen at
  `journal-robustness-assurance-v1` was produced this way.

## V2 correction

### 1. Swept mission contacts (`mission_contact_mode = "swept"`)

For two bodies A, B moving linearly over a step (τ ∈ [0, 1]):

```
r0 = pA0 − pB0,   dr = (pA1 − pA0) − (pB1 − pB0),   r(τ) = r0 + τ·dr
```

The first contact is the smallest τ with ‖r(τ)‖ ≤ R. It is computed through
the closest point of approach,

```
τ_cpa = −(r0·dr)/(dr·dr),   d_cpa = ‖r(τ_cpa)‖,
τ_in  = τ_cpa − sqrt(max(R² − d_cpa², 0)/(dr·dr)),
```

which stays accurate for fast, nearly tangent passes (no cancellation).
Handled explicitly: already inside at τ = 0 (fraction 0), zero relative
displacement, tangent contact (1e-9 m tolerance), pass-through with both
endpoints outside, both bodies moving, and the contact window lying outside
[0, 1]. Implemented in `uav_defend/obstacles/swept_contact.py`
(`first_contact`) with independent tests (`tests/test_swept_contact.py`),
including a dense-sampling cross-check. Radii are **not** inflated.

* **Attacker→asset capture** uses the existing `threat_radius` (2.0 m), with
  the asset's own motion included.
* **Defender→attacker interception** uses the existing `intercept_radius`
  (2.5 m). During pre-detection standby the defender's synchronisation onto
  the asset is bookkeeping, not flight, so it is treated as stationary that
  step (the same rule already used for its obstacle sweep).
* The existing safe/unsafe classification (`unsafe_intercept_radius`, 3.5 m)
  is evaluated on the **interpolated states at the interception instant**.

### 2. Event ordering within a step

Candidate events: defender obstacle collision, attacker obstacle collision,
attacker captures asset, defender intercepts attacker. All are timed as
fractions of the step. **The earliest wins.** Ties (within 1e-9 of a step)
use a fixed, documented priority:

> defender collision > attacker collision > asset capture > interception

(mission failure outranks mission success; collisions outrank both — the same
ordering V1 used at the end of a step.) Events that lose are discarded; a
collision that would have happened *after* a mission event is rolled back, so
it is neither counted nor allowed to end the episode. In V1 any collision in
a step overrode any mission outcome regardless of timing; V2 uses true time
order. The state at a mission-contact instant is the interpolated state, and
`info` reports `terminal_event`, `terminal_event_fraction`,
`capture_contact_fraction/position/distance` and the interception equivalents.

### 3. Corrected scripted A0 (`enemy_nominal_guidance_mode = "direct_obstacle_only"`)

* **Clear path:** command = `unit(asset − attacker)`, passed through the
  unchanged acceleration / turn-rate / climb-descent / speed limits. No weave,
  no heading noise (so A0 consumes no random numbers), no defender-reactive
  evasion. Tactical evasion belongs to the learned A1 attacker.
* **Obstructed path:** the existing perception and avoidance machinery
  applies, with the hierarchy *physical safety > required bypass > direct
  pursuit*. Level B uses locally perceived obstacles; Level C uses mapped
  static plus locally perceived dynamic obstacles (no ground-truth leakage —
  unchanged).
* **Return to direct pursuit:** avoidance is released as soon as the base
  command is predicted collision-free **and** the straight line to the asset
  touches no obstacle. There is no persistence margin and no stale waypoint.
  The straight-line test uses the same physical obstacle geometry that
  triggers avoidance; a clearance-inflated test can never release when the
  asset lies within the clearance margin of an obstacle, which left the
  attacker hovering at its bypass waypoint (found during validation).
* **Look-ahead ends at the asset:** the avoidance planner's 12-step
  prediction is truncated where the predicted path first enters the capture
  sphere. A predicted collision *beyond* the asset is not a conflict because
  capture ends the episode first. Without this, obstacles behind the asset
  triggered spurious bypasses that pulled the attacker off its descent and
  caused overshoot loops above the asset.
* **Route planner (opt-in, off in canonical C2):** used only while the direct
  line is obstructed; as soon as it is clear the route is discarded
  (`RoutePlanner.invalidate_route`) so no stale waypoint survives.
* The legacy hysteresis path is untouched and is selected by the legacy mode.

### 4. Mission-efficiency instrumentation (no battery model)

Reported in `info` every step (observational only; never used by reward or
termination): `attacker_initial_range`, `attacker_path_length`,
`attacker_flight_time_s`, `attacker_heading_change_total_deg`,
`attacker_overshoot_count` (approach→recede transitions of range-to-asset,
1 m/s deadband), `attacker_path_efficiency` (= initial straight-line range /
path length) and its capped value. The asset random-walks, so the ratio can
marginally exceed 1 (hence the capped value).

### 5. Trace export

`experiments/journal_v2_trace.py` writes one JSONL row per step with asset,
attacker and defender positions, attacker velocity, mission direction, final
command, behavior mode (`DIRECT` / `OBSTACLE_AVOID` / `ROUTE_BYPASS` /
`EVADE`), blocking state, predicted-collision data, nearest-obstacle
clearance, perceived obstacle ids, detection state, contact fractions,
terminal event and outcome. It is the input contract for the visualizer.

## Validation (behavioral, not a defender study)

`experiments/journal_v2_micro_validation.py` runs 50 seeds (300,000–300,049)
per canonical level under V1 and V2 semantics. The defender is parked far away
and never acts, so only the attacker's motion is measured. Results are in
`results/journal_v2_corrected_semantics/micro_validation/`.

| | C0 V1 | C0 V2 | C2 V1 | C2 V2 | C3 V1 | C3 V2 |
|---|---|---|---|---|---|---|
| Asset captured | 86% | **100%** | 68% | **100%** | 28% | **100%** |
| Attacker obstacle collision | 0% | 0% | 4% | 0% | 24% | 0% |
| Timeout | 14% | 0% | 28% | 0% | 48% | 0% |
| Missed sphere crossings (total) | 41 | **0** | 27 | **0** | 10 | **0** |
| Median steps to capture | 673 | 10 | 759 | 10.5 | 1200 | 10.5 |
| Median path efficiency | 0.014 | 1.04 | 0.011 | 1.03 | 0.007 | 1.02 |
| Mean cumulative heading change (°) | 13,827 | 20 | 18,116 | 52 | 23,847 | 116 |
| Episodes with an approach→recede overshoot | 47 | 0 | 50 | 2 | 49 | 5 |
| Episodes orbiting the asset ≥ 1 rev (within 20 m) | 42 | **0** | 39 | **0** | 35 | **0** |
| Mean avoidance segments / episode | 0.1 | 0 | 67 | 0.4 | 79 | 1.3 |
| Mean angle to mission direction after the last deviation | 41.6° | 0° | 31.9° | 0° | 12.8° | 0° |

Interpretation: V2 has no missed crossings, no sustained orbiting, and
returns to the exact mission direction after every obstacle deviation. The
residual overshoots in clutter (2 of 50 at C2, 5 of 50 at C3) are single
approach→recede transitions after an obstacle bypass, not loops. These numbers
describe attacker motion only; **no defender-performance conclusion should be
drawn from them.**

Two defects were found and fixed *during* this validation, both in the V2
code path only: (1) the avoidance look-ahead flew through the asset and
reacted to obstacles beyond it; (2) a clearance-inflated release test could
never release when the asset sat within the clearance margin of an obstacle,
leaving the attacker hovering at its bypass waypoint (4 of 50 C3 episodes
exceeded 150 steps before the fix; none after).

## Compatibility

* Defaults are V1: `EnvConfig()` reproduces V1 exactly. Verified bit-for-bit
  against the frozen V1 code (full-trajectory hashes, C0/C2/C3 × 3 seeds,
  `tests/test_corrected_attacker_semantics_v2.py::V1_GOLDEN`).
* `experiments.journal_v2_protocol.build_v1_env_config` builds an explicit V1
  config; `build_v2_env_config` builds the canonical V2 config and asserts both
  switches (and canonical standby). Publication experiments must use these
  builders and never rely on a default.
* The learned attacker's tactical override still replaces the nominal command
  exactly as in V1. Existing A1 checkpoints were trained under V1 contact
  semantics; evaluating them under V2 is a separate, deliberate step.

## Publication implication

V1 datasets are preserved for provenance and sensitivity comparison but are
**not** automatically the final V2 results. Defender, attacker and robustness
results that depend on A0 behavior or mission contact must be regenerated
under V2 before they are used as V2 publication evidence. No V2 defender
result has been produced yet.

## Battery / endurance

V2 **measures** efficiency (path length, flight time, heading change) but
does **not** impose a hard battery cutoff. An explicit endurance model is
deferred until the corrected motion has been validated visually.
