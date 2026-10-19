# Journal results provenance: V1 vs V2

This document records which simulation semantics produced which datasets, so
that no result is ever attributed to the wrong physics.

## V1 — original semantics (frozen)

* **Contact:** attacker→asset capture and defender→attacker interception are
  tested **only at the final positions of each 0.5 s step** (`endpoint`).
  A body moving ~6 m per step can pass through a 2.0 m capture sphere
  (4 m diameter) or a 2.5 m interception sphere between samples without
  registering contact.
* **Scripted A0:** pursuit + AR(1) lateral weave + heading noise + reactive
  defender evasion (`legacy`). Weave and evasion are active even in clear
  space.
* **Frozen at:** branch `research/lr-ppo-clutter`, tag
  `journal-robustness-assurance-v1` (commit `a3bc6c4`), plus the
  documentation-only commit that follows it. All earlier journal tags
  (`journal-canonical-c0-evaluation-v1`, `journal-lrppo-environment-audit-v1`,
  `journal-architecture-diagnostics-v1`, …) are V1.
* **V1 datasets (preserved, never modified):**
  `results/defender_adaptation_*`, `results/journal_robustness_assurance/`,
  `results/adaptive_attacker_*`, and every other directory present at the V1
  freeze.
* **Use:** provenance and *sensitivity comparison*. They remain valid
  descriptions of the V1 simulator. They are **not** automatically the final
  V2 publication results.

## V2 — corrected semantics

* **Contact:** `mission_contact_mode="swept"` — continuous moving-sphere
  contact over each step; the physically earliest event of the step wins.
* **Scripted A0:** `enemy_nominal_guidance_mode="direct_obstacle_only"` —
  direct pursuit; deviation only while an obstacle blocks the direct path;
  immediate return to direct pursuit once it is clear; no weave, noise or
  evasion.
* **Branch:** `research/journal-corrected-attacker-v2`
  (created from V1 HEAD `e72d421`).
* **Results namespace:** `results/journal_v2_corrected_semantics/`. Every
  future corrected-semantics result is written there.
* **Builder:** `experiments/journal_v2_protocol.build_v2_env_config` sets both
  switches explicitly and asserts them. Publication runs never rely on an
  `EnvConfig` default.

## Guarantees

1. V1 data are **not deleted, renamed or rewritten**.
2. V2 data will **never overwrite** V1 data (separate directory; the V2
   builder and result paths cannot point into V1 directories).
3. Large V1 datasets are **not copied**; V2 analyses that need a V1 baseline
   reference it in place and cite its tag.
4. `EnvConfig()` defaults are still the V1 values. Old builders reproduce V1
   bit-for-bit (verified: identical full-trajectory hashes against the frozen
   V1 code for C0/C2/C3 × 3 seeds — see
   `tests/test_corrected_attacker_semantics_v2.py::V1_GOLDEN`).

## Seed namespaces

| Block | Range | Use |
|---|---|---|
| V1 robustness evaluation | 200,000–200,099 | V1 only |
| V2 micro-validation (this task) | 300,000–300,049 | attacker-motion diagnostics only; do not reuse for publication evaluation |

## Before-state: the legacy A0 motion stack (actual code order)

`SoldierEnv.step()` in `uav_defend/envs/soldier_env.py`:

```
step()
 A  dynamic obstacles advance                       (if enabled)
 B  _move_soldier():  asset random walk (σ = v_s·dt per axis), blocked by obstacles
 C  standby: defender synchronised onto the asset   (until detection)
 D  _move_enemy():
      D1  mission command   r̂ = unit(asset − attacker)
      D2  perception update  (EnemyObstaclePerception; Level B/C)
      D3  defender-reactive evasion  α = clip(1 − d/20, 0, 1),  component = 0.75·α·û_away   ← evasion introduced here
      D4  route arbitration  (suspend route while evading / hysteresis)
      D5  optional route planner  (replaces r̂ with a waypoint direction; off in canonical C2)
      D6  weave   a ← 0.85·a + 0.5·η ;  weave = a·1.5·lateral(r̂)                          ← periodic weave introduced here
      D7  heading noise  0.15·z   (3-D normal)
      D8  u_base = normalize(r̂ + weave + noise + evasion)
      D9  learned tactical override replaces u_base      (A1 only)
      D10 ObstacleAvoidancePlanner on u_base:
            activate  if a 12-step constant-command prediction hits a physical obstacle;
            release   after 4 consecutive clear checks AND clearance ≥ clearance + v·dt
      D11 constrained dynamics (accel 6 m/s², turn 75°/s, climb/descent 5 m/s) → position
 E  _update_detection()
 F  defender action executed through its constrained dynamics    (if controller in control)
 G  swept OBSTACLE collisions (attacker and defender), clamp to first contact
 H  END-OF-STEP contact tests on final positions only:
        asset capture   : |attacker − asset|   ≤ threat_radius (2.0)      ← capture evaluated here
        interception    : |defender − attacker| ≤ intercept_radius (2.5)  ← interception evaluated here
        unsafe          : |attacker − asset|   ≤ unsafe_intercept_radius (3.5)
      precedence: obstacle collision > soldier_caught > unsafe_intercept > intercepted > timeout
```

Where each behavior enters:

| Behavior | Where | V2 |
|---|---|---|
| Periodic weave | D6 | removed for A0 |
| Heading noise | D7 | removed for A0 |
| Defender-reactive evasion | D3 (component), D4 (route suspension), D8 | removed for A0 |
| Obstacle avoidance overrides the base command | D10, only after a predicted collision | unchanged trigger; faster release; look-ahead truncated at the asset |
| Asset capture / defender interception | H | continuous (swept) test with earliest-event arbitration |

## After-state: the V2 A0 motion stack

```
step()
 A  dynamic obstacles advance
 B  _move_soldier()
 C  standby sync
 D  _move_enemy():
      D1  mission command   r̂ = unit(asset − attacker)
      D2  perception update
      D3  evasion DIAGNOSTICS only (never enters the command)
      D4  (route planning, opt-in) direct line clear? → discard route, keep r̂
                                  blocked?            → route waypoint direction
      D6/D7  skipped (no weave, no noise, no RNG draws)
      D8  u_base = r̂
      D9  learned tactical override (A1) unchanged
      D10 ObstacleAvoidancePlanner (physical safety > bypass > direct pursuit):
            activate  if the look-ahead, TRUNCATED at the asset capture sphere, hits an obstacle;
            release   as soon as the base command is predicted clear AND the straight line to the asset touches no obstacle
      D11 constrained dynamics unchanged
 E  _update_detection()
 F  defender action
 G  obstacle-collision times (static and dynamic) for both movers
 H  SWEPT mission contacts over the whole step (moving sphere vs moving sphere):
        capture τ_c, interception τ_i  (first-contact fractions)
        earliest of {defender collision, attacker collision, capture, interception} wins;
        tie within 1e-9 → defender collision > attacker collision > capture > interception
        losing collisions are rolled back; safe/unsafe classification is evaluated at the interpolated instant
```
