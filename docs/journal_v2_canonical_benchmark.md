# Canonical V2 A0 benchmark (frozen)

**Status: FROZEN.** The corrected V2 scripted attacker (A0) was inspected manually and approved on
2026-10-25. Any scientific change to the components listed under *Frozen components* requires a new
version (a new tag, not an edit of this one). This document is the reproducible record of what was frozen
before any V2 A1 or defender retraining.

## Repository provenance

| | |
|---|---|
| Research repository | `mikemishal/RL-UAV-Extended` (local worktree `RL-UAV-Extended-lrppo-clutter`) |
| Branch | `research/journal-corrected-attacker-v2` |
| A0 environment commit (last commit that changed simulation code) | `8e2d59115423d329156d5bc468bf594019f4dc03` (`feat: add finite mission endurance to corrected V2 attacker`, 2026-10-25) |
| Frozen benchmark tag | `journal-v2-a0-canonical-v1` (annotated; resolves to the commit that adds this document and the training-contract guard; no file under `uav_defend/` differs from `8e2d591`) |
| Superseded, unmoved tag | `journal-corrected-attacker-semantics-v2` -> `db8c2995fb08279fff3d98e5fe1fef9a7a72e2ce` (the first V2 semantics checkpoint; kept as is) |
| Visualizer (tooling, not part of the science) | branch `tooling/journal-visualizer`, tag `journal-v2-a0-visualizer-v1`, pinned to the research commit above |

Every V2 replay records the research commit and a source-dirty flag; the visualizer refuses to run against
any research commit other than its pinned one.

## Canonical V2 configuration

Built only through `experiments.journal_v2_protocol.build_v2_env_config(level, estimator, awareness_mode=...)`.
Effective values (identical for C0-C3 unless stated; recorded from the frozen revision):

| Group | Field | Value |
|---|---|---|
| Contact | `mission_contact_mode` | `swept` (attacker->asset capture and defender->attacker interception are swept-sphere contacts) |
| Guidance | `enemy_nominal_guidance_mode` | `direct_obstacle_only` |
| Avoidance | `SoldierEnv.STATEFUL_ENEMY_AVOIDANCE` | `True` (stateful receding-horizon controller: DIRECT / LEFT / RIGHT / CLIMB) |
| | `enemy_obstacle_avoidance_enabled` | `True` for C1-C3; `False` for C0 (C0 has no obstacles) |
| | `enemy_obstacle_awareness_mode` | `map_local` (also `local` selectable); C0 `disabled` |
| | `enemy_obstacle_clearance` | 2.0 m |
| | `enemy_obstacle_prediction_horizon_steps` | 12 |
| | `enemy_obstacle_release_steps` | 4 |
| | `enemy_obstacle_sensor_range` | 15.0 m |
| | `enemy_route_planning_enabled` | `False` |
| Asset | `asset_motion_model` | `legacy_gaussian_random_walk` (`asset_motion_tau_s=1.0` is unused under this model) |
| | `v_s` | 1.5 m/s |
| Endurance | `enemy_endurance_enabled` | `True` |
| | `enemy_endurance_path_factor` | `5.0` |
| | terminal outcome / reward | `enemy_endurance_exhausted` / `-100.0` (placeholder equal to `reward_timeout`) |
| Attacker dynamics | `v_e` | 12.0 m/s |
| | `enemy_max_accel` | 6.0 m/s^2 |
| | `enemy_max_turn_rate_deg` | 75.0 deg/s |
| | `enemy_max_climb_rate` / `enemy_max_descent_rate` | 5.0 / 5.0 m/s |
| Time / domain | `dt` | 0.5 s |
| | `max_steps` | 2000 |
| | `L` / `max_altitude` | 50.0 m half-size / 30.0 m |
| Radii | `threat_radius` (capture) | 2.0 m |
| | `intercept_radius` | 2.5 m |
| | `unsafe_intercept_radius` | 3.5 m |
| | `detection_radius` | 15.0 m |
| Defender | `defender_standby_until_detection` | `True` (`v_d = 18.0 m/s`) |
| Clutter (C0 / C1 / C2 / C3) | static / moving obstacles | 0/0, 3/1, 8/2, 15/4 |

Resolved-config fingerprints (SHA-256 of the sorted-key JSON of `dataclasses.asdict(config)`, `awareness_mode="map_local"`;
the same recipe the visualizer records as `config_sha256`):

| Level | measurement | kalman |
|---|---|---|
| C0 | `858758ea1fc1f2749affd8f836713318981f77df6b99b0b8ad13fb16d1df5cd6` | `5d1ff5c1ac0593a07129ca3a0718eb5fcf1df5eaf533680b2179e5c4c00b9eaf` |
| C1 | `ca28f37f2e314bf16a26dc28a5cc11bc738935765f9d0ef413d995054d01c276` | `2bbbb254b3f5caa59dfa0d4c6bb0f6e5306a6cb7f6fc954db88969e7a321c324` |
| C2 | `7f922d0569e4dc5327aecaec6603975a620ccfad42ada33c73457bcc4ace8a23` | `999159d67f7030ec43027e7a320043d17481937a57297b0704724881d5f88a60` |
| C3 | `734d50cc2496601fd754f79aebbabd5d78bbf4733f585ede8e8a106eb88bd186` | `360c56680b3ce3ee1745bb47aea965e1c53e36df8c3a5f571ea89ec5717d78d7` |

## A0 semantics

Priority order: **physical safety > obstacle avoidance > direct mission pursuit.**

A0 attacks the asset directly when its path is clear; it never deliberately evades defenders (its trajectory
is independent of the defender's behavior -- verified: maximum attacker-position delta 0.0 m between passive,
Greedy and Lead over 76 paired episodes); it has no periodic weave; it deviates only for obstacle safety (the
stateful controller releases back to DIRECT as soon as the executed DIRECT rollout is collision free); and it
flies under a finite mission endurance, so it may legitimately end an episode with `enemy_endurance_exhausted`.
Clear-space behavior (C0): pure direct pursuit, always in the mission mode, path-excess about 0.97 (e.g. C0 seed
130000: 12 steps, 68.4 m for a 70.2 m chord).

Within a step, the physically earliest event wins (swept contacts, obstacle collisions and endurance exhaustion
share one first-event resolver; ties use the fixed order defender collision > attacker collision > capture >
interception > endurance). Event ordering is frozen.

## Asset model

Canonical: **`legacy_gaussian_random_walk`** (an independent Gaussian displacement every step, sigma = `v_s * dt`).

A temporally correlated Gauss-Markov asset model (`gauss_markov_velocity`) is implemented, config-gated and
**not canonical**. It was evaluated over 2000 paired episodes (C0-C3 x 100 seeds x 5 settings) and rejected:
every tested persistence setting introduced new collision failures in C2/C3 (1-3 per setting, against 0/2000 for
the legacy model), moved rather than removed the long-pursuit pathologies, and changed benchmark capture
difficulty. See `docs/journal_v2_asset_motion_model_study.md`.

## Finite mission endurance

\[
B_{\text{path}} = 5\,d_0, \qquad L_{t+1} = L_t + \lVert p_{t+1} - p_t \rVert ,
\]

with `d0` the initial 3-D attacker-to-asset distance and `p` the actually integrated attacker position (never
commanded or nominal speed). The episode ends as `enemy_endurance_exhausted` when `L` reaches `B_path`; the
within-step exhaustion fraction is `tau_E = clip((B - L_t)/||p_{t+1}-p_t||, 0, 1)` and competes with the other
step events as the lowest-priority candidate.

**This is a normalized finite mission-endurance / path budget. It is not a calibrated battery-energy model**;
no empirical UAV energy-consumption model exists in this project and `k = 5` makes no claim about real flight
time. No guidance code reads the remaining budget, so the attacker's trajectory is bit-identical to the
no-endurance trajectory up to the exhaustion instant (the budget only ever terminates an episode).
Full derivation: `docs/journal_v2_mission_endurance.md`.

## Known rare failure: C3 seed 130017 (part of the benchmark characterization)

- Without endurance (stateful avoidance, legacy motion): after a brief avoidance the attacker passes the asset
  inside its turning circle and the unchanged DIRECT command circles it for 346 steps / 172.7 s / 1828.5 m
  (path-excess 27.06x) before capture. Four independent attempts to resolve this through attacker guidance or target
  motion (terminal speed governor, lead intercept, reactive reacquisition, correlated asset motion) either left it
  essentially unfixed or fixed it only by introducing a new collision elsewhere; see
  `docs/journal_v2_terminal_*.md` and the asset-motion study.
- With canonical `k = 5`: `enemy_endurance_exhausted` at step 54 (26.8 s of flight, 285.65 m = 5 x 57.13 m),
  no collision, attacker still at 9.6 m from the asset.
- Population effect (400 passive-A0 episodes per condition, C0-C3 x seeds 130000-130099): for each of
  k = 3, 4, 5 exactly one episode changed (this one, `soldier_caught` -> `enemy_endurance_exhausted`); 0
  collisions in all 1600 sweep episodes; the preceding trajectory is bit-identical. k = 5 was selected as the
  least restrictive of the tested factors.
- This case is not hidden: under a defender that engages (Greedy, Lead) 130017 is intercepted before the budget
  binds, so it only appears as an endurance failure against a passive/ineffective defender (the visualizer's
  `passive` option reproduces it).

## Verification evidence

### Endurance (docs/journal_v2_mission_endurance.md)

| k | C3 outcomes | C3 max path | C3 max path-excess | changed seeds | collisions (1600-episode sweep) |
|---|---|---|---|---|---|
| disabled | 100 soldier_caught | 1828.5 m | 27.058 | - | 0 |
| 3 | 99 caught + 1 exhausted | 171.4 m | 3.409 | 130017 | 0 |
| 4 | 99 caught + 1 exhausted | 228.5 m | 4.539 | 130017 | 0 |
| 5 (canonical) | 99 caught + 1 exhausted | 285.6 m | 5.407 | 130017 | 0 |

C0, C1, C2 are unchanged at every k. Greedy/Lead attacker-trajectory independence under k = 5: 76 pairs, max delta 0.0 m.

### Stateful avoidance (docs/journal_v2_planner_prediction_fidelity.md)

Measured on this frozen revision (passive A0; chord = straight-line distance to the asset's final position):

| Case | Historical (documented) | Frozen revision |
|---|---|---|
| C3 130006 | 12.98x detour under the waypoint planner (0.97x before) | 10 steps, 53.9 m, 0.987x |
| C2 130065 | 2.09x (legacy), 1.32x (waypoint planner): large climb detour | 10 steps, 56.0 m, 0.974x |
| C3 130063 | 4.47x (legacy), 0.99x (waypoint planner) | 12 steps, 69.7 m, 1.002x |
| C3 130081 | 3.06x (legacy), 0.98x (waypoint planner) | 11 steps, 61.7 m, 0.980x |
| C3 130064 | 1.20x foundation (11.8x under a rejected variant) | 12 steps, 66.1 m, 1.000x |
| C2 130054 | 11 steps foundation (346 under a rejected variant) | 11 steps, 57.3 m, 1.003x |
| C0 130000 (clear space) | - | 12 steps, 68.4 m, 0.974x, no avoidance |

Population gates for the stateful controller (300 paired seeds C1-C3, then C0-C3 re-checks): no attacker collision
gained or lost, capture rate 1.00 per level, C3 mean path-excess 0.974 excluding 130017; C0, C1 and every V1
trajectory bit-identical to before. Precision of the historical column is limited to what the tracked
diagnostic documents record.

## Frozen components

A0 mission guidance; swept-contact semantics; stateful obstacle avoidance (DIRECT / LEFT / RIGHT / CLIMB and the
blocker set); obstacle perception; event ordering; the canonical `legacy_gaussian_random_walk` asset model; finite
mission endurance (`k = 5`); capture / interception radii; attacker dynamics. Changing any of them is a new
benchmark version.

## Canonical V2 Training Contract

All future V2 A1 and defender training must use, from this frozen revision: canonical V2 semantics (swept mission
contact, direct obstacle-only A0), `legacy_gaussian_random_walk` asset motion, the stateful obstacle controller
(`SoldierEnv.STATEFUL_ENEMY_AVOIDANCE = True`) and finite mission endurance with `k = 5`. No training script may
silently disable these; a named counterfactual experiment must say so explicitly and must not be reported as a
canonical V2 result.

Enforcement (configuration only, no change to dynamics), in `experiments/journal_v2_protocol.py`:

- `assert_canonical_v2_training_config(config)` / `canonical_v2_config_violations(config)`: fail fast (or list every
  violation) unless the config has the V2 semantics, endurance `True` with `k = 5`, the legacy asset model, the
  frozen attacker dynamics / radii / `dt`, canonical standby and route-planning settings, and the stateful switch.
- `build_randomized_v2_training_config(rng, estimator)`: the existing randomized C1-C3 training-clutter
  distribution with the canonical V2 fields applied.
- `guard_canonical_v2_config_fn(config_fn)` and `assert_canonical_v2_env_fn(env_fn, seed)`: wrap a config factory, or
  reset one environment from a training `env_fn` (so wrappers that re-sample the config on reset are seen) and assert
  it.
- `require_canonical_v2: bool = False` on `run_adaptive_attacker_training`, `run_clutter_training` and
  `run_defender_adaptation_training`. **Every V2 campaign must pass `require_canonical_v2=True`**; the check runs
  before anything is created, loaded or trained. The default stays `False` so that V1 reproductions are unchanged.

Audit finding that motivated this: the existing journal training environments are V1. The attacker env
(`AdaptiveAttackerTrainingEnv`) defaults to `build_randomized_training_config`, the defender clutter env
(`RandomizedClutterWrapper.reset`) replaces the config with the V1 randomized one on every reset, and the
`build_ppo_d1_env` / `build_lr_ppo_d1_env` wrappers reuse it, so a V2 campaign built on them would silently train on
V1 (endpoint contact, no endurance) unless guarded. Existing A1 and defender checkpoints are V1-trained.

## Handoff

`docs/journal_v2_training_handoff.json` records the identifiers for the next phase, including the V1 A1 checkpoint
hashes (provenance only). **Existing A1 checkpoints are V1-trained and MUST NOT be used as canonical V2 results.**
