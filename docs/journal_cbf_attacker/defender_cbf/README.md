# A1-CBF — defender-avoidance stage (Prompt 4: joint obstacle + defender CBF-QP)

**Result: PARTIAL. Primary safety gate PASSES; primary defender-avoidance EFFECT gate and the 200-case joint
stress set's zero-collision requirement do NOT both pass. Generalization across 4 additional defender families is
strongly positive (PN -33pp, RMPC -38pp interception; both statistically significant). Verdict B — the mechanism
is correctly implemented and demonstrably effective against predictive/model-based defenders, but the numeric
acceptance thresholds chosen around the Greedy baseline are not met, and the 200-case stress set surfaced a rare
(1.5%) edge-case obstacle-collision limitation in the shared soft-slack mechanism. Manual review is required
before any controller freeze/tag.**

## 0. Motivation

Prompts 1-3 built a deterministic, optimization-based "evasive attacker" (pursuit law + quadrotor dynamics +
static-obstacle CBF-QP + viability-conditioned initialization) as a drop-in replacement for the failed PPO/BC
learned attacker. This stage (Prompt 4) is the FINAL controller-development stage: it adds active avoidance of the
defender UAV as an adversarial MOVING hazard, jointly with static-clutter avoidance and moving-asset pursuit, in
the SAME CBF-QP (never solved sequentially). Unlike Prompts 1-3 (which always disabled the defender via
`intercept_radius=1e-6`), this stage runs the REAL canonical defender dynamics and an ACTIVE policy
(Greedy/Kalman-Greedy/PN/Lead/RMPC) for the first time.

## 1. Architecture (frozen)

```
moving asset pursuit -> static route planner -> v_nom -> joint obstacle + defender CBF-QP -> v_safe -> velocity executor
```

The route planner handles ONLY static geometry (global/local-minimum deadlock around clutter); it never sees the
defender. The defender is a MOVING adversarial hazard, handled LOCALLY by the CBF-QP every `dt_sub=0.1 s` substep
-- replanning a global route around a constantly-moving interceptor would be architecturally wrong (brief
section 3). See `fig01_architecture_diagram.png`.

## 2. Defender information contract (verified from live code, not assumed)

`EnvConfig.enemy_defender_detection_range = 25.0` (`uav_defend/config/env_config.py:114`) is the SAME,
pre-existing gate the PPO/BC curriculum's own attacker observation uses
(`experiments/adaptive_attacker_env.py:build_attacker_observation`): defender position AND ground-truth velocity
are both zeroed unless `||defender_pos - attacker_pos|| <= 25 m`. A1-CBF reuses this EXACT gate/radius but
deliberately does **not** read the ground-truth velocity component that contract technically exposes -- it
independently estimates defender velocity via finite differences of ONLY the gated POSITION stream
(`DefenderVelocityEstimator` in `uav_defend/attackers/cbf_defender.py`), a stricter, more defensible sensing
assumption for a safety-critical filter (an onboard position sensor is a far more realistic real-world assumption
than a perfect velocity feed).

**Estimator contract (frozen):**
- First-ever valid observation: velocity estimate = zero, `valid=False` (no prior sample to differentiate against).
- A gap (defender unobservable for >=1 step, then observable again): treated as a FRESH first observation, never
  differentiated across the gap (would silently understate the TRUE closing speed).
- Deterministic: a pure function of the observation sequence, no randomness.
- When the defender is unavailable: NO defender CBF constraint exists at all (the joint QP degenerates exactly to
  the Prompt-2/3 obstacle-only QP, bit-for-bit -- verified by `test_full_variant_degenerates_to_obstacle_only_when_defender_unobservable`).

## 3. Defender safety/interception radii (verified from live code)

| quantity | value | meaning |
|---|---:|---|
| `R_intercept` (`EnvConfig.intercept_radius`) | 2.5 m | physical interception distance (attacker-defender), UNCHANGED |
| `R_D` (`DEFENDER_SAFETY_RADIUS`) | 3.5 m | CBF defender safety radius (attacker-defender) |
| `EnvConfig.unsafe_intercept_radius` | 3.5 m | **a DIFFERENT geometric quantity**: gates the ENEMY-ASSET (attacker-soldier) distance at the moment of physical interception, used by the env's own win/loss classification -- NOT the attacker-defender distance |

**Important disambiguation** (verified by reading `SoldierEnv.step()`'s termination logic, not assumed): the brief
directs reuse of the "canonical unsafe radius" NUMBER (3.5 m) for `R_D`, but `unsafe_intercept_radius` in the live
environment is defined over `dist_es` (enemy-soldier distance, a collateral-risk proxy), not `dist_de`
(defender-enemy distance). `R_D` is a NEW application of the SAME frozen number to the attacker-defender
relationship the brief's own equations define (section 6: `d_D = ||p_A - p_D|| - R_D`). This gives the CBF a 1 m
margin outside the actual physical interception radius (2.5 m), unchanged.

## 4. Relative-state / stopping-margin / CBF equations (frozen, as implemented in `uav_defend/attackers/cbf_defender.py`)

```
r_D = p_A - p_D                      n_D = r_D / ||r_D||            d_D = ||r_D|| - R_D
v_rel = v_A - v_D                    s_D = max(0, -n_D . v_rel)     (closing speed)
d_stop,D = s_D^2 / (2 a_max) + s_D * dt_sub          (a_max=6 m/s^2, dt_sub=0.1 s)
h_D = d_D - d_stop,D
CBF constraint:  n_D^T (v_safe - v_D) >= -alpha_D h_D     (alpha_D = 1.0 s^-1, frozen)
```

Substep prediction (brief section 9): `p_D(tau) = p_D(t) + v_D(t) tau` for `tau in {0,0.1,0.2,0.3,0.4}` within the
current 0.5 s external step, using the SINGLE velocity estimate computed once at the start of the step (never
re-differentiated mid-step; refreshed only at the next external step). **Limitation** (documented per brief
section 7): this is a receding-horizon dynamic CBF using the defender's CURRENTLY OBSERVED motion. It is NOT
claimed to guarantee avoidance against arbitrary unknown future defender acceleration.

## 5. Joint QP (frozen)

```
min_v  1/2 ||v - v_nom||^2
s.t.   all active static-obstacle CBF constraints
       defender CBF constraint, when observable
       velocity bounds (horizontal speed / climb / descent)
```

Solved as ONE QP (never sequentially). Hard -> shared-slack soft (`1e4 * delta^2` penalty) -> emergency
brake/hover fallback (`uav_defend/attackers/cbf_obstacle.py:solve_velocity_cbf_qp`). Separate residuals
(`obstacle_constraint_residual`, `defender_constraint_residual`) and an `intervention_source`/
`infeasibility_source` classification (`none`/`obstacle`/`defender`/`joint`) are tracked at every stage.

## 6. Experiments run

| Campaign | N | Namespace |
|---|---:|---|
| C0 + scripted A0 legacy regression | 1,000 | 620000-620999 |
| Static-clutter (no-defender) regression | 500 | 610000-610499 |
| Primary matched (Greedy, obstacle-only vs. full) | 1,000 x 2 | 600000-600999 |
| 200-case joint obstacle+defender stress set | 200 | 630000+ (deterministic construction) |
| Generalization diagnostic (Kalman-Greedy/PN/Lead/RMPC) | 200 x 2 each | 640000-640199 |
| Broad project regression (`pytest`) | 1,233 tests | n/a |

## 7. Results

### 7.1 Mandatory regressions
- **C0/A0 legacy**: 1,000 seeds checked against the historical parent-commit baseline, **0 mismatches** -- bit-for-bit identical (the legacy `enemy_control_mode=="legacy"` code path is untouched by this stage).
- **Static-clutter (no-defender)**: 500 episodes, **0 obstacle collisions**, 0 scenario-generation failures, 0 route errors -- frozen Prompt-3 gate fully passes (identical mechanism: `defender_cbf_enabled=False` means the defender is never even queried).

### 7.2 Primary campaign (Greedy, 1,000 matched episodes)
| | obstacle-only | full A1-CBF |
|---|---:|---:|
| Defender interception | 96.4% | 95.1% |
| Asset capture | 3.6% | 4.9% |
| Obstacle collisions | 0 | 0 |
| Mean episode duration | 5.59 s | 6.12 s |
| QP hard/soft/emergency | 100% / 0% / 0% | 100% / 0.13% / 0% |

**Primary acceptance gate: PASS** (0 obstacle collisions, 0 numerical failures, 0 route errors, 0
scenario-generation failures, 100% QP completion, soft-fallback 0.13% <= 2%, emergency-brake 0% <= 1%).

**Defender-avoidance EFFECT gate: FAIL.**
- Interception reduction: **1.3 pp** (required >= 10 pp). FAIL.
- Paired bootstrap 95% CI for the beneficial-direction interception difference: `[0.002, 0.025]` (excludes 0).
  PASS -- the small reduction IS statistically real, just far below the frozen threshold.
- Median minimum separation increase: **~0 m** (both variants: 2.500 m -- episodes overwhelmingly terminate via
  physical interception at the 2.5 m boundary in BOTH variants). FAIL (required >= 1.0 m).
- Capture tradeoff: capture went UP 3.6% -> 4.9% (+1.3 pp, well within the allowed <= 5 pp decrease -- an
  improvement, reported honestly). PASS.

**Root cause (not a bug): Greedy's defender speed `v_d=18 m/s` is 1.5x the attacker's `v_e=12 m/s`.** A
receding-horizon local CBF bounded by `a_max=6 m/s^2` can shift the interception point by a few meters/seconds but
cannot out-run a persistently faster, always-correctly-aimed pursuer in open space -- this is a genuine physical
limitation of local reactive avoidance against raw speed advantage, not an implementation defect (confirmed by the
generalization diagnostic below, where the SAME mechanism produces large effects against defenders that do not
rely on instantaneous re-aiming).

### 7.3 200-case joint obstacle+defender stress set
Deterministically constructed (brief section 19): attacker placed within 10 m of an inflated obstacle (moving
inward) AND within 25 m of a defender with positive closing speed, both active simultaneously by construction
(and validated against the FULL obstacle set via the same stopping-margin viability test Prompt 3 uses for normal
spawns, to exclude construction artifacts).

| metric | value |
|---|---:|
| Scenarios | 200 |
| Simultaneous obstacle+defender constraint rate | 100% |
| Soft-QP episodes | 50 (25%) |
| Emergency-brake episodes | 3 (1.5%) |
| **Obstacle collisions** | **3 (1.5%)** |
| Numerical failures | 0 |
| Defender interceptions | 61 (30.5%) |
| Asset captures | 133 (66.5%) |

**Stress-set acceptance: FAIL** on the zero-obstacle-collision requirement. Root-caused (seed 630069 traced
step-by-step): when the obstacle and defender constraints point in GEOMETRICALLY OPPOSING directions and are both
severely violated simultaneously, the shared soft-slack mechanism (ONE scalar `delta` relaxing both constraint
classes, as the brief's frozen design specifies) can accept an arbitrarily large `delta` to keep the QP solvable,
which in the worst case translates into real physical obstacle penetration rather than triggering emergency-brake.
This is a genuine, narrow limitation of the FROZEN shared-slack design under severe joint conflict -- it did not
appear even once in the 1,000-episode organic primary campaign (where such extreme geometric traps are rare), only
in the deliberately adversarial stress construction. **Not silently patched** (would be "tuning after evaluation");
documented here as a limitation requiring human review of the soft-fallback/emergency-escalation policy.

### 7.4 Generalization diagnostic (200 matched episodes x 2 variants, no parameter tuning between defenders)

| Defender | Interception (oo -> full) | Capture (oo -> full) | Obstacle collisions | Effect gate (strict) |
|---|---|---|---:|---|
| Kalman-Greedy | 97% -> 96% (-1 pp) | 3% -> 4% | 0 / 0 | fail (small effect) |
| **PN** | **95% -> 62% (-33 pp)** | **5% -> 38%** | 0 / 0 | fail only on separation metric |
| Lead | 97% -> 95% (-2 pp) | 3% -> 5% | 0 / 0 | fail (small effect) |
| **RMPC** | **75% -> 37% (-38 pp)** | **25% -> 63%** | 0 / 0 | fail only on separation metric |

See `fig06_interception_capture_comparison.png`. **Zero obstacle collisions across all 4 families (0/800
additional episodes)**. Per the brief's OWN generalization success criterion (section 21: "defender CBF reduces
interception for the MAJORITY of defender families... report honestly if one benefits and another doesn't"): **ALL
FOUR** additional families show a non-negative interception reduction (none made worse), with TWO (PN, RMPC)
showing dramatic, bootstrap-significant improvements and large capture-rate gains. The strict numeric EFFECT_GATE
(calibrated for the primary Greedy experiment) does not pass for any family on the separation-increase sub-check
specifically, because median separation is dominated by episodes still ending at the 2.5 m interception boundary.

**Interpretation**: the defender-CBF's evasive maneuver is highly effective against PREDICTIVE/model-based
defenders (PN extrapolates constant velocity; RMPC plans over a receding horizon) -- a sudden CBF-triggered
velocity change throws off their prediction. It is only marginally effective against REACTIVE,
always-correctly-aimed, high-raw-speed pursuers (Greedy, Kalman-Greedy, Lead) that re-aim at the attacker's CURRENT
position every step regardless of maneuvering. This is valuable, honestly-reported counter-UAS evidence (brief
section 21).

### 7.5 Broad regression
`pytest tests -q -p no:cacheprovider --ignore=tests/test_defender_adaptation_training_runner.py
--ignore=tests/test_vectorized_clutter_training.py --ignore=tests/test_journal_v2_a1_r2_training_runner.py`:
**1,233 passed, 0 failed** (1,199 baseline from Prompt 3 + 34 new defender-CBF tests), 3,699.73 s. Confirms zero
regression across the entire PPO/BC curriculum suite and every prior CBF-stage test.

## 8. Targeted tests

`tests/test_cbf_defender.py`, 34 tests covering all 24 required areas (brief section 25): detection gate (entry/
exit), no ground-truth leakage, velocity estimation (first-observation, gap handling, determinism), relative-state
geometry, safety-radius use, closing-speed calculation, acceleration-aware stopping margin, defender CBF
inequality, substep prediction, defender-only QP, joint QP, conflicting constraints, soft-fallback/emergency
classification, intervention-source classification, 2.5 m/3.5 m radii unchanged, Greedy unchanged,
obstacle-only/full equivalence when defender absent, env-level wiring (bit-for-bit when disabled, activation when
observable, joint-constraint counting).

## 9. Failure cases and limitations (brief section 28)

1. **Raw speed disadvantage**: against a defender with `v_d > v_e` that always re-aims at the attacker's current
   position (Greedy/Kalman-Greedy/Lead), local reactive evasion provides only a small, though statistically real,
   tactical benefit -- it cannot "out-run" superior raw speed.
2. **Shared soft-slack under severe joint conflict**: 1.5% of deliberately adversarial stress scenarios produced a
   genuine physical obstacle collision when both constraint classes were severely and oppositely violated
   simultaneously. A future revision should consider escalating to emergency-brake when the accepted slack implies
   physical penetration, rather than accepting any slack that merely satisfies the relaxed inequality.
3. **Receding-horizon defender modeling**: the defender-CBF uses ONLY the currently observed (estimated) defender
   velocity, extrapolated at constant velocity across substeps. It provides no guarantee against defender
   acceleration the attacker has not yet observed.
4. **Median-separation metric insensitivity**: when a large fraction of episodes terminate via physical
   interception at exactly the 2.5 m boundary (in both variants), the median-separation effect-size metric is
   insensitive to real tactical improvements that occur in the EARLIER part of matched episodes (time-to-
   interception, path deviation) -- a metric limitation, not a controller limitation.

## 10. Terminology

Per brief section 27: this remains a **guidance-and-CBF evasive attacker** (equivalently, an
**optimization-based evasive attacker**) -- never a "learned attacker."

## 11. Reproduction

```
python -m experiments.journal_cbf_attacker_defender_figures
python -m experiments.journal_cbf_attacker_defender_visualizer_export
pytest tests/test_cbf_defender.py -q
```

Campaign results are cached (content+protocol-hash verified) under
`docs/journal_cbf_attacker/defender_cbf/results/*.json`; re-running `experiments.journal_cbf_attacker_defender_foundation`
functions reuses them unless the controller/config changes.
