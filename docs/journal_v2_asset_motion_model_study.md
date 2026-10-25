# V2 journal benchmark: asset-motion model study (correlated vs. IID random walk)

Starting research HEAD for this task: `366af116df0f8d4d3488fd0fe368c0be86724043` (diagnostics/
tests/docs only beyond the accepted production commit `873026ea15d0794bc45bf767af2df3aaa45e49fa`).

This is a MODEL STUDY, not a guidance-controller change. Three independent terminal-guidance
mechanisms (speed-only, lead-intercept, reactive curvature recovery -- see
`docs/journal_v2_terminal_speed_governor.md`, `docs/journal_v2_terminal_lead_intercept.md`,
`docs/journal_v2_terminal_reacquisition.md`) all failed on the same isolated seed (C3 130017) via
the identical mechanism: the asset (soldier) follows a true, zero-velocity-persistence Gaussian
random walk (a fresh, independent per-step displacement), which is both mathematically odd as a
"human on foot" model and, more importantly, makes any attacker-side intervention near that model's
IID noise structurally unpredictable. This study asks whether REPLACING that asset-motion model
with a temporally correlated one is scientifically preferable, config-gated and legacy-preserving,
with no attacker-guidance changes at all.

## Legacy equation (unchanged, default, bit-identical)

```python
sigma = v_s * dt
displacement = rng.normal(0, sigma, size=2)        # IID, fresh draw every step
position += displacement
position = reflect_boundary(position)
```

Verified bit-identical to the accepted production commit `873026ea` via trajectory digests across
64 (semantics x level x controller x seed) combinations: **0 of 64 digests changed**
(`git diff 873026ea -- uav_defend` for the legacy path's own equations is otherwise empty; the
new code only ADDS a sibling method and a config field, it does not touch the legacy method body).

## Correlated (Gauss-Markov velocity) equation, config-gated, non-default

```python
rho = exp(-dt / tau)                               # asset_motion_tau_s -> rho
v_{t+1} = rho * v_t + sqrt(1 - rho^2) * v_s * xi_t,  xi_t ~ N(0, I)     # first step: v_0 = v_s * xi_0
p_{t+1} = p_t + v_{t+1} * dt
```

Implemented in `uav_defend/envs/soldier_env.py:_move_soldier_gauss_markov`, selected by
`EnvConfig.asset_motion_model = "gauss_markov_velocity"` (default remains
`"legacy_gaussian_random_walk"`; no canonical builder selects anything else).

* **RNG handling**: exactly one `rng.normal(0, 1, size=(2,))` draw per step, the SAME shape as the
  legacy model's `size=(2,)` draw, confirmed by a test that the RNG bit-generator state advances
  identically regardless of which model is selected -- switching models does not shift any other
  consumer of the same stream.
* **Boundary handling**: the existing reflecting-boundary position rule is reused; the
  corresponding velocity component is ALSO reflected (sign-flipped) on any axis that reflects, so
  velocity never keeps pointing outward after a position reflection. The legacy model (position-
  only, no velocity state) and `_reflect_boundary` (used by the enemy/defender) are both untouched;
  a new `_reflect_horizontal_boundary_with_velocity` exists alongside, used only by the new model.
* **Obstacle-blocked motion**: the candidate segment is rejected-in-place exactly as the legacy
  model's is (never resampled); the velocity STATE update is not rejected (mirrors the legacy
  model's own rule that the proposed draw is computed "regardless of obstacles").

## rho / tau table (dt = 0.5 s)

| tau (s) | rho = exp(-dt/tau) | empirical lag-1 autocorrelation (measured) |
|---:|---:|---:|
| 0.5 | 0.3679 | 0.383 (vx) / 0.385 (avg) |
| 1.0 | 0.6065 | 0.613 |
| 2.0 | 0.7788 | 0.770 |
| 4.0 | 0.8825 | 0.864 |
| legacy (IID reference) | 0 (by construction) | 0.025 (measurement noise floor) |

The close match between the theoretical rho and the measured lag-1 autocorrelation confirms the
recurrence is implemented correctly.

## Target-motion-only statistics (20 seeds x 200 s, obstacles disabled, no attacker/defender)

| | legacy | tau=0.5 | tau=1.0 | tau=2.0 | tau=4.0 |
|---|---:|---:|---:|---:|---:|
| mean speed (m/s) | 1.868 | 1.887 | 1.898 | 1.907 | 1.908 |
| median speed | 1.741 | 1.756 | 1.776 | 1.784 | 1.794 |
| std speed | 0.989 | 0.996 | 0.996 | 0.998 | 0.994 |
| p95 speed | 3.69 | 3.70 | 3.71 | 3.75 | 3.74 |
| mean |accel| (m/s^2) | 5.22 | 4.17 | 3.31 | 2.52 | 1.87 |
| p95 |accel| | 10.17 | 8.18 | 6.45 | 4.92 | 3.66 |
| mean heading change (deg/step) | 88.1 | 66.7 | 51.5 | 38.6 | 28.4 |
| p95 heading change | 169.6 | 161.5 | 146.8 | 127.2 | 100.6 |
| lag-1 autocorr | 0.025 | 0.385 | 0.613 | 0.770 | 0.864 |
| lag-2 autocorr | -0.002 | 0.137 | 0.362 | 0.576 | 0.733 |
| displacement RMS/step (m) | 1.057 | 1.067 | 1.072 | 1.077 | 1.076 |
| path length (200 s, 1 seed) | 373.6 | 377.4 | 379.7 | 381.5 | 381.5 |
| net displacement (200 s, 1 seed) | 20.4 | 28.7 | 34.5 | 35.4 | 37.7 |
| boundary reflections (20 seeds x 200 s) | 0 | 0 | 0 | 1 | 0 |

**Mean/median/std/p95 speed and displacement RMS are essentially unchanged across every setting**
(confirming the construction preserves the marginal motion SCALE as designed), while acceleration,
heading-change magnitude, and both autocorrelation lags change monotonically and smoothly with
`tau` -- this is unambiguously a change in TEMPORAL PERSISTENCE, not merely "slower" or "less
noisy" motion. Plots: `results/asset_motion_model_study/fig_trajectories.png` (representative
200 s paths), `fig_autocorrelation.png` (velocity autocorrelation curves), and
`fig_heading_speed_distributions.png` (heading-change and speed distributions, 10 seeds pooled).

## Physical interpretation (qualitative only -- no human-motion calibration data exists in this
repository, so none of these labels should be read as empirically validated against real foot
movement)

* **legacy (IID)**: near-Brownian with an unphysically abrupt direction change EVERY step (mean
  88 deg/step heading change) -- the "random walk" label is accurate but the implied instantaneous
  re-randomization of direction every 0.5 s has no obvious physical analogue.
* **tau = 0.5 s**: still rapidly changing direction (67 deg/step mean), only mildly smoothed.
* **tau = 1.0-2.0 s**: moderately persistent walking-like motion (39-52 deg/step mean heading
  change over a 0.5 s step), plausibly closer to an ambling/meandering gait change-of-mind
  frequency, but this is a plausibility judgement, not a calibrated one.
* **tau = 4.0 s**: much more persistent, closer to "walking roughly in one direction with
  occasional correction" (28 deg/step mean) -- still far from ballistic, but the most
  directionally-committed of the tested settings.

## Passive-A0 population sweep (C0-C3, seeds 130000-130099, 2000 episodes total)

| | legacy | tau=0.5 | tau=1.0 | tau=2.0 | tau=4.0 |
|---|---:|---:|---:|---:|---:|
| C0 capture / collisions | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 |
| C1 capture / collisions | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 | 1.00 / 0 |
| C2 capture / collisions | 1.00 / 0 | 0.98 / **2** | 0.99 / **1** | 0.99 / **1** | 0.99 / **1** |
| C3 capture / collisions | 1.00 / 0 | 0.98 / **2** | 0.98 / **2** | 0.97 / **3** | 0.97 / **3** |
| C3 excess mean / max | 1.235 / 27.06 | 1.084 / 7.41 | 1.019 / 3.26 | 1.008 / 2.39 | 1.040 / 5.07 |
| C3 >1.3x / >2x / >5x | 1 / 1 / 1 | 3 / 3 / 1 | 3 / 2 / 0 | 3 / 2 / 0 | 4 / 1 / 1 |
| C3 terminal-flyby detections | 1 (130017) | 0 | 0 | 0 | 0 |
| C0 terminal-flyby detections | 0 | 2 (130028, 130054) | 1 (130028) | 1 (130028) | 1 (130017) |
| C3 avoidance emergency (all-candidates-collide) selections | 4 | 6 | 6 | 10 | 9 |
| mean episode length (C3, steps) | 13.9 | 11.6 | 11.1 | 11.0 | 11.4 |

**Every single correlated-motion setting introduces NEW collisions that do not exist under the
legacy model (0 across all 2000 legacy-model episodes)** -- 2-4 new
`enemy_static_obstacle_collision` episodes per setting, spread across C2 and C3 (and, at
tau=4.0, the collision lands back on 130017 itself). The avoidance controller's
"all-candidates-collide" emergency path is invoked substantially more often (4 -> 6-10 times in
C3) and, unlike under legacy motion where it always eventually resolves safely, sometimes fails to
under the altered trajectory shapes the correlated asset motion induces in the attacker's own
pursuit path.

## C3 seed 130017 across every setting

| | legacy | tau=0.5 | tau=1.0 | tau=2.0 | tau=4.0 |
|---|---:|---:|---:|---:|---:|
| outcome | `soldier_caught` | `soldier_caught` | `soldier_caught` | `soldier_caught` | **`enemy_static_obstacle_collision`** |
| steps | 346 | 12 | 12 | 13 | 17 |
| path length (m) | 1828.5 | 56.9 | 57.5 | 59.4 | 86.4 |
| path-excess | 27.06x | 1.07x | 1.10x | 1.18x | 1.74x |
| fly-by count | 22 | 0 | 0 | 0 | 0 |
| cumulative bearing revolutions | 14.46 | 0.65 | 0.77 | 0.63 | 0.00 |
| avoidance activations | 2 | 1 | 1 | 1 | 2 |
| flight time (s) | 172.7 | 5.8 | 5.8 | 6.0 | 8.1 |

Three of the four correlated settings (tau 0.5/1.0/2.0) genuinely eliminate 130017's orbit (it now
captures in 12-13 steps, essentially the population mean) -- but this is exactly the seed-130017-
only improvement the study is explicitly warned not to select a model on, and `tau=4.0`
**re-introduces a failure on this very seed, now a collision instead of an orbit**. See
`results/asset_motion_model_study/fig_130017_overlay.png` for the overlaid top-down trajectories.

## Replacement-pathology search (Phase 11): the pathology MOVES, it does not disappear

Legacy's only C3 pathology is 130017 itself (0 collisions, 0 orbit events, exactly 1 >5x episode).
Every correlated setting trades that single, SAFE (no collision) orbit pathology for 1-3 NEW,
UNSAFE (collision) episodes on DIFFERENT seeds each time:

* tau=0.5: new collisions at 130028, 130054 (C2/C3); new worst path-excess at 130054 (7.41x).
* tau=1.0: new collisions at 130028, 130064; worst excess 130014 (3.26x).
* tau=2.0: new collisions at 130014, 130028, 130096; worst excess 130096 (2.39x).
* tau=4.0: new collisions at 130014, **130017**, 130070; worst excess 130096 (5.07x).

This directly answers Phase 11's key question: correlation does **not** genuinely improve the
population distribution -- it relocates a non-fatal pathology (orbit) into several fatal ones
(collision) on different seeds, at every tested persistence level. The mechanism is the same one
found in all three previously rejected guidance-controller experiments: the (deliberately
unmodified, per every one of these four tasks' hard restrictions) stateful obstacle-avoidance
controller's "all-candidates-collide" emergency fallback has a rare, pre-existing gap that the
LEGACY model's induced population of attacker trajectories happens never to trigger into an actual
collision, but that ANY sufficiently different trajectory shape -- whether from a different
attacker guidance law (three prior experiments) or, as shown here, from a different TARGET motion
model -- can expose.

## Benchmark-difficulty changes (Phase 12)

Capture rate drops from a clean 1.00 (legacy, all four levels) to 0.97-0.99 in C2/C3 under every
correlated setting (purely from the new collisions above; C0/C1 are unaffected since clutter is
absent/light enough that the avoidance controller's emergency path is rarely invoked at all).
Mean C3 episode length actually DECREASES slightly (13.9 -> 11.0-11.6 steps) because the single
very long legacy episode (130017) is removed from the mean; excluding 130017, legacy and
correlated mean lengths are comparable. This is flagged as a non-trivial difficulty change: the
benchmark would go from "perfect capture, one known inefficient outlier" to "97-99% capture with a
small but nonzero collision rate" -- a materially different (and worse, from a safety-benchmark
perspective) failure profile, not merely a cosmetic change.

## Finite-endurance interpretation of legacy 130017 (Phase 16, report only, no cutoff implemented)

* Episode duration: 172.68 s (346 steps at dt=0.5 s).
* Attacker path length: 1828.5 m.
* Straight-line (direct) distance for this engagement: path_length / path_excess_ratio ~= 67.6 m.
* A "normal" direct engagement at this distance would take ~67.6 / v_e ~= 5.6 s and ~68 m.
* Extra time attributable to the orbit: ~167.1 s (97% of the episode).
* Extra path attributable to the orbit: ~1760.9 m (96% of the path).

This episode's path-excess (27.06x) is already far beyond any of the ">1.3x / >2x / >5x"
thresholds used throughout every related investigation in this journal. A plausible finite
mission-endurance or path-budget limit anywhere in a wide, defensible range (e.g. 3-5x the direct
distance, ~200-340 m / ~17-28 s) would have reclassified this specific episode as an ordinary
`attacker mission failure / endurance expiration` well before step 60 of 346 -- a natural,
non-arbitrary resolution that requires no further controller modification and does not touch any
of the three previously-rejected guidance mechanisms or this task's asset-motion model.

## Selection criteria assessment (Phase 14)

1. Eliminates the zero-persistence velocity process? Yes, by construction, for any tau > 0.
2. Preserves comparable stochastic motion scale? Yes (speed/displacement-RMS statistics above are
   within ~2% of legacy across every tau).
3. Introduces no new collisions attributable to the model/controller interaction? **No** -- every
   tested tau introduces 2-4 new collisions across the 400-seed C0-C3 population.
4. No new severe >5x path tail? Borderline: the >5x COUNT stays at 0-1 per setting (not worse than
   legacy's 1), but the specific SEEDS that produce it change every time, and tau=4.0 regresses
   130017 itself from a safe 27x orbit into a collision.
5. Materially reduces terminal fly-by/orbit pathology? Yes for tau 0.5/1.0/2.0 (130017 fixed); no
   at tau=4.0 (a new, different fly-by/collision reappears on 130017 itself, plus a new one at
   C0/C2 130017/130064 depending on level).
6. Does not radically alter overall benchmark difficulty? **No** -- capture rate drops and a new
   nonzero collision rate appears at every tested persistence level.
7. Defensible physical interpretation? Only qualitatively (criterion met, with the explicit caveat
   above that no calibration data exists in this repository).

Criterion 3 and 6 are both violated, by a population-wide, population-scale, non-seed-specific
margin (criterion 3 fails in 4,4 of 4 tested settings), which the task's selection framework treats
as disqualifying regardless of criteria 1/2/5 being satisfied.

## Tests and regression

* 12 new unit tests (`tests/test_asset_motion_model.py`): legacy default/exactness/determinism,
  Gauss-Markov recurrence exactness (replayed against the env's own RNG stream), stationary-scale
  preservation, rho/tau conversion, boundary velocity reflection, identical per-step RNG draw count
  between models (no information leakage to other consumers of the stream), config
  serialization/round-trip, and config validation. All pass.
* Legacy-equivalence digests: 0/64 changed vs the accepted `873026ea` baseline.
* Targeted suite (soldier/obstacle motion, prediction fidelity, corrected attacker, journal V2
  validation, swept contact, stateful avoidance, enemy obstacle avoidance, new motion tests): 148
  passed.
* Non-training regression (both training modules excluded, never executed): **578 passed**
  (566 previously accepted + 12 new).
* Checkpoint hashes: 30/30 unchanged. Greedy/Lead invariance: 76 pairs, 0.0 m max delta, no
  evasion (unaffected, since the default/canonical configuration is untouched).

## Commit policy

Per Phase 18, this is a config-gated, legacy-preserving model STUDY: the optional implementation
is committed as `feat:` even though the canonical default is not changed and the correlated model
is NOT recommended for adoption (see verdict). The visualizer pin and validation tag are untouched.
