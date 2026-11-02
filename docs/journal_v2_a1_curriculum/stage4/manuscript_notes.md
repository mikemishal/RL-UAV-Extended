# Manuscript notes — curriculum Stage 4 (moving canonical asset)

## Claim supported (negative / partial result — state it plainly)
Under the frozen PPO configuration and a 16-rollout (524,288-step) budget, curriculum initialization from the Stage3-v2 checkpoints **did not** let the attacker master the canonical moving-asset task: 0/3 seeds met the pre-registered gate. Held-out capture is 52.5 / 60.5 / 59.5 % (scripted A0: 0 %; own Stage3-v2 parent on the same scenarios: 50.5 / 46.0 / 54.5 %). Previously learned skills were retained (pursuit 0.99–1.00; static-defender evasion 0.94–0.97; moving-defender evasion 0.87–0.93), so the failure is not catastrophic forgetting. The training benefit over the parent is small and significant for one seed only (+14.5 pp [7.5, 21.5] for seed 202; +2 [−6, 10] and +5 [−2, 12] for 201 and 203).

## Hypothesis (pre-registered)
> After learning pursuit and moving-defender evasion in simplified conditions, curriculum initialization enables PPO to adapt to stochastic target motion without catastrophic loss of previously learned pursuit/evasion behavior.

Supported in the "no catastrophic loss" part, **not** in the "adapt to stochastic target motion" (mastery) part. Stage 4 isolates target-motion complexity: clutter, obstacles, scaffold removal and defender training are excluded.

## Why only the asset motion was added
Stage 3 used a stationary asset; the canonical V2 asset random walk is the single remaining canonical difference of the target that is not an obstacle/clutter effect. Isolating it lets the result be attributed to target motion alone (C0, same scaffold defender, same obs/action/reward/PPO).

## Canonical asset-motion specification (from live code)
`asset_motion_model = legacy_gaussian_random_walk`, `v_s = 1.5 m/s`, `dt = 0.5 s` → i.i.d. Gaussian horizontal displacement with σ = 0.75 m per axis per step, z = 0, reflecting boundaries at ±50 m, start at the origin, no velocity persistence; draws from the dedicated `_rng_soldier` stream. No correlated motion, retuned speed, reduced noise or curriculum-specific easier walk was used. Over the short episodes (captures ≈ 14 steps) the asset drifts a median 2.5–2.7 m; long exhaustion episodes ≈ 8–10 m.

## Observation sufficiency (document, do not hide)
`a1_v2_obs_v1` (37-D) carries the current normalised asset position; the walk is memoryless, so a velocity or prediction feature would add no information, and no future position/velocity is exposed. The observation was therefore not augmented.

## Common-random-number protocol
Placement (initial-state nominal-A0 path, λ ~ U(0.35, 0.65), r ~ U(0, 1.5 m)) uses only initial attacker and asset states; the asset stream is independent of attacker, defender and sensor streams. On every evaluation seed A0, the Stage3 parent and the Stage4 policy therefore see identical initial states, defender placement and realised asset trajectory (verified: 0.0 m maximum difference over all completed steps; only the terminal swept-contact event state of an episode differs, ≤ 1.7 m, and is excluded). Paired bootstrap CIs rely on this matching.

## Pre-training difficulty validation (1,000 scenarios)
A0: capture 0 %, interception 100 %. Stage3 parents: capture 53.8 / 54.6 / 58.3 %, interception 19.5–25.7 %, endurance exhaustion 19.7–22.2 %. **The stationary-asset parents lose ≈ 25–30 pp capture simply from asset jitter, before any Stage-4 training** — the headline curriculum finding.

## Protocol
Gate (frozen before training): two consecutive validation passes with capture ≥ 0.70, ≥ +20 pp over matched A0, interception ≤ 0.30, timeout ≤ 0.03, exhaustion ≤ 0.03, median path-excess ≤ 2.0, >3× ≤ 0.05, >5× = 0, no persistent orbit, anchors ≥ 0.98 / 0.80 / 0.75, no numerical failure. Retention stop = two consecutive anchor failures. Validation blocks 165000–165199 (alternating), anchors 166000–166299, diagnostic 167000–167099, difficulty check 168000–168999; all audited disjoint from earlier blocks. Initialization: Stage3-v2 actor + critic + log_std, fresh optimizer, counter 0; parent SHA in every manifest; protocol SHA `5a1a9254…cfaa26`. Restart safety: atomic writes, per-rollout manifests with SHA-256, per-seed locks, scratch outside OneDrive; real-SB3 interrupted-vs-uninterrupted equivalence (including a crash between model write and manifest) validated before training.

## Results to quote
* Held-out (200 matched scenarios): capture 52.5 / 60.5 / 59.5 %; interception 24.0 / 22.5 / 18.0 %; exhaustion 23.5 / 17.0 / 22.5 %; timeout 0; median path-excess 1.11 / 1.10 / 1.13, but >3× in 30 / 27 / 31 % and >5× in 11.5 / 8.5 / 12 % of episodes; orbit episodes 2 / 0 / 0.
* Paired Stage4 − A0: +52.5 [45.5, 59.5] / +60.5 [53.5, 67.5] / +59.5 [52.5, 66.5] pp. Paired Stage4 − Stage3 parent: +2.0 [−6.0, 10.0] / +14.5 [7.5, 21.5] / +5.0 [−2.0, 12.0] pp.
* Training curves are flat (validation capture 0.40–0.67 with no trend; training capture ≈ 0.50 → 0.55).

## Failure mechanism
Repeated fly-bys: the attacker passes the asset at ≈ 3.4 m (capture radius 2 m), re-approaches (median 3 overshoots, ≈ 60 steps) and exhausts its endurance budget; first-pass capture is 74–84 % of successes. Pursuit speed (closing ≈ 9.1–9.3 m/s) and evasion are intact.

## Moving-target and evasion analyses (descriptive; never rewards)
Whether the policy aims at the *current* asset cannot be decided from these episodes: current and initial asset directions differ by only ~3° (median drift 2.7 m), so the diagnostic (30–46 % "closer to current") is uninformative; no claim of explicit tracking is made. Evasion pattern retained: no deviation before the 25 m gate, 0.5 s detection-to-deviation, closest approach ≈ 5.7–5.9 m (A0 2.5 m), post-encounter cosine 0.88–0.94, reacquisition 0.5–1.5 s; the evasive direction is uncorrelated with the asset's drift (cosine ≈ 0).

## Generalization (diagnostic only)
Against Kalman-Greedy / PN / Lead (moving asset, same scaffold) capture is 55–72 % for Stage4 vs 45–63 % for the parents (+6…+13 pp); other controllers are not harder than Greedy; A0 is 0 % everywhere.

## Limitations to state
1. Short episodes → small asset displacement; this stage mostly probes end-game precision under jitter, not long-range tracking.
2. Hold-until-detection scaffold retained (defender ≤ 8 m/s at encounter).
3. One frozen PPO configuration and budget; flat curves are not proof of unlearnability under other settings (e.g. longer training, other hyper-parameters — each would be a new registered version).
4. Parents are provisional curriculum checkpoints; no manual review of Stage3-v2 or Stage 4 recorded (visualizer package only prepared for a pass / verdict B).
5. Three seeds, 200 paired scenarios; the seed-202 gain is within what a different budget or seed could change.

## Figures
`figures/fig1…fig6` (PNG + PDF): capture vs Stage-4 timesteps, interception vs timesteps, three retention anchors, path-excess, Stage3 parent vs Stage4 capture, representative A0 / Stage3 / Stage4 moving-target trajectories.
