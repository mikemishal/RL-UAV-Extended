# Manuscript notes — curriculum Stage 1: learning to capture a stationary asset (negative result under the pre-registered budget)

## Motivation for a curriculum
Seven single-stage A1 interfaces (continuous steering, bounded residual, filtered residual, obstacle-primitive reuse, finite tactical pulses, evasion blend, route-level intents) were each rejected under simultaneous mission, evasion and clutter complexity, and the first learned attempt (continuous direction, reward v1) never learned the mission at all (capture 0/6,000). We therefore decompose A1 learning: first establish *mission competence* (reach and capture a stationary asset in open space), then add an active defender, target motion and obstacles. Stage 1 deliberately changes only the environment (configuration/wrapper), never the objective (reward v2) or the learner (PPO settings).

## Pre-registered hypothesis
> PPO can learn reliable asset capture under canonical attacker dynamics when the asset is stationary and defender/clutter complexity is removed.

Accepted for a seed if two consecutive rollout validations (alternating held-out sets A/B of 100 scenarios) all satisfy: capture ≥ 98%, 0 collisions, exhaustion ≤ 1%, timeout ≤ 1%, median path-excess ≤ 1.20×, >2× ≤ 2%, >5× = 0, no orbit and ≤ 2% pathological episodes (orbit ≥ 1 revolution, ≥ 2 fly-bys, or ≥ 10 steps closing slower than 1 m/s), no non-finite values; Stage 1 passes if ≥ 2/3 seeds master, each mastered seed passes the final 200-episode gate and no pathology is common to the seeds. Budget: 8 rollouts × 32,768 steps per seed, early stop at mastery, no PPO tuning.

## Setup
Asset `v_s = 0` (canonical origin); parked defender at a ground corner; C0; canonical 37-D observation; reward v2; action = continuous desired direction normalized to a unit vector (near-zero action holds the current heading; never A0 guidance); canonical reset geometry plus a random initial heading offset of up to 90°. Three PPO seeds (201–203) from fresh initialization, 16 `SubprocVecEnv` workers, local scratch with atomic per-rollout checkpoints and a real-SB3-verified bit-exact resume protocol (interrupted ≡ uninterrupted).

## Quantitative findings
- **No seed mastered** (262,144 steps each). Validation capture at rollouts 1–8: seed 201 .00/.03/.02/.00/.01/.05/.01/.30; seed 202 ≈ 0 throughout; seed 203 ≤ .04.
- Final 200-episode held-out evaluation of the rollout-8 policies (not accepted): capture 33.5% / 0% / 2.5%; timeouts 66.5% / 97% / 97.5%; median path-excess 1.17× / 3.99× / 1.83×; >5× 0% / 23.5% / 7%; mean return −149 / −227 / −217. Sample efficiency (timesteps-to-mastery): not reached.
- Scripted A0 on the identical scenarios: capture 100%, median path-excess 1.02×, 5.6 s per episode — the task is easy for a mission-aware controller and the gate is attainable.
- Diagnostics: cosine between the deterministic action direction and the line of sight 0.07 / −0.005 / 0.014; learned action std ≈ 1.02; stochastic training episodes end by endurance exhaustion in ~73 steps in 96–98% of cases, capture in 0.5–4%, and hit the parked defender in 1.4–3.2%.

## Interpretation
The hypothesis is **not supported under this budget and configuration**. The data point to an exploration/credit-assignment problem rather than a representation problem: the action is a unit direction sampled from a Gaussian with σ ≈ 1 over a [−1, 1] box, so sampled directions are close to uniform on the sphere; with only a few captures per ~440 training episodes per rollout (0.5–4%) and only eight policy updates, the progress shaping of reward v2 does not yet bias the mean direction toward the asset (cosine ≈ 0). Seed 201 starts to improve at the end (training return −18.0 → −17.0, validation capture 30%), which argues against impossibility and for an under-budgeted or poorly-conditioned protocol. We do **not** claim PPO cannot learn this task.

## Limitations
Single environment variant; three seeds; eight rollouts only (pre-registered cap); deterministic validation policy differs from the stochastic training policy; the parked defender is not completely inert in stochastic training (1.4–3.2% interceptions) and is visible in ~7.5% of initial states; the initial-heading randomization (up to 90°) is an addition to the canonical reset and makes the task harder than canonical geometry; diagnostics are post-hoc and were not used for tuning.

## Decision and next step
**Stage 1 fails; Stage 2 (stationary active defender) is not started and no checkpoint is promoted.** Any remedy (a larger rollout budget, lower initial exploration noise, a bounded/atanh-squashed or heading-relative action parameterization) changes the pre-registered protocol and must be registered as a new Stage-1 version before it is run.

## Architecture-history sentence (for the ablation chronology)
Single-stage A1 architectures failed under simultaneous mission/evasion/clutter complexity; curriculum learning therefore decomposes the problem beginning with stationary-target acquisition — and the first curriculum stage, run with the unchanged reward and PPO settings, did not master even that within 262,144 steps (0/3 seeds).
