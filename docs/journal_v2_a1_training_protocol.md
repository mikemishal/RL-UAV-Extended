# V2 A1 training protocol (`a1_v2_training_v1`, reward `a1_v2_reward_v1`)

Status: **defined and frozen; no PPO training has been run.** Frozen benchmark: tag `journal-v2-a0-canonical-v1`
(commit `bdbee45dc08a583a0da5ca882dc6b1269207a8cd`, see [journal_v2_canonical_benchmark.md](journal_v2_canonical_benchmark.md)).
Machine-readable summary: `a1_v2_training_protocol` in [journal_v2_training_handoff.json](journal_v2_training_handoff.json).
Implementation: `experiments/journal_v2_a1.py` (environment, reward, observation, seeds),
`experiments/journal_v2_a1_training.py` (runner), `experiments/journal_v2_a1_evaluation.py` (evaluation, gates).

Any change to the reward, observation, distributions, seeds, budget or gates requires a **new** version identifier.

## 1. Role of A1: a learned tactical attacker

A1 learns *where to head*, not how to fly. The pipeline is unchanged from the frozen A0 benchmark:

```
PPO tactical command u_t ∈ [-1,1]^3  →  common V2 obstacle/safety layer  →  common constrained dynamics  →  motion
```

The action is a desired 3-D direction (normalized by the environment; it replaces the nominal pursuit direction `u_base`
before the common obstacle planner). A1 does not learn vehicle physics, swept-contact semantics, collision response, or the
frozen stateful avoidance. The attacker kinematic limits (v_e = 12 m/s, 6 m/s² acceleration, 75°/s turn rate, ±5 m/s
climb) and the finite mission endurance apply to A1 exactly as to A0.

**Design finding (safety layer).** The frozen stateful controller's *direct* rollout re-derives the mission direction toward
the asset every step and does not re-validate an off-axis tactical command against obstacles; the command is used once
avoidance activates. Measured on seeds 140000–140039 with a parked defender: A0 had 0 obstacle collisions at every level;
uniformly random commands collided in 3/40 (C1), 7/40 (C2), 8/40 (C3) episodes and purely lateral commands in 2/40, 14/40,
27/40. A1 therefore must learn obstacle awareness from its 20 sector features and the −10 collision terminal, and the
acceptance gate (§9) bounds the obstacle-collision rate. The frozen safety layer is deliberately not modified.

## 2. Audited V1 A1 → V2 A1

| Item | V1 A1 (code-authoritative) | V2 A1 (`a1_v2_training_v1`) |
|---|---|---|
| Entry point | `run_adaptive_attacker_400k_campaign.py` → `run_adaptive_attacker_training` | `python -m experiments.journal_v2_a1_training` |
| Environment | `AdaptiveAttackerTrainingEnv` | `A1V2TrainingEnv`, canonical guard asserted per episode |
| Observation | 36-D (16 base + 20 sector) | **37-D** (same 36 + `remaining_endurance_fraction`) |
| Action | Box(−1,1,3) tactical direction | unchanged (3-D) |
| Policy | SB3 `MlpPolicy` [64,64] tanh, no VecNormalize | unchanged |
| PPO | lr 3e-4, γ 0.99, λ 0.95, n_steps 2048, batch 64, 10 epochs, clip 0.2, ent 0.01, vf 0.5, max-grad-norm 0.5 | identical (imported from `experiments/ppo_hyperparams.py`) |
| Budget | 400,000 requested, 16 envs → 425,984 actual (13 × 32,768) | identical |
| Seeds | PPO 201/202/203; worker seeds = seed + i·10,000 | identical PPO seeds; scenario stream separate (§6) |
| Defenders | Lead + 9 V1 learned clutter-trained policies (RMPC excluded) | greedy, kalman_greedy, pn, lead, rmpc, 0.20 each; no passive; no learned defenders |
| Clutter | V1 randomized configs spanning C1–C3, never C0 | C0–C3 uniform (0.25), canonical V2 randomized scenarios |
| Asset motion / endurance | V1 semantics, no endurance | `legacy_gaussian_random_walk`; endurance k = 5, fail-fast |
| Terminal reward | +100 capture, −100 intercepted, −50 unsafe, −50 collision, −20 timeout, none for defender-collision, no endurance | +10 capture, **−10 every other terminal** |
| Dense reward | 5.0 × (raw metres of range decrease) − 0.05 per step | potential progress + path + time (§4) |
| Selection | `best_model.zip` by noisy 20-episode success rate | `final_model.zip` is canonical; best-validation is diagnostic only |
| Logging | SB3 evaluation | per-episode JSONL, rolling summaries, validation JSONL |

V1's raw-metre progress reward paid for every range decrease (approach/retreat cycles were exploitable), had no path or
endurance cost, and early V1 policies frequently ended in `timeout` by circling. These are the behaviors V2 removes.

## 3. Observation (`a1_v2_obs_v1`, 37-D)

Indices 0–35 keep the V1 order; index 36 is new.

| Idx | Feature | Class |
|---|---|---|
| 0–2 | attacker position / 3 scale | attacker own-state |
| 3–5 | asset position | asset state |
| 6–8 | attacker velocity | attacker own-state |
| 9 | defender-detected flag (true range ≤ 25 m) | defender state |
| 10–12 | defender position (zero unless detected) | defender state |
| 13–15 | defender velocity (zero unless detected) | defender state |
| 16–35 | 10 sectors (front, front_left, left, rear_left, rear, rear_right, right, front_right, up, down) × (static clearance, dynamic clearance), from the attacker's own perceived obstacle set | obstacle/perception state |
| 36 | `remaining_endurance_fraction` | mission/endurance state |

No future information is exposed, ground-truth obstacles are not exposed, and the defender block is gated by a
detection range. No equivalent resource information existed in the 36-D vector (the asset position and own position do not
reveal path already flown), so exactly one scalar was added:

$$e_t=\mathrm{clip}\!\left(1-\frac{L_t}{B_{path}},0,1\right),\qquad B_{path}=5\,d_0,$$

read from the environment's own endurance accounting. Only the network input layer changes (36 → 37).

## 4. Reward `a1_v2_reward_v1`

Let $d_t$ be the 3-D attacker–asset range, $d_0$ its initial value, $\Phi(s)=-d/d_0$, $\Delta L_t$ the *actual* integrated
distance flown in the step, $B_{path}=5d_0$, $T_{ref}=B_{path}/v_e$, and $\gamma=0.99$ (the PPO discount):

$$
r_t=r^{term}_t+2.0\big(\gamma\Phi(s_{t+1})-\Phi(s_t)\big)-1.0\,\frac{\Delta L_t}{B_{path}}-0.25\,\frac{\Delta t}{T_{ref}}
$$

$$
r^{term}=\begin{cases}+10 & \text{asset capture}\\ -10 & \text{defender interception, unsafe interception, obstacle collision, endurance exhaustion, timeout, any other terminal}\\ 0 & \text{otherwise}\end{cases}
$$

Deliberately absent: yaw-rate/heading/action-change penalties, any reward for defender distance, lateral motion or evasion.
Evasion is valuable only through eventual capture.

Properties (verified in `tests/test_journal_v2_a1_protocol.py`):

* **Terminal handling.** Φ is evaluated at the actual terminal (swept-contact) state and is *not* zeroed at termination. The
  discounted shaping telescopes exactly to $2(\gamma^T\Phi(s_T)-\Phi(s_0))$, which is bounded and path-independent; it is not
  claimed to be policy-invariant, and the ±10 terminal dominates it.
* **Path term.** A full endurance budget of flight costs exactly −1; with $T_{ref}=B_{path}/v_e$ a full nominal endurance
  time costs −0.25.
* **Discount leak (known, bounded).** At constant long range the shaping contributes $+2(1-\gamma)d/d_0=+0.02\,d/d_0$ per
  step, which exceeds path+time cost for $d\gtrsim 0.5\,d_0$. Because the vehicle cannot hover (measured mean speed ≥ 8 m/s
  under braking/alternating commands) and the path budget ends the episode, the *undiscounted* extra reward from circling
  before exhaustion is bounded below ≈ +1.5 (≤ 15 % of the terminal scale) and the *discounted* objective that PPO optimizes
  strictly prefers direct capture (the discounted terminal alone loses $10(1-\gamma^n)$ for $n$ wasted steps).
  Approach/retreat oscillation therefore cannot create unbounded reward.

### Reward-magnitude audit (real episodes, seeds 140000+, deterministic)

| Case | Scenario | Outcome | Terminal | Progress | Path | Time | **Total** |
|---|---|---|---|---|---|---|---|
| Efficient direct capture (path-excess 1.00) | A0, C0, greedy, 140001 | capture | +10.0 | +2.029 | −0.204 | −0.062 | **+11.763** |
| Defender-evasion capture (scripted evader, excess 2.62) | C0, rmpc, 140030 | capture | +10.0 | +2.241 | −0.574 | −0.161 | **+11.506** |
| Obstacle-detour capture (excess 3.10, 7 avoidance steps) | A0, C2, greedy, 140006 | capture | +10.0 | +2.216 | −0.691 | −0.192 | **+11.332** |
| Interception | A0, C1, rmpc, 140000 | intercepted | −10.0 | +1.975 | −0.189 | −0.049 | **−8.263** |
| Obstacle collision | lateral command, C3, greedy, 140015 | static collision | −10.0 | +0.337 | −0.034 | −0.012 | **−9.709** |
| Endurance exhaustion by circling (excess 5.36) | tangential command, C0, greedy, 140000 | exhausted | −10.0 | +1.156 | −1.000 | −0.260 | **−10.105** |

Every success (≥ +11.33) exceeds every failure (≤ −8.26); efficient > evasive > detoured success, while the efficiency
spread (≈ 0.4) is two orders of magnitude below the terminal gap. Synthetic traces cover successful-but-longer capture,
circling-then-capture, temporary retreat followed by capture, and approach/retreat oscillation.

## 5. Training distribution

* **Defender (per episode, uniform):** greedy, kalman_greedy, pn, lead, rmpc — 0.20 each. Passive is never a training
  opponent (it is a diagnostic/visualization condition). Learned V1 defenders are not used, which avoids circular dependence
  on defenders that have not yet been retrained for V2.
* **Clutter (per episode, uniform):** C0, C1, C2, C3 — 0.25 each, using the canonical V2 randomized scenario generator, one
  general policy per PPO seed.
* **Fixed contract:** `asset_motion_model = legacy_gaussian_random_walk`, `enemy_endurance_enabled = True`,
  `enemy_endurance_path_factor = 5`. The correlated (`gauss_markov_velocity`) asset model never enters training. Every episode
  configuration passes `assert_canonical_v2_training_config`; drift raises `CanonicalV2ConfigError` before any output is created.

Cost note: RMPC costs ≈ 0.057 s/step versus ≈ 0.003 s for the other defenders (≈ 19×); with 20 % RMPC this dominates run time
but remains tractable with 16 workers.

## 6. Seeds and RNG separation

| Stream | Definition |
|---|---|
| PPO/model seed | 201, 202, 203 (independent policies) |
| Training scenario stream | `SeedSequence([0xA1D2, ppo_seed, worker_index, resume_index])`; draws (level, defender, scenario seed ∈ [10⁶, 2³¹−1)). Independent of PPO's seed and of global NumPy state; `resume_index` increments on resume so scenarios are never replayed |
| In-training validation | fixed 40-episode stratified set, seeds 150000–150039 (each level × defender cell twice) |
| Post-training evaluation | seeds 140000–140099 |
| Reserved, never used | 130000–130099 (A0 validation), 95000–95999, 180000–180999, 190000–190999 |

The training scenario seed range is disjoint from every reserved range, so evaluation scenarios can never appear as
training episodes.

## 7. Training budget and checkpoints

Same budget as the final V1 A1 campaign: 400,000 requested steps, 16 environments, rounded up to whole 32,768-step rollouts
(**425,984** steps). No PPO retuning. Periodic checkpoints every 50,000 steps are immutable; `final_model.zip` is the
canonical model of a run; `best_validation.zip` (replaceable) is diagnostic only. Resume uses `resume_latest.zip` +
`resume_state.json`; a completed run (`final_model.zip` present) is never resumed or overwritten.

Output namespace (V1 artifacts are never written or overwritten; any path inside a V1 results directory raises):

```
results/journal_v2_corrected_semantics/a1_training/a1_v2_training_v1/seed_{201,202,203}/
    checkpoints/   *.zip  + *.manifest.json  (final_model.zip is canonical)
    logs/          episodes.jsonl, training_summary.jsonl
    summaries/     validation.jsonl
    manifests/     run_manifest.json, resume_state.json
```

Every saved model has an atomically written sidecar manifest: PPO seed, canonical research tag/commit and live HEAD,
per-(level, estimator) config SHA-256 fingerprints, observation/action dimensions and schema hash, reward version, protocol
version, total and requested steps, PPO hyper-parameters, defender/clutter sampling distributions, endurance factor,
asset-motion model, seed namespaces, UTC timestamp, and the SHA-256 of the model file. No checkpoint is canonical without it.

## 8. Evaluation protocol (frozen, not yet run)

Seeds 140000–140099; every A1 checkpoint (`A1-201`, `A1-202`, `A1-203`, deterministic policy) against greedy,
kalman_greedy, pn, lead, rmpc at C0–C3, **plus matched A0 on the same scenario seeds**:
(3 A1 + A0) × 5 × 4 × 100 = 8,000 episodes. Reported: capture, interception, obstacle-collision, endurance-exhaustion and
timeout rates; path efficiency (flagging path-excess > 1.3×, > 2×, > 5×); duration; minimum defender separation;
overshoot/orbit diagnostics. Primary comparison: A1 − A0.

**Seen/unseen.** A1-201 and A1-202 may be used as *seen* attackers during later defender adaptation; A1-203 is held out
from adaptation and is the primary *unseen* learned attacker. "Unseen" means unseen by the adaptation process, not untrained:
A1-203 is trained exactly like the others.

## 9. Pre-registered acceptance gates (before A1 may be frozen for defender training)

1. **C0 directness / endurance.** Per seed at C0, endurance-exhaustion rate ≤ 5 % (above is automatic rejection) and ≤ 5 % of
   episodes with path-excess > 2×.
2. **Mission advantage.** For at least one defender family, A1 − A0 capture rate ≥ +5 pp in ≥ 2 of 3 seeds. (A1 need not
   dominate every defender.)
3. **Seed generalization.** Pooled capture spread across seeds ≤ 0.20, and no seed more than 10 pp below A0.
4. **Obstacle safety.** Collision rate ≤ 2 % per seed and level (A0 reference: 0), given the §1 finding.
5. **Path efficiency.** Given capture: > 2× path-excess in ≤ 10 % of captures and > 5× in ≤ 1 %.

A0 baseline note: with a parked defender, 1/40 A0 scenarios in 140000–140039 exhausts endurance at every level including
C0, so exhaustion is reported against the A0 reference, not against zero.

## 10. Manual inspection tooling

`select_representative_episodes` deterministically chooses, per evaluation, the best mission-success, a successful evasive
(minimum defender separation ≤ 15 m, excess ≥ 1.05), an endurance-exhaustion, the worst path-excess and an obstacle-heavy
successful (C2/C3) episode, each with (attacker, level, defender, seed) as exact reload parameters.
`VISUAL_INSPECTION_MATRIX` lists the cells (greedy C0/C2/C3, lead C0/C3, rmpc C0/C3) for each of A1-201/202/203.
Loading V2 A1 checkpoints in the visualizer (new attacker ids, 37-D observation builder, registry) is deliberately the first step
of the next task; it needs no change to the research benchmark.

## 11. Running

```
python -m experiments.journal_v2_a1_training --seed 201 --dry-run    # verifies everything up to PPO construction; writes nothing
python -m experiments.journal_v2_a1_training --seed 201 --train      # the campaign (next task); likewise 202, 203
```
