# Corrected V2 A1 campaign (`a1_v2_training_v2`) — status: STOPPED AT THE θ_max GATE, NO TRAINING STARTED

Failed campaign `a1_v2_training_v1` / `a1_v2_reward_v1` stays preserved as a **NON-CANONICAL — FAILED REWARD/ACTION FORMULATION** ablation
(see [../journal_v2_a1_campaign_v1/README.md](../journal_v2_a1_campaign_v1/README.md)). Its checkpoints are never resumed into this campaign.
Canonical A0 and the V2 environment are unchanged. No A1 PPO training was run; the protocol was **not** frozen and **no tag** was created.

## 1. Reward `a1_v2_reward_v2` (implemented in `experiments/journal_v2_a1_r2.py`)

$$r_t=r^{term}+2(\gamma\Phi_{t+1}-\Phi_t)-\frac{\Delta L_t}{B_{path}}-0.1,\qquad \Phi=\mathrm{clip}\!\left(1-\frac{d}{d_0},0,1\right),\ \gamma=0.99,\ B_{path}=5d_0$$

Terminal: +10 capture, −10 for every other outcome. At constant Φ the shaping is $2(\gamma-1)\Phi\le 0$. The step cost is the
discount-compensating $-(1-\gamma)R_F=-0.1$ with $R_F=10$, because $\sum_t\gamma^t(-0.1)=-10$: surviving forever without capture is worth no more than failing at once.

### Offline proof on the failed campaign's saved trajectories (`data/reward_audit.json`) — ALL CHECKS PASS

| Case | Steps | Terminal | Progress | Path | Step | Undiscounted | PPO-discounted |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 efficient capture | 12 | +10.00 | 1.79 | −0.20 | −1.20 | **+10.38** | **+9.33** |
| A0 inefficient capture (3.1× path) | 33 | +10.00 | 1.56 | −0.69 | −3.30 | **+7.56** | **+5.21** |
| A1-201 C0 loiter / 2000-step timeout | 2000 | −10.00 | −4.17 | −0.28 | −200.00 | −214.45 | −10.25 |
| A1-202 C0 loiter / timeout | 2000 | −10.00 | −3.60 | −0.58 | −200.00 | −214.17 | −10.22 |
| A1-203 C0 loiter / timeout | 2000 | −10.00 | −4.17 | −0.29 | −200.00 | −214.46 | −10.25 |
| A1-201 C0 endurance exhaustion | 313 | −10.00 | 0.10 | −1.00 | −31.30 | −42.20 | −10.22 |
| A0 early interception | 10 | −10.00 | 1.77 | −0.19 | −1.00 | −9.41 | −8.57 |
| A1 late interception | 1857 | −10.00 | −6.03 | −0.98 | −185.70 | −202.71 | −10.17 |
| A1-203 obstacle collision | 536 | −10.00 | 0.09 | −1.00 | −53.60 | −64.50 | −10.25 |
| synthetic approach/retreat oscillation | 49 | −10.00 | 0.15 | −0.98 | −4.90 | −15.73 | −10.72 |

Every capture exceeds every failure (undiscounted and discounted); efficient > inefficient capture; the 2000-step timeout is worse than an early
interception (−10.25 vs −8.57 discounted); constant-range shaping ≤ 0; oscillation is negative and bounded.

## 2. Action contract `a1_v2_action_v2`

$a_\perp=a-(a^\top u_b)u_b$ (clamped to norm ≤ 1), $u_{A1}=\mathrm{normalize}(u_b+\tan\theta_{max}\,a_\perp)$, so $\angle(u_{A1},u_b)\le\theta_{max}$.
`u_b` is read inside the environment's override call from the canonical base guidance (direct pursuit in V2). An exactly zero residual returns the
zero vector, which the environment treats as "no override", so a zero action reproduces canonical A0 **bit-for-bit** (verified on 4 scenarios in the tests).
The parallel component is discarded, so A1 can neither command speed nor reverse the mission direction.

## 3. θ_max selection study — NO CANDIDATE PASSES (training must not start)

Pre-frozen rule (see `experiments/journal_v2_a1_r2_theta_study.py`): the largest θ ∈ {15, 30, 45, 60}° with collision ≤ 2% at every clutter level (pooled over the
six stress commands), positive C0 progress, no reverse command, no ceiling-hover exploit, numerical stability. Seeds 145000–145099, C0–C3, six stress
commands (random, persistent left/right/up/down, alternating), parked passive defender: 4 θ × 6 × 4 × 100 = 9,600 episodes.

| θ_max | pooled collision C0 / C1 / C2 / C3 | random residual alone C1 / C2 / C3 | worst single command (any level) |
|---:|---|---|---:|
| 15° | 0 / 1.8% / 10.0% / 21.0% | 1% / 6% / 8% | 45% |
| 30° | 0 / 6.8% / 28.2% / 51.7% | 5% / 26% / 54% | 79% |
| 45° | 0 / 10.0% / 32.3% / 51.2% | 10% / 36% / 64% | 79% |
| 60° | 0 / 8.7% / 28.7% / 49.3% | 10% / 32% / 70% | 80% |

All other checks pass for every θ: C0 progress is positive for every command (0.46–0.97), the command never deviates by more than θ, no episode is a ceiling-hover, and
the simulation is numerically stable. **Only the collision criterion fails**, at every candidate and at clutter levels C2/C3 (and C1 for θ ≥ 30°).

**Supplementary, post-hoc, ineligible for selection** (`data/theta_study_supplementary_5_10_summary.json`): θ = 10° pooled 0 / 0.8% / 3.8% / 12.0%; θ = 5° pooled 0 / 0.2% / 1.7% / 3.0%
(random residual alone 0 / 0 / 1% / 2%). Even 5° exceeds 2% at C3 under the persistent stress commands.

### Why collisions occur
Under persistent off-axis commands the vehicle follows a constant-offset curve around the asset (C0 capture falls from 93% at 15° random to 0% at 30° left/right/up, with exhaustion near 100%), so it spends hundreds of
steps in the obstacle field. The frozen stateful avoidance layer's direct rollout re-derives the mission direction toward the asset and does not re-check an off-axis tactical command (the finding recorded at the
V1 protocol freeze). Constraining the angle bounds the *authority* but does not make the command obstacle-safe.

## 4. Power-loss-safe runner (implemented and tested; not used for A1 training)

`experiments/journal_v2_a1_r2_training.py`: exactly 13 rollouts × 32,768 = 425,984 steps per seed, one `learn(32768)` call per rollout; atomic recovery checkpoint after every
completed update (model ZIP, RNG state, episode log, manifest renamed last with SHA-256); active state in a local non-synced scratch directory (`C:\Users\Public\journal_v2_a1_training_v2`
selected automatically: writable, atomic rename tested, not under OneDrive; the per-user AppData tree is avoided because Store Python virtualises it), mirrored atomically into the results directory; automatic resume
from the highest verified rollout, discarding temp/partial files; scratch can be rebuilt from the verified mirror; final manifest with SHA-256; the failed-v1 namespace is write-protected.
`tests/test_journal_v2_a1_r2_training_runner.py` (18 tests, tiny PPO on a dummy environment) covers power loss at every stage, corrupt checkpoints, lost scratch, exact accounting, and that **resumed training equals uninterrupted training exactly**.
`protocol_description()` loads `selected_theta.json`, which does not exist, so the A1 training entry points fail loudly until a θ is frozen.

## 5. Decision required before any training

The frozen θ rule cannot be satisfied by constraining the angle alone. Options (none applied): (a) keep the contract but pre-register a different collision criterion (for example the random-residual stress only, which θ = 5° satisfies at 2%, leaving A1 almost no tactical authority);
(b) make the contract obstacle-aware **inside A1's tactical stage** (for example reject a residual whose constrained-dynamics rollout, via the frozen `predict_constrained_trajectory`, predicts a collision, and fall back to `u_base`), leaving the common obstacle/dynamics stack untouched, as `a1_v2_action_v3`, then repeat this study; (c) both.

---

# Addendum — Option B: obstacle-aware action `a1_v2_action_v3` — θ_max gate FAILS, still no training

Status: **no protocol freeze, no `journal-v2-a1-training-protocol-v2` tag, no A1 PPO training.** `selected_theta.json` does not exist.
Reward `a1_v2_reward_v2` is preserved exactly (its offline proof and 53 tests still pass). Canonical A0, the V2 environment and the common obstacle layer are unchanged.

## Action-v3 semantics (implemented in `experiments/journal_v2_a1_r2.py`)

Bounded residual of action-v2 followed by a **binary A1 tactical safety filter**: the proposed `u_c` is accepted iff the frozen rollout
`predict_constrained_trajectory` (current float32 state, `SoldierEnv._enemy_dynamics_limits()`, `enemy_obstacle_prediction_horizon_steps`, truncated at the capture sphere)
over the attacker's PERCEIVED obstacles (static + threatening dynamic swept boxes through `SoldierEnv._combined_avoidance_layout`) has no collision and a minimum predicted clearance of at least
`enemy_obstacle_clearance` (2 m); otherwise the executed command is exactly `u_base` (zero vector = no override). The frozen stateful common obstacle layer and dynamics run afterwards unchanged. The filter
is pure (verified: no environment/planner mutation), fails closed without a perceived set, never runs for A0/V1/defenders, and diagnostics (proposed angle, safe flag, intervention, clearance, obstacle count) are logged per step.
20 new unit tests pass, including the regression (θ = 30°, persistent left, C3, seed 145003): action-v2 collided at step 19; action-v3 does not.

## Pre-registered stress study (9,600 filtered + 400 zero-reference episodes; seeds 145000–145099; passive defender)

Gate language was frozen in `experiments/journal_v2_a1_r3_theta_study.py` before the run. **Zero-residual reference: 0 collisions at C0–C3** (C3 has one endurance exhaustion).

Filtered collisions per 100 matched scenarios (reference = 0, so every one is a gained collision):

| θ_max | command | C1 | C2 | C3 |
|---:|---|---:|---:|---:|
| 15° | alternating / down / left / random / right / up | 2 / 0 / 1 / 1 / 0 / 1 | 3 / 0 / 4 / 2 / 3 / 15 | 6 / 0 / 7 / 6 / 6 / 26 |
| 30° | same order | 11 / 0 / 9 / 5 / 2 / 1 | 16 / 0 / 25 / 15 / 24 / 12 | 17 / 1 / 29 / 14 / 36 / 43 |
| 45° | same order | 9 / 0 / 10 / 5 / 10 / 1 | 18 / 0 / 41 / 14 / 45 / 2 | 14 / 1 / 54 / 30 / 50 / 16 |
| 60° | same order | 12 / 0 / 16 / 2 / 11 / 0 | 28 / 0 / 45 / 28 / 50 / 0 | 27 / 0 / 62 / 42 / 64 / 0 |

Total filtered collisions by level (all 600 stress episodes per level): 15° → C1 5, C2 27, C3 51; 30° → 28 / 92 / 140; 45° → 35 / 120 / 165; 60° → 41 / 151 / 195. C0: 0 at every θ.
The filter roughly halves the action-v2 collision rates (for example θ = 15°: pooled C3 21.0% → 8.5%) but does not remove them. Intervention rate (usability diagnostic, not gated): 15° C1/C2/C3 = 7.9% / 27.7% / 50.0%; 60° = 7.5% / 22.4% / 34.1%; C0 0%.

| θ_max | G1 no gained collisions | G2 C0 persistent mission | G3 angle bound | G4 ceiling hover | G5 numerical | selected |
|---:|:--:|:--:|:--:|:--:|:--:|:--:|
| 15° | FAIL (83 gained) | FAIL | pass | pass | pass | no |
| 30° | FAIL (260) | FAIL | pass | pass | pass | no |
| 45° | FAIL (320) | FAIL | pass | pass | pass | no |
| 60° | FAIL (387) | FAIL | pass | pass | pass | no |

**Why G1 fails — mechanism (all 83 θ = 15° collisions re-simulated, `data/action_v3_collision_mechanism_15deg.json`).** In every case, at the collision the filter had rejected the residual and the canonical `u_base` itself was already predicted unsafe: the vehicle had been carried by earlier
accepted residuals into a state from which even the frozen common layer cannot avoid the obstacle (22 of 83 had an accepted residual within the last 6 steps; the rest earlier). A one-step binary filter evaluates only whether *this* command is safe over 12 steps; it cannot guarantee that the *state it leads to* stays
recoverable. Canonical A0 is collision-free here precisely because it never leaves its own trajectory.

**Why G2 fails (independent of the filter).** At C0 there are no obstacles, so the filter is a no-op and C0 results equal action-v2's. A persistent maximum residual is a constant-bearing pursuit that is kinematically unable to close the final metres: with v = 12 m/s, turn rate 75°/s, the minimum
orbit radius is v·sinθ/ω ≈ 2.4 m at 15° (> the 2.0 m capture radius), larger for bigger θ. Result at 15°: left/right 80% capture + 20% exhaustion, up 100% exhaustion (52% of episodes > 5× path), alternating 17% exhaustion; ≥ 30° → ~100% exhaustion. Only the persistent "down" command passes. The candidate set {15,30,45,60}° therefore cannot satisfy G2 for any filter. (θ ≲ 12° would be kinematically capture-feasible; not a candidate.)

## Conclusion and next decision

Per the pre-registered rule: **STOP, do not train**. The bounded-residual tactical interface cannot be made obstacle-safe by a one-step binary filter on top of the frozen common layer, and the C0 persistent-command gate is incompatible with every candidate angle.
Options (none applied, each needs its own pre-registration): (i) a **look-ahead recoverability** filter (accept `u_c` only if, after the horizon, the canonical `u_base` rollout is still safe) — a larger, still A1-only change; (ii) a **smaller candidate set** (θ ≤ 10°) combined with (i), accepting very limited authority;
(iii) a different interface (for example a discrete high-level tactical choice such as {base, bypass-left, bypass-right, climb} executed *through* the frozen bypass guidance), which gives A1 genuine tactical choices while the frozen layer remains the only thing that steers around obstacles; (iv) accept the evidence that PPO tactical control on top of canonical guidance gives no advantage over A0 in this benchmark.
