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
