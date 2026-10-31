# Defender-evasion gating (action-v6) — generated evidence tables

Specification SHA-256 `249da482ac7d1c11bdbe5954589526645ce40fc1e72d19a80d61da82def4645a`; 18,000 episodes = 20 cells (4 clutter levels x 5 defenders) x 100 matched seeds (145000-145099) x 9 policies. BASE (blend 0 through the env) is bit-identical to canonical A0 in all 2,000 matched scenarios; SWITCH_b is trajectory-identical to FIXED_b in all cells (flags: {'base_equals_a0': True, 'switch_equals_fixed': True, 'deterministic_replay': True, 'finite': True}). Pooled over the five defenders per level (500 episodes per level).

## 1. Gates per policy

| policy | G1 collisions | G2 mission | G3 authority | G3 positive | G4 path | G6 obstacle authority | gained collisions |
|---|---|---|---|---|---|---|---|
| FIXED_0.25 | FAIL | FAIL | FAIL | FAIL | PASS | PASS | 11 |
| FIXED_0.50 | FAIL | FAIL | PASS | PASS | PASS | FAIL | 4 |
| FIXED_0.75 | FAIL | FAIL | PASS | PASS | FAIL | FAIL | 31 |
| FIXED_1.00 | FAIL | FAIL | PASS | PASS | FAIL | FAIL | 38 |
| SWITCH_0.25 | FAIL | FAIL | FAIL | FAIL | PASS | PASS | 11 |
| SWITCH_0.50 | FAIL | FAIL | PASS | PASS | PASS | FAIL | 4 |
| SWITCH_0.75 | FAIL | FAIL | PASS | PASS | FAIL | FAIL | 31 |
| SWITCH_1.00 | FAIL | FAIL | PASS | PASS | FAIL | FAIL | 38 |

G5 (BASE == canonical A0): PASS.

## 2. Obstacle collisions by clutter level (BASE has 0 in every cell)

| blend | C0 | C1 | C2 | C3 | total gained / 2,000 |
|---|---|---|---|---|---|
| 0.25 | 0 | 1 | 3 | 7 | 11 |
| 0.50 | 0 | 0 | 2 | 2 | 4 |
| 0.75 | 0 | 0 | 11 | 20 | 31 |
| 1.00 | 0 | 1 | 14 | 23 | 38 |

## 3. Collisions by defender family (gained, pooled over levels)

| blend | greedy | kalman_greedy | pn | lead | rmpc |
|---|---|---|---|---|---|
| 0.25 | 3 | 3 | 3 | 1 | 1 |
| 0.50 | 0 | 0 | 0 | 2 | 2 |
| 0.75 | 8 | 8 | 7 | 4 | 4 |
| 1.00 | 7 | 8 | 10 | 6 | 7 |

## 4. Mission gate G2: endurance exhaustion / timeout / orbit / fly-by (BASE: 0 / 0 / 0 / 0.0%)

| blend | exhausted C0 / C1 / C2 / C3 | exhausted total | timeouts | orbit >=1 rev | episodes with >=1 overshoot (fly-by) |
|---|---|---|---|---|---|
| 0.25 | 7 / 7 / 5 / 4 | 23 | 0 | 0 | 2.4% |
| 0.50 | 9 / 11 / 8 / 8 | 36 | 0 | 0 | 3.1% |
| 0.75 | 16 / 18 / 11 / 4 | 49 | 0 | 0 | 5.7% |
| 1.00 | 18 / 15 / 11 / 4 | 48 | 0 | 0 | 6.9% |

## 5. Tactical authority G3 (vs matched BASE)

| blend | episodes diverging >= 1 m | median / max divergence (m) | outcome changed | BASE interception -> capture | capture -> interception | mean min-separation diff (m) | mean extra steps | mean extra path (m) | max deflection (deg) |
|---|---|---|---|---|---|---|---|---|---|
| 0.25 | 3.0% | 0.05 / 4.1 | 80 | 38 | 3 | +0.032 | +0.72 | +3.8 | 9.2 |
| 0.50 | 6.3% | 0.12 / 5.0 | 101 | 47 | 4 | +0.048 | +1.15 | +6.0 | 18.2 |
| 0.75 | 8.8% | 0.21 / 5.8 | 145 | 47 | 2 | +0.073 | +1.63 | +8.6 | 28.3 |
| 1.00 | 11.3% | 0.31 / 6.5 | 180 | 62 | 6 | +0.099 | +1.85 | +9.8 | 40.4 |

## 6. Path efficiency G4 (BASE: max 3.8x, 0 episodes > 5x)

| blend | mean | median | max | >1.3x | >2x | >5x | exhausted | timeout |
|---|---|---|---|---|---|---|---|---|
| 0.25 | 1.063 | 1.001 | 4.91 | 47 | 37 | 0 | 23 | 0 |
| 0.50 | 1.101 | 1.001 | 5.16 | 63 | 59 | 6 | 36 | 0 |
| 0.75 | 1.147 | 1.001 | 7.61 | 109 | 83 | 18 | 49 | 0 |
| 1.00 | 1.159 | 1.002 | 7.63 | 125 | 96 | 15 | 48 | 0 |

## 7. Mechanisms (re-simulated)

| blend | gained-collision mechanism | exhaustions: with fly-by / total (mean overshoots; mean extra steps vs BASE) | >5x cases (all endurance exhaustion?) |
|---|---|---|---|
| 0.25 | evasion ended >3 steps earlier; frozen layer owned the collision step: 11 | 23 / 23 (2.0) | 0 (True) |
| 0.50 | evasion ended >3 steps earlier; frozen layer owned the collision step: 3; evasion ended 1-3 steps earlier; frozen layer owned the collision step: 1 | 36 / 36 (2.5) | 6 (False) |
| 0.75 | evasion ended >3 steps earlier; frozen layer owned the collision step: 19; evasion ended 1-3 steps earlier; frozen layer owned the collision step: 12 | 49 / 49 (2.4) | 18 (True) |
| 1.00 | evasion ended >3 steps earlier; frozen layer owned the collision step: 24; evasion ended 1-3 steps earlier; frozen layer owned the collision step: 13; evasion executed on the collision step; frozen layer not owning the collision step: 1 | 48 / 48 (2.6) | 15 (False) |

## 8. Net capture effect (BASE: 232 captures / 2,000; matched episodes)

| blend | captures (C0 / C1 / C2 / C3) | total | change vs BASE | captures lost to endurance exhaustion (BASE captured -> exhausted) |
|---|---|---|---|---|
| 0.25 | 34 / 38 / 69 / 92 | 233 | +1 | 23 |
| 0.50 | 32 / 35 / 68 / 99 | 234 | +2 | 36 |
| 0.75 | 29 / 32 / 60 / 87 | 208 | -24 | 43 |
| 1.00 | 34 / 34 / 61 / 90 | 219 | -13 | 38 |

## 9. Suppression bookkeeping (steps, pooled)

| blend | no defender information | obstacle authority | primitive inactive (alpha=0) | executed while planner owned steering |
|---|---|---|---|---|
| 0.25 | 14025 | 16 | 1896 | 0 |
| 0.50 | 14636 | 16 | 1973 | 0 |
| 0.75 | 15321 | 36 | 2073 | 0 |
| 1.00 | 15657 | 47 | 2088 | 0 |
