# Robustness and Assurance — Claims / Evidence Ledger

Campaign: `experiments/journal_robustness_assurance.py` (code); results in `results/journal_robustness_assurance/`.

- Attacker: A1-unseen-203 only. Defenders: 18 frozen instances (PPO-C seeds 71–73, PPO-Kalman-C 74–76, LR-PPO-C 77–79; D0+ and D1). No retraining.
- New episodes: 21,600 = 18 defenders × 6 off-nominal settings × 2 levels (C0, C2) × 100 seeds. Evaluation seeds 200,000–200,099 (new block; common random numbers across D0+/D1 and all cells).
- Checkpoint integrity verified before and after the run (all hashes OK). Every cell asserts config values (noise, radius, v_d, v_e, standby=True, estimator, obstacle/route config) before launch.
- Nominal cells are reused, not rerun: C0 from `results/defender_adaptation_canonical_c0_evaluation/episodes.csv` (seeds 130000–130099); C2 from `results/defender_adaptation_evaluation/episodes.csv`. Nominal and off-nominal cells therefore use **different seed blocks**.
- Statistic: D1 − D0+ in percentage points (pp), hierarchical paired bootstrap over the 3 training seeds and matched evaluation seeds, 10,000 replicates, 95% CI. n = 300 episodes per cell per generation (3 policies × 100 seeds); these are not 300 independent policies.
- Source tables: `d1_minus_d0plus_robustness.csv` (all effects/CIs), `d1_minus_d0plus_robustness_full.json` (per-seed effects), `outcome_decomposition_challenging.csv`, `interaction_tests.csv`.

## Original-protocol values used (provenance)

| Variable | Original sweep values | Nominal | Journal values |
|---|---|---|---|
| Measurement variance (R1) | 0.05, 0.25, 0.50, 1.00, 2.00, 4.00 | 0.50 | low 0.05, high 4.00 |
| Detection radius (R2) | 5, 10, 15, 20, 25, 30 | 15 | short 5, long 30 |
| Defender speed v_d (R3) | 12, 15, 18, 21, 24 | 18 | 12 (disadvantaged), 18 |
| Attacker speed v_e (R3) | 8, 10, 12, 14, 16 | 12 | 12 (disadvantaged), 8 (advantaged) |

R3 pairs (v_d, v_e): disadvantaged (12, 12) ratio 1.00; nominal (18, 12) ratio 1.50; advantaged (18, 8) ratio 2.25. The original protocol never tested v_e > v_d, so ratio 1.00 is the least favourable tested point.

## Effects, D1 − D0+ (pp [95% CI])

| Architecture | Level | Nominal | R1 low | R1 high | R2 short | R2 long | R3 disadv. | R3 adv. |
|---|---|---|---|---|---|---|---|---|
| PPO-C | C0 | −26.3 [−51.3, −6.0] | −26.0 [−45.0, −10.3] | +4.7 [−6.0, +15.0] | −7.7 [−21.0, +4.0] | −23.0 [−48.0, −1.7] | −21.7 [−35.3, −8.0] | −8.7 [−20.0, +2.0] |
| PPO-C | C2 | +0.3 [−10.7, +11.0] | −9.7 [−20.3, +1.0] | −0.3 [−12.7, +11.7] | −3.7 [−15.0, +7.0] | −5.7 [−19.0, +7.0] | −10.0 [−20.0, 0.0] | −0.3 [−9.7, +8.3] |
| PPO-Kalman-C | C0 | +18.7 [+3.7, +35.0] | +19.0 [+5.3, +32.3] | +22.0 [+6.0, +37.0] | +11.3 [+3.0, +19.7] | +18.7 [+8.7, +29.0] | +17.3 [+2.7, +34.0] | +23.7 [+8.3, +37.0] |
| PPO-Kalman-C | C2 | +16.7 [+3.7, +28.3] | +10.3 [−4.0, +24.0] | +8.7 [−1.0, +18.3] | +1.3 [−7.7, +10.7] | +5.0 [−9.0, +19.3] | +14.3 [−3.0, +32.0] | +17.3 [+7.0, +27.3] |
| LR-PPO-C | C0 | −12.7 [−30.0, +5.0] | −5.0 [−19.7, +6.3] | −6.0 [−22.0, +8.7] | +1.3 [−6.7, +9.0] | −3.7 [−15.0, +8.0] | +1.0 [−9.3, +11.0] | −9.3 [−21.7, +2.7] |
| LR-PPO-C | C2 | −9.3 [−21.0, +2.3] | −4.0 [−14.0, +5.3] | −4.0 [−18.0, +10.0] | −3.3 [−10.7, +4.0] | 0.0 [−9.7, +9.7] | −1.7 [−10.3, +7.0] | −3.3 [−13.0, +6.7] |

## Claims

| # | Claim | Quantitative evidence | Dataset | Strength |
|---|---|---|---|---|
| 1 | PPO-Kalman-C's D1 benefit over D0+ persists at C0 across all seven operating points. | +11.3 to +23.7 pp; all seven 95% CIs exclude 0; per-seed effects positive in all 21 (3 seeds × 7 points) cases. | New cells + canonical C0 nominal | **Strong** (C0, this attacker) |
| 2 | PPO-Kalman-C's D1 benefit is positive in point estimate at every C2 operating point, but is attenuated under reduced warning time. | C2 effects +1.3 to +17.3 pp; CIs exclude 0 only at nominal (+16.7) and defender-advantaged (+17.3); short detection +1.3 [−7.7, +10.7]. | New cells + C2 nominal | **Moderate** (direction consistent, magnitude condition-dependent) |
| 3 | The PPO-Kalman-C D1 benefit under high noise is not larger than nominal. | C0 difference-of-differences (high − nominal) +3.3 [−14.7, +21.7]. | `interaction_tests.csv` | **Unsupported** as an increase |
| 4 | Short detection reduces PPO-Kalman-C's D1 benefit at C0. | Short − nominal = −7.3 [−23.0, +9.0]; CI includes 0. | `interaction_tests.csv` | **Descriptive** only |
| 5 | The larger attenuation at C2 short detection (+1.3 vs +16.7 nominal) reflects a detection × clutter interaction. | Not tested formally. | — | **Descriptive** only |
| 6 | PPO-C's D1 is worse than D0+ at C0 under most conditions, consistent with the previously reported C0 regression. | Negative in 6 of 7 points; CIs exclude 0 at nominal, low noise, long detection, disadvantaged mobility. High noise +4.7 [−6.0, +15.0]. | New cells + canonical C0 nominal | **Moderate** (seed 72 dominates: −53 pp at nominal) |
| 7 | PPO-C shows no reliable D1 effect at C2. | Range −10.0 to +0.3 pp; all CIs include 0 or touch 0 (disadvantaged [−20.0, 0.0]). | New cells + C2 nominal | **Moderate** (neutral to mildly negative) |
| 8 | LR-PPO-C shows no reliable D1 effect at C0 or C2 under any tested perturbation. | All 14 CIs include 0; point estimates −12.7 to +1.3 pp; per-seed signs mixed in 12 of 14 cells. | New cells + nominal | **Moderate** (no detectable effect; limited power) |
| 9 | Higher measurement noise and shorter detection range degrade absolute safe-interception for the D0+ and D1 defenders. | E.g. LR-PPO-C D0+ C0: 0.80 (low) → 0.46 (high noise); 0.34 at short detection. PPO-Kalman-C D0+ C0: 0.53 → 0.42 (high noise). | New cells | **Strong** (large, consistent) |
| 10 | The Kalman filter caused PPO-Kalman-C's D1 improvement. | Kalman position error is similar between D0+ and D1 (e.g. C0 high noise 5.29 vs 4.49). The estimator is shared; the data do not isolate it as the cause. | New cells | **Unsupported** |
| 11 | D1 does not increase obstacle collisions or unsafe interceptions materially in the challenging settings. | Defender collision rates ≤ 5% everywhere; unsafe-intercept rates 0.3–22%, mostly similar between D0+ and D1 (Table 3). | `outcome_decomposition_challenging.csv` | **Descriptive** |
| 12 | LR-PPO's residual remains well inside the 30° authority bound under perturbation. | Mean residual angle 6.6–12.0°; fraction within the near-bound margin ≤ 0.1%. | New cells | **Strong** (descriptive of this policy) |
| 13 | Lower defender speed does not monotonically hurt safe interception. | LR-PPO-C C0 D0+: 0.77 at (12, 12) vs 0.70 at nominal. The original LR-PPO sweep shows the same direction (PPO 0.87 at v_d=12 vs 0.85 at 18). | New cells + original mobility sweep | **Descriptive** |

## Caveats attached to all claims

- A single unseen attacker (A1-unseen-203); results do not cover A1-201/202 or A0 under perturbation.
- N = 100 seeds × 3 training seeds per architecture cell; 95% CIs are wide for PPO-C and LR-PPO-C.
- Nominal cells come from earlier seed blocks; differences between nominal and off-nominal effects mix operating-point and seed-block variation. Only the pre-specified difference-of-differences tests were run.
- The defender-disadvantaged point has v_d = v_e; no point has the attacker faster than the defender.
