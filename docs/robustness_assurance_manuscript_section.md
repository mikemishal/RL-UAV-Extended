# Robustness and Assurance Under Off-Nominal Conditions

Numbers are from `results/journal_robustness_assurance/`; evidence strength for each claim is in [robustness_assurance_claims_evidence.md](./robustness_assurance_claims_evidence.md). Effects are D1 − D0+ in percentage points (pp) with hierarchical paired-bootstrap 95% CIs (10,000 replicates; 3 training seeds per architecture).

## 1. Purpose

This section tests whether adversarial defender adaptation (D1) remains beneficial, neutral, or harmful relative to the equal-budget control (D0+) when sensing and kinematic conditions depart from nominal. All episodes use the unseen adaptive attacker A1-203 and frozen checkpoints, in the no-obstacle (C0) and medium-clutter (C2) environments.

Operating points come from the original LR-PPO sweep grids:

| Study | Low / disadvantaged | Nominal | High / advantaged |
|---|---|---|---|
| R1 measurement variance | 0.05 | 0.50 | 4.00 |
| R2 detection radius | 5 (short) | 15 | 30 (long) |
| R3 (v_d, v_e), ratio | (12, 12), 1.00 | (18, 12), 1.50 | (18, 8), 2.25 |

New episodes: 21,600 (18 defenders × 6 off-nominal settings × 2 levels × 100 seeds; seeds 200,000–200,099). Nominal cells are reused from the canonical C0 evaluation (C0) and the 32,400-episode evaluation (C2). Nominal and off-nominal cells therefore use different seed blocks.

## 2. Results

### 2.1 Measurement noise (R1)

Higher noise reduces absolute safe interception for every defender. D0+ success at C0 falls from 0.80 (low) to 0.46 (high) for LR-PPO-C, from 0.69 to 0.55 for PPO-C, and from 0.53 to 0.42 for PPO-Kalman-C.

PPO-Kalman-C's D1 benefit remains positive and significant at C0 at both noise extremes (+19.0 [+5.3, +32.3] low; +22.0 [+6.0, +37.0] high). At C2 it stays positive in point estimate (+10.3 low; +8.7 high) but the CIs include 0. The C0 difference between high noise and nominal is +3.3 pp [−14.7, +21.7], so the data do not show a larger benefit under noise.

PPO-C's D1 effect is negative at C0 for low noise (−26.0 [−45.0, −10.3]) and indistinguishable from zero at high noise (+4.7 [−6.0, +15.0]). At C2 it is near zero or mildly negative. LR-PPO-C shows no detectable effect at either level (−6.0 to −4.0 pp; all CIs include 0).

### 2.2 Detection radius (R2)

Short detection lowers success for all defenders (e.g. C0 D0+: 0.41 PPO-C, 0.33 PPO-Kalman-C, 0.34 LR-PPO-C) and raises the unsafe-intercept share to 13–22% for D0+ (9–22% across D0+ and D1; Table 3).

PPO-Kalman-C retains a significant D1 benefit at C0 under short (+11.3 [+3.0, +19.7]) and long (+18.7 [+8.7, +29.0]) detection. At C2 the benefit shrinks to +1.3 [−7.7, +10.7] at short and +5.0 [−9.0, +19.3] at long detection, compared with +16.7 [+3.7, +28.3] at nominal. This C2 contrast was not formally tested. At C0 the short-minus-nominal difference is −7.3 [−23.0, +9.0].

PPO-C's D1 effect is negative at C0 under long detection (−23.0 [−48.0, −1.7]) and not distinguishable from zero under short detection. LR-PPO-C shows no detectable effect (+1.3 short; −3.7 long at C0; −3.3 and 0.0 at C2). LR-PPO-C's D0+ success at long detection (0.61 at C0, 0.65 at C2) is higher than at short detection (0.34, 0.49).

### 2.3 Relative mobility (R3)

| Level | Architecture | (12, 12) | nominal (18, 12) | (18, 8) |
|---|---|---|---|---|
| C0 | PPO-C | −21.7 [−35.3, −8.0] | −26.3 [−51.3, −6.0] | −8.7 [−20.0, +2.0] |
| C0 | PPO-Kalman-C | +17.3 [+2.7, +34.0] | +18.7 [+3.7, +35.0] | +23.7 [+8.3, +37.0] |
| C0 | LR-PPO-C | +1.0 [−9.3, +11.0] | −12.7 [−30.0, +5.0] | −9.3 [−21.7, +2.7] |
| C2 | PPO-C | −10.0 [−20.0, 0.0] | +0.3 [−10.7, +11.0] | −0.3 [−9.7, +8.3] |
| C2 | PPO-Kalman-C | +14.3 [−3.0, +32.0] | +16.7 [+3.7, +28.3] | +17.3 [+7.0, +27.3] |
| C2 | LR-PPO-C | −1.7 [−10.3, +7.0] | −9.3 [−21.0, +2.3] | −3.3 [−13.0, +6.7] |

PPO-Kalman-C's effect is positive at all three points at both levels. PPO-C is negative at C0 at all three points. LR-PPO-C shows no detectable effect.

Absolute success is not monotone in the speed ratio. For example LR-PPO-C D0+ at C0 is 0.77 at (12, 12), 0.70 at nominal and 0.69 at (18, 8). The original LR-PPO defender-speed sweep shows the same direction of change for PPO (0.87 at v_d = 12 vs 0.85 at 18).

### 2.4 C0 versus C2

Clutter lowers absolute success for PPO-C and PPO-Kalman-C at every operating point; LR-PPO-C D0+ is comparable (0.74 at C2 nominal vs 0.70 at C0). PPO-Kalman-C's D1 benefit is smaller in point estimate at C2 than at C0 at all seven points, and CIs include 0 at five of seven C2 points. PPO-C's negative D1 effect at C0 is not reproduced at C2 (near zero except at the defender-disadvantaged point). LR-PPO-C is neutral at both levels.

### 2.5 Failure modes, efficiency and safety

At the challenging settings (Table 3), non-success outcomes are mostly attacker asset-reach, with unsafe interceptions highest at short detection (up to 22%). Defender obstacle collisions are 0% at C0 and at most 5% at C2. Over all cells, D1 and D0+ collision rates are similar (e.g. PPO-C C2 3.1% vs 2.2%; PPO-Kalman-C C2 2.7% vs 2.5%).

PPO-Kalman-C's D1 episodes are shorter on average at C0 (97 vs 151 steps) and C2 (234 vs 270), with shorter defender paths. PPO-C's D1 episodes are longer than D0+ (147 vs 115 steps at C0). Kalman position error is similar between D0+ and D1 (e.g. 5.29 vs 4.49 at C0 high noise). LR-PPO's mean residual angle is 6.6–12.0°, with at most 0.1% of steps near the 30° bound.

## 3. Scorecard

D1 − D0+ in pp. Labels follow the quantitative results above.

| Architecture | Nominal | High noise | Short detection | Defender disadvantaged | C0 verdict | C2 verdict |
|---|---|---|---|---|---|---|
| PPO-C | C0 −26.3 / C2 +0.3 | −0.3 to +4.7 | −7.7 / −3.7 | −21.7 / −10.0 | Negative adaptation | Neutral |
| PPO-Kalman-C | +18.7 / +16.7 | +22.0 / +8.7 | +11.3 / +1.3 | +17.3 / +14.3 | Robust positive adaptation | Condition-dependent (positive; attenuated under short detection and noise) |
| LR-PPO-C | −12.7 / −9.3 | −6.0 / −4.0 | +1.3 / −3.3 | +1.0 / −1.7 | Neutral | Neutral |

Each cell lists C0 / C2 where two values are shown.

## 4. Assurance interpretation

**Sensing degradation.** Noisier measurements lower success for all architectures. PPO-Kalman-C keeps a positive D1 effect at C0 under high noise; the benefit is not shown to be larger than at nominal. This does not isolate the Kalman filter as the cause (claim 10).

**Reduced warning time.** Short detection is the most damaging single perturbation for absolute success. PPO-Kalman-C's D1 benefit remains significant at C0 but is no longer distinguishable from zero at C2.

**Kinematic disadvantage.** Reducing the defender to equal speed does not reverse the sign of PPO-Kalman-C's effect. Absolute success does not fall monotonically with the speed ratio, as in the original sweep.

**Environmental complexity.** C2 attenuates PPO-Kalman-C's benefit and removes PPO-C's negative effect. LR-PPO-C is neutral in both.

## 5. Discussion

The adaptation effect depends on the architecture rather than on the perturbation. Within the tested range, perturbations change absolute performance much more than they change the D1 − D0+ sign. PPO-Kalman-C's gain is the most consistent result, and is strongest in the no-obstacle case. PPO-C's C0 loss continues to look like an architecture × environment interaction, and its D0+ control is strong at C0, which makes the D1 deficit larger. LR-PPO-C's structured Lead-based guidance neither gains nor loses measurably from adversarial adaptation, which fits its bounded 30° residual.

Statements that an effect is larger or smaller under a given condition are descriptive unless stated. Only two difference-of-differences tests were run, and neither CI excludes 0.

## 6. Limitations

- One unseen attacker (A1-203); no A0 or A1-201/202 off-nominal runs.
- Three training seeds per architecture; PPO-C and LR-PPO-C CIs are wide, and PPO-C's C0 effect is dominated by one seed (72).
- Off-nominal conditions are single-factor changes; interactions (e.g. noise × detection) were not tested.
- Nominal cells are reused from earlier seed blocks.
- The "defender-disadvantaged" point has equal speeds; the original protocol contains no point with the attacker faster.
- Success is measured against scripted-class attackers; no hardware, communication degradation, or self-play was evaluated.
