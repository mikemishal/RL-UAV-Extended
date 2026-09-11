# RA-LAPG 2x2 Factorial Ablation Study

**Status: DEVELOPMENT STUDY, not the final journal evaluation.** Uses
`ra_lapg_development` seeds (100000-100199), N=200/scenario. Environment
frozen at `clutter-env-v1` (`93474b98a6a9ff909b0bc808b2117afb2d601958`).

## Multiscale predictor result

Stratified (equal-count-per-band) horizon training substantially reduced,
but did not eliminate, the short-horizon regression found in the original
uniformly-trained horizon-conditioned predictor:

| horizon | CV | old uniform GRU | new multiscale GRU |
|---|---|---|---|
| 0.5s | 2.95 | 8.81 | 4.69 |
| 1.0s | 5.90 | 10.72 | 6.25 |
| 2.0s | 15.25 | 15.41 | 10.27 |
| 4.0s | 44.54 | 25.95 | 19.90 |
| 8.0s | 114.56 | 31.04 | 28.69 |
| 12.0s | 158.26 | 31.74 | 30.87 |
| 16.0s | 191.24 | 35.43 | 37.29 |

## Hybrid predictor decision

Since the multiscale GRU is still worse than CV at 0.5s/1.0s, a
deterministic hybrid rule is used everywhere in this study:
`p_pred(h) = p_CV(h)` for `h <= 2.0s`, `p_CV(h) + Delta_p(h)` for `h > 2.0s`.
`h_switch=2.0s` chosen from this validation crossover, not final-evaluation
data. **Methodological note**: the multiscale GRU already beats CV exactly
at 2.0s, so this switch is conservative; a finer crossover search between
1.0-2.0s is deferred to future work and must not be mixed with the
reachability question this study addresses.

## 2x2 controller definitions

|                    | KINEMATIC | REACHABILITY |
|---|---|---|
| CV prediction      | Lead      | RA-Lead      |
| Learned prediction | LP-Lead   | RA-LAPG      |

- **Lead**: existing deployable CV pure-pursuit intercept controller.
- **RA-Lead**: CV prediction + dynamic-reachability candidate selection (no learned model).
- **LP-Lead**: hybrid-predictor-corrected CV Lead, no reachability.
- **RA-LAPG**: hybrid predictor + dynamic-reachability candidate selection.

All four share the identical candidate-time grid `(1,2,4,6,8,12,16)`s,
defender dynamics model, reachability metric, `intercept_radius=2.5`
threshold, and lexicographic selection rule (`uav_defend.guidance.
reachability_selection`) -- the ONLY difference is the target predictor
(CV vs. hybrid) and whether reachability filtering is applied at all.

## N and seed range

N=200 matched seeds per scenario, `ra_lapg_development` = 100000-100199.
4 open scenarios (`nominal_open`, `strong_maneuver`, `strong_evasion`,
`mobility_mismatch`). Clutter explicitly deferred.

## Success table (N=200, Wilson CI)

| scenario | Lead | RA-Lead | LP-Lead | RA-LAPG | True-State | LR-PPO |
|---|---|---|---|---|---|---|
| nominal_open | 0.855 | 0.865 | 0.865 | 0.850 | 0.920 | 0.895 |
| strong_maneuver | 0.655 | 0.650 | 0.735 | 0.690 | 0.735 | 0.690 |
| strong_evasion | 0.830 | 0.820 | 0.925 | 0.830 | 0.920 | 0.905 |
| mobility_mismatch | 0.695 | 0.795 | 0.655 | 0.905 | 0.870 | 0.885 |

## Factorial main effects (bootstrap 95% CI)

| scenario | delta_prediction_kinematic | delta_reachability_cv | delta_reachability_learned | interaction |
|---|---|---|---|---|
| nominal_open | +0.010 [-.04,.06] | +0.010 [-.03,.06] | -0.015 [-.07,.04] | -0.025 |
| strong_maneuver | **+0.080** [.005,.155] | -0.005 [-.08,.07] | -0.045 [-.12,.03] | -0.040 |
| strong_evasion | **+0.095** [.05,.145] | -0.010 [-.065,.045] | **-0.095** [-.145,-.05] | -0.085 |
| mobility_mismatch | -0.040 [-.12,.04] | **+0.100** [.02,.175] | **+0.250** [.185,.32] | **+0.150** |

## Mobility-mismatch positive synergy

Reachability with learned prediction (RA-LAPG) gives a +25.0pp
improvement over Lead, with a significant +15.0pp positive interaction
term -- RA-LAPG (0.905) exceeds even the True-State sensing oracle
(0.870) in this scenario. Both reachability main effects (CV and
learned) are independently significant here.

## Strong-evasion negative reachability interaction

Learned prediction alone (LP-Lead, 0.925) is the single best-performing
ablation controller in this scenario, matching the True-State oracle
(0.920). Adding reachability (RA-LAPG, 0.830) **erases almost the entire
benefit** (-9.5pp vs. LP-Lead, CI excludes 0) -- reachability filtering
is actively interfering with an already-effective learned-prediction
correction in this regime.

## Policy-induced covariate-shift finding

Predictor error under RA-LAPG-controlled rollouts is ~30% higher than
under Lead-controlled rollouts specifically in `mobility_mismatch` (h=4s:
28.4 vs 21.6; h=8s: 39.4 vs 32.6), while nearly identical in
`nominal_open`. RA-LAPG intercepts much faster in this scenario, driving
the hostile into a trajectory distribution the predictor (trained on
Lead rollouts) was not trained on.

## Current limitations

- Fixed-vs-receding-target reachability rollout comparison not yet run (this study only used FIXED_DIRECTION).
- No per-step saturation/D_turn diagnostic logging collected for this run (only coarse outcome-based failure partition).
- No dedicated reachability-calibration analysis.
- Hybrid switch horizon (2.0s) not finely re-optimized.
