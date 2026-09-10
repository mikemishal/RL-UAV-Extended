# FA-LAPG Foundation Validation Study

Status: development/foundation results, **not the final journal evaluation**.
Environment frozen at tag `clutter-env-v1`
(`93474b98a6a9ff909b0bc808b2117afb2d601958`). Foundation code committed at
`b8d9234883e4b9da54d42887ecce06ccd8eeeb1e` /
`ff587ef9e2b3e1ce203063fe3ad88e192e093440` (diagnostic study) and
`ff58c65d74a709c57c16f92cbcbd3b21073818f2` (FA-LAPG foundation code) on
branch `research/fa-lapg-foundation`.

## A. GRU dataset design

Supervised, non-RL trajectory-prediction dataset
(`experiments/fa_lapg_foundation/trajectory_dataset.py`). Rollouts driven by
the deployable measurement-mode `LeadInterceptPolicy`. Each sample contains
ONLY causally available information: estimated hostile position/velocity
(from the noisy measurement), defender position/velocity, protected-asset
(soldier) position, time-since-detection, and hostile-defender range, over
the last 8 detected steps (zero-padded/masked at the start of detection).
Ground-truth hostile position is used ONLY as the supervised target, never
as a model input. Targets are computed at 5 fixed horizons, exactly
grid-aligned to the environment's `dt=0.5s` (no interpolation needed).
Targets beyond the recorded episode length are omitted, never fabricated.

## B. Seed ranges

Train `70000-70019` (20/scenario), validation `71000-71007` (8/scenario),
test `72000-72007` (8/scenario) across all 6 scenarios. Disjoint from the
learning-benefit diagnostic pilot (`60000-60199`) and all other locked
training/robustness ranges (see `experiments/fa_lapg_foundation/seed_config.py`
and, going forward, `docs/experiment_seed_registry.md`).

## C. History length

8 most-recent DETECTED steps (4 seconds of causal history at `dt=0.5s`).

## D. Prediction horizons

0.5, 1.0, 2.0, 3.0, 4.0 seconds (fixed, multi-output GRU head).

## E. CV results

Constant-velocity baseline error grows from mean 2.88 (0.5s horizon) to
42.96 (4.0s horizon), pooled across all 6 scenarios (N≈17.5-17.8k samples
per horizon).

## F. CA results

Constant-acceleration (finite-difference acceleration, same methodology as
the diagnostic study's `ConstantAccelerationLeadPolicy`) is **consistently
worse than CV at every single horizon and scenario tested** (e.g. mean
error 3.74 at 0.5s rising to 107.26 at 4.0s) -- naive finite-difference
acceleration amplifies noise rather than correcting for it, exactly
mirroring the diagnostic study's CA-Lead guidance finding.

## G. GRU prediction results

The small single-layer GRU residual predictor (hidden size 32, multi-output
head over all 5 horizons) beats both CV and CA at every horizon and every
scenario:

| horizon | CV mean | CA mean | GRU mean | GRU vs CV reduction |
|---|---|---|---|---|
| 0.5s | 2.88 | 3.74 | 2.06 | 28% |
| 1.0s | 5.80 | 9.36 | 3.51 | 40% |
| 2.0s | 14.94 | 29.55 | 6.88 | 54% |
| 3.0s | 27.70 | 62.06 | 10.39 | 62% |
| 4.0s | 42.96 | 107.26 | 13.95 | 68% |

The relative advantage GROWS with horizon, confirming the hypothesis that
learned temporal correction becomes increasingly valuable as CV
extrapolation degrades.

## H. OOD/restricted-training results

A GRU trained ONLY on `nominal_open` + `strong_evasion`, evaluated on the 4
harder held-out scenarios, shows higher error than a model trained on all 6
scenarios evaluated on the same held-out set (e.g. `mobility_mismatch`:
pooled=6.81 vs restricted=10.02, +47% worse). Critically, the restricted
model STILL beats CV in every held-out scenario -- unlike the existing
LR-PPO guidance residual, the learned-prediction residual degrades
gracefully rather than becoming harmful under distribution shift.

## I. Dynamic-feasibility results

`uav_defend/guidance/dynamic_feasibility.py` computes individually
interpretable, policy-independent feasibility diagnostics (`theta_req_deg`,
`theta_available_deg`, `D_turn`, `required_speed`, `closing_speed_ratio`,
predicted accel/turn/climb saturation fractions, `E_reach`,
`obstacle_path_conflict`) using the EXACT SAME `advance_velocity`/
`apply_boundary` dynamics `SoldierEnv` uses. `D_turn` correlates negatively
with Lead success (r=-0.18, episode-level, N=238). Stratifying diagnostic
episodes into LOW/MEDIUM/HIGH `D_turn` tertiles:

| stratum | Lead success | LR-PPO success | True-State success | paired diff (LR-PPO-Lead) |
|---|---|---|---|---|
| LOW | 0.850 | 1.000 | 0.988 | +0.150 |
| MEDIUM | 0.924 | 0.937 | 0.924 | +0.013 |
| HIGH | 0.722 | 0.747 | 0.772 | +0.025 |

Learning benefit is LARGEST under low feasibility stress and SHRINKS under
high stress -- where even the True-State (perfect sensing) oracle also
struggles, indicating dynamic feasibility, not sensing, is the binding
constraint there.

## J. Joint prediction-error x feasibility analysis

Four-regime classification (median split on mean Lead CV prediction error x
mean `D_turn`, episode-level, N=225):

| regime | n | Lead success | LR-PPO success | mean LR-PPO benefit |
|---|---|---|---|---|
| high_error_low_stress | 75 | 0.880 | 1.000 | **+0.120** |
| low_error_high_stress | 75 | 0.800 | 0.840 | +0.040 |
| low_error_low_stress | 38 | 0.921 | 0.974 | +0.053 |
| high_error_high_stress | 37 | 0.703 | 0.730 | +0.027 (smallest) |

LR-PPO's benefit is largest exactly where prediction error is high but
dynamic stress is low, and smallest where both stressors compound --
motivating a combined (not single-mechanism) architecture.

## K. Clutter negative-transfer result

In `clutter_strong_maneuver`, 39/40 diagnostic episodes had obstacle
avoidance active; within that group LR-PPO's benefit is **-20.5pp**
(Lead=87.2%, LR-PPO=66.7%). LR-PPO (trained without obstacles) is an
out-of-distribution reference baseline in clutter, not a validated
obstacle-aware controller.

## L. Interpretation

- CA prediction was consistently inferior to CV in every scenario/horizon
  tested; it should not be pursued further as a target-prediction or
  guidance model.
- Learned temporal residual prediction beat CV/CA in every tested
  scenario/horizon, with the advantage growing at longer horizons.
- Learning benefit was greatest in high-prediction-error/low-dynamic-stress
  episodes.
- High dynamic stress reduced the benefit of both better sensing
  (True-State oracle) and the existing LR-PPO -- feasibility, not sensing
  alone, appears to bind in that regime.
- Existing LR-PPO showed harmful out-of-distribution transfer in clutter +
  strong maneuver (statistically resolved negative effect in the original
  diagnostic pilot; concentrated in obstacle-active episodes here).

These are development/foundation results guiding the design of a possible
future controller; they are **not** the final journal evaluation and use
non-final (development) seed ranges throughout.
