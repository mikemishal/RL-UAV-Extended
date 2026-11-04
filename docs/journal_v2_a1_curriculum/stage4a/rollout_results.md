# Stage4A per-rollout validation results

Moving canonical asset (`legacy_gaussian_random_walk`), defender parked + masked + no interception, C0. Alternating validation: odd rollouts on set A (`169000–169099`), even rollouts on set B (`169100–169199`).
Retention anchors are evaluated at every rollout on their own fixed blocks (Anchor 1 `169200–169299`, Anchor 2 `169300–169399`, Anchor 3 `169400–169499`).

Frozen gate (all of): capture ≥ 0.95, exhaustion ≤ 0.02, timeout ≤ 0.01, median path-excess ≤ 1.50, >2× ≤ 0.05, >3× ≤ 0.02, >5× = 0, no persistent orbit, repeated fly-by episodes ≤ 0.05, anchors ≥ 0.98 / ≥ 0.80 / ≥ 0.75,
no numerical failure. Mastery = two consecutive passing rollouts. Retention stop = two consecutive failures of one anchor (never triggered). Timeout and non-finite entries are 0 at every rollout; orbit episodes occurred in 4 of the 36 rollout validations (1 episode each).
The median-path-excess check failed in 3 of 36 rollouts (all seed 202, set B): the path-excess distribution is bimodal (first-pass captures ≈ 1.05, multi-fly-by episodes ≫ 2), so the median jumps to 2.6–2.8 when just over half of the episodes exceed 2× (>2× = 0.51–0.52). Matched canonical A0 capture is exactly 1.00 on every validation set.
The gate fails at all 36 rollout validations on capture, exhaustion, >2×, >3×, >5× and repeated fly-by rate (the last three columns of the gate are omitted from the tables; see the seed manifests).

Mean validation capture, first six / last six rollouts: seed 201 0.757 → 0.798 (best 0.85); seed 202 0.710 → 0.745 (best 0.78); seed 203 0.797 → 0.827 (best 0.89). Mean training-episode capture, first three / last three rollouts:
201 0.653 → 0.698; 202 0.633 → 0.640; 203 0.659 → 0.681. Stage3-v2 parents on the 1,000-scenario baseline: 69.2 / 67.8 / 71.0 %.

## Seed 201 — FAIL after 12 rollouts (393,216 Stage-4A timesteps)
| Rollout | Set | Train capture | Val capture | Val exhaustion | Val repeated fly-by | Val >3× | Val median path-excess | Anchor 1 | Anchor 2 | Anchor 3 | Gate |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | A | 0.664 | 0.73 | 0.27 | 0.29 | 0.33 | 1.07 | 1.00 | 0.94 | 0.91 | fail |
| 2 | B | 0.652 | 0.76 | 0.24 | 0.29 | 0.36 | 1.11 | 1.00 | 0.93 | 0.89 | fail |
| 3 | A | 0.643 | 0.74 | 0.26 | 0.28 | 0.33 | 1.05 | 1.00 | 0.91 | 0.86 | fail |
| 4 | B | 0.662 | 0.79 | 0.21 | 0.26 | 0.33 | 1.12 | 1.00 | 0.92 | 0.85 | fail |
| 5 | A | 0.670 | 0.70 | 0.30 | 0.32 | 0.35 | 1.07 | 1.00 | 0.89 | 0.81 | fail |
| 6 | B | 0.691 | 0.82 | 0.18 | 0.22 | 0.27 | 1.08 | 1.00 | 0.90 | 0.81 | fail |
| 7 | A | 0.674 | 0.76 | 0.24 | 0.27 | 0.34 | 1.07 | 1.00 | 0.88 | 0.81 | fail |
| 8 | B | 0.676 | 0.84 | 0.16 | 0.23 | 0.31 | 1.08 | 1.00 | 0.88 | 0.81 | fail |
| 9 | A | 0.680 | 0.76 | 0.24 | 0.28 | 0.31 | 1.06 | 1.00 | 0.90 | 0.82 | fail |
| 10 | B | 0.698 | 0.82 | 0.18 | 0.22 | 0.28 | 1.07 | 1.00 | 0.88 | 0.80 | fail |
| 11 | A | 0.680 | 0.76 | 0.24 | 0.26 | 0.29 | 1.06 | 1.00 | 0.85 | 0.79 | fail |
| 12 | B | 0.715 | 0.85 | 0.15 | 0.19 | 0.26 | 1.07 | 1.00 | 0.85 | 0.79 | fail |

## Seed 202 — FAIL after 12 rollouts (393,216)
| Rollout | Set | Train capture | Val capture | Val exhaustion | Val repeated fly-by | Val >3× | Val median path-excess | Anchor 1 | Anchor 2 | Anchor 3 | Gate |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | A | 0.619 | 0.71 | 0.29 | 0.31 | 0.37 | 1.13 | 1.00 | 0.97 | 0.91 | fail |
| 2 | B | 0.647 | 0.73 | 0.27 | 0.32 | 0.41 | 2.58 | 1.00 | 0.97 | 0.86 | fail |
| 3 | A | 0.633 | 0.70 | 0.30 | 0.32 | 0.36 | 1.11 | 1.00 | 0.96 | 0.86 | fail |
| 4 | B | 0.633 | 0.68 | 0.32 | 0.35 | 0.47 | 2.81 | 1.00 | 0.96 | 0.82 | fail |
| 5 | A | 0.624 | 0.73 | 0.27 | 0.32 | 0.37 | 1.10 | 1.00 | 0.95 | 0.84 | fail |
| 6 | B | 0.636 | 0.71 | 0.29 | 0.31 | 0.42 | 2.71 | 1.00 | 0.96 | 0.85 | fail |
| 7 | A | 0.625 | 0.76 | 0.24 | 0.28 | 0.33 | 1.08 | 1.00 | 0.95 | 0.87 | fail |
| 8 | B | 0.640 | 0.74 | 0.26 | 0.27 | 0.42 | 1.17 | 1.00 | 0.96 | 0.84 | fail |
| 9 | A | 0.649 | 0.76 | 0.24 | 0.27 | 0.32 | 1.07 | 1.00 | 0.95 | 0.85 | fail |
| 10 | B | 0.614 | 0.71 | 0.29 | 0.31 | 0.40 | 1.21 | 1.00 | 0.94 | 0.84 | fail |
| 11 | A | 0.651 | 0.78 | 0.22 | 0.24 | 0.28 | 1.06 | 1.00 | 0.94 | 0.84 | fail |
| 12 | B | 0.653 | 0.72 | 0.28 | 0.29 | 0.38 | 1.12 | 1.00 | 0.93 | 0.80 | fail |

## Seed 203 — FAIL after 12 rollouts (393,216)
| Rollout | Set | Train capture | Val capture | Val exhaustion | Val repeated fly-by | Val >3× | Val median path-excess | Anchor 1 | Anchor 2 | Anchor 3 | Gate |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | A | 0.615 | 0.83 | 0.17 | 0.20 | 0.27 | 1.07 | 1.00 | 0.92 | 0.89 | fail |
| 2 | B | 0.667 | 0.74 | 0.26 | 0.31 | 0.38 | 1.12 | 1.00 | 0.92 | 0.90 | fail |
| 3 | A | 0.694 | 0.83 | 0.17 | 0.19 | 0.25 | 1.06 | 1.00 | 0.95 | 0.90 | fail |
| 4 | B | 0.655 | 0.76 | 0.24 | 0.28 | 0.35 | 1.12 | 1.00 | 0.94 | 0.90 | fail |
| 5 | A | 0.655 | 0.84 | 0.16 | 0.17 | 0.24 | 1.06 | 1.00 | 0.93 | 0.90 | fail |
| 6 | B | 0.640 | 0.78 | 0.22 | 0.27 | 0.33 | 1.12 | 1.00 | 0.92 | 0.87 | fail |
| 7 | A | 0.684 | 0.86 | 0.14 | 0.14 | 0.22 | 1.06 | 1.00 | 0.92 | 0.87 | fail |
| 8 | B | 0.670 | 0.80 | 0.20 | 0.25 | 0.31 | 1.08 | 1.00 | 0.91 | 0.84 | fail |
| 9 | A | 0.660 | 0.82 | 0.18 | 0.22 | 0.30 | 1.07 | 1.00 | 0.91 | 0.86 | fail |
| 10 | B | 0.688 | 0.80 | 0.20 | 0.23 | 0.29 | 1.08 | 1.00 | 0.90 | 0.86 | fail |
| 11 | A | 0.669 | 0.89 | 0.11 | 0.15 | 0.22 | 1.06 | 1.00 | 0.91 | 0.84 | fail |
| 12 | B | 0.686 | 0.79 | 0.21 | 0.24 | 0.30 | 1.09 | 1.00 | 0.90 | 0.82 | fail |
