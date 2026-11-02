# Stage4 per-rollout validation results

Moving canonical asset (`legacy_gaussian_random_walk`) + hold-until-detection Greedy, C0. Alternating validation: odd rollouts on set A (`165000–165099`), even rollouts on set B (`165100–165199`). Matched canonical A0 capture is 0 % on every set (interception 100 %), so the improvement-over-A0 gate (≥ +20 pp) equals the capture and is met whenever capture ≥ 0.20.

Frozen gate (all of): capture ≥ 0.70, improvement over A0 ≥ +20 pp, interception ≤ 0.30, timeout ≤ 0.03, exhaustion ≤ 0.03, median path-excess ≤ 2.00, >3× ≤ 0.05, >5× = 0, no persistent orbit, anchors ≥ 0.98 (Stage-1 basic pursuit) / ≥ 0.80 (Stage-2 static defender) / ≥ 0.75 (Stage-3 moving defender), no non-finite values. Mastery = two consecutive passing rollouts. Retention stop = two consecutive failures of one anchor (never triggered). Timeout is 0 at every rollout. The gate fails at every rollout of every seed on capture, exhaustion, >3× and >5× path-excess (>3× / >5× columns are omitted here; see the seed manifests).

Held-out results of the final (rollout-16) checkpoints are in `README.md` and `final_evaluation.json`.

## Seed 201 — FAIL after 16 rollouts (524,288 Stage-4 timesteps)
| Rollout | Set | Train capture | Val capture | Val interception | Val exhaustion | Val median path-excess | Anchor 1 | Anchor 2 | Anchor 3 | Gate |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | A | 0.502 | 0.55 | 0.23 | 0.22 | 1.08 | 1.00 | 0.94 | 0.87 | fail |
| 2 | B | 0.473 | 0.47 | 0.26 | 0.27 | 1.14 | 1.00 | 0.95 | 0.88 | fail |
| 3 | A | 0.508 | 0.59 | 0.22 | 0.19 | 1.08 | 1.00 | 0.95 | 0.87 | fail |
| 4 | B | 0.534 | 0.43 | 0.26 | 0.31 | 1.16 | 1.00 | 0.96 | 0.87 | fail |
| 5 | A | 0.491 | 0.54 | 0.25 | 0.21 | 1.09 | 1.00 | 0.95 | 0.88 | fail |
| 6 | B | 0.511 | 0.40 | 0.27 | 0.33 | 1.18 | 1.00 | 0.95 | 0.89 | fail |
| 7 | A | 0.510 | 0.50 | 0.24 | 0.26 | 1.09 | 1.00 | 0.95 | 0.87 | fail |
| 8 | B | 0.482 | 0.44 | 0.30 | 0.26 | 1.15 | 1.00 | 0.94 | 0.90 | fail |
| 9 | A | 0.469 | 0.54 | 0.22 | 0.24 | 1.10 | 1.00 | 0.95 | 0.90 | fail |
| 10 | B | 0.473 | 0.45 | 0.30 | 0.25 | 1.13 | 1.00 | 0.95 | 0.91 | fail |
| 11 | A | 0.495 | 0.57 | 0.23 | 0.20 | 1.09 | 1.00 | 0.94 | 0.89 | fail |
| 12 | B | 0.533 | 0.49 | 0.27 | 0.24 | 1.16 | 1.00 | 0.88 | 0.88 | fail |
| 13 | A | 0.541 | 0.54 | 0.25 | 0.21 | 1.08 | 1.00 | 0.91 | 0.87 | fail |
| 14 | B | 0.542 | 0.50 | 0.24 | 0.26 | 1.14 | 1.00 | 0.91 | 0.87 | fail |
| 15 | A | 0.528 | 0.55 | 0.22 | 0.23 | 1.08 | 1.00 | 0.93 | 0.88 | fail |
| 16 | B | 0.557 | 0.52 | 0.26 | 0.22 | 1.13 | 0.99 | 0.94 | 0.87 | fail |

## Seed 202 — FAIL after 16 rollouts (524,288)
| Rollout | Set | Train capture | Val capture | Val interception | Val exhaustion | Val median path-excess | Anchor 1 | Anchor 2 | Anchor 3 | Gate |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | A | 0.491 | 0.45 | 0.34 | 0.21 | 1.09 | 1.00 | 0.94 | 0.86 | fail |
| 2 | B | 0.496 | 0.42 | 0.32 | 0.26 | 1.14 | 1.00 | 0.96 | 0.87 | fail |
| 3 | A | 0.517 | 0.51 | 0.30 | 0.19 | 1.09 | 1.00 | 0.97 | 0.88 | fail |
| 4 | B | 0.476 | 0.44 | 0.36 | 0.20 | 1.09 | 1.00 | 0.95 | 0.88 | fail |
| 5 | A | 0.520 | 0.58 | 0.28 | 0.14 | 1.07 | 1.00 | 0.95 | 0.87 | fail |
| 6 | B | 0.514 | 0.43 | 0.35 | 0.22 | 1.10 | 1.00 | 0.95 | 0.86 | fail |
| 7 | A | 0.531 | 0.61 | 0.29 | 0.10 | 1.07 | 1.00 | 0.95 | 0.87 | fail |
| 8 | B | 0.527 | 0.44 | 0.43 | 0.13 | 1.09 | 1.00 | 0.96 | 0.89 | fail |
| 9 | A | 0.549 | 0.58 | 0.27 | 0.15 | 1.07 | 1.00 | 0.98 | 0.91 | fail |
| 10 | B | 0.558 | 0.50 | 0.32 | 0.18 | 1.09 | 1.00 | 0.98 | 0.90 | fail |
| 11 | A | 0.581 | 0.60 | 0.31 | 0.09 | 1.09 | 1.00 | 0.96 | 0.90 | fail |
| 12 | B | 0.571 | 0.48 | 0.36 | 0.16 | 1.09 | 1.00 | 0.95 | 0.89 | fail |
| 13 | A | 0.577 | 0.60 | 0.26 | 0.14 | 1.09 | 1.00 | 0.98 | 0.90 | fail |
| 14 | B | 0.560 | 0.52 | 0.33 | 0.15 | 1.09 | 1.00 | 0.94 | 0.91 | fail |
| 15 | A | 0.576 | 0.67 | 0.20 | 0.13 | 1.09 | 1.00 | 0.95 | 0.91 | fail |
| 16 | B | 0.554 | 0.56 | 0.24 | 0.20 | 1.13 | 1.00 | 0.96 | 0.93 | fail |

## Seed 203 — FAIL after 16 rollouts (524,288)
| Rollout | Set | Train capture | Val capture | Val interception | Val exhaustion | Val median path-excess | Anchor 1 | Anchor 2 | Anchor 3 | Gate |
|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| 1 | A | 0.504 | 0.56 | 0.23 | 0.21 | 1.08 | 1.00 | 0.97 | 0.87 | fail |
| 2 | B | 0.518 | 0.56 | 0.19 | 0.25 | 1.12 | 1.00 | 0.97 | 0.88 | fail |
| 3 | A | 0.538 | 0.55 | 0.28 | 0.17 | 1.07 | 1.00 | 0.97 | 0.88 | fail |
| 4 | B | 0.515 | 0.51 | 0.25 | 0.24 | 1.13 | 1.00 | 0.97 | 0.89 | fail |
| 5 | A | 0.501 | 0.56 | 0.22 | 0.22 | 1.09 | 1.00 | 0.97 | 0.89 | fail |
| 6 | B | 0.546 | 0.45 | 0.21 | 0.34 | 1.20 | 1.00 | 0.96 | 0.89 | fail |
| 7 | A | 0.541 | 0.57 | 0.25 | 0.18 | 1.09 | 1.00 | 0.95 | 0.89 | fail |
| 8 | B | 0.511 | 0.59 | 0.19 | 0.22 | 1.15 | 1.00 | 0.97 | 0.89 | fail |
| 9 | A | 0.551 | 0.60 | 0.22 | 0.18 | 1.09 | 1.00 | 0.98 | 0.91 | fail |
| 10 | B | 0.521 | 0.54 | 0.18 | 0.28 | 1.16 | 1.00 | 0.98 | 0.92 | fail |
| 11 | A | 0.525 | 0.57 | 0.25 | 0.18 | 1.09 | 1.00 | 0.97 | 0.92 | fail |
| 12 | B | 0.519 | 0.52 | 0.18 | 0.30 | 1.20 | 1.00 | 0.97 | 0.93 | fail |
| 13 | A | 0.538 | 0.59 | 0.21 | 0.20 | 1.09 | 1.00 | 0.96 | 0.91 | fail |
| 14 | B | 0.531 | 0.58 | 0.14 | 0.28 | 1.19 | 1.00 | 0.95 | 0.93 | fail |
| 15 | A | 0.550 | 0.63 | 0.17 | 0.20 | 1.09 | 1.00 | 0.97 | 0.92 | fail |
| 16 | B | 0.565 | 0.56 | 0.18 | 0.26 | 1.22 | 1.00 | 0.97 | 0.92 | fail |
