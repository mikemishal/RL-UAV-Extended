# Stage3-v2 per-rollout validation results

Alternating validation: odd rollouts on set A (`154000–154099`), even on B (`154100–154199`); matched A0 capture is 0 % on every set (A0 interception 100 %), so the improvement-over-A0 gate (≥ +20 pp) equals the capture. Gate (all of): capture ≥ 0.80, interception ≤ 0.20, timeout ≤ 0.02, exhaustion ≤ 0.02, median path-excess ≤ 1.80, >3× ≤ 0.05, >5× = 0, no persistent orbit, Stage-1 anchor ≥ 0.98, Stage-2 static anchor ≥ 0.80, no non-finite. Mastery = two consecutive passing rollouts. All timeout / exhaustion / >3× / >5× / orbit / non-finite entries are 0 and median path-excess is 1.04–1.06 at every rollout.

## Seed 201 — MASTERED at rollout 3 (98,304 Stage-3 timesteps)
| Rollout | Set | Train capture | Val capture | Val interception | Stage-1 anchor | Static anchor | Gate |
|---:|:--:|---:|---:|---:|---:|---:|:--:|
| 1 | A | 0.672 | 0.77 | 0.23 | 1.00 | 0.84 | fail |
| 2 | B | 0.719 | 0.82 | 0.18 | 1.00 | 0.86 | **pass** |
| 3 | A | 0.766 | 0.81 | 0.19 | 1.00 | 0.90 | **pass** |

## Seed 202 — MASTERED at rollout 5 (163,840)
| Rollout | Set | Train capture | Val capture | Val interception | Stage-1 anchor | Static anchor | Gate |
|---:|:--:|---:|---:|---:|---:|---:|:--:|
| 1 | A | 0.662 | 0.73 | 0.27 | 1.00 | 0.91 | fail |
| 2 | B | 0.675 | 0.81 | 0.19 | 1.00 | 0.92 | pass |
| 3 | A | 0.721 | 0.77 | 0.23 | 1.00 | 0.94 | fail |
| 4 | B | 0.759 | 0.85 | 0.15 | 1.00 | 0.94 | **pass** |
| 5 | A | 0.789 | 0.80 | 0.20 | 1.00 | 0.94 | **pass** |

## Seed 203 — MASTERED at rollout 7 (229,376)
| Rollout | Set | Train capture | Val capture | Val interception | Stage-1 anchor | Static anchor | Gate |
|---:|:--:|---:|---:|---:|---:|---:|:--:|
| 1 | A | 0.676 | 0.72 | 0.28 | 1.00 | 0.90 | fail |
| 2 | B | 0.688 | 0.87 | 0.13 | 1.00 | 0.93 | pass |
| 3 | A | 0.738 | 0.76 | 0.24 | 1.00 | 0.94 | fail |
| 4 | B | 0.749 | 0.91 | 0.09 | 1.00 | 0.93 | pass |
| 5 | A | 0.793 | 0.78 | 0.22 | 1.00 | 0.94 | fail |
| 6 | B | 0.791 | 0.90 | 0.10 | 1.00 | 0.95 | pass |
| 7 | A | 0.787 | 0.84 | 0.16 | 1.00 | 0.95 | **pass** |

Note that set B scored higher than the adjacent set A in most pairs (e.g. seed 203: 0.87 vs 0.72/0.76, 0.91 vs 0.78): the 100-scenario validation sets are not exchangeable at this precision, which is why mastery requires two consecutive passes across *alternating* sets and why the 200-episode held-out evaluation (A ∪ B) is the confirmation.
