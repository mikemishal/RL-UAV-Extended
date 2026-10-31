# Finite tactical-pulse geometry sweep — generated evidence tables

Specification SHA-256 `daffa22c31cd0c03c1b349d974296ecb0f9e2332bc7533dc7f77db1348a33386`; 204,800 episode records (16 candidates x 4 levels x 4 contexts x 8 policies x 100 seeds).
BASE here is the forced-BASE policy through the same state machine (bit-identical to canonical A0, see `base_equivalence.json`). 'Gained' = tactical collision where the matched BASE episode (same seed/context) did not collide.

## 1. Gates per candidate

| theta | T (s) | steps | G1 collisions | G2 mission | G3 authority | G4 path | G5 machine | pass |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.5 | 1 | FAIL | FAIL | FAIL | PASS | PASS | FAIL |
| 10 | 1.0 | 2 | FAIL | FAIL | FAIL | PASS | PASS | FAIL |
| 10 | 1.5 | 3 | FAIL | FAIL | PASS | PASS | PASS | FAIL |
| 10 | 2.0 | 4 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 15 | 0.5 | 1 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 15 | 1.0 | 2 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 15 | 1.5 | 3 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 15 | 2.0 | 4 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 20 | 0.5 | 1 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 20 | 1.0 | 2 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 20 | 1.5 | 3 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 20 | 2.0 | 4 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 30 | 0.5 | 1 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 30 | 1.0 | 2 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 30 | 1.5 | 3 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |
| 30 | 2.0 | 4 | FAIL | FAIL | PASS | FAIL | PASS | FAIL |

## 2. Gained obstacle collisions (matched BASE collisions are 0 in every cell)

Active = real fixed defenders (greedy/lead/pn; 7 tactical policies x 3 defenders x 100 seeds = 2,100 episodes per level). Passive-forced = mission-only stress diagnostic (parked defender, gate forced open).

| theta | T (s) | active C0 | active C1 | active C2 | active C3 | passive C0 | passive C1 | passive C2 | passive C3 | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.5 | 0 | 2 | 15 | 24 | 0 | 2 | 4 | 15 | 62 |
| 10 | 1.0 | 0 | 0 | 25 | 31 | 0 | 3 | 15 | 31 | 105 |
| 10 | 1.5 | 0 | 0 | 12 | 24 | 0 | 1 | 7 | 23 | 67 |
| 10 | 2.0 | 0 | 0 | 15 | 34 | 0 | 2 | 8 | 28 | 87 |
| 15 | 0.5 | 0 | 2 | 31 | 51 | 0 | 6 | 40 | 64 | 194 |
| 15 | 1.0 | 0 | 0 | 25 | 55 | 0 | 8 | 47 | 90 | 225 |
| 15 | 1.5 | 0 | 5 | 34 | 132 | 0 | 5 | 38 | 87 | 301 |
| 15 | 2.0 | 0 | 20 | 65 | 141 | 0 | 22 | 71 | 121 | 440 |
| 20 | 0.5 | 0 | 1 | 45 | 92 | 0 | 12 | 49 | 109 | 308 |
| 20 | 1.0 | 0 | 7 | 96 | 231 | 0 | 25 | 142 | 297 | 798 |
| 20 | 1.5 | 0 | 31 | 176 | 527 | 0 | 20 | 121 | 270 | 1145 |
| 20 | 2.0 | 0 | 41 | 245 | 581 | 0 | 29 | 183 | 368 | 1447 |
| 30 | 0.5 | 0 | 0 | 56 | 110 | 0 | 12 | 60 | 129 | 367 |
| 30 | 1.0 | 0 | 95 | 515 | 945 | 0 | 38 | 213 | 412 | 2218 |
| 30 | 1.5 | 0 | 144 | 501 | 1007 | 0 | 40 | 210 | 398 | 2300 |
| 30 | 2.0 | 0 | 154 | 589 | 1026 | 0 | 42 | 216 | 420 | 2447 |

## 3. Mechanism of the gained collisions (re-simulated; <= 400 episodes per candidate)

| theta | T (s) | classified | on a tactical pulse step | frozen common layer owned steering (post-maneuver state) | other |
|---|---|---|---|---|---|
| 10 | 0.5 | 62 | 20 | 42 | 0 |
| 10 | 1.0 | 105 | 27 | 78 | 0 |
| 10 | 1.5 | 67 | 30 | 37 | 0 |
| 10 | 2.0 | 87 | 30 | 57 | 0 |
| 15 | 0.5 | 194 | 21 | 173 | 0 |
| 15 | 1.0 | 225 | 32 | 193 | 0 |
| 15 | 1.5 | 301 | 42 | 259 | 0 |
| 15 | 2.0 | 400 | 49 | 351 | 0 |
| 20 | 0.5 | 308 | 23 | 285 | 0 |
| 20 | 1.0 | 400 | 32 | 368 | 0 |
| 20 | 1.5 | 400 | 30 | 367 | 3 |
| 20 | 2.0 | 400 | 22 | 377 | 1 |
| 30 | 0.5 | 367 | 27 | 340 | 0 |
| 30 | 1.0 | 400 | 19 | 381 | 0 |
| 30 | 1.5 | 400 | 19 | 381 | 0 |
| 30 | 2.0 | 400 | 15 | 385 | 0 |

## 4. Mission gate G2 — tactical vs matched BASE episode counts (summed over policies, contexts, levels)

| theta | T (s) | exhausted (tact / base) | timeout (tact / base) | orbit>=1 rev (tact / base) | ceiling hover | sustained hover | cells failing G2 |
|---|---|---|---|---|---|---|---|
| 10 | 0.5 | 87 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 10 | 1.0 | 136 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 10 | 1.5 | 57 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 10 | 2.0 | 238 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 15 | 0.5 | 199 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 15 | 1.0 | 395 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 15 | 1.5 | 610 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 15 | 2.0 | 977 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 20 | 0.5 | 419 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 20 | 1.0 | 1683 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 20 | 1.5 | 2133 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 20 | 2.0 | 2885 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 30 | 0.5 | 581 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 30 | 1.0 | 5356 / 7 | 0 / 0 | 5 / 0 | 0 | 0 | 12+ |
| 30 | 1.5 | 5328 / 7 | 0 / 0 | 0 / 0 | 0 | 0 | 12+ |
| 30 | 2.0 | 5909 / 7 | 0 / 0 | 4 / 0 | 0 | 0 | 12+ |

## 5. Path efficiency G4 (tactical episodes)

| theta | T (s) | mean | median | max | >1.3x | >2x | >5x | >5x fraction (BASE fraction) |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.5 | 1.07 | 1.00 | 8.0 | 255 | 225 | 18 | 0.0014 (0.0000) |
| 10 | 1.0 | 1.07 | 1.00 | 7.9 | 253 | 224 | 32 | 0.0025 (0.0000) |
| 10 | 1.5 | 1.04 | 1.00 | 7.9 | 137 | 122 | 19 | 0.0015 (0.0000) |
| 10 | 2.0 | 1.12 | 1.00 | 7.9 | 376 | 346 | 93 | 0.0073 (0.0000) |
| 15 | 0.5 | 1.15 | 1.01 | 8.7 | 552 | 472 | 102 | 0.0080 (0.0000) |
| 15 | 1.0 | 1.23 | 1.01 | 9.2 | 779 | 722 | 215 | 0.0168 (0.0000) |
| 15 | 1.5 | 1.34 | 1.01 | 19.1 | 1149 | 1065 | 299 | 0.0234 (0.0000) |
| 15 | 2.0 | 1.53 | 1.01 | 11.7 | 1749 | 1619 | 516 | 0.0403 (0.0000) |
| 20 | 0.5 | 1.26 | 1.01 | 8.6 | 956 | 826 | 225 | 0.0176 (0.0000) |
| 20 | 1.0 | 1.91 | 1.01 | 14.3 | 3128 | 2924 | 899 | 0.0702 (0.0000) |
| 20 | 1.5 | 2.19 | 1.03 | 11.0 | 4069 | 3777 | 1116 | 0.0872 (0.0000) |
| 20 | 2.0 | 2.59 | 1.13 | 11.9 | 5360 | 4978 | 1645 | 0.1285 (0.0000) |
| 30 | 0.5 | 1.35 | 1.01 | 10.9 | 1220 | 1071 | 295 | 0.0230 (0.0000) |
| 30 | 1.0 | 3.82 | 3.97 | 15.9 | 9074 | 8423 | 3119 | 0.2437 (0.0000) |
| 30 | 1.5 | 3.86 | 3.91 | 16.4 | 9288 | 8640 | 3111 | 0.2430 (0.0000) |
| 30 | 2.0 | 4.02 | 4.13 | 13.0 | 9684 | 9008 | 3354 | 0.2620 (0.0000) |

## 6. Tactical authority G3 (best active-defender policy; share of 400 matched episodes diverging >= 2 m from BASE)

| theta | T (s) | best context / policy | diverging fraction | outcome differs | mean min-separation difference (m) | G3 |
|---|---|---|---|---|---|---|
| 10 | 0.5 | greedy / ALT_LR | 0.05 | 26 | +0.00 | FAIL |
| 10 | 1.0 | pn / RANDOM | 0.08 | 39 | +0.09 | FAIL |
| 10 | 1.5 | lead / ALT_LR | 0.23 | 36 | +0.03 | PASS |
| 10 | 2.0 | lead / CYCLE_LCR | 0.23 | 46 | +0.10 | PASS |
| 15 | 0.5 | greedy / RANDOM | 0.14 | 38 | +0.11 | PASS |
| 15 | 1.0 | lead / LEFT | 0.99 | 51 | +0.12 | PASS |
| 15 | 1.5 | lead / LEFT | 0.99 | 73 | +0.21 | PASS |
| 15 | 2.0 | lead / LEFT | 0.99 | 84 | +0.26 | PASS |
| 20 | 0.5 | greedy / LEFT | 0.29 | 46 | +0.04 | PASS |
| 20 | 1.0 | lead / LEFT | 0.99 | 63 | +0.14 | PASS |
| 20 | 1.5 | lead / LEFT | 0.99 | 99 | +0.38 | PASS |
| 20 | 2.0 | lead / LEFT | 0.99 | 118 | +0.46 | PASS |
| 30 | 0.5 | greedy / LEFT | 0.31 | 50 | +0.06 | PASS |
| 30 | 1.0 | lead / LEFT | 0.99 | 347 | +1.51 | PASS |
| 30 | 1.5 | lead / LEFT | 0.99 | 376 | +1.87 | PASS |
| 30 | 2.0 | lead / LEFT | 0.99 | 382 | +1.98 | PASS |

## 7. State machine G5

| theta | T (s) | pulses started | cancelled by obstacle | aborted (geometry) | machine violations | longest tactical run (steps) | deterministic replay |
|---|---|---|---|---|---|---|---|
| 10 | 0.5 | 24105 | 0 | 0 | 0 | 1 (<= 1) | True |
| 10 | 1.0 | 18371 | 78 | 0 | 0 | 2 (<= 2) | True |
| 10 | 1.5 | 15821 | 88 | 0 | 0 | 3 (<= 3) | True |
| 10 | 2.0 | 15464 | 151 | 0 | 0 | 4 (<= 4) | True |
| 15 | 0.5 | 26793 | 0 | 0 | 0 | 1 (<= 1) | True |
| 15 | 1.0 | 22829 | 153 | 0 | 0 | 2 (<= 2) | True |
| 15 | 1.5 | 20811 | 238 | 0 | 0 | 3 (<= 3) | True |
| 15 | 2.0 | 21496 | 443 | 0 | 0 | 4 (<= 4) | True |
| 20 | 0.5 | 30535 | 0 | 0 | 0 | 1 (<= 1) | True |
| 20 | 1.0 | 37215 | 352 | 0 | 0 | 2 (<= 2) | True |
| 20 | 1.5 | 32776 | 571 | 0 | 0 | 3 (<= 3) | True |
| 20 | 2.0 | 34848 | 1126 | 0 | 0 | 4 (<= 4) | True |
| 30 | 0.5 | 32653 | 0 | 0 | 0 | 1 (<= 1) | True |
| 30 | 1.0 | 59109 | 701 | 0 | 0 | 2 (<= 2) | True |
| 30 | 1.5 | 50568 | 1287 | 0 | 0 | 3 (<= 3) | True |
| 30 | 2.0 | 46411 | 1719 | 0 | 0 | 4 (<= 4) | True |

## 8. Supplementary: active defenders only (real 25 m information gate; passive-forced diagnostic excluded)

Mission counts are over 8,400 matched tactical episodes per candidate (7 policies x 3 defenders x 4 levels x 100 seeds). 'Defender-relevant flips' count only episodes whose outcome changed between two defender-attributable terminal outcomes (attacker capture vs defender interception/timeout); collisions and endurance exhaustion are NOT credited as authority.

| theta | T (s) | gained collisions | exhausted (tact / base) | tactical pulses / episode | flips intercepted -> captured | flips captured -> intercepted | mean min-separation diff (m) | share with any pulse |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.5 | 41 | 48 / 0 | 1.54 | 207 | 174 | +0.042 | 1.00 |
| 10 | 1.0 | 56 | 45 / 0 | 1.05 | 261 | 271 | +0.047 | 1.00 |
| 10 | 1.5 | 36 | 20 / 0 | 1.02 | 371 | 262 | +0.073 | 1.00 |
| 10 | 2.0 | 49 | 139 / 0 | 1.05 | 431 | 271 | +0.087 | 1.00 |
| 15 | 0.5 | 84 | 69 / 0 | 1.59 | 261 | 221 | +0.056 | 1.00 |
| 15 | 1.0 | 80 | 55 / 0 | 1.08 | 582 | 350 | +0.106 | 1.00 |
| 15 | 1.5 | 171 | 313 / 0 | 1.23 | 905 | 320 | +0.236 | 1.00 |
| 15 | 2.0 | 226 | 422 / 0 | 1.25 | 960 | 298 | +0.285 | 1.00 |
| 20 | 0.5 | 138 | 137 / 0 | 1.69 | 284 | 250 | +0.080 | 1.00 |
| 20 | 1.0 | 334 | 590 / 0 | 1.53 | 997 | 343 | +0.338 | 1.00 |
| 20 | 1.5 | 734 | 1275 / 0 | 1.89 | 1022 | 246 | +0.615 | 1.00 |
| 20 | 2.0 | 867 | 1388 / 0 | 1.84 | 1069 | 239 | +0.695 | 1.00 |
| 30 | 0.5 | 166 | 198 / 0 | 1.78 | 307 | 250 | +0.113 | 1.00 |
| 30 | 1.0 | 1555 | 3621 / 0 | 3.27 | 1636 | 103 | +1.777 | 1.00 |
| 30 | 1.5 | 1652 | 3676 / 0 | 3.08 | 1827 | 81 | +2.110 | 1.00 |
| 30 | 2.0 | 1769 | 3897 / 0 | 2.81 | 1685 | 72 | +2.235 | 1.00 |
