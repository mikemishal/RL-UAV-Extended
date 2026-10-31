# Route-level A1 (`a1_route_action_v1`) — generated evidence tables

Specification SHA-256 `5cdb448a669f4519d2949ef90fc596a073b5600fe9da9cca0003f18ff711dd7d`; 48,000 episodes = 6 configurations (R, T_replan) x verified cells x 100 matched seeds (145000-145099) x 8 policies. **Coverage (staged sweep; RMPC and the remaining C1-C3 defenders were not run, see README):** {'C0': ['greedy', 'kalman_greedy', 'lead', 'pn'], 'C1': ['greedy', 'pn'], 'C2': ['greedy', 'pn'], 'C3': ['greedy', 'pn']}. DIRECT through the wrapper is the matched reference and is bit-identical to canonical A0 (flags: {'direct_equals_a0': True, 'deterministic_replay': True}); forbidden references in the controller module: none. Per-level pooling: C0 = 400 episodes per policy and configuration; C1-C3 = 200.

## 1. Gates per configuration

| R (m) | T_replan (s) | G1 collisions | G2 mission | G3 authority | G3 positive | G4 path | G5 perception | G6 stability | pass | gained collisions (7 policies) |
|---|---|---|---|---|---|---|---|---|---|---|
| 5.0 | 1.0 | FAIL | FAIL | PASS | PASS | FAIL | PASS | FAIL | FAIL | 673 |
| 5.0 | 2.0 | FAIL | FAIL | PASS | PASS | FAIL | PASS | FAIL | FAIL | 903 |
| 7.5 | 1.0 | FAIL | FAIL | PASS | PASS | FAIL | PASS | FAIL | FAIL | 821 |
| 7.5 | 2.0 | FAIL | FAIL | PASS | PASS | FAIL | PASS | FAIL | FAIL | 954 |
| 10.0 | 1.0 | FAIL | FAIL | PASS | PASS | FAIL | PASS | FAIL | FAIL | 990 |
| 10.0 | 2.0 | FAIL | FAIL | PASS | PASS | FAIL | PASS | FAIL | FAIL | 973 |

## 2. Gained obstacle collisions (DIRECT has 0 in every cell): per policy, summed over levels (C0 / C1 / C2 / C3 for the worst policy)

| R | T | LEFT | RIGHT | ABOVE | ALT_LR | RANDOM | MAX_SEP | ADVERSARIAL | worst policy by level C0/C1/C2/C3 |
|---|---|---|---|---|---|---|---|---|---|
| 5.0 | 1.0 | 131 | 130 | 140 | 45 | 104 | 18 | 105 | ABOVE: 0 / 12 / 35 / 93 |
| 5.0 | 2.0 | 150 | 146 | 139 | 149 | 161 | 31 | 127 | RANDOM: 0 / 13 / 60 / 88 |
| 7.5 | 1.0 | 152 | 161 | 153 | 103 | 130 | 15 | 107 | RIGHT: 0 / 16 / 48 / 97 |
| 7.5 | 2.0 | 171 | 168 | 144 | 165 | 161 | 18 | 127 | LEFT: 0 / 25 / 53 / 93 |
| 10.0 | 1.0 | 191 | 183 | 141 | 168 | 142 | 26 | 139 | LEFT: 0 / 25 / 67 / 99 |
| 10.0 | 2.0 | 181 | 166 | 140 | 179 | 155 | 28 | 124 | LEFT: 0 / 17 / 66 / 98 |

## 3. Tactical authority: conversions (DIRECT interception -> capture) / reversals (DIRECT capture -> any failure) / net captures (of 1,000 matched episodes per policy: C0 x 4 defenders + C1-C3 x 2 defenders; DIRECT captures 97 / 1,000 per configuration)

| R | T | LEFT | RIGHT | ABOVE | ALT_LR | RANDOM | MAX_SEP | ADVERSARIAL |
|---|---|---|---|---|---|---|---|---|
| 5.0 | 1.0 | 175 / 75 / +100 | 184 / 72 / +112 | 254 / 65 / +189 | 50 / 57 / -7 | 148 / 79 / +69 | 39 / 58 / -19 | 242 / 74 / +168 |
| 5.0 | 2.0 | 144 / 78 / +66 | 177 / 72 / +105 | 351 / 68 / +283 | 192 / 75 / +117 | 229 / 80 / +149 | 37 / 48 / -11 | 292 / 54 / +238 |
| 7.5 | 1.0 | 172 / 70 / +102 | 188 / 64 / +124 | 214 / 70 / +144 | 174 / 66 / +108 | 190 / 80 / +110 | 28 / 48 / -20 | 200 / 72 / +128 |
| 7.5 | 2.0 | 130 / 75 / +55 | 172 / 76 / +96 | 263 / 81 / +182 | 161 / 75 / +86 | 201 / 77 / +124 | 28 / 50 / -22 | 235 / 68 / +167 |
| 10.0 | 1.0 | 121 / 76 / +45 | 139 / 74 / +65 | 174 / 72 / +102 | 209 / 60 / +149 | 196 / 71 / +125 | 31 / 53 / -22 | 154 / 71 / +83 |
| 10.0 | 2.0 | 114 / 77 / +37 | 147 / 73 / +74 | 148 / 84 / +64 | 143 / 75 / +68 | 137 / 68 / +69 | 26 / 53 / -27 | 173 / 68 / +105 |

## 4. Mission gate G2: endurance exhaustion / timeouts (DIRECT: 0 / 0) and fly-by fraction

| R | T | LEFT | RIGHT | ABOVE | ALT_LR | RANDOM | MAX_SEP | ADVERSARIAL |
|---|---|---|---|---|---|---|---|---|
| 5.0 | 1.0 | 468 / 0 | 440 / 0 | 425 / 0 | 74 / 0 | 260 / 0 | 62 / 0 | 256 / 0 |
| 5.0 | 2.0 | 520 / 0 | 444 / 0 | 353 / 0 | 438 / 0 | 370 / 0 | 46 / 0 | 302 / 0 |
| 7.5 | 1.0 | 588 / 0 | 535 / 0 | 452 / 0 | 250 / 0 | 384 / 0 | 41 / 0 | 396 / 0 |
| 7.5 | 2.0 | 619 / 0 | 565 / 0 | 432 / 0 | 552 / 0 | 452 / 0 | 41 / 0 | 378 / 0 |
| 10.0 | 1.0 | 597 / 0 | 568 / 0 | 507 / 0 | 424 / 0 | 432 / 0 | 63 / 0 | 434 / 0 |
| 10.0 | 2.0 | 611 / 0 | 581 / 0 | 542 / 0 | 566 / 0 | 505 / 0 | 64 / 0 | 437 / 0 |

## 5. Path efficiency (tactical episodes; DIRECT max below); worst-case policy shown

| R | T | mean (range over policies) | median (range) | max | >1.3x | >2x | >5x | exhausted | timeouts |
|---|---|---|---|---|---|---|---|---|---|
| 5.0 | 1.0 | 1.34-3.64 | 1.00-3.59 | 9.6 | 3715 | 3487 | 1031 | 1985 | 0 |
| 5.0 | 2.0 | 1.30-3.58 | 1.00-3.79 | 8.8 | 4693 | 4401 | 1249 | 2473 | 0 |
| 7.5 | 1.0 | 1.23-3.97 | 1.00-3.97 | 17.5 | 4569 | 4322 | 1330 | 2646 | 0 |
| 7.5 | 2.0 | 1.23-4.00 | 1.00-4.05 | 10.0 | 5098 | 4829 | 1593 | 3039 | 0 |
| 10.0 | 1.0 | 1.37-3.90 | 1.00-4.03 | 10.9 | 5020 | 4699 | 1513 | 3025 | 0 |
| 10.0 | 2.0 | 1.37-4.08 | 1.00-4.23 | 10.8 | 5095 | 4815 | 1686 | 3306 | 0 |

DIRECT reference: path-excess max 1.06x, >1.3x: 0 of 6000 (all configurations share the same DIRECT episodes), >5x: 0, exhaustion 0, timeouts 0, captures 582, interceptions 5418.

## 6. Orbit / fly-by diagnostics (episodes; DIRECT in last row)

| R | T | orbit>=1 rev | episodes with >=1 fly-by overshoot (share of 7,000 tactical episodes per configuration) |
|---|---|---|---|
| 5.0 | 1.0 | 4 | 57.8% |
| 5.0 | 2.0 | 1 | 74.7% |
| 7.5 | 1.0 | 8 | 71.1% |
| 7.5 | 2.0 | 2 | 80.0% |
| 10.0 | 1.0 | 28 | 78.2% |
| 10.0 | 2.0 | 14 | 79.6% |
| DIRECT | - | 0 | 0.0% |

## 7. Perception audit G5 (planner calls pooled over the 7 tactical policies)

| R | T | planner calls | calls without defender information | obstacle ids outside the perceived set | calls with unperceived static obstacles | frozen-layer suppressions | commands executed while the layer owned steering |
|---|---|---|---|---|---|---|---|
| 5.0 | 1.0 | 28567 | 0 | 0 | 0 | 228 | 0 |
| 5.0 | 2.0 | 21073 | 0 | 0 | 0 | 267 | 0 |
| 7.5 | 1.0 | 32445 | 0 | 0 | 0 | 298 | 0 |
| 7.5 | 2.0 | 22221 | 0 | 0 | 0 | 365 | 0 |
| 10.0 | 1.0 | 34959 | 0 | 0 | 0 | 297 | 0 |
| 10.0 | 2.0 | 21542 | 0 | 0 | 0 | 290 | 0 |

## 8. Planner stability G6

| R | T | plans/episode | replans/episode | mean waypoints | mean route length (m) | failed-plan share | longest route run (steps) | max plans in 4 steps | non-finite | top rejection reasons |
|---|---|---|---|---|---|---|---|---|---|---|
| 5.0 | 1.0 | 4.08 | 1.21 | 2.0 | 19 | 29.7% | 19 | 6 | 0 | defender_not_ahead: 6714, no_path: 1211, ceiling: 561 |
| 5.0 | 2.0 | 3.01 | 0.29 | 2.1 | 22 | 35.3% | 16 | 6 | 0 | defender_not_ahead: 5843, no_path: 1179, ceiling: 407 |
| 7.5 | 1.0 | 4.64 | 1.40 | 2.1 | 23 | 29.7% | 22 | 6 | 0 | defender_not_ahead: 7558, no_path: 1477, ceiling: 602 |
| 7.5 | 2.0 | 3.17 | 0.29 | 2.1 | 25 | 37.1% | 19 | 6 | 0 | defender_not_ahead: 6542, no_path: 1287, ceiling: 424 |
| 10.0 | 1.0 | 4.99 | 1.41 | 2.1 | 26 | 30.7% | 18 | 6 | 0 | defender_not_ahead: 8258, no_path: 1917, ceiling: 538 |
| 10.0 | 2.0 | 3.08 | 0.27 | 2.1 | 29 | 35.9% | 17 | 6 | 0 | defender_not_ahead: 6031, no_path: 1273, ceiling: 437 |

## 9. Mechanisms (re-simulated gained collisions; >5x classification)

| config | gained (sampled) | gained-collision mechanism | >5x cases by outcome | exhaustions: DIRECT outcome |
|---|---|---|---|---|
| R10.0_T1.0 | 990 (40) | on a route step; frozen layer not owning the collision step: 1; on a route step; frozen layer owned the collision step: 2; route ended 1-3 steps earlier; frozen layer owned the collision step: 8; route ended >3 steps earlier; frozen layer owned the collision step: 29 | 1513: {'defender_static_obstacle_collision': 2, 'enemy_endurance_exhausted': 1325, 'enemy_static_obstacle_collision': 26, 'intercepted': 42, 'soldier_caught': 118} | {'intercepted': 2724, 'soldier_caught': 255, 'unsafe_intercept': 46} |
| R10.0_T2.0 | 973 (40) | on a route step; frozen layer owned the collision step: 4; route ended 1-3 steps earlier; frozen layer owned the collision step: 10; route ended >3 steps earlier; frozen layer owned the collision step: 26 | 1686: {'enemy_endurance_exhausted': 1477, 'enemy_static_obstacle_collision': 30, 'intercepted': 37, 'soldier_caught': 142} | {'intercepted': 2967, 'soldier_caught': 286, 'unsafe_intercept': 53} |
| R5.0_T1.0 | 673 (40) | on a route step; frozen layer owned the collision step: 2; route ended 1-3 steps earlier; frozen layer owned the collision step: 9; route ended >3 steps earlier; frozen layer owned the collision step: 29 | 1031: {'enemy_endurance_exhausted': 942, 'enemy_static_obstacle_collision': 7, 'intercepted': 8, 'soldier_caught': 74} | {'intercepted': 1723, 'soldier_caught': 234, 'unsafe_intercept': 28} |
| R5.0_T2.0 | 903 (40) | on a route step; frozen layer owned the collision step: 1; route ended 1-3 steps earlier; frozen layer not owning the collision step: 2; route ended 1-3 steps earlier; frozen layer owned the collision step: 4; route ended >3 steps earlier; frozen layer owned the collision step: 33 | 1249: {'defender_dynamic_obstacle_collision': 1, 'enemy_endurance_exhausted': 1092, 'enemy_static_obstacle_collision': 9, 'intercepted': 17, 'soldier_caught': 130} | {'intercepted': 2177, 'soldier_caught': 248, 'unsafe_intercept': 48} |
| R7.5_T1.0 | 821 (40) | on a route step; frozen layer owned the collision step: 4; route ended 1-3 steps earlier; frozen layer owned the collision step: 8; route ended >3 steps earlier; frozen layer owned the collision step: 28 | 1330: {'defender_dynamic_obstacle_collision': 1, 'enemy_endurance_exhausted': 1185, 'enemy_static_obstacle_collision': 7, 'intercepted': 33, 'soldier_caught': 104} | {'intercepted': 2352, 'soldier_caught': 247, 'unsafe_intercept': 47} |
| R7.5_T2.0 | 954 (40) | on a route step; frozen layer owned the collision step: 2; route ended 1-3 steps earlier; frozen layer owned the collision step: 4; route ended >3 steps earlier; frozen layer owned the collision step: 34 | 1593: {'enemy_endurance_exhausted': 1409, 'enemy_static_obstacle_collision': 19, 'intercepted': 36, 'soldier_caught': 129} | {'intercepted': 2720, 'soldier_caught': 273, 'unsafe_intercept': 46} |
