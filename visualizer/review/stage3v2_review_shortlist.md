# Stage3-v2 manual-review shortlist — Stage3-v2 REVIEW — PROVISIONAL CURRICULUM CHECKPOINT / NON-FINAL / NON-CANONICAL

Research commit `3a2b23f3d461490ab10cc9e1e9e032edf8b73eae`. All 15 replays verified against an independent research run (outcome, steps, attacker AND defender trajectory ≤ 0.001 m over every frame, release step, checkpoint SHA).

| # | Recording | Defender | Scenario | Outcome | Steps | Closest sep. | Path-excess | Open |
|---|---|---|---|---|---|---|---|---|
| 1 | Seed 203 — representative HIGH-QUALITY successful evasion of the moving defender | greedy | 154129 | soldier_caught | 11 | 5.82 m | 1.035 | `http://127.0.0.1:5173/?replay=553cbec4db3867c54f7b309ee95f7660` |
| 2 | Seed 203 — CLOSEST successful defender pass | greedy | 154140 | soldier_caught | 13 | 3.31 m | 1.083 | `http://127.0.0.1:5173/?replay=e7f9607bf6f2e9e7ee514417b63536ce` |
| 3 | Seed 203 — INTERCEPTION failure (longest attempted evasion) | greedy | 154087 | intercepted | 7 | 2.50 m | 1.119 | `http://127.0.0.1:5173/?replay=0511b9a8e00ce623c3c613f1752c5627` |
| 4 | Matched canonical A0 reference (same scenario as the seed-203 high-quality success, 154129) | greedy | 154129 | intercepted | 6 | 2.50 m | 1.000 | `http://127.0.0.1:5173/?replay=352ebf2dc8f1a327a92aa252f572773c` |
| 5 | Seed 201 — representative HIGH-QUALITY successful evasion of the moving defender | greedy | 154139 | soldier_caught | 12 | 7.13 m | 1.030 | `http://127.0.0.1:5173/?replay=d8df7318de4b375d72fcd054fb435099` |
| 6 | Seed 201 — CLOSEST successful defender pass | greedy | 154142 | soldier_caught | 15 | 3.55 m | 1.152 | `http://127.0.0.1:5173/?replay=d1b766eecf379d7174a77becc9287c0e` |
| 7 | Seed 201 — INTERCEPTION failure (longest attempted evasion) | greedy | 154047 | intercepted | 8 | 2.50 m | 1.001 | `http://127.0.0.1:5173/?replay=4770861aa0d321428d12d8bdb2c1a49d` |
| 8 | Seed 201 — VERTICAL evasion (largest altitude change) | greedy | 154182 | soldier_caught | 14 | 6.44 m | 1.076 | `http://127.0.0.1:5173/?replay=88a23680ae13deae3a574881770c7a22` |
| 9 | Seed 202 — CLOSEST successful defender pass | greedy | 154151 | soldier_caught | 12 | 3.42 m | 1.135 | `http://127.0.0.1:5173/?replay=97fd28dc9bc9aa636c129c583f385a59` |
| 10 | Seed 202 — INTERCEPTION failure (longest attempted evasion) | greedy | 154087 | intercepted | 8 | 2.50 m | 1.129 | `http://127.0.0.1:5173/?replay=f64b1e0fc9d8d663aa147c8e65be12bf` |
| 11 | Seed 202 — representative successful capture | greedy | 154014 | soldier_caught | 14 | 6.06 m | 1.061 | `http://127.0.0.1:5173/?replay=286eb527f935725a41af2ce5bd80ec45` |
| 12 | Seed 203 — LATERAL evasion (largest horizontal detour) | greedy | 154072 | soldier_caught | 14 | 5.70 m | 1.072 | `http://127.0.0.1:5173/?replay=62b4f97147bfe01d72e2cf61abec18d5` |
| 13 | Seed 203 — MIXED lateral + vertical evasion (largest displacement) | greedy | 154182 | soldier_caught | 13 | 5.26 m | 1.086 | `http://127.0.0.1:5173/?replay=8c3ed1376a284185c0a0abadca70e992` |
| 14 | Seed 203 vs LEAD (never trained against) — captures where the Stage2-v2 parent is intercepted | lead | 156018 | soldier_caught | 13 | 4.79 m | 1.030 | `http://127.0.0.1:5173/?replay=531bdf15a56eb982cf9e6021a5a208d6` |
| 15 | Seed 203 vs PN (never trained against) — captures where the Stage2-v2 parent is intercepted | pn | 156095 | soldier_caught | 11 | 4.77 m | 1.036 | `http://127.0.0.1:5173/?replay=39ca1dd0f48d5debeb0de1f7870b9b61` |

## What to look for (all recordings)

1. Does A1 fly toward the asset normally before the defender becomes visible (inside the 25 m information gate) — no early swerving?
2. Does it deviate only AFTER defender information becomes available (gate entry / detection), and does the deviation start promptly?
3. Is the evasion smooth rather than random or wandering?
4. Does it create enough separation to avoid the defender (interception radius 2.5 m) while the defender accelerates toward it after release?
5. After clearing the defender, does it turn back toward the asset?
6. Does it capture on the first reasonable approach?
7. Any unnecessary loops, oscillations, fly-bys or excessive detours? (Note the defender keeps chasing after the pass — does the attacker stay ahead of it?)
