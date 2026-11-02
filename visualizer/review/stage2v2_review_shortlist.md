# Stage2-v2 manual-review shortlist — Stage2-v2 REVIEW — NOT MASTERED / NON-CANONICAL

Research commit `1214c3807ca9adfc5aa95b01f2a0e1d5a7699526`. All 13 replays verified against an independent research run (outcome, steps, trajectory ≤ 0.001 m, defender position, checkpoint SHA).

| # | Recording | Seed | Scenario | Outcome | Steps | Closest sep. | Path-excess | Open |
|---|---|---|---|---|---|---|---|---|
| 1 | Seed 201 — representative HIGH-QUALITY successful evasion | 201 | 147149 | soldier_caught | 10 | 3.25 m | 1.020 | `http://127.0.0.1:5173/?replay=ff5a2c388c131b2a0a1856a5155c3bf0` |
| 2 | Seed 203 — representative HIGH-QUALITY successful evasion | 203 | 147145 | soldier_caught | 11 | 5.51 m | 1.019 | `http://127.0.0.1:5173/?replay=93844ff11116e7ead6abf89c885f0e52` |
| 3 | Seed 203 — INTERCEPTION failure (largest attempted evasion) | 203 | 147007 | intercepted | 6 | 2.50 m | 1.068 | `http://127.0.0.1:5173/?replay=7da7770775975b634ce04d2ef50ec9d8` |
| 4 | Matched canonical A0 reference (same scenario as the seed-201 high-quality success, 147149) | A0 | 147149 | intercepted | 5 | 2.50 m | 1.025 | `http://127.0.0.1:5173/?replay=7e6cced512d8d2c1484db16b14204082` |
| 5 | Seed 201 — CLOSEST successful defender pass | 201 | 147038 | soldier_caught | 10 | 2.88 m | 1.038 | `http://127.0.0.1:5173/?replay=4a05c88ea4334e6036dd2545412a6d45` |
| 6 | Seed 201 — INTERCEPTION failure (largest attempted evasion) | 201 | 147094 | intercepted | 6 | 2.50 m | 1.079 | `http://127.0.0.1:5173/?replay=eb6c1337bcc720f952fa695180bae9b5` |
| 7 | Seed 201 — LARGEST successful detour | 201 | 147071 | soldier_caught | 15 | 7.93 m | 1.203 | `http://127.0.0.1:5173/?replay=4a80a7cbe4ec0670b4eb63bd4cde5c5a` |
| 8 | Seed 202 — CLOSEST successful defender pass | 202 | 147061 | soldier_caught | 10 | 2.70 m | 1.041 | `http://127.0.0.1:5173/?replay=bb706f8281df1e870a2739a326cd40ad` |
| 9 | Seed 202 — INTERCEPTION failure (largest attempted evasion) | 202 | 147161 | intercepted | 7 | 2.50 m | 1.079 | `http://127.0.0.1:5173/?replay=5f872d8d1d3ba6e8a2d34af6f30dfc68` |
| 10 | Seed 202 — LARGEST successful detour | 202 | 147050 | soldier_caught | 13 | 4.18 m | 1.253 | `http://127.0.0.1:5173/?replay=0ed25a62acbec8b300b8471b475b035d` |
| 11 | Seed 202 — representative successful capture | 202 | 147171 | soldier_caught | 9 | 5.37 m | 1.034 | `http://127.0.0.1:5173/?replay=27025811823fdbd032eecd852e26fd34` |
| 12 | Seed 203 — CLOSEST successful defender pass | 203 | 147118 | soldier_caught | 11 | 2.77 m | 1.046 | `http://127.0.0.1:5173/?replay=1b8e093051610586c392905aa967c77d` |
| 13 | Seed 203 — LARGEST successful detour | 203 | 147071 | soldier_caught | 16 | 9.31 m | 1.211 | `http://127.0.0.1:5173/?replay=54faa848d99fa8940ef377e3ff2e87aa` |

## What to look for (all recordings)

1. Before the defender becomes visible (inside the 25 m information gate), does A1 fly toward the asset normally (no early swerving)?
2. Does it deviate only AFTER defender information becomes available?
3. Is the evasion smooth rather than random or wandering?
4. Does it create enough separation to avoid the defender (interception radius 2.5 m)?
5. After clearing the defender, does it turn back toward the asset?
6. Does it capture on the first reasonable approach?
7. Any unnecessary loops, oscillations, fly-bys or excessive detours?
