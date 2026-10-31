# Journal V2 A1 — finite tactical-pulse geometry study (`a1_v2_action_v5`) — NEGATIVE RESULT

**Verdict C: tactical geometry still creates unsafe / non-mission behavior. No geometry selected, no protocol tag, no PPO trained.**
Starting point: research `cd6bc38` (clean), A0 tag `journal-v2-a0-canonical-v1` = `bdbee45`, `uav_defend` byte-identical to that tag, reward `a1_v2_reward_v2` unchanged.

## Pre-registered design (frozen before the sweep; `maneuver_specification.json`, SHA-256 in `task_state.json`)
- Actions `Discrete(4)`: BASE, LEFT_PULSE, RIGHT_PULSE, CLIMB_PULSE. Authority: contact > frozen common obstacle layer > tactical pulse > BASE.
- Geometry with `m` recomputed each step: `l = normalize(z × m_h)`; LEFT `normalize(m + tanθ·l)`, RIGHT `normalize(m − tanθ·l)`, CLIMB `normalize(m + tanθ·ẑ)`; degenerate horizontal projection → BASE; CLIMB → BASE when less than one step of maximum climb remains below the ceiling.
- Activation only when defender information is available (true range ≤ 25 m, the observation's existing gate), the frozen layer is not owning steering, no pulse is active and rejoin is complete.
- State machine `BASE → TACTICAL_{LEFT,RIGHT,CLIMB} (n steps) → REJOIN (2 steps) → BASE`, `OBSTACLE_OVERRIDE` whenever the frozen common layer owns steering (its own trigger, reproduced exactly: perceived layout non-empty and direct rollout collides). Requests during a pulse/rejoin are ignored (no restart/extension).
- Observation `a1_v2_obs_v2` = unchanged 37 features + 5-way mode one-hot [BASE, LEFT, RIGHT, CLIMB, REJOIN/OBSTACLE-SUPPRESSED] + remaining fraction = **43-D**.
- Grid (never extended): θ ∈ {10,15,20,30}° × T ∈ {0.5,1.0,1.5,2.0} s (1–4 steps), rejoin 1.0 s. Seeds 145000–145099, C0–C3; contexts greedy/lead/pn (real gate) + `passive_forced` (mission-only stress diagnostic); 8 scripted policies (BASE, LEFT, RIGHT, CLIMB, alternate, cycle, random, adversarial rapid switching).
- Gates G1–G5 and the selection order are quoted verbatim in `maneuver_specification.json`.

## Results (full tables: `sweep_results.md`; machine-readable: `sweep_summary.json`)
- **BASE equivalence**: BASE through the machine is bit-identical to scripted A0 in 2,000/2,000 episodes (C0–C3 × 5 defender families × 100 seeds; max deviation 0.0 m; 0 gained collisions; same endurance behaviour) — `base_equivalence.json`.
- **G5 (state machine): 16/16 pass** — 0 invariant violations over 204,800 episodes, longest tactical run ≤ n, deterministic replay, obstacle override observed (up to 1,719 cancelled pulses). A first version of the invariant checker produced false positives; it was corrected and mutation-tested (7 invalid sequences detected) before the sweep.
- **G1 (zero gained collisions): 0/16 pass.** BASE has 0 collisions in every cell. Active-defender gained collisions (of 2,100 tactical episodes per level) range from 2/15/24 (C1/C2/C3, gentlest 10°/0.5 s) to 154/589/1,026 (30°/2.0 s); C0 is 0 everywhere.
- **Mechanism**: 55–96% of gained collisions happen while the frozen common layer already owns steering (post-maneuver state unrecoverable); 4–45% on the pulse step itself.
- **G2 (mission): 0/16 pass.** Endurance exhaustion in active-defender episodes 48 (10°/0.5 s) … 3,897 (30°/2.0 s) vs 0 for BASE. Ceiling hover 0, sustained hover 0, orbit ≥ 1 revolution only 9 episodes overall (30° with T = 1.0 and 2.0 s). The >5× path cases are 90% endurance exhaustion: a long tail of failure to close on the asset (mean 57 steps vs ≈ 10 for BASE), not hover/orbit.
- **G3 (authority): 14/16 pass and the authority is real**: for θ ≥ 15° pulses convert BASE interceptions into captures more often than the reverse (e.g. 15°/1.5 s: 905 vs 320; 30°/1.0 s: 1,636 vs 103), mean minimum defender separation +0.04 … +2.2 m. The two weakest candidates (10°, T ≤ 1.0 s) do not pass.
- **G4 (path): 3/16 pass** (10° with T ≤ 1.5 s); BASE max path-excess 3.76×, tactical max 19.1×.
- **Selection**: no candidate passes G1–G5 → none selected; `selected_geometry.json` deliberately absent; training-runner preparation (`a1_v2_training_v3`) not performed.

## Files
`maneuver_specification.json`, `base_equivalence.json`, `sweep_summary.json`, `sweep_results.md`, `diagnostic_recordings.json` (7 deterministic NON-CANONICAL examples: gained collisions at the gentlest candidate, LEFT/RIGHT/CLIMB defender-outcome changes, obstacle-override cases, each with the matched BASE reference; not wired into the visualizer), `ablation_record_v2.md`, `unit_test_result.json`, `repository_verification.json`, `task_state.json`. Raw per-group sweep records (64 files, per-group SHA-256) remain in the gitignored `results/.../a1_v2_training_v3/tactical_geometry_sweep/`.
Code: `experiments/journal_v2_a1_tg.py` (geometry, machine, obs-v2, env), `journal_v2_a1_tg_sweep.py`, `journal_v2_a1_tg_equiv.py`, `journal_v2_a1_tg_report.py`, `journal_v2_a1_tg_state.py`; tests `tests/test_journal_v2_a1_tg.py` (51).

## Lesson and what this does not show
A finite, defender-triggered pulse with a mandatory rejoin and perfect state-machine behaviour still cannot be executed safely on top of the frozen reactive obstacle layer, because the layer's recoverability is state-dependent and a pulse displaces the vehicle into states it cannot always recover from. Amplitude and duration only trade authority against collisions/exhaustion; no point of the grid is free. Not tested (and not claimed): a pulse admission rule that validates pulse + rejoin + planner response against the same perception (a *look-ahead recoverability gate*), an obstacle-aware maneuver set, or modifying the common planner. Any such design must be pre-registered as a new action version; the grid here was not extended after seeing results.
