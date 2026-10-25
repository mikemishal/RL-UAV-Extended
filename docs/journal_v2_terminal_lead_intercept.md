# V2 attacker: capture-assured terminal lead-intercept guidance (rejected)

Starting research HEAD for this task: `f3d28e084924bde2dccfee09e792fccfb13e3534` (diagnostics/
tests/docs only beyond the accepted production commit `873026ea15d0794bc45bf767af2df3aaa45e49fa`).
Production code at the end of this task is again byte-identical to `873026ea`
(`git diff 873026ea -- uav_defend` is empty); only this document and a diagnostic module/tests were
kept.

## Phase 1: fly-by population scan

`terminal_flyby_metrics` (kept from the previous round) was run against the accepted production
behavior over passive A0, seeds 130000-130099, for C0/C1/C2/C3 (400 episodes total). Result:
**zero detections in C0, C1 and C2; exactly one in C3 (seed 130017, 22 fly-bys, 14.46 cumulative
bearing revolutions)**. C3 130017 is not a member of a larger class of terminal-guidance failures
in this population -- it is fully isolated.

## Design tried: capture-assured terminal lead-intercept controller

A new `TERMINAL_INTERCEPT` mode, strictly below obstacle avoidance in priority and strictly above
normal pursuit, applying only in V2 A0 DIRECT mode:

* **Asset-state source**: current asset position (already used by pure pursuit); asset velocity
  estimated by finite difference between the two most recently OBSERVED asset positions (the
  controller's own one-step memory) -- never a privileged or future ground-truth field.
* **Lead-intercept equation**: for commanded speed `s` and `r = p_asset - p_attacker`, solve the
  constant-velocity intercept quadratic `||r + v_asset*t|| = s*t` for the smallest positive `t`;
  initial direction `normalize(p_asset + v_asset*t - p_attacker)`. Re-solved from the predicted
  current state at every step of a rollout (no persisted waypoint).
* **Candidate speeds**: a 1 m/s grid over `[0, v_e]` (`v_e = 12`), matching the already-audited
  physical range (no minimum-speed/hover constraint exists for the attacker).
* **Capture-assurance gate**: a candidate is usable only if its own rollout predicts (a) swept
  capture within the horizon, (b) strictly before any obstacle conflict, and (c) strictly before
  the step at which the EXISTING, unmodified avoidance trigger (the same full-speed pure-pursuit
  rollout `uav_defend/obstacles/stateful_avoidance.py` uses) would itself have activated. Condition
  (c) exists specifically because of the rejected speed-governor's failure mode (below).
* **Nominal-pursuit gate (added after an initial bug)**: if full-speed pure pursuit ALREADY
  predicts a clean capture, the controller must not act at all (bit-identical to not existing).
  Without this check the controller changed *every single recorded trajectory*, including every
  C0 episode (zero obstacles, nothing to fix) -- it was quietly picking a marginally different but
  still-capturing lead candidate even when plain pursuit was already fine. With the gate, C0/C1 are
  bit-identical to the foundation and only 1-2 of 100 C2/C3 episodes change at all.
* **Persistence/anti-chatter**: the previously committed speed is re-evaluated fresh from the
  current state each step and kept if it is still capture-assured, unless a freshly searched
  candidate predicts a strictly earlier capture.
* **Ranking**: lexicographic among capture-assured candidates -- earliest capture, shortest path,
  largest minimum clearance, smallest heading change, highest speed.

## Why it was rejected

### The compliant (hard-gate-passing) variant barely moves 130017

With the nominal-pursuit gate and persistence in place -- required for Test-A compatibility and
confirmed zero new collisions across the full 400-seed population (`git diff`-clean trajectories
for 396/400 episodes; the 4 that change show no collision and no outcome change; see
`results/journal_v2_planner_fidelity/ti2*`) -- **C3 130017 is essentially unfixed**:
path-excess `27.058x -> 26.853x` (323 of 346 steps), nowhere close to materially below 27x and far
above the 5x threshold the task itself flags as requiring further investigation.

### Why: the asset has no velocity to lead

`SoldierEnv._move_soldier` moves the asset with "a true Gaussian random walk": a **fresh,
independent** horizontal displacement is drawn every step (scale `v_s * dt`), with **no velocity
persistence** at all. The Phase-4-mandated constant-velocity extrapolation of a finite-difference
"velocity" estimate therefore has no real signal to extrapolate beyond one step -- it is
structurally, not just empirically, an unreliable predictor over any horizon long enough to matter
for an orbiting fly-by. This was measured directly: of the 19 real `TERMINAL_INTERCEPT` activations
that occurred while reproducing 130017 with an earlier (less-gated) variant, only **3 (16%)**
produced swept capture anywhere near their own predicted capture step; 16 were simply abandoned at
the next real step once the controller's own fresh prediction (now built from a new, different
asset-velocity estimate) no longer agreed with the one it had just committed to. Shortening the
required look-ahead to the 2 nearest predicted steps (hypothesising that very-near-term lead
predictions would be more trustworthy) did **not** help capture reliability, and in a separate run
introduced a **new static-obstacle collision** in the same episode -- the identical failure mode
that sank the earlier speed-only governor (rerouting the attacker, this time via a different
heading law, into the pre-existing, unmodifiable avoidance controller's "all candidates collide"
emergency path). This reproduces, under a materially different terminal-guidance law, the exact
chaotic-sensitivity finding from the rejected terminal speed governor: **any** intervention that
measurably perturbs this seed's approach trajectory risks the one known pre-existing gap in the
avoidance controller's emergency handling, and the asset's zero-persistence motion model makes any
multi-step capture *prediction* for this seed unreliable regardless of which guidance law commands
it.

### Net conclusion

A local, receding-horizon terminal controller -- whether it changes only speed (previous round) or
both heading and speed via lead-intercept (this round) -- cannot reliably and safely resolve C3
130017:

* a version permissive enough to meaningfully shorten the fly-by either fails Test-A compatibility
  (changes unrelated, already-fine trajectories) or risks a new collision;
* a version strict enough to guarantee compatibility and avoid new collisions leaves the fly-by
  essentially unresolved, because the fundamental asset-motion assumption the task requires
  (constant-velocity extrapolation) does not hold for this soldier model.

Production code has been reverted to `873026ea` (bit-identical). The diagnostic module
(`experiments/terminal_lead_intercept_diagnostic.py`, the pure lead-intercept geometry and
rollout search, never wired into `uav_defend/`) and its unit tests
(`tests/test_terminal_lead_intercept_diagnostic.py`) are kept, since the geometry itself is
correct and may be useful for a genuinely different (route-level or asset-model-aware) future
design.

## Evidence files (gitignored)

* `results/journal_v2_planner_fidelity/terminal_c0/` -- C0 100-seed foundation recording (added
  this round; the earlier accepted foundation set only covered C1/C2/C3).
* `results/journal_v2_planner_fidelity/ti2*` -- the compliant (nominal-gate + persistence) variant,
  named cases and 300/100-seed populations.
* `results/journal_v2_planner_fidelity/terminal_flyby_scan.json` -- the Phase-1 population scan.
