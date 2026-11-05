# A1-CBF — pursuit/dynamics foundation (analytical relative-target guidance + quadrotor velocity executor)

**Result: the foundation PASSES the pre-registered acceptance gate on all 7 checks; verdict A.** A deterministic,
literature-motivated attacker — proportional guidance on the current relative target position, an identity CBF/QP
safety-filter interface, and an acceleration-limited (not turn-rate-limited) quadrotor velocity executor — captures
the canonical moving asset in **100.0 %** of 1,000 matched scenarios (A0: 99.3 %), with **0 repeated fly-by episodes**
(A0: 9), **0 orbit episodes**, and **0 numerical failures**, with NO RL training of any kind. This directly repairs
the fly-by/covariate-shift failure mode that the PPO/BC A1 curriculum (Stages 4, 4A, 4A-1) could not fix. The
opt-in `enemy_control_mode` switch is verified bit-for-bit no-op (1,000/1,000 identical episodes against the branch
point commit) when left at its default.

## 0. Why this direction, and what this stage is NOT
The prior PPO/BC curriculum (`docs/journal_v2_a1_curriculum/`) spent four stages (Stage 4, 4A, 4A-1) trying to teach
a neural-network attacker to track a moving target under the canonical benchmark, and failed each time with the
SAME symptom — repeated near-miss fly-bys — for different diagnosed reasons (missing asset-input sensitivity,
then behaviour-cloning covariate shift). The research focus from here on is the **counter-UAS defender**, not a
novel attacker. A1-CBF exists to give the defender research a non-degenerate, literature-grounded attacker
(deterministic velocity-level pursuit + a safety-filter interface that a later stage turns into an actual CBF/QP),
without consuming more effort on attacker learning. **This stage implements ONLY** (1) analytical moving-target
pursuit and (2) the velocity-command/quadrotor execution model, plus the interfaces a later stage needs for
obstacle/defender CBF constraints. **It does NOT implement obstacle avoidance, defender avoidance, or any CBF/QP
solve** — the safety filter is the identity function, `hazards` is always empty, and `cbf_qp_status` always reads
`"identity_no_constraints"`.

## 1. Architecture (frozen)
```
current relative target state -> nominal velocity guidance -> CBF/QP safety filter -> quadrotor velocity executor
```
Implemented in `uav_defend/attackers/cbf_attacker.py` (pure functions / a stateless `CBFAttackerController`
dataclass; nothing in this module imports `SoldierEnv`, so it is independently unit-testable):
* `nominal_velocity_command` / `CBFAttackerController.nominal_velocity` — the guidance law (§2).
* `CBFAttackerController.filter_velocity` — the safety-filter **interface**: Prompt 1 is `v_safe = v_nom` exactly,
  `hazards=()` always. A later prompt replaces only this method's body with a QP solve; no other component changes.
* `integrate_quadrotor_substeps` — the velocity executor (§3).

## 2. Nominal guidance (frozen; never tuned after the acceptance run)
```
r = p_asset - p_attacker                      (CURRENT relative position; never the initial/obsolete location)
v_nom = min(max_speed, kp * ||r||) * r / ||r||    for ||r|| > eps,  else [0, 0, 0] (hover)
kp = 1.0 s^-1,  max_speed = 12.0 m/s
```
No target-velocity feedforward (the canonical asset is an i.i.d. Gaussian random walk with no persistent velocity —
see `docs/journal_v2_a1_curriculum/stage4/README.md` — so there is no well-defined target velocity to feed forward;
this is a deliberate, documented omission). No minimum forward speed: hover (`v_nom = 0`) is always reachable. With
`kp = 1`, the nominal command slows naturally: ≈12 m/s at ≥12 m, 6 m/s at 6 m, 3 m/s at 3 m, 2 m/s at 2 m.

## 3. Quadrotor velocity executor (frozen)
* External step `dt = 0.5 s` (unchanged, for compatibility with every existing experiment). Internal substep
  `dt_sub = 0.1 s`, so **5 substeps per external step**; `v_cmd` (= `v_safe`, computed once per external step) is
  **held constant across all 5 substeps**.
* Per substep: `Δv = v_cmd - v`; clip `‖Δv‖ ≤ a_max·dt_sub` (full 3-D, no decomposition); `v += Δv`; hard-clip
  horizontal speed to `max_horizontal_speed`; hard-clip `v_z` to `[-max_descent_rate, max_climb_rate]`;
  `p += v·dt_sub`; apply the existing (bit-identical) arena-boundary handling.
* **No legacy heading/turn-rate limit.** A quadrotor can accelerate laterally, decelerate, hover and reverse
  without maintaining forward airspeed, unlike the fixed-wing-style legacy model — this is the single physical
  change from the legacy dynamics, besides the guidance law itself.
* Physical limits (identical to the canonical hostile-UAV values already in `uav_defend/config/env_config.py`, kept
  deliberately so only the guidance+executor change, not the vehicle's physical envelope): max horizontal speed
  12 m/s, max acceleration 6 m/s², max climb 5 m/s, max descent 5 m/s.

## 4. Swept mission-contact semantics: unchanged, not double-tested
`SoldierEnv.step()`'s swept capture/interception test is **completely untouched**: it still tests exactly one
linear segment from the pre-step to the post-(all-5-substeps) position, over the full external 0.5 s step — the
SAME mechanism every other controller (A0, Greedy, PN, Lead, PPO/BC) uses, and the same mechanism validated
throughout the moving-asset curriculum work. The 5-substep integration only changes **how** the attacker's one-step
motion is produced (acceleration-limited substepping replacing a single turn-rate-limited update); it introduces no
new or parallel contact test, so no event can be double-counted. This is verified directly: 993/1,000 A0 episodes
and 1,000/1,000 A1-CBF episodes terminate in `soldier_caught` with `terminal_distance_m <= 2.0 m` (the unchanged
capture radius), and a dedicated test asserts this on a smaller sample.

## 5. The opt-in control-mode switch and legacy no-op
`SoldierEnv.__init__(..., enemy_control_mode: str = "legacy")` — a constructor argument, deliberately kept OUT of
`EnvConfig` (the same way `render_mode` already is), validated to be `"legacy"` or `"cbf_velocity"`. Keeping it off
`EnvConfig` matters: every prior PPO/BC curriculum stage pins a frozen SHA-256 fingerprint of
`dataclasses.asdict(config)` as a reproducibility guard (`docs/journal_v2_training_handoff.json` and each stage's
`docs/journal_v2_a1_curriculum/stage*/protocol.json`); adding a new `EnvConfig` field — even an inert, defaulted one
— changes that fingerprint for every one of those already-completed, protected historical stages. Putting the
switch on the constructor instead leaves `EnvConfig`, and every historical fingerprint, completely untouched.
`"legacy"` is the sole default and reproduces the ENTIRE prior hand-designed guidance+dynamics pipeline — the
single code change on this path is one `if` check at the top of `SoldierEnv._move_enemy()` that evaluates to
`False` and falls through to the unmodified body. **Legacy no-op verification:** 1,000 matched canonical-greedy
scenarios (seeds `510000–510999`), each run through (a) the CURRENT code with `enemy_control_mode="legacy"` and (b)
`uav_defend/envs/soldier_env.py` **as it existed at the branch-point commit `2f45914`** (loaded dynamically via
`git show`, reusing every other current — unmodified — `uav_defend` submodule), comparing outcome, terminal step,
every entity's position/velocity trajectory, attacker path length, and the collision/contact result. **1,000/1,000
identical, 0 mismatches.** The full project regression suite (1,143 tests, including every pre-existing
config/protocol-hash guard test across the whole PPO/BC curriculum) also passes unchanged.

`"cbf_velocity"` replaces the entire hand-designed guidance (pursuit + weave + noise + evasion + obstacle avoidance
+ route planning) and the turn-rate-limited dynamics with the A1-CBF pipeline (§1–3). It draws no randomness, does
not consult `enemy_tactical_override` (a *different*, direction-only hook used by the learned-attacker-v1 family),
and ignores `enemy_evasion_*` / `enemy_obstacle_avoidance_enabled` (out of scope for this prompt).

## 6. Foundation acceptance experiment (1,000 matched scenarios, seeds `500000–500999`)
Environment: C0, canonical moving asset (`v_s=1.5`, `legacy_gaussian_random_walk`, unchanged), finite mission
endurance (k=5, unchanged), swept mission contact, capture radius 2.0 m (unchanged) — **no active defender
interaction**: the established curriculum mechanism (`intercept_radius=1e-6`, reused verbatim from the Stage1-v2 /
Stage4A/4A-1 work) is applied so the structurally-present defender can never terminate the episode. Canonical A0
(`enemy_control_mode="legacy"`, the existing `enemy_nominal_guidance_mode="direct_obstacle_only"`) and A1-CBF
(`enemy_control_mode="cbf_velocity"`) are evaluated on the **identical** seeds; common random numbers hold because
the asset's `_rng_soldier` stream is independent of `enemy_control_mode` (verified directly, every step, over 5
matched scenarios, in addition to being true by construction).

| | canonical A0 | A1-CBF |
|---|---:|---:|
| Capture | 99.3 % | **100.0 %** |
| Endurance exhaustion | 0.7 % | 0 % |
| Timeout | 0 | 0 |
| First-pass capture (of all episodes) | 99.0 % | 99.3 % |
| First-pass fraction of captures | 99.7 % | 99.3 % |
| Repeated fly-by episodes (≥2 overshoots) | **9** | **0** |
| Fly-by episodes (≥1 overshoot) | 10 | 7 |
| Orbit episodes | 0 | 0 |
| Numerical failures | 0 | 0 |
| Median / mean path-excess | 0.972 / 1.012 | 0.975 / 0.976 |
| Mean duration / path length | 5.47 s / 61.8 m | 5.37 s / 59.7 m |
| Mean closest-approach count | 1.025 | 1.011 |
| Median miss distance (near-miss episodes) | 4.37 m | 2.59 m |
| Mean speed / max speed | 11.82 / 12.0 m/s | 11.24 / 12.0 m/s |
| Max acceleration observed | 6.0 m/s² | 6.0 m/s² |
| Mean fraction of time below 3 m/s | 0.02 % | 0.26 % |
| Total hover events (speed < 0.5 m/s) | 0 | 1 |

A1-CBF is never worse than A0 on any primary metric and strictly better on the two the curriculum could never fix
(repeated fly-bys: 9 → 0; capture: 99.3 → 100.0 %). The *mechanism* is visible in the speed-vs-distance figure
(fig. 1): the nominal law slows the command smoothly as range shrinks, and the executor's acceleration limit
produces a brief, physically-expected lag (a quadrotor cannot instantaneously match a falling-range command), but
this never causes an overshoot the way the legacy controller's fixed-speed, heading-rate-limited approach did.

## 7. Acceptance gate (frozen BEFORE the run; §16/§17 of the task brief) — all 7 checks PASS
| Check | Required | Observed | Pass |
|---|---:|---:|:--:|
| Capture | ≥ 98 % | 100.0 % | ✓ |
| Exhaustion | ≤ 2 % | 0 % | ✓ |
| Timeout | = 0 | 0 | ✓ |
| First-pass capture | ≥ 97 % | 99.3 % | ✓ |
| Repeated fly-by episodes | ≤ 2 % | 0 % | ✓ |
| Persistent orbit | = 0 | 0 | ✓ |
| Numerical failures | = 0 | 0 | ✓ |

**Dynamics (constraint-limit verification):** over all 1,000 A1-CBF episodes, the maximum observed horizontal speed
is 12.000 m/s (limit 12.0) and the maximum observed acceleration is 6.000 m/s² (limit 6.0) — within floating-point
tolerance of, never exceeding, their configured limits. No speed, acceleration, or climb/descent violation
occurred. `kp`, the four physical limits, the substep, and the capture radius were frozen before this run and were
**not** tuned afterward.

## 8. Limitations
1. No obstacle avoidance and no defender avoidance: the safety filter is the identity function; this is the
   explicit scope of this prompt, not a limitation of the architecture.
2. The foundation experiment uses C0 and a non-interacting defender; it says nothing about A1-CBF's behaviour once
   obstacle/defender CBF constraints are added (the next stage).
3. `kp=1`, the limits, and the capture radius were frozen before this run, as required; whether a different `kp`
   would perform even better was deliberately not explored (per the "do not tune after seeing results" rule).
4. A1-CBF is a deterministic, hand-specified controller, not a learned one — by design (the research direction is
   the defender, not a novel attacker).
5. The acceleration-limiting "shelf" visible in fig. 1 near 4–10 m range is a genuine physical lag of the quadrotor
   model, not a bug; it does not prevent capture at this benchmark's scale.

## 9. Reproduce / verify
```
python -m experiments.journal_cbf_attacker_run all        # stages 1-7 (restartable; writes docs/journal_cbf_attacker/foundation/*)
python -m experiments.journal_cbf_attacker_figures         # figures 1-3
pytest tests/test_cbf_attacker.py -q
```
State: `task_state.json` (10 stages, atomic, hash-verified). Results (untracked):
`results/journal_v2_corrected_semantics/journal_cbf_attacker/foundation/` (chunked, restart-safe). Protocol SHA-256:
`dbe29eaa41c24b4b…` (`protocol.json`). Namespaces: foundation seeds `500000–500999`, legacy no-op seeds
`510000–510999` (both far from every A1-curriculum evaluation block, which tops out at 192199).

## 10. Scope reminder
Research focus beyond this point is the counter-UAS **defender**. A1-CBF is not claimed as a novel attacker
contribution. The next stage (obstacle CBF) is **not started here**; neither is defender CBF, RL training of any
kind, or a canonical tag for this controller.

## Figures
`figures/fig1…fig3` (PNG + PDF): speed vs. asset distance (A1-CBF, vs. the nominal law); capture/exhaustion/
first-pass/repeated-fly-by rate comparison (A0 vs A1-CBF); a representative matched trajectory (A0 vs A1-CBF on the
identical moving-asset path, the longest-path capture in the validation set).
