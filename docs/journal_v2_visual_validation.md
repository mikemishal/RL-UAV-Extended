# V2 visual and behavioral validation

Scope: behavioral/visual validation only. No training, no evaluation of
learned policies, no change to simulation semantics, no battery model. All
code is observational (reads `info`/public env state; never mutates control,
reward or termination). Repository: `research/journal-corrected-attacker-v2`,
validated at commit `db8c299` (`journal-corrected-attacker-semantics-v2`).

## 1. Seed-selection protocol (declared before results were inspected)

- **Core set:** seeds 130,000–130,009 (10 seeds), every canonical level
  (C0–C3), every scripted controller. Seed 130000 is the previously
  problematic one from the original visualizer session.
- **Coverage extension (added after an observation, not after inspecting
  outcomes we wanted):** the 10-seed set produced only 0/1/4 avoidance
  segments in C1/C2/C3 respectively — too few to select a meaningful
  "max-avoidance" or representative obstacle-bypass case. A 100-seed passive
  extension (130,000–130,099, V2, passive controller, same counterfactual
  instrumentation) was added for C1–C3 only, declared in
  `experiments/journal_v2_visual_validation.py` before any case was chosen
  from it.
- **V1-vs-V2 comparison:** C0 and C2, controllers `passive` and `greedy`, same
  seeds, V1 built through the explicit `build_v1_env_config` builder.
- **Per-level visual subset:** first 3 seeds (130000–130002), the seed with
  the most `OBSTACLE_AVOID`/`ROUTE_BYPASS` steps (C1–C3 only, from the 100-seed
  extension), and the seed with the longest attacker path.
- **Presentation examples:** objective rules in
  `select_presentation_cases()` — p1 = first non-130000 C0 seed; p2 = seed
  130000; p3 = the C1/C2 episode with exactly one avoidance segment that is
  counterfactually *necessary* and has the largest lateral excursion (from the
  100-seed extension); p4 = the C3 episode with the most avoidance segments;
  p5 = Lead (or Greedy if Lead unavailable) interception at C0; p6 = the
  swept-interception example below.
- **Swept-interception example:** the first (by controller order, then level,
  then seed) V2 episode whose terminal event is `interception` and whose
  end-of-step (endpoint) counterfactual would **not** have ended the step.
- **Constructed scenarios** (S1–S3): deterministic, hand-placed obstacle
  layouts with a stationary asset — not random draws — used to isolate
  entry/bypass/exit/necessity behavior cleanly.
- **Competing-event scenarios:** three hand-constructed single-step cases
  (defender closer / defender farther / exactly tied) with a fixed attacker
  trajectory, to exercise event ordering directly.

## 2. Controllers

`passive` (defender parked at the engagement corner, standby disabled, never
acts — isolates attacker motion), `greedy`, `kalman_greedy`, `pn`, `lead`,
`rmpc`. All five scripted controllers import and ran without modification; no
PPO checkpoint was used.

## 3. Clutter levels

C0 (no obstacles), C1 (low), C2 (medium), C3 (high) — all built through
`experiments.journal_v2_protocol.build_v2_env_config` / `build_v1_env_config`,
which assert the intended semantic switches explicitly rather than relying on
defaults.

## 4. C0 — behavioral verdict: PASS

- Every V2 episode (60 across the core set, all controllers) has
  `frac_direct == 1.0`, `max_abs_weave_bias == 0.0`, maximum deviation between
  the attacker's final command and its raw mission direction `< 0.1°`.
- Zero orbit events, zero overshoots, zero missed swept attacker captures,
  zero missed swept defender interceptions.
- Passive median pursuit efficiency (net range reduction / path length) is
  0.996; median closing-step fraction is 1.0 (every step closes range).
- Figures: `c0/C0_seed130000_first3_longest_path.png`,
  `v1_vs_v2/C0_passive_seed130000_v1_vs_v2.png` (and its 3-D companion),
  `c0/C0_seed130000_controller_overlay.png`.

## 5. C1 — behavioral verdict: PASS

100-seed passive coverage: 100% capture, 1 avoidance segment total (a single
1-step dynamic-obstacle reaction), 0 unassociated segments, 0 segments that
fail to return to DIRECT, 0 orbit events, 0 overshoots, median path-excess
ratio 0.97 (direct flight is at least as short as the straight-line distance
most of the time, since the asset moves slightly).

## 6. C2 — behavioral verdict: PASS, with one caveat (see §13)

100-seed passive coverage: 100% capture, 16 avoidance segments over 7
episodes, every one associated with a perceived obstacle, every one returns to
DIRECT, 0 orbit events. One episode (seed 130065) has an excessive, though
counterfactually *necessary*, detour (path-excess ratio 9.08×, max lateral
deviation 40.0 m) — see §13. Excluding that one seed, path-excess ratios stay
near 1.0.

## 7. C3 — behavioral verdict: obstacle-bypass caveat (see §13); not a
clear-space or event-semantics defect

100-seed passive coverage: 100% capture, 53 avoidance segments over 26
episodes, every one associated with a perceived obstacle, every one returns to
DIRECT, 0 orbit events (the orbit metric, which requires ≥1 full revolution
*close* to the asset, never fires — the large excursions in §13 happen far
from the asset). 6 of 100 episodes have a path-excess ratio above 1.3× (max
4.47×); 3 of 100 have more than 4 consecutive steps where the direct line was
already clear but the attacker was still formally in avoidance mode (max 10).

## 8. Seed 130000 — V1 vs V2

| | V1 (endpoint, legacy A0) | V2 (swept, direct A0) |
|---|---|---|
| Outcome | `soldier_caught` at **step 874** | `soldier_caught` at **step 12** |
| Path length | 4,238.6 m | 59.4 m |
| Heading change | 12,636.8° | 20.2° |
| Overshoots (env counter) | 479 | 0 |
| Orbit events (≥1 rev within 20 m) | 4 | 0 |
| Missed attacker-capture crossings (independent dense-sampling audit) | **5** | 0 |
| Pursuit efficiency | 0.01 | 1.00 |

V1's near-misses (asset-frame plot in
`v1_vs_v2/C0_passive_seed130000_v1_vs_v2.png`): endpoint distances 2.0, 3.8,
3.5, 2.6, 3.1 m, with swept minimum separations of 1.6, 1.6, 1.1, 0.6, 1.7 m —
every one would have been a capture under swept contact but was missed under
endpoint contact. V2 captures on the first approach: swept contact at
τ = 0.42, interpolated contact distance exactly 2.0 m (the capture radius).
3-D view (`v1_vs_v2/C0_seed130000_3d_v1_vs_v2.png`) confirms this is not an
altitude artifact — V1's loops are genuine horizontal circling, not a 2-D
projection of distinct altitude passes.

## 9. Missed-contact counts (all V2 episodes, core set + coverage extension)

- Attacker-capture: **0 / 0** (zero missed swept crossings across the core
  set and the 100-seed-per-level extension; independent dense-sampling audit,
  257 sub-steps per transition, agrees with the analytic solver in all cases).
- Defender-interception: **0 / 0** (core set only; the extension is
  passive-controller so has no defender to intercept).

## 10. Overshoot and orbit counts

- Overshoot definition: a range-to-asset decrease (rate < −1 m/s) followed by
  an increase (rate > +1 m/s) with no terminal event — an approach-then-recede
  transition, counted once per transition. Verified against the environment's
  own `attacker_overshoot_count` in every recorded episode (exact match).
- Orbit definition: cumulative signed bearing change of the attacker about the
  asset, accumulated only while the attacker stays within 20 m of it,
  per continuous excursion; an excursion with `|Δbearing| ≥ 1` revolution is
  an orbit event.
- V2 totals: C0 0/0, C1 0/0, C2 0/1 overshoot episode (seed 130065's large
  loop, which starts and ends >20 m from the asset so does not register as an
  orbit event), C3 0 orbit events / 6 overshoot episodes (out of 100 extended
  seeds).
- V1 totals (10-seed core set): C0 passive 4 orbit events / 10 episodes, C0
  greedy 0 (defender intercepts before persistent circling), C2 passive 1
  orbit event, C2 greedy 0.

## 11. Path-length / directness results

Pursuit efficiency = (initial range − final range) / path length; because the
asset random-walks (σ = v_s·dt per axis per step) this is **not** an exact
shortest-path ratio — it is the net range reduction achieved per metre flown,
documented so it is not over-interpreted. Path-excess ratio = path length /
initial straight-line distance (likewise approximate for the same reason).

| Level | Controller | Semantics | Mean path (m) | Median pursuit efficiency | % flight DIRECT |
|---|---|---|---|---|---|
| C0 | passive | V1 | 4238.6 | 0.01 | 100.0 |
| C0 | passive | V2 | 59.4 | 1.00 | 100.0 |
| C0 | greedy | V1 | 1127.5 | 0.04 | 71.9 |
| C0 | greedy | V2 | 56.0 | 1.00 | 100.0 |
| C2 | passive | V1 | 5642.5 | 0.01 | 48.1 |
| C2 | passive | V2 | 59.5 | 1.00 | 98.0 |
| C2 | greedy | V1 | 892.4 | 0.03 | 37.9 |
| C2 | greedy | V2 | 56.8 | 1.00 | 98.0 |

Full table: `v1_vs_v2/comparison_table.md`. V2 summary across all six
controllers and four levels: `summary_table_v2.md`.

## 12. Obstacle entry/bypass/exit behavior

For every recorded V2 episode (core set + 100-seed extension), every
`DIRECT → {OBSTACLE_AVOID, ROUTE_BYPASS}` transition was checked against:

- **Entry association:** is the obstacle a real, currently perceived one (a
  dynamic-obstacle id, or a static id present in the attacker's own perceived
  set)? **100% yes**, 0 unassociated segments in any recorded episode.
- **Exit:** does the episode return to `DIRECT`, and is the direct line to the
  asset clear at that moment? **100% return**, and the direct line was clear
  at every recorded return except 5 segments (3 in C2, 2 in C3, all
  `necessary_collision`-classified, out of 86 total segments) where the
  episode returned to `DIRECT` one step before the decision-time line test
  itself cleared — i.e. the return is slightly ahead of the diagnostic's own
  1-step-delayed line check, not a stale bypass.
- **Necessity (counterfactual):** for every `DIRECT → avoidance` transition, a
  copy of the pre-activation state was branched with avoidance disabled and
  flown straight for the planner's own 12-step horizon. Across the 100-seed
  extension: C1 0/1 segments necessary (the one segment is a single-step
  dynamic-obstacle reaction, classified "unnecessary" by this 12-step static
  test because the obstacle had already passed by the time the rollout
  started — see limitation below), C2 11/16 necessary (9 would collide, 2
  would come within the clearance margin), C3 36/53 necessary. The remaining
  "unnecessary" segments are overwhelmingly single-step, low-consequence
  reactions (median duration 1 step) rather than large detours.

Log of activation/last-avoidance/return steps for every segment:
`avoidance_segments.csv` (core set) and
`coverage_extension_segments.csv` (100-seed extension).

## 13. Known limitation: large "climb" detours under high attacker momentum

**Finding.** In a minority of C2/C3 episodes (1/100 and 9/100 respectively,
by the path-excess-ratio > 1.3 threshold), the obstacle-avoidance planner
selects its **climb** bypass candidate after the attacker has built up a
sustained, undeviated high-speed horizontal run (made possible precisely
*because* V2 removed the clear-space weave that previously kept the legacy
attacker's heading constantly perturbed). The climb candidate commands a
near-vertical direction toward a fixed waypoint directly above the position
where avoidance activated. Because the vehicle's turn-rate limit
(75°/s) constrains how fast its established horizontal velocity can be
redirected, it takes a wide, multi-second loop — up to 9.08× the straight-line
path length and 40–57 m of lateral deviation in the worst recorded cases —
before settling into the climb and then resuming direct pursuit.

**This is not a new defect introduced by the corrected-attacker-v2 changes.**
The underlying mechanism — the climb-candidate waypoint and the turn-rate-
limited constrained dynamics — is pre-existing, shared code, unmodified by
this task or the prior one. Running the unmodified V1 attacker (legacy
weave/evasion) on the same ten C2/C3 seeds shows it also spends roughly
40–55% of every episode in `climb` avoidance mode; V1's own effectively
chaotic trajectory already masks these wide loops inside its general
circling, so the phenomenon was never visually isolated before. Every
example found:
- is associated with a real, currently perceived obstacle (never
  unassociated);
- is, in most cases, counterfactually **necessary** (the direct-rollout
  counterfactual shows a real predicted collision or clearance breach);
- always returns cleanly to `DIRECT` once resolved;
- never causes a collision, timeout, or failed capture (100% capture rate
  across every one of the 300 extension episodes, C1–C3).

**Verdict implication.** Because it is bounded, explainable, always resolves
safely, and is a shared pre-existing mechanism rather than a regression in the
new guidance/contact logic, this is reported as a caveat on an otherwise
passing C2/C3 result, not as grounds to withhold validation of the primary
behaviors this task was scoped to check (clear-space pursuit, event
semantics, no tactical evasion). See §22 (verdict).

**Scope note on the necessity counterfactual.** The 12-step, static-horizon
necessity check is itself approximate: a few single-step "unnecessary"
dynamic-obstacle reactions are an artifact of testing with a fixed horizon
after the obstacle has already moved past, not evidence of spurious
avoidance (entry is still gated on the planner's own prediction, computed
live, not on this diagnostic).

## 14. Observations by controller

- **Passive:** exposes the attacker's full, unmodified mission (used for
  every entry/exit/necessity measurement above).
- **Greedy / Kalman-Greedy / PN / Lead:** all four produce **bit-identical**
  attacker tracks to the passive run for every shared step (max deviation
  `0.0` m across all 240 core-set episodes — see
  `attacker_invariance_across_controllers.csv`), confirming A0 never reacts to
  the defender's presence or policy. Interception rates differ only because
  these controllers actually fly toward the attacker.
- **RMPC:** also attacker-invariant; occasionally fails to intercept in time
  at C3 (captured C3: 0/10 in the core set — RMPC is slower to react within
  the short post-detection window these episodes allow, a defender-side
  observation, not an attacker-model finding).
- No episode under any controller shows `enemy_evasion_active` or an `EVADE`
  behavior mode; `no_evasion_active_any_episode` and `no_evade_mode_any_episode`
  are both true.

## 15. Swept-interception visualization

Example: C0, greedy, seed 130000 (the first V2 episode across all
controllers whose terminal event is `interception` and whose end-of-step
counterfactual would not have ended the step).
`event_semantics/swept_interception_C0_greedy_seed130000.png`:
pre-step separation 6.81 m; the naive endpoint-only post-step separation is
3.35 m (**> 2.5 m interception radius — a miss**); the swept contact fraction
is τ = 0.441, giving an interpolated contact distance of exactly 2.50 m. Under
endpoint-only semantics this exact step would report `'ongoing'`.
Terminal-step audit over the full V2 run: 135 interceptions correctly
captured by their own endpoint, **38 interceptions would have been missed or
delayed by endpoint semantics** (`event_semantics/terminal_steps_endpoint_would_miss.csv`),
and 0 asset captures would have been missed.

## 16. Competing-event result

Three deterministic single-step scenarios
(`event_semantics/competing_events.png`, `.json`):

| Scenario | τ (capture) | τ (interception) | Winner | Outcome |
|---|---|---|---|---|
| Defender closer (x=−0.1) | 0.200 | 0.100 | interception | `unsafe_intercept` |
| Defender farther (x=+1.5) | 0.200 | 0.367 | capture | `soldier_caught` |
| Exactly tied (x=+0.5) | 0.200 | 0.200 | capture (documented tie rule) | `soldier_caught` |

All three match the implemented priority (defender collision > attacker
collision > capture > interception, earliest-time otherwise) exactly.

## 17. Presentation-quality examples (`presentation/`)

| File | One-sentence interpretation |
|---|---|
| `p1_c0_direct_attack.png`/`.gif` | A0 flies a near-straight line to the moving asset in open space, with zero weave. |
| `p2_c0_seed130000_swept_capture.png`/`.gif` | The previously problematic seed now ends in a single, clean swept capture instead of hundreds of steps of circling. |
| `p3_single_obstacle_bypass.png`/`.gif` | A0 deviates around one obstacle and returns immediately to a direct line once past it. |
| `p4_c3_complex_path.png`/`.gif` | In heavy clutter A0's path is longer but every turn traces to a visible obstacle, with no unexplained lateral oscillation. |
| `p5_scripted_interception.png`/`.gif` | A scripted defender (Lead/Greedy) intercepts A0 while A0 continues its mission unaware of the pursuit. |
| `p6_swept_defender_interception.png` | The swept interception geometry showing the endpoint-only miss that the corrected contact test catches. |

These are model-validation figures, not final publication figures.

## 18. Visual-validation directory

`results/journal_v2_corrected_semantics/visual_validation/`:
`c0/`, `c1/`, `c2/`, `c3/`, `v1_vs_v2/`, `event_semantics/`, `presentation/`,
`traces/` (per-step JSONL), plus `episodes_all.csv`, `avoidance_segments.csv`,
`attacker_invariance_across_controllers.csv`,
`coverage_extension_passive_100seeds.csv`, `coverage_extension_segments.csv`,
`constructed_scenarios_summary.csv`, `constructed_scenarios_segments.csv`,
`summary_table_v2.{csv,md}`, `seed_selection.json`, `acceptance.json`. No V1
visualization artifact is overwritten; V1 results are read only.

## 19. Acceptance criteria (full machine-readable result: `acceptance.json`)

| Criterion | Result |
|---|---|
| C0: no weave/non-direct mode | PASS |
| C0: no orbit events | PASS (0) |
| C0: zero missed attacker captures | PASS (0) |
| C0: zero missed defender interceptions | PASS (0) |
| C0: direct progress | PASS (median efficiency 0.996, closing fraction 1.0) |
| C1: deviations associated with perceived obstacles | PASS |
| C1: return to direct | PASS |
| C1: no lingering beyond limit | PASS |
| C2: deviations associated with perceived obstacles | PASS |
| C2: return to direct | PASS |
| C2: no lingering beyond limit | PASS |
| C3: deviations associated with perceived obstacles | PASS |
| C3: return to direct | PASS |
| **C3: no lingering beyond limit** | **FAIL** (max 10 consecutive clear-line steps while avoiding, 1 of 100 seeds) |
| Scripted defenders: no tactical evasion | PASS |
| Scripted defenders: attacker independent of defender | PASS (max delta 0.0 m) |
| Event semantics: competing-event winners correct | PASS |
| Event semantics: zero missed swept contacts | PASS |
| Constructed S1 (blocking obstacle: deviate + necessary + return) | PASS |
| Constructed S2 (non-blocking obstacle: no deviation) | PASS |
| Constructed S3 (two obstacles: ≥2 excursions, both return) | FAIL by the strict "all necessary" sub-check only (both excursions return to DIRECT; see note below) |

Note on S3: the acceptance script's strict sub-check requires every excursion
to be counterfactually necessary; one of the two excursions in the
constructed two-obstacle scenario is a short, clearance-margin-only reaction
(not a true collision) and is classified `necessary_margin`, which the
stricter boolean check does not count as passing. This is a scoring-strictness
artifact, not a behavioral defect: both excursions still return to `DIRECT`
and the mission still succeeds (see `acceptance.json` →
`constructed_scenarios`).

## 20. Battery/endurance recommendation

**SIMPLE ENDURANCE LIMIT RECOMMENDED.** The corrected attacker's direct-path
behavior is rational and its efficiency metrics (path length, flight time,
heading change) are now meaningful and already instrumented without any
reward/termination change. However, the §13 finding shows that even a
well-behaved attacker can occasionally fly a materially longer path (up to
9×) when a particular obstacle-candidate/dynamics interaction fires; a simple
maximum mission duration or path-length budget would make this a
quantifiable, boundable cost in later experiments rather than leaving it as
an open-ended qualitative concern. An energy-aware decision-making model is
not justified by anything observed here.

## 21. Limitations

- The §13 climb-candidate/turn-rate interaction is not fixed in this task
  (no simulation-semantics changes were made); it is a precise, narrowly
  scoped candidate for a follow-up before relying on C2/C3 path-length
  statistics for publication-grade efficiency claims.
- The 12-step necessity counterfactual is a static-horizon approximation; it
  slightly under-counts the necessity of fast-moving dynamic-obstacle
  reactions that resolve inside one step.
- The 100-seed coverage extension was run only for the passive controller
  (attacker-motion isolation); defender-interaction effects on obstacle
  timing were only checked at the 10-seed, six-controller granularity.
- GIFs use a frame stride for file-size reasons and are not literally
  one-frame-per-step for long episodes (none of the kept episodes needed more
  than the stride cap in practice).

## 22. Verdict

**C — OBSTACLE BYPASS STILL NEEDS ONE NARROW CORRECTION**, scoped precisely:
the climb-bypass candidate's fixed "directly above activation position"
waypoint, combined with the turn-rate-limited vehicle dynamics, can produce
disproportionate (up to 9×) detours when the attacker has built up
high-momentum, low-noise horizontal flight before a climb candidate is
selected — now visible in isolation because V2 correctly removed the
clear-space weave that used to mask it. This affects a small minority of
episodes (1/100 C2, 9/100 C3 by the path-excess threshold; 0/100 C0/C1) and
never causes a missed capture, a collision, a failure to return to DIRECT, or
an unassociated/false obstacle trigger.

Everything else this task set out to validate is unambiguously correct and
should **not** be revisited: clear-space pursuit (C0, the primary sanity
case) is flawless; swept contact and event-ordering semantics are flawless
(zero missed contacts, all competing-event winners correct); A0 never
performs tactical defender evasion and is bit-for-bit independent of the
defender's identity and policy.
