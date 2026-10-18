# Phase 6 policy comparison

Phase 6 adds a read-only comparison workspace to the existing local replay
viewer. It renders two real validated replay traces in separate Three.js
battlefields, synchronizes the display clock, and reuses Phase 5 inspectors and
diagnostics. The browser never runs simulation dynamics, launches a model,
performs inference, substitutes one episode's attacker path into another, or
edits a replay.

## Selection and availability

Policy A and Policy B have independent controller-family, generation, and
training-seed selectors. Options come from the live `/api/policies` registry;
unavailable policies, generations, or seeds are not selectable. For learned
policies, the generation advertises its manifest checkpoint role and the seed
option shows that role and the checkpoint SHA-256 prefix. That role is not
silently overridden. The replay-list record and downloaded trace are checked
against the expected role/hash. Analytical controllers do not have a model
checkpoint.

The shared scenario selection includes A0/A1-201/A1-202/A1-203, C0–C3,
perception mode, evaluation seed start, and 1/3/5/10 episodes. Only scenario
options supported by both chosen policies are eligible. The default is one
matched seed and two defender rollouts; the UI calculates and displays the
number of required simulations before the user starts anything. Evaluation
seeds are independent from the policies' training seeds.

## Replay matching and pair status

For each requested seed, replay discovery requires the exact defender policy,
generation, training seed, registry-selected checkpoint role/hash, attacker ID
and frozen attacker checkpoint hash, clutter level, perception, and evaluation
seed. Candidates are read from the existing replay index; no selection change
generates an episode. Each full bundle is loaded lazily and checked for schema
version, integrity, one-episode shape, at least two frames, selected metadata,
and the pair-level invariants below.

Pair status has three meanings:

- **Matched** — the pair passes all scientific compatibility checks with clean
  source provenance. This means shared verified initial conditions and
  compatible runtime/configuration, not shared future random trajectories.
- **Exploratory** — schema/integrity and physical initial conditions are valid,
  but an estimator contract/config differs or simulator/visualizer source
  provenance is dirty. The reason is shown; the application does not label the
  comparison controlled.
- **Incompatible** — a mismatch in a required invariant prevents comparison.
  The pair is rejected with an actionable reason rather than shown as matched.

The pair validator checks:

1. Same evaluation seed and frozen attacker identity/checkpoint SHA-256.
2. Same replay schema, coordinate convention, display convention, timestep,
   canonical clutter/perception/builder and deterministic-inference contract.
3. Same resolved physical environment config and recorded initial defender,
   attacker, protected-asset and obstacle geometry. Static geometry and
   first-recorded dynamic obstacle identity/position/velocity/radius are
   compared from the two traces.
4. Same simulation source commit, Python version, scientific package-version
   map and model inference device. A behavior-equivalent runtime is not
   assumed when these differ.
5. Exact selected defender and attacker checkpoint identities/roles/hashes.

The sole normalized physical-config exception is
`use_kalman_tracking`, an estimator-specific setting. It is reported as an
estimator/configuration difference and makes the pair exploratory; it is not
silently erased from the replay or forced to match. Any other physical
configuration mismatch is incompatible. Differences in estimator and
observation contracts are legitimate where required by the controller and
remain visible in each pane. The application never forces PPO-Kalman and
LR-PPO to share an estimator or leaks either controller's observations to the
other pane.

Dirty-source provenance is conservative. The comparison cases below were
recorded with dirty-source provenance and are therefore exploratory, even when
they share the same recorded simulator revision, checkpoint hashes, resolved
physical parameters and reset state. A matched evaluation seed does not prove
fully coupled common-random-number realizations. A defender's actions can
change subsequent environment evolution; a learned attacker may condition on
defender motion, and separate stochastic streams may diverge. The two views
always use their own authoritative attacker, asset, obstacle, and defender
states. No attacker or protected-asset path is synchronized or copied.

## Missing replay orchestration

Existing valid exact-match traces are reused. If either side is missing, the
researcher must explicitly activate **Generate missing replays**. The UI
reports required rollouts (two per seed) and only runs the missing A/B
episodes. It loops through requested seeds sequentially, generating A and then
B, validating both full artifacts, and retaining every successful replay
artifact. Completed seed pairs remain available if a later rollout fails; the
batch is visibly incomplete rather than treated as successful.

The orchestration calls the existing Phase 3 single-episode job API and Phase 2
CLI. It does not create a second process manager or comparison-specific
simulator. The established single-active-job guard also prevents a regular
replay job from running concurrently. Existing timeout, cancellation,
atomic-persistence and interrupted-job-recovery behavior remains in force.
Changing selectors or navigating seeds never starts a job. Persisted replay
artifacts can be rediscovered and paired after a page reload; comparison batch
state itself is not a separate scientific record.

## Synchronized 3D playback and cameras

Two independent instances of the existing `Battlefield` renderer use the same
scene scale, merged world bounds and coordinate mapping. Each receives its own
episode, selected sample, trajectories, obstacle states, terminal status and
diagnostics. The shared playback controls provide play, pause, restart,
back/forward frame step, scrub, jump to start/end and 0.25×/0.5×/1×/2×/4×/8×
rates.

**Absolute simulation time** is the default. Each replay frame is selected
using that episode's recorded timestamps and time-step metadata; matching
frame indices are not assumed. Display interpolation is clamped between that
episode's adjacent recorded states. The shared axis ends at the later terminal
time. An earlier episode is clamped to its actual last frame, explicitly marked
ended, and never extrapolated.

Optional **time relative to first detection** subtracts each episode's own
recorded first-detection time. If either trace lacks a valid detection time,
this mode is disabled with an explanation. Both local simulation times remain
visible; alignment is an analysis transform only and does not imply identical
world states.

Free-orbit and top-down cameras link compatible world-space poses by default;
updates are echo-suppressed to prevent OrbitControls feedback loops. The
researcher may unlink cameras for independent inspection. Follow-defender and
follow-attacker modes follow the corresponding entity in each pane, not the
other replay. Reset applies to both views.

## Diagnostics, charts, and descriptive metrics

The shared Ground Truth / Policy-Visible choice applies to both panes. Ground
Truth is explicitly researcher-only. Policy-Visible hides privileged attacker
truth, outcomes, collision/event details, privileged ranges and aggregate
outcome counts. It retains Phase 5's explicit partial-reconstruction
limitation. Each pane reads only its own policy observation/estimator
telemetry.

Where recorded, each pane shows defender-attacker and attacker-asset ranges,
actual movement, requested and safety-adjusted/executed actions, turn angle,
detection state, safety intervention and native outcome. LR-PPO retains the
Phase 5 Lead, final requested direction, actual movement, native 3D correction,
30-degree reference cone, and horizontal/elevation decompositions. It does not
invent raw PPO residuals or clipping internals. PPO-Kalman retains its own
policy-visible estimate, researcher-only privileged truth in Ground Truth
mode, and recorded/derived tracking diagnostics. Metrics not emitted by a
controller remain unavailable, not zero.

Comparative charts show:

1. Recorded defender-to-attacker range for A and B.
2. Recorded attacker-to-asset range for A and B.
3. Accumulated defender path length for A and B.
4. Requested action-turn angle where that controller records it.
5. Recorded safety intervention state over actual telemetry decision times.
6. Recorded detection and terminal events.
7. Available policy-specific diagnostics in separately labeled views.

The same timeline cursor drives both panels and all charts; chart selection
seeks the shared axis. No series generates points after its episode terminates.
In detection-relative mode chart x-coordinates use seconds relative to each
trace's own first detection.

Path length is a derived metric:

`sum(norm(position[i] - position[i-1]))`

over consecutive authoritative recorded defender positions in native 3D
meters. It does not sum interpolated render samples. First significant path
separation is a display-only threshold crossing over authoritative coordinates
at timestamps actually recorded in both traces; it is configurable from 1 to
50 meters (default 5 meters). This exploratory timestamp/threshold does not
identify a cause. Jump buttons exist only for recorded/validly-derived first
detection, path divergence, safety intervention, collision and terminal
events.

Each episode summary displays the native recorded outcome and whether it maps
to canonical `intercepted` success, duration, recorded detection/interception
times, derived path length, recorded collisions, available safety-decision
count and final privileged ranges. A−B duration/path differences are
descriptive only. The paired-seed table reports replay availability, pair
validation state, outcomes and duration differences for validated pairs.
Outcome counts only include fully loaded/validated pairs. Batch sizes 1/3/5/10
support exploratory inspection; the application does not calculate
publication-grade confidence intervals or rank controllers.

## Reproducibility and research interpretation

Preserve the replay files and their full `provenance` record when using these
views in research discussions. Record the pair status and reasons, policy and
attacker checkpoint roles/hashes, evaluation seed, resolved physical config,
initial-state equality, simulator commit, runtime versions, estimator
differences and whether source dirtiness was present. A seed match alone is not
evidence that later exogenous trajectories are identical. The viewer is a
diagnostic aid, not a causal analysis, paired statistical test, or replacement
for the canonical evaluation harness.

The native coordinates and recorded geometry remain authoritative. The scene
does not add altitude control, collision physics or vehicle dynamics. Rendering
and sample interpolation are presentation-only.

## Live verification and evidence

The actual local FastAPI service was started on `127.0.0.1:8000` with the
configured scientific Python interpreter and the research data root, while
Vite ran on `127.0.0.1:5173`. Existing user `.env` files were not overwritten.
The live check verified `/api/health`, `/api/policies`, `/api/scenarios`, and
the replay catalog/list. Full trace bundles were fetched from the live API;
Chromium compared both rendered frame-zero coordinates to those authoritative
bundles and exercised shared playback/step, scrub, terminal freeze,
detection-relative local-time mapping, linked/unlinked cameras, top-down mode,
per-pane LR-PPO diagnostic overlays, and Policy-Visible masking. A separate
opt-in browser run generated only the missing requested one-seed traces via
the live API and existing job worker.

Three requested seed-0 pairings used A1-203, C3, `map_local`, evaluation seed
0, and the same frozen attacker checkpoint:

| Comparison | Existing replay reused | Newly generated replay | Recorded outcomes |
|---|---|---|---|
| LR-PPO D0 seed 77 vs LR-PPO D1 seed 77 | LR-PPO D1 `final` | LR-PPO D0 `d0_start` | `soldier_caught` at 760.0 s vs `intercepted` at 486.5 s |
| LR-PPO D1 seed 77 vs PPO-Kalman D1 seed 74 | LR-PPO D1 `final` | PPO-Kalman D1 `final` | `intercepted` at 486.5 s vs `intercepted` at 442.5 s |
| Lead vs LR-PPO D1 seed 77 | LR-PPO D1 `final` | Lead | `soldier_caught` at 27.0 s vs `intercepted` at 486.5 s |

The existing LR-PPO D1 artifact was reused in all three pairs, alongside
exactly three newly generated single-episode rollouts. The PPO-Kalman pair's
estimator-only tracking flag differs. All inspected episodes report dirty
source provenance, so these pairings are shown as **exploratory**, not clean
controlled comparisons. Do not infer that their later attacker or asset
trajectories were identical.

Chromium screenshots are ignored local artifacts under
`frontend/test-results/`:

- `phase6-real-lrppo-d0-vs-d1.png`
- `phase6-real-lrppo-d1-vs-ppo-kalman-d1.png`
- `phase6-real-lead-vs-lrppo-d1.png`
- `phase6-real-synchronized-timeline-and-charts.png`
- `phase6-real-detection-aligned.png`
- `phase6-real-linked-top-down.png`
- `phase6-real-different-terminal-outcomes.png`
- `phase6-real-paired-episode-list.png`

The comparison tests include pure pairing/runtime/checkpoint matching,
initial-world rejection, actual timestamp alignment, unequal-duration
terminal clamping, camera/playback behavior, privacy masking, chart seeking,
divergence, replay immutability, and opt-in live API/browser and generation
flows. The README lists the exact local commands and which services must be
running.

## Phase boundary

Phase 6 comparison semantics remain unchanged. Three Phase 7 research-case
sidecars preserve the documented exploratory seed-zero comparisons using four
distinct existing replay artifacts; no Phase 7 traces were generated. The case
IDs, replay IDs, outcomes, and limitations are recorded in the
[research case library](research_case_library.md). Reports and comparison
exports remain factual summaries of the validated saved data; they do not
upgrade an exploratory pairing into a controlled or causal result.
