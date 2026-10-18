# Phase 7 research case library

Research cases are local JSON sidecars that organize existing, validated replay
artifacts. A case stores opaque replay IDs, SHA-256 references to the compressed
replay files, researcher-authored metadata, and optional timestamped bookmarks.
It does not copy, regenerate, or modify an episode. Case files live below the
ignored `visualizer/data/local/cases/` directory; deleting a case removes only
its sidecar.

## Integrity and comparison status

Case create, update, delete, bookmark, and list operations are served by the
local FastAPI application. Mutations use an expected sidecar version and reject
stale edits. Replay references are checked against the current compressed-file
hash before export or capture; a mismatch blocks evidence generation instead
of silently accepting changed data.

Comparison cases use the Phase 6 replay-pair validator. `matched` denotes the
documented matching invariants and clean provenance; it does not establish
causality or couple future random trajectories. `exploratory` remains explicit
for dirty sources, recorder/runtime differences, or permitted estimator
differences. `incompatible` identifies cases that fail required pair checks.
The API preserves each comparison reason for display and export.

Bookmarks record shared-axis time, replay-local times, frame and decision
steps, absolute or detection-relative synchronization, optional observation
and interpretation, and tags. The API verifies the referenced frame/decision
interval and maps comparison-axis time to bounded local replay times. Empty
research interpretation fields remain empty; the software does not create
scientific conclusions.

## Reused Phase 6 comparison cases

Three case sidecars were saved from existing seed-zero replays. All use A1-203,
C3, `map_local`, and evaluation seed 0. The LR-PPO D1 replay
`32b7d3bf27c755da0d73821c9c495603` is reused across all three comparisons.

| Case ID | Pair | Existing replay IDs | Outcomes / durations | Case status |
|---|---|---|---|---|
| `395eba85acb041a59bad313e4b56ab36` | LR-PPO D0 seed 77 vs LR-PPO D1 seed 77 | `8da13f073f6e6838892af7c8d46c7b78`, `32b7d3bf27c755da0d73821c9c495603` | `soldier_caught` 760.0 s vs `intercepted` 486.5 s | Exploratory |
| `fa6e378eb71f43e0b24c4da7031d182e` | LR-PPO D1 seed 77 vs PPO-Kalman D1 seed 74 | `32b7d3bf27c755da0d73821c9c495603`, `54953a906c037857cb46f605435ba8f2` | `intercepted` 486.5 s vs `intercepted` 442.5 s | Exploratory |
| `b6402c6d691140f98209216bb90d14d3` | Lead vs LR-PPO D1 seed 77 | `187f2af642e023239b5416964579bcfa`, `32b7d3bf27c755da0d73821c9c495603` | `soldier_caught` 27.0 s vs `intercepted` 486.5 s | Exploratory |

The D1 / PPO-Kalman pair reports its estimator-contract and full environment
fingerprint differences. All three cases report dirty source provenance and
replay-recorder revision differences. They are useful for recorded-behavior
inspection, not clean controlled comparisons. Phase 7 reused four distinct
replay artifacts across six case references and generated **zero** episodes.

## Scene capture

From a saved case, capture either referenced replay at its currently selected
local time. The frontend restores that actual replay into the single-episode
Three.js view, reads the current WebGL scene into an on-demand 2560 × 1440 PNG,
and posts the PNG with replay ID, replay hash, time, camera mode and projection.
The API verifies case membership and artifact integrity before saving a
visualization-only capture under the case's ignored export directory.

The capture renders the Three.js scene only; DOM-rendered Drei `Html` labels
and other surrounding application chrome are not part of the WebGL image.
Capture does not enable global `preserveDrawingBuffer`, alter the replay, or
introduce browser-side dynamics. A real capture is stored under:

`visualizer/data/local/exports/395eba85acb041a59bad313e4b56ab36/fc7d2e007f474cf4bc1d11c6c778eaa9/battlefield-scene.png`

The capture API and renderer behavior are described in
[scientific exports](scientific_exports.md).
