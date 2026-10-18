# Phase 7 scientific exports

Exports are reproducibility aids derived from validated local replay bundles
and case-sidecar metadata. They are written below the ignored
`visualizer/data/local/exports/<case-id>/` directory. They never write into
research experiment or checkpoint directories.

## Export contents

The case workspace provides:

- Markdown case reports with policy/checkpoint identity, replay and source
  hashes, environment and seed provenance, outcomes, annotations, and explicit
  availability limitations.
- Single-episode or comparison CSV values taken from recorded frames and
  telemetry. Terminal rows do not invent policy actions or decisions.
- Trajectory SVG and telemetry SVG charts. Missing policy telemetry stays
  unavailable instead of being substituted with zero or reconstructed from
  unrecorded behavior.
- ZIP packages containing the requested report/data/charts, case annotations,
  and integrity manifest. Previously rasterized chart PNGs are included under
  `charts_png/`.

Optional finite increasing start/end bounds select a segment of the recorded
timeline. Comparison exports use the selected absolute or detection-relative
shared axis and map that interval independently to each replay's local clock.
Blank bounds retain the full recorded duration. Selected time ranges do not
truncate or rewrite source replay files.

## Chart PNGs

Chart SVGs are rasterized in the browser at 2100 pixels wide, corresponding to
seven inches at 300 DPI. The frontend preserves the SVG `viewBox` aspect ratio.
The backend verifies the source SVG SHA-256, expected dimensions, PNG chunk
checksums, and the 300-DPI `pHYs` metadata before saving a derived PNG. The
package ZIP includes these PNGs after rasterization; the SVG source remains
available alongside them.

## Native Three.js scene captures

Scene capture is an explicit user action from a saved case. The app loads the
chosen recorded replay, pauses playback, and renders the current scene to a
temporary 2560 × 1440 WebGL render target. Pixel readback is vertically
corrected and encoded as PNG. The previous render target, viewport, scissor,
scissor-test state, and camera projection are restored; the shared renderer's
global `preserveDrawingBuffer` option is not changed.

The API accepts the capture only when the replay ID and compressed artifact
SHA-256 match a reference in that unchanged case. It validates PNG dimensions
and checksums and records the replay hash, replay-local time, camera mode,
projection, resolution, and screenshot type in export metadata. Captures are
visualization artifacts, not simulation output. Drei HTML overlays are outside
the WebGL render target and therefore are not included in the PNG.

## Local verification and limitations

Focused API tests cover recorded-value export intervals, terminal action
omission, chart-source checksums, aspect-ratio enforcement, PNG validation,
300-DPI metadata, ZIP inclusion, scene-capture provenance, changed-replay
rejection, and case-only deletion. A live Chromium/API run captured the
existing LR-PPO D0/C3/A1-203 replay at 0.00 s; the saved WebGL PNG is
2560 × 1440 and 388,112 bytes. The local artifact paths are ignored by Git.
The live browser also rasterized six charts and produced package export
`104896408c1648eb9454f94638732ad8` (4,032,956 bytes); its ZIP contains all six
files below `charts_png/`. The frontend/API workflow completed with no browser
page errors.

The case-report output is a factual export, not an automatically generated
scientific interpretation or publication-ready statistical report. Derived
metrics and pair statuses retain the limitations documented in
[policy comparison](policy_comparison.md) and
[research cases](research_case_library.md).
