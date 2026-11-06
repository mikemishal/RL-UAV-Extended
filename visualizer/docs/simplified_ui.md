# Simplified two-scenario UI

Companion to `visualizer/docs/simplified_episode_backend.md` (the backend this UI consumes) and the existing
`visualizer/README.md` (the full, historical replay/case-library interface, still available unchanged at
`/legacy`). This document covers only the new, default-root simplified research interface.

## 1. Layout

One screen, three columns (`SimplifiedApp.tsx`):

- **Left — control column**: scenario cards, defender dropdown, optional seed, Run Episode button.
- **Center — viewer column**: the trajectory replay (`TrajectoryViewer.tsx`), a compact current-episode summary,
  and a collapsible Details drawer.
- **Right — episode library column**: New/Saved tabs, filters, sort, and the episode list (with single/bulk
  delete).

A small header shows the page title, a `Simulation Ready` / `Simulation Unavailable` status indicator (backed by
periodic, lightweight `/api/health` polling), and a `Legacy interface` link to `/legacy`.

## 2. The two scenarios, and automatic attacker resolution

Exactly two large, keyboard-selectable scenario cards: **No Clutter** and **Clutter**, defaulting to **Clutter**
(the primary journal scenario). There is no separate attacker selector anywhere in this UI — the attacker is
always resolved from the scenario alone (`GET /api/scenario-presets`), and shown only as a subtle info line:
"Attacker: Scripted" (No Clutter) or "Attacker: Evasive CBF" (Clutter). Internal names (C0/C2/A0/A1-CBF/commit
SHAs/tags) never appear in the primary controls — they are available only in the collapsed Details drawer.

## 3. Defender, seed, and Run Episode

A single `Defender` dropdown lists all 9 supported defenders with friendly labels (Passive, Greedy, Kalman
Greedy, Proportional Navigation (PN), Lead, RMPC, PPO, PPO + Kalman, LR-PPO), defaulting to **LR-PPO**. Backend
defender availability (from the existing `/api/policies` catalog, used opportunistically) disables unavailable
options rather than silently substituting a different defender. A `Seed (optional)` field is left blank by
default (the backend generates one); a 🎲 Randomize button fills a random valid seed locally without a round
trip. `Run Episode` disables itself while a run is in flight (and whenever the selected defender is reported
unavailable) and shows `Running…`.

**Important**: generating an episode does **not** call `GET /api/episodes/{id}` — doing so would immediately
mark it viewed. A newly generated episode stays under **New** until the user explicitly chooses **Replay**.

## 4. New vs Saved workflow

Two tabs, `New (N)` / `Saved (N)`, defaulting to New if any new episodes exist on first load, otherwise Saved.
Each episode row shows a scenario badge (`No Clutter`/`Clutter` — never the internal `C0`/`C2` level), the
defender's friendly label, a friendly outcome label (Asset Captured / Defender Intercepted / Obstacle Collision /
Endurance Exhausted / Timeout / Ongoing — falling back to a humanized raw string for anything unrecognized), and
a `NEW` badge when applicable. Clicking **Replay** calls `GET /api/episodes/{id}` once: it loads the stored
trajectory into the viewer (never re-running the simulation) and, because the backend marks the episode viewed
on that same call, the episode moves from New to Saved and its `NEW` badge disappears — all without a full page
reload (the list is refreshed from the same response).

## 5. Replay viewer

Reuses the existing A1-CBF review trajectory player's rendering approach and CSS classes
(`cbf-review-scene`/`cbf-review-trail-*`/`cbf-review-marker-*` from `CbfDefenderReview.tsx`/`styles.css`),
extracted into a new, smaller, reusable `TrajectoryViewer.tsx` component (parameterized by the simplified
episode replay schema) — `CbfDefenderReview.tsx` itself is untouched. Shows attacker/defender/asset markers and
trails, static obstacles when present, and playback controls: restart, play/pause, a timeline scrubber, and a
0.5x/1x/2x/4x speed selector.

## 6. Current-episode summary and Details drawer

Below the viewer, a compact summary always shows scenario, attacker label, defender, seed, outcome, duration,
and steps, plus minimum defender separation / minimum obstacle clearance / path length when available. A
collapsed-by-default **Details** drawer reveals reproducibility metadata on demand: resolved attacker id, source
commit, controller version/tag (e.g. `journal-cbf-attacker-v1`), the exact backend defender id, the episode id,
and the three boolean outcome flags (asset captured / defender interception / obstacle collision).

## 7. Delete workflow

Every episode (New or Saved) has a delete (🗑) action, always behind the same confirmation dialog (a focus-trapped
modal, not the native `confirm()`), worded appropriately ("Delete this episode?" for New, "Delete this saved
episode? This removes its stored replay." for Saved). Confirming calls `DELETE /api/episodes/{id}`, removes the
row, clears the viewer if that episode was active, and refreshes the New/Saved counts.

In the Saved tab, a **Select** mode adds checkboxes, a **Select all visible** button, and **Delete selected (N)**
— which, after its own confirmation ("Delete N saved episodes? Their stored replay data will also be removed."),
calls `POST /api/episodes/delete` once with the selected ids. Per-id failures are surfaced ("Could not delete:
...") rather than assuming total success.

## 8. Filters and sorting

Saved (and New) support a `Scenario` filter (All / No Clutter / Clutter) and a `Defender` filter (All / the 9
defenders), both forwarded as query parameters to `GET /api/episodes`. Sort defaults to `Newest first`, with
`Oldest first` as the only other option. The list view never fetches full replay trajectories — only metadata —
so opening either tab stays cheap regardless of library size.

## 9. Empty states and errors

Clean, specific empty-state messages ("No new episodes. Run an episode to create one.", "No saved episodes
yet.", "No episodes match these filters.") replace blank lists. Run/replay/delete/backend-unavailable failures
surface as inline text or toast messages using the backend's own error message where available — never a raw
Python stack trace.

## 10. Legacy interface location

The full, historical interface (semantics V1/V2, C0–C3 clutter levels, attacker-policy/checkpoint selection,
curriculum attackers, comparison explorer, research case library, diagnostics, etc.) is completely unchanged and
remains available at **`/legacy`** (`main.tsx` routes on `window.location.pathname`; `App.tsx` itself was not
modified). The simplified root page links to it via a small, non-prominent "Legacy interface" link in the
header.
