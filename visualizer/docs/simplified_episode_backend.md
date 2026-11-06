# Simplified two-scenario episode backend

Companion to the existing `visualizer/README.md` (which documents the full, historical replay/case-library
system). This document covers ONLY the new, additive "simplified" API introduced to support exactly two
user-facing scenarios and simple episode management (New / Saved, replay, delete). It does not replace or
redesign any existing endpoint.

## 1. The two scenarios

The app exposes exactly two scenarios. The attacker is **always** auto-resolved from the scenario and is never a
user-facing selection.

| Scenario id | Label | Clutter level | Resolved attacker | Dynamic obstacles | Viability-conditioned init | Defender-CBF (defender avoidance) |
|---|---|---|---|---|---|---|
| `no_clutter` | No Clutter | C0 | A0 (scripted legacy pursuit) | off | off | n/a |
| `clutter` | Clutter | C2 (canonical) | A1-CBF (`journal-cbf-attacker-v1`, commit `a989217`) | off | on | on |

`GET /api/scenario-presets` returns this mapping verbatim for introspection/debugging; the simplified run API
never accepts an explicit attacker selection (`EpisodeCreateRequest` has no `attacker` field and rejects unknown
fields, so one cannot be smuggled in).

## 2. Attacker auto-resolution logic

`visualizer/backend/episode_models.py: SCENARIO_PRESETS` is the single source of truth. `episode_cli.py`
(the subprocess entry point that actually runs the simulation) branches ONLY on `--scenario`:

- `no_clutter` → `experiments.journal_v2_validation_core.build_config("C0", "V2", defender)` + plain
  `SoldierEnv(cfg)` (scripted A0, the research environment's default `enemy_control_mode="legacy"`).
- `clutter` → `experiments.journal_cbf_attacker_defender_foundation.defender_config(defender)` +
  `SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=True)`
  — the exact, frozen A1-CBF configuration validated and tagged `journal-cbf-attacker-v1`.

Neither attacker requires a trained-model checkpoint (A0 is scripted; A1-CBF is an optimization-based
controller), so there is no checkpoint registry/hash verification on the attacker side at all — unlike the
EXISTING `/api/replay-jobs` system, which replays trained attacker checkpoints via `ReplayRequest`/
`replay_engine.py`'s registry machinery. The two systems do not share an attacker code path.

## 3. Supported defenders

All defender ids already supported by the research/visualizer stack remain supported, unchanged:
`passive, greedy, kalman_greedy, pn, lead, rmpc, ppo, ppo_kalman, lr_ppo`. Scripted/analytical defenders
(`passive, greedy, kalman_greedy, pn, lead, rmpc`) are constructed directly with no checkpoint. The three
PPO-family defenders (`ppo, ppo_kalman, lr_ppo`) reuse the visualizer's OWN verified, sha256-checked checkpoint
resolution (`replay_engine._resolve_defender_checkpoint`/`_load_defender`), with ONE canonical, frozen
`(generation, seed)` default per architecture (the `"d1"` generation's first registered seed) — the simplified
API never asks the caller to choose a generation or training seed; "everything else comes from the preset."

## 4. Episode data model

`visualizer/backend/episode_models.py: EpisodeMetadata` — one record per episode:

```
episode_id, created_at, scenario, scenario_label, resolved_attacker, defender, seed, status, outcome,
duration, steps, asset_captured, defender_interception, obstacle_collision, replay_path, source_commit,
controller_version, is_new, min_defender_separation_m?, min_obstacle_clearance_m?, path_length_m?, error_message?
```

`source_commit`/`controller_version` provenance (brief section 15):
- `clutter` episodes: `source_commit = a989217b38dc0316718d04e41b065943fec29ce7`,
  `controller_version = "journal-cbf-attacker-v1"`.
- `no_clutter` episodes: `source_commit` = the research repository's live `git rev-parse HEAD` at generation
  time, `controller_version = "A0 (scripted legacy attacker, research environment baseline)"` — A0 has no
  separate research tag of its own, so none is invented.

The full step-by-step trajectory (`replay`) is stored SEPARATELY from the metadata record (one JSON per episode
under `replays/`), never duplicated inside the metadata.

## 5. New vs Saved semantics

- A NEW episode is one created this session (or any prior session) that has never been opened/replayed:
  `is_new=true`.
- A SAVED episode is any persisted episode that HAS been opened/replayed at least once: `is_new=false`.
- Episodes are persisted immediately on creation (`is_new=true`); there is no explicit "save" action — opening
  an episode (`GET /api/episodes/{id}`) marks it viewed (`is_new=false`) and this is idempotent (re-opening an
  already-viewed episode is a no-op, never an error).
- `GET /api/episodes?new=true|false` filters by this flag; the unfiltered list and `/api/health` both report
  `new_episode_count`/`saved_episode_count`.

## 6. Delete semantics

- `DELETE /api/episodes/{episode_id}` removes exactly that episode's metadata JSON and its own replay JSON, and
  nothing else. 404 if the id does not exist.
- `POST /api/episodes/delete` with `{"episode_ids": [...]}` performs the same deletion for each id in the list
  and returns a per-id `{episode_id, deleted, reason}` result; a missing id is reported as `deleted=false,
  reason="not_found"`, never an error for the whole batch.
- `EpisodeStore._safe_path` validates every episode id against a strict `^[0-9a-f]{32}$` pattern and re-resolves
  the final path against the store's own `metadata`/`replays` directories before any file operation, rejecting
  path traversal, symlinks, and any id that would resolve outside the store root.
- There is no "delete everything on disk" command. `EpisodeStore.delete_saved_older_than(cutoff_iso)` is an
  optional, NOT yet wired into an API route, helper that only ever deletes SAVED (`is_new=false`) episodes with
  `created_at < cutoff`; it is never exposed as a bulk-wipe and never touches research source, checkpoints, or
  any file outside its own `metadata`/`replays` directories.

## 7. Replay compatibility

The simplified API's `GET /api/episodes/{id}` always returns a `replay` object with the schema
`"app_episode_replay/1"` (`{schema, outcome, obstacles, steps}`, where each step has `t`, `attacker_pos`,
`defender_pos`, `asset_pos`) regardless of whether the episode came from the scripted A0 path or the A1-CBF
path — the frontend does not need to know which controller produced it. This is a deliberately small, new,
dedicated schema for app-generated episodes; it does NOT attempt to normalize the existing, much richer SB3
`ReplayBundle` schema (`/api/replays/{id}`) or the precomputed A1-CBF manual-review schema (`/api/cbf-review/*`)
into itself in this stage — those two existing, read-only paths remain fully available, unchanged, for
historical/research tooling.

## 8. Provenance / reproducibility

Every `clutter` episode is reproducible from `(scenario="clutter", defender, seed)` alone, against the frozen,
tagged `journal-cbf-attacker-v1` controller at commit `a989217`. Every `no_clutter` episode is reproducible
against the research repository's current A0 baseline at whatever commit was live at generation time (recorded
in `source_commit`).

## 9. Storage layout

`visualizer/data/episodes/` (a SEPARATE, additive directory from the existing `visualizer/data/local/`
replay-job/case-library area, which has its own checkpoint-verification-oriented schema):

```
visualizer/data/episodes/
  metadata/{episode_id}.json
  replays/{episode_id}.json
```

Every write is atomic: a temporary file in the SAME directory, then `os.replace` (mirroring `case_library.py`'s
established `_atomic_json` pattern), with a short retry loop against transient `PermissionError`s.

## 10. Backward compatibility

No existing endpoint was removed, renamed, or had its response schema narrowed. `HealthResponse` gained six new,
defaulted, additive fields (`research_source_available`, `a0_available`, `a1_cbf_available`,
`episode_storage_writable`, `new_episode_count`, `saved_episode_count`); every previously-existing field is
unchanged. `/api/replays`, `/api/cases`, `/api/replay-jobs`, `/api/policies`, `/api/scenarios`, and
`/api/cbf-review/*` are all untouched.

## 11. Health

`GET /api/health` additionally reports: `research_source_available` (the configured research root exists and
looks like a research worktree), `a0_available` (`uav_defend/envs/soldier_env.py` present),
`a1_cbf_available` (`uav_defend/attackers/cbf_obstacle.py` present), `episode_storage_writable` (a cheap
write-probe file, created and immediately removed), and the current new/saved episode counts. All of these are
cheap filesystem checks — none of them runs a simulation.
