"""
No-obstacle (C0) robustness-sweep reproduction for the D0/D0+/D1
defender-adaptation checkpoints (journal "accepted-paper reproduction"
sub-task, Phase 5/6).

Reproduces the accepted paper's 4 named 1-D robustness sweeps (detection
radius, measurement noise, defender speed, hostile speed) plus the 25-cell
2-D defender x hostile speed feasibility grid -- using the EXACT x-axis
values and field names from the locked protocol files
(`robustness_detection_protocol.py`, `robustness_measurement_noise_protocol.py`,
`robustness_mobility_protocol.py`, `robustness_mobility_grid_protocol.py`;
see `docs/original_paper_protocol_audit.md` Section 2) -- against the
original scripted hostile (A0-paper; `attacker_model=None`, never the
Adaptive Attacker v1 substitutes, per Phase 7) -- for the 27 D0/D0+/D1
defender instances, under C0 (zero obstacles).

SCOPE DECISION (documented in `docs/original_paper_protocol_audit.md`
Section 5): the original sweeps used N=500-1,000 episodes/point across 6
scripted+2 PPO controllers. Reproducing at that exact N for 27 NEW
defender instances across 21 sweep points + 25 grid cells would be
69,000+ fresh episodes for the 1-D sweeps alone. This module uses a
reduced-but-real N (explicitly labeled, never silently substituted for
the original N when reporting the ORIGINAL paper's own numbers):
N_SWEEP_EPISODES=50 per 1-D sweep point, N_GRID_EPISODES=30 per grid
cell. All x-axis parameter VALUES are identical to the original protocol;
only the per-point episode count is reduced for this new, additional
D0/D0+/D1 extension experiment.

No training or model modification happens anywhere in this module.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.registry import get_policy

from experiments.journal_clutter_protocol import build_canonical_env_config
from experiments.robustness_detection_protocol import DETECTION_RADIUS_VALUES, NOMINAL_DETECTION_RADIUS
from experiments.robustness_measurement_noise_protocol import MEASUREMENT_VAR_VALUES, NOMINAL_MEASUREMENT_VAR
from experiments.robustness_mobility_protocol import (
    DEFENDER_SPEED_VALUES, HOSTILE_SPEED_VALUES,
    NOMINAL_DEFENDER_SPEED, NOMINAL_HOSTILE_SPEED,
    MOBILITY_CONDITIONS,
)
from experiments.robustness_mobility_grid_protocol import GRID_CELLS

from experiments.defender_adaptation_evaluation import (
    PROJECT_ROOT,
    MODEL_REGISTRY_PATH,
    load_model_registry,
    build_model_registry,
    verify_checkpoint_integrity,
    build_defender_instances,
    _run_matched_episode,
    _config_hash,
    _eval_git_commit,
    load_completed_episodes,
    append_episode_result,
)

OUT_DIR = PROJECT_ROOT / "results" / "defender_adaptation_c0_robustness"

C0_LEVEL = "C0"

# Reduced-but-real N per sweep point / grid cell (see module docstring).
N_SWEEP_EPISODES = 50
N_GRID_EPISODES = 30

# New, disjoint seed namespace (per docs/original_paper_protocol_audit.md
# Section 4: next free 10,000-aligned block after C0_EVAL_SEED_START=100_000
# reserving 100,000-100,099). This module reserves 110,000-110,999 for the
# 1-D sweeps (only the first N_SWEEP_EPISODES=50 of each 1,000-wide block
# are used, same seeds reused identically across all 21 sweep points/instances
# -- common-random-numbers, matching the original protocol's own convention)
# and 120,000-120,999 for the 2-D grid (first N_GRID_EPISODES=30 used).
SWEEP_SEED_START = 110_000
SWEEP_SEEDS: tuple[int, ...] = tuple(range(SWEEP_SEED_START, SWEEP_SEED_START + N_SWEEP_EPISODES))

GRID_SEED_START = 120_000
GRID_SEEDS: tuple[int, ...] = tuple(range(GRID_SEED_START, GRID_SEED_START + N_GRID_EPISODES))


# =============================================================================
# Sweep point definitions (reusing the ORIGINAL protocol's x-axis values).
# =============================================================================

def _detection_radius_points() -> list[dict]:
    return [
        {"sweep": "detection_radius", "x_value": r, "overrides": {"detection_radius": r},
         "is_nominal": (r == NOMINAL_DETECTION_RADIUS)}
        for r in DETECTION_RADIUS_VALUES
    ]


def _measurement_noise_points() -> list[dict]:
    return [
        {"sweep": "measurement_noise", "x_value": v, "overrides": {"measurement_var": v},
         "is_nominal": (v == NOMINAL_MEASUREMENT_VAR)}
        for v in MEASUREMENT_VAR_VALUES
    ]


def _mobility_points() -> list[dict]:
    points = []
    for c in MOBILITY_CONDITIONS:
        points.append({
            "sweep": f"mobility_{c.sweep_axis}" if c.sweep_axis != "nominal" else "mobility_nominal",
            "x_value": c.defender_speed if c.sweep_axis == "defender_speed" else c.hostile_speed,
            "overrides": {"v_d": c.defender_speed, "v_e": c.hostile_speed},
            "is_nominal": c.sweep_axis == "nominal",
        })
    return points


SWEEP_POINTS: list[dict] = _detection_radius_points() + _measurement_noise_points() + _mobility_points()
EXPECTED_N_SWEEP_POINTS = len(DETECTION_RADIUS_VALUES) + len(MEASUREMENT_VAR_VALUES) + len(MOBILITY_CONDITIONS)
assert len(SWEEP_POINTS) == EXPECTED_N_SWEEP_POINTS == 21


def _grid_points() -> list[dict]:
    return [
        {"sweep": "mobility_grid", "x_value": f"{cell.defender_speed}/{cell.hostile_speed}",
         "overrides": {"v_d": cell.defender_speed, "v_e": cell.hostile_speed},
         "is_nominal": (cell.defender_speed == NOMINAL_DEFENDER_SPEED and cell.hostile_speed == NOMINAL_HOSTILE_SPEED)}
        for cell in GRID_CELLS
    ]


GRID_POINTS: list[dict] = _grid_points()
assert len(GRID_POINTS) == 25


def _point_env_config(point: dict, estimator: str) -> SoldierEnv:
    cfg = build_canonical_env_config(C0_LEVEL, estimator, extra_overrides=point["overrides"])
    return cfg


def _episode_key(d, sweep_name: str, x_value, eval_seed: int) -> str:
    return "|".join(str(x) for x in (
        d.architecture, d.defender_seed, d.generation, d.checkpoint_role, sweep_name, x_value, eval_seed,
    ))


EPISODE_KEY_FIELDS = ("architecture", "defender_seed", "generation", "checkpoint_role", "sweep", "x_value", "eval_seed")


def _row_key(row: dict) -> str:
    return "|".join(str(row[f]) for f in EPISODE_KEY_FIELDS)


def _run_point(jsonl_path: Path, completed: dict, d, points: list[dict], seeds: tuple[int, ...],
                lead_policy_cache: dict, eval_code_commit: str, rows: list, counters: dict) -> None:
    for point in points:
        base_config = _point_env_config(point, d.estimator)
        defender_policy = d.build(base_config)
        lead_policy_for_residual = None
        if d.is_lr_ppo:
            cache_key = f"{point['sweep']}:{point['x_value']}:{d.estimator}"
            if cache_key not in lead_policy_cache:
                lead_policy_cache[cache_key] = get_policy("lead", config=base_config)
            lead_policy_for_residual = lead_policy_cache[cache_key]
        env = SoldierEnv(base_config)
        t0 = time.time()
        n_new = 0
        for eval_seed in seeds:
            key = _episode_key(d, point["sweep"], point["x_value"], eval_seed)
            if key in completed:
                counters["skipped"] += 1
                continue
            metrics = _run_matched_episode(env, d, defender_policy, lead_policy_for_residual, None, eval_seed)
            metrics.update(
                architecture=d.architecture, defender_seed=d.defender_seed, generation=d.generation,
                checkpoint_role=d.checkpoint_role, sweep=point["sweep"], x_value=point["x_value"],
                is_nominal=point["is_nominal"], eval_seed=eval_seed, eval_code_commit=eval_code_commit,
            )
            append_episode_result_local(jsonl_path, metrics, completed)
            rows.append(metrics)
            counters["executed"] += 1
            n_new += 1
        if n_new:
            print(f"{d.architecture:>22s} seed={d.defender_seed} {d.generation:<8s} "
                  f"sweep={point['sweep']:<20s} x={point['x_value']} (+{n_new} new, {time.time()-t0:.1f}s)", flush=True)


def append_episode_result_local(jsonl_path: Path, row: dict, completed: dict) -> None:
    key = _row_key(row)
    if key in completed:
        if completed[key] != row:
            raise ValueError(f"refusing to append duplicate episode key {key!r} with different content")
        return
    import os
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    completed[key] = row


def load_completed_episodes_local(jsonl_path: Path) -> dict:
    completed: dict = {}
    if not jsonl_path.exists():
        return completed
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            is_last = (i == len(lines) - 1)
            if not is_last:
                raise ValueError(f"malformed JSONL line {i + 1} in {jsonl_path} is NOT the final line.")
            continue
        for field in EPISODE_KEY_FIELDS:
            if field not in row:
                continue
        key = _row_key(row)
        if key in completed and completed[key] != row:
            raise ValueError(f"duplicate episode key {key!r} in {jsonl_path} with DIFFERING content")
        completed[key] = row
    return completed


def evaluate_sweeps(jsonl_path: Path | None = None, registry: dict | None = None) -> pd.DataFrame:
    jsonl_path = jsonl_path if jsonl_path is not None else (OUT_DIR / "sweep_episodes.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    registry = registry if registry is not None else load_model_registry()
    completed = load_completed_episodes_local(jsonl_path)
    defenders = build_defender_instances(registry)
    lead_policy_cache: dict = {}
    eval_code_commit = _eval_git_commit()
    rows = list(completed.values())
    counters = {"executed": 0, "skipped": 0}
    for d in defenders:
        _run_point(jsonl_path, completed, d, SWEEP_POINTS, SWEEP_SEEDS, lead_policy_cache, eval_code_commit, rows, counters)
    print(f"=== sweep loop done: {counters['executed']} executed, {counters['skipped']} skipped ===", flush=True)
    return pd.DataFrame(rows)


def evaluate_grid(jsonl_path: Path | None = None, registry: dict | None = None) -> pd.DataFrame:
    jsonl_path = jsonl_path if jsonl_path is not None else (OUT_DIR / "grid_episodes.jsonl")
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    registry = registry if registry is not None else load_model_registry()
    completed = load_completed_episodes_local(jsonl_path)
    defenders = build_defender_instances(registry)
    lead_policy_cache: dict = {}
    eval_code_commit = _eval_git_commit()
    rows = list(completed.values())
    counters = {"executed": 0, "skipped": 0}
    for d in defenders:
        _run_point(jsonl_path, completed, d, GRID_POINTS, GRID_SEEDS, lead_policy_cache, eval_code_commit, rows, counters)
    print(f"=== grid loop done: {counters['executed']} executed, {counters['skipped']} skipped ===", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild-registry", action="store_true")
    parser.add_argument("--only", choices=["sweeps", "grid", "both"], default="both")
    args = parser.parse_args()

    if args.rebuild_registry or not MODEL_REGISTRY_PATH.exists():
        print("Building model registry (shared)...", flush=True)
        build_model_registry()

    registry = load_model_registry()
    print("Checkpoint integrity check (before)...", flush=True)
    problems = verify_checkpoint_integrity(registry)
    if problems:
        raise RuntimeError("Checkpoint integrity FAILED before evaluation:\n" + "\n".join(problems))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    expected_sweep_rows = 27 * EXPECTED_N_SWEEP_POINTS * N_SWEEP_EPISODES
    expected_grid_rows = 27 * 25 * N_GRID_EPISODES
    print(f"expected sweep rows: {expected_sweep_rows}; expected grid rows: {expected_grid_rows}", flush=True)

    if args.only in ("sweeps", "both"):
        df_sweep = evaluate_sweeps(OUT_DIR / "sweep_episodes.jsonl", registry)
        df_sweep.to_csv(OUT_DIR / "sweep_episodes.csv", index=False)
        print(f"wrote {len(df_sweep)} sweep rows (expected {expected_sweep_rows})", flush=True)
        if len(df_sweep) != expected_sweep_rows:
            raise RuntimeError(f"sweep incomplete: got {len(df_sweep)}, expected {expected_sweep_rows}")

    if args.only in ("grid", "both"):
        df_grid = evaluate_grid(OUT_DIR / "grid_episodes.jsonl", registry)
        df_grid.to_csv(OUT_DIR / "grid_episodes.csv", index=False)
        print(f"wrote {len(df_grid)} grid rows (expected {expected_grid_rows})", flush=True)
        if len(df_grid) != expected_grid_rows:
            raise RuntimeError(f"grid incomplete: got {len(df_grid)}, expected {expected_grid_rows}")

    print("Checkpoint integrity check (after)...", flush=True)
    problems = verify_checkpoint_integrity(registry)
    if problems:
        raise RuntimeError("Checkpoint integrity FAILED after evaluation:\n" + "\n".join(problems))
