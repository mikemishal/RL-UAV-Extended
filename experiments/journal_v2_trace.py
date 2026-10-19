"""Per-step trajectory trace export for visual validation of V2 semantics.

Pure instrumentation: it reads `SoldierEnv.step()` info dicts and never
alters guidance, dynamics, reward or termination. The row schema is the
contract for the later visualization task.

Row fields (one row per environment step, plus a step-0 row):
    step, time_s
    asset_pos, attacker_pos, defender_pos, attacker_vel        (x, y, z)
    attacker_mission_direction     raw unit vector toward the asset
    attacker_final_command         command after obstacle handling
    behavior_mode                  DIRECT | OBSTACLE_AVOID | ROUTE_BYPASS | EVADE
    obstacle_blocking              reactive avoidance active this step
    predicted_obstacle_index / predicted_collision_step / predicted_min_clearance
    nearest_obstacle_clearance     attacker -> nearest static obstacle surface (m)
    perceived_static_ids           obstacle ids currently perceived by the attacker
    detected                       defender has acquired the attacker
    capture_contact_fraction / interception_contact_fraction / terminal_event
    outcome
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable

import numpy as np

BEHAVIOR_LABELS = {
    "MISSION": "DIRECT",
    "COLLISION_AVOID": "OBSTACLE_AVOID",
    "ROUTE": "ROUTE_BYPASS",
    "EVADE": "EVADE",
}


def _vec(value) -> list[float] | None:
    if value is None:
        return None
    return [float(x) for x in np.asarray(value, dtype=np.float64).reshape(-1)]


def trace_row(step: int, dt: float, info: dict) -> dict:
    """Convert one env info dict into a JSON-serialisable trace row."""
    clearance = info.get("enemy_min_obstacle_clearance")
    return {
        "step": int(step),
        "time_s": float(info.get("attacker_flight_time_s", step * dt)),
        "asset_pos": _vec(info["soldier_pos"]),
        "attacker_pos": _vec(info["enemy_pos"]),
        "defender_pos": _vec(info["defender_pos"]),
        "attacker_vel": _vec(info["enemy_vel"]),
        "attacker_mission_direction": _vec(info.get("enemy_mission_command")),
        "attacker_final_command": _vec(info.get("enemy_final_command")),
        "behavior_mode": BEHAVIOR_LABELS.get(info.get("enemy_behavior_mode"), str(info.get("enemy_behavior_mode"))),
        "obstacle_blocking": bool(info.get("enemy_obstacle_avoidance_active", False)),
        "predicted_obstacle_index": info.get("enemy_predicted_obstacle_index"),
        "predicted_collision_step": info.get("enemy_predicted_collision_step"),
        "predicted_min_clearance": _finite_or_none(info.get("enemy_predicted_min_clearance")),
        "nearest_obstacle_clearance": _finite_or_none(clearance),
        "perceived_static_ids": list(info.get("enemy_perceived_static_ids", ())),
        "detected": bool(info.get("enemy_detected", False)),
        "capture_contact_fraction": info.get("capture_contact_fraction"),
        "interception_contact_fraction": info.get("interception_contact_fraction"),
        "terminal_event": info.get("terminal_event"),
        "outcome": info.get("outcome"),
    }


def _finite_or_none(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def run_trace_episode(
    env,
    seed: int,
    defender_action: Callable[[object, dict], np.ndarray] | None = None,
    max_steps: int | None = None,
) -> list[dict]:
    """Run one episode on `env` and return its per-step trace rows.

    `defender_action(env, info)` returns the defender action each step; the
    default holds the defender passive (zero action).
    """
    _, info = env.reset(seed=seed)
    rows = [trace_row(0, env.config.dt, info)]
    limit = max_steps if max_steps is not None else env.config.max_steps
    for step in range(1, limit + 1):
        action = np.zeros(3, dtype=np.float32) if defender_action is None else defender_action(env, info)
        _, _, terminated, truncated, info = env.step(action)
        rows.append(trace_row(step, env.config.dt, info))
        if terminated or truncated:
            break
    return rows


def write_trace_jsonl(rows: list[dict], path: Path) -> None:
    """Atomic write (temp file + rename) of a trace as JSON Lines."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row))
            f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
