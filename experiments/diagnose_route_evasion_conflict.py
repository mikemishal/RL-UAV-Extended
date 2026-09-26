"""
Phase 4: reproduce and classify the route-planning-vs-evasion failure
mechanism using the command-conflict diagnostics added in Phase 3 (never
inferred from trajectories alone).

Uses the SAME seeds (80000-80014) and clutter levels (C1-C3) as the Phase
20-25 matched Monte Carlo study, evasion enabled, comparing
E_route_planner_levelB against B_reactive_only, and drills into the
"route ON substantially worse than route OFF" episodes.

Usage:
    python -m experiments.diagnose_route_evasion_conflict
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from experiments.clutter_benchmark_protocol import CLUTTER_LEVELS
from uav_defend.config.env_config import EnvConfig
from uav_defend.envs import SoldierEnv

SEEDS = tuple(range(80000, 80015))
LEVELS = ("C1", "C2", "C3")


def _run(config: EnvConfig, seed: int, max_steps: int = 2000) -> dict:
    env = SoldierEnv(config=config)
    obs, info = env.reset(seed=seed)
    start_pos = info["enemy_pos"].copy()
    path_length = 0.0
    prev = info["enemy_pos"].copy()
    max_angle_evasion_route = 0.0
    replans_while_evading = 0
    waypoint_invalidations = 0
    resumptions = 0
    obstacle_conflicts_during_evasion = 0
    mode_switch_count = 0
    step = 0
    for step in range(max_steps):
        obs, reward, done, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        path_length += float(np.linalg.norm(info["enemy_pos"] - prev))
        prev = info["enemy_pos"].copy()
        if info["enemy_angle_evasion_route_deg"] is not None:
            max_angle_evasion_route = max(max_angle_evasion_route, info["enemy_angle_evasion_route_deg"])
        replans_while_evading = info["enemy_replans_while_evading"]
        waypoint_invalidations = info["enemy_waypoint_invalidations_from_evasion"]
        resumptions = info["enemy_route_resumptions_after_evasion"]
        obstacle_conflicts_during_evasion = info["enemy_obstacle_conflicts_during_evasion"]
        mode_switch_count = info["enemy_behavior_mode_switch_count"]
        if done or truncated:
            break
    direct = float(np.linalg.norm(info["soldier_pos"] - start_pos))
    return dict(
        outcome=info["outcome"], steps=step + 1,
        detour_ratio=path_length / direct if direct > 1e-6 else float("nan"),
        max_angle_evasion_route=max_angle_evasion_route,
        replans_while_evading=replans_while_evading,
        waypoint_invalidations_from_evasion=waypoint_invalidations,
        route_resumptions_after_evasion=resumptions,
        obstacle_conflicts_during_evasion=obstacle_conflicts_during_evasion,
        mode_switch_count=mode_switch_count,
    )


def _config(level: str, routed: bool) -> EnvConfig:
    return EnvConfig(
        **CLUTTER_LEVELS[level], dynamic_obstacles_enabled=False,
        enemy_obstacle_avoidance_enabled=True, enemy_obstacle_awareness_mode="local",
        enemy_evasion_enabled=True, enemy_route_planning_enabled=routed,
    )


def classify(row: dict) -> str:
    """Categorize the failure mechanism using ONLY command diagnostics
    (Phase 4 requirement -- never infer from trajectory alone)."""
    if row["waypoint_invalidations_from_evasion"] >= 5:
        return "B. route replans repeatedly because evasion invalidates the planned corridor"
    if row["max_angle_evasion_route"] > 150.0 and row["replans_while_evading"] >= 3:
        return "C. evasion pushes attacker opposite the route's chosen side"
    if row["obstacle_conflicts_during_evasion"] >= 5:
        return "D. evasion causes repeated obstacle-cluster re-entry"
    if row["route_resumptions_after_evasion"] == 0 and row["replans_while_evading"] >= 3:
        return "A. route waypoint becomes stale after evasion (never cleanly resumed)"
    if row["mode_switch_count"] > row["steps"] * 0.3:
        return "G. high-frequency mode oscillation (ROUTE<->EVADE ping-pong)"
    return "G. other/undetermined (see raw diagnostics)"


def run_phase4() -> None:
    findings = []
    for level in LEVELS:
        config_b = _config(level, routed=False)
        config_e = _config(level, routed=True)
        for seed in SEEDS:
            result_b = _run(config_b, seed)
            result_e = _run(config_e, seed)
            worse = result_e["detour_ratio"] > result_b["detour_ratio"] * 1.5
            print(f"level={level} seed={seed} B_detour={result_b['detour_ratio']:.1f} "
                  f"E_detour={result_e['detour_ratio']:.1f} worse={worse}", flush=True)
            if worse:
                category = classify(result_e)
                findings.append(dict(level=level, seed=seed, category=category, **result_e))
                print(f"  -> CATEGORY: {category} | diagnostics={result_e}", flush=True)

    print("\n=== Phase 4 category summary ===")
    counts = Counter(f["category"] for f in findings)
    for category, count in counts.most_common():
        print(f"{count:3d}  {category}")
    print(f"\nTotal failure episodes classified: {len(findings)} / {len(SEEDS) * len(LEVELS)}")


if __name__ == "__main__":
    run_phase4()
