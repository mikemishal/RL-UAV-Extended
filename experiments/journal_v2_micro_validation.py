"""Phase 14: legacy-vs-corrected A0 behavioral micro-validation.

NOT a publication evaluation and NOT a defender-performance study. It only
exposes the ATTACKER's motion: the defender is parked far from the engagement
(standby disabled so it is never synchronized onto the asset) and never acts,
so interceptions cannot pre-empt attacker-asset contact.

For each canonical clutter level it runs the same seeds under
  * V1 semantics  (endpoint contact, weave/evasion A0), and
  * V2 semantics  (swept contact, direct/obstacle-only A0),
and reports missed sphere crossings, capture, path, heading, overshoot/orbit
and obstacle-deviation metrics.

Seed namespace: 300,000-300,049 (diagnostic only; never reuse for
publication evaluation).
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.swept_contact import first_contact

from experiments.journal_v2_protocol import V2_RESULTS_DIR, build_v1_env_config, build_v2_env_config

OUT_DIR = V2_RESULTS_DIR / "micro_validation"
MICRO_VALIDATION_SEED_START = 300_000
N_SEEDS = 50
SEEDS = tuple(range(MICRO_VALIDATION_SEED_START, MICRO_VALIDATION_SEED_START + N_SEEDS))
LEVELS = ("C0", "C2", "C3")
ORBIT_RANGE_M = 20.0
PARKED_DEFENDER = (-45.0, -45.0, 29.0)
HARNESS_OVERRIDES = {"defender_standby_until_detection": False}
BUILDERS = {"V1_legacy": build_v1_env_config, "V2_corrected": build_v2_env_config}


def run_episode(level: str, semantics: str, seed: int) -> dict:
    cfg = dataclasses.replace(BUILDERS[semantics](level, "measurement"), **HARNESS_OVERRIDES)
    env = SoldierEnv(cfg)
    env.enemy_tactical_override = None
    _, info = env.reset(seed=seed)
    env._defender_pos = np.asarray(PARKED_DEFENDER, dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)

    R = cfg.threat_radius
    missed_crossings = 0
    deviation_flags: list[bool] = []
    mission_angle_after: list[float] = []
    bearing_prev = None
    orbit_angle = 0.0
    done = False
    steps = 0
    while not done:
        asset_prev = env._soldier_pos.copy()
        enemy_prev = env._enemy_pos.copy()
        _, _, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        done = terminated or truncated
        steps += 1

        # Independent audit of the endpoint-contact failure mode: did the
        # attacker's segment cross the capture sphere on a step that did NOT
        # end in capture?
        endpoint_dist = float(np.linalg.norm(info["enemy_pos"] - info["soldier_pos"]))
        crossed = first_contact(enemy_prev, info["enemy_pos"], asset_prev, info["soldier_pos"], R) is not None
        if crossed and info["outcome"] != "soldier_caught" and endpoint_dist > R:
            missed_crossings += 1

        deviation_flags.append(info["enemy_behavior_mode"] != "MISSION")
        final_cmd = np.asarray(info["enemy_final_command"], dtype=np.float64)
        mission_cmd = np.asarray(info["enemy_mission_command"], dtype=np.float64)
        denom = np.linalg.norm(final_cmd) * np.linalg.norm(mission_cmd)
        mission_angle_after.append(
            float(np.degrees(np.arccos(np.clip(np.dot(final_cmd, mission_cmd) / denom, -1, 1)))) if denom > 0 else 0.0
        )

        rel = np.asarray(info["enemy_pos"], dtype=np.float64) - np.asarray(info["soldier_pos"], dtype=np.float64)
        if float(np.linalg.norm(rel)) <= ORBIT_RANGE_M and float(np.hypot(rel[0], rel[1])) > 1e-6:
            bearing = math.atan2(rel[1], rel[0])
            if bearing_prev is not None:
                orbit_angle += (bearing - bearing_prev + math.pi) % (2 * math.pi) - math.pi
            bearing_prev = bearing
        else:
            bearing_prev = None

    deviation_steps = int(sum(deviation_flags))
    segments = sum(1 for i, f in enumerate(deviation_flags) if f and (i == 0 or not deviation_flags[i - 1]))
    last_dev = max((i for i, f in enumerate(deviation_flags) if f), default=None)
    after = mission_angle_after[last_dev + 1:] if last_dev is not None else mission_angle_after
    flight_time = info["attacker_flight_time_s"]
    return {
        "level": level, "semantics": semantics, "seed": seed, "outcome": info["outcome"], "steps": steps,
        "captured": info["outcome"] == "soldier_caught",
        "flight_time_s": flight_time,
        "path_length_m": info["attacker_path_length"],
        "path_efficiency": info["attacker_path_efficiency"],
        "heading_change_deg": info["attacker_heading_change_total_deg"],
        "mean_heading_rate_deg_s": info["attacker_heading_change_total_deg"] / flight_time if flight_time > 0 else float("nan"),
        "overshoots": info["attacker_overshoot_count"],
        "orbit_revolutions_within_20m": abs(orbit_angle) / (2 * math.pi),
        "missed_crossings": missed_crossings,
        "deviation_steps": deviation_steps,
        "deviation_segments": segments,
        "post_deviation_mean_angle_to_mission_deg": float(np.mean(after)) if after else 0.0,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (level, semantics), g in df.groupby(["level", "semantics"], sort=False):
        cap = g[g.captured]
        rows.append({
            "level": level, "semantics": semantics, "n": len(g),
            "capture_rate": g.captured.mean(),
            "obstacle_collision_rate": g.outcome.str.contains("obstacle_collision").mean(),
            "timeout_rate": (g.outcome == "timeout").mean(),
            "missed_crossings_total": int(g.missed_crossings.sum()),
            "episodes_with_missed_crossing": int((g.missed_crossings > 0).sum()),
            "median_steps_captured": cap.steps.median() if len(cap) else float("nan"),
            "median_path_length_m": g.path_length_m.median(),
            "median_path_efficiency": g.path_efficiency.median(),
            "mean_heading_change_deg": g.heading_change_deg.mean(),
            "mean_heading_rate_deg_s": g.mean_heading_rate_deg_s.mean(),
            "mean_overshoots": g.overshoots.mean(),
            "episodes_with_overshoot": int((g.overshoots > 0).sum()),
            "mean_orbit_revolutions_within_20m": g.orbit_revolutions_within_20m.mean(),
            "episodes_with_orbit_ge_1_rev": int((g.orbit_revolutions_within_20m >= 1.0).sum()),
            "mean_deviation_steps": g.deviation_steps.mean(),
            "mean_deviation_segments": g.deviation_segments.mean(),
            "post_deviation_mean_angle_to_mission_deg": g.post_deviation_mean_angle_to_mission_deg.mean(),
        })
    return pd.DataFrame(rows)


def _run_cell(args: tuple[str, str]) -> list[dict]:
    level, semantics = args
    return [run_episode(level, semantics, seed) for seed in SEEDS]


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cells = [(level, semantics) for level in LEVELS for semantics in BUILDERS]
    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(len(cells), os.cpu_count() or 1)) as pool:
        for cell, cell_records in zip(cells, pool.map(_run_cell, cells)):
            records.extend(cell_records)
            print(f"done {cell[0]} {cell[1]}", flush=True)
    df = pd.DataFrame(records)
    summary = summarize(df)
    tmp = OUT_DIR / "episodes.csv.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, OUT_DIR / "episodes.csv")
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps({
        "seed_range": [SEEDS[0], SEEDS[-1]], "n_seeds_per_cell": N_SEEDS, "levels": list(LEVELS),
        "harness": "defender parked at %s, standby disabled, zero action" % (PARKED_DEFENDER,),
        "summary": json.loads(summary.to_json(orient="records")),
    }, indent=2))
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(summary.round(3).T.to_string())


if __name__ == "__main__":
    main()
