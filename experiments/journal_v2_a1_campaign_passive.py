"""Small passive-defender diagnostic for the V2 A1 policies (visualization only; NOT primary performance evidence).

The defender is parked far away and never acts (the research harness convention in
`journal_v2_validation_core.record_episode(controller="passive")`), so the attacker's own behavior is visible
without early defender interception. Scenario seeds are chosen from the primary evaluation records.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd

import experiments.journal_v2_a1_campaign_analysis as AN
import experiments.journal_v2_a1_campaign_eval as E
import experiments.journal_v2_a1_training as T
from experiments import journal_v2_a1 as A
from experiments.journal_v2_a1_evaluation import OBSTACLE_COLLISION_OUTCOMES
from experiments.journal_v2_validation_core import PARKED_DEFENDER, build_config
from uav_defend.envs.soldier_env import SoldierEnv

PASSIVE_LEVELS = ("C0", "C2", "C3")


def passive_episode(model, attacker_id: str, level: str, seed: int) -> dict:
    cfg, _estimator = build_config(level, "V2", "passive")
    env = SoldierEnv(cfg)
    pending = {"a": np.zeros(3, dtype=np.float32)}
    env.enemy_tactical_override = lambda: pending["a"]
    env.reset(seed=int(seed))
    env._defender_pos = np.asarray(PARKED_DEFENDER, dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    detection = float(cfg.enemy_defender_detection_range)
    start = np.asarray(env._enemy_pos, dtype=np.float64).copy()
    initial_range = float(np.linalg.norm(np.asarray(env._soldier_pos, dtype=np.float64) - start))
    done, steps, info, path = False, 0, {}, []
    while not done:
        obs = A.build_a1_v2_observation(env, detection)
        pending["a"] = np.asarray(model.predict(obs, deterministic=True)[0], dtype=np.float32).reshape(3)
        _o, _r, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        path.append(np.asarray(info["enemy_pos"], dtype=np.float64))
        done, steps = bool(terminated or truncated), steps + 1
    traveled = float(info["enemy_endurance_traveled_m"])
    chord = float(np.linalg.norm(path[-1] - start))
    return {"attacker": attacker_id, "level": level, "seed": int(seed), "defender": "passive", "outcome": info["outcome"], "steps": steps,
            "path_length_m": traveled, "path_excess": traveled / chord if chord > 1e-9 else float("nan"),
            "initial_range_m": initial_range, "captured": info["outcome"] == "soldier_caught",
            "obstacle_collision": info["outcome"] in OBSTACLE_COLLISION_OUTCOMES,
            "endurance_exhausted": info["outcome"] == "enemy_endurance_exhausted"}


def select_scenarios(df: pd.DataFrame, attacker_id: str, level: str) -> list[tuple[str, int]]:
    """(category, scenario seed): one ordinary, one obstacle-heavy, one high-path-excess episode at this level."""
    g = df[(df.attacker == attacker_id) & (df.level == level)].copy()
    g["pe"] = g.path_excess.replace([np.inf], np.nan)
    picks: list[tuple[str, int]] = []

    def add(category, row):
        if row is not None and int(row["seed"]) not in {s for _c, s in picks}:
            picks.append((category, int(row["seed"])))

    cap = g[g.captured]
    pool = cap if len(cap) else g
    ordinary = pool.iloc[(pool.pe - pool.pe.median()).abs().argsort().iloc[:1]] if len(pool) else pool
    add("ordinary", ordinary.iloc[0] if len(ordinary) else None)
    add("obstacle_heavy", g.sort_values(["avoidance_steps", "seed"], ascending=[False, True]).iloc[0] if len(g) else None)
    add("high_path_excess", g.sort_values(["pe", "seed"], ascending=[False, True]).iloc[0] if len(g) else None)
    return picks


def run_passive_diagnostics() -> list[dict]:
    records = E.scan_tasks(AN.derive)
    df = AN.frame(records)
    out = []
    for attacker_id in AN.A1_IDS:
        model = E._load_attacker(attacker_id)
        for level in PASSIVE_LEVELS:
            for category, seed in select_scenarios(df, attacker_id, level):
                result = passive_episode(model, attacker_id, level, seed)
                out.append({**result, "category": category})
    path = E.EVAL_ROOT / "passive_diagnostic.json"
    T.write_json_atomic(path, {"note": "visualization diagnostic only; not primary performance evidence", "episodes": out})
    return out


if __name__ == "__main__":
    for row in run_passive_diagnostics():
        print(row)
