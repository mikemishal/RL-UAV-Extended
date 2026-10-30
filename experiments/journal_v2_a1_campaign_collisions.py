"""Re-simulate every A1 obstacle-collision episode to identify the obstacle, its perception status and the
pre-collision sector observation (the evaluation trace stores no obstacle id: SoldierEnv reports None).

Deterministic: same model, same scenario seed, same defender as the evaluation. Read-only with respect to all results.
"""

from __future__ import annotations

import json

import numpy as np

import experiments.journal_v2_a1_campaign_analysis as AN
import experiments.journal_v2_a1_campaign_eval as E
from experiments import journal_v2_a1 as A

SECTORS = AN.SECTOR_NAMES


def box_distance(point, lo, hi) -> float:
    p, lo, hi = (np.asarray(x, dtype=np.float64) for x in (point, lo, hi))
    return float(np.linalg.norm(np.maximum(np.maximum(lo - p, 0.0), p - hi)))


def resim(model, attacker_id, level, family, seed) -> dict:
    env = A.A1V2TrainingEnv(0, 0, defender_bank=A.A1V2DefenderBank())
    obs, _ = env.reset(options={"scenario": (level, family, int(seed))})
    layout = env.soldier_env._obstacle_layout.obstacles
    history, done = [], False
    while not done:
        action = np.asarray(model.predict(obs, deterministic=True)[0], dtype=np.float32)
        before = obs.copy()
        obs, _r, terminated, truncated, info = env.step(action)
        history.append({"obs": before, "action": action, "pos": np.asarray(info["enemy_pos"], dtype=np.float64),
                        "perceived": set(int(i) for i in info.get("enemy_perceived_static_ids", ())),
                        "avoid": bool(info.get("enemy_obstacle_avoidance_active")), "mode": str(info.get("enemy_obstacle_avoidance_mode")),
                        "mission": np.asarray(info["enemy_mission_command"], dtype=np.float64),
                        "clearance": float(info["enemy_actual_min_clearance"]), "outcome": info["outcome"]})
        done = terminated or truncated
    last = history[-1]
    distances = [box_distance(last["pos"], o.min_corner, o.max_corner) for o in layout]
    index = int(np.argmin(distances))
    obstacle = layout[index]
    sector = np.asarray(history[-1]["obs"][16:36], dtype=np.float64).reshape(10, 2)
    angles = [AN.angle_deg(h["action"], h["mission"]) for h in history[-3:]]
    return {
        "attacker": attacker_id, "level": level, "defender": family, "seed": int(seed), "steps": len(history),
        "outcome": last["outcome"], "obstacle_index": index, "obstacle_distance_to_final_pos_m": distances[index],
        "obstacle_map_known": bool(getattr(obstacle, "map_known", False)),
        "obstacle_in_attacker_perceived_set_last_step": index in last["perceived"],
        "obstacle_in_perceived_set_any_last10": any(index in h["perceived"] for h in history[-10:]),
        "n_obstacles": len(layout), "n_perceived_last": len(last["perceived"]),
        "avoidance_mode_last_step": last["mode"], "avoidance_active_last_step": last["avoid"],
        "avoidance_active_last5": [h["avoid"] for h in history[-5:]],
        "tactical_action_last": [round(float(x), 3) for x in last["action"]],
        "angle_action_vs_asset_last_deg": angles[-1], "angle_action_vs_asset_max_last3_deg": max(a for a in angles if not np.isnan(a)),
        "sector_static_clearance_before": {n: round(float(v), 3) for n, v in zip(SECTORS, sector[:, 0])},
        "min_sector_static_clearance_before": float(sector[:, 0].min()),
        "min_clearance_prior_steps_m": [round(h["clearance"], 2) for h in history[-6:-1]],
        "min_clearance_over_episode_before_last_m": min([h["clearance"] for h in history[:-1]] or [float("nan")]),
        "altitude_last_m": float(last["pos"][2]),
    }


def classify(row: dict) -> str:
    if not row["obstacle_in_perceived_set_any_last10"]:
        return "perception_limitation"
    if row["angle_action_vs_asset_max_last3_deg"] > 45.0:
        return "learned_policy_obstacle_mistake"
    if row["avoidance_active_last_step"]:
        return "avoidance_layer_limitation"
    return "unclear"


def main() -> list[dict]:
    summary = json.loads((AN.SUMMARY_DIR / "evaluation_summary.json").read_text())
    models = {a: E._load_attacker(a) for a in AN.A1_IDS}
    rows = []
    for item in summary["collisions"]:
        row = resim(models[item["attacker"]], item["attacker"], item["level"], item["defender"], item["seed"])
        assert row["outcome"] == item["outcome"] and row["steps"] == item["steps"], (row["outcome"], item["outcome"], row["steps"], item["steps"])
        row["classification"] = classify(row)
        row["a0_same_scenario_outcome"] = item["a0_same_scenario_outcome"]
        rows.append(row)
    (AN.SUMMARY_DIR / "collision_forensics.json").write_text(json.dumps(rows, indent=1, default=float))
    return rows


if __name__ == "__main__":
    import collections

    rows = main()
    print(len(rows), collections.Counter((r["attacker"], r["level"], r["classification"]) for r in rows))
