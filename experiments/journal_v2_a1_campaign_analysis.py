"""Analysis of the pre-registered V2 A1 evaluation: metrics, paired statistics, gates, forensics, shortlist.

All inputs are the evaluation task files written by `journal_v2_a1_campaign_eval`; nothing is re-simulated here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import experiments.journal_v2_a1_campaign_eval as E
import experiments.journal_v2_a1_training as T
from experiments import journal_v2_a1 as A
from experiments.final_stats import paired_bootstrap_ci
from experiments.journal_v2_a1_evaluation import (
    A1_V2_ACCEPTANCE_THRESHOLDS,
    OBSTACLE_COLLISION_OUTCOMES,
    evaluate_acceptance,
    select_representative_episodes,
)

SUMMARY_DIR = E.EVAL_ROOT / "summary"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816  # the repository convention (experiments.adaptive_attacker_statistics)
A1_IDS = ("A1-201", "A1-202", "A1-203")
SECTOR_NAMES = ("front", "front_left", "left", "rear_left", "rear", "rear_right", "right", "front_right", "up", "down")
INTERCEPT_OUTCOMES = ("intercepted", "unsafe_intercept")


def angle_deg(a, b) -> float:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["intercepted"] = df.outcome.isin(INTERCEPT_OUTCOMES)
    df["timed_out"] = df.outcome == "timeout"
    df["flight_time_success"] = df.flight_time_s.where(df.captured)
    df["path_excess_success"] = df.path_excess.where(df.captured)
    return df


def group_metrics(g: pd.DataFrame) -> dict:
    cap = g[g.captured]
    pe = g.path_excess.replace([np.inf, -np.inf], np.nan)
    return {
        "n": len(g), "capture_rate": g.captured.mean(), "interception_rate": g.intercepted.mean(),
        "obstacle_collision_rate": g.obstacle_collision.mean(), "endurance_exhaustion_rate": g.endurance_exhausted.mean(),
        "timeout_rate": g.timed_out.mean(), "mean_duration_s": g.flight_time_s.mean(),
        "mean_success_duration_s": cap.flight_time_s.mean() if len(cap) else float("nan"),
        "mean_path_length_m": g.path_length_m.mean(), "mean_path_excess": pe.mean(), "median_path_excess": pe.median(),
        "mean_path_excess_success": cap.path_excess.mean() if len(cap) else float("nan"),
        "median_path_excess_success": cap.path_excess.median() if len(cap) else float("nan"),
        "gt1.3x": int((pe > 1.3).sum()), "gt2x": int((pe > 2).sum()), "gt5x": int((pe > 5).sum()),
        "success_gt1.3x": int((cap.path_excess > 1.3).sum()), "success_gt2x": int((cap.path_excess > 2).sum()),
        "success_gt5x": int((cap.path_excess > 5).sum()),
        "mean_min_defender_separation_m": g.min_defender_separation_m.mean(),
        "terminal_orbit_or_repeated_flyby": int(((g.orbit_revolutions_max >= 1.0) | (g.overshoots >= 3)).sum()),
        "episodes_with_any_overshoot": int((g.overshoots >= 1).sum()), "mean_return": g.return_undiscounted.mean(),
    }


def cell_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = [{"attacker": a, "defender": d, "level": l, **group_metrics(g)} for (a, d, l), g in df.groupby(["attacker", "defender", "level"])]
    return pd.DataFrame(rows)


def margins(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {}
    for name, keys in (("attacker", ["attacker"]), ("attacker_level", ["attacker", "level"]), ("attacker_defender", ["attacker", "defender"])):
        out[name] = pd.DataFrame([{**dict(zip(keys, k if isinstance(k, tuple) else (k,))), **group_metrics(g)} for k, g in df.groupby(keys)])
    return out


def paired_table(df: pd.DataFrame) -> pd.DataFrame:
    """A1 - A0 capture-rate differences with paired bootstrap CIs over the matched (level, defender, seed) scenarios."""
    wide = df.pivot_table(index=["level", "defender", "seed"], columns="attacker", values="captured", aggfunc="first").astype(float)
    wide["A1-mean"] = wide[list(A1_IDS)].mean(axis=1)
    rows = []

    def add(scope, level, defender, subset):
        for a1 in (*A1_IDS, "A1-mean"):
            observed, lo, hi = paired_bootstrap_ci(subset[a1].to_numpy(), subset["A0"].to_numpy(), BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
            rows.append({"attacker": a1, "scope": scope, "level": level, "defender": defender, "pairs": len(subset),
                         "a1_capture": subset[a1].mean(), "a0_capture": subset["A0"].mean(),
                         "diff_pp": 100 * observed, "ci95_low_pp": 100 * lo, "ci95_high_pp": 100 * hi,
                         "ci_excludes_zero": bool(lo > 0 or hi < 0)})

    for (level, defender), sub in wide.groupby(level=["level", "defender"]):
        add("cell", level, defender, sub)
    for level, sub in wide.groupby(level="level"):
        add("by_level_all_defenders", level, "ALL", sub)
    for defender, sub in wide.groupby(level="defender"):
        add("by_defender_all_levels", "ALL", defender, sub)
    add("overall", "ALL", "ALL", wide)
    return pd.DataFrame(rows)


def _ids_for(step: dict, collision_type) -> tuple:
    return step["perceived_dynamic_ids"] if collision_type and "dynamic" in str(collision_type).lower() else step["perceived_static_ids"]


def collision_forensics(df: pd.DataFrame, traces: dict) -> list[dict]:
    rows = []
    for r in df[(df.attacker.isin(A1_IDS)) & df.obstacle_collision].sort_values(["attacker", "level", "defender", "seed"]).to_dict("records"):
        trace = traces[(r["attacker"], r["level"], r["defender"], r["seed"])]
        last = trace[-1]
        prev = trace[-2] if len(trace) > 1 else last
        ctype = last["collision_type"] or ("dynamic" if "dynamic" in r["outcome"] else "static")
        obstacle_id = last["collision_obstacle_id"]
        perceived_last = obstacle_id is not None and int(obstacle_id) in _ids_for(prev, ctype)
        perceived_any = obstacle_id is not None and any(int(obstacle_id) in _ids_for(s, ctype) for s in trace[-6:])
        angles = [angle_deg(s["action"], s["mission_cmd"]) for s in trace[-3:]]
        angles = [a for a in angles if not math.isnan(a)]
        obs = last["obs_before"]
        sector = np.asarray(obs[16:36], dtype=np.float64).reshape(10, 2)
        avoid_recent = [bool(s["avoid_active"]) for s in trace[-5:]]
        a0 = df[(df.attacker == "A0") & (df.level == r["level"]) & (df.defender == r["defender"]) & (df.seed == r["seed"])].iloc[0]
        row = {
            "attacker": r["attacker"], "seed": r["seed"], "defender": r["defender"], "level": r["level"], "outcome": r["outcome"],
            "obstacle_type": str(ctype), "obstacle_id": None if obstacle_id is None else int(obstacle_id), "steps": r["steps"],
            "attacker_mode_at_collision": last["avoid_mode"], "avoidance_active_last_step": bool(last["avoid_active"]),
            "avoidance_active_last5": avoid_recent, "tactical_action_last": [round(float(x), 3) for x in last["action"]],
            "action_vs_asset_angle_last_deg": angles[-1] if angles else float("nan"),
            "action_vs_asset_angle_max_last3_deg": max(angles) if angles else float("nan"),
            "final_cmd_vs_action_deg": angle_deg(last["final_cmd"], last["action"]),
            "sector_static_clearance_before": {n: round(float(v), 3) for n, v in zip(SECTOR_NAMES, sector[:, 0])},
            "sector_dynamic_clearance_before": {n: round(float(v), 3) for n, v in zip(SECTOR_NAMES, sector[:, 1])},
            "min_sector_static_before": float(sector[:, 0].min()),
            "obstacle_perceived_prev_step": bool(perceived_last), "obstacle_perceived_last6": bool(perceived_any),
            "min_clearance_prior_5_steps_m": min([s["actual_min_clearance"] for s in trace[:-1][-5:]] or [float("nan")]),
            "min_clearance_prior_all_m": min([s["actual_min_clearance"] for s in trace[:-1]] or [float("nan")]),
            "a0_same_scenario_outcome": a0["outcome"],
        }
        row["large_off_axis_command"] = bool(row["action_vs_asset_angle_max_last3_deg"] > 45.0)
        if not perceived_any:
            row["classification"] = "perception_limitation"
        elif a0["obstacle_collision"]:
            row["classification"] = "avoidance_layer_limitation"
        elif row["avoidance_active_last_step"] and not row["large_off_axis_command"]:
            row["classification"] = "avoidance_layer_limitation"
        elif row["large_off_axis_command"]:
            row["classification"] = "learned_policy_obstacle_mistake"
        else:
            row["classification"] = "unclear"
        rows.append(row)
    return rows


def endurance_forensics(df: pd.DataFrame, traces: dict) -> list[dict]:
    rows = []
    for r in df[(df.attacker.isin(A1_IDS)) & df.endurance_exhausted].sort_values(["attacker", "level", "defender", "seed"]).to_dict("records"):
        trace = traces[(r["attacker"], r["level"], r["defender"], r["seed"])]
        ranges = np.array([s["range_to_asset"] for s in trace])
        def_ranges = np.array([s["range_to_defender"] for s in trace])
        avoid = np.mean([bool(s["avoid_active"]) for s in trace])
        a0 = df[(df.attacker == "A0") & (df.level == r["level"]) & (df.defender == r["defender"]) & (df.seed == r["seed"])].iloc[0]
        closest = float(ranges.min())
        defender_near = float(np.mean(def_ranges < 25.0))
        actions = np.array([s["action"] for s in trace], dtype=np.float64)
        unit = actions / np.maximum(np.linalg.norm(actions, axis=1, keepdims=True), 1e-9)
        altitude = np.array([s["enemy_pos"][2] for s in trace], dtype=np.float64)
        angle_to_asset = np.array([angle_deg(s["action"], s["mission_cmd"]) for s in trace], dtype=np.float64)
        circling = r["orbit_revolutions_max"] >= 1.0 or r["overshoots"] >= 3 or closest > 0.5 * r["initial_range_m"]
        if circling and defender_near < 0.25 and avoid < 0.4:
            cls = "circling_loitering"
        elif circling and defender_near < 0.25:
            cls = "obstacle_induced_detour" if avoid >= 0.4 else "circling_loitering"
        elif avoid >= 0.4:
            cls = "obstacle_induced_detour"
        elif defender_near >= 0.25:
            cls = "tactical_evasion_consumed_endurance"
        elif a0["endurance_exhausted"]:
            cls = "difficult_but_sensible_pursuit"
        else:
            cls = "unclear"
        rows.append({
            "attacker": r["attacker"], "seed": r["seed"], "defender": r["defender"], "level": r["level"], "steps": r["steps"],
            "duration_s": r["flight_time_s"], "path_excess": r["path_excess"], "path_length_m": r["path_length_m"],
            "initial_range_m": r["initial_range_m"], "closest_asset_approach_m": closest, "closest_over_initial": closest / r["initial_range_m"],
            "flybys_overshoots": r["overshoots"], "orbit_revolutions_max": r["orbit_revolutions_max"], "avoidance_steps": r["avoidance_steps"],
            "avoidance_fraction": float(avoid), "fraction_steps_defender_within_25m": defender_near,
            "a0_same_scenario_outcome": a0["outcome"], "classification": cls,
            "mean_speed_mps": float(r["path_length_m"] / max(r["flight_time_s"], 1e-9)),
            "mean_altitude_m": float(altitude.mean()), "max_altitude_m": float(altitude.max()),
            "mean_action_z": float(unit[:, 2].mean()), "fraction_steps_action_z_gt_0.5": float((unit[:, 2] > 0.5).mean()),
            "median_angle_action_vs_asset_deg": float(np.nanmedian(angle_to_asset)) if np.isfinite(angle_to_asset).any() else float("nan")})
    return rows


def derive(record: dict, trace: list[dict]) -> dict:
    """Per-episode scalars computed while streaming the (large) trace files."""
    gamma = A.A1_V2_PPO_HYPERPARAMS["gamma"]
    rewards = np.array([s["reward"] for s in trace], dtype=np.float64)
    pos = np.array([s["enemy_pos"] for s in trace], dtype=np.float64)
    if len(pos) < 3:
        lateral = 0.0
    else:
        line = pos[-1] - pos[0]
        norm = np.linalg.norm(line)
        lateral = float(np.linalg.norm(pos - pos[0], axis=1).max() if norm < 1e-9
                        else np.linalg.norm(np.cross(pos - pos[0], line / norm), axis=1).max())
    return {"discounted_return": float((gamma ** np.arange(len(rewards)) * rewards).sum()),
            "lateral_excursion_m": lateral, "min_clearance_all_m": float(min(s["actual_min_clearance"] for s in trace))}


def pathology_screen(df: pd.DataFrame, traces=None) -> dict:
    out = {}
    for a1 in A1_IDS:
        g = df[df.attacker == a1]
        cap = g[g.captured]

        def brief(row, extra=None):
            if row is None:
                return None
            return {"level": row["level"], "defender": row["defender"], "seed": int(row["seed"]), "outcome": row["outcome"],
                    "path_excess": float(row["path_excess"]), "steps": int(row["steps"]), "return": float(row["return_undiscounted"]), **(extra or {})}

        nonc = g[~g.obstacle_collision]
        closest = nonc.loc[nonc.min_clearance_all_m.idxmin()].to_dict() if len(nonc) else None
        lateral = g.loc[g.lateral_excursion_m.idxmax()].to_dict()
        pick = lambda frame_, col, fn: frame_.loc[fn(frame_[col])].to_dict() if len(frame_) else None  # noqa: E731
        out[a1] = {
            "worst_path_excess": brief(pick(g.assign(pe=g.path_excess.replace([np.inf], np.nan)), "pe", lambda s: s.idxmax())),
            "longest_episode": brief(pick(g, "steps", lambda s: s.idxmax())),
            "closest_obstacle_precursor_non_collision": None if closest is None else brief(closest, {"min_clearance_m": closest["min_clearance_all_m"]}),
            "terminal_flyby_or_orbit_count": int(((g.orbit_revolutions_max >= 1.0) | (g.overshoots >= 3)).sum()),
            "c0_endurance_exhaustions": [brief(r) for r in g[(g.level == "C0") & g.endurance_exhausted].head(10).to_dict("records")],
            "c0_endurance_exhaustion_count": int(((g.level == "C0") & g.endurance_exhausted).sum()),
            "largest_lateral_excursion": brief(lateral, {"max_lateral_excursion_m": lateral["lateral_excursion_m"]}),
            "highest_return_success": brief(pick(cap, "return_undiscounted", lambda s: s.idxmax())),
            "lowest_return_success": brief(pick(cap, "return_undiscounted", lambda s: s.idxmin())),
        }
    return out


def discounted_return_by_outcome(df: pd.DataFrame) -> list[dict]:
    """Discounted training return (what PPO optimizes) per attacker x outcome, from the recorded per-step rewards
    (`discounted_return` is computed in `derive`). Diagnoses whether late failure (endurance exhaustion) is cheaper
    than early failure under the frozen gamma."""
    table = df.groupby(["attacker", "outcome"]).agg(
        episodes=("steps", "size"), mean_steps=("steps", "mean"), mean_undiscounted=("return_undiscounted", "mean"),
        mean_discounted=("discounted_return", "mean")).reset_index()
    return table.to_dict("records")


def seed_consistency(df: pd.DataFrame) -> dict:
    out = {}
    for a1 in A1_IDS:
        g = df[df.attacker == a1]
        pe = g[g.captured].path_excess
        out[a1] = {"capture": g.captured.mean(), "collision": g.obstacle_collision.mean(), "exhaustion": g.endurance_exhausted.mean(),
                   "interception": g.intercepted.mean(), "timeout": g.timed_out.mean(),
                   "path_excess_success_quantiles": {str(q): float(pe.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)} if len(pe) else {}}
    caps = [v["capture"] for v in out.values()]
    return {"per_seed": out, "capture_spread": max(caps) - min(caps), "capture_range": [min(caps), max(caps)],
            "dispersion_flag": (max(caps) - min(caps)) > A1_V2_ACCEPTANCE_THRESHOLDS["generalization_max_seed_spread"]}


LOOK_FOR = {
    "normal_c0_success": "A direct, near-straight run to the asset in clear space with no weaving, loop or orbit.",
    "clear_space_evasion": "Whether any lateral move around the defender is purposeful (it should still end in capture) and not a loiter.",
    "obstacle_heavy_success": "Smooth passage through the clutter: the policy and the common avoidance layer should not fight each other.",
    "dense_clutter_success": "Tactical choices against Lead/RMPC while threading dense obstacles; progress toward the asset should be steady.",
    "worst_path_excess": "The most wasteful trajectory: is it circling/weaving or a justified detour?",
    "endurance_exhaustion": "Whether the attacker loiters or circles until its path budget runs out.",
    "collision": "Whether the obstacle was visible and why the attacker flew into it (command vs avoidance layer).",
    "lead_rmpc_behavior": "How the policy handles the stronger predictive defenders (timing of turns, committing vs hedging).",
    "heldout_a1_203": "Representative behavior of the policy that will be held out from defender adaptation.",
}


def build_shortlist(df: pd.DataFrame, traces: dict, max_items: int = 15) -> list[dict]:
    chosen: dict[tuple, dict] = {}

    def add(row, category):
        if row is None:
            return
        key = (row["attacker"], row["level"], row["defender"], int(row["seed"]))
        entry = chosen.setdefault(key, {"attacker": key[0], "level": key[1], "defender": key[2], "seed": key[3], "outcome": row["outcome"],
                                        "path_excess": float(row["path_excess"]), "categories": [], "look_for": []})
        if category not in entry["categories"]:
            entry["categories"].append(category)
            entry["look_for"].append(LOOK_FOR[category])

    a1 = df[df.attacker.isin(A1_IDS)]
    for seed_id in A1_IDS:
        g = a1[a1.attacker == seed_id]
        c0 = g[(g.level == "C0") & (g.defender == "greedy") & g.captured].sort_values(["seed"])
        if len(c0):
            mid = c0.iloc[(c0.path_excess - c0.path_excess.median()).abs().argsort().iloc[0]]
            add(mid.to_dict(), "normal_c0_success")
        hard = g[(g.level.isin(["C2", "C3"])) & g.captured].sort_values(["avoidance_steps", "seed"], ascending=[False, True])
        if len(hard):
            add(hard.iloc[0].to_dict(), "obstacle_heavy_success")
        dense = g[(g.level == "C3") & g.defender.isin(["lead", "rmpc"]) & g.captured].sort_values(["path_excess", "seed"])
        if len(dense):
            add(dense.iloc[len(dense) // 2].to_dict(), "dense_clutter_success" if seed_id != "A1-203" else "heldout_a1_203")
            add(dense.iloc[len(dense) // 2].to_dict(), "lead_rmpc_behavior")
    evasive = a1[(a1.level.isin(["C0", "C1"])) & a1.captured & (a1.path_excess > 1.05)].sort_values(["min_defender_separation_m", "seed"])
    if len(evasive):
        add(evasive.iloc[0].to_dict(), "clear_space_evasion")
    worst = a1.assign(pe=a1.path_excess.replace([np.inf], np.nan)).sort_values("pe", ascending=False)
    for row in worst.head(2).to_dict("records"):
        add(row, "worst_path_excess")
    exh = a1[a1.endurance_exhausted].sort_values(["level", "attacker", "seed"])
    for row in exh.head(2).to_dict("records"):
        add(row, "endurance_exhaustion")
    col = a1[a1.obstacle_collision].sort_values(["attacker", "level", "seed"])
    for row in col.head(2).to_dict("records"):
        add(row, "collision")
    items = list(chosen.values())
    # keep within budget: drop single-category, non-essential entries first
    items.sort(key=lambda e: (-len(e["categories"]), e["attacker"], e["level"], e["seed"]))
    return items[:max_items]


def run_analysis() -> dict:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    records = E.scan_tasks(derive)
    traces = E.TraceStore()
    integrity = E.integrity_check(records)
    df = frame(records)
    cells = cell_table(df)
    cells.to_csv(SUMMARY_DIR / "cell_metrics_80_cells.csv", index=False)
    for name, table in margins(df).items():
        table.to_csv(SUMMARY_DIR / f"margin_{name}.csv", index=False)
    paired = paired_table(df)
    paired.to_csv(SUMMARY_DIR / "paired_a1_minus_a0_capture.csv", index=False)
    gates = evaluate_acceptance(records)
    collisions = collision_forensics(df, traces)
    exhaustions = endurance_forensics(df, traces)
    pathology = pathology_screen(df, traces)
    consistency = seed_consistency(df)
    shortlist = build_shortlist(df, traces)
    representative = select_representative_episodes(records)
    payload = {"integrity": integrity, "gates": gates, "seed_consistency": consistency, "collisions": collisions,
               "endurance_exhaustions": exhaustions, "pathology": pathology, "shortlist": shortlist, "representative": representative,
               "discounted_return_by_outcome": discounted_return_by_outcome(df)}
    (SUMMARY_DIR / "evaluation_summary.json").write_text(json.dumps(payload, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return payload


if __name__ == "__main__":
    out = run_analysis()
    print(json.dumps({"integrity": out["integrity"], "all_gates_passed": out["gates"]["all_passed"],
                      "gates": {k: v["passed"] for k, v in out["gates"].items() if isinstance(v, dict) and "passed" in v},
                      "collisions": len(out["collisions"]), "exhaustions": len(out["endurance_exhaustions"])}, indent=1))
