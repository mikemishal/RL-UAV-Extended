"""Phase 9-12 analysis for the asset motion model study: mission/efficiency/terminal/obstacle
metrics, C3 130017 tracking, worst-case pathology search, and difficulty-preservation comparison."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.journal_v2_planner_fidelity import population_metrics
from experiments.journal_v2_validation_core import summarize, terminal_flyby_metrics

R = Path("results/asset_motion_model_study")
LABELS = ("legacy", "tau0.5", "tau1.0", "tau2.0", "tau4.0")


def load(label):
    return {p.stem: json.loads(p.read_text()) for p in sorted((R / label).glob("C?_*.json"))}


def main():
    report = {}
    for label in LABELS:
        records = load(label)
        recs = list(records.values())
        pop = population_metrics(recs)
        per_level = {}
        for level in ("C0", "C1", "C2", "C3"):
            level_recs = [r for r in recs if r["meta"]["level"] == level]
            summaries = [summarize(r) for r in level_recs]
            flyby = [terminal_flyby_metrics(r) for r in level_recs]
            per_level[level] = {
                **pop[level],
                "terminal_flyby_detections": sum(f["detected"] for f in flyby),
                "terminal_flyby_seeds": [r["meta"]["seed"] for r, f in zip(level_recs, flyby) if f["detected"]],
                "mean_episode_length": float(np.mean([s["steps"] for s in summaries])),
                "mean_min_range_m": float(np.mean([s["min_range_m"] for s in summaries])),
            }
        report[label] = per_level

        # Phase 10: C3 130017 explicitly
        rec_130017 = records.get("C3_130017")
        if rec_130017 is not None:
            s = summarize(rec_130017)
            f = terminal_flyby_metrics(rec_130017)
            report.setdefault("_130017", {})[label] = {
                "outcome": s["outcome"], "steps": s["steps"], "path_length_m": s["path_length_m"],
                "path_excess_ratio": s["path_excess_ratio"], "fly_by_count": f["fly_by_count"],
                "angular_travel_revolutions": round(f["angular_travel_revolutions"], 2),
                "avoidance_activations": s["n_avoidance_segments"], "min_range_m": s["min_range_m"],
                "flight_time_s": s["flight_time_s"],
            }

        # Phase 11: worst-case pathologies per setting
        c3_recs = [r for r in recs if r["meta"]["level"] == "C3"]
        c3_summaries = [(r["meta"]["seed"], summarize(r)) for r in c3_recs]
        by_excess = sorted(c3_summaries, key=lambda rs: -rs[1]["path_excess_ratio"])[:5]
        by_length = sorted(c3_summaries, key=lambda rs: -rs[1]["steps"])[:5]
        by_clearance = sorted(
            [(seed, s, min((r["nearest_obstacle_clearance"] for r in rec["rows"] if r["nearest_obstacle_clearance"] is not None), default=float("inf")))
             for (seed, s), rec in zip(c3_summaries, c3_recs)], key=lambda t: t[2])[:5]
        collisions = [(seed, s["outcome"]) for seed, s in c3_summaries if "collision" in s["outcome"]]
        orbit_events = [(seed, s["orbit_revolutions_max"]) for seed, s in c3_summaries if s["orbit_event"]]
        report.setdefault("_pathologies", {})[label] = {
            "top5_path_excess": [(seed, round(s["path_excess_ratio"], 2)) for seed, s in by_excess],
            "top5_longest": [(seed, s["steps"]) for seed, s in by_length],
            "top5_lowest_clearance": [(seed, round(c, 3)) for seed, s, c in by_clearance],
            "collisions": collisions, "orbit_events": orbit_events,
        }

    json.dump(report, open(R / "population_report.json", "w"), indent=1, default=str)
    for label in LABELS:
        print(f"\n=== {label} ===")
        for level in ("C0", "C1", "C2", "C3"):
            m = report[label][level]
            print(f"  {level}: capture={m['capture_rate']:.2f} coll={m['collisions']} "
                  f"excess(mean/max)={m['excess_mean']:.3f}/{m['excess_max']:.2f} "
                  f">1.3x={m['excess_gt_1.3']} >2x={m['excess_gt_2']} >5x={m['excess_gt_5']} "
                  f"flyby={m['terminal_flyby_detections']}{m['terminal_flyby_seeds']} "
                  f"avoid_act={m['activations']} emerg={m['all_candidates_collide_selections']} "
                  f"mean_len={m['mean_episode_length']:.1f}")
    print("\n=== C3 130017 across settings ===")
    for label in LABELS:
        print(" ", label, report["_130017"][label])
    print("\n=== pathologies ===")
    for label in LABELS:
        print(" ", label, report["_pathologies"][label])


if __name__ == "__main__":
    main()
