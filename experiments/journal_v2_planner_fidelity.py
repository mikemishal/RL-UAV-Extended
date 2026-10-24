"""Predictor-vs-execution fidelity diagnostics for the V2 attacker's obstacle planner.

Purely observational (no training, no learned policy, no change to any decision).
For every candidate selection (activation or reselection) of the scripted attacker's
`ObstacleAvoidancePlanner` it compares the positions the attacker actually flew until
the next selection (at most the 12-step horizon) with four predictions:

  fixed            the original model: the initial direction held for the horizon (float64)
  persist          persisted-waypoint re-aiming without release (`predict_waypoint_following_trajectory`)
  persist_float64  an independent float64 reference of `persist`, showing rounding sensitivity
  executed         the planner's release-aware predictor (`predict_executed_trajectory`)

Episodes are passive-A0 (defender parked), V2 semantics, map_local perception. Results go
to the gitignored results/journal_v2_planner_fidelity/<label>/.

Usage:
    python -m experiments.journal_v2_planner_fidelity cases LABEL    # C2/130065, C3/130063, C3/130081, C3/130064
    python -m experiments.journal_v2_planner_fidelity seeds LABEL    # C1-C3 x seeds 130000-130099 (300 episodes)
    python -m experiments.journal_v2_planner_fidelity analyze LABEL  # fidelity + behavior summary.json
    python -m experiments.journal_v2_planner_fidelity compare A B    # paired seeds: collisions gained/lost, metrics
    python -m experiments.journal_v2_planner_fidelity controlled     # exactness with the external changes removed

`cases`/`seeds` only record, so they can be run on an older revision (e.g. a `git archive`
extraction of the baseline) and analyzed with this file afterwards. Existing labels are
never overwritten.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from uav_defend.dynamics.constrained_point_mass import advance_velocity, apply_boundary
from uav_defend.obstacles.avoidance import DynamicsLimits, predict_constrained_trajectory

from experiments.journal_v2_avoidance_recorder import CASES, SEEDS, aggregate, plain, record

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "journal_v2_planner_fidelity"
CRITICAL_CASES = CASES + (("C3", 130006), ("C3", 130064), ("C2", 130054))
HORIZON_STEPS = 12


def _unit(vector, fallback, eps):
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > eps else fallback


def _follow(position, velocity, waypoint, fallback, limits, horizon_steps):
    """Pure persisted-waypoint model (no release); absent in revisions that predate it."""
    from uav_defend.obstacles.avoidance import predict_waypoint_following_trajectory

    return predict_waypoint_following_trajectory(position, velocity, waypoint, fallback, limits, horizon_steps)


def _executed(decision, limits, steps):
    """The planner's release-aware executed-policy predictor (absent in older revisions).

    Always predicted over the planner's full horizon (the release test looks ahead
    that far) and then cut to the `steps` of the comparison interval."""
    from uav_defend.obstacles.avoidance import predict_executed_trajectory
    from uav_defend.obstacles.geometry import AABBObstacle
    from uav_defend.obstacles.layout import ObstacleLayout

    layout = ObstacleLayout(obstacles=tuple(
        AABBObstacle(min_corner=np.array(b["min"]), max_corner=np.array(b["max"])) for b in decision["layout"]))
    positions, velocities, release = predict_executed_trajectory(
        np.asarray(decision["position"], dtype=np.float32), np.asarray(decision["velocity"], dtype=np.float32),
        decision["waypoint"], decision["direct"], np.asarray(decision["target"], dtype=np.float32), layout, limits,
        HORIZON_STEPS, decision["stop_radius"],
    )
    return positions[:steps + 1], velocities[:steps + 1], release if release is not None and release <= steps else None


def waypoint_following_model_float64(position, velocity, waypoint, fallback, limits: DynamicsLimits, horizon_steps):
    """Independent float64 reference of the persisted-waypoint control law.

    Differs from execution only in numeric precision. Kept to show how sensitive
    the executed motion is to float32 rounding near a heading reversal."""
    p = np.array(position, dtype=np.float64)
    v = np.array(velocity, dtype=np.float64)
    waypoint = np.asarray(waypoint, dtype=np.float64)
    fallback = np.asarray(fallback, dtype=np.float64)
    positions, velocities = [p.copy()], [v.copy()]
    for _ in range(horizon_steps):
        direction = _unit(waypoint - p, fallback, limits.eps)
        v, _ = advance_velocity(
            current_velocity=v, desired_velocity=limits.max_speed * direction, max_speed=limits.max_speed,
            max_accel=limits.max_accel, max_turn_rate_rad=limits.max_turn_rate_rad,
            max_climb_rate=limits.max_climb_rate, max_descent_rate=limits.max_descent_rate,
            dt=limits.dt, eps=limits.eps,
        )
        v = np.asarray(v, dtype=np.float64)
        p, v = apply_boundary(p + v * limits.dt, v, L=limits.L, max_altitude=limits.max_altitude)
        positions.append(p.copy())
        velocities.append(v.copy())
    return positions, velocities


def box_clearance(point, boxes):
    """Point-to-surface distance to the nearest static AABB (0 inside)."""
    best = float("inf")
    for box in boxes:
        lo, hi = np.asarray(box["min"]), np.asarray(box["max"])
        gap = np.maximum(np.maximum(lo - point, point - hi), 0.0)
        best = min(best, float(np.linalg.norm(gap)))
    return best


def _angle_deg(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def selection_events(rec):
    """One record per candidate selection. The comparison interval runs from the
    selection to the next selection (activation/reselection), the horizon, or
    the end of the episode -- i.e. as long as the selected candidate and the
    policy that follows it (release to DIRECT) are what actually flies."""
    decisions, rows, extras = rec["decisions"], rec["rows"], rec["extras"]
    selected = [i for i, d in enumerate(decisions) if d["selection"] is not None]
    events = []
    for i in selected:
        d = decisions[i]
        if not d["active"] or d["waypoint"] is None:
            continue
        following = [j for j in selected if j > i]
        n = (following[0] if following else len(decisions)) - i
        usable = min(n, HORIZON_STEPS, len(rows) - 2 - i)  # last row may be an interpolated terminal contact
        if usable < 1:
            continue
        obstacle_type = extras[i + 1]["avoid_obstacle_type"]
        category = "dynamic" if obstacle_type not in (None, "static") else (
            "climb" if d["mode"] == "climb" else "lateral")
        events.append({"step": i + 1, "mode": d["mode"], "category": category, "persisted_steps": n,
                       "usable_steps": usable, "decision": d, "reselection": not d["activation"]})
    return events


def event_fidelity(rec, event, boxes):
    d, i, k = event["decision"], event["step"] - 1, event["usable_steps"]
    limits = DynamicsLimits(**d["limits"])
    p0, v0 = np.array(d["position"]), np.array(d["velocity"])
    waypoint, fallback = np.array(d["waypoint"]), np.array(d["direct"])
    direction = _unit(waypoint - p0, fallback, limits.eps)
    fixed_p, fixed_v = predict_constrained_trajectory(p0, v0, direction, limits, k)
    persist_p, persist_v = _follow(p0, v0, waypoint, fallback, limits, k)
    persist64_p, persist64_v = waypoint_following_model_float64(p0, v0, waypoint, fallback, limits, k)
    executed_p, executed_v, release = _executed(d, limits, k)
    actual_p = [np.array(rec["rows"][i]["attacker_pos"])] + [np.array(rec["rows"][i + j]["attacker_pos"]) for j in range(1, k + 1)]
    actual_v = [v0] + [np.array(rec["rows"][i + j]["attacker_vel"]) for j in range(1, k + 1)]
    actual_release = next((j + 1 for j in range(k) if not rec["decisions"][i + j]["active"]), None)
    result = {"step": event["step"], "category": event["category"], "mode": event["mode"], "steps": k,
              "persisted_steps": event["persisted_steps"], "reselection": event["reselection"],
              "predicted_release_step": release, "actual_release_step": actual_release}
    for name, (pp, vv) in (("fixed", (fixed_p, fixed_v)), ("persist", (persist_p, persist_v)),
                           ("persist_float64", (persist64_p, persist64_v)), ("executed", (executed_p, executed_v))):
        err = [float(np.linalg.norm(a - b)) for a, b in zip(pp[1:], actual_p[1:])]
        head = [_angle_deg(a, b) for a, b in zip(vv[1:], actual_v[1:])]
        result[name] = {
            "mean_pos_err": float(np.mean(err)), "max_pos_err": float(np.max(err)), "final_pos_err": err[-1],
            "mean_heading_err_deg": float(np.mean(head)), "max_heading_err_deg": float(np.max(head)),
            "pred_min_clearance": min(box_clearance(p, boxes) for p in pp),
            "pred_path": float(sum(np.linalg.norm(b - a) for a, b in zip(pp, pp[1:]))),
        }
    result["actual"] = {
        "min_clearance": min(box_clearance(p, boxes) for p in actual_p),
        "path": float(sum(np.linalg.norm(b - a) for a, b in zip(actual_p, actual_p[1:]))),
    }
    selection = d["selection"]
    result["planner_selection"] = selection
    return result


def episode_fidelity(rec):
    boxes = rec["static_obstacles"]
    return [event_fidelity(rec, e, boxes) for e in selection_events(rec)]


def summarize_fidelity(events):
    out = {}
    for category in ("climb", "lateral", "dynamic", "all"):
        subset = [e for e in events if category == "all" or e["category"] == category]
        multi = [e for e in subset if e["steps"] >= 2]
        entry = {"events": len(subset), "events_with_2plus_steps": len(multi)}
        for model in ("fixed", "persist", "persist_float64", "executed"):
            if not multi:
                continue
            entry[model] = {
                "mean_pos_err_m": float(np.mean([e[model]["mean_pos_err"] for e in multi])),
                "max_pos_err_m": float(np.max([e[model]["max_pos_err"] for e in multi])),
                "mean_final_pos_err_m": float(np.mean([e[model]["final_pos_err"] for e in multi])),
                "mean_heading_err_deg": float(np.mean([e[model]["mean_heading_err_deg"] for e in multi])),
                "max_heading_err_deg": float(np.max([e[model]["max_heading_err_deg"] for e in multi])),
                "mean_pred_min_clearance_m": float(np.mean([e[model]["pred_min_clearance"] for e in multi])),
                "mean_abs_clearance_err_m": float(np.mean(
                    [abs(e[model]["pred_min_clearance"] - e["actual"]["min_clearance"]) for e in multi])),
                "mean_signed_clearance_err_m": float(np.mean(
                    [e[model]["pred_min_clearance"] - e["actual"]["min_clearance"] for e in multi])),
                "events_predicted_safer_by_1m": sum(
                    e[model]["pred_min_clearance"] - e["actual"]["min_clearance"] > 1.0 for e in multi),
            }
        if multi:
            entry["mean_actual_min_clearance_m"] = float(np.mean([e["actual"]["min_clearance"] for e in multi]))
            entry["release_step_agreement"] = (
                f"{sum(e['predicted_release_step'] == e['actual_release_step'] for e in multi)}/{len(multi)}")
        out[category] = entry
    return out


def episode_extras(rec):
    boxes = rec["static_obstacles"]
    clearances = [box_clearance(np.array(r["attacker_pos"]), boxes) for r in rec["rows"]] if boxes else [float("inf")]
    return {"min_static_clearance_m": min(clearances), "frac_avoid": rec["summary"]["frac_obstacle_avoid"],
            "frac_direct": rec["summary"]["frac_direct"]}


def _provenance():
    try:
        return {
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "diff_uav_defend": subprocess.check_output(["git", "diff", "HEAD", "--", "uav_defend"], cwd=ROOT, text=True),
        }
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return {"head": os.environ.get("FIDELITY_PROVENANCE", "unknown (git archive extraction)"), "diff_uav_defend": ""}


def record_run(kind, label):
    """Record episodes only (needs nothing newer than the simulator under test)."""
    directory = OUT / label
    if (directory / "provenance.json").exists():
        raise FileExistsError(f"Completed evidence already exists: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    jobs = list(CRITICAL_CASES) if kind == "cases" else [(lv, s) for lv in ("C1", "C2", "C3") for s in SEEDS]
    for level, seed in jobs:
        rec = record(level, seed)
        (directory / f"{level}_{seed}.json").write_text(json.dumps(rec, default=plain), encoding="utf-8")
        print(label, level, seed, rec["summary"]["outcome"], round(rec["summary"]["path_excess_ratio"], 3), flush=True)
    (directory / "provenance.json").write_text(json.dumps({**_provenance(), "kind": kind}, indent=2), encoding="utf-8")


def analyze(label):
    """Fidelity of every selection plus paired behavior aggregates, from saved records."""
    directory = OUT / label
    records = [json.loads(p.read_text()) for p in sorted(directory.glob("C?_*.json"))]
    provenance = json.loads((directory / "provenance.json").read_text())
    fidelity = []
    for rec in records:
        rec["episode_extras"] = episode_extras(rec)
        fidelity += [{**f, "level": rec["meta"]["level"], "seed": rec["meta"]["seed"]} for f in episode_fidelity(rec)]
    summary = {**provenance, "fidelity": summarize_fidelity(fidelity), "episodes": len(records)}
    if provenance["kind"] == "seeds":
        levels = ("C1", "C2", "C3")
        summary["aggregate"] = aggregate(records)
        summary["min_static_clearance_m"] = {
            lv: min(r["episode_extras"]["min_static_clearance_m"] for r in records if r["meta"]["level"] == lv) for lv in levels}
        for key in ("frac_avoid", "frac_direct"):
            summary[f"mean_{key}"] = {
                lv: float(np.mean([r["episode_extras"][key] for r in records if r["meta"]["level"] == lv])) for lv in levels}
    (directory / "events.json").write_text(json.dumps(fidelity, default=plain), encoding="utf-8")
    (directory / "summary.json").write_text(json.dumps(summary, default=plain, indent=2), encoding="utf-8")
    return summary


def _load(label):
    return {p.name: json.loads(p.read_text()) for p in sorted((OUT / label).glob("C?_*.json"))}


def population_metrics(records):
    """Per-level population metrics for the acceptance gates (tail counts, clearance distribution)."""
    result = {}
    for level in sorted({r["meta"]["level"] for r in records}):
        subset = [r for r in records if r["meta"]["level"] == level]
        summaries = [r["summary"] for r in subset]
        segments = [s for r in subset for s in r["segments"]]
        paths = np.array([s["path_length_m"] for s in summaries])
        ratios = np.array([s["path_excess_ratio"] for s in summaries])
        minimum = np.array([
            min(box_clearance(np.array(row["attacker_pos"]), r["static_obstacles"]) for row in r["rows"])
            if r["static_obstacles"] else np.inf for r in subset])
        selections = [d for r in subset for d in r["decisions"] if d["selection"] is not None]
        result[level] = {
            "episodes": len(subset),
            "capture_rate": float(np.mean([s["captured"] for s in summaries])),
            "collisions": int(sum("collision" in s["outcome"] for s in summaries)),
            "path_mean_m": float(paths.mean()), "path_median_m": float(np.median(paths)), "path_max_m": float(paths.max()),
            "excess_mean": float(ratios.mean()), "excess_max": float(ratios.max()),
            "excess_gt_1.3": int((ratios > 1.3).sum()), "excess_gt_2": int((ratios > 2).sum()), "excess_gt_5": int((ratios > 5).sum()),
            "activations": len(segments),
            "avoid_mean_steps": float(np.mean([g["duration_steps"] for g in segments])) if segments else 0.0,
            "avoid_max_steps": max((g["duration_steps"] for g in segments), default=0),
            "linger_cases": int(sum(s["max_linger_run_steps"] > 4 for s in summaries)),
            "overshoots": int(sum(s["overshoots"] for s in summaries)),
            "orbit_events": int(sum(s["orbit_event"] for s in summaries)),
            "nonreturning_segments": int(sum(g["return_to_direct_step"] is None for g in segments)),
            "unassociated_segments": int(sum(not g["associated_with_perceived_obstacle"] for g in segments)),
            "all_candidates_collide_selections": int(sum(bool(d["fallback"]) for d in selections)),
            "selections": len(selections),
            "evasion_episodes": int(sum(s["any_evasion_active"] for s in summaries)),
            "min_clearance_m": {
                "min": float(minimum.min()), "p1": float(np.percentile(minimum, 1)), "p5": float(np.percentile(minimum, 5)),
                "median": float(np.median(minimum)),
                **{f"lt_{t}": int((minimum < t).sum()) for t in (0.05, 0.10, 0.25, 0.50, 1.0)},
            },
        }
    return result


def compare(base_label, candidate_label):
    """Paired comparison of two recorded sets: collisions gained/lost, tails and clearance."""
    base, candidate = _load(base_label), _load(candidate_label)
    shared = sorted(set(base) & set(candidate))
    collided = lambda rec: "collision" in rec["summary"]["outcome"]
    return {
        "episodes": len(shared),
        "changed_trajectories": [k for k in shared if base[k]["rows"] != candidate[k]["rows"]],
        "collisions_gained": [k for k in shared if collided(candidate[k]) and not collided(base[k])],
        "collisions_lost": [k for k in shared if collided(base[k]) and not collided(candidate[k])],
        "tail_over_5x": {"base": sorted(k for k in shared if base[k]["summary"]["path_excess_ratio"] > 5),
                         "candidate": sorted(k for k in shared if candidate[k]["summary"]["path_excess_ratio"] > 5)},
        "base": population_metrics([base[k] for k in shared]),
        "candidate": population_metrics([candidate[k] for k in shared]),
    }


def controlled(label="controlled", seeds=range(130000, 130200)):
    """Exactness of the planner's predictor with every external change removed.

    The asset is frozen, the attacker perceives the whole static layout and there are no
    moving obstacles, so execution should follow the prediction to float32 precision."""
    def transform(cfg):
        return dataclasses.replace(cfg, enemy_obstacle_awareness_mode="disabled", dynamic_obstacle_count=0)

    events = []
    for level in ("C2", "C3"):
        for seed in seeds:
            rec = record(level, seed, cfg_transform=transform, freeze_asset=True)
            events += [{**e, "level": level, "seed": seed} for e in episode_fidelity(rec)]
    summary = {
        "events": len(events),
        "by_category": {c: sum(e["category"] == c for e in events) for c in ("climb", "lateral", "dynamic")},
        "max_position_error_m": {m: max(e[m]["max_pos_err"] for e in events) for m in ("fixed", "persist", "executed")},
        "events_above_1mm": {m: sum(e[m]["max_pos_err"] > 1e-3 for e in events) for m in ("fixed", "persist", "executed")},
        "release_step_agreement": f"{sum(e['predicted_release_step'] == e['actual_release_step'] for e in events)}/{len(events)}",
    }
    directory = OUT / label
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "controlled.json").write_text(json.dumps(summary, default=plain, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("cases", "seeds", "analyze", "compare", "controlled"))
    parser.add_argument("label", nargs="?", help="result label (compare: the baseline label)")
    parser.add_argument("other", nargs="?", help="compare: the candidate label")
    arguments = parser.parse_args()
    if arguments.action == "controlled":
        print(json.dumps(controlled(), indent=2))
    elif arguments.action == "compare":
        print(json.dumps(compare(arguments.label, arguments.other), default=plain, indent=1))
    elif arguments.action == "analyze":
        print(json.dumps(analyze(arguments.label)["fidelity"], default=plain, indent=1))
    else:
        record_run(arguments.action, arguments.label)
