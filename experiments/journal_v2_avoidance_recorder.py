"""Observational recorder for the V2 scripted attacker's obstacle planner (diagnostics only).

`record(level, seed)` flies one passive-A0 episode (defender parked, no learned policy, no
training) and returns the usual `journal_v2_validation_core` episode record plus, for every
planner call: the inputs (position, velocity, direct direction, target, dynamics limits),
the output mode and persisted waypoint, the selected candidate's evaluation, and the
obstacle layout when a candidate was (re)selected. The planner is wrapped only to read; no
decision is ever changed. `aggregate` summarizes paired-seed behavior per clutter level.

Used by `journal_v2_planner_fidelity` (the command-line driver); see
docs/journal_v2_planner_prediction_fidelity.md.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from experiments.journal_v2_validation_core import deviation_segments, record_episode, summarize

SEEDS = range(130000, 130100)
CASES = (("C2", 130065), ("C3", 130063), ("C3", 130081))


def plain(value):
    """`json.dumps(default=...)` hook for numpy values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value)}")


def record(level, seed, controller="passive", cfg_transform=None, freeze_asset=False):
    """One recorded V2 episode with planner decisions.

    `cfg_transform` edits the EnvConfig before construction and `freeze_asset` stops the
    asset's random walk; both exist only for controlled-conditions diagnostics."""
    import uav_defend.obstacles.avoidance as avoidance

    decisions = []
    captured = []
    original_evaluate = avoidance.evaluate_and_select_candidate

    def capturing_evaluate(*args, **kwargs):
        best, fallback_used = original_evaluate(*args, **kwargs)
        captured.append(best)
        return best, fallback_used

    def setup(env):
        if freeze_asset:
            env._move_soldier = lambda: None
        planner = env._enemy_obstacle_planner
        original = planner.plan

        def observed(position, velocity, u_base, layout, target_position, limits):
            before = planner.active
            captured.clear()
            result = original(position, velocity, u_base, layout, target_position, limits)
            selected = captured[-1] if captured else None
            index = result["obstacle_index"]
            obstacle = layout[index] if index is not None else None
            decisions.append({
                "position": np.array(position).tolist(), "velocity": np.array(velocity).tolist(),
                "target": np.array(target_position).tolist(), "direct": np.array(u_base).tolist(),
                "limits": dataclasses.asdict(limits), "active": result["active"],
                "activation": result["active"] and not before, "mode": result["mode"],
                "waypoint": None if planner.waypoint is None else planner.waypoint.tolist(),
                "obstacle_min": None if obstacle is None else obstacle.min_corner.tolist(),
                "obstacle_max": None if obstacle is None else obstacle.max_corner.tolist(),
                "fallback": result["fallback_used"],
                "selection": None if selected is None else {
                    "mode": selected.mode, "collision": bool(selected.collision),
                    "min_clearance": float(selected.min_clearance), "progress": float(selected.progress),
                    "meets_clearance": bool(selected.meets_clearance),
                },
                "layout": None if selected is None else [
                    {"min": o.min_corner.tolist(), "max": o.max_corner.tolist()} for o in layout.obstacles],
                "stop_radius": planner.mission_stop_radius,
            })
            return result

        planner.plan = observed

    avoidance.evaluate_and_select_candidate = capturing_evaluate
    try:
        rec = record_episode(level, "V2", controller, seed, detour_cf=False, setup=setup, cfg_transform=cfg_transform)
    finally:
        avoidance.evaluate_and_select_candidate = original_evaluate
    rec["decisions"] = decisions
    rec["summary"] = summarize(rec)
    rec["segments"] = segment_details(rec)
    return rec


def segment_details(rec):
    """Per-avoidance-segment activation state, candidate, path length, turning and progress."""
    result = []
    for segment in deviation_segments(rec):
        start, end = segment["activation_step"], segment["last_avoid_step"]
        decision = rec["decisions"][start - 1]
        positions = np.array([r["attacker_pos"] for r in rec["rows"][start - 1:end + 1]])
        velocities = np.array([r["attacker_vel"] for r in rec["rows"][start - 1:end + 1]])
        norms = np.linalg.norm(velocities, axis=1)
        headings = velocities / np.maximum(norms[:, None], 1e-12)
        turns = np.degrees(np.arccos(np.clip(np.sum(headings[1:] * headings[:-1], axis=1), -1, 1)))
        direction = np.array(decision["direct"])
        velocity = np.array(decision["velocity"])
        result.append({
            **segment, "activation_position": decision["position"], "velocity": decision["velocity"],
            "speed": float(np.linalg.norm(velocity)),
            "heading_deg": math.degrees(math.atan2(velocity[1], velocity[0])),
            "asset_bearing_deg": math.degrees(math.atan2(direction[1], direction[0])),
            "candidate": decision["mode"], "waypoint": decision["waypoint"],
            "obstacle_min": decision["obstacle_min"], "obstacle_max": decision["obstacle_max"],
            "path_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
            "heading_change_deg": float(turns.sum()),
            "direct_progress_m": float(np.dot(positions[-1] - positions[0], direction)),
        })
    return result


def aggregate(records):
    """Capture, path, avoidance, orbit, collision and association metrics per clutter level."""
    result = {}
    for level in sorted({r["meta"]["level"] for r in records}):
        subset = [r for r in records if r["meta"]["level"] == level]
        summaries = [r["summary"] for r in subset]
        segments = [s for r in subset for s in r["segments"]]
        paths = [s["path_length_m"] for s in summaries]
        ratios = [s["path_excess_ratio"] for s in summaries]
        result[level] = {
            "episodes": len(subset), "capture_rate": np.mean([s["captured"] for s in summaries]),
            "path_median_m": np.median(paths), "path_mean_m": np.mean(paths), "path_max_m": max(paths),
            "excess_median": np.median(ratios), "excess_mean": np.mean(ratios), "excess_max": max(ratios),
            "path_excess_cases": sum(x > 1.3 for x in ratios),
            "activations": len(segments),
            "avoid_mean_steps": np.mean([s["duration_steps"] for s in segments]) if segments else 0,
            "avoid_max_steps": max((s["duration_steps"] for s in segments), default=0),
            "linger_cases": sum(s["max_linger_run_steps"] > 4 for s in summaries),
            "linger_max_steps": max(s["max_linger_run_steps"] for s in summaries),
            "heading_mean_deg": np.mean([s["heading_change_deg"] for s in summaries]),
            "heading_max_deg": max(s["heading_change_deg"] for s in summaries),
            "orbit_count": sum(s["orbit_event"] for s in summaries),
            "overshoot_count": sum(s["overshoots"] for s in summaries),
            "nonreturning_segments": sum(s["return_to_direct_step"] is None for s in segments),
            "unassociated_segments": sum(not s["associated_with_perceived_obstacle"] for s in segments),
            "missed_contacts": sum(s["missed_attacker_contacts"] + s["missed_defender_contacts"] for s in summaries),
            "collisions": sum("collision" in s["outcome"] for s in summaries),
            "weave_max": max(s["max_abs_weave_bias"] for s in summaries),
            "direct_error_max_deg": max(s["max_direct_command_error_deg"] for s in summaries),
            "evasion_episodes": sum(s["any_evasion_active"] for s in summaries),
            "chatter_pairs": sum(chatter_pairs(r) for r in subset),
        }
    return result


def chatter_pairs(rec):
    """AVOID -> DIRECT -> AVOID on the same obstacle with at most one DIRECT step between."""
    segments = rec["segments"]
    return sum(
        a["return_to_direct_step"] is not None
        and b["activation_step"] - a["return_to_direct_step"] <= 1
        and (a["obstacle_type"], a["obstacle_id"]) == (b["obstacle_type"], b["obstacle_id"])
        for a, b in zip(segments, segments[1:])
    )
