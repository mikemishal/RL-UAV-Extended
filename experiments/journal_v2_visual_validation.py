"""V2 visual and behavioral validation driver (validation only).

No training, no evaluation of learned policies, no change to simulation
semantics. Scripted controllers only (passive, greedy, kalman_greedy, pn,
lead, rmpc). Outputs go to results/journal_v2_corrected_semantics/
visual_validation/.

PRE-DECLARED PROTOCOL (fixed before any outcome was inspected)
  * Seeds: 130,000-130,009 for every level and controller (10 seeds; the
    first seed is the previously problematic 130000). Behavioral validation
    only: this block is NOT claimed as unseen for any later evaluation.
  * Levels: C0, C1, C2, C3 (canonical V2 builder, standby True).
  * Controllers: passive (parked far from the engagement, standby disabled,
    never acts: exposes the attacker's full mission), greedy, kalman_greedy,
    pn, lead, rmpc.
  * V1-vs-V2 comparison: C0 and C2, controllers passive and greedy, same
    seeds, V1 built with the explicit V1 builder.
  * Visual subset per level (passive attacker view paired with greedy):
      first 3 seeds, the seed with the most OBSTACLE_AVOID/ROUTE_BYPASS steps
      (levels with obstacles only; lowest seed on ties), and the seed with the
      longest attacker path (lowest seed on ties).
  * Presentation examples: rules are in `select_presentation_cases`.
  * Acceptance criteria: `evaluate_acceptance`.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from uav_defend.envs.soldier_env import SoldierEnv

from experiments import journal_v2_visual_plots as vp
from experiments.journal_v2_protocol import V2_RESULTS_DIR, build_v2_env_config
from experiments.journal_v2_trace import write_trace_jsonl
from experiments.journal_v2_validation_core import (
    CONTROLLERS, LINGER_RUN_LIMIT_STEPS, deviation_segments, record_episode, summarize,
)

OUT = V2_RESULTS_DIR / "visual_validation"
SEEDS = tuple(range(130_000, 130_010))
EXT_SEEDS = tuple(range(130_000, 130_100))
LEVELS = ("C0", "C1", "C2", "C3")
V1_LEVELS = ("C0", "C2")
V1_CONTROLLERS = ("passive", "greedy")
PROBLEM_SEED = 130_000
PRACTICAL_PURSUIT_EFFICIENCY = 0.85
PRACTICAL_CLOSING_FRACTION = 0.90


# --------------------------------------------------------------- pass 1
def _cell(args):
    semantics, level, controller = args
    summaries, tracks, segment_rows = [], {}, []
    for seed in SEEDS:
        rec = record_episode(level, semantics, controller, seed,
                             counterfactual=(semantics == "V2"), detour_cf=(controller == "passive"))
        summaries.append(summarize(rec))
        tracks[seed] = np.array([r["attacker_pos"] for r in rec["rows"]])
        for seg in deviation_segments(rec):
            segment_rows.append({**rec["meta"], **seg})
    return summaries, tracks, segment_rows


def _ext_cell(level):
    summaries, segment_rows = [], []
    for seed in EXT_SEEDS:
        rec = record_episode(level, "V2", "passive", seed, counterfactual=True, detour_cf=True)
        summaries.append(summarize(rec))
        for seg in deviation_segments(rec):
            segment_rows.append({**rec["meta"], **seg})
    return summaries, segment_rows


def run_extension() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Coverage extension E1 (added AFTER the 10-seed set produced only 0/1/4
    avoidance segments in C1/C2/C3): V2 passive attacker, seeds
    130,000-130,099, C1-C3, with the counterfactual necessity test."""
    levels = ("C1", "C2", "C3")
    summaries, segments = [], []
    with ProcessPoolExecutor(max_workers=len(levels)) as pool:
        for level, (s, g) in zip(levels, pool.map(_ext_cell, levels)):
            summaries += s
            segments += g
            print("extension done", level, flush=True)
    return pd.DataFrame(summaries), pd.DataFrame(segments)


def run_pass1() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    jobs = [("V2", lv, c) for lv in LEVELS for c in CONTROLLERS] + [("V1", lv, c) for lv in V1_LEVELS for c in V1_CONTROLLERS]
    summaries, segments, tracks = [], [], {}
    with ProcessPoolExecutor(max_workers=min(len(jobs), os.cpu_count() or 1)) as pool:
        for job, (s, t, g) in zip(jobs, pool.map(_cell, jobs)):
            summaries += s
            segments += g
            for seed, arr in t.items():
                tracks[(job[0], job[1], job[2], seed)] = arr
            print("pass1 done", job, flush=True)
    return pd.DataFrame(summaries), pd.DataFrame(segments), tracks


def invariance_table(tracks: dict) -> pd.DataFrame:
    """Max attacker-position difference between each controller and the passive
    run for the same (level, seed), over steps whose positions are full-step
    (the final step of an episode ending in a swept contact is interpolated)."""
    rows = []
    for level in LEVELS:
        for seed in SEEDS:
            ref = tracks[("V2", level, "passive", seed)]
            for c in CONTROLLERS:
                if c == "passive":
                    continue
                other = tracks[("V2", level, c, seed)]
                m = min(len(ref), len(other)) - 1  # exclude the last (possibly interpolated) row
                delta = float(np.max(np.linalg.norm(ref[:m] - other[:m], axis=1))) if m > 0 else 0.0
                rows.append({"level": level, "seed": seed, "controller": c, "common_steps": m, "max_attacker_delta_m": delta})
    return pd.DataFrame(rows)


# ------------------------------------------------------------- selection
def select_visual_cases(df: pd.DataFrame, ext: pd.DataFrame) -> dict:
    """C0 uses the 10-seed set; C1-C3 take the max-avoidance and longest-path
    seeds from the 100-seed passive coverage extension (the 10-seed set has too
    few avoidance events to select from)."""
    out = {}
    for level in LEVELS:
        if level == "C0":
            p = df[(df.semantics == "V2") & (df.level == level) & (df.controller == "passive")]
        else:
            p = ext[ext.level == level]
        p = p.sort_values("seed").reset_index(drop=True)
        cases = {int(s): ["first3"] for s in SEEDS[:3]}
        avoid_steps = (p.frac_obstacle_avoid + p.frac_route_bypass) * p.steps
        if level != "C0" and avoid_steps.max() > 0:
            cases.setdefault(int(p.loc[int(avoid_steps.idxmax()), "seed"]), []).append("max_avoidance")
        cases.setdefault(int(p.loc[int(p.path_length_m.idxmax()), "seed"]), []).append("longest_path")
        out[level] = cases
    return out


def select_presentation_cases(df: pd.DataFrame, ext: pd.DataFrame) -> dict:
    v2 = df[df.semantics == "V2"]
    passive = v2[v2.controller == "passive"].sort_values(["level", "seed"])
    sel = {"p1_c0_direct_attack": ("C0", "passive", int(passive[(passive.level == "C0") & (passive.seed != PROBLEM_SEED)].seed.min()))}
    sel["p2_c0_seed130000_swept_capture"] = ("C0", "passive", PROBLEM_SEED)
    # p3/p4 are chosen from the 100-seed extension by objective rules (see
    # docs): p3 = a single, counterfactually NECESSARY deviation with the
    # largest lateral excursion in C1/C2; p4 = the C3 episode with the most
    # deviation segments (largest lateral excursion on ties).
    one = ext[ext.level.isin(["C1", "C2"]) & (ext.n_avoidance_segments == 1) & (ext.detours_necessary_collision == 1)]
    row = one.sort_values(["max_lateral_deviation_m", "level", "seed"], ascending=[False, True, True]).iloc[0]
    sel["p3_single_obstacle_bypass"] = (str(row.level), "passive", int(row.seed))
    c3 = ext[ext.level == "C3"].sort_values(["n_avoidance_segments", "max_lateral_deviation_m", "seed"],
                                            ascending=[False, False, True]).iloc[0]
    sel["p4_c3_complex_path"] = ("C3", "passive", int(c3.seed))
    lead = v2[(v2.level == "C0") & (v2.controller == "lead") & (v2.outcome == "intercepted")].sort_values("seed")
    ctrl = "lead" if len(lead) else "greedy"
    pool = lead if len(lead) else v2[(v2.level == "C0") & (v2.controller == "greedy") & (v2.outcome == "intercepted")].sort_values("seed")
    sel["p5_scripted_interception"] = ("C0", ctrl, int(pool.iloc[0].seed))
    return sel


SCENARIO_START = (-20.0, 0.0, 3.0)
SCENARIO_ASSET = (45.0, 0.0, 0.0)
CONSTRUCTED = {
    "S1_single_blocking_obstacle": ((15.0, 0.0, 5.0, 4.0, 4.0, 5.0),),
    "S2_nearby_nonblocking_obstacle": ((15.0, 14.0, 5.0, 4.0, 4.0, 5.0),),
    "S3_two_staggered_obstacles": ((5.0, 2.0, 5.0, 3.0, 3.0, 5.0), (35.0, -2.0, 5.0, 3.0, 3.0, 5.0)),
}


def _scenario_transform(spec):
    def transform(cfg):
        return dataclasses.replace(cfg, obstacle_layout_mode="fixed", obstacle_fixed_spec=spec,
                                   dynamic_obstacles_enabled=False)
    return transform


def _scenario_setup(env) -> None:
    env._enemy_pos = np.array(SCENARIO_START, dtype=np.float32)
    env._enemy_vel = np.zeros(3, dtype=np.float32)
    env._soldier_pos = np.array(SCENARIO_ASSET, dtype=np.float32)
    env._move_soldier = lambda: None  # stationary asset isolates the attacker's obstacle behavior


def run_constructed_scenarios() -> dict:
    """Deterministic, constructed obstacle scenarios (NOT random draws): a
    horizontally visible blocking case, a near-miss that must not trigger
    avoidance, and two staggered obstacles."""
    return {name: record_episode("C2", "V2", "passive", 0, counterfactual=True, detour_cf=True,
                                 cfg_transform=_scenario_transform(spec), setup=_scenario_setup)
            for name, spec in CONSTRUCTED.items()}


def select_swept_interception_example(df: pd.DataFrame):
    v2 = df[(df.semantics == "V2") & (df.controller != "passive") & (df.terminal_event == "interception")
            & (df.endpoint_would_miss_terminal)].copy()
    if v2.empty:
        return None
    order = {c: i for i, c in enumerate(CONTROLLERS)}
    v2["_c"] = v2.controller.map(order)
    v2 = v2.sort_values(["level", "_c", "seed"])
    row = v2.iloc[0]
    return str(row.level), str(row.controller), int(row.seed)


# --------------------------------------------------------- event semantics
def competing_event_cases() -> list[dict]:
    cases = []
    for name, defender_x in (("defender interception earlier (defender at x=-0.1)", -0.1),
                             ("asset capture earlier (defender at x=+1.5)", 1.5),
                             ("simultaneous contact (defender at x=+0.5): documented tie rule", 0.5)):
        cfg = dataclasses.replace(build_v2_env_config("C0", "measurement"), defender_standby_until_detection=False)
        env = SoldierEnv(cfg)
        env.reset(seed=0)
        env._enemy_pos = np.array([-3.2, 0.0, 0.0], dtype=np.float32)
        env._enemy_vel = np.array([cfg.v_e, 0.0, 0.0], dtype=np.float32)
        env._soldier_pos = np.zeros(3, dtype=np.float32)
        env._defender_pos = np.array([defender_x, 0.0, 0.0], dtype=np.float32)
        env._defender_vel = np.zeros(3, dtype=np.float32)
        env._move_soldier = lambda: None
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        env.enemy_tactical_override = lambda: direction
        _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
        cases.append({
            "name": name, "defender_x": defender_x, "x_start": -3.2, "x_end": -3.2 + cfg.v_e * cfg.dt,
            "threat_radius": cfg.threat_radius, "intercept_radius": cfg.intercept_radius,
            "capture_fraction": float(info["capture_contact_fraction"]),
            "interception_fraction": float(info["interception_contact_fraction"]),
            "terminal_event": info["terminal_event"], "outcome": info["outcome"],
            "terminal_fraction": float(info["terminal_event_fraction"]),
        })
    return cases


# ------------------------------------------------------------ acceptance
def evaluate_acceptance(df: pd.DataFrame, segs: pd.DataFrame, inv: pd.DataFrame, competing: list[dict],
                        ext: pd.DataFrame, constructed: dict) -> dict:
    v2 = df[df.semantics == "V2"]
    result: dict = {}
    c0 = v2[v2.level == "C0"]
    passive_c0 = c0[c0.controller == "passive"]
    result["C0"] = {
        "no_weave_or_nondirect_mode": bool((c0.max_abs_weave_bias == 0).all() and (c0.frac_direct == 1.0).all()
                                           and (c0.max_direct_command_error_deg < 0.1).all()),
        "no_orbit_events": int(c0.orbit_event.sum()) == 0, "orbit_events": int(c0.orbit_event.sum()),
        "zero_missed_attacker_captures": int(c0.missed_attacker_contacts.sum()) == 0,
        "missed_attacker_captures": int(c0.missed_attacker_contacts.sum()),
        "zero_missed_defender_interceptions": int(c0.missed_defender_contacts.sum()) == 0,
        "missed_defender_interceptions": int(c0.missed_defender_contacts.sum()),
        "overshoots_total": int(c0.overshoots.sum()),
        "direct_progress": bool(passive_c0.pursuit_efficiency.median() >= PRACTICAL_PURSUIT_EFFICIENCY
                                and passive_c0.closing_step_fraction.median() >= PRACTICAL_CLOSING_FRACTION),
        "median_pursuit_efficiency_passive": float(passive_c0.pursuit_efficiency.median()),
        "median_closing_step_fraction_passive": float(passive_c0.closing_step_fraction.median()),
    }
    for level in ("C1", "C2", "C3"):
        d = v2[v2.level == level]  # 10-seed set, all six controllers
        e = ext[ext.level == level]  # 100-seed passive coverage extension
        result[level] = {
            "all_deviations_associated_with_perceived_obstacle": bool((d.segments_unassociated.sum() + e.segments_unassociated.sum()) == 0),
            "unassociated_segments": int(d.segments_unassociated.sum() + e.segments_unassociated.sum()),
            "all_segments_return_to_direct": bool((d.segments_not_returned.sum() + e.segments_not_returned.sum()) == 0),
            "returns_with_line_still_blocked": int(d.segments_bad_return.sum() + e.segments_bad_return.sum()),
            "no_lingering_beyond_limit": bool(max(d.max_linger_run_steps.max(), e.max_linger_run_steps.max()) <= LINGER_RUN_LIMIT_STEPS),
            "max_linger_run_steps": int(max(d.max_linger_run_steps.max(), e.max_linger_run_steps.max())),
            "no_orbit_events": int(d.orbit_event.sum() + e.orbit_event.sum()) == 0,
            "orbit_events": int(d.orbit_event.sum() + e.orbit_event.sum()),
            "zero_missed_attacker_captures": int(d.missed_attacker_contacts.sum() + e.missed_attacker_contacts.sum()) == 0,
            "zero_missed_defender_interceptions": int(d.missed_defender_contacts.sum()) == 0,
            "no_evasion_mode": bool((d.frac_evade == 0).all() and (e.frac_evade == 0).all()),
            "extension_episodes": int(len(e)),
            "extension_capture_rate_passive": float(e.captured.mean()),
            "extension_avoidance_segments": int(e.n_avoidance_segments.sum()),
            "extension_episodes_with_avoidance": int((e.n_avoidance_segments > 0).sum()),
            "extension_necessary_collision": int(e.detours_necessary_collision.sum()),
            "extension_necessary_margin": int(e.detours_necessary_margin.sum()),
            "extension_unnecessary_detours": int(e.detours_unnecessary.sum()),
            "extension_median_segment_steps": float(np.median([x for x in e.max_segment_steps if x > 0])) if (e.max_segment_steps > 0).any() else 0.0,
            "extension_max_segment_steps": int(e.max_segment_steps.max()),
            "extension_overshoot_episodes": int((e.overshoots > 0).sum()),
            "extension_median_path_excess_ratio": float(e.path_excess_ratio.median()),
            "extension_max_path_excess_ratio": float(e.path_excess_ratio.max()),
        }
    summaries = {name: summarize(rec) for name, rec in constructed.items()}
    segs_c = {name: deviation_segments(rec) for name, rec in constructed.items()}
    s1, s2, s3 = (summaries[n] for n in CONSTRUCTED)
    result["constructed_scenarios"] = {
        "S1_deviates_and_returns": bool(s1["n_avoidance_segments"] >= 1 and s1["segments_not_returned"] == 0 and s1["captured"]
                                        and all(s["necessity"] == "necessary_collision" for s in segs_c["S1_single_blocking_obstacle"])),
        "S2_no_unnecessary_deviation": bool(s2["n_avoidance_segments"] == 0 and s2["captured"]),
        "S3_two_obstacles_two_excursions_both_returned": bool(s3["n_avoidance_segments"] >= 2 and s3["segments_not_returned"] == 0 and s3["captured"]),
        "no_collision_any_scenario": bool(all("collision" not in s["outcome"] for s in summaries.values())),
        "S1_max_lateral_deviation_m": float(s1["max_lateral_deviation_m"]),
        "S3_max_lateral_deviation_m": float(s3["max_lateral_deviation_m"]),
    }
    result["scripted_defenders"] = {
        "no_evasion_active_any_episode": bool(not v2.any_evasion_active.any()),
        "no_evade_mode_any_episode": bool((v2.frac_evade == 0).all()),
        "max_attacker_delta_across_controllers_m": float(inv.max_attacker_delta_m.max()),
        "attacker_independent_of_defender": bool(inv.max_attacker_delta_m.max() <= 1e-6),
    }
    ok_event = [c["terminal_event"] == ("interception" if c["interception_fraction"] < c["capture_fraction"] else "capture")
                for c in competing]
    result["event_semantics"] = {
        "competing_event_winners_correct": bool(all(ok_event)),
        "missed_attacker_contacts_all_v2": int(v2.missed_attacker_contacts.sum() + ext.missed_attacker_contacts.sum()),
        "missed_defender_contacts_all_v2": int(v2.missed_defender_contacts.sum()),
        "zero_missed_contacts_all_v2": bool((v2.missed_attacker_contacts.sum() + ext.missed_attacker_contacts.sum()
                                             + v2.missed_defender_contacts.sum()) == 0),
    }
    flags = []
    for key, section in result.items():
        for name, value in section.items():
            if isinstance(value, bool):
                flags.append((f"{key}.{name}", value))
    result["_failed_checks"] = [n for n, v in flags if not v]
    result["_all_pass"] = not result["_failed_checks"]
    return result


# ---------------------------------------------------------------- tables
def v2_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    v2 = df[df.semantics == "V2"]
    rows = []
    for (level, controller), g in v2.groupby(["level", "controller"], sort=False):
        rows.append({
            "Level": level, "Controller": controller, "N": len(g),
            "Asset-capture rate": float(g.captured.mean()),
            "Interception rate": float((g.outcome == "intercepted").mean()),
            "Mean capture time (s)": float(g.capture_time_s.mean()) if g.captured.any() else float("nan"),
            "Mean path length (m)": float(g.path_length_m.mean()),
            "Mean heading change (deg)": float(g.heading_change_deg.mean()),
            "Overshoots": int(g.overshoots.sum()), "Orbit events": int(g.orbit_event.sum()),
            "Missed swept contacts (att+def)": int(g.missed_attacker_contacts.sum() + g.missed_defender_contacts.sum()),
        })
    return pd.DataFrame(rows)


def v1_v2_table(df: pd.DataFrame, segs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level in V1_LEVELS:
        for controller in V1_CONTROLLERS:
            for semantics in ("V1", "V2"):
                g = df[(df.level == level) & (df.controller == controller) & (df.semantics == semantics)]
                rows.append({
                    "Level": level, "Controller": controller, "Semantics": semantics, "N": len(g),
                    "Asset-capture rate": float(g.captured.mean()),
                    "Interception rate": float((g.outcome == "intercepted").mean()),
                    "Timeout rate": float((g.outcome == "timeout").mean()),
                    "Mean capture time (s)": float(g.capture_time_s.mean()) if g.captured.any() else float("nan"),
                    "Mean path length (m)": float(g.path_length_m.mean()),
                    "Mean heading change (deg)": float(g.heading_change_deg.mean()),
                    "Median pursuit efficiency": float(g.pursuit_efficiency.median()),
                    "Overshoots": int(g.overshoots.sum()), "Orbit events": int(g.orbit_event.sum()),
                    "Missed attacker contacts": int(g.missed_attacker_contacts.sum()),
                    "Missed defender contacts": int(g.missed_defender_contacts.sum()),
                    "Avoidance/evasion segments": int(g.n_avoidance_segments.sum()),
                    "Segments w/o blocked line or dynamic obstacle": int(g.segments_prediction_only.sum()),
                    "Evasion segments": int(g.segments_evasion.sum()),
                    "Detours necessary (direct flight would collide or breach 2 m)": int(g.detours_necessary_collision.sum() + g.detours_necessary_margin.sum()),
                    "Detours UNnecessary": int(g.detours_unnecessary.sum()),
                    "Detours not evaluated": int(g.detours_not_evaluated.sum()),
                    "% flight DIRECT": float(100 * g.frac_direct.mean()),
                    "% flight OBSTACLE_AVOID": float(100 * g.frac_obstacle_avoid.mean()),
                    "% flight EVADE": float(100 * g.frac_evade.mean()),
                })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------ main
def _dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def main() -> None:
    for sub in ("c0", "c1", "c2", "c3", "v1_vs_v2", "event_semantics", "presentation", "traces"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    df, segs, tracks = run_pass1()
    df.to_csv(OUT / "episodes_all.csv", index=False)
    segs.to_csv(OUT / "avoidance_segments.csv", index=False)
    inv = invariance_table(tracks)
    inv.to_csv(OUT / "attacker_invariance_across_controllers.csv", index=False)
    ext, ext_segs = run_extension()
    ext.to_csv(OUT / "coverage_extension_passive_100seeds.csv", index=False)
    ext_segs.to_csv(OUT / "coverage_extension_segments.csv", index=False)

    visual = select_visual_cases(df, ext)
    presentation = select_presentation_cases(df, ext)
    swept_example = select_swept_interception_example(df)
    _dump(OUT / "seed_selection.json", {
        "seeds": list(SEEDS), "extension_seeds": [EXT_SEEDS[0], EXT_SEEDS[-1]], "levels": list(LEVELS),
        "controllers": list(CONTROLLERS),
        "visual_cases_per_level": {lv: {str(k): v for k, v in c.items()} for lv, c in visual.items()},
        "presentation_cases": {k: list(v) for k, v in presentation.items()},
        "swept_interception_example": swept_example,
        "disclosure": "The 10-seed set produced only 0/1/4 avoidance segments in C1/C2/C3, so a 100-seed passive "
                      "coverage extension was added AFTER that observation; presentation cases p3/p4 and the C1-C3 "
                      "max-avoidance/longest-path visual cases are selected from it by the objective rules in "
                      "select_visual_cases/select_presentation_cases.",
        "rules": "see module docstring of experiments/journal_v2_visual_validation.py",
    })

    cache: dict = {}

    def rec(level, semantics, controller, seed, cf=False):
        key = (level, semantics, controller, seed, cf)
        if key not in cache:
            cache[key] = record_episode(level, semantics, controller, seed, counterfactual=cf)
        return cache[key]

    # Per-level cases: passive (full attacker mission) beside greedy (with defender).
    for level, cases in visual.items():
        folder = OUT / level.lower()
        for seed, reasons in cases.items():
            a, b = rec(level, "V2", "passive", seed), rec(level, "V2", "greedy", seed)
            tag = "_".join(reasons)
            vp.plot_pair(a, b, folder / f"{level}_seed{seed}_{tag}.png",
                         f"V2 {level} seed {seed} ({', '.join(reasons)}): attacker-only mission (left) and with greedy defender (right)")
            write_trace_jsonl(a["rows"], OUT / "traces" / f"trace_V2_{level}_passive_seed{seed}.jsonl")
            write_trace_jsonl(b["rows"], OUT / "traces" / f"trace_V2_{level}_greedy_seed{seed}.jsonl")
        longest = [s for s, r in cases.items() if "longest_path" in r][0]
        vp.plot_3d([rec(level, "V2", "passive", longest)], folder / f"{level}_seed{longest}_3d.png",
                   f"V2 {level} seed {longest}: 3-D attacker mission", views=((22, -60), (60, -90)))
        overlay = [rec(level, "V2", c, PROBLEM_SEED) for c in CONTROLLERS]
        delta = float(inv[(inv.level == level) & (inv.seed == PROBLEM_SEED)].max_attacker_delta_m.max())
        vp.plot_controller_overlay(overlay, folder / f"{level}_seed{PROBLEM_SEED}_controller_overlay.png", delta,
                                   f"V2 {level} seed {PROBLEM_SEED}: A0 ignores the defender (all six controllers)")

    # V1 vs V2.
    for level, controller in (("C0", "passive"), ("C0", "greedy"), ("C2", "passive"), ("C2", "greedy")):
        v1, v2 = rec(level, "V1", controller, PROBLEM_SEED), rec(level, "V2", controller, PROBLEM_SEED)
        vp.plot_v1_vs_v2(v1, v2, OUT / "v1_vs_v2" / f"{level}_{controller}_seed{PROBLEM_SEED}_v1_vs_v2.png",
                         f"{level} {controller} seed {PROBLEM_SEED}: V1 endpoint contact vs V2 swept contact")
        write_trace_jsonl(v1["rows"], OUT / "traces" / f"trace_V1_{level}_{controller}_seed{PROBLEM_SEED}.jsonl")
        write_trace_jsonl(v2["rows"], OUT / "traces" / f"trace_V2_{level}_{controller}_seed{PROBLEM_SEED}.jsonl")
    vp.plot_3d([rec("C0", "V1", "passive", PROBLEM_SEED), rec("C0", "V2", "passive", PROBLEM_SEED)],
               OUT / "v1_vs_v2" / f"C0_seed{PROBLEM_SEED}_3d_v1_vs_v2.png",
               f"C0 seed {PROBLEM_SEED}: 3-D, V1 (left) vs V2 (right)", views=((22, -60), (75, -90)))
    vp.plot_3d([rec("C0", "V2", "greedy", PROBLEM_SEED)], OUT / "v1_vs_v2" / f"C0_seed{PROBLEM_SEED}_3d_v2_greedy.png",
               f"C0 seed {PROBLEM_SEED}: 3-D, V2 with greedy defender", views=((22, -60),))

    # Event semantics.
    competing = competing_event_cases()
    vp.plot_competing_events(competing, OUT / "event_semantics" / "competing_events.png")
    _dump(OUT / "event_semantics" / "competing_events.json", competing)
    swept_info = None
    if swept_example is not None:
        level, controller, seed = swept_example
        sw = rec(level, "V2", controller, seed, cf=True)
        swept_info = vp.plot_swept_interception(sw, OUT / "event_semantics" / f"swept_interception_{level}_{controller}_seed{seed}.png")
        swept_info.update(level=level, controller=controller, seed=seed)
        _dump(OUT / "event_semantics" / "swept_interception_example.json", swept_info)
    v2 = df[df.semantics == "V2"]
    tunnel = (v2[v2.terminal_event.isin(["capture", "interception"])]
              .groupby(["terminal_event", "endpoint_would_miss_terminal"]).size().rename("episodes").reset_index())
    tunnel.to_csv(OUT / "event_semantics" / "terminal_steps_endpoint_would_miss.csv", index=False)

    # Presentation examples (PNG + GIF).
    interpretations = {}
    for name, (level, controller, seed) in presentation.items():
        r = rec(level, "V2", controller, seed)
        vp.plot_case(r, OUT / "presentation" / f"{name}.png")
        vp.save_gif(r, OUT / "presentation" / f"{name}.gif")
        s = summarize(r)
        interpretations[name] = {"case": [level, controller, seed], "summary": {
            k: s[k] for k in ("outcome", "steps", "path_length_m", "heading_change_deg", "overshoots", "n_avoidance_segments",
                              "pursuit_efficiency", "frac_direct")}}
    for level, controller in (("C0", "passive"), ("C0", "greedy")):
        vp.save_gif(rec(level, "V1", controller, PROBLEM_SEED), OUT / "v1_vs_v2" / f"{level}_{controller}_seed{PROBLEM_SEED}_V1.gif")
        vp.save_gif(rec(level, "V2", controller, PROBLEM_SEED), OUT / "v1_vs_v2" / f"{level}_{controller}_seed{PROBLEM_SEED}_V2.gif")
    if swept_example is not None:
        shutil.copy(next((OUT / "event_semantics").glob("swept_interception_*.png")), OUT / "presentation" / "p6_swept_defender_interception.png")

    # Constructed deterministic obstacle scenarios (horizontal bypass made visible).
    constructed = run_constructed_scenarios()
    scen_rows = []
    for name, srec in constructed.items():
        folder = OUT / "c2" if name.startswith("S") else OUT
        vp.plot_case(srec, OUT / "presentation" / f"constructed_{name}.png")
        vp.plot_3d([srec], OUT / "presentation" / f"constructed_{name}_3d.png", f"Constructed scenario {name}: 3-D",
                   views=((22, -60), (70, -90)))
        write_trace_jsonl(srec["rows"], OUT / "traces" / f"trace_constructed_{name}.jsonl")
        for seg in deviation_segments(srec):
            scen_rows.append({"scenario": name, **{k: seg[k] for k in (
                "activation_step", "last_avoid_step", "return_to_direct_step", "duration_steps", "kind", "necessity",
                "line_clear_at_return", "associated_with_perceived_obstacle")}})
        if name != "S2_nearby_nonblocking_obstacle":
            vp.save_gif(srec, OUT / "presentation" / f"constructed_{name}.gif")
    pd.DataFrame([{**{"scenario": n}, **{k: v for k, v in summarize(r).items() if k in (
        "outcome", "steps", "path_length_m", "straight_line_distance_m", "path_excess_ratio", "max_lateral_deviation_m",
        "n_avoidance_segments", "detours_necessary_collision", "detours_unnecessary", "overshoots", "heading_change_deg")}}
        for n, r in constructed.items()]).to_csv(OUT / "constructed_scenarios_summary.csv", index=False)
    pd.DataFrame(scen_rows).to_csv(OUT / "constructed_scenarios_segments.csv", index=False)
    _dump(OUT / "presentation" / "cases.json", interpretations)

    # Tables and acceptance.
    t1 = v2_summary_table(df)
    t1.to_csv(OUT / "summary_table_v2.csv", index=False)
    (OUT / "summary_table_v2.md").write_text(to_markdown(t1))
    t2 = v1_v2_table(df, segs)
    t2.to_csv(OUT / "v1_vs_v2" / "comparison_table.csv", index=False)
    (OUT / "v1_vs_v2" / "comparison_table.md").write_text(to_markdown(t2))
    acceptance = evaluate_acceptance(df, segs, inv, competing, ext, constructed)
    _dump(OUT / "acceptance.json", acceptance)
    print(to_markdown(t1))
    print(to_markdown(t2))
    print(json.dumps(acceptance, indent=2, default=str))


if __name__ == "__main__":
    main()
