"""Deterministic route-level sweep + analysis (`a1_route_action_v1`). NO PPO.

Pre-registered BEFORE the sweep (serialized to `route_interface_specification.json`, hashed in the task state, re-checked before the sweep and the analysis):

  intents     DIRECT (canonical A0, zero override) / EVADE_LEFT / EVADE_RIGHT / EVADE_ABOVE; the deterministic existing visibility-graph A* planner plans
              attacker -> tactical subgoal -> asset from the PERCEIVED static obstacle set only; frozen common obstacle layer keeps final authority
  subgoals    q = closest point of the attacker->asset segment to the defender; q_L = q + R l, q_R = q - R l, q_U = q + R z (ceiling-capped); R in {5, 7.5, 10} m
  cadence     T_replan in {1.0, 2.0} s (2 / 4 steps); re-decision (mode switch allowed) only when the cadence expires; other replans: route blocked by a perceived
              obstacle, asset moved (existing RoutePlanner criterion), defender information lost -> DIRECT
  validation  commit only a finite, perceived-collision-free route; rejections: defender_not_ahead (s < 1 m), degenerate_geometry, ceiling (capped vertical offset
              < 0.5 R), no_path, non_finite, planner_collision -> DIRECT.  Log every rejection.
  completion  subgoal reached (2 m) / passed (abeam) and the direct line to the asset is perceived-clear -> DIRECT.
  lockout     after a completion the controller stays DIRECT for one replan interval (design rule added after a 4-scenario smoke test, BEFORE any sweep, because a
              route that can re-arm on the step it completes yields perpetual evasion loops; exposed through the countdown observation slot)
  policies    DIRECT, LEFT, RIGHT, ABOVE (whenever the defender is visible), ALT_LR (alternate at plan commits), RANDOM, MAX_SEP (candidate route with greatest
              defender closest-approach, DIRECT included), ADVERSARIAL rapid switching
  grid        6 configurations (R, T); seeds 145000-145099; C0-C3; greedy, kalman_greedy, pn, lead, rmpc; DIRECT is the matched reference

  G1  per config, level (pooled over defenders) and EVERY tactical policy: collisions <= DIRECT and matched gained collisions == 0.
  G2  per config, level and policy: exhaustion, timeout, orbit(>=1 rev), ceiling/sustained hover <= DIRECT; fly-by (>=1 overshoot) fraction <= DIRECT + 0.5 pp.
  G3  some tactical policy: pooled over its 2,000 matched episodes >= 5% diverge >= 1.0 m from DIRECT AND (>= 10 interception->capture conversions OR
      |mean min-separation difference| >= 0.25 m); positive iff conversions >= reversals.
  G4  per policy: fraction path-excess > 5x <= DIRECT + 0.5 pp (every >5x case classified).
  G5  perception: 0 planner calls without defender information, 0 obstacle ids outside the perceived set, 0 forbidden references in the controller module,
      DIRECT == canonical A0 bit-for-bit.
  G6  stability: all episodes finite, deterministic replay, longest route run <= 40 steps, <= 3 planner calls in any 4 steps, no exceptions.
  Selection among passing configs: (1) zero gained collisions (2) no mission pathology (3) strongest positive tactical effect (max conversions - reversals)
  (4) fewer reversals (5) lower exhaustion (6) lower median path-excess (7) fewer replans (8) smaller R (9) longer cadence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_route as RT
from experiments import journal_v2_a1_route_env as RE
from experiments import journal_v2_a1_rt_state as TS

SEEDS = tuple(range(145_000, 145_100))
LEVELS = A.A1_V2_CLUTTER_LEVELS
FAMILIES = A.A1_V2_DEFENDER_FAMILIES
CONFIGS = tuple((R, T) for R in RT.RADII_M for T in RT.REPLAN_S)
POLICIES = ("DIRECT", "LEFT", "RIGHT", "ABOVE", "ALT_LR", "RANDOM", "MAX_SEP", "ADVERSARIAL")
TACTICAL = POLICIES[1:]
SWEEP_DIR = RE.RESULTS_ROOT / "route_sweep"
CELL_DIR = SWEEP_DIR / "cells"
TRAJ_STEPS = 60
EQUIV_SEEDS_PER_CELL = 5
DIVERGENCE_M, DIVERGENCE_FRACTION, CONVERSIONS_MIN, MIN_SEP_DIFF_M = 1.0, 0.05, 10, 0.25
PATH_EXCESS_5X_ALLOWANCE, OVERSHOOT_ALLOWANCE = 0.005, 0.005
MAX_PLANS_IN_4_STEPS = 3
FORBIDDEN_IN_CONTROLLER = ("_obstacle_layout", "_dynamic_obstacles", "obstacle_layout", "_defender_vel", "soldier_env", "SoldierEnv", "_enemy_perception")


def frozen_specification() -> dict:
    return {
        "action_version": RT.ACTION_VERSION, "observation_version": RT.OBSERVATION_VERSION, "training_protocol": RT.TRAINING_PROTOCOL,
        "reward_version": RE.REWARD_VERSION, "intents": list(RT.ACTIONS), "radii_m": list(RT.RADII_M), "replan_s": list(RT.REPLAN_S),
        "configs": [list(c) for c in CONFIGS], "seeds": [SEEDS[0], SEEDS[-1]], "levels": list(LEVELS), "defenders": list(FAMILIES), "policies": list(POLICIES),
        "planner": "uav_defend.obstacles.route_planner.build_and_search_route (existing visibility-graph + A*, perceived static obstacles only)",
        "validation_rejections": ["defender_not_ahead(s<1 m)", "degenerate_geometry", "ceiling(capped vertical offset < 0.5 R)", "no_path", "non_finite", "planner_collision"],
        "lockout_rule": "after a route completion the controller stays DIRECT for one replan interval (added after a 4-scenario smoke test, before any sweep)",
        "observation": {"dim": RE.OBS_ROUTE_DIM, "extra": [n for n, _ in RE.OBS_ROUTE_EXTRA_SCHEMA],
                        "progress": "1 - |subgoal - p_a| / initial distance to the subgoal, 1 after it is passed, 0 in DIRECT",
                        "countdown": "route: steps until the cadence expires / replan_steps; DIRECT: post-completion hold remaining / replan_steps"},
        "thresholds": {"divergence_m": DIVERGENCE_M, "divergence_fraction": DIVERGENCE_FRACTION, "conversions_min": CONVERSIONS_MIN, "min_sep_diff_m": MIN_SEP_DIFF_M,
                       "path_excess_5x_allowance": PATH_EXCESS_5X_ALLOWANCE, "overshoot_allowance": OVERSHOOT_ALLOWANCE, "route_run_bound_steps": RT.ROUTE_RUN_BOUND_STEPS,
                       "max_plans_in_4_steps": MAX_PLANS_IN_4_STEPS, "min_ahead_m": RT.MIN_AHEAD_M, "min_vertical_fraction": RT.MIN_VERTICAL_FRACTION,
                       "trajectory_steps_compared": TRAJ_STEPS},
        "gates": {"G1": "every tactical policy, every level: collisions <= DIRECT and gained == 0", "G2": "exhausted/timeout/orbit/hover <= DIRECT; fly-by fraction <= DIRECT+0.5pp",
                  "G3": ">=5% diverge >=1 m AND (>=10 conversions OR |mean min-sep diff|>=0.25 m); positive: conversions >= reversals",
                  "G4": "fraction path-excess>5x <= DIRECT + 0.5pp", "G5": "0 plans without info, 0 ids outside perceived, 0 forbidden references, DIRECT == A0",
                  "G6": "finite, deterministic replay, route run <= 40 steps, <= 3 plans in 4 steps"},
        "selection": ["zero gained collisions", "no mission pathology", "strongest positive tactical effect", "fewer reversals", "lower exhaustion",
                      "lower median path-excess", "fewer replans", "smaller R", "longer cadence"],
    }


def spec_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_specification(), sort_keys=True).encode()).hexdigest()


def forbidden_references() -> list[str]:
    """G5: the controller module must not mention ground-truth world state, the simulator, defender velocity or the perception object."""
    import re

    source = Path(RT.__file__).read_text(encoding="utf-8")
    code = re.sub(r'"""[\s\S]*?"""', "", source)                      # docstrings may describe the rule
    code = "\n".join(line.split("#", 1)[0] for line in code.splitlines())
    return [name for name in FORBIDDEN_IN_CONTROLLER if name in code]


# ----------------------------------------------------------------------------- scripted policies
def action_for(name: str, t: int, env: RE.A1RouteTrainingEnv, rng: np.random.Generator) -> int:
    if name == "DIRECT":
        return 0
    if name == "LEFT":
        return 1
    if name == "RIGHT":
        return 2
    if name == "ABOVE":
        return 3
    if name == "ALT_LR":
        return 1 + (env.controller.counters["plans_committed"] % 2)
    if name == "RANDOM":
        return int(rng.integers(0, 4))
    if name == "MAX_SEP":
        if not env.visible():
            return 0
        seps = env.candidate_separations()
        return max(sorted(seps), key=lambda a: seps[a])
    if name == "ADVERSARIAL":
        return (3 * t + t // 3) % 4
    raise ValueError(name)


def run_episode(env: RE.A1RouteTrainingEnv, scenario, name: str, seed: int) -> dict:
    rng = np.random.default_rng(int(seed) * 13 + 5)
    env.reset(options={"scenario": scenario})
    done, t, info = False, 0, {}
    while not done:
        _o, _r, te, tr, info = env.step(action_for(name, t, env, rng))
        done, t = te or tr, t + 1
    return info["a1_episode"]


def divergence(a: list, b: list) -> float:
    n = min(len(a), len(b), TRAJ_STEPS)
    return float(np.max(np.linalg.norm(np.asarray(a[:n]) - np.asarray(b[:n]), axis=1))) if n else 0.0


def compact(ep: dict, base: dict, name: str, level: str, family: str) -> dict:
    is_col = lambda e: e["outcome"] in RE.COLLISIONS
    a = ep["audit"]
    return {
        "level": level, "defender": family, "policy": name, "seed": ep["scenario_seed"], "outcome": ep["outcome"], "steps": ep["steps"], "collision": is_col(ep),
        "exhausted": bool(ep["endurance_exhausted"]), "timeout": ep["outcome"] == "timeout", "captured": ep["outcome"] == "soldier_caught",
        "intercepted": ep["outcome"] in ("intercepted", "unsafe_intercept"), "path_excess": ep["path_excess"], "path_length_m": ep["path_length_m"],
        "min_sep": ep["min_defender_separation_m"], "ceiling_hover": bool(ep["ceiling_hover"]), "sustained_hover": bool(ep["sustained_hover"]),
        "orbit": float(ep["orbit_revolutions_max"]), "overshoots": ep["overshoots"], "finite": bool(ep["finite"]), "route_steps": ep["route_steps"],
        "longest_route_run": ep["longest_route_run"], "plans_attempted": ep["plans_attempted"], "plans_committed": ep["plans_committed"], "replans": ep["replans"],
        "completions": ep["completions"], "invalid": ep["invalid"], "cancels": ep["cancels"], "suppressed_by_obstacle": ep["suppressed_by_obstacle"],
        "waypoint_mean": ep["waypoint_mean"], "route_length_mean": ep["route_length_mean"], "max_plans_in_4_steps": ep["max_plans_in_4_steps"],
        "executed_while_owned": ep["executed_while_owned"], "steps_info_available": ep["steps_info_available"],
        "plan_calls": a["plan_calls"], "plan_calls_without_info": a["plan_calls_without_info"], "ids_outside_perceived": a["ids_outside_perceived"],
        "plan_calls_with_unperceived": a["plan_calls_with_unperceived_obstacles"], "div_m": divergence(ep["trajectory"], base["trajectory"]),
        "base_outcome": base["outcome"], "outcome_differs": ep["outcome"] != base["outcome"], "base_collision": is_col(base),
        "min_sep_diff": ep["min_defender_separation_m"] - base["min_defender_separation_m"], "steps_diff": ep["steps"] - base["steps"],
        "path_diff_m": ep["path_length_m"] - base["path_length_m"], "base_steps": base["steps"],
    }


def cell_path(R: float, T: float, level: str, family: str) -> Path:
    return CELL_DIR / f"sweep_R{str(R).replace('.', 'p')}_T{str(T).replace('.', 'p')}_{level}_{family}.json"


def expected_records() -> int:
    return len(SEEDS) * len(POLICIES)


def verify_cell(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
        body = json.dumps(payload["records"], sort_keys=True).encode()
        return (payload.get("complete") is True and payload.get("spec_sha256") == spec_sha256() and len(payload["records"]) == expected_records()
                and hashlib.sha256(body).hexdigest() == payload["records_sha256"])
    except (OSError, ValueError, KeyError):
        return False


def run_cell(task) -> dict:
    R, T, level, family = task
    path = cell_path(R, T, level, family)
    if verify_cell(path):
        return {"task": task, "skipped": True}
    lock = TS.CellLock(path)
    if not lock.acquire():
        return {"task": task, "skipped": True, "reason": "another worker owns this cell"}
    try:
        started = time.time()
        bank = A.A1V2DefenderBank()
        env = RE.A1RouteTrainingEnv(0, 0, defender_bank=bank, radius_m=R, replan_s=T, record_trajectory=True)
        a0 = RE.A1RouteTrainingEnv(0, 0, defender_bank=bank, scripted_a0=True, radius_m=R, replan_s=T, record_trajectory=True)
        records, a0_ok, det_ok = [], True, True
        for i, seed in enumerate(SEEDS):
            scenario = (level, family, int(seed))
            base = run_episode(env, scenario, "DIRECT", seed)
            if i < EQUIV_SEEDS_PER_CELL:
                ref = run_episode(a0, scenario, "DIRECT", seed)
                a0_ok &= (base["trajectory"] == ref["trajectory"] and base["outcome"] == ref["outcome"] and base["steps"] == ref["steps"])
            kept = {}
            for name in POLICIES:
                ep = base if name == "DIRECT" else run_episode(env, scenario, name, seed)
                records.append(compact(ep, base, name, level, family))
                if i < 3 and name in ("RANDOM", "ADVERSARIAL", "MAX_SEP"):
                    kept[name] = ep
            for name, first in kept.items():
                again = run_episode(env, scenario, name, seed)
                det_ok &= (again["trajectory"] == first["trajectory"] and again["outcome"] == first["outcome"] and again["modes"] == first["modes"])
        body = json.dumps(records, sort_keys=True).encode()
        TS.atomic_write_bytes(path, json.dumps({"complete": True, "spec_sha256": spec_sha256(), "radius_m": R, "replan_s": T, "level": level, "defender": family,
                                                "direct_equals_a0": bool(a0_ok), "deterministic_replay": bool(det_ok), "records_sha256": hashlib.sha256(body).hexdigest(),
                                                "records": json.loads(body)}).encode())
        return {"task": task, "episodes": len(records), "seconds": round(time.time() - started, 1), "direct_equals_a0": bool(a0_ok), "deterministic": bool(det_ok)}
    finally:
        lock.release()


def all_tasks(families: tuple = FAMILIES, levels: tuple = LEVELS) -> list[tuple]:
    tasks = [(R, T, lv, fam) for (R, T) in CONFIGS for lv in LEVELS if lv in levels for fam in FAMILIES if fam in families]
    tasks.sort(key=lambda t: (t[2] != "C3", t[2] != "C2"))  # slowest levels first within a stage; RMPC (very slow: ~1 s/step) is only ever requested explicitly
    return tasks


def run_all(workers: int = 16, families: tuple = FAMILIES, levels: tuple = LEVELS) -> dict:
    CELL_DIR.mkdir(parents=True, exist_ok=True)
    tasks = all_tasks(families, levels)
    pending = [t for t in tasks if not verify_cell(cell_path(*t))]
    print(f"{len(tasks) - len(pending)} cells already verified, {len(pending)} to run (families: {', '.join(families)}; levels: {', '.join(levels)})", flush=True)
    done = 0
    if pending:
        with mp.get_context("spawn").Pool(min(workers, len(pending))) as pool:
            for result in pool.imap_unordered(run_cell, pending):
                done += 1
                print(done, len(pending), result, flush=True)
    return {"cells": len(tasks), "verified": sum(verify_cell(cell_path(*t)) for t in tasks)}


def load_available() -> tuple[list[dict], dict, dict]:
    """All verified cells that exist (the staged sweep covers C0 x 4 defenders and C1-C3 x {greedy, pn}); returns records, flags and an explicit coverage map."""
    records, flags, coverage = [], {"direct_equals_a0": True, "deterministic_replay": True}, {}
    for task in all_tasks():
        path = cell_path(*task)
        if not verify_cell(path):
            continue
        payload = json.loads(path.read_text())
        for r in payload["records"]:
            r["radius_m"], r["replan_s"] = task[0], task[1]
        records.extend(payload["records"])
        for k in flags:
            flags[k] &= bool(payload[k])
        coverage.setdefault(task[2], set()).add(task[3])
    return records, flags, {k: sorted(v) for k, v in sorted(coverage.items())}


def load_cells(families: tuple = FAMILIES, levels: tuple = LEVELS) -> tuple[list[dict], dict]:
    records, flags = [], {"direct_equals_a0": True, "deterministic_replay": True}
    for task in all_tasks(families, levels):
        path = cell_path(*task)
        if not verify_cell(path):
            raise RuntimeError(f"cell {path.name} missing/unverified; re-run the sweep")
        payload = json.loads(path.read_text())
        for r in payload["records"]:
            r["radius_m"], r["replan_s"] = task[0], task[1]
        records.extend(payload["records"])
        for k in flags:
            flags[k] &= bool(payload[k])
    return records, flags


# ----------------------------------------------------------------------------- analysis
def _sum_dicts(series) -> dict:
    total: dict = {}
    for d in series:
        for k, v in d.items():
            total[k] = total.get(k, 0) + v
    return total


def analyze(records: list[dict], flags: dict, forbidden: list[str] | None = None) -> dict:
    import pandas as pd

    forbidden = forbidden_references() if forbidden is None else forbidden
    df = pd.DataFrame(records)
    configs = {}
    for (R, T), cfg in df.groupby(["radius_m", "replan_s"]):
        base = cfg[cfg.policy == "DIRECT"]
        pol = {}
        for policy, cand in cfg[cfg.policy != "DIRECT"].groupby("policy"):
            levels, g1, g2 = {}, True, True
            for level in LEVELS:
                c, b = cand[cand.level == level], base[base.level == level]
                gained = int((c.collision.values & ~c.base_collision.values).sum())
                counts = {k: int(c[k].sum()) for k in ("collision", "exhausted", "timeout", "ceiling_hover", "sustained_hover")}
                ref = {k: int(b[k].sum()) for k in counts}
                counts["orbit"], ref["orbit"] = int((c.orbit >= 1).sum()), int((b.orbit >= 1).sum())
                over, over_ref = float((c.overshoots >= 1).mean()), float((b.overshoots >= 1).mean())
                ok1 = counts["collision"] <= ref["collision"] and gained == 0
                bad = {k: (counts[k], ref[k]) for k in ("exhausted", "timeout", "orbit", "ceiling_hover", "sustained_hover") if counts[k] > ref[k]}
                ok2 = not bad and over <= over_ref + OVERSHOOT_ALLOWANCE
                g1, g2 = g1 and ok1, g2 and ok2
                levels[level] = {"collisions": counts["collision"], "base_collisions": ref["collision"], "gained": gained, "G1": ok1, "exhausted": counts["exhausted"],
                                 "base_exhausted": ref["exhausted"], "timeout": counts["timeout"], "orbit": counts["orbit"], "overshoot_fraction": over,
                                 "base_overshoot_fraction": over_ref, "G2": ok2, "G2_violations": bad}
            conv = int((cand.base_outcome.isin(["intercepted", "unsafe_intercept"]) & (cand.outcome == "soldier_caught")).sum())
            rev = int(((cand.base_outcome == "soldier_caught") & (cand.outcome != "soldier_caught")).sum())
            frac = float((cand.div_m >= DIVERGENCE_M).mean())
            sep = float(cand.min_sep_diff.mean())
            meaningful = bool(frac >= DIVERGENCE_FRACTION and (conv >= CONVERSIONS_MIN or abs(sep) >= MIN_SEP_DIFF_M))
            pe, peb = cand.path_excess.replace([np.inf], np.nan), base.path_excess.replace([np.inf], np.nan)
            f5, f5b = float((pe > 5).mean()), float((peb > 5).mean())
            pol[policy] = {
                "G1": bool(g1), "G2": bool(g2), "G3": meaningful, "G3_positive": bool(meaningful and conv >= rev), "G4": bool(f5 <= f5b + PATH_EXCESS_5X_ALLOWANCE),
                "gained_collisions": int(sum(v["gained"] for v in levels.values())), "per_level": levels,
                "authority": {"fraction_diverging": frac, "conversions": conv, "reversals": rev, "net_captures": int(cand.captured.sum()) - int(base.captured.sum()),
                              "mean_min_sep_diff_m": sep, "outcome_changed": int(cand.outcome_differs.sum()), "median_div_m": float(cand.div_m.median()),
                              "mean_extra_path_m": float(cand.path_diff_m.mean()), "mean_extra_steps": float(cand.steps_diff.mean())},
                "path_excess": {"mean": float(pe.mean()), "median": float(cand.path_excess.median()), "max": float(cand.path_excess.max()), "gt1.3": int((cand.path_excess > 1.3).sum()),
                                "gt2": int((cand.path_excess > 2).sum()), "gt5": int((cand.path_excess > 5).sum()), "fraction_gt5": f5, "base_fraction_gt5": f5b},
                "exhausted": int(cand.exhausted.sum()), "timeouts": int(cand.timeout.sum()), "captures": int(cand.captured.sum()),
                "replans_per_episode": float(cand.replans.mean()), "plans_per_episode": float(cand.plans_attempted.mean()), "route_steps": float(cand.route_steps.mean()),
                "longest_route_run": int(cand.longest_route_run.max()), "max_plans_in_4_steps": int(cand.max_plans_in_4_steps.max()),
                "waypoint_mean": float(cand[cand.plans_committed > 0].waypoint_mean.mean()) if (cand.plans_committed > 0).any() else 0.0,
                "route_length_mean": float(cand[cand.plans_committed > 0].route_length_mean.mean()) if (cand.plans_committed > 0).any() else 0.0,
                "invalid": _sum_dicts(cand.invalid), "cancels": _sum_dicts(cand.cancels), "plans_attempted": int(cand.plans_attempted.sum()),
                "plans_committed": int(cand.plans_committed.sum()), "non_finite": int((~cand.finite).sum()), "executed_while_owned": int(cand.executed_while_owned.sum()),
                "plan_calls_without_info": int(cand.plan_calls_without_info.sum()), "ids_outside_perceived": int(cand.ids_outside_perceived.sum()),
                "plan_calls_with_unperceived": int(cand.plan_calls_with_unperceived.sum()), "suppressed_by_obstacle": int(cand.suppressed_by_obstacle.sum()),
            }
        g1, g2 = all(p["G1"] for p in pol.values()), all(p["G2"] for p in pol.values())
        g3 = any(p["G3"] for p in pol.values())
        g3p = any(p["G3_positive"] for p in pol.values())
        g4 = all(p["G4"] for p in pol.values())
        g5 = (not forbidden and bool(flags["direct_equals_a0"]) and all(p["plan_calls_without_info"] == 0 and p["ids_outside_perceived"] == 0 for p in pol.values()))
        g6 = (all(p["non_finite"] == 0 and p["longest_route_run"] <= RT.ROUTE_RUN_BOUND_STEPS and p["max_plans_in_4_steps"] <= MAX_PLANS_IN_4_STEPS for p in pol.values())
              and bool(flags["deterministic_replay"]))
        best = max(pol.values(), key=lambda p: p["authority"]["conversions"] - p["authority"]["reversals"])
        configs[(float(R), float(T))] = {
            "radius_m": float(R), "replan_s": float(T), "G1": g1, "G2": g2, "G3": g3, "G3_positive": g3p, "G4": g4, "G5": bool(g5), "G6": bool(g6),
            "passed": bool(g1 and g2 and g3p and g4 and g5 and g6), "gained_collisions_total": int(sum(p["gained_collisions"] for p in pol.values())),
            "policies": pol, "best_effect": int(best["authority"]["conversions"] - best["authority"]["reversals"]),
            "reversals_at_best": int(best["authority"]["reversals"]), "exhausted_total": int(sum(p["exhausted"] for p in pol.values())),
            "median_path_excess": float(np.median([p["path_excess"]["median"] for p in pol.values()])),
            "replans_per_episode": float(np.mean([p["replans_per_episode"] for p in pol.values()])),
        }
    passing = [c for c in configs.values() if c["passed"]]
    selected = None
    if passing:
        passing.sort(key=lambda c: (-c["best_effect"], c["reversals_at_best"], c["exhausted_total"], round(c["median_path_excess"], 2), c["replans_per_episode"],
                                    c["radius_m"], -c["replan_s"]))
        selected = {"radius_m": passing[0]["radius_m"], "replan_s": passing[0]["replan_s"]}
    return {"spec_sha256": spec_sha256(), "n_records": int(len(df)), "flags": flags, "forbidden_references": forbidden,
            "configs": [configs[k] for k in sorted(configs)], "selected": selected}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("sweep", "analyze"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--families", type=str, default=",".join(FAMILIES))
    parser.add_argument("--levels", type=str, default=",".join(LEVELS))
    args = parser.parse_args()
    families = tuple(f for f in args.families.split(",") if f)
    levels = tuple(f for f in args.levels.split(",") if f)
    if args.stage == "sweep":
        print(run_all(args.workers, families, levels))
    else:
        recs, flags = load_cells(families, levels)
        result = analyze(recs, flags)
        result["families"], result["levels"] = list(families), list(levels)
        TS.atomic_write_json(SWEEP_DIR / "analysis.json", result)
        print(json.dumps({"selected": result["selected"], "flags": flags, "forbidden": result["forbidden_references"],
                          "configs": [{k: c[k] for k in ("radius_m", "replan_s", "G1", "G2", "G3", "G3_positive", "G4", "G5", "G6", "passed", "gained_collisions_total")}
                                      for c in result["configs"]]}, indent=1))
