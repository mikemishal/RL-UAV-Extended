"""Deterministic fixed-blend / scripted-switching sweep + analysis for the defender-evasion gating interface (`a1_v2_action_v6`). NO PPO.

Pre-registered BEFORE any sweep (serialized to `interface_specification.json`, hashed in the task state, re-checked before the sweep and before the analysis):

  blends      b in {0.00 (BASE), 0.25, 0.50, 0.75, 1.00}: u_base(b) = normalize(m + b * evasion_component) with the unmodified `_compute_enemy_evasion`
  policies    BASE; FIXED_b (request b every step; the env forces 0 when defender information is unavailable, true range > 25 m);
              SWITCH_b (BASE unless defender information is available, then b)  [expected trajectory-identical to FIXED_b: the primitive itself is zero
              beyond 20 m and the gate is 25 m; the equality is MEASURED, not assumed]
  seeds       145000-145099; levels C0-C3; defenders greedy, kalman_greedy, pn, lead, rmpc  (matched scenarios; BASE is the matched reference)

  G1  per level (pooled over defenders): collisions(candidate) <= collisions(BASE); matched gained collisions reported (expected 0).
  G2  per level: endurance-exhaustion, timeout, orbit(>=1 rev), ceiling-hover, sustained-hover episode counts each <= BASE; fraction of episodes with
      >=1 fly-by overshoot <= BASE fraction + 0.5 pp.
  G3  nonzero blend has authority iff, pooled over all matched episodes, >= 5% diverge >= 1.0 m from the matched BASE trajectory AND
      (>= 10 matched BASE-interception -> capture conversions OR |mean minimum defender separation difference| >= 0.25 m).  'Positive' additionally
      requires conversions >= reversals (capture -> interception).
  G4  pooled fraction of path-excess > 5x <= BASE fraction + 0.5 pp (every >5x case is investigated).
  G5  BASE (blend 0) bit-identical to canonical A0 (trajectory, outcome, collisions) over the whole grid.
  G6  obstacle authority: evasion is applied before the frozen planner AND is suppressed whenever the planner owns steering (executed-while-owned = 0)
      AND zero gained collisions are attributable to a step on which evasion was executed within the last 3 steps.
  Selection (nonzero blends passing G1-G4, positive effect): (1) zero gained collisions, (2) no mission pathology, (3) positive tactical effect,
  (4) lower median path-excess, (5) lower endurance+timeout, (6) smaller blend; then the smallest useful set (BASE+LOW+HIGH only if LOW and HIGH
  differ meaningfully: >= 5% of matched episodes diverge >= 1 m between them), no redundant strengths.
  Switching stress (final set only): alternate BASE/max every step, max blend whenever visible, random, adversarial rapid switching: require zero gained
  collisions, no orbit/loiter pathology, endurance exhaustion <= BASE, finite/numerically stable.
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
from experiments import journal_v2_a1_eg as E
from experiments import journal_v2_a1_eg_state as TS

SEEDS = tuple(range(145_000, 145_100))
LEVELS = A.A1_V2_CLUTTER_LEVELS
FAMILIES = A.A1_V2_DEFENDER_FAMILIES
NONZERO = E.BLENDS[1:]
POLICIES = ("BASE", *[f"FIXED_{b:.2f}" for b in NONZERO], *[f"SWITCH_{b:.2f}" for b in NONZERO])
SWEEP_DIR = E.RESULTS_ROOT / "evasion_gate_sweep"
CELL_DIR = SWEEP_DIR / "cells"
STRESS_DIR = SWEEP_DIR / "stress"
TRAJ_STEPS = 60
DIVERGENCE_M, DIVERGENCE_FRACTION, CONVERSIONS_MIN, MIN_SEP_DIFF_M = 1.0, 0.05, 10, 0.25
PATH_EXCESS_5X_ALLOWANCE, OVERSHOOT_ALLOWANCE = 0.005, 0.005
STRESS_POLICIES = ("ALTERNATE", "MAX_WHEN_VISIBLE", "RANDOM", "ADVERSARIAL")


def frozen_specification() -> dict:
    return {
        "action_version": E.ACTION_VERSION, "observation_version": E.OBSERVATION_VERSION, "training_protocol": E.TRAINING_PROTOCOL,
        "reward_version": E.REWARD_VERSION, "blends": list(E.BLENDS), "primitive": "SoldierEnv._compute_enemy_evasion (unmodified)",
        "combination": "u = normalize(m + b * evasion_component); b = 0 -> env u_base bit-for-bit", "seeds": [SEEDS[0], SEEDS[-1]], "levels": list(LEVELS),
        "defenders": list(FAMILIES), "policies": list(POLICIES), "stress_policies": list(STRESS_POLICIES),
        "gating": "executed blend forced to 0 when true defender range > enemy_defender_detection_range (A1 observation gate); suppressed when the frozen "
                  "common layer owns steering",
        "thresholds": {"divergence_m": DIVERGENCE_M, "divergence_fraction": DIVERGENCE_FRACTION, "conversions_min": CONVERSIONS_MIN,
                       "min_sep_diff_m": MIN_SEP_DIFF_M, "path_excess_5x_allowance": PATH_EXCESS_5X_ALLOWANCE, "overshoot_allowance": OVERSHOOT_ALLOWANCE,
                       "trajectory_steps_compared": TRAJ_STEPS},
        "gates": {"G1": "per level pooled collisions(candidate) <= collisions(BASE); matched gained collisions == 0",
                  "G2": "per level exhausted/timeout/orbit/ceiling-hover/sustained-hover <= BASE; overshoot-episode fraction <= BASE + 0.5pp",
                  "G3": ">=5% episodes diverge >=1 m AND (>=10 interception->capture conversions OR |mean min-sep diff|>=0.25 m); positive: conversions >= reversals",
                  "G4": "fraction path-excess>5x <= BASE + 0.5pp", "G5": "blend 0 bit-identical to canonical A0",
                  "G6": "evasion precedes frozen planner, suppressed when planner owns steering, no gained collision within 3 steps of executed evasion"},
        "selection": ["zero gained collisions", "no mission pathology", "positive tactical effect", "lower median path-excess", "lower endurance+timeout",
                      "smaller blend", "smallest useful set (BASE+LOW+HIGH only if distinct), no redundant strengths"],
    }


def spec_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_specification(), sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------- scripted policies
def action_for(name: str, t: int, env: E.A1V6TrainingEnv, rng: np.random.Generator, n_actions: int) -> int:
    top = n_actions - 1
    if name == "BASE":
        return 0
    if name.startswith("FIXED_"):
        return E.BLENDS.index(float(name.split("_")[1]))
    if name.startswith("SWITCH_"):
        return E.BLENDS.index(float(name.split("_")[1])) if env.visible() else 0
    if name == "ALTERNATE":
        return top if t % 2 == 0 else 0
    if name == "MAX_WHEN_VISIBLE":
        return top if env.visible() else 0
    if name == "RANDOM":
        return int(rng.integers(0, n_actions))
    if name == "ADVERSARIAL":  # rapid switching through every strength including BASE, phase-shifted so it never settles
        return (3 * t + t // 3) % n_actions
    raise ValueError(name)


def run_episode(env: E.A1V6TrainingEnv, scenario, name: str, seed: int) -> dict:
    rng = np.random.default_rng(int(seed) * 17 + 3)
    env.reset(options={"scenario": scenario})
    done, t, info = False, 0, {}
    while not done:
        _o, _r, te, tr, info = env.step(action_for(name, t, env, rng, env.action_space.n))
        done, t = te or tr, t + 1
    return info["a1_episode"]


def divergence(a: list, b: list) -> float:
    n = min(len(a), len(b), TRAJ_STEPS)
    return float(np.max(np.linalg.norm(np.asarray(a[:n]) - np.asarray(b[:n]), axis=1))) if n else 0.0


def compact(ep: dict, base: dict, name: str, level: str, family: str) -> dict:
    is_col = lambda e: e["outcome"] in E.COLLISIONS
    return {
        "level": level, "defender": family, "policy": name, "seed": ep["scenario_seed"], "outcome": ep["outcome"], "steps": ep["steps"],
        "collision": is_col(ep), "exhausted": bool(ep["endurance_exhausted"]), "timeout": ep["outcome"] == "timeout", "captured": ep["outcome"] == "soldier_caught",
        "intercepted": ep["outcome"] in ("intercepted", "unsafe_intercept"), "path_excess": ep["path_excess"], "path_length_m": ep["path_length_m"],
        "min_sep": ep["min_defender_separation_m"], "progress_net": ep["progress_net"], "ceiling_hover": bool(ep["ceiling_hover"]),
        "sustained_hover": bool(ep["sustained_hover"]), "orbit": float(ep["orbit_revolutions_max"]), "overshoots": ep["overshoots"],
        "evasion_steps": ep["evasion_steps"], "requested_steps": ep["requested_steps"], "suppressed": ep["suppressed"],
        "steps_primitive_active": ep["steps_primitive_active"], "steps_info_available": ep["steps_info_available"], "steps_obstacle_owned": ep["steps_obstacle_owned"],
        "max_deflection_deg": ep["max_deflection_deg"], "evasion_in_last3": ep["evasion_in_last3"], "executed_while_owned": ep["executed_while_owned"],
        "div_m": divergence(ep["trajectory"], base["trajectory"]), "base_outcome": base["outcome"], "outcome_differs": ep["outcome"] != base["outcome"],
        "base_collision": is_col(base), "min_sep_diff": ep["min_defender_separation_m"] - base["min_defender_separation_m"], "steps_diff": ep["steps"] - base["steps"],
        "path_diff_m": ep["path_length_m"] - base["path_length_m"], "base_steps": base["steps"],
    }


def cell_path(kind: str, level: str, family: str, tag: str = "") -> Path:
    return (CELL_DIR if kind == "sweep" else STRESS_DIR) / f"{kind}{tag}_{level}_{family}.json"


def expected_records(kind: str) -> int:
    return len(SEEDS) * (len(POLICIES) if kind == "sweep" else 1 + len(STRESS_POLICIES))


def verify_cell(path: Path, kind: str = "sweep", extra_sha: str | None = None) -> bool:
    try:
        payload = json.loads(path.read_text())
        body = json.dumps(payload["records"], sort_keys=True).encode()
        return (payload.get("complete") is True and payload.get("spec_sha256") == spec_sha256() and payload.get("extra_sha") == extra_sha
                and len(payload["records"]) == expected_records(kind) and hashlib.sha256(body).hexdigest() == payload["records_sha256"])
    except (OSError, ValueError, KeyError):
        return False


def run_cell(task) -> dict:
    kind, level, family, levels = task  # levels: tuple of blend levels for the env (stress cells use the selected final set)
    extra = hashlib.sha256(json.dumps(levels).encode()).hexdigest()[:16] if kind == "stress" else None
    path = cell_path(kind, level, family, "" if extra is None else "_" + extra)
    if verify_cell(path, kind, extra):
        return {"task": task, "skipped": True}
    lock = TS.CellLock(path)
    if not lock.acquire():
        return {"task": task, "skipped": True, "reason": "another worker owns this cell"}
    try:
        started = time.time()
        names = POLICIES if kind == "sweep" else ("BASE", *STRESS_POLICIES)
        env = E.A1V6TrainingEnv(0, 0, defender_bank=A.A1V2DefenderBank(), blend_levels=tuple(levels), record_trajectory=True)
        a0 = E.A1V6TrainingEnv(0, 0, defender_bank=A.A1V2DefenderBank(), scripted_a0=True, blend_levels=tuple(levels), record_trajectory=True)
        records, det_ok, a0_ok, sw_fix_ok, finite_ok = [], True, True, True, True
        for seed in SEEDS:
            scenario = (level, family, int(seed))
            base = run_episode(env, scenario, "BASE", seed)
            ref = run_episode(a0, scenario, "BASE", seed)
            a0_ok &= (base["trajectory"] == ref["trajectory"] and base["outcome"] == ref["outcome"] and base["steps"] == ref["steps"])
            eps = {}
            for name in names:
                ep = base if name == "BASE" else run_episode(env, scenario, name, seed)
                eps[name] = ep
                finite_ok &= bool(np.isfinite(np.asarray(ep["trajectory"], dtype=float)).all()) and np.isfinite(ep["path_length_m"])
                records.append(compact(ep, base, name, level, family))
            if kind == "sweep":
                for b in NONZERO:
                    x, y = eps[f"FIXED_{b:.2f}"], eps[f"SWITCH_{b:.2f}"]
                    sw_fix_ok &= (x["trajectory"] == y["trajectory"] and x["outcome"] == y["outcome"])
                fixed = {b: eps[f"FIXED_{b:.2f}"]["trajectory"] for b in NONZERO}
                for r in records[-len(names):]:
                    if r["policy"].startswith("FIXED_"):
                        mine = float(r["policy"].split("_")[1])
                        for other in NONZERO:
                            r[f"div_to_{other:.2f}"] = divergence(fixed[mine], fixed[other]) if other != mine else 0.0
            if seed < SEEDS[0] + 3:  # deterministic replay of the stressing policies
                probe = "FIXED_1.00" if kind == "sweep" else "ADVERSARIAL"
                again = run_episode(env, scenario, probe, seed)
                det_ok &= (again["trajectory"] == eps[probe]["trajectory"] and again["outcome"] == eps[probe]["outcome"])
        body = json.dumps(records, sort_keys=True).encode()
        TS.atomic_write_bytes(path, json.dumps({
            "complete": True, "spec_sha256": spec_sha256(), "extra_sha": extra, "kind": kind, "level": level, "defender": family,
            "base_equals_a0": bool(a0_ok), "switch_equals_fixed": bool(sw_fix_ok), "deterministic_replay": bool(det_ok), "finite": bool(finite_ok),
            "records_sha256": hashlib.sha256(body).hexdigest(), "records": json.loads(body)}).encode())
        return {"task": task, "episodes": len(records), "seconds": round(time.time() - started, 1), "base_equals_a0": bool(a0_ok),
                "switch_equals_fixed": bool(sw_fix_ok), "deterministic": bool(det_ok)}
    finally:
        lock.release()


def all_tasks(kind: str = "sweep", levels: tuple = E.BLENDS) -> list[tuple]:
    tasks = [(kind, lv, fam, tuple(levels)) for lv in LEVELS for fam in FAMILIES]
    tasks.sort(key=lambda t: (t[2] != "rmpc", t[1] != "C3", t[1] != "C2"))  # slowest cells first
    return tasks


def cell_for(task) -> Path:
    kind, level, family, levels = task
    return cell_path(kind, level, family, "" if kind == "sweep" else "_" + hashlib.sha256(json.dumps(levels).encode()).hexdigest()[:16])


def cell_ok(task) -> bool:
    kind, _l, _f, levels = task
    extra = hashlib.sha256(json.dumps(levels).encode()).hexdigest()[:16] if kind == "stress" else None
    return verify_cell(cell_for(task), kind, extra)


def run_all(kind: str = "sweep", levels: tuple = E.BLENDS, workers: int = 10) -> dict:
    (CELL_DIR if kind == "sweep" else STRESS_DIR).mkdir(parents=True, exist_ok=True)
    tasks = all_tasks(kind, levels)
    pending = [t for t in tasks if not cell_ok(t)]
    print(f"{len(tasks) - len(pending)} cells already verified, {len(pending)} to run", flush=True)
    done = 0
    if pending:
        with mp.get_context("spawn").Pool(min(workers, len(pending))) as pool:
            for result in pool.imap_unordered(run_cell, pending):
                done += 1
                print(done, len(pending), result, flush=True)
    return {"cells": len(tasks), "verified": sum(cell_ok(t) for t in tasks)}


def load_cells(kind: str = "sweep", levels: tuple = E.BLENDS) -> tuple[list[dict], dict]:
    records, flags = [], {"base_equals_a0": True, "switch_equals_fixed": True, "deterministic_replay": True, "finite": True}
    for task in all_tasks(kind, levels):
        if not cell_ok(task):
            raise RuntimeError(f"cell {cell_for(task).name} missing/unverified; re-run the sweep")
        payload = json.loads(cell_for(task).read_text())
        records.extend(payload["records"])
        for k in flags:
            flags[k] &= bool(payload[k])
    return records, flags


# ----------------------------------------------------------------------------- analysis
def _cnt(g, col):
    return int(g[col].sum())


def analyze(records: list[dict], flags: dict) -> dict:
    import pandas as pd

    df = pd.DataFrame(records)
    base = df[df.policy == "BASE"]
    out = {}
    for policy, cand in df[df.policy != "BASE"].groupby("policy"):
        per_level, g1, g2 = {}, True, True
        for level in LEVELS:
            c, b = cand[cand.level == level], base[base.level == level]
            gained = int((c.collision.values & ~c.base_collision.values).sum())
            counts = {k: _cnt(c, k) for k in ("collision", "exhausted", "timeout", "ceiling_hover", "sustained_hover")}
            ref = {k: _cnt(b, k) for k in counts}
            counts["orbit"], ref["orbit"] = int((c.orbit >= 1).sum()), int((b.orbit >= 1).sum())
            over, over_ref = float((c.overshoots >= 1).mean()), float((b.overshoots >= 1).mean())
            ok1 = counts["collision"] <= ref["collision"] and gained == 0
            bad = {k: (counts[k], ref[k]) for k in ("exhausted", "timeout", "orbit", "ceiling_hover", "sustained_hover") if counts[k] > ref[k]}
            ok2 = not bad and over <= over_ref + OVERSHOOT_ALLOWANCE
            g1, g2 = g1 and ok1, g2 and ok2
            per_level[level] = {"collisions": counts["collision"], "base_collisions": ref["collision"], "gained": gained, "G1": ok1,
                                "exhausted": counts["exhausted"], "base_exhausted": ref["exhausted"], "timeout": counts["timeout"], "base_timeout": ref["timeout"],
                                "orbit": counts["orbit"], "base_orbit": ref["orbit"], "overshoot_fraction": over, "base_overshoot_fraction": over_ref,
                                "G2": ok2, "G2_violations": bad}
        conv = int(((cand.base_outcome.isin(["intercepted", "unsafe_intercept"])) & (cand.outcome == "soldier_caught")).sum())
        rev = int(((cand.base_outcome == "soldier_caught") & cand.outcome.isin(["intercepted", "unsafe_intercept"])).sum())
        frac_div = float((cand.div_m >= DIVERGENCE_M).mean())
        sep = float(cand.min_sep_diff.mean())
        meaningful = bool(frac_div >= DIVERGENCE_FRACTION and (conv >= CONVERSIONS_MIN or abs(sep) >= MIN_SEP_DIFF_M))
        pe, pe_b = cand.path_excess.replace([np.inf], np.nan), base.path_excess.replace([np.inf], np.nan)
        f5, f5b = float((pe > 5).mean()), float((pe_b > 5).mean())
        gained_all = cand[cand.collision & ~cand.base_collision]
        veto_gap = int((gained_all.evasion_in_last3 > 0).sum())
        g6 = int(cand.executed_while_owned.sum()) == 0 and veto_gap == 0
        out[policy] = {
            "policy": policy, "G1": bool(g1), "G2": bool(g2), "G3": meaningful, "G3_positive": bool(meaningful and conv >= rev), "G4": bool(f5 <= f5b + PATH_EXCESS_5X_ALLOWANCE),
            "G6": bool(g6), "gained_collisions_total": int(sum(v["gained"] for v in per_level.values())), "per_level": per_level,
            "authority": {"fraction_diverging_ge_1m": frac_div, "conversions_int_to_capture": conv, "reversals_capture_to_int": rev,
                          "mean_min_sep_diff_m": sep, "outcome_changed_episodes": int(cand.outcome_differs.sum()), "mean_steps_diff": float(cand.steps_diff.mean()),
                          "mean_path_diff_m": float(cand.path_diff_m.mean()), "median_div_m": float(cand.div_m.median()), "max_div_m": float(cand.div_m.max()),
                          "episodes_with_evasion": int((cand.evasion_steps > 0).sum()), "mean_evasion_steps": float(cand.evasion_steps.mean()),
                          "max_deflection_deg": float(cand.max_deflection_deg.max())},
            "path_excess": {"mean": float(pe.mean()), "median": float(cand.path_excess.median()), "max": float(cand.path_excess.max()), "gt1.3": int((cand.path_excess > 1.3).sum()),
                            "gt2": int((cand.path_excess > 2).sum()), "gt5": int((cand.path_excess > 5).sum()), "fraction_gt5": f5, "base_fraction_gt5": f5b},
            "exhausted": int(cand.exhausted.sum()), "timeouts": int(cand.timeout.sum()), "base_exhausted": int(base.exhausted.sum()), "base_timeouts": int(base.timeout.sum()),
            "veto_gap_collisions": veto_gap, "executed_while_owned": int(cand.executed_while_owned.sum()),
            "suppression_totals": _sum_dicts(cand.suppressed),
        }
    result = {"spec_sha256": spec_sha256(), "n_records": int(len(df)), "flags": flags, "policies": out}
    result["G5"] = bool(flags["base_equals_a0"])
    result["selection"] = select_action_set(out, df, flags)
    return result


def _sum_dicts(series) -> dict:
    total: dict = {}
    for d in series:
        for k, v in d.items():
            total[k] = total.get(k, 0) + v
    return total


def select_action_set(out: dict, df, flags: dict) -> dict:
    ok = [b for b in NONZERO if (p := out[f"FIXED_{b:.2f}"])["G1"] and p["G2"] and p["G3_positive"] and p["G4"] and p["G6"]]
    if not flags["base_equals_a0"] or not ok:
        return {"passing_blends": ok, "final_blends": None, "reason": "no nonzero blend passes G1-G4/G6 with positive effect" if not ok else "G5 failed"}
    ok.sort(key=lambda b: (out[f"FIXED_{b:.2f}"]["path_excess"]["median"] // 0.01, out[f"FIXED_{b:.2f}"]["exhausted"] + out[f"FIXED_{b:.2f}"]["timeouts"], b))
    ranked = list(ok)
    low, high = min(ok), max(ok)
    chosen = [low]
    if high != low:
        rows = df[df.policy == f"FIXED_{low:.2f}"]
        frac = float((rows[f"div_to_{high:.2f}"] >= DIVERGENCE_M).mean())
        if frac >= DIVERGENCE_FRACTION:
            chosen = [low, high]
    return {"passing_blends": sorted(ok), "ranked": ranked, "final_blends": [0.0, *sorted(chosen)], "low_high_divergence_fraction":
            (float((df[df.policy == f"FIXED_{low:.2f}"][f"div_to_{high:.2f}"] >= DIVERGENCE_M).mean()) if high != low else None)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("sweep", "stress", "analyze"))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--levels", type=str, default="")
    args = parser.parse_args()
    if args.stage == "sweep":
        print(run_all("sweep", E.BLENDS, args.workers))
    elif args.stage == "stress":
        print(run_all("stress", tuple(json.loads(args.levels)), args.workers))
    else:
        recs, flags = load_cells()
        result = analyze(recs, flags)
        TS.atomic_write_json(SWEEP_DIR / "analysis.json", result)
        print(json.dumps({"G5": result["G5"], "flags": flags, "selection": result["selection"],
                          "policies": {k: {g: v[g] for g in ("G1", "G2", "G3", "G3_positive", "G4", "G6", "gained_collisions_total")} for k, v in result["policies"].items()}}, indent=1))
