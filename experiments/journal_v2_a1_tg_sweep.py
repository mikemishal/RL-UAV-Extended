"""Deterministic geometry sweep + analysis for the finite tactical-pulse action (`a1_v2_action_v5`). NO PPO.

Pre-registered BEFORE the sweep (this specification is serialized to `maneuver_specification.json`, hashed in the task state, and re-checked
against these constants before the sweep and again before the analysis; changing any constant afterwards therefore invalidates the run):

  grid        theta in {10, 15, 20, 30} deg  x  pulse T in {0.5, 1.0, 1.5, 2.0} s  (1/2/3/4 steps at dt = 0.5 s); rejoin = 1.0 s (2 steps)
  seeds       145000-145099; levels C0-C3
  contexts    active fixed defenders {greedy, lead, pn} (real 25 m information gate)  +  `passive_forced` (parked defender, information gate forced
              open: a mission-only STUDY diagnostic that stresses the geometry itself)
  policies    BASE, LEFT, RIGHT, CLIMB, alternate L/R, cycle L/C/R, random {0..3}, adversarial rapid switching (state machine/rejoin still apply)
  reference   `BASE` policy through the same machine (proved bit-identical to canonical A0 in the BASE-equivalence stage)

  G1  per candidate, level, context and policy: collisions(policy) <= collisions(BASE) over the same 100 matched seeds (expected 0).
  G2  per candidate, level, context and policy: ceiling-hover, sustained-hover, orbit (>= 1 revolution), endurance-exhaustion and timeout episode counts
      each <= the BASE count in that cell. (Interception by an active defender is not an architectural failure.)
  G3  exists an active defender and a non-BASE policy such that, pooled over C0-C3 (400 matched episodes), >= 10% of episodes diverge from the matched BASE
      trajectory by >= 2.0 m AND (outcome differs in >= 1 episode OR |mean minimum defender separation difference| >= 0.5 m).
  G4  pooled over the candidate's episodes, fraction with path-excess > 5x <= BASE fraction + 0.5 percentage points.
  G5  zero state-machine invariant violations; longest tactical run <= pulse steps; deterministic replay of sampled episodes identical.
  Selection among candidates passing G1-G5, lexicographic: (1) lowest endurance+timeout excess over BASE, (2) lowest median path-excess (0.01 resolution),
  (3) shorter pulse, (4) smaller angle.  [zero gained collisions, zero orbit/hover and meaningful authority are gates G1-G3.]
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
from experiments import journal_v2_a1_tg as G
from experiments import journal_v2_a1_tg_state as TS

SEEDS = tuple(range(145_000, 145_100))
LEVELS = A.A1_V2_CLUTTER_LEVELS
ACTIVE_DEFENDERS = ("greedy", "lead", "pn")
CONTEXTS = (*ACTIVE_DEFENDERS, "passive_forced")
POLICY_NAMES = ("BASE", "LEFT", "RIGHT", "CLIMB", "ALT_LR", "CYCLE_LCR", "RANDOM", "ADVERSARIAL")
TACTICAL_POLICIES = POLICY_NAMES[1:]
SWEEP_DIR = G.RESULTS_ROOT / "tactical_geometry_sweep"
GROUP_DIR = SWEEP_DIR / "groups"
TRAJ_STEPS = 60
DIVERGENCE_M, DIVERGENCE_FRACTION, MIN_SEP_DIFF_M = 2.0, 0.10, 0.5
PATH_EXCESS_5X_ALLOWANCE = 0.005


def frozen_specification() -> dict:
    return {
        "action_version": G.ACTION_VERSION, "observation_version": G.OBSERVATION_VERSION, "training_protocol": G.TRAINING_PROTOCOL,
        "reward_version": G.REWARD_VERSION, "actions": list(G.ACTIONS), "modes": list(G.MODES),
        "theta_grid_deg": list(G.THETA_GRID_DEG), "pulse_grid_s": list(G.PULSE_GRID_S), "rejoin_s": G.REJOIN_S, "dt_s": G.DT_S,
        "degenerate_horizontal_eps": G.DEGENERATE_HORIZONTAL_EPS, "climb_ceiling_rule": "BASE if (z_max - z) < max_climb_rate * dt",
        "defender_gate": "true range <= config.enemy_defender_detection_range (25 m) [the observation's existing gate]",
        "activation_requires": ["defender information available", "frozen common layer not owning steering (direct rollout safe)",
                                "no active pulse", "rejoin complete"],
        "geometry": {"left": "normalize(m + tan(theta) l)", "right": "normalize(m - tan(theta) l)", "climb": "normalize(m + tan(theta_z) z_hat)",
                     "l": "normalize(z_hat x normalize([m_x, m_y, 0]))", "m": "current unit attacker->asset direction, recomputed every step"},
        "observation_v2": {"dim": G.OBS_V2_DIM, "extra": [n for n, _ in G.OBS_V2_EXTRA_SCHEMA]},
        "seeds": [SEEDS[0], SEEDS[-1]], "levels": list(LEVELS), "contexts": list(CONTEXTS), "policies": list(POLICY_NAMES),
        "thresholds": {"divergence_m": DIVERGENCE_M, "divergence_fraction": DIVERGENCE_FRACTION, "min_sep_diff_m": MIN_SEP_DIFF_M,
                       "path_excess_5x_allowance": PATH_EXCESS_5X_ALLOWANCE, "trajectory_steps_compared": TRAJ_STEPS},
        "gates": {"G1": "collisions(policy) <= collisions(BASE) per candidate x level x context x policy",
                  "G2": "ceiling-hover, sustained-hover, orbit, exhaustion, timeout counts <= BASE per cell",
                  "G3": "exists active defender & policy: >=10% episodes diverge >=2.0 m and (outcome differs or |mean min-separation diff|>=0.5 m)",
                  "G4": "fraction path-excess>5x <= BASE fraction + 0.5pp", "G5": "no machine violations; run<=n; deterministic replay"},
        "selection": ["min endurance+timeout excess over BASE", "min median path-excess (0.01)", "shorter pulse", "smaller angle"],
    }


def spec_sha256() -> str:
    return hashlib.sha256(json.dumps(frozen_specification(), sort_keys=True).encode()).hexdigest()


# ----------------------------------------------------------------------------- scripted policies
def policy_action(name: str, t: int, rng: np.random.Generator) -> int:
    if name == "BASE":
        return 0
    if name == "LEFT":
        return 1
    if name == "RIGHT":
        return 2
    if name == "CLIMB":
        return 3
    if name == "ALT_LR":
        return 1 + (t % 2)
    if name == "CYCLE_LCR":
        return (1, 3, 2)[t % 3]
    if name == "RANDOM":
        return int(rng.integers(0, 4))
    if name == "ADVERSARIAL":
        return 1 + ((5 * t + t // 2) % 3)  # a different tactical request almost every step, including during pulses and rejoin
    raise ValueError(name)


def run_episode(env: G.A1V5TrainingEnv, scenario, name: str, seed: int) -> dict:
    rng = np.random.default_rng(int(seed) * 31 + POLICY_NAMES.index(name))
    _obs, _ = env.reset(options={"scenario": scenario})
    done, t, info = False, 0, {}
    while not done:
        _obs, _r, te, tr, info = env.step(policy_action(name, t, rng))
        done, t = te or tr, t + 1
    return info["a1_episode"]


def divergence(traj_a: list, traj_b: list) -> float:
    n = min(len(traj_a), len(traj_b), TRAJ_STEPS)
    if n == 0:
        return 0.0
    return float(np.max(np.linalg.norm(np.asarray(traj_a[:n]) - np.asarray(traj_b[:n]), axis=1)))


def compact(ep: dict, base: dict, name: str, theta: float, pulse_s: float, level: str, context: str) -> dict:
    return {
        "theta": theta, "pulse_s": pulse_s, "level": level, "context": context, "policy": name, "seed": ep["scenario_seed"],
        "outcome": ep["outcome"], "steps": ep["steps"], "collision": ep["outcome"] in ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"),
        "exhausted": bool(ep["endurance_exhausted"]), "timeout": ep["outcome"] == "timeout", "captured": ep["outcome"] == "soldier_caught",
        "intercepted": ep["outcome"] in ("intercepted", "unsafe_intercept"), "path_excess": ep["path_excess"], "min_sep": ep["min_defender_separation_m"],
        "progress_net": ep["progress_net"], "ceiling_hover": bool(ep["ceiling_hover"]), "sustained_hover": bool(ep["sustained_hover"]),
        "orbit": float(ep["orbit_revolutions_max"]), "overshoots": ep["overshoots"], "tactical_steps": ep["tactical_steps"],
        "longest_tactical_run": ep["longest_tactical_run"], "pulses": int(sum(ep["pulses_started"].values())),
        "pulses_cancelled": ep["pulses_cancelled_by_obstacle"], "pulses_aborted": ep["pulses_aborted_geometry"],
        "violations": ep["state_machine_violations"], "suppressed": ep["suppressed"], "transitions": ep["transitions"],
        "steps_by_mode": ep["steps_by_mode"],
        "div_m": divergence(ep["trajectory"], base["trajectory"]), "base_outcome": base["outcome"],
        "outcome_differs": ep["outcome"] != base["outcome"], "base_collision": base["outcome"] in ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"),
        "min_sep_diff": ep["min_defender_separation_m"] - base["min_defender_separation_m"], "steps_diff": ep["steps"] - base["steps"],
        "base_steps": base["steps"], "base_min_sep": base["min_defender_separation_m"], "mean_altitude_m": ep["mean_altitude_m"],
    }


def group_path(theta: float, pulse_s: float, level: str) -> Path:
    return GROUP_DIR / f"theta{int(theta)}_T{str(pulse_s).replace('.', 'p')}_{level}.json"


def expected_records() -> int:
    return len(POLICY_NAMES) * len(CONTEXTS) * len(SEEDS)


def verify_group(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text())
        body = json.dumps(payload["records"], sort_keys=True).encode()
        return (payload.get("complete") is True and payload.get("spec_sha256") == spec_sha256() and len(payload["records"]) == expected_records()
                and hashlib.sha256(body).hexdigest() == payload["records_sha256"])
    except (OSError, ValueError, KeyError):
        return False


def run_group(task) -> dict:
    theta, pulse_s, level = task
    path = group_path(theta, pulse_s, level)
    if verify_group(path):
        return {"task": task, "skipped": True}
    started = time.time()
    bank = A.A1V2DefenderBank()
    records, determinism_ok = [], True
    for context in CONTEXTS:
        family = G.PASSIVE_DIAGNOSTIC if context == "passive_forced" else context
        env = G.A1V5TrainingEnv(0, 0, defender_bank=bank, theta_deg=theta, pulse_s=pulse_s, gate_override=(context == "passive_forced"),
                                record_trajectory=True)
        for seed in SEEDS:
            scenario = (level, family, int(seed))
            base = run_episode(env, scenario, "BASE", seed)
            sample = seed < SEEDS[0] + 3
            kept = {}
            for name in POLICY_NAMES:
                ep = base if name == "BASE" else run_episode(env, scenario, name, seed)
                records.append(compact(ep, base, name, theta, pulse_s, level, context))
                if sample and name in ("RANDOM", "ADVERSARIAL"):
                    kept[name] = ep
            for name, first in kept.items():  # deterministic-replay sample: a second run must reproduce the first exactly
                again = run_episode(env, scenario, name, seed)
                determinism_ok &= (again["outcome"], again["steps"], again["trajectory"], again["modes"]) == (
                    first["outcome"], first["steps"], first["trajectory"], first["modes"])
    body = json.dumps(records, sort_keys=True).encode()
    TS.atomic_write_bytes(path, json.dumps({"complete": True, "spec_sha256": spec_sha256(), "theta": theta, "pulse_s": pulse_s, "level": level,
                                            "determinism_ok": bool(determinism_ok), "records_sha256": hashlib.sha256(body).hexdigest(),
                                            "records": json.loads(body)}).encode())
    return {"task": task, "episodes": len(records), "seconds": time.time() - started, "determinism_ok": bool(determinism_ok)}


def all_tasks() -> list[tuple]:
    tasks = [(th, T, lv) for th in G.THETA_GRID_DEG for T in G.PULSE_GRID_S for lv in LEVELS]
    tasks.sort(key=lambda t: (t[2] != "C3", t[2] != "C2", t[2] != "C1"))  # heaviest levels first
    return tasks


def run_sweep(workers: int = 16) -> dict:
    GROUP_DIR.mkdir(parents=True, exist_ok=True)
    pending = [t for t in all_tasks() if not verify_group(group_path(*t))]
    print(f"{len(all_tasks()) - len(pending)} groups already verified, {len(pending)} to run", flush=True)
    done = 0
    if pending:
        with mp.get_context("spawn").Pool(min(workers, len(pending))) as pool:
            for result in pool.imap_unordered(run_group, pending):
                done += 1
                print(done, len(pending), result, flush=True)
    return {"groups": len(all_tasks()), "verified": sum(verify_group(group_path(*t)) for t in all_tasks())}


# ----------------------------------------------------------------------------- analysis
def load_all() -> list[dict]:
    records = []
    for task in all_tasks():
        path = group_path(*task)
        if not verify_group(path):
            raise RuntimeError(f"group {path.name} is missing or unverified; re-run the sweep")
        records.extend(json.loads(path.read_text())["records"])
    return records


def analyze(records: list[dict], determinism: dict | None = None) -> dict:
    import pandas as pd

    df = pd.DataFrame(records)
    cell = ["theta", "pulse_s", "level", "context"]
    base = df[df.policy == "BASE"].set_index(cell + ["seed"])
    candidates = {}
    for (theta, pulse_s), cand in df.groupby(["theta", "pulse_s"]):
        n_steps = G.pulse_steps(pulse_s)
        tact = cand[cand.policy != "BASE"]
        cb = cand[cand.policy == "BASE"]
        g1_cells, g2_cells = [], []
        for (level, context, policy), g in tact.groupby(["level", "context", "policy"]):
            ref = cb[(cb.level == level) & (cb.context == context)]
            counts = {k: int(g[k].sum()) for k in ("collision", "ceiling_hover", "sustained_hover", "exhausted", "timeout")}
            counts["orbit"] = int((g.orbit >= 1.0).sum())
            refc = {k: int(ref[k].sum()) for k in ("collision", "ceiling_hover", "sustained_hover", "exhausted", "timeout")}
            refc["orbit"] = int((ref.orbit >= 1.0).sum())
            gained = int(((g.collision.values) & (~g.base_collision.values)).sum())
            g1_cells.append({"level": level, "context": context, "policy": policy, "tactical": counts["collision"], "base": refc["collision"],
                             "gained": gained, "ok": counts["collision"] <= refc["collision"]})
            bad = {k: (counts[k], refc[k]) for k in ("ceiling_hover", "sustained_hover", "orbit", "exhausted", "timeout") if counts[k] > refc[k]}
            g2_cells.append({"level": level, "context": context, "policy": policy, "violations": bad, "ok": not bad})
        g1 = all(c["ok"] for c in g1_cells)
        g2 = all(c["ok"] for c in g2_cells)
        # G3 authority (active defenders only)
        authority = []
        for context in ACTIVE_DEFENDERS:
            for policy in TACTICAL_POLICIES:
                g = tact[(tact.context == context) & (tact.policy == policy)]
                frac = float((g.div_m >= DIVERGENCE_M).mean())
                differs = int(g.outcome_differs.sum())
                sep = float(g.min_sep_diff.mean())
                authority.append({"context": context, "policy": policy, "fraction_diverging": frac, "outcome_differs": differs,
                                  "mean_min_sep_diff": sep, "meaningful": bool(frac >= DIVERGENCE_FRACTION and (differs >= 1 or abs(sep) >= MIN_SEP_DIFF_M))})
        g3 = any(a["meaningful"] for a in authority)
        # G4 path-excess
        pe = cand.path_excess.replace([np.inf], np.nan)
        pe_base = cb.path_excess.replace([np.inf], np.nan)
        frac5, frac5_base = float((pe > 5).mean()), float((pe_base > 5).mean())
        g4 = frac5 <= frac5_base + PATH_EXCESS_5X_ALLOWANCE
        # G5
        violations = int(tact.violations.sum())
        run_ok = bool((tact.longest_tactical_run <= n_steps).all())
        det_ok = all(determinism.get((theta, pulse_s, lv), True) for lv in LEVELS) if determinism is not None else True
        g5 = violations == 0 and run_ok and det_ok
        excess = int(sum(max(int(g.exhausted.sum() + g.timeout.sum()) - int(cb[(cb.level == lv) & (cb.context == cx)][["exhausted", "timeout"]].sum().sum()), 0)
                         for (lv, cx, _p), g in tact.groupby(["level", "context", "policy"])))
        candidates[(float(theta), float(pulse_s))] = {
            "theta": float(theta), "pulse_s": float(pulse_s), "pulse_steps": n_steps,
            "G1": bool(g1), "G2": bool(g2), "G3": bool(g3), "G4": bool(g4), "G5": bool(g5), "passed": bool(g1 and g2 and g3 and g4 and g5),
            "collisions_tactical_total": int(tact.collision.sum()), "collisions_base_total": int(cb.collision.sum()),
            "gained_collisions_total": int(sum(c["gained"] for c in g1_cells)), "g1_failing_cells": [c for c in g1_cells if not c["ok"]][:12],
            "g2_failing_cells": [c for c in g2_cells if not c["ok"]][:12], "authority_best": sorted(authority, key=lambda a: -a["fraction_diverging"])[:3],
            "path_excess": {"mean": float(tact.path_excess.replace([np.inf], np.nan).mean()),
                            "median": float(tact.path_excess.median()), "max": float(tact.path_excess.max()), "gt1.3": int((tact.path_excess > 1.3).sum()),
                            "gt2": int((tact.path_excess > 2).sum()), "gt5": int((tact.path_excess > 5).sum()), "base_gt5": int((cb.path_excess > 5).sum()),
                            "fraction_gt5": frac5, "base_fraction_gt5": frac5_base},
            "endurance_exhausted": int(tact.exhausted.sum()), "timeouts": int(tact.timeout.sum()), "base_exhausted": int(cb.exhausted.sum()),
            "base_timeouts": int(cb.timeout.sum()), "excess_endurance_timeout_over_base": excess,
            "machine_violations": violations, "longest_tactical_run_max": int(tact.longest_tactical_run.max()), "deterministic": bool(det_ok),
            "pulses_started": int(tact.pulses.sum()), "pulses_cancelled_by_obstacle": int(tact.pulses_cancelled.sum()), "pulses_aborted": int(tact.pulses_aborted.sum()),
            "median_path_excess_bin": round(float(tact.path_excess.median()), 2),
        }
    passing = [c for c in candidates.values() if c["passed"]]
    selected = None
    if passing:
        passing.sort(key=lambda c: (c["excess_endurance_timeout_over_base"], c["median_path_excess_bin"], c["pulse_s"], c["theta"]))
        selected = {"theta_deg": passing[0]["theta"], "pulse_s": passing[0]["pulse_s"]}
    return {"spec_sha256": spec_sha256(), "candidates": [candidates[k] for k in sorted(candidates)], "selected": selected, "n_records": int(len(df))}


def load_determinism() -> dict:
    out = {}
    for task in all_tasks():
        payload = json.loads(group_path(*task).read_text())
        out[task] = bool(payload.get("determinism_ok", False))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("sweep", "analyze"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.stage == "sweep":
        print(run_sweep(args.workers))
    else:
        result = analyze(load_all(), load_determinism())
        TS.atomic_write_json(SWEEP_DIR / "analysis.json", result)
        print(json.dumps({"selected": result["selected"], "candidates": [{k: c[k] for k in ("theta", "pulse_s", "G1", "G2", "G3", "G4", "G5", "passed",
                           "gained_collisions_total")} for c in result["candidates"]]}, indent=1))
