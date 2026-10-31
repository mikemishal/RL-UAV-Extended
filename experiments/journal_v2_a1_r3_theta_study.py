"""theta_max selection study for `a1_v2_action_v3` (bounded residual + A1 tactical safety filter). NO PPO training.

Seeds 145000-145099, C0-C3, parked passive defender (only attacker / frozen obstacle layer / dynamics matter), theta in
{15, 30, 45, 60} deg, six stress commands (random, persistent left / right / up / down, alternating) PLUS the matched
zero-residual reference (canonical u_base: the scripted A0 attacker, no tactical override) on the same scenarios.

GATES, frozen before the study is run (calculation language):

 G1 (primary, safety)  For every theta, clutter level and stress command:
        N_collision(filtered stress command)  <=  N_collision(zero-residual reference)
    counted over the same 100 matched scenarios. The reference is expected to be 0 everywhere; then the filtered action
    must also have 0 collisions. "Gained" collisions (filtered collides where the reference does not) are also reported.
 G2 (C0 mission direction)  For every PERSISTENT command (left, right, up, down, alternating) at C0, per theta:
        mean net mission progress > 0;  timeout count == 0;  endurance-exhaustion count <= reference count;
        episodes with path-excess > 5x <= reference count;  no ceiling-hover episode.
 G3  executed residual angle never exceeds theta (+0.01 deg).
 G4  ceiling-hover episodes <= 1% for every theta x clutter pooled over commands.
 G5  numerical stability: no exception, no non-finite state/command.
Selection: the LARGEST theta in {15, 30, 45, 60} passing G1-G5. If none passes, training must not start.
Filter-intervention statistics are reported as usability diagnostics and are NOT gated.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_a1 as A
from experiments.journal_v2_a1_r2_theta_study import COMMANDS, LEVELS, SEEDS, COLLISION_OUTCOMES, stress_action
from experiments.journal_v2_validation_core import PARKED_DEFENDER, build_config
from uav_defend.envs.soldier_env import SoldierEnv

STUDY_DIR = R.RESULTS_ROOT / "theta_study_action_v3"
PERSISTENT = ("left", "right", "up", "down", "alternating")
ALL_KINDS = ("zero", *COMMANDS)


def run_episode(theta: float, kind: str, level: str, seed: int, safety_filter: bool = True) -> dict:
    cfg, _ = build_config(level, "V2", "passive")
    env = SoldierEnv(cfg)
    rng = np.random.default_rng(seed + 17 * (COMMANDS.index(kind) + 1)) if kind != "zero" else None
    state = {"step": 0, "proposals": 0, "interventions": 0, "angle_max": 0.0, "proposed_sum": 0.0, "executed_sum": 0.0,
             "min_clearance": float("inf")}

    def override():
        u_b = R.base_direction(env)
        if u_b is None:
            return np.zeros(3)
        command, angle = R.residual_command(stress_action(kind, u_b, state["step"], rng), u_b, theta)
        if not command.any():
            return np.zeros(3)
        state["proposals"] += 1
        state["proposed_sum"] += angle
        if safety_filter:
            check = R.tactical_safety_check(env, command)
            state["min_clearance"] = min(state["min_clearance"], check["min_clearance"])
            if not check["safe"]:
                state["interventions"] += 1
                return np.zeros(3)
        state["angle_max"] = max(state["angle_max"], angle)
        state["executed_sum"] += angle
        return command

    env.enemy_tactical_override = None if kind == "zero" else override
    env.reset(seed=int(seed))
    env._defender_pos = np.asarray(PARKED_DEFENDER, dtype=np.float32)
    env._defender_vel = np.zeros(3, dtype=np.float32)
    start = np.asarray(env._enemy_pos, dtype=np.float64).copy()
    d0 = float(np.linalg.norm(np.asarray(env._soldier_pos, dtype=np.float64) - start))
    prev, speeds, alts, finite = start.copy(), [], [], True
    done, info = False, {}
    while not done:
        _o, _r, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float32))
        state["step"] += 1
        pos = np.asarray(info["enemy_pos"], dtype=np.float64)
        finite &= bool(np.isfinite(pos).all() and np.isfinite(np.asarray(info["enemy_final_command"])).all())
        speeds.append(float(np.linalg.norm(pos - prev)) / cfg.dt)
        alts.append(float(pos[2]))
        prev = pos
        done = bool(terminated or truncated)
    chord = float(np.linalg.norm(prev - start))
    path = float(info["enemy_endurance_traveled_m"])
    hover = R.hover_metrics(np.array(speeds), np.array(alts), float(cfg.max_altitude))
    outcome = info["outcome"]
    return {"theta": theta, "command": kind, "level": level, "seed": int(seed), "outcome": outcome, "steps": state["step"],
            "collision": outcome in COLLISION_OUTCOMES, "exhausted": outcome == "enemy_endurance_exhausted",
            "timeout": outcome == "timeout", "captured": outcome == "soldier_caught",
            "path_excess": path / chord if chord > 1e-9 else float("nan"),
            "progress_net": 1.0 - float(info["enemy_soldier_dist"]) / d0, "max_residual_angle_deg": state["angle_max"],
            "proposals": state["proposals"], "interventions": state["interventions"],
            "mean_proposed_angle_deg": state["proposed_sum"] / max(state["proposals"], 1),
            "mean_executed_angle_deg": state["executed_sum"] / max(state["proposals"] - state["interventions"], 1),
            "mean_speed_mps": float(np.mean(speeds)), "finite": bool(finite), **hover}


def run_task(task):
    theta, kind, level = task
    t0 = time.time()
    rows = []
    for seed in SEEDS:
        try:
            rows.append(run_episode(theta, kind, level, seed))
        except Exception as exc:
            rows.append({"theta": theta, "command": kind, "level": level, "seed": int(seed), "error": repr(exc), "finite": False})
    return rows, time.time() - t0


def run_study(workers: int = 16) -> list[dict]:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(0.0, "zero", lv) for lv in LEVELS] + [(th, k, lv) for th in R.THETA_CANDIDATES_DEG for k in COMMANDS for lv in LEVELS]
    tasks.sort(key=lambda t: t[2] != "C3")
    records = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, (rows, seconds) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            records.extend(rows)
            print(i, len(tasks), rows[0]["theta"], rows[0]["command"], rows[0]["level"], round(seconds, 1), flush=True)
    (STUDY_DIR / "records.json").write_text(json.dumps(records, default=float))
    return records


def evaluate(records: list[dict]) -> dict:
    import pandas as pd

    df = pd.DataFrame(records)
    errors = int(df["error"].notna().sum()) if "error" in df else 0
    ok = df[df["error"].isna()] if "error" in df else df
    ref = ok[ok.command == "zero"].set_index(["level", "seed"])
    out = {"episodes": len(df), "errors": errors, "finite_all": bool(df["finite"].all()),
           "reference": {lv: {"collisions": int(g.collision.sum()), "exhausted": int(g.exhausted.sum()), "captured": int(g.captured.sum()),
                              "path_gt5x": int((g.path_excess > 5).sum()), "n": int(len(g))} for lv, g in ok[ok.command == "zero"].groupby("level")}}
    verdicts, tables = {}, {"collision": [], "c0": [], "intervention": []}
    for theta in R.THETA_CANDIDATES_DEG:
        t = ok[(ok.theta == theta) & (ok.command != "zero")]
        g1 = g2 = True
        for (level, kind), g in t.groupby(["level", "command"]):
            r = ref.loc[level]
            gained = int(sum(bool(c) and not bool(ref.loc[(level, s), "collision"]) for c, s in zip(g.collision, g.seed)))
            n_f, n_z = int(g.collision.sum()), int(r.collision.sum())
            tables["collision"].append({"theta": theta, "level": level, "command": kind, "filtered": n_f, "reference": n_z, "gained": gained,
                                        "ok": n_f <= n_z})
            g1 &= n_f <= n_z
            if level == "C0" and kind in PERSISTENT:
                gp = r.path_excess.gt(5).sum()
                row = {"theta": theta, "command": kind, "progress": float(g.progress_net.mean()), "timeouts": int(g.timeout.sum()),
                       "exhausted": int(g.exhausted.sum()), "ref_exhausted": int(r.exhausted.sum()), "path_gt5x": int((g.path_excess > 5).sum()),
                       "ref_path_gt5x": int(gp), "ceiling_hover": int(g.ceiling_hover.sum()), "captured": int(g.captured.sum())}
                row["ok"] = bool(row["progress"] > 0 and row["timeouts"] == 0 and row["exhausted"] <= row["ref_exhausted"]
                                 and row["path_gt5x"] <= row["ref_path_gt5x"] and row["ceiling_hover"] == 0)
                tables["c0"].append(row)
                g2 &= row["ok"]
        for level, g in t.groupby("level"):
            tables["intervention"].append({"theta": theta, "level": level, "proposals": int(g.proposals.sum()), "interventions": int(g.interventions.sum()),
                                           "rate": float(g.interventions.sum() / max(g.proposals.sum(), 1))})
        for kind, g in t.groupby("command"):
            tables["intervention"].append({"theta": theta, "command": kind, "proposals": int(g.proposals.sum()), "interventions": int(g.interventions.sum()),
                                           "rate": float(g.interventions.sum() / max(g.proposals.sum(), 1))})
        g3 = bool((t.max_residual_angle_deg <= theta + 0.01).all())
        g4 = bool(all(g.ceiling_hover.mean() <= 0.01 for _, g in t.groupby("level")))
        g5 = bool(out["finite_all"] and errors == 0)
        verdicts[theta] = {"passed": bool(g1 and g2 and g3 and g4 and g5),
                           "G1_no_gained_collisions": bool(g1), "G2_c0_persistent_mission": bool(g2), "G3_angle_bound": g3,
                           "G4_ceiling_hover": g4, "G5_numerical": g5,
                           "filtered_collisions_by_level": {lv: int(t[t.level == lv].collision.sum()) for lv in LEVELS},
                           "executed_angle_mean_deg": float(t.mean_executed_angle_deg.mean()),
                           "overall_intervention_rate": float(t.interventions.sum() / max(t.proposals.sum(), 1))}
    passing = [t for t, v in verdicts.items() if v["passed"]]
    out.update(verdicts=verdicts, tables=tables, selected_theta_max_deg=max(passing) if passing else None)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    records = json.loads((STUDY_DIR / "records.json").read_text()) if args.summarize_only else run_study(args.workers)
    result = evaluate(records)
    (STUDY_DIR / "summary.json").write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({"selected": result["selected_theta_max_deg"], "verdicts": result["verdicts"], "reference": result["reference"]}, indent=1, default=float))
