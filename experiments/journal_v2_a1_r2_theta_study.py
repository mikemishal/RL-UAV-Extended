"""theta_max selection study for `a1_v2_action_v2` (NO PPO training).

Stress commands drive the residual action contract for theta in {15, 30, 45, 60} degrees on seeds 145000-145099
at C0-C3, against a parked passive defender (so that only attacker, obstacle layer and dynamics matter).
The selection rule is frozen in `select_theta` BEFORE the study is run:

  choose the LARGEST theta such that ALL of
    1. obstacle-collision rate <= 2% at every clutter level (pooled over the six stress commands; worst case reported);
    2. net mission progress > 0 at C0 for every stress command;
    3. no reverse-direction command: the tactical command never deviates from base guidance by more than theta
       (+0.01 deg) and theta < 90 deg. (The executed direction may deviate further where the frozen avoidance layer
       bypasses an obstacle, exactly as it does for canonical A0; it is reported, not gated.)
    4. no ceiling-hover exploit: pooled fraction of ceiling-hover episodes <= 1% for every theta-level pair;
    5. numerical stability: no non-finite state, no exception.
If none passes, training must not start.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_a1 as A
from experiments.journal_v2_validation_core import PARKED_DEFENDER, build_config
from uav_defend.envs.soldier_env import SoldierEnv

STUDY_DIR = R.RESULTS_ROOT / "theta_study"
SEEDS = tuple(range(145_000, 145_100))
LEVELS = A.A1_V2_CLUTTER_LEVELS
COMMANDS = ("random", "left", "right", "up", "down", "alternating")
Z = np.array([0.0, 0.0, 1.0])
COLLISION_OUTCOMES = ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision")
MAX_COLLISION_RATE = 0.02
CEILING_HOVER_MAX = 0.01


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def stress_action(kind: str, u_b: np.ndarray, step: int, rng: np.random.Generator) -> np.ndarray:
    """Raw action whose perpendicular residual is a UNIT vector (maximum authority) for the persistent commands."""
    if kind == "random":
        return rng.uniform(-1.0, 1.0, 3)
    left = _unit(np.cross(Z, u_b))
    if left is None:
        left = np.array([-1.0, 0.0, 0.0])
    up = _unit(Z - float(np.dot(Z, u_b)) * u_b)
    if kind == "left":
        return left
    if kind == "right":
        return -left
    if kind == "up":
        return up if up is not None else np.zeros(3)
    if kind == "down":
        return -up if up is not None else np.zeros(3)
    if kind == "alternating":
        return left if step % 2 == 0 else -left
    raise ValueError(kind)


def run_episode(theta: float, kind: str, level: str, seed: int) -> dict:
    cfg, _ = build_config(level, "V2", "passive")
    env = SoldierEnv(cfg)
    state = {"step": 0, "angle_max": 0.0, "exec_angle_max": 0.0, "angle_sum": 0.0}
    rng = np.random.default_rng(seed + 17 * (COMMANDS.index(kind) + 1))

    def override():
        u_b = R.base_direction(env)
        if u_b is None:
            return np.zeros(3)
        command, angle = R.residual_command(stress_action(kind, u_b, state["step"], rng), u_b, theta)
        state["angle_max"] = max(state["angle_max"], angle)
        state["angle_sum"] += angle
        state["u_b"] = u_b
        return command

    env.enemy_tactical_override = override
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
        final = np.asarray(info["enemy_final_command"], dtype=np.float64)
        if "u_b" in state and np.linalg.norm(final) > 1e-9:
            cosine = float(np.clip(np.dot(final / np.linalg.norm(final), state["u_b"]), -1.0, 1.0))
            state["exec_angle_max"] = max(state["exec_angle_max"], float(np.degrees(np.arccos(cosine))))
        done = bool(terminated or truncated)
    chord = float(np.linalg.norm(prev - start))
    path = float(info["enemy_endurance_traveled_m"])
    hover = R.hover_metrics(np.array(speeds), np.array(alts), float(cfg.max_altitude))
    outcome = info["outcome"]
    return {"theta": theta, "command": kind, "level": level, "seed": int(seed), "outcome": outcome, "steps": state["step"],
            "collision": outcome in COLLISION_OUTCOMES, "exhausted": outcome == "enemy_endurance_exhausted",
            "captured": outcome == "soldier_caught", "path_excess": path / chord if chord > 1e-9 else float("nan"),
            "progress_net": 1.0 - float(info["enemy_soldier_dist"]) / d0, "max_residual_angle_deg": state["angle_max"],
            "mean_residual_angle_deg": state["angle_sum"] / max(state["step"], 1), "max_executed_vs_base_deg": state["exec_angle_max"],
            "mean_speed_mps": float(np.mean(speeds)), "finite": bool(finite), **hover}


def run_task(task):
    theta, kind, level = task
    t0 = time.time()
    rows, error = [], None
    for seed in SEEDS:
        try:
            rows.append(run_episode(theta, kind, level, seed))
        except Exception as exc:  # numerical/stability failures are results, not crashes
            error = repr(exc)
            rows.append({"theta": theta, "command": kind, "level": level, "seed": int(seed), "error": error, "finite": False})
    return rows, time.time() - t0


def run_study(workers: int = 18, thetas=R.THETA_CANDIDATES_DEG, name: str = "records.json") -> list[dict]:
    STUDY_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(th, k, lv) for th in thetas for k in COMMANDS for lv in LEVELS]
    records = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for i, (rows, seconds) in enumerate(pool.imap_unordered(run_task, tasks), 1):
            records.extend(rows)
            print(i, len(tasks), rows[0]["theta"], rows[0]["command"], rows[0]["level"], round(seconds, 1), flush=True)
    (STUDY_DIR / name).write_text(json.dumps(records, default=float))
    return records


def summarize(records: list[dict]) -> dict:
    import pandas as pd

    df = pd.DataFrame(records)
    ok = df[df.get("error").isna()] if "error" in df else df
    out = {"episodes": len(df), "errors": int(len(df) - len(ok))}
    cols = {"collision_rate": ("collision", "mean"), "exhaustion_rate": ("exhausted", "mean"), "capture_rate": ("captured", "mean"),
            "mean_path_excess": ("path_excess", "mean"), "mean_progress_net": ("progress_net", "mean"),
            "ceiling_occupancy": ("ceiling_occupancy", "mean"), "near_hover_fraction": ("near_hover_fraction", "mean"),
            "sustained_hover_rate": ("sustained_hover", "mean"), "ceiling_hover_rate": ("ceiling_hover", "mean"),
            "max_residual_angle_deg": ("max_residual_angle_deg", "max"), "max_executed_vs_base_deg": ("max_executed_vs_base_deg", "max"),
            "mean_speed_mps": ("mean_speed_mps", "mean"), "n": ("seed", "size")}
    out["by_theta_level"] = ok.groupby(["theta", "level"]).agg(**cols).reset_index().to_dict("records")
    out["by_theta_command_level"] = ok.groupby(["theta", "command", "level"]).agg(**cols).reset_index().to_dict("records")
    out["finite_all"] = bool(df["finite"].all()) if "finite" in df else False
    return out


def select_theta(summary: dict, thetas=R.THETA_CANDIDATES_DEG) -> dict:
    import pandas as pd

    tl = pd.DataFrame(summary["by_theta_level"])
    tcl = pd.DataFrame(summary["by_theta_command_level"])
    verdicts = {}
    for theta in thetas:
        a, b = tl[tl.theta == theta], tcl[tcl.theta == theta]
        c0 = b[b.level == "C0"]
        checks = {
            "collision_le_2pct_every_level_pooled": bool((a.collision_rate <= MAX_COLLISION_RATE).all()),
            "positive_c0_progress_every_command": bool((c0.mean_progress_net > 0).all()),
            "no_reverse_command": bool((a.max_residual_angle_deg <= theta + 0.01).all() and theta < 90.0),
            "no_ceiling_hover_exploit": bool((a.ceiling_hover_rate <= CEILING_HOVER_MAX).all()),
            "numerically_stable": bool(summary["finite_all"] and summary["errors"] == 0),
        }
        verdicts[theta] = {"passed": all(checks.values()), "checks": checks,
                           "worst_case_collision_rate": float(b.collision_rate.max()),
                           "pooled_collision_by_level": {r.level: float(r.collision_rate) for r in a.itertuples()}}
    passing = [t for t, v in verdicts.items() if v["passed"]]
    return {"selected_theta_max_deg": max(passing) if passing else None, "verdicts": verdicts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--supplementary", default=None, help="comma-separated extra thetas (INELIGIBLE for selection)")
    args = parser.parse_args()
    if args.supplementary:
        thetas = tuple(float(x) for x in args.supplementary.split(","))
        records = run_study(args.workers, thetas, "records_supplementary.json")
        summary = summarize(records)
        (STUDY_DIR / "summary_supplementary.json").write_text(json.dumps(
            {"note": "SUPPLEMENTARY, post-hoc, ineligible for theta selection", "summary": summary,
             "checks": select_theta(summary, thetas)}, indent=1, default=float))
        raise SystemExit(0)
    if args.summarize_only:
        records = json.loads((STUDY_DIR / "records.json").read_text())
    else:
        records = run_study(args.workers)
    summary = summarize(records)
    selection = select_theta(summary)
    (STUDY_DIR / "summary.json").write_text(json.dumps({"summary": summary, "selection": selection}, indent=1, default=float))
    print(json.dumps(selection, indent=1, default=float))
