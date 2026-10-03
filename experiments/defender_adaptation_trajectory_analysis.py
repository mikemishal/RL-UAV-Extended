"""
Matched trajectory-pair analysis for the defender-adaptation held-out
evaluation (journal extension, 38-phase defender-adaptation-EVALUATION
sub-task, Phase 33). Selects representative scenarios from the COMPLETED
`episodes.csv` by predefined category (never "most visually impressive"),
then re-runs those EXACT (defender, attacker, level, eval_seed) scenarios
with full position-trajectory capture for plotting D0+ vs D1 side by side.
No new STATISTICAL episodes are added to episodes.csv/episodes.jsonl by
this module -- trajectory re-runs are deterministic replays of already-
counted episodes (same seed, same frozen checkpoints), used only for
visualization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.sector_features import SECTOR_OBSERVATION_V1, compute_sector_features
from uav_defend.policies.sanitize import build_policy_info

from experiments.journal_clutter_protocol import build_canonical_env_config
from experiments.adaptive_attacker_env import build_attacker_observation
from experiments.defender_adaptation_evaluation import (
    OUT_DIR, build_attacker_conditions, build_defender_instances, load_model_registry,
)
from experiments.defender_adaptation_statistics import load_episodes

TRAJ_OUT_DIR = OUT_DIR / "figures"


def _run_episode_with_trajectory(env, defender_instance, defender_policy, attacker_model, eval_seed):
    pending = {"action": np.zeros(3, dtype=np.float32)}
    env.enemy_tactical_override = (lambda: pending["action"]) if attacker_model is not None else None
    obs, info = env.reset(seed=eval_seed)
    defender_policy.reset()
    enemy_traj = [env._enemy_pos.copy()]
    defender_traj = [env._defender_pos.copy()]
    soldier_traj = [env._soldier_pos.copy()]
    intervention_flags = [bool(info.get("defender_avoidance_active", False))]
    steps = 0
    done = False
    while not done:
        if attacker_model is not None:
            attacker_obs = build_attacker_observation(env, env.config.enemy_defender_detection_range)
            action, _ = attacker_model.predict(attacker_obs, deterministic=True)
            pending["action"] = np.asarray(action, dtype=np.float32)
        defender_obs = env._get_obs()
        defender_info = build_policy_info(env._get_info(), defender_instance.estimator)
        sector = None
        if defender_instance.needs_sector_features:
            sector = compute_sector_features(
                defender_pos=env._defender_pos, defender_vel=env._defender_vel,
                perceived=env._defender_last_perceived, sensor_range=SECTOR_OBSERVATION_V1.sensor_range,
            )
        if defender_instance.is_lr_ppo:
            defender_action = defender_policy.act(defender_obs, defender_info, extra_features=sector)
        elif defender_instance.needs_sector_features:
            defender_action = defender_policy.act(np.concatenate([defender_obs, sector]), defender_info)
        else:
            defender_action = defender_policy.act(defender_obs, defender_info)
        obs, reward, terminated, truncated, info = env.step(np.asarray(defender_action, dtype=np.float32))
        steps += 1
        enemy_traj.append(env._enemy_pos.copy())
        defender_traj.append(env._defender_pos.copy())
        soldier_traj.append(env._soldier_pos.copy())
        intervention_flags.append(bool(info.get("defender_avoidance_active", False)))
        done = terminated or truncated
    return {
        "outcome": info["outcome"], "steps": steps,
        "enemy_trajectory": np.array(enemy_traj), "defender_trajectory": np.array(defender_traj),
        "soldier_trajectory": np.array(soldier_traj), "intervention_flags": np.array(intervention_flags),
    }


def _plot_matched_pair(case_name: str, title: str, d0_plus_result: dict, d1_result: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, result, label in zip(axes, (d0_plus_result, d1_result), ("D0+", "D1")):
        et, dt, st = result["enemy_trajectory"], result["defender_trajectory"], result["soldier_trajectory"]
        ax.plot(et[:, 0], et[:, 1], color="tab:red", label="attacker", linewidth=1.5)
        ax.scatter(et[0, 0], et[0, 1], color="tab:red", marker="o", s=40, zorder=5)
        ax.plot(dt[:, 0], dt[:, 1], color="tab:blue", label="defender", linewidth=1.5)
        ax.scatter(dt[0, 0], dt[0, 1], color="tab:blue", marker="o", s=40, zorder=5)
        ax.plot(st[:, 0], st[:, 1], color="tab:green", label="asset", linewidth=1.0, linestyle="--")
        if result["intervention_flags"].any():
            idx = np.where(result["intervention_flags"])[0]
            ax.scatter(dt[idx, 0], dt[idx, 1], color="orange", marker=".", s=15, alpha=0.6, label="safety intervention")
        marker = "*" if result["outcome"] == "intercepted" else "X"
        color = "tab:blue" if result["outcome"] == "intercepted" else "black"
        ax.scatter(dt[-1, 0] if result["outcome"] == "intercepted" else et[-1, 0],
                    dt[-1, 1] if result["outcome"] == "intercepted" else et[-1, 1],
                    marker=marker, s=150, color=color, zorder=6, label="terminal event")
        ax.set_title(f"{label}: outcome={result['outcome']}, steps={result['steps']}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.legend(fontsize=7)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.3)
    fig.suptitle(f"Fig 9 ({case_name}): {title}")
    fig.tight_layout()
    fig.savefig(out_dir / f"fig9_trajectory_{case_name}.png", dpi=150)
    plt.close(fig)


def _select_case_a(df: pd.DataFrame) -> dict | None:
    """D0+ fails against A1-unseen while D1 succeeds."""
    pivot = df[df.attacker_type == "A1-unseen-203"].pivot_table(
        index=["architecture", "defender_seed", "level", "eval_seed"],
        columns="generation", values="defender_safe_intercept",
    ).dropna()
    hits = pivot[(pivot["d0_plus"] == 0) & (pivot["d1"] == 1)]
    if hits.empty:
        return None
    arch, seed, level, eval_seed = hits.index[0]
    return {"architecture": arch, "defender_seed": int(seed), "level": level, "eval_seed": int(eval_seed)}


def _select_case_b(df: pd.DataFrame) -> dict | None:
    """Both succeed, D1 intercepts more efficiently (shorter episode)."""
    sub = df[(df.attacker_type == "A1-unseen-203") & (df.defender_safe_intercept == 1)]
    pivot = sub.pivot_table(index=["architecture", "defender_seed", "level", "eval_seed"],
                             columns="generation", values="episode_length").dropna()
    if "d0_plus" not in pivot.columns or "d1" not in pivot.columns:
        return None
    hits = pivot[pivot["d1"] < pivot["d0_plus"]]
    if hits.empty:
        return None
    hits = hits.assign(gap=hits["d0_plus"] - hits["d1"]).sort_values("gap", ascending=False)
    arch, seed, level, eval_seed = hits.index[0]
    return {"architecture": arch, "defender_seed": int(seed), "level": level, "eval_seed": int(eval_seed)}


def _select_case_c(df: pd.DataFrame) -> dict | None:
    """D1 still fails against A1-unseen."""
    sub = df[(df.attacker_type == "A1-unseen-203") & (df.generation == "d1") & (df.defender_safe_intercept == 0)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return {"architecture": row.architecture, "defender_seed": int(row.defender_seed), "level": row.level, "eval_seed": int(row.eval_seed)}


def _select_case_d(df: pd.DataFrame) -> dict | None:
    """D1 retains A0 performance (representative success vs scripted)."""
    sub = df[(df.attacker_type == "A0") & (df.generation == "d1") & (df.defender_safe_intercept == 1)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return {"architecture": row.architecture, "defender_seed": int(row.defender_seed), "level": row.level, "eval_seed": int(row.eval_seed)}


def _select_case_e(df: pd.DataFrame) -> dict | None:
    """C3 high-clutter adaptive case: D1 succeeds vs A1-unseen at C3."""
    sub = df[(df.attacker_type == "A1-unseen-203") & (df.generation == "d1") & (df.level == "C3") & (df.defender_safe_intercept == 1)]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return {"architecture": row.architecture, "defender_seed": int(row.defender_seed), "level": row.level, "eval_seed": int(row.eval_seed)}


CASE_SELECTORS = {
    "case_a_d0plus_fails_d1_succeeds": (_select_case_a, "D0+ fails vs A1-unseen-203 while D1 succeeds"),
    "case_b_both_succeed_d1_more_efficient": (_select_case_b, "Both succeed; D1 intercepts earlier/more efficiently"),
    "case_c_d1_still_fails": (_select_case_c, "D1 still fails against A1-unseen-203"),
    "case_d_d1_retains_a0": (_select_case_d, "D1 retains A0 (scripted) performance"),
    "case_e_c3_high_clutter_adaptive": (_select_case_e, "C3 high-clutter adaptive case (D1 vs A1-unseen-203)"),
}


def run_trajectory_analysis(df: pd.DataFrame | None = None) -> dict:
    df = df if df is not None else load_episodes()
    TRAJ_OUT_DIR.mkdir(parents=True, exist_ok=True)
    registry = load_model_registry()
    defenders_by_key = {(d.architecture, d.defender_seed, d.generation): d for d in build_defender_instances(registry)}
    attackers_by_name = {a.name: a for a in build_attacker_conditions(registry)}

    results = {}
    for case_name, (selector, title) in CASE_SELECTORS.items():
        scenario = selector(df)
        if scenario is None:
            results[case_name] = {"status": "no matching scenario found"}
            continue
        arch, seed, level, eval_seed = scenario["architecture"], scenario["defender_seed"], scenario["level"], scenario["eval_seed"]
        attacker_type = "A0" if case_name == "case_d_d1_retains_a0" else "A1-unseen-203"
        attacker = attackers_by_name[attacker_type]

        episode_results = {}
        for generation in ("d0_plus", "d1"):
            d = defenders_by_key[(arch, seed, generation)]
            cfg = build_canonical_env_config(level, d.estimator)
            policy = d.build(cfg)
            env = SoldierEnv(cfg)
            episode_results[generation] = _run_episode_with_trajectory(env, d, policy, attacker.model, eval_seed)

        _plot_matched_pair(case_name, f"{title} ({arch}, seed {seed}, {level}, seed={eval_seed})",
                            episode_results["d0_plus"], episode_results["d1"], TRAJ_OUT_DIR)
        results[case_name] = {
            "status": "plotted", "scenario": scenario, "attacker_type": attacker_type,
            "d0_plus_outcome": episode_results["d0_plus"]["outcome"],
            "d1_outcome": episode_results["d1"]["outcome"],
        }
    return results


if __name__ == "__main__":
    import json
    results = run_trajectory_analysis()
    print(json.dumps(results, indent=2, default=str))
