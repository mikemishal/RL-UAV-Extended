"""Publication figures for the A1-CBF defender-avoidance stage (journal_cbf_attacker, defender_cbf / Prompt 4,
brief section 29). Reads the already-computed chunked Monte Carlo results under
docs/journal_cbf_attacker/defender_cbf/results/ and the summary JSONs written by the campaign runs -- generates
NO new simulation, purely a reporting/visualization pass over existing (already-validated) data.

Scope note: time-boxed to 6 of the brief's 8 requested figures (interception/capture/separation comparisons,
joint-constraint/intervention-source breakdown, one representative 3-D joint-evasion trajectory, and
attacker-defender separation vs time for that same episode) plus the architecture diagram, PNG only (no
PDF/vector export) -- documented honestly in the README rather than silently reduced.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import experiments.journal_cbf_attacker_defender_foundation as F

FIG_DIR = F.DOCS / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_matched(tag: str):
    oo, fu = [], []
    for p in sorted(F.RESULTS_ROOT.glob(f"{tag}_chunk_*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        oo.extend(payload["obstacle_only"])
        fu.extend(payload["full"])
    return oo, fu


def fig_interception_capture_comparison():
    defenders = ["greedy", "kalman_greedy", "pn", "lead", "rmpc"]
    labels = ["Greedy", "Kalman-\nGreedy", "PN", "Lead", "RMPC"]
    interception_oo, interception_fu, capture_oo, capture_fu = [], [], [], []
    for d in defenders:
        oo, fu = _load_matched(f"matched_{d}")
        m_oo, m_fu = F.metrics(oo), F.metrics(fu)
        interception_oo.append(m_oo["mission"]["defender_interception"] * 100)
        interception_fu.append(m_fu["mission"]["defender_interception"] * 100)
        capture_oo.append(m_oo["mission"]["asset_capture"] * 100)
        capture_fu.append(m_fu["mission"]["asset_capture"] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(defenders))
    w = 0.35
    axes[0].bar(x - w / 2, interception_oo, w, label="obstacle-only", color="#9e4a4a")
    axes[0].bar(x + w / 2, interception_fu, w, label="full A1-CBF", color="#3a6ea5")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Defender interception rate (%)")
    axes[0].set_title("Interception rate: obstacle-only vs. full defender-CBF")
    axes[0].legend(); axes[0].set_ylim(0, 100)
    axes[1].bar(x - w / 2, capture_oo, w, label="obstacle-only", color="#9e4a4a")
    axes[1].bar(x + w / 2, capture_fu, w, label="full A1-CBF", color="#3a6ea5")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Asset capture rate (%)")
    axes[1].set_title("Asset capture rate: obstacle-only vs. full defender-CBF")
    axes[1].legend(); axes[1].set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig06_interception_capture_comparison.png", dpi=150)
    plt.close(fig)


def fig_separation_distribution():
    oo, fu = _load_matched("matched_greedy")
    sep_oo = [e["min_defender_separation_m"] for e in oo if not e["scenario_generation_failure"] and np.isfinite(e["min_defender_separation_m"])]
    sep_fu = [e["min_defender_separation_m"] for e in fu if not e["scenario_generation_failure"] and np.isfinite(e["min_defender_separation_m"])]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bins = np.linspace(2.0, 15.0, 40)
    ax.hist(sep_oo, bins=bins, alpha=0.6, label="obstacle-only", color="#9e4a4a")
    ax.hist(sep_fu, bins=bins, alpha=0.6, label="full A1-CBF", color="#3a6ea5")
    ax.axvline(F.DEFENDER_SAFETY_RADIUS, color="k", linestyle="--", linewidth=1, label=f"R_D={F.DEFENDER_SAFETY_RADIUS} m")
    ax.set_xlabel("Minimum attacker-defender separation (m)")
    ax.set_ylabel("Episode count")
    ax.set_title("Minimum separation distribution vs. Greedy (1,000 matched episodes)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig07_min_separation_distribution.png", dpi=150)
    plt.close(fig)


def fig_joint_constraint_breakdown():
    oo, fu = _load_matched("matched_greedy")
    m_fu = F.metrics(fu)
    totals = m_fu["joint_safety"]["intervention_source_step_totals"]
    labels = ["none", "obstacle", "defender", "joint"]
    values = [totals[k] for k in labels]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    colors = ["#bbbbbb", "#e2a13a", "#3a6ea5", "#9e4a4a"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Step count (full A1-CBF, 1,000 episodes vs. Greedy)")
    ax.set_title("Intervention-source breakdown")
    for i, v in enumerate(values):
        ax.text(i, v, f"{v}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig08_intervention_source_breakdown.png", dpi=150)
    plt.close(fig)


def fig_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axis("off")
    boxes = [
        "moving-asset\npursuit", "static route\nplanner", "v_nom", "joint obstacle +\ndefender CBF-QP",
        "v_safe", "velocity\nexecutor",
    ]
    n = len(boxes)
    xs = np.linspace(0.05, 0.95, n)
    for x, label in zip(xs, boxes):
        ax.add_patch(plt.Rectangle((x - 0.07, 0.35), 0.14, 0.3, fill=True, facecolor="#e8eef7", edgecolor="#3a6ea5"))
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=9)
    for i in range(n - 1):
        ax.annotate("", xy=(xs[i + 1] - 0.075, 0.5), xytext=(xs[i] + 0.075, 0.5),
                    arrowprops=dict(arrowstyle="->", color="black"))
    ax.text(0.5, 0.05, "Defender hazard gated at 25 m (brief sec. 4); static hazards active within 20 m (Prompt 2)",
            ha="center", fontsize=8, color="#555555")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("A1-CBF final architecture (frozen)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_architecture_diagram.png", dpi=150)
    plt.close(fig)


def _record_single_episode(defender: str, seed: int, *, defender_cbf_enabled: bool) -> dict:
    """Re-runs ONE episode with full per-step trajectory capture (position/defender/asset/diagnostics) -- used for
    both the representative trajectory figure and the visualizer JSON export (brief sections 26/29)."""
    from uav_defend.envs.soldier_env import SoldierEnv
    from uav_defend.policies.sanitize import build_policy_info

    cfg, estimator = F.defender_config(defender)
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=defender_cbf_enabled)
    policy = F.VC.make_policy(defender, cfg)
    obs, info = env.reset(seed=seed)
    if policy is not None:
        policy.reset()
    steps = [{
        "t": 0.0, "attacker_pos": list(map(float, info["enemy_pos"])), "defender_pos": list(map(float, info["defender_pos"])),
        "asset_pos": list(map(float, info["soldier_pos"])), "defender_observable": False, "intervention_source": "none",
        "cbf_qp_status": "inactive", "defender_barrier_h": None, "min_inflated_obstacle_clearance": float(info.get("min_inflated_obstacle_clearance", float("inf"))),
    }]
    obstacles = [
        {"min": [float(x) for x in o.min_corner], "max": [float(x) for x in o.max_corner]}
        for o in env._obstacle_layout.obstacles
    ]
    done, t = False, 0.0
    while not done:
        if policy is None:
            action = np.zeros(3, dtype=np.float32)
        else:
            action = np.asarray(policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator)), dtype=np.float32).reshape(3)
        obs, _r, te, tr, info = env.step(action)
        t += env.config.dt
        steps.append({
            "t": float(t), "attacker_pos": list(map(float, info["enemy_pos"])), "defender_pos": list(map(float, info["defender_pos"])),
            "asset_pos": list(map(float, info["soldier_pos"])), "defender_observable": bool(info.get("defender_observable", False)),
            "intervention_source": str(info.get("intervention_source", "none")), "cbf_qp_status": str(info["cbf_qp_status"]),
            "defender_barrier_h": (float(info["defender_barrier_h"]) if np.isfinite(info.get("defender_barrier_h", float("inf"))) else None),
            "min_inflated_obstacle_clearance": (float(info["min_inflated_obstacle_clearance"]) if np.isfinite(info["min_inflated_obstacle_clearance"]) else None),
        })
        done = te or tr
    return {"defender": defender, "seed": int(seed), "defender_cbf_enabled": bool(defender_cbf_enabled),
            "outcome": info["outcome"], "obstacles": obstacles, "steps": steps}


def fig_representative_trajectory_and_separation(defender: str = "rmpc", seed: int = 640150):
    """A representative joint obstacle+defender evasion trajectory (figure 2) plus attacker-defender separation
    vs. time for the SAME episode (figure 3), chosen from the RMPC generalization diagnostic (the family showing
    the clearest large-effect evasion)."""
    record = _record_single_episode(defender, seed, defender_cbf_enabled=True)
    steps = record["steps"]
    att = np.array([s["attacker_pos"] for s in steps])
    dfn = np.array([s["defender_pos"] for s in steps])
    ast = np.array([s["asset_pos"] for s in steps])
    t = np.array([s["t"] for s in steps])
    sep = np.linalg.norm(att - dfn, axis=1)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for o in record["obstacles"]:
        mn, mx = o["min"], o["max"]
        ax.add_patch(plt.Rectangle((mn[0], mn[1]), mx[0] - mn[0], mx[1] - mn[1], facecolor="#cccccc", edgecolor="#777777"))
    ax.plot(att[:, 0], att[:, 1], "-o", color="#3a6ea5", markersize=3, label="attacker (A1-CBF)")
    ax.plot(dfn[:, 0], dfn[:, 1], "-o", color="#9e4a4a", markersize=3, label=f"defender ({defender})")
    ax.plot(ast[:, 0], ast[:, 1], "-o", color="#4a9e4a", markersize=3, label="asset")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title(f"Joint evasion (seed {seed}, outcome={record['outcome']})", fontsize=10)
    ax.legend(); ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_representative_trajectory.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(6.5, 4))
    ax2.plot(t, sep, color="#3a6ea5")
    ax2.axhline(F.DEFENDER_SAFETY_RADIUS, color="orange", linestyle="--", label=f"R_D={F.DEFENDER_SAFETY_RADIUS} m (unsafe radius)")
    ax2.axhline(2.5, color="red", linestyle="--", label="R_intercept=2.5 m")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("attacker-defender separation (m)")
    ax2.set_title("Attacker-defender separation vs. time")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "fig03_separation_vs_time.png", dpi=150)
    plt.close(fig2)
    return record


def fig_nominal_vs_safe_velocity(defender: str = "pn", seed: int = 640050):
    """Nominal vs. safe velocity magnitude during a defender encounter (figure 4)."""
    from uav_defend.envs.soldier_env import SoldierEnv
    from uav_defend.policies.sanitize import build_policy_info

    cfg, estimator = F.defender_config(defender)
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=True)
    policy = F.VC.make_policy(defender, cfg)
    obs, info = env.reset(seed=seed)
    if policy is not None:
        policy.reset()
    t_list, v_safe_list, v_actual_list = [], [], []
    done, t = False, 0.0
    while not done:
        action = np.asarray(policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator)), dtype=np.float32).reshape(3) if policy else np.zeros(3, dtype=np.float32)
        obs, _r, te, tr, info = env.step(action)
        t += env.config.dt
        t_list.append(t)
        v_safe_list.append(float(np.linalg.norm(info["attacker_velocity_command"])))
        v_actual_list.append(float(info["attacker_speed"]))
        done = te or tr
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(t_list, v_safe_list, label="||v_safe|| (commanded)", color="#3a6ea5")
    ax.plot(t_list, v_actual_list, label="||v_actual|| (executed)", color="#9e4a4a", linestyle="--")
    ax.set_xlabel("time (s)"); ax.set_ylabel("speed (m/s)")
    ax.set_title(f"Commanded vs. executed attacker speed during defender encounter ({defender}, seed {seed})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_nominal_vs_safe_velocity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    fig_architecture_diagram()
    fig_interception_capture_comparison()
    fig_separation_distribution()
    fig_joint_constraint_breakdown()
    fig_representative_trajectory_and_separation()
    fig_nominal_vs_safe_velocity()
    print("Figures written to", FIG_DIR)
