"""Publication figures for the A1-CBF static-obstacle-navigation acceptance campaign (journal_cbf_attacker,
obstacle_cbf stage). Run after experiments/journal_cbf_attacker_obstacle_run.py has produced cached results.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import journal_cbf_attacker_obstacle_foundation as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "docs" / "journal_cbf_attacker" / "obstacle_cbf" / "figures"


def rel_fig(p: Path) -> str:
    return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")


def main() -> list[str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out: list[str] = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = FIG_DIR / f"{name}.{ext}"
            fig.savefig(p, dpi=200)
            out.append(rel_fig(p))
        plt.close(fig)

    cbf_eps = F.run_obstacle_validation(F.RESULTS_ROOT / "obstacle", enemy_control_mode="cbf_velocity")
    a0_eps = F.run_obstacle_validation(F.RESULTS_ROOT / "obstacle", enemy_control_mode="legacy")
    m0, mcbf = F.metrics(a0_eps), F.metrics(cbf_eps)

    from uav_defend.envs.soldier_env import SoldierEnv

    def traced(env, seed):
        obs, info = env.reset(seed=seed)
        pos, asset = [np.asarray(info["enemy_pos"], dtype=np.float64)], [np.asarray(info["soldier_pos"], dtype=np.float64)]
        clearances, interventions = [], []
        done = False
        while not done:
            obs, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
            pos.append(np.asarray(info["enemy_pos"], dtype=np.float64))
            asset.append(np.asarray(info["soldier_pos"], dtype=np.float64))
            clearances.append(float(info["min_physical_obstacle_clearance"]))
            interventions.append((float(np.linalg.norm(info["attacker_velocity_command"])), float(info["cbf_intervention_norm"])))
            done = te or tr
        return {"pos": np.array(pos), "asset": np.array(asset), "outcome": info["outcome"], "clearances": clearances, "interventions": interventions}

    env_cbf = SoldierEnv(F.obstacle_config(), enemy_control_mode="cbf_velocity")

    # fig1: representative 3-D route through C2 (a successful, obstacle-engaging episode)
    engaging = [e for e in cbf_eps if e["outcome"] == "soldier_caught" and e["active_steps"] > 3 and not e["obstacle_collision"]]
    pick_seed = max(engaging, key=lambda e: e["active_steps"])["scenario_seed"] if engaging else cbf_eps[0]["scenario_seed"]
    tr = traced(env_cbf, pick_seed)
    obstacles = env_cbf._obstacle_layout.obstacles
    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*tr["pos"].T, "-o", ms=3, color="#1f77b4", label=f"A1-CBF route (seed {pick_seed}, {tr['outcome']})")
    ax.plot(*tr["asset"].T, "-", color="blue", lw=2, label="moving asset")
    for o in obstacles:
        c, h = o.center, o.half_extents
        ax.bar3d(c[0] - h[0], c[1] - h[1], c[2] - h[2], 2 * h[0], 2 * h[1], 2 * h[2], color="gray", alpha=0.25, shade=True)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]"); ax.legend(fontsize=7, loc="upper left")
    save(fig, "fig1_representative_route_c2")

    # fig2: nominal vs CBF-safe velocity near the obstacle, for the SAME representative episode
    speeds = [v[0] for v in tr["interventions"]]
    intervs = [v[1] for v in tr["interventions"]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.8, 5.2), sharex=True)
    ax1.plot(tr["clearances"], "k-", lw=1); ax1.axhline(0, color="r", ls="--", lw=1, label="raw obstacle surface")
    ax1.set_ylabel("physical clearance [m]"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2.plot(intervs, color="#d62728"); ax2.set_xlabel("step"); ax2.set_ylabel("CBF intervention norm [m/s]"); ax2.grid(alpha=0.3)
    save(fig, "fig2_nominal_vs_safe_velocity_near_obstacle")

    # fig3: minimum obstacle clearance distribution (physical, across all A1-CBF episodes)
    clearances_all = [e["min_physical_obstacle_clearance"] for e in cbf_eps if np.isfinite(e["min_physical_obstacle_clearance"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(clearances_all, bins=40, color="#1f77b4", alpha=0.8)
    ax.axvline(0, color="r", ls="--", lw=1.5, label="collision (clearance=0)")
    ax.set_xlabel("minimum physical obstacle clearance per episode [m]"); ax.set_ylabel("episodes"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    save(fig, "fig3_min_clearance_distribution")

    # fig4: CBF intervention distribution (per-episode mean intervention norm, episodes with >=1 active step only)
    engaged_eps = [e for e in cbf_eps if e["active_steps"] > 0]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist([e["mean_intervention_norm"] for e in engaged_eps], bins=40, color="#9467bd", alpha=0.8)
    ax.set_xlabel("mean CBF intervention norm per obstacle-engaging episode [m/s]"); ax.set_ylabel("episodes"); ax.grid(alpha=0.3, axis="y")
    save(fig, "fig4_cbf_intervention_distribution")

    # fig5: A0 vs A1-CBF mission outcomes
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = ["capture", "exhaustion", "timeout", "obstacle_collision_rate", "repeated_flyby_fraction"]
    x = np.arange(len(labels)); width = 0.35
    ax.bar(x - width / 2, [m0[k] for k in labels], width, label="canonical A0", color="#999999")
    ax.bar(x + width / 2, [mcbf[k] for k in labels], width, label="A1-CBF", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(["capture", "exhaustion", "timeout", "obstacle\ncollision", "repeated\nfly-by"])
    ax.set_ylabel("rate"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    save(fig, "fig5_a0_vs_cbf_outcomes")

    # fig6: path-excess distribution
    pe0 = [e["path_excess"] for e in a0_eps if np.isfinite(e["path_excess"])]
    pecbf = [e["path_excess"] for e in cbf_eps if np.isfinite(e["path_excess"])]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    bins = np.linspace(0.8, 2.0, 40)
    ax.hist(pe0, bins=bins, alpha=0.5, label="canonical A0", color="#999999")
    ax.hist(pecbf, bins=bins, alpha=0.5, label="A1-CBF", color="#1f77b4")
    ax.set_xlabel("path-excess (path length / chord)"); ax.set_ylabel("episodes"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    save(fig, "fig6_path_excess_distribution")

    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
