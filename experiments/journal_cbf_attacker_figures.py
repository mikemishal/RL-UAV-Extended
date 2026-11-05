"""Figures for the A1-CBF foundation stage (`python -m experiments.journal_cbf_attacker_figures`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments import journal_cbf_attacker_foundation as F
from experiments import journal_cbf_attacker_state as TS

FIG_DIR = TS.STATE_DIR / "figures"


def rel_fig(p: Path) -> str:
    return str(p.resolve().relative_to(TS.PROJECT_ROOT.resolve())).replace("\\", "/")


def make_figures() -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = FIG_DIR / f"{name}.{ext}"
            fig.savefig(p, dpi=200)
            out.append(rel_fig(p))
        plt.close(fig)

    cbf_eps = F.run_foundation_validation(F.RESULTS_ROOT / "foundation", enemy_control_mode="cbf_velocity")
    a0_eps = F.run_foundation_validation(F.RESULTS_ROOT / "foundation", enemy_control_mode="legacy")
    m0, mcbf = F.metrics(a0_eps), F.metrics(cbf_eps)

    # fig1: speed vs distance
    pairs = [p for e in cbf_eps for p in e["speed_vs_distance"]]
    d = np.array([p[0] for p in pairs])
    s = np.array([p[1] for p in pairs])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    idx = np.random.default_rng(0).choice(len(d), size=min(6000, len(d)), replace=False)
    ax.scatter(d[idx], s[idx], s=2, alpha=0.15, color="#1f77b4", label="A1-CBF (sampled decisions)")
    rng = np.linspace(0, 30, 200)
    ax.plot(rng, np.minimum(12.0, 1.0 * rng), "k--", lw=1.5, label="nominal law v=min(12, kp*r)")
    ax.set_xlabel("asset distance [m]"); ax.set_ylabel("attacker speed [m/s]"); ax.set_xlim(0, 30); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "fig1_speed_vs_distance")

    # fig2: capture/exhaustion/first-pass/repeated-flyby comparison
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    labels = ["capture", "exhaustion", "first_pass_capture", "repeated_flyby_fraction"]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, [m0[k] for k in labels], width, label="canonical A0", color="#999999")
    ax.bar(x + width / 2, [mcbf[k] for k in labels], width, label="A1-CBF", color="#1f77b4")
    ax.set_xticks(x); ax.set_xticklabels(["capture", "exhaustion", "first-pass\ncapture", "repeated\nfly-by frac."])
    ax.set_ylabel("rate"); ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)
    save(fig, "fig2_a0_vs_cbf_rates")

    # fig3: representative trajectory (longest-path capture, i.e. the most visually informative moving-target case)
    def traced(env, seed):
        obs, info = env.reset(seed=seed)
        pos, asset = [np.asarray(info["enemy_pos"], dtype=np.float64)], [np.asarray(info["soldier_pos"], dtype=np.float64)]
        done = False
        while not done:
            obs, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
            pos.append(np.asarray(info["enemy_pos"], dtype=np.float64))
            asset.append(np.asarray(info["soldier_pos"], dtype=np.float64))
            done = te or tr
        return {"pos": np.array(pos), "asset": np.array(asset), "outcome": info["outcome"]}

    from uav_defend.envs.soldier_env import SoldierEnv

    captures = [e for e in cbf_eps if e["outcome"] == "soldier_caught"]
    pick_seed = max(captures, key=lambda e: e["path_length_m"])["scenario_seed"]
    env_cbf = SoldierEnv(F.foundation_config(), enemy_control_mode="cbf_velocity")
    env_a0 = SoldierEnv(F.foundation_config(), enemy_control_mode="legacy")
    tr_cbf = traced(env_cbf, pick_seed)
    tr_a0 = traced(env_a0, pick_seed)
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*tr_a0["pos"].T, "k--o", ms=3, label=f"canonical A0 ({tr_a0['outcome']})")
    ax.plot(*tr_cbf["pos"].T, "-o", ms=3, color="#1f77b4", label=f"A1-CBF ({tr_cbf['outcome']})")
    ax.plot(*tr_cbf["asset"].T, "-", color="blue", lw=2, label="moving asset (common realized path)")
    ax.scatter(*tr_cbf["asset"][0], color="blue", marker="*", s=90)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"scenario {pick_seed}: A0 vs A1-CBF on the same moving-asset trajectory")
    save(fig, "fig3_representative_trajectory")

    return out


if __name__ == "__main__":
    print(json.dumps(make_figures(), indent=1))
