"""Phase 6 plots: representative trajectories, velocity autocorrelation, heading-change and speed
distributions, for the target-motion-only characterization."""
import dataclasses
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv

OUT = "results/asset_motion_model_study"
SETTINGS = [("legacy", "legacy_gaussian_random_walk", 1.0, "tab:gray"),
            ("tau=0.5s", "gauss_markov_velocity", 0.5, "tab:blue"),
            ("tau=1.0s", "gauss_markov_velocity", 1.0, "tab:green"),
            ("tau=2.0s", "gauss_markov_velocity", 2.0, "tab:orange"),
            ("tau=4.0s", "gauss_markov_velocity", 4.0, "tab:red")]
STEPS = 400
DT = EnvConfig().dt


def trajectory(model, tau, seed):
    cfg = dataclasses.replace(EnvConfig(), obstacles_enabled=False, asset_motion_model=model, asset_motion_tau_s=tau)
    env = SoldierEnv(cfg)
    env.reset(seed=seed)
    pos = [env._soldier_pos.copy()]
    for _ in range(STEPS):
        env.step(np.zeros(3, dtype=np.float32))
        pos.append(env._soldier_pos.copy())
    return np.array(pos)


def autocorr_curve(vx, max_lag=20):
    vx = vx - vx.mean()
    den = np.sum(vx * vx)
    return [float(np.sum(vx[:-lag] * vx[lag:]) / den) if lag > 0 else 1.0 for lag in range(max_lag + 1)]


fig, axes = plt.subplots(1, 5, figsize=(22, 4.4))
for ax, (label, model, tau, color) in zip(axes, SETTINGS):
    pos = trajectory(model, tau, seed=7)
    ax.plot(pos[:, 0], pos[:, 1], color=color, lw=1)
    ax.plot(pos[0, 0], pos[0, 1], "ko", ms=5)
    ax.set_title(label, fontsize=10)
    ax.set_aspect("equal"); ax.grid(alpha=.2)
fig.suptitle("Representative target (soldier) trajectory, 200 s, seed 7")
fig.tight_layout()
fig.savefig(f"{OUT}/fig_trajectories.png", dpi=110)

fig, ax = plt.subplots(figsize=(6, 4.5))
for label, model, tau, color in SETTINGS:
    pos = trajectory(model, tau, seed=3)
    vel = np.diff(pos, axis=0) / DT
    curve = autocorr_curve(vel[:, 0])
    ax.plot(range(len(curve)), curve, color=color, label=label)
ax.axhline(0, color="k", lw=0.5)
ax.set_xlabel("lag (steps, dt=0.5s)"); ax.set_ylabel("velocity autocorrelation (v_x)")
ax.legend(fontsize=8); ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_autocorrelation.png", dpi=110)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for label, model, tau, color in SETTINGS:
    all_heading, all_speed = [], []
    for seed in range(10):
        pos = trajectory(model, tau, seed=seed)
        vel = np.diff(pos, axis=0) / DT
        speed = np.linalg.norm(vel, axis=1)
        heading = np.arctan2(vel[:, 1], vel[:, 0])
        dheading = np.degrees(np.abs(((np.diff(heading) + np.pi) % (2 * np.pi)) - np.pi))
        all_heading.extend(dheading.tolist()); all_speed.extend(speed.tolist())
    axes[0].hist(all_heading, bins=40, histtype="step", color=color, label=label, density=True)
    axes[1].hist(all_speed, bins=40, histtype="step", color=color, label=label, density=True)
axes[0].set_xlabel("heading change per step (deg)"); axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
axes[1].set_xlabel("speed (m/s)"); axes[1].set_ylabel("density"); axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_heading_speed_distributions.png", dpi=110)
print("wrote", OUT, "fig_trajectories.png, fig_autocorrelation.png, fig_heading_speed_distributions.png")
