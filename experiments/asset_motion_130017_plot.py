"""Phase 10: overlaid top-down attacker trajectories for C3 seed 130017 across asset-motion settings."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path("results/asset_motion_model_study")
LABELS = [("legacy", "tab:gray"), ("tau0.5", "tab:blue"), ("tau1.0", "tab:green"),
          ("tau2.0", "tab:orange"), ("tau4.0", "tab:red")]

fig, axes = plt.subplots(1, 5, figsize=(22, 4.6))
for ax, (label, color) in zip(axes, LABELS):
    rec = json.loads((R / label / "C3_130017.json").read_text())
    pos = np.array([r["attacker_pos"] for r in rec["rows"]])
    asset = np.array([r["asset_pos"] for r in rec["rows"]])
    for o in rec["static_obstacles"]:
        lo, hi = o["min"], o["max"]
        ax.add_patch(plt.Rectangle((lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1], color="gray", alpha=0.3))
    ax.plot(pos[:, 0], pos[:, 1], color=color, lw=1)
    ax.plot(asset[:, 0], asset[:, 1], ":", color="g", lw=0.8)
    ax.plot(pos[0, 0], pos[0, 1], "ko", ms=6)
    ax.plot(asset[-1, 0], asset[-1, 1], "g*", ms=14)
    outcome = rec["summary"]["outcome"]
    ax.set_title(f"{label}\n{len(rec['rows'])} steps, {outcome}", fontsize=9)
    ax.set_xlim(-52, 52); ax.set_ylim(-52, 52); ax.set_aspect("equal"); ax.grid(alpha=.2)
fig.suptitle("C3 seed 130017: attacker (solid) / asset (dotted) across asset-motion settings")
fig.tight_layout()
fig.savefig(R / "fig_130017_overlay.png", dpi=110)
print("wrote", R / "fig_130017_overlay.png")
