"""Static and animated trajectory views for the V2 visual validation.

Read-only consumers of `journal_v2_validation_core.record_episode` records:
they never touch the environment, so they cannot change simulation behavior.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

from experiments.journal_v2_validation_core import deviation_segments  # noqa: E402

MODE_COLORS = {"DIRECT": "#1f77b4", "OBSTACLE_AVOID": "#ff7f0e", "ROUTE_BYPASS": "#9467bd", "EVADE": "#d62728"}
ATTACKER_COLOR, DEFENDER_COLOR, ASSET_COLOR = "#c0392b", "#27ae60", "#2c3e50"


def _arrays(rec: dict):
    rows = rec["rows"]
    att = np.array([r["attacker_pos"] for r in rows])
    ast = np.array([r["asset_pos"] for r in rows])
    dfn = np.array([r["defender_pos"] for r in rows])
    return att, ast, dfn


def _title(rec: dict) -> str:
    m, info = rec["meta"], rec["final_info"]
    return (f"{m['semantics']} | {m['level']} | {m['controller']} | seed {m['seed']}\n"
            f"{info['outcome']} at step {len(rec['rows']) - 1}"
            + (f" ({info['terminal_event']})" if info.get("terminal_event") else ""))


def legend_handles() -> list:
    return [Line2D([0], [0], color=c, lw=2.5, label=m) for m, c in MODE_COLORS.items()] + [
        Line2D([0], [0], color=DEFENDER_COLOR, lw=2, label="defender"),
        Line2D([0], [0], color="0.5", ls=":", label="asset trail"),
        Line2D([0], [0], marker="*", color="w", mfc="#00bcd4", mec="k", ms=10, label="detection"),
        Line2D([0], [0], marker="v", color="w", mfc=MODE_COLORS["OBSTACLE_AVOID"], mec="k", label="avoidance entry"),
        Line2D([0], [0], marker="s", color="w", mfc=MODE_COLORS["DIRECT"], mec="k", label="return to DIRECT"),
        Line2D([0], [0], marker="*", color="w", mfc="gold", mec="k", ms=11, label="terminal event"),
        Line2D([0], [0], color="red", ls="--", label="capture radius"),
        Line2D([0], [0], color="red", lw=3, alpha=0.7, label="swept contact / missed crossing"),
    ]


def add_figure_legend(fig) -> None:
    fig.legend(handles=legend_handles(), loc="lower center", ncol=6, fontsize=7.5, frameon=False)


# ----------------------------------------------------------------- 2-D
def draw_topdown(ax, rec: dict, upto: int | None = None, show_line: bool = False, legend: bool = False,
                 mark_misses: bool = True) -> None:
    rows, ex = rec["rows"], rec["extras"]
    n = len(rows) - 1
    k = n if upto is None else max(0, min(upto, n))
    cfg = rec["config"]
    L = cfg["L"]
    att, ast, dfn = _arrays(rec)
    modes = [r["behavior_mode"] for r in rows]

    ax.set_xlim(-L - 3, L + 3)
    ax.set_ylim(-L - 3, L + 3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.add_patch(Rectangle((-L, -L), 2 * L, 2 * L, fill=False, ls=":", ec="0.55"))

    blocking = ex[k]["line_blocked_static"]
    for i, o in enumerate(rec["static_obstacles"]):
        lo, hi = np.array(o["min"]), np.array(o["max"])
        ax.add_patch(Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], fc="0.45" if o["map_known"] else "0.75",
                               alpha=0.55, ec="red" if (show_line and blocking == i) else "0.2",
                               lw=2.2 if (show_line and blocking == i) else 0.8, zorder=2))
    for (x, y, _z, r, _id) in ex[k]["dynamic_obstacles"]:
        ax.add_patch(Circle((x, y), r, fc="#b8b800", alpha=0.35, ec="#7a7a00", zorder=2))

    ax.plot(ast[: k + 1, 0], ast[: k + 1, 1], color="0.5", ls=":", lw=1.2, zorder=3)
    ax.add_patch(Circle(ast[k, :2], cfg["threat_radius"], fill=False, ec="red", ls="--", lw=1.2, zorder=4))
    ax.plot(*ast[k, :2], marker="s", color=ASSET_COLOR, ms=7, zorder=6)
    ax.plot(*ast[0, :2], marker="s", mfc="none", color=ASSET_COLOR, ms=7, zorder=5)

    if k >= 1:
        segs = [[att[i - 1, :2], att[i, :2]] for i in range(1, k + 1)]
        ax.add_collection(LineCollection(segs, colors=[MODE_COLORS.get(modes[i], "k") for i in range(1, k + 1)],
                                         linewidths=2.2, zorder=5))
        step = max(1, n // 7)
        for i in range(step, k, step):
            ax.annotate("", xy=att[i, :2], xytext=att[i - 1, :2],
                        arrowprops=dict(arrowstyle="->", color=MODE_COLORS.get(modes[i], "k"), lw=1.4), zorder=6)
        ax.plot(dfn[: k + 1, 0], dfn[: k + 1, 1], color=DEFENDER_COLOR, lw=1.8, alpha=0.9, zorder=4)
    ax.plot(*att[0, :2], marker="^", mfc="none", color=ATTACKER_COLOR, ms=9, zorder=6)
    ax.plot(*att[k, :2], marker="^", color=ATTACKER_COLOR, ms=9, zorder=7)
    ax.plot(*dfn[0, :2], marker="D", mfc="none", color=DEFENDER_COLOR, ms=7, zorder=5)
    ax.plot(*dfn[k, :2], marker="D", color=DEFENDER_COLOR, ms=7, zorder=7)
    if rec["meta"]["controller"] != "passive":
        ax.add_patch(Circle(dfn[k, :2], cfg["intercept_radius"], fill=False, ec=DEFENDER_COLOR, ls="--", lw=1.0, zorder=4))

    if show_line:
        blocked = ex[k]["line_blocked_static"] is not None or ex[k]["line_blocked_dynamic"] is not None
        ax.plot([att[k, 0], ast[k, 0]], [att[k, 1], ast[k, 1]], color="red" if blocked else "0.35",
                ls="--", lw=1.0, zorder=3)

    det = next((i for i in range(0, k + 1) if rows[i]["detected"]), None)
    if det is not None:
        ax.plot(*att[det, :2], marker="*", color="#00bcd4", ms=13, mec="k", mew=0.6, zorder=8)
    for seg in deviation_segments(rec):
        if seg["activation_step"] <= k:
            ax.plot(*att[seg["activation_step"] - 1, :2], marker="v", color=MODE_COLORS["OBSTACLE_AVOID"], mec="k",
                    ms=9, zorder=8)
            r = seg["return_to_direct_step"]
            if r is not None and r <= k:
                ax.plot(*att[r - 1, :2], marker="s", color=MODE_COLORS["DIRECT"], mec="k", ms=8, zorder=8)
    if mark_misses:
        for miss in rec["missed_attacker"][:8]:
            if miss["step"] <= k:
                ax.plot([att[miss["step"] - 1, 0], att[miss["step"], 0]], [att[miss["step"] - 1, 1], att[miss["step"], 1]],
                        color="red", lw=3.2, alpha=0.7, zorder=7)
    if k == n and rec["final_info"].get("terminal_event"):
        event = rec["final_info"]["terminal_event"]
        frac = rec["final_info"].get("terminal_event_fraction")
        ax.plot(*att[n, :2], marker="*", color="gold", ms=17, mec="k", zorder=9)
        if frac is not None and event in ("capture", "interception"):
            ax.plot([att[n - 1, 0], att[n, 0]], [att[n - 1, 1], att[n, 1]], color="red", lw=3.2, alpha=0.8, zorder=7)
            ax.annotate(f"swept {event}\ntau={frac:.2f}", att[n, :2], xytext=(10, 12), textcoords="offset points",
                        fontsize=7, bbox=dict(fc="w", ec="0.5", alpha=0.85), zorder=10)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")


def plot_case(rec: dict, path: Path, extra: dict | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 7.8), dpi=130)
    draw_topdown(ax, rec, show_line=False)
    info = rec["final_info"]
    ax.set_title(_title(rec), fontsize=9)
    ax.text(0.01, 0.01, f"path {info['attacker_path_length']:.0f} m | heading change {info['attacker_heading_change_total_deg']:.0f} deg | "
            f"overshoots {info['attacker_overshoot_count']} | avoid segments {len(deviation_segments(rec))}",
            transform=ax.transAxes, fontsize=7, bbox=dict(fc="w", alpha=0.8, ec="0.6"))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    add_figure_legend(fig)
    fig.savefig(path)
    plt.close(fig)


def plot_pair(rec_a: dict, rec_b: dict, path: Path, heading: str) -> None:
    """Two top-down panels (e.g. passive attacker-only | with controller)."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 7.6), dpi=120)
    for ax, rec in zip(axes, (rec_a, rec_b)):
        draw_topdown(ax, rec, show_line=False)
        ax.set_title(_title(rec), fontsize=9)
    fig.suptitle(heading, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    add_figure_legend(fig)
    fig.savefig(path)
    plt.close(fig)


def _relative_panel(ax, rec: dict, title: str) -> None:
    """Asset-centered relative geometry of every attacker-vs-asset sphere
    crossing (missed in V1, or the terminal swept capture in V2)."""
    att, ast, _ = _arrays(rec)
    R = rec["config"]["threat_radius"]
    ax.add_patch(Circle((0, 0), R, fc="#ffd6d6", ec="red", ls="--", lw=1.4, zorder=1))
    ax.plot(0, 0, marker="s", color=ASSET_COLOR, ms=7, zorder=5)
    events = []
    for m in rec["missed_attacker"]:
        k = m["step"]
        events.append((att[k - 1] - ast[k - 1], att[k] - ast[k], "miss", f"endpoint {m['endpoint_distance']:.1f} m\nswept min {m['min_distance']:.1f} m"))
    info = rec["final_info"]
    if info.get("terminal_event") == "capture":
        n = len(rec["rows"]) - 1
        pre = rec["pre_terminal"]
        r0 = pre["att"] - pre["ast"]
        r_contact = att[n] - ast[n]
        events.append((r0, r_contact, "capture", f"tau={info['terminal_event_fraction']:.2f}\ncontact {np.linalg.norm(r_contact):.1f} m"))
    lim = 4.0
    for r0, r1, kind, label in events:
        color = "red" if kind == "miss" else "green"
        ax.annotate("", xy=r1[:2], xytext=r0[:2], arrowprops=dict(arrowstyle="->", color=color, lw=2), zorder=4)
        ax.plot(*r1[:2], marker="x" if kind == "miss" else "*", color=color, ms=8 if kind == "miss" else 14, zorder=6)
        ax.annotate(label, (r0 + r1)[:2] / 2, fontsize=6.5, xytext=(4, 6), textcoords="offset points",
                    bbox=dict(fc="w", alpha=0.7, ec="0.7"))
        lim = max(lim, float(np.max(np.abs(r0[:2]))) * 1.1, float(np.max(np.abs(r1[:2]))) * 1.1)
    ax.set_xlim(-min(lim, 12), min(lim, 12))
    ax.set_ylim(-min(lim, 12), min(lim, 12))
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("attacker - asset, x (m)")
    ax.set_ylabel("attacker - asset, y (m)")


def plot_v1_vs_v2(rec_v1: dict, rec_v2: dict, path: Path, heading: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 13.5), dpi=110)
    draw_topdown(axes[0, 0], rec_v1)
    axes[0, 0].set_title("V1 (endpoint contact, legacy A0)\n" + _title(rec_v1).split("\n", 1)[1], fontsize=9)
    draw_topdown(axes[0, 1], rec_v2)
    axes[0, 1].set_title("V2 (swept contact, direct A0)\n" + _title(rec_v2).split("\n", 1)[1], fontsize=9)
    _relative_panel(axes[1, 0], rec_v1, f"V1: {len(rec_v1['missed_attacker'])} sphere crossings NOT registered (asset frame)")
    _relative_panel(axes[1, 1], rec_v2, "V2: swept capture contact (asset frame)")
    fig.suptitle(heading, fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    add_figure_legend(fig)
    fig.savefig(path)
    plt.close(fig)


def plot_controller_overlay(recs: list[dict], path: Path, max_delta: float, heading: str) -> None:
    """Attacker trails of every controller on one seed (they must coincide)
    with each controller's own defender trail."""
    fig, ax = plt.subplots(figsize=(8, 8), dpi=120)
    base = recs[0]
    L = base["config"]["L"]
    ax.set_xlim(-L - 3, L + 3)
    ax.set_ylim(-L - 3, L + 3)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    for o in base["static_obstacles"]:
        lo, hi = np.array(o["min"]), np.array(o["max"])
        ax.add_patch(Rectangle(lo[:2], hi[0] - lo[0], hi[1] - lo[1], fc="0.55", alpha=0.5, ec="0.2"))
    palette = plt.get_cmap("tab10")
    for idx, rec in enumerate(recs):
        att, ast, dfn = _arrays(rec)
        c = palette(idx)
        ax.plot(att[:, 0], att[:, 1], color=ATTACKER_COLOR, lw=4 - 0.5 * idx, alpha=0.55, zorder=3)
        if rec["meta"]["controller"] != "passive":
            ax.plot(dfn[:, 0], dfn[:, 1], color=c, lw=1.6, label=f"{rec['meta']['controller']} defender "
                    f"({rec['final_info']['outcome']}, step {len(rec['rows']) - 1})", zorder=4)
        ax.plot(*att[-1, :2], marker="*", color=c, ms=11, mec="k", zorder=6)
    ast = _arrays(base)[1]
    ax.plot(*ast[0, :2], marker="s", color=ASSET_COLOR, ms=8)
    ax.add_patch(Circle(ast[0, :2], base["config"]["threat_radius"], fill=False, ec="red", ls="--"))
    ax.plot([], [], color=ATTACKER_COLOR, lw=3, alpha=0.55, label=f"A0 attacker (all controllers; max |delta| = {max_delta:.2g} m)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title(heading, fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------- 3-D
def _box_faces(lo, hi):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
    idx = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 3, 7, 4)]
    return [[v[i] for i in face] for face in idx]


def draw_3d(ax, rec: dict) -> None:
    rows = rec["rows"]
    n = len(rows) - 1
    cfg = rec["config"]
    L, H = cfg["L"], cfg["max_altitude"]
    att, ast, dfn = _arrays(rec)
    modes = [r["behavior_mode"] for r in rows]
    for o in rec["static_obstacles"]:
        ax.add_collection3d(Poly3DCollection(_box_faces(o["min"], o["max"]), alpha=0.25, facecolor="0.4", edgecolor="0.2"))
    i = 1
    while i <= n:
        j = i
        while j + 1 <= n and modes[j + 1] == modes[i]:
            j += 1
        ax.plot(att[i - 1: j + 1, 0], att[i - 1: j + 1, 1], att[i - 1: j + 1, 2], color=MODE_COLORS.get(modes[i], "k"), lw=2.4)
        i = j + 1
    ax.plot(att[:, 0], att[:, 1], np.zeros(len(att)), color=ATTACKER_COLOR, lw=0.8, alpha=0.35)  # ground shadow
    ax.plot(dfn[:, 0], dfn[:, 1], dfn[:, 2], color=DEFENDER_COLOR, lw=1.8)
    ax.plot(ast[:, 0], ast[:, 1], ast[:, 2], color="0.5", ls=":", lw=1.2)
    ax.scatter(*att[0], marker="^", facecolors="none", edgecolors=ATTACKER_COLOR, s=50)
    ax.scatter(*att[-1], marker="*", color="gold", edgecolors="k", s=160)
    ax.scatter(*dfn[0], marker="D", facecolors="none", edgecolors=DEFENDER_COLOR, s=40)
    ax.scatter(*dfn[-1], marker="D", color=DEFENDER_COLOR, s=40)
    ax.scatter(*ast[-1], marker="s", color=ASSET_COLOR, s=50)
    u, v = np.mgrid[0: 2 * np.pi: 24j, 0: np.pi: 12j]
    R = cfg["threat_radius"]
    ax.plot_wireframe(ast[-1, 0] + R * np.cos(u) * np.sin(v), ast[-1, 1] + R * np.sin(u) * np.sin(v),
                      ast[-1, 2] + R * np.cos(v), color="red", alpha=0.25, lw=0.4)
    ax.set_xlim(-L, L)
    ax.set_ylim(-L, L)
    ax.set_zlim(0, H)
    ax.set_box_aspect((1, 1, 0.5))
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")


def plot_3d(recs: list[dict], path: Path, heading: str, views=((22, -60),)) -> None:
    fig = plt.figure(figsize=(7.5 * len(recs), 6.5 * len(views)), dpi=110)
    for r, (elev, azim) in enumerate(views):
        for c, rec in enumerate(recs):
            ax = fig.add_subplot(len(views), len(recs), r * len(recs) + c + 1, projection="3d")
            draw_3d(ax, rec)
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(_title(rec), fontsize=8)
    fig.suptitle(heading, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    add_figure_legend(fig)
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------ animation
def save_gif(rec: dict, path: Path, fps: int = 8, max_frames: int = 110, size: float = 5.6) -> None:
    rows = rec["rows"]
    n = len(rows) - 1
    att, ast, dfn = _arrays(rec)
    stride = max(1, math.ceil(n / max_frames))
    frames = list(range(0, n + 1, stride))
    if frames[-1] != n:
        frames.append(n)
    frames += [n] * 8  # hold the terminal frame
    fig, ax = plt.subplots(figsize=(size, size + 0.7), dpi=70)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    add_figure_legend(fig)

    def update(f):
        ax.clear()
        draw_topdown(ax, rec, upto=f, show_line=True, mark_misses=True)
        mode = rows[f]["behavior_mode"]
        r = float(np.linalg.norm(att[f] - ast[f]))
        d = float(np.linalg.norm(att[f] - dfn[f]))
        ax.text(0.02, 0.98, f"step {f}   t = {rows[f]['time_s']:.1f} s\nattacker-asset {r:.1f} m\n"
                f"attacker-defender {d:.1f} m", transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="w", alpha=0.85, ec="0.5"))
        ax.text(0.98, 0.02, mode, transform=ax.transAxes, ha="right", fontsize=10, color="w", weight="bold",
                bbox=dict(fc=MODE_COLORS.get(mode, "k"), ec="k"))
        ax.set_title(_title(rec).split("\n")[0], fontsize=8)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    anim.save(str(path), writer=PillowWriter(fps=fps))
    plt.close(fig)


# ------------------------------------------------------- event semantics
def plot_swept_interception(rec: dict, path: Path) -> dict:
    """Pre-step, post-step (endpoint counterfactual) and swept-contact
    geometry of the terminal interception step."""
    cf, pre, info, cfg = rec["counterfactual"], rec["pre_terminal"], rec["final_info"], rec["config"]
    att, ast, dfn = _arrays(rec)
    R = cfg["intercept_radius"]
    tau = float(info["terminal_event_fraction"])
    a0, d0 = pre["att"], pre["defender_start"]
    a1_end, d1_end = np.array(cf["attacker_pos"]), np.array(cf["defender_pos"])
    a_c, d_c = att[-1], dfn[-1]
    r0, r1 = a0 - d0, a1_end - d1_end
    r_c = r0 + tau * (r1 - r0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), dpi=120)
    ax = axes[0]
    ax.plot([a0[0], a1_end[0]], [a0[1], a1_end[1]], color=ATTACKER_COLOR, ls=":", lw=1.3)
    ax.plot([d0[0], d1_end[0]], [d0[1], d1_end[1]], color=DEFENDER_COLOR, ls=":", lw=1.3)
    ax.plot([a0[0], a_c[0]], [a0[1], a_c[1]], color=ATTACKER_COLOR, lw=2.5)
    ax.plot([d0[0], d_c[0]], [d0[1], d_c[1]], color=DEFENDER_COLOR, lw=2.5)
    ax.plot(*a0[:2], marker="^", mfc="none", color=ATTACKER_COLOR, ms=11, label="attacker: pre-step")
    ax.plot(*d0[:2], marker="D", mfc="none", color=DEFENDER_COLOR, ms=9, label="defender: pre-step")
    ax.plot(*a1_end[:2], marker="x", color=ATTACKER_COLOR, ms=11, mew=2.5, label="attacker: endpoint-only post-step")
    ax.plot(*d1_end[:2], marker="x", color=DEFENDER_COLOR, ms=11, mew=2.5, label="defender: endpoint-only post-step")
    ax.plot(*a_c[:2], marker="*", color="gold", mec="k", ms=18, label=f"swept contact (tau={tau:.2f})")
    ax.add_patch(Circle(d_c[:2], R, fill=False, ec=DEFENDER_COLOR, ls="--", lw=1.5))
    ax.add_patch(Circle(d1_end[:2], R, fill=False, ec=DEFENDER_COLOR, ls=":", lw=1))
    pts = np.array([a0, a1_end, d0, d1_end])[:, :2]
    pad = 4.0
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, loc="best")
    ax.set_title("World frame, terminal step", fontsize=10)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax = axes[1]
    ax.add_patch(Circle((0, 0), R, fc="#d6f5d6", ec=DEFENDER_COLOR, ls="--", lw=1.6))
    ax.annotate("", xy=r1[:2], xytext=r0[:2], arrowprops=dict(arrowstyle="->", color="0.4", lw=1.6, ls=":"))
    ax.plot([r0[0], r_c[0]], [r0[1], r_c[1]], color="red", lw=3)
    ax.plot(*r0[:2], marker="o", color="k", ms=6)
    ax.plot(*r1[:2], marker="x", color="k", ms=10, mew=2.2)
    ax.plot(*r_c[:2], marker="*", color="gold", mec="k", ms=18)
    ax.plot(0, 0, marker="D", color=DEFENDER_COLOR, ms=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    span = max(abs(r0[:2]).max(), abs(r1[:2]).max()) * 1.15
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_title("Relative frame: r(tau) = attacker - defender", fontsize=10)
    ax.text(0.02, 0.02, f"|r0| = {np.linalg.norm(r0):.2f} m   |r(endpoint)| = {np.linalg.norm(r1):.2f} m (> {R} m)\n"
            f"|r(tau)| = {np.linalg.norm(r_c):.2f} m = interception radius\ncontact fraction tau = {tau:.3f}\n"
            f"endpoint-only result on this step: '{cf['outcome']}'", transform=ax.transAxes, fontsize=8,
            bbox=dict(fc="w", alpha=0.9, ec="0.5"))
    m = rec["meta"]
    fig.suptitle(f"Swept defender interception - {m['level']} {m['controller']} seed {m['seed']} (V2)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"tau": tau, "contact_distance": float(np.linalg.norm(r_c)), "endpoint_distance": float(np.linalg.norm(r1)),
            "endpoint_only_outcome": cf["outcome"], "contact_position": [float(x) for x in a_c]}


def plot_competing_events(cases: list[dict], path: Path) -> None:
    """Deterministic micro-scenarios: both an interception and an asset capture
    are geometrically possible inside one step."""
    fig, axes = plt.subplots(len(cases), 1, figsize=(11, 3.1 * len(cases)), dpi=120, sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cases):
        x0, x1 = c["x_start"], c["x_end"]
        ax.axhline(0, color="0.8", lw=0.8)
        ax.add_patch(Circle((0, 0), c["threat_radius"], fc="#ffd6d6", ec="red", ls="--", alpha=0.9))
        ax.add_patch(Circle((c["defender_x"], 0), c["intercept_radius"], fc="#d6f5d6", ec=DEFENDER_COLOR, ls="--", alpha=0.7))
        ax.plot([x0, x1], [0, 0], color=ATTACKER_COLOR, lw=2.4, ls=":", label="full-step attacker path")
        win_x = x0 + c["terminal_fraction"] * (x1 - x0)
        ax.plot([x0, win_x], [0, 0], color=ATTACKER_COLOR, lw=3.4)
        ax.plot(x0, 0, marker="^", mfc="none", color=ATTACKER_COLOR, ms=11)
        ax.plot(0, 0, marker="s", color=ASSET_COLOR, ms=8)
        ax.plot(c["defender_x"], 0, marker="D", color=DEFENDER_COLOR, ms=8)
        ax.plot(win_x, 0, marker="*", color="gold", mec="k", ms=18)
        ax.set_aspect("equal")
        ax.set_ylim(-4, 4)
        ax.grid(alpha=0.25)
        ax.set_title(f"{c['name']}: capture tau_c = {c['capture_fraction']:.3f}, interception tau_i = {c['interception_fraction']:.3f}"
                     f"  ->  winner: {c['terminal_event']} ({c['outcome']})", fontsize=9)
    np.atleast_1d(axes)[-1].set_xlabel("x (m): attacker moves left to right; asset at 0, defender as marked")
    fig.suptitle("Event ordering inside one step (V2): earliest contact wins", fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
