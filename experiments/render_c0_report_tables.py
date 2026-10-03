"""
Phase 12-14: finalize C0 reproduction results into clean markdown tables
for the journal artifact / final report (journal "accepted-paper
reproduction" sub-task). Pure post-processing of
`experiments/defender_adaptation_c0_statistics.py`'s JSON/CSV outputs --
no new episodes run here.
"""

from __future__ import annotations

import json

import pandas as pd

from experiments.defender_adaptation_c0_evaluation import OUT_DIR

ARCH_LABELS = {"ppo_clutter_v1": "PPO-C", "ppo_kalman_clutter_v1": "PPO-Kalman-C", "lrppo_clutter_v1": "LR-PPO-C"}


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def render_retention_table(path) -> str:
    rows = json.loads(path.read_text())
    lines = ["| Architecture | Comparison | D0/D0+ rate | D1/comparison rate | Diff | 95% CI |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {ARCH_LABELS[r['architecture']]} | {r['generation_a']} vs {r['generation_b']} "
            f"| {_fmt(r['success_b_mean'])} | {_fmt(r['success_a_mean'])} "
            f"| {r['observed_diff']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |"
        )
    return "\n".join(lines)


def render_primary_endpoint_table(path) -> str:
    rows = json.loads(path.read_text())
    lines = ["| Architecture | D0+ rate (vs A1-unseen-203) | D1 rate | Diff (D1-D0+) | 95% CI |",
             "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {ARCH_LABELS[r['architecture']]} | {_fmt(r['success_b_mean'])} | {_fmt(r['success_a_mean'])} "
            f"| {r['observed_diff']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    out = OUT_DIR
    print("## C0 scripted (A0) retention: D0/D0+/D1\n")
    print(render_retention_table(out / "c0_scripted_retention.json"))
    print("\n## C0 primary endpoint (D1-D0+) vs A1-unseen-203\n")
    print(render_primary_endpoint_table(out / "c0_primary_endpoint_unseen.json"))
    print("\n## 4-condition decomposition\n")
    decomp = pd.read_csv(out / "four_condition_decomposition.csv")
    print(decomp.to_string(index=False))
    print("\n## LR-PPO residual-angle usage in C0\n")
    lrppo = pd.read_csv(out / "lrppo_residual_c0_summary.csv")
    print(lrppo.to_string(index=False))
