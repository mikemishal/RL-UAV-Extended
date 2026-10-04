"""Phase 16: render canonical C0 publication tables as markdown. Pure
post-processing of defender_adaptation_canonical_c0_statistics.py outputs."""

from __future__ import annotations

import json

import pandas as pd

from experiments.defender_adaptation_canonical_c0_evaluation import OUT_DIR

ARCH_LABELS = {"ppo_clutter_v1": "PPO-C", "ppo_kalman_clutter_v1": "PPO-Kalman-C", "lrppo_clutter_v1": "LR-PPO-C"}


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def table1_a0_retention(path) -> str:
    rows = json.loads(path.read_text())
    by_arch = {}
    for r in rows:
        by_arch.setdefault(r["architecture"], {})[(r["generation_a"], r["generation_b"])] = r
    lines = ["| Architecture | D0 | D0+ | D1 | D1-D0+ (95% CI) |", "|---|---|---|---|---|"]
    for arch, cells in by_arch.items():
        d0plus_vs_d0 = cells[("d0_plus", "d0")]
        d1_vs_d0plus = cells[("d1", "d0_plus")]
        d0 = d0plus_vs_d0["success_b_mean"]
        d0plus = d0plus_vs_d0["success_a_mean"]
        d1 = d1_vs_d0plus["success_a_mean"]
        lines.append(
            f"| {ARCH_LABELS[arch]} | {_fmt(d0)} | {_fmt(d0plus)} | {_fmt(d1)} | "
            f"{d1_vs_d0plus['observed_diff']:+.3f} [{d1_vs_d0plus['ci_low']:+.3f}, {d1_vs_d0plus['ci_high']:+.3f}] |"
        )
    return "\n".join(lines)


def table2_a1_unseen(path) -> str:
    rows = json.loads(path.read_text())
    lines = ["| Architecture | D0+ | D1 | D1-D0+ | 95% CI |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {ARCH_LABELS[r['architecture']]} | {_fmt(r['success_b_mean'])} | {_fmt(r['success_a_mean'])} "
            f"| {r['observed_diff']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |"
        )
    return "\n".join(lines)


def table3_a1_seen(path) -> str:
    rows = json.loads(path.read_text())
    lines = ["| Architecture | Attacker | D0+ | D1 | D1-D0+ | 95% CI |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {ARCH_LABELS[r['architecture']]} | {r['attacker_type']} | {_fmt(r['success_b_mean'])} | {_fmt(r['success_a_mean'])} "
            f"| {r['observed_diff']:+.3f} | [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    out = OUT_DIR
    print("## Table 1 -- Canonical C0 A0 retention\n")
    print(table1_a0_retention(out / "canonical_a0_retention.json"))
    print("\n## Table 2 -- Canonical C0 A1-unseen-203\n")
    print(table2_a1_unseen(out / "canonical_primary_endpoint_unseen.json"))
    print("\n## Table 3 -- Canonical C0 A1-seen (201/202 + mean)\n")
    print(table3_a1_seen(out / "canonical_seen_attacker_analysis.json"))
    print("\n## Table 4 -- C0 vs clutter\n")
    t4 = pd.read_csv(out / "c0_vs_clutter_table.csv")
    print(t4.to_string(index=False))
    print("\n## Table 5 -- Canonical C0 LR-PPO residual behavior\n")
    t5 = pd.read_csv(out / "lrppo_residual_canonical_summary.csv")
    print(t5.to_string(index=False))
