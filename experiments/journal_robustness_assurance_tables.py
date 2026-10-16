"""Phase 16 publication tables for the journal robustness/assurance
evaluation. Pure post-processing of journal_robustness_assurance_stats.py
outputs -- renders compact manuscript-ready markdown tables."""

from __future__ import annotations

import pandas as pd

from experiments.journal_robustness_assurance import OUT_DIR

TABLES_DIR = OUT_DIR / "tables"


def table1_safe_interception() -> str:
    df = pd.read_csv(OUT_DIR / "d1_minus_d0plus_robustness.csv")
    lines = ["## Robustness Table 1 -- Safe-interception rate (D0+ / D1)\n",
             "| Architecture | Level | Condition | D0+ | D1 |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        cond = r.setting_label if r.dimension == "nominal" else f"{r.dimension.split('_', 1)[1]}: {r.setting_label}"
        lines.append(f"| {r.architecture} | {r.level} | {cond} | {r.d0plus_mean:.3f} | {r.d1_mean:.3f} |")
    return "\n".join(lines)


def table2_d1_minus_d0plus() -> str:
    df = pd.read_csv(OUT_DIR / "d1_minus_d0plus_robustness.csv")
    lines = ["## Robustness Table 2 -- D1\u2212D0+ effects (pp, 95% CI)\n",
             "| Architecture | Level | Condition | Diff (pp) | 95% CI |", "|---|---|---|---|---|"]
    for _, r in df.iterrows():
        cond = r.setting_label if r.dimension == "nominal" else f"{r.dimension.split('_', 1)[1]}: {r.setting_label}"
        lines.append(f"| {r.architecture} | {r.level} | {cond} | {r["diff"]*100:+.1f} | [{r.ci_low*100:+.1f}, {r.ci_high*100:+.1f}] |")
    return "\n".join(lines)


def table3_failure_modes() -> str:
    df = pd.read_csv(OUT_DIR / "outcome_decomposition_challenging.csv")
    lines = ["## Robustness Table 3 -- Failure-mode decomposition (challenging conditions)\n",
             "| Condition | Architecture | Level | Gen | Safe-int. | Asset reach | Unsafe | Timeout | Collision |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        lines.append(f"| {r.dimension}:{r.setting_label} | {r.architecture} | {r.level} | {r.generation} | "
                      f"{r.safe_intercept:.3f} | {r.asset_reach:.3f} | {r.unsafe_intercept:.3f} | {r.timeout:.3f} | {r.defender_collision:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    t1, t2, t3 = table1_safe_interception(), table2_d1_minus_d0plus(), table3_failure_modes()
    (TABLES_DIR / "table1_safe_interception.md").write_text(t1, encoding="utf-8")
    (TABLES_DIR / "table2_d1_minus_d0plus.md").write_text(t2, encoding="utf-8")
    (TABLES_DIR / "table3_failure_modes.md").write_text(t3, encoding="utf-8")
    print(f"tables written to {TABLES_DIR}")
