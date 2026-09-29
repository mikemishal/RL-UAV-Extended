"""
Phase 19/21 (journal extension, scale-up sub-task): full-scale LR-PPO
residual-saturation analysis, re-run against the Phase 10 400k campaign
checkpoints (all 3 seeds: 77, 78, 79), NOT the 40k diagnostic checkpoint.

Answers the "do NOT change the 30-degree residual bound absent genuine
evidence of restriction" question with FULL-SCALE data: if
P(near_theta_max) is small (few percent) across all seeds, pooled and
individually, that is evidence the bound is NOT restrictive at full
training scale either, and should remain unchanged. Read-only diagnostic
-- never modifies theta_max.

Usage:
    python -m experiments.run_lrppo_saturation_400k
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments.lrppo_residual_saturation_analysis import run_analysis, summarize

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "lrppo_residual_saturation_400k"
CAMPAIGN_ROOT = PROJECT_ROOT / "results" / "clutter_training_400k" / "lrppo_clutter_v1"
SEEDS = (77, 78, 79)
N_EPISODES = 200


def main() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed_summaries = {}
    all_rows = []
    for seed in SEEDS:
        run_dir = CAMPAIGN_ROOT / f"seed_{seed}"
        path = run_dir / "best_model.zip"
        if not path.exists():
            path = run_dir / "final_model.zip"
        df = run_analysis(str(path), n_episodes=N_EPISODES)
        df["training_seed"] = seed
        all_rows.append(df)
        s = summarize(df)
        per_seed_summaries[seed] = s
        print(f"seed={seed}: {s}", flush=True)

    pooled_df = pd.concat(all_rows, ignore_index=True)
    pooled_df.to_csv(OUT_DIR / "residual_saturation_episodes.csv", index=False)
    pooled_summary = summarize(pooled_df)
    print(f"pooled (n={len(SEEDS)} seeds x {N_EPISODES} episodes): {pooled_summary}", flush=True)

    out = {"per_seed": per_seed_summaries, "pooled": pooled_summary, "n_episodes_per_seed": N_EPISODES,
           "seeds": list(SEEDS)}
    (OUT_DIR / "residual_saturation_summary.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
