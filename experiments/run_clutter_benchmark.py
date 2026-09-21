"""
Run the journal clutter benchmark (Phase 8 smoke test / Phase 9 main
benchmark) for every controller instance x clutter level, using the
matched-seed contract from experiments/clutter_benchmark_protocol.py.

Usage:
    python -m experiments.run_clutter_benchmark --smoke
    python -m experiments.run_clutter_benchmark --episodes 500
    python -m experiments.run_clutter_benchmark --episodes 500 --levels C0 C1

Writes one CSV per (level, instance) under results/clutter_benchmark/<run>/
and a combined results/clutter_benchmark/<run>/all_episodes.csv.

No training. No modification of any controller/environment logic.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from uav_defend.envs import SoldierEnv

from experiments.clutter_benchmark_protocol import (
    CLUTTER_LEVEL_ORDER,
    CLUTTER_MAIN_EPISODES,
    PROJECT_ROOT,
    SMOKE_SEEDS,
    build_env_config,
    controller_instances,
    main_seeds,
)
from experiments.clutter_eval_utils import run_clutter_episode

RESULTS_ROOT = PROJECT_ROOT / "results" / "clutter_benchmark"


def run_benchmark(run_name: str, seeds: tuple[int, ...], levels: tuple[str, ...]) -> pd.DataFrame:
    out_dir = RESULTS_ROOT / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    instances = controller_instances()
    all_rows = []
    t_start = time.time()

    for level in levels:
        for inst in instances:
            config = build_env_config(level, inst.estimator)
            env = SoldierEnv(config=config)
            policy = inst.build(config)

            t0 = time.time()
            rows = []
            for seed in seeds:
                metrics = run_clutter_episode(env, policy, seed)
                metrics["level"] = level
                metrics["family"] = inst.family
                metrics["instance_id"] = inst.instance_id
                metrics["estimator"] = inst.estimator
                rows.append(metrics)
            elapsed = time.time() - t0

            df = pd.DataFrame(rows)
            df.to_csv(out_dir / f"{level}_{inst.instance_id}.csv", index=False)
            all_rows.extend(rows)

            n_feasible = int(df["feasible"].sum())
            n_success = int(df.loc[df["feasible"], "success"].sum()) if n_feasible else 0
            print(
                f"[{run_name}] {level} {inst.instance_id:>10s}: "
                f"{n_feasible}/{len(seeds)} feasible, {n_success} success, "
                f"{elapsed:.1f}s ({elapsed / len(seeds):.3f}s/ep)"
            )

    combined = pd.DataFrame(all_rows)
    combined.to_csv(out_dir / "all_episodes.csv", index=False)
    print(f"[{run_name}] total wall time: {time.time() - t_start:.1f}s")
    print(f"[{run_name}] wrote {out_dir / 'all_episodes.csv'}")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Journal clutter benchmark runner")
    parser.add_argument("--smoke", action="store_true", help="Run the Phase-8 smoke test (20 episodes/instance).")
    parser.add_argument("--episodes", type=int, default=CLUTTER_MAIN_EPISODES, help="Episodes per instance per level.")
    parser.add_argument("--levels", nargs="+", default=list(CLUTTER_LEVEL_ORDER), choices=list(CLUTTER_LEVEL_ORDER))
    parser.add_argument("--run-name", type=str, default=None)
    args = parser.parse_args()

    if args.smoke:
        run_name = args.run_name or "smoke"
        seeds = SMOKE_SEEDS
    else:
        run_name = args.run_name or f"main_{args.episodes}"
        seeds = main_seeds(args.episodes)

    run_benchmark(run_name, seeds, tuple(args.levels))


if __name__ == "__main__":
    main()
