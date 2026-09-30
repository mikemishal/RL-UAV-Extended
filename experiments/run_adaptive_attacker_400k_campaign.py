"""
Phase 17: full 3-seed x 400k Adaptive Attacker v1 training campaign
(journal extension). Mirrors experiments/run_clutter_400k_campaign.py's
structure but is a SEPARATE pipeline (Phase 4) for the attacker.

Usage:
    python -m experiments.run_adaptive_attacker_400k_campaign
    python -m experiments.run_adaptive_attacker_400k_campaign --seed 201
    python -m experiments.run_adaptive_attacker_400k_campaign --num-envs 8
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.adaptive_attacker_env import (
    ATTACKER_TRAINING_SEEDS,
    ATTACKER_VALIDATION_SEED_START,
    AdaptiveAttackerTrainingEnv,
    build_defender_pool,
)
from experiments.adaptive_attacker_training_runner import run_adaptive_attacker_training

PROJECT_ROOT = Path(__file__).parent.parent
OUT_ROOT = PROJECT_ROOT / "results" / "adaptive_attacker_400k"

CAMPAIGN_TOTAL_TIMESTEPS = 400_000
CAMPAIGN_EVAL_FREQ = 20_000
CAMPAIGN_EVAL_EPISODES = 20
CAMPAIGN_CHECKPOINT_FREQ = 50_000
DEFAULT_NUM_ENVS = 16  # measured value validated for the defender clutter campaign; re-verify below


def _pool_spec(pool) -> list[dict]:
    return [{"family": m.family, "seed": m.seed, "estimator": m.estimator} for m in pool]


def _env_fn(worker_seed: int) -> AdaptiveAttackerTrainingEnv:
    # Each SubprocVecEnv worker calls this ONCE at startup, in its OWN
    # process -- builds its own defender pool (loads its own copies of
    # the 9 checkpoints + Lead) rather than pickling shared state.
    return AdaptiveAttackerTrainingEnv(defender_pool=build_defender_pool())


def run_one(seed: int, num_envs: int) -> dict:
    output_dir = OUT_ROOT / f"seed_{seed}"
    pool_spec = _pool_spec(build_defender_pool())
    print(f"=== adaptive_attacker_v1 seed={seed} (num_envs={num_envs}, "
          f"total_timesteps={CAMPAIGN_TOTAL_TIMESTEPS}) ===", flush=True)
    t0 = time.time()
    manifest = run_adaptive_attacker_training(
        env_fn=_env_fn,
        seed=seed,
        output_dir=output_dir,
        total_timesteps=CAMPAIGN_TOTAL_TIMESTEPS,
        eval_freq=CAMPAIGN_EVAL_FREQ,
        eval_episodes=CAMPAIGN_EVAL_EPISODES,
        eval_seed=ATTACKER_VALIDATION_SEED_START,
        checkpoint_freq=CAMPAIGN_CHECKPOINT_FREQ,
        training_config_summary={"architecture": "adaptive_attacker_v1"},
        opponent_pool_spec=pool_spec,
        verbose=0,
        num_envs=num_envs,
    )
    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s, actual_total_timesteps={manifest['actual_total_timesteps']}, "
          f"final eval={manifest['eval_history'][-1] if manifest['eval_history'] else None}", flush=True)
    return manifest


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Run only this seed (default: all 3).")
    parser.add_argument("--num-envs", type=int, default=DEFAULT_NUM_ENVS)
    args = parser.parse_args()

    seeds = [args.seed] if args.seed else list(ATTACKER_TRAINING_SEEDS)
    results = {}
    campaign_t0 = time.time()
    for seed in seeds:
        manifest = run_one(seed, args.num_envs)
        results[seed] = {
            "wall_time_s": manifest["wall_time_s"],
            "actual_total_timesteps": manifest["actual_total_timesteps"],
            "num_envs": manifest["num_envs"],
            "final_eval": manifest["eval_history"][-1] if manifest["eval_history"] else None,
            "output_dir": str(OUT_ROOT / f"seed_{seed}"),
        }
        OUT_ROOT.mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "campaign_summary.json").write_text(json.dumps(results, indent=2, default=str))

    total_elapsed = time.time() - campaign_t0
    print(f"=== campaign complete in {total_elapsed:.1f}s ===", flush=True)
    results["_campaign_wall_time_s"] = total_elapsed
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "campaign_summary.json").write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    main()
