"""
Phase 10 (journal extension, scale-up sub-task): the FULL 3-seed x 400k
training campaign for all three obstacle-aware architectures (PPO,
PPO-Kalman, LR-PPO) -- 9 runs total.

Vectorization / throughput (Phases 3-4, 9): uses `num_envs=16` via
SubprocVecEnv, chosen from REAL measured throughput on this machine (NOT
assumed): 93.76 env-steps/sec at num_envs=1 vs 322.22 env-steps/sec at
num_envs=16 (3.44x speedup, see results/vectorization_benchmark/
phase9_throughput_benchmark.json). Deterministic per-worker seeding uses
`worker_seed = seed + worker_idx * SEED_STRIDE` (SEED_STRIDE=10_000,
see experiments/clutter_training_runner.py) so every run is fully
reproducible from its base `seed` alone.

Seed reuse note (Phase 10 explicit documentation requirement): this
campaign reuses the SAME base seeds as the prior diagnostic run
(experiments/run_clutter_diagnostic_training.py) for methodological
continuity -- ppo=(71,72,73), ppo_kalman=(74,75,76), lr_ppo=(77,78,79),
from experiments.journal_clutter_protocol.CLUTTER_TRAINING_SEEDS. The
FIRST seed per architecture (71/74/77) is therefore the SAME base seed
used for the 40k-step diagnostic checkpoint, but this campaign trains for
400k steps (10x) into a SEPARATE output directory
(results/clutter_training_400k/...), so it does NOT overwrite or conflict
with the diagnostic checkpoints under results/clutter_diagnostic_training/.
Additionally, because num_envs=16 draws 16 DIFFERENT worker seeds per run
(seed + i*SEED_STRIDE), the actual training data distribution seen by
each 400k run is far broader than the diagnostic run's single-worker data,
even when the base seed coincides.

Checkpoint policy (Phase 11):
  - `checkpoints/<arch>_seed<seed>_<n>_steps.zip` every `checkpoint_freq`
    environment steps (recovery / mid-training inspection).
  - `best_model.zip`: the checkpoint with the HIGHEST periodic-eval
    success rate seen during training (same-distribution, NOT held-out --
    see `experiments/clutter_retraining_evaluation.py` for the held-out
    Phase 15 evaluation used for the final reported numbers).
  - `final_model.zip`: the model state at the end of training.
  - Model-selection rule for all downstream evaluation/reporting: use
    `best_model.zip` (periodic-eval-best), falling back to `final_model.zip`
    only if `best_model.zip` is missing.

Usage:
    python -m experiments.run_clutter_400k_campaign
    python -m experiments.run_clutter_400k_campaign --arch ppo_clutter_v1
    python -m experiments.run_clutter_400k_campaign --arch ppo_clutter_v1 --seed 71
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from experiments.clutter_training_runner import run_clutter_training
from experiments.clutter_training_env import build_ppo_clutter_env
from experiments.lr_ppo_clutter_training_env import build_lr_ppo_clutter_env
from experiments.journal_clutter_protocol import CLUTTER_TRAINING_SEEDS

PROJECT_ROOT = Path(__file__).parent.parent
OUT_ROOT = PROJECT_ROOT / "results" / "clutter_training_400k"

CAMPAIGN_TOTAL_TIMESTEPS = 400_000
CAMPAIGN_EVAL_FREQ = 20_000
CAMPAIGN_EVAL_EPISODES = 20
CAMPAIGN_CHECKPOINT_FREQ = 50_000
CAMPAIGN_NUM_ENVS = 16  # see module docstring: measured, not assumed

ARCHITECTURES = {
    "ppo_clutter_v1": {
        "seeds": CLUTTER_TRAINING_SEEDS["ppo"],
        "env_fn": lambda worker_seed: build_ppo_clutter_env("measurement"),
        "training_config_summary": {"architecture": "ppo_clutter_v1", "estimator": "measurement"},
    },
    "ppo_kalman_clutter_v1": {
        "seeds": CLUTTER_TRAINING_SEEDS["ppo_kalman"],
        "env_fn": lambda worker_seed: build_ppo_clutter_env("kalman"),
        "training_config_summary": {"architecture": "ppo_kalman_clutter_v1", "estimator": "kalman"},
    },
    "lrppo_clutter_v1": {
        "seeds": CLUTTER_TRAINING_SEEDS["lr_ppo"],
        "env_fn": lambda worker_seed: build_lr_ppo_clutter_env(root_seed=worker_seed),
        "training_config_summary": {"architecture": "lrppo_clutter_v1", "residual_max_angle_deg": 30.0},
    },
}


def run_one(architecture: str, seed: int) -> dict:
    spec = ARCHITECTURES[architecture]
    output_dir = OUT_ROOT / architecture / f"seed_{seed}"
    print(f"=== {architecture} seed={seed} (num_envs={CAMPAIGN_NUM_ENVS}, "
          f"total_timesteps={CAMPAIGN_TOTAL_TIMESTEPS}) ===", flush=True)
    t0 = time.time()
    manifest = run_clutter_training(
        architecture=architecture,
        env_fn=spec["env_fn"],
        seed=seed,
        output_dir=output_dir,
        total_timesteps=CAMPAIGN_TOTAL_TIMESTEPS,
        eval_freq=CAMPAIGN_EVAL_FREQ,
        eval_episodes=CAMPAIGN_EVAL_EPISODES,
        checkpoint_freq=CAMPAIGN_CHECKPOINT_FREQ,
        training_config_summary=spec["training_config_summary"],
        verbose=0,
        num_envs=CAMPAIGN_NUM_ENVS,
    )
    elapsed = time.time() - t0
    print(f"    done in {elapsed:.1f}s, actual_total_timesteps={manifest['actual_total_timesteps']}, "
          f"final eval={manifest['eval_history'][-1] if manifest['eval_history'] else None}", flush=True)
    return manifest


def main() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=list(ARCHITECTURES), default=None,
                        help="Run only this architecture (default: all three).")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only this seed (default: all three seeds for the architecture).")
    args = parser.parse_args()

    architectures = [args.arch] if args.arch else list(ARCHITECTURES)
    results = {}
    campaign_t0 = time.time()
    for architecture in architectures:
        seeds = [args.seed] if args.seed else list(ARCHITECTURES[architecture]["seeds"])
        results[architecture] = {}
        for seed in seeds:
            manifest = run_one(architecture, seed)
            results[architecture][seed] = {
                "wall_time_s": manifest["wall_time_s"],
                "actual_total_timesteps": manifest["actual_total_timesteps"],
                "num_envs": manifest["num_envs"],
                "worker_seeds": manifest["worker_seeds"],
                "final_eval": manifest["eval_history"][-1] if manifest["eval_history"] else None,
                "output_dir": str(OUT_ROOT / architecture / f"seed_{seed}"),
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
