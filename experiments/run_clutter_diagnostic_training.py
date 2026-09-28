"""
Phase 17 diagnostic training driver (journal extension): a REDUCED-budget
sanity-check training run for each of the three obstacle-aware
architectures (PPO, PPO-Kalman, LR-PPO), one seed each, run sequentially.

Budget note (measured, not estimated): a smoke test on this machine
measured ~38 env-steps/sec for this single-environment CPU setup. The full
Phase 18 budget (3 seeds x 400k steps x 3 architectures) is far beyond
what is feasible within one interactive session at this throughput; this
script intentionally uses a much smaller budget suitable for a diagnostic
health check (reward curve, collision rate, interception rate, safety
intervention rate, numerical stability), per Phase 17's own instruction to
"begin with a diagnostic budget rather than immediately launching the
final expensive run."

Usage:
    python -m experiments.run_clutter_diagnostic_training
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.clutter_training_runner import run_clutter_training
from experiments.clutter_training_env import build_ppo_clutter_env
from experiments.lr_ppo_clutter_training_env import build_lr_ppo_clutter_env
from experiments.journal_clutter_protocol import CLUTTER_TRAINING_SEEDS

PROJECT_ROOT = Path(__file__).parent.parent
OUT_ROOT = PROJECT_ROOT / "results" / "clutter_diagnostic_training"

DIAGNOSTIC_TOTAL_TIMESTEPS = 40_000
DIAGNOSTIC_EVAL_FREQ = 10_000
DIAGNOSTIC_EVAL_EPISODES = 15
DIAGNOSTIC_CHECKPOINT_FREQ = 20_000


def main() -> dict:
    results = {}

    seed_ppo = CLUTTER_TRAINING_SEEDS["ppo"][0]
    print(f"=== PPO clutter diagnostic (seed {seed_ppo}) ===", flush=True)
    results["ppo_clutter_v1"] = run_clutter_training(
        architecture="ppo_clutter_v1",
        env_fn=lambda: build_ppo_clutter_env("measurement"),
        seed=seed_ppo,
        output_dir=OUT_ROOT / "ppo_clutter_v1" / f"seed_{seed_ppo}",
        total_timesteps=DIAGNOSTIC_TOTAL_TIMESTEPS,
        eval_freq=DIAGNOSTIC_EVAL_FREQ,
        eval_episodes=DIAGNOSTIC_EVAL_EPISODES,
        checkpoint_freq=DIAGNOSTIC_CHECKPOINT_FREQ,
        training_config_summary={"architecture": "ppo_clutter_v1", "estimator": "measurement"},
        verbose=0,
    )

    seed_ppok = CLUTTER_TRAINING_SEEDS["ppo_kalman"][0]
    print(f"=== PPO-Kalman clutter diagnostic (seed {seed_ppok}) ===", flush=True)
    results["ppo_kalman_clutter_v1"] = run_clutter_training(
        architecture="ppo_kalman_clutter_v1",
        env_fn=lambda: build_ppo_clutter_env("kalman"),
        seed=seed_ppok,
        output_dir=OUT_ROOT / "ppo_kalman_clutter_v1" / f"seed_{seed_ppok}",
        total_timesteps=DIAGNOSTIC_TOTAL_TIMESTEPS,
        eval_freq=DIAGNOSTIC_EVAL_FREQ,
        eval_episodes=DIAGNOSTIC_EVAL_EPISODES,
        checkpoint_freq=DIAGNOSTIC_CHECKPOINT_FREQ,
        training_config_summary={"architecture": "ppo_kalman_clutter_v1", "estimator": "kalman"},
        verbose=0,
    )

    seed_lrppo = CLUTTER_TRAINING_SEEDS["lr_ppo"][0]
    print(f"=== LR-PPO clutter diagnostic (seed {seed_lrppo}) ===", flush=True)
    results["lrppo_clutter_v1"] = run_clutter_training(
        architecture="lrppo_clutter_v1",
        env_fn=lambda: build_lr_ppo_clutter_env(root_seed=seed_lrppo),
        seed=seed_lrppo,
        output_dir=OUT_ROOT / "lrppo_clutter_v1" / f"seed_{seed_lrppo}",
        total_timesteps=DIAGNOSTIC_TOTAL_TIMESTEPS,
        eval_freq=DIAGNOSTIC_EVAL_FREQ,
        eval_episodes=DIAGNOSTIC_EVAL_EPISODES,
        checkpoint_freq=DIAGNOSTIC_CHECKPOINT_FREQ,
        training_config_summary={"architecture": "lrppo_clutter_v1", "residual_max_angle_deg": 30.0},
        verbose=0,
    )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "diagnostic_summary.json").write_text(json.dumps(results, indent=2, default=str))
    for arch, manifest in results.items():
        print(arch, "wall_time_s=", manifest["wall_time_s"], "eval_history=", manifest["eval_history"])
    return results


if __name__ == "__main__":
    main()
