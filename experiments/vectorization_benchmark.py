"""
Phase 8 (scientific equivalence smoke run) and Phase 9 (throughput
benchmark) for the vectorized clutter-training infrastructure.

Phase 8: trains PPO-clutter at a small, matched transition budget via (A)
the num_envs=1 path and (B) a num_envs=4 SubprocVecEnv path, and reports
reward/success-rate ranges for both -- NOT expecting numerically
identical trajectories (different num_envs changes rollout composition,
so bitwise equivalence is not the goal), just confirming BOTH train
stably and land in comparable ranges.

Phase 9: measures REAL (not assumed) wall-clock env-steps/sec for
num_envs in {1, 2, 4, 8, 16} using SubprocVecEnv (num_envs=1 uses
DummyVecEnv, matching production behavior), and reports scaling
efficiency relative to the num_envs=1 baseline. This determines the
`num_envs` used for the full Phase 10 3x400k campaign.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from experiments.clutter_training_env import build_ppo_clutter_env
from experiments.clutter_training_runner import run_clutter_training

RESULTS_DIR = Path("results/vectorization_benchmark")

SMOKE_TOTAL_TIMESTEPS = 20_480  # 10 rollouts at n_steps=2048
# Chosen so that EVERY num_envs value in the sweep (1,2,4,8,16) completes an
# EXACT whole number of rollouts (>=2 rollouts even at num_envs=16) with NO
# overshoot -- this avoids the rollout-granularity ambiguity where SB3
# always finishes a full rollout (n_steps*num_envs transitions) before
# checking the stop condition, which would otherwise bias steps/sec low
# for small num_envs (more, smaller rollouts fit exactly) and require
# careful overshoot-correction for large num_envs (see actual_total_timesteps).
THROUGHPUT_TOTAL_TIMESTEPS = 65_536


def _env_fn(worker_seed: int):
    return build_ppo_clutter_env("measurement")


def run_equivalence_smoke_test() -> dict:
    print("=== Phase 8: equivalence smoke test (num_envs=1 vs num_envs=4) ===", flush=True)
    out = {}

    t0 = time.time()
    manifest_1 = run_clutter_training(
        architecture="ppo_clutter_v1_smoke_single", env_fn=_env_fn, seed=71,
        output_dir=RESULTS_DIR / "smoke_num_envs_1", total_timesteps=SMOKE_TOTAL_TIMESTEPS,
        eval_freq=SMOKE_TOTAL_TIMESTEPS // 2, eval_episodes=10, checkpoint_freq=SMOKE_TOTAL_TIMESTEPS,
        num_envs=1, verbose=0,
    )
    out["num_envs_1"] = {
        "wall_time_s": time.time() - t0,
        "eval_history": manifest_1["eval_history"],
        "worker_seeds": manifest_1["worker_seeds"],
    }
    print("num_envs=1:", out["num_envs_1"], flush=True)

    t0 = time.time()
    manifest_4 = run_clutter_training(
        architecture="ppo_clutter_v1_smoke_vec4", env_fn=_env_fn, seed=71,
        output_dir=RESULTS_DIR / "smoke_num_envs_4", total_timesteps=SMOKE_TOTAL_TIMESTEPS,
        eval_freq=SMOKE_TOTAL_TIMESTEPS // 2, eval_episodes=10, checkpoint_freq=SMOKE_TOTAL_TIMESTEPS,
        num_envs=4, vec_env_cls="subproc", verbose=0,
    )
    out["num_envs_4"] = {
        "wall_time_s": time.time() - t0,
        "eval_history": manifest_4["eval_history"],
        "worker_seeds": manifest_4["worker_seeds"],
    }
    print("num_envs=4:", out["num_envs_4"], flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "phase8_equivalence_smoke.json").write_text(json.dumps(out, indent=2, default=str))
    return out


def run_throughput_benchmark(num_envs_values=(1, 2, 4, 8, 16)) -> dict:
    print("=== Phase 9: throughput benchmark ===", flush=True)
    out = {}
    baseline_steps_per_sec = None
    for num_envs in num_envs_values:
        t0 = time.time()
        manifest = run_clutter_training(
            architecture=f"ppo_clutter_v1_throughput_{num_envs}", env_fn=_env_fn, seed=71,
            output_dir=RESULTS_DIR / f"throughput_num_envs_{num_envs}",
            total_timesteps=THROUGHPUT_TOTAL_TIMESTEPS,
            eval_freq=THROUGHPUT_TOTAL_TIMESTEPS, eval_episodes=2, checkpoint_freq=THROUGHPUT_TOTAL_TIMESTEPS,
            num_envs=num_envs, vec_env_cls=("dummy" if num_envs == 1 else "subproc"), verbose=0,
        )
        elapsed = time.time() - t0
        # IMPORTANT: SB3 always completes a full rollout (n_steps*num_envs
        # transitions) before checking the stop condition, so the actual
        # transitions collected can overshoot THROUGHPUT_TOTAL_TIMESTEPS
        # for large num_envs -- use the manifest's actual count, not the
        # nominal request, for a correct steps/sec measurement.
        actual_steps = manifest["actual_total_timesteps"]
        steps_per_sec = actual_steps / elapsed
        if baseline_steps_per_sec is None:
            baseline_steps_per_sec = steps_per_sec
        scaling_efficiency = (steps_per_sec / baseline_steps_per_sec) / num_envs
        out[num_envs] = {
            "wall_time_s": elapsed,
            "actual_total_timesteps": actual_steps,
            "env_steps_per_sec": steps_per_sec,
            "speedup_vs_num_envs_1": steps_per_sec / baseline_steps_per_sec,
            "scaling_efficiency": scaling_efficiency,
        }
        print(f"num_envs={num_envs}: {out[num_envs]}", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "phase9_throughput_benchmark.json").write_text(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    smoke = run_equivalence_smoke_test()
    throughput = run_throughput_benchmark()
    print(json.dumps({"phase8": smoke, "phase9": throughput}, indent=2, default=str))
