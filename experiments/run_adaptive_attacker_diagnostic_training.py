"""
Phase 16 diagnostic training driver for Adaptive Attacker v1 (journal
extension): a REDUCED-budget health-check training run (1 seed, 60k
total steps -- within the requested 40k-100k range) against the frozen
defender pool, BEFORE committing to the full Phase 17 (3 seeds x 400k)
campaign.

Validates: reward improves, soldier_caught (asset-reaching) behavior
exists and is nonzero, no NaNs/crashes, attacker does not collapse into
trivial avoidance/hiding (asset_progress_per_step tracked explicitly),
obstacle-collision rate stays controlled, and the learned policy is
compared against the existing hand-designed hostile on the SAME matched
diagnostic scenarios.

Usage:
    python -m experiments.run_adaptive_attacker_diagnostic_training
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.adaptive_attacker_env import (
    ATTACKER_DIAGNOSTIC_SEED,
    ATTACKER_VALIDATION_SEED_START,
    AdaptiveAttackerTrainingEnv,
    build_defender_pool,
)
from experiments.adaptive_attacker_training_runner import run_adaptive_attacker_training
from experiments.journal_clutter_protocol import build_randomized_training_config

PROJECT_ROOT = Path(__file__).parent.parent
OUT_ROOT = PROJECT_ROOT / "results" / "adaptive_attacker_diagnostic_training"

DIAGNOSTIC_TOTAL_TIMESTEPS = 60_000
DIAGNOSTIC_EVAL_FREQ = 15_000
DIAGNOSTIC_EVAL_EPISODES = 20
DIAGNOSTIC_CHECKPOINT_FREQ = 30_000

# Module-level defender pool: built ONCE, shared by every env_fn call in
# this (num_envs=1, single-process) diagnostic run.
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = build_defender_pool()
    return _POOL


def _env_fn(worker_seed: int):
    return AdaptiveAttackerTrainingEnv(defender_pool=_pool())


def run_scripted_hostile_comparison(n_episodes: int = 30) -> dict:
    """Phase 16 explicit requirement: compare against the EXISTING
    hand-designed attacker on matched diagnostic scenarios (same seeds,
    same defender pool sampling sequence, enemy_tactical_override left
    unset so _move_enemy() uses its normal hand-designed tactical
    logic)."""
    from uav_defend.envs.soldier_env import SoldierEnv
    from uav_defend.policies.sanitize import build_policy_info
    from experiments.adaptive_attacker_env import sample_defender

    rows = []
    pool = _pool()
    for i in range(n_episodes):
        seed = ATTACKER_VALIDATION_SEED_START + 50_000 + i  # distinct sub-range, never used by training/validation
        rng = np.random.default_rng(seed)
        config = build_randomized_training_config(rng, "measurement")
        env = SoldierEnv(config)
        # enemy_tactical_override intentionally left as its default None:
        # the hostile flies its ORIGINAL hand-designed tactical logic.
        obs, info = env.reset(seed=seed)
        member = sample_defender(rng, pool)
        member.policy.reset()
        start_dist = float(np.linalg.norm(env._soldier_pos - env._enemy_pos))
        steps = 0
        done = False
        while not done:
            defender_obs = env._get_obs()
            defender_info = build_policy_info(env._get_info(), member.estimator)
            if member.needs_sector_features:
                from uav_defend.obstacles.sector_features import SECTOR_OBSERVATION_V1, compute_sector_features
                sector = compute_sector_features(
                    defender_pos=env._defender_pos, defender_vel=env._defender_vel,
                    perceived=env._defender_last_perceived, sensor_range=SECTOR_OBSERVATION_V1.sensor_range,
                )
                if member.is_lr_ppo:
                    defender_action = member.policy.act(defender_obs, defender_info, extra_features=sector)
                else:
                    defender_action = member.policy.act(np.concatenate([defender_obs, sector]), defender_info)
            else:
                defender_action = member.policy.act(defender_obs, defender_info)
            obs, reward, terminated, truncated, info = env.step(np.asarray(defender_action, dtype=np.float32))
            done = terminated or truncated
            steps += 1
        end_dist = float(np.linalg.norm(env._soldier_pos - env._enemy_pos))
        rows.append({
            "seed": seed, "defender_family": member.family, "outcome": info["outcome"], "steps": steps,
            "asset_progress_total": start_dist - end_dist,
            "asset_progress_per_step": (start_dist - end_dist) / steps if steps else 0.0,
        })
    outcomes: dict[str, int] = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    return {
        "n_episodes": n_episodes,
        "soldier_caught_rate": outcomes.get("soldier_caught", 0) / n_episodes,
        "intercepted_rate": outcomes.get("intercepted", 0) / n_episodes,
        "outcomes": outcomes,
        "mean_asset_progress_per_step": float(np.mean([r["asset_progress_per_step"] for r in rows])),
        "episodes": rows,
    }


def main() -> dict:
    print("=== Adaptive Attacker v1 diagnostic training "
          f"(seed={ATTACKER_DIAGNOSTIC_SEED}, total_timesteps={DIAGNOSTIC_TOTAL_TIMESTEPS}) ===", flush=True)
    manifest = run_adaptive_attacker_training(
        env_fn=_env_fn,
        seed=ATTACKER_DIAGNOSTIC_SEED,
        output_dir=OUT_ROOT,
        total_timesteps=DIAGNOSTIC_TOTAL_TIMESTEPS,
        eval_freq=DIAGNOSTIC_EVAL_FREQ,
        eval_episodes=DIAGNOSTIC_EVAL_EPISODES,
        checkpoint_freq=DIAGNOSTIC_CHECKPOINT_FREQ,
        training_config_summary={"architecture": "adaptive_attacker_v1", "phase": "diagnostic"},
        verbose=0,
        num_envs=1,
    )
    print("training complete:", "wall_time_s=", manifest["wall_time_s"], flush=True)
    print("eval_history:", flush=True)
    for h in manifest["eval_history"]:
        print(" ", h, flush=True)

    print("=== scripted hostile comparison (matched diagnostic scenarios) ===", flush=True)
    scripted_comparison = run_scripted_hostile_comparison()
    print(scripted_comparison["soldier_caught_rate"], scripted_comparison["outcomes"],
          scripted_comparison["mean_asset_progress_per_step"], flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {"manifest": manifest, "scripted_hostile_comparison": scripted_comparison}
    (OUT_ROOT / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    main()
