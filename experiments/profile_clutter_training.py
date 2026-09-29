"""
Phase 2: runtime profiling for the obstacle-aware clutter training loop
(journal extension, scale-up sub-task). Uses `cProfile` (not assumptions)
against a short but representative training run, then reports cumulative
time by component, grouped into: env.step() physics, perception, static
avoidance, dynamic avoidance, sector-feature construction, collision
checking, PPO inference/update, reset/layout generation, and "other".

Usage:
    python -m experiments.profile_clutter_training
"""

from __future__ import annotations

import cProfile
import pstats
import io
from pathlib import Path

from experiments.clutter_training_env import build_ppo_clutter_env
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "clutter_training_profile"

PROFILE_TIMESTEPS = 8192  # ~4 rollouts at n_steps=2048; representative, still fast

# Function-name substrings -> component bucket (checked in order; first match wins).
COMPONENT_BUCKETS = [
    ("generate_layout", "reset/layout generation"),
    ("generate_dynamic_obstacles", "reset/layout generation"),
    ("_fixed_layout", "reset/layout generation"),
    ("_random_layout", "reset/layout generation"),
    ("EnemyObstaclePerception", "perception"),
    ("perception.py", "perception"),
    ("compute_sector_features", "obstacle-sector construction"),
    ("sector_features.py", "obstacle-sector construction"),
    ("ObstacleAvoidancePlanner", "avoidance planning (static+dynamic)"),
    ("avoidance.py", "avoidance planning (static+dynamic)"),
    ("_combined_avoidance_layout", "avoidance planning (static+dynamic)"),
    ("dynamic_obstacle_threatens", "avoidance planning (static+dynamic)"),
    ("dynamic_obstacle_swept_aabb", "avoidance planning (static+dynamic)"),
    ("first_segment_intersection", "collision checking"),
    ("swept_sphere_vs_sphere", "collision checking"),
    ("first_segment_intersection", "collision checking"),
    ("geometry.py", "collision checking"),
    ("layout.py:", "collision checking"),
    ("_move_defender", "env.step() core physics"),
    ("_move_enemy", "env.step() core physics"),
    ("_move_soldier", "env.step() core physics"),
    ("_advance_velocity", "env.step() core physics"),
    ("_update_detection", "env.step() core physics"),
    ("_get_obs", "env.step() core physics"),
    ("_get_info", "env.step() core physics"),
    ("soldier_env.py", "env.step() core physics"),
    ("torch", "PPO inference/update (torch)"),
    ("policies.py", "PPO inference/update (torch)"),
    ("ppo.py", "PPO inference/update (torch)"),
    ("buffers.py", "PPO inference/update (torch)"),
    ("distributions.py", "PPO inference/update (torch)"),
]


def _bucket_for(func_key) -> str:
    filename, _lineno, funcname = func_key
    haystack = f"{filename}:{funcname}"
    for needle, bucket in COMPONENT_BUCKETS:
        if needle in haystack:
            return bucket
    return "other"


def profile_training(total_timesteps: int = PROFILE_TIMESTEPS) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    def _make():
        env = build_ppo_clutter_env("measurement")
        env.reset(seed=71001)
        return Monitor(env, str(OUT_DIR / "monitor"))

    train_env = DummyVecEnv([_make])
    model = PPO(policy="MlpPolicy", env=train_env, n_steps=2048, verbose=0, seed=1)

    profiler = cProfile.Profile()
    profiler.enable()
    model.learn(total_timesteps=total_timesteps)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.dump_stats(str(OUT_DIR / "profile.pstats"))

    totals: dict[str, float] = {}
    grand_total = 0.0
    for func_key, (_cc, _nc, tt, _ct, _callers) in stats.stats.items():
        bucket = _bucket_for(func_key)
        totals[bucket] = totals.get(bucket, 0.0) + tt
        grand_total += tt

    percentages = {k: (v / grand_total * 100.0 if grand_total else 0.0) for k, v in totals.items()}
    ranked = dict(sorted(percentages.items(), key=lambda kv: -kv[1]))

    # Also dump the top 40 individual functions by cumulative time for detail.
    buf = io.StringIO()
    stats.sort_stats("cumulative").stream = buf
    stats.print_stats(40)
    (OUT_DIR / "top_functions.txt").write_text(buf.getvalue())

    result = {"total_timesteps": total_timesteps, "grand_total_seconds": grand_total, "percent_by_component": ranked}
    (OUT_DIR / "profile_summary.json").write_text(__import__("json").dumps(result, indent=2))
    return result


if __name__ == "__main__":
    result = profile_training()
    print(f"grand_total_seconds={result['grand_total_seconds']:.2f}")
    for k, v in result["percent_by_component"].items():
        print(f"  {k:40s} {v:6.2f}%")
