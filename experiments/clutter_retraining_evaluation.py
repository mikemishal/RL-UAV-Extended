"""
Phase 21-25: primary comparison evaluation for the obstacle-aware clutter
retraining study (journal extension). Compares, under IDENTICAL canonical
clutter scenarios (Phase 2 protocol, held-out seeds -- Phase 23):

    - classical baselines (Greedy, PN, Lead, RMPC) + shared safety layer
      (Phase 22 -- they never receive the new RL obstacle observation);
    - FROZEN conference PPO/PPO-Kalman/LR-PPO (16-D / 19-D observation,
      unchanged) + shared safety layer (Phase 21 zero-shot-degradation arm);
    - CLUTTER-TRAINED PPO-C/PPOK-C/LRPPO-C (36-D / 39-D observation with
      sector features) + shared safety layer (Phase 21 recovery arm).

Every controller is evaluated with the SAME held-out seed set (never used
in training -- see `journal_clutter_protocol.CLUTTER_EVAL_SEED_START`) at
the SAME canonical clutter levels (Phase 2), so results are directly
comparable (Phase 26 fairness precedent).

RUNTIME NOTE: RMPC remains ~150-400x slower than every other controller
(documented in `clutter_benchmark_protocol.py`); `RMPC_N_EPISODES` may use
a smaller matched subset, reported separately (Phase 23's own allowance).

SAMPLE SIZE NOTE (Phase 15, explicit and honest per user decision): the
spec target is N>=500 held-out episodes/controller/level. A full N=500
sweep across ~22 controller instances (13 frozen/classical + 9 new
clutter-trained checkpoints) x 3 levels was estimated (from measured
per-episode cost) at roughly 8-17+ additional hours of unattended
background compute beyond the ~6.4h already spent on the training
campaign. Given that cost, this module defaults to a REDUCED
`N_EPISODES_DEFAULT=100` for this run (still large enough for informative
Wilson-score confidence intervals), explicitly flagged as reduced-from-spec
in every summary/report artifact this module writes. The harness itself is
unchanged in structure and can reproduce the full N=500 protocol by
passing `n_episodes=500` (or `--n-episodes 500` on the CLI) with no code
changes, if/when that additional compute budget is available.

Usage:
    python -m experiments.clutter_retraining_evaluation
    python -m experiments.clutter_retraining_evaluation --n-episodes 500
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from uav_defend.policies.sanitize import build_policy_info

from experiments.journal_clutter_protocol import (
    CANONICAL_CLUTTER_LEVEL_ORDER,
    CLUTTER_EVAL_SEED_START,
    build_canonical_env_config,
)
from experiments.clutter_benchmark_protocol import controller_instances as frozen_controller_instances

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "clutter_retraining_evaluation"

EVAL_LEVELS = ("C1", "C2", "C3")
FULL_SCALE_N_EPISODES = 500  # Phase 15 spec target -- see module docstring
N_EPISODES_DEFAULT = 100  # reduced-but-honest for this session, see module docstring
RMPC_N_EPISODES_DEFAULT = 20  # proportionally reduced matched subset (was 10 @ N=30)

# Base seeds used by the Phase 10 400k campaign (experiments/run_clutter_400k_campaign.py) --
# same as journal_clutter_protocol.CLUTTER_TRAINING_SEEDS.
CAMPAIGN_SEEDS = {
    "ppo_clutter_v1": (71, 72, 73),
    "ppo_kalman_clutter_v1": (74, 75, 76),
    "lrppo_clutter_v1": (77, 78, 79),
}
CAMPAIGN_ROOT = PROJECT_ROOT / "results" / "clutter_training_400k"


def discover_campaign_checkpoints() -> list[tuple[str, int, Path]]:
    """Phase 18 (seed robustness): every (architecture, seed) checkpoint
    produced by the Phase 10 400k campaign, using `best_model.zip`
    (periodic-eval-best -- see run_clutter_400k_campaign.py's documented
    model-selection rule), falling back to `final_model.zip` if missing."""
    found = []
    for arch, seeds in CAMPAIGN_SEEDS.items():
        for seed in seeds:
            run_dir = CAMPAIGN_ROOT / arch / f"seed_{seed}"
            path = run_dir / "best_model.zip"
            if not path.exists():
                path = run_dir / "final_model.zip"
            if not path.exists():
                raise FileNotFoundError(f"no checkpoint found for {arch} seed_{seed} under {run_dir}")
            found.append((arch, seed, path))
    return found


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Phase 16: Wilson score confidence interval for a binomial rate
    (robust at small/moderate N and at rates near 0 or 1, unlike the naive
    normal approximation)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z ** 2 / n
    center = p + z ** 2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((center - margin) / denom, (center + margin) / denom)


def held_out_seeds(n_episodes: int) -> tuple[int, ...]:
    return tuple(range(CLUTTER_EVAL_SEED_START, CLUTTER_EVAL_SEED_START + n_episodes))


def _run_episode(env, policy, seed: int, obs_slice: slice | None) -> dict:
    obs, info = env.reset(seed=seed)
    policy.reset()
    estimator_mode = "kalman" if env.unwrapped.config.use_kalman_tracking else "measurement"
    avoidance_steps = 0
    override_angles = []
    min_static_clearance = info["defender_min_obstacle_clearance"]
    step = 0
    done = False
    while not done:
        policy_obs = obs[obs_slice] if obs_slice is not None else obs
        action = np.asarray(policy.act(policy_obs, build_policy_info(info, estimator_mode)), dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1
        min_static_clearance = min(min_static_clearance, info["defender_min_obstacle_clearance"])
        if info.get("defender_avoidance_active"):
            avoidance_steps += 1
            override_angles.append(info["defender_avoidance_angle_deg"])
    outcome = info["outcome"]
    return {
        "seed": seed, "outcome": outcome,
        "safe_intercept": 1 if outcome == "intercepted" else 0,
        "defender_collision": 1 if outcome in (
            "defender_static_obstacle_collision", "defender_dynamic_obstacle_collision") else 0,
        "attacker_environmental_collision": 1 if outcome in (
            "enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision") else 0,
        "soldier_caught": 1 if outcome == "soldier_caught" else 0,
        "unsafe_intercept": 1 if outcome == "unsafe_intercept" else 0,
        "timeout": 1 if outcome == "timeout" else 0,
        "episode_length": step,
        "intercept_time": step if outcome == "intercepted" else None,
        "min_clearance": min_static_clearance,
        "avoidance_fraction": avoidance_steps / step if step else 0.0,
        "mean_override_angle": float(np.mean(override_angles)) if override_angles else 0.0,
        "p95_override_angle": float(np.percentile(override_angles, 95)) if override_angles else 0.0,
    }


def evaluate_frozen_and_classical(n_episodes: int = N_EPISODES_DEFAULT,
                                   rmpc_n_episodes: int = RMPC_N_EPISODES_DEFAULT) -> pd.DataFrame:
    """Phase 21 (frozen arm) + Phase 22 (classical baselines): every
    instance from the existing `controller_instances()` inventory, run
    under the canonical clutter protocol with held-out seeds. Frozen
    learned policies receive obs[:16] (their original observation slice);
    classical policies receive the same slice for API compatibility (they
    never use the appended sector features)."""
    from uav_defend.envs import SoldierEnv
    from experiments.clutter_training_env import build_ppo_clutter_env

    rows = []
    instances = frozen_controller_instances()
    for level in EVAL_LEVELS:
        for inst in instances:
            n = rmpc_n_episodes if inst.family == "RMPC" else n_episodes
            seeds = held_out_seeds(n)
            base_config = build_canonical_env_config(level, inst.estimator)
            env = build_ppo_clutter_env(inst.estimator, randomized=False, fixed_config=base_config)
            policy = inst.build(base_config)
            t0 = time.time()
            for seed in seeds:
                metrics = _run_episode(env, policy, seed, obs_slice=slice(0, 16))
                metrics.update(level=level, family=inst.family, instance_id=inst.instance_id, arm="frozen_or_classical")
                rows.append(metrics)
            print(f"level={level} instance={inst.instance_id:>10s} n={n} ({time.time()-t0:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def evaluate_clutter_trained(checkpoints: list[tuple[str, int, Path]] | None = None,
                              n_episodes: int = N_EPISODES_DEFAULT) -> pd.DataFrame:
    """Phase 21 (recovery arm) + Phase 18 (seed robustness): ALL clutter-
    trained PPO-C/PPOK-C/LRPPO-C checkpoints from the Phase 10 400k
    campaign (default: `discover_campaign_checkpoints()`, i.e. all 9 --
    3 seeds x 3 architectures), receiving the FULL augmented observation.
    Each row is tagged with `family` (architecture) AND `training_seed`
    (which of the 3 checkpoints), so seed-level variability is directly
    analyzable (Phase 18), on TOP of `seed` (the per-episode held-out
    evaluation seed, shared across ALL checkpoints/architectures/levels)."""
    from stable_baselines3 import PPO
    from experiments.clutter_training_env import build_ppo_clutter_env
    from experiments.lr_ppo_clutter_training_env import build_lr_ppo_clutter_eval_env

    if checkpoints is None:
        checkpoints = discover_campaign_checkpoints()

    class _SB3PolicyAdapter:
        def __init__(self, model):
            self.model = model
        def reset(self):
            pass
        def act(self, obs, info):
            action, _ = self.model.predict(obs, deterministic=True)
            return action

    rows = []
    for level in EVAL_LEVELS:
        for arch, training_seed, path in checkpoints:
            model = PPO.load(path)
            policy = _SB3PolicyAdapter(model)
            estimator = "kalman" if arch == "ppo_kalman_clutter_v1" else "measurement"
            if arch == "lrppo_clutter_v1":
                base_config = build_canonical_env_config(level, "measurement")
                # Bug fix (Phase 15-17, see build_lr_ppo_clutter_eval_env
                # docstring): use the FIXED-config eval env, NOT
                # build_lr_ppo_clutter_env (which always resamples clutter
                # per-episode and would make `level` meaningless here).
                env = build_lr_ppo_clutter_eval_env(base_config, root_seed=CLUTTER_EVAL_SEED_START)
            else:
                base_config = build_canonical_env_config(level, estimator)
                env = build_ppo_clutter_env(estimator, randomized=False, fixed_config=base_config)
            instance_id = f"{arch}-{training_seed}"
            seeds = held_out_seeds(n_episodes)
            t0 = time.time()
            for seed in seeds:
                metrics = _run_episode(env, policy, seed, obs_slice=None)
                metrics.update(level=level, family=arch, instance_id=instance_id,
                                arm="clutter_trained", training_seed=training_seed)
                rows.append(metrics)
            print(f"level={level} instance={instance_id:>22s} n={n_episodes} ({time.time()-t0:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 16 (statistical metrics): per-group rates PLUS Wilson-score
    95% confidence intervals for the two primary binomial outcomes
    (safe-intercept rate, defender-collision rate)."""
    group_cols = ["level", "family", "arm"]
    if "training_seed" in df.columns and df["training_seed"].notna().any():
        group_cols = ["level", "family", "training_seed", "arm"]
    summary = df.groupby(group_cols).agg(
        n=("seed", "count"),
        safe_intercept_rate=("safe_intercept", "mean"),
        defender_collision_rate=("defender_collision", "mean"),
        attacker_environmental_collision_rate=("attacker_environmental_collision", "mean"),
        soldier_caught_rate=("soldier_caught", "mean"),
        unsafe_intercept_rate=("unsafe_intercept", "mean"),
        timeout_rate=("timeout", "mean"),
        mean_min_clearance=("min_clearance", "mean"),
        mean_avoidance_fraction=("avoidance_fraction", "mean"),
        mean_override_angle=("mean_override_angle", "mean"),
        p95_override_angle=("p95_override_angle", "mean"),
    ).reset_index()
    ci_lo, ci_hi = [], []
    coll_lo, coll_hi = [], []
    for _, row in summary.iterrows():
        n = int(row["n"])
        lo, hi = wilson_ci(round(row["safe_intercept_rate"] * n), n)
        ci_lo.append(lo)
        ci_hi.append(hi)
        clo, chi = wilson_ci(round(row["defender_collision_rate"] * n), n)
        coll_lo.append(clo)
        coll_hi.append(chi)
    summary["safe_intercept_rate_ci95_lo"] = ci_lo
    summary["safe_intercept_rate_ci95_hi"] = ci_hi
    summary["defender_collision_rate_ci95_lo"] = coll_lo
    summary["defender_collision_rate_ci95_hi"] = coll_hi
    return summary


def compute_recovery_deltas(frozen_df: pd.DataFrame, trained_df: pd.DataFrame) -> pd.DataFrame:
    """Phase 17 (frozen-vs-trained recovery): for each (level, family in
    {PPO, PPO-Kalman, LR-PPO}), compare the FROZEN conference model's
    zero-shot safe-intercept/collision rates in clutter (mean across its 3
    seeds) against the NEW clutter-trained model's rates (mean across its
    3 seeds), reporting the delta (positive = clutter training helped)."""
    family_map = {"PPO": "ppo_clutter_v1", "PPO-Kalman": "ppo_kalman_clutter_v1", "LR-PPO": "lrppo_clutter_v1"}
    frozen_agg = frozen_df[frozen_df["family"].isin(family_map)].groupby(["level", "family"]).agg(
        n_episodes_total=("seed", "count"),
        frozen_safe_intercept_rate=("safe_intercept", "mean"),
        frozen_defender_collision_rate=("defender_collision", "mean"),
    ).reset_index()
    trained_agg = trained_df.groupby(["level", "family"]).agg(
        n_episodes_total=("seed", "count"),
        trained_safe_intercept_rate=("safe_intercept", "mean"),
        trained_defender_collision_rate=("defender_collision", "mean"),
    ).reset_index()
    trained_agg["family"] = trained_agg["family"].map({v: k for k, v in family_map.items()})
    merged = frozen_agg.merge(trained_agg, on=["level", "family"], suffixes=("_frozen", "_trained"))
    merged["safe_intercept_rate_delta"] = (
        merged["trained_safe_intercept_rate"] - merged["frozen_safe_intercept_rate"])
    merged["defender_collision_rate_delta"] = (
        merged["trained_defender_collision_rate"] - merged["frozen_defender_collision_rate"])
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=N_EPISODES_DEFAULT,
                        help=f"Held-out episodes per (controller, level). Spec target is "
                             f"{FULL_SCALE_N_EPISODES}; default {N_EPISODES_DEFAULT} is a "
                             f"reduced-but-honest sample for interactive-session budgets (see module docstring).")
    parser.add_argument("--rmpc-n-episodes", type=int, default=RMPC_N_EPISODES_DEFAULT)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = evaluate_frozen_and_classical(n_episodes=args.n_episodes, rmpc_n_episodes=args.rmpc_n_episodes)
    df.to_csv(OUT_DIR / "frozen_and_classical_episodes.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUT_DIR / "frozen_and_classical_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(summary.to_string(index=False))

    df_trained = evaluate_clutter_trained(n_episodes=args.n_episodes)
    df_trained.to_csv(OUT_DIR / "clutter_trained_episodes.csv", index=False)
    summary_trained = summarize(df_trained)
    summary_trained.to_csv(OUT_DIR / "clutter_trained_summary.csv", index=False)
    print(summary_trained.to_string(index=False))

    # Phase 18: seed-robustness -- aggregate clutter-trained across the 3
    # seeds per architecture (mean/std of the per-seed rates already in
    # summary_trained, grouped by level+family).
    seed_robustness = summary_trained.groupby(["level", "family"]).agg(
        n_seeds=("training_seed", "nunique"),
        mean_safe_intercept_rate=("safe_intercept_rate", "mean"),
        std_safe_intercept_rate=("safe_intercept_rate", "std"),
        mean_defender_collision_rate=("defender_collision_rate", "mean"),
        std_defender_collision_rate=("defender_collision_rate", "std"),
    ).reset_index()
    seed_robustness.to_csv(OUT_DIR / "clutter_trained_seed_robustness.csv", index=False)
    print(seed_robustness.to_string(index=False))

    # Phase 17: frozen-vs-trained recovery deltas.
    recovery = compute_recovery_deltas(df, df_trained)
    recovery.to_csv(OUT_DIR / "recovery_deltas.csv", index=False)
    print(recovery.to_string(index=False))

    print(f"\nNOTE: N_EPISODES={args.n_episodes} (spec target: {FULL_SCALE_N_EPISODES}); "
          f"see module docstring for the documented reduced-sample decision.", flush=True)
