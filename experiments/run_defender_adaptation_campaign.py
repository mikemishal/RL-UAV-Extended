"""
The FULL 18-run D0+/D1 continuation-training campaign driver (journal
extension, 34-phase defender-adaptation sub-task, Phase 13/17).

3 architectures (PPO, PPO-Kalman, LR-PPO) x 3 existing D0 training seeds
each x {D0+ (matched scripted-only control), D1 (50/25/25 A0/A1-seen
mixture)} = 18 continuation-training runs. Each run continues from its
EXISTING 400k-step D0 checkpoint (`results/clutter_training_400k/<arch>/
seed_<seed>/best_model.zip`, NEVER modified by this module -- see
`experiments.defender_adaptation_training_runner._verify_frozen_attacker_hashes`
for the analogous frozen-ATTACKER-checkpoint guarantee; the frozen D0
STARTING checkpoints are separately re-verified by the pre-launch hash
audit, never written to by `run_defender_adaptation_training` since its
`output_dir` is always a NEW, separate directory under
`results/defender_adaptation/`).

Cadence/vectorization/hyperparameters are IDENTICAL to the original D0
campaign (`experiments/run_clutter_400k_campaign.py`): num_envs=16,
eval_freq=20_000, eval_episodes=20, checkpoint_freq=50_000,
total_additional_timesteps=400_000 -- the ONLY permitted difference
between a matched D0+/D1 pair is `env_fn` (opponent distribution).

Usage:
    python -m experiments.run_defender_adaptation_campaign --dry-run
    python -m experiments.run_defender_adaptation_campaign --arch ppo_clutter_v1 --seed 71 --generation d0_plus
    python -m experiments.run_defender_adaptation_campaign  # (after explicit launch authorization)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.adaptive_attacker_env import _clutter_checkpoint_path
from experiments.clutter_training_env import build_ppo_clutter_env
from experiments.lr_ppo_clutter_training_env import build_lr_ppo_clutter_env
from experiments.defender_adaptation_env import (
    build_ppo_d1_env, build_lr_ppo_d1_env, build_opponent_pool, opponent_pool_spec, A1_SEEN_SEEDS, A1_UNSEEN_SEED,
)
from experiments.defender_adaptation_training_runner import run_defender_adaptation_training
from experiments.journal_clutter_protocol import CLUTTER_TRAINING_SEEDS

PROJECT_ROOT = Path(__file__).parent.parent
OUT_ROOT = PROJECT_ROOT / "results" / "defender_adaptation"

CAMPAIGN_TOTAL_ADDITIONAL_TIMESTEPS = 400_000
CAMPAIGN_EVAL_FREQ = 20_000
CAMPAIGN_EVAL_EPISODES = 20
CAMPAIGN_CHECKPOINT_FREQ = 50_000
CAMPAIGN_NUM_ENVS = 16  # identical to the D0 campaign (measured, not assumed)

ARCHITECTURES = {
    "ppo_clutter_v1": {"seeds": CLUTTER_TRAINING_SEEDS["ppo"], "estimator": "measurement"},
    "ppo_kalman_clutter_v1": {"seeds": CLUTTER_TRAINING_SEEDS["ppo_kalman"], "estimator": "kalman"},
    "lrppo_clutter_v1": {"seeds": CLUTTER_TRAINING_SEEDS["lr_ppo"], "estimator": None},
}


def _d0_plus_env_fn(architecture: str, estimator: str | None):
    """IDENTICAL env builder to the original D0 training (Phase 18:
    'nothing else may differ between D0+ and D1 except the opponent
    distribution') -- 100% scripted opponent, achieved by NEVER touching
    `enemy_tactical_override` at all (not even via the mixture wrapper with
    weight 1.0 on A0), for the strongest possible equivalence guarantee."""
    if architecture == "lrppo_clutter_v1":
        return lambda worker_seed: build_lr_ppo_clutter_env(root_seed=worker_seed)
    return lambda worker_seed: build_ppo_clutter_env(estimator=estimator)


def _d1_env_fn(architecture: str, estimator: str | None):
    """50/25/25 A0/A1-seen-201/A1-seen-202 opponent mixture (Phase 7), on
    top of the SAME underlying D0 clutter-training env builder."""
    if architecture == "lrppo_clutter_v1":
        return lambda worker_seed: build_lr_ppo_d1_env(root_seed=worker_seed, pool=None)
    return lambda worker_seed: build_ppo_d1_env(estimator=estimator, pool=None)


def build_run_manifest() -> list[dict]:
    """The proposed 18-run manifest (Phase 17/25): architecture, defender
    seed, D0+/D1, starting checkpoint, training seed, output directory --
    fully explicit, no cherry-picking (Phase 17: ALL 3 seeds per arch)."""
    runs = []
    for architecture, spec in ARCHITECTURES.items():
        for seed in spec["seeds"]:
            starting_checkpoint = _clutter_checkpoint_path(architecture, seed)
            for generation in ("d0_plus", "d1"):
                subdir = "d0_plus_scripted" if generation == "d0_plus" else "d1_adaptive_mixture"
                output_dir = OUT_ROOT / subdir / architecture / f"seed_{seed}"
                runs.append({
                    "architecture": architecture,
                    "defender_seed": seed,
                    "generation": generation,
                    "starting_checkpoint": str(starting_checkpoint),
                    "training_seed": seed,
                    "output_dir": str(output_dir),
                })
    return runs


def run_one(architecture: str, seed: int, generation: str) -> dict:
    spec = ARCHITECTURES[architecture]
    estimator = spec["estimator"]
    starting_checkpoint = _clutter_checkpoint_path(architecture, seed)
    subdir = "d0_plus_scripted" if generation == "d0_plus" else "d1_adaptive_mixture"
    output_dir = OUT_ROOT / subdir / architecture / f"seed_{seed}"

    if generation == "d0_plus":
        env_fn = _d0_plus_env_fn(architecture, estimator)
        pool_spec = None
        training_config_summary = {"architecture": architecture, "generation": "d0_plus", "opponent": "A0-only"}
    else:
        env_fn = _d1_env_fn(architecture, estimator)
        pool_spec = opponent_pool_spec(build_opponent_pool())
        training_config_summary = {
            "architecture": architecture, "generation": "d1",
            "opponent_mixture": pool_spec["weights"], "a1_seen_seeds": list(A1_SEEN_SEEDS),
            "a1_unseen_seed": A1_UNSEEN_SEED,
        }

    return run_defender_adaptation_training(
        architecture=architecture,
        generation=generation,
        env_fn=env_fn,
        seed=seed,
        starting_checkpoint_path=starting_checkpoint,
        output_dir=output_dir,
        total_additional_timesteps=CAMPAIGN_TOTAL_ADDITIONAL_TIMESTEPS,
        opponent_pool_spec_dict=pool_spec,
        eval_freq=CAMPAIGN_EVAL_FREQ,
        eval_episodes=CAMPAIGN_EVAL_EPISODES,
        checkpoint_freq=CAMPAIGN_CHECKPOINT_FREQ,
        training_config_summary=training_config_summary,
        verbose=0,
        num_envs=CAMPAIGN_NUM_ENVS,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=list(ARCHITECTURES), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--generation", choices=["d0_plus", "d1"], default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the 18-run manifest and exit.")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(build_run_manifest(), indent=2))
        return

    architectures = [args.arch] if args.arch else list(ARCHITECTURES)
    generations = [args.generation] if args.generation else ["d0_plus", "d1"]
    for architecture in architectures:
        seeds = [args.seed] if args.seed else list(ARCHITECTURES[architecture]["seeds"])
        for seed in seeds:
            for generation in generations:
                print(f"=== {architecture} seed={seed} generation={generation} ===", flush=True)
                manifest = run_one(architecture, seed, generation)
                print(f"    actual_additional_timesteps={manifest['actual_additional_timesteps']}", flush=True)


if __name__ == "__main__":
    main()
