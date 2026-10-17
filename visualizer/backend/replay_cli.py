"""Generate one deterministic real simulation replay from frozen policies."""

from __future__ import annotations

import argparse
import json

from pydantic import ValidationError

from .application_error import ApplicationError
from .config import load_settings
from .replay_engine import ATTACKER_SEEDS, ReplayRequest, run_replay_episode, save_replay_bundle


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one real SoldierEnv episode and save a compressed native-3D replay."
    )
    parser.add_argument(
        "--defender",
        required=True,
        choices=("greedy", "kalman_greedy", "pn", "lead", "rmpc", "ppo", "ppo_kalman", "lr_ppo"),
    )
    parser.add_argument("--generation", choices=("conference", "d0", "d0_plus", "d1"))
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--attacker", choices=("A0", "A1-201", "A1-202", "A1-203"), default="A0")
    parser.add_argument("--clutter", required=True, choices=("C0", "C1", "C2", "C3"))
    parser.add_argument("--perception", choices=("local", "map_local"), default="map_local")
    parser.add_argument("--evaluation-seed", required=True, type=int)
    parser.add_argument(
        "--output-name",
        help="Optional filename stem only; output remains under visualizer/data/local and is never overwritten.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        request = ReplayRequest(
            defender_policy_id=args.defender,
            defender_generation=args.generation,
            defender_training_seed=args.training_seed,
            attacker_condition=args.attacker,
            attacker_model_seed=ATTACKER_SEEDS.get(args.attacker),
            clutter_level=args.clutter,
            perception_mode=args.perception,
            evaluation_seed=args.evaluation_seed,
            deterministic_inference=True,
        )
        bundle = run_replay_episode(request, load_settings())
        output_path = save_replay_bundle(bundle, args.output_name)
    except ValidationError:
        print(json.dumps({
            "error": {
                "code": "INVALID_REQUEST",
                "message": "Replay selection does not match the CLI contract.",
            }
        }))
        return 2
    except ApplicationError as exc:
        print(json.dumps({"error": {"code": exc.code, "message": exc.message}}))
        return 1

    episode = bundle.episodes[0]
    checkpoint_details = {}
    for role, policy in (("defender", episode.defender), ("attacker", episode.attacker)):
        checkpoint = policy.checkpoint
        if checkpoint is not None:
            checkpoint_details[role] = {
                "artifact_id": checkpoint.artifact_id,
                "relative_path": checkpoint.research_relative_path,
                "sha256": checkpoint.sha256,
                "role": checkpoint.role,
            }
    print(json.dumps({
        "simulation_source_commit": bundle.provenance.simulation_source_commit,
        "defender": episode.defender.model_dump(mode="json"),
        "attacker": episode.attacker.model_dump(mode="json"),
        "checkpoints": checkpoint_details,
        "scenario": {
            "clutter": episode.environment.clutter,
            "perception": episode.environment.awareness,
            "config_sha256": episode.environment.config_sha256,
        },
        "evaluation_seed": episode.simulation_seed,
        "frame_count": len(episode.frames),
        "outcome": episode.summary.outcome,
        "steps": episode.summary.steps,
        "total_reward": episode.summary.total_reward,
        "output_path": str(output_path),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
