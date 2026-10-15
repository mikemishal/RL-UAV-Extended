"""Phase 4: LR-PPO residual-authority (30 degree) inference-time diagnostic.
Reuses the FROZEN D1 LR-PPO checkpoints UNCHANGED -- only the inference-time
`residual_max_angle_deg` passed to `LeadResidualPPOPolicyWrapper` varies
(15/30/45/60). This is NOT equivalent to a policy trained with those
bounds (the policy was trained with 30 degrees); results must be labeled
"inference-time authority diagnostic" per the task's explicit instruction.
Reuses `_run_matched_episode` from experiments/defender_adaptation_evaluation.py
UNCHANGED -- only the defender_policy instance passed to it differs.
"""

from __future__ import annotations

import json

import pandas as pd

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.registry import get_policy
from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper

from experiments.defender_adaptation_evaluation import (
    load_model_registry, build_defender_instances, build_attacker_conditions,
    verify_checkpoint_integrity, _run_matched_episode,
)
from experiments.mechanism_diagnostic import OUT_DIR, _env_config_for, CANONICAL_C0_EVAL_SEEDS, C1C3_EVAL_SEEDS

AUTHORITY_ANGLES_DEG = (15.0, 30.0, 45.0, 60.0)
AUTHORITY_LEVELS = ("C0", "C3")
N_AUTHORITY_SEEDS = 100  # explicit Phase 4 request
AUTHORITY_SEEDS_C0 = tuple(CANONICAL_C0_EVAL_SEEDS[:N_AUTHORITY_SEEDS])
AUTHORITY_SEEDS_C3 = tuple(C1C3_EVAL_SEEDS[:N_AUTHORITY_SEEDS])


def _authority_seeds_for(level: str) -> tuple[int, ...]:
    return AUTHORITY_SEEDS_C0 if level == "C0" else AUTHORITY_SEEDS_C3


def run_authority_diagnostic() -> pd.DataFrame:
    registry = load_model_registry()
    problems = verify_checkpoint_integrity(registry)
    if problems:
        raise RuntimeError("Checkpoint integrity check failed:\n" + "\n".join(problems))

    d1_lrppo = [d for d in build_defender_instances(registry)
                if d.architecture == "lrppo_clutter_v1" and d.generation == "d1"]
    attacker = next(a for a in build_attacker_conditions(registry) if a.name == "A1-unseen-203")

    rows = []
    for level in AUTHORITY_LEVELS:
        seeds = _authority_seeds_for(level)
        for d in d1_lrppo:
            base_config = _env_config_for(level, d.estimator)
            lead_policy_for_residual = get_policy("lead", config=base_config)
            env = SoldierEnv(base_config)
            for angle in AUTHORITY_ANGLES_DEG:
                policy = LeadResidualPPOPolicyWrapper.load(
                    d.checkpoint_path, residual_max_angle_deg=angle, config=base_config, deterministic=True)
                for eval_seed in seeds:
                    metrics = _run_matched_episode(env, d, policy, lead_policy_for_residual, attacker.model, eval_seed)
                    metrics.update(defender_seed=d.defender_seed, level=level, eval_seed=eval_seed,
                                   inference_residual_max_angle_deg=angle,
                                   is_canonical_training_angle=(angle == 30.0))
                    rows.append(metrics)
            print(f"authority diagnostic: seed={d.defender_seed} level={level} done", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "lrppo_authority_diagnostic_episodes.csv", index=False)
    return df


if __name__ == "__main__":
    manifest = {
        "angles_deg": list(AUTHORITY_ANGLES_DEG), "levels": AUTHORITY_LEVELS, "n_seeds": N_AUTHORITY_SEEDS,
        "note": "15/45/60 deg are INFERENCE-TIME authority diagnostics on the D1 checkpoint TRAINED with 30 deg; "
                "NOT equivalent to policies trained with those bounds.",
    }
    (OUT_DIR / "authority_diagnostic_manifest.json").write_text(json.dumps(manifest, indent=2))
    run_authority_diagnostic()
    print("=== authority diagnostic complete ===", flush=True)
