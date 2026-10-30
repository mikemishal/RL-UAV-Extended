"""Campaign driver for the first canonical V2 A1 training run (`a1_v2_training_v1`).

Wraps the frozen runner (`experiments.journal_v2_a1_training`) without changing it:

    python -m experiments.journal_v2_a1_campaign preflight
    python -m experiments.journal_v2_a1_campaign train            # seeds 201, 202, 203 in sequence; resumes/skips
    python -m experiments.journal_v2_a1_campaign registry         # campaign-level model registry (after all 3)

No reward, environment, protocol or hyper-parameter setting is touched here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

import experiments.journal_v2_a1_training as T
from experiments import journal_v2_a1 as A
from experiments.journal_v2_protocol import assert_canonical_v2_training_config

CAMPAIGN_ROOT = T.A1_V2_RESULTS_ROOT
CAMPAIGN_MANIFEST = CAMPAIGN_ROOT / "campaign_manifest.json"
CAMPAIGN_REGISTRY = CAMPAIGN_ROOT / "a1_v2_model_registry.json"
PROTOCOL_TAG = "journal-v2-a1-training-protocol-v1"
EXPECTED_PROTOCOL_COMMIT = "b881bddad2cb19f6f6ff0cb1bcc1bf05f31a0fd1"
TORCH_THREADS = 2  # main-process torch threads; the 16 env workers own the CPU during rollouts


def git(*args):
    return subprocess.check_output(["git", *args], cwd=T.PROJECT_ROOT, text=True).strip()


def verify_frozen_state() -> dict:
    state = {
        "branch": git("branch", "--show-current"),
        "head": git("rev-parse", "HEAD"),
        "protocol_tag_commit": git("rev-parse", f"{PROTOCOL_TAG}^{{commit}}"),
        "a0_tag_commit": git("rev-parse", f"{T.CANONICAL_A0_TAG}^{{commit}}"),
        "tracked_changes": [line for line in git("status", "--porcelain", "--untracked-files=no").splitlines() if line],
    }
    assert state["branch"] == "research/journal-corrected-attacker-v2", state
    assert state["a0_tag_commit"] == T.CANONICAL_A0_COMMIT, state
    assert state["protocol_tag_commit"] == EXPECTED_PROTOCOL_COMMIT, state
    # HEAD may advance only by non-protocol commits (campaign metadata); the frozen modules must equal the tag.
    changed = git("diff", "--name-only", PROTOCOL_TAG, "HEAD", "--", "uav_defend", "experiments/journal_v2_a1.py",
                  "experiments/journal_v2_a1_training.py", "experiments/journal_v2_protocol.py",
                  "docs/journal_v2_a1_training_protocol.md")
    assert not changed, f"frozen files differ from the protocol tag: {changed}"
    assert not state["tracked_changes"], state["tracked_changes"]
    state["frozen_files_identical_to_protocol_tag"] = True
    assert A.A1_V2_REWARD_VERSION == "a1_v2_reward_v1" and A.A1_V2_TRAINING_PROTOCOL == "a1_v2_training_v1"
    assert A.A1_V2_OBSERVATION_DIM == 37 and A.A1_V2_ACTION_DIM == 3
    return state


def verify_registered_checkpoints() -> dict:
    registry = json.loads((T.PROJECT_ROOT / "results" / "defender_adaptation_model_registry.json").read_text())
    entries = list(registry["defenders"].values()) + list(registry["attackers"].values())
    bad = [e["path"] for e in entries if T.sha256_file(T.PROJECT_ROOT / e["path"] if not os.path.isabs(e["path"]) else e["path"]) != e["sha256"]]
    assert not bad, f"historical checkpoint hash mismatch: {bad}"
    return {"verified": len(entries), "mismatches": 0}


def compute_environment() -> dict:
    import stable_baselines3
    import torch

    return {
        "python": sys.version.split()[0], "platform": platform.platform(), "torch": torch.__version__,
        "stable_baselines3": stable_baselines3.__version__, "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()), "ppo_device_requested": "auto",
        "ppo_device_resolved": "cuda" if torch.cuda.is_available() else "cpu",
        "cpu_count_logical": os.cpu_count(), "cpu": platform.processor(), "num_vector_envs": A.A1_V2_NUM_ENVS,
        "vector_env": "SubprocVecEnv", "torch_threads_main_process": TORCH_THREADS,
    }


def preflight_environment_checks(max_steps: int = 400) -> dict:
    """One env per (clutter level, defender family): guard, shapes, finiteness, endurance range, terminal mapping."""
    rng = np.random.default_rng(0)
    bank = A.A1V2DefenderBank()
    outcomes_seen: dict[str, float] = {}
    cells = {}
    for level in A.A1_V2_CLUTTER_LEVELS:
        for family in A.A1_V2_DEFENDER_FAMILIES:
            config = A.build_episode_config(level, family)
            assert_canonical_v2_training_config(config)
            env = A.A1V2TrainingEnv(201, 0, defender_bank=bank)
            for mode in ("random", "asset_ward"):
                obs, info = env.reset(options={"scenario": (level, family, 3_000_000 + len(cells))})
                assert obs.shape == (37,) and env.action_space.shape == (3,) and np.isfinite(obs).all()
                assert info["a1_scenario"] == {"level": level, "defender": family, "scenario_seed": env.scenario[2]}
                assert obs[-1] == 1.0
                done, steps = False, 0
                while not done and steps < max_steps:
                    if mode == "random":
                        action = env.action_space.sample() if False else rng.uniform(-1, 1, 3).astype(np.float32)
                    else:
                        direction = obs[3:6] - obs[0:3]
                        action = (direction / (np.linalg.norm(direction) + 1e-9)).astype(np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    assert obs.shape == (37,) and np.isfinite(obs).all() and 0.0 <= obs[-1] <= 1.0, (level, family)
                    assert np.isfinite(reward) and reward == reward
                    assert abs(reward - sum(info["a1_reward_components"].values())) < 1e-9
                    done, steps = terminated or truncated, steps + 1
                assert done, f"episode did not terminate within {max_steps} steps for {level}/{family}/{mode}"
                episode = info["a1_episode"]
                expected = A.terminal_reward(episode["outcome"])
                assert episode["components"]["terminal"] == expected
                assert expected == (10.0 if episode["outcome"] == "soldier_caught" else -10.0)
                outcomes_seen[episode["outcome"]] = expected
                cells[f"{level}/{family}/{mode}"] = episode["outcome"]
    # sampling sanity
    stream = A.training_scenario_stream(201, 0)
    draws = [A.sample_training_scenario(stream) for _ in range(2000)]
    assert {d[0] for d in draws} == set(A.A1_V2_CLUTTER_LEVELS) and {d[1] for d in draws} == set(A.A1_V2_DEFENDER_FAMILIES)
    return {"cells_checked": len(cells), "terminal_mapping_observed": outcomes_seen, "episode_outcomes": cells}


def worker_memory_check(num_envs: int = A.A1_V2_NUM_ENVS) -> dict:
    """Spawn the real 16 subprocess workers once (no PPO), reset and step, and measure memory/availability."""
    from stable_baselines3.common.vec_env import SubprocVecEnv

    vec = SubprocVecEnv(T.worker_env_fns(201, num_envs))
    try:
        vec.reset()
        t0 = time.time()
        for _ in range(5):
            vec.step(np.zeros((num_envs, 3), dtype=np.float32) + 0.5)
        elapsed = time.time() - t0
    finally:
        vec.close()
    return {"workers": num_envs, "five_vector_steps_s": elapsed}


def preflight(write: bool = True) -> dict:
    manifest = {
        "campaign": "v2_a1_first_canonical_training", "protocol": A.A1_V2_TRAINING_PROTOCOL,
        "reward_version": A.A1_V2_REWARD_VERSION, "observation_dim": 37, "action_dim": 3,
        "frozen_state": verify_frozen_state(), "historical_checkpoints": verify_registered_checkpoints(),
        "compute_environment": compute_environment(), "config_sha256": T.config_fingerprints(),
        "preflight_checks": preflight_environment_checks(), "worker_spawn_check": worker_memory_check(),
        "ppo_seeds": list(A.A1_V2_PPO_SEEDS), "requested_total_timesteps": A.A1_V2_TOTAL_TIMESTEPS,
        "actual_rollout_aligned_timesteps": 13 * A.A1_V2_NUM_ENVS * A.A1_V2_PPO_HYPERPARAMS["n_steps"],
        "ppo_hyperparameters": dict(A.A1_V2_PPO_HYPERPARAMS),
        "preflight_utc": datetime.now(timezone.utc).isoformat(), "training": {},
    }
    assert manifest["actual_rollout_aligned_timesteps"] == 425_984
    if write:
        T.write_json_atomic(CAMPAIGN_MANIFEST, manifest)
    return manifest


def train_all(seeds=A.A1_V2_PPO_SEEDS) -> None:
    import torch

    torch.set_num_threads(TORCH_THREADS)
    manifest = json.loads(CAMPAIGN_MANIFEST.read_text())
    for seed in seeds:
        paths = T.run_paths(seed)
        if paths.final_model.exists():
            print(f"seed {seed}: complete, skipping", flush=True)
            continue
        started = time.time()
        record = manifest["training"].setdefault(str(seed), {})
        record.setdefault("starts_utc", []).append(datetime.now(timezone.utc).isoformat())
        record["git_head"] = git("rev-parse", "HEAD")
        T.write_json_atomic(CAMPAIGN_MANIFEST, manifest)
        print(f"seed {seed}: starting ({T.decide_run_state(paths)})", flush=True)
        try:
            final = T.run_a1_v2_training(seed)
        except BaseException as exc:
            # Recorded, then the process exits; an outer loop re-invokes `train`, which continues the same run through
            # the protocol-defined resume mechanism (never restarted with different settings).
            record.setdefault("interruptions", []).append({"utc": datetime.now(timezone.utc).isoformat(), "error": repr(exc)})
            T.write_json_atomic(CAMPAIGN_MANIFEST, manifest)
            raise
        record.update(status="complete", wall_time_s_this_invocation=time.time() - started,
                      final_model_sha256=final["model_sha256"], num_timesteps=final["num_timesteps"],
                      finished_utc=datetime.now(timezone.utc).isoformat())
        T.write_json_atomic(CAMPAIGN_MANIFEST, manifest)
        print(f"seed {seed}: done {final['num_timesteps']} steps sha {final['model_sha256'][:12]}", flush=True)


def build_registry() -> dict:
    registry = {"protocol": A.A1_V2_TRAINING_PROTOCOL, "reward_version": A.A1_V2_REWARD_VERSION,
                "research_commit_at_registry": git("rev-parse", "HEAD"), "protocol_tag": PROTOCOL_TAG,
                "canonical_a0_commit": T.CANONICAL_A0_COMMIT, "canonical": False,
                "note": "NOT yet canonical: pending quantitative gates and manual visual inspection", "attackers": {}}
    for seed in A.A1_V2_PPO_SEEDS:
        paths = T.run_paths(seed)
        sidecar = json.loads(paths.final_model.with_suffix(".manifest.json").read_text())
        actual = T.sha256_file(paths.final_model)
        assert actual == sidecar["model_sha256"], f"seed {seed}: manifest SHA mismatch"
        registry["attackers"][f"A1-{seed}-V2"] = {
            "ppo_seed": seed, "path": str(paths.final_model.relative_to(T.PROJECT_ROOT)).replace("\\", "/"),
            "sha256": actual, "num_timesteps": sidecar["num_timesteps"], "observation_dim": sidecar["observation_dim"],
            "action_dim": sidecar["action_dim"], "training_research_commit": sidecar["research_commit"],
            "held_out_from_defender_adaptation": seed == A.A1_V2_HELD_OUT_FROM_DEFENDER_ADAPTATION}
    T.write_json_atomic(CAMPAIGN_REGISTRY, registry)
    return registry


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("preflight", "train", "registry"))
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
        print(json.dumps({k: v for k, v in result.items() if k != "preflight_checks"}, indent=1, default=str))
        print("preflight cells:", result["preflight_checks"]["cells_checked"], result["preflight_checks"]["terminal_mapping_observed"])
    elif args.stage == "train":
        train_all()
    else:
        print(json.dumps(build_registry(), indent=1))
