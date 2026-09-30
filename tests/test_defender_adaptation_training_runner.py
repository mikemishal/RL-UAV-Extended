"""
Tests for the continuation-aware, restart-safe defender-adaptation
training runner (journal extension, 34-phase defender-adaptation
sub-task, Phase 8/13/16/17/18).

Real (but tiny) PPO continuation runs, following the established
`tests/test_vectorized_clutter_training.py` convention: fast, in-process
`DummyVecEnv`/num_envs=1, tiny `total_additional_timesteps`. Output goes
under pytest's `tmp_path` fixture (OS temp dir, OUTSIDE the OneDrive-synced
workspace tree) rather than `results/` -- several of these tests
write-then-delete-then-rewrite the SAME directory (to simulate a restart),
and doing that under OneDrive's sync tree was observed to hit persistent
`PermissionError: [WinError 5] Access is denied` while the sync client
locks recently-written files (see repo memory note) -- `tmp_path` sidesteps
this at the root cause instead of retrying around it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.adaptive_attacker_env import _clutter_checkpoint_path
from experiments.clutter_training_env import build_ppo_clutter_env
from experiments.defender_adaptation_env import build_opponent_pool, opponent_pool_spec
from experiments.defender_adaptation_training_runner import (
    ADAPTIVE_ATTACKER_HASHES_PATH,
    _sha256_file,
    _verify_frozen_attacker_hashes,
    run_defender_adaptation_training,
)


def test_sha256_file_is_stable_and_content_sensitive(tmp_path):
    f1 = tmp_path / "a.bin"
    f1.write_bytes(b"hello world")
    f2 = tmp_path / "b.bin"
    f2.write_bytes(b"hello world")
    f3 = tmp_path / "c.bin"
    f3.write_bytes(b"hello WORLD")
    assert _sha256_file(f1) == _sha256_file(f2)
    assert _sha256_file(f1) != _sha256_file(f3)


def test_verify_frozen_attacker_hashes_passes_for_real_unchanged_checkpoints():
    spec = opponent_pool_spec(build_opponent_pool())
    _verify_frozen_attacker_hashes(spec)  # must not raise


def test_verify_frozen_attacker_hashes_noop_when_no_learned_members():
    _verify_frozen_attacker_hashes(None)
    _verify_frozen_attacker_hashes({"members": [{"kind": "scripted", "seed": None}]})


def test_verify_frozen_attacker_hashes_raises_on_missing_baseline(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "experiments.defender_adaptation_training_runner.ADAPTIVE_ATTACKER_HASHES_PATH",
        tmp_path / "does_not_exist.json",
    )
    spec = {"members": [{"name": "A1-seen-201", "kind": "learned", "seed": 201}]}
    with pytest.raises(RuntimeError, match="baseline missing"):
        _verify_frozen_attacker_hashes(spec)


def test_verify_frozen_attacker_hashes_raises_on_hash_mismatch(monkeypatch, tmp_path):
    tampered = tmp_path / "model_hashes.json"
    tampered.write_text(json.dumps([{"seed": 201, "sha256": "0" * 64}]))
    monkeypatch.setattr(
        "experiments.defender_adaptation_training_runner.ADAPTIVE_ATTACKER_HASHES_PATH", tampered,
    )
    spec = {"members": [{"name": "A1-seen-201", "kind": "learned", "seed": 201}]}
    with pytest.raises(RuntimeError, match="FROZEN ATTACKER CHECKPOINT MODIFIED"):
        _verify_frozen_attacker_hashes(spec)


def _d0_checkpoint() -> Path:
    return _clutter_checkpoint_path("ppo_clutter_v1", 71)


def test_continuation_run_advances_num_timesteps_not_reinitialized(tmp_path):
    """Continuing from a real D0 checkpoint must start `num_timesteps` at
    the D0 checkpoint's OWN (large, nonzero) value, not 0 -- proving the
    policy is loaded/continued, never reinitialized from scratch."""
    starting_checkpoint = _d0_checkpoint()
    from stable_baselines3 import PPO
    baseline_model = PPO.load(str(starting_checkpoint))
    baseline_timesteps = int(baseline_model.num_timesteps)
    assert baseline_timesteps > 0

    manifest = run_defender_adaptation_training(
        architecture="ppo_clutter_v1",
        generation="d0_plus",
        env_fn=lambda worker_seed: build_ppo_clutter_env("measurement"),
        seed=71,
        starting_checkpoint_path=starting_checkpoint,
        output_dir=tmp_path / "continuation_basic",
        total_additional_timesteps=2048,
        eval_freq=2048,
        eval_episodes=2,
        checkpoint_freq=2048,
        num_envs=1,
        verbose=0,
    )
    assert manifest["starting_num_timesteps"] == baseline_timesteps
    assert manifest["model_num_timesteps"] > baseline_timesteps
    assert manifest["actual_additional_timesteps"] >= 2048
    assert manifest["resumed_from_interrupt"] is False


def test_resume_after_simulated_interrupt_does_not_exceed_budget(tmp_path):
    """Simulates an interruption by running with a small budget, deleting
    the final-completion artifacts but keeping the mid-run resume state,
    then re-invoking with the SAME (larger) target budget -- the total
    additional transitions actually trained must never exceed the
    requested target by more than one rollout's worth, and must be
    STRICTLY MORE than the first partial run alone (proving real
    resumption, not a restart from the D0 checkpoint)."""
    starting_checkpoint = _d0_checkpoint()
    output_dir = tmp_path / "resume_case"

    first_pass = run_defender_adaptation_training(
        architecture="ppo_clutter_v1",
        generation="d0_plus",
        env_fn=lambda worker_seed: build_ppo_clutter_env("measurement"),
        seed=71,
        starting_checkpoint_path=starting_checkpoint,
        output_dir=output_dir,
        total_additional_timesteps=2048,
        eval_freq=2048,
        eval_episodes=2,
        checkpoint_freq=2048,
        num_envs=1,
        verbose=0,
    )
    first_pass_completed = first_pass["actual_additional_timesteps"]

    # Simulate the process being interrupted BEFORE final completion by
    # reverting the resume manifest's "completed" flag to False (the
    # on-disk resume_latest.zip is otherwise untouched -- exactly the
    # state a real interruption mid-loop would leave behind).
    resume_manifest_path = output_dir / "checkpoints" / "resume_manifest.json"
    manifest_data = json.loads(resume_manifest_path.read_text())
    manifest_data["completed"] = False
    resume_manifest_path.write_text(json.dumps(manifest_data))

    second_pass = run_defender_adaptation_training(
        architecture="ppo_clutter_v1",
        generation="d0_plus",
        env_fn=lambda worker_seed: build_ppo_clutter_env("measurement"),
        seed=71,
        starting_checkpoint_path=starting_checkpoint,
        output_dir=output_dir,
        total_additional_timesteps=4096,
        eval_freq=2048,
        eval_episodes=2,
        checkpoint_freq=2048,
        num_envs=1,
        verbose=0,
    )
    assert second_pass["resumed_from_interrupt"] is True
    assert second_pass["starting_num_timesteps"] == first_pass["starting_num_timesteps"]
    assert second_pass["actual_additional_timesteps"] > first_pass_completed
    # Rollout width for num_envs=1 is n_steps=2048 -- overshoot past the
    # 4096 target is bounded by one rollout.
    assert second_pass["actual_additional_timesteps"] <= 4096 + 2048


def test_resume_rejects_mismatched_starting_checkpoint_lineage(tmp_path):
    output_dir = tmp_path / "lineage_mismatch"
    run_defender_adaptation_training(
        architecture="ppo_clutter_v1",
        generation="d0_plus",
        env_fn=lambda worker_seed: build_ppo_clutter_env("measurement"),
        seed=71,
        starting_checkpoint_path=_clutter_checkpoint_path("ppo_clutter_v1", 71),
        output_dir=output_dir,
        total_additional_timesteps=2048,
        eval_freq=2048,
        eval_episodes=2,
        checkpoint_freq=2048,
        num_envs=1,
        verbose=0,
    )
    resume_manifest_path = output_dir / "checkpoints" / "resume_manifest.json"
    manifest_data = json.loads(resume_manifest_path.read_text())
    manifest_data["completed"] = False
    resume_manifest_path.write_text(json.dumps(manifest_data))

    with pytest.raises(RuntimeError, match="mismatched lineage"):
        run_defender_adaptation_training(
            architecture="ppo_clutter_v1",
            generation="d0_plus",
            env_fn=lambda worker_seed: build_ppo_clutter_env("measurement"),
            seed=72,  # DIFFERENT seed -> different starting checkpoint
            starting_checkpoint_path=_clutter_checkpoint_path("ppo_clutter_v1", 72),
            output_dir=output_dir,
            total_additional_timesteps=2048,
            eval_freq=2048,
            eval_episodes=2,
            checkpoint_freq=2048,
            num_envs=1,
            verbose=0,
        )


def test_d0_plus_and_d1_matched_pair_identical_actual_transitions(tmp_path):
    """Phase 18: a matched D0+/D1 pair sharing the SAME starting checkpoint
    and target budget must train for the IDENTICAL number of actual
    additional transitions (same rollout arithmetic), differing only in
    opponent distribution."""
    from experiments.defender_adaptation_env import build_ppo_d1_env

    starting_checkpoint = _d0_checkpoint()
    d0_plus_manifest = run_defender_adaptation_training(
        architecture="ppo_clutter_v1",
        generation="d0_plus",
        env_fn=lambda worker_seed: build_ppo_clutter_env("measurement"),
        seed=71,
        starting_checkpoint_path=starting_checkpoint,
        output_dir=tmp_path / "matched_pair" / "d0_plus",
        total_additional_timesteps=2048,
        eval_freq=2048,
        eval_episodes=2,
        checkpoint_freq=2048,
        num_envs=1,
        verbose=0,
    )
    pool_spec = opponent_pool_spec(build_opponent_pool())
    d1_manifest = run_defender_adaptation_training(
        architecture="ppo_clutter_v1",
        generation="d1",
        env_fn=lambda worker_seed: build_ppo_d1_env("measurement"),
        seed=71,
        starting_checkpoint_path=starting_checkpoint,
        output_dir=tmp_path / "matched_pair" / "d1",
        total_additional_timesteps=2048,
        opponent_pool_spec_dict=pool_spec,
        eval_freq=2048,
        eval_episodes=2,
        checkpoint_freq=2048,
        num_envs=1,
        verbose=0,
    )
    assert d0_plus_manifest["starting_num_timesteps"] == d1_manifest["starting_num_timesteps"]
    assert d0_plus_manifest["actual_additional_timesteps"] == d1_manifest["actual_additional_timesteps"]


def test_d0_checkpoint_never_modified_by_continuation_run(tmp_path):
    starting_checkpoint = _d0_checkpoint()
    before_hash = _sha256_file(starting_checkpoint)
    run_defender_adaptation_training(
        architecture="ppo_clutter_v1",
        generation="d0_plus",
        env_fn=lambda worker_seed: build_ppo_clutter_env("measurement"),
        seed=71,
        starting_checkpoint_path=starting_checkpoint,
        output_dir=tmp_path / "no_overwrite_check",
        total_additional_timesteps=2048,
        eval_freq=2048,
        eval_episodes=2,
        checkpoint_freq=2048,
        num_envs=1,
        verbose=0,
    )
    after_hash = _sha256_file(starting_checkpoint)
    assert before_hash == after_hash
