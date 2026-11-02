"""Stage runner for curriculum Stage 2 (static-defender evasion). Safe to re-run after ANY interruption: verified stages are skipped, an interrupted seed resumes from its highest verified rollout.

    python -m experiments.journal_v2_a1_cur2_run all        # stages 1-9 (final evaluation / visualizer / documentation / commit are separate steps)
    python -m experiments.journal_v2_a1_cur2_run status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur2_state as TS
from experiments import journal_v2_a1_cur2_training as T2
from experiments import journal_v2_a1_r2 as R

DOCS = "docs/journal_v2_a1_curriculum/stage2/"
EXPECTED_START_HEAD = "dac0230"


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(TS.PROJECT_ROOT.resolve())).replace("\\", "/")


def parent_hashes() -> dict:
    return {str(s): {"attacker_id": S.BC_SEEDS[s], "model_sha256": T2.bc_parent(s)["model_sha256"]} for s in S.PPO_SEEDS}


def checkpoint_integrity() -> dict:
    import hashlib as _h

    registry = json.loads((TS.PROJECT_ROOT / "results" / "defender_adaptation_model_registry.json").read_text())
    entries = list(registry["defenders"].values()) + list(registry["attackers"].values())
    bad = [e["path"] for e in entries if _h.sha256((Path(e["path"]) if Path(e["path"]).is_absolute() else TS.PROJECT_ROOT / e["path"]).read_bytes()).hexdigest() != e["sha256"]]
    failed_v1 = json.loads((R.FAILED_V1_RESULTS_ROOT / "a1_v2_model_registry.json").read_text())
    fbad = [n for n, e in failed_v1["attackers"].items() if _h.sha256((TS.PROJECT_ROOT / e["path"]).read_bytes()).hexdigest() != e["sha256"]]
    stage1v1 = {}
    for s in C.PPO_SEEDS:
        m = json.loads((C.RESULTS_ROOT / f"seed_{s}" / "checkpoints" / "seed_manifest.json").read_text())
        stage1v1[str(s)] = {"status": m["status"], "sha256_matches": TS.sha256_file(C.RESULTS_ROOT / f"seed_{s}" / "checkpoints" / "accepted_model.zip") == m["accepted_model_sha256"],
                            "marker_present": (C.RESULTS_ROOT / f"seed_{s}" / "checkpoints" / "NOT_ACCEPTED_stage1_failed.txt").exists()}
    bc = {str(s): {"sha256_matches_handoff": True, "model_sha256": T2.bc_parent(s)["model_sha256"]} for s in S.PPO_SEEDS}   # bc_parent raises on mismatch
    rec = {"historical_registered_checkpoints": len(entries), "historical_mismatches": bad, "failed_a1_v1_checkpoints": len(failed_v1["attackers"]), "failed_a1_v1_mismatches": fbad,
           "stage1_v1_failed_curriculum_models": stage1v1, "stage1_v2_bc_models": bc}
    assert not bad and not fbad and len(entries) == 30 and len(failed_v1["attackers"]) == 3
    assert all(v["status"] == "FAIL" and v["sha256_matches"] and v["marker_present"] for v in stage1v1.values())
    return rec


def verify_repository() -> dict:
    head = TS.git("rev-parse", "HEAD")
    tag = TS.git("rev-list", "-n", "1", S.STAGE1_V2_TAG)
    handoff = json.loads((TS.PROJECT_ROOT / S.STAGE1_V2_HANDOFF).read_text())
    frozen = ("experiments/journal_v2_a1.py", "experiments/journal_v2_a1_r2.py", "experiments/journal_v2_a1_cur1.py", "experiments/journal_v2_a1_cur1v2.py", "experiments/journal_v2_a1_cur1v2_bc.py", "uav_defend")
    schema = json.dumps(A.A1_V2_OBSERVATION_SCHEMA).encode()
    rec = {"head": head, "expected_start_head": EXPECTED_START_HEAD, "head_is_expected_or_descendant": subprocess.run(["git", "merge-base", "--is-ancestor", EXPECTED_START_HEAD, "HEAD"], cwd=TS.PROJECT_ROOT).returncode == 0,
           "branch": TS.git("rev-parse", "--abbrev-ref", "HEAD"), "stage1_tag": S.STAGE1_V2_TAG, "stage1_tag_commit": tag, "stage1_tag_is_ancestor": subprocess.run(["git", "merge-base", "--is-ancestor", tag, "HEAD"], cwd=TS.PROJECT_ROOT).returncode == 0,
           "tracked_tree_clean": TS.git("status", "--porcelain", "--untracked-files=no") == "",
           "a0_tag_commit": TS.git("rev-list", "-n", "1", "journal-v2-a0-canonical-v1"), "uav_defend_identical_to_a0_tag": TS.git("diff", "--stat", "journal-v2-a0-canonical-v1", "--", "uav_defend") == "",
           "frozen_files_identical_to_stage1_tag": TS.git("diff", "--stat", S.STAGE1_V2_TAG, "--", *frozen) == "", "frozen_files": list(frozen),
           "handoff_stage1_v2_pass": bool(handoff["decision"]["stage1_v2_pass"]), "handoff_versions": handoff["versions"],
           "versions_unchanged": handoff["versions"] == {"action": S.ACTION_VERSION, "observation": S.OBSERVATION_VERSION, "reward": S.REWARD_VERSION, "bc_protocol": "a1_cur1v2_bc_v1"},
           "reward_version": R.REWARD_VERSION, "observation_dim": A.A1_V2_OBSERVATION_DIM, "observation_schema_sha256": hashlib.sha256(schema).hexdigest(), "action_dim": A.A1_V2_ACTION_DIM,
           "ppo_hyperparameters": A.A1_V2_PPO_HYPERPARAMS, "n_envs": A.A1_V2_NUM_ENVS, "bc_checkpoints": parent_hashes(), "integrity": checkpoint_integrity()}
    assert rec["head_is_expected_or_descendant"] and rec["stage1_tag_commit"].startswith(EXPECTED_START_HEAD) and rec["tracked_tree_clean"] and rec["uav_defend_identical_to_a0_tag"]
    assert rec["frozen_files_identical_to_stage1_tag"] and rec["handoff_stage1_v2_pass"] and rec["versions_unchanged"] and rec["reward_version"] == "a1_v2_reward_v2"
    assert rec["observation_dim"] == 37 and rec["action_dim"] == 3 and rec["ppo_hyperparameters"]["n_steps"] == 2048 and rec["n_envs"] == 16
    return rec


def scenario_validation_record() -> dict:
    full = S.scenario_validation()
    again = S.evaluate(None, list(range(S.PLACEMENT_VALIDATION[0], S.PLACEMENT_VALIDATION[0] + 50)), scripted_a0=True)
    first = S.evaluate(None, list(range(S.PLACEMENT_VALIDATION[0], S.PLACEMENT_VALIDATION[0] + 50)), scripted_a0=True)
    deterministic = all(a["defender_position"] == b["defender_position"] and a["outcome"] == b["outcome"] and a["steps"] == b["steps"] for a, b in zip(first, again))
    ok = full["a0_interception_fraction"] >= S.A0_INTERCEPTION_MIN
    return {"scenario_seeds": list(S.PLACEMENT_VALIDATION), "placement_protocol": S.PLACEMENT_PROTOCOL, "summary": full, "deterministic_replay_first_50": bool(deterministic),
            "stop_rule": f"A0 interception fraction >= {S.A0_INTERCEPTION_MIN}", "stop_rule_passed": bool(ok),
            "superseded_design": "docs/journal_v2_a1_curriculum/stage2/scenario_validation_initial_STOP.json (initial-chord placement: A0 interception 51.5%)"}


def run_all() -> None:
    state = TS.load_state()
    parent = parent_hashes()
    psha = S.protocol_sha256(parent)
    scratch = T2.select_scratch_root()
    print(f"[stage2] scratch={scratch} protocol_sha={psha[:16]}", flush=True)
    if not TS.stage_is_verified(state, "01_repository_verification"):
        TS.begin(state, "01_repository_verification")
        TS.atomic_write_json(TS.STATE_DIR / "repository_verification.json", verify_repository())
        TS.complete(state, "01_repository_verification", [DOCS + "repository_verification.json"], "HEAD dac0230, clean, A0/versions/BC hashes verified")
    if not TS.stage_is_verified(state, "02_stage2_protocol_freeze", psha):
        TS.begin(state, "02_stage2_protocol_freeze", psha)
        TS.atomic_write_json(TS.STATE_DIR / "protocol.json", S.protocol_specification(parent) | {"protocol_sha256": psha})
        TS.complete(state, "02_stage2_protocol_freeze", [DOCS + "protocol.json"], "protocol sha " + psha[:16])
    code = ["experiments/journal_v2_a1_cur2.py", "experiments/journal_v2_a1_cur2_training.py", "experiments/journal_v2_a1_cur2_state.py", "experiments/journal_v2_a1_cur2_run.py"]
    if not TS.stage_is_verified(state, "03_environment_implementation", psha):
        TS.begin(state, "03_environment_implementation", psha)
        TS.complete(state, "03_environment_implementation", code, "Stage-2 env + runner (tests/test_journal_v2_a1_cur2.py)")
    if not TS.stage_is_verified(state, "04_deterministic_scenario_validation", psha):
        TS.begin(state, "04_deterministic_scenario_validation", psha)
        record = scenario_validation_record()
        TS.atomic_write_json(TS.STATE_DIR / "scenario_validation.json", record)
        if not (record["stop_rule_passed"] and record["deterministic_replay_first_50"]):
            TS.fail(state, "04_deterministic_scenario_validation", "A0 interception below 70% or non-deterministic placement: STOP", [DOCS + "scenario_validation.json"])
            raise SystemExit("STOP: scenario validation failed")
        TS.complete(state, "04_deterministic_scenario_validation", [DOCS + "scenario_validation.json"], f"A0 interception {record['summary']['a0_interception_fraction']:.3f}")
    test_path = TS.STATE_DIR / "unit_test_result.json"
    if not TS.stage_is_verified(state, "05_tests", psha):
        if not test_path.exists() or json.loads(test_path.read_text()).get("failed", 1) != 0:
            raise SystemExit("run the targeted tests first and write unit_test_result.json (failed == 0)")
        TS.begin(state, "05_tests", psha)
        TS.complete(state, "05_tests", [DOCS + "unit_test_result.json"], "targeted tests passed")
    if not TS.stage_is_verified(state, "06_real_sb3_resume_verification", psha):
        TS.begin(state, "06_real_sb3_resume_verification", psha)
        result = T2.resume_test(scratch.parent / "stage2_resume_test")
        TS.atomic_write_json(TS.STATE_DIR / "resume_validation.json", result)
        if not (result["all_states_equal"] and result["all_episode_logs_equal"] and result["init_record_equal"] and result["crash_injected_between_model_write_and_manifest"]):
            TS.fail(state, "06_real_sb3_resume_verification", "interrupted run differs from uninterrupted run", [DOCS + "resume_validation.json"])
            raise SystemExit("STOP: real-SB3 resume validation failed")
        TS.complete(state, "06_real_sb3_resume_verification", [DOCS + "resume_validation.json"], "interrupted == uninterrupted (policy+optimizer state hashes)")
    for stage_name, seed in (("07_seed_201_ppo", 201), ("08_seed_202_ppo", 202), ("09_seed_203_ppo", 203)):
        if TS.stage_is_verified(state, stage_name, psha):
            continue
        TS.begin(state, stage_name, psha)
        result = T2.train_seed(seed, scratch_root=scratch)
        if result.get("status") == "INCOMPLETE":
            raise SystemExit(f"seed {seed} incomplete")
        path = TS.STATE_DIR / f"seed_{seed}_manifest.json"
        TS.atomic_write_json(path, result)
        TS.complete(state, stage_name, [DOCS + path.name], f"seed {seed}: {result['status']} rollouts={result['rollouts_trained']} mastery_rollout={result['mastery_rollout']}")
    print(json.dumps(T2.status(scratch), indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "status"))
    args = parser.parse_args(argv)
    if args.command == "all":
        run_all()
    else:
        s = TS.load_state()
        print(json.dumps({k: v["status"] for k, v in s["stages"].items()} | {"next": TS.next_stage(s)}, indent=1))


if __name__ == "__main__":
    main()
