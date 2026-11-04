"""Stage runner for curriculum Stage 4A (moving canonical asset, NO active defender). Safe to re-run after ANY interruption: verified stages are skipped, an interrupted seed resumes from its highest verified rollout.

    python -m experiments.journal_v2_a1_cur4a_run resume-test   # stage 6 evidence (real SB3, interrupted vs uninterrupted, injected crash)
    python -m experiments.journal_v2_a1_cur4a_run all           # stages 1-9: verification, protocol, environment, baseline, tests gate, resume gate, PPO training of seeds 201/202/203
    python -m experiments.journal_v2_a1_cur4a_run status
"""

from __future__ import annotations

import argparse
import json
import subprocess

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur2_run as RUN2
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur3v2_run as RUN3
from experiments import journal_v2_a1_cur4 as P
from experiments import journal_v2_a1_cur4_run as RUN4
from experiments import journal_v2_a1_cur4a as Y
from experiments import journal_v2_a1_cur4a_state as TS
from experiments import journal_v2_a1_cur4a_training as T4

DOCS = Y.DOCS
START_HEAD = "15faece"
FROZEN = RUN4.FROZEN + ("experiments/journal_v2_a1_cur4.py", "experiments/journal_v2_a1_cur4_training.py", "experiments/journal_v2_a1_cur4_final.py", "experiments/journal_v2_a1_cur4_run.py",
                        "experiments/journal_v2_a1_cur4_state.py", "tests/test_journal_v2_a1_cur4.py")


def stage4_integrity() -> dict:
    """The failed Stage-4 checkpoints are preserved byte-for-byte as non-canonical evidence."""
    out = {}
    for s in Y.PPO_SEEDS:
        root = P.RESULTS_ROOT / f"seed_{s}"
        m = json.loads((root / "checkpoints" / "seed_manifest.json").read_text())
        rolls = {r["rollout"]: TS.sha256_file(root / "rollouts" / f"rollout_{r['rollout']:02d}.zip") == r["model_sha256"] for r in m["rollouts"]}
        out[f"stage4_seed_{s}"] = {"status": m["status"], "accepted_sha256": TS.sha256_file(root / "checkpoints" / "accepted_model.zip"),
                                   "matches_recorded_final_hash": TS.sha256_file(root / "checkpoints" / "accepted_model.zip") == Y.STAGE4_ACCEPTED_FINAL_SHA256[s], "all_rollouts_match": all(rolls.values())}
    assert all(v["status"] == "FAIL" and v["matches_recorded_final_hash"] and v["all_rollouts_match"] for v in out.values())
    return out


def verify_repository() -> dict:
    head = TS.git("rev-parse", "HEAD")
    decision = json.loads((TS.PROJECT_ROOT / P.DOCS / "stage4_decision.json").read_text())
    rec = {"head": head, "expected_start_head": START_HEAD, "head_is_expected_or_descendant": subprocess.run(["git", "merge-base", "--is-ancestor", START_HEAD, "HEAD"], cwd=TS.PROJECT_ROOT).returncode == 0,
           "branch": TS.git("rev-parse", "--abbrev-ref", "HEAD"), "tracked_tree_clean": TS.git("status", "--porcelain", "--untracked-files=no") == "",
           "untracked_at_start": [l[3:] for l in TS.git("status", "--porcelain", "--untracked-files=all").splitlines() if l.startswith("??")],
           "uav_defend_identical_to_a0_tag": TS.git("diff", "--stat", "journal-v2-a0-canonical-v1", "--", "uav_defend") == "",
           "frozen_files_identical_to_start_head": TS.git("diff", "--stat", START_HEAD, "--", *FROZEN) == "", "frozen_files": list(FROZEN),
           "stage4": {"decision": decision, "status": "failed Stage 4 preserved as non-canonical evidence; its checkpoints are never used for Stage 4A"},
           "stage3_v2_parents": {str(s): Y.parent_checkpoint(s) for s in Y.PPO_SEEDS}, "parent_status": Y.PARENT_STATUS, "asset_motion": Y.asset_motion_parameters(),
           "asset_motion_check_vs_canonical": Y.canonical_asset_motion_check(), "config_differences_from_canonical": Y.config_differences_from_canonical(), "reward_version": S.REWARD_VERSION,
           "observation_version": S.OBSERVATION_VERSION, "action_version": S.ACTION_VERSION, "ppo_hyperparameters": A.A1_V2_PPO_HYPERPARAMS, "integrity": RUN2.checkpoint_integrity(),
           "stage2_integrity": RUN3.stage2_integrity(), "stage3_integrity": RUN4.stage3_integrity(), "stage4_integrity": stage4_integrity(), "namespace_overlaps": Y.namespace_overlaps()}
    assert rec["head_is_expected_or_descendant"] and rec["tracked_tree_clean"] and rec["uav_defend_identical_to_a0_tag"] and rec["frozen_files_identical_to_start_head"]
    assert rec["asset_motion_check_vs_canonical"]["identical"] and rec["reward_version"] == "a1_v2_reward_v2" and rec["ppo_hyperparameters"]["n_steps"] == 2048
    assert set(rec["config_differences_from_canonical"]) == {"intercept_radius", "defender_standby_until_detection"}
    assert not any(rec["namespace_overlaps"][k] for k in ("validation_A", "validation_B", "anchor1", "anchor2", "anchor3", "baseline", "mutual", "training_overlap"))
    return rec


def baseline_record() -> dict:
    summary = Y.baseline_validation()
    verdict = Y.baseline_verdict(summary)
    return {"scenario_protocol": Y.SCENARIO_PROTOCOL, "seeds": list(Y.BASELINE), "asset_motion": Y.asset_motion_parameters(), "summary": summary, "verdict": verdict, "protocol_sha256": Y.protocol_sha256()}


def run_all() -> None:
    state = TS.load_state()
    psha = Y.protocol_sha256()
    scratch = T4.select_scratch_root()
    print(f"[stage4a] scratch={scratch} protocol_sha={psha[:16]}", flush=True)
    if not TS.stage_is_verified(state, "01_repository_verification"):
        TS.begin(state, "01_repository_verification")
        TS.atomic_write_json(TS.STATE_DIR / "repository_verification.json", verify_repository())
        TS.complete(state, "01_repository_verification", [DOCS + "repository_verification.json"], "HEAD 15faece, tracked tree clean, Stage3-v2 parents + Stage1-4 checkpoint integrity verified")
    if not TS.stage_is_verified(state, "02_stage4a_protocol_freeze", psha):
        TS.begin(state, "02_stage4a_protocol_freeze", psha)
        TS.atomic_write_json(TS.STATE_DIR / "protocol.json", Y.protocol_specification() | {"protocol_sha256": psha})
        TS.complete(state, "02_stage4a_protocol_freeze", [DOCS + "protocol.json"], "protocol sha " + psha[:16])
    code = ["experiments/journal_v2_a1_cur4a.py", "experiments/journal_v2_a1_cur4a_training.py", "experiments/journal_v2_a1_cur4a_state.py", "experiments/journal_v2_a1_cur4a_run.py"]
    if not TS.stage_is_verified(state, "03_environment_implementation", psha):
        TS.begin(state, "03_environment_implementation", psha)
        TS.complete(state, "03_environment_implementation", code, "Stage4AEnv (canonical legacy_gaussian_random_walk asset, masked parked defender) + runner (tests/test_journal_v2_a1_cur4a.py)")
    if not TS.stage_is_verified(state, "04_baseline_validation", psha):
        TS.begin(state, "04_baseline_validation", psha)
        rec = baseline_record()
        TS.atomic_write_json(TS.STATE_DIR / "baseline_validation.json", rec)
        artifacts = [DOCS + "baseline_validation.json"]
        if not rec["verdict"]["passed"]:
            TS.fail(state, "04_baseline_validation", "STAGE4A CONFIGURATION VALIDATION FAILED - DO NOT TRAIN: " + json.dumps(rec["verdict"]), artifacts)
            raise SystemExit("baseline configuration validation failed: no training")
        TS.complete(state, "04_baseline_validation", artifacts, f"A0 capture {rec['summary']['a0']['capture']:.3f}; parents capture {rec['verdict']['parent_capture']}")
    test_path = TS.STATE_DIR / "unit_test_result.json"
    if not TS.stage_is_verified(state, "05_targeted_tests", psha):
        if not test_path.exists() or json.loads(test_path.read_text()).get("failed", 1) != 0:
            raise SystemExit("run the targeted tests first and write unit_test_result.json (failed == 0)")
        TS.begin(state, "05_targeted_tests", psha)
        TS.complete(state, "05_targeted_tests", [DOCS + "unit_test_result.json"], "targeted tests passed")
    resume_path = TS.STATE_DIR / "resume_validation.json"
    if not TS.stage_is_verified(state, "06_sb3_resume_validation", psha):
        if not resume_path.exists():
            raise SystemExit("run `resume-test` first (writes resume_validation.json)")
        res = json.loads(resume_path.read_text())
        if not (res["all_states_equal"] and res["all_episode_logs_equal"] and res["init_record_equal"] and res["crash_injected_between_model_write_and_manifest"] and res["protocol_sha256"] == psha):
            raise SystemExit("real-SB3 resume validation FAILED - do not train")
        TS.begin(state, "06_sb3_resume_validation", psha)
        TS.complete(state, "06_sb3_resume_validation", [DOCS + "resume_validation.json"], "interrupted == uninterrupted (state hashes, episode logs) incl. injected crash")
    for stage_name, seed in (("07_seed_201_training", 201), ("08_seed_202_training", 202), ("09_seed_203_training", 203)):
        if TS.stage_is_verified(state, stage_name, psha):
            continue
        TS.begin(state, stage_name, psha)
        parent = Y.parent_checkpoint(seed)
        result = T4.train_seed(seed, scratch_root=scratch, parent=parent)
        if result.get("status") == "INCOMPLETE":
            raise SystemExit(f"seed {seed} incomplete")
        path = TS.STATE_DIR / f"seed_{seed}_manifest.json"
        TS.atomic_write_json(path, result | {"stage4a_protocol_sha256": psha})
        last = result["rollouts"][-1]
        TS.complete(state, stage_name, [DOCS + path.name], f"seed {seed}: {result['status']} rollouts={result['rollouts_trained']} mastery_rollout={result['mastery_rollout']}",
                    {"parent_sha256": parent["model_sha256"], "completed_rollout": result["rollouts_trained"], "final_model_sha256": last["model_sha256"], "final_state_sha256": last["state_sha256"],
                     "accepted_model_sha256": result["accepted_model_sha256"]})
    print(json.dumps(T4.status(scratch), indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "status", "resume-test"))
    args = parser.parse_args(argv)
    TS.STATE_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        print(json.dumps(T4.status(), indent=1))
    elif args.command == "resume-test":
        T4.T.keep_awake()
        scratch = T4.select_scratch_root()
        result = T4.resume_test(scratch.parent / "stage4a_resume_test")
        TS.atomic_write_json(TS.STATE_DIR / "resume_validation.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    else:
        T4.T.keep_awake()
        run_all()


if __name__ == "__main__":
    main()
