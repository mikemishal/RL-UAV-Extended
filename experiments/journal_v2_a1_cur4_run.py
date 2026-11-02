"""Stage runner for curriculum Stage 4 (moving canonical asset + hold-until-detection Greedy). Safe to re-run after ANY interruption: verified stages are skipped, an interrupted seed resumes from its highest verified rollout.

    python -m experiments.journal_v2_a1_cur4_run resume-test   # stage 6 evidence (real SB3, interrupted vs uninterrupted, injected crash)
    python -m experiments.journal_v2_a1_cur4_run all           # stages 1-9: parent verification, protocol, config, scenario validation, tests gate, resume gate, PPO training of seeds 201/202/203
    python -m experiments.journal_v2_a1_cur4_run status
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur2_run as RUN2
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur3v2_run as RUN3
from experiments import journal_v2_a1_cur4 as Y
from experiments import journal_v2_a1_cur4_state as TS
from experiments import journal_v2_a1_cur4_training as T4

DOCS = Y.DOCS
START_HEAD = "3a2b23f"
FROZEN = RUN3.FROZEN + ("experiments/journal_v2_a1_cur3.py", "experiments/journal_v2_a1_cur3v2.py", "experiments/journal_v2_a1_cur3v2_training.py", "experiments/journal_v2_a1_cur3v2_final.py",
                        "experiments/journal_v2_a1_cur3v2_run.py", "experiments/journal_v2_a1_cur3v2_state.py")


def stage3_integrity() -> dict:
    out = {}
    for s in Y.PPO_SEEDS:
        m = json.loads((X.RESULTS_ROOT / f"seed_{s}" / "checkpoints" / "seed_manifest.json").read_text())
        rolls = {r["rollout"]: TS.sha256_file(X.RESULTS_ROOT / f"seed_{s}" / "rollouts" / f"rollout_{r['rollout']:02d}.zip") == r["model_sha256"] for r in m["rollouts"]}
        out[f"stage3_v2_seed_{s}"] = {"status": m["status"], "accepted_sha256_matches_preregistered": TS.sha256_file(X.RESULTS_ROOT / f"seed_{s}" / "checkpoints" / "accepted_model.zip") == Y.PARENT_SHA256[s],
                                      "all_rollouts_match": all(rolls.values())}
    assert all(v["accepted_sha256_matches_preregistered"] and v["all_rollouts_match"] for v in out.values())
    return out


def verify_repository() -> dict:
    head = TS.git("rev-parse", "HEAD")
    v3_state = json.loads((TS.PROJECT_ROOT / X.DOCS / "task_state.json").read_text())
    v3_handoff = json.loads((TS.PROJECT_ROOT / X.DOCS / "stage4_handoff.json").read_text())
    rec = {"head": head, "expected_start_head": START_HEAD, "head_is_expected_or_descendant": subprocess.run(["git", "merge-base", "--is-ancestor", START_HEAD, "HEAD"], cwd=TS.PROJECT_ROOT).returncode == 0,
           "branch": TS.git("rev-parse", "--abbrev-ref", "HEAD"), "tracked_tree_clean": TS.git("status", "--porcelain", "--untracked-files=no") == "",
           "untracked_at_start": [l[3:] for l in TS.git("status", "--porcelain", "--untracked-files=all").splitlines() if l.startswith("??")],
           "uav_defend_identical_to_a0_tag": TS.git("diff", "--stat", "journal-v2-a0-canonical-v1", "--", "uav_defend") == "",
           "frozen_files_identical_to_start_head": TS.git("diff", "--stat", START_HEAD, "--", *FROZEN) == "", "frozen_files": list(FROZEN),
           "stage3_v2": {"overall_status": v3_state.get("overall_status"), "handoff_status": v3_handoff["status"], "protocol_sha256": json.loads((TS.PROJECT_ROOT / X.DOCS / "protocol.json").read_text())["protocol_sha256"],
                         "manual_review_acceptance_recorded": False,
                         "recorded_as": Y.PARENT_STATUS, "note": "no manual-review acceptance is recorded in the repository; Stage3-v2 is recorded as a quantitatively accepted curriculum parent; earlier mastery results are unchanged"},
           "stage3_v2_parents": {str(s): Y.parent_checkpoint(s) for s in Y.PPO_SEEDS}, "parent_status": Y.PARENT_STATUS, "detection_radius_m": X.detection_radius(),
           "asset_motion": Y.asset_motion_parameters(), "asset_motion_check_vs_canonical": Y.canonical_asset_motion_check(), "reward_version": S.REWARD_VERSION, "observation_version": S.OBSERVATION_VERSION,
           "action_version": S.ACTION_VERSION, "ppo_hyperparameters": A.A1_V2_PPO_HYPERPARAMS, "integrity": RUN2.checkpoint_integrity(), "stage2_integrity": RUN3.stage2_integrity(), "stage3_integrity": stage3_integrity(),
           "namespace_overlaps": Y.namespace_overlaps()}
    assert rec["head_is_expected_or_descendant"] and rec["tracked_tree_clean"] and rec["uav_defend_identical_to_a0_tag"] and rec["frozen_files_identical_to_start_head"]
    assert rec["detection_radius_m"] == 15.0 and rec["asset_motion_check_vs_canonical"]["identical"] and rec["reward_version"] == "a1_v2_reward_v2" and rec["ppo_hyperparameters"]["n_steps"] == 2048
    assert not any(rec["namespace_overlaps"][k] for k in ("validation_A", "validation_B", "anchor1", "anchor2", "anchor3", "diagnostic", "scenario_validation", "mutual", "training_overlap"))
    return rec


def scenario_validation_record() -> dict:
    from stable_baselines3 import PPO

    from experiments import journal_v2_a1_cur1 as C

    summary = Y.scenario_validation()
    seeds = list(range(Y.SCENARIO_VALIDATION[0], Y.SCENARIO_VALIDATION[0] + 50))
    policies = {"a0": None} | {f"parent_{s}": C.model_actor(PPO.load(Y.parent_checkpoint(s)["path"].removesuffix(".zip"), device="cpu")) for s in Y.PPO_SEEDS}
    summary["crn"] = Y.crn_report(Y.asset_traces(seeds, policies))
    verdict = Y.validation_verdict(summary)
    return {"scenario_protocol": Y.SCENARIO_PROTOCOL, "seeds": list(Y.SCENARIO_VALIDATION), "detection_radius_m": X.detection_radius(), "asset_motion": Y.asset_motion_parameters(), "summary": summary,
            "verdict": verdict, "protocol_sha256": Y.protocol_sha256()}


def run_all() -> None:
    state = TS.load_state()
    psha = Y.protocol_sha256()
    scratch = T4.select_scratch_root()
    print(f"[stage4] scratch={scratch} protocol_sha={psha[:16]}", flush=True)
    if not TS.stage_is_verified(state, "01_verify_stage3_parent"):
        TS.begin(state, "01_verify_stage3_parent")
        TS.atomic_write_json(TS.STATE_DIR / "repository_verification.json", verify_repository())
        TS.complete(state, "01_verify_stage3_parent", [DOCS + "repository_verification.json"], "HEAD 3a2b23f, clean, Stage3-v2 parents (full SHA-256) + integrity verified; recorded as quantitatively accepted curriculum parent")
    if not TS.stage_is_verified(state, "02_stage4_protocol_freeze", psha):
        TS.begin(state, "02_stage4_protocol_freeze", psha)
        TS.atomic_write_json(TS.STATE_DIR / "protocol.json", Y.protocol_specification() | {"protocol_sha256": psha})
        TS.complete(state, "02_stage4_protocol_freeze", [DOCS + "protocol.json"], "protocol sha " + psha[:16])
    code = ["experiments/journal_v2_a1_cur4.py", "experiments/journal_v2_a1_cur4_training.py", "experiments/journal_v2_a1_cur4_state.py", "experiments/journal_v2_a1_cur4_run.py"]
    if not TS.stage_is_verified(state, "03_moving_asset_curriculum_config", psha):
        TS.begin(state, "03_moving_asset_curriculum_config", psha)
        TS.complete(state, "03_moving_asset_curriculum_config", code, "Stage4Env (canonical legacy_gaussian_random_walk asset) + runner (tests/test_journal_v2_a1_cur4.py)")
    if not TS.stage_is_verified(state, "04_scenario_validation", psha):
        TS.begin(state, "04_scenario_validation", psha)
        rec = scenario_validation_record()
        TS.atomic_write_json(TS.STATE_DIR / "scenario_validation.json", rec)
        artifacts = [DOCS + "scenario_validation.json"]
        if not rec["verdict"]["passed"]:
            TS.fail(state, "04_scenario_validation", "STAGE4 SCENARIO VALIDATION FAILED - DO NOT TRAIN (verdict D): " + json.dumps(rec["verdict"]), artifacts)
            raise SystemExit("scenario validation failed: verdict D, no training")
        TS.complete(state, "04_scenario_validation", artifacts, f"A0 capture {rec['summary']['a0']['capture']:.3f} / interception {rec['summary']['a0']['interception']:.3f}; parents capture {rec['verdict']['parent_capture']}")
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
        TS.atomic_write_json(path, result | {"stage4_protocol_sha256": psha})
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
        result = T4.resume_test(scratch.parent / "stage4_resume_test")
        TS.atomic_write_json(TS.STATE_DIR / "resume_validation.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    else:
        T4.T.keep_awake()
        run_all()


if __name__ == "__main__":
    main()
