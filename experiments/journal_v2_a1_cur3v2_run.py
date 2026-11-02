"""Stage runner for curriculum Stage3-v2 (moving Greedy defender, hold-until-detection). Safe to re-run after ANY interruption: verified stages are skipped, an interrupted seed resumes from its highest
verified rollout.

    python -m experiments.journal_v2_a1_cur3v2_run resume-test   # stage 6 evidence (real SB3, interrupted vs uninterrupted)
    python -m experiments.journal_v2_a1_cur3v2_run all           # stages 1-9: verification, protocol, scenario validation, tests gate, resume gate, PPO training of seeds 201/202/203
    python -m experiments.journal_v2_a1_cur3v2_run status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur2_run as RUN2
from experiments import journal_v2_a1_cur3 as V1
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur3v2_state as TS
from experiments import journal_v2_a1_cur3v2_training as T3

DOCS = X.DOCS
START_HEAD = "1214c38"
FROZEN = ("experiments/journal_v2_a1.py", "experiments/journal_v2_a1_r2.py", "experiments/journal_v2_a1_cur1.py", "experiments/journal_v2_a1_cur1v2.py", "experiments/journal_v2_a1_cur1v2_bc.py",
          "experiments/journal_v2_a1_cur2.py", "experiments/journal_v2_a1_cur2_training.py", "experiments/journal_v2_a1_cur2_final.py", "experiments/journal_v2_a1_cur2_run.py",
          "experiments/journal_v2_a1_cur2v2.py", "experiments/journal_v2_validation_core.py", "uav_defend")


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(TS.PROJECT_ROOT.resolve())).replace("\\", "/")


def stage2_integrity() -> dict:
    out = {}
    for label, root in (("stage2_v1", S.RESULTS_ROOT), ("stage2_v2", X.STAGE2V2_RESULTS_ROOT)):
        for s in S.PPO_SEEDS:
            m = json.loads((root / f"seed_{s}" / "checkpoints" / "seed_manifest.json").read_text())
            out[f"{label}_seed_{s}"] = {"status": m["status"], "accepted_sha256_matches": TS.sha256_file(root / f"seed_{s}" / "checkpoints" / "accepted_model.zip") == m["accepted_model_sha256"]}
    assert all(v["accepted_sha256_matches"] for v in out.values())
    return out


def verify_repository() -> dict:
    head = TS.git("rev-parse", "HEAD")
    v1_state = json.loads((TS.PROJECT_ROOT / X.V1_DOCS / "task_state.json").read_text())
    v1_evidence = json.loads((TS.PROJECT_ROOT / X.V1_DOCS / "scenario_validation.json").read_text())
    parents = {str(s): X.parent_checkpoint(s) for s in X.PPO_SEEDS}
    rec = {"head": head, "expected_start_head": START_HEAD, "head_is_expected_or_descendant": subprocess.run(["git", "merge-base", "--is-ancestor", START_HEAD, "HEAD"], cwd=TS.PROJECT_ROOT).returncode == 0,
           "branch": TS.git("rev-parse", "--abbrev-ref", "HEAD"), "tracked_tree_clean": TS.git("status", "--porcelain", "--untracked-files=no") == "",
           "untracked_at_start": [l[3:] for l in TS.git("status", "--porcelain", "--untracked-files=all").splitlines() if l.startswith("??")],
           "uav_defend_identical_to_a0_tag": TS.git("diff", "--stat", "journal-v2-a0-canonical-v1", "--", "uav_defend") == "",
           "frozen_files_identical_to_start_head": TS.git("diff", "--stat", START_HEAD, "--", *FROZEN) == "", "frozen_files": list(FROZEN),
           "stage3_v1_evidence": {"path": X.V1_DOCS + "scenario_validation.json", "verdict": v1_evidence["verdict"], "a0_interception": v1_evidence["summary"]["a0_interception_fraction"],
                                  "sha256": TS.sha256_file(TS.PROJECT_ROOT / X.V1_DOCS / "scenario_validation.json"), "v1_protocol_sha256": v1_evidence["protocol_sha256"], "v1_task_state_status": v1_state["overall_status"]},
           "stage2_v2_parents": parents, "parent_status": X.PARENT_STATUS, "detection_radius_m": X.detection_radius(), "reward_version": S.REWARD_VERSION, "observation_version": S.OBSERVATION_VERSION,
           "action_version": S.ACTION_VERSION, "ppo_hyperparameters": A.A1_V2_PPO_HYPERPARAMS, "integrity": RUN2.checkpoint_integrity(), "stage2_integrity": stage2_integrity(),
           "namespace_overlaps": X.namespace_overlaps()}
    assert rec["head_is_expected_or_descendant"] and rec["tracked_tree_clean"] and rec["uav_defend_identical_to_a0_tag"] and rec["frozen_files_identical_to_start_head"]
    assert rec["stage3_v1_evidence"]["a0_interception"] == 0.0 and rec["detection_radius_m"] == 15.0 and rec["reward_version"] == "a1_v2_reward_v2" and rec["ppo_hyperparameters"]["n_steps"] == 2048
    assert not any(rec["namespace_overlaps"][k] for k in ("validation_A", "validation_B", "stage1_anchor", "static_anchor", "diagnostic", "scenario_validation", "mutual", "training_overlap"))
    return rec


def scenario_validation_record() -> dict:
    summary = X.scenario_validation()
    verdict = X.validation_verdict(summary)
    first = X.evaluate(None, list(range(X.SCENARIO_VALIDATION[0], X.SCENARIO_VALIDATION[0] + 50)), scripted_a0=True)
    again = X.evaluate(None, list(range(X.SCENARIO_VALIDATION[0], X.SCENARIO_VALIDATION[0] + 50)), scripted_a0=True)
    deterministic = all(a["defender_position"] == b["defender_position"] and a["outcome"] == b["outcome"] and a["steps"] == b["steps"] and a["detection_step"] == b["detection_step"]
                        and a["defender_final_position"] == b["defender_final_position"] for a, b in zip(first, again))
    return {"scenario_protocol": X.SCENARIO_PROTOCOL, "seeds": list(X.SCENARIO_VALIDATION), "detection_radius_m": X.detection_radius(), "summary": summary, "verdict": verdict,
            "deterministic_replay_first_50": bool(deterministic), "protocol_sha256": X.protocol_sha256(),
            "note": "examined the A0 baseline before PPO training only; the immediate-contact criterion was defined as 'episode ends within 1 step of reset / < 2 policy decisions with the defender in the 25 m gate' "
                    "(interceptions within 1 step after release are inherent in the geometry and reported informationally)"}


def baseline_comparison() -> dict:
    cmp = X.v1_vs_v2_baseline()
    v1_doc = json.loads((TS.PROJECT_ROOT / X.V1_DOCS / "scenario_validation.json").read_text())["summary"]
    table = [
        {"metric": "A0 interception", "stage3_v1_escort": cmp["stage3_v1_escort"]["a0_interception"], "stage3_v2_hold": cmp["stage3_v2_hold"]["a0_interception"]},
        {"metric": "A0 capture", "stage3_v1_escort": cmp["stage3_v1_escort"]["a0_capture"], "stage3_v2_hold": cmp["stage3_v2_hold"]["a0_capture"]},
        {"metric": "Defender speed at detection (median, p95) [m/s]", "stage3_v1_escort": [cmp["stage3_v1_escort"]["defender_speed_at_detection_mps_median"], cmp["stage3_v1_escort"]["defender_speed_at_detection_mps_p95"]],
         "stage3_v2_hold": [cmp["stage3_v2_hold"]["defender_speed_at_detection_mps_median"], cmp["stage3_v2_hold"]["defender_speed_at_detection_mps_p95"]]},
        {"metric": "Defender velocity toward attacker at detection (median) [m/s]", "stage3_v1_escort": cmp["stage3_v1_escort"]["defender_vel_toward_attacker_at_detection_mps_median"],
         "stage3_v2_hold": cmp["stage3_v2_hold"]["defender_vel_toward_attacker_at_detection_mps_median"]},
        {"metric": "Median min defender separation [m]", "stage3_v1_escort": cmp["stage3_v1_escort"]["median_min_separation_m"], "stage3_v2_hold": cmp["stage3_v2_hold"]["median_min_separation_m"]},
        {"metric": "Median episode duration [s]", "stage3_v1_escort": cmp["stage3_v1_escort"]["episode_duration_s_median"], "stage3_v2_hold": cmp["stage3_v2_hold"]["episode_duration_s_median"]},
    ]
    return {**cmp, "table": table, "stage3_v1_recorded_evidence": {"seeds": v1_doc and [153000, 153999], "a0_interception": v1_doc["a0_interception_fraction"],
                                                                   "median_min_separation_m": v1_doc["a0_min_defender_separation_m"]["median"], "file": X.V1_DOCS + "scenario_validation.json"},
            "note": "both columns re-run on the identical Stage3-v2 scenario block 157000-157999 through the same code path (`hold_until_detection` False vs True); the recorded v1 evidence used seeds 153000-153999"}


def run_all() -> None:
    state = TS.load_state()
    psha = X.protocol_sha256()
    scratch = T3.select_scratch_root()
    print(f"[stage3-v2] scratch={scratch} protocol_sha={psha[:16]}", flush=True)
    if not TS.stage_is_verified(state, "01_repository_cleanup_verification"):
        TS.begin(state, "01_repository_cleanup_verification")
        TS.atomic_write_json(TS.STATE_DIR / "repository_verification.json", verify_repository())
        TS.complete(state, "01_repository_cleanup_verification", [DOCS + "repository_verification.json", X.V1_DOCS + "scenario_validation.json"],
                    "HEAD 1214c38, tracked tree clean, Stage3-v1 evidence preserved, parents/integrity verified")
    if not TS.stage_is_verified(state, "02_stage3_v2_protocol_freeze", psha):
        TS.begin(state, "02_stage3_v2_protocol_freeze", psha)
        TS.atomic_write_json(TS.STATE_DIR / "protocol.json", X.protocol_specification() | {"protocol_sha256": psha})
        TS.complete(state, "02_stage3_v2_protocol_freeze", [DOCS + "protocol.json"], "protocol sha " + psha[:16])
    code = ["experiments/journal_v2_a1_cur3v2.py", "experiments/journal_v2_a1_cur3v2_training.py", "experiments/journal_v2_a1_cur3v2_state.py", "experiments/journal_v2_a1_cur3v2_run.py"]
    if not TS.stage_is_verified(state, "03_hold_wrapper_implementation", psha):
        TS.begin(state, "03_hold_wrapper_implementation", psha)
        TS.complete(state, "03_hold_wrapper_implementation", code, "Stage3V2Env hold/release wrapper + runner (tests/test_journal_v2_a1_cur3v2.py)")
    if not TS.stage_is_verified(state, "04_scenario_validation", psha):
        TS.begin(state, "04_scenario_validation", psha)
        rec = scenario_validation_record()
        TS.atomic_write_json(TS.STATE_DIR / "scenario_validation.json", rec)
        TS.atomic_write_json(TS.STATE_DIR / "baseline_comparison.json", baseline_comparison())
        artifacts = [DOCS + "scenario_validation.json", DOCS + "baseline_comparison.json"]
        if not rec["verdict"]["passed"] or not rec["deterministic_replay_first_50"]:
            TS.fail(state, "04_scenario_validation", "SCENARIO VALIDATION FAILED - DO NOT TRAIN (verdict D): " + json.dumps(rec["verdict"]), artifacts)
            raise SystemExit("scenario validation failed: verdict D, no training")
        TS.complete(state, "04_scenario_validation", artifacts, f"A0 interception {rec['summary']['a0_interception_fraction']:.3f} on 1000 scenarios; validity checks passed")
    test_path = TS.STATE_DIR / "unit_test_result.json"
    if not TS.stage_is_verified(state, "05_tests", psha):
        if not test_path.exists() or json.loads(test_path.read_text()).get("failed", 1) != 0:
            raise SystemExit("run the targeted tests first and write unit_test_result.json (failed == 0)")
        TS.begin(state, "05_tests", psha)
        TS.complete(state, "05_tests", [DOCS + "unit_test_result.json"], "targeted tests passed")
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
        parent = X.parent_checkpoint(seed)
        result = T3.train_seed(seed, scratch_root=scratch, parent=parent)
        if result.get("status") == "INCOMPLETE":
            raise SystemExit(f"seed {seed} incomplete")
        path = TS.STATE_DIR / f"seed_{seed}_manifest.json"
        TS.atomic_write_json(path, result | {"stage3_v2_protocol_sha256": psha})
        last = result["rollouts"][-1]
        TS.complete(state, stage_name, [DOCS + path.name], f"seed {seed}: {result['status']} rollouts={result['rollouts_trained']} mastery_rollout={result['mastery_rollout']}",
                    {"parent_sha256": parent["model_sha256"], "completed_rollout": result["rollouts_trained"], "final_model_sha256": last["model_sha256"], "final_state_sha256": last["state_sha256"],
                     "accepted_model_sha256": result["accepted_model_sha256"]})
    print(json.dumps(T3.status(scratch), indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "status", "resume-test"))
    args = parser.parse_args(argv)
    TS.STATE_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        print(json.dumps(T3.status(), indent=1))
    elif args.command == "resume-test":
        T3.T.keep_awake()
        scratch = T3.select_scratch_root()
        result = T3.resume_test(scratch.parent / "stage3_v2_resume_test")
        TS.atomic_write_json(TS.STATE_DIR / "resume_validation.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    else:
        T3.T.keep_awake()
        run_all()


if __name__ == "__main__":
    main()
