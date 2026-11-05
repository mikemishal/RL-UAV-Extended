"""Stage runner for curriculum Stage 4A-1 (goal-conditioned pursuit via translated A0 imitation + Stage3-v2 retention). Safe to re-run after ANY interruption.

    python -m experiments.journal_v2_a1_cur4a1_run all     # stages 1-8: verification, protocol, env/tests gate, dataset, PRE sensitivity, BC fine-tuning of seeds 201/202/203
    python -m experiments.journal_v2_a1_cur4a1_run status
"""

from __future__ import annotations

import argparse
import json
import subprocess

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur4 as P4
from experiments import journal_v2_a1_cur4_run as RUN4
from experiments import journal_v2_a1_cur4a as P4A
from experiments import journal_v2_a1_cur4a1 as Z
from experiments import journal_v2_a1_cur4a1_bc as BC4
from experiments import journal_v2_a1_cur4a1_data as D
from experiments import journal_v2_a1_cur4a1_retention as RT
from experiments import journal_v2_a1_cur4a1_state as TS
from experiments import journal_v2_a1_cur4a_run as RUN4A

DOCS = Z.DOCS
START_HEAD = "9984495"
FROZEN = RUN4A.FROZEN + ("experiments/journal_v2_a1_cur4a.py", "experiments/journal_v2_a1_cur4a_final.py", "experiments/journal_v2_a1_cur4a_run.py", "experiments/journal_v2_a1_cur4a_state.py",
                        "experiments/journal_v2_a1_cur4a_training.py", "tests/test_journal_v2_a1_cur4a.py")
MAX_EPOCHS = Z.MAX_EPOCHS


def stage4a_integrity() -> dict:
    out = {}
    for s in Z.PPO_SEEDS:
        root = P4A.RESULTS_ROOT / f"seed_{s}"
        m = json.loads((root / "checkpoints" / "seed_manifest.json").read_text())
        rolls = {r["rollout"]: TS.sha256_file(root / "rollouts" / f"rollout_{r['rollout']:02d}.zip") == r["model_sha256"] for r in m["rollouts"]}
        out[f"stage4a_seed_{s}"] = {"status": m["status"], "matches_recorded_final_hash": TS.sha256_file(root / "checkpoints" / "accepted_model.zip") == Z.STAGE4A_ACCEPTED_FINAL_SHA256[s],
                                    "all_rollouts_match": all(rolls.values())}
    assert all(v["status"] == "FAIL" and v["matches_recorded_final_hash"] and v["all_rollouts_match"] for v in out.values())
    return out


def verify_repository() -> dict:
    head = TS.git("rev-parse", "HEAD")
    decision = json.loads((TS.PROJECT_ROOT / P4A.DOCS / "stage4a_decision.json").read_text())
    rec = {"head": head, "expected_start_head": START_HEAD, "head_is_expected_or_descendant": subprocess.run(["git", "merge-base", "--is-ancestor", START_HEAD, "HEAD"], cwd=TS.PROJECT_ROOT).returncode == 0,
           "branch": TS.git("rev-parse", "--abbrev-ref", "HEAD"), "tracked_tree_clean": TS.git("status", "--porcelain", "--untracked-files=no") == "",
           "untracked_at_start": [l[3:] for l in TS.git("status", "--porcelain", "--untracked-files=all").splitlines() if l.startswith("??")],
           "uav_defend_identical_to_a0_tag": TS.git("diff", "--stat", "journal-v2-a0-canonical-v1", "--", "uav_defend") == "",
           "frozen_files_identical_to_start_head": TS.git("diff", "--stat", START_HEAD, "--", *FROZEN) == "", "frozen_files": list(FROZEN),
           "stage4a": {"verdict": decision.get("verdict"), "stage4a_pass": decision.get("stage4a_pass"), "status": "failed Stage 4A preserved as non-canonical evidence; its checkpoints are never used for Stage 4A-1"},
           "stage3_v2_parents": {str(s): Z.parent_checkpoint(s) for s in Z.PPO_SEEDS}, "parent_status": Z.PARENT_STATUS, "env_config_sha256": Z.config_sha256(),
           "config_differences_from_canonical": Z.config_differences_from_canonical(), "reward_version": S.REWARD_VERSION, "observation_version": S.OBSERVATION_VERSION, "action_version": S.ACTION_VERSION,
           "ppo_architecture": A.A1_V2_POLICY, "integrity": RUN4.RUN2.checkpoint_integrity(), "stage2_integrity": RUN4.RUN3.stage2_integrity(), "stage3_integrity": RUN4.stage3_integrity(),
           "stage4_integrity": RUN4A.stage4_integrity(), "stage4a_integrity": stage4a_integrity(), "namespace_overlaps": Z.namespace_overlaps()}
    assert rec["head_is_expected_or_descendant"] and rec["tracked_tree_clean"] and rec["uav_defend_identical_to_a0_tag"] and rec["frozen_files_identical_to_start_head"]
    assert rec["reward_version"] == "a1_v2_reward_v2" and set(rec["config_differences_from_canonical"]) == {"v_s", "intercept_radius", "defender_standby_until_detection"}
    assert not any(rec["namespace_overlaps"][k] for k in ("expert", "validation_A", "validation_B", "anchor1", "anchor2", "anchor3", "generalization", "retention_source", "mutual", "training_overlap"))
    return rec


def translated_env_checks() -> dict:
    """Stage 3 evidence (`03_translated_scenario_generator`): translation preserves relative geometry and respects the safety margin, over a probe sample."""
    probe = range(Z.EXPERT_SEED_START, Z.EXPERT_SEED_START + 200)
    max_abs, min_range_diff = 0.0, 0.0
    for s in probe:
        env_t = Z.TranslatedStage1V2Env(0, 0, 0, scripted_a0=True)
        env_u = Z.V1V2.Stage1V2Env(0, 0, 0, scripted_a0=True)
        env_t.reset(options={"scenario": int(s)})
        env_u.reset(options={"scenario": int(s)})
        max_abs = max(max_abs, float(abs(env_t.soldier_env._enemy_pos[:2] - env_u.soldier_env._enemy_pos[:2] - env_t.translation).max()))
        rel_t = float(((env_t.soldier_env._soldier_pos - env_t.soldier_env._enemy_pos) ** 2).sum() ** 0.5)
        rel_u = float(((env_u.soldier_env._soldier_pos - env_u.soldier_env._enemy_pos) ** 2).sum() ** 0.5)
        min_range_diff = max(min_range_diff, abs(rel_t - rel_u))
        L, margin = float(env_t._config.L), Z.MARGIN_M
        assert abs(env_t.soldier_env._soldier_pos[0]) <= L - margin + 1e-6 and abs(env_t.soldier_env._soldier_pos[1]) <= L - margin + 1e-6
        assert abs(env_t.soldier_env._enemy_pos[0]) <= L - margin + 1e-6 and abs(env_t.soldier_env._enemy_pos[1]) <= L - margin + 1e-6
    return {"scenarios_checked": len(probe), "max_abs_translation_consistency_error_m": max_abs, "max_relative_range_change_m": min_range_diff, "margin_m": Z.MARGIN_M,
            "relative_geometry_preserved": min_range_diff < 1e-3, "translation_applied_consistently": max_abs < 1e-3}


def run_all() -> None:
    state = TS.load_state()
    psha = Z.protocol_sha256()
    print(f"[stage4a1] protocol_sha={psha[:16]}", flush=True)
    if not TS.stage_is_verified(state, "01_repository_verification"):
        TS.begin(state, "01_repository_verification")
        TS.atomic_write_json(TS.STATE_DIR / "repository_verification.json", verify_repository())
        TS.complete(state, "01_repository_verification", [DOCS + "repository_verification.json"], "HEAD 9984495, tracked tree clean, Stage3-v2 parents + Stage1-4A checkpoint integrity verified")
    if not TS.stage_is_verified(state, "02_protocol_freeze", psha):
        TS.begin(state, "02_protocol_freeze", psha)
        TS.atomic_write_json(TS.STATE_DIR / "protocol.json", Z.protocol_specification() | {"protocol_sha256": psha})
        TS.complete(state, "02_protocol_freeze", [DOCS + "protocol.json"], "protocol sha " + psha[:16])
    code = ["experiments/journal_v2_a1_cur4a1.py", "experiments/journal_v2_a1_cur4a1_data.py", "experiments/journal_v2_a1_cur4a1_retention.py", "experiments/journal_v2_a1_cur4a1_bc.py",
            "experiments/journal_v2_a1_cur4a1_state.py", "experiments/journal_v2_a1_cur4a1_run.py"]
    if not TS.stage_is_verified(state, "03_translated_scenario_generator", psha):
        TS.begin(state, "03_translated_scenario_generator", psha)
        rec = translated_env_checks()
        TS.atomic_write_json(TS.STATE_DIR / "translated_scenario_checks.json", rec)
        assert rec["relative_geometry_preserved"] and rec["translation_applied_consistently"]
        TS.complete(state, "03_translated_scenario_generator", code + [DOCS + "translated_scenario_checks.json"], "relative geometry preserved; translation consistent; safety margin respected")
    if not TS.stage_is_verified(state, "04_expert_dataset_generation", psha):
        TS.begin(state, "04_expert_dataset_generation", psha)
        manifest = D.generate(workers=10)
        TS.atomic_write_json(TS.STATE_DIR / "dataset_manifest.json", manifest)
        TS.complete(state, "04_expert_dataset_generation", [DOCS + "dataset_manifest.json"], f"{manifest['pairs']} pairs from {manifest['episodes']} translated A0 episodes; dataset sha "
                    + manifest["dataset_sha256"][:16])
    if not TS.stage_is_verified(state, "05_baseline_sensitivity_analysis", psha):
        from stable_baselines3 import PPO

        TS.begin(state, "05_baseline_sensitivity_analysis", psha)
        pre = {}
        for s in Z.PPO_SEEDS:
            model = PPO.load(Z.parent_checkpoint(s)["path"].removesuffix(".zip"), device="cpu")
            pre[str(s)] = Z.representation_record(model)
        TS.atomic_write_json(TS.STATE_DIR / "pre_sensitivity.json", pre)
        TS.complete(state, "05_baseline_sensitivity_analysis", [DOCS + "pre_sensitivity.json"],
                    "PRE asset/attacker sensitivity ratio " + str({s: round(v["sensitivity"]["asset_to_attacker_sensitivity_ratio"], 4) for s, v in pre.items()}))
    for stage_name, seed in (("06_seed_201_bc_fine_tuning", 201), ("07_seed_202_bc_fine_tuning", 202), ("08_seed_203_bc_fine_tuning", 203)):
        if TS.stage_is_verified(state, stage_name, psha):
            continue
        TS.begin(state, stage_name, psha)
        result = run_seed(seed)
        path = TS.STATE_DIR / f"seed_{seed}_manifest.json"
        TS.atomic_write_json(path, result)
        TS.complete(state, stage_name, [DOCS + path.name], f"seed {seed}: {result['status']} epochs={result['epochs_trained']} mastery_epoch={result.get('mastery_epoch')}",
                    {"parent_sha256": result["parent"]["model_sha256"], "accepted_model_sha256": result["accepted_model_sha256"]})
    print(json.dumps(status(), indent=1))


def run_seed(seed: int, *, scratch_root=None, max_epochs: int = MAX_EPOCHS) -> dict:
    """BC fine-tune one seed end-to-end: combined-loss epochs, validate each epoch (translated held-out + three anchors), stop at 2 consecutive mastery passes / retention failure / max_epochs."""
    import time
    from pathlib import Path

    import torch
    from stable_baselines3 import PPO

    from experiments import journal_v2_a1_eg_state as base

    scratch_root = Path(scratch_root) if scratch_root else select_scratch_root()
    layout_dir, results_dir = scratch_root / f"seed_{seed}", Z.RESULTS_ROOT / f"seed_{seed}"
    manifest_path = results_dir / "seed_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    results_dir.mkdir(parents=True, exist_ok=True)
    data = D.load_all()
    sp = D.split(data)
    parent = Z.parent_checkpoint(seed)
    parent_model = PPO.load(parent["path"].removesuffix(".zip"), device="cpu")
    ret_ds = RT.collect_retention_dataset(parent_model, target_states=Z.RETENTION_TARGET_PER_SEED, seeds=range(*Z.RETENTION_SOURCE))
    base.atomic_write_json(results_dir / "retention_dataset_summary.json", RT.retention_coverage(ret_ds) | {"sha256": RT.dataset_sha256(ret_ds)})
    ret_sp = RT.split(ret_ds)
    model, init_record = BC4.initialize_model(seed, parent=parent)
    base.atomic_write_json(results_dir / "init_record.json", init_record)
    hyper = {"epochs": max_epochs, "lr": 1e-3, "batch_size": 256}
    started = time.time()
    tag = f"stage4a1_seed{seed}"
    history, status_, decided_at = [], None, None
    for stop in range(1, max_epochs + 1):
        out = BC4.train_combined(model, sp["train"], ret_sp["train"], sp["val"], ret_sp["val"], work_dir=layout_dir, seed=seed, tag=tag, hyper=hyper, stop_after=stop)
        history = out["history"]
        validation = epoch_validation(model, history[-1])
        base.atomic_write_json(results_dir / f"epoch_{stop:03d}_validation.json", validation)
        all_validations = [json.loads((results_dir / f"epoch_{k:03d}_validation.json").read_text()) for k in range(1, stop + 1)]
        status_, decided_at = seed_decision(all_validations)
        if status_ is not None:
            break
    accepted_epoch = decided_at if status_ == "MASTERED" else len(history)
    accepted_ckpt = layout_dir / BC4.epoch_names(accepted_epoch)["ckpt"]
    state_dict = torch.load(accepted_ckpt, weights_only=False)
    BC4.BC.load_actor_state(model, state_dict["actor"])
    sha = BC4.BC.save_model(model, results_dir / "accepted_model.zip")
    manifest = {"seed": seed, "status": status_ or "FAIL", "epochs_trained": len(history), "mastery_epoch": decided_at if status_ == "MASTERED" else None, "accepted_epoch": accepted_epoch,
                "accepted_model_sha256": sha, "parent": parent, "parent_status": Z.PARENT_STATUS, "bc_protocol": Z.BC_PROTOCOL, "protocol_sha256": Z.protocol_sha256(),
                "hyper": hyper, "retain_loss_weight": Z.RETAIN_LOSS_WEIGHT,
                "epochs": [{"epoch": h["epoch"], "train_loss": h["train_loss"], "val": h["val"], "validation": json.loads((results_dir / f"epoch_{h['epoch']:03d}_validation.json").read_text())} for h in history],
                "wall_time_s": time.time() - started, "research_commit": base.git("rev-parse", "HEAD"), "created_utc": base._now()}
    base.atomic_write_json(manifest_path, manifest)
    return manifest


def epoch_validation(model, epoch_record: dict) -> dict:
    act = C.model_actor(model)
    m = Z.translated_metrics(Z.evaluate_translated(act, Z.final_evaluation_seeds()[:40]))
    an1 = Z.evaluate_anchor1(act)
    an2 = Z.anchor_metrics_static(Z.evaluate_anchor2(act))
    an3 = Z.anchor_metrics_moving(Z.evaluate_anchor3(act))
    an1m = Z.metrics(an1)
    gate = Z.gate_status(m, anchor1=an1m["capture"], anchor2=an2["capture"], anchor3=an3["capture"])
    return {"epoch": epoch_record["epoch"], "metrics": m, "anchor1_capture": an1m["capture"], "anchor2_capture": an2["capture"], "anchor3_capture": an3["capture"], "gate": gate, "passed": gate["passed"]}


def seed_decision(validations: list[dict]) -> tuple:
    ANCHOR_NAMES = {"anchor1_capture": "anchor1_basic_pursuit", "anchor2_capture": "anchor2_static_defender", "anchor3_capture": "anchor3_moving_defender"}
    for k in range(1, len(validations) + 1):
        passes = [v["passed"] for v in validations[:k]]
        if k >= 2 and passes[-1] and passes[-2]:
            return "MASTERED", k
        for key, name in ANCHOR_NAMES.items():
            vals = [v[key] for v in validations[:k]]
            if k >= 2 and vals[-1] < Z.ANCHOR_MIN[name] and vals[-2] < Z.ANCHOR_MIN[name]:
                return "RETENTION_FAILURE", k
    return None, None


def select_scratch_root():
    from pathlib import Path

    from experiments import journal_v2_a1_cur4_training as T4

    candidates = [Path(r"C:\Users\Public\journal_v2_a1_curriculum\stage4a1")]
    for c in candidates:
        if not T4.T._is_synced(c) and T4.T._writable_roundtrip(c):
            return c.resolve()
    raise RuntimeError("no writable local non-synced scratch directory")


def status() -> dict:
    out = {}
    for s in Z.PPO_SEEDS:
        p = Z.RESULTS_ROOT / f"seed_{s}" / "seed_manifest.json"
        out[s] = json.loads(p.read_text())["status"] if p.exists() else "NOT_STARTED"
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "status"))
    args = parser.parse_args(argv)
    TS.STATE_DIR.mkdir(parents=True, exist_ok=True)
    if args.command == "status":
        print(json.dumps(status(), indent=1))
    else:
        run_all()


if __name__ == "__main__":
    main()
