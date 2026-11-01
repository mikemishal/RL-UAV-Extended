"""Stage runner for curriculum Stage 1 V2 (expert initialization). Safe to re-run after ANY interruption: verified stages, dataset shards, BC epochs, relabel states and
evaluations are skipped; the first incomplete piece is resumed. NO PPO learn().

    python -m experiments.journal_v2_a1_cur1v2_run all         # stages 1-9 (documentation / visualizer / commit stages are recorded by hand)
    python -m experiments.journal_v2_a1_cur1v2_run status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1_training as CT   # scratch-root selection only
from experiments import journal_v2_a1_cur1v2 as V
from experiments import journal_v2_a1_cur1v2_bc as BC
from experiments import journal_v2_a1_cur1v2_data as D
from experiments import journal_v2_a1_cur1v2_state as TS
from experiments import journal_v2_a1_r2 as R

DOCS = "docs/journal_v2_a1_curriculum/stage1_v2/"
EVAL_PATH = TS.STATE_DIR / "final_evaluation.json"
DECISION_PATH = TS.STATE_DIR / "stage1_v2_decision.json"
HANDOFF_PATH = TS.STATE_DIR / "stage2_handoff.json"
DEFAULT_SCRATCH = Path(r"C:\Users\Public\journal_v2_a1_curriculum\stage1_v2")


def scratch_root() -> Path:
    from experiments import journal_v2_a1_r2_training as T

    for candidate in (DEFAULT_SCRATCH, Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "journal_v2_a1_curriculum" / "stage1_v2"):
        if not T._is_synced(candidate) and T._writable_roundtrip(candidate):
            return candidate.resolve()
    raise RuntimeError("no writable local non-synced scratch directory")


def seed_results(seed: int) -> Path:
    return V.RESULTS_ROOT / f"seed_{seed}"


def seed_scratch(seed: int, scratch: Path | None = None) -> Path:
    return (scratch or scratch_root()) / f"seed_{seed}"


# ----------------------------------------------------------------------------- round 0 (stages 5-7)
def round0_ok(seed: int) -> dict | None:
    try:
        s = json.loads((seed_results(seed) / "round0_summary.json").read_text())
        if TS.sha256_file(seed_results(seed) / "round0_model.zip") == s["model_sha256"]:
            return s
    except (OSError, ValueError, KeyError):
        pass
    return None


def run_round0(seed: int, expert: dict, scratch: Path | None = None, results: Path | None = None) -> dict:
    out = Path(results) if results is not None else seed_results(seed)
    existing = round0_ok(seed) if results is None else None
    if existing:
        return existing
    out.mkdir(parents=True, exist_ok=True)
    model = BC.build_model(seed)
    value_before = BC.value_state_sha256(model)
    train, val = expert["train"], expert["val"]
    result = BC.train_bc(model, train, val, work_dir=(scratch or scratch_root()) / f"seed_{seed}" / "round0", seed=seed, tag="round0", hyper=V.BC)
    BC.set_log_std(model)
    sha = BC.save_model(model, out / "round0_model.zip")
    best = result["selected_epoch"]
    summary = {"seed": seed, "phase": "round0", "selected_epoch": best, "epochs": result["completed_epochs"], "model_sha256": sha, "state_sha256": BC.model_state_sha256(model),
               "value_network_unchanged": BC.value_state_sha256(model) == value_before, "log_std": [float(x) for x in model.policy.log_std.detach().numpy()],
               "initial_action_std": float(np.exp(model.policy.log_std.detach().numpy()).mean()), "train_metrics": BC.bc_metrics(model, *train), "val_metrics": BC.bc_metrics(model, *val),
               "history": [{"epoch": r["epoch"], "train_loss": r["train_loss"], "val_loss": r["val"]["loss"], "val_cos_mean": r["val"]["cos_mean"], "val_mse": r["val"]["mse"]} for r in result["history"]],
               "hyper": V.BC, "dataset_sha256": D.dataset_sha256()}
    TS.atomic_write_json(out / "round0_summary.json", summary)
    return summary


# ----------------------------------------------------------------------------- relabeling + fine-tuning (stage 8)
def relabel_paths(seed: int, results: Path | None = None) -> tuple[Path, Path]:
    base = Path(results) if results is not None else seed_results(seed)
    return base / "relabel_states.npz", base / "relabel_states.json"


def collect_relabel(seed: int, model, results: Path | None = None) -> dict:
    npz, side = relabel_paths(seed, results)
    if npz.exists() and side.exists():
        try:
            s = json.loads(side.read_text())
            if TS.sha256_file(npz) == s["npz_sha256"]:
                with np.load(npz) as z:
                    return {k: z[k] for k in z.files} | {"meta": s}
        except (OSError, ValueError, KeyError):
            pass
    visited = BC.collect_visited_states(model, V.relabel_seeds())
    payload = D.npz_bytes({"obs": visited["obs"], "act": visited["act"], "ep_seed": visited["ep_seed"]})
    TS.atomic_write_bytes(npz, payload)
    meta = {"seed": seed, "episodes": len(visited["steps_per_episode"]), "visited_states": int(visited["obs"].shape[0]), "outcomes_round0_policy": visited["outcomes"],
            "max_states_per_episode": V.RELABEL_MAX_STEPS, "mean_states_per_episode": float(np.mean(visited["steps_per_episode"])), "npz_sha256": hashlib.sha256(payload).hexdigest(),
            "scenario_range": [V.RELABEL_SEED_START, V.RELABEL_SEED_START + V.RELABEL_EPISODES - 1]}
    TS.atomic_write_json(side, meta)
    return {"obs": visited["obs"], "act": visited["act"], "ep_seed": visited["ep_seed"], "meta": meta}


def finetune_ok(seed: int) -> dict | None:
    try:
        s = json.loads((seed_results(seed) / "final_summary.json").read_text())
        if TS.sha256_file(seed_results(seed) / "bc_final_model.zip") == s["model_sha256"]:
            return s
    except (OSError, ValueError, KeyError):
        pass
    return None


def run_relabel_finetune(seed: int, expert: dict, scratch: Path | None = None, results: Path | None = None) -> dict:
    existing = finetune_ok(seed) if results is None else None
    if existing:
        return existing
    out = Path(results) if results is not None else seed_results(seed)
    r0 = round0_ok(seed) if results is None else json.loads((out / "round0_summary.json").read_text())
    model = BC.load_model(out / "round0_model.zip")
    relabel = collect_relabel(seed, model, out)
    visited = {"obs": relabel["obs"], "act": relabel["act"], "ep_seed": relabel["ep_seed"]}
    extra = BC.relabel_arrays(visited)
    union = BC.append_datasets(expert, extra)
    pre_expert_val = BC.bc_metrics(model, *expert["val"])
    pre_relabel_val = BC.bc_metrics(model, *extra["val"])
    result = BC.train_bc(model, union["train"], union["val"], work_dir=(scratch or scratch_root()) / f"seed_{seed}" / "round1", seed=seed, tag="round1", hyper=V.FINE_TUNE)
    BC.set_log_std(model)
    sha = BC.save_model(model, out / "bc_final_model.zip")
    summary = {"seed": seed, "phase": "round1_finetune", "selected_epoch": result["selected_epoch"], "epochs": result["completed_epochs"], "model_sha256": sha, "state_sha256": BC.model_state_sha256(model),
               "log_std": [float(x) for x in model.policy.log_std.detach().numpy()], "initial_action_std": float(np.exp(model.policy.log_std.detach().numpy()).mean()),
               "relabel": {**{k: v for k, v in relabel["meta"].items() if k != "npz_sha256"}, "relabel_npz_sha256": relabel["meta"]["npz_sha256"], "added_train_states": int(extra["train"][0].shape[0]),
                           "added_val_states": int(extra["val"][0].shape[0]), "added_expert_labels_total": int(visited["obs"].shape[0])},
               "pre_relabel_round0_metrics": {"expert_val": pre_expert_val, "relabel_val": pre_relabel_val}, "post_relabel_metrics": {"expert_val": BC.bc_metrics(model, *expert["val"]),
               "relabel_val": BC.bc_metrics(model, *extra["val"]), "union_val": BC.bc_metrics(model, *union["val"])}, "value_network_unchanged": r0["value_network_unchanged"],
               "history": [{"epoch": r["epoch"], "train_loss": r["train_loss"], "val_loss": r["val"]["loss"], "val_cos_mean": r["val"]["cos_mean"]} for r in result["history"]], "hyper": V.FINE_TUNE,
               "round0_model_sha256": r0["model_sha256"], "dataset_sha256": D.dataset_sha256()}
    TS.atomic_write_json(out / "final_summary.json", summary)
    return summary


# ----------------------------------------------------------------------------- held-out evaluation (stage 9)
def _eval_chunk(args):
    path, seeds, scripted = args
    if scripted:
        return V.evaluate(None, seeds, scripted_a0=True)
    return V.evaluate(BC.model_actor(BC.load_model(path)), seeds)


def parallel_evaluate(path: str | None, seeds: list[int], *, scripted_a0: bool = False, workers: int = 18) -> list[dict]:
    size = max(1, int(np.ceil(len(seeds) / workers)))
    chunks = [seeds[i:i + size] for i in range(0, len(seeds), size)]
    with mp.get_context("spawn").Pool(min(workers, len(chunks))) as pool:
        parts = pool.map(_eval_chunk, [(path, c, scripted_a0) for c in chunks])
    return [e for part in parts for e in part]


def brief(episodes: list[dict]) -> list[dict]:
    keys = ("scenario_seed", "outcome", "steps", "path_length_m", "chord_m", "path_excess", "initial_range_m", "terminal_distance_m", "initial_altitude_m", "initial_heading_offset_deg",
            "overshoots", "orbit_revolutions_max", "nonclosing_run", "pathology")
    return [{k: e[k] for k in keys} | {"expert_cos_mean": (float(np.mean(e["expert_cosines"])) if e["expert_cosines"] else None)} for e in episodes]


def evaluate_all(seeds_list=V.PPO_SEEDS) -> dict:
    seeds = C.final_evaluation_seeds()
    result = json.loads(EVAL_PATH.read_text()) if EVAL_PATH.exists() else {}
    result.update({"seeds": [seeds[0], seeds[-1]], "episodes": len(seeds), "gate": V.GATE_V2})
    result.setdefault("per_seed", {})
    if "a0" not in result:
        a0 = parallel_evaluate(None, seeds, scripted_a0=True)
        m = V.metrics(a0)
        result["a0"] = {"metrics": m, "gate": V.gate_status(m), "episodes_summary": brief(a0)}
        TS.atomic_write_json(EVAL_PATH, result)
    for seed in seeds_list:
        r0, ft = round0_ok(seed), finetune_ok(seed)
        if not r0 or not ft:
            continue
        key = str(seed)
        if result["per_seed"].get(key, {}).get("final_model_sha256") == ft["model_sha256"]:
            continue
        entry = {"seed": seed, "round0_model_sha256": r0["model_sha256"], "final_model_sha256": ft["model_sha256"]}
        for name, path in (("round0", seed_results(seed) / "round0_model.zip"), ("final", seed_results(seed) / "bc_final_model.zip")):
            eps = parallel_evaluate(str(path), seeds)
            m = V.metrics(eps)
            entry[name] = {"metrics": m, "gate": V.gate_status(m), "episodes_summary": brief(eps)}
        result["per_seed"][key] = entry
        TS.atomic_write_json(EVAL_PATH, result)
    return result


def decide(evaluation: dict) -> dict:
    passed = [int(s) for s, v in evaluation["per_seed"].items() if v["final"]["gate"]["passed"]]
    failed = [int(s) for s, v in evaluation["per_seed"].items() if not v["final"]["gate"]["passed"]]
    categories = {"orbit": "orbit_episodes", "repeated_flyby": "repeated_flyby_episodes", "nonclosing": "nonclosing_episodes"}
    common = {}
    for name, key in categories.items():
        hit = [s for s in passed if evaluation["per_seed"][str(s)]["final"]["metrics"][key] / evaluation["episodes"] > V.COMMON_PATHOLOGY_RATE]
        if len(hit) >= V.COMMON_PATHOLOGY_SEEDS:
            common[name] = hit
    ok = bool(len(passed) >= 2 and not common)
    near = bool(not ok and any(v["final"]["metrics"]["capture"] >= 0.9 for v in evaluation["per_seed"].values()))
    return {"seeds_passed": sorted(passed), "seeds_failed": sorted(failed), "all_three_passed": len(passed) == 3, "common_structural_pathology": common, "stage1_v2_pass": ok,
            "near_pass_needs_review": near, "rule": V.protocol_specification()["pass_rule"]}


def handoff(evaluation: dict, decision: dict, dataset_manifest: dict) -> dict:
    return {"note": "These are curriculum initialization checkpoints, not final intelligent-attacker models.", "stage": "journal_v2_a1_curriculum/stage1_v2",
            "next_stage": "Stage 2: stationary asset, C0, ACTIVE stationary defender, PPO to learn evasion while preserving pursuit competence (not started)",
            "architecture": {"policy": "SB3 PPO MlpPolicy", "net_arch": {"pi": [64, 64], "vf": [64, 64]}, "activation": "tanh", "obs_dim": 37, "action_dim": 3},
            "versions": {"observation": V.OBSERVATION_VERSION, "action": V.ACTION_VERSION, "reward": V.REWARD_VERSION, "bc_protocol": V.BC_PROTOCOL}, "log_std": V.LOG_STD_INIT,
            "initial_action_std": float(np.exp(V.LOG_STD_INIT)), "value_function": "randomly initialized (NOT trained by BC); PPO must learn it", "curriculum_mode": V.CURRICULUM_MODE,
            "env_config_sha256": V.config_sha256(), "protocol_sha256": V.protocol_sha256(), "bc_dataset": {"sha256": dataset_manifest["dataset_sha256"], "pairs": dataset_manifest["pairs"],
            "train_pairs": dataset_manifest["train_pairs"], "val_pairs": dataset_manifest["val_pairs"], "expert_scenarios": dataset_manifest["expert_scenarios"]},
            "checkpoints": {str(s): {"path": str((seed_results(s) / "bc_final_model.zip").relative_to(TS.PROJECT_ROOT)), "model_sha256": evaluation["per_seed"][str(s)]["final_model_sha256"],
                                     "relabel_provenance": {k: v for k, v in json.loads((seed_results(s) / "final_summary.json").read_text())["relabel"].items()}} for s in decision["seeds_passed"]},
            "evaluation": {str(s): evaluation["per_seed"][str(s)]["final"]["metrics"] for s in decision["seeds_passed"]}, "a0_reference": evaluation["a0"]["metrics"], "decision": decision}


# ----------------------------------------------------------------------------- driver
def verify_repository() -> dict:
    import hashlib as _h

    head = TS.git("rev-parse", "HEAD")
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
    from experiments import journal_v2_a1 as A

    A.build_episode_config("C0", "greedy")
    rec = {"head": head, "branch": TS.git("rev-parse", "--abbrev-ref", "HEAD"), "expected_head_prefix": "67cf934", "head_matches": head.startswith("67cf934"),
           "a0_tag_commit": TS.git("rev-list", "-n", "1", "journal-v2-a0-canonical-v1"), "uav_defend_identical_to_a0_tag": TS.git("diff", "--stat", "journal-v2-a0-canonical-v1", "--", "uav_defend") == "",
           "tracked_tree_clean": TS.git("status", "--porcelain", "--untracked-files=no") == "", "canonical_v2_guard": "passed", "reward_version": R.REWARD_VERSION, "historical_checkpoints": len(entries),
           "historical_mismatches": bad, "failed_a1_v1_checkpoints": len(failed_v1["attackers"]), "failed_a1_v1_mismatches": fbad, "stage1_v1_failed_models": stage1v1,
           "failed_experiments_preserved": [p for p in ("docs/journal_v2_a1_campaign_v1", "docs/journal_v2_a1_discrete_v4", "docs/journal_v2_a1_tactical_geometry_v1", "docs/journal_v2_a1_evasion_gate_v1",
                                                        "docs/journal_v2_a1_route_level_v1", "docs/journal_v2_a1_curriculum/stage1") if (TS.PROJECT_ROOT / p).is_dir()]}
    assert rec["head_matches"] and rec["uav_defend_identical_to_a0_tag"] and rec["tracked_tree_clean"] and not bad and not fbad and len(rec["failed_experiments_preserved"]) == 6
    assert all(v["status"] == "FAIL" and v["sha256_matches"] and v["marker_present"] for v in stage1v1.values())
    return rec


def run_all(workers: int = 10) -> None:
    psha = V.protocol_sha256()
    state = TS.load_state()
    root = TS.PROJECT_ROOT
    if not TS.stage_is_verified(state, "01_verify_repository"):
        TS.begin(state, "01_verify_repository")
        TS.atomic_write_json(TS.STATE_DIR / "repository_verification.json", verify_repository())
        TS.complete(state, "01_verify_repository", [DOCS + "repository_verification.json"], "clean 67cf934; checkpoints and Stage-1-v1 failed models verified")
    if not TS.stage_is_verified(state, "02_freeze_stage1_v2_protocol", psha):
        TS.begin(state, "02_freeze_stage1_v2_protocol", psha)
        TS.atomic_write_json(TS.STATE_DIR / "protocol.json", V.protocol_specification() | {"protocol_sha256": psha})
        TS.complete(state, "02_freeze_stage1_v2_protocol", [DOCS + "protocol.json"], "protocol sha " + psha[:16])
    code = ["experiments/journal_v2_a1_cur1v2.py", "experiments/journal_v2_a1_cur1v2_data.py", "experiments/journal_v2_a1_cur1v2_bc.py", "experiments/journal_v2_a1_cur1v2_state.py",
            "experiments/journal_v2_a1_cur1v2_run.py"]
    if not TS.stage_is_verified(state, "03_implement_clean_curriculum_wrapper", psha):
        TS.begin(state, "03_implement_clean_curriculum_wrapper", psha)
        TS.complete(state, "03_implement_clean_curriculum_wrapper", code, "clean wrapper + BC pipeline (see tests/test_journal_v2_a1_cur1v2.py)")
    # ---- 4 dataset
    manifest_path = D.DATA_DIR / "dataset_manifest.json"
    if not TS.stage_is_verified(state, "04_generate_expert_dataset", psha):
        TS.begin(state, "04_generate_expert_dataset", psha)
        manifest = D.generate(workers)
        TS.atomic_write_json(TS.STATE_DIR / "dataset_manifest.json", manifest)
        TS.complete(state, "04_generate_expert_dataset", [DOCS + "dataset_manifest.json"], f"{manifest['pairs']} pairs, sha {manifest['dataset_sha256'][:16]}")
    dataset_manifest = json.loads((TS.STATE_DIR / "dataset_manifest.json").read_text())
    assert D.dataset_sha256() == dataset_manifest["dataset_sha256"], "dataset changed on disk"
    expert = D.split(D.load_all())
    # ---- 5-7 round 0
    for stage, seed in (("05_train_bc_seed_201", 201), ("06_train_bc_seed_202", 202), ("07_train_bc_seed_203", 203)):
        art = f"results/journal_v2_corrected_semantics/a1_curriculum/stage1_v2/seed_{seed}/round0_summary.json"
        if not TS.stage_is_verified(state, stage, psha):
            TS.begin(state, stage, psha)
            s = run_round0(seed, expert)
            TS.complete(state, stage, [art], f"round-0 BC seed {seed}: val cos {s['val_metrics']['cos_mean']:.5f}, epoch {s['selected_epoch']}")
    # ---- 8 relabel + fine-tune
    stage8 = "08_fixed_expert_relabeling_round"
    if not TS.stage_is_verified(state, stage8, psha):
        TS.begin(state, stage8, psha)
        arts = []
        for seed in V.PPO_SEEDS:
            run_relabel_finetune(seed, expert)
            arts.append(f"results/journal_v2_corrected_semantics/a1_curriculum/stage1_v2/seed_{seed}/final_summary.json")
        TS.complete(state, stage8, arts, "one fixed expert-relabeling round + fine-tune for seeds 201/202/203")
    # ---- 9 evaluation
    stage9 = "09_heldout_evaluation"
    if not TS.stage_is_verified(state, stage9, psha):
        TS.begin(state, stage9, psha)
        ev = evaluate_all()
        dec = decide(ev)
        TS.atomic_write_json(DECISION_PATH, dec)
        arts = [DOCS + "final_evaluation.json", DOCS + "stage1_v2_decision.json"]
        if dec["stage1_v2_pass"]:
            TS.atomic_write_json(HANDOFF_PATH, handoff(ev, dec, dataset_manifest))
            arts.append(DOCS + "stage2_handoff.json")
        TS.complete(state, stage9, arts, f"seeds passed {dec['seeds_passed']}; stage pass={dec['stage1_v2_pass']}")


def status() -> dict:
    state = TS.load_state()
    return {s: state["stages"][s]["status"] for s in TS.STAGES} | {"next": TS.next_stage(state, V.protocol_sha256())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "status"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps(status(), indent=1))
    else:
        run_all(args.workers)
        print(json.dumps(status(), indent=1))
