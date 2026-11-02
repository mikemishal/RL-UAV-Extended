"""Stage runner for curriculum Stage2-v2 (budget-only continuation). Safe to re-run after ANY interruption.

    python -m experiments.journal_v2_a1_cur2v2_run all      # stages 1-7: verification, protocol, parent seeding, tests gate, PPO rollouts 9-16 for seeds 201/202/203
    python -m experiments.journal_v2_a1_cur2v2_run final    # stage 8: final evaluation + budget-sensitivity + learning-curve analysis + figures
    python -m experiments.journal_v2_a1_cur2v2_run status
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur2_final as F
from experiments import journal_v2_a1_cur2_run as RUN1
from experiments import journal_v2_a1_cur2_training as T2
from experiments import journal_v2_a1_cur2v2 as V2
from experiments import journal_v2_a1_cur2v2_state as TS

DOCS = "docs/journal_v2_a1_curriculum/stage2_v2/"
V1_COMMIT = "b94659d"


def verify_repository() -> dict:
    head = TS.git("rev-parse", "HEAD")
    frozen = ("experiments/journal_v2_a1.py", "experiments/journal_v2_a1_r2.py", "experiments/journal_v2_a1_cur1.py", "experiments/journal_v2_a1_cur1v2.py", "experiments/journal_v2_a1_cur1v2_bc.py",
              "experiments/journal_v2_a1_cur2.py", "experiments/journal_v2_a1_cur2_training.py", "experiments/journal_v2_a1_cur2_final.py", "experiments/journal_v2_a1_cur2_run.py", "uav_defend")
    v1_tracked = TS.git("ls-tree", "-r", "--name-only", V1_COMMIT, "--", "docs/journal_v2_a1_curriculum/stage2").splitlines()
    rec = {"head": head, "expected_start_head": V1_COMMIT, "head_is_expected_or_descendant": subprocess.run(["git", "merge-base", "--is-ancestor", V1_COMMIT, "HEAD"], cwd=TS.PROJECT_ROOT).returncode == 0,
           "branch": TS.git("rev-parse", "--abbrev-ref", "HEAD"), "tracked_tree_clean": TS.git("status", "--porcelain", "--untracked-files=no") == "",
           "uav_defend_identical_to_a0_tag": TS.git("diff", "--stat", "journal-v2-a0-canonical-v1", "--", "uav_defend") == "",
           "frozen_files_identical_to_stage2_v1_commit": TS.git("diff", "--stat", V1_COMMIT, "--", *frozen) == "", "frozen_files": list(frozen),
           "stage2_v1_docs_identical_to_stage2_v1_commit": TS.git("diff", "--stat", V1_COMMIT, "--", "docs/journal_v2_a1_curriculum/stage2") == "", "stage2_v1_tracked_files": len(v1_tracked),
           "stage2_v1_protocol_sha256_stored": V2.v1_protocol_sha256(), "stage2_v1_protocol_sha256_recomputed": S.protocol_sha256(V2.parent_hashes()),
           "reward_version": S.REWARD_VERSION, "observation_version": S.OBSERVATION_VERSION, "action_version": S.ACTION_VERSION, "ppo_hyperparameters": S.A.A1_V2_PPO_HYPERPARAMS,
           "stage2_v1_parents": {str(s): V2.verify_v1_parent(s) for s in S.PPO_SEEDS}, "integrity": RUN1.checkpoint_integrity()}
    assert rec["head_is_expected_or_descendant"] and rec["tracked_tree_clean"] and rec["uav_defend_identical_to_a0_tag"] and rec["frozen_files_identical_to_stage2_v1_commit"]
    assert rec["stage2_v1_docs_identical_to_stage2_v1_commit"] and rec["stage2_v1_protocol_sha256_stored"] == rec["stage2_v1_protocol_sha256_recomputed"]
    assert rec["reward_version"] == "a1_v2_reward_v2" and rec["ppo_hyperparameters"]["n_steps"] == 2048
    return rec


def rel(path: Path) -> str:
    return str(Path(path).resolve().relative_to(TS.PROJECT_ROOT.resolve())).replace("\\", "/")


def run_all() -> None:
    state = TS.load_state()
    psha = V2.protocol_sha256()
    scratch = V2.select_scratch_root()
    print(f"[stage2-v2] scratch={scratch} protocol_sha={psha[:16]}", flush=True)
    if not TS.stage_is_verified(state, "01_repository_verification"):
        TS.begin(state, "01_repository_verification")
        TS.atomic_write_json(TS.STATE_DIR / "repository_verification.json", verify_repository())
        TS.complete(state, "01_repository_verification", [DOCS + "repository_verification.json"], "HEAD >= b94659d, clean, Stage2-v1 parents + integrity verified")
    if not TS.stage_is_verified(state, "02_stage2_v2_protocol_freeze", psha):
        TS.begin(state, "02_stage2_v2_protocol_freeze", psha)
        diff = V2.differences_from_v1()
        if set(diff) != {"ppo.max_rollouts", "ppo.max_timesteps"}:
            raise SystemExit(f"protocol differs from Stage2-v1 beyond the budget: {sorted(diff)}")
        TS.atomic_write_json(TS.STATE_DIR / "protocol.json", V2.protocol_specification() | {"protocol_sha256": psha, "differences_from_stage2_v1": diff})
        TS.complete(state, "02_stage2_v2_protocol_freeze", [DOCS + "protocol.json"], "protocol sha " + psha[:16] + "; only the budget differs from Stage2-v1")
    if not TS.stage_is_verified(state, "03_parent_checkpoint_seeding", psha):
        TS.begin(state, "03_parent_checkpoint_seeding", psha)
        parents = {str(s): V2.seed_from_v1(s, scratch) for s in S.PPO_SEEDS}
        TS.atomic_write_json(TS.STATE_DIR / "parent_checkpoints.json", {"stage2_v1_rollout_8": parents, "scratch": str(scratch)})
        TS.complete(state, "03_parent_checkpoint_seeding", [DOCS + "parent_checkpoints.json"], "rollouts 1-8 of the three seeds verified and seeded", {"parents": {k: v["model_sha256"] for k, v in parents.items()}})
    test_path = TS.STATE_DIR / "unit_test_result.json"
    if not TS.stage_is_verified(state, "04_tests", psha):
        if not test_path.exists() or json.loads(test_path.read_text()).get("failed", 1) != 0:
            raise SystemExit("run the targeted tests first and write unit_test_result.json (failed == 0)")
        TS.begin(state, "04_tests", psha)
        TS.complete(state, "04_tests", [DOCS + "unit_test_result.json"], "targeted tests passed")
    with V2.activated():
        for stage_name, seed in (("05_seed_201_ppo_rollouts_9_16", 201), ("06_seed_202_ppo_rollouts_9_16", 202), ("07_seed_203_ppo_rollouts_9_16", 203)):
            if TS.stage_is_verified(state, stage_name, psha):
                continue
            TS.begin(state, stage_name, psha)
            parent = V2.seed_from_v1(seed, scratch)
            result = T2.train_seed(seed, scratch_root=scratch, spec=V2.SPEC)
            if result.get("status") == "INCOMPLETE":
                raise SystemExit(f"seed {seed} incomplete")
            path = TS.STATE_DIR / f"seed_{seed}_manifest.json"
            TS.atomic_write_json(path, result | {"stage2_v1_parent": parent, "stage2_v2_protocol_sha256": psha})
            last = result["rollouts"][-1]
            TS.complete(state, stage_name, [DOCS + path.name], f"seed {seed}: {result['status']} rollouts={result['rollouts_trained']} mastery_rollout={result['mastery_rollout']}",
                        {"source_checkpoint_sha256": parent["model_sha256"], "completed_rollout": result["rollouts_trained"], "final_model_sha256": last["model_sha256"], "final_state_sha256": last["state_sha256"],
                         "accepted_model_sha256": result["accepted_model_sha256"]})
    print(json.dumps(status(scratch), indent=1))


def status(scratch=None) -> dict:
    scratch = scratch or V2.select_scratch_root()
    out = {}
    with V2.activated():
        for seed in S.PPO_SEEDS:
            layout = T2.make_layout(seed, scratch)
            if layout.seed_manifest.exists():
                m = json.loads(layout.seed_manifest.read_text())
                out[seed] = {"status": m["status"], "mastery_rollout": m["mastery_rollout"], "rollouts_trained": m["rollouts_trained"]}
                continue
            best = 0
            for k in range(V2.MAX_ROLLOUTS, 0, -1):
                if T2.verify_rollout(layout.scratch_rollouts, k, seed=seed, spec=V2.SPEC):
                    best = k
                    break
            out[seed] = {"status": "IN_PROGRESS" if best else "NOT_STARTED", "verified_rollouts": best}
    return out


# ----------------------------------------------------------------------------- final evaluation + analyses
def budget_sensitivity(v1_eval: dict, v2_eval: dict) -> dict:
    out = {}
    for s in S.PPO_SEEDS:
        a1, a2 = v1_eval["per_seed"][str(s)], v2_eval["per_seed"][str(s)]
        o1 = np.array([e["outcome"] == "soldier_caught" for e in a1["episodes"]], dtype=float)
        o2 = np.array([e["outcome"] == "soldier_caught" for e in a2["episodes"]], dtype=float)
        assert [e["scenario_seed"] for e in a1["episodes"]] == [e["scenario_seed"] for e in a2["episodes"]]
        out[str(s)] = {"stage2_v1_capture": float(o1.mean()), "stage2_v2_capture": float(o2.mean()), "paired_v2_minus_v1_capture": S.paired_bootstrap(o2, o1),
                       "stage2_v1_interception": a1["metrics"]["interception"], "stage2_v2_interception": a2["metrics"]["interception"], "stage2_v2_training_status": a2["training_status"]}
    return out


def learning_curves(seeds=S.PPO_SEEDS) -> dict:
    out = {}
    for s in seeds:
        m = json.loads((TS.STATE_DIR / f"seed_{s}_manifest.json").read_text())
        rows = m["rollouts"]
        series = {"capture": [r["validation_metrics"]["capture"] for r in rows], "interception": [r["validation_metrics"]["interception"] for r in rows],
                  "mean_return": [r["validation_metrics"]["mean_return"] for r in rows], "median_path_excess": [r["validation_metrics"]["median_path_excess"] for r in rows],
                  "anchor_capture": [r["anchor_capture"] for r in rows], "train_capture": [r["training_summary"]["capture_rate"] for r in rows]}
        out[str(s)] = {"rollouts": len(rows), "timesteps": [r["num_timesteps"] for r in rows], "series": series, "validation_passed": [r["validation_passed"] for r in rows],
                       "trend_rollouts_9_end": {k: V2.describe_trend(v[8:]) for k, v in series.items() if k in ("capture", "interception", "mean_return", "median_path_excess", "anchor_capture")},
                       "trend_rollouts_1_8": {k: V2.describe_trend(v[:8]) for k, v in series.items() if k in ("capture", "interception")},
                       "trend_train_capture_9_end": V2.describe_trend(series["train_capture"][8:]), "status": m["status"], "mastery_rollout": m["mastery_rollout"]}
    return out


def write_rollout_table(path: Path) -> None:
    L = ["# Stage2-v2 per-rollout results (rollouts 1-8 are the unchanged Stage2-v1 rollouts)", "",
         "| seed | rollout | timesteps | set | train capture | val capture | val interception | A0 capture | improvement (pp) | val mean return | val median path-excess | val exhaustion | val timeout | anchor capture | gate |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in S.PPO_SEEDS:
        m = json.loads((TS.STATE_DIR / f"seed_{s}_manifest.json").read_text())
        for r in m["rollouts"]:
            v, a = r["validation_metrics"], r["a0_metrics"]
            L.append(f"| {s} | {r['rollout']} | {r['num_timesteps']} | {r['validation_set']} | {r['training_summary']['capture_rate']:.3f} | {v['capture']:.2f} | {v['interception']:.2f} | {a['capture']:.2f} | "
                     f"{r['gate']['improvement_over_a0_pp']:.0f} | {v['mean_return']:.2f} | {v['median_path_excess']:.3f} | {v['exhaustion']:.2f} | {v['timeout']:.2f} | {r['anchor_capture']:.2f} | "
                     f"{'PASS' if r['validation_passed'] else 'fail'} |")
    Path(path).write_text("\n".join(L) + "\n", encoding="utf-8")


def extra_curve_figure(curves: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"201": "#1f77b4", "202": "#d62728", "203": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for s, c in curves.items():
        ax.plot(c["timesteps"], c["series"]["mean_return"], marker="o", color=colors[s], label=f"A1-CUR2-{s}")
    ax.axvline(262_144, color="gray", ls=":", label="Stage2-v1 budget (8 rollouts)")
    ax.set_xlabel("PPO timesteps"); ax.set_ylabel("validation mean return"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = []
    for ext in ("png", "pdf"):
        p = F.FIG_DIR / f"fig6_return_vs_timesteps.{ext}"
        fig.savefig(p, dpi=200)
        out.append(rel(p))
    plt.close(fig)
    return out


def run_final() -> None:
    state = TS.load_state()
    psha = V2.protocol_sha256()
    if not all(TS.stage_is_verified(state, n, psha) for n in TS.STAGES[4:7]):
        raise SystemExit("training stages are not complete")
    F.DOCS, F.FIG_DIR = TS.STATE_DIR, TS.STATE_DIR / "figures"
    TS.begin(state, "08_final_evaluation", psha)
    with V2.activated():
        F.run_all()
    v1_eval = json.loads((TS.PROJECT_ROOT / V2.V1_DOCS / "final_evaluation.json").read_text())
    v2_eval = json.loads((TS.STATE_DIR / "final_evaluation.json").read_text())
    sensitivity = budget_sensitivity(v1_eval, v2_eval)
    TS.atomic_write_json(TS.STATE_DIR / "budget_sensitivity.json", sensitivity)
    curves = learning_curves()
    TS.atomic_write_json(TS.STATE_DIR / "learning_curves.json", curves)
    write_rollout_table(TS.STATE_DIR / "rollout_results.md")
    figs = extra_curve_figure(curves)
    decision = json.loads((TS.STATE_DIR / "stage2_decision.json").read_text())
    decision["rule"] = decision["rule"].replace("within 8 rollouts", "within 16 total rollouts (Stage2-v2 budget)")
    decision["figures"] = decision["figures"] + figs
    decision["stage"] = V2.VERSION
    decision["protocol_sha256"] = psha
    TS.atomic_write_json(TS.STATE_DIR / "stage2_decision.json", decision)
    arts = [DOCS + n for n in ("final_evaluation.json", "evasion_analysis.json", "bc_fidelity.json", "stage2_decision.json", "budget_sensitivity.json", "learning_curves.json", "rollout_results.md")]
    TS.complete(state, "08_final_evaluation", arts, f"stage2_pass={decision['stage2_pass']} mastered={decision['seeds_mastered']}")
    print(json.dumps(decision, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "final", "status"))
    args = parser.parse_args(argv)
    if args.command == "all":
        run_all()
    elif args.command == "final":
        run_final()
    else:
        s = TS.load_state()
        print(json.dumps({k: v["status"] for k, v in s["stages"].items()} | {"next": TS.next_stage(s)}, indent=1))


if __name__ == "__main__":
    main()
