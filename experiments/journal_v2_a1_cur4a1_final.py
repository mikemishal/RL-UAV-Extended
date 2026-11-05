"""Final evaluation + analysis for curriculum Stage 4A-1 (run after BC fine-tuning; `python -m experiments.journal_v2_a1_cur4a1_final all`).

* every seed's accepted policy on 200 matched TRANSLATED held-out scenarios, together with canonical A0 and the seed's own (untranslated-trained) Stage3-v2 parent on the SAME scenarios;
* retention anchors across epochs and at the accepted checkpoint;
* POST representation sensitivity (asset vs attacker input dependence) compared with the PRE (parent) snapshot from stage 5, and the representation-gate decision;
* the translation-GENERALIZATION test (8 quadrant x magnitude strata, NOT used in BC/held-out), capture/cosine by region;
* the Stage-4A-1 decision and figures 1-6 (PNG + PDF).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur4a1 as Z
from experiments import journal_v2_a1_cur4a1_state as TS

DOCS = TS.STATE_DIR
FIG_DIR = DOCS / "figures"


def accepted_model_path(seed: int) -> Path:
    return Z.RESULTS_ROOT / f"seed_{seed}" / "accepted_model.zip"


def seed_manifest(seed: int) -> dict:
    return json.loads((Z.RESULTS_ROOT / f"seed_{seed}" / "seed_manifest.json").read_text())


def _model_actor(path: str):
    from stable_baselines3 import PPO

    return C.model_actor(PPO.load(path.removesuffix(".zip"), device="cpu"))


def _model(path: str):
    from stable_baselines3 import PPO

    return PPO.load(path.removesuffix(".zip"), device="cpu")


def anchors_of(act) -> dict:
    an1 = Z.metrics(Z.evaluate_anchor1(act))
    an2 = Z.anchor_metrics_static(Z.evaluate_anchor2(act))
    an3 = Z.anchor_metrics_moving(Z.evaluate_anchor3(act))
    return {"anchor1_basic_pursuit": an1["capture"], "anchor2_static_defender": an2["capture"], "anchor3_moving_defender": an3["capture"]}


def compact(ep: dict) -> dict:
    keys = ("scenario_seed", "outcome", "steps", "duration_s", "path_length_m", "path_excess", "terminal_distance_m", "translation_m", "asset_position_m", "expert_cosines")
    out = {k: ep[k] for k in keys if k in ep}
    if "expert_cosines" in out:
        out["expert_cosine_mean"] = float(np.mean(out["expert_cosines"])) if out["expert_cosines"] else None
        del out["expert_cosines"]
    return out


def evaluate_seed(seed: int, seeds: list[int], a0_eps: list[dict]) -> dict:
    act = _model_actor(str(accepted_model_path(seed)))
    model = _model(str(accepted_model_path(seed)))
    pact = _model_actor(Z.parent_checkpoint(seed)["path"])
    eps = Z.evaluate_translated(act, seeds)
    peps = Z.evaluate_translated(pact, seeds)
    m, m0, mp = Z.translated_metrics(eps), Z.translated_metrics(a0_eps), Z.translated_metrics(peps)
    anchors = anchors_of(act)
    gate = Z.gate_status(m, anchor1=anchors["anchor1_basic_pursuit"], anchor2=anchors["anchor2_static_defender"], anchor3=anchors["anchor3_moving_defender"])
    manifest = seed_manifest(seed)
    rep_post = Z.representation_record(model, held_out_seeds=seeds)
    return {"seed": seed, "accepted_model_sha256": manifest["accepted_model_sha256"], "training_status": manifest["status"], "mastery_epoch": manifest["mastery_epoch"], "metrics": m,
            "a0_metrics": m0, "parent_metrics": mp, "anchor_metrics": anchors, "gate_on_200": gate, "representation_post": rep_post, "episodes": [compact(e) for e in eps],
            "parent_episodes": [compact(e) for e in peps]}


def pre_sensitivity() -> dict:
    return json.loads((DOCS / "pre_sensitivity.json").read_text())


def representation_improved(seed_result: dict, pre: dict) -> dict:
    """Representation success gate (frozen before training): POST asset/attacker sensitivity ratio >= 2x PRE, and no translation-generalization region captures below 95%."""
    pre_ratio = pre["sensitivity"]["asset_to_attacker_sensitivity_ratio"]
    post_ratio = seed_result["representation_post"]["sensitivity"]["asset_to_attacker_sensitivity_ratio"]
    gain = post_ratio / pre_ratio if pre_ratio else float("inf")
    regions = seed_result["generalization"]["regions"]
    region_min = min((r["capture"] for r in regions.values() if r["capture"] is not None), default=0.0)
    return {"pre_ratio": pre_ratio, "post_ratio": post_ratio, "sensitivity_gain": gain, "sensitivity_gate": bool(gain >= Z.SENSITIVITY_RATIO_MIN_GAIN),
            "region_min_capture": region_min, "region_gate": bool(region_min >= Z.QUADRANT_CAPTURE_MIN), "passed": bool(gain >= Z.SENSITIVITY_RATIO_MIN_GAIN and region_min >= Z.QUADRANT_CAPTURE_MIN)}


def generalization_for_seed(seed: int, per_stratum: int = 40) -> dict:
    act = _model_actor(str(accepted_model_path(seed)))
    pact = _model_actor(Z.parent_checkpoint(seed)["path"])
    table = Z.generalization_scenario_table(per_stratum=per_stratum)
    regions, parent_regions = {}, {}
    for name, target in Z.generalization_strata().items():
        eps_a1 = [e for s in table[name] for e in Z.evaluate_translated(act, [s], target=target, jitter_m=3.0)]
        eps_p = [e for s in table[name] for e in Z.evaluate_translated(pact, [s], target=target, jitter_m=3.0)]
        regions[name] = Z.translated_metrics(eps_a1) if eps_a1 else {"capture": None, "episodes": 0}
        parent_regions[name] = Z.translated_metrics(eps_p) if eps_p else {"capture": None, "episodes": 0}
    return {"scenario_table": table, "regions": {k: {kk: regions[k].get(kk) for kk in ("capture", "expert_cosine_mean", "episodes")} for k in regions},
            "parent_regions": {k: {kk: parent_regions[k].get(kk) for kk in ("capture", "expert_cosine_mean", "episodes")} for k in parent_regions}}


def retention_across_epochs(seed: int) -> dict:
    manifest = seed_manifest(seed)
    rows = [{"epoch": e["epoch"], "train_loss": e["train_loss"], "val_loss": e["val"]["loss"], "goal_cos_mean": e["val"]["goal"]["cos_mean"], "retain_cos_mean": e["val"]["retain"]["cos_mean"],
             "capture": e["validation"]["metrics"]["capture"], "anchor1": e["validation"]["anchor1_capture"], "anchor2": e["validation"]["anchor2_capture"], "anchor3": e["validation"]["anchor3_capture"],
             "passed": e["validation"]["passed"]} for e in manifest["epochs"]]
    return {"seed": seed, "rows": rows}


def decide(per_seed: dict, trainings: dict, pre: dict) -> dict:
    mastered = [s for s, t in trainings.items() if t["status"] == "MASTERED"]
    retained = [s for s in mastered if all(per_seed[s]["anchor_metrics"][k] >= v for k, v in (("anchor1_basic_pursuit", Z.ANCHOR_MIN["anchor1_basic_pursuit"]),
                                                                                              ("anchor2_static_defender", Z.ANCHOR_MIN["anchor2_static_defender"]),
                                                                                              ("anchor3_moving_defender", Z.ANCHOR_MIN["anchor3_moving_defender"])))]
    rep = {s: representation_improved(per_seed[s], pre[str(s)]) for s in per_seed}
    rep_pass = [s for s in retained if rep[s]["passed"]]
    pathology = {}
    for key in ("orbit_episodes", "gt5", "nonfinite"):
        hit = [s for s in rep_pass if per_seed[s]["metrics"].get(key, 0) > 0]
        if len(hit) >= 2:
            pathology[key] = hit
    passed = len(mastered) >= 2 and len(retained) >= 2 and len(rep_pass) >= 2 and not pathology
    improving = [s for s in per_seed if per_seed[s]["metrics"]["capture"] >= 0.80 and rep[s]["sensitivity_gain"] >= 1.5]
    promising = (not passed) and (len(mastered) >= 1 or len(improving) >= 2)
    verdict = "A" if passed else ("B" if promising else "C")
    return {"seeds_mastered": mastered, "seeds_retaining_all_anchors": retained, "seeds_passing_representation_gate": rep_pass, "training_status": {str(s): t["status"] for s, t in trainings.items()},
            "representation": {str(s): v for s, v in rep.items()}, "common_structural_pathology": pathology, "stage4a1_pass": bool(passed), "all_three_mastered": len(mastered) == 3,
            "promising_seeds": improving, "needs_manual_review": bool(promising), "verdict": verdict,
            "rule": ">= 2/3 seeds master (2 consecutive epoch passes), retain all three anchors, AND pass the representation gate (sensitivity ratio >= 2x PRE, no region < 95%); no shared pathology; "
                    "verdict B if not passed but >= 1 mastered or >= 2 seeds with capture >= 0.80 and sensitivity gain >= 1.5x"}


# ----------------------------------------------------------------------------- figures
def rel_fig(p: Path) -> str:
    return str(p.resolve().relative_to(TS.PROJECT_ROOT.resolve())).replace("\\", "/")


def make_figures(trainings: dict, per_seed: dict, pre: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    colors = {201: "#1f77b4", 202: "#d62728", 203: "#2ca02c"}

    def save(fig, name):
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = FIG_DIR / f"{name}.{ext}"
            fig.savefig(p, dpi=200)
            out.append(rel_fig(p))
        plt.close(fig)

    rows = {s: retention_across_epochs(s)["rows"] for s in trainings}
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for s, rs in rows.items():
        ax.plot([r["epoch"] for r in rs], [r["capture"] for r in rs], marker="o", color=colors[s], label=f"A1-CUR4A1-{s}")
    ax.axhline(Z.GATE["capture_min"], color="gray", ls=":", label=f"mastery {Z.GATE['capture_min']}")
    ax.set_xlabel("BC epoch"); ax.set_ylabel("translated-target capture (40-scenario epoch check)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "fig1_translated_capture_vs_epoch")

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for s, rs in rows.items():
        ax.plot([r["epoch"] for r in rs], [r["goal_cos_mean"] for r in rs], marker="o", color=colors[s], label=f"A1-CUR4A1-{s}")
    ax.axhline(Z.GATE["cosine_mean_min"], color="gray", ls=":", label=f"mastery {Z.GATE['cosine_mean_min']}")
    ax.set_xlabel("BC epoch"); ax.set_ylabel("mean cosine to A0 (goal validation set)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "fig2_a0_cosine_vs_epoch")

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    seeds_ = sorted(per_seed)
    pre_r = [pre[str(s)]["sensitivity"]["asset_to_attacker_sensitivity_ratio"] * 100 for s in seeds_]
    post_r = [per_seed[s]["representation_post"]["sensitivity"]["asset_to_attacker_sensitivity_ratio"] * 100 for s in seeds_]
    width = 0.35
    xs = np.arange(len(seeds_))
    ax.bar(xs - width / 2, pre_r, width, label="PRE (Stage3-v2 parent)", color="#999999")
    ax.bar(xs + width / 2, post_r, width, label="POST (Stage4A-1)", color=[colors[s] for s in seeds_])
    ax.set_xticks(xs); ax.set_xticklabels([str(s) for s in seeds_]); ax.set_ylabel("asset/attacker input-sensitivity ratio [%]"); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    save(fig, "fig3_sensitivity_pre_vs_post")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    for ax, (key, title, thr) in zip(axes, (("anchor1", "Anchor 1: fixed-origin pursuit", 0.99), ("anchor2", "Anchor 2: static defender", 0.80), ("anchor3", "Anchor 3: moving defender", 0.75))):
        for s, rs in rows.items():
            ax.plot([r["epoch"] for r in rs], [r[key] for r in rs], marker="o", color=colors[s], label=f"A1-CUR4A1-{s}")
        ax.axhline(thr, color="gray", ls=":"); ax.set_title(title, fontsize=10); ax.set_xlabel("BC epoch"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("anchor capture"); axes[0].legend(fontsize=7)
    save(fig, "fig4_retention_anchors_vs_epoch")

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    strata = list(Z.generalization_strata())
    for s in seeds_:
        caps = [per_seed[s]["generalization"]["regions"][name]["capture"] for name in strata]
        ax.plot(strata, caps, marker="o", color=colors[s], label=f"A1-CUR4A1-{s}")
    ax.axhline(Z.QUADRANT_CAPTURE_MIN, color="gray", ls=":", label=f"gate {Z.QUADRANT_CAPTURE_MIN}")
    ax.set_ylabel("capture"); ax.tick_params(axis="x", rotation=60); ax.grid(alpha=0.3); ax.legend(fontsize=7)
    save(fig, "fig5_capture_by_translation_region")
    return out


def make_trajectory_figure(per_seed: dict, seed_pick: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = per_seed[seed_pick]
    act = _model_actor(str(accepted_model_path(seed_pick)))
    pact = _model_actor(Z.parent_checkpoint(seed_pick)["path"])
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 9.0), subplot_kw={"projection": "3d"})
    picks = [("+x/+y", (10.6, 10.6)), ("-x/-y", (-24.7, -24.7)), ("+x/-y outer", (24.7, -24.7)), ("-x/+y inner", (-10.6, 10.6))]
    for ax, (label, target) in zip(axes.ravel(), picks):
        table = Z.stratified_scenarios(label, target, 1, jitter_m=3.0)
        if not table:
            continue
        env = Z.TranslatedStage1V2Env(0, 0, 0)
        envp = Z.TranslatedStage1V2Env(0, 0, 0)
        obs, _ = env.reset(options={"scenario": table[0], "target_translation": target, "translation_jitter_m": 3.0})
        obsp, _ = envp.reset(options={"scenario": table[0], "target_translation": target, "translation_jitter_m": 3.0})
        pos, posp, done, donep = [np.asarray(env.soldier_env._enemy_pos, dtype=np.float64)], [np.asarray(envp.soldier_env._enemy_pos, dtype=np.float64)], False, False
        while not done:
            obs, _r, te, tr, info = env.step(np.asarray(act(obs), dtype=np.float32))
            pos.append(np.asarray(info["enemy_pos"], dtype=np.float64)); done = te or tr
        while not donep:
            obsp, _r, te, tr, info = envp.step(np.asarray(pact(obsp), dtype=np.float32))
            posp.append(np.asarray(info["enemy_pos"], dtype=np.float64)); donep = te or tr
        pos, posp = np.array(pos), np.array(posp)
        ax.plot(*pos.T, "-o", ms=3, color="#2ca02c", label="Stage4A-1")
        ax.plot(*posp.T, "-s", ms=3, color="#ff7f0e", label="Stage3-v2 parent")
        ax.scatter(*env._asset0, color="blue", marker="*", s=100, label="asset")
        ax.set_title(label, fontsize=9); ax.legend(fontsize=6)
    fig.suptitle(f"seed {seed_pick}: same relative geometry, translated asset positions")
    fig.tight_layout()
    out = []
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"fig6_translated_trajectories_{seed_pick}.{ext}"
        fig.savefig(p, dpi=200)
        out.append(rel_fig(p))
    plt.close(fig)
    return out


def checkpoint_integrity() -> dict:
    out = {"through_stage4a": "verified by 01_repository_verification (see repository_verification.json)", "stage4a1": {}}
    for s in Z.PPO_SEEDS:
        root = Z.RESULTS_ROOT / f"seed_{s}"
        m = seed_manifest(s)
        out["stage4a1"][f"seed_{s}"] = {"status": m["status"], "accepted_sha256": TS.sha256_file(root / "accepted_model.zip"), "matches_manifest": TS.sha256_file(root / "accepted_model.zip") == m["accepted_model_sha256"],
                                        "parent_sha256": m["parent"]["model_sha256"], "parent_matches_preregistered": m["parent"]["model_sha256"] == Z.PARENT_SHA256[s]}
    out["all_ok"] = all(v["matches_manifest"] and v["parent_matches_preregistered"] for v in out["stage4a1"].values())
    return out


def run_all() -> None:
    seeds = Z.final_evaluation_seeds()
    trainings = {s: seed_manifest(s) for s in Z.PPO_SEEDS}
    a0_eps = Z.evaluate_translated(None, seeds, scripted_a0=True)
    per_seed = {s: evaluate_seed(s, seeds, a0_eps) for s in Z.PPO_SEEDS}
    for s in Z.PPO_SEEDS:
        per_seed[s]["generalization"] = generalization_for_seed(s)
    pre = pre_sensitivity()
    decision = decide(per_seed, trainings, pre)
    figs = make_figures(trainings, per_seed, pre)
    pick = 203 if 203 in decision["seeds_mastered"] else max(per_seed, key=lambda s: per_seed[s]["metrics"]["capture"])
    figs += make_trajectory_figure(per_seed, pick)
    TS.atomic_write_json(DOCS / "final_evaluation.json", {"scenarios": [seeds[0], seeds[-1]], "episodes": len(seeds), "a0": {"metrics": Z.translated_metrics(a0_eps), "episodes": [compact(e) for e in a0_eps]},
                                                           "per_seed": {str(s): {k: v for k, v in r.items() if k not in ("episodes", "parent_episodes")} for s, r in per_seed.items()}})
    TS.atomic_write_json(DOCS / "sensitivity_pre_post.json", {str(s): {"pre": pre[str(s)]["sensitivity"], "post": per_seed[s]["representation_post"]["sensitivity"],
                                                                    "gate": decision["representation"][str(s)]} for s in Z.PPO_SEEDS})
    TS.atomic_write_json(DOCS / "retention_evaluation.json", {str(s): retention_across_epochs(s) for s in Z.PPO_SEEDS})
    TS.atomic_write_json(DOCS / "generalization_evaluation.json", {str(s): {"regions": per_seed[s]["generalization"]["regions"], "parent_regions": per_seed[s]["generalization"]["parent_regions"]}
                                                                 for s in Z.PPO_SEEDS})
    TS.atomic_write_json(DOCS / "stage4a1_decision.json", decision | {"figures": figs, "trajectory_figure_seed": pick})
    TS.atomic_write_json(DOCS / "checkpoint_integrity.json", checkpoint_integrity())
    print(json.dumps(decision, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "integrity"))
    args = parser.parse_args(argv)
    if args.command == "integrity":
        TS.atomic_write_json(DOCS / "checkpoint_integrity.json", checkpoint_integrity())
    else:
        run_all()


if __name__ == "__main__":
    main()
