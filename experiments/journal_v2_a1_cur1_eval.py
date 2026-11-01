"""Curriculum Stage 1: final held-out evaluation, A0 reference, learning curves, handoff manifest and verdict.

    python -m experiments.journal_v2_a1_cur1_eval evaluate     # per-seed 200-episode held-out evaluation + scripted A0 on the identical scenarios (atomic, restartable)
    python -m experiments.journal_v2_a1_cur1_eval curves       # PNG + PDF learning curves for all three seeds
    python -m experiments.journal_v2_a1_cur1_eval decide       # Stage-1 verdict (+ handoff manifest when passing)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1_state as TS
from experiments import journal_v2_a1_cur1_training as TR

OUT_DIR = TS.STATE_DIR
FIG_DIR = OUT_DIR / "figures"
EVAL_PATH = OUT_DIR / "final_evaluation.json"
HANDOFF_PATH = OUT_DIR / "stage2_handoff_manifest.json"
DECISION_PATH = OUT_DIR / "stage1_decision.json"
COMMON_PATHOLOGY_RATE = 0.01      # a pathology category in > 1% of a seed's 200 episodes ...
COMMON_PATHOLOGY_SEEDS = 2        # ... in at least two seeds = a common structural pathology


def seed_manifests(scratch_root: Path | None = None) -> dict:
    out = {}
    for seed in C.PPO_SEEDS:
        layout = TR.make_layout(seed, scratch_root or TR.select_scratch_root())
        if layout.seed_manifest.exists():
            out[seed] = json.loads(layout.seed_manifest.read_text())
    return out


def _eval_chunk(args):
    model_path, seeds, scripted_a0 = args
    if scripted_a0:
        return C.evaluate(None, seeds, scripted_a0=True)
    from stable_baselines3 import PPO

    model = PPO.load(model_path.removesuffix(".zip"), device="cpu")
    return C.evaluate(C.model_actor(model), seeds)


def parallel_evaluate(model_path: str | None, seeds: list[int], *, scripted_a0: bool = False, workers: int = 18) -> list[dict]:
    """Deterministic evaluation split into contiguous chunks over a spawn pool; the result order equals `seeds`."""
    import multiprocessing as mp

    size = max(1, int(np.ceil(len(seeds) / workers)))
    chunks = [seeds[i:i + size] for i in range(0, len(seeds), size)]
    with mp.get_context("spawn").Pool(min(workers, len(chunks))) as pool:
        parts = pool.map(_eval_chunk, [(model_path, chunk, scripted_a0) for chunk in chunks])
    return [e for part in parts for e in part]


def evaluate_all(scratch_root: Path | None = None) -> dict:
    seeds = C.final_evaluation_seeds()
    existing = json.loads(EVAL_PATH.read_text()) if EVAL_PATH.exists() else {}
    result = {"seeds": [seeds[0], seeds[-1]], "episodes": len(seeds), "gate": C.GATE, "per_seed": existing.get("per_seed", {}), "a0": existing.get("a0")}
    if result["a0"] is None:
        a0 = parallel_evaluate(None, seeds, scripted_a0=True)
        result["a0"] = {"metrics": C.metrics(a0), "gate": C.gate_status(C.metrics(a0)), "episodes_summary": _brief(a0)}
        TS.atomic_write_json(EVAL_PATH, result)
    for seed, m in seed_manifests(scratch_root).items():
        key = str(seed)
        layout = TR.make_layout(seed, scratch_root or TR.select_scratch_root())
        if key in result["per_seed"] and result["per_seed"][key].get("accepted_model_sha256") == m["accepted_model_sha256"]:
            continue
        sha = TR.T.sha256_file(layout.accepted_model)
        if sha != m["accepted_model_sha256"]:
            raise TR.RunConflictError(f"accepted model of seed {seed} does not match its manifest")
        episodes = parallel_evaluate(str(layout.accepted_model), seeds)
        met = C.metrics(episodes)
        result["per_seed"][key] = {"seed": seed, "status": m["status"], "mastery_rollout": m["mastery_rollout"], "timesteps_to_mastery": m["timesteps_to_mastery"],
                                   "accepted_rollout": m["accepted_rollout"], "model_role": m["model_role"], "accepted_model_sha256": sha, "metrics": met,
                                   "gate": C.gate_status(met), "episodes_summary": _brief(episodes)}
        TS.atomic_write_json(EVAL_PATH, result)
    return result


def _brief(episodes: list[dict]) -> list[dict]:
    keys = ("scenario_seed", "outcome", "steps", "path_length_m", "chord_m", "path_excess", "initial_range_m", "terminal_distance_m", "initial_altitude_m",
            "initial_heading_offset_deg", "return_undiscounted", "overshoots", "orbit_revolutions_max", "nonclosing_run", "pathology")
    return [{k: e[k] for k in keys} for e in episodes]


def decide(evaluation: dict, manifests: dict) -> dict:
    mastered = [s for s, m in manifests.items() if m["status"] == "MASTERED"]
    final_ok = {s: evaluation["per_seed"][str(s)]["gate"]["passed"] for s in mastered if str(s) in evaluation["per_seed"]}
    categories = {"orbit": "orbit_episodes", "repeated_flyby": "repeated_flyby_episodes", "nonclosing": "nonclosing_episodes"}
    common = {}
    for name, key in categories.items():
        hit = [s for s in mastered if str(s) in evaluation["per_seed"] and evaluation["per_seed"][str(s)]["metrics"][key] / evaluation["episodes"] > COMMON_PATHOLOGY_RATE]
        if len(hit) >= COMMON_PATHOLOGY_SEEDS:
            common[name] = hit
    passed = bool(len(mastered) >= 2 and all(final_ok.get(s, False) for s in mastered) and not common)
    return {"mastered_seeds": mastered, "seeds_failed": [s for s, m in manifests.items() if m["status"] == "FAIL"], "final_heldout_gates_passed": final_ok,
            "common_structural_pathology": common, "stage1_pass": passed, "all_three_mastered": len(mastered) == 3,
            "rule": ">= 2/3 seeds mastered AND every mastered seed passes the final 200-episode gate AND no common structural pathology"}


def handoff(manifests: dict, evaluation: dict, decision: dict) -> dict:
    return {
        "note": "Stage-1 checkpoints are curriculum initialization checkpoints, not final A1 evaluation models.", "stage": "journal_v2_a1_curriculum/stage1",
        "next_stage": "Stage 2: stationary asset, C0, ACTIVE stationary defender (not started)",
        "versions": {"action": C.ACTION_VERSION, "observation": C.OBSERVATION_VERSION, "reward": C.REWARD_VERSION, "training": C.TRAINING_PROTOCOL},
        "stationary_asset_config": {"differences_from_canonical_C0_V2": C.config_differences_from_canonical(), "env_config_sha256": C.config_sha256(), "defender_park": list(C.PARKED_DEFENDER)},
        "protocol_sha256": C.protocol_sha256(), "ppo_hyperparameters": TR.A.A1_V2_PPO_HYPERPARAMS, "rng_namespaces": {"scenario_stream_tag": hex(C.SCENARIO_STREAM_TAG),
                                                                                                                   "heading_stream_tag": hex(C.HEADING_STREAM_TAG), "validation": [C.VALIDATION_A, C.VALIDATION_B]},
        "accepted_checkpoints": {str(s): {"path": m["accepted_model"], "model_sha256": m["accepted_model_sha256"], "state_sha256": m["accepted_state_sha256"],
                                          "mastery_rollout": m["mastery_rollout"], "timesteps_to_mastery": m["timesteps_to_mastery"], "results_dir": str(TR.make_layout(s, Path("."), None).canonical)}
                                 for s, m in manifests.items() if m["status"] == "MASTERED"},
        "evaluation": {str(s): evaluation["per_seed"][str(s)]["metrics"] for s in decision["mastered_seeds"]}, "a0_reference": evaluation["a0"]["metrics"], "decision": decision,
    }


def plot_curves(manifests: dict, out_dir: Path = FIG_DIR) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {201: "#1f77b4", 202: "#d62728", 203: "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    panels = [("Validation capture rate", lambda r: r["validation"]["capture"], (0, 1.02), C.GATE["capture_min"]),
              ("Validation median path-excess", lambda r: r["validation"]["median_path_excess"], None, C.GATE["median_path_excess_max"]),
              ("Validation exhaustion (solid) / timeout (dashed)", None, (0, 1.02), C.GATE["exhaustion_max"]),
              ("Mean training episode return", lambda r: r["training_summary"].get("mean_return", np.nan), None, None)]
    for ax, (title, getter, ylim, ref) in zip(axes.ravel(), panels):
        for seed, m in sorted(manifests.items()):
            x = [r["num_timesteps"] for r in m["rollouts"]]
            if getter is None:
                ax.plot(x, [r["validation"]["exhaustion"] for r in m["rollouts"]], "-o", color=colors[seed], label=f"seed {seed} exhaustion", ms=4)
                ax.plot(x, [r["validation"]["timeout"] for r in m["rollouts"]], "--s", color=colors[seed], label=f"seed {seed} timeout", ms=4, alpha=0.8)
            else:
                ax.plot(x, [getter(r) for r in m["rollouts"]], "-o", color=colors[seed], label=f"seed {seed}", ms=4)
            if m["status"] == "MASTERED":
                ax.axvline(m["timesteps_to_mastery"], color=colors[seed], ls=":", alpha=0.7)
        if ref is not None:
            ax.axhline(ref, color="k", ls="-.", lw=0.8, alpha=0.6)
        if ylim:
            ax.set_ylim(*ylim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("PPO timesteps")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("Curriculum Stage 1 (stationary asset, C0): learning curves, seeds 201/202/203\n(dotted verticals: mastery; dash-dot: gate threshold; validation alternates sets A/B)", fontsize=10)
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"stage1_learning_curves.{ext}"
        fig.savefig(path, dpi=200 if ext == "png" else None)
        paths.append(str(path))
    plt.close(fig)
    return paths


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("evaluate", "curves", "decide"))
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        r = evaluate_all()
        print(json.dumps({"a0": {k: r["a0"]["metrics"][k] for k in ("capture", "median_path_excess", "mean_duration_s")},
                          "seeds": {s: {k: v["metrics"][k] for k in ("capture", "exhaustion", "timeout", "median_path_excess", "gt2", "gt5")} | {"gate": v["gate"]["passed"]} for s, v in r["per_seed"].items()}}, indent=1))
    elif args.command == "curves":
        print(plot_curves(seed_manifests()))
    else:
        evaluation, manifests = json.loads(EVAL_PATH.read_text()), seed_manifests()
        decision = decide(evaluation, manifests)
        TS.atomic_write_json(DECISION_PATH, decision)
        if decision["stage1_pass"]:
            TS.atomic_write_json(HANDOFF_PATH, handoff(manifests, evaluation, decision))
        print(json.dumps(decision, indent=1))


if __name__ == "__main__":
    main()
