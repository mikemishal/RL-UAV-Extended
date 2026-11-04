"""Final evaluation + analysis for curriculum Stage 4A (run after training; `python -m experiments.journal_v2_a1_cur4a_final all`).

* every seed's accepted / final policy on the 200 matched held-out scenarios 169000-169199 together with canonical A0 AND the seed's own Stage3-v2 parent on the SAME scenarios (identical realized asset random walk,
  verified); paired Stage4A - parent and Stage4A - A0 capture differences (paired bootstrap 95% CI); the failed Stage-4 checkpoint of the same seed is evaluated on the same scenarios as context only;
* moving-target tracking analysis: per decision, the cosine between the executed direction and the direction to the CURRENT asset vs to the INITIAL asset location (an obsolete target), restricted to the
  decisions where the two directions are actually distinguishable; closing velocity, closest approaches and miss distances, overshoots, first-pass capture, asset displacement;
* retention anchors on the final checkpoint, the Stage-4A decision (pre-registered rules) and figures 1-5 (PNG + PDF).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur4 as P
from experiments import journal_v2_a1_cur4_final as F4
from experiments import journal_v2_a1_cur4a as Y
from experiments import journal_v2_a1_cur4a_state as TS

DOCS = TS.STATE_DIR
FIG_DIR = DOCS / "figures"
DT = 0.5
FINAL_APPROACH = 3
DISTINGUISHABLE_DEG = 3.0       # decisions where the current and the initial asset locations differ by at least this angle as seen from the attacker


def accepted_model_path(seed: int) -> Path:
    return Y.RESULTS_ROOT / f"seed_{seed}" / "checkpoints" / "accepted_model.zip"


def seed_manifest(seed: int) -> dict:
    return json.loads((Y.RESULTS_ROOT / f"seed_{seed}" / "checkpoints" / "seed_manifest.json").read_text())


def tracking_row(trace: dict, episode: dict) -> dict:
    """Per-episode tracking features. `cos_current` / `cos_initial` = cosine between the executed direction and the direction to the asset's CURRENT / INITIAL position at each decision."""
    pos, u, los = trace["pos"][:-1], trace["u"], trace["los"]
    a0 = trace["asset_trace"][0]
    init_dir = np.array([F4._unit(a0 - p) for p in pos])
    cos_cur = np.einsum("ij,ij->i", u, los)
    cos_init = np.einsum("ij,ij->i", u, init_dir)
    separation = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", los, init_dir), -1.0, 1.0)))
    rng = np.linalg.norm(trace["asset_at_decision"] - pos, axis=1)
    closing = -np.diff(np.append(rng, np.linalg.norm(trace["asset"] - trace["pos"][-1]))) / DT
    tail = slice(max(len(u) - FINAL_APPROACH, 0), len(u))
    informative = separation >= DISTINGUISHABLE_DEG
    return {"scenario_seed": episode["scenario_seed"], "outcome": episode["outcome"], "steps": episode["steps"], "cos_current_mean": float(cos_cur.mean()), "cos_initial_mean": float(cos_init.mean()),
            "cos_current_final": float(cos_cur[tail].mean()), "cos_initial_final": float(cos_init[tail].mean()), "heading_error_current_final_deg": float(np.degrees(np.arccos(np.clip(cos_cur[tail], -1, 1))).mean()),
            "informative_decisions": int(informative.sum()), "informative_current_better": int((cos_cur[informative] > cos_init[informative]).sum()),
            "informative_cos_gain_sum": float((cos_cur[informative] - cos_init[informative]).sum()), "separation_final_deg": float(separation[tail].mean()), "mean_closing_speed_mps": float(closing.mean()),
            "overshoots": int(episode["overshoots"]), "closest_approach_count": int(episode["closest_approach_count"]), "miss_distances_m": [float(x) for x in episode["miss_distances_m"]],
            "first_pass": int(episode["overshoots"]) == 0, "asset_final_displacement_m": float(episode["asset_final_displacement_m"]), "asset_displacement_max_m": float(episode["asset_displacement_max_m"]),
            "path_excess": float(episode["path_excess"])}


def summarize_tracking(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}

    def stat(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        return {"mean": float(v.mean()), "median": float(np.median(v)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    inf_n = sum(r["informative_decisions"] for r in rows)
    inf_good = sum(r["informative_current_better"] for r in rows)
    misses = [m for r in rows for m in r["miss_distances_m"]]
    return {"n": len(rows), "cos_current_mean": stat("cos_current_mean"), "cos_initial_mean": stat("cos_initial_mean"), "cos_current_final": stat("cos_current_final"), "cos_initial_final": stat("cos_initial_final"),
            "heading_error_current_final_deg": stat("heading_error_current_final_deg"), "mean_closing_speed_mps": stat("mean_closing_speed_mps"), "asset_final_displacement_m": stat("asset_final_displacement_m"),
            "asset_displacement_max_m": stat("asset_displacement_max_m"), "overshoots": stat("overshoots"), "closest_approach_count": stat("closest_approach_count"),
            "first_pass_fraction": float(np.mean([r["first_pass"] for r in rows])), "miss_distance_m": ({"n": len(misses), "median": float(np.median(misses)), "p5": float(np.percentile(misses, 5)),
                                                                                                         "p95": float(np.percentile(misses, 95))} if misses else {"n": 0}),
            "informative_decisions": inf_n, "informative_fraction_current_better": (inf_good / inf_n) if inf_n else None,
            "informative_mean_cos_gain": (sum(r["informative_cos_gain_sum"] for r in rows) / inf_n) if inf_n else None,
            "definitions": {"cos_current / cos_initial": "cosine of the executed direction with the direction to the asset's CURRENT / INITIAL (obsolete) position at each decision",
                            "informative decisions": f"decisions where the two target directions differ by >= {DISTINGUISHABLE_DEG} deg as seen from the attacker (otherwise tracking is indistinguishable from aiming at the start point)",
                            "informative_fraction_current_better": "fraction of informative decisions with cos_current > cos_initial; 1.0 = always aims toward the current rather than the obsolete location"}}


def compact(ep: dict) -> dict:
    keys = ("scenario_seed", "outcome", "steps", "duration_s", "path_length_m", "path_excess", "overshoots", "closest_approach_count", "miss_distances_m", "terminal_distance_m", "min_range_m",
            "asset_final_displacement_m", "asset_displacement_max_m", "initial_heading_offset_deg", "endurance_exhausted", "orbit_revolutions_max")
    return {k: ep[k] for k in keys if k in ep}


def outcome_breakdown(eps: list[dict]) -> dict:
    n = len(eps)
    return {name: {"n": len(sub), "fraction": len(sub) / n, "median_steps": float(np.median([e["steps"] for e in sub])), "median_asset_final_displacement_m": float(np.median([e["asset_final_displacement_m"] for e in sub])),
                   "median_terminal_distance_m": float(np.median([e["terminal_distance_m"] for e in sub])), "median_closest_approaches": float(np.median([e["closest_approach_count"] for e in sub]))}
            for name in sorted({e["outcome"] for e in eps}) for sub in [[e for e in eps if e["outcome"] == name]]}


def _model_actor(path: str):
    from stable_baselines3 import PPO

    return C.model_actor(PPO.load(path.removesuffix(".zip"), device="cpu"))


def anchors_of(act) -> dict:
    return {"anchor1": V1V2.metrics(Y.evaluate_anchor1(act)), "anchor2": Y.anchor_metrics(Y.evaluate_anchor2(act)), "anchor3": Y.anchor_metrics(Y.evaluate_anchor3(act))}


def evaluate_seed(seed: int, seeds: list[int], a0_eps: list[dict]) -> dict:
    act, pact = _model_actor(str(accepted_model_path(seed))), _model_actor(Y.parent_checkpoint(seed)["path"])
    fact = _model_actor(str(P.RESULTS_ROOT / f"seed_{seed}" / "checkpoints" / "accepted_model.zip"))       # failed Stage-4 checkpoint: context only
    env, penv, aenv = Y.Stage4AEnv(0, 0, 0), Y.Stage4AEnv(0, 0, 0), Y.Stage4AEnv(0, 0, 0, scripted_a0=True)
    eps, traces, peps, ptraces, atr = [], [], [], [], {}
    for s in seeds:
        ep, tr = F4.traced_episode(env, s, act)
        pe, ptr = F4.traced_episode(penv, s, pact)
        _ae, at = F4.traced_episode(aenv, s, lambda _o: np.zeros(3, dtype=np.float32))
        eps.append(ep), traces.append(tr), peps.append(pe), ptraces.append(ptr)
        atr[s] = at["asset_trace"]
    crn = Y.crn_report({"a0": atr, "stage3_parent": {s: t["asset_trace"] for s, t in zip(seeds, ptraces)}, "stage4a": {s: t["asset_trace"] for s, t in zip(seeds, traces)}})
    feps = Y.evaluate(fact, seeds)
    m, m0, mp, mf = Y.metrics(eps), Y.metrics(a0_eps), Y.metrics(peps), Y.metrics(feps)
    anm, pam = anchors_of(act), anchors_of(pact)
    gate = Y.gate_status(m, a0_capture=m0["capture"], anchor1=anm["anchor1"]["capture"], anchor2=anm["anchor2"]["capture"], anchor3=anm["anchor3"]["capture"])
    succ = lambda es: np.array([e["outcome"] == "soldier_caught" for e in es], dtype=float)                   # noqa: E731
    rep = lambda es: np.array([e["overshoots"] >= C.OVERSHOOT_PATHOLOGY for e in es], dtype=float)            # noqa: E731
    exh = lambda es: np.array([e["outcome"] == "enemy_endurance_exhausted" for e in es], dtype=float)         # noqa: E731
    track = [tracking_row(traces[i], eps[i]) for i in range(len(eps))]
    ptrack = [tracking_row(ptraces[i], peps[i]) for i in range(len(peps))]
    win = [r for r in track if r["outcome"] == "soldier_caught"]
    pwin = [r for r in ptrack if r["outcome"] == "soldier_caught"]
    manifest = seed_manifest(seed)
    return {"seed": seed, "accepted_model_sha256": manifest["accepted_model_sha256"], "training_status": manifest["status"], "mastery_rollout": manifest["mastery_rollout"], "crn": crn,
            "metrics": m, "a0_metrics": m0, "parent_metrics": mp, "failed_stage4_metrics_context_only": mf, "anchor_metrics": anm, "parent_anchor_metrics": pam, "gate_on_200": gate,
            "paired_capture_vs_a0": S.paired_bootstrap(succ(eps), succ(a0_eps)), "paired_capture_vs_parent": S.paired_bootstrap(succ(eps), succ(peps)),
            "paired_repeated_flyby_vs_parent": S.paired_bootstrap(rep(eps), rep(peps)), "paired_exhaustion_vs_parent": S.paired_bootstrap(exh(eps), exh(peps)),
            "paired_capture_vs_failed_stage4": S.paired_bootstrap(succ(eps), succ(feps)), "outcomes_stage4a": outcome_breakdown(eps), "outcomes_parent": outcome_breakdown(peps),
            "tracking_stage4a_all": summarize_tracking(track), "tracking_stage4a_captures": summarize_tracking(win), "tracking_parent_all": summarize_tracking(ptrack), "tracking_parent_captures": summarize_tracking(pwin),
            "tracking_rows": track, "episodes": [compact(e) for e in eps], "parent_episodes": [compact(e) for e in peps], "_traces": traces, "_parent_traces": ptraces, "_episodes_full": eps}


def tracking_improved(r: dict) -> dict:
    """Pre-registered: paired Stage4A - parent capture CI lower bound > 0 AND the repeated fly-by rate is below the parent's."""
    ci_ok = r["paired_capture_vs_parent"]["ci95_pp"][0] > 0.0
    fly_ok = r["metrics"]["repeated_flyby_fraction"] < r["parent_metrics"]["repeated_flyby_fraction"]
    return {"capture_ci_lower_bound_above_zero": bool(ci_ok), "repeated_flyby_rate_below_parent": bool(fly_ok), "improved": bool(ci_ok and fly_ok)}


def decide(per_seed: dict, trainings: dict) -> dict:
    mastered = [s for s, t in trainings.items() if t["status"] == "MASTERED"]
    retained = [s for s in mastered if all(per_seed[s]["anchor_metrics"][k]["capture"] >= v for k, v in (("anchor1", Y.ANCHOR_MIN["anchor1_basic_pursuit"]), ("anchor2", Y.ANCHOR_MIN["anchor2_static_defender"]),
                                                                                                          ("anchor3", Y.ANCHOR_MIN["anchor3_moving_defender"])))]
    improved = {s: tracking_improved(per_seed[s]) for s in per_seed}
    improved_mastered = [s for s in retained if improved[s]["improved"]]
    pathology = {}
    for key in ("orbit_episodes", "repeated_flyby_episodes", "nonclosing_episodes", "nonfinite"):
        hit = [s for s in retained if per_seed[s]["metrics"][key] / per_seed[s]["metrics"]["episodes"] > Y.COMMON_PATHOLOGY_RATE]
        if len(hit) >= Y.COMMON_PATHOLOGY_SEEDS:
            pathology[key] = hit
    passed = len(mastered) >= 2 and len(retained) >= 2 and len(improved_mastered) >= 2 and not pathology
    improving = [s for s in per_seed if per_seed[s]["metrics"]["capture"] >= 0.90 and per_seed[s]["paired_capture_vs_parent"]["ci95_pp"][0] > 0.0]
    promising = (not passed) and (len(mastered) >= 1 or len(improving) >= 2)
    return {"seeds_mastered": mastered, "seeds_retaining_all_anchors": retained, "training_status": {str(s): t["status"] for s, t in trainings.items()}, "tracking_improved": {str(s): v for s, v in improved.items()},
            "mastered_seeds_with_improved_tracking": improved_mastered, "common_structural_pathology": pathology, "stage4a_pass": bool(passed), "all_three_mastered": len(mastered) == 3,
            "seeds_improving_over_parent_with_capture_ge_0.90": improving, "improves_needs_manual_review": bool(promising),
            "verdict": "A" if passed else ("B" if promising else "C"),
            "rule": ">= 2/3 seeds master (two consecutive validation passes within 12 rollouts), mastered seeds retain all three anchors, tracking improves (paired capture CI lower bound > 0 and a lower repeated "
                    "fly-by rate than the parent), no shared pathology; verdict B if not passed but >= 1 seed mastered or >= 2 seeds reach held-out capture >= 0.90 with a paired-CI lower bound above the parent"}


# ----------------------------------------------------------------------------- figures
def rel_fig(p: Path) -> str:
    return str(p.resolve().relative_to(TS.PROJECT_ROOT.resolve())).replace("\\", "/")


def make_figures(trainings: dict, per_seed: dict) -> list[str]:
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

    def curve(ylabel, fname, getter, a0_key=None, hline=None, parent_key=None):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for s, t in trainings.items():
            ax.plot([r["num_timesteps"] for r in t["rollouts"]], [getter(r) for r in t["rollouts"]], marker="o", color=colors[s], label=f"A1-CUR4A-{s}")
            if parent_key:
                ax.axhline(per_seed[s]["parent_metrics"][parent_key], color=colors[s], ls="--", lw=0.8, alpha=0.6, label="Stage3-v2 parent (200 held-out)" if s == min(trainings) else None)
        if a0_key:
            r0 = next(iter(trainings.values()))["rollouts"]
            ax.plot([r["num_timesteps"] for r in r0], [r["a0_metrics"][a0_key] for r in r0], "k:", label="canonical A0 (matched)")
        if hline is not None:
            ax.axhline(hline[0], color="gray", ls="-.", label=hline[1])
        ax.set_xlabel("Stage-4A PPO timesteps"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        save(fig, fname)

    curve("validation capture rate", "fig1_capture_vs_timesteps", lambda r: r["validation_metrics"]["capture"], "capture", (0.95, "mastery 0.95"), "capture")
    curve("repeated fly-by episode rate", "fig2_flyby_rate_vs_timesteps", lambda r: r["validation_metrics"]["repeated_flyby_fraction"], "repeated_flyby_fraction", (0.05, "gate 0.05"),
          "repeated_flyby_fraction")
    curve("validation median path-excess", "fig3_path_excess_vs_timesteps", lambda r: r["validation_metrics"]["median_path_excess"], "median_path_excess", (1.5, "gate 1.50"), "median_path_excess")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    for ax, (key, title, thr) in zip(axes, (("anchor1_capture", "Anchor 1: stationary basic pursuit", 0.98), ("anchor2_capture", "Anchor 2: static defender", 0.80), ("anchor3_capture", "Anchor 3: moving defender", 0.75))):
        for s, t in trainings.items():
            ax.plot([r["num_timesteps"] for r in t["rollouts"]], [r[key] for r in t["rollouts"]], marker="o", color=colors[s], label=f"A1-CUR4A-{s}")
        ax.axhline(thr, color="gray", ls=":"); ax.set_title(title, fontsize=10); ax.set_xlabel("Stage-4A PPO timesteps"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("anchor capture"); axes[0].legend(fontsize=7)
    save(fig, "fig4_retention_anchors_vs_timesteps")
    return out


def make_trajectory_figure(per_seed: dict, seed_pick: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = per_seed[seed_pick]
    eps, peps = res["_episodes_full"], res["parent_episodes"]
    cand = [(i, e) for i, e in enumerate(eps) if e["outcome"] == "soldier_caught" and peps[i]["outcome"] != "soldier_caught"] or [(i, e) for i, e in enumerate(eps) if e["outcome"] == "soldier_caught"]
    if not cand:
        return []
    cand.sort(key=lambda t: -(peps[t[0]]["overshoots"] + 0.01 * t[1]["asset_final_displacement_m"]))
    idx, ep = cand[0]
    target = ep["scenario_seed"]
    tr, ptr = res["_traces"][idx], res["_parent_traces"][idx]
    _ae, a0 = F4.traced_episode(Y.Stage4AEnv(0, 0, 0, scripted_a0=True), target, lambda _o: np.zeros(3, dtype=np.float32))
    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*a0["pos"].T, "k--o", ms=3, label="canonical A0 (captured)")
    ax.plot(*ptr["pos"].T, "-s", ms=3, color="#ff7f0e", label=f"Stage3-v2 parent ({peps[idx]['outcome']}, {peps[idx]['overshoots']} fly-bys)")
    ax.plot(*tr["pos"].T, "-o", ms=3, color="#2ca02c", label=f"Stage4A A1-CUR4A-{seed_pick} (captured)")
    ax.plot(*tr["asset_trace"].T, "-", color="blue", lw=2, label="moving asset (common realized path)")
    ax.scatter(*tr["asset_trace"][0], color="blue", marker="*", s=80)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"scenario {target}: asset displacement {ep['asset_final_displacement_m']:.1f} m")
    fig.tight_layout()
    out = []
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"fig5_trajectory_a0_stage3_stage4a_{seed_pick}.{ext}"
        fig.savefig(p, dpi=200)
        out.append(rel_fig(p))
    plt.close(fig)
    return out


ASSET_SLOTS, ATTACKER_SLOTS = (3, 4, 5), (0, 1, 2)       # `asset_pos_*` and `attacker_pos_*` of a1_v2_obs_v1 (both normalised by the arena size L)


def collect_observations(act, seeds, max_episodes: int = 40) -> np.ndarray:
    env = Y.Stage4AEnv(0, 0, 0)
    rows = []
    for s in list(seeds)[:max_episodes]:
        obs, _ = env.reset(options={"scenario": int(s)})
        done = False
        while not done:
            rows.append(obs.copy())
            obs, _r, te, tr, _i = env.step(np.asarray(act(obs), dtype=np.float32))
            done = te or tr
    return np.array(rows, dtype=np.float32)


def asset_input_analysis_of(model, obs: np.ndarray) -> dict:
    """Does the policy use the asset position? (i) first-layer weight norm of the asset slots vs the attacker slots; (ii) the Jacobian of the executed unit direction w.r.t. the asset xy slots vs the attacker xy slots.
    A perfect relative tracker has d(direction)/d(asset) = -d(direction)/d(attacker) (ratio ~1, alignment ~1); an absolute-coordinate aimer at a fixed point has ratio ~0."""
    import torch

    w = model.policy.mlp_extractor.policy_net[0].weight.detach().numpy()
    x = torch.tensor(obs, dtype=torch.float32, requires_grad=True)
    mean = model.policy.get_distribution(x).distribution.mean
    unit = mean / mean.norm(dim=1, keepdim=True).clamp_min(1e-8)
    jac = np.stack([torch.autograd.grad(unit[:, k].sum(), x, retain_graph=True)[0].numpy() for k in range(3)], axis=1)      # (N, 3 outputs, 37 inputs)
    a, b = jac[:, :, 3:5].reshape(len(obs), -1), jac[:, :, 0:2].reshape(len(obs), -1)
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    align = -np.einsum("ij,ij->i", a, b) / np.maximum(na * nb, 1e-12)
    return {"first_layer_column_norm_asset_xyz": [float(v) for v in np.linalg.norm(w[:, list(ASSET_SLOTS)], axis=0)], "first_layer_column_norm_attacker_xyz": [float(v) for v in np.linalg.norm(w[:, list(ATTACKER_SLOTS)], axis=0)],
            "orthogonal_init_column_norm": float(np.sqrt(2.0)), "jacobian_norm_asset_xy_mean": float(na.mean()), "jacobian_norm_attacker_xy_mean": float(nb.mean()), "asset_to_attacker_sensitivity_ratio": float(na.mean() / nb.mean()),
            "relative_alignment_mean": float(align.mean()), "states": int(len(obs))}


def asset_input_analysis() -> dict:
    from stable_baselines3 import PPO

    seeds = list(Y.final_evaluation_seeds())
    out = {"definition": asset_input_analysis_of.__doc__, "states_from": "Stage-4A held-out scenarios, first 40 of 169000-169199, trajectories of each analysed policy"}
    for s in Y.PPO_SEEDS:
        entry = {}
        for tag, path in (("stage3_parent", Y.parent_checkpoint(s)["path"]), ("stage4a", str(accepted_model_path(s))), ("stage4_failed", str(P.RESULTS_ROOT / f"seed_{s}" / "checkpoints" / "accepted_model.zip"))):
            model = PPO.load(path.removesuffix(".zip"), device="cpu")
            entry[tag] = asset_input_analysis_of(model, collect_observations(C.model_actor(model), seeds))
        out[str(s)] = entry
    return out


def checkpoint_integrity() -> dict:
    from experiments import journal_v2_a1_cur4_run as RUN4
    from experiments import journal_v2_a1_cur4a_run as RUN

    out = {"stage1_to_3": RUN4.RUN3.stage2_integrity() | RUN4.stage3_integrity(), "stage4_failed_evidence": RUN.stage4_integrity(), "stage4a": {}}
    for s in Y.PPO_SEEDS:
        root = Y.RESULTS_ROOT / f"seed_{s}"
        m = seed_manifest(s)
        rolls = {r["rollout"]: TS.sha256_file(root / "rollouts" / f"rollout_{r['rollout']:02d}.zip") == r["model_sha256"] for r in m["rollouts"]}
        out["stage4a"][f"seed_{s}"] = {"status": m["status"], "accepted_sha256": TS.sha256_file(root / "checkpoints" / "accepted_model.zip"), "matches_manifest": TS.sha256_file(root / "checkpoints" / "accepted_model.zip") == m["accepted_model_sha256"],
                                      "all_rollouts_match": all(rolls.values()), "parent_sha256": m["parent"]["model_sha256"], "parent_matches_preregistered": m["parent"]["model_sha256"] == Y.PARENT_SHA256[s]}
    out["all_ok"] = all(v["matches_manifest"] and v["all_rollouts_match"] and v["parent_matches_preregistered"] for v in out["stage4a"].values())
    return out


def run_all() -> None:
    seeds = Y.final_evaluation_seeds()
    trainings = {s: seed_manifest(s) for s in Y.PPO_SEEDS}
    a0_eps = Y.evaluate(None, seeds, scripted_a0=True)
    per_seed = {s: evaluate_seed(s, seeds, a0_eps) for s in Y.PPO_SEEDS}
    decision = decide(per_seed, trainings)
    figs = make_figures(trainings, per_seed)
    pick = 203 if 203 in decision["seeds_mastered"] else max(per_seed, key=lambda s: per_seed[s]["metrics"]["capture"])
    figs += make_trajectory_figure(per_seed, pick)
    TS.atomic_write_json(DOCS / "final_evaluation.json", {"scenarios": [seeds[0], seeds[-1]], "episodes": len(seeds), "a0": {"metrics": Y.metrics(a0_eps), "episodes": [compact(e) for e in a0_eps]},
                                                           "per_seed": {str(s): {k: v for k, v in r.items() if not k.startswith("_") and k != "tracking_rows"} for s, r in per_seed.items()}})
    TS.atomic_write_json(DOCS / "moving_target_analysis.json", {str(s): {"stage4a_all": r["tracking_stage4a_all"], "stage4a_captures": r["tracking_stage4a_captures"], "stage3_parent_all": r["tracking_parent_all"],
                                                                      "stage3_parent_captures": r["tracking_parent_captures"], "outcomes_stage4a": r["outcomes_stage4a"], "outcomes_parent": r["outcomes_parent"],
                                                                      "tracking_improved_rule": decision["tracking_improved"][str(s)], "rows": r["tracking_rows"]} for s, r in per_seed.items()})
    TS.atomic_write_json(DOCS / "retention_evaluation.json", {str(s): {"stage4a_final": {k: v["capture"] for k, v in r["anchor_metrics"].items()}, "stage3_parent": {k: v["capture"] for k, v in r["parent_anchor_metrics"].items()},
                                                                      "per_rollout": [{"rollout": x["rollout"], "anchor1": x["anchor1_capture"], "anchor2": x["anchor2_capture"], "anchor3": x["anchor3_capture"]}
                                                                                      for x in trainings[s]["rollouts"]], "minimums": Y.ANCHOR_MIN} for s, r in per_seed.items()})
    TS.atomic_write_json(DOCS / "stage4a_decision.json", decision | {"figures": figs, "trajectory_figure_seed": pick})
    TS.atomic_write_json(DOCS / "checkpoint_integrity.json", checkpoint_integrity())
    print(json.dumps(decision, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all", "integrity", "figures", "asset-input"))
    args = parser.parse_args(argv)
    if args.command == "asset-input":
        TS.atomic_write_json(DOCS / "asset_input_analysis.json", asset_input_analysis())
    elif args.command == "integrity":
        TS.atomic_write_json(DOCS / "checkpoint_integrity.json", checkpoint_integrity())
    elif args.command == "figures":     # regenerate figures 1-4 from the recorded evaluation (no re-simulation)
        recorded = json.loads((DOCS / "final_evaluation.json").read_text())["per_seed"]
        print(make_figures({s: seed_manifest(s) for s in Y.PPO_SEEDS}, {int(k): v for k, v in recorded.items()}))
    else:
        run_all()


if __name__ == "__main__":
    main()
