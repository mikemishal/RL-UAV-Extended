"""Final evaluation + analysis for curriculum Stage3-v2 (run after training; `python -m experiments.journal_v2_a1_cur3v2_final all`).

* every seed's accepted / final policy on the 200 matched scenarios 154000-154199 together with canonical A0 and the seed's own Stage2-v2 parent on the SAME scenarios; paired A1-A0 and A1-parent capture /
  interception differences (paired bootstrap 95% CI);
* moving-defender evasion analysis of successful episodes (pursuit -> detection -> evasion of the moving Greedy -> resume pursuit -> capture), measured against the same policy flown WITHOUT the defender;
* generalization DIAGNOSTIC (Kalman-Greedy / PN / Lead with the same hold-until-detection semantics) - never a mastery gate;
* the Stage3-v2 decision, learning-curve figures (PNG + PDF), the Stage-4 handoff (only if accepted).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur3v2 as X
from experiments import journal_v2_a1_cur3v2_state as TS

DOCS = TS.STATE_DIR
FIG_DIR = DOCS / "figures"
DIVERGENCE_DEG = 10.0
MEANINGFUL_DEVIATION_M = 1.0          # displacement from the same policy's no-defender path
UNCLEAR_BELOW_M = 1.0
VERTICAL_SHARE = (0.30, 0.70)         # vertical share of the perpendicular displacement: <= 0.30 lateral, >= 0.70 vertical, else mixed
REACQUIRE_COS = math.cos(math.radians(DIVERGENCE_DEG))
COMMON_PATHOLOGY_RATE, COMMON_PATHOLOGY_SEEDS = 0.01, 2
DT = 0.5


def accepted_model_path(seed: int) -> Path:
    return X.RESULTS_ROOT / f"seed_{seed}" / "checkpoints" / "accepted_model.zip"


def seed_manifest(seed: int) -> dict:
    return json.loads((X.RESULTS_ROOT / f"seed_{seed}" / "checkpoints" / "seed_manifest.json").read_text())


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def traced_episode(env, scenario_seed: int, act) -> tuple[dict, dict]:
    """One deterministic episode with a per-decision trace: attacker position, executed direction, line of sight to the asset, defender position / velocity (moving-defender envs)."""
    obs, _ = env.reset(options={"scenario": int(scenario_seed)})
    e = env.soldier_env
    pos, u_list, los_list = [], [], []
    done, info = False, {}
    while not done:
        action = np.asarray(act(obs), dtype=np.float32)
        u, _held = C.desired_direction(action, e._enemy_vel)
        p = np.asarray(e._enemy_pos, dtype=np.float64)
        pos.append(p.copy())
        u_list.append(u)
        los_list.append(_unit(np.asarray(e._soldier_pos, dtype=np.float64) - p))
        obs, _r, te, tr, info = env.step(action)
        done = te or tr
    pos.append(np.asarray(env._att[-1], dtype=np.float64))
    trace = {"pos": np.array(pos), "u": np.array(u_list), "los": np.array(los_list), "asset": np.asarray(e._soldier_pos, dtype=np.float64)}
    if hasattr(env, "_def_pos"):
        trace["def_pos"], trace["def_vel"] = np.array(env._def_pos), np.array(env._def_vel)
    return dict(info["a1_episode"]), trace


def behavior_features(trace: dict, trace0: dict, episode: dict) -> dict:
    """Features of one SUCCESSFUL episode against a moving defender, relative to the same policy's no-defender trajectory `trace0`."""
    pos, u, los, dp, dv, asset = trace["pos"], trace["u"], trace["los"], trace["def_pos"], trace["def_vel"], trace["asset"]
    n = min(len(pos), len(trace0["pos"]), len(dp))
    dist = np.linalg.norm(pos[:n] - dp[:n], axis=1)
    i_star = int(np.argmin(dist))
    t_det, t_vis = episode["detection_step"], episode["defender_first_visible_step"]
    disp_vec = pos[:n] - trace0["pos"][:n]
    disp = np.linalg.norm(disp_vec, axis=1)
    div_idx = np.where(disp > 1e-4)[0]
    dev_idx = np.where(disp > MEANINGFUL_DEVIATION_M)[0]
    first_div = int(div_idx[0]) if len(div_idx) else None
    first_dev = int(dev_idx[0]) if len(dev_idx) else None
    chord_u = _unit(asset - pos[0])
    along = np.abs(disp_vec @ chord_u)
    perp_vec = disp_vec - np.outer(disp_vec @ chord_u, chord_u)
    perp = np.linalg.norm(perp_vec, axis=1)
    vert = np.abs(perp_vec[:, 2])
    horiz = np.linalg.norm(perp_vec[:, :2], axis=1)
    max_disp = float(disp.max())
    share = float(vert.max() / max(vert.max() + horiz.max(), 1e-9))
    if max_disp < UNCLEAR_BELOW_M:
        klass = "unclear"
    elif along.max() >= perp.max():
        klass = "timing_turn"
    elif share >= VERTICAL_SHARE[1]:
        klass = "vertical"
    elif share <= VERTICAL_SHARE[0]:
        klass = "lateral"
    else:
        klass = "mixed"
    # relative geometry at first visibility (25 m gate) and at the closest approach
    iv = t_vis if t_vis is not None and t_vis < n - 1 else None
    rel_at_vis = None
    if iv is not None:
        v_att = (pos[iv + 1] - pos[iv]) / DT
        r = pos[iv] - dp[iv]
        v_rel = v_att - dv[iv]
        rel_at_vis = {"closing_speed_mps": float(-np.dot(r, v_rel) / max(np.linalg.norm(r), 1e-9)), "defender_speed_mps": float(np.linalg.norm(dv[iv])), "range_m": float(np.linalg.norm(r)),
                      "relative_speed_mps": float(np.linalg.norm(v_rel))}
    v_def = dv[i_star]
    w = pos[i_star] - dp[i_star]
    ahead = None
    if np.linalg.norm(v_def) > 0.5:
        ahead = bool(np.dot(w, _unit(v_def)) > 0.0)        # True: the attacker passes in FRONT of the defender's velocity vector; False: behind it
    cos_los = np.einsum("ij,ij->i", u, los)
    post = slice(min(i_star + 1, len(u) - 1), len(u))
    post_cos = float(cos_los[post].mean()) if len(cos_los[post]) else float("nan")
    reacq = None
    for j in range(i_star, len(cos_los)):
        if cos_los[j] >= REACQUIRE_COS:
            reacq = (j - i_star) * DT
            break
    return {
        "scenario_seed": episode["scenario_seed"], "class": klass, "detection_step": t_det, "first_visible_step": t_vis, "closest_step": i_star, "min_distance_m": float(dist[i_star]),
        "first_trajectory_divergence_step": first_div, "first_meaningful_deviation_step": first_dev, "deviation_before_visibility": bool(first_div is not None and t_vis is not None and first_div < t_vis),
        "time_detection_to_meaningful_deviation_s": (float((first_dev - t_det) * DT) if first_dev is not None and t_det is not None else None),
        "time_visibility_to_meaningful_deviation_s": (float((first_dev - t_vis) * DT) if first_dev is not None and t_vis is not None else None),
        "max_displacement_vs_no_defender_m": max_disp, "max_lateral_m": float(horiz.max()), "max_vertical_m": float(vert.max()), "max_along_track_m": float(along.max()), "vertical_share": share,
        "defender_speed_at_closest_mps": float(np.linalg.norm(v_def)), "defender_speed_at_detection_mps": episode["defender_speed_at_detection_mps"], "passes_ahead_of_defender": ahead,
        "relative_at_first_visibility": rel_at_vis, "post_encounter_mean_cos_to_asset": post_cos, "reacquisition_time_s": reacq, "path_excess": episode["path_excess"], "steps": episode["steps"],
        "min_separation_m": episode["min_defender_separation_m"],
    }


def summarize_behavior(rows: list[dict]) -> dict:
    if not rows:
        return {"n_successful": 0}

    def stat(values):
        v = np.array([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], dtype=np.float64)
        return None if v.size == 0 else {"n": int(v.size), "mean": float(v.mean()), "median": float(np.median(v)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    classes = [r["class"] for r in rows]
    ahead = [r["passes_ahead_of_defender"] for r in rows if r["passes_ahead_of_defender"] is not None]
    return {"n_successful": len(rows), "class_counts": {k: classes.count(k) for k in ("lateral", "vertical", "mixed", "timing_turn", "unclear")},
            "min_distance_m": stat([r["min_distance_m"] for r in rows]), "time_detection_to_meaningful_deviation_s": stat([r["time_detection_to_meaningful_deviation_s"] for r in rows]),
            "time_visibility_to_meaningful_deviation_s": stat([r["time_visibility_to_meaningful_deviation_s"] for r in rows]),
            "fraction_deviation_before_defender_visible": float(np.mean([r["deviation_before_visibility"] for r in rows])),
            "fraction_with_meaningful_deviation": float(np.mean([r["first_meaningful_deviation_step"] is not None for r in rows])),
            "max_lateral_m": stat([r["max_lateral_m"] for r in rows]), "max_vertical_m": stat([r["max_vertical_m"] for r in rows]), "max_displacement_vs_no_defender_m": stat([r["max_displacement_vs_no_defender_m"] for r in rows]),
            "defender_speed_at_closest_mps": stat([r["defender_speed_at_closest_mps"] for r in rows]), "defender_speed_at_detection_mps": stat([r["defender_speed_at_detection_mps"] for r in rows]),
            "passes_ahead_of_defender_fraction": (float(np.mean(ahead)) if ahead else None), "closing_speed_at_first_visibility_mps": stat([(r["relative_at_first_visibility"] or {}).get("closing_speed_mps") for r in rows]),
            "defender_speed_at_first_visibility_mps": stat([(r["relative_at_first_visibility"] or {}).get("defender_speed_mps") for r in rows]),
            "post_encounter_mean_cos_to_asset": stat([r["post_encounter_mean_cos_to_asset"] for r in rows]), "reacquisition_time_s": stat([r["reacquisition_time_s"] for r in rows]),
            "path_excess": stat([r["path_excess"] for r in rows]),
            "definitions": {"deviation": f"displacement of the attacker from the SAME policy's no-defender trajectory (identical scenario) > {MEANINGFUL_DEVIATION_M} m; a trajectory divergence > 1e-4 m "
                            "marks the first observation difference (the defender entering the 25 m gate)", "classes": f"unclear: max displacement < {UNCLEAR_BELOW_M} m; timing_turn: along-chord displacement >= "
                            f"perpendicular displacement; else vertical share of perpendicular displacement >= {VERTICAL_SHARE[1]} -> vertical, <= {VERTICAL_SHARE[0]} -> lateral, otherwise mixed",
                            "reacquisition": f"first decision at/after the closest approach whose executed direction is within {DIVERGENCE_DEG} deg of the line of sight to the asset",
                            "passes_ahead": "attacker lies in front of the defender velocity vector at the closest approach (False: behind it); undefined when the defender moves < 0.5 m/s",
                            "labels": "descriptive only; never used for reward"}}


def compact(ep: dict) -> dict:
    keep = ("scenario_seed", "outcome", "steps", "duration_s", "path_length_m", "chord_m", "path_excess", "initial_range_m", "initial_altitude_m", "initial_heading_offset_deg", "min_defender_separation_m",
            "defender_visible_s", "defender_visible_steps", "defender_first_visible_step", "defender_position", "defender_final_position", "placement_lambda", "placement_offset_m", "overshoots",
            "orbit_revolutions_max", "nonclosing_run", "intercepted", "pathology", "detection_step", "detection_time_s", "release_step", "defender_speed_at_release_mps", "defender_speed_at_detection_mps",
            "time_release_to_end_s", "closest_step", "time_detection_to_closest_s", "defender_speed_at_closest_mps", "defender_max_speed_mps")
    return {k: ep[k] for k in keep if k in ep}


def no_defender_traces(act, seeds: list[int]) -> list[dict]:
    env = V1V2.Stage1V2Env(0, 0, 0)
    return [traced_episode(env, s, act)[1] for s in seeds]


def evaluate_seed(seed: int, seeds: list[int], a0_eps: list[dict]) -> dict:
    from stable_baselines3 import PPO

    model = PPO.load(str(accepted_model_path(seed)).removesuffix(".zip"), device="cpu")
    parent = PPO.load(X.parent_checkpoint(seed)["path"].removesuffix(".zip"), device="cpu")
    act, pact = C.model_actor(model), C.model_actor(parent)
    env = X.Stage3V2Env(0, 0, 0)
    eps, traces = [], []
    for s in seeds:
        ep, tr = traced_episode(env, s, act)
        eps.append(ep)
        traces.append(tr)
    base = no_defender_traces(act, seeds)
    parent_eps = X.evaluate(pact, seeds)
    m, m0, mp = X.metrics(eps), X.metrics(a0_eps), X.metrics(parent_eps)
    a1m = V1V2.metrics(X.evaluate_stage1_anchor(act))
    a2m = X.metrics(X.evaluate_static_anchor(act))
    parent_a1 = V1V2.metrics(X.evaluate_stage1_anchor(pact))["capture"]
    parent_a2 = X.metrics(X.evaluate_static_anchor(pact))["capture"]
    gate = X.gate_status(m, a0_capture=m0["capture"], stage1_anchor=a1m["capture"], static_anchor=a2m["capture"])
    succ = lambda es: np.array([e["outcome"] == "soldier_caught" for e in es], dtype=float)   # noqa: E731
    icpt = lambda es: np.array([e["intercepted"] for e in es], dtype=float)                    # noqa: E731
    rows = [behavior_features(tr, b, ep) for ep, tr, b in zip(eps, traces, base) if ep["outcome"] == "soldier_caught"]
    manifest = seed_manifest(seed)
    return {"seed": seed, "accepted_model_sha256": manifest["accepted_model_sha256"], "training_status": manifest["status"], "mastery_rollout": manifest["mastery_rollout"],
            "metrics": m, "a0_metrics": m0, "parent_metrics": mp, "stage1_anchor_metrics": a1m, "static_anchor_metrics": a2m, "parent_anchor_captures": {"stage1": parent_a1, "static": parent_a2},
            "gate_on_200": gate, "paired_capture_vs_a0": S.paired_bootstrap(succ(eps), succ(a0_eps)), "paired_interception_vs_a0": S.paired_bootstrap(icpt(eps), icpt(a0_eps)),
            "paired_capture_vs_parent": S.paired_bootstrap(succ(eps), succ(parent_eps)), "paired_interception_vs_parent": S.paired_bootstrap(icpt(eps), icpt(parent_eps)),
            "behavior": summarize_behavior(rows), "behavior_rows": rows, "episodes": [compact(e) for e in eps], "parent_episodes": [compact(e) for e in parent_eps],
            "_traces": traces, "_baseline": base, "_episodes_full": eps}


def generalization(seed: int) -> dict:
    """DIAGNOSTIC ONLY: the final policy (and its Stage2-v2 parent) against other scripted interceptors under the SAME hold-until-detection semantics, plus matched canonical A0."""
    from stable_baselines3 import PPO

    model = PPO.load(str(accepted_model_path(seed)).removesuffix(".zip"), device="cpu")
    parent = PPO.load(X.parent_checkpoint(seed)["path"].removesuffix(".zip"), device="cpu")
    act, pact = C.model_actor(model), C.model_actor(parent)
    seeds = list(X.seeds_of(X.DIAGNOSTIC))
    out = {}
    for controller in ("greedy",) + tuple(X.DIAGNOSTIC_CONTROLLERS):
        a1 = X.metrics(X.evaluate(act, seeds, controller=controller))
        p = X.metrics(X.evaluate(pact, seeds, controller=controller))
        a0 = X.metrics(X.evaluate(None, seeds, scripted_a0=True, controller=controller))
        pick = ("capture", "interception", "timeout", "exhaustion", "median_path_excess", "min_defender_separation_median_m", "orbit_episodes", "repeated_flyby_episodes")
        out[controller] = {"stage3_v2": {k: a1[k] for k in pick}, "stage2_v2_parent": {k: p[k] for k in pick}, "matched_a0": {k: a0[k] for k in pick},
                           "capture_minus_a0_pp": 100.0 * (a1["capture"] - a0["capture"]), "capture_minus_parent_pp": 100.0 * (a1["capture"] - p["capture"])}
    return {"seed": seed, "scenarios": [seeds[0], seeds[-1]], "episodes": len(seeds), "controllers": out, "note": "diagnostic only; not part of mastery or training"}


def decide(per_seed: dict, trainings: dict) -> dict:
    mastered = [s for s, t in trainings.items() if t["status"] == "MASTERED"]
    retained = [s for s in mastered if per_seed[s]["stage1_anchor_metrics"]["capture"] >= X.STAGE1_ANCHOR_MIN and per_seed[s]["static_anchor_metrics"]["capture"] >= X.STATIC_ANCHOR_MIN]
    pathology = {}
    for key in ("orbit_episodes", "repeated_flyby_episodes", "nonclosing_episodes", "nonfinite"):
        hit = [s for s in retained if per_seed[s]["metrics"][key] / per_seed[s]["metrics"]["episodes"] > COMMON_PATHOLOGY_RATE]
        if len(hit) >= COMMON_PATHOLOGY_SEEDS:
            pathology[key] = hit
    passed = len(mastered) >= 2 and len(retained) >= 2 and not pathology
    improving = [s for s in per_seed if per_seed[s]["metrics"]["capture"] >= 0.75 and per_seed[s]["paired_capture_vs_parent"]["ci95_pp"][0] > 0.0]
    promising = (not passed) and (len(mastered) >= 1 or len(improving) >= 2)
    return {"seeds_mastered": mastered, "seeds_retaining_both_anchors": retained, "training_status": {str(s): t["status"] for s, t in trainings.items()}, "common_structural_pathology": pathology,
            "stage3_v2_pass": bool(passed), "all_three_mastered": len(mastered) == 3, "seeds_improving_over_parent_with_capture_ge_0.75": improving, "promising_manual_review": bool(promising),
            "rule": ">= 2/3 seeds master (two consecutive validation passes within 16 rollouts), mastered seeds retain both anchors, no shared pathology; verdict B if not passed but >= 1 seed mastered or >= 2 seeds "
                    "reach held-out capture >= 0.75 with a paired-CI lower bound above the Stage2-v2 parent"}


# ----------------------------------------------------------------------------- figures
def rel_fig(p: Path) -> str:
    return str(p.resolve().relative_to(TS.PROJECT_ROOT.resolve())).replace("\\", "/")


def make_figures(trainings: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    colors = {201: "#1f77b4", 202: "#d62728", 203: "#2ca02c"}

    def curve(ylabel, fname, getter, a0_key=None, hline=None):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for s, t in trainings.items():
            ax.plot([r["num_timesteps"] for r in t["rollouts"]], [getter(r) for r in t["rollouts"]], marker="o", color=colors[s], label=f"A1-CUR3V2-{s}")
        if a0_key:
            r0 = next(iter(trainings.values()))["rollouts"]
            ax.plot([r["num_timesteps"] for r in r0], [r["a0_metrics"][a0_key] for r in r0], "k--", label="canonical A0 (matched)")
        if hline is not None:
            ax.axhline(hline[0], color="gray", ls=":", label=hline[1])
        ax.set_xlabel("Stage-3 PPO timesteps"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = FIG_DIR / f"{fname}.{ext}"
            fig.savefig(p, dpi=200)
            out.append(rel_fig(p))
        plt.close(fig)

    curve("validation capture rate", "fig1_capture_vs_timesteps", lambda r: r["validation_metrics"]["capture"], "capture", (0.80, "mastery 0.80"))
    curve("validation interception rate", "fig2_interception_vs_timesteps", lambda r: r["validation_metrics"]["interception"], "interception", (0.20, "gate 0.20"))
    curve("Stage-1 pursuit anchor capture", "fig3_pursuit_retention_vs_timesteps", lambda r: r["stage1_anchor_capture"], None, (0.98, "retention 0.98"))
    curve("Stage-2 static-defender anchor capture", "fig4_static_evasion_retention_vs_timesteps", lambda r: r["static_anchor_capture"], None, (0.80, "retention 0.80"))
    curve("validation median path-excess", "fig5_path_excess_vs_timesteps", lambda r: r["validation_metrics"]["median_path_excess"], "median_path_excess", (1.80, "gate 1.80"))
    return out


def a0_traced(seeds: list[int]) -> dict:
    env = X.Stage3V2Env(0, 0, 0, scripted_a0=True)
    return {s: traced_episode(env, s, lambda _o: np.zeros(3, dtype=np.float32)) for s in seeds}


def make_trajectory_figure(per_seed: dict, seed_pick: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = per_seed[seed_pick]
    rows = [r for r in res["behavior_rows"] if r["class"] != "unclear"] or res["behavior_rows"]
    if not rows:
        return []
    rows.sort(key=lambda r: abs(r["min_distance_m"] - 4.0) + 0.01 * r["path_excess"])
    target = rows[0]["scenario_seed"]
    idx = [e["scenario_seed"] for e in res["_episodes_full"]].index(target)
    tr = res["_traces"][idx]
    a0_ep, a0 = a0_traced([target])[target]
    fig = plt.figure(figsize=(7.4, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*a0["pos"].T, "k--o", ms=3, label="canonical A0 (intercepted)")
    ax.plot(*a0["def_pos"].T, ":", color="gray", marker="x", ms=4, label="Greedy defender vs A0")
    ax.plot(*tr["pos"].T, "-o", ms=3, color="#2ca02c", label=f"A1-CUR3V2-{seed_pick} (captured)")
    ax.plot(*tr["def_pos"].T, "-", color="#d62728", marker="x", ms=4, label="Greedy defender vs A1")
    ax.scatter(*tr["def_pos"][0], color="red", s=30, marker="s", label="defender hold position")
    ax.scatter(*tr["asset"], color="blue", marker="*", s=90, label="asset")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"scenario {target}: min separation {rows[0]['min_distance_m']:.2f} m ({rows[0]['class']})")
    fig.tight_layout()
    out = []
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"fig6_trajectory_a0_vs_a1_{seed_pick}.{ext}"
        fig.savefig(p, dpi=200)
        out.append(rel_fig(p))
    plt.close(fig)
    return out


def run_all() -> None:
    seeds = X.final_evaluation_seeds()
    trainings = {s: seed_manifest(s) for s in X.PPO_SEEDS}
    a0_eps = X.evaluate(None, seeds, scripted_a0=True)
    per_seed = {s: evaluate_seed(s, seeds, a0_eps) for s in X.PPO_SEEDS}
    gen = {str(s): generalization(s) for s in X.PPO_SEEDS}
    decision = decide(per_seed, trainings)
    figs = make_figures(trainings)
    pick = 203 if 203 in decision["seeds_mastered"] else max(per_seed, key=lambda s: per_seed[s]["metrics"]["capture"])
    if per_seed[pick]["behavior_rows"]:
        figs += make_trajectory_figure(per_seed, pick)
    payload = {"scenarios": [seeds[0], seeds[-1]], "episodes": len(seeds), "a0": {"metrics": X.metrics(a0_eps), "episodes": [compact(e) for e in a0_eps]},
               "per_seed": {str(s): {k: v for k, v in r.items() if not k.startswith("_")} for s, r in per_seed.items()}}
    TS.atomic_write_json(DOCS / "final_evaluation.json", payload)
    TS.atomic_write_json(DOCS / "behavior_analysis.json", {str(s): {"summary": r["behavior"], "rows": r["behavior_rows"]} for s, r in per_seed.items()})
    TS.atomic_write_json(DOCS / "generalization_diagnostic.json", gen)
    TS.atomic_write_json(DOCS / "stage3_v2_decision.json", decision | {"figures": figs, "trajectory_figure_seed": pick})
    print(json.dumps(decision, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all",))
    parser.parse_args(argv)
    run_all()


if __name__ == "__main__":
    main()
