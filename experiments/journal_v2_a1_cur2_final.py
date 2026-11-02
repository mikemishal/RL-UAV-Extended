"""Final evaluation + analysis for curriculum Stage 2 (run after training; `python -m experiments.journal_v2_a1_cur2_final all`).

* every seed's accepted / final policy on the 200 matched scenarios 147000-147199 together with canonical A0 on the SAME scenarios; paired A1-A0 capture / interception differences (paired bootstrap 95% CI);
* learned-evasion analysis of successful A1 episodes (pursuit -> evasion -> pursuit);
* BC-fidelity / forgetting analysis against the seed's own Stage-1 BC initializer in no-defender C0;
* the Stage-2 decision, learning-curve figures (PNG + PDF), the Stage-3 handoff (only if Stage 2 passes).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_cur1 as C
from experiments import journal_v2_a1_cur1v2 as V1V2
from experiments import journal_v2_a1_cur1v2_bc as BC
from experiments import journal_v2_a1_cur2 as S
from experiments import journal_v2_a1_cur2_state as TS
from experiments import journal_v2_a1_cur2_training as T2

DOCS = TS.STATE_DIR
FIG_DIR = DOCS / "figures"
DIVERGENCE_DEG = 10.0
RESUME_MEAN_COS = 0.95
COMMON_PATHOLOGY_RATE, COMMON_PATHOLOGY_SEEDS = 0.01, 2


def accepted_model_path(seed: int) -> Path:
    return S.RESULTS_ROOT / f"seed_{seed}" / "checkpoints" / "accepted_model.zip"


def seed_manifest(seed: int) -> dict:
    return json.loads((S.RESULTS_ROOT / f"seed_{seed}" / "checkpoints" / "seed_manifest.json").read_text())


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def traced_episode(env, scenario_seed: int, act, bc_act=None) -> tuple[dict, dict]:
    """One deterministic episode with a per-decision trace: position, executed direction, line of sight to the asset, BC-parent direction, distance to the defender."""
    obs, _ = env.reset(options={"scenario": int(scenario_seed)})
    e = env.soldier_env
    pos, u_list, los_list, ub_list, dd = [], [], [], [], []
    done, info = False, {}
    while not done:
        action = np.asarray(act(obs), dtype=np.float32)
        u, _held = C.desired_direction(action, e._enemy_vel)
        p = np.asarray(e._enemy_pos, dtype=np.float64)
        pos.append(p.copy())
        u_list.append(u)
        los_list.append(_unit(np.asarray(e._soldier_pos, dtype=np.float64) - p))
        dd.append(float(np.linalg.norm(np.asarray(e._defender_pos, dtype=np.float64) - p)) if hasattr(e, "_defender_pos") else float("nan"))
        if bc_act is not None:
            ub_list.append(C.desired_direction(np.asarray(bc_act(obs), dtype=np.float32), e._enemy_vel)[0])
        obs, _r, te, tr, info = env.step(action)
        done = te or tr
    pos.append(np.asarray(env._att[-1], dtype=np.float64))
    trace = {"pos": np.array(pos), "u": np.array(u_list), "los": np.array(los_list), "u_bc": np.array(ub_list) if ub_list else None, "defender_dist": np.array(dd),
             "defender": np.asarray(e._defender_pos, dtype=np.float64), "asset": np.asarray(e._soldier_pos, dtype=np.float64)}
    return dict(info["a1_episode"]), trace


def evasion_features(trace: dict, episode: dict) -> dict:
    pos, u, los, ub, dd, d = trace["pos"], trace["u"], trace["los"], trace["u_bc"], trace["defender_dist"], trace["defender"]
    p0, asset = pos[0], trace["asset"]
    chord = asset - p0
    chord_u = chord / np.linalg.norm(chord)
    rel = pos - p0
    par = rel @ chord_u
    perp_vec = rel - np.outer(par, chord_u)
    lateral = np.linalg.norm(perp_vec, axis=1)
    z_line = p0[2] + (par / np.linalg.norm(chord)) * (asset[2] - p0[2])
    vertical = np.abs(pos[:, 2] - z_line)
    dist_all = np.linalg.norm(pos - d[None, :], axis=1)
    i_star = int(np.argmin(dist_all))
    cos_los = np.einsum("ij,ij->i", u, los)
    cos_bc = np.einsum("ij,ij->i", u, ub) if ub is not None else None
    dev = cos_los < math.cos(math.radians(DIVERGENCE_DEG))
    pre, post = slice(0, max(i_star, 1)), slice(min(i_star + 1, len(u) - 1), len(u))
    first_div = None
    if ub is not None:
        idx = np.where(cos_bc < math.cos(math.radians(DIVERGENCE_DEG)))[0]
        first_div = int(idx[0]) if len(idx) else None
    # side of the defender at the closest approach: decompose the defender->attacker vector perpendicular to the approach direction
    approach = _unit(pos[min(i_star + 1, len(pos) - 1)] - pos[max(i_star - 1, 0)])
    w = pos[i_star] - d
    w_perp = w - float(np.dot(w, approach)) * approach
    up = np.array([0.0, 0.0, 1.0])
    vert_c = float(np.dot(w_perp, up))
    side_h = float(approach[0] * w_perp[1] - approach[1] * w_perp[0])      # >0: attacker passes to the LEFT of the defender (horizontal plane)
    horiz_c = side_h / max(float(np.linalg.norm(approach[:2])), 1e-9)
    side = ("over" if vert_c > 0 else "under") if abs(vert_c) >= abs(horiz_c) else ("left" if horiz_c > 0 else "right")
    visible = dd <= 25.0
    pursuit_before = bool((~dev[pre]).any()) if len(dev[pre]) else True
    deviated = bool(dev[visible].any()) if visible.any() else bool(dev.any())
    pursuit_after = bool((~dev[post]).any()) if len(dev[post]) else True
    resume_mean = float(cos_los[post].mean()) if len(cos_los[post]) else float("nan")
    return {"scenario_seed": episode["scenario_seed"], "max_lateral_dev_m": float(lateral.max()), "max_vertical_dev_m": float(vertical.max()), "min_defender_distance_m": float(dist_all[i_star]),
            "closest_step": i_star, "side": side, "first_divergence_step": first_div, "first_divergence_defender_distance_m": float(dd[first_div]) if first_div is not None else None,
            "first_divergence_while_visible": bool(first_div is not None and visible[first_div]), "defender_first_visible_step": episode.get("defender_first_visible_step"),
            "deviation_steps": int(dev.sum()), "post_encounter_mean_cos_to_asset": resume_mean, "resumes_pursuit": bool(post.start < len(u) and resume_mean >= RESUME_MEAN_COS),
            "pursuit_evasion_pursuit": bool(pursuit_before and deviated and pursuit_after and episode["outcome"] == "soldier_caught"), "path_excess": episode["path_excess"],
            "final_cos_to_asset": float(cos_los[-1])}


def summarize_evasion(rows: list[dict]) -> dict:
    if not rows:
        return {"n_successful": 0}
    f = lambda key: np.array([r[key] for r in rows], dtype=np.float64)  # noqa: E731
    stat = lambda v: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)), "p5": float(np.nanpercentile(v, 5)), "p95": float(np.nanpercentile(v, 95)), "max": float(np.nanmax(v))}  # noqa: E731
    sides = [r["side"] for r in rows]
    div = [r for r in rows if r["first_divergence_step"] is not None]
    return {"n_successful": len(rows), "max_lateral_deviation_from_initial_chord_m": stat(f("max_lateral_dev_m")), "max_vertical_deviation_from_chord_m": stat(f("max_vertical_dev_m")),
            "min_defender_distance_m": stat(f("min_defender_distance_m")), "side_counts": {k: sides.count(k) for k in ("over", "under", "left", "right")},
            "fraction_diverging_from_bc_direction": len(div) / len(rows),
            "first_divergence_defender_distance_m": stat(np.array([r["first_divergence_defender_distance_m"] for r in div])) if div else None,
            "first_divergence_while_defender_visible_fraction": (sum(r["first_divergence_while_visible"] for r in div) / len(div)) if div else None,
            "resumes_pursuit_fraction": float(np.mean([r["resumes_pursuit"] for r in rows])), "post_encounter_mean_cos_to_asset": stat(f("post_encounter_mean_cos_to_asset")),
            "pursuit_evasion_pursuit_fraction": float(np.mean([r["pursuit_evasion_pursuit"] for r in rows])), "path_excess": stat(f("path_excess")),
            "definitions": {"divergence": f"angle between executed direction and the BC-parent direction at the same state > {DIVERGENCE_DEG} deg", "resume": f"mean cos(executed direction, line of sight) after the closest "
                            f"defender approach >= {RESUME_MEAN_COS}", "pursuit_evasion_pursuit": "pursuit steps (angle to line of sight <= 10 deg) before the closest approach, deviating steps while the defender is "
                            "visible, pursuit steps afterwards, episode captured"}}


def compact(ep: dict) -> dict:
    keep = ("scenario_seed", "outcome", "steps", "duration_s", "path_length_m", "chord_m", "path_excess", "initial_range_m", "initial_altitude_m", "initial_heading_offset_deg", "min_defender_separation_m",
            "defender_visible_s", "defender_visible_steps", "defender_first_visible_step", "defender_position", "placement_lambda", "placement_offset_m", "overshoots", "orbit_revolutions_max",
            "nonclosing_run", "intercepted", "pathology")
    return {k: ep[k] for k in keep if k in ep}


def reactive_evasion(seed: int, seeds: list[int], stage2_traces: list[dict], stage2_eps: list[dict]) -> dict:
    """Causal check: the same final policy on the same scenarios WITHOUT the defender (Stage-1 masked no-defender env). Before the defender enters the 25 m gate the two observations are identical, so
    the paired trajectories must coincide; the displacement after gate entry is the policy's reaction to the defender."""
    from stable_baselines3 import PPO

    model = PPO.load(str(accepted_model_path(seed)).removesuffix(".zip"), device="cpu")
    act = C.model_actor(model)
    env = V1V2.Stage1V2Env(0, 0, 0)
    rows = []
    for s, tr, ep in zip(seeds, stage2_traces, stage2_eps):
        _e0, tr0 = traced_episode(env, s, act)
        n = min(len(tr["pos"]), len(tr0["pos"]))
        disp = np.linalg.norm(tr["pos"][:n] - tr0["pos"][:n], axis=1)
        first_visible = ep.get("defender_first_visible_step")
        diverged = np.where(disp > 1e-4)[0]
        first_div = int(diverged[0]) if len(diverged) else None
        chord = tr["asset"] - tr["pos"][0]
        cu = chord / np.linalg.norm(chord)

        def lat(p):
            rel = p - p[0]
            return float(np.linalg.norm(rel - np.outer(rel @ cu, cu), axis=1).max())
        rows.append({"scenario_seed": s, "max_displacement_m": float(disp.max()), "first_trajectory_divergence_step": first_div, "defender_first_visible_step": first_visible,
                     "diverges_before_visibility": bool(first_div is not None and first_visible is not None and first_div < first_visible), "lateral_dev_with_defender_m": lat(tr["pos"]),
                     "lateral_dev_without_defender_m": lat(tr0["pos"]), "captured_with": ep["outcome"] == "soldier_caught"})
    disp = np.array([r["max_displacement_m"] for r in rows])
    caught = [r for r in rows if r["captured_with"]]
    intercepted = [r for r in rows if not r["captured_with"]]
    return {"episodes": len(rows), "fraction_reacting_gt_1m": float((disp > 1.0).mean()), "max_displacement_m": {"median": float(np.median(disp)), "mean": float(disp.mean()), "p95": float(np.percentile(disp, 95))},
            "fraction_diverging_before_gate_entry": float(np.mean([r["diverges_before_visibility"] for r in rows])),
            "mean_extra_lateral_deviation_m": float(np.mean([r["lateral_dev_with_defender_m"] - r["lateral_dev_without_defender_m"] for r in rows])),
            "captured_mean_displacement_m": float(np.mean([r["max_displacement_m"] for r in caught])) if caught else None,
            "intercepted_mean_displacement_m": float(np.mean([r["max_displacement_m"] for r in intercepted])) if intercepted else None, "rows": rows}


def evaluate_seed(seed: int, seeds: list[int], a0_eps: list[dict]) -> dict:
    from stable_baselines3 import PPO

    model = PPO.load(str(accepted_model_path(seed)).removesuffix(".zip"), device="cpu")
    bc = BC.load_model(Path(T2.bc_parent(seed)["path"]))
    act, bc_act = C.model_actor(model), C.model_actor(bc)
    env = S.Stage2Env(0, 0, 0)
    eps, traces = [], []
    for s in seeds:
        ep, tr = traced_episode(env, s, act, bc_act)
        eps.append(ep)
        traces.append(tr)
    m = S.metrics(eps)
    m0 = S.metrics(a0_eps)
    anchor = V1V2.metrics(S.evaluate_anchor(act))
    gate = S.gate_status(m, a0_capture=m0["capture"], anchor_capture=anchor["capture"])
    succ = np.array([e["outcome"] == "soldier_caught" for e in eps], dtype=float)
    succ0 = np.array([e["outcome"] == "soldier_caught" for e in a0_eps], dtype=float)
    icpt = np.array([e["intercepted"] for e in eps], dtype=float)
    icpt0 = np.array([e["intercepted"] for e in a0_eps], dtype=float)
    rows = [evasion_features(tr, ep) for ep, tr in zip(eps, traces) if ep["outcome"] == "soldier_caught"]
    return {"seed": seed, "accepted_model_sha256": seed_manifest(seed)["accepted_model_sha256"], "training_status": seed_manifest(seed)["status"], "metrics": m, "a0_metrics": m0, "anchor_metrics": anchor,
            "gate_on_200": gate, "paired_capture": S.paired_bootstrap(succ, succ0), "paired_interception": S.paired_bootstrap(icpt, icpt0),
            "evasion": summarize_evasion(rows), "evasion_rows": rows, "reactive": reactive_evasion(seed, seeds, traces, eps), "episodes": [compact(e) for e in eps], "_traces": traces, "_episodes_full": eps}


def bc_fidelity(seed: int) -> dict:
    from stable_baselines3 import PPO

    final = PPO.load(str(accepted_model_path(seed)).removesuffix(".zip"), device="cpu")
    bc = BC.load_model(Path(T2.bc_parent(seed)["path"]))
    seeds = list(C.final_evaluation_seeds())                 # Stage-1 held-out 146000-146199 (never used for BC)
    act_f, act_b = C.model_actor(final), C.model_actor(bc)
    mf, mb = V1V2.metrics(V1V2.evaluate(act_f, seeds)), V1V2.metrics(V1V2.evaluate(act_b, seeds))
    env = V1V2.Stage1V2Env(0, 0, 0)
    cos_on_final, cos_on_bc = [], []
    for s in seeds:
        for runner, store in ((act_f, cos_on_final), (act_b, cos_on_bc)):
            obs, _ = env.reset(options={"scenario": int(s)})
            done = False
            while not done:
                a, ab = np.asarray(act_f(obs)), np.asarray(act_b(obs))
                store.append(float(np.dot(C.desired_direction(a, env.soldier_env._enemy_vel)[0], C.desired_direction(ab, env.soldier_env._enemy_vel)[0])))
                obs, _r, te, tr, _i = env.step(runner(obs).astype(np.float32))
                done = te or tr
    stat = lambda v: {"mean": float(np.mean(v)), "median": float(np.median(v)), "p5": float(np.percentile(v, 5)), "min": float(np.min(v)), "steps": len(v)}  # noqa: E731
    pick = ("capture", "median_path_excess", "mean_path_excess", "mean_duration_s", "mean_path_length_m", "exhaustion", "timeout", "gt2", "expert_cosine_mean", "expert_cosine_min")
    return {"seed": seed, "scenarios": [seeds[0], seeds[-1]], "stage1_bc": {k: mb[k] for k in pick}, "stage2_final": {k: mf[k] for k in pick},
            "action_cosine_final_vs_bc_on_final_trajectories": stat(cos_on_final), "action_cosine_final_vs_bc_on_bc_trajectories": stat(cos_on_bc)}


def decide(per_seed: dict, trainings: dict) -> dict:
    mastered = [s for s, t in trainings.items() if t["status"] == "MASTERED"]
    retained = [s for s in mastered if per_seed[s]["anchor_metrics"]["capture"] >= S.ANCHOR_RETENTION_MIN]
    pathology = {}
    for key in ("orbit_episodes", "repeated_flyby_episodes", "nonclosing_episodes", "nonfinite"):
        hit = [s for s in retained if per_seed[s]["metrics"][key] / per_seed[s]["metrics"]["episodes"] > COMMON_PATHOLOGY_RATE]
        if len(hit) >= COMMON_PATHOLOGY_SEEDS:
            pathology[key] = hit
    passed = len(mastered) >= 2 and len(retained) >= 2 and not pathology
    near = (not passed) and any(per_seed[s]["metrics"]["capture"] >= 0.5 for s in per_seed)
    return {"seeds_mastered": mastered, "seeds_retaining_stage1_pursuit": retained, "training_status": {str(s): t["status"] for s, t in trainings.items()}, "common_structural_pathology": pathology,
            "stage2_pass": bool(passed), "all_three_mastered": len(mastered) == 3, "learning_without_mastery": bool(near),
            "rule": ">= 2/3 seeds master (two consecutive validation passes within 8 rollouts), mastered seeds retain Stage-1 pursuit (anchor >= 0.98), no shared pathology"}


# ----------------------------------------------------------------------------- figures
def make_figures(trainings: dict, per_seed: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    colors = {201: "#1f77b4", 202: "#d62728", 203: "#2ca02c"}

    def curve(key, ylabel, fname, a0_key=None, getter=None, hline=None):
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for s, t in trainings.items():
            xs = [r["num_timesteps"] for r in t["rollouts"]]
            ys = [getter(r) for r in t["rollouts"]]
            ax.plot(xs, ys, marker="o", color=colors[s], label=f"A1-CUR2-{s}")
        if a0_key:
            r0 = next(iter(trainings.values()))["rollouts"]
            ax.plot([r["num_timesteps"] for r in r0], [r["a0_metrics"][a0_key] for r in r0], "k--", label="canonical A0 (matched)")
        if hline is not None:
            ax.axhline(hline[0], color="gray", ls=":", label=hline[1])
        ax.set_xlabel("PPO timesteps"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = FIG_DIR / f"{fname}.{ext}"
            fig.savefig(p, dpi=200)
            out.append(rel_fig(p))
        plt.close(fig)

    curve("capture", "validation capture rate", "fig1_capture_vs_timesteps", "capture", lambda r: r["validation_metrics"]["capture"], (0.90, "mastery 0.90"))
    curve("interception", "validation interception rate", "fig2_interception_vs_timesteps", "interception", lambda r: r["validation_metrics"]["interception"])
    curve("anchor", "Stage-1 anchor capture", "fig3_anchor_capture_vs_timesteps", None, lambda r: r["anchor_capture"], (0.98, "retention 0.98"))
    curve("path", "validation median path-excess", "fig4_path_excess_vs_timesteps", "median_path_excess", lambda r: r["validation_metrics"]["median_path_excess"], (1.60, "gate 1.60"))
    return out


def rel_fig(p: Path) -> str:
    return str(p.resolve().relative_to(TS.PROJECT_ROOT.resolve())).replace("\\", "/")


def make_trajectory_figure(per_seed: dict, a0_eps_traced: dict, seed_pick: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    res = per_seed[seed_pick]
    rows = [r for r in res["evasion_rows"] if r["side"] in ("over", "under", "left", "right")]
    if not rows:
        return []
    rows.sort(key=lambda r: abs(r["min_defender_distance_m"] - 3.0) + 0.01 * r["path_excess"])
    target = rows[0]["scenario_seed"]
    idx = [e["scenario_seed"] for e in res["_episodes_full"]].index(target)
    tr = res["_traces"][idx]
    a0 = a0_eps_traced[target]
    fig = plt.figure(figsize=(7.2, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*a0["pos"].T, "k--o", ms=3, label="canonical A0 (intercepted)")
    ax.plot(*tr["pos"].T, "-o", ms=3, color="#2ca02c", label=f"A1-CUR2-{seed_pick} (captured)")
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    d = tr["defender"]
    ax.plot_surface(d[0] + 2.5 * np.cos(u) * np.sin(v), d[1] + 2.5 * np.sin(u) * np.sin(v), d[2] + 2.5 * np.cos(v), color="red", alpha=0.25)
    ax.scatter(*d, color="red", s=20, label="stationary defender (r = 2.5 m)")
    ax.scatter(*tr["asset"], color="blue", marker="*", s=90, label="asset")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"scenario {target}: min separation {rows[0]['min_defender_distance_m']:.2f} m")
    fig.tight_layout()
    out = []
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"fig5_trajectory_a0_vs_a1_{seed_pick}.{ext}"
        fig.savefig(p, dpi=200)
        out.append(rel_fig(p))
    plt.close(fig)
    return out


def a0_traced(seeds: list[int]) -> dict:
    env = S.Stage2Env(0, 0, 0, scripted_a0=True)
    out = {}
    for s in seeds:
        ep, tr = traced_episode(env, s, lambda _o: np.zeros(3, dtype=np.float32))
        out[s] = tr
    return out


def run_all() -> None:
    seeds = S.final_evaluation_seeds()
    trainings = {s: seed_manifest(s) for s in S.PPO_SEEDS}
    a0_eps = S.evaluate(None, seeds, scripted_a0=True)
    per_seed = {s: evaluate_seed(s, seeds, a0_eps) for s in S.PPO_SEEDS}
    fidelity = {str(s): bc_fidelity(s) for s in S.PPO_SEEDS}
    decision = decide(per_seed, trainings)
    figs = make_figures(trainings, per_seed)
    pick = 203 if 203 in decision["seeds_mastered"] else max(per_seed, key=lambda s: per_seed[s]["metrics"]["capture"])
    if per_seed[pick]["evasion_rows"]:
        figs += make_trajectory_figure(per_seed, a0_traced(seeds), pick)
    payload = {"scenarios": [seeds[0], seeds[-1]], "episodes": len(seeds), "a0": {"metrics": S.metrics(a0_eps), "episodes": [compact(e) for e in a0_eps]},
               "per_seed": {str(s): {k: v for k, v in r.items() if not k.startswith("_")} for s, r in per_seed.items()}}
    TS.atomic_write_json(DOCS / "final_evaluation.json", payload)
    TS.atomic_write_json(DOCS / "evasion_analysis.json", {str(s): {"summary": r["evasion"], "reactive_vs_no_defender": r["reactive"], "rows": r["evasion_rows"]} for s, r in per_seed.items()})
    TS.atomic_write_json(DOCS / "bc_fidelity.json", fidelity)
    TS.atomic_write_json(DOCS / "stage2_decision.json", decision | {"figures": figs, "trajectory_figure_seed": pick})
    print(json.dumps(decision, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all",))
    parser.parse_args(argv)
    run_all()


if __name__ == "__main__":
    main()
