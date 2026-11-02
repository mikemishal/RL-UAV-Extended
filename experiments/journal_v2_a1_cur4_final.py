"""Final evaluation + analysis for curriculum Stage 4 (run after training; `python -m experiments.journal_v2_a1_cur4_final all`).

* every seed's accepted / final policy on the 200 matched scenarios 165000-165199 together with canonical A0 AND the seed's own Stage3-v2 parent on the SAME scenarios, with a verified common-random-number asset trajectory;
  paired Stage4 - A0 and Stage4 - Stage3-parent capture / interception differences (paired bootstrap 95% CI);
* moving-target pursuit analysis (is the policy tracking the CURRENT asset or an obsolete location?), failure analysis, and defender-evasion retention against the same policy flown without the defender (same asset path);
* the three retention anchors on the accepted checkpoint; generalization DIAGNOSTIC (Kalman-Greedy / PN / Lead, moving asset, same hold-until-detection scaffold) - never a mastery gate;
* the Stage-4 decision and figures 1-6 (PNG + PDF).
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
from experiments import journal_v2_a1_cur3v2_final as F3
from experiments import journal_v2_a1_cur4 as Y
from experiments import journal_v2_a1_cur4_state as TS

DOCS = TS.STATE_DIR
FIG_DIR = DOCS / "figures"
DT = 0.5
COMMON_PATHOLOGY_RATE, COMMON_PATHOLOGY_SEEDS = 0.01, 2
FIRST_PASS_OVERSHOOTS = 0
APPROACH_STEPS = 3


def accepted_model_path(seed: int) -> Path:
    return Y.RESULTS_ROOT / f"seed_{seed}" / "checkpoints" / "accepted_model.zip"


def seed_manifest(seed: int) -> dict:
    return json.loads((Y.RESULTS_ROOT / f"seed_{seed}" / "checkpoints" / "seed_manifest.json").read_text())


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def traced_episode(env, scenario_seed: int, act) -> tuple[dict, dict]:
    """One deterministic episode with a per-decision trace (attacker position, executed direction, line of sight to the CURRENT asset, asset position at the decision, defender state) + the full asset trace."""
    obs, _ = env.reset(options={"scenario": int(scenario_seed)})
    e = env.soldier_env
    pos, u_list, los_list, asset_list = [], [], [], []
    done, info = False, {}
    while not done:
        action = np.asarray(act(obs), dtype=np.float32)
        u, _held = C.desired_direction(action, e._enemy_vel)
        p = np.asarray(e._enemy_pos, dtype=np.float64)
        a = np.asarray(e._soldier_pos, dtype=np.float64)
        pos.append(p.copy())
        asset_list.append(a.copy())
        u_list.append(u)
        los_list.append(_unit(a - p))
        obs, _r, te, tr, info = env.step(action)
        done = te or tr
    pos.append(np.asarray(env._att[-1], dtype=np.float64))
    trace = {"pos": np.array(pos), "u": np.array(u_list), "los": np.array(los_list), "asset_at_decision": np.array(asset_list), "asset": np.asarray(e._soldier_pos, dtype=np.float64),
             "asset_trace": env.asset_trace()}
    if hasattr(env, "_def_pos"):
        trace["def_pos"], trace["def_vel"] = np.array(env._def_pos), np.array(env._def_vel)
    return dict(info["a1_episode"]), trace


def angle_deg(u, v) -> float:
    return float(np.degrees(np.arccos(np.clip(float(np.dot(_unit(u), _unit(v))), -1.0, 1.0))))


def tracking_features(trace: dict, episode: dict) -> dict:
    """Is the attacker flying at the CURRENT asset or at an obsolete (initial) location? Computed over the decisions of one episode."""
    pos, u, los = trace["pos"][:-1], trace["u"], trace["los"]
    a0 = trace["asset_trace"][0]
    cur = np.array([angle_deg(ui, li) for ui, li in zip(u, los)])
    obs_loc = np.array([angle_deg(ui, a0 - p) for ui, p in zip(u, pos)])
    rng = np.linalg.norm(trace["asset_at_decision"] - pos, axis=1)
    closing = -np.diff(np.append(rng, np.linalg.norm(trace["asset"] - trace["pos"][-1]))) / DT
    tail = slice(max(len(u) - APPROACH_STEPS, 0), len(u))
    disp = float(np.linalg.norm(trace["asset_trace"][-1] - trace["asset_trace"][0]))
    return {"scenario_seed": episode["scenario_seed"], "asset_final_displacement_m": disp, "asset_path_length_m": episode["asset_path_length_m"], "asset_displacement_max_m": episode["asset_displacement_max_m"],
            "heading_error_current_asset_mean_deg": float(cur.mean()), "heading_error_initial_asset_location_mean_deg": float(obs_loc.mean()),
            "final_approach_error_current_deg": float(cur[tail].mean()), "final_approach_error_initial_location_deg": float(obs_loc[tail].mean()),
            "tracks_current_asset": bool(cur[tail].mean() < obs_loc[tail].mean()), "mean_closing_speed_mps": float(closing.mean()), "overshoots": int(episode["overshoots"]),
            "first_pass": int(episode["overshoots"]) == FIRST_PASS_OVERSHOOTS, "path_excess": episode["path_excess"], "steps": episode["steps"]}


def summarize_tracking(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}

    def stat(key):
        v = np.array([r[key] for r in rows], dtype=np.float64)
        return {"mean": float(v.mean()), "median": float(np.median(v)), "p5": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)), "max": float(v.max())}

    disp = np.array([r["asset_final_displacement_m"] for r in rows])
    far = disp >= np.median(disp)
    return {"n": len(rows), "asset_final_displacement_m": stat("asset_final_displacement_m"), "asset_path_length_m": stat("asset_path_length_m"),
            "heading_error_current_asset_mean_deg": stat("heading_error_current_asset_mean_deg"), "heading_error_initial_asset_location_mean_deg": stat("heading_error_initial_asset_location_mean_deg"),
            "final_approach_error_current_deg": stat("final_approach_error_current_deg"), "final_approach_error_initial_location_deg": stat("final_approach_error_initial_location_deg"),
            "fraction_tracking_current_asset": float(np.mean([r["tracks_current_asset"] for r in rows])),
            "fraction_tracking_current_asset_when_asset_moved_more_than_median": float(np.mean([r["tracks_current_asset"] for r, f in zip(rows, far) if f])),
            "mean_closing_speed_mps": stat("mean_closing_speed_mps"), "first_pass_capture_fraction": float(np.mean([r["first_pass"] for r in rows])), "overshoots": stat("overshoots"),
            "definitions": {"heading error": "angle between the executed direction and the line of sight to the asset's CURRENT position at the decision (vs to the asset's INITIAL position = an obsolete location)",
                            "tracks_current_asset": f"over the last {APPROACH_STEPS} decisions the mean heading error to the current asset is smaller than to the initial asset location",
                            "first_pass": "no overshoot / re-approach in the episode"}}


def evasion_target_motion(trace: dict, trace0: dict, episode: dict) -> dict | None:
    """Horizontal target-motion direction (detection -> closest defender approach) vs the lateral deviation from the same policy's no-defender path at the closest approach (same asset trajectory)."""
    t_det = episode.get("detection_step")
    if t_det is None:
        return None
    n = min(len(trace["pos"]), len(trace0["pos"]), len(trace["def_pos"]))
    dist = np.linalg.norm(trace["pos"][:n] - trace["def_pos"][:n], axis=1)
    i = int(np.argmin(dist))
    motion = (trace["asset_trace"][min(i, len(trace["asset_trace"]) - 1)] - trace["asset_trace"][min(t_det, len(trace["asset_trace"]) - 1)])[:2]
    dev = (trace["pos"][i] - trace0["pos"][i])[:2]
    if np.linalg.norm(motion) < 1e-6 or np.linalg.norm(dev) < 1e-3:
        return {"scenario_seed": episode["scenario_seed"], "cosine": None}
    return {"scenario_seed": episode["scenario_seed"], "cosine": float(np.dot(_unit(motion), _unit(dev))), "asset_motion_m": float(np.linalg.norm(motion)), "deviation_m": float(np.linalg.norm(dev))}


def no_defender_traces(act, seeds: list[int]) -> list[dict]:
    env = Y.Stage4NoDefenderEnv(0, 0, 0)
    return [traced_episode_nodef(env, s, act) for s in seeds]


def traced_episode_nodef(env, s: int, act) -> dict:
    obs, _ = env.reset(options={"scenario": int(s)})
    e = env.soldier_env
    pos, done = [], False
    while not done:
        pos.append(np.asarray(e._enemy_pos, dtype=np.float64).copy())
        obs, _r, te, tr, _i = env.step(np.asarray(act(obs), dtype=np.float32))
        done = te or tr
    pos.append(np.asarray(env._att[-1], dtype=np.float64))
    return {"pos": np.array(pos)}


def compact(ep: dict) -> dict:
    keep = F3.compact(ep) | {k: ep[k] for k in ("asset_final_displacement_m", "asset_path_length_m", "asset_displacement_max_m", "terminal_distance_m", "overshoots", "endurance_exhausted") if k in ep}
    return keep


def outcome_breakdown(eps: list[dict]) -> dict:
    n = len(eps)
    out = {}
    for name in sorted({e["outcome"] for e in eps}):
        sub = [e for e in eps if e["outcome"] == name]
        out[name] = {"n": len(sub), "fraction": len(sub) / n, "median_asset_final_displacement_m": float(np.median([e["asset_final_displacement_m"] for e in sub])),
                     "median_terminal_distance_m": float(np.median([e["terminal_distance_m"] for e in sub])), "median_path_length_m": float(np.median([e["path_length_m"] for e in sub])),
                     "median_steps": float(np.median([e["steps"] for e in sub]))}
    return out


def evaluate_seed(seed: int, seeds: list[int], a0_eps: list[dict]) -> dict:
    from stable_baselines3 import PPO

    model = PPO.load(str(accepted_model_path(seed)).removesuffix(".zip"), device="cpu")
    parent = PPO.load(Y.parent_checkpoint(seed)["path"].removesuffix(".zip"), device="cpu")
    act, pact = C.model_actor(model), C.model_actor(parent)
    env, penv, aenv = Y.Stage4Env(0, 0, 0), Y.Stage4Env(0, 0, 0), Y.Stage4Env(0, 0, 0, scripted_a0=True)
    eps, traces, peps, ptraces, atr = [], [], [], [], {}
    for s in seeds:
        ep, tr = traced_episode(env, s, act)
        pe, ptr = traced_episode(penv, s, pact)
        _ae, at = traced_episode(aenv, s, lambda _o: np.zeros(3, dtype=np.float32))
        eps.append(ep), traces.append(tr), peps.append(pe), ptraces.append(ptr)
        atr[s] = at["asset_trace"]
    crn = Y.crn_report({"a0": atr, "stage3_parent": {s: t["asset_trace"] for s, t in zip(seeds, ptraces)}, "stage4": {s: t["asset_trace"] for s, t in zip(seeds, traces)}})
    base = no_defender_traces(act, seeds)
    m, m0, mp = Y.metrics(eps), Y.metrics(a0_eps), Y.metrics(peps)
    an = {k: f(act) for k, f in (("anchor1", Y.evaluate_anchor1), ("anchor2", Y.evaluate_anchor2), ("anchor3", Y.evaluate_anchor3))}
    pan = {k: f(pact) for k, f in (("anchor1", Y.evaluate_anchor1), ("anchor2", Y.evaluate_anchor2), ("anchor3", Y.evaluate_anchor3))}
    anm = {"anchor1": V1V2.metrics(an["anchor1"]), "anchor2": Y.metrics(an["anchor2"]), "anchor3": Y.metrics(an["anchor3"])}
    pam = {"anchor1": V1V2.metrics(pan["anchor1"])["capture"], "anchor2": Y.metrics(pan["anchor2"])["capture"], "anchor3": Y.metrics(pan["anchor3"])["capture"]}
    gate = Y.gate_status(m, a0_capture=m0["capture"], anchor1=anm["anchor1"]["capture"], anchor2=anm["anchor2"]["capture"], anchor3=anm["anchor3"]["capture"])
    succ = lambda es: np.array([e["outcome"] == "soldier_caught" for e in es], dtype=float)   # noqa: E731
    icpt = lambda es: np.array([e["intercepted"] for e in es], dtype=float)                    # noqa: E731
    exh = lambda es: np.array([e["outcome"] == "enemy_endurance_exhausted" for e in es], dtype=float)  # noqa: E731
    wins = [i for i, e in enumerate(eps) if e["outcome"] == "soldier_caught"]
    pwins = [i for i, e in enumerate(peps) if e["outcome"] == "soldier_caught"]
    track = [tracking_features(traces[i], eps[i]) for i in wins]
    ptrack = [tracking_features(ptraces[i], peps[i]) for i in pwins]
    behavior = [F3.behavior_features(traces[i], base[i], eps[i]) for i in wins if eps[i].get("detection_step") is not None]
    tmotion = [r for r in (evasion_target_motion(traces[i], base[i], eps[i]) for i in wins) if r is not None]
    manifest = seed_manifest(seed)
    return {"seed": seed, "accepted_model_sha256": manifest["accepted_model_sha256"], "training_status": manifest["status"], "mastery_rollout": manifest["mastery_rollout"], "crn": crn,
            "metrics": m, "a0_metrics": m0, "parent_metrics": mp, "anchor_metrics": anm, "parent_anchor_captures": pam, "gate_on_200": gate,
            "paired_capture_vs_a0": S.paired_bootstrap(succ(eps), succ(a0_eps)), "paired_interception_vs_a0": S.paired_bootstrap(icpt(eps), icpt(a0_eps)),
            "paired_capture_vs_parent": S.paired_bootstrap(succ(eps), succ(peps)), "paired_interception_vs_parent": S.paired_bootstrap(icpt(eps), icpt(peps)),
            "paired_exhaustion_vs_parent": S.paired_bootstrap(exh(eps), exh(peps)), "outcomes_stage4": outcome_breakdown(eps), "outcomes_parent": outcome_breakdown(peps),
            "tracking_stage4": summarize_tracking(track), "tracking_parent": summarize_tracking(ptrack), "evasion_retention": F3.summarize_behavior(behavior), "evasion_rows": behavior,
            "target_motion_vs_evasion": {"n": len(tmotion), "cosine_mean": float(np.mean([r["cosine"] for r in tmotion if r["cosine"] is not None])) if any(r["cosine"] is not None for r in tmotion) else None,
                                         "fraction_deviating_with_target_motion": float(np.mean([r["cosine"] > 0 for r in tmotion if r["cosine"] is not None])) if any(r["cosine"] is not None for r in tmotion) else None,
                                         "note": "cosine between the horizontal asset displacement (detection -> closest defender approach) and the attacker's horizontal deviation from its own no-defender path; descriptive only"},
            "tracking_rows": track, "episodes": [compact(e) for e in eps], "parent_episodes": [compact(e) for e in peps], "_traces": traces, "_parent_traces": ptraces, "_episodes_full": eps}


def generalization(seed: int) -> dict:
    """DIAGNOSTIC ONLY: Stage-4 policy, its Stage3-v2 parent and canonical A0 against other scripted interceptors under the moving asset and the same hold-until-detection scaffold."""
    from stable_baselines3 import PPO

    act = C.model_actor(PPO.load(str(accepted_model_path(seed)).removesuffix(".zip"), device="cpu"))
    pact = C.model_actor(PPO.load(Y.parent_checkpoint(seed)["path"].removesuffix(".zip"), device="cpu"))
    seeds = list(Y.seeds_of(Y.DIAGNOSTIC))
    out = {}
    pick = ("capture", "interception", "timeout", "exhaustion", "median_path_excess", "min_defender_separation_median_m", "orbit_episodes", "repeated_flyby_episodes")
    for controller in ("greedy",) + tuple(Y.DIAGNOSTIC_CONTROLLERS):
        a1 = Y.metrics(Y.evaluate(act, seeds, controller=controller))
        p = Y.metrics(Y.evaluate(pact, seeds, controller=controller))
        a0 = Y.metrics(Y.evaluate(None, seeds, scripted_a0=True, controller=controller))
        out[controller] = {"stage4": {k: a1[k] for k in pick}, "stage3_parent": {k: p[k] for k in pick}, "matched_a0": {k: a0[k] for k in pick},
                           "capture_minus_a0_pp": 100.0 * (a1["capture"] - a0["capture"]), "capture_minus_parent_pp": 100.0 * (a1["capture"] - p["capture"])}
    return {"seed": seed, "scenarios": [seeds[0], seeds[-1]], "episodes": len(seeds), "controllers": out, "note": "diagnostic only; not part of mastery or training"}


def decide(per_seed: dict, trainings: dict) -> dict:
    mastered = [s for s, t in trainings.items() if t["status"] == "MASTERED"]
    retained = [s for s in mastered if all(per_seed[s]["anchor_metrics"][k]["capture"] >= v for k, v in (("anchor1", Y.ANCHOR_MIN["anchor1_basic_pursuit"]), ("anchor2", Y.ANCHOR_MIN["anchor2_static_defender"]),
                                                                                                          ("anchor3", Y.ANCHOR_MIN["anchor3_moving_defender"])))]
    pathology = {}
    for key in ("orbit_episodes", "repeated_flyby_episodes", "nonclosing_episodes", "nonfinite"):
        hit = [s for s in retained if per_seed[s]["metrics"][key] / per_seed[s]["metrics"]["episodes"] > COMMON_PATHOLOGY_RATE]
        if len(hit) >= COMMON_PATHOLOGY_SEEDS:
            pathology[key] = hit
    passed = len(mastered) >= 2 and len(retained) >= 2 and not pathology
    improving = [s for s in per_seed if per_seed[s]["metrics"]["capture"] >= 0.60 and per_seed[s]["paired_capture_vs_parent"]["ci95_pp"][0] > 0.0]
    promising = (not passed) and (len(mastered) >= 1 or len(improving) >= 2)
    return {"seeds_mastered": mastered, "seeds_retaining_all_anchors": retained, "training_status": {str(s): t["status"] for s, t in trainings.items()}, "common_structural_pathology": pathology,
            "stage4_pass": bool(passed), "all_three_mastered": len(mastered) == 3, "seeds_improving_over_parent_with_capture_ge_0.60": improving, "promising_manual_review": bool(promising),
            "rule": ">= 2/3 seeds master (two consecutive validation passes within 16 rollouts), mastered seeds retain all three anchors, no shared pathology; verdict B if not passed but >= 1 seed mastered or >= 2 seeds "
                    "reach held-out capture >= 0.60 with a paired-CI lower bound above the Stage3-v2 parent"}


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
            ax.plot([r["num_timesteps"] for r in t["rollouts"]], [getter(r) for r in t["rollouts"]], marker="o", color=colors[s], label=f"A1-CUR4-{s}")
            if parent_key:
                ax.axhline(per_seed[s]["parent_metrics"][parent_key], color=colors[s], ls="--", lw=0.8, alpha=0.6)
        if a0_key:
            r0 = next(iter(trainings.values()))["rollouts"]
            ax.plot([r["num_timesteps"] for r in r0], [r["a0_metrics"][a0_key] for r in r0], "k--", label="canonical A0 (matched)")
        if hline is not None:
            ax.axhline(hline[0], color="gray", ls=":", label=hline[1])
        ax.set_xlabel("Stage-4 PPO timesteps"); ax.set_ylabel(ylabel); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        save(fig, fname)

    curve("validation capture rate", "fig1_capture_vs_timesteps", lambda r: r["validation_metrics"]["capture"], "capture", (0.70, "mastery 0.70"), "capture")
    curve("validation interception rate", "fig2_interception_vs_timesteps", lambda r: r["validation_metrics"]["interception"], "interception", (0.30, "gate 0.30"), "interception")
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharex=True)
    for ax, (key, title, thr) in zip(axes, (("anchor1_capture", "Anchor 1: basic pursuit", 0.98), ("anchor2_capture", "Anchor 2: static defender", 0.80), ("anchor3_capture", "Anchor 3: moving defender", 0.75))):
        for s, t in trainings.items():
            ax.plot([r["num_timesteps"] for r in t["rollouts"]], [r[key] for r in t["rollouts"]], marker="o", color=colors[s], label=f"A1-CUR4-{s}")
        ax.axhline(thr, color="gray", ls=":"); ax.set_title(title, fontsize=10); ax.set_xlabel("Stage-4 PPO timesteps"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("anchor capture"); axes[0].legend(fontsize=7)
    save(fig, "fig3_retention_anchors_vs_timesteps")
    curve("validation median path-excess", "fig4_path_excess_vs_timesteps", lambda r: r["validation_metrics"]["median_path_excess"], "median_path_excess", (2.0, "gate 2.00"))
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    seeds_ = sorted(per_seed)
    d = [per_seed[s]["paired_capture_vs_parent"] for s in seeds_]
    ax.bar([str(s) for s in seeds_], [x["difference_pp"] for x in d], color=[colors[s] for s in seeds_],
           yerr=[[x["difference_pp"] - x["ci95_pp"][0] for x in d], [x["ci95_pp"][1] - x["difference_pp"] for x in d]], capsize=4)
    ax.axhline(0, color="k", lw=0.8); ax.set_ylabel("Stage4 - Stage3-parent capture [pp] (paired 95% CI)"); ax.set_xlabel("PPO seed"); ax.grid(alpha=0.3, axis="y")
    save(fig, "fig5_stage3_parent_vs_stage4_improvement")
    return out


def make_trajectory_figure(per_seed: dict, seed_pick: int) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = per_seed[seed_pick]
    cand = [(i, e) for i, e in enumerate(res["_episodes_full"]) if e["outcome"] == "soldier_caught" and res["parent_episodes"][i]["outcome"] != "soldier_caught"] or \
           [(i, e) for i, e in enumerate(res["_episodes_full"]) if e["outcome"] == "soldier_caught"]
    if not cand:
        return []
    cand.sort(key=lambda t: -t[1]["asset_final_displacement_m"])
    idx, ep = cand[0]
    target = ep["scenario_seed"]
    tr, ptr = res["_traces"][idx], res["_parent_traces"][idx]
    a0e = Y.Stage4Env(0, 0, 0, scripted_a0=True)
    _ae, a0 = traced_episode(a0e, target, lambda _o: np.zeros(3, dtype=np.float32))
    fig = plt.figure(figsize=(7.6, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*a0["pos"].T, "k--o", ms=3, label="canonical A0 (intercepted)")
    ax.plot(*ptr["pos"].T, "-s", ms=3, color="#ff7f0e", label=f"Stage3-v2 parent ({res['parent_episodes'][idx]['outcome']})")
    ax.plot(*tr["pos"].T, "-o", ms=3, color="#2ca02c", label=f"Stage4 A1-CUR4-{seed_pick} (captured)")
    ax.plot(*tr["def_pos"].T, "-x", ms=4, color="#d62728", label="Greedy defender vs Stage4")
    ax.plot(*tr["asset_trace"].T, "-", color="blue", lw=2, label="moving asset (common realized path)")
    ax.scatter(*tr["asset_trace"][0], color="blue", marker="*", s=80)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]"); ax.legend(fontsize=7, loc="upper left")
    ax.set_title(f"scenario {target}: asset displacement {ep['asset_final_displacement_m']:.1f} m")
    fig.tight_layout()
    out = []
    for ext in ("png", "pdf"):
        p = FIG_DIR / f"fig6_trajectory_a0_stage3_stage4_{seed_pick}.{ext}"
        fig.savefig(p, dpi=200)
        out.append(rel_fig(p))
    plt.close(fig)
    return out


def run_all() -> None:
    seeds = Y.final_evaluation_seeds()
    trainings = {s: seed_manifest(s) for s in Y.PPO_SEEDS}
    a0_eps = Y.evaluate(None, seeds, scripted_a0=True)
    per_seed = {s: evaluate_seed(s, seeds, a0_eps) for s in Y.PPO_SEEDS}
    gen = {str(s): generalization(s) for s in Y.PPO_SEEDS}
    decision = decide(per_seed, trainings)
    figs = make_figures(trainings, per_seed)
    pick = 203 if 203 in decision["seeds_mastered"] else max(per_seed, key=lambda s: per_seed[s]["metrics"]["capture"])
    figs += make_trajectory_figure(per_seed, pick)
    payload = {"scenarios": [seeds[0], seeds[-1]], "episodes": len(seeds), "a0": {"metrics": Y.metrics(a0_eps), "episodes": [compact(e) for e in a0_eps]},
               "per_seed": {str(s): {k: v for k, v in r.items() if not k.startswith("_")} for s, r in per_seed.items()}}
    TS.atomic_write_json(DOCS / "final_evaluation.json", payload)
    TS.atomic_write_json(DOCS / "moving_target_analysis.json", {str(s): {"stage4": r["tracking_stage4"], "stage3_parent": r["tracking_parent"], "outcomes_stage4": r["outcomes_stage4"],
                                                                      "outcomes_parent": r["outcomes_parent"], "target_motion_vs_evasion": r["target_motion_vs_evasion"], "rows": r["tracking_rows"]} for s, r in per_seed.items()})
    TS.atomic_write_json(DOCS / "evasion_retention_analysis.json", {str(s): {"summary": r["evasion_retention"], "rows": r["evasion_rows"]} for s, r in per_seed.items()})
    TS.atomic_write_json(DOCS / "generalization_diagnostic.json", gen)
    TS.atomic_write_json(DOCS / "stage4_decision.json", decision | {"figures": figs, "trajectory_figure_seed": pick})
    print(json.dumps(decision, indent=1))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("all",))
    parser.parse_args(argv)
    run_all()


if __name__ == "__main__":
    main()
