"""Build and verify the Stage3-v2 HUMAN-REVIEW recordings (visualizer support only; the research repository is read, never written).

    python -m visualizer.review.build_stage3v2_review            # select -> replay -> verify -> save recordings + shortlist (+ JSON/MD)
    python -m visualizer.review.build_stage3v2_review --cases    # additionally create research cases (notes + bookmarks) through the running backend

Selection uses only the committed research results of Stage3-v2 (`final_evaluation.json`, `behavior_analysis.json`, seed manifests) plus - for the generalization recordings - a fresh research evaluation of the
diagnostic block. Every recording is replayed through the visualizer engine and then compared with an independent research evaluation of the same checkpoint on the same scenario (outcome, step count, attacker AND
defender trajectories over every frame, release step, checkpoint SHA-256). No scientific status is changed: the policies remain PROVISIONAL curriculum checkpoints (non-final, non-canonical).
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REVIEW_DIR = Path(__file__).resolve().parent
SHORTLIST_JSON = REVIEW_DIR / "stage3v2_review_shortlist.json"
SHORTLIST_MD = REVIEW_DIR / "stage3v2_review_shortlist.md"
DOCS = "docs/journal_v2_a1_curriculum/stage3_v2"
LABEL = "Stage3-v2 REVIEW — PROVISIONAL CURRICULUM CHECKPOINT / NON-FINAL / NON-CANONICAL"
POSITION_TOLERANCE_M = 1e-3          # replay frames are float32 simulator state; the research trace is the same simulator state in float64
PATH_EXCESS_TOLERANCE = 1e-3
BACKEND = "http://127.0.0.1:8000"
FRONTEND = "http://127.0.0.1:5173"
DT = 0.5
DIAGNOSTIC_SEEDS = range(156_000, 156_100)

CHECKLIST = [
    "Does A1 fly toward the asset normally before the defender becomes visible (inside the 25 m information gate) — no early swerving?",
    "Does it deviate only AFTER defender information becomes available (gate entry / detection), and does the deviation start promptly?",
    "Is the evasion smooth rather than random or wandering?",
    "Does it create enough separation to avoid the defender (interception radius 2.5 m) while the defender accelerates toward it after release?",
    "After clearing the defender, does it turn back toward the asset?",
    "Does it capture on the first reasonable approach?",
    "Any unnecessary loops, oscillations, fly-bys or excessive detours? (Note the defender keeps chasing after the pass — does the attacker stay ahead of it?)",
]
KIND_NOTE = {
    "representative_high_quality_success": "Showcase case (no deviation before the gate, prompt reactive deviation, return to pursuit, efficient path, comfortable separation). Use it as the reference for what the policy does when it works.",
    "closest_successful_pass": "Edge case: the narrowest successful pass; check that the margin to the 2.5 m interception sphere is real and the maneuver deliberate.",
    "interception_failure": "Failure case: the longest attempted evasion before interception; look at whether the dodge was too late, too small, or toward the side the defender was already steering.",
    "lateral_evasion": "Lateral example: the largest horizontal displacement from the same policy's no-defender path.",
    "vertical_evasion": "Vertical example: the largest vertical displacement from the same policy's no-defender path (altitude change to clear the defender).",
    "mixed_evasion": "Mixed example: a combined horizontal + vertical maneuver (the most common class).",
    "representative_success": "Typical case: nothing special selected; judge overall smoothness and efficiency.",
    "matched_a0_reference": "Reference: canonical scripted A0 in the SAME scenario — it flies straight at the asset and is intercepted. Compare its path with the A1 recording of the same scenario.",
    "generalization_capture": "Generalization case: a controller the policy was NEVER trained against (same hold-until-detection semantics). The Stage2-v2 parent is intercepted in this scenario; check that the Stage-3 policy still shows the same evade-then-resume pattern.",
}


def load_research(root: Path) -> dict:
    final = json.loads((root / DOCS / "final_evaluation.json").read_text())
    behavior = json.loads((root / DOCS / "behavior_analysis.json").read_text())
    manifests = {s: json.loads((root / DOCS / f"seed_{s}_manifest.json").read_text()) for s in (201, 202, 203)}
    return {"final": final, "behavior": behavior, "manifests": manifests}


def episode_table(data: dict, seed: int) -> list[dict]:
    eps = {e["scenario_seed"]: dict(e) for e in data["final"]["per_seed"][str(seed)]["episodes"]}
    for r in data["behavior"][str(seed)]["rows"]:
        eps[r["scenario_seed"]]["behavior"] = r
    return [eps[k] for k in sorted(eps)]


def attacker_id(seed: int) -> str:
    return f"A1-CUR3V2-{seed}"


def generalization_pick(research_root: Path, controller: str, seed: int = 203) -> dict:
    """Seed-203 policy vs `controller` on the diagnostic block: a scenario the Stage2-v2 parent LOSES and the Stage-3 policy WINS, with the winner's closest approach nearest the median of such wins."""
    import experiments.journal_v2_a1_cur1 as C
    import experiments.journal_v2_a1_cur3v2 as X
    from stable_baselines3 import PPO

    act = C.model_actor(PPO.load(str(research_root / "results/journal_v2_corrected_semantics/a1_curriculum/stage3_v2" / f"seed_{seed}" / "checkpoints" / "accepted_model.zip").removesuffix(".zip"), device="cpu"))
    pact = C.model_actor(PPO.load(X.parent_checkpoint(seed)["path"].removesuffix(".zip"), device="cpu"))
    seeds = list(DIAGNOSTIC_SEEDS)
    a1, parent = X.evaluate(act, seeds, controller=controller), X.evaluate(pact, seeds, controller=controller)
    wins = [(e["scenario_seed"], e["min_defender_separation_m"]) for e, p in zip(a1, parent) if e["outcome"] == "soldier_caught" and p["intercepted"]]
    med = float(np.median([w[1] for w in wins]))
    chosen = min(wins, key=lambda w: (abs(w[1] - med), w[0]))
    return {"scenario": chosen[0], "wins_over_parent": len(wins), "a1_capture": float(np.mean([e["outcome"] == "soldier_caught" for e in a1])), "parent_capture": float(np.mean([not p["intercepted"] for p in parent]))}


def select_recordings(data: dict, research_root: Path) -> list[dict]:
    chosen: dict[int, set[int]] = {201: set(), 202: set(), 203: set()}
    out: list[dict] = []

    def add(seed, scenario, kind, title, defender="greedy", extra=None):
        chosen.setdefault(seed, set()).add(scenario)
        out.append({"attacker": attacker_id(seed), "model_seed": seed, "defender": defender, "evaluation_seed": scenario, "kind": kind, "title": title, **(extra or {})})

    tables = {s: episode_table(data, s) for s in (201, 202, 203)}

    def wins(seed):
        return [e for e in tables[seed] if e["outcome"] == "soldier_caught" and "behavior" in e]

    for seed in (203, 201, 202):
        w = wins(seed)
        med_pe = float(np.median([e["path_excess"] for e in w]))
        med_md = float(np.median([e["behavior"]["min_distance_m"] for e in w]))
        med_disp = float(np.median([e["behavior"]["max_displacement_vs_no_defender_m"] for e in w]))
        if seed in (201, 203):
            good = [e for e in w if e["behavior"]["class"] != "unclear" and not e["behavior"]["deviation_before_visibility"] and e["behavior"]["reacquisition_time_s"] is not None
                    and e["behavior"]["first_meaningful_deviation_step"] is not None and e["behavior"]["first_meaningful_deviation_step"] <= e["behavior"]["closest_step"]
                    and e["behavior"]["max_displacement_vs_no_defender_m"] >= med_disp
                    and (e.get("defender_first_visible_step") or 0) >= 2 and e["path_excess"] <= med_pe and e["behavior"]["min_distance_m"] >= med_md]
            best = max(good or w, key=lambda e: e["behavior"]["post_encounter_mean_cos_to_asset"] - 2.0 * (e["path_excess"] - 1.0) - 1e-6 * e["scenario_seed"])
            add(seed, best["scenario_seed"], "representative_high_quality_success", f"Seed {seed} — representative HIGH-QUALITY successful evasion of the moving defender", extra={"high_quality": True})
        else:
            best = min(w, key=lambda e: abs(e["path_excess"] - med_pe) / 0.05 + abs(e["behavior"]["min_distance_m"] - med_md) + 1e-6 * e["scenario_seed"])
            add(seed, best["scenario_seed"], "representative_success", f"Seed {seed} — representative successful capture")
        rest = [e for e in w if e["scenario_seed"] not in chosen[seed]]
        closest = min(rest, key=lambda e: (e["behavior"]["min_distance_m"], e["scenario_seed"]))
        add(seed, closest["scenario_seed"], "closest_successful_pass", f"Seed {seed} — CLOSEST successful defender pass")
        fails = [e for e in tables[seed] if e["outcome"] != "soldier_caught"]
        if fails:
            worst = max(fails, key=lambda e: (e["steps"], -e["scenario_seed"]))
            add(seed, worst["scenario_seed"], "interception_failure", f"Seed {seed} — INTERCEPTION failure (longest attempted evasion)")
    # class examples (seed 203 for lateral/mixed; the seed with the strongest vertical maneuver for vertical)
    w203 = [e for e in wins(203) if e["scenario_seed"] not in chosen[203]]
    lat = max([e for e in w203 if e["behavior"]["class"] == "lateral"], key=lambda e: (e["behavior"]["max_lateral_m"], -e["scenario_seed"]))
    add(203, lat["scenario_seed"], "lateral_evasion", "Seed 203 — LATERAL evasion (largest horizontal detour)")
    mixed = max([e for e in wins(203) if e["behavior"]["class"] == "mixed" and e["scenario_seed"] not in chosen[203]], key=lambda e: (e["behavior"]["max_displacement_vs_no_defender_m"], -e["scenario_seed"]))
    add(203, mixed["scenario_seed"], "mixed_evasion", "Seed 203 — MIXED lateral + vertical evasion (largest displacement)")
    vert_candidates = [(s, e) for s in (201, 202, 203) for e in wins(s) if e["behavior"]["class"] == "vertical" and e["scenario_seed"] not in chosen[s]]
    vs, ve = max(vert_candidates, key=lambda t: (t[1]["behavior"]["max_vertical_m"], -t[1]["scenario_seed"]))
    add(vs, ve["scenario_seed"], "vertical_evasion", f"Seed {vs} — VERTICAL evasion (largest altitude change)")
    first = next(o for o in out if o["model_seed"] == 203 and o["kind"] == "representative_high_quality_success")
    out.append({"attacker": "A0-CUR3V2-REFERENCE", "model_seed": None, "defender": "greedy", "evaluation_seed": first["evaluation_seed"], "kind": "matched_a0_reference",
                "title": f"Matched canonical A0 reference (same scenario as the seed-203 high-quality success, {first['evaluation_seed']})"})
    for controller in ("lead", "pn"):
        pick = generalization_pick(research_root, controller)
        add(203, pick["scenario"], "generalization_capture", f"Seed 203 vs {controller.upper()} (never trained against) — captures where the Stage2-v2 parent is intercepted", defender=controller,
            extra={"generalization": pick, "diagnostic_block": [DIAGNOSTIC_SEEDS[0], DIAGNOSTIC_SEEDS[-1]]})
    priority = {("A1-CUR3V2-203", "representative_high_quality_success"): 1, ("A1-CUR3V2-203", "closest_successful_pass"): 2, ("A1-CUR3V2-203", "interception_failure"): 3, ("A0-CUR3V2-REFERENCE", "matched_a0_reference"): 4,
                ("A1-CUR3V2-201", "representative_high_quality_success"): 5}
    for o in out:
        o["priority"] = priority.get((o["attacker"], o["kind"]))
    return sorted(out, key=lambda o: (o["priority"] is None, o["priority"] or 0, o["kind"] == "generalization_capture", o["attacker"], o["kind"]))


def notes_for(rec: dict, table_row: dict | None) -> str:
    """Per-recording `what_to_look_for`: the fixed questions plus recording-specific facts from the research analysis / the verified replay."""
    facts = []
    s = rec["stats"]
    gate = s.get("defender_first_visible_step")
    facts.append(f"Defender controller: {rec['defender']} (unmodified research policy), HELD stationary until canonical detection (15 m), then released"
                 + (f" at step {s['release_step']} (t = {s['release_step'] * DT:.1f} s)" if s.get("release_step") is not None else " (never released: the episode ended first)")
                 + " — the held phase is a curriculum scaffold, not the final defender benchmark.")
    if gate is not None:
        facts.append(f"The defender is inside the 25 m information gate from step {gate} (t = {gate * DT:.1f} s)" + (" — i.e. already at the start" if gate == 0 else "; before that A1 sees no defender features") + ".")
    b = (table_row or {}).get("behavior")
    if rec["kind"] == "matched_a0_reference":
        facts.append("Canonical A0 flies straight at the asset, ignores the defender and is INTERCEPTED (A0 captures 0/200 in this scenario family; contact separation 2.5 m).")
    elif b:
        facts.append(f"Closest defender approach at step {b['closest_step']} (t = {b['closest_step'] * DT:.1f} s), separation {b['min_distance_m']:.2f} m (A0: exactly 2.5 m = contact); defender speed there "
                     f"{b['defender_speed_at_closest_mps']:.1f} m/s; class: {b['class']}; max displacement from the same policy's no-defender path {b['max_displacement_vs_no_defender_m']:.1f} m "
                     f"(lateral {b['max_lateral_m']:.1f} m, vertical {b['max_vertical_m']:.1f} m).")
        if b.get("time_detection_to_meaningful_deviation_s") is not None:
            t_det, t_vis = b["time_detection_to_meaningful_deviation_s"], b["time_visibility_to_meaningful_deviation_s"]
            when = lambda t, ref: f"{abs(t):.1f} s {'before' if t < 0 else 'after'} {ref}"  # noqa: E731
            facts.append(f"First >1 m deviation from the no-defender path {when(t_det, 'detection (= release)')} and {when(t_vis, 'gate entry')} "
                         f"(the deviation starts as soon as defender information exists; the defender is detected, and released, once it is within 15 m); deviation before gate entry: "
                         f"{'yes' if b['deviation_before_visibility'] else 'no'}.")
        ahead = b.get("passes_ahead_of_defender")
        facts.append(f"Passes {'in FRONT of' if ahead else 'BEHIND'} the defender's velocity vector; post-encounter mean cosine toward the asset {b['post_encounter_mean_cos_to_asset']:.3f}; "
                     f"reacquisition {b['reacquisition_time_s'] if b['reacquisition_time_s'] is not None else 'n/a'} s; path-excess {b['path_excess']:.3f}.")
    elif rec["kind"] == "interception_failure":
        facts.append(f"Interception failure after {s['steps']} steps ({s['duration_s']:.1f} s): the policy survived the longest of this seed's failures but still entered the 2.5 m radius.")
    elif rec["kind"] == "generalization_capture":
        g = rec["generalization"]
        facts.append(f"Diagnostic block {rec['diagnostic_block'][0]}–{rec['diagnostic_block'][1]}: the seed-203 policy captures {g['a1_capture']:.0%} against {rec['defender']} (Stage2-v2 parent {g['parent_capture']:.0%}); "
                     f"this scenario is one of {g['wins_over_parent']} the parent loses and the Stage-3 policy wins (closest approach nearest the median of those wins).")
    questions = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(CHECKLIST))
    return f"{LABEL}\n{KIND_NOTE[rec['kind']]}\n\nFacts (research analysis / verified replay):\n- " + "\n- ".join(facts) + f"\n\nWhat to look for:\n{questions}"


def run_replays(records: list[dict]) -> None:
    from visualizer.backend.config import load_settings
    from visualizer.backend.replay_engine import ReplayRequest, run_replay_episode, save_replay_bundle
    from visualizer.backend.replay_jobs import _replay_id_for_path
    from visualizer.backend.replay_requests import ALL_ATTACKER_SEEDS

    settings = load_settings()
    # Save in REVERSE priority order so that the first recording to inspect is the newest file (the replay list is sorted newest-first).
    for rec in reversed(records):
        request = ReplayRequest(defender_policy_id=rec["defender"], attacker_condition=rec["attacker"], attacker_model_seed=ALL_ATTACKER_SEEDS.get(rec["attacker"]), clutter_level="C0",
                                perception_mode="map_local", semantics="V2", evaluation_seed=rec["evaluation_seed"], deterministic_inference=True)
        bundle = run_replay_episode(request, settings)
        path = save_replay_bundle(bundle, None)
        rec["recording_name"] = path.name.removesuffix(".json.gz")
        rec["replay_id"] = _replay_id_for_path(path)                      # the opaque id the API / UI use (sha-256 of the file name)
        rec["replay_file"] = str(path.relative_to(Path(__file__).resolve().parents[2])).replace("\\", "/")
        rec["_bundle"] = bundle
        print(f"[replay] {rec['attacker']} vs {rec['defender']} {rec['evaluation_seed']} {bundle.episodes[0].summary.outcome} steps={bundle.episodes[0].summary.steps} -> {rec['recording_name']} ({rec['replay_id']})", flush=True)


def research_reference(research_root: Path, rec: dict) -> dict:
    """Independent research evaluation of the same scenario (fresh PPO load of the manifest model; research Stage3V2Env; exact float64 trace)."""
    import hashlib

    import experiments.journal_v2_a1_cur1 as C             # research modules (the visualizer engine has already put the research root first on sys.path)
    import experiments.journal_v2_a1_cur3v2 as X
    import experiments.journal_v2_a1_cur3v2_final as F

    assert Path(X.__file__).resolve().is_relative_to(research_root.resolve()), "research modules must come from the research worktree"
    env = X.Stage3V2Env(0, 0, 0, scripted_a0=rec["model_seed"] is None, controller=rec["defender"])
    if rec["model_seed"] is None:
        ep, tr = F.traced_episode(env, rec["evaluation_seed"], lambda _o: np.zeros(3, dtype=np.float32))
        return {"episode": ep, "trace": tr, "model_sha256": None}
    from stable_baselines3 import PPO

    path = research_root / "results/journal_v2_corrected_semantics/a1_curriculum/stage3_v2" / f"seed_{rec['model_seed']}" / "checkpoints" / "accepted_model.zip"
    model = PPO.load(str(path).removesuffix(".zip"), device="cpu")
    ep, tr = F.traced_episode(env, rec["evaluation_seed"], C.model_actor(model))
    return {"episode": ep, "trace": tr, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_and_describe(research_root: Path, records: list[dict], research_data: dict) -> None:
    for rec in records:
        episode = rec["_bundle"].episodes[0]
        ref = research_reference(research_root, rec)
        rep_pos = np.array([f.attacker.position_m for f in episode.frames], dtype=np.float64)
        rep_def = np.array([f.defender.position_m for f in episode.frames], dtype=np.float64)
        ref_pos, ref_def = ref["trace"]["pos"], ref["trace"]["def_pos"]
        same_len = len(rep_pos) == len(ref_pos) == len(ref_def)
        pos_diff = float(np.abs(rep_pos - ref_pos).max()) if same_len else float("inf")
        def_diff = float(np.abs(rep_def - ref_def).max()) if same_len else float("inf")
        chord = float(np.linalg.norm(rep_pos[-1] - rep_pos[0]))
        replay_pe = float(episode.summary.safety_metrics["attacker_path_length_m"]) / chord
        ref_pe = float(ref["episode"]["path_excess"])
        sha_replay = episode.attacker.checkpoint.sha256 if episode.attacker.checkpoint else None
        in_final = rec["defender"] == "greedy"
        compact = None
        manifest_sha = None
        if rec["model_seed"]:
            manifest_sha = research_data["manifests"][rec["model_seed"]]["accepted_model_sha256"]
            if in_final:
                compact = next((e for e in research_data["final"]["per_seed"][str(rec["model_seed"])]["episodes"] if e["scenario_seed"] == rec["evaluation_seed"]), None)
        elif in_final:
            compact = next((e for e in research_data["final"]["a0"]["episodes"] if e["scenario_seed"] == rec["evaluation_seed"]), None)
        replay_release = rec["_bundle"].episodes[0].attacker.parameters.get("defender_release_step")
        checks = {
            "outcome_equal": episode.summary.outcome == ref["episode"]["outcome"] and (compact is None or compact["outcome"] == ref["episode"]["outcome"]),
            "steps_equal": episode.summary.steps == ref["episode"]["steps"] and (compact is None or compact["steps"] == ref["episode"]["steps"]),
            "attacker_trajectory_within_tolerance": bool(same_len and pos_diff <= POSITION_TOLERANCE_M),
            "defender_trajectory_within_tolerance": bool(same_len and def_diff <= POSITION_TOLERANCE_M),
            "release_step_equal": replay_release == ref["episode"]["release_step"],
            "path_excess_within_tolerance": abs(replay_pe - ref_pe) <= PATH_EXCESS_TOLERANCE and (compact is None or abs(ref_pe - compact["path_excess"]) <= 1e-9),
            "checkpoint_sha_equal": (sha_replay == ref["model_sha256"] == manifest_sha) if rec["model_seed"] else sha_replay is None,
        }
        rec["verification"] = {"checks": checks, "passed": bool(all(checks.values())), "max_attacker_trajectory_difference_m": pos_diff, "max_defender_trajectory_difference_m": def_diff, "tolerance_m": POSITION_TOLERANCE_M,
                               "replay_path_excess": replay_pe, "research_path_excess": ref_pe, "replay_release_step": replay_release, "research_release_step": ref["episode"]["release_step"],
                               "replay_sha256": sha_replay, "research_sha256": ref["model_sha256"], "manifest_sha256": manifest_sha, "compared_frames": len(rep_pos)}
        table_row = next((e for e in episode_table(research_data, rec["model_seed"]) if e["scenario_seed"] == rec["evaluation_seed"]), None) if (rec["model_seed"] and in_final) else None
        dists = np.linalg.norm(rep_pos - rep_def, axis=1)
        rec["stats"] = {"outcome": episode.summary.outcome, "steps": episode.summary.steps, "duration_s": episode.summary.duration_s, "path_excess": replay_pe,
                        "closest_defender_separation_m": float(episode.summary.safety_metrics["minimum_defender_attacker_distance_m"]), "closest_step": int(np.argmin(dists)),
                        "defender_first_visible_step": next((i for i, d in enumerate(dists) if d <= 25.0), None), "release_step": replay_release,
                        "defender_initial_position_m": [float(x) for x in rep_def[0]], "initial_attacker_position_m": [float(x) for x in rep_pos[0]]}
        rec["what_to_look_for"] = notes_for(rec, table_row)
        print(f"[verify] {rec['attacker']} vs {rec['defender']} {rec['evaluation_seed']} passed={rec['verification']['passed']} att={pos_diff:.2e} def={def_diff:.2e}", flush=True)
        if not rec["verification"]["passed"]:
            raise SystemExit(f"replay equivalence FAILED for {rec['attacker']} {rec['evaluation_seed']}: {checks}")


def post(path: str, payload: dict) -> dict:
    req = urllib.request.Request(BACKEND + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read())


def create_cases(records: list[dict]) -> None:
    for rec in reversed(records):
        episode = rec["_bundle"].episodes[0]
        frames = episode.frames
        visible_step = rec["stats"]["defender_first_visible_step"]
        marks = []
        if visible_step is not None and 0 < visible_step < len(frames):
            marks.append(("Defender enters the 25 m information gate", visible_step, "Before this step A1 has no defender information; check that it is flying normally toward the asset."))
        if rec["stats"].get("release_step") is not None and rec["stats"]["release_step"] < len(frames):
            marks.append(("Defender detects the attacker and is RELEASED", rec["stats"]["release_step"], "The defender was stationary until now; from the next step the unmodified controller drives it at the attacker."))
        dists = [float(np.linalg.norm(np.array(f.attacker.position_m) - np.array(f.defender.position_m))) for f in frames]
        closest = int(np.argmin(dists))
        marks.append((f"Closest defender approach ({dists[closest]:.2f} m)", closest, "Is this a deliberate, smooth clearance? (interception radius 2.5 m)"))
        marks.append((f"Episode end: {episode.summary.outcome}", len(frames) - 1, "Did it turn back toward the asset and capture on the first reasonable approach (or, for a failure, why was it intercepted)?"))
        bookmarks = [{"title": t, "time_s": step * DT, "frame_step": step, "observation": "", "interpretation": hint, "tags": ["stage3v2-review"]} for t, step, hint in marks]
        body = {"case_type": "single", "replay_ids": [rec["replay_id"]], "title": f"[Stage3-v2 REVIEW] {rec['title']}"[:200],
                "description": f"{LABEL}. Scenario {rec['evaluation_seed']}, {rec['attacker']} vs {rec['defender']}, outcome {rec['stats']['outcome']}."[:2000],
                "tags": ["stage3v2-review", "non-canonical", "provisional-curriculum", rec["kind"].replace("_", "-")], "notes": rec["what_to_look_for"][:12000], "bookmarks": bookmarks}
        created = post("/api/cases", body)
        rec["case_id"] = created.get("case_id")
        print(f"[case] {rec['attacker']} {rec['evaluation_seed']} -> {rec['case_id']}", flush=True)


def write_outputs(records: list[dict], research_commit: str) -> None:
    rows = []
    for i, rec in enumerate(records, start=1):
        rows.append({k: v for k, v in rec.items() if not k.startswith("_")} | {"order": i, "url": f"{FRONTEND}/?replay={rec['replay_id']}",
                                                                              "live_url": f"{FRONTEND}/?semantics=V2&clutter=C0&defender={rec['defender']}&attacker={rec['attacker']}&seed={rec['evaluation_seed']}"})
    payload = {"label": LABEL, "scientific_status": "PROVISIONAL curriculum checkpoints (non-final, non-canonical); no scientific status changed", "research_commit": research_commit,
               "environment": "exact research Stage3-v2 scenario: V2, C0, stationary asset, scripted defender on A0's nominal path HELD until canonical detection then released (unchanged controller), 37-D observation, continuous 3-D action",
               "recordings": rows, "all_verified": all(r["verification"]["passed"] for r in rows), "tolerance_m": POSITION_TOLERANCE_M, "checklist": CHECKLIST}
    SHORTLIST_JSON.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    lines = [f"# Stage3-v2 manual-review shortlist — {LABEL}", "", f"Research commit `{research_commit}`. All {len(rows)} replays verified against an independent research run (outcome, steps, attacker AND defender trajectory ≤ {POSITION_TOLERANCE_M} m over every frame, release step, checkpoint SHA).", "",
             "| # | Recording | Defender | Scenario | Outcome | Steps | Closest sep. | Path-excess | Open |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["stats"]
        lines.append(f"| {r['order']} | {r['title']} | {r['defender']} | {r['evaluation_seed']} | {s['outcome']} | {s['steps']} | {s['closest_defender_separation_m']:.2f} m | {s['path_excess']:.3f} | `{r['url']}` |")
    lines += ["", "## What to look for (all recordings)", ""] + [f"{i + 1}. {q}" for i, q in enumerate(CHECKLIST)]
    SHORTLIST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", action="store_true", help="also create research cases (notes + bookmarks) via the running backend")
    args = parser.parse_args(argv)
    from visualizer.backend.config import load_settings
    from visualizer.backend.replay_engine import _research_modules

    research_root = Path(load_settings().research_data_root).resolve()
    _research_modules(research_root)                                  # puts the research root first on sys.path and imports the research modules
    data = load_research(research_root)
    commit = subprocess.run(["git", "-C", str(research_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    records = select_recordings(data, research_root)
    run_replays(records)
    verify_and_describe(research_root, records, data)
    if args.cases:
        create_cases(records)
    write_outputs(records, commit)
    print(json.dumps([(i + 1, r["attacker"], r["defender"], r["evaluation_seed"], r["kind"]) for i, r in enumerate(records)]))


if __name__ == "__main__":
    sys.exit(main())
