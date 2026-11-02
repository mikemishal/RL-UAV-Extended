"""Build and verify the Stage2-v2 HUMAN-REVIEW recordings (visualizer support only; the research repository is read, never written).

    python -m visualizer.review.build_stage2v2_review            # select -> replay -> verify -> save recordings + shortlist (+ JSON/MD)
    python -m visualizer.review.build_stage2v2_review --cases    # additionally create research cases (notes + bookmarks) through the running backend

Selection uses only the committed research results of Stage2-v2 (`final_evaluation.json`, `evasion_analysis.json`, seed manifests). Every recording is replayed through the visualizer engine and then
compared with an independent research evaluation of the same checkpoint on the same scenario (outcome, step count, attacker trajectory, defender position, checkpoint SHA-256).
No scientific status is changed: the policies remain NOT MASTERED / NON-CANONICAL.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

REVIEW_DIR = Path(__file__).resolve().parent
SHORTLIST_JSON = REVIEW_DIR / "stage2v2_review_shortlist.json"
SHORTLIST_MD = REVIEW_DIR / "stage2v2_review_shortlist.md"
DOCS = "docs/journal_v2_a1_curriculum/stage2_v2"
LABEL = "Stage2-v2 REVIEW — NOT MASTERED / NON-CANONICAL"
POSITION_TOLERANCE_M = 1e-3          # replay frames are float32 simulator state; the research trace is the same simulator state in float64
PATH_EXCESS_TOLERANCE = 1e-3
BACKEND = "http://127.0.0.1:8000"
FRONTEND = "http://127.0.0.1:5173"
DT = 0.5

CHECKLIST = [
    "Before the defender becomes visible (inside the 25 m information gate), does A1 fly toward the asset normally (no early swerving)?",
    "Does it deviate only AFTER defender information becomes available?",
    "Is the evasion smooth rather than random or wandering?",
    "Does it create enough separation to avoid the defender (interception radius 2.5 m)?",
    "After clearing the defender, does it turn back toward the asset?",
    "Does it capture on the first reasonable approach?",
    "Any unnecessary loops, oscillations, fly-bys or excessive detours?",
]


def load_research(root: Path) -> dict:
    final = json.loads((root / DOCS / "final_evaluation.json").read_text())
    evasion = json.loads((root / DOCS / "evasion_analysis.json").read_text())
    manifests = {s: json.loads((root / DOCS / f"seed_{s}_manifest.json").read_text()) for s in (201, 202, 203)}
    return {"final": final, "evasion": evasion, "manifests": manifests}


def episode_table(data: dict, seed: int) -> list[dict]:
    """One row per scenario: compact episode + (for captures) the evasion-analysis row + (always) the with/without-defender displacement."""
    eps = {e["scenario_seed"]: dict(e) for e in data["final"]["per_seed"][str(seed)]["episodes"]}
    for r in data["evasion"][str(seed)]["rows"]:
        eps[r["scenario_seed"]]["evasion"] = r
    for r in data["evasion"][str(seed)]["reactive_vs_no_defender"]["rows"]:
        eps[r["scenario_seed"]]["reactive"] = r
    return [eps[k] for k in sorted(eps)]


def select_recordings(data: dict) -> list[dict]:
    chosen: dict[int, set[int]] = {201: set(), 202: set(), 203: set()}
    out: list[dict] = []

    def add(seed, scenario, kind, title, extra=None):
        chosen[seed].add(scenario)
        out.append({"attacker": f"A1-CUR2V2-{seed}-REVIEW", "model_seed": seed, "evaluation_seed": scenario, "kind": kind, "title": title, **(extra or {})})

    for seed in (201, 202, 203):
        table = episode_table(data, seed)
        wins = [e for e in table if e["outcome"] == "soldier_caught"]
        med_pe = float(np.median([e["path_excess"] for e in wins]))
        med_md = float(np.median([e["evasion"]["min_defender_distance_m"] for e in wins]))
        if seed in (201, 203):          # high-quality success: clear pre-gate pursuit, reactive deviation, resume, efficient, comfortable separation
            good = [e for e in wins if e["evasion"]["pursuit_evasion_pursuit"] and e["evasion"]["resumes_pursuit"] and (e.get("defender_first_visible_step") or 0) >= 2
                    and e["evasion"]["first_divergence_while_visible"] and e["path_excess"] <= med_pe and e["evasion"]["min_defender_distance_m"] >= med_md]
            pool = good or wins
            best = max(pool, key=lambda e: e["evasion"]["post_encounter_mean_cos_to_asset"] - 2.0 * (e["path_excess"] - 1.0) - 1e-6 * e["scenario_seed"])
            add(seed, best["scenario_seed"], "representative_high_quality_success", f"Seed {seed} — representative HIGH-QUALITY successful evasion", {"high_quality": True})
        else:
            best = min(wins, key=lambda e: abs(e["path_excess"] - med_pe) / 0.05 + abs(e["evasion"]["min_defender_distance_m"] - med_md) / 1.0 + 1e-6 * e["scenario_seed"])
            add(seed, best["scenario_seed"], "representative_success", f"Seed {seed} — representative successful capture")
        rest = [e for e in wins if e["scenario_seed"] not in chosen[seed]]
        closest = min(rest, key=lambda e: (e["evasion"]["min_defender_distance_m"], e["scenario_seed"]))
        add(seed, closest["scenario_seed"], "closest_successful_pass", f"Seed {seed} — CLOSEST successful defender pass")
        rest = [e for e in wins if e["scenario_seed"] not in chosen[seed]]
        widest = max(rest, key=lambda e: (e["evasion"]["max_lateral_dev_m"] ** 2 + e["evasion"]["max_vertical_dev_m"] ** 2, -e["scenario_seed"]))
        add(seed, widest["scenario_seed"], "largest_successful_detour", f"Seed {seed} — LARGEST successful detour")
        fails = [e for e in table if e["outcome"] != "soldier_caught"]
        if fails:                       # the most evasive failed attempt: the policy reacted (largest displacement) but still hit the defender
            worst = max(fails, key=lambda e: (e["reactive"]["max_displacement_m"], -e["scenario_seed"]))
            add(seed, worst["scenario_seed"], "interception_failure", f"Seed {seed} — INTERCEPTION failure (largest attempted evasion)")
    first = next(o for o in out if o["model_seed"] == 201 and o["kind"] == "representative_high_quality_success")
    out.append({"attacker": "A0-CUR2V2-REFERENCE", "model_seed": None, "evaluation_seed": first["evaluation_seed"], "kind": "matched_a0_reference",
                "title": f"Matched canonical A0 reference (same scenario as the seed-201 high-quality success, {first['evaluation_seed']})"})
    priority = {("A1-CUR2V2-201-REVIEW", "representative_high_quality_success"): 1, ("A1-CUR2V2-203-REVIEW", "representative_high_quality_success"): 2,
                ("A1-CUR2V2-203-REVIEW", "interception_failure"): 3, ("A0-CUR2V2-REFERENCE", "matched_a0_reference"): 4}
    for o in out:
        o["priority"] = priority.get((o["attacker"], o["kind"]))
    return sorted(out, key=lambda o: (o["priority"] is None, o["priority"] or 0, o["attacker"], o["kind"]))


def notes_for(rec: dict, table_row: dict | None) -> str:
    """Per-recording `what_to_look_for`: the seven fixed questions plus recording-specific facts taken from the research analysis."""
    facts = []
    gate = rec["stats"].get("defender_first_visible_step")
    if gate is not None:
        facts.append(f"The defender is inside the 25 m information gate from step {gate} (t = {gate * DT:.1f} s)" + (" — i.e. already at the start" if gate == 0 else "; before that A1 sees no defender features") + ".")
    ev = (table_row or {}).get("evasion")
    if rec["kind"] == "matched_a0_reference":
        facts.append("Canonical A0 flies straight at the asset, ignores the defender and is INTERCEPTED (A0 captures 0/200 in this scenario family). Compare its path with the A1 recording of the same scenario.")
    elif ev:
        facts.append(f"Closest defender approach at step {ev['closest_step']} (t = {ev['closest_step'] * DT:.1f} s), separation {ev['min_defender_distance_m']:.2f} m (A0: exactly 2.5 m = contact); "
                     f"passes {ev['side']} of the defender; max lateral deviation from the initial chord {ev['max_lateral_dev_m']:.1f} m, vertical {ev['max_vertical_dev_m']:.1f} m.")
        if ev.get("first_divergence_step") is not None:
            facts.append(f"First >10° departure from the Stage-1 BC direction at step {ev['first_divergence_step']} (defender distance {ev['first_divergence_defender_distance_m']:.1f} m).")
        facts.append(f"After the closest approach the mean cosine toward the asset is {ev['post_encounter_mean_cos_to_asset']:.3f}; path-excess {ev['path_excess']:.3f}.")
    elif rec["kind"] == "interception_failure":
        r = (table_row or {}).get("reactive", {})
        facts.append(f"Interception failure: the policy reacted (max displacement {r.get('max_displacement_m', float('nan')):.1f} m from its no-defender path) but still entered the 2.5 m interception radius. "
                     "Look at whether the dodge was too late, too small, or in a direction that stays inside the radius.")
    kind_note = {
        "closest_successful_pass": "Edge case: this is the narrowest successful pass; check that the margin to the 2.5 m interception sphere is real and the maneuver is deliberate.",
        "largest_successful_detour": "Edge case: this is the widest successful detour; judge whether it is purposeful (clearing the defender, then returning) or an excessive/unnecessary arc.",
        "representative_success": "Typical case: nothing special selected; judge overall smoothness and efficiency.",
        "representative_high_quality_success": "Showcase case (clear pre-gate pursuit, reactive deviation, return to pursuit, efficient path, comfortable separation). Use it as the reference for what the policy does when it works.",
        "interception_failure": "Failure case: use it to understand how the remaining ≈ 10% of scenarios fail.",
        "matched_a0_reference": "Reference: the scripted attacker that the learned policies are being compared against.",
    }[rec["kind"]]
    questions = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(CHECKLIST))
    return f"{LABEL}\n{kind_note}\n\nFacts (research analysis):\n- " + "\n- ".join(facts or ["(no additional analysis row)"]) + f"\n\nWhat to look for:\n{questions}"


def run_replays(records: list[dict]) -> None:
    from visualizer.backend.config import load_settings
    from visualizer.backend.replay_engine import ReplayRequest, run_replay_episode, save_replay_bundle
    from visualizer.backend.replay_jobs import _replay_id_for_path
    from visualizer.backend.replay_requests import ALL_ATTACKER_SEEDS

    settings = load_settings()
    # Save in REVERSE priority order so that the first recording to inspect is the newest file (the replay list is sorted newest-first).
    for rec in reversed(records):
        request = ReplayRequest(defender_policy_id="passive", attacker_condition=rec["attacker"], attacker_model_seed=ALL_ATTACKER_SEEDS.get(rec["attacker"]), clutter_level="C0",
                                perception_mode="map_local", semantics="V2", evaluation_seed=rec["evaluation_seed"], deterministic_inference=True)
        bundle = run_replay_episode(request, settings)
        path = save_replay_bundle(bundle, None)
        rec["recording_name"] = path.name.removesuffix(".json.gz")
        rec["replay_id"] = _replay_id_for_path(path)                      # the opaque id the API / UI use (sha-256 of the file name)
        rec["replay_file"] = str(path.relative_to(Path(__file__).resolve().parents[2])).replace("\\", "/")
        rec["_bundle"] = bundle
        print(f"[replay] {rec['attacker']} {rec['evaluation_seed']} {bundle.episodes[0].summary.outcome} steps={bundle.episodes[0].summary.steps} -> {rec['recording_name']} ({rec['replay_id']})", flush=True)


def research_reference(research_root: Path, rec: dict) -> dict:
    """Independent research evaluation of the same scenario (fresh PPO load of the manifest model; research Stage2Env; exact float64 trace)."""
    import hashlib

    import experiments.journal_v2_a1_cur1 as C             # research modules (the visualizer engine has already put the research root first on sys.path)
    import experiments.journal_v2_a1_cur2 as S
    import experiments.journal_v2_a1_cur2_final as F

    assert Path(S.__file__).resolve().is_relative_to(research_root.resolve()), "research modules must come from the research worktree"
    env = S.Stage2Env(0, 0, 0, scripted_a0=rec["model_seed"] is None)
    if rec["model_seed"] is None:
        ep, tr = F.traced_episode(env, rec["evaluation_seed"], lambda _o: np.zeros(3, dtype=np.float32))
        return {"episode": ep, "trace": tr, "model_sha256": None}
    from stable_baselines3 import PPO

    path = research_root / "results/journal_v2_corrected_semantics/a1_curriculum/stage2_v2" / f"seed_{rec['model_seed']}" / "checkpoints" / "accepted_model.zip"
    model = PPO.load(str(path).removesuffix(".zip"), device="cpu")
    ep, tr = F.traced_episode(env, rec["evaluation_seed"], C.model_actor(model))
    return {"episode": ep, "trace": tr, "model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def verify_and_describe(research_root: Path, records: list[dict], research_data: dict) -> None:
    for rec in records:
        episode = rec["_bundle"].episodes[0]
        ref = research_reference(research_root, rec)
        rep_pos = np.array([f.attacker.position_m for f in episode.frames], dtype=np.float64)
        ref_pos = ref["trace"]["pos"]
        same_len = len(rep_pos) == len(ref_pos)
        pos_diff = float(np.abs(rep_pos - ref_pos).max()) if same_len else float("inf")
        defender_diff = float(np.abs(np.array(episode.frames[0].defender.position_m) - ref["trace"]["defender"]).max())
        chord = float(np.linalg.norm(rep_pos[-1] - rep_pos[0]))
        replay_pe = float(episode.summary.safety_metrics["attacker_path_length_m"]) / chord
        ref_pe = float(ref["episode"]["path_excess"])
        sha_replay = episode.attacker.checkpoint.sha256 if episode.attacker.checkpoint else None
        if rec["model_seed"]:
            manifest_sha = research_data["manifests"][rec["model_seed"]]["accepted_model_sha256"]
            compact = next(e for e in research_data["final"]["per_seed"][str(rec["model_seed"])]["episodes"] if e["scenario_seed"] == rec["evaluation_seed"])
        else:
            manifest_sha = None
            compact = next(e for e in research_data["final"]["a0"]["episodes"] if e["scenario_seed"] == rec["evaluation_seed"])
        checks = {
            "outcome_equal": episode.summary.outcome == ref["episode"]["outcome"] == compact["outcome"],
            "steps_equal": episode.summary.steps == ref["episode"]["steps"] == compact["steps"],
            "trajectory_within_tolerance": bool(same_len and pos_diff <= POSITION_TOLERANCE_M),
            "defender_position_equal": defender_diff <= 1e-4,
            "path_excess_within_tolerance": abs(replay_pe - ref_pe) <= PATH_EXCESS_TOLERANCE and abs(ref_pe - compact["path_excess"]) <= 1e-9,
            "checkpoint_sha_equal": (sha_replay == ref["model_sha256"] == manifest_sha) if rec["model_seed"] else sha_replay is None,
        }
        rec["verification"] = {"checks": checks, "passed": bool(all(checks.values())), "max_trajectory_difference_m": pos_diff, "tolerance_m": POSITION_TOLERANCE_M, "defender_position_difference_m": defender_diff,
                               "replay_path_excess": replay_pe, "research_path_excess": ref_pe, "replay_sha256": sha_replay, "research_sha256": ref["model_sha256"], "manifest_sha256": manifest_sha}
        table_row = next((e for e in episode_table(research_data, rec["model_seed"]) if e["scenario_seed"] == rec["evaluation_seed"]), None) if rec["model_seed"] else None
        rec["stats"] = {"outcome": episode.summary.outcome, "steps": episode.summary.steps, "duration_s": episode.summary.duration_s, "path_excess": replay_pe,
                        "closest_defender_separation_m": float(episode.summary.safety_metrics["minimum_defender_attacker_distance_m"]), "defender_first_visible_step": compact.get("defender_first_visible_step"),
                        "defender_position_m": [float(x) for x in episode.frames[0].defender.position_m], "initial_attacker_position_m": [float(x) for x in rep_pos[0]]}
        rec["what_to_look_for"] = notes_for(rec, table_row)
        print(f"[verify] {rec['attacker']} {rec['evaluation_seed']} passed={rec['verification']['passed']} maxdiff={pos_diff:.2e}", flush=True)
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
        dists = [float(np.linalg.norm(np.array(f.attacker.position_m) - np.array(f.defender.position_m))) for f in frames]
        closest = int(np.argmin(dists))
        marks.append((f"Closest defender approach ({dists[closest]:.2f} m)", closest, "Is this a deliberate, smooth clearance? (interception radius 2.5 m)"))
        marks.append((f"Episode end: {episode.summary.outcome}", len(frames) - 1, "Did it turn back toward the asset and capture on the first reasonable approach (or, for a failure, why was it intercepted)?"))
        bookmarks = [{"title": t, "time_s": step * DT, "frame_step": step, "observation": "", "interpretation": hint, "tags": ["stage2v2-review"]} for t, step, hint in marks]
        body = {"case_type": "single", "replay_ids": [rec["replay_id"]], "title": f"[Stage2-v2 REVIEW] {rec['title']}"[:200],
                "description": f"{LABEL}. Scenario {rec['evaluation_seed']}, {rec['attacker']}, outcome {rec['stats']['outcome']}."[:2000],
                "tags": ["stage2v2-review", "non-canonical", "not-mastered", rec["kind"].replace("_", "-")], "notes": rec["what_to_look_for"][:12000], "bookmarks": bookmarks}
        created = post("/api/cases", body)
        rec["case_id"] = created.get("case_id")
        print(f"[case] {rec['attacker']} {rec['evaluation_seed']} -> {rec['case_id']}", flush=True)


def write_outputs(records: list[dict], research_commit: str) -> None:
    rows = []
    for i, rec in enumerate(records, start=1):
        rows.append({k: v for k, v in rec.items() if not k.startswith("_")} | {"order": i, "url": f"{FRONTEND}/?replay={rec['replay_id']}",
                                                                              "live_url": f"{FRONTEND}/?semantics=V2&clutter=C0&defender=passive&attacker={rec['attacker']}&seed={rec['evaluation_seed']}"})
    payload = {"label": LABEL, "scientific_status": "NOT MASTERED / NON-CANONICAL; no scientific status changed", "research_commit": research_commit,
               "environment": "exact research Stage-2 scenario: V2, C0, stationary asset, stationary ACTIVE defender placed on A0's nominal path, 37-D observation, continuous 3-D action",
               "recordings": rows, "all_verified": all(r["verification"]["passed"] for r in rows), "tolerance_m": POSITION_TOLERANCE_M, "checklist": CHECKLIST}
    SHORTLIST_JSON.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    lines = [f"# Stage2-v2 manual-review shortlist — {LABEL}", "", f"Research commit `{research_commit}`. All {len(rows)} replays verified against an independent research run (outcome, steps, trajectory ≤ {POSITION_TOLERANCE_M} m, defender position, checkpoint SHA).", "",
             "| # | Recording | Seed | Scenario | Outcome | Steps | Closest sep. | Path-excess | Open |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s = r["stats"]
        lines.append(f"| {r['order']} | {r['title']} | {r['model_seed'] or 'A0'} | {r['evaluation_seed']} | {s['outcome']} | {s['steps']} | {s['closest_defender_separation_m']:.2f} m | {s['path_excess']:.3f} | `{r['url']}` |")
    lines += ["", "## What to look for (all recordings)", ""] + [f"{i + 1}. {q}" for i, q in enumerate(CHECKLIST)]
    SHORTLIST_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> None:
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", action="store_true", help="also create research cases (notes + bookmarks) via the running backend")
    args = parser.parse_args(argv)
    from visualizer.backend.config import load_settings

    research_root = Path(load_settings().research_data_root).resolve()
    data = load_research(research_root)
    commit = subprocess.run(["git", "-C", str(research_root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    records = select_recordings(data)
    run_replays(records)
    verify_and_describe(research_root, records, data)
    if args.cases:
        create_cases(records)
    write_outputs(records, commit)
    print(json.dumps([(i + 1, r["attacker"], r["evaluation_seed"], r["kind"]) for i, r in enumerate(records)]))


if __name__ == "__main__":
    sys.exit(main())
