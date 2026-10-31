"""Report builder for the finite tactical-pulse geometry sweep: per-candidate gate table, collision table, mechanism of gained collisions, authority and path-efficiency."""

from __future__ import annotations

import collections
import json

import numpy as np
import pandas as pd

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_tg as G
from experiments import journal_v2_a1_tg_state as TS
from experiments import journal_v2_a1_tg_sweep as S

OUT_DIR = TS.STATE_DIR


def collision_table(df: pd.DataFrame) -> pd.DataFrame:
    tact = df[df.policy != "BASE"].copy()
    tact["gained"] = tact.collision & ~tact.base_collision
    tact["lost"] = ~tact.collision & tact.base_collision
    rows = []
    for (theta, pulse, level, kind), g in tact.assign(kind=np.where(tact.context == "passive_forced", "passive_forced", "active")).groupby(
            ["theta", "pulse_s", "level", "kind"]):
        rows.append({"theta": theta, "pulse_s": pulse, "level": level, "kind": kind, "episodes": len(g), "tactical_collisions": int(g.collision.sum()),
                     "matched_base_collisions": int(g.base_collision.sum()), "gained": int(g.gained.sum()), "lost": int(g.lost.sum()),
                     "cells_with_gain": int(g[g.gained].groupby(["context", "policy"]).ngroups)})
    return pd.DataFrame(rows)


def mechanism(theta: float, pulse_s: float, records: pd.DataFrame, limit_per_candidate: int = 400) -> dict:
    """Re-run the gained-collision episodes and classify the executed mode on the collision step and on the step before it."""
    bad = records[(records.theta == theta) & (records.pulse_s == pulse_s) & records.collision & ~records.base_collision].head(limit_per_candidate)
    bank = A.A1V2DefenderBank()
    counts = collections.Counter()
    envs = {}
    for r in bad.itertuples():
        key = r.context
        if key not in envs:
            envs[key] = G.A1V5TrainingEnv(0, 0, defender_bank=bank, theta_deg=theta, pulse_s=pulse_s, gate_override=(key == "passive_forced"),
                                          record_trajectory=True)
        family = G.PASSIVE_DIAGNOSTIC if key == "passive_forced" else key
        ep = S.run_episode(envs[key], (r.level, family, int(r.seed)), r.policy, int(r.seed))
        modes = ep["modes"]
        last = modes[-1]
        cause = ("collided during a tactical pulse step" if last in G.TACTICAL_MODES else
                 "collided while the frozen common layer owned steering (post-maneuver state)" if last == "OBSTACLE_OVERRIDE" else f"collided in {last}")
        counts[cause] += 1
    return {"theta": theta, "pulse_s": pulse_s, "classified": int(sum(counts.values())), "counts": dict(counts)}


def render_markdown(df: pd.DataFrame, result: dict, coll: pd.DataFrame, mech: list[dict]) -> str:
    lines = ["# Finite tactical-pulse geometry sweep — generated evidence tables", "",
             f"Specification SHA-256 `{result['spec_sha256']}`; {result['n_records']:,} episode records (16 candidates x 4 levels x 4 contexts x 8 policies x 100 seeds).",
             "BASE here is the forced-BASE policy through the same state machine (bit-identical to canonical A0, see `base_equivalence.json`). "
             "'Gained' = tactical collision where the matched BASE episode (same seed/context) did not collide.", "",
             "## 1. Gates per candidate", "", "| theta | T (s) | steps | G1 collisions | G2 mission | G3 authority | G4 path | G5 machine | pass |", "|---|---|---|---|---|---|---|---|---|"]
    for c in result["candidates"]:
        flag = lambda b: "PASS" if b else "FAIL"
        lines.append(f"| {c['theta']:.0f} | {c['pulse_s']} | {c['pulse_steps']} | {flag(c['G1'])} | {flag(c['G2'])} | {flag(c['G3'])} | {flag(c['G4'])} | {flag(c['G5'])} | "
                     f"{'PASS' if c['passed'] else 'FAIL'} |")
    lines += ["", "## 2. Gained obstacle collisions (matched BASE collisions are 0 in every cell)", "",
              "Active = real fixed defenders (greedy/lead/pn; 7 tactical policies x 3 defenders x 100 seeds = 2,100 episodes per level). "
              "Passive-forced = mission-only stress diagnostic (parked defender, gate forced open).", "",
              "| theta | T (s) | active C0 | active C1 | active C2 | active C3 | passive C0 | passive C1 | passive C2 | passive C3 | total |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in result["candidates"]:
        cc = coll[(coll.theta == c["theta"]) & (coll.pulse_s == c["pulse_s"])]
        g = lambda kind, lv: int(cc[(cc.kind == kind) & (cc.level == lv)].gained.sum())
        lines.append(f"| {c['theta']:.0f} | {c['pulse_s']} | " + " | ".join(str(g("active", lv)) for lv in S.LEVELS) + " | "
                     + " | ".join(str(g("passive_forced", lv)) for lv in S.LEVELS) + f" | {c['gained_collisions_total']} |")
    lines += ["", "## 3. Mechanism of the gained collisions (re-simulated; <= 400 episodes per candidate)", "",
              "| theta | T (s) | classified | on a tactical pulse step | frozen common layer owned steering (post-maneuver state) | other |", "|---|---|---|---|---|---|"]
    for m in mech:
        pulse = m["counts"].get("collided during a tactical pulse step", 0)
        owned = m["counts"].get("collided while the frozen common layer owned steering (post-maneuver state)", 0)
        lines.append(f"| {m['theta']:.0f} | {m['pulse_s']} | {m['classified']} | {pulse} | {owned} | {m['classified'] - pulse - owned} |")
    lines += ["", "## 4. Mission gate G2 — tactical vs matched BASE episode counts (summed over policies, contexts, levels)", "",
              "| theta | T (s) | exhausted (tact / base) | timeout (tact / base) | orbit>=1 rev (tact / base) | ceiling hover | sustained hover | cells failing G2 |",
              "|---|---|---|---|---|---|---|---|"]
    for c in result["candidates"]:
        g = df[(df.theta == c["theta"]) & (df.pulse_s == c["pulse_s"])]
        t, b = g[g.policy != "BASE"], g[g.policy == "BASE"]
        # BASE is replicated once per tactical policy so the matched comparison is per policy
        reps = len(S.TACTICAL_POLICIES)
        lines.append(f"| {c['theta']:.0f} | {c['pulse_s']} | {int(t.exhausted.sum())} / {int(b.exhausted.sum()) * reps} | {int(t.timeout.sum())} / {int(b.timeout.sum()) * reps} | "
                     f"{int((t.orbit >= 1).sum())} / {int((b.orbit >= 1).sum()) * reps} | {int(t.ceiling_hover.sum())} | {int(t.sustained_hover.sum())} | "
                     f"{len([x for x in json.loads(json.dumps(c['g2_failing_cells']))])}{'+' if len(c['g2_failing_cells']) >= 12 else ''} |")
    lines += ["", "## 5. Path efficiency G4 (tactical episodes)", "", "| theta | T (s) | mean | median | max | >1.3x | >2x | >5x | >5x fraction (BASE fraction) |", "|---|---|---|---|---|---|---|---|---|"]
    for c in result["candidates"]:
        p = c["path_excess"]
        lines.append(f"| {c['theta']:.0f} | {c['pulse_s']} | {p['mean']:.2f} | {p['median']:.2f} | {p['max']:.1f} | {p['gt1.3']} | {p['gt2']} | {p['gt5']} | "
                     f"{p['fraction_gt5']:.4f} ({p['base_fraction_gt5']:.4f}) |")
    lines += ["", "## 6. Tactical authority G3 (best active-defender policy; share of 400 matched episodes diverging >= 2 m from BASE)", "",
              "| theta | T (s) | best context / policy | diverging fraction | outcome differs | mean min-separation difference (m) | G3 |", "|---|---|---|---|---|---|---|"]
    for c in result["candidates"]:
        a = c["authority_best"][0]
        lines.append(f"| {c['theta']:.0f} | {c['pulse_s']} | {a['context']} / {a['policy']} | {a['fraction_diverging']:.2f} | {a['outcome_differs']} | "
                     f"{a['mean_min_sep_diff']:+.2f} | {'PASS' if c['G3'] else 'FAIL'} |")
    lines += ["", "## 7. State machine G5", "", "| theta | T (s) | pulses started | cancelled by obstacle | aborted (geometry) | machine violations | longest tactical run (steps) | deterministic replay |",
              "|---|---|---|---|---|---|---|---|"]
    for c in result["candidates"]:
        lines.append(f"| {c['theta']:.0f} | {c['pulse_s']} | {c['pulses_started']} | {c['pulses_cancelled_by_obstacle']} | {c['pulses_aborted']} | {c['machine_violations']} | "
                     f"{c['longest_tactical_run_max']} (<= {c['pulse_steps']}) | {c['deterministic']} |")
    return "\n".join(lines) + "\n"


def supplementary_markdown(df: pd.DataFrame, candidates: list[dict]) -> str:
    """Active-defender-only mission gate and a defender-relevant reading of authority (outcome flips that are NOT obstacle collisions / exhaustion)."""
    act = df[df.context.isin(S.ACTIVE_DEFENDERS) & (df.policy != "BASE")].copy()
    lines = ["", "## 8. Supplementary: active defenders only (real 25 m information gate; passive-forced diagnostic excluded)", "",
             "Mission counts are over 8,400 matched tactical episodes per candidate (7 policies x 3 defenders x 4 levels x 100 seeds). 'Defender-relevant flips' count only episodes whose "
             "outcome changed between two defender-attributable terminal outcomes (attacker capture vs defender interception/timeout); collisions and endurance exhaustion are NOT credited as authority.", "",
             "| theta | T (s) | gained collisions | exhausted (tact / base) | tactical pulses / episode | flips intercepted -> captured | flips captured -> intercepted | mean min-separation diff (m) | share with any pulse |",
             "|---|---|---|---|---|---|---|---|---|"]
    for c in candidates:
        g = act[(act.theta == c["theta"]) & (act.pulse_s == c["pulse_s"])]
        gained = int((g.collision & ~g.base_collision).sum())
        base_df = df[(df.theta == c["theta"]) & (df.pulse_s == c["pulse_s"]) & df.context.isin(S.ACTIVE_DEFENDERS) & (df.policy == "BASE")]
        exh_base = int(base_df.exhausted.sum()) * len(S.TACTICAL_POLICIES)
        to_cap = int(((g.base_outcome == "intercepted") & (g.outcome == "soldier_caught")).sum() + ((g.base_outcome == "unsafe_intercept") & (g.outcome == "soldier_caught")).sum())
        to_int = int(((g.base_outcome == "soldier_caught") & g.outcome.isin(["intercepted", "unsafe_intercept"])).sum())
        lines.append(f"| {c['theta']:.0f} | {c['pulse_s']} | {gained} | {int(g.exhausted.sum())} / {exh_base} | {g.pulses.mean():.2f} | {to_cap} | {to_int} | "
                     f"{g.min_sep_diff.mean():+.3f} | {(g.pulses > 0).mean():.2f} |")
    return "\n".join(lines) + "\n"


def build(include_mechanism: bool = True) -> dict:
    records = S.load_all()
    df = pd.DataFrame(records)
    result = S.analyze(records, S.load_determinism())
    coll = collision_table(df)
    mech = []
    if include_mechanism:
        for c in result["candidates"]:
            mech.append(mechanism(c["theta"], c["pulse_s"], df))
    TS.atomic_write_json(S.SWEEP_DIR / "analysis.json", result)
    TS.atomic_write_json(OUT_DIR / "sweep_summary.json", {"spec_sha256": result["spec_sha256"], "selected": result["selected"], "candidates": result["candidates"],
                                                         "collision_table": coll.to_dict("records"), "mechanism": mech, "n_records": result["n_records"]})
    return {"result": result, "collisions": coll, "mechanism": mech, "df": df}


if __name__ == "__main__":
    out = build()
    print(json.dumps({"selected": out["result"]["selected"]}, indent=1))
    for c in out["result"]["candidates"]:
        print({k: c[k] for k in ("theta", "pulse_s", "G1", "G2", "G3", "G4", "G5", "passed", "gained_collisions_total")})
