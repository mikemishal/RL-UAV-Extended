"""Evidence tables + mechanism analysis for the evasion-gating sweep (re-simulates the gained-collision, exhaustion and >5x episodes)."""

from __future__ import annotations

import collections
import json

import numpy as np
import pandas as pd

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_eg as E
from experiments import journal_v2_a1_eg_state as TS
from experiments import journal_v2_a1_eg_sweep as S


def _resim(env, row, policy: str):
    return S.run_episode(env, (row.level, row.defender, int(row.seed)), policy, int(row.seed))


def mechanisms(df: pd.DataFrame) -> dict:
    bank = A.A1V2DefenderBank()
    env = E.A1V6TrainingEnv(0, 0, defender_bank=bank, record_trajectory=True)
    out = {}
    for b in S.NONZERO:
        name = f"FIXED_{b:.2f}"
        c = df[df.policy == name]
        gained = c[c.collision & ~c.base_collision]
        exhausted = c[c.exhausted]
        big = c[c.path_excess > 5]
        coll = collections.Counter()
        since_all = []
        for r in gained.itertuples():
            ep = _resim(env, r, name)
            blends, owned = ep["executed_blends"], ep["owned_flags"]
            last_evasion = max((i for i, v in enumerate(blends) if v > 0.0), default=None)
            since = None if last_evasion is None else len(blends) - 1 - last_evasion
            since_all.append(since)
            when = "evasion never executed" if since is None else ("evasion executed on the collision step" if since == 0 else
                                                                    "evasion ended 1-3 steps earlier" if since <= 3 else "evasion ended >3 steps earlier")
            coll[f"{when}; frozen layer {'owned' if owned[-1] else 'not owning'} the collision step"] += 1
        known = [s for s in since_all if s is not None]
        ex = {"count": int(len(exhausted)), "mean_steps": float(exhausted.steps.mean()) if len(exhausted) else None,
              "with_overshoot_ge1": int((exhausted.overshoots >= 1).sum()), "mean_overshoots": float(exhausted.overshoots.mean()) if len(exhausted) else None,
              "by_defender": exhausted.defender.value_counts().to_dict(), "by_level": exhausted.level.value_counts().to_dict(),
              "base_outcome": exhausted.base_outcome.value_counts().to_dict(), "mean_evasion_steps": float(exhausted.evasion_steps.mean()) if len(exhausted) else None,
              "mean_max_deflection_deg": float(exhausted.max_deflection_deg.mean()) if len(exhausted) else None,
              "mean_base_steps": float(exhausted.base_steps.mean()) if len(exhausted) else None}
        out[name] = {"gained_collision_mechanism": dict(coll), "gained_collision_steps_since_last_evasion": {"median": float(np.median(known)) if known else None, "max": max(known) if known else None},
                     "exhausted": ex,
                     "path_excess_gt5": {"count": int(len(big)), "outcomes": big.outcome.value_counts().to_dict(), "defenders": big.defender.value_counts().to_dict(),
                                         "all_exhausted": bool(len(big) == 0 or big.exhausted.all())}}
    return out


def render_markdown(df: pd.DataFrame, result: dict, mech: dict) -> str:
    P = result["policies"]
    L = ["# Defender-evasion gating (action-v6) — generated evidence tables", "",
         f"Specification SHA-256 `{result['spec_sha256']}`; {result['n_records']:,} episodes = 20 cells (4 clutter levels x 5 defenders) x 100 matched seeds (145000-145099) x 9 policies. "
         "BASE (blend 0 through the env) is bit-identical to canonical A0 in all 2,000 matched scenarios; SWITCH_b is trajectory-identical to FIXED_b in all cells "
         f"(flags: {result['flags']}). Pooled over the five defenders per level (500 episodes per level).", "",
         "## 1. Gates per policy", "", "| policy | G1 collisions | G2 mission | G3 authority | G3 positive | G4 path | G6 obstacle authority | gained collisions |", "|---|---|---|---|---|---|---|---|"]
    f = lambda x: "PASS" if x else "FAIL"
    for k, p in P.items():
        L.append(f"| {k} | {f(p['G1'])} | {f(p['G2'])} | {f(p['G3'])} | {f(p['G3_positive'])} | {f(p['G4'])} | {f(p['G6'])} | {p['gained_collisions_total']} |")
    L += ["", f"G5 (BASE == canonical A0): {f(result['G5'])}.", "", "## 2. Obstacle collisions by clutter level (BASE has 0 in every cell)", "",
          "| blend | C0 | C1 | C2 | C3 | total gained / 2,000 |", "|---|---|---|---|---|---|"]
    for b in S.NONZERO:
        p = P[f"FIXED_{b:.2f}"]
        L.append(f"| {b:.2f} | " + " | ".join(str(p["per_level"][lv]["collisions"]) for lv in S.LEVELS) + f" | {p['gained_collisions_total']} |")
    L += ["", "## 3. Collisions by defender family (gained, pooled over levels)", "", "| blend | " + " | ".join(S.FAMILIES) + " |", "|---|" + "---|" * len(S.FAMILIES)]
    for b in S.NONZERO:
        c = df[df.policy == f"FIXED_{b:.2f}"]
        L.append(f"| {b:.2f} | " + " | ".join(str(int((c[c.defender == fam].collision & ~c[c.defender == fam].base_collision).sum())) for fam in S.FAMILIES) + " |")
    L += ["", "## 4. Mission gate G2: endurance exhaustion / timeout / orbit / fly-by (BASE: 0 / 0 / 0 / 0.0%)", "",
          "| blend | exhausted C0 / C1 / C2 / C3 | exhausted total | timeouts | orbit >=1 rev | episodes with >=1 overshoot (fly-by) |", "|---|---|---|---|---|---|"]
    for b in S.NONZERO:
        p = P[f"FIXED_{b:.2f}"]
        c = df[df.policy == f"FIXED_{b:.2f}"]
        L.append(f"| {b:.2f} | " + " / ".join(str(p["per_level"][lv]["exhausted"]) for lv in S.LEVELS) + f" | {p['exhausted']} | {p['timeouts']} | "
                 f"{sum(p['per_level'][lv]['orbit'] for lv in S.LEVELS)} | {float((c.overshoots >= 1).mean()) * 100:.1f}% |")
    L += ["", "## 5. Tactical authority G3 (vs matched BASE)", "", "| blend | episodes diverging >= 1 m | median / max divergence (m) | outcome changed | BASE interception -> capture | capture -> interception | "
          "mean min-separation diff (m) | mean extra steps | mean extra path (m) | max deflection (deg) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b in S.NONZERO:
        a = P[f"FIXED_{b:.2f}"]["authority"]
        L.append(f"| {b:.2f} | {a['fraction_diverging_ge_1m'] * 100:.1f}% | {a['median_div_m']:.2f} / {a['max_div_m']:.1f} | {a['outcome_changed_episodes']} | {a['conversions_int_to_capture']} | "
                 f"{a['reversals_capture_to_int']} | {a['mean_min_sep_diff_m']:+.3f} | {a['mean_steps_diff']:+.2f} | {a['mean_path_diff_m']:+.1f} | {a['max_deflection_deg']:.1f} |")
    L += ["", "## 6. Path efficiency G4 (BASE: max 3.8x, 0 episodes > 5x)", "", "| blend | mean | median | max | >1.3x | >2x | >5x | exhausted | timeout |", "|---|---|---|---|---|---|---|---|---|"]
    for b in S.NONZERO:
        p = P[f"FIXED_{b:.2f}"]
        pe = p["path_excess"]
        L.append(f"| {b:.2f} | {pe['mean']:.3f} | {pe['median']:.3f} | {pe['max']:.2f} | {pe['gt1.3']} | {pe['gt2']} | {pe['gt5']} | {p['exhausted']} | {p['timeouts']} |")
    L += ["", "## 7. Mechanisms (re-simulated)", "", "| blend | gained-collision mechanism | exhaustions: with fly-by / total (mean overshoots; mean extra steps vs BASE) | >5x cases (all endurance exhaustion?) |", "|---|---|---|---|"]
    for b in S.NONZERO:
        m = mech[f"FIXED_{b:.2f}"]
        ex = m["exhausted"]
        g = "; ".join(f"{k}: {v}" for k, v in m["gained_collision_mechanism"].items()) or "none"
        L.append(f"| {b:.2f} | {g} | {ex['with_overshoot_ge1']} / {ex['count']} ({(ex['mean_overshoots'] or 0):.1f}) | {m['path_excess_gt5']['count']} ({m['path_excess_gt5']['all_exhausted']}) |")
    L += ["", "## 8. Net capture effect (BASE: 232 captures / 2,000; matched episodes)", "", "| blend | captures (C0 / C1 / C2 / C3) | total | change vs BASE | captures lost to endurance exhaustion (BASE captured -> exhausted) |", "|---|---|---|---|---|"]
    base_c = df[df.policy == "BASE"]
    for b in S.NONZERO:
        c = df[df.policy == f"FIXED_{b:.2f}"]
        lost = int(((c.base_outcome == "soldier_caught") & c.exhausted).sum())
        L.append(f"| {b:.2f} | " + " / ".join(str(int(c[c.level == lv].captured.sum())) for lv in S.LEVELS) + f" | {int(c.captured.sum())} | {int(c.captured.sum()) - int(base_c.captured.sum()):+d} | {lost} |")
    L += ["", "## 9. Suppression bookkeeping (steps, pooled)", "", "| blend | no defender information | obstacle authority | primitive inactive (alpha=0) | executed while planner owned steering |", "|---|---|---|---|---|"]
    for b in S.NONZERO:
        p = P[f"FIXED_{b:.2f}"]
        s = p["suppression_totals"]
        L.append(f"| {b:.2f} | {s.get('no_defender_info', 0)} | {s.get('obstacle_authority', 0)} | {s.get('evasion_inactive', 0)} | {p['executed_while_owned']} |")
    return "\n".join(L) + "\n"


def build() -> dict:
    records, flags = S.load_cells()
    df = pd.DataFrame(records)
    result = S.analyze(records, flags)
    mech = mechanisms(df)
    md = render_markdown(df, result, mech)
    TS.atomic_write_json(S.SWEEP_DIR / "analysis.json", result)
    TS.atomic_write_json(TS.STATE_DIR / "sweep_summary.json", {"spec_sha256": result["spec_sha256"], "flags": flags, "selection": result["selection"], "G5": result["G5"],
                                                              "policies": {k: v for k, v in result["policies"].items()}, "mechanisms": mech})
    TS.atomic_write_bytes(TS.STATE_DIR / "sweep_results.md", md.encode("utf-8"))
    return {"result": result, "mech": mech, "md": md, "df": df}


if __name__ == "__main__":
    print(build()["md"])
