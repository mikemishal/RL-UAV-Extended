"""Evidence tables + mechanism analysis for the route-level sweep."""

from __future__ import annotations

import collections
import json

import numpy as np
import pandas as pd

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_route_env as RE
from experiments import journal_v2_a1_rt_state as TS
from experiments import journal_v2_a1_route_sweep as S

KEY = ("LEFT", "RIGHT", "ABOVE", "MAX_SEP")


def mechanisms(df: pd.DataFrame, limit: int = 40) -> dict:
    """Re-simulate gained collisions (per config) and classify them; classify every >5x path-excess case by outcome."""
    bank = A.A1V2DefenderBank()
    out = {}
    for (R, T), cfg in df.groupby(["radius_m", "replan_s"]):
        tact = cfg[cfg.policy != "DIRECT"]
        gained = tact[tact.collision & ~tact.base_collision].sort_values(["level", "defender", "seed", "policy"]).head(limit)
        env = RE.A1RouteTrainingEnv(0, 0, defender_bank=bank, radius_m=R, replan_s=T, record_trajectory=True)
        counts = collections.Counter()
        for r in gained.itertuples():
            ep = S.run_episode(env, (r.level, r.defender, int(r.seed)), r.policy, int(r.seed))
            modes, owned = ep["modes"], ep["owned_flags"]
            last_route = max((i for i, m in enumerate(modes) if m != "DIRECT"), default=None)
            since = None if last_route is None else len(modes) - 1 - last_route
            when = "no route ever active" if since is None else ("on a route step" if since == 0 else ("route ended 1-3 steps earlier" if since <= 3 else "route ended >3 steps earlier"))
            counts[f"{when}; frozen layer {'owned' if owned[-1] else 'not owning'} the collision step"] += 1
        big = tact[tact.path_excess > 5]
        out[f"R{R}_T{T}"] = {"gained_collision_mechanism_sample": dict(counts), "gained_total": int((tact.collision & ~tact.base_collision).sum()), "sampled": int(len(gained)),
                              "path_excess_gt5": {"count": int(len(big)), "by_outcome": big.outcome.value_counts().to_dict(), "by_policy": big.policy.value_counts().to_dict(),
                                                  "base_outcome": big.base_outcome.value_counts().to_dict()},
                              "exhausted_base_outcome": tact[tact.exhausted].base_outcome.value_counts().to_dict()}
    return out


def render_markdown(df: pd.DataFrame, result: dict, mech: dict) -> str:
    f = lambda x: "PASS" if x else "FAIL"
    L = ["# Route-level A1 (`a1_route_action_v1`) — generated evidence tables", "",
         f"Specification SHA-256 `{result['spec_sha256']}`; {result['n_records']:,} episodes = 6 configurations (R, T_replan) x verified cells x 100 matched seeds (145000-145099) x 8 policies. "
         f"**Coverage (staged sweep; RMPC and the remaining C1-C3 defenders were not run, see README):** {result.get('coverage')}. "
         f"DIRECT through the wrapper is the matched reference and is bit-identical to canonical A0 (flags: {result['flags']}); forbidden references in the "
         f"controller module: {result['forbidden_references'] or 'none'}. Per-level pooling: C0 = 400 episodes per policy and configuration; C1-C3 = 200.", "",
         "## 1. Gates per configuration", "", "| R (m) | T_replan (s) | G1 collisions | G2 mission | G3 authority | G3 positive | G4 path | G5 perception | G6 stability | pass | gained collisions (7 policies) |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in result["configs"]:
        L.append(f"| {c['radius_m']} | {c['replan_s']} | {f(c['G1'])} | {f(c['G2'])} | {f(c['G3'])} | {f(c['G3_positive'])} | {f(c['G4'])} | {f(c['G5'])} | {f(c['G6'])} | {f(c['passed'])} | {c['gained_collisions_total']} |")
    L += ["", "## 2. Gained obstacle collisions (DIRECT has 0 in every cell): per policy, summed over levels (C0 / C1 / C2 / C3 for the worst policy)", "",
          "| R | T | " + " | ".join(S.TACTICAL) + " | worst policy by level C0/C1/C2/C3 |", "|---|---|" + "---|" * (len(S.TACTICAL) + 1)]
    for c in result["configs"]:
        worst = max(c["policies"].items(), key=lambda kv: kv[1]["gained_collisions"])
        lv = " / ".join(str(worst[1]["per_level"][l]["gained"]) for l in S.LEVELS)
        L.append(f"| {c['radius_m']} | {c['replan_s']} | " + " | ".join(str(c["policies"][p]["gained_collisions"]) for p in S.TACTICAL) + f" | {worst[0]}: {lv} |")
    L += ["", "## 3. Tactical authority: conversions (DIRECT interception -> capture) / reversals (DIRECT capture -> any failure) / net captures (of 1,000 matched episodes per policy: C0 x 4 defenders + C1-C3 x 2 defenders; DIRECT captures 97 / 1,000 per configuration)", "",
          "| R | T | " + " | ".join(S.TACTICAL) + " |", "|---|---|" + "---|" * len(S.TACTICAL)]
    for c in result["configs"]:
        L.append(f"| {c['radius_m']} | {c['replan_s']} | " + " | ".join(
            f"{c['policies'][p]['authority']['conversions']} / {c['policies'][p]['authority']['reversals']} / {c['policies'][p]['authority']['net_captures']:+d}" for p in S.TACTICAL) + " |")
    L += ["", "## 4. Mission gate G2: endurance exhaustion / timeouts (DIRECT: 0 / 0) and fly-by fraction", "", "| R | T | " + " | ".join(S.TACTICAL) + " |", "|---|---|" + "---|" * len(S.TACTICAL)]
    for c in result["configs"]:
        L.append(f"| {c['radius_m']} | {c['replan_s']} | " + " | ".join(f"{c['policies'][p]['exhausted']} / {c['policies'][p]['timeouts']}" for p in S.TACTICAL) + " |")
    L += ["", "## 5. Path efficiency (tactical episodes; DIRECT max below); worst-case policy shown", "", "| R | T | mean (range over policies) | median (range) | max | >1.3x | >2x | >5x | exhausted | timeouts |", "|---|---|---|---|---|---|---|---|---|---|"]
    for c in result["configs"]:
        ps = list(c["policies"].values())
        L.append(f"| {c['radius_m']} | {c['replan_s']} | {min(p['path_excess']['mean'] for p in ps):.2f}-{max(p['path_excess']['mean'] for p in ps):.2f} | "
                 f"{min(p['path_excess']['median'] for p in ps):.2f}-{max(p['path_excess']['median'] for p in ps):.2f} | {max(p['path_excess']['max'] for p in ps):.1f} | "
                 f"{sum(p['path_excess']['gt1.3'] for p in ps)} | {sum(p['path_excess']['gt2'] for p in ps)} | {sum(p['path_excess']['gt5'] for p in ps)} | "
                 f"{sum(p['exhausted'] for p in ps)} | {sum(p['timeouts'] for p in ps)} |")
    base = df[df.policy == "DIRECT"]
    L += ["", f"DIRECT reference: path-excess max {base.path_excess.max():.2f}x, >1.3x: {int((base.path_excess > 1.3).sum())} of {len(base)} (all configurations share the same DIRECT episodes), >5x: {int((base.path_excess > 5).sum())}, "
          f"exhaustion {int(base.exhausted.sum())}, timeouts {int(base.timeout.sum())}, captures {int(base.captured.sum())}, interceptions {int(base.intercepted.sum())}.", "",
          "## 6. Orbit / fly-by diagnostics (episodes; DIRECT in last row)", "", "| R | T | orbit>=1 rev | episodes with >=1 fly-by overshoot (share of 7,000 tactical episodes per configuration) |", "|---|---|---|---|"]
    for (R, T), cfg in df.groupby(["radius_m", "replan_s"]):
        t = cfg[cfg.policy != "DIRECT"]
        L.append(f"| {R} | {T} | {int((t.orbit >= 1).sum())} | {float((t.overshoots >= 1).mean()) * 100:.1f}% |")
    L.append(f"| DIRECT | - | {int((base.orbit >= 1).sum())} | {float((base.overshoots >= 1).mean()) * 100:.1f}% |")
    L += ["", "## 7. Perception audit G5 (planner calls pooled over the 7 tactical policies)", "", "| R | T | planner calls | calls without defender information | obstacle ids outside the perceived set | calls with unperceived static obstacles | frozen-layer suppressions | commands executed while the layer owned steering |", "|---|---|---|---|---|---|---|---|"]
    for c in result["configs"]:
        ps = list(c["policies"].values())
        L.append(f"| {c['radius_m']} | {c['replan_s']} | {sum(p['plans_attempted'] for p in ps)} | {sum(p['plan_calls_without_info'] for p in ps)} | {sum(p['ids_outside_perceived'] for p in ps)} | "
                 f"{sum(p['plan_calls_with_unperceived'] for p in ps)} | {sum(p['suppressed_by_obstacle'] for p in ps)} | {sum(p['executed_while_owned'] for p in ps)} |")
    L += ["", "## 8. Planner stability G6", "", "| R | T | plans/episode | replans/episode | mean waypoints | mean route length (m) | failed-plan share | longest route run (steps) | max plans in 4 steps | non-finite | top rejection reasons |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in result["configs"]:
        ps = list(c["policies"].values())
        inv = collections.Counter()
        for p in ps:
            inv.update(p["invalid"])
        attempts = sum(p["plans_attempted"] for p in ps)
        L.append(f"| {c['radius_m']} | {c['replan_s']} | {np.mean([p['plans_per_episode'] for p in ps]):.2f} | {np.mean([p['replans_per_episode'] for p in ps]):.2f} | "
                 f"{np.mean([p['waypoint_mean'] for p in ps]):.1f} | {np.mean([p['route_length_mean'] for p in ps]):.0f} | {sum(inv.values()) / max(attempts, 1) * 100:.1f}% | "
                 f"{max(p['longest_route_run'] for p in ps)} | {max(p['max_plans_in_4_steps'] for p in ps)} | {sum(p['non_finite'] for p in ps)} | {', '.join(f'{k}: {v}' for k, v in inv.most_common(3))} |")
    L += ["", "## 9. Mechanisms (re-simulated gained collisions; >5x classification)", "", "| config | gained (sampled) | gained-collision mechanism | >5x cases by outcome | exhaustions: DIRECT outcome |", "|---|---|---|---|---|"]
    for k, m in mech.items():
        g = "; ".join(f"{a}: {b}" for a, b in m["gained_collision_mechanism_sample"].items()) or "none"
        L.append(f"| {k} | {m['gained_total']} ({m['sampled']}) | {g} | {m['path_excess_gt5']['count']}: {m['path_excess_gt5']['by_outcome']} | {m['exhausted_base_outcome']} |")
    return "\n".join(L) + "\n"


def build(with_mechanisms: bool = True, reuse_mechanisms: bool = False) -> dict:
    records, flags, coverage = S.load_available()
    df = pd.DataFrame(records)
    result = S.analyze(records, flags)
    result["coverage"] = coverage
    cache = TS.STATE_DIR / "sweep_summary.json"
    if reuse_mechanisms and cache.exists():
        mech = json.loads(cache.read_text())["mechanisms"]
    else:
        mech = mechanisms(df) if with_mechanisms else {}
    md = render_markdown(df, result, mech)
    TS.atomic_write_json(S.SWEEP_DIR / "analysis.json", result)
    TS.atomic_write_json(TS.STATE_DIR / "sweep_summary.json", {"spec_sha256": result["spec_sha256"], "flags": flags, "selected": result["selected"], "coverage": coverage,
                                                              "forbidden_references": result["forbidden_references"], "configs": result["configs"], "mechanisms": mech})
    TS.atomic_write_bytes(TS.STATE_DIR / "sweep_results.md", md.encode("utf-8"))
    return {"result": result, "mech": mech, "md": md, "df": df}


if __name__ == "__main__":
    import sys

    print(build(reuse_mechanisms="--reuse" in sys.argv)["md"])
