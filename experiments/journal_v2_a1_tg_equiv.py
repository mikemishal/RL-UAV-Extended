"""BASE-equivalence study: forcing BASE for every decision must reproduce canonical scripted A0 exactly (trajectory, outcome, collisions, endurance)."""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp

import numpy as np

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_tg as G
from experiments import journal_v2_a1_tg_state as TS

SEEDS = tuple(range(145_000, 145_100))
FAMILIES = A.A1_V2_DEFENDER_FAMILIES
OUT_PATH = G.RESULTS_ROOT / "tactical_geometry_sweep" / "base_equivalence.json"


def _run(env: G.A1V5TrainingEnv, scenario, scripted: bool) -> dict:
    env.reset(options={"scenario": scenario})
    done, info = False, {}
    while not done:
        _o, _r, te, tr, info = env.step(0)
        done = te or tr
    return info["a1_episode"]


def level_task(level: str) -> dict:
    bank = A.A1V2DefenderBank()
    machine_env = G.A1V5TrainingEnv(0, 0, defender_bank=bank, theta_deg=20.0, pulse_s=1.0, record_trajectory=True)
    a0_env = G.A1V5TrainingEnv(0, 0, defender_bank=bank, scripted_a0=True, theta_deg=20.0, pulse_s=1.0, record_trajectory=True)
    n = identical_traj = identical_outcome = 0
    max_dev = 0.0
    collisions_machine = collisions_a0 = exhausted_machine = exhausted_a0 = 0
    mismatches, tactical_steps = [], 0
    for family in FAMILIES:
        for seed in SEEDS:
            scenario = (level, family, int(seed))
            a = _run(a0_env, scenario, True)
            b = _run(machine_env, scenario, False)
            n += 1
            ta, tb = np.asarray(a["trajectory"]), np.asarray(b["trajectory"])
            same_len = len(ta) == len(tb)
            dev = float(np.max(np.linalg.norm(ta - tb, axis=1))) if same_len and len(ta) else float("inf")
            max_dev = max(max_dev, dev)
            identical_traj += int(same_len and dev == 0.0)
            same_out = (a["outcome"], a["steps"]) == (b["outcome"], b["steps"])
            identical_outcome += int(same_out)
            col = ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision")
            collisions_a0 += int(a["outcome"] in col)
            collisions_machine += int(b["outcome"] in col)
            exhausted_a0 += int(bool(a["endurance_exhausted"]))
            exhausted_machine += int(bool(b["endurance_exhausted"]))
            tactical_steps += int(b["tactical_steps"])
            if not (same_len and dev == 0.0 and same_out) and len(mismatches) < 5:
                mismatches.append({"scenario": list(scenario), "a0": a["outcome"], "machine": b["outcome"], "dev": dev})
    return {"level": level, "episodes": n, "identical_trajectory": identical_traj, "identical_outcome": identical_outcome, "max_positional_deviation_m": max_dev,
            "a0_collisions": collisions_a0, "machine_base_collisions": collisions_machine, "gained_collisions": max(collisions_machine - collisions_a0, 0),
            "a0_endurance_exhausted": exhausted_a0, "machine_endurance_exhausted": exhausted_machine, "tactical_steps_in_base": tactical_steps,
            "mismatches": mismatches}


def run(workers: int = 4) -> dict:
    with mp.get_context("spawn").Pool(workers) as pool:
        levels = pool.map(level_task, list(A.A1_V2_CLUTTER_LEVELS))
    passed = all(r["identical_trajectory"] == r["episodes"] and r["identical_outcome"] == r["episodes"] and r["gained_collisions"] == 0
                 and r["a0_endurance_exhausted"] == r["machine_endurance_exhausted"] and r["tactical_steps_in_base"] == 0 for r in levels)
    payload = {"definition": "policy forced to BASE (action 0) through the tactical machine vs unmodified scripted A0, same scenarios",
               "seeds": [SEEDS[0], SEEDS[-1]], "families": list(FAMILIES), "levels": levels, "passed": bool(passed),
               "episodes": int(sum(r["episodes"] for r in levels))}
    payload["sha256_of_levels"] = hashlib.sha256(json.dumps(levels, sort_keys=True).encode()).hexdigest()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TS.atomic_write_json(OUT_PATH, payload)
    return payload


if __name__ == "__main__":
    out = run()
    print(json.dumps({k: v for k, v in out.items() if k != "levels"}, indent=1))
    for r in out["levels"]:
        print({k: r[k] for k in r if k != "mismatches"}, r["mismatches"][:2])
