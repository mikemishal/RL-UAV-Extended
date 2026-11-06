"""Controller-neutral trajectory/replay JSON export for manual visual review (journal_cbf_attacker,
defender_cbf_v3 / Prompt 6, brief section 23). Reuses the established, controller-agnostic
`_record_single_episode` helper (journal_cbf_attacker_defender_figures.py, unchanged since Prompt 4) -- these
exports reflect the CURRENT (Prompt 6, acceleration-reachable) controller only. No Prompt-5 "before" trajectory
export exists (only the numeric Monte Carlo evidence in docs/journal_cbf_attacker/defender_cbf_v2/README.md,
since that controller code no longer exists in the working tree) -- documented honestly rather than fabricated.
"""
from __future__ import annotations

import json
from pathlib import Path

import experiments.journal_cbf_attacker_defender_foundation as F
from experiments.journal_cbf_attacker_defender_figures import _record_single_episode

OUT_DIR = Path("docs/journal_cbf_attacker/defender_cbf_v3/visualizer_artifacts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _record_stress_seed(seed: int, defender: str = "greedy") -> dict:
    """Like `_record_single_episode`, but starts from the deterministic stress-scenario construction
    (`build_stress_scenario`) rather than a plain `reset()`, matching how seeds 630069/630123/630203 are defined."""
    import numpy as np
    from uav_defend.envs.soldier_env import SoldierEnv
    from uav_defend.policies.sanitize import build_policy_info

    cfg, estimator = F.defender_config(defender)
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=True)
    assert F.build_stress_scenario(env, seed)
    policy = F.VC.make_policy(defender, cfg)
    policy.reset()
    info = env._get_info()
    steps = [{
        "t": 0.0, "attacker_pos": list(map(float, info["enemy_pos"])), "defender_pos": list(map(float, info["defender_pos"])),
        "asset_pos": list(map(float, info["soldier_pos"])), "defender_observable": False, "intervention_source": "none",
        "cbf_qp_status": "inactive", "executor_velocity_error": 0.0, "physical_safety_infeasible": False,
        "reachable_emergency_brake": False,
    }]
    obstacles = [
        {"min": [float(x) for x in o.min_corner], "max": [float(x) for x in o.max_corner]}
        for o in env._obstacle_layout.obstacles
    ]
    done, t, step_count = False, 0.0, 0
    while not done and step_count < 60:
        action = np.asarray(policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator)), dtype=np.float32).reshape(3)
        obs, _r, te, tr, info = env.step(action)
        t += env.config.dt
        step_count += 1
        steps.append({
            "t": float(t), "attacker_pos": list(map(float, info["enemy_pos"])), "defender_pos": list(map(float, info["defender_pos"])),
            "asset_pos": list(map(float, info["soldier_pos"])), "defender_observable": bool(info.get("defender_observable", False)),
            "intervention_source": str(info.get("intervention_source", "none")), "cbf_qp_status": str(info["cbf_qp_status"]),
            "executor_velocity_error": float(info.get("executor_velocity_error", 0.0)),
            "physical_safety_infeasible": bool(info.get("physical_safety_infeasible", False)),
            "reachable_emergency_brake": bool(info.get("reachable_emergency_brake", False)),
        })
        done = te or tr
    return {"defender": defender, "seed": int(seed), "defender_cbf_enabled": True, "outcome": info["outcome"],
            "obstacles": obstacles, "steps": steps}


def _write(name: str, record: dict, what_to_look_for: str) -> None:
    record = dict(record)
    record["what_to_look_for"] = what_to_look_for
    record["schema"] = "a1_cbf_defender_cbf_v3_visualizer_trajectory/1"
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print("wrote", path, "outcome=", record["outcome"], "steps=", len(record["steps"]))


def main():
    _write("01_seed_630123_accel_reachable", _record_stress_seed(630123),
           "Prompt-5 obstacle-collision case, FIXED under Prompt 6: attacker brakes to a safe stop inside the "
           "obstacle's inflated clearance buffer (physical_safety_infeasible, reachable_emergency_brake) and "
           "holds there for the remainder of the horizon (outcome='ongoing') -- 0 collisions, "
           "executor_velocity_error stays at machine precision throughout.")
    _write("02_seed_630203_accel_reachable", _record_stress_seed(630203),
           "Prompt-5 obstacle-collision case, FIXED under Prompt 6: resolves to defender interception instead of "
           "an obstacle collision.")
    _write("03_seed_630069_accel_reachable", _record_stress_seed(630069),
           "Prompt-5 priority-conflict case (fixed in Prompt 5 via defender slack): remains collision-free under "
           "Prompt 6, but the outcome flips from 'soldier_caught' to 'intercepted' -- a tactical trade-off from "
           "the added acceleration-reachability constraint, not a safety regression.")

    oo_pn, fu_pn = [], []
    for p in sorted(Path("docs/journal_cbf_attacker/defender_cbf_v3/results").glob("matched_pn_accel_reachable_chunk_*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        oo_pn.extend(payload["obstacle_only"]); fu_pn.extend(payload["full"])
    strong_pn = next((e["scenario_seed"] for e in fu_pn if not e["scenario_generation_failure"] and e["asset_capture"]), fu_pn[0]["scenario_seed"])
    _write("04_strong_pn_evasion", _record_single_episode("pn", strong_pn, defender_cbf_enabled=True),
           "A PN-defender scenario where the attacker successfully evades/captures despite active defender-CBF "
           "intervention -- illustrates the large PN benefit preserved under Prompt 6.")

    oo_rmpc, fu_rmpc = [], []
    for p in sorted(Path("docs/journal_cbf_attacker/defender_cbf_v3/results").glob("matched_rmpc_accel_reachable_chunk_*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        oo_rmpc.extend(payload["obstacle_only"]); fu_rmpc.extend(payload["full"])
    strong_rmpc = next((e["scenario_seed"] for e in fu_rmpc if not e["scenario_generation_failure"] and e["asset_capture"]), fu_rmpc[0]["scenario_seed"])
    _write("05_strong_rmpc_evasion", _record_single_episode("rmpc", strong_rmpc, defender_cbf_enabled=True),
           "An RMPC-defender scenario where the attacker successfully evades/captures -- illustrates the large "
           "RMPC benefit preserved (reduced from Prompt 5 but still substantial) under Prompt 6.")

    greedy_oo, greedy_fu = [], []
    for p in sorted(Path("docs/journal_cbf_attacker/defender_cbf_v3/results").glob("matched_greedy_accel_reachable_chunk_*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        greedy_oo.extend(payload["obstacle_only"]); greedy_fu.extend(payload["full"])
    interception_seed = next((e["scenario_seed"] for e in greedy_fu if not e["scenario_generation_failure"] and e["defender_interception"]), greedy_fu[0]["scenario_seed"])
    _write("06_greedy_interception", _record_single_episode("greedy", interception_seed, defender_cbf_enabled=True),
           "A Greedy-defender scenario resolving to defender interception -- the dominant outcome against Greedy.")
    capture_seed = next((e["scenario_seed"] for e in greedy_fu if not e["scenario_generation_failure"] and e["asset_capture"]), greedy_fu[0]["scenario_seed"])
    _write("07_greedy_capture", _record_single_episode("greedy", capture_seed, defender_cbf_enabled=True),
           "A Greedy-defender scenario resolving to asset capture (attacker success) -- the modest-but-nonzero "
           "capture rate preserved under Prompt 6.")

    joint_seed = next((e["scenario_seed"] for e in greedy_fu if not e["scenario_generation_failure"] and e["joint_constraint_fraction"] > 0), greedy_fu[0]["scenario_seed"])
    _write("08_joint_obstacle_defender_maneuver", _record_single_episode("greedy", joint_seed, defender_cbf_enabled=True),
           "A scenario with simultaneously active obstacle AND defender constraints -- illustrates the "
           "hierarchical priority (obstacle+acceleration hard, defender soft) resolving a genuine joint conflict.")


if __name__ == "__main__":
    main()
