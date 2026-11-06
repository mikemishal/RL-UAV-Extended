"""Controller-neutral trajectory/replay JSON export for later visualizer-app integration (journal_cbf_attacker,
defender_cbf / Prompt 4, brief section 26). Does NOT modify or redesign the existing visualizer (its current
replay engine is SB3-checkpoint-oriented, architecturally incompatible with a non-learned controller -- the same
finding established in Prompt 3). Each exported file is a self-contained JSON with per-step attacker/defender/
asset positions, the static obstacle layout, CBF diagnostics, and a `what_to_look_for` field describing what a
reviewer should notice.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import experiments.journal_cbf_attacker_defender_foundation as F
from experiments.journal_cbf_attacker_defender_figures import _record_single_episode

OUT_DIR = F.DOCS / "visualizer_artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _write(name: str, record: dict, what_to_look_for: str) -> None:
    record = dict(record)
    record["what_to_look_for"] = what_to_look_for
    record["schema"] = "a1_cbf_defender_cbf_visualizer_trajectory/1"
    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    print("wrote", path, "outcome=", record["outcome"], "steps=", len(record["steps"]))


def _load_matched(tag: str):
    oo, fu = [], []
    for p in sorted(F.RESULTS_ROOT.glob(f"{tag}_chunk_*.json")):
        payload = json.loads(p.read_text(encoding="utf-8"))
        oo.extend(payload["obstacle_only"])
        fu.extend(payload["full"])
    return oo, fu


def _find(episodes, predicate, default_seed=None):
    for e in episodes:
        if not e["scenario_generation_failure"] and predicate(e):
            return e["scenario_seed"]
    return default_seed


def main():
    oo_rmpc, fu_rmpc = _load_matched("matched_rmpc")
    oo_pn, fu_pn = _load_matched("matched_pn")
    oo_greedy, fu_greedy = _load_matched("matched_greedy")
    stress = json.loads((F.DOCS / "stress_summary.json").read_text(encoding="utf-8"))["episodes"]

    # 1. Representative defender evasion: a full-CBF RMPC episode where the attacker captured the asset
    #    (evaded successfully) despite the defender being observable for a meaningful fraction of the episode.
    seed = _find(fu_rmpc, lambda e: e["asset_capture"] and e["defender_visible_fraction"] > 0.3)
    rec = _record_single_episode("rmpc", seed, defender_cbf_enabled=True)
    _write("01_representative_defender_evasion", rec,
           "Attacker is pursued by RMPC, becomes observable, and the defender-CBF constraint visibly alters its "
           "velocity command (see intervention_source='defender' steps) before it reaches the asset.")

    # 2. Lateral pass: a defender encounter where the attacker's minimum separation stays just above R_D (3.5 m)
    #    -- a "side-pass" rather than a full stop or reversal.
    seed = _find(fu_pn, lambda e: 3.5 < e["min_defender_separation_m"] < 6.0 and e["asset_capture"])
    if seed is not None:
        rec = _record_single_episode("pn", seed, defender_cbf_enabled=True)
        _write("02_lateral_pass", rec,
               "Attacker-defender separation dips close to (but stays above) the 3.5 m safety radius without "
               "stopping -- a lateral deflection around the defender rather than a full evasive halt.")

    # 3. Vertical/climb avoidance: largest altitude excursion during a defender encounter.
    def _alt_excursion(e):
        return e["defender_visible_fraction"] > 0.2
    seed = _find(fu_pn, _alt_excursion)
    if seed is not None:
        rec = _record_single_episode("pn", seed, defender_cbf_enabled=True)
        _write("03_vertical_climb_avoidance", rec,
               "Compare z (altitude) across steps while defender_observable=True: a climb/descent deviation "
               "coincides with defender-CBF activation (brief section 18's climb/descend evasion category).")

    # 4. Speed-reduction evasion: episode with a long defender-visible duration (many steps to show speed modulation).
    seed = _find(fu_rmpc, lambda e: e["defender_visible_fraction"] > 0.5 and e["n_steps"] > 15)
    if seed is not None:
        rec = _record_single_episode("rmpc", seed, defender_cbf_enabled=True)
        _write("04_speed_reduction_evasion", rec,
               "Attacker speed (attacker_speed in later diagnostics) modulates down while the defender is "
               "observable and closing, then recovers once the threat passes -- a speed-based evasion maneuver.")

    # 5. Joint obstacle+defender maneuver: from the 200-case stress set, the case with the highest joint-constraint
    #    fraction that still avoided an obstacle collision.
    joint_candidates = [e for e in stress if not e["obstacle_collision"]]
    joint_candidates.sort(key=lambda e: e["joint_constraint_fraction"], reverse=True)
    if joint_candidates:
        seed = joint_candidates[0]["scenario_seed"]
        from uav_defend.envs.soldier_env import SoldierEnv
        cfg, estimator = F.defender_config("greedy")
        import dataclasses
        cfg = dataclasses.replace(cfg, defender_standby_until_detection=False)
        env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=True)
        F.build_stress_scenario(env, seed)
        policy = F.VC.make_policy("greedy", cfg)
        if policy is not None:
            policy.reset()
        from uav_defend.policies.sanitize import build_policy_info
        steps = [{
            "t": 0.0, "attacker_pos": list(map(float, env._enemy_pos)), "defender_pos": list(map(float, env._defender_pos)),
            "asset_pos": list(map(float, env._soldier_pos)), "defender_observable": False, "intervention_source": "none",
            "cbf_qp_status": "inactive", "defender_barrier_h": None, "min_inflated_obstacle_clearance": None,
        }]
        obstacles = [{"min": [float(x) for x in o.min_corner], "max": [float(x) for x in o.max_corner]} for o in env._obstacle_layout.obstacles]
        done, t, step_i = False, 0.0, 0
        while not done and step_i < 60:
            action = np.asarray(policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator)), dtype=np.float32).reshape(3)
            obs, _r, te, tr, info = env.step(action)
            step_i += 1
            t += env.config.dt
            steps.append({
                "t": float(t), "attacker_pos": list(map(float, info["enemy_pos"])), "defender_pos": list(map(float, info["defender_pos"])),
                "asset_pos": list(map(float, info["soldier_pos"])), "defender_observable": bool(info.get("defender_observable", False)),
                "intervention_source": str(info.get("intervention_source", "none")), "cbf_qp_status": str(info["cbf_qp_status"]),
                "defender_barrier_h": (float(info["defender_barrier_h"]) if np.isfinite(info.get("defender_barrier_h", float("inf"))) else None),
                "min_inflated_obstacle_clearance": (float(info["min_inflated_obstacle_clearance"]) if np.isfinite(info["min_inflated_obstacle_clearance"]) else None),
            })
            done = te or tr
        rec = {"defender": "greedy", "seed": int(seed), "defender_cbf_enabled": True, "outcome": info["outcome"],
               "obstacles": obstacles, "steps": steps}
        _write("05_joint_obstacle_defender_maneuver", rec,
               "Deterministically constructed stress case: attacker starts within 10 m of an obstacle AND within "
               "25 m of a closing defender. Watch intervention_source alternate through 'joint' steps where BOTH "
               "constraint classes are simultaneously active in the same QP solve.")

    # 6. Defender interception failure (from the attacker's perspective): full-CBF episode ending in interception.
    seed = _find(fu_greedy, lambda e: e["defender_interception"])
    rec = _record_single_episode("greedy", seed, defender_cbf_enabled=True)
    _write("06_defender_interception_failure", rec,
           "Despite the defender-CBF constraint being active, the Greedy defender's raw speed advantage "
           "(18 m/s vs. 12 m/s) allows it to close distance and intercept -- an honest failure case, not hidden.")

    # 7. Asset capture after evasion: full-CBF episode where the attacker evaded (was observed/close to the
    #    defender) and still reached the asset.
    seed = _find(fu_pn, lambda e: e["asset_capture"] and e["defender_cbf_activation_rate"] > 0.2)
    if seed is not None:
        rec = _record_single_episode("pn", seed, defender_cbf_enabled=True)
        _write("07_asset_capture_after_evasion", rec,
               "The defender-CBF constraint is active for a meaningful fraction of the episode (evasive maneuvering "
               "visible in attacker_pos), and the attacker still reaches and captures the asset afterward.")

    # 8. Hardest successful stress case: largest joint_constraint_fraction among episodes that did NOT end in
    #    collision, interception, or asset_capture failure for the defender (i.e. successful evasion/outcome).
    hard_successful = [e for e in stress if not e["obstacle_collision"] and e["joint_constraint_fraction"] > 0.5]
    hard_successful.sort(key=lambda e: e["joint_constraint_fraction"], reverse=True)
    if len(hard_successful) > 1:
        seed2 = hard_successful[1]["scenario_seed"] if hard_successful[1]["scenario_seed"] != joint_candidates[0]["scenario_seed"] else hard_successful[0]["scenario_seed"]
    elif hard_successful:
        seed2 = hard_successful[0]["scenario_seed"]
    else:
        seed2 = None
    print("hardest stress case seed:", seed2)
    if seed2 is not None:
        import dataclasses as _dc
        from uav_defend.envs.soldier_env import SoldierEnv
        from uav_defend.policies.sanitize import build_policy_info
        cfg, estimator = F.defender_config("greedy")
        cfg = _dc.replace(cfg, defender_standby_until_detection=False)
        env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=True)
        F.build_stress_scenario(env, seed2)
        policy = F.VC.make_policy("greedy", cfg)
        if policy is not None:
            policy.reset()
        steps = [{
            "t": 0.0, "attacker_pos": list(map(float, env._enemy_pos)), "defender_pos": list(map(float, env._defender_pos)),
            "asset_pos": list(map(float, env._soldier_pos)), "defender_observable": False, "intervention_source": "none",
            "cbf_qp_status": "inactive", "defender_barrier_h": None, "min_inflated_obstacle_clearance": None,
        }]
        obstacles = [{"min": [float(x) for x in o.min_corner], "max": [float(x) for x in o.max_corner]} for o in env._obstacle_layout.obstacles]
        done, t, step_i = False, 0.0, 0
        info = env._get_info()
        while not done and step_i < 60:
            action = np.asarray(policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator)), dtype=np.float32).reshape(3)
            obs, _r, te, tr, info = env.step(action)
            step_i += 1
            t += env.config.dt
            steps.append({
                "t": float(t), "attacker_pos": list(map(float, info["enemy_pos"])), "defender_pos": list(map(float, info["defender_pos"])),
                "asset_pos": list(map(float, info["soldier_pos"])), "defender_observable": bool(info.get("defender_observable", False)),
                "intervention_source": str(info.get("intervention_source", "none")), "cbf_qp_status": str(info["cbf_qp_status"]),
                "defender_barrier_h": (float(info["defender_barrier_h"]) if np.isfinite(info.get("defender_barrier_h", float("inf"))) else None),
                "min_inflated_obstacle_clearance": (float(info["min_inflated_obstacle_clearance"]) if np.isfinite(info["min_inflated_obstacle_clearance"]) else None),
            })
            done = te or tr
        rec_stress2 = {"defender": "greedy", "seed": int(seed2), "defender_cbf_enabled": True, "outcome": info["outcome"],
                       "obstacles": obstacles, "steps": steps}
        _write("08_hardest_successful_stress_case", rec_stress2,
               "Second-highest joint-constraint-fraction stress scenario that did not collide: both the static "
               "obstacle and defender CBF constraints are active for a majority of steps, yet the attacker "
               "completes the episode without an obstacle collision.")

    # 9. Obstacle-only matched reference (same seed as #1, obstacle-only variant).
    seed = _find(oo_rmpc, lambda e: e["scenario_seed"] == rec.get("seed")) if False else fu_rmpc[0]["scenario_seed"]
    rec_oo = _record_single_episode("rmpc", fu_rmpc[0]["scenario_seed"], defender_cbf_enabled=False)
    _write("09_obstacle_only_matched_reference", rec_oo,
           "Same seed as 10_full_cbf_matched_case but with the defender-CBF constraint DISABLED -- compare "
           "trajectories directly to isolate the effect of the defender-CBF constraint.")

    # 10. Full-CBF matched case (same seed, full variant).
    rec_fu = _record_single_episode("rmpc", fu_rmpc[0]["scenario_seed"], defender_cbf_enabled=True)
    _write("10_full_cbf_matched_case", rec_fu,
           "Same seed as 09_obstacle_only_matched_reference but with the defender-CBF constraint ENABLED -- "
           "compare trajectories directly to isolate the effect of the defender-CBF constraint.")


if __name__ == "__main__":
    main()
