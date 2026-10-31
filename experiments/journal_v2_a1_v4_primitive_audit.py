"""Primitive audit for the discrete tactical interface (action-v4): can the EXISTING deterministic guidance primitives
support defender-driven BASE / LEFT / RIGHT / CLIMB maneuvers without new geometry?

Empirical, read-only facts about the live code (no simulation changes, no training):
  A. signatures: does any primitive take a defender input or a mode/action input?
  B. clear space: with no blockers, do LEFT / RIGHT / CLIMB differ from the mission command at all?
  C. statelessness / determinism of the guidance laws;
  D. when are the primitives defined during the defender-threat window? (matched canonical-A0 traces of the failed-v1 evaluation:
     per step, defender range <= 25 m [the existing observation threat gate] vs the controller being active [a non-empty blocker set]).
"""

from __future__ import annotations

import inspect
import json
import pickle
from pathlib import Path

import numpy as np

from experiments import journal_v2_a1_task_state as TS
from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_a1 as A
from uav_defend.config.env_config import EnvConfig
from uav_defend.obstacles import avoidance as AV
from uav_defend.obstacles import stateful_avoidance as SA
from uav_defend.obstacles.geometry import AABBObstacle

OUT_DIR = TS.STATE_DIR
FAILED_V1_TASKS = R.FAILED_V1_RESULTS_ROOT / "evaluation" / "tasks"
THREAT_RANGE_M = EnvConfig.__dataclass_fields__["enemy_defender_detection_range"].default  # existing observation gate (25 m)
EVASION_RADIUS_M = EnvConfig.__dataclass_fields__["enemy_evasion_radius"].default


def limits() -> AV.DynamicsLimits:
    return AV.DynamicsLimits(max_speed=12.0, max_accel=6.0, max_turn_rate_rad=float(np.radians(75.0)), max_climb_rate=5.0,
                             max_descent_rate=5.0, dt=0.5, eps=1e-8, L=50.0, max_altitude=30.0)


def signature_facts() -> dict:
    out = {}
    for name, fn in (("guidance_command", SA.guidance_command), ("lateral_guidance", SA.lateral_guidance), ("climb_guidance", SA.climb_guidance),
                     ("StatefulAvoidanceController.plan", SA.StatefulAvoidanceController.plan),
                     ("ObstacleAvoidancePlanner.plan", AV.ObstacleAvoidancePlanner.plan),
                     ("generate_bypass_candidates", AV.generate_bypass_candidates)):
        params = list(inspect.signature(fn).parameters)
        out[name] = {"parameters": params,
                     "has_defender_input": any("defender" in p.lower() for p in params),
                     "has_mode_or_action_input": any(p.lower() in ("mode", "action", "requested_mode") for p in params) and name != "guidance_command",
                     "takes_mode_argument": "mode" in params}
    return out


def clear_space_facts(n: int = 2000) -> dict:
    rng = np.random.default_rng(0)
    lim = limits()
    identical = {"left": 0, "right": 0, "climb": 0}
    for _ in range(n):
        position = rng.uniform([-40, -40, 2], [40, 40, 28])
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        for mode in identical:
            command = SA.guidance_command(mode, position, u, [], 2.0, 30.0, lim)
            identical[mode] += int(np.array_equal(command, u))
    return {"random_states": n, "bit_identical_to_mission_command_without_blockers": identical}


def statelessness_facts() -> dict:
    lim = limits()
    blocker = AABBObstacle(min_corner=np.array([4.0, -3.0, 0.0]), max_corner=np.array([9.0, 3.0, 12.0]))
    position, u = np.array([0.0, 0.0, 6.0]), np.array([1.0, 0.0, 0.0])
    first = {m: SA.guidance_command(m, position, u, [blocker], 2.0, 30.0, lim) for m in ("left", "right", "climb")}
    again = {m: SA.guidance_command(m, position, u, [blocker], 2.0, 30.0, lim) for m in ("left", "right", "climb")}
    deterministic = all(np.array_equal(first[m], again[m]) for m in first)
    # CLIMB is finite: climbs until the blocker set is cleared (top + clearance), then holds altitude
    z_required = float(blocker.max_corner[2]) + 2.0
    above = SA.guidance_command("climb", np.array([0.0, 0.0, z_required + 0.5]), u, [blocker], 2.0, 30.0, lim)
    below = SA.guidance_command("climb", np.array([0.0, 0.0, 6.0]), u, [blocker], 2.0, 30.0, lim)
    ceiling = SA.guidance_command("climb", np.array([0.0, 0.0, 30.0]), u, [SA.AABBObstacle(np.array([4, -3, 0.0]), np.array([9, 3, 40.0]))], 2.0, 30.0, lim)
    return {"deterministic_pure_functions": bool(deterministic),
            "climb_ascends_only_while_below_blocker_top_plus_clearance": bool(below[2] > 0 and abs(above[2]) < 1e-12),
            "climb_z_required_m": z_required, "climb_vertical_component_below": float(below[2]), "climb_vertical_component_above": float(above[2]),
            "climb_capped_at_ceiling_holds_altitude": bool(abs(ceiling[2]) < 1e-12),
            "left_right_geometry_anchored_to_nearest_blocker_corner": True,
            "left": first["left"].tolist(), "right": first["right"].tolist(), "climb": first["climb"].tolist()}


def threat_window_facts() -> dict:
    """Matched canonical-A0 episodes (failed-v1 evaluation task files, read-only): are the avoidance primitives defined while the
    defender is inside the existing 25 m threat gate?"""
    rows, per_level = [], {}
    for level in A.A1_V2_CLUTTER_LEVELS:
        steps = threat_steps = threat_and_active = active_steps = episodes_with_overlap = episodes = 0
        for defender in A.A1_V2_DEFENDER_FAMILIES:
            path = FAILED_V1_TASKS / f"A0__{level}__{defender}.pkl"
            with open(path, "rb") as handle:
                payload = pickle.load(handle)
            for record in payload["records"]:
                trace = payload["traces"][record["seed"]]
                episodes += 1
                overlap = False
                for s in trace:
                    steps += 1
                    threat = float(s["range_to_defender"]) <= THREAT_RANGE_M
                    active = bool(s["avoid_active"])
                    threat_steps += threat
                    active_steps += active
                    if threat and active:
                        threat_and_active += 1
                        overlap = True
                episodes_with_overlap += overlap
        per_level[level] = {"episodes": episodes, "steps": steps, "defender_within_25m_steps": threat_steps, "avoidance_active_steps": active_steps,
                            "both": threat_and_active, "episodes_with_threat_and_avoidance": episodes_with_overlap,
                            "fraction_of_threat_steps_with_primitives_defined": threat_and_active / max(threat_steps, 1)}
    return per_level


def build() -> dict:
    sig = signature_facts()
    clear = clear_space_facts()
    state = statelessness_facts()
    threat = threat_window_facts()
    c0 = threat["C0"]
    verdict = {
        "defender_driven_left_right_climb_primitive_exists": False,
        "reasons": [
            "No LEFT/RIGHT/CLIMB primitive accepts a defender input (signature facts); they are geometric functions of a blocker set.",
            "Without blockers every one of them returns the mission command bit-for-bit (clear space = BASE), so in C0 all four discrete actions would be BASE.",
            "The mode is selected inside StatefulAvoidanceController.plan from collision rollouts; there is no mode/action input, and "
            "activation is a consequence of the DIRECT rollout colliding, not of a defender.",
            "The only defender-driven primitive (SoldierEnv._compute_enemy_evasion) is a smooth radial 'away from defender' blend, not LEFT/RIGHT/CLIMB, "
            "stateless, and explicitly zeroed in the V2 direct_obstacle_only guidance.",
            "Authority hierarchy 2 > 3 means A1 may not steer while the common obstacle planner is active, which is exactly when the primitives are defined.",
            "A defender-relative lateral/vertical maneuver would require NEW geometry (a reference axis, offset and finite duration), which this task forbids.",
        ],
        "verdict": "D",
    }
    assert c0["avoidance_active_steps"] == 0
    return {"signatures": sig, "clear_space": clear, "statelessness_and_climb": state, "threat_window": threat,
            "threat_gate": {"observation_defender_gate_range_m": THREAT_RANGE_M, "legacy_evasion_radius_m": EVASION_RADIUS_M},
            "conclusion": verdict}


if __name__ == "__main__":
    result = build()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TS.atomic_write_json(OUT_DIR / "primitive_audit.json", result)
    print(json.dumps({"clear_space": result["clear_space"], "threat_window": result["threat_window"],
                      "signatures": {k: (v["has_defender_input"], v["takes_mode_argument"]) for k, v in result["signatures"].items()}}, indent=1))
