"""Stage 2 — audit of the existing visibility-graph / A* route-planning infrastructure for use by the ATTACKER (route-level A1). Facts are taken from the live code."""

from __future__ import annotations

import inspect
import json

import numpy as np

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles import perception as PC
from uav_defend.obstacles import route_planner as RP

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_rt_state as TS


def build_audit() -> dict:
    sig_build = str(inspect.signature(RP.build_and_search_route))
    sig_plan = str(inspect.signature(RP.RoutePlanner.plan))
    sig_direct = str(inspect.signature(RP.RoutePlanner.direct_path_clear))
    levels = {}
    for level in ("C0", "C1", "C2", "C3"):
        config = A.build_episode_config(level, "greedy")
        env = SoldierEnv(config)
        env.reset(seed=145002)
        env.step(np.zeros(3, dtype=np.float32))
        perceived = getattr(env, "_enemy_last_perceived", None)
        levels[level] = {"awareness_mode": config.enemy_obstacle_awareness_mode, "avoidance_enabled": config.enemy_obstacle_avoidance_enabled,
                         "route_planning_enabled_in_canonical_A0": config.enemy_route_planning_enabled, "guidance": config.enemy_nominal_guidance_mode,
                         "ground_truth_static_obstacles": len(env._obstacle_layout), "ground_truth_dynamic_obstacles": len(env._dynamic_obstacles),
                         "perceived_static_at_step_1": None if perceived is None else len(perceived.static),
                         "perceived_dynamic_at_step_1": None if perceived is None else len(perceived.dynamic)}
    mode_doc = PC.__doc__
    cfg = A.build_episode_config("C3", "greedy")
    return {
        "planner_functions": ["uav_defend/obstacles/route_planner.py: build_and_search_route (pure visibility graph + A*)", "RoutePlanner (stateful wrapper)",
                              "RouteState (waypoints, index, obstacle ids, replan reason)"],
        "signatures": {"build_and_search_route": sig_build, "RoutePlanner.plan": sig_plan, "RoutePlanner.direct_path_clear": sig_direct},
        "route_representation": "RouteState: tuple of 3-D waypoints, current_waypoint_index, stable obstacle ids, replan reason, estimated length, goal_at_creation",
        "waypoint_representation": "np.ndarray(3); graph nodes = start, goal, 4 inflated horizontal AABB corners per perceived static obstacle (+1 vertical-over waypoint when below the ceiling)",
        "static_obstacle_support": "yes: inflated AABBs from PerceivedObstacleSet.static; collision-free edges are a hard constraint, not a cost",
        "dynamic_obstacle_support": "none in the planner (it never reads PerceivedObstacleSet.dynamic); dynamic obstacles are handled only by the frozen reactive layer",
        "replan_logic": "RoutePlanner.plan: replans on no_active_route, route_unsafe_or_new_obstacle, target_moved (goal shift > max(0.5*planning_range, 4*clearance)), progress_stall, route_complete; "
                        "final waypoint tracks the moving (local) goal every step",
        "moving_target_support": "yes (live final-waypoint tracking + target_moved replan); here the target is the protected asset",
        "perception_inputs": "PerceivedObstacleSet ONLY (docstring: 'NEVER receives the ground-truth world'); build_and_search_route(position, goal, perceived, clearance, max_altitude, planning_range, max_graph_nodes)",
        "route_following": "RoutePlanner returns the unit direction to the current waypoint; env feeds it as r_hat (legacy) into the common obstacle layer",
        "integration_with_common_controller": "route command -> StatefulAvoidanceController (frozen, last stage) -> dynamics; the controller returns its input unchanged when its DIRECT mission rollout is predicted safe",
        "canonical_v2_A0": "route planning is DISABLED (enemy_route_planning_enabled False, direct_obstacle_only): A0 = direct pursuit + frozen reactive layer; the planner is therefore never run for A0",
        "canonical_perception_per_level": levels,
        "map_knowledge_note": "in C1-C3 every static obstacle is map-known (perceived at step 1 == ground truth static), dynamic obstacles are only locally perceived (sensor range "
                              f"{cfg.enemy_dynamic_obstacle_sensor_range} m) and are not used by the planner; this IS the canonical attacker perception contract",
        "can_be_used_for_attacker": True,
        "requires_privileged_information": False,
        "privileged_information_analysis": "the planner accepts only a PerceivedObstacleSet and positions; the A1 route layer passes `env._enemy_last_perceived` (the set the frozen layer itself uses) and the current "
                                           "attacker/asset/defender positions (defender only while the A1 defender-information gate is open). The asset position is the mission target A0 already uses.",
        "limitations": ["static obstacles only", "visibility graph limited to planning_range (40 m) and 150 nodes", "dynamics feasibility enforced only by the frozen reactive layer",
                        "route legs are not checked against dynamic obstacles or against the vehicle's turning radius"],
        "docstring_of_perception_module_excerpt": mode_doc.split("Awareness modes:")[1][:900] if "Awareness modes:" in mode_doc else "",
    }


if __name__ == "__main__":
    audit = build_audit()
    TS.atomic_write_json(TS.STATE_DIR / "route_planner_audit.json", audit)
    print(json.dumps({k: audit[k] for k in ("can_be_used_for_attacker", "requires_privileged_information", "canonical_perception_per_level")}, indent=1)[:1800])
