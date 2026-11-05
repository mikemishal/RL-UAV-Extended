"""Adversarial near-viability-boundary stress suite (journal_cbf_attacker, obstacle_cbf_v2 / Prompt 3, brief
section 15): deterministic, synthetic initial states satisfying `0 <= margin <= 0.5 m` for at least one static
obstacle, drawn from REAL canonical-C2 obstacle layouts (never a fictitious/relaxed layout). These are LEGAL
(viability-passing) but difficult states -- diagnostic only, separate from the 1,000-episode Monte Carlo rate
metrics. Requires zero physical obstacle collisions and zero numerical failures.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from uav_defend.attackers.cbf_obstacle import initial_state_dynamically_viable
from uav_defend.envs.soldier_env import SoldierEnv

from experiments import journal_v2_validation_core as VC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = "docs/journal_cbf_attacker/obstacle_cbf_v2/"

# Source C2 layouts (distinct seeds, disjoint from every other namespace used in this repo).
STRESS_LAYOUT_SEEDS = range(550_000, 550_020)   # 20 distinct C2 obstacle layouts
MARGIN_TARGETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)  # 6 targets x 20 layouts = 120 >= 100 required states
V_E = 12.0
A_MAX = 6.0
DT_SUB = 0.1


def clutter_config():
    cfg2, _ = VC.build_config("C2", "V2", "passive")
    return dataclasses.replace(cfg2, intercept_radius=1e-6, dynamic_obstacles_enabled=False)


def _face_center_and_normal(obstacle, clearance: float) -> tuple[np.ndarray, np.ndarray]:
    """Picks the horizontal face of the inflated AABB facing TOWARD the arena center (0, 0) -- so that backing off
    outward from the face (to construct the stopping-margin distance) moves the attacker further INTO the domain,
    never outside the [-L, L] boundary (a construction bug found and fixed during development: approaching from a
    FIXED -y face placed some attackers outside the arena, where `apply_boundary` clipping interacted with the
    crafted velocity in ways the test never intended to exercise). Returns `(face_center, outward_normal)`."""
    min_c = obstacle.min_corner - clearance
    max_c = obstacle.max_corner + clearance
    center = (min_c + max_c) / 2.0
    to_center_xy = -center[:2]
    norm = float(np.linalg.norm(to_center_xy))
    direction_xy = to_center_xy / norm if norm > 1e-6 else np.array([1.0, 0.0])
    axis = 0 if abs(direction_xy[0]) >= abs(direction_xy[1]) else 1
    sign = 1.0 if direction_xy[axis] > 0 else -1.0
    normal = np.zeros(3)
    normal[axis] = sign
    face_center = center.copy()
    face_center[axis] = max_c[axis] if sign > 0 else min_c[axis]
    return face_center, normal


def generate_stress_states() -> list[dict]:
    """Returns a list of `{seed, obstacle_id, margin_target, position, velocity}` synthetic near-boundary states,
    each constructed from a REAL C2 obstacle layout (canonical, unmodified)."""
    cfg = clutter_config()
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=False)
    states = []
    for seed in STRESS_LAYOUT_SEEDS:
        env.reset(seed=seed)
        obstacles = env._obstacle_layout.obstacles
        if not obstacles:
            continue
        # Use the obstacle closest to the arena center for consistency/reproducibility across layouts.
        obstacle_id, obstacle = min(enumerate(obstacles), key=lambda item: float(np.linalg.norm(item[1].center[:2])))
        for margin_target in MARGIN_TARGETS:
            face_center, normal = _face_center_and_normal(obstacle, clearance=2.0)
            stopping_at_v_e = (V_E ** 2) / (2 * A_MAX) + V_E * DT_SUB
            d_target = stopping_at_v_e + margin_target
            position = face_center + normal * d_target  # OUTWARD from the face, at the target stopping distance
            position[2] = max(position[2], 1.0)  # stay above ground
            velocity = -normal * V_E  # purely radial, full speed, directly INTO the face -- the hardest case
            states.append({
                "seed": int(seed), "obstacle_id": int(obstacle_id), "margin_target": margin_target,
                "position": position.tolist(), "velocity": velocity.tolist(),
            })
    return states


def run_stress_suite() -> dict:
    states = generate_stress_states()
    cfg = clutter_config()
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=False)
    results = []
    for s in states:
        env.reset(seed=s["seed"])
        position = np.array(s["position"], dtype=np.float32)
        velocity = np.array(s["velocity"], dtype=np.float32)
        viability = initial_state_dynamically_viable(position, velocity, env._obstacle_layout.obstacles, max_accel=A_MAX, dt_sub=DT_SUB)
        env._enemy_pos = position.copy()
        env._enemy_vel = velocity.copy()
        collided = False
        nonfinite = False
        min_clearance = float("inf")
        steps = 0
        done = False
        outcome = "ongoing"
        while not done and steps < 3000:
            obs, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
            steps += 1
            min_clearance = min(min_clearance, float(info["min_physical_obstacle_clearance"]))
            if not (np.isfinite(info["enemy_pos"]).all() and np.isfinite(info["enemy_speed"])):
                nonfinite = True
            outcome = info["outcome"]
            if outcome == "enemy_static_obstacle_collision":
                collided = True
            done = te or tr
        results.append({
            **s, "viable_at_construction": viability["viable"], "margin_at_construction": viability["min_margin"],
            "outcome": outcome, "steps": steps, "collision": collided, "nonfinite": nonfinite, "min_physical_clearance": min_clearance,
        })
    n = len(results)
    collisions = sum(r["collision"] for r in results)
    nonfinites = sum(r["nonfinite"] for r in results)
    construction_viable = sum(r["viable_at_construction"] for r in results)
    qualifying_boundary_states = sum(1 for r in results if 0.0 <= r["margin_at_construction"] <= 0.5)
    return {
        "n_states": n, "construction_viable_count": construction_viable,
        "qualifying_boundary_states_0_to_0.5m": qualifying_boundary_states,
        "collisions": collisions, "nonfinite_failures": nonfinites,
        "min_physical_clearance_observed": float(min((r["min_physical_clearance"] for r in results), default=float("inf"))),
        "passed": bool(collisions == 0 and nonfinites == 0 and qualifying_boundary_states >= 100),
        "margin_target_range": [min(MARGIN_TARGETS), max(MARGIN_TARGETS)],
        "results": results,
    }


if __name__ == "__main__":
    result = run_stress_suite()
    out_path = PROJECT_ROOT / "docs" / "journal_cbf_attacker" / "obstacle_cbf_v2" / "stress_test_result.json"
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "results"}, indent=1))
