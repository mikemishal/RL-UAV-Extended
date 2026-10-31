"""Stage 2 — audit of `SoldierEnv._compute_enemy_evasion` as the tactical basis for A1 (action-v6). Produces `primitive_audit.json` from the LIVE code
(source inspection + numerical probes), not from documentation."""

from __future__ import annotations

import ast
import inspect
import json
import textwrap

import numpy as np

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles import stateful_avoidance as SA

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_eg_state as TS


def _line_of(source_lines: list[str], needle: str, start: int = 0) -> int | None:
    for i in range(start, len(source_lines)):
        if needle in source_lines[i]:
            return i + 1
    return None


def build_audit() -> dict:
    src_fn = inspect.getsource(SoldierEnv._compute_enemy_evasion)
    tree = ast.parse(textwrap.dedent(src_fn))
    self_writes = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Store)
                   and isinstance(n.value, ast.Name) and n.value.id == "self"]
    move_src, move_start = inspect.getsourcelines(SoldierEnv._move_enemy)
    compute_call = _line_of(move_src, "self._compute_enemy_evasion(")
    zeroed = _line_of(move_src, 'evasion_info["evasion_component"] = np.zeros(3')
    in_command = _line_of(move_src, "u_base_raw = r_hat + weave_component")
    override = _line_of(move_src, "self.enemy_tactical_override()")
    planner = _line_of(move_src, "self._enemy_obstacle_planner.plan(")
    ctl_src, ctl_start = inspect.getsourcelines(SA.StatefulAvoidanceController.plan)
    release = next((ctl_start + i for i, l in enumerate(ctl_src) if "return self._result(u_base, direct, False)" in l), None)

    cfg = A.build_episode_config("C0", "greedy")
    env = SoldierEnv(cfg)
    env.reset(seed=145000)
    probes = []
    for d in (0.0, 1e-9, float(cfg.eps) * 0.5, float(cfg.eps) * 2.0, 1.0, 5.0, 10.0, 19.99, 20.0, 25.0, 60.0):
        defender = np.array([0.0, 0.0, 10.0])
        enemy = defender + np.array([d, 0.0, 0.0])
        out = env._compute_enemy_evasion(defender, enemy)
        again = env._compute_enemy_evasion(defender, enemy)
        probes.append({"distance_m": d, "activation": out["activation"], "effective_weight": out["effective_weight"],
                       "away_direction": [float(x) for x in out["away_direction"]], "component_norm": float(np.linalg.norm(out["evasion_component"])),
                       "finite": bool(np.isfinite(out["evasion_component"]).all()), "repeat_identical": bool(np.array_equal(out["evasion_component"], again["evasion_component"]))})
    return {
        "function": "uav_defend/envs/soldier_env.py: SoldierEnv._compute_enemy_evasion(defender_pos, enemy_pos) -> dict",
        "inputs": "true 3-D defender and attacker positions (ground truth, independent of detection/sensor state); config.enemy_evasion_{enabled,radius,gain}, config.eps",
        "output": {"distance": "true 3-D range", "activation": "alpha = clip(1 - d/R_e, 0, 1) (0 if disabled)", "effective_weight": "gain*alpha",
                   "away_direction": "unit (enemy-defender)/d; fallback +z if d <= eps", "evasion_component": "effective_weight * away_direction (pre-combination vector)"},
        "defender_geometry": "radial 'away' direction only: full 3-D line from the defender through the attacker; no defender velocity, no prediction, no route",
        "activation_condition": f"smooth: alpha > 0 iff d < R_e = {cfg.enemy_evasion_radius} m (no binary switch); weight at d=0 is gain = {cfg.enemy_evasion_gain}",
        "original_gain_blend": f"legacy command: u_raw = r_hat + weave + sigma_e*z + evasion_component with gain {cfg.enemy_evasion_gain}; weave/noise draw RNG and are not part of this interface",
        "historical_injection": "inside SoldierEnv._move_enemy, computed before route arbitration (Phase 5) and summed into u_base_raw (legacy hostile model); route arbitration suspended route following while it was active",
        "config_in_V2_episode": {"enemy_nominal_guidance_mode": cfg.enemy_nominal_guidance_mode, "enemy_evasion_enabled": cfg.enemy_evasion_enabled,
                                "enemy_evasion_radius": cfg.enemy_evasion_radius, "enemy_evasion_gain": cfg.enemy_evasion_gain,
                                "enemy_defender_detection_range": cfg.enemy_defender_detection_range, "eps": cfg.eps},
        "v2_forces_contribution_to_zero": f"yes: in _move_enemy (line {zeroed} of the method) `direct_obstacle_only` replaces evasion_component with zeros and the legacy u_base_raw "
                                          f"expression (line {in_command}) is in the non-direct branch, so canonical A0 never uses it; diagnostics (distance, activation) are kept",
        "stateless": {"self_attribute_writes_inside_function": self_writes, "repeat_calls_identical_in_probes": all(p["repeat_identical"] for p in probes),
                      "hidden_state": "none: pure function of the two positions and static config"},
        "changes_speed_or_direction": "direction only: the result is combined into a unit direction that the point-mass dynamics (max accel / climb limits) then track; no speed term",
        "near_zero_distance": {"probes": probes, "all_finite": all(p["finite"] for p in probes),
                               "note": "alpha -> 1 and the direction is discontinuous only at d <= eps (fallback +z); no division by zero or NaN"},
        "order_in_move_enemy_lines_of_method": {"compute_enemy_evasion": compute_call, "zeroed_in_v2": zeroed, "legacy_command_sum": in_command,
                                                "tactical_override_replaces_u_base": override, "frozen_obstacle_planner_plan": planner,
                                                "evasion_and_override_precede_planner": bool(compute_call and override and planner and compute_call < override < planner)},
        "obstacle_planner_can_override": {
            "ordering": "yes: the planner runs AFTER the evasion/override stage on the same step and its returned direction is the final command",
            "condition": f"the frozen controller returns u_base UNCHANGED whenever its direct (mission-direction) rollout is predicted safe (stateful_avoidance.py line {release}); "
                         "its rollout re-derives the mission direction from the target position and does NOT test u_base. The planner therefore vetoes an evasion command only "
                         "in states where the DIRECT mission line is itself predicted unsafe.",
            "consequence": "an evasion-blended u_base that points toward an obstacle while the direct line is predicted clear is NOT checked by the frozen layer (the same veto gap that "
                           "defeated action-v2, v3 and v5). The ordering criterion of the STOP rule is not met (planner is last), so the study proceeds, and the gap is measured "
                           "empirically by G1/G6 rather than assumed."},
        "a1_only_without_changing_a0": "yes: applied through the existing `enemy_tactical_override` hook (None for A0); `uav_defend` is byte-identical to the A0 tag",
        "a1_blend_definition": "u(b) = normalize(m + b * evasion_component); b = 0 -> zero override vector -> the environment's own u_base bit-for-bit",
        "threat_gate_redundancy": "evasion activates only for d < 20 m and the A1 observation gate is 25 m, so the gate never removes an otherwise active evasion; it is structurally redundant here "
                                  "(scripted SWITCH_b and FIXED_b are expected to coincide; measured in the sweep)",
        "maximum_deflection_from_mission_direction_deg_if_away_is_orthogonal": float(np.degrees(np.arctan(float(cfg.enemy_evasion_gain)))),
        "observation": "stateless -> `a1_v2_obs_v1` (37-D) retained; the 43-D pulse-state observation is NOT used",
    }


if __name__ == "__main__":
    audit = build_audit()
    TS.atomic_write_json(TS.STATE_DIR / "primitive_audit.json", audit)
    print(json.dumps({k: audit[k] for k in ("stateless", "order_in_move_enemy_lines_of_method", "v2_forces_contribution_to_zero", "maximum_deflection_from_mission_direction_deg_if_away_is_orthogonal")}, indent=1)[:2500])
