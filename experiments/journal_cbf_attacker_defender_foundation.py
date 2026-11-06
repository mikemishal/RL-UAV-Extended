"""Monte Carlo acceptance experiment for the A1-CBF defender-avoidance stage (journal_cbf_attacker, defender_cbf /
Prompt 4): joint static-obstacle + defender CBF-QP evasion against an ACTIVE canonical defender policy (brief
section 12-21). Reuses the FROZEN Prompt-2/3 CBF-QP/route-planner parameters and the new Prompt-4 defender-CBF
parameters verbatim -- nothing about the controller itself is tuned in this stage.

Unlike Prompts 1-3 (which used the `intercept_radius=1e-6` no-defender-interaction trick), this stage runs the
REAL canonical defender dynamics (`intercept_radius=2.5`, `unsafe_intercept_radius=3.5`,
`defender_standby_until_detection=True`) with an ACTUAL policy (Greedy primary; Kalman-Greedy/PN/Lead/RMPC for the
generalization diagnostic) stepping the defender every external step -- see `experiments.journal_v2_validation_core`
for the established active-defender episode-loop pattern this module reuses.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from uav_defend.attackers.cbf_defender import DEFENDER_DETECTION_RANGE, DEFENDER_SAFETY_RADIUS
from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.sanitize import build_policy_info

from experiments import journal_v2_validation_core as VC
from experiments import journal_v2_a1_eg_state as base

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = PROJECT_ROOT / "docs" / "journal_cbf_attacker" / "defender_cbf"
RESULTS_ROOT = DOCS / "results"

# ----------------------------------------------------------------------------- pre-registered namespaces (frozen before any validation)
# All fresh blocks, non-overlapping with every namespace used by Prompts 1-3 (500_000-559_999 are already taken).
PRIMARY_SEEDS = (600_000, 600_999)           # 1,000 matched obstacle-only vs full-CBF scenarios vs Greedy
STATIC_CLUTTER_REGRESSION_SEEDS = (610_000, 610_499)  # 500 Prompt-2B no-defender clutter episodes, re-verified
C0_REGRESSION_SEEDS = (620_000, 620_999)      # 1,000-seed C0+scripted-A0 legacy no-op regression
STRESS_SEED_BASE = 630_000                    # base for the 200 deterministic joint obstacle+defender stress cases
GENERALIZATION_SEEDS = (640_000, 640_199)     # 200 matched episodes per additional defender family
CHUNK_SIZE = 100
GENERALIZATION_DEFENDERS = ("kalman_greedy", "pn", "lead", "rmpc")

# Frozen primary acceptance gate (brief section 16; never weakened after execution).
PRIMARY_GATE = {
    "obstacle_collision_max": 0, "numerical_failures_max": 0, "route_errors_max": 0,
    "scenario_generation_failures_max": 0, "dynamics_violations_max": 0,
    "qp_completion_min": 1.0, "soft_fallback_step_fraction_max": 0.02, "emergency_brake_step_fraction_max": 0.01,
}
# Frozen defender-avoidance EFFECT gate (brief section 17; never weakened after execution).
EFFECT_GATE = {
    "interception_reduction_min_pp": 10.0, "min_separation_increase_min_m": 1.0, "capture_drop_max_pp": 5.0,
}


def seeds_of(rng: tuple[int, int]) -> range:
    return range(rng[0], rng[1] + 1)


def chunk_seeds(namespace: tuple[int, int], chunk_index: int, chunk_size: int = CHUNK_SIZE) -> list[int]:
    start = namespace[0] + chunk_index * chunk_size
    stop = min(namespace[1], start + chunk_size - 1)
    return list(range(start, stop + 1))


def num_chunks(namespace: tuple[int, int], chunk_size: int = CHUNK_SIZE) -> int:
    total = namespace[1] - namespace[0] + 1
    return (total + chunk_size - 1) // chunk_size


def _content_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------- config
def defender_config(defender: str = "greedy", *, dynamic_obstacles_enabled: bool = False) -> tuple[EnvConfig, str]:
    """Canonical C2 V2 config with the REAL (never-disabled) canonical defender dynamics -- the established
    `VC.build_config("C2", "V2", defender)` already sets `intercept_radius=2.5`, `unsafe_intercept_radius=3.5`,
    `defender_standby_until_detection=True` (canonical final defender behavior, brief section 12), and the correct
    Kalman/measurement estimator track for the requested defender family. Dynamic clutter obstacles are disabled
    throughout this stage (brief section 2)."""
    cfg, estimator = VC.build_config("C2", "V2", defender)
    cfg = dataclasses.replace(cfg, dynamic_obstacles_enabled=dynamic_obstacles_enabled)
    return cfg, estimator


def config_sha256(defender: str = "greedy") -> str:
    cfg, _ = defender_config(defender)
    return hashlib.sha256(json.dumps(dataclasses.asdict(cfg), sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------- episode runner
def run_episode(env: SoldierEnv, policy, estimator: str, seed: int) -> dict:
    """Runs one C2 scenario with an ACTIVE defender policy stepping every external step (brief sections 12-15).
    `env` must already be constructed with `enemy_control_mode="cbf_velocity"`, `viability_conditioned_init=True`,
    and the desired `defender_cbf_enabled` (True=`A1-CBF-full`, False=`A1-CBF-obstacle-only`)."""
    obs, info = env.reset(seed=seed)
    if policy is not None:
        policy.reset()
    if info["scenario_generation_failure"]:
        return {
            "scenario_seed": int(seed), "outcome": "scenario_generation_failure", "scenario_generation_failure": True,
            "defender_cbf_enabled": bool(env._defender_cbf_enabled),
        }

    start_pos = np.asarray(info["enemy_pos"], dtype=np.float64).copy()
    asset0 = np.asarray(info["soldier_pos"], dtype=np.float64).copy()
    prev_pos = start_pos.copy()
    path_length = 0.0
    min_defender_sep = float("inf")
    first_defender_obs_step: int | None = None
    defender_obs_steps = 0
    defender_constraint_active_steps = 0
    joint_constraint_steps = 0
    unsafe_radius_steps = 0           # time spent inside 3.5 m
    interception_radius_steps = 0     # time spent inside 2.5 m (prior to termination)
    min_h_d = float("inf")
    intervention_sources: list[str] = []
    qp_statuses: list[str] = []
    route_statuses: list[str] = []
    route_errors_seen = 0
    defender_residuals: list[float] = []
    finite = True
    done, info_final, step = False, info, 0
    while not done:
        if policy is None:
            action = np.zeros(3, dtype=np.float32)
        else:
            action = np.asarray(
                policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator)), dtype=np.float32,
            ).reshape(3)
        obs, _r, te, tr, info = env.step(action)
        step += 1
        pos = np.asarray(info["enemy_pos"], dtype=np.float64)
        path_length += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos
        dist_de = float(np.linalg.norm(np.asarray(info["defender_pos"], dtype=np.float64) - pos))
        min_defender_sep = min(min_defender_sep, dist_de)
        if dist_de <= DEFENDER_SAFETY_RADIUS:
            unsafe_radius_steps += 1
        if dist_de <= env.config.intercept_radius:
            interception_radius_steps += 1
        if info.get("defender_observable"):
            defender_obs_steps += 1
            if first_defender_obs_step is None:
                first_defender_obs_step = step
            if info.get("defender_constraint_active"):
                defender_constraint_active_steps += 1
                defender_residuals.append(float(info.get("defender_constraint_residual", 0.0)))
            min_h_d = min(min_h_d, float(info.get("defender_barrier_h", float("inf"))))
        if info.get("simultaneous_obstacle_defender_constraints"):
            joint_constraint_steps += 1
        intervention_sources.append(str(info.get("intervention_source", "none")))
        qp_statuses.append(str(info["cbf_qp_status"]))
        route_statuses.append(str(info["route_status"]))
        if info["route_status"] == "route_error":
            route_errors_seen += 1
        finite = finite and bool(np.isfinite(pos).all() and np.isfinite(info["enemy_speed"]))
        done = te or tr
        info_final = info

    n_steps = step
    outcome = info_final["outcome"]
    hard_steps = sum(1 for s in qp_statuses if s == "hard")
    soft_steps = sum(1 for s in qp_statuses if s == "soft_fallback")
    emergency_steps = sum(1 for s in qp_statuses if s == "emergency_brake")
    chord = float(np.linalg.norm(start_pos - asset0))
    return {
        "scenario_seed": int(seed), "outcome": outcome, "scenario_generation_failure": False,
        "defender_cbf_enabled": bool(env._defender_cbf_enabled),
        "n_steps": n_steps, "duration_s": n_steps * env.config.dt,
        "path_length_m": path_length, "path_excess": (path_length / chord) if chord > 1e-9 else float("nan"),
        "finite": finite,
        "asset_capture": outcome == "soldier_caught",
        "defender_interception": outcome in ("intercepted", "unsafe_intercept"),
        "obstacle_collision": outcome in ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"),
        "endurance_exhausted": outcome == "enemy_endurance_exhausted",
        "timeout": outcome == "timeout",
        "min_defender_separation_m": min_defender_sep,
        "first_defender_observation_step": first_defender_obs_step,
        "defender_visible_steps": defender_obs_steps,
        "defender_visible_fraction": (defender_obs_steps / n_steps) if n_steps else 0.0,
        "defender_cbf_activation_rate": (defender_constraint_active_steps / n_steps) if n_steps else 0.0,
        "mean_defender_constraint_residual": (float(np.mean(defender_residuals)) if defender_residuals else 0.0),
        "min_h_d": min_h_d,
        "unsafe_radius_steps": unsafe_radius_steps, "interception_radius_steps": interception_radius_steps,
        "joint_constraint_steps": joint_constraint_steps, "joint_constraint_fraction": (joint_constraint_steps / n_steps) if n_steps else 0.0,
        "route_replan_count": int(info_final.get("route_replan_count", 0)),
        "route_errors": int(info_final.get("route_errors", 0)), "route_errors_observed": route_errors_seen,
        "emergency_brake_count": int(info_final.get("emergency_brake_count", 0)),
        "hard_step_fraction": (hard_steps / n_steps) if n_steps else 1.0,
        "soft_step_fraction": (soft_steps / n_steps) if n_steps else 0.0,
        "emergency_step_fraction": (emergency_steps / n_steps) if n_steps else 0.0,
        "qp_hard_or_soft_success": bool(emergency_steps == 0),
        "intervention_source_counts": {k: int(intervention_sources.count(k)) for k in ("none", "obstacle", "defender", "joint")},
        "max_speed_mps": float(info_final.get("attacker_speed", 0.0)),
        "viability_resample_count": int(info_final.get("viability_resample_count", 0)),
    }


def _point_near_inflated_obstacle(obstacle, *, clearance: float, distance: float, direction_angle: float, z: float) -> np.ndarray:
    """A deterministic point exactly `distance` meters (horizontally) outward from the INFLATED AABB's surface
    along direction `direction_angle`, at altitude `z` -- used to place the attacker "within 10 m of an inflated
    obstacle" (brief section 19) without any randomness beyond the caller-supplied angle/distance."""
    from uav_defend.attackers.cbf_obstacle import inflate_aabb

    min_c, max_c = inflate_aabb(obstacle, clearance)
    center = (min_c + max_c) / 2.0
    half = (max_c - min_c) / 2.0
    d = np.array([math.cos(direction_angle), math.sin(direction_angle), 0.0])
    ratios = [half[a] / abs(d[a]) for a in (0, 1) if abs(d[a]) > 1e-9]
    t_exit = min(ratios) if ratios else 0.0
    pos = center.copy()
    pos[:2] += d[:2] * t_exit
    pos += d * distance
    pos[2] = z
    return pos


def build_stress_scenario(env: SoldierEnv, seed: int, *, clearance: float = 2.0, obstacle_distance_m: float = 6.0,
                           defender_distance_m: float = 15.0) -> bool:
    """Deterministically constructs one joint obstacle+defender stress scenario (brief section 19) on top of the
    canonical C2 obstacle field drawn by `seed`: the attacker is placed `obstacle_distance_m` (< 10 m) from an
    inflated static obstacle with velocity directed INTO it (activates the obstacle CBF), and the defender is
    placed `defender_distance_m` (< 25 m, observable) away with velocity directed AT the attacker (positive closing
    speed, activates the defender CBF) -- both simultaneously, by construction. Returns False (scenario skipped) if
    the drawn seed has no static obstacles or scenario generation otherwise failed; the caller should then try the
    next seed in sequence (deterministic, no retries/resampling of this same seed)."""
    obs, info = env.reset(seed=seed)
    if info["scenario_generation_failure"] or not len(env._obstacle_layout):
        return False
    obstacles = env._obstacle_layout.obstacles
    rng = np.random.default_rng(seed)
    obstacle = obstacles[int(rng.integers(0, len(obstacles)))]
    z = float(np.clip(obstacle.min_corner[2] + 1.0, 0.5, env.config.max_altitude - 0.5))
    angle = float(rng.uniform(0.0, 2.0 * math.pi))
    attacker_pos = _point_near_inflated_obstacle(obstacle, clearance=clearance, distance=obstacle_distance_m, direction_angle=angle, z=z)
    inward_dir = -attacker_pos[:2] + (obstacle.min_corner[:2] + obstacle.max_corner[:2]) / 2.0
    inward_norm = np.linalg.norm(inward_dir)
    inward_unit = (inward_dir / inward_norm) if inward_norm > 1e-9 else np.array([1.0, 0.0])
    attacker_vel = np.array([inward_unit[0], inward_unit[1], 0.0]) * (env.config.v_e * 0.5)

    defender_angle = float(rng.uniform(0.0, 2.0 * math.pi))
    defender_offset = np.array([math.cos(defender_angle), math.sin(defender_angle), 0.0]) * defender_distance_m
    defender_pos = attacker_pos + defender_offset
    defender_pos[2] = float(np.clip(defender_pos[2], 0.5, env.config.max_altitude - 0.5))
    closing_dir = attacker_pos - defender_pos
    closing_norm = np.linalg.norm(closing_dir)
    closing_unit = (closing_dir / closing_norm) if closing_norm > 1e-9 else np.array([1.0, 0.0, 0.0])
    defender_vel = closing_unit * (env.config.v_d * 0.3)  # positive closing speed, by construction

    env._enemy_pos = attacker_pos.astype(np.float32)
    env._enemy_vel = attacker_vel.astype(np.float32)
    env._defender_pos = defender_pos.astype(np.float32)
    env._defender_vel = defender_vel.astype(np.float32)
    # Physical-validity guard: the constructed point is checked only against the TARGETED obstacle's inflated
    # clearance above -- in a dense C2 field it can still land inside (or too close to) a DIFFERENT obstacle.
    # Reject (caller tries the next seed) rather than accept a scenario that starts in an already-collided state.
    if env._obstacle_layout.contains_point(attacker_pos) or env._obstacle_layout.contains_point(defender_pos):
        return False
    if env._obstacle_layout.min_clearance(attacker_pos) < clearance or env._obstacle_layout.min_clearance(defender_pos) < clearance:
        return False
    if not (0.0 <= attacker_pos[2] <= env.config.max_altitude) or not (0.0 <= defender_pos[2] <= env.config.max_altitude):
        return False
    # Dynamic-viability guard (journal_cbf_attacker, obstacle_cbf_v2 / Prompt 3 precedent): a manually constructed
    # state bypasses the normal viability-conditioned spawn resampling entirely -- it must be checked against the
    # SAME stopping-margin criterion directly, against the FULL obstacle set (not just the targeted obstacle),
    # or a rare spawn-time physics infeasibility (the Prompt-2 failure mode) can produce a genuine collision that
    # the in-flight CBF cannot avoid regardless of alpha/clearance tuning.
    from uav_defend.attackers.cbf_obstacle import CBF_DT_SUB, initial_state_dynamically_viable

    viability = initial_state_dynamically_viable(
        attacker_pos, attacker_vel, obstacles, clearance=clearance,
        max_accel=env._cbf_controller.limits.max_accel, dt_sub=CBF_DT_SUB,
    )
    if not viability["viable"]:
        return False
    env._update_detection()  # refreshes enemy_detected/enemy_measurement/e_hat for the manually placed positions
    return True


def run_stress_campaign(n_scenarios: int = 200, *, seed_base: int = STRESS_SEED_BASE, defender: str = "greedy",
                         horizon_steps: int = 60) -> list[dict]:
    """Runs the 200-case joint obstacle+defender stress set (brief section 19) under the FULL A1-CBF variant
    (`defender_cbf_enabled=True`) with `defender_standby_until_detection=False` (the stress set targets the joint
    CBF mechanism directly, not canonical pre-detection timing) and the REAL canonical Greedy policy thereafter."""
    cfg, estimator = defender_config(defender)
    cfg = dataclasses.replace(cfg, defender_standby_until_detection=False)
    episodes: list[dict] = []
    seed = seed_base
    constructed = 0
    while constructed < n_scenarios:
        env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=True)
        if not build_stress_scenario(env, seed):
            seed += 1
            continue
        policy = VC.make_policy(defender, cfg)
        if policy is not None:
            policy.reset()
        record = _run_from_current_state(env, policy, estimator, max_steps=horizon_steps, scenario_seed=seed)
        episodes.append(record)
        constructed += 1
        seed += 1
    return episodes


def _run_from_current_state(env: SoldierEnv, policy, estimator: str, *, max_steps: int, scenario_seed: int) -> dict:
    """Like `run_episode`, but starts from the env's CURRENT (already-manipulated) state rather than calling
    `reset()` -- used by the stress-set runner, whose scenario construction happens AFTER `reset()`."""
    start_pos = np.asarray(env._enemy_pos, dtype=np.float64).copy()
    asset0 = np.asarray(env._soldier_pos, dtype=np.float64).copy()
    prev_pos = start_pos.copy()
    path_length = 0.0
    min_defender_sep = float("inf")
    joint_constraint_steps = 0
    min_h_d = float("inf")
    qp_statuses: list[str] = []
    intervention_sources: list[str] = []
    finite = True
    done, info_final, step = False, env._get_info(), 0
    while not done and step < max_steps:
        if policy is None:
            action = np.zeros(3, dtype=np.float32)
        else:
            action = np.asarray(
                policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator)), dtype=np.float32,
            ).reshape(3)
        obs, _r, te, tr, info = env.step(action)
        step += 1
        pos = np.asarray(info["enemy_pos"], dtype=np.float64)
        path_length += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos
        dist_de = float(np.linalg.norm(np.asarray(info["defender_pos"], dtype=np.float64) - pos))
        min_defender_sep = min(min_defender_sep, dist_de)
        if info.get("simultaneous_obstacle_defender_constraints"):
            joint_constraint_steps += 1
        if info.get("defender_observable"):
            min_h_d = min(min_h_d, float(info.get("defender_barrier_h", float("inf"))))
        qp_statuses.append(str(info["cbf_qp_status"]))
        intervention_sources.append(str(info.get("intervention_source", "none")))
        finite = finite and bool(np.isfinite(pos).all())
        done = te or tr
        info_final = info
    n_steps = step
    outcome = info_final["outcome"] if done else "stress_horizon_reached"
    hard_steps = sum(1 for s in qp_statuses if s == "hard")
    soft_steps = sum(1 for s in qp_statuses if s == "soft_fallback")
    emergency_steps = sum(1 for s in qp_statuses if s == "emergency_brake")
    return {
        "scenario_seed": int(scenario_seed), "outcome": outcome, "scenario_generation_failure": False,
        "n_steps": n_steps, "finite": finite, "path_length_m": path_length,
        "asset_capture": outcome == "soldier_caught",
        "defender_interception": outcome in ("intercepted", "unsafe_intercept"),
        "obstacle_collision": outcome in ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"),
        "min_defender_separation_m": min_defender_sep, "min_h_d": min_h_d,
        "joint_constraint_steps": joint_constraint_steps, "joint_constraint_fraction": (joint_constraint_steps / n_steps) if n_steps else 0.0,
        "hard_step_fraction": (hard_steps / n_steps) if n_steps else 1.0,
        "soft_step_fraction": (soft_steps / n_steps) if n_steps else 0.0,
        "emergency_step_fraction": (emergency_steps / n_steps) if n_steps else 0.0,
        "qp_hard_or_soft_success": bool(emergency_steps == 0),
        "intervention_source_counts": {k: int(intervention_sources.count(k)) for k in ("none", "obstacle", "defender", "joint")},
    }


def stress_campaign_summary(episodes: list[dict]) -> dict:
    n = len(episodes)
    return {
        "n_scenarios": n,
        "obstacle_collisions": int(sum(e["obstacle_collision"] for e in episodes)),
        "numerical_failures": int(sum(not e["finite"] for e in episodes)),
        "simultaneous_constraint_rate": float(np.mean([e["joint_constraint_fraction"] > 0 for e in episodes])) if n else 0.0,
        "soft_qp_episodes": int(sum(e["soft_step_fraction"] > 0 for e in episodes)),
        "emergency_brake_episodes": int(sum(e["emergency_step_fraction"] > 0 for e in episodes)),
        "defender_interceptions": int(sum(e["defender_interception"] for e in episodes)),
        "asset_captures": int(sum(e["asset_capture"] for e in episodes)),
        "qp_hard_or_soft_success_fraction": float(np.mean([e["qp_hard_or_soft_success"] for e in episodes])) if n else 1.0,
    }


# ----------------------------------------------------------------------------- generalization diagnostic (brief section 20)
def run_generalization_diagnostic(results_root: Path, *, defenders: tuple[str, ...] = GENERALIZATION_DEFENDERS,
                                   namespace: tuple[int, int] = GENERALIZATION_SEEDS, chunk_size: int = CHUNK_SIZE) -> dict:
    """200 matched obstacle-only/full episodes PER additional defender family (brief section 20), with NO
    controller-parameter tuning between defenders (reuses the exact same `defender_cbf_enabled` A1-CBF code path).
    Returns `{defender_name: {"obstacle_only": metrics, "full": metrics, "effect": effect_gate_status}}`."""
    results: dict = {}
    for name in defenders:
        oo, fu = run_matched_campaign(results_root, defender=name, namespace=namespace, chunk_size=chunk_size, tag=f"matched_{name}")
        results[name] = {
            "obstacle_only_metrics": metrics(oo), "full_metrics": metrics(fu),
            "effect": effect_gate_status(oo, fu),
            "obstacle_only_primary_gate": primary_gate_status(metrics(oo)),
            "full_primary_gate": primary_gate_status(metrics(fu)),
        }
    return results


def run_matched_pair(defender: str, seed: int) -> tuple[dict, dict]:
    """Runs the SAME scenario (identical seed -> identical common random numbers: obstacle layout, asset path,
    attacker viability-conditioned spawn, defender spawn/noise) under both A1-CBF variants (brief section 13):
    obstacle-only (`defender_cbf_enabled=False`) and full (`defender_cbf_enabled=True`). Two fresh environments are
    used (not a deepcopy) since both are reset from the SAME seed, which is sufficient for common random numbers
    given the env's RNG is seeded entirely from `reset(seed=...)`."""
    cfg, estimator = defender_config(defender)
    env_obstacle_only = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=False)
    env_full = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=True)
    policy_obstacle_only = VC.make_policy(defender, cfg)
    policy_full = VC.make_policy(defender, cfg)
    obstacle_only = run_episode(env_obstacle_only, policy_obstacle_only, estimator, seed)
    full = run_episode(env_full, policy_full, estimator, seed)
    return obstacle_only, full


# ----------------------------------------------------------------------------- chunked, restart-safe campaign
def protocol_sha256(defender: str = "greedy") -> str:
    return hashlib.sha256(json.dumps({
        "config_sha256": config_sha256(defender), "primary_gate": PRIMARY_GATE, "effect_gate": EFFECT_GATE,
        "detection_range": DEFENDER_DETECTION_RANGE, "safety_radius": DEFENDER_SAFETY_RADIUS,
    }, sort_keys=True, default=str).encode()).hexdigest()


def run_matched_campaign(results_root: Path, *, defender: str = "greedy", namespace: tuple[int, int] = PRIMARY_SEEDS,
                          chunk_size: int = CHUNK_SIZE, tag: str | None = None) -> tuple[list[dict], list[dict]]:
    """Chunked/restart-safe matched campaign (brief section 14): for each chunk, if a verified chunk file already
    exists (matching protocol hash + content hash), it is reused unchanged; otherwise it is computed and
    atomically written. Returns `(obstacle_only_episodes, full_episodes)`."""
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    psha = protocol_sha256(defender)
    tag = tag or f"matched_{defender}"
    all_obstacle_only: list[dict] = []
    all_full: list[dict] = []
    for i in range(num_chunks(namespace, chunk_size)):
        path = results_root / f"{tag}_chunk_{i:03d}.json"
        payload = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol_sha256") != psha or payload.get("tag") != tag:
                payload = None
            elif _content_sha256([payload["obstacle_only"], payload["full"]]) != payload.get("content_sha256"):
                payload = None
        except (OSError, ValueError, KeyError, TypeError):
            payload = None
        if payload is None:
            seeds = chunk_seeds(namespace, i, chunk_size)
            obstacle_only_eps, full_eps = [], []
            for s in seeds:
                oo, fu = run_matched_pair(defender, s)
                obstacle_only_eps.append(oo)
                full_eps.append(fu)
            payload = {
                "schema": "a1_cbf_defender_cbf_matched_chunk/1", "chunk": i, "tag": tag, "seeds": seeds,
                "obstacle_only": obstacle_only_eps, "full": full_eps,
                "content_sha256": _content_sha256([obstacle_only_eps, full_eps]), "protocol_sha256": psha,
            }
            base.atomic_write_json(path, payload)
        all_obstacle_only.extend(payload["obstacle_only"])
        all_full.extend(payload["full"])
    return all_obstacle_only, all_full


# ----------------------------------------------------------------------------- metrics
def mission_metrics(episodes: list[dict]) -> dict:
    valid = [e for e in episodes if not e["scenario_generation_failure"]]
    n = len(valid)
    n_all = len(episodes)
    return {
        "episodes_total": n_all, "episodes_valid": n, "scenario_generation_failures": n_all - n,
        "asset_capture": float(np.mean([e["asset_capture"] for e in valid])) if n else 0.0,
        "defender_interception": float(np.mean([e["defender_interception"] for e in valid])) if n else 0.0,
        "obstacle_collision": int(sum(e["obstacle_collision"] for e in valid)),
        "obstacle_collision_rate": float(np.mean([e["obstacle_collision"] for e in valid])) if n else 0.0,
        "endurance_exhausted": float(np.mean([e["endurance_exhausted"] for e in valid])) if n else 0.0,
        "timeout": float(np.mean([e["timeout"] for e in valid])) if n else 0.0,
        "mean_duration_s": float(np.mean([e["duration_s"] for e in valid])) if n else None,
        "nonfinite": int(sum(not e["finite"] for e in valid)),
    }


def defender_interaction_metrics(episodes: list[dict]) -> dict:
    valid = [e for e in episodes if not e["scenario_generation_failure"]]
    n = len(valid)
    seps = [e["min_defender_separation_m"] for e in valid if np.isfinite(e["min_defender_separation_m"])]
    first_obs = [e["first_defender_observation_step"] for e in valid if e["first_defender_observation_step"] is not None]
    min_hs = [e["min_h_d"] for e in valid if np.isfinite(e["min_h_d"])]
    return {
        "min_separation_median_m": float(np.median(seps)) if seps else None,
        "min_separation_mean_m": float(np.mean(seps)) if seps else None,
        "first_observation_time_mean_s": (float(np.mean(first_obs)) * 0.5) if first_obs else None,
        "defender_visible_fraction_mean": float(np.mean([e["defender_visible_fraction"] for e in valid])) if n else 0.0,
        "defender_cbf_activation_rate_mean": float(np.mean([e["defender_cbf_activation_rate"] for e in valid])) if n else 0.0,
        "mean_defender_constraint_residual": float(np.mean([e["mean_defender_constraint_residual"] for e in valid])) if n else 0.0,
        "min_h_d": float(np.min(min_hs)) if min_hs else None,
        "unsafe_radius_steps_total": int(sum(e["unsafe_radius_steps"] for e in valid)),
        "interception_radius_steps_total": int(sum(e["interception_radius_steps"] for e in valid)),
    }


def navigation_metrics(episodes: list[dict]) -> dict:
    valid = [e for e in episodes if not e["scenario_generation_failure"]]
    n = len(valid)
    pe = [e["path_excess"] for e in valid if np.isfinite(e["path_excess"])]
    return {
        "mean_path_length_m": float(np.mean([e["path_length_m"] for e in valid])) if n else None,
        "mean_path_excess": float(np.mean(pe)) if pe else None,
        "mean_duration_s": float(np.mean([e["duration_s"] for e in valid])) if n else None,
        "total_route_replans": int(sum(e["route_replan_count"] for e in valid)),
        "total_route_errors": int(sum(e["route_errors"] for e in valid)),
    }


def joint_safety_metrics(episodes: list[dict]) -> dict:
    valid = [e for e in episodes if not e["scenario_generation_failure"]]
    n = len(valid)
    total_steps = sum(e["n_steps"] for e in valid)
    source_totals = {k: sum(e["intervention_source_counts"][k] for e in valid) for k in ("none", "obstacle", "defender", "joint")}
    return {
        "mean_hard_step_fraction": float(np.mean([e["hard_step_fraction"] for e in valid])) if n else None,
        "mean_soft_step_fraction": float(np.mean([e["soft_step_fraction"] for e in valid])) if n else None,
        "mean_emergency_step_fraction": float(np.mean([e["emergency_step_fraction"] for e in valid])) if n else None,
        "qp_hard_or_soft_success_fraction": float(np.mean([e["qp_hard_or_soft_success"] for e in valid])) if n else 1.0,
        "joint_constraint_fraction_mean": float(np.mean([e["joint_constraint_fraction"] for e in valid])) if n else 0.0,
        "total_steps": int(total_steps), "intervention_source_step_totals": source_totals,
    }


def metrics(episodes: list[dict]) -> dict:
    return {
        "mission": mission_metrics(episodes), "defender_interaction": defender_interaction_metrics(episodes),
        "navigation": navigation_metrics(episodes), "joint_safety": joint_safety_metrics(episodes),
    }


def primary_gate_status(m: dict) -> dict:
    mission, safety = m["mission"], m["joint_safety"]
    checks = {
        "obstacle_collision": mission["obstacle_collision"] <= PRIMARY_GATE["obstacle_collision_max"],
        "numerical_failures": mission["nonfinite"] <= PRIMARY_GATE["numerical_failures_max"],
        "route_errors": m["navigation"]["total_route_errors"] <= PRIMARY_GATE["route_errors_max"],
        "scenario_generation_failures": mission["scenario_generation_failures"] <= PRIMARY_GATE["scenario_generation_failures_max"],
        "qp_completion": safety["qp_hard_or_soft_success_fraction"] >= PRIMARY_GATE["qp_completion_min"],
        "soft_fallback_fraction": (safety["mean_soft_step_fraction"] or 0.0) <= PRIMARY_GATE["soft_fallback_step_fraction_max"],
        "emergency_brake_fraction": (safety["mean_emergency_step_fraction"] or 0.0) <= PRIMARY_GATE["emergency_brake_step_fraction_max"],
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


# ----------------------------------------------------------------------------- effect gate (brief section 17)
def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, *, n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05) -> dict:
    """Paired bootstrap 95% CI for `mean(b) - mean(a)` (e.g. `P(intercept_full) - P(intercept_obstacle_only)`),
    resampling MATCHED (a[i], b[i]) pairs together (brief section 17: 'Paired bootstrap')."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    n = a.shape[0]
    rng = np.random.default_rng(seed)
    diffs = b - a
    observed = float(diffs.mean())
    boot = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = diffs[idx].mean()
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return {"observed_diff": observed, "ci_lo": lo, "ci_hi": hi, "n": int(n), "n_boot": int(n_boot)}


def effect_gate_status(obstacle_only: list[dict], full: list[dict]) -> dict:
    """Evaluates brief section 17's frozen defender-avoidance effect gate on MATCHED (same-seed) episode pairs."""
    oo = {e["scenario_seed"]: e for e in obstacle_only if not e["scenario_generation_failure"]}
    fu = {e["scenario_seed"]: e for e in full if not e["scenario_generation_failure"]}
    common = sorted(set(oo) & set(fu))
    n = len(common)
    oo_intercept = np.array([float(oo[s]["defender_interception"]) for s in common])
    fu_intercept = np.array([float(fu[s]["defender_interception"]) for s in common])
    oo_capture = np.array([float(oo[s]["asset_capture"]) for s in common])
    fu_capture = np.array([float(fu[s]["asset_capture"]) for s in common])

    interception_oo = float(oo_intercept.mean()) if n else 0.0
    interception_fu = float(fu_intercept.mean()) if n else 0.0
    interception_reduction_pp = (interception_oo - interception_fu) * 100.0

    capture_oo = float(oo_capture.mean()) if n else 0.0
    capture_fu = float(fu_capture.mean()) if n else 0.0
    capture_change_pp = (capture_fu - capture_oo) * 100.0

    # Separation increase among matched NON-immediate-terminal scenarios (brief section 17: excludes scenarios
    # that terminate almost immediately, where "minimum separation" is not a meaningful evasion metric) --
    # operationalized as episodes lasting at least 4 steps (2.0 s) in BOTH variants.
    non_immediate = [s for s in common if oo[s]["n_steps"] >= 4 and fu[s]["n_steps"] >= 4]
    sep_oo = np.array([oo[s]["min_defender_separation_m"] for s in non_immediate if np.isfinite(oo[s]["min_defender_separation_m"])])
    sep_fu = np.array([fu[s]["min_defender_separation_m"] for s in non_immediate if np.isfinite(fu[s]["min_defender_separation_m"])])
    median_sep_oo = float(np.median(sep_oo)) if len(sep_oo) else None
    median_sep_fu = float(np.median(sep_fu)) if len(sep_fu) else None
    separation_increase_m = (median_sep_fu - median_sep_oo) if (median_sep_oo is not None and median_sep_fu is not None) else None

    # Beneficial direction: full < obstacle-only, i.e. (fu - oo) < 0; equivalently the CI for (oo - fu) must
    # exclude 0 on the positive (beneficial) side.
    ci_beneficial = paired_bootstrap_ci(fu_intercept, oo_intercept, seed=1) if n else {"observed_diff": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0, "n_boot": 0}

    checks = {
        "interception_reduction": interception_reduction_pp >= EFFECT_GATE["interception_reduction_min_pp"],
        "bootstrap_ci_excludes_zero_beneficial": (ci_beneficial["ci_lo"] > 0.0) if n else False,
        "separation_increase": (separation_increase_m is not None and separation_increase_m >= EFFECT_GATE["min_separation_increase_min_m"]),
        "capture_tradeoff": capture_change_pp >= -EFFECT_GATE["capture_drop_max_pp"],
    }
    return {
        "n_matched": n, "interception_obstacle_only": interception_oo, "interception_full": interception_fu,
        "interception_reduction_pp": interception_reduction_pp, "capture_obstacle_only": capture_oo,
        "capture_full": capture_fu, "capture_change_pp": capture_change_pp,
        "median_separation_obstacle_only_m": median_sep_oo, "median_separation_full_m": median_sep_fu,
        "separation_increase_m": separation_increase_m, "n_non_immediate": len(non_immediate),
        "bootstrap_ci_beneficial_direction": ci_beneficial,
        "checks": checks, "passed": bool(all(checks.values())),
    }
