"""Monte Carlo acceptance experiment for the viability-conditioned A1-CBF clutter benchmark (journal_cbf_attacker,
obstacle_cbf_v2 / Prompt 3): canonical C2 obstacle generation + dynamically-feasible attacker initialization +
corrected route-status handling (route_ok / route_temporarily_unavailable / route_error). Reuses the FROZEN
Prompt-2 CBF-QP/route-planner parameters verbatim (brief section 2) -- nothing about the controller itself is
tuned in this stage.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from uav_defend.attackers.cbf_attacker import CBFAttackerLimits
from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv

from experiments import journal_v2_validation_core as VC
from experiments import journal_v2_a1_r2 as R
from experiments.journal_v2_a1_cur4a import closest_approaches
from experiments.journal_cbf_attacker_foundation import (
    PARENT_COMMIT as FOUNDATION_PARENT_COMMIT,
    _load_legacy_module,
    run_legacy_episode,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = "docs/journal_cbf_attacker/obstacle_cbf_v2/"
RESULTS_ROOT = R.V2_RESULTS_DIR / "journal_cbf_attacker" / "obstacle_cbf_v2"

# ----------------------------------------------------------------------------- pre-registered namespaces (frozen before any validation)
CLUTTER_V2_SEEDS = (530_000, 530_999)      # fresh 1,000-seed block for the revised Clutter Monte Carlo
LEGACY_NOOP_SEEDS = (540_000, 540_999)     # fresh block, far from every other namespace
C0_IDENTITY_SEEDS = (500_000, 500_999)     # reuses the Prompt-1 foundation block (same config, pair-identified)
STRESS_SEED_BASE = 550_000                 # base for synthetic near-viability-boundary stress states
CHUNK_SIZE = 100
PARENT_COMMIT = FOUNDATION_PARENT_COMMIT

# Frozen acceptance gate (brief section 13; do not weaken after execution).
GATE = {
    "capture_min": 0.95, "obstacle_collision_max": 0, "timeout_max": 0.01, "exhaustion_max": 0.04,
    "orbit_episodes_max": 0, "repeated_flyby_max": 0.02, "path_excess_5x_max": 0.01, "route_errors_max": 0,
    "scenario_generation_failures_max": 0, "nonfinite_max": 0, "qp_hard_or_soft_success_min": 1.0,
    "emergency_brake_step_fraction_max": 0.01, "soft_fallback_step_fraction_max": 0.01,
}
REPEATED_FLYBY_OVERSHOOTS = 2
ORBIT_REVOLUTION_THRESHOLD = 1.0


def seeds_of(rng: tuple[int, int]) -> range:
    return range(rng[0], rng[1] + 1)


def clutter_config() -> EnvConfig:
    """Canonical C2 V2 config, UNCHANGED (same obstacle count/positions/sizes/randomization, same moving asset,
    same endurance, same world boundaries as Prompt 2) with `intercept_radius=1e-6` (established no-defender-
    interaction mechanism) and `dynamic_obstacles_enabled=False` (existing switch; static geometry bit-identical)."""
    cfg, _estimator = VC.build_config("C2", "V2", "passive")
    return dataclasses.replace(cfg, intercept_radius=1e-6, dynamic_obstacles_enabled=False)


def config_sha256() -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(clutter_config()), sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------- episode runner
def run_episode(env: SoldierEnv, seed: int) -> dict:
    """Runs one C2 scenario with viability-conditioned initialization. If scenario generation fails (100 resample
    attempts exhausted), the episode is NOT stepped further -- it is recorded as `scenario_generation_failure` and
    excluded from rate metrics by the caller (brief section 5: 'STOP that seed')."""
    obs, info = env.reset(seed=seed)
    if info["scenario_generation_failure"]:
        return {
            "scenario_seed": int(seed), "outcome": "scenario_generation_failure", "scenario_generation_failure": True,
            "viability_resample_count": int(info["viability_resample_count"]), "initial_viability_margin": float(info["initial_viability_margin"]),
        }

    start_pos = np.asarray(info["enemy_pos"], dtype=np.float64).copy()
    asset0 = np.asarray(info["soldier_pos"], dtype=np.float64).copy()
    initial_range = float(np.linalg.norm(asset0 - start_pos))
    att, ast, speeds, accels, ranges, finite = [start_pos], [asset0], [], [], [initial_range], True
    prev_pos = start_pos.copy()
    path_length = 0.0
    min_physical_clearance = float("inf")
    min_inflated_clearance = float("inf")
    cbf_h_min = float("inf")
    intervention_norms, active_counts, qp_statuses, qp_successes = [], [], [], []
    route_statuses, hover_steps = [], 0
    done, info_final = False, info
    while not done:
        obs, _r, te, tr, info = env.step(np.zeros(3, dtype=np.float32))
        pos = np.asarray(info["enemy_pos"], dtype=np.float64)
        asset_pos = np.asarray(info["soldier_pos"], dtype=np.float64)
        path_length += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos
        att.append(pos)
        ast.append(asset_pos)
        d = float(np.linalg.norm(asset_pos - pos))
        ranges.append(d)
        speeds.append(float(info["enemy_speed"]))
        accels.append(float(info["enemy_accel"]))
        finite = finite and bool(np.isfinite(pos).all() and np.isfinite(d) and np.isfinite(info["enemy_speed"]))
        min_physical_clearance = min(min_physical_clearance, float(info["min_physical_obstacle_clearance"]))
        min_inflated_clearance = min(min_inflated_clearance, float(info["min_inflated_obstacle_clearance"]))
        cbf_h_min = min(cbf_h_min, float(info["cbf_h_min"]))
        intervention_norms.append(float(info["cbf_intervention_norm"]))
        active_counts.append(int(info["active_obstacle_count"]))
        qp_statuses.append(str(info["cbf_qp_status"]))
        qp_successes.append(bool(info["qp_success"]))
        route_statuses.append(str(info["route_status"]))
        if info["route_hover_fallback"]:
            hover_steps += 1
        done = te or tr
        info_final = info
    ranges_arr = np.array(ranges, dtype=np.float64)
    overshoots = int(VC.count_overshoots(ranges_arr, env.config.dt))
    closest_n, misses = closest_approaches(ranges_arr)
    orbit_max, orbit_total = VC.orbit_revolutions(np.array(att), np.array(ast))
    chord = float(np.linalg.norm(start_pos - asset0))
    outcome = info_final["outcome"]
    speeds_arr = np.array(speeds, dtype=np.float64) if speeds else np.zeros(1)
    accels_arr = np.array(accels, dtype=np.float64) if accels else np.zeros(1)
    n_steps = len(speeds)
    hard_steps = sum(1 for s in qp_statuses if s == "hard")
    soft_steps = sum(1 for s in qp_statuses if s == "soft_fallback")
    emergency_steps = sum(1 for s in qp_statuses if s == "emergency_brake")
    active_steps = sum(1 for c in active_counts if c > 0)
    intervention_steps = sum(1 for v in intervention_norms if v > 1e-6)
    unavailable_steps = sum(1 for s in route_statuses if s == "route_temporarily_unavailable")
    error_steps = sum(1 for s in route_statuses if s == "route_error")
    return {
        "scenario_seed": int(seed), "outcome": outcome, "scenario_generation_failure": False,
        "viability_resample_count": int(info["viability_resample_count"]), "initial_viability_margin": float(info["initial_viability_margin"]),
        "steps": n_steps, "duration_s": n_steps * env.config.dt,
        "path_length_m": path_length, "path_excess": (path_length / chord) if chord > 1e-9 else float("nan"),
        "initial_range_m": initial_range, "terminal_distance_m": float(ranges_arr[-1]), "min_distance_m": float(ranges_arr.min()),
        "overshoots": overshoots, "closest_approach_count": int(closest_n), "miss_distances_m": [float(m) for m in misses],
        "first_pass": bool(outcome == "soldier_caught" and overshoots == 0), "repeated_flyby": bool(overshoots >= REPEATED_FLYBY_OVERSHOOTS),
        "orbit_revolutions_max": float(orbit_max), "orbit": bool(orbit_max >= ORBIT_REVOLUTION_THRESHOLD),
        "mean_speed_mps": float(speeds_arr.mean()), "max_speed_mps": float(speeds_arr.max()), "min_speed_mps": float(speeds_arr.min()),
        "max_accel_mps2": float(accels_arr.max()), "finite": finite,
        "enemy_endurance_exhausted": outcome == "enemy_endurance_exhausted",
        "obstacle_collision": outcome in ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"),
        "min_physical_obstacle_clearance": min_physical_clearance, "min_inflated_obstacle_clearance": min_inflated_clearance,
        "cbf_h_min": cbf_h_min,
        "route_replan_count": int(info_final.get("route_replan_count", 0)),
        "route_temporarily_unavailable_steps": unavailable_steps, "route_error_steps": error_steps,
        "route_errors": int(info_final.get("route_errors", 0)),
        "route_longest_unavailable_streak": int(info_final.get("route_longest_unavailable_streak", 0)),
        "route_hover_fallback_count": int(info_final.get("route_hover_fallback_count", 0)),
        "hover_steps": hover_steps,
        "emergency_brake_count": int(info_final.get("emergency_brake_count", 0)),
        "active_steps": active_steps, "n_steps": n_steps,
        "mean_intervention_norm": float(np.mean(intervention_norms)) if intervention_norms else 0.0,
        "max_intervention_norm": float(np.max(intervention_norms)) if intervention_norms else 0.0,
        "intervention_step_fraction": (intervention_steps / n_steps) if n_steps else 0.0,
        "hard_step_fraction_of_active": (hard_steps / active_steps) if active_steps else None,
        "soft_step_fraction": (soft_steps / n_steps) if n_steps else 0.0,
        "emergency_step_fraction": (emergency_steps / n_steps) if n_steps else 0.0,
        "qp_success_fraction": (sum(qp_successes) / n_steps) if n_steps else 1.0,
        "qp_hard_or_soft_success": bool(emergency_steps == 0),
    }


def evaluate(seeds, *, enemy_control_mode: str, viability_conditioned_init: bool = True) -> list[dict]:
    env = SoldierEnv(clutter_config(), enemy_control_mode=enemy_control_mode, viability_conditioned_init=viability_conditioned_init)
    return [run_episode(env, s) for s in seeds]


def metrics(episodes: list[dict]) -> dict:
    valid = [e for e in episodes if not e["scenario_generation_failure"]]
    n_all = len(episodes)
    n = len(valid)
    scenario_generation_failures = n_all - n
    outcome = np.array([e["outcome"] for e in valid])
    pe = np.array([e["path_excess"] for e in valid], dtype=np.float64)
    cap = outcome == "soldier_caught"
    first_pass_of_captures = [e["first_pass"] for e in valid if e["outcome"] == "soldier_caught"]
    misses = [m for e in valid for m in e["miss_distances_m"]]
    hard_fracs = [e["hard_step_fraction_of_active"] for e in valid if e["hard_step_fraction_of_active"] is not None]
    resample_counts = [e["viability_resample_count"] for e in episodes]
    return {
        "episodes_total": n_all, "episodes_valid": n, "scenario_generation_failures": scenario_generation_failures,
        "capture": float(cap.mean()) if n else 0.0, "exhaustion": float((outcome == "enemy_endurance_exhausted").mean()) if n else 0.0,
        "timeout": float((outcome == "timeout").mean()) if n else 0.0,
        "obstacle_collision": int(sum(e["obstacle_collision"] for e in valid)),
        "obstacle_collision_rate": float(np.mean([e["obstacle_collision"] for e in valid])) if n else 0.0,
        "outcomes": {k: int((outcome == k).sum()) for k in sorted(set(outcome.tolist()))} if n else {},
        "first_pass_capture": float(np.mean([e["first_pass"] for e in valid])) if n else 0.0,
        "first_pass_fraction_of_captures": (float(np.mean(first_pass_of_captures)) if first_pass_of_captures else None),
        "repeated_flyby_episodes": int(sum(e["repeated_flyby"] for e in valid)),
        "repeated_flyby_fraction": float(np.mean([e["repeated_flyby"] for e in valid])) if n else 0.0,
        "orbit_episodes": int(sum(e["orbit"] for e in valid)), "nonfinite": int(sum(not e["finite"] for e in valid)),
        "median_path_excess": float(np.nanmedian(pe)) if n else None, "mean_path_excess": float(np.nanmean(pe)) if n else None,
        "path_excess_5x_episodes": int(sum(e["path_excess"] > 5.0 for e in valid if np.isfinite(e["path_excess"]))),
        "path_excess_5x_fraction": float(np.mean([e["path_excess"] > 5.0 if np.isfinite(e["path_excess"]) else False for e in valid])) if n else 0.0,
        "mean_duration_s": float(np.mean([e["duration_s"] for e in valid])) if n else None,
        "mean_path_length_m": float(np.mean([e["path_length_m"] for e in valid])) if n else None,
        "mean_closest_approaches": float(np.mean([e["closest_approach_count"] for e in valid])) if n else None,
        "miss_distance_median_m": (float(np.median(misses)) if misses else None), "miss_distance_count": len(misses),
        "mean_speed_mps": float(np.mean([e["mean_speed_mps"] for e in valid])) if n else None,
        "max_speed_mps": float(np.max([e["max_speed_mps"] for e in valid])) if n else None,
        "max_accel_mps2": float(np.max([e["max_accel_mps2"] for e in valid])) if n else None,
        "min_physical_obstacle_clearance": float(np.min([e["min_physical_obstacle_clearance"] for e in valid])) if n else None,
        "min_inflated_obstacle_clearance": float(np.min([e["min_inflated_obstacle_clearance"] for e in valid])) if n else None,
        "cbf_h_min": float(np.min([e["cbf_h_min"] for e in valid])) if n else None,
        "total_route_replans": int(sum(e["route_replan_count"] for e in valid)),
        "mean_route_replans": float(np.mean([e["route_replan_count"] for e in valid])) if n else None,
        "total_route_errors": int(sum(e["route_errors"] for e in valid)),
        "route_error_episodes": int(sum(e["route_errors"] > 0 for e in valid)),
        "episodes_with_temporarily_unavailable_route": int(sum(e["route_temporarily_unavailable_steps"] > 0 for e in valid)),
        "total_temporarily_unavailable_steps": int(sum(e["route_temporarily_unavailable_steps"] for e in valid)),
        "longest_unavailable_streak": int(max((e["route_longest_unavailable_streak"] for e in valid), default=0)),
        "total_hover_fallback_events": int(sum(e["route_hover_fallback_count"] for e in valid)),
        "hover_steps_total": int(sum(e["hover_steps"] for e in valid)),
        "total_emergency_brake_events": int(sum(e["emergency_brake_count"] for e in valid)),
        "mean_intervention_step_fraction": float(np.mean([e["intervention_step_fraction"] for e in valid])) if n else None,
        "mean_intervention_norm": float(np.mean([e["mean_intervention_norm"] for e in valid])) if n else None,
        "max_intervention_norm": float(np.max([e["max_intervention_norm"] for e in valid])) if n else None,
        "mean_hard_step_fraction_of_active": (float(np.mean(hard_fracs)) if hard_fracs else None),
        "mean_soft_step_fraction": float(np.mean([e["soft_step_fraction"] for e in valid])) if n else None,
        "mean_emergency_step_fraction": float(np.mean([e["emergency_step_fraction"] for e in valid])) if n else None,
        "qp_hard_or_soft_success_fraction": float(np.mean([e["qp_hard_or_soft_success"] for e in valid])) if n else 1.0,
        "viability_resample_rate": float(np.mean([c > 0 for c in resample_counts])), "mean_viability_resamples": float(np.mean(resample_counts)),
        "max_viability_resamples": int(np.max(resample_counts)) if resample_counts else 0,
    }


def gate_status(m: dict) -> dict:
    checks = {
        "capture": m["capture"] >= GATE["capture_min"],
        "obstacle_collision": m["obstacle_collision"] <= GATE["obstacle_collision_max"],
        "timeout": m["timeout"] <= GATE["timeout_max"], "exhaustion": m["exhaustion"] <= GATE["exhaustion_max"],
        "orbit": m["orbit_episodes"] <= GATE["orbit_episodes_max"], "repeated_flyby": m["repeated_flyby_fraction"] <= GATE["repeated_flyby_max"],
        "path_excess_5x": m["path_excess_5x_fraction"] <= GATE["path_excess_5x_max"],
        "route_errors": m["total_route_errors"] <= GATE["route_errors_max"],
        "scenario_generation_failures": m["scenario_generation_failures"] <= GATE["scenario_generation_failures_max"],
        "nonfinite": m["nonfinite"] <= GATE["nonfinite_max"],
        "qp_hard_or_soft_success": m["qp_hard_or_soft_success_fraction"] >= GATE["qp_hard_or_soft_success_min"],
        "emergency_brake_fraction": (m["mean_emergency_step_fraction"] or 0.0) <= GATE["emergency_brake_step_fraction_max"],
        "soft_fallback_fraction": (m["mean_soft_step_fraction"] or 0.0) <= GATE["soft_fallback_step_fraction_max"],
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def constraint_verification(episodes: list[dict], *, limits: CBFAttackerLimits) -> dict:
    valid = [e for e in episodes if not e["scenario_generation_failure"]]
    max_speed_seen = max((e["max_speed_mps"] for e in valid), default=0.0)
    max_accel_seen = max((e["max_accel_mps2"] for e in valid), default=0.0)
    speed_ok = max_speed_seen <= limits.max_horizontal_speed + 1e-6
    accel_ok = max_accel_seen <= limits.max_accel + 1e-6
    return {"max_observed_horizontal_speed_mps": max_speed_seen, "max_allowed_horizontal_speed_mps": limits.max_horizontal_speed,
            "horizontal_speed_within_limit": speed_ok, "max_observed_accel_mps2": max_accel_seen, "max_allowed_accel_mps2": limits.max_accel,
            "accel_within_limit": accel_ok, "passed": bool(speed_ok and accel_ok)}


# ----------------------------------------------------------------------------- protocol
def protocol_specification() -> dict:
    return {
        "version": "a1_cbf_clutter_v2",
        "hypothesis": "Conditioning ONLY the attacker's initial spawn/heading on braking feasibility (never "
                      "altering canonical C2 obstacle generation, the attacker dynamics, or any frozen CBF/route "
                      "parameter) eliminates static-obstacle collisions caused by spawn-time physical "
                      "infeasibility, while route-status handling (route_ok/route_temporarily_unavailable/"
                      "route_error) correctly distinguishes transient planner limitations from real bugs.",
        "frozen_controller_parameters": {
            "kp_per_s": 1.0, "v_xy_max_mps": 12.0, "a_max_mps2": 6.0, "climb_descent_mps": 5.0, "dt_s": 0.5,
            "dt_sub_s": 0.1, "obstacle_clearance_m": 2.0, "active_cbf_range_m": 20.0, "alpha_per_s": 1.0,
            "velocity_polygon_sides": 32, "qp_policy": "hard -> soft -> emergency_brake", "solver": "scipy SLSQP",
            "note": "all identical to Prompt 2; NONE tuned in this stage",
        },
        "viability_equation": "d_i >= s_i^2/(2*a_max) + s_i*dt_sub for every static obstacle i, "
                              "s_i = max(0, -n_i . v_0); no additional buffer",
        "resampling": "attacker spawn/heading only (existing _spawn_enemy_at_edge mechanism), obstacle field and "
                      "asset initialization fixed, up to 100 attempts, else scenario_generation_failure",
        "config_sha256": config_sha256(),
        "validation": {"clutter_v2_seeds": list(CLUTTER_V2_SEEDS), "c0_identity_seeds": list(C0_IDENTITY_SEEDS),
                       "legacy_noop_seeds": list(LEGACY_NOOP_SEEDS), "chunk_size": CHUNK_SIZE, "parent_commit": PARENT_COMMIT},
        "gate": GATE,
        "freeze_rule": "kp, physical limits, alpha, clearance, active range, polygon sides, replan rules, QP "
                       "tolerances are frozen (Prompt 2 values) and were not tuned in this stage",
        "scope": "viability-conditioned static-obstacle navigation only; no defender CBF, no RL training",
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol_specification(), sort_keys=True, default=str).encode()).hexdigest()


def namespace_overlaps() -> dict:
    curriculum_known_max = 192_199
    prior_blocks = [(500_000, 500_999), (510_000, 510_999), (520_000, 520_999)]
    def overlaps(a, b):
        return not (a[1] < b[0] or b[1] < a[0])
    return {
        "clutter_v2_seeds": list(CLUTTER_V2_SEEDS), "legacy_noop_seeds": list(LEGACY_NOOP_SEEDS),
        "curriculum_known_max_evaluation_seed": curriculum_known_max,
        "gap_to_curriculum_m": CLUTTER_V2_SEEDS[0] - curriculum_known_max,
        "clutter_v2_vs_legacy_noop_overlap": overlaps(CLUTTER_V2_SEEDS, LEGACY_NOOP_SEEDS),
        "clutter_v2_vs_prior_blocks_overlap": any(overlaps(CLUTTER_V2_SEEDS, b) for b in prior_blocks),
        "legacy_noop_vs_prior_blocks_overlap": any(overlaps(LEGACY_NOOP_SEEDS, b) for b in prior_blocks),
    }


# ----------------------------------------------------------------------------- C0-identity regression
def c0_identity_check(seeds) -> dict:
    from experiments import journal_cbf_attacker_foundation as F1

    cached_by_seed: dict[int, dict] = {}
    chunk_dir = F1.RESULTS_ROOT / "foundation"
    for path in sorted(chunk_dir.glob("cbf_velocity_chunk_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for ep in payload["episodes"]:
            cached_by_seed[ep["scenario_seed"]] = ep

    cfg = F1.foundation_config()
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=False)
    keys = ["outcome", "steps", "path_length_m", "terminal_distance_m", "overshoots", "mean_speed_mps", "max_speed_mps", "max_accel_mps2"]
    mismatches = []
    checked = 0
    for s in seeds:
        if s not in cached_by_seed:
            continue
        checked += 1
        fresh = F1.run_episode(env, s)
        diffs = {k: (cached_by_seed[s][k], fresh[k]) for k in keys if cached_by_seed[s][k] != fresh[k]}
        if diffs:
            mismatches.append({"seed": s, "diffs": diffs})
    return {"seeds_checked": checked, "mismatches": mismatches, "identical": len(mismatches) == 0}


# ----------------------------------------------------------------------------- restart-safe chunked validation
def chunk_seeds(namespace: tuple[int, int], chunk_index: int, chunk_size: int = CHUNK_SIZE) -> list[int]:
    lo = namespace[0] + chunk_index * chunk_size
    hi = min(lo + chunk_size - 1, namespace[1])
    return list(range(lo, hi + 1))


def num_chunks(namespace: tuple[int, int], chunk_size: int = CHUNK_SIZE) -> int:
    total = namespace[1] - namespace[0] + 1
    return (total + chunk_size - 1) // chunk_size


def _content_sha256(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def run_clutter_validation(results_root: Path, *, enemy_control_mode: str, viability_conditioned_init: bool = True,
                           namespace: tuple[int, int] = CLUTTER_V2_SEEDS, chunk_size: int = CHUNK_SIZE) -> list[dict]:
    from experiments import journal_v2_a1_eg_state as base

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    psha = protocol_sha256()
    tag = f"{enemy_control_mode}_{'viable' if viability_conditioned_init else 'unconditioned'}"
    env = None
    all_episodes: list[dict] = []
    for i in range(num_chunks(namespace, chunk_size)):
        path = results_root / f"{tag}_chunk_{i:03d}.json"
        episodes = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol_sha256") == psha and payload.get("tag") == tag and _content_sha256(payload["episodes"]) == payload.get("content_sha256"):
                episodes = payload["episodes"]
        except (OSError, ValueError, KeyError, TypeError):
            episodes = None
        if episodes is None:
            env = env or SoldierEnv(clutter_config(), enemy_control_mode=enemy_control_mode, viability_conditioned_init=viability_conditioned_init)
            seeds = chunk_seeds(namespace, i, chunk_size)
            episodes = [run_episode(env, s) for s in seeds]
            base.atomic_write_json(path, {"schema": "a1_cbf_clutter_v2_chunk/1", "chunk": i, "tag": tag, "seeds": seeds,
                                           "episodes": episodes, "content_sha256": _content_sha256(episodes), "protocol_sha256": psha})
        all_episodes.extend(episodes)
    return all_episodes


def run_legacy_noop_validation(results_root: Path, *, namespace: tuple[int, int] = LEGACY_NOOP_SEEDS, chunk_size: int = CHUNK_SIZE, parent_commit: str = PARENT_COMMIT) -> dict:
    from experiments import journal_v2_a1_eg_state as base

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    psha = protocol_sha256()
    legacy_mod = _load_legacy_module(parent_commit)
    cfg, _ = VC.build_config("C0", "V2", "greedy")
    mismatches: list[dict] = []
    checked = 0
    for i in range(num_chunks(namespace, chunk_size)):
        path = results_root / f"legacy_noop_chunk_{i:03d}.json"
        payload = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol_sha256") != psha or payload.get("parent_commit") != parent_commit:
                payload = None
        except (OSError, ValueError, KeyError, TypeError):
            payload = None
        if payload is None:
            seeds = chunk_seeds(namespace, i, chunk_size)
            chunk_mismatches = []
            for s in seeds:
                current = run_legacy_episode(SoldierEnv, cfg, s)
                baseline = run_legacy_episode(legacy_mod.SoldierEnv, cfg, s)
                if current != baseline:
                    chunk_mismatches.append({"seed": int(s), "current": current, "baseline": baseline})
            payload = {"schema": "a1_cbf_clutter_v2_legacy_noop_chunk/1", "chunk": i, "seeds": seeds,
                       "mismatches": chunk_mismatches, "parent_commit": parent_commit, "protocol_sha256": psha}
            base.atomic_write_json(path, payload)
        mismatches.extend(payload["mismatches"])
        checked += len(payload["seeds"])
    return {"parent_commit": parent_commit, "seeds_checked": checked, "mismatches": mismatches, "identical": len(mismatches) == 0}
