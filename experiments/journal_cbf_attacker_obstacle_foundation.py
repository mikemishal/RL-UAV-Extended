"""Monte Carlo acceptance experiment for A1-CBF static-obstacle navigation (journal_cbf_attacker, Prompt 2):
route-planner + CBF-QP obstacle avoidance on canonical C2 static clutter. NO RL, NO defender/dynamic-obstacle CBF --
static-obstacle navigation only. Defender interaction disabled via the established intercept_radius=1e-6 mechanism
(reused verbatim from Prompt 1); dynamic clutter disabled via the EXISTING dynamic_obstacles_enabled=False switch
(preserves C2's static geometry bit-for-bit -- see test_c2_static_obstacles_are_all_map_known... in
tests/test_cbf_obstacle.py).

Final-benchmark simplification (brief section 0): "No Clutter" == canonical C0; "Clutter" == canonical C2. Both
used UNCHANGED; no new clutter generator.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from uav_defend.attackers.cbf_attacker import CBFAttackerLimits
from uav_defend.attackers.cbf_obstacle import CBF_ACTIVE_OBSTACLE_RANGE, CBF_ALPHA, CBF_OBSTACLE_CLEARANCE
from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv

from experiments import journal_v2_validation_core as VC
from experiments import journal_v2_a1_r2 as R
from experiments.journal_v2_a1_cur4a import closest_approaches
from experiments.journal_cbf_attacker_foundation import (
    PARENT_COMMIT as FOUNDATION_PARENT_COMMIT,
    _load_legacy_module,
    legacy_noop_episode_keys,
    run_legacy_episode,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS = "docs/journal_cbf_attacker/obstacle_cbf/"
RESULTS_ROOT = R.V2_RESULTS_DIR / "journal_cbf_attacker" / "obstacle_cbf"

# ----------------------------------------------------------------------------- pre-registered namespaces (frozen before any validation)
# Re-uses the SAME 1,000-seed block as the Prompt-1 foundation stage: a scenario is identified by the (seed,
# EnvConfig) PAIR, not the seed alone, and this stage uses a DIFFERENT config (C2, not C0) -- no collision with
# Prompt 1's own use of these seeds on C0.
OBSTACLE_SEEDS = (500_000, 500_999)          # 1,000 matched A0 vs A1-CBF C2 scenarios
C0_IDENTITY_SEEDS = (500_000, 500_999)       # >=500-seed C0 behavioral-identity regression (brief section 20)
LEGACY_NOOP_SEEDS = (520_000, 520_999)       # fresh block, far from every other namespace used in this repo
CHUNK_SIZE = 100
PARENT_COMMIT = FOUNDATION_PARENT_COMMIT     # unchanged: no new legacy-affecting commit since Prompt 1

# Frozen acceptance gate (brief section 24; do not alter after evaluation starts).
GATE = {
    "capture_min": 0.95, "obstacle_collision_max": 0, "timeout_max": 0.01, "exhaustion_max": 0.04,
    "orbit_episodes_max": 0, "repeated_flyby_max": 0.02, "path_excess_5x_max": 0.01, "planner_failure_episodes_max": 0,
    "nonfinite_max": 0, "qp_hard_or_soft_success_min": 1.0, "emergency_brake_step_fraction_max": 0.01,
    "soft_fallback_step_fraction_max": 0.01,
}
REPEATED_FLYBY_OVERSHOOTS = 2
ORBIT_REVOLUTION_THRESHOLD = 1.0


def seeds_of(rng: tuple[int, int]) -> range:
    return range(rng[0], rng[1] + 1)


def obstacle_config(*, enemy_control_mode_placeholder: str | None = None) -> EnvConfig:
    """Canonical C2 V2 config with: (1) `intercept_radius=1e-6` (established no-defender-interaction mechanism,
    unchanged), (2) `dynamic_obstacles_enabled=False` (the EXISTING supported switch -- disables ONLY the dynamic
    clutter subset; the static AABB geometry is bit-for-bit UNCHANGED for a given seed, see
    tests/test_cbf_obstacle.py). Nothing else differs from the canonical C2 V2 config."""
    cfg, _estimator = VC.build_config("C2", "V2", "passive")
    return dataclasses.replace(cfg, intercept_radius=1e-6, dynamic_obstacles_enabled=False)


def config_sha256() -> str:
    return hashlib.sha256(json.dumps(dataclasses.asdict(obstacle_config()), sort_keys=True, default=str).encode()).hexdigest()


# ----------------------------------------------------------------------------- episode runner (ground truth only; no policy/observation interface)
def run_episode(env: SoldierEnv, seed: int) -> dict:
    """Runs one C2 scenario with `env`'s own control mode (A0 scripted legacy, or A1-CBF) with a constant zero
    defender action. Returns ground-truth diagnostics read back from `info` every step."""
    obs, info = env.reset(seed=seed)
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
    return {
        "scenario_seed": int(seed), "outcome": outcome, "steps": n_steps, "duration_s": n_steps * env.config.dt,
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
        "route_planner_failures": int(info_final.get("route_planner_failures", 0)),
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


def evaluate(seeds, *, enemy_control_mode: str) -> list[dict]:
    env = SoldierEnv(obstacle_config(), enemy_control_mode=enemy_control_mode)
    return [run_episode(env, s) for s in seeds]


def metrics(episodes: list[dict]) -> dict:
    n = len(episodes)
    outcome = np.array([e["outcome"] for e in episodes])
    pe = np.array([e["path_excess"] for e in episodes], dtype=np.float64)
    cap = outcome == "soldier_caught"
    first_pass_of_captures = [e["first_pass"] for e in episodes if e["outcome"] == "soldier_caught"]
    misses = [m for e in episodes for m in e["miss_distances_m"]]
    hard_fracs = [e["hard_step_fraction_of_active"] for e in episodes if e["hard_step_fraction_of_active"] is not None]
    return {
        "episodes": n, "capture": float(cap.mean()), "exhaustion": float((outcome == "enemy_endurance_exhausted").mean()),
        "timeout": float((outcome == "timeout").mean()), "obstacle_collision": int(sum(e["obstacle_collision"] for e in episodes)),
        "obstacle_collision_rate": float(np.mean([e["obstacle_collision"] for e in episodes])),
        "outcomes": {k: int((outcome == k).sum()) for k in sorted(set(outcome.tolist()))},
        "first_pass_capture": float(np.mean([e["first_pass"] for e in episodes])),
        "first_pass_fraction_of_captures": (float(np.mean(first_pass_of_captures)) if first_pass_of_captures else None),
        "repeated_flyby_episodes": int(sum(e["repeated_flyby"] for e in episodes)), "repeated_flyby_fraction": float(np.mean([e["repeated_flyby"] for e in episodes])),
        "orbit_episodes": int(sum(e["orbit"] for e in episodes)), "nonfinite": int(sum(not e["finite"] for e in episodes)),
        "median_path_excess": float(np.nanmedian(pe)), "mean_path_excess": float(np.nanmean(pe)),
        "path_excess_5x_episodes": int(sum(e["path_excess"] > 5.0 for e in episodes if np.isfinite(e["path_excess"]))),
        "path_excess_5x_fraction": float(np.mean([e["path_excess"] > 5.0 if np.isfinite(e["path_excess"]) else False for e in episodes])),
        "mean_duration_s": float(np.mean([e["duration_s"] for e in episodes])), "mean_path_length_m": float(np.mean([e["path_length_m"] for e in episodes])),
        "mean_closest_approaches": float(np.mean([e["closest_approach_count"] for e in episodes])),
        "miss_distance_median_m": (float(np.median(misses)) if misses else None), "miss_distance_count": len(misses),
        "mean_speed_mps": float(np.mean([e["mean_speed_mps"] for e in episodes])), "max_speed_mps": float(np.max([e["max_speed_mps"] for e in episodes])),
        "max_accel_mps2": float(np.max([e["max_accel_mps2"] for e in episodes])),
        "min_physical_obstacle_clearance": float(np.min([e["min_physical_obstacle_clearance"] for e in episodes])),
        "min_inflated_obstacle_clearance": float(np.min([e["min_inflated_obstacle_clearance"] for e in episodes])),
        "cbf_h_min": float(np.min([e["cbf_h_min"] for e in episodes])),
        "barrier_violation_episodes": int(sum(e["cbf_h_min"] < 0 for e in episodes)),
        "total_route_replans": int(sum(e["route_replan_count"] for e in episodes)),
        "mean_route_replans": float(np.mean([e["route_replan_count"] for e in episodes])),
        "planner_failure_episodes": int(sum(e["route_planner_failures"] > 0 for e in episodes)),
        "total_planner_failures": int(sum(e["route_planner_failures"] for e in episodes)),
        "total_emergency_brake_events": int(sum(e["emergency_brake_count"] for e in episodes)),
        "mean_intervention_step_fraction": float(np.mean([e["intervention_step_fraction"] for e in episodes])),
        "mean_intervention_norm": float(np.mean([e["mean_intervention_norm"] for e in episodes])),
        "max_intervention_norm": float(np.max([e["max_intervention_norm"] for e in episodes])),
        "mean_hard_step_fraction_of_active": (float(np.mean(hard_fracs)) if hard_fracs else None),
        "mean_soft_step_fraction": float(np.mean([e["soft_step_fraction"] for e in episodes])),
        "mean_emergency_step_fraction": float(np.mean([e["emergency_step_fraction"] for e in episodes])),
        "qp_hard_or_soft_success_fraction": float(np.mean([e["qp_hard_or_soft_success"] for e in episodes])),
    }


def gate_status(m: dict) -> dict:
    checks = {
        "capture": m["capture"] >= GATE["capture_min"],
        "obstacle_collision": m["obstacle_collision"] <= GATE["obstacle_collision_max"],
        "timeout": m["timeout"] <= GATE["timeout_max"], "exhaustion": m["exhaustion"] <= GATE["exhaustion_max"],
        "orbit": m["orbit_episodes"] <= GATE["orbit_episodes_max"], "repeated_flyby": m["repeated_flyby_fraction"] <= GATE["repeated_flyby_max"],
        "path_excess_5x": m["path_excess_5x_fraction"] <= GATE["path_excess_5x_max"],
        "planner_failure": m["planner_failure_episodes"] <= GATE["planner_failure_episodes_max"],
        "nonfinite": m["nonfinite"] <= GATE["nonfinite_max"],
        "qp_hard_or_soft_success": m["qp_hard_or_soft_success_fraction"] >= GATE["qp_hard_or_soft_success_min"],
        "emergency_brake_fraction": m["mean_emergency_step_fraction"] <= GATE["emergency_brake_step_fraction_max"],
        "soft_fallback_fraction": m["mean_soft_step_fraction"] <= GATE["soft_fallback_step_fraction_max"],
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def constraint_verification(episodes: list[dict], *, limits: CBFAttackerLimits) -> dict:
    max_speed_seen = max((e["max_speed_mps"] for e in episodes), default=0.0)
    max_accel_seen = max((e["max_accel_mps2"] for e in episodes), default=0.0)
    speed_ok = max_speed_seen <= limits.max_horizontal_speed + 1e-6
    accel_ok = max_accel_seen <= limits.max_accel + 1e-6
    return {"max_observed_horizontal_speed_mps": max_speed_seen, "max_allowed_horizontal_speed_mps": limits.max_horizontal_speed,
            "horizontal_speed_within_limit": speed_ok, "max_observed_accel_mps2": max_accel_seen, "max_allowed_accel_mps2": limits.max_accel,
            "accel_within_limit": accel_ok, "passed": bool(speed_ok and accel_ok)}


# ----------------------------------------------------------------------------- C0-identity regression (brief section 20)
def c0_identity_check(seeds) -> dict:
    """Compares A1-CBF running the Prompt-2-modified SoldierEnv on C0 against the CACHED Prompt-1 foundation
    per-episode results (docs/journal_cbf_attacker/foundation/ -- the exact same seeds, exact same C0 config): the
    obstacle-aware code path must be a perfect no-op whenever `self._obstacle_layout` is empty."""
    from experiments import journal_cbf_attacker_foundation as F1

    cached_by_seed: dict[int, dict] = {}
    chunk_dir = F1.RESULTS_ROOT / "foundation"
    for path in sorted(chunk_dir.glob("cbf_velocity_chunk_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for ep in payload["episodes"]:
            cached_by_seed[ep["scenario_seed"]] = ep

    cfg = F1.foundation_config()
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity")
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


# ----------------------------------------------------------------------------- protocol
def protocol_specification() -> dict:
    return {
        "version": "a1_cbf_obstacle_v1",
        "hypothesis": "A route planner (reused, unmodified visibility-graph + A*) for global deadlock avoidance, "
                      "combined with a per-instant CBF-QP safety filter over inflated static-obstacle AABBs, lets "
                      "A1-CBF navigate canonical C2 static clutter while pursuing the moving asset, without any RL.",
        "architecture": "moving asset -> static route waypoint -> v_nom -> CBF-QP -> v_safe -> velocity executor",
        "benchmark_simplification": {"no_clutter": "C0", "clutter": "C2", "note": "canonical, unchanged; no new clutter generator"},
        "environment": {"clutter": "C2", "dynamic_obstacles_enabled": False, "asset": "canonical V2 motion (v_s=1.5, legacy_gaussian_random_walk)",
                        "defender": "structurally present, cannot terminate the episode (intercept_radius=1e-6, reused unchanged from Prompt 1)",
                        "mission_contact_mode": "swept", "enemy_endurance_enabled": True, "capture_radius_m": 2.0},
        "config_sha256": config_sha256(),
        "route_planner": {"reused_unmodified_function": "uav_defend.obstacles.route_planner.build_and_search_route",
                          "clearance_m": CBF_OBSTACLE_CLEARANCE, "planning_range_m": 40.0, "max_graph_nodes": 150,
                          "waypoint_tolerance_m": 2.0, "replan_period_s": 1.0, "asset_move_trigger_m": 2.0,
                          "role": "global/local-minimum deadlock avoidance ONLY -- never safety margins"},
        "cbf_qp": {"obstacle_clearance_m": CBF_OBSTACLE_CLEARANCE, "active_obstacle_range_m": CBF_ACTIVE_OBSTACLE_RANGE,
                   "alpha_per_s": CBF_ALPHA, "velocity_polygon_sides": 32, "solver": "scipy.optimize.minimize(method='SLSQP')",
                   "fallback_chain": "hard -> soft-slack (1e4*delta^2 penalty) -> emergency_brake (v_safe=0)",
                   "substep_recomputation": "re-solved every dt_sub=0.1s substep when >=1 hazard is active at a step's start "
                                            "(NOT held constant across the full 0.5s external step, unlike Prompt 1's identity "
                                            "path) -- required so the stopping-margin formula's own dt_sub reaction-time "
                                            "assumption is actually realized; see README.md"},
        "validation": {"obstacle_seeds": list(OBSTACLE_SEEDS), "c0_identity_seeds": list(C0_IDENTITY_SEEDS),
                       "legacy_noop_seeds": list(LEGACY_NOOP_SEEDS), "chunk_size": CHUNK_SIZE, "parent_commit": PARENT_COMMIT},
        "gate": GATE,
        "freeze_rule": "clearance, alpha, active_range, kp, physical limits, polygon sides, replan period/triggers are "
                       "frozen BEFORE the acceptance run and never tuned afterward",
        "scope": "static-obstacle navigation ONLY; no defender CBF, no dynamic-obstacle CBF, no RL training",
    }


def protocol_sha256() -> str:
    return hashlib.sha256(json.dumps(protocol_specification(), sort_keys=True, default=str).encode()).hexdigest()


def namespace_overlaps() -> dict:
    curriculum_known_max = 192_199
    foundation_legacy_noop = (510_000, 510_999)
    return {"obstacle_seeds": list(OBSTACLE_SEEDS), "legacy_noop_seeds": list(LEGACY_NOOP_SEEDS),
            "curriculum_known_max_evaluation_seed": curriculum_known_max,
            "gap_to_curriculum_m": OBSTACLE_SEEDS[0] - curriculum_known_max,
            "obstacle_vs_legacy_noop_overlap": not (OBSTACLE_SEEDS[1] < LEGACY_NOOP_SEEDS[0] or LEGACY_NOOP_SEEDS[1] < OBSTACLE_SEEDS[0]),
            "obstacle_vs_foundation_legacy_noop_overlap": not (LEGACY_NOOP_SEEDS[1] < foundation_legacy_noop[0] or foundation_legacy_noop[1] < LEGACY_NOOP_SEEDS[0])}


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


def run_obstacle_validation(results_root: Path, *, enemy_control_mode: str, namespace: tuple[int, int] = OBSTACLE_SEEDS, chunk_size: int = CHUNK_SIZE) -> list[dict]:
    """Restart-safe: each chunk is written atomically with an embedded content hash + this stage's protocol hash; a
    chunk whose file verifies is never regenerated."""
    from experiments import journal_v2_a1_eg_state as base

    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    psha = protocol_sha256()
    env = None
    all_episodes: list[dict] = []
    for i in range(num_chunks(namespace, chunk_size)):
        path = results_root / f"{enemy_control_mode}_chunk_{i:03d}.json"
        episodes = None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol_sha256") == psha and payload.get("enemy_control_mode") == enemy_control_mode and _content_sha256(payload["episodes"]) == payload.get("content_sha256"):
                episodes = payload["episodes"]
        except (OSError, ValueError, KeyError, TypeError):
            episodes = None
        if episodes is None:
            env = env or SoldierEnv(obstacle_config(), enemy_control_mode=enemy_control_mode)
            seeds = chunk_seeds(namespace, i, chunk_size)
            episodes = [run_episode(env, s) for s in seeds]
            base.atomic_write_json(path, {"schema": "a1_cbf_obstacle_chunk/1", "chunk": i, "enemy_control_mode": enemy_control_mode, "seeds": seeds,
                                           "episodes": episodes, "content_sha256": _content_sha256(episodes), "protocol_sha256": psha})
        all_episodes.extend(episodes)
    return all_episodes


def run_legacy_noop_validation(results_root: Path, *, namespace: tuple[int, int] = LEGACY_NOOP_SEEDS, chunk_size: int = CHUNK_SIZE, parent_commit: str = PARENT_COMMIT) -> dict:
    """Restart-safe chunked legacy no-op check reusing the Prompt-1 machinery (_load_legacy_module,
    legacy_noop_episode_keys, run_legacy_episode) on a FRESH, non-overlapping seed block."""
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
            payload = {"schema": "a1_cbf_obstacle_legacy_noop_chunk/1", "chunk": i, "seeds": seeds,
                       "mismatches": chunk_mismatches, "parent_commit": parent_commit, "protocol_sha256": psha}
            base.atomic_write_json(path, payload)
        mismatches.extend(payload["mismatches"])
        checked += len(payload["seeds"])
    return {"parent_commit": parent_commit, "seeds_checked": checked, "mismatches": mismatches, "identical": len(mismatches) == 0}
