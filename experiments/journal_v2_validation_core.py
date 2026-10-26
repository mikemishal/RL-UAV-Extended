"""Observational recorder and behavioral metrics for the V2 visual validation.

Nothing here changes simulation behavior: episodes are run through the normal
`SoldierEnv.step()` with the controller's own action, and everything else is
read back from the returned `info` dict and the environment's public state.

Metric definitions (also documented in docs/journal_v2_visual_validation.md):

* Missed swept contact (independent audit): on every NON-terminal step, the
  relative motion of the two bodies is sampled densely (257 sub-steps, no use
  of the analytic solver). If the minimum separation is <= the contact radius
  the step crossed the sphere without ending the episode. For
  attacker->asset the radius is `threat_radius`; for defender->attacker it is
  `intercept_radius`, with the defender treated as stationary on pre-detection
  standby steps (synchronisation is bookkeeping, not flight).
* Target overshoot: the range to the asset decreases (rate < -1 m/s) and then
  increases (rate > +1 m/s) with no terminal event. Counted once per
  approach->recede transition. Recomputed here from the trace and checked
  against the environment's own counter.
* Orbit/circling: cumulative signed bearing change of the attacker about the
  asset while it stays within 20 m of it, taken per continuous excursion; an
  excursion with |bearing change| >= 1 revolution is an orbit event.
* Pursuit efficiency: (range_initial - range_final) / attacker_path_length.
  Because the asset moves slightly this is NOT a shortest-path ratio; it is
  the net range reduction per metre flown (<= 1 for a straight pursuit of a
  stationary target that ends on the capture sphere).
* Deviation segment: a maximal run of steps whose behavior mode is not DIRECT.
"""

from __future__ import annotations

import copy
import dataclasses
import math

import numpy as np

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info

from experiments.journal_v2_protocol import build_v1_env_config, build_v2_env_config
from experiments.journal_v2_trace import trace_row

CONTROLLERS = ("passive", "greedy", "kalman_greedy", "pn", "lead", "rmpc")
PARKED_DEFENDER = (-45.0, -45.0, 29.0)
ORBIT_RANGE_M = 20.0
ORBIT_REVOLUTION_THRESHOLD = 1.0
OVERSHOOT_DEADBAND_M_S = 1.0
# Longest run of non-DIRECT steps tolerated while the straight line to the
# asset is clear (the V1 release-hysteresis length, 4 steps = 2 s).
LINGER_RUN_LIMIT_STEPS = 4
AUDIT_SAMPLES = 257
DIRECT = "DIRECT"
FINAL_INFO_KEYS = (
    "outcome", "terminal_event", "terminal_event_fraction", "capture_contact_fraction",
    "capture_contact_position", "capture_contact_distance", "interception_contact_fraction",
    "interception_contact_position", "interception_contact_distance", "attacker_initial_range",
    "attacker_path_length", "attacker_flight_time_s", "attacker_heading_change_total_deg",
    "attacker_overshoot_count", "attacker_path_efficiency", "enemy_soldier_dist", "defender_enemy_dist",
    "step_count", "mission_contact_mode", "enemy_nominal_guidance_mode",
    "enemy_endurance_enabled", "enemy_endurance_budget_m", "enemy_endurance_traveled_m",
    "enemy_endurance_remaining_m",
)


# --------------------------------------------------------------- geometry
def dense_min_distance(p0a, p1a, p0b, p1b, n: int = AUDIT_SAMPLES) -> tuple[float, float]:
    """Minimum separation of two linearly moving points over a step, by dense
    sampling (independent of the analytic solver). Returns (distance, tau)."""
    p0a, p1a, p0b, p1b = (np.asarray(p, dtype=np.float64) for p in (p0a, p1a, p0b, p1b))
    taus = np.linspace(0.0, 1.0, n)[:, None]
    rel = (p0a - p0b)[None, :] + taus * ((p1a - p0a) - (p1b - p0b))[None, :]
    dist = np.linalg.norm(rel, axis=1)
    k = int(np.argmin(dist))
    return float(dist[k]), float(taus[k, 0])


def _point_segment_distance(point, a, b) -> float:
    point, a, b = (np.asarray(x, dtype=np.float64) for x in (point, a, b))
    ab = b - a
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom < 1e-12 else float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def line_blocked_state(env: SoldierEnv, clearance: float) -> tuple[int | None, int | None]:
    """Ground-truth (diagnostic only) test of the straight attacker->asset
    line: index of the first static obstacle (inflated by `clearance`) or
    dynamic obstacle id (radius + clearance) that it touches, else None."""
    a = np.asarray(env._enemy_pos, dtype=np.float64)
    s = np.asarray(env._soldier_pos, dtype=np.float64)
    static_hit = None
    for i, o in enumerate(env._obstacle_layout.obstacles):
        inflated = AABBObstacle.from_center_half_extents(o.center, o.half_extents + clearance)
        if inflated.segment_intersects(a, s):
            static_hit = i
            break
    dynamic_hit = None
    for d in env._dynamic_obstacles:
        if _point_segment_distance(d.position, a, s) <= d.radius + clearance:
            dynamic_hit = int(d.id)
            break
    return static_hit, dynamic_hit


# ----------------------------------------------------------- definitions
def count_overshoots(ranges: np.ndarray, dt: float, deadband: float = OVERSHOOT_DEADBAND_M_S) -> int:
    """Approach->recede transitions of the range to the asset."""
    count, trend, prev = 0, 0, float(ranges[0])
    for r in ranges[1:]:
        rate = (float(r) - prev) / dt
        if rate < -deadband:
            trend = -1
        elif rate > deadband:
            if trend == -1:
                count += 1
            trend = 1
        prev = float(r)
    return count


def orbit_revolutions(attacker: np.ndarray, asset: np.ndarray, radius: float = ORBIT_RANGE_M) -> tuple[float, float]:
    """(max, total) |cumulative signed bearing change| in revolutions about the
    asset over continuous excursions within `radius` of it."""
    rel = np.asarray(attacker, dtype=np.float64) - np.asarray(asset, dtype=np.float64)
    excursions: list[float] = []
    acc, prev = 0.0, None
    for v in rel:
        inside = float(np.linalg.norm(v)) <= radius and float(np.hypot(v[0], v[1])) > 1e-6
        if inside:
            bearing = math.atan2(v[1], v[0])
            if prev is not None:
                acc += (bearing - prev + math.pi) % (2 * math.pi) - math.pi
            prev = bearing
        else:
            if prev is not None:
                excursions.append(abs(acc))
            acc, prev = 0.0, None
    if prev is not None:
        excursions.append(abs(acc))
    if not excursions:
        return 0.0, 0.0
    return max(excursions) / (2 * math.pi), sum(excursions) / (2 * math.pi)


def deviation_segments(rec: dict) -> list[dict]:
    """Maximal runs of non-DIRECT behavior with entry/exit diagnostics."""
    rows, ex = rec["rows"], rec["extras"]
    n = len(rows) - 1
    ranges = np.linalg.norm(
        np.array([r["attacker_pos"] for r in rows]) - np.array([r["asset_pos"] for r in rows]), axis=1,
    )
    segments, k = [], 1
    while k <= n:
        if rows[k]["behavior_mode"] == DIRECT:
            k += 1
            continue
        j = k
        while j + 1 <= n and rows[j + 1]["behavior_mode"] != DIRECT:
            j += 1
        decision = ex[k - 1]  # state the planner saw when choosing step k
        line_blocked = decision["line_blocked_static"] is not None or decision["line_blocked_dynamic"] is not None
        obstacle_type = ex[k]["avoid_obstacle_type"]
        obstacle_id = ex[k]["avoid_obstacle_id"]
        if rows[k]["behavior_mode"] == "EVADE":
            kind = "evasion"
        elif obstacle_type is not None and obstacle_type != "static":
            kind = "dynamic"
        elif line_blocked:
            kind = "static_line_blocked"
        else:
            kind = "static_prediction_only"
        # Associated with an actual PERCEIVED obstacle: a dynamic obstacle
        # id, or a static id present in the attacker's perceived set.
        associated = obstacle_id is not None and (
            kind == "dynamic" or obstacle_id in rows[k]["perceived_static_ids"]
        )
        # Counterfactual necessity (only where a pre-activation branch was run).
        cf = rec.get("detour_cf", {}).get(k)
        if cf is None:
            necessity = None
        elif cf["collides"]:
            necessity = "necessary_collision"
        elif cf["min_clearance"] < rec.get("config", {}).get("clearance_margin", 2.0):
            necessity = "necessary_margin"
        else:
            necessity = "unnecessary"
        run, longest = 0, 0
        for i in range(k, j + 1):
            clear = ex[i - 1]["line_blocked_static"] is None and ex[i - 1]["line_blocked_dynamic"] is None
            run = run + 1 if clear else 0
            longest = max(longest, run)
        exit_step = j + 1 if j + 1 <= n else None
        segments.append({
            "activation_step": k, "last_avoid_step": j, "return_to_direct_step": exit_step,
            "duration_steps": j - k + 1, "kind": kind, "line_blocked_at_activation": line_blocked,
            "obstacle_type": obstacle_type, "obstacle_id": obstacle_id,
            "associated_with_perceived_obstacle": bool(associated),
            "necessity": necessity,
            "direct_rollout_min_clearance": None if cf is None else (cf["min_clearance"] if math.isfinite(cf["min_clearance"]) else None),
            "range_at_activation": float(ranges[k - 1]), "range_at_last_avoid": float(ranges[j]),
            "line_clear_at_return": None if exit_step is None else (
                ex[j]["line_blocked_static"] is None and ex[j]["line_blocked_dynamic"] is None),
            "max_linger_run_steps": longest, "terminated_while_avoiding": exit_step is None,
        })
        k = j + 1
    return segments


# Thresholds for `terminal_flyby_metrics`'s diagnostic-only orbit/fly-by detector (see its
# docstring for why these are needed in addition to `orbit_revolutions`/`ORBIT_REVOLUTION_THRESHOLD`).
TERMINAL_ORBIT_FLYBY_THRESHOLD = 3
TERMINAL_ORBIT_REVOLUTION_THRESHOLD = 1.5


def terminal_flyby_metrics(rec: dict) -> dict:
    """Diagnostic-only (never used to steer the vehicle): detects a sustained post-avoidance
    fly-by/orbit around the asset that `orbit_revolutions` can miss because it resets its
    cumulative-bearing accumulator every time the attacker leaves a FIXED `ORBIT_RANGE_M` (20 m)
    radius -- C3 seed 130017 repeatedly swings out well past 20 m between near misses, so no
    single continuous within-20m excursion ever reaches the 1-revolution threshold even though
    the attacker visibly loiters around the asset for hundreds of steps.

    Computed over the TERMINAL WINDOW: from the end of the last avoidance segment (or episode
    start, if none) to the end of the episode -- independent of any fixed radius. A normal
    capture's terminal window is short and close to a straight final approach, so none of the
    three signals below grows large; a sustained fly-by/orbit keeps returning to a small range
    without ever being captured, and the accumulated bearing swings keep growing regardless of
    how far out each loop goes.

    Returns:
      - fly_by_count: local-minima-of-range events (closest approaches) in the terminal window
        that do not reach the capture sphere;
      - angular_travel_revolutions: cumulative |bearing change| about the asset over the terminal
        window, in revolutions (never reset by a fixed radius, unlike `orbit_revolutions`);
      - closing_sign_changes: sign changes of the range rate (closing <-> opening) in the window
        (supporting evidence only, not gated on);
      - detected: fly_by_count >= TERMINAL_ORBIT_FLYBY_THRESHOLD AND angular_travel_revolutions >=
        TERMINAL_ORBIT_REVOLUTION_THRESHOLD -- both signals, chosen well above what a single
        normal curved intercept produces (see docs/journal_v2_terminal_speed_governor.md for the
        calibration evidence across the recorded population).
    """
    rows = rec["rows"]
    segments = deviation_segments(rec)
    start = max(1, max((s["return_to_direct_step"] or (s["last_avoid_step"] + 1)) for s in segments)) if segments else 1
    att = np.array([r["attacker_pos"] for r in rows[start - 1:]])
    ast = np.array([r["asset_pos"] for r in rows[start - 1:]])
    if len(att) < 3:
        return {"fly_by_count": 0, "angular_travel_revolutions": 0.0, "closing_sign_changes": 0, "detected": False}
    ranges = np.linalg.norm(att - ast, axis=1)
    rel = att[:, :2] - ast[:, :2]
    bearings = np.arctan2(rel[:, 1], rel[:, 0])
    deltas = (np.diff(bearings) + math.pi) % (2 * math.pi) - math.pi
    angular_travel = float(np.sum(np.abs(deltas))) / (2 * math.pi)
    dt = rec["config"]["dt"]
    rates = np.diff(ranges) / dt
    signs = np.sign(np.where(np.abs(rates) > 1e-6, rates, 0.0))
    nonzero = signs[signs != 0]
    closing_sign_changes = int(np.sum(np.diff(nonzero) != 0)) if len(nonzero) > 1 else 0
    threat_radius = rec["config"]["threat_radius"]
    fly_by_count = sum(
        1 for i in range(1, len(ranges) - 1)
        if ranges[i] < ranges[i - 1] and ranges[i] <= ranges[i + 1] and ranges[i] > threat_radius
    )
    return {
        "fly_by_count": fly_by_count, "angular_travel_revolutions": angular_travel,
        "closing_sign_changes": closing_sign_changes,
        "detected": fly_by_count >= TERMINAL_ORBIT_FLYBY_THRESHOLD and angular_travel >= TERMINAL_ORBIT_REVOLUTION_THRESHOLD,
    }


# ------------------------------------------------------------- recording
def make_policy(controller: str, cfg):
    if controller == "passive":
        return None
    if controller in ("greedy", "kalman_greedy"):
        return get_policy(controller)
    if controller == "rmpc":
        from experiments.clutter_benchmark_protocol import _frozen_rmpc_config
        from uav_defend.policies.mpc.rmpc_policy import RMPCPolicy

        return RMPCPolicy(config=cfg, rmpc_config=_frozen_rmpc_config())
    return get_policy(controller, config=cfg)


def build_config(level: str, semantics: str, controller: str):
    estimator = "kalman" if controller == "kalman_greedy" else "measurement"
    cfg = (build_v2_env_config if semantics == "V2" else build_v1_env_config)(level, estimator)
    if controller == "passive":
        # Attacker-motion harness: defender parked far away, never acts, and is
        # NOT synchronised onto the asset (so it cannot pre-empt capture).
        cfg = dataclasses.replace(cfg, defender_standby_until_detection=False)
    return cfg, estimator


def _step_extras(env: SoldierEnv, info: dict, clearance: float) -> dict:
    static_hit, dynamic_hit = line_blocked_state(env, clearance)
    return {
        "dynamic_obstacles": [
            (float(o.position[0]), float(o.position[1]), float(o.position[2]), float(o.radius), int(o.id))
            for o in env._dynamic_obstacles
        ],
        "line_blocked_static": static_hit, "line_blocked_dynamic": dynamic_hit,
        "avoid_obstacle_type": info.get("enemy_avoidance_obstacle_type"),
        "avoid_obstacle_id": info.get("enemy_avoidance_obstacle_id"),
        "defender_standby": bool(info.get("defender_standby", False)),
        "weave_bias": float(info.get("weave_bias", 0.0)),
        "evasion_active": bool(info.get("enemy_evasion_active", False)),
    }


def _detour_counterfactual(snapshot: SoldierEnv, horizon: int = 12) -> dict:
    """Branch from the PRE-activation state with obstacle avoidance disabled and
    fly the same direct mission command for the planner's own horizon (12
    steps). Reports whether that direct flight would have collided with an
    obstacle and how close it would have come. The defender is parked and
    standby disabled so only the attacker/obstacle interaction is measured."""
    snapshot.config = dataclasses.replace(
        snapshot.config, enemy_obstacle_avoidance_enabled=False, defender_standby_until_detection=False,
    )
    snapshot._defender_pos = np.asarray(PARKED_DEFENDER, dtype=np.float32)
    snapshot._defender_vel = np.zeros(3, dtype=np.float32)
    zero = np.zeros(3, dtype=np.float32)
    collides, min_clear, steps, outcome = False, float("inf"), 0, "ongoing"
    for _ in range(horizon):
        _, _, terminated, truncated, info = snapshot.step(zero)
        steps += 1
        outcome = info["outcome"]
        min_clear = min(min_clear, float(info["enemy_min_obstacle_clearance"]),
                        float(info["enemy_min_dynamic_obstacle_clearance"]))
        if info["enemy_obstacle_collision"] or info["enemy_dynamic_obstacle_collision"]:
            collides = True
            break
        if terminated or truncated:
            break
    return {"collides": collides, "min_clearance": min_clear, "steps": steps, "outcome": outcome}


def record_episode(level: str, semantics: str, controller: str, seed: int, *, counterfactual: bool = False,
                   detour_cf: bool = False, cfg_transform=None, setup=None) -> dict:
    """Run one episode and return its full observational record.

    `counterfactual`: on a terminal swept contact, also step a copy of the
    pre-step state under endpoint contact semantics.
    `detour_cf`: for every DIRECT -> avoidance activation, branch a copy of the
    pre-step state with avoidance disabled (see `_detour_counterfactual`).
    `cfg_transform` / `setup`: optional hooks (config edit before construction,
    state placement after reset) for constructed deterministic scenarios.
    """
    cfg, estimator = build_config(level, semantics, controller)
    if cfg_transform is not None:
        cfg = cfg_transform(cfg)
    env = SoldierEnv(cfg)
    env.enemy_tactical_override = None
    policy = make_policy(controller, cfg)
    obs, info = env.reset(seed=seed)
    if policy is not None:
        policy.reset()
    if controller == "passive":
        env._defender_pos = np.asarray(PARKED_DEFENDER, dtype=np.float32)
        env._defender_vel = np.zeros(3, dtype=np.float32)
        info = env._get_info()
    if setup is not None:
        setup(env)
        info = env._get_info()

    clearance = 0.0  # physical (uninflated) geometry: the same test that governs V2 release
    dt = cfg.dt
    rows = [trace_row(0, dt, info)]
    extras = [_step_extras(env, info, clearance)]
    static_obstacles = [
        {"min": [float(x) for x in o.min_corner], "max": [float(x) for x in o.max_corner], "map_known": bool(o.map_known)}
        for o in env._obstacle_layout.obstacles
    ]
    missed_attacker: list[dict] = []
    missed_defender: list[dict] = []
    counterfactual_record = None
    detour_results: dict[int, dict] = {}
    pre_terminal = None
    done, step = False, 0
    prev = {
        "att": np.asarray(info["enemy_pos"], dtype=np.float64), "ast": np.asarray(info["soldier_pos"], dtype=np.float64),
        "dfn": np.asarray(info["defender_pos"], dtype=np.float64),
    }
    while not done:
        if policy is None:
            action = np.zeros(3, dtype=np.float32)
        else:
            action = np.asarray(
                policy.act(env._get_obs(), build_policy_info(env._get_info(), estimator)), dtype=np.float32,
            ).reshape(3)
        snapshot = copy.deepcopy(env) if (counterfactual or detour_cf) else None
        _, _, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        step += 1
        rows.append(trace_row(step, dt, info))
        extras.append(_step_extras(env, info, clearance))
        if (detour_cf and rows[-1]["behavior_mode"] in ("OBSTACLE_AVOID", "ROUTE_BYPASS")
                and rows[-2]["behavior_mode"] == DIRECT):
            detour_results[step] = _detour_counterfactual(copy.deepcopy(snapshot))
        att = np.asarray(info["enemy_pos"], dtype=np.float64)
        ast = np.asarray(info["soldier_pos"], dtype=np.float64)
        dfn = np.asarray(info["defender_pos"], dtype=np.float64)
        if not done:
            d_att, tau_att = dense_min_distance(prev["att"], att, prev["ast"], ast)
            if d_att <= cfg.threat_radius:
                missed_attacker.append({"step": step, "min_distance": d_att, "tau": tau_att,
                                        "endpoint_distance": float(info["enemy_soldier_dist"])})
            d_start = dfn if info.get("defender_standby") else prev["dfn"]
            d_def, tau_def = dense_min_distance(d_start, dfn, prev["att"], att)
            if d_def <= cfg.intercept_radius:
                missed_defender.append({"step": step, "min_distance": d_def, "tau": tau_def,
                                        "endpoint_distance": float(info["defender_enemy_dist"])})
        else:
            pre_terminal = {k: v.copy() for k, v in prev.items()}
            pre_terminal["defender_start"] = dfn.copy() if info.get("defender_standby") else prev["dfn"].copy()
            if snapshot is not None and info.get("terminal_event") in ("capture", "interception"):
                # Finite mission endurance (journal extension) requires swept contact mode, so it
                # is explicitly disabled for this endpoint-semantics counterfactual branch (the
                # budget mechanism has no meaning under V1's endpoint contact model).
                snapshot.config = dataclasses.replace(
                    snapshot.config, mission_contact_mode="endpoint", enemy_endurance_enabled=False)
                _, _, _, _, cf = snapshot.step(action)
                counterfactual_record = {
                    "outcome": cf["outcome"], "terminal": bool(cf["outcome"] != "ongoing"),
                    "attacker_pos": [float(x) for x in cf["enemy_pos"]],
                    "defender_pos": [float(x) for x in cf["defender_pos"]],
                    "asset_pos": [float(x) for x in cf["soldier_pos"]],
                    "defender_enemy_dist": float(cf["defender_enemy_dist"]),
                    "enemy_soldier_dist": float(cf["enemy_soldier_dist"]),
                }
        prev = {"att": att, "ast": ast, "dfn": dfn}

    return {
        "meta": {"level": level, "semantics": semantics, "controller": controller, "seed": int(seed)},
        "config": {
            "threat_radius": cfg.threat_radius, "intercept_radius": cfg.intercept_radius,
            "unsafe_radius": cfg.unsafe_intercept_radius, "dt": dt, "L": cfg.L, "clearance": clearance,
            "max_altitude": cfg.max_altitude, "v_e": cfg.v_e, "clearance_margin": cfg.enemy_obstacle_clearance,
        },
        "rows": rows, "extras": extras, "static_obstacles": static_obstacles,
        "final_info": {k: _plain(info.get(k)) for k in FINAL_INFO_KEYS},
        "missed_attacker": missed_attacker, "missed_defender": missed_defender,
        "counterfactual": counterfactual_record, "pre_terminal": pre_terminal, "detour_cf": detour_results,
    }


def _plain(value):
    if isinstance(value, np.ndarray):
        return [float(x) for x in value]
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


# --------------------------------------------------------------- summary
def summarize(rec: dict) -> dict:
    rows, info, cfg = rec["rows"], rec["final_info"], rec["config"]
    att = np.array([r["attacker_pos"] for r in rows])
    ast = np.array([r["asset_pos"] for r in rows])
    ranges = np.linalg.norm(att - ast, axis=1)
    n = len(rows) - 1
    modes = [r["behavior_mode"] for r in rows[1:]]
    fraction = lambda m: (sum(x == m for x in modes) / n) if n else 0.0
    orbit_max, orbit_total = orbit_revolutions(att, ast)
    segments = deviation_segments(rec)
    path = float(info["attacker_path_length"])
    command_angles = []
    for r in rows[1:]:
        if r["behavior_mode"] == DIRECT and r["attacker_mission_direction"] and r["attacker_final_command"]:
            a, b = np.array(r["attacker_mission_direction"]), np.array(r["attacker_final_command"])
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom > 0:
                command_angles.append(math.degrees(math.acos(float(np.clip(np.dot(a, b) / denom, -1, 1)))))
    cf = rec["counterfactual"]
    event = info["terminal_event"]
    chord = ast[-1] - att[0]
    straight = float(np.linalg.norm(chord))
    rel2 = att[:, :2] - att[0, :2]
    d2 = chord[:2]
    nd2 = float(np.linalg.norm(d2))
    lateral = float(np.max(np.abs(d2[0] * rel2[:, 1] - d2[1] * rel2[:, 0])) / nd2) if nd2 > 1e-9 else 0.0
    return {
        **rec["meta"],
        "outcome": info["outcome"], "terminal_event": event, "steps": n,
        "captured": info["outcome"] == "soldier_caught",
        "flight_time_s": float(info["attacker_flight_time_s"]),
        "capture_time_s": float(info["attacker_flight_time_s"]) if info["outcome"] == "soldier_caught" else float("nan"),
        "path_length_m": path, "initial_range_m": float(ranges[0]), "min_range_m": float(ranges.min()),
        "final_range_m": float(ranges[-1]), "range_reduction_m": float(ranges[0] - ranges[-1]),
        "pursuit_efficiency": float((ranges[0] - ranges[-1]) / path) if path > 1e-9 else float("nan"),
        "closing_step_fraction": float(np.mean(np.diff(ranges) < 0)) if n else float("nan"),
        "heading_change_deg": float(info["attacker_heading_change_total_deg"]),
        "overshoots": int(info["attacker_overshoot_count"]),
        "overshoots_from_trace": count_overshoots(ranges, cfg["dt"]),
        "orbit_revolutions_max": orbit_max, "orbit_revolutions_total": orbit_total,
        "orbit_event": orbit_max >= ORBIT_REVOLUTION_THRESHOLD,
        "missed_attacker_contacts": len(rec["missed_attacker"]),
        "missed_defender_contacts": len(rec["missed_defender"]),
        "frac_direct": fraction(DIRECT), "frac_obstacle_avoid": fraction("OBSTACLE_AVOID"),
        "frac_route_bypass": fraction("ROUTE_BYPASS"), "frac_evade": fraction("EVADE"),
        "n_avoidance_segments": len(segments),
        "max_segment_steps": max((s["duration_steps"] for s in segments), default=0),
        "segments_line_blocked": sum(s["kind"] == "static_line_blocked" for s in segments),
        "segments_dynamic": sum(s["kind"] == "dynamic" for s in segments),
        "segments_prediction_only": sum(s["kind"] == "static_prediction_only" for s in segments),
        "segments_evasion": sum(s["kind"] == "evasion" for s in segments),
        "segments_unassociated": sum(not s["associated_with_perceived_obstacle"] for s in segments),
        "segments_not_returned": sum(s["return_to_direct_step"] is None and not s["terminated_while_avoiding"] for s in segments),
        "segments_bad_return": sum(s["line_clear_at_return"] is False for s in segments),
        "detours_necessary_collision": sum(s["necessity"] == "necessary_collision" for s in segments),
        "detours_necessary_margin": sum(s["necessity"] == "necessary_margin" for s in segments),
        "detours_unnecessary": sum(s["necessity"] == "unnecessary" for s in segments),
        "detours_not_evaluated": sum(s["necessity"] is None for s in segments),
        "max_lateral_deviation_m": lateral,
        "straight_line_distance_m": straight,
        "path_excess_ratio": float(path / straight) if straight > 1e-9 else float("nan"),
        "max_linger_run_steps": max((s["max_linger_run_steps"] for s in segments), default=0),
        "max_direct_command_error_deg": max(command_angles, default=0.0),
        "max_abs_weave_bias": max(abs(e["weave_bias"]) for e in rec["extras"]),
        "any_evasion_active": any(e["evasion_active"] for e in rec["extras"]),
        "endpoint_would_miss_terminal": (cf is not None and not cf["terminal"]),
        "counterfactual_outcome": None if cf is None else cf["outcome"],
        "min_defender_attacker_range_m": float(np.min(np.linalg.norm(
            np.array([r["defender_pos"] for r in rows]) - att, axis=1))),
    }
