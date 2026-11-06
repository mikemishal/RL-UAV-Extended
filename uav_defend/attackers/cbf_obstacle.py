"""A1-CBF static-obstacle navigation (journal_cbf_attacker, Prompt 2): AABB signed-distance/normal geometry,
acceleration-aware stopping-margin hazard construction, a fixed 32-sided velocity-bound polygon, and the
CBF-QP safety filter -- hard obstacle+velocity-bounds+acceleration-reachability safety, with defender avoidance
the only constraint class permitted a soft slack (journal_cbf_attacker, defender_cbf_v3 / Prompt 6, brief
section 11; see `solve_velocity_cbf_qp`'s docstring) -- plus a stateful route navigator that reuses the EXISTING
tested visibility-graph + A* static-obstacle planner (`uav_defend.obstacles.route_planner`).

Architecture (frozen, see docs/journal_cbf_attacker/obstacle_cbf/README.md):

    moving asset -> static route waypoint -> v_nom -> CBF-QP -> v_safe -> velocity executor

Division of responsibility (do not blur): the ROUTE PLANNER solves global/local-minimum obstacle-cluster deadlock
(it picks WHICH waypoint to steer toward); the CBF-QP provides LOCAL safety only, minimally perturbing the
requested velocity near obstacles (it never re-plans a path). Nothing here imports `SoldierEnv`; everything is a
pure function or a self-contained stateful class, independently testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from uav_defend.attackers.cbf_attacker import CBF_DT_SUB, DEFAULT_MAX_ACCEL, CBFAttackerController, apply_one_substep, substep_count
from uav_defend.attackers.cbf_defender import (
    CBF_ALPHA_DEFENDER,
    DEFENDER_SAFETY_RADIUS,
    DefenderHazard,
    build_defender_hazard,
    predict_defender_positions,
)
from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.perception import PerceivedObstacleSet, PerceivedStaticObstacle
from uav_defend.obstacles.route_planner import build_and_search_route

_EPS = 1e-9

# ----------------------------------------------------------------------------------------------- frozen constants
# Canonical obstacle clearance (matches EnvConfig.enemy_obstacle_clearance's canonical default); frozen before the
# Prompt-2 acceptance evaluation -- never tuned after seeing results (brief section 5/17).
CBF_OBSTACLE_CLEARANCE = 2.0
# CBF barrier gain alpha (s^-1); frozen (brief section 11/17).
CBF_ALPHA = 1.0
# Only inflated static AABBs within this geometric distance of the attacker become active CBF constraints
# (brief section 12). All obstacles within range are simultaneously active -- never just the single nearest one.
CBF_ACTIVE_OBSTACLE_RANGE = 20.0
# Fixed inscribed polygon used to linearize the horizontal speed-bound circle for the convex QP (brief section 14).
VELOCITY_POLYGON_SIDES = 32
# Soft-fallback slack penalty weight (brief section 16).
SOFT_SLACK_PENALTY = 1.0e4
# Route planner: reuses the SAME canonical defaults as the legacy enemy_route_planning_* config fields (frozen,
# independent of EnvConfig so this attacker's obstacle policy can never be perturbed by unrelated config changes --
# same design rationale as keeping enemy_control_mode off EnvConfig in Prompt 1).
ROUTE_PLANNING_RANGE = 40.0
ROUTE_MAX_GRAPH_NODES = 150
ROUTE_WAYPOINT_TOLERANCE = 2.0   # m; matches canonical enemy_route_waypoint_tolerance
ROUTE_REPLAN_PERIOD_S = 1.0      # brief section 7: replan every 1.0 s (every two external 0.5 s steps)
ROUTE_ASSET_MOVE_TRIGGER_M = 2.0  # brief section 7: replan if the asset moved > 2.0 m from the route's goal


# ========================================================================================= AABB signed distance
def aabb_signed_distance_and_normal(min_corner: np.ndarray, max_corner: np.ndarray, point: np.ndarray, eps: float = _EPS) -> tuple[float, np.ndarray, np.ndarray]:
    """Exact signed distance from `point` to the (inflated) axis-aligned box [min_corner, max_corner], plus the
    outward unit normal and the closest boundary point. NEVER approximates the box as a sphere/circle.

    Outside (or exactly on the boundary): `distance = ||clip(point) - point|| >= 0`, where `clip` clamps `point`
    into the box on every axis; the clamped point is automatically the correct nearest point whether one axis is
    clamped (face case), two (edge case), or three (corner case) -- the standard box-distance algorithm handles all
    three uniformly, no case-by-case branching is needed. `normal = (point - closest) / distance` (points FROM the
    box surface TOWARD the point, i.e. outward).

    Inside: `distance < 0` (penetration depth, negated), where the magnitude is the distance to the NEAREST face
    (the smallest of the six face distances); `normal` points outward through that nearest face (the direction of
    steepest/shortest escape -- the standard convention for an interior-point box normal). On an exact tie between
    two candidate exit faces, the lower-indexed axis (x before y before z) is chosen; this is a rare, deterministic
    tie-break, not a physically meaningful distinction.
    """
    p = np.asarray(point, dtype=np.float64).reshape(3)
    min_c = np.asarray(min_corner, dtype=np.float64).reshape(3)
    max_c = np.asarray(max_corner, dtype=np.float64).reshape(3)
    clipped = np.clip(p, min_c, max_c)
    outside_vec = p - clipped
    outside_dist = float(np.linalg.norm(outside_vec))
    if outside_dist > eps:
        normal = outside_vec / outside_dist
        return outside_dist, normal, clipped
    # Inside (or exactly on the boundary): penetration depth = distance to the nearest face.
    dist_to_min = p - min_c   # distance to each of the 3 "min" faces, if inside
    dist_to_max = max_c - p  # distance to each of the 3 "max" faces, if inside
    per_axis_margin = np.minimum(dist_to_min, dist_to_max)
    axis = int(np.argmin(per_axis_margin))
    depth = float(per_axis_margin[axis])
    normal = np.zeros(3, dtype=np.float64)
    exit_via_min_face = dist_to_min[axis] <= dist_to_max[axis]
    normal[axis] = -1.0 if exit_via_min_face else 1.0
    closest = p.copy()
    closest[axis] = min_c[axis] if exit_via_min_face else max_c[axis]
    return -depth, normal, closest


def inflate_aabb(obstacle: AABBObstacle, clearance: float) -> tuple[np.ndarray, np.ndarray]:
    """Returns (min_corner, max_corner) of `obstacle` inflated by `clearance` on every face."""
    return obstacle.min_corner - clearance, obstacle.max_corner + clearance


# ========================================================================================= stopping-margin hazards
@dataclass(frozen=True)
class StaticHazard:
    """One active static-obstacle CBF constraint (brief sections 9-12). `normal`/`d`/`h` are all computed against
    the INFLATED AABB; `physical_clearance` is the (uninflated) ground-truth clearance, kept only for diagnostics."""

    obstacle_id: int
    normal: np.ndarray      # outward unit normal at the closest inflated-boundary point (toward the attacker)
    d: float                # signed clearance distance to the INFLATED boundary (>0 outside, <0 inside)
    h: float                # barrier margin: d - stopping_distance_margin
    inward_speed: float     # s_i = max(0, -n_i . v_A)
    stopping_margin: float  # d_stop_i
    physical_clearance: float  # ground-truth (uninflated) clearance, for reporting only


def build_static_hazards(
    position: np.ndarray, velocity: np.ndarray, obstacles: tuple[AABBObstacle, ...], *,
    clearance: float = CBF_OBSTACLE_CLEARANCE, active_range: float = CBF_ACTIVE_OBSTACLE_RANGE,
    max_accel: float, dt_sub: float, alpha: float = CBF_ALPHA,
) -> list[StaticHazard]:
    """Builds one `StaticHazard` per inflated static AABB within `active_range` (brief section 12: ALL qualifying
    obstacles are simultaneously active, never only the nearest). Static obstacles have v_obs=0 (brief section 11),
    so the stopping margin uses only the attacker's own velocity component directed into the obstacle."""
    position = np.asarray(position, dtype=np.float64).reshape(3)
    velocity = np.asarray(velocity, dtype=np.float64).reshape(3)
    hazards: list[StaticHazard] = []
    for idx, obstacle in enumerate(obstacles):
        inflated_min, inflated_max = inflate_aabb(obstacle, clearance)
        d, normal, _closest = aabb_signed_distance_and_normal(inflated_min, inflated_max, position)
        if d > active_range:
            continue
        s = max(0.0, float(-np.dot(normal, velocity)))
        d_stop = (s * s) / (2.0 * max_accel) + s * dt_sub
        h = d - d_stop
        physical_clearance = obstacle.clearance_to_point(position)
        hazards.append(StaticHazard(
            obstacle_id=idx, normal=normal, d=d, h=h, inward_speed=s, stopping_margin=d_stop,
            physical_clearance=physical_clearance,
        ))
    return hazards


# ========================================================================================= initial-state viability (Prompt 3)
def initial_state_dynamically_viable(
    position: np.ndarray, velocity: np.ndarray, obstacles: tuple[AABBObstacle, ...], *,
    clearance: float = CBF_OBSTACLE_CLEARANCE, max_accel: float, dt_sub: float = CBF_DT_SUB,
) -> dict:
    """Initial-state viability test (journal_cbf_attacker, obstacle_cbf_v2 / Prompt 3): is `(position, velocity)`
    dynamically reachable WITHOUT a guaranteed collision, given the frozen acceleration limit alone?

    Tests the SAME stopping-margin barrier used by the live CBF (`build_static_hazards`) against EVERY static
    obstacle (brief section 4: no active-range cutoff here -- unlike in-flight hazard selection, the initial-state
    check must consider every obstacle in the layout, since the attacker has not yet had any chance to react to
    anything). The state is dynamically viable iff `d_i >= d_req,i` for EVERY obstacle, i.e. `h_i >= 0` for every
    hazard (no additional arbitrary buffer, per the brief's explicit instruction) -- `alpha` never enters this
    computation (`h_i = d_i - d_req,i` is alpha-independent; the CBF-QP's `alpha` only shapes how aggressively a
    LIVE, in-flight constraint is enforced, not whether the STARTING state is physically reachable at all).

    Returns `{"viable": bool, "min_margin": float (min h_i across all obstacles, or +inf if none), "hazards": [...]}`.
    """
    hazards = build_static_hazards(position, velocity, obstacles, clearance=clearance, active_range=float("inf"), max_accel=max_accel, dt_sub=dt_sub)
    viable = all(h.h >= 0.0 for h in hazards)
    min_margin = min((h.h for h in hazards), default=float("inf"))
    return {"viable": bool(viable), "min_margin": float(min_margin), "hazards": hazards}


# ========================================================================================= velocity-bound polygon
def velocity_polygon_halfplanes(radius: float, n_sides: int = VELOCITY_POLYGON_SIDES) -> tuple[np.ndarray, float]:
    """A fixed `n_sides`-sided polygon INSCRIBED in the horizontal speed-bound circle of radius `radius` (i.e. the
    polygon lies entirely WITHIN the disk -- a conservative inner approximation, never exceeding the true physical
    bound). Returns `(unit_normals, apothem)` where each row of `unit_normals` is `(cos(theta_k), sin(theta_k))` and
    the half-plane constraint is `unit_normals[k] . (vx, vy) <= apothem`.

    Conservatism: the maximum gap between the polygon boundary and the true circle (at each edge's midpoint) is
    `radius * (1 - cos(pi / n_sides))`; for n_sides=32 this is ~0.058 m/s (0.48% of a 12 m/s bound) -- negligible
    relative to the physical executor's own saturation behavior, and the EXACT physical clip in
    `integrate_quadrotor_substeps` remains the final numerical guard regardless (brief section 14)."""
    apothem = radius * math.cos(math.pi / n_sides)
    angles = 2.0 * math.pi * np.arange(n_sides) / n_sides
    normals = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return normals, apothem


def velocity_polygon_max_conservatism(radius: float, n_sides: int = VELOCITY_POLYGON_SIDES) -> float:
    """The maximum conservatism (m/s) introduced by the inscribed polygon approximation (see
    `velocity_polygon_halfplanes`'s docstring) -- documented explicitly per brief section 14."""
    return radius * (1.0 - math.cos(math.pi / n_sides))


# ========================================================================================= CBF-QP safety filter
def _cbf_constraint_funcs(hazards: list[StaticHazard], alpha: float, slack: bool) -> list:
    funcs = []
    for hz in hazards:
        n, h = hz.normal, hz.h
        if slack:
            funcs.append(lambda v, n=n, h=h: float(np.dot(n, v[:3])) + alpha * h + v[3])
        else:
            funcs.append(lambda v, n=n, h=h: float(np.dot(n, v[:3])) + alpha * h)
    return funcs


def _defender_constraint_funcs(defender_hazard, alpha_defender: float, slack: bool) -> list:
    """Defender-avoidance constraint (journal_cbf_attacker, defender_cbf / Prompt 4, brief section 8):
    `n_D^T (v_safe - v_D) >= -alpha_D h_D`  <=>  `n_D^T v_safe >= n_D^T v_D - alpha_D h_D`. Returns [] when
    `defender_hazard` is None (no observable defender this step -- the joint QP then degenerates exactly to the
    Prompt-2/3 obstacle-only QP, bit-for-bit)."""
    if defender_hazard is None:
        return []
    n, h, v_d = defender_hazard.normal, defender_hazard.h, defender_hazard.defender_velocity
    rhs = float(np.dot(n, v_d)) - alpha_defender * h
    if slack:
        return [lambda v, n=n, rhs=rhs: float(np.dot(n, v[:3])) - rhs + v[3]]
    return [lambda v, n=n, rhs=rhs: float(np.dot(n, v[:3])) - rhs]


def _velocity_bound_funcs(max_horizontal_speed: float, max_climb_rate: float, max_descent_rate: float, n_sides: int) -> list:
    normals, apothem = velocity_polygon_halfplanes(max_horizontal_speed, n_sides)
    funcs = []
    for nx, ny in normals:
        funcs.append(lambda v, nx=nx, ny=ny: apothem - (nx * v[0] + ny * v[1]))
    funcs.append(lambda v: v[2] + max_descent_rate)
    funcs.append(lambda v: max_climb_rate - v[2])
    return funcs


def _clip_initial_guess(v_nom: np.ndarray, max_horizontal_speed: float, max_climb_rate: float, max_descent_rate: float) -> np.ndarray:
    v = np.asarray(v_nom, dtype=np.float64).copy()
    horiz = v[:2]
    horiz_speed = float(np.linalg.norm(horiz))
    if horiz_speed > max_horizontal_speed and horiz_speed > _EPS:
        v[:2] = horiz / horiz_speed * max_horizontal_speed
    v[2] = float(np.clip(v[2], -max_descent_rate, max_climb_rate))
    return v


def _max_residual(funcs, v) -> float:
    return max((max(0.0, -f(v)) for f in funcs), default=0.0)


def _classify_intervention_source(v_nom: np.ndarray, hazards: list[StaticHazard], defender_hazard, alpha: float, alpha_defender: float) -> str:
    """Classifies WHY the QP would need to intervene, evaluated at the RAW nominal command v_nom (before solving):
    a constraint class is "active" iff v_nom itself already violates at least one of its constraints. Brief
    section 24: one of "none" / "obstacle" / "defender" / "joint"."""
    v3 = np.asarray(v_nom, dtype=np.float64).reshape(3)
    obstacle_active = any(f(v3) < 0.0 for f in _cbf_constraint_funcs(hazards, alpha, slack=False))
    defender_active = any(f(v3) < 0.0 for f in _defender_constraint_funcs(defender_hazard, alpha_defender, slack=False))
    if obstacle_active and defender_active:
        return "joint"
    if obstacle_active:
        return "obstacle"
    if defender_active:
        return "defender"
    return "none"


def _accel_reachable_func(v_current: np.ndarray, max_accel: float, dt_sub: float):
    """Exact Euclidean acceleration-reachability constraint (journal_cbf_attacker, defender_cbf_v3 / Prompt 6,
    brief sections 3-4): `(a_max*dt_sub)^2 - ||v - v_current||^2 >= 0`. HARD, never relaxed -- matches the
    EXECUTOR's own norm-based acceleration limiter (`apply_one_substep`'s `delta_v = v_cmd - vel; clip
    ||delta_v|| <= max_accel*dt_sub`, preserving direction) EXACTLY, so a velocity the optimizer certifies safe is
    the SAME velocity the executor actually applies (brief section 9's executor contract). This is a QUADRATIC
    (not linear) inequality in `v` -- SLSQP handles general smooth nonlinear constraints directly; see this
    module's terminology note in `solve_velocity_cbf_qp`'s docstring."""
    v_current = np.asarray(v_current, dtype=np.float64).reshape(3)
    limit_sq = (max_accel * dt_sub) ** 2

    def f(v):
        diff = np.asarray(v[:3], dtype=np.float64) - v_current
        return limit_sq - float(np.dot(diff, diff))

    return f


def _reachable_initial_guess(
    v_nom: np.ndarray, v_current: np.ndarray, max_accel: float, dt_sub: float,
    max_horizontal_speed: float, max_climb_rate: float, max_descent_rate: float,
) -> np.ndarray:
    """Deterministic initial guess for SLSQP that is ITSELF acceleration-reachable (brief section 10): project
    `v_nom` onto the current acceleration ball around `v_current` FIRST, THEN apply the existing velocity-limit
    projection (`_clip_initial_guess`) -- never feed SLSQP a raw, physically-unreachable nominal velocity as x0."""
    v_nom = np.asarray(v_nom, dtype=np.float64).reshape(3)
    v_current = np.asarray(v_current, dtype=np.float64).reshape(3)
    delta = v_nom - v_current
    delta_norm = float(np.linalg.norm(delta))
    limit = max_accel * dt_sub
    if delta_norm > limit and delta_norm > _EPS:
        delta = delta / delta_norm * limit
    v_init = v_current + delta
    return _clip_initial_guess(v_init, max_horizontal_speed, max_climb_rate, max_descent_rate)


def _reachable_emergency_brake(v_current: np.ndarray, max_accel: float, dt_sub: float) -> np.ndarray:
    """Maximum physically reachable deceleration toward zero velocity for ONE substep (journal_cbf_attacker,
    defender_cbf_v3 / Prompt 6, brief section 12) -- replaces any previous emergency behavior that requested a
    physically unreachable instantaneous hover. `v_brake = v_current - min(a_max*dt_sub, ||v_current||) *
    v_current / ||v_current||` for nonzero `v_current`; exactly zero only when `v_current` is already (near)
    zero. The caller still applies the existing hard velocity-limit projection on top of this (brief section 12:
    "Then apply physical velocity limits.")."""
    v_current = np.asarray(v_current, dtype=np.float64).reshape(3)
    speed = float(np.linalg.norm(v_current))
    if speed <= _EPS:
        return np.zeros(3, dtype=np.float64)
    step = min(max_accel * dt_sub, speed)
    return v_current - step * (v_current / speed)


def solve_velocity_cbf_qp(
    v_nom: np.ndarray, hazards: list[StaticHazard], *, v_current: np.ndarray, defender_hazard=None,
    alpha: float = CBF_ALPHA, alpha_defender: float = CBF_ALPHA_DEFENDER,
    max_horizontal_speed: float, max_climb_rate: float, max_descent_rate: float,
    max_accel: float = DEFAULT_MAX_ACCEL, dt_sub: float = CBF_DT_SUB,
    n_sides: int = VELOCITY_POLYGON_SIDES, maxiter: int = 200, ftol: float = 1e-12, residual_tol: float = 1e-4,
) -> tuple[np.ndarray, dict]:
    """ACCELERATION-REACHABLE, hierarchical obstacle+defender CBF velocity-CONSTRAINED optimization
    (journal_cbf_attacker, defender_cbf_v3 / Prompt 6). Terminology note (brief section 4): because the
    acceleration-reachability constraint below is an exact Euclidean-norm (quadratic, not linear) inequality,
    this solver is more precisely a "CBF constrained velocity optimization" over a convex feasible region, not a
    strictly linear-constraint QP at every stage -- the objective remains quadratic and the problem remains
    convex in the intended feasible region throughout.

    Corrects a physical-consistency gap discovered in Prompt 5 (journal_cbf_attacker, defender_cbf_v2): the
    optimizer selected a velocity satisfying every CBF inequality in velocity space with no awareness of the
    vehicle's SEPARATE, downstream acceleration-rate limit -- the executor then silently clipped that velocity,
    so the certified-safe `v_safe` and the ACTUALLY EXECUTED velocity could differ, invalidating the safety
    certificate for the motion actually flown (two persistent Prompt-5 stress collisions, seeds 630123/630203).
    This adds the exact one-substep acceleration-reachable set directly into the optimization as a HARD
    constraint (brief section 3): `||v - v_current||^2 <= (a_max*dt_sub)^2`, matching the EXECUTOR's own
    norm-based clip exactly (no independent per-axis approximation; brief section 4). Given a feasible solve,
    `v_optimized == v_executed` to within numerical tolerance (brief section 9) -- verified downstream by
    `integrate_obstacle_cbf_substeps`'s `executor_velocity_error` diagnostic.

    FROZEN priority (brief section 2 -- acceleration-reachability now JOINS obstacle safety and velocity limits
    as an always-hard, never-relaxed "physical safety" class; only this changes relative to Prompt 5):
        Priority 1 (HARD, never relaxed): static-obstacle safety.
        Priority 2 (HARD, never relaxed): physical velocity limits AND one-substep acceleration-reachability.
        Priority 3 (SOFT): defender avoidance, via a dedicated non-negative slack `delta_D`.
        Objective:  minimize ||v - v_nom||^2 (+ 1e4*delta_D^2 whenever delta_D is part of the solve).

    Step 1 solves physical safety alone (obstacle + velocity bounds + acceleration-reachability, ALL hard),
    entirely ignoring any defender (brief sections 5/11) -- infeasible here is a genuine physical crisis, NEVER a
    defender-avoidance side effect; the defender is dropped entirely and classified `physical_safety_infeasible`,
    returning the REACHABLE emergency brake (brief section 12) -- never a physically-unreachable instantaneous
    hover (this corrects every prior emergency-brake code path's `v=0` return). If physical safety is feasible and
    no defender is observable, that Step-1 solution is returned directly. Otherwise Step 2a tries obstacle +
    acceleration + bounds + defender ALL hard (the common, no-conflict case, warm-started from Step 1); on
    infeasibility (a genuine priority conflict) Step 2b solves the hierarchical soft QP over `[vx, vy, vz,
    delta_D]` with ONLY the defender constraint relaxed -- physical safety (obstacle/bounds/acceleration) is
    NEVER relaxed to help the defender term, matching Prompt 5's established rule exactly, now extended to
    acceleration-reachability. New diagnostics (brief section 21): `optimizer_accel_request`,
    `optimizer_accel_norm`, `accel_constraint_active`, `accel_reachable_margin`, `reachable_emergency_brake`,
    `physical_safety_infeasible`-tagged `priority_classification`. (`executor_velocity_error` and
    `optimizer_executor_contract_violation` are computed one level up, in `integrate_obstacle_cbf_substeps`, since
    this function has no visibility into the executor's own separate clipping.)
    """
    v_nom = np.asarray(v_nom, dtype=np.float64).reshape(3)
    v_current = np.asarray(v_current, dtype=np.float64).reshape(3)
    bound_funcs = _velocity_bound_funcs(max_horizontal_speed, max_climb_rate, max_descent_rate, n_sides)
    accel_func = _accel_reachable_func(v_current, max_accel, dt_sub)
    x0 = _reachable_initial_guess(
        v_nom, v_current, max_accel, dt_sub, max_horizontal_speed, max_climb_rate, max_descent_rate,
    )
    intervention_source = _classify_intervention_source(v_nom, hazards, defender_hazard, alpha, alpha_defender)
    accel_request = v_nom - v_current
    accel_request_norm = float(np.linalg.norm(accel_request))

    def objective3(v):
        diff = v[:3] - v_nom
        return 0.5 * float(np.dot(diff, diff))

    def _accel_diagnostics(v_point: np.ndarray) -> dict:
        residual = max(0.0, -accel_func(v_point))
        return {
            "optimizer_accel_request": accel_request.tolist(), "optimizer_accel_norm": accel_request_norm,
            "accel_constraint_active": bool(residual > residual_tol),
            "accel_reachable_margin": -residual,
        }

    # ---- Step 1: physical safety alone (obstacle + velocity bounds + acceleration-reachability), ALWAYS hard,
    # defender entirely ignored (brief sections 5/11: a pure physical-crisis test, independent of any tactical
    # consideration). ----
    obstacle_funcs_hard = _cbf_constraint_funcs(hazards, alpha, slack=False)
    physical_hard_funcs = obstacle_funcs_hard + bound_funcs + [accel_func]
    physical_constraints = [{"type": "ineq", "fun": f} for f in physical_hard_funcs]
    physical_res = minimize(objective3, x0, method="SLSQP", constraints=physical_constraints, options={"maxiter": maxiter, "ftol": ftol})
    physical_residual = _max_residual(physical_hard_funcs, physical_res.x)
    obstacle_residual_physical = _max_residual(obstacle_funcs_hard, physical_res.x)

    if physical_residual > residual_tol:
        # Priority 1/2 (obstacle safety, velocity bounds, acceleration-reachability) cannot be satisfied
        # regardless of the defender -- NEVER relaxed to chase defender avoidance (brief section 11). Command the
        # maximum physically reachable braking velocity instead (brief section 12), never an unreachable hover.
        v_brake = _reachable_emergency_brake(v_current, max_accel, dt_sub)
        v_brake = _clip_initial_guess(v_brake, max_horizontal_speed, max_climb_rate, max_descent_rate)
        obstacle_residual_brake = _max_residual(obstacle_funcs_hard, v_brake)
        return v_brake, {
            "cbf_qp_status": "emergency_brake", "qp_success": False, "qp_iterations": int(physical_res.nit),
            "qp_objective": float("nan"), "qp_max_constraint_residual": physical_residual, "soft_slack": 0.0,
            "obstacle_constraint_residual": obstacle_residual_brake, "defender_constraint_residual": 0.0,
            "intervention_source": intervention_source, "infeasibility_source": "obstacle",
            "priority_classification": "physical_safety_infeasible", "obstacle_constraints_hard": True,
            "physical_safety_infeasible": True, "reachable_emergency_brake": True,
            "defender_slack": 0.0, "defender_constraint_satisfied_without_slack": True,
            "defender_constraint_violation_with_slack": 0.0,
            "obstacle_min_residual": obstacle_residual_brake, "defender_min_residual": 0.0,
            "priority_conflict": False, "defender_ignored_for_obstacle_safety": defender_hazard is not None,
            **_accel_diagnostics(v_brake),
        }

    if defender_hazard is None:
        v_safe = physical_res.x[:3]
        return v_safe, {
            "cbf_qp_status": "hard", "qp_success": bool(physical_res.success), "qp_iterations": int(physical_res.nit),
            "qp_objective": float(objective3(physical_res.x)), "qp_max_constraint_residual": physical_residual,
            "soft_slack": 0.0, "obstacle_constraint_residual": obstacle_residual_physical,
            "defender_constraint_residual": 0.0, "intervention_source": intervention_source,
            "infeasibility_source": None, "priority_classification": None, "obstacle_constraints_hard": True,
            "physical_safety_infeasible": False, "reachable_emergency_brake": False,
            "defender_slack": 0.0, "defender_constraint_satisfied_without_slack": True,
            "defender_constraint_violation_with_slack": 0.0,
            "obstacle_min_residual": obstacle_residual_physical, "defender_min_residual": 0.0,
            "priority_conflict": False, "defender_ignored_for_obstacle_safety": False,
            **_accel_diagnostics(physical_res.x),
        }

    # ---- Step 2a: try ALL constraints hard (obstacle + acceleration + bounds + defender) -- the common,
    # no-conflict case. A dedicated hard-only (3-dim, no slack variable) solve first, exactly mirroring the
    # established hard-first pattern used everywhere else in this module (Prompt 5's rationale, unchanged):
    # introducing the slack variable into EVERY solve (even when zero slack is optimal) makes the QP numerically
    # harder for SLSQP to reliably converge to delta_D=0 from a cold start. Warm-started from Step 1's OWN
    # physical-safety-feasible optimum, keeping this hard solve consistent with Priority 1/2.
    defender_funcs_hard = _defender_constraint_funcs(defender_hazard, alpha_defender, slack=False)
    all_hard_funcs = physical_hard_funcs + defender_funcs_hard
    all_hard_constraints = [{"type": "ineq", "fun": f} for f in all_hard_funcs]
    hard_res = minimize(objective3, physical_res.x, method="SLSQP", constraints=all_hard_constraints, options={"maxiter": maxiter, "ftol": ftol})
    all_hard_residual = _max_residual(all_hard_funcs, hard_res.x)
    if all_hard_residual <= residual_tol:
        v_safe = hard_res.x[:3]
        obstacle_residual = _max_residual(obstacle_funcs_hard, hard_res.x)
        defender_residual = _max_residual(defender_funcs_hard, hard_res.x)
        return v_safe, {
            "cbf_qp_status": "hard", "qp_success": bool(hard_res.success), "qp_iterations": int(hard_res.nit),
            "qp_objective": float(objective3(hard_res.x)), "qp_max_constraint_residual": all_hard_residual,
            "soft_slack": 0.0, "obstacle_constraint_residual": obstacle_residual,
            "defender_constraint_residual": defender_residual,
            "intervention_source": intervention_source, "infeasibility_source": None,
            "priority_classification": None, "obstacle_constraints_hard": True,
            "physical_safety_infeasible": False, "reachable_emergency_brake": False,
            "defender_slack": 0.0, "defender_constraint_satisfied_without_slack": True,
            "defender_constraint_violation_with_slack": defender_residual,
            "obstacle_min_residual": obstacle_residual, "defender_min_residual": defender_residual,
            "priority_conflict": False, "defender_ignored_for_obstacle_safety": False,
            **_accel_diagnostics(hard_res.x),
        }

    # ---- Step 2b: all-hard infeasible -- a genuine priority conflict. Obstacle+bounds+acceleration stay HARD;
    # ONLY the defender constraint is permitted to relax, via the dedicated non-negative slack `delta_D` (brief
    # section 4). ----
    defender_funcs_soft = _defender_constraint_funcs(defender_hazard, alpha_defender, slack=True)
    physical_hard_funcs_4d = [lambda v, f=f: f(v[:3]) for f in physical_hard_funcs]
    primary_funcs = physical_hard_funcs_4d + defender_funcs_soft

    def primary_objective(v):
        diff = v[:3] - v_nom
        delta = v[3]
        return 0.5 * float(np.dot(diff, diff)) + SOFT_SLACK_PENALTY * delta * delta

    x0_primary = np.concatenate([physical_res.x[:3], [0.0]])
    primary_constraints = [{"type": "ineq", "fun": f} for f in primary_funcs]
    primary_res = minimize(
        primary_objective, x0_primary, method="SLSQP", constraints=primary_constraints,
        bounds=[(None, None)] * 3 + [(0.0, None)], options={"maxiter": maxiter, "ftol": ftol},
    )
    primary_residual = _max_residual(primary_funcs, primary_res.x)
    obstacle_residual = _max_residual(obstacle_funcs_hard, primary_res.x)
    defender_hard_residual = _max_residual(defender_funcs_hard, primary_res.x)
    delta_d = float(primary_res.x[3])

    if primary_residual <= residual_tol:
        v_safe = primary_res.x[:3]
        status = "hard" if delta_d <= residual_tol else "soft_fallback"
        infeasibility_source = None if status == "hard" else "defender"
        priority_conflict = bool(delta_d > residual_tol)
        return v_safe, {
            "cbf_qp_status": status, "qp_success": bool(primary_res.success), "qp_iterations": int(primary_res.nit),
            "qp_objective": float(primary_objective(primary_res.x)), "qp_max_constraint_residual": primary_residual,
            "soft_slack": delta_d, "obstacle_constraint_residual": obstacle_residual,
            "defender_constraint_residual": defender_hard_residual,
            "intervention_source": intervention_source, "infeasibility_source": infeasibility_source,
            "priority_classification": None, "obstacle_constraints_hard": True,
            "physical_safety_infeasible": False, "reachable_emergency_brake": False,
            "defender_slack": delta_d,
            "defender_constraint_satisfied_without_slack": bool(defender_hard_residual <= residual_tol),
            "defender_constraint_violation_with_slack": defender_hard_residual,
            "obstacle_min_residual": obstacle_residual, "defender_min_residual": defender_hard_residual,
            "priority_conflict": priority_conflict, "defender_ignored_for_obstacle_safety": False,
            **_accel_diagnostics(primary_res.x),
        }

    # ---- Defensive numerical safety net (this QP is provably feasible whenever Step 1 passed, since delta_D has
    # no upper bound -- should not occur in practice; kept only against SLSQP non-convergence pathologies).
    # Physical safety is STILL never relaxed: the REACHABLE emergency brake, never an invented escape direction
    # and never a physically-unreachable instantaneous hover. ----
    v_brake = _reachable_emergency_brake(v_current, max_accel, dt_sub)
    v_brake = _clip_initial_guess(v_brake, max_horizontal_speed, max_climb_rate, max_descent_rate)
    obstacle_residual_brake = _max_residual(obstacle_funcs_hard, v_brake)
    return v_brake, {
        "cbf_qp_status": "emergency_brake", "qp_success": False, "qp_iterations": int(primary_res.nit),
        "qp_objective": float("nan"), "qp_max_constraint_residual": primary_residual,
        "soft_slack": delta_d, "obstacle_constraint_residual": obstacle_residual_brake,
        "defender_constraint_residual": defender_hard_residual,
        "intervention_source": intervention_source, "infeasibility_source": "numerical",
        "priority_classification": "numerical_fallback", "obstacle_constraints_hard": True,
        "physical_safety_infeasible": False, "reachable_emergency_brake": True,
        "defender_slack": delta_d, "defender_constraint_satisfied_without_slack": False,
        "defender_constraint_violation_with_slack": defender_hard_residual,
        "obstacle_min_residual": obstacle_residual_brake, "defender_min_residual": defender_hard_residual,
        "priority_conflict": bool(delta_d > residual_tol), "defender_ignored_for_obstacle_safety": False,
        **_accel_diagnostics(v_brake),
    }


# ========================================================================================= safety-critical executor
def integrate_obstacle_cbf_substeps(
    position: np.ndarray, velocity: np.ndarray, *, route_target: np.ndarray | None, controller: CBFAttackerController,
    obstacles: tuple[AABBObstacle, ...], dt: float, dt_sub: float = CBF_DT_SUB, L: float, max_altitude: float,
    eps: float = 1e-8, clearance: float = CBF_OBSTACLE_CLEARANCE, active_range: float = CBF_ACTIVE_OBSTACLE_RANGE,
    alpha: float = CBF_ALPHA, defender_pos: np.ndarray | None = None, defender_vel_estimate: np.ndarray | None = None,
    defender_safety_radius: float | None = None, alpha_defender: float = CBF_ALPHA_DEFENDER,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Safety-critical variant of `cbf_attacker.integrate_quadrotor_substeps`: re-solves the nominal-guidance +
    joint obstacle/defender CBF-QP safety filter EVERY substep (dt_sub, default 0.1 s), instead of holding one
    command constant across all 5 substeps of the external 0.5 s step.

    WHY this differs from Prompt 1's "hold the command constant across substeps" rule: the stopping-margin barrier
    (`build_static_hazards`'s `d_stop_i = s_i^2/(2 a_max) + s_i * dt_sub`) is explicitly derived assuming the
    attacker's velocity can be corrected again within `dt_sub` (brief section 10 -- `dt_sub` is the reaction-time
    term). That assumption is only true if the safety filter is ACTUALLY re-evaluated at that cadence; holding
    `v_safe` fixed for a full 0.5 s (5x `dt_sub`) while the executor's OWN acceleration limit (6 m/s^2) often cannot
    reach a large commanded deceleration within even one 0.5 s step creates a genuine, measurable safety gap --
    found empirically during development (a 200-episode C2 smoke test produced ONE static-obstacle collision; a
    high-closing-speed attacker's velocity command changed by ~2x more than the 5-substep acceleration budget could
    deliver before the obstacle was reached). Re-solving every substep makes the formula's own reaction-time
    assumption true by construction, which eliminates the collision (verified: the exact failing seed, replayed
    with this executor, no longer collides -- see the dedicated regression test). This is a FIX to the control-loop
    timing, not a tuning of `alpha`, `kp`, the clearance, or any physical limit (all remain exactly as frozen).

    The guidance TARGET (`route_target`) itself is NOT recomputed here -- it is supplied once per external step by
    the (slower-cadence, brief section 7) route planner, matching the division of responsibility in brief section
    3: the route planner solves global deadlock; the CBF-QP provides only local, per-instant safety. `route_target`
    may be `None` (journal_cbf_attacker, obstacle_cbf_v2 / Prompt 3, brief section 9: `ROUTE_TEMPORARILY_UNAVAILABLE`
    with no previously retained waypoint) -- in that case the nominal command is `v_nom = 0` every substep (hover/
    brake), never an invented escape direction; the CBF-QP may still intervene if a hazard demands outward motion.

    Defender avoidance (journal_cbf_attacker, defender_cbf / Prompt 4, brief section 9): when `defender_pos` is
    provided (the defender was observable -- gated position -- at the START of this external step),
    `defender_vel_estimate` (the finite-difference ESTIMATE, refreshed only once per external step, never
    repredicted mid-step) is used to extrapolate the defender's position at CONSTANT velocity for each substep
    (`predict_defender_positions`), and a fresh `DefenderHazard` is built every substep from the attacker's
    CURRENT substep state and the PREDICTED defender position. No defender acceleration is predicted. When
    `defender_pos` is None (not observable this external step), no defender constraint exists at all -- the joint
    QP degenerates exactly to the Prompt-2/3 obstacle-only QP.

    Returns `(position, velocity, exec_diagnostics, cbf_diagnostics)` where `exec_diagnostics` has the SAME schema
    as `integrate_quadrotor_substeps` (accel/speed/climb saturation flags) and `cbf_diagnostics` holds the LAST
    substep's CBF-QP diagnostics plus aggregates (`emergency_brake_substeps`, `min_h_over_substeps`,
    `max_active_obstacle_count`, `min_defender_h_over_substeps`, `max_defender_closing_speed`) over all 5 substeps.
    """
    pos = np.asarray(position, dtype=np.float64).copy()
    vel = np.asarray(velocity, dtype=np.float64).copy()
    route_target = np.asarray(route_target, dtype=np.float64).reshape(3) if route_target is not None else None
    n = substep_count(dt, dt_sub)
    limits = controller.limits
    safety_radius = defender_safety_radius if defender_safety_radius is not None else DEFENDER_SAFETY_RADIUS
    defender_predicted_positions = (
        predict_defender_positions(defender_pos, defender_vel_estimate, dt_sub=dt_sub, n_substeps=n)
        if defender_pos is not None else None
    )

    max_accel_used = 0.0
    accel_saturated = horizontal_saturated = vertical_saturated = False
    emergency_brake_substeps = 0
    min_h_over_substeps = float("inf")
    min_d_over_substeps = float("inf")
    max_active_obstacle_count = 0
    min_defender_h_over_substeps = float("inf")
    max_defender_closing_speed = 0.0
    defender_active_substeps = 0
    simultaneous_obstacle_defender_substeps = 0
    max_defender_slack_over_substeps = 0.0
    priority_conflict_substeps = 0
    obstacle_safety_infeasible_substeps = 0
    physical_safety_infeasible_substeps = 0
    reachable_emergency_brake_substeps = 0
    accel_constraint_active_substeps = 0
    max_executor_velocity_error_over_substeps = 0.0
    optimizer_executor_contract_violation_substeps = 0
    last_cbf_diag: dict = {}
    last_defender_hazard: DefenderHazard | None = None
    last_v_nom = np.zeros(3)
    last_v_safe = np.zeros(3)

    for substep_idx in range(n):
        if route_target is None:
            v_nom = np.zeros(3, dtype=np.float64)   # hover/brake (brief section 9): no target available this step
        else:
            r_g = route_target - pos
            v_nom = controller.nominal_velocity(r_g)
        hazards = build_static_hazards(
            pos, vel, obstacles, clearance=clearance, active_range=active_range, max_accel=limits.max_accel,
            dt_sub=dt_sub, alpha=alpha,
        )
        defender_hazard = None
        if defender_predicted_positions is not None:
            defender_hazard = build_defender_hazard(
                pos, vel, defender_predicted_positions[substep_idx], defender_vel_estimate,
                safety_radius=safety_radius, max_accel=limits.max_accel, dt_sub=dt_sub,
            )
            min_defender_h_over_substeps = min(min_defender_h_over_substeps, defender_hazard.h)
            max_defender_closing_speed = max(max_defender_closing_speed, defender_hazard.closing_speed)
            defender_active_substeps += 1
            if hazards:
                simultaneous_obstacle_defender_substeps += 1
        v_safe, cbf_diag = controller.filter_velocity(
            v_nom, state={"position": pos.copy(), "velocity": vel.copy()}, hazards=tuple(hazards), defender_hazard=defender_hazard,
        )
        if cbf_diag["cbf_qp_status"] == "emergency_brake":
            emergency_brake_substeps += 1
        if hazards:
            min_h_over_substeps = min(min_h_over_substeps, min(h.h for h in hazards))
            min_d_over_substeps = min(min_d_over_substeps, min(h.d for h in hazards))
        max_active_obstacle_count = max(max_active_obstacle_count, len(hazards))
        max_defender_slack_over_substeps = max(max_defender_slack_over_substeps, float(cbf_diag.get("defender_slack", 0.0)))
        if cbf_diag.get("priority_conflict", False):
            priority_conflict_substeps += 1
        if cbf_diag.get("priority_classification") == "obstacle_safety_infeasible":
            obstacle_safety_infeasible_substeps += 1
        if cbf_diag.get("physical_safety_infeasible", False):
            physical_safety_infeasible_substeps += 1
        if cbf_diag.get("reachable_emergency_brake", False):
            reachable_emergency_brake_substeps += 1
        if cbf_diag.get("accel_constraint_active", False):
            accel_constraint_active_substeps += 1

        pos, vel, sub_diag = apply_one_substep(pos, vel, v_safe, limits=limits, dt_sub=dt_sub, L=L, max_altitude=max_altitude, eps=eps)
        # ---- Optimizer/executor contract (journal_cbf_attacker, defender_cbf_v3 / Prompt 6, brief section 9):
        # because `v_safe` is already acceleration-reachable (hard-constrained in `solve_velocity_cbf_qp`), the
        # executor's OWN separate acceleration clip should make NO material modification -- verify this here,
        # where both the QP output (`v_safe`) and the ACTUAL post-executor velocity are both in scope. Only
        # meaningful when `solve_velocity_cbf_qp` actually ran this substep (the accel-ball constraint is part of
        # THAT solve): when `filter_velocity` took its identity shortcut instead (no hazards/defender active for
        # THIS specific substep -- a legitimate occurrence mid-step, e.g. right after a hazard-triggered maneuver
        # moves the attacker back out of range), `v_safe` is the raw `v_nom` with NO reachability guarantee at
        # all, and the executor's ordinary acceleration-rate convergence toward it (identical to Prompt 1's
        # always-has-been behavior) must never be miscounted as a Prompt-6 contract violation. ----
        if cbf_diag["cbf_qp_status"] != "identity_no_constraints":
            executor_velocity_error = float(np.linalg.norm(vel - v_safe))
            max_executor_velocity_error_over_substeps = max(max_executor_velocity_error_over_substeps, executor_velocity_error)
            if executor_velocity_error > 1e-6:
                optimizer_executor_contract_violation_substeps += 1
        max_accel_used = max(max_accel_used, sub_diag["accel_used"])
        accel_saturated = accel_saturated or sub_diag["accel_saturated"]
        horizontal_saturated = horizontal_saturated or sub_diag["horizontal_speed_saturated"]
        vertical_saturated = vertical_saturated or sub_diag["vertical_rate_saturated"]
        last_cbf_diag = cbf_diag
        last_defender_hazard = defender_hazard
        last_v_nom = v_nom
        last_v_safe = v_safe

    exec_diag = {
        "accel_used": max_accel_used, "turn_rate_used_deg": 0.0, "accel_saturated": bool(accel_saturated),
        "turn_saturated": False, "horizontal_speed_saturated": bool(horizontal_saturated),
        "vertical_rate_saturated": bool(vertical_saturated), "climb_saturated": bool(vertical_saturated),
    }
    cbf_diagnostics = {
        **last_cbf_diag, "v_nom_last_substep": last_v_nom, "v_safe_last_substep": last_v_safe,
        "min_d_over_substeps": min_d_over_substeps,
        "emergency_brake_substeps": emergency_brake_substeps, "min_h_over_substeps": min_h_over_substeps,
        "max_active_obstacle_count": max_active_obstacle_count,
        "min_defender_h_over_substeps": min_defender_h_over_substeps, "max_defender_closing_speed": max_defender_closing_speed,
        "defender_constraint_present": defender_predicted_positions is not None,
        "defender_active_substeps": defender_active_substeps,
        "simultaneous_obstacle_defender_substeps": simultaneous_obstacle_defender_substeps,
        "last_defender_distance": (last_defender_hazard.distance if last_defender_hazard is not None else float("inf")),
        "last_defender_closing_speed": (last_defender_hazard.closing_speed if last_defender_hazard is not None else 0.0),
        "last_defender_stopping_margin": (last_defender_hazard.stopping_margin if last_defender_hazard is not None else 0.0),
        "last_defender_h": (last_defender_hazard.h if last_defender_hazard is not None else float("inf")),
        "max_defender_slack_over_substeps": max_defender_slack_over_substeps,
        "priority_conflict_substeps": priority_conflict_substeps,
        "obstacle_safety_infeasible_substeps": obstacle_safety_infeasible_substeps,
        "physical_safety_infeasible_substeps": physical_safety_infeasible_substeps,
        "reachable_emergency_brake_substeps": reachable_emergency_brake_substeps,
        "accel_constraint_active_substeps": accel_constraint_active_substeps,
        "max_executor_velocity_error_over_substeps": max_executor_velocity_error_over_substeps,
        "optimizer_executor_contract_violation_substeps": optimizer_executor_contract_violation_substeps,
    }
    return pos.astype(np.float32), vel.astype(np.float32), exec_diag, cbf_diagnostics


# ========================================================================================= route navigator
@dataclass
class _RouteState:
    waypoints: tuple[np.ndarray, ...]
    index: int
    goal_at_creation: np.ndarray
    created_time_s: float


def _direct_path_clear(position: np.ndarray, goal: np.ndarray, inflated_obstacles: list[tuple[np.ndarray, np.ndarray]]) -> bool:
    for min_c, max_c in inflated_obstacles:
        box = AABBObstacle(min_corner=min_c, max_corner=max_c)
        if box.segment_intersects(position, goal):
            return False
    return True


# Route-status vocabulary (journal_cbf_attacker, obstacle_cbf_v2 / Prompt 3, brief section 8):
#   ROUTE_OK                     -- a valid route/direct-shot target is available this step.
#   ROUTE_TEMPORARILY_UNAVAILABLE -- the planner executed correctly (no exception, finite output) but found no
#       safe path at this instant (e.g. start or goal transiently inside/overlapping an inflated clearance zone --
#       see docs/journal_cbf_attacker/obstacle_cbf_v2/planner_failure_audit.json). NOT an implementation failure.
#   ROUTE_ERROR                  -- an exception, non-finite/invalid waypoint output, or a planner-contract
#       violation (e.g. a returned waypoint whose segment from the current position is STILL blocked). Acceptance
#       requires this to be exactly 0; the audit found zero such cases across all 32 Prompt-2 failure events.
ROUTE_OK = "route_ok"
ROUTE_TEMPORARILY_UNAVAILABLE = "route_temporarily_unavailable"
ROUTE_ERROR = "route_error"


def _waypoints_valid(waypoints: tuple[np.ndarray, ...]) -> bool:
    """Contract check (brief section 8/10): every returned waypoint must be a finite, well-formed 3-vector."""
    return all(np.asarray(w, dtype=np.float64).shape == (3,) and np.all(np.isfinite(w)) for w in waypoints)


@dataclass
class CBFRouteNavigator:
    """Stateful route-target selector (brief sections 3/4/6/7). Reuses the EXISTING tested visibility-graph + A*
    static-obstacle planner (`uav_defend.obstacles.route_planner.build_and_search_route`) for the actual geometric
    search -- this class owns only route PERSISTENCE, the replanning triggers, and the route-status
    classification/fallback behavior (brief sections 8/9), never a second independent planning algorithm.

    Role separation (brief section 3): this navigator handles GLOBAL/local-minimum deadlock around obstacle
    clusters by choosing which waypoint to steer toward; it has no notion of safety margins or velocity -- that is
    the CBF-QP's job, entirely downstream of this component.
    """

    clearance: float = CBF_OBSTACLE_CLEARANCE
    max_altitude: float = 30.0
    planning_range: float = ROUTE_PLANNING_RANGE
    max_graph_nodes: int = ROUTE_MAX_GRAPH_NODES
    waypoint_tolerance: float = ROUTE_WAYPOINT_TOLERANCE
    replan_period_s: float = ROUTE_REPLAN_PERIOD_S
    asset_move_trigger_m: float = ROUTE_ASSET_MOVE_TRIGGER_M

    _route: _RouteState | None = field(default=None, init=False)
    _replan_count: int = field(default=0, init=False)
    _planner_failure_count: int = field(default=0, init=False)   # ROUTE_TEMPORARILY_UNAVAILABLE events
    _route_error_count: int = field(default=0, init=False)       # ROUTE_ERROR events (must stay 0)
    _last_status: str = field(default="direct", init=False)
    _last_route_status: str = field(default=ROUTE_OK, init=False)
    _consecutive_unavailable_steps: int = field(default=0, init=False)
    _longest_unavailable_streak: int = field(default=0, init=False)
    _hover_fallback_count: int = field(default=0, init=False)

    def reset(self) -> None:
        self._route = None
        self._replan_count = 0
        self._planner_failure_count = 0
        self._route_error_count = 0
        self._last_status = "direct"
        self._last_route_status = ROUTE_OK
        self._consecutive_unavailable_steps = 0
        self._longest_unavailable_streak = 0
        self._hover_fallback_count = 0

    @property
    def replan_count(self) -> int:
        return self._replan_count

    @property
    def planner_failure_count(self) -> int:
        return self._planner_failure_count

    @property
    def route_error_count(self) -> int:
        return self._route_error_count

    @property
    def longest_unavailable_streak(self) -> int:
        return self._longest_unavailable_streak

    @property
    def hover_fallback_count(self) -> int:
        return self._hover_fallback_count

    def update(self, position: np.ndarray, asset_position: np.ndarray, obstacles: tuple[AABBObstacle, ...], elapsed_time_s: float) -> tuple[np.ndarray | None, dict]:
        """Returns `(guidance_target, diagnostics)`. `guidance_target` is the current asset position (direct line
        clear), the next/previous route waypoint (blocked but a target is available), or `None` -- meaning "hover/
        brake" (brief section 9: command zero velocity; NEVER aim at a known-blocked target, NEVER an invented
        escape direction) -- which occurs only when the route is `ROUTE_TEMPORARILY_UNAVAILABLE` or `ROUTE_ERROR`
        AND there is no previously retained waypoint to fall back on."""
        position = np.asarray(position, dtype=np.float64).reshape(3)
        asset_position = np.asarray(asset_position, dtype=np.float64).reshape(3)
        inflated = [inflate_aabb(o, self.clearance) for o in obstacles]
        waypoint_retained = False

        if _direct_path_clear(position, asset_position, inflated):
            self._route = None
            self._last_status = "direct"
            self._last_route_status = ROUTE_OK
            self._consecutive_unavailable_steps = 0
            return asset_position, self._diagnostics(route_length=0.0, waypoint_retained=False, hover=False)

        replan_reason = self._replan_reason(position, asset_position, inflated, elapsed_time_s)
        if replan_reason is not None:
            route_status = ROUTE_OK
            try:
                perceived = PerceivedObstacleSet(static=tuple(
                    PerceivedStaticObstacle(id=i, min_corner=o.min_corner, max_corner=o.max_corner, source="map")
                    for i, o in enumerate(obstacles)
                ))
                waypoints = build_and_search_route(
                    position, asset_position, perceived, clearance=self.clearance, max_altitude=self.max_altitude,
                    planning_range=self.planning_range, max_graph_nodes=self.max_graph_nodes,
                )
            except Exception:
                waypoints = None
                route_status = ROUTE_ERROR
            else:
                if waypoints is not None and not _waypoints_valid(waypoints):
                    # Planner-contract violation (brief section 8): non-finite/malformed output. A genuine
                    # implementation bug, NOT a geometric search failure -- never seen in the 32-event audit, but
                    # detected explicitly here rather than silently relabeled.
                    waypoints = None
                    route_status = ROUTE_ERROR
            self._replan_count += 1

            if waypoints is None:
                if route_status == ROUTE_ERROR:
                    self._route_error_count += 1
                else:
                    route_status = ROUTE_TEMPORARILY_UNAVAILABLE
                    self._planner_failure_count += 1
                self._last_status = "planner_failed" if route_status == ROUTE_TEMPORARILY_UNAVAILABLE else "route_error"
                self._last_route_status = route_status
                self._consecutive_unavailable_steps += 1
                self._longest_unavailable_streak = max(self._longest_unavailable_streak, self._consecutive_unavailable_steps)
                # Brief section 9: retain the previous waypoint if one remains; otherwise hover/brake -- NEVER fall
                # back to the (by construction, still-blocked) direct asset target, and NEVER an invented escape.
                if self._route is not None and self._route.waypoints:
                    target = self._route.waypoints[self._route.index]
                    waypoint_retained = True
                    route_length = float(np.linalg.norm(target - position))
                    return target, self._diagnostics(route_length=route_length, waypoint_retained=True, hover=False)
                self._hover_fallback_count += 1
                return None, self._diagnostics(route_length=0.0, waypoint_retained=False, hover=True)

            self._route = _RouteState(waypoints=waypoints, index=0, goal_at_creation=asset_position.copy(), created_time_s=elapsed_time_s)
            self._last_status = "replanned"
            self._last_route_status = ROUTE_OK
            self._consecutive_unavailable_steps = 0
        else:
            self._last_status = "following"
            self._last_route_status = ROUTE_OK
            self._consecutive_unavailable_steps = 0

        route = self._route
        # Advance past any waypoint already reached (defensive -- normally index==0 right after a replan).
        while route.index < len(route.waypoints) - 1 and float(np.linalg.norm(route.waypoints[route.index] - position)) <= self.waypoint_tolerance:
            route.index += 1
        target = route.waypoints[route.index]
        route_length = float(sum(
            np.linalg.norm(route.waypoints[i + 1] - route.waypoints[i]) for i in range(len(route.waypoints) - 1)
        )) + float(np.linalg.norm(route.waypoints[0] - position))
        return target, self._diagnostics(route_length=route_length, waypoint_retained=waypoint_retained, hover=False)

    def _replan_reason(self, position: np.ndarray, asset_position: np.ndarray, inflated, elapsed_time_s: float) -> str | None:
        route = self._route
        if route is None or not route.waypoints:
            return "no_route"
        if float(np.linalg.norm(asset_position - route.goal_at_creation)) > self.asset_move_trigger_m:
            return "asset_moved"
        target = route.waypoints[route.index]
        if not _direct_path_clear(position, target, inflated):
            return "segment_invalid"
        if float(np.linalg.norm(target - position)) <= self.waypoint_tolerance:
            return "waypoint_reached"
        if elapsed_time_s - route.created_time_s >= self.replan_period_s - 1e-9:
            return "periodic"
        return None

    def _diagnostics(self, *, route_length: float, waypoint_retained: bool, hover: bool) -> dict:
        target = self._route.waypoints[self._route.index] if self._route is not None else None
        return {
            "route_planner_status": self._last_status, "route_status": self._last_route_status,
            "route_waypoint": target, "route_replan_count": self._replan_count,
            "route_planner_failures": self._planner_failure_count, "route_errors": self._route_error_count,
            "route_length_m": route_length, "route_unavailable_streak": self._consecutive_unavailable_steps,
            "route_longest_unavailable_streak": self._longest_unavailable_streak,
            "route_waypoint_retained": waypoint_retained, "route_hover_fallback": hover,
            "route_hover_fallback_count": self._hover_fallback_count,
        }

