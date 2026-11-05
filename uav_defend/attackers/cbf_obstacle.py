"""A1-CBF static-obstacle navigation (journal_cbf_attacker, Prompt 2): AABB signed-distance/normal geometry,
acceleration-aware stopping-margin hazard construction, a fixed 32-sided velocity-bound polygon, and the
CBF-QP safety filter (hard -> soft-slack -> emergency-brake fallback chain), plus a stateful route navigator that
reuses the EXISTING tested visibility-graph + A* static-obstacle planner (`uav_defend.obstacles.route_planner`).

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

from uav_defend.attackers.cbf_attacker import CBF_DT_SUB, CBFAttackerController, apply_one_substep, substep_count
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


def solve_velocity_cbf_qp(
    v_nom: np.ndarray, hazards: list[StaticHazard], *, alpha: float = CBF_ALPHA,
    max_horizontal_speed: float, max_climb_rate: float, max_descent_rate: float,
    n_sides: int = VELOCITY_POLYGON_SIDES, maxiter: int = 200, ftol: float = 1e-12, residual_tol: float = 1e-4,
) -> tuple[np.ndarray, dict]:
    """Solves `min_v 0.5*||v - v_nom||^2` subject to every active CBF constraint (section 11) and the velocity
    bounds (section 14), via `scipy.optimize.minimize(method="SLSQP")` -- deterministic initial guess (clipped
    `v_nom`), fixed tolerances/iteration cap. Hard-first / soft-slack-fallback / emergency-brake chain (section 16).

    No obstacle constraints: identity (NOT routed through SciPy at all; preserves Prompt-1 bit-for-bit behavior
    exactly -- see `CBFAttackerController.filter_velocity`, which only calls this function when `hazards` is
    non-empty)."""
    v_nom = np.asarray(v_nom, dtype=np.float64).reshape(3)
    bound_funcs = _velocity_bound_funcs(max_horizontal_speed, max_climb_rate, max_descent_rate, n_sides)
    x0 = _clip_initial_guess(v_nom, max_horizontal_speed, max_climb_rate, max_descent_rate)

    def objective(v):
        diff = v[:3] - v_nom
        return 0.5 * float(np.dot(diff, diff))

    # ---- (1) HARD QP ----
    # Acceptance is based on CONSTRAINT FEASIBILITY (max residual within tolerance), not SLSQP's own internal
    # `success` flag: for borderline-degenerate QPs (e.g. a solution lying exactly on two constraint boundaries
    # simultaneously) SLSQP's line search can report "Positive directional derivative" (status 8) even though the
    # returned point is numerically feasible to well within tolerance -- what matters for safety is feasibility,
    # not the optimizer's own termination diagnostic. `qp_success` still records the raw solver flag for
    # transparency/diagnostics.
    hard_funcs = _cbf_constraint_funcs(hazards, alpha, slack=False) + bound_funcs
    constraints = [{"type": "ineq", "fun": f} for f in hard_funcs]
    res = minimize(objective, x0, method="SLSQP", constraints=constraints, options={"maxiter": maxiter, "ftol": ftol})
    hard_residual = _max_residual(hard_funcs, res.x)
    if hard_residual <= residual_tol:
        v_safe = res.x[:3]
        return v_safe, {
            "cbf_qp_status": "hard", "qp_success": bool(res.success), "qp_iterations": int(res.nit), "qp_objective": float(objective(res.x)),
            "qp_max_constraint_residual": hard_residual, "soft_slack": 0.0,
        }

    # ---- (2) SOFT-SLACK FALLBACK ----
    soft_cbf_funcs = _cbf_constraint_funcs(hazards, alpha, slack=True)
    soft_bound_funcs = [lambda v, f=f: f(v[:3]) for f in bound_funcs]  # bounds stay HARD (never relaxed by slack)
    soft_funcs = soft_cbf_funcs + soft_bound_funcs

    def soft_objective(v):
        diff = v[:3] - v_nom
        delta = v[3]
        return 0.5 * float(np.dot(diff, diff)) + SOFT_SLACK_PENALTY * delta * delta

    x0_soft = np.concatenate([x0, [0.0]])
    soft_constraints = [{"type": "ineq", "fun": f} for f in soft_funcs]
    soft_res = minimize(
        soft_objective, x0_soft, method="SLSQP", constraints=soft_constraints,
        bounds=[(None, None)] * 3 + [(0.0, None)], options={"maxiter": maxiter, "ftol": ftol},
    )
    soft_residual = _max_residual(soft_funcs, soft_res.x)
    if soft_residual <= residual_tol:
        v_safe = soft_res.x[:3]
        return v_safe, {
            "cbf_qp_status": "soft_fallback", "qp_success": bool(soft_res.success), "qp_iterations": int(soft_res.nit),
            "qp_objective": float(soft_objective(soft_res.x)), "qp_max_constraint_residual": soft_residual,
            "soft_slack": float(soft_res.x[3]),
        }

    # ---- (3) EMERGENCY BRAKE ----
    # The acceleration-feasible command toward zero velocity -- never a random/invented escape direction. The
    # route planner re-evaluates on the next step (brief section 16).
    return np.zeros(3, dtype=np.float64), {
        "cbf_qp_status": "emergency_brake", "qp_success": False,
        "qp_iterations": int(soft_res.nit), "qp_objective": float("nan"), "qp_max_constraint_residual": soft_residual,
        "soft_slack": float(soft_res.x[3]) if soft_res.x is not None else float("nan"),
    }


# ========================================================================================= safety-critical executor
def integrate_obstacle_cbf_substeps(
    position: np.ndarray, velocity: np.ndarray, *, route_target: np.ndarray | None, controller: CBFAttackerController,
    obstacles: tuple[AABBObstacle, ...], dt: float, dt_sub: float = CBF_DT_SUB, L: float, max_altitude: float,
    eps: float = 1e-8, clearance: float = CBF_OBSTACLE_CLEARANCE, active_range: float = CBF_ACTIVE_OBSTACLE_RANGE,
    alpha: float = CBF_ALPHA,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Safety-critical variant of `cbf_attacker.integrate_quadrotor_substeps`: re-solves the nominal-guidance +
    CBF-QP safety filter EVERY substep (dt_sub, default 0.1 s), instead of holding one command constant across all
    5 substeps of the external 0.5 s step.

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

    Returns `(position, velocity, exec_diagnostics, cbf_diagnostics)` where `exec_diagnostics` has the SAME schema
    as `integrate_quadrotor_substeps` (accel/speed/climb saturation flags) and `cbf_diagnostics` holds the LAST
    substep's CBF-QP diagnostics plus aggregates (`emergency_brake_substeps`, `min_h_over_substeps`,
    `max_active_obstacle_count`) over all 5 substeps.
    """
    pos = np.asarray(position, dtype=np.float64).copy()
    vel = np.asarray(velocity, dtype=np.float64).copy()
    route_target = np.asarray(route_target, dtype=np.float64).reshape(3) if route_target is not None else None
    n = substep_count(dt, dt_sub)
    limits = controller.limits

    max_accel_used = 0.0
    accel_saturated = horizontal_saturated = vertical_saturated = False
    emergency_brake_substeps = 0
    min_h_over_substeps = float("inf")
    min_d_over_substeps = float("inf")
    max_active_obstacle_count = 0
    last_cbf_diag: dict = {}
    last_v_nom = np.zeros(3)
    last_v_safe = np.zeros(3)

    for _ in range(n):
        if route_target is None:
            v_nom = np.zeros(3, dtype=np.float64)   # hover/brake (brief section 9): no target available this step
        else:
            r_g = route_target - pos
            v_nom = controller.nominal_velocity(r_g)
        hazards = build_static_hazards(
            pos, vel, obstacles, clearance=clearance, active_range=active_range, max_accel=limits.max_accel,
            dt_sub=dt_sub, alpha=alpha,
        )
        v_safe, cbf_diag = controller.filter_velocity(v_nom, state={"position": pos.copy(), "velocity": vel.copy()}, hazards=tuple(hazards))
        if cbf_diag["cbf_qp_status"] == "emergency_brake":
            emergency_brake_substeps += 1
        if hazards:
            min_h_over_substeps = min(min_h_over_substeps, min(h.h for h in hazards))
            min_d_over_substeps = min(min_d_over_substeps, min(h.d for h in hazards))
        max_active_obstacle_count = max(max_active_obstacle_count, len(hazards))

        pos, vel, sub_diag = apply_one_substep(pos, vel, v_safe, limits=limits, dt_sub=dt_sub, L=L, max_altitude=max_altitude, eps=eps)
        max_accel_used = max(max_accel_used, sub_diag["accel_used"])
        accel_saturated = accel_saturated or sub_diag["accel_saturated"]
        horizontal_saturated = horizontal_saturated or sub_diag["horizontal_speed_saturated"]
        vertical_saturated = vertical_saturated or sub_diag["vertical_rate_saturated"]
        last_cbf_diag = cbf_diag
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

