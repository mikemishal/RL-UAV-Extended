"""Stateful receding-horizon avoidance controller for the V2 scripted attacker.

Deterministic controlled scenarios (no environment randomness, no learned policy, no training):
mode selection, guidance laws that depend only on the current state, the DIRECT executed-rollout
rejoin gate, anti-chatter, the emergency ordering and rollout-vs-execution exactness."""

from __future__ import annotations

import numpy as np

from uav_defend.obstacles.avoidance import DynamicsLimits, _executed_step, _mission_direction
from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout
from uav_defend.obstacles.stateful_avoidance import (
    Evaluation,
    StatefulAvoidanceController,
    _Context,
    climb_guidance,
    direct_is_safe,
    guidance_command,
    lateral_guidance,
    mode_rollout,
    select_candidate,
)

LIMITS = DynamicsLimits(
    max_speed=12.0, max_accel=6.0, max_turn_rate_rad=np.radians(75.0), max_climb_rate=5.0, max_descent_rate=5.0,
    dt=0.5, eps=1e-8, L=50.0, max_altitude=30.0,
)
STEP = LIMITS.max_speed * LIMITS.dt
STOP = 2.0


def box(x0, x1, y0, y1, z0, z1) -> AABBObstacle:
    return AABBObstacle(min_corner=np.array([x0, y0, z0]), max_corner=np.array([x1, y1, z1]))


def controller() -> StatefulAvoidanceController:
    return StatefulAvoidanceController(clearance=2.0, horizon_steps=12, max_altitude=30.0, mission_stop_radius=STOP)


def _segment_distance(a, b, point) -> float:
    """Distance from `point` to the segment a-b (the environment's contact is swept, not end-point only)."""
    a, b, point = (np.asarray(x, dtype=np.float64) for x in (a, b, point))
    ab = b - a
    t = 0.0 if float(ab @ ab) == 0.0 else float(np.clip((point - a) @ ab / (ab @ ab), 0.0, 1.0))
    return float(np.linalg.norm(a + t * ab - point))


def fly(ctrl, obstacles, start, velocity, target, steps=60):
    """Closed loop with a stationary asset; the environment's float32 update. Returns the trace."""
    layout = ObstacleLayout(tuple(obstacles))
    target = np.asarray(target, dtype=np.float32)
    p, v = np.asarray(start, dtype=np.float32), np.asarray(velocity, dtype=np.float32)
    trace = {"pos": [p.copy()], "active": [], "mode": [], "plans": [], "captured": False, "collided": False}
    for _ in range(steps):
        u_base = _mission_direction(p, target, np.array([1.0, 0.0, 0.0]), LIMITS.eps)
        plan = ctrl.plan(p, v, u_base, layout, target, LIMITS)
        new_p, new_v = _executed_step(p, v, plan["direction"], LIMITS)
        trace["active"].append(plan["active"])
        trace["mode"].append(plan["mode"])
        trace["plans"].append(dict(ctrl.last))
        if layout.first_segment_intersection(p.astype(np.float64), new_p.astype(np.float64)).intersects:
            trace["collided"] = True
            break
        p, v = new_p, new_v
        trace["pos"].append(p.copy())
        if _segment_distance(trace["pos"][-2], p, target) <= STOP:
            trace["captured"] = True
            break
    trace["pos"] = np.array(trace["pos"])
    return trace


def short_cycles(active, gap=2) -> int:
    count, i = 0, 0
    while i < len(active):
        if not active[i] and i > 0 and active[i - 1]:
            j = i
            while j < len(active) and not active[j]:
                j += 1
            if j < len(active) and j - i <= gap:
                count += 1
            i = j
        else:
            i += 1
    return count


# ----------------------------------------------------------------------- DIRECT
def test_clear_environment_selects_direct_and_returns_the_mission_command_exactly():
    ctrl = controller()
    layout = ObstacleLayout((box(-40, -35, 30, 35, 0, 10),))
    p, v = np.array([-10.0, 0.0, 10.0], dtype=np.float32), np.array([12.0, 0.0, 0.0], dtype=np.float32)
    target = np.array([30.0, 0.0, 10.0], dtype=np.float32)
    u_base = _mission_direction(p, target, np.array([1.0, 0.0, 0.0]), LIMITS.eps)
    plan = ctrl.plan(p, v, u_base, layout, target, LIMITS)
    assert not plan["active"] and plan["mode"] is None and np.array_equal(plan["direction"], u_base)


def test_an_empty_layout_is_a_noop():
    ctrl = controller()
    u_base = np.array([1.0, 0.0, 0.0])
    plan = ctrl.plan(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), u_base, ObstacleLayout(()),
                     np.array([20.0, 0.0, 0.0]), LIMITS)
    assert not plan["active"] and np.array_equal(plan["direction"], u_base)


# ------------------------------------------------------------------------ CLIMB
def test_a_full_width_low_wall_selects_climb_with_forward_progress_and_no_reversal():
    wall = box(10, 14, -50, 50, 0, 6)  # no way around: only over
    ctrl = controller()
    trace = fly(ctrl, [wall], (-10, 0, 3), (12, 0, 0), (45, 0, 9))
    assert not trace["collided"] and trace["captured"]
    first = trace["mode"].index("climb")
    assert trace["mode"][first] == "climb" and set(m for m in trace["mode"] if m) == {"climb"}
    x = trace["pos"][:, 0]
    assert np.all(np.diff(x) > 0.0)  # forward progress every step: never back toward the activation point
    assert trace["pos"][:, 2].max() >= 6.0 + 2.0 - 1e-6  # cleared the top by the existing vertical clearance
    assert not trace["active"][-1]  # climb exits: DIRECT again before capture


def test_climb_guidance_is_a_pure_function_of_the_current_state():
    blockers = [box(10, 14, -50, 50, 0, 6)]
    mission = np.array([0.8, 0.0, -0.6])
    below = climb_guidance(np.array([0.0, 0.0, 3.0]), mission, blockers, 2.0, 30.0, LIMITS)
    above = climb_guidance(np.array([0.0, 0.0, 9.0]), mission, blockers, 2.0, 30.0, LIMITS)
    assert below[2] > 0.0 and np.isclose(np.linalg.norm(below), 1.0)  # climbs, at the vehicle's climb rate
    assert np.isclose(below[2], LIMITS.max_climb_rate / LIMITS.max_speed)
    assert above[2] == 0.0 and above[0] > 0.0  # already clear: hold level, keep going forward
    # the horizontal direction depends only on the mission direction, never on where the avoidance began
    assert np.allclose(below[:2] / np.linalg.norm(below[:2]), above[:2] / np.linalg.norm(above[:2]))
    assert below[0] > 0.0 and above[0] > 0.0


def test_climb_is_capped_at_the_ceiling():
    out = climb_guidance(np.array([0.0, 0.0, 29.5]), np.array([1.0, 0.0, 0.0]), [box(10, 14, -5, 5, 0, 29)], 2.0, 30.0, LIMITS)
    assert out[2] > 0.0
    out = climb_guidance(np.array([0.0, 0.0, 30.0]), np.array([1.0, 0.0, 0.0]), [box(10, 14, -5, 5, 0, 29)], 2.0, 30.0, LIMITS)
    assert out[2] == 0.0


# ---------------------------------------------------------------------- LATERAL
def test_only_the_left_side_is_open():
    # the obstacle runs from the arena's right wall (y = -50) to y = 2, full height: no climb, no right
    ctrl = controller()
    trace = fly(ctrl, [box(10, 14, -50, 2, 0, 30)], (-10, 0, 10), (12, 0, 0), (40, 0, 10))
    assert not trace["collided"] and trace["captured"]
    assert set(m for m in trace["mode"] if m) == {"left"}
    assert trace["pos"][:, 1].max() > 2.0


def test_only_the_right_side_is_open():
    ctrl = controller()
    trace = fly(ctrl, [box(10, 14, -2, 50, 0, 30)], (-10, 0, 10), (12, 0, 0), (40, 0, 10))
    assert not trace["collided"] and trace["captured"]
    assert set(m for m in trace["mode"] if m) == {"right"}
    assert trace["pos"][:, 1].min() < -2.0


def test_lateral_guidance_never_points_behind_the_vehicle_and_keeps_the_mission_vertical():
    blockers = [box(1, 5, -2, 2, 0, 30)]
    mission = np.array([0.6, 0.0, -0.8])  # diving
    for side in ("left", "right"):
        for position in ((-6.0, 0.0, 10.0), (0.0, 0.5, 10.0), (3.0, 0.0, 10.0), (4.9, 1.9, 10.0), (6.0, 0.0, 10.0)):
            out = lateral_guidance(np.array(position), mission, blockers, 2.0, side, LIMITS.eps)
            assert np.isclose(np.linalg.norm(out), 1.0) and out[2] == mission[2]
            assert out[0] * mission[0] + out[1] * mission[1] >= -1e-12


# ----------------------------------------------------------------------- CLUSTER
def test_a_perceived_cluster_is_one_continuous_maneuver_without_release_chatter():
    # B sits just beyond A and a little to the side: clearing A alone would re-aim straight at B
    a = box(10, 13, -2, 2, 0, 30)
    b = box(17, 20, 1, 5, 0, 30)
    ctrl = controller()
    trace = fly(ctrl, [a, b], (-10, 0, 10), (12, 0, 0), (45, 0, 10))
    assert not trace["collided"] and trace["captured"]
    first = next(plan for plan in trace["plans"] if plan.get("direct_collides"))
    assert len(first["blockers"]) == 2  # both were already perceived and predictable
    assert short_cycles(trace["active"]) == 0
    assert len(set(m for m in trace["mode"] if m)) == 1  # one maneuver, one side


# ------------------------------------------------------------------ DIRECT REJOIN
def _context(layout, target, p):
    return _Context(
        target=np.asarray(target, dtype=np.float32), layout=layout, limits=LIMITS, horizon=12, stop_radius=STOP,
        clearance=2.0, max_altitude=30.0, fallback=np.array([1.0, 0.0, 0.0]),
    )


def test_direct_becomes_eligible_when_the_actual_rollout_is_safe():
    layout = ObstacleLayout((box(10, 14, -2, 2, 0, 30),))
    ctrl = controller()
    trace = fly(ctrl, [layout.obstacles[0]], (-10, 0, 10), (12, 0, 0), (45, 0, 10))
    assert trace["active"][0] and not trace["active"][-1]  # avoided, then rejoined DIRECT
    # the executed DIRECT flight from the rejoin point onward really is collision-free all the way to contact
    assert not trace["collided"] and trace["captured"]
    assert all(m is None for m in trace["mode"][trace["active"].index(False):])
    assert ctrl.diagnostics["release_accepted"] == 1


def test_a_clear_straight_line_does_not_release_when_the_turn_limited_flight_would_collide():
    # fast along +x, the asset 90 deg to the left: the straight line vehicle->asset is geometrically clear,
    # but the vehicle cannot turn that quickly and flies into the obstacle ahead of it.
    obstacle = box(5, 9, -3, 3, 0, 30)
    layout = ObstacleLayout((obstacle,))
    p, v = np.array([0.0, 0.0, 10.0], dtype=np.float32), np.array([12.0, 0.0, 0.0], dtype=np.float32)
    target = np.array([0.0, 30.0, 10.0], dtype=np.float32)
    assert not layout.segment_intersects(p.astype(np.float64), target.astype(np.float64))  # the line is clear
    ctx = _context(layout, target, p)
    assert not direct_is_safe(p, v, ctx)  # ... the executed DIRECT flight is not

    for already_active in (False, True):
        ctrl = controller()
        if already_active:
            ctrl.active, ctrl.mode = True, "left"
        u_base = _mission_direction(p, target, np.array([1.0, 0.0, 0.0]), LIMITS.eps)
        plan = ctrl.plan(p, v, u_base, layout, target, LIMITS)
        assert plan["active"] and plan["mode"] in ("left", "right", "climb")
        assert not np.array_equal(plan["direction"], u_base)
    assert ctrl.diagnostics["release_rejected"] == 1 and ctrl.diagnostics["release_accepted"] == 0


# -------------------------------------------------------------------- ANTI-CHATTER
def _evaluation(mode, order, *, collision=False, clearance=3.0, progress=20.0, excess=0.0, heading=0.5, first=None,
                conflicts=0, flyby=False):
    return Evaluation(
        mode=mode, order=order, collision=collision, first_collision_step=first, min_clearance=clearance,
        meets_clearance=(not collision) and clearance >= 2.0, progress=progress, excess=excess, heading_change=heading,
        switch_step=None, positions=[], conflict_steps=conflicts, flyby=flyby,
    )


def test_equivalent_safe_candidates_keep_the_current_mode():
    left, right, climb = _evaluation("left", 0, excess=0.0), _evaluation("right", 1, excess=3.0), _evaluation("climb", 2, excess=4.0)
    for current in ("left", "right", "climb"):
        assert select_candidate([left, right, climb], current, STEP).mode == current
    assert select_candidate([left, right, climb], None, STEP).mode == "left"  # fixed order breaks a true tie


def test_a_clearly_better_candidate_still_wins_over_the_current_mode():
    left, right = _evaluation("left", 0, excess=0.0), _evaluation("right", 1, excess=STEP + 1.0)
    assert select_candidate([left, right], "right", STEP).mode == "left"
    # and a safe candidate beats a collision-free but margin-deficient current mode
    margin = _evaluation("right", 1, clearance=1.0)
    assert select_candidate([left, margin], "right", STEP).mode == "left"


def test_a_symmetric_obstacle_gives_a_stable_mode_through_the_whole_maneuver():
    ctrl = controller()
    trace = fly(ctrl, [box(10, 14, -2, 2, 0, 30)], (-10, 0, 10), (12, 0, 0), (45, 0, 10))
    modes = [m for m in trace["mode"] if m]
    assert modes and len(set(modes)) == 1 and ctrl.diagnostics["mode_switches"] == 0


# ------------------------------------------------------------------ SELECTION ORDER
def test_lexicographic_order_collision_free_then_capture_then_clearance_then_progress_then_cost():
    safe_flyby = _evaluation("left", 0, flyby=True)
    margin_capture = _evaluation("right", 1, clearance=1.0)
    colliding = _evaluation("climb", 2, collision=True, first=3, clearance=0.0)
    assert select_candidate([safe_flyby, margin_capture, colliding], None, STEP).mode == "right"  # captures
    assert select_candidate([safe_flyby, colliding], None, STEP).mode == "left"  # collision-free over colliding
    ahead, behind = _evaluation("left", 0, progress=-5.0), _evaluation("right", 1, progress=5.0, excess=20.0)
    assert select_candidate([ahead, behind], None, STEP).mode == "right"  # positive progress first
    cheap, costly = _evaluation("left", 0, excess=30.0), _evaluation("right", 1, excess=2.0)
    assert select_candidate([cheap, costly], None, STEP).mode == "right"  # then lower mission-path cost
    smooth, sharp = _evaluation("left", 0, heading=2.0), _evaluation("right", 1, heading=0.5)
    assert select_candidate([smooth, sharp], None, STEP).mode == "right"  # then less steering disruption


# ---------------------------------------------------------------------- EMERGENCY
def test_when_every_candidate_collides_the_latest_and_shortest_conflict_is_chosen():
    a = _evaluation("left", 0, collision=True, first=0, conflicts=5, clearance=0.0)
    b = _evaluation("right", 1, collision=True, first=0, conflicts=2, clearance=0.0)
    c = _evaluation("climb", 2, collision=True, first=4, conflicts=9, clearance=0.0)
    assert select_candidate([a, b], None, STEP).mode == "right"  # leave the conflict region soonest
    assert select_candidate([a, b, c], None, STEP).mode == "climb"  # later first contact wins
    assert select_candidate([a, b, c], "left", STEP).mode == "climb"  # no hysteresis inside an emergency


def test_an_all_collide_state_is_counted_and_reported_as_a_fallback():
    # surrounded: a pocket of full-height walls on three sides and a ceiling-height block ahead
    walls = [box(3, 6, -3, 3, 0, 30), box(-3, 6, 3, 6, 0, 30), box(-3, 6, -6, -3, 0, 30)]
    ctrl = controller()
    layout = ObstacleLayout(tuple(walls))
    p, v = np.array([0.0, 0.0, 10.0], dtype=np.float32), np.array([12.0, 0.0, 0.0], dtype=np.float32)
    target = np.array([30.0, 0.0, 10.0], dtype=np.float32)
    plan = ctrl.plan(p, v, _mission_direction(p, target, np.array([1.0, 0.0, 0.0]), LIMITS.eps), layout, target, LIMITS)
    assert plan["active"] and plan["fallback_used"] and ctrl.diagnostics["all_candidates_collide"] == 1


# ---------------------------------------------------- PREDICTION / EXECUTION SHARING
def test_the_rollout_of_the_selected_law_is_exactly_what_is_flown():
    wall = box(10, 14, -50, 50, 0, 6)
    layout = ObstacleLayout((wall,))
    target = np.array([40.0, 0.0, 3.0], dtype=np.float32)
    ctrl = controller()
    p, v = np.array([-10.0, 0.0, 3.0], dtype=np.float32), np.array([12.0, 0.0, 0.0], dtype=np.float32)
    rollout, flown, switch = None, [p.astype(np.float64)], None
    for step in range(40):
        u_base = _mission_direction(p, target, np.array([1.0, 0.0, 0.0]), LIMITS.eps)
        plan = ctrl.plan(p, v, u_base, layout, target, LIMITS)
        if plan["active"] and rollout is None:
            context = _Context(target=target, layout=layout, limits=LIMITS, horizon=12, stop_radius=STOP, clearance=2.0,
                               max_altitude=30.0, fallback=u_base)
            evaluation = mode_rollout(plan["mode"], p, v, [wall], context, 0)
            rollout, switch, start = evaluation.positions, evaluation.switch_step, step
            assert evaluation.mode == "climb"
        p, v = _executed_step(p, v, plan["direction"], LIMITS)
        flown.append(p.astype(np.float64))
        if rollout is not None and step - start + 1 == switch:
            break
    assert rollout is not None and switch is not None and switch >= 2
    n = switch + 1
    assert max(np.linalg.norm(a - b) for a, b in zip(rollout[:n], flown[start:start + n])) < 1e-5


def test_the_guidance_command_is_unit_length_and_direct_is_the_mission_command():
    blockers = [box(10, 14, -5, 5, 0, 8)]
    mission = np.array([0.6, 0.0, -0.8])
    assert np.array_equal(guidance_command("direct", np.zeros(3), mission, blockers, 2.0, 30.0, LIMITS), mission)
    for mode in ("left", "right", "climb"):
        out = guidance_command(mode, np.array([0.0, 0.0, 4.0]), mission, blockers, 2.0, 30.0, LIMITS)
        assert np.isclose(np.linalg.norm(out), 1.0)
