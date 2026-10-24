"""Stateful receding-horizon obstacle-avoidance controller for the V2 scripted attacker.

Replaces the fixed-waypoint bypass (`ObstacleAvoidancePlanner`, still used by V1 and the defender
planners) for the corrected A0 only. It stays a LOCAL reactive controller: no global map, no
defender evasion, no weave, no tactical intelligence. Hierarchy: physical safety > obstacle
avoidance > direct mission pursuit.

Guidance laws (`guidance_command`) are pure functions of the CURRENT state -- there is no stored
x/y/z target. The same function is used by real execution and by the rollouts that evaluate it,
and the rollouts replay the environment's float32 update through the shared constrained dynamics
(`_executed_step`), so a prediction is the trajectory that will actually be flown.

    DIRECT   fly the mission command (re-aimed at the asset every step)
    CLIMB    forward progress toward the asset; climb at the vehicle's climb rate until the
             blocker set is cleared (top + clearance, capped at the ceiling), then hold altitude
    LEFT/RIGHT  steer toward the clearance-inflated corner of the nearest blocker on that side,
             recomputed from the current position every step (never pointing behind the
             vehicle: a sideways step is used instead)

Per planning step (`StatefulAvoidanceController.plan`):

    1. DIRECT rollout (actual turn-limited flight, <= horizon steps, cut at the capture sphere).
       If it is collision-free -> DIRECT (and release any avoidance). This is the same test that
       triggers avoidance, so trigger and release cannot disagree.
    2. Otherwise the blocker set = perceived obstacles the DIRECT rollout hits or passes within
       `clearance` of, extended by those threatened by the chosen candidate's rollout.
    3. Evaluate CLIMB/LEFT/RIGHT by hybrid rollouts (the law until DIRECT becomes collision-free
       from the predicted state, then DIRECT) and pick lexicographically (`select_candidate`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_defend.obstacles.avoidance import (
    DynamicsLimits,
    _executed_step,
    _horizontal_bypass_candidate,
    _lateral_axis,
    _mission_direction,
    predict_collision,
    truncate_prediction_at_target,
)
from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.layout import ObstacleLayout

# Candidate order is also the final tie-break (same as the original planner).
MODES = ("left", "right", "climb")


# ------------------------------------------------------------------ guidance
def _horizontal_unit(vector: np.ndarray, eps: float) -> np.ndarray:
    horizontal = np.asarray(vector[:2], dtype=np.float64)
    norm = float(np.linalg.norm(horizontal))
    return horizontal / norm if norm > eps else np.array([1.0, 0.0])


def nearest_blocker(position: np.ndarray, blockers: list[AABBObstacle]) -> AABBObstacle:
    """The blocker closest to `position` (ties: first in the list)."""
    return min(blockers, key=lambda obstacle: obstacle.distance_to_point(position))


def climb_guidance(
    position: np.ndarray, u_mission: np.ndarray, blockers: list[AABBObstacle], clearance: float, max_altitude: float,
    limits: DynamicsLimits,
) -> np.ndarray:
    """CLIMB: keep moving toward the asset horizontally; climb at the vehicle's climb rate until the
    blocker set is cleared (highest top + the existing vertical `clearance`, capped at the ceiling),
    then hold altitude. Horizontal direction is always the current direct-to-asset one."""
    u_h = _horizontal_unit(u_mission, limits.eps)
    z_required = float(position[2])
    if blockers:
        z_required = min(max(float(obstacle.max_corner[2]) for obstacle in blockers) + clearance, max_altitude)
    if float(position[2]) < z_required - limits.eps:
        pitch = min(1.0, limits.max_climb_rate / limits.max_speed)
        return np.array([*(u_h * np.sqrt(1.0 - pitch * pitch)), pitch], dtype=np.float64)
    return np.array([u_h[0], u_h[1], 0.0], dtype=np.float64)


def lateral_guidance(
    position: np.ndarray, u_mission: np.ndarray, blockers: list[AABBObstacle], clearance: float, side: str,
    eps: float,
) -> np.ndarray:
    """LEFT/RIGHT: head horizontally for the visible clearance-inflated corner of the nearest blocker on
    `side`, from the current position (the original tangent-corner rule, evaluated every step). If that
    would point behind the vehicle (relative to the mission direction), sidestep instead. The vertical
    component stays the mission's own (the asset's elevation is unchanged by a sidestep), so a diving
    attacker keeps descending toward the asset instead of bleeding its descent off and flying past it."""
    primary = nearest_blocker(position, blockers)
    direction = _horizontal_bypass_candidate(position, primary, u_mission, clearance, side, eps).direction
    horizontal = _horizontal_unit(direction, eps)
    if float(np.dot(horizontal, _horizontal_unit(u_mission, eps))) < 0.0:
        sign = 1.0 if side == "left" else -1.0
        horizontal = sign * _lateral_axis(u_mission, eps)
    vertical = float(u_mission[2])
    return np.array([*(horizontal * np.sqrt(max(1.0 - vertical * vertical, 0.0))), vertical], dtype=np.float64)


def guidance_command(
    mode: str, position: np.ndarray, u_mission: np.ndarray, blockers: list[AABBObstacle], clearance: float,
    max_altitude: float, limits: DynamicsLimits,
) -> np.ndarray:
    """Unit command of guidance law `mode` from the current state. Shared by execution and rollouts."""
    if mode == "direct" or not blockers:
        return np.asarray(u_mission, dtype=np.float64)
    if mode == "climb":
        return climb_guidance(position, u_mission, blockers, clearance, max_altitude, limits)
    return lateral_guidance(position, u_mission, blockers, clearance, mode, limits.eps)


# ------------------------------------------------------------------ rollouts
@dataclass
class _Context:
    target: np.ndarray
    layout: ObstacleLayout
    limits: DynamicsLimits
    horizon: int
    stop_radius: float | None
    clearance: float
    max_altitude: float
    fallback: np.ndarray


def truncate_at_closest_approach(positions: list[np.ndarray], target: np.ndarray) -> tuple[list[np.ndarray], bool]:
    """Cut a DIRECT rollout at its closest approach to the asset: once the vehicle starts moving away
    (a fly-by it cannot turn in on), the rest is re-pursuit that is re-planned from the then-current
    state, not part of this engagement. The immediate next step is always kept. Returns the cut path and
    whether the vehicle was moving away at the cut."""
    distances = [float(np.linalg.norm(p - target)) for p in positions]
    for index in range(len(positions) - 1):
        if distances[index + 1] > distances[index]:
            return positions[: max(index + 1, 2)], True
    return positions, False


def direct_rollout(position32: np.ndarray, velocity32: np.ndarray, context: _Context) -> list[np.ndarray]:
    """What the attacker flies if it commands DIRECT from this state: the mission command is re-derived
    from the (predicted) position every step, through the executed dynamics. Cut at the capture sphere
    or, if the asset is not reached, at the closest approach to it."""
    p, v = position32, velocity32
    positions = [p.astype(np.float64)]
    for _ in range(context.horizon):
        command = _mission_direction(p, context.target, context.fallback, context.limits.eps)
        p, v = _executed_step(p, v, command, context.limits)
        positions.append(p.astype(np.float64))
    if context.stop_radius is not None:
        positions = truncate_prediction_at_target(positions, context.target, context.stop_radius)
    return truncate_at_closest_approach(positions, np.asarray(context.target, dtype=np.float64))[0]


def direct_is_safe(position32: np.ndarray, velocity32: np.ndarray, context: _Context) -> bool:
    """DIRECT eligibility: the actual turn-limited DIRECT rollout is collision-free under the planner's
    physical obstacle geometry (no added clearance distance)."""
    return not predict_collision(direct_rollout(position32, velocity32, context), context.layout).collision


@dataclass
class Evaluation:
    mode: str
    order: int
    collision: bool
    first_collision_step: int | None
    min_clearance: float          # over the part flown under the avoidance law
    meets_clearance: bool
    progress: float
    excess: float                 # predicted path length minus progress (mission-path cost)
    heading_change: float
    switch_step: int | None       # predicted step at which DIRECT becomes eligible
    positions: list[np.ndarray]
    conflict_steps: int = 0       # predicted segments that intersect an obstacle
    flyby: bool = False           # predicted to pass the asset outside the capture sphere and move away


def mode_rollout(
    mode: str, position32: np.ndarray, velocity32: np.ndarray, blockers: list[AABBObstacle], context: _Context,
    order: int,
) -> Evaluation:
    """Fly guidance law `mode` until DIRECT is eligible from the predicted state, then fly DIRECT."""
    limits = context.limits
    p, v = position32, velocity32
    positions = [p.astype(np.float64)]
    velocities = [v.astype(np.float64)]
    switch = None
    for step in range(context.horizon):
        mission = _mission_direction(p, context.target, context.fallback, limits.eps)
        if switch is None and step > 0 and direct_is_safe(p, v, context):
            switch = step
        command = mission if switch is not None else guidance_command(
            mode, p, mission, blockers, context.clearance, context.max_altitude, limits)
        p, v = _executed_step(p, v, command, limits)
        positions.append(p.astype(np.float64))
        velocities.append(v.astype(np.float64))
    if context.stop_radius is not None:
        positions = truncate_prediction_at_target(positions, context.target, context.stop_radius)
    target = np.asarray(context.target, dtype=np.float64)
    cut = False
    if switch is not None:
        tail, cut = truncate_at_closest_approach(positions[switch:], target)
        positions = positions[:switch] + tail
    distances = [float(np.linalg.norm(point - target)) for point in positions]
    stop = context.stop_radius if context.stop_radius is not None else 0.0
    flyby = min(distances) > stop and (cut or int(np.argmin(distances)) < len(distances) - 1)
    prediction = predict_collision(positions, context.layout)
    avoidance_part = positions if switch is None else positions[: switch + 1]
    clearance = min(context.layout.min_clearance(point) for point in avoidance_part)
    start_distance = float(np.linalg.norm(positions[0] - context.target))
    progress = start_distance - float(np.linalg.norm(positions[-1] - context.target))
    path_length = float(sum(np.linalg.norm(b - a) for a, b in zip(positions, positions[1:])))
    heading_change = 0.0
    for a, b in zip(velocities, velocities[1:len(positions)]):
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        if norm > limits.eps:
            heading_change += float(np.arccos(np.clip(np.dot(a, b) / norm, -1.0, 1.0)))
    conflicts = 0
    if prediction.collision:
        conflicts = sum(
            context.layout.first_segment_intersection(a, b).intersects for a, b in zip(positions, positions[1:]))
    return Evaluation(
        mode=mode, order=order, collision=prediction.collision, first_collision_step=prediction.step_index,
        min_clearance=clearance, meets_clearance=(not prediction.collision) and clearance >= context.clearance,
        progress=progress, excess=max(path_length - progress, 0.0), heading_change=heading_change,
        switch_step=switch, positions=positions, conflict_steps=conflicts, flyby=flyby,
    )


# ----------------------------------------------------------------- selection
def _tier(evaluation: Evaluation) -> tuple[int, int]:
    """(feasibility, margin): 0 collision-free, 1 predicted fly-by (collision-free but misses the asset),
    2 collision; and 0 meeting the required clearance, 1 not."""
    if evaluation.collision:
        return (2, 2)
    return (1 if evaluation.flyby else 0, 0 if evaluation.meets_clearance else 1)


def _key(evaluation: Evaluation, step_length: float):
    tier = _tier(evaluation)
    ahead = 0 if evaluation.progress > 0.0 else 1
    cost_bucket = int(evaluation.excess // step_length)
    if tier[0] < 2 and tier[1] == 0:
        return (tier[0], 0, ahead, cost_bucket, evaluation.heading_change, -evaluation.min_clearance, evaluation.order)
    if tier[0] < 2:
        return (tier[0], 1, -evaluation.min_clearance, ahead, cost_bucket, evaluation.heading_change, evaluation.order)
    # Emergency (every candidate collides): latest first contact, then the fewest conflicted steps
    # (leave the conflict region soonest), then the roomiest, then fixed order.
    first = evaluation.first_collision_step if evaluation.first_collision_step is not None else 0
    return (2, 2, -first, evaluation.conflict_steps, -evaluation.min_clearance, evaluation.order)


def select_candidate(evaluations: list[Evaluation], current_mode: str | None, step_length: float) -> Evaluation:
    """Lexicographic choice among avoidance candidates (no weights):

      1. collision-free beats colliding; among collision-free, meeting the required clearance over
         the avoided part beats not meeting it (the original safe / margin-deficient categories);
         if everything collides, the roomiest candidate (largest predicted minimum clearance);
      2. positive predicted target progress;
      3. lower predicted mission-path cost (path length minus progress), compared in steps of one
         step's travel (`step_length` = max_speed * dt) so near-equal costs are treated as equal;
      4. smaller steering disruption (cumulative predicted heading change);
      5. larger predicted minimum clearance, then the fixed order left, right, climb.

    Anti-chatter: the currently active mode is kept whenever it is as safe as the best candidate
    (same tier, collision-free), still makes progress (or the best does not), and its mission-path
    cost is within one step's travel of the best -- i.e. the two are effectively equivalent."""
    best = min(evaluations, key=lambda e: _key(e, step_length))
    if current_mode is not None:
        current = next((e for e in evaluations if e.mode == current_mode), None)
        if (
            current is not None and current is not best
            and _tier(current) == _tier(best) and _tier(current)[0] < 2
            and (current.progress > 0.0 or best.progress <= 0.0)
            and current.excess - best.excess <= step_length
        ):
            return current
    return best


def threatened_indices(layout: ObstacleLayout, positions: list[np.ndarray], clearance: float) -> set[int]:
    """Obstacles a predicted path hits or passes within `clearance` of."""
    return {
        index for index, obstacle in enumerate(layout.obstacles)
        if any(obstacle.distance_to_point(point) < clearance for point in positions)
    }


# ---------------------------------------------------------------- controller
class StatefulAvoidanceController:
    """V2 attacker obstacle avoidance. Drop-in for `ObstacleAvoidancePlanner` (same `plan` signature and
    result keys, `active`, `mode`, `obstacle_index`, `reset`)."""

    def __init__(self, clearance: float, horizon_steps: int, max_altitude: float, eps: float = 1e-8,
                 mission_stop_radius: float | None = None):
        self.clearance = clearance
        self.horizon_steps = horizon_steps
        self.max_altitude = max_altitude
        self.eps = eps
        self.mission_stop_radius = mission_stop_radius
        self.diagnostics = self._empty_diagnostics()
        self.last: dict = {}
        self.reset()

    @staticmethod
    def _empty_diagnostics() -> dict:
        return {"activations": 0, "mode_switches": 0, "release_accepted": 0, "release_rejected": 0,
                "all_candidates_collide": 0, "selections": {m: 0 for m in MODES}}

    def reset(self) -> None:
        """Clear the avoidance state (call on episode reset). Diagnostics are kept; see `reset_diagnostics`."""
        self.active = False
        self.mode: str | None = None
        self.obstacle_index: int | None = None
        self.blockers: tuple[int, ...] = ()
        self.entry_step: int | None = None
        self.step_count = getattr(self, "step_count", 0)

    def reset_diagnostics(self) -> None:
        self.diagnostics = self._empty_diagnostics()
        self.step_count = 0

    def plan(self, position, velocity, u_base, layout: ObstacleLayout, target_position, limits: DynamicsLimits) -> dict:
        u_base = np.asarray(u_base, dtype=np.float64)
        self.step_count += 1
        if len(layout) == 0:
            self.reset()
            self.last = {"direct_collides": False}
            return self._result(u_base, None, False)

        p32 = np.asarray(position, dtype=np.float32)
        v32 = np.asarray(velocity, dtype=np.float32)
        context = _Context(
            target=np.asarray(target_position), layout=layout, limits=limits, horizon=self.horizon_steps,
            stop_radius=self.mission_stop_radius, clearance=self.clearance, max_altitude=self.max_altitude,
            fallback=u_base,
        )
        direct_positions = direct_rollout(p32, v32, context)
        direct = predict_collision(direct_positions, layout)
        if not direct.collision:
            if self.active:
                self.diagnostics["release_accepted"] += 1
            self.reset()
            self.last = {"direct_collides": False}
            return self._result(u_base, direct, False)

        if self.active:
            self.diagnostics["release_rejected"] += 1
        blockers = threatened_indices(layout, direct_positions, self.clearance) | {direct.obstacle_index}
        step_length = limits.max_speed * limits.dt
        chosen = None
        for _ in range(2):  # one refinement: add obstacles the chosen candidate itself threatens
            objects = [layout[i] for i in sorted(blockers)]
            evaluations = [mode_rollout(mode, p32, v32, objects, context, order) for order, mode in enumerate(MODES)]
            chosen = select_candidate(evaluations, self.mode if self.active else None, step_length)
            extended = blockers | threatened_indices(layout, chosen.positions, self.clearance)
            if extended == blockers:
                break
            blockers = extended

        fallback = all(e.collision for e in evaluations)
        if not self.active:
            self.diagnostics["activations"] += 1
            self.entry_step = self.step_count
        elif chosen.mode != self.mode:
            self.diagnostics["mode_switches"] += 1
        if fallback:
            self.diagnostics["all_candidates_collide"] += 1
        self.diagnostics["selections"][chosen.mode] += 1
        self.active, self.mode = True, chosen.mode
        self.blockers = tuple(sorted(blockers))
        self.obstacle_index = direct.obstacle_index
        self.last = {
            "direct_collides": True, "selected": chosen.mode, "fallback": fallback, "blockers": list(self.blockers),
            "candidates": {e.mode: {"collision": e.collision, "meets_clearance": e.meets_clearance,
                                    "min_clearance": e.min_clearance, "progress": e.progress, "excess": e.excess,
                                    "heading_change": e.heading_change, "switch_step": e.switch_step}
                           for e in evaluations},
        }
        command = guidance_command(
            chosen.mode, position, u_base, [layout[i] for i in sorted(blockers)], self.clearance, self.max_altitude, limits)
        return self._result(command, direct, fallback)

    def _result(self, direction, direct, fallback_used: bool) -> dict:
        return {
            "direction": np.asarray(direction, dtype=np.float64),
            "active": self.active,
            "mode": self.mode,
            "obstacle_index": self.obstacle_index,
            "predicted_collision_step": None if direct is None else direct.step_index,
            "predicted_min_clearance": float("inf") if direct is None else direct.min_clearance,
            "fallback_used": fallback_used,
        }
