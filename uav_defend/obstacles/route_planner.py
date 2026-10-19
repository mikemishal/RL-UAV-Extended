"""
Perception-constrained multi-obstacle local route planner (journal
extension, Phase 3-15 of the attacker-navigation task).

Replaces the attacker's previous "always react to the single most
immediate obstacle" mission command with a short, deterministic,
locally-planned ROUTE through/around a perceived obstacle cluster, while
the EXISTING reactive `ObstacleAvoidancePlanner` (unmodified) remains the
low-level last-mile safety layer:

    mission/evasion command (u_mission)
        -> RoutePlanner (THIS module): next waypoint direction (u_route)
        -> ObstacleAvoidancePlanner (unchanged): u_safe = avoid(u_route, O_perceived)
        -> vehicle dynamics

CRITICAL CONSTRAINT (Phase 4): this planner NEVER receives the
ground-truth world -- only `PerceivedObstacleSet` (see
uav_defend/obstacles/perception.py). It plans exactly what the current
awareness mode (Level B/C) currently perceives, nothing more.

Architecture (visibility-graph + A*, Phase 5-6):
    Nodes: current position, local goal, and -- for each perceived static
        obstacle within `enemy_route_planning_range` -- 4 horizontal
        inflated-AABB corner waypoints plus (if geometrically/dynamically
        possible) 1 vertical-over waypoint above the obstacle.
    Edges: connect two nodes only if the straight segment between them is
        collision-free against EVERY inflated perceived static obstacle
        AND does not require an immediate near-reversal turn (a light,
        deterministic proxy for "dynamically feasible" -- the existing
        reactive layer remains the authoritative dynamics-feasibility
        enforcer during actual flight).
    Cost: path_length + turn_penalty + vertical_motion_penalty +
        low_clearance_penalty (see `_edge_cost`). Safety (collision-free)
        is a hard feasibility constraint, never traded off in the cost.
    Search: A* (Euclidean-distance-to-goal heuristic, admissible since
        real cost >= straight-line distance).

Replanning (Phase 8), progress-stall detection (Phase 12), and loop/
revisit suppression (Phase 13) are handled by the stateful `RoutePlanner`
wrapper, which also emits the diagnostics consumed by
uav_defend/envs/soldier_env.py.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from uav_defend.obstacles.perception import PerceivedObstacleSet, PerceivedStaticObstacle

_EPS = 1e-8


@dataclass(frozen=True)
class _Node:
    position: np.ndarray
    kind: str  # "start" | "goal" | "corner" | "vertical"
    obstacle_id: int | None = None


@dataclass(frozen=True)
class RouteState:
    """Explicit route state (Phase 11) -- stable obstacle IDs only, never
    transient list indices."""

    route_id: int
    waypoints: tuple[np.ndarray, ...]
    current_waypoint_index: int
    awareness_mode: str
    obstacle_ids: tuple[int, ...]
    created_step: int
    replan_reason: str
    estimated_length: float
    goal_at_creation: np.ndarray

    def current_target(self) -> np.ndarray:
        return self.waypoints[min(self.current_waypoint_index, len(self.waypoints) - 1)]

    def advance_if_reached(self, position: np.ndarray, tolerance: float) -> "RouteState":
        idx = self.current_waypoint_index
        while idx < len(self.waypoints) - 1 and float(np.linalg.norm(self.waypoints[idx] - position)) <= tolerance:
            idx += 1
        if idx == self.current_waypoint_index:
            return self
        return RouteState(self.route_id, self.waypoints, idx, self.awareness_mode,
                           self.obstacle_ids, self.created_step, self.replan_reason,
                           self.estimated_length, self.goal_at_creation)


def _inflated_corners_xy(obstacle: PerceivedStaticObstacle, clearance: float) -> list[np.ndarray]:
    # Push corners a small extra margin BEYOND the inflated-clearance
    # boundary: a waypoint exactly ON that boundary would make every
    # segment ending there register as "touching" the inflated box under
    # this repository's boundary-inclusive collision convention (see
    # AABBObstacle's `tolerance`), spuriously blocking all edges to it.
    extra = clearance * 0.15 + 0.25
    xmin, ymin = obstacle.min_corner[0] - clearance - extra, obstacle.min_corner[1] - clearance - extra
    xmax, ymax = obstacle.max_corner[0] + clearance + extra, obstacle.max_corner[1] + clearance + extra
    z = float((obstacle.min_corner[2] + obstacle.max_corner[2]) / 2.0)
    return [np.array([xmin, ymin, z]), np.array([xmax, ymin, z]),
            np.array([xmax, ymax, z]), np.array([xmin, ymax, z])]


def _vertical_waypoint(obstacle: PerceivedStaticObstacle, position: np.ndarray, clearance: float, max_altitude: float) -> np.ndarray | None:
    extra = clearance * 0.15 + 0.25
    target_z = float(obstacle.max_corner[2]) + clearance + extra
    if target_z > max_altitude - _EPS:
        return None  # climbing high enough to clear is infeasible (Phase 6/14)
    center_xy = (obstacle.min_corner[:2] + obstacle.max_corner[:2]) / 2.0
    return np.array([center_xy[0], center_xy[1], target_z])


def _segment_blocked(p0: np.ndarray, p1: np.ndarray, inflated_obstacles: list) -> bool:
    return any(obs.segment_intersects(p0, p1) for obs in inflated_obstacles)


def _min_clearance_along(p0: np.ndarray, p1: np.ndarray, raw_boxes: list, samples: int = 5) -> float:
    # `raw_boxes` must be a small, PRE-BUILT list of raw (non-inflated)
    # AABBObstacle instances -- callers must bound this to the obstacles
    # already known to be within `planning_range` (never the full
    # perceived-static set, and never rebuilt per-call) to keep per-edge
    # cost evaluation cheap (this runs on every A* state expansion).
    if not raw_boxes:
        return float("inf")
    best = float("inf")
    for t in np.linspace(0.0, 1.0, samples):
        point = p0 + t * (p1 - p0)
        for box in raw_boxes:
            best = min(best, box.clearance_to_point(point))
    return best


def _edge_cost(
    p0: np.ndarray, p1: np.ndarray, incoming_dir: np.ndarray | None, raw_boxes: list,
    turn_weight: float, vertical_weight: float, clearance_weight: float, required_clearance: float,
) -> float:
    segment = p1 - p0
    length = float(np.linalg.norm(segment))
    if length < _EPS:
        return 0.0
    direction = segment / length
    turn_penalty = 0.0
    if incoming_dir is not None:
        cos_angle = float(np.clip(np.dot(direction, incoming_dir), -1.0, 1.0))
        turn_penalty = turn_weight * float(np.arccos(cos_angle))
    vertical_penalty = vertical_weight * abs(float(segment[2]))
    clearance = _min_clearance_along(p0, p1, raw_boxes)
    low_clearance_penalty = clearance_weight * max(0.0, required_clearance - clearance) if np.isfinite(clearance) else 0.0
    return length + turn_penalty + vertical_penalty + low_clearance_penalty


def build_and_search_route(
    position: np.ndarray,
    goal: np.ndarray,
    perceived: PerceivedObstacleSet,
    clearance: float,
    max_altitude: float,
    planning_range: float,
    max_graph_nodes: int,
    turn_weight: float = 3.0,
    vertical_weight: float = 1.5,
    clearance_weight: float = 2.0,
) -> tuple[np.ndarray, ...] | None:
    """Build a visibility graph from `perceived.static` (obstacles beyond
    `planning_range` from `position` are excluded -- Phase 7 bounded
    planning) and return the A*-shortest FEASIBLE waypoint sequence
    (excluding the start node) from `position` to `goal`, or None if no
    feasible route exists within the graph (Phase 24: a geometric-search
    failure, NOT a ground-truth feasibility judgment)."""
    position = np.asarray(position, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)

    nearby = [o for o in perceived.static if float(np.linalg.norm(
        ((o.min_corner + o.max_corner) / 2.0) - position)) <= planning_range + clearance]

    from uav_defend.obstacles.geometry import AABBObstacle
    inflated = [
        AABBObstacle.from_center_half_extents(
            (o.min_corner + o.max_corner) / 2.0,
            (o.max_corner - o.min_corner) / 2.0 + clearance,
        ) for o in nearby
    ]
    # Raw (non-inflated) boxes for clearance scoring -- built ONCE here and
    # reused for every edge-cost evaluation below (a real perf bug fixed in
    # this task: previously each A* state expansion rebuilt AABBObstacle
    # instances from the FULL perceived-static set on every call, which
    # under Level-C map-known awareness (many always-perceived obstacles)
    # made individual replans take seconds instead of milliseconds).
    raw_boxes = [AABBObstacle(min_corner=o.min_corner, max_corner=o.max_corner) for o in nearby]

    nodes: list[_Node] = [_Node(position, "start"), _Node(goal, "goal")]
    for o in nearby:
        for corner in _inflated_corners_xy(o, clearance):
            nodes.append(_Node(corner, "corner", obstacle_id=o.id))
        vwp = _vertical_waypoint(o, position, clearance, max_altitude)
        if vwp is not None:
            nodes.append(_Node(vwp, "vertical", obstacle_id=o.id))
        if len(nodes) >= max_graph_nodes:
            break
    nodes = nodes[:max_graph_nodes]

    # Direct collision-free shot: skip graph search entirely (common case, cheap).
    if not _segment_blocked(position, goal, inflated):
        return (goal,)

    start_idx, goal_idx = 0, 1
    n = len(nodes)
    adjacency: list[list[int]] = [[] for _ in range(n)]
    # Precompute the direction-INDEPENDENT part of each edge's cost once
    # (length + vertical + clearance penalty never change across A* state
    # expansions for a fixed pair of nodes) -- only the turn penalty is
    # direction-dependent and is computed cheaply (no obstacle loop) during
    # the search itself.
    edge_base_cost: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            p0, p1 = nodes[i].position, nodes[j].position
            if _segment_blocked(p0, p1, inflated):
                continue
            adjacency[i].append(j)
            adjacency[j].append(i)
            segment = p1 - p0
            length = float(np.linalg.norm(segment))
            vertical_penalty = vertical_weight * abs(float(segment[2]))
            clearance_val = _min_clearance_along(p0, p1, raw_boxes)
            low_clearance_penalty = (
                clearance_weight * max(0.0, clearance - clearance_val) if np.isfinite(clearance_val) else 0.0
            )
            base = length + vertical_penalty + low_clearance_penalty
            edge_base_cost[(i, j)] = base
            edge_base_cost[(j, i)] = base

    def heuristic(i: int) -> float:
        return float(np.linalg.norm(nodes[i].position - goal))

    # A* with incoming-direction-aware edge cost (state = (node_index, incoming_dir_key)).
    # Bounded expansion count guards against pathological state-space growth
    # in dense clusters (Phase 25 runtime concern) -- if exceeded, treat as
    # a geometric search failure (same as "no route found").
    max_expansions = max(2000, 50 * n * n)
    open_heap = [(heuristic(start_idx), 0.0, start_idx, None)]
    best_cost: dict[tuple, float] = {(start_idx, None): 0.0}
    came_from: dict[tuple, tuple] = {}
    visited_goal_state = None
    expansions = 0

    while open_heap:
        expansions += 1
        if expansions > max_expansions:
            break
        _, cost_so_far, node_idx, incoming_dir_tuple = heapq.heappop(open_heap)
        state = (node_idx, incoming_dir_tuple)
        if cost_so_far > best_cost.get(state, float("inf")) + 1e-9:
            continue
        if node_idx == goal_idx:
            visited_goal_state = state
            break
        incoming_dir = np.asarray(incoming_dir_tuple) if incoming_dir_tuple is not None else None
        for neighbor_idx in adjacency[node_idx]:
            direction = nodes[neighbor_idx].position - nodes[node_idx].position
            norm = np.linalg.norm(direction)
            direction = direction / norm if norm > _EPS else np.zeros(3)
            turn_penalty = 0.0
            if incoming_dir is not None:
                cos_angle = float(np.clip(np.dot(direction, incoming_dir), -1.0, 1.0))
                turn_penalty = turn_weight * float(np.arccos(cos_angle))
            edge_cost = edge_base_cost[(node_idx, neighbor_idx)] + turn_penalty
            new_cost = cost_so_far + edge_cost
            new_state = (neighbor_idx, tuple(np.round(direction, 6)))
            if new_cost < best_cost.get(new_state, float("inf")) - 1e-9:
                best_cost[new_state] = new_cost
                came_from[new_state] = state
                heapq.heappush(open_heap, (new_cost + heuristic(neighbor_idx), new_cost, neighbor_idx, new_state[1]))

    if visited_goal_state is None:
        return None  # no feasible route found in this perceived graph (Phase 24)

    path_states = [visited_goal_state]
    while path_states[-1] in came_from:
        path_states.append(came_from[path_states[-1]])
    path_states.reverse()
    waypoints = tuple(nodes[state[0]].position for state in path_states[1:])  # exclude start
    total_length = sum(
        float(np.linalg.norm(nodes[path_states[i + 1][0]].position - nodes[path_states[i][0]].position))
        for i in range(len(path_states) - 1)
    )
    return waypoints if waypoints else (goal,)


@dataclass
class RoutePlanner:
    """Stateful wrapper: owns route persistence, replanning triggers
    (Phase 8), progress-stall detection (Phase 12), and loop/revisit
    suppression (Phase 13). Consumes ZERO randomness -- purely a function
    of perceived state, so identical seeds reproduce identical routes.

    One instance tracks route state for ONE mover (the hostile UAV).
    """

    clearance: float
    max_altitude: float
    planning_range: float
    max_graph_nodes: int
    waypoint_tolerance: float
    progress_window_steps: int
    stall_distance_threshold: float
    dt: float

    _route: RouteState | None = field(default=None, init=False)
    _next_route_id: int = field(default=0, init=False)
    _replan_count: int = field(default=0, init=False)
    _last_replan_reason: str | None = field(default=None, init=False)
    _goal_distance_history: list = field(default_factory=list, init=False)
    _position_history: list = field(default_factory=list, init=False)  # thinned, for revisit suppression
    _step_count: int = field(default=0, init=False)
    _vertical_bypass_count: int = field(default=0, init=False)
    _last_progress_efficiency_num: float = field(default=0.0, init=False)
    _last_progress_efficiency_den: float = field(default=0.0, init=False)

    def reset(self) -> None:
        self._route = None
        self._next_route_id = 0
        self._replan_count = 0
        self._last_replan_reason = None
        self._goal_distance_history = []
        self._position_history = []
        self._step_count = 0
        self._vertical_bypass_count = 0
        self._last_progress_efficiency_num = 0.0
        self._last_progress_efficiency_den = 0.0

    def invalidate_route(self) -> None:
        """Discard the current route (V2 corrected A0): once the direct path
        to the target is feasible again, no stale bypass waypoint may
        survive; the next `plan()` call replans from scratch. Unlike
        `reset()`, keeps cumulative replan/diagnostic counters."""
        self._route = None

    def direct_path_clear(self, position: np.ndarray, true_target: np.ndarray, perceived: PerceivedObstacleSet) -> bool:
        """True iff the straight segment position -> true_target avoids every
        PERCEIVED static obstacle inflated by `clearance` (same inflation the
        route graph uses). Uses perception only -- never ground truth."""
        from uav_defend.obstacles.geometry import AABBObstacle
        position = np.asarray(position, dtype=np.float64)
        true_target = np.asarray(true_target, dtype=np.float64)
        for o in perceived.static:
            inflated = AABBObstacle.from_center_half_extents(
                (o.min_corner + o.max_corner) / 2.0, (o.max_corner - o.min_corner) / 2.0 + self.clearance,
            )
            if inflated.segment_intersects(position, true_target):
                return False
        return True

    def _local_goal(self, position: np.ndarray, true_target: np.ndarray) -> np.ndarray:
        """Phase 9: use the true target directly if within the planning
        horizon, else project a point toward it at `planning_range`."""
        offset = true_target - position
        distance = float(np.linalg.norm(offset))
        if distance <= self.planning_range:
            return true_target
        return position + offset / distance * self.planning_range

    def _route_is_blocked(self, position: np.ndarray, perceived: PerceivedObstacleSet) -> bool:
        if self._route is None:
            return True
        from uav_defend.obstacles.geometry import AABBObstacle
        inflated = [
            AABBObstacle.from_center_half_extents(
                (o.min_corner + o.max_corner) / 2.0, (o.max_corner - o.min_corner) / 2.0 + self.clearance,
            ) for o in perceived.static
        ]
        remaining = (position,) + self._route.waypoints[self._route.current_waypoint_index:]
        return any(_segment_blocked(remaining[i], remaining[i + 1], inflated) for i in range(len(remaining) - 1))

    def _detect_stall(self) -> bool:
        if len(self._goal_distance_history) < self.progress_window_steps:
            return False
        window = self._goal_distance_history[-self.progress_window_steps:]
        return (window[0] - window[-1]) < self.stall_distance_threshold

    def _record_revisit_history(self, position: np.ndarray) -> bool:
        """Returns True if `position` revisits a location seen >=
        `progress_window_steps` steps ago (Phase 13)."""
        is_revisit = False
        if len(self._position_history) > self.progress_window_steps // 5:
            earlier = np.array(self._position_history[: -max(1, self.progress_window_steps // 5)])
            if len(earlier):
                if np.any(np.linalg.norm(earlier - position, axis=1) < 3.0):
                    is_revisit = True
        if self._step_count % 5 == 0:
            self._position_history.append(position.copy())
            if len(self._position_history) > 200:
                self._position_history.pop(0)
        return is_revisit

    def plan(
        self, position: np.ndarray, true_target: np.ndarray, perceived: PerceivedObstacleSet, awareness_mode: str,
    ) -> dict:
        """Compute this step's next-waypoint direction (u_route) and
        advance all replanning/diagnostic state. Returns a dict with keys:
        "direction" (unit np.ndarray, feed as u_mission into the EXISTING
        reactive avoidance layer), plus diagnostics (Phase 18)."""
        position = np.asarray(position, dtype=np.float64)
        true_target = np.asarray(true_target, dtype=np.float64)
        self._step_count += 1

        goal_distance = float(np.linalg.norm(true_target - position))
        self._goal_distance_history.append(goal_distance)
        if len(self._goal_distance_history) > self.progress_window_steps * 2:
            self._goal_distance_history.pop(0)

        is_revisit = self._record_revisit_history(position)

        replan_reason = None
        if self._route is None:
            replan_reason = "no_active_route"
        elif self._route_is_blocked(position, perceived):
            replan_reason = "route_unsafe_or_new_obstacle"
        elif float(np.linalg.norm(self._route.goal_at_creation - self._local_goal(position, true_target))) > max(self.planning_range * 0.5, self.clearance * 4):
            # NOTE: this threshold intentionally scales with the planning
            # horizon rather than a small fixed distance. The pursuit
            # TARGET (the defender) moves continuously and can easily cover
            # >10m between calls; a small fixed threshold caused a full
            # geometric replan almost every step even with no meaningful
            # change in the obstacle-avoidance situation, which (a) was
            # very expensive and (b) let the bypass STRATEGY (e.g. "go left"
            # vs "go right" around a cluster) flip-flop step to step,
            # producing net zigzag non-progress -- the very pathology this
            # task exists to fix. Small target motion is now absorbed cheaply
            # every step via the live final-waypoint tracking below (outside
            # this trigger block); a full replan here is reserved for when
            # the target has moved far enough that the existing bypass
            # geometry may no longer make sense. Safety is never affected by
            # this threshold: `route_unsafe_or_new_obstacle` (above) still
            # forces an immediate replan the instant the current route is
            # actually blocked, regardless of how recently it was planned.
            replan_reason = "target_moved"
        elif self._detect_stall():
            replan_reason = "progress_stall"
        elif self._route.current_waypoint_index >= len(self._route.waypoints) - 1 and \
                float(np.linalg.norm(self._route.waypoints[-1] - position)) <= self.waypoint_tolerance:
            replan_reason = "route_complete"

        if replan_reason is not None:
            local_goal = self._local_goal(position, true_target)
            # Loop suppression (Phase 13): a stalled/revisiting episode
            # replans with a temporarily larger clearance margin, which
            # shifts the visibility graph enough to select a materially
            # different route rather than deterministically reselecting
            # the same cyclic path.
            effective_clearance = self.clearance * 1.5 if (replan_reason == "progress_stall" or is_revisit) else self.clearance
            waypoints = build_and_search_route(
                position, local_goal, perceived, effective_clearance, self.max_altitude,
                self.planning_range, self.max_graph_nodes,
            )
            self._replan_count += 1
            self._last_replan_reason = replan_reason
            if waypoints is None:
                # No feasible route found in the perceived graph (Phase 24):
                # fall back to a direct-toward-goal single waypoint, letting
                # the reactive layer be the sole safety mechanism this step
                # (safe/deterministic degradation, never a crash/NaN/teleport).
                waypoints = (local_goal,)
            obstacle_ids = tuple(sorted({o.id for o in perceived.static
                                         if float(np.linalg.norm(((o.min_corner + o.max_corner) / 2.0) - position)) <= self.planning_range + self.clearance}))
            self._vertical_bypass_count += sum(1 for w in waypoints if abs(float(w[2]) - float(position[2])) > 1.0)
            estimated_length = float(sum(
                np.linalg.norm(waypoints[i] - (position if i == 0 else waypoints[i - 1])) for i in range(len(waypoints))
            ))
            self._route = RouteState(
                route_id=self._next_route_id, waypoints=waypoints, current_waypoint_index=0,
                awareness_mode=awareness_mode, obstacle_ids=obstacle_ids, created_step=self._step_count,
                replan_reason=replan_reason, estimated_length=estimated_length, goal_at_creation=local_goal,
            )
            self._next_route_id += 1
        else:
            self._route = self._route.advance_if_reached(position, self.waypoint_tolerance)
            # Live final-waypoint tracking: the LAST waypoint of any route
            # always represents "the (local) goal" (see build_and_search_route),
            # which for this task is the CONTINUOUSLY MOVING pursuit target
            # (the defender), not a fixed destination. Bypass corner
            # waypoints earlier in the route are legitimately tied to fixed
            # obstacle geometry and are left untouched, but refusing to
            # refresh the final leg every step (even when no full replan is
            # needed) would make the attacker chase a stale, minutes-old
            # snapshot of where the target USED to be -- breaking continuous
            # pursuit. Small movements (below the `target_moved` replan
            # threshold) are absorbed here cheaply; large movements still
            # trigger a full geometric replan above.
            local_goal = self._local_goal(position, true_target)
            if not np.allclose(self._route.waypoints[-1], local_goal, atol=1e-9):
                new_waypoints = self._route.waypoints[:-1] + (local_goal,)
                self._route = RouteState(
                    route_id=self._route.route_id, waypoints=new_waypoints,
                    current_waypoint_index=self._route.current_waypoint_index,
                    awareness_mode=self._route.awareness_mode, obstacle_ids=self._route.obstacle_ids,
                    created_step=self._route.created_step, replan_reason=self._route.replan_reason,
                    estimated_length=self._route.estimated_length, goal_at_creation=self._route.goal_at_creation,
                )

        target_point = self._route.current_target()
        offset = target_point - position
        norm = float(np.linalg.norm(offset))
        direction = offset / norm if norm > _EPS else np.zeros(3)

        return {
            "direction": direction,
            "route_active": True,
            "route_waypoint_count": len(self._route.waypoints),
            "route_current_waypoint": self._route.current_waypoint_index,
            "route_replan_count": self._replan_count,
            "route_replan_reason": self._last_replan_reason,
            "route_obstacle_ids": self._route.obstacle_ids,
            "route_estimated_length": self._route.estimated_length,
            "progress_stall_active": replan_reason == "progress_stall",
            "route_revisit": is_revisit,
            "vertical_bypass_count": self._vertical_bypass_count,
        }

    def resume_or_rejoin(
        self, position: np.ndarray, true_target: np.ndarray, perceived: PerceivedObstacleSet, awareness_mode: str,
    ) -> dict:
        """Behavior-arbitration Phase 6: called ONCE when the caller
        transitions out of a suspension window (e.g. defender-evasion just
        ended) instead of `plan()`. While suspended, the caller does not
        call `plan()` at all -- so no stall/target-moved/revisit judgment
        accumulates evasion-induced deviation against the route (Phase 6:
        "preserve the route if it remains useful; suspend waypoint
        tracking temporarily").

        On resuming:
          1. If no route exists yet, or the existing route is no longer
             collision-safe from the CURRENT (post-evasion) position,
             this is exactly a `no_active_route` / `route_unsafe_or_new_
             obstacle` replan -- delegate to `plan()` (a fresh replan is
             the correct, safe behavior; this is NOT counted as an
             evasion-induced "waypoint invalidation" since it reflects a
             genuine safety/geometry change, not policy churn).
          2. Otherwise, SKIP any waypoints that are effectively already
             behind the current position (a waypoint is skipped if the
             segment straight to a LATER waypoint is collision-free AND
             that later waypoint is no farther away -- i.e. detouring
             back to the skipped waypoint would only add path length),
             then resume following the route from there -- never forcing
             a return to an obsolete waypoint behind the attacker.
        """
        position = np.asarray(position, dtype=np.float64)
        true_target = np.asarray(true_target, dtype=np.float64)

        if self._route is None or self._route_is_blocked(position, perceived):
            return self.plan(position, true_target, perceived, awareness_mode)

        from uav_defend.obstacles.geometry import AABBObstacle
        inflated = [
            AABBObstacle.from_center_half_extents(
                (o.min_corner + o.max_corner) / 2.0, (o.max_corner - o.min_corner) / 2.0 + self.clearance,
            ) for o in perceived.static
        ]
        idx = self._route.current_waypoint_index
        waypoints = self._route.waypoints
        while idx < len(waypoints) - 1:
            later_is_closer = float(np.linalg.norm(waypoints[idx + 1] - position)) <= float(np.linalg.norm(waypoints[idx] - position))
            direct_to_later_safe = not _segment_blocked(position, waypoints[idx + 1], inflated)
            if later_is_closer and direct_to_later_safe:
                idx += 1
            else:
                break

        local_goal = self._local_goal(position, true_target)
        if idx == len(waypoints) - 1 and not np.allclose(waypoints[-1], local_goal, atol=1e-9):
            waypoints = waypoints[:-1] + (local_goal,)

        self._route = RouteState(
            route_id=self._route.route_id, waypoints=waypoints, current_waypoint_index=idx,
            awareness_mode=self._route.awareness_mode, obstacle_ids=self._route.obstacle_ids,
            created_step=self._route.created_step, replan_reason=self._route.replan_reason,
            estimated_length=self._route.estimated_length, goal_at_creation=self._route.goal_at_creation,
        )

        target_point = self._route.current_target()
        offset = target_point - position
        norm = float(np.linalg.norm(offset))
        direction = offset / norm if norm > _EPS else np.zeros(3)

        return {
            "direction": direction,
            "route_active": True,
            "route_waypoint_count": len(self._route.waypoints),
            "route_current_waypoint": self._route.current_waypoint_index,
            "route_replan_count": self._replan_count,
            "route_replan_reason": self._last_replan_reason,
            "route_obstacle_ids": self._route.obstacle_ids,
            "route_estimated_length": self._route.estimated_length,
            "progress_stall_active": False,
            "route_revisit": False,
            "vertical_bypass_count": self._vertical_bypass_count,
        }

