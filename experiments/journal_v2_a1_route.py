"""Route-level tactical intent controller for the V2 A1 attacker: action `a1_route_action_v1`, observation `a1_v2_obs_route_v1`.
See docs/journal_v2_a1_route_level_v1/.

A1 selects a ROUTE INTENT {DIRECT, EVADE_LEFT, EVADE_RIGHT, EVADE_ABOVE}; it never chooses heading, residual steering, velocity, acceleration or duration.
The deterministic existing visibility-graph/A* planner (`uav_defend.obstacles.route_planner.build_and_search_route`) generates the path
`attacker -> tactical subgoal -> asset` from the PERCEIVED obstacle set only; this module follows it with a persistent waypoint follower and hands the
resulting direction to the frozen common obstacle layer, which keeps final authority. DIRECT returns no command (the environment flies canonical A0).

PRIVILEGED-INFORMATION RULE (enforced by construction and by tests): this module receives ONLY
  * the attacker / asset / defender POSITIONS at the current step (defender position is used only while the A1 defender-information gate is open),
  * a `PerceivedObstacleSet` (what the canonical attacker perception contract currently provides),
  * static configuration numbers.
It never imports or touches the simulator, its ground-truth obstacle layout, dynamic-obstacle list, defender velocity or any future state.

Tactical geometry (m = unit attacker->asset, m_h = horizontal projection, l = normalize(z x m_h)):
    s = clip((p_d - p_a).m, 0, |p_s - p_a|);  q = p_a + s m;   q_L = q + R l,  q_R = q - R l,  q_U = q + R z   (q_U.z capped at the ceiling)
The defender is tactical guidance only: it is never added to the obstacle set and no collision geometry or margin is changed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uav_defend.obstacles.geometry import AABBObstacle
from uav_defend.obstacles.perception import PerceivedObstacleSet
from uav_defend.obstacles.route_planner import _segment_blocked, build_and_search_route

ACTION_VERSION = "a1_route_action_v1"
OBSERVATION_VERSION = "a1_v2_obs_route_v1"
TRAINING_PROTOCOL = "a1_route_training_v1"

MODES = ("DIRECT", "LEFT", "RIGHT", "ABOVE")
ACTIONS = ("DIRECT_ROUTE", "EVADE_LEFT_ROUTE", "EVADE_RIGHT_ROUTE", "EVADE_ABOVE_ROUTE")
RADII_M = (5.0, 7.5, 10.0)          # pre-registered, never extended
REPLAN_S = (1.0, 2.0)               # pre-registered; dt = 0.5 s -> every 2 / 4 steps
DT_S = 0.5
MIN_AHEAD_M = 1.0                   # candidate rejected unless the closest mission-segment point to the defender is at least this far ahead
MIN_VERTICAL_FRACTION = 0.5         # ABOVE rejected unless the ceiling-capped vertical offset is at least this fraction of R
LATERAL_EPS = 1e-3
ROUTE_RUN_BOUND_STEPS = 40          # a single tactical route active longer than this is a deadlock (G6)
EMPTY_PERCEIVED = PerceivedObstacleSet()


@dataclass(frozen=True)
class RouteParams:
    radius_m: float
    replan_steps: int
    clearance: float
    max_altitude: float
    planning_range: float
    max_graph_nodes: int
    waypoint_tolerance: float

    @staticmethod
    def from_config(config, radius_m: float, replan_s: float) -> "RouteParams":
        return RouteParams(radius_m=float(radius_m), replan_steps=int(round(replan_s / float(config.dt))), clearance=float(config.enemy_obstacle_clearance),
                           max_altitude=float(config.max_altitude), planning_range=float(config.enemy_route_planning_range),
                           max_graph_nodes=int(config.enemy_route_max_graph_nodes), waypoint_tolerance=float(config.enemy_route_waypoint_tolerance))


# ----------------------------------------------------------------------------- geometry (pure)
def mission_geometry(p_a, p_s, p_d) -> dict | None:
    p_a, p_s, p_d = (np.asarray(x, dtype=np.float64) for x in (p_a, p_s, p_d))
    offset = p_s - p_a
    distance = float(np.linalg.norm(offset))
    if distance < 1e-6 or not np.isfinite(offset).all() or not np.isfinite(p_d).all():
        return None
    m = offset / distance
    s = float(np.clip(float(np.dot(p_d - p_a, m)), 0.0, distance))
    return {"m": m, "distance": distance, "s": s, "q": p_a + s * m}


def lateral_axis(m: np.ndarray) -> np.ndarray | None:
    mh = np.array([m[0], m[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(mh))
    if norm < LATERAL_EPS:
        return None
    lateral = np.cross(np.array([0.0, 0.0, 1.0]), mh / norm)
    return lateral / np.linalg.norm(lateral)


def tactical_subgoal(mode: str, p_a, p_s, p_d, params: RouteParams) -> tuple[np.ndarray | None, str | None]:
    """(subgoal, None) or (None, rejection reason)."""
    geo = mission_geometry(p_a, p_s, p_d)
    if geo is None:
        return None, "degenerate_geometry"
    if geo["s"] < MIN_AHEAD_M:
        return None, "defender_not_ahead"
    q, R = geo["q"], params.radius_m
    if mode == "ABOVE":
        ceiling = params.max_altitude - params.clearance
        z = min(float(q[2]) + R, ceiling)
        if z - float(q[2]) < MIN_VERTICAL_FRACTION * R:
            return None, "ceiling"
        goal = np.array([q[0], q[1], z])
    elif mode in ("LEFT", "RIGHT"):
        lateral = lateral_axis(geo["m"])
        if lateral is None:
            return None, "degenerate_geometry"
        goal = q + (R if mode == "LEFT" else -R) * lateral
    else:
        return None, "invalid_mode"
    if not np.isfinite(goal).all() or goal[2] < 0.0 or goal[2] > params.max_altitude:
        return None, "ceiling" if np.isfinite(goal).all() else "non_finite"
    return goal, None


def local_goal(p_a, p_s, planning_range: float) -> np.ndarray:
    """Same horizon rule as the existing `RoutePlanner._local_goal`: the asset itself within the planning range, else a point toward it at that range."""
    p_a, p_s = np.asarray(p_a, dtype=np.float64), np.asarray(p_s, dtype=np.float64)
    offset = p_s - p_a
    distance = float(np.linalg.norm(offset))
    return p_s.copy() if distance <= planning_range or distance < 1e-9 else p_a + offset / distance * planning_range


def inflated_boxes(perceived: PerceivedObstacleSet, clearance: float) -> list:
    return [AABBObstacle.from_center_half_extents((o.min_corner + o.max_corner) / 2.0, (o.max_corner - o.min_corner) / 2.0 + clearance) for o in perceived.static]


def direct_path_clear(p_a, p_s, perceived: PerceivedObstacleSet, clearance: float) -> bool:
    return not _segment_blocked(np.asarray(p_a, dtype=np.float64), np.asarray(p_s, dtype=np.float64), inflated_boxes(perceived, clearance))


def route_blocked(p_a, waypoints: tuple, idx: int, perceived: PerceivedObstacleSet, clearance: float) -> bool:
    boxes = inflated_boxes(perceived, clearance)
    path = (np.asarray(p_a, dtype=np.float64),) + tuple(waypoints[idx:])
    return any(_segment_blocked(path[i], path[i + 1], boxes) for i in range(len(path) - 1))


def polyline_min_distance(point, start, waypoints) -> float:
    """Minimum distance from `point` to the polyline start -> waypoints (closest-approach separation of a candidate route from a position)."""
    point = np.asarray(point, dtype=np.float64)
    pts = [np.asarray(start, dtype=np.float64)] + [np.asarray(w, dtype=np.float64) for w in waypoints]
    best = float("inf")
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        t = 0.0 if denom < 1e-12 else float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
        best = min(best, float(np.linalg.norm(point - (a + t * ab))))
    return best


@dataclass(frozen=True)
class Candidate:
    mode: str
    subgoal: np.ndarray
    waypoints: tuple
    subgoal_index: int
    length: float
    obstacle_ids: tuple


def plan_candidate(mode: str, p_a, p_s, p_d, perceived: PerceivedObstacleSet, params: RouteParams) -> tuple[Candidate | None, str | None]:
    """Validate-before-commit: a finite, collision-free (against the PERCEIVED inflated static set) path attacker -> subgoal -> asset, else a rejection reason."""
    p_a = np.asarray(p_a, dtype=np.float64)
    subgoal, reason = tactical_subgoal(mode, p_a, p_s, p_d, params)
    if subgoal is None:
        return None, reason
    args = (perceived, params.clearance, params.max_altitude, params.planning_range, params.max_graph_nodes)
    leg1 = build_and_search_route(p_a, subgoal, *args)
    if leg1 is None:
        return None, "no_path"
    goal2 = local_goal(subgoal, p_s, params.planning_range)
    leg2 = build_and_search_route(subgoal, goal2, *args)
    if leg2 is None:
        return None, "no_path"
    waypoints = tuple(np.asarray(w, dtype=np.float64) for w in (*leg1, *leg2))
    if not waypoints or any(not np.isfinite(w).all() for w in waypoints):
        return None, "non_finite"
    if any(float(w[2]) > params.max_altitude + 1e-9 or float(w[2]) < -1e-9 for w in waypoints):
        return None, "ceiling"
    boxes = inflated_boxes(perceived, params.clearance)
    path = (p_a,) + waypoints
    if any(_segment_blocked(path[i], path[i + 1], boxes) for i in range(len(path) - 1)):
        return None, "planner_collision"
    length = float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))
    near = tuple(sorted(o.id for o in perceived.static))
    return Candidate(mode, subgoal, waypoints, len(leg1) - 1, length, near), None


# ----------------------------------------------------------------------------- persistent route state machine
@dataclass
class RouteDecision:
    direction: np.ndarray | None
    mode: str                      # route mode in force while this step is flown (after this step's decisions)
    events: list = field(default_factory=list)
    planned: bool = False          # a planner call was made this step
    audit: dict | None = None      # data used by the last planner call (for the perception audit)
    command_applied: bool = False


class RouteController:
    def __init__(self, params: RouteParams):
        self.params = params
        self.reset()

    def reset(self) -> None:
        self.mode = "DIRECT"
        self.waypoints: tuple = ()
        self.idx = 0
        self.subgoal: np.ndarray | None = None
        self.subgoal_index = 0
        self.subgoal_passed = False
        self.countdown = 0
        self.initial_subgoal_distance = 1.0
        self.goal_at_creation: np.ndarray | None = None
        self.progress = 0.0
        self.run_steps = 0
        self.longest_run = 0
        self.lockout = 0
        self.counters = {"plans_attempted": 0, "plans_committed": 0, "replans": 0, "completions": 0, "route_steps": 0, "suppressed_by_obstacle": 0,
                         "invalid": {}, "cancels": {}, "waypoint_total": 0, "length_total": 0.0, "plan_steps": [], "steps": 0}

    # -- observation: mode one-hot (4) + normalized route progress + normalized replan countdown --
    def observation(self) -> np.ndarray:
        onehot = np.zeros(4, dtype=np.float32)
        onehot[MODES.index(self.mode)] = 1.0
        countdown = (float(np.clip(self.lockout / max(self.params.replan_steps, 1), 0.0, 1.0)) if self.mode == "DIRECT"
                     else float(np.clip(self.countdown / max(self.params.replan_steps, 1), 0.0, 1.0)))
        return np.concatenate([onehot, np.array([float(np.clip(self.progress, 0.0, 1.0)), countdown], dtype=np.float32)])

    def _bump(self, bucket: str, key: str) -> None:
        self.counters[bucket][key] = self.counters[bucket].get(key, 0) + 1

    def _cancel(self, reason: str, events: list) -> None:
        self._bump("cancels", reason)
        events.append(f"cancel:{reason}")
        self.mode, self.waypoints, self.idx, self.subgoal, self.countdown, self.progress, self.subgoal_passed = "DIRECT", (), 0, None, 0, 0.0, False

    def _commit(self, cand: Candidate, p_a, p_s, events: list, replan: bool) -> None:
        self.mode, self.waypoints, self.idx = cand.mode, cand.waypoints, 0
        self.subgoal, self.subgoal_index, self.subgoal_passed = cand.subgoal, cand.subgoal_index, False
        self.countdown = self.params.replan_steps
        self.initial_subgoal_distance = max(float(np.linalg.norm(cand.subgoal - np.asarray(p_a))), 1e-6)
        self.goal_at_creation = local_goal(p_a, p_s, self.params.planning_range)
        self.progress = 0.0
        self.counters["plans_committed"] += 1
        self.counters["replans"] += int(replan)
        self.counters["waypoint_total"] += len(cand.waypoints)
        self.counters["length_total"] += cand.length
        events.append("replan" if replan else "start")

    def _plan(self, mode, p_a, p_s, p_d, perceived, step_index, events):
        self.counters["plans_attempted"] += 1
        self.counters["plan_steps"].append(step_index)
        cand, reason = plan_candidate(mode, p_a, p_s, p_d, perceived, self.params)
        audit = {"mode": mode, "obstacle_ids": tuple(sorted(o.id for o in perceived.static)), "n_perceived_static": len(perceived.static)}
        if cand is None:
            self._bump("invalid", reason)
            events.append(f"invalid:{reason}")
        return cand, audit

    def step(self, request: int, p_a, p_s, p_d, info_available: bool, perceived: PerceivedObstacleSet, owner_required: bool) -> RouteDecision:
        p_a, p_s, p_d = (np.asarray(x, dtype=np.float64) for x in (p_a, p_s, p_d))
        step_index = self.counters["steps"]
        self.counters["steps"] += 1
        events: list = []
        planned, audit = False, None
        just_completed = False
        geo = mission_geometry(p_a, p_s, p_d)
        m = None if geo is None else geo["m"]
        cancelled_to_direct_by_request = False

        if self.mode != "DIRECT":
            if not info_available:
                self._cancel("defender_info_lost", events)
            else:
                while self.idx < len(self.waypoints) - 1 and float(np.linalg.norm(self.waypoints[self.idx] - p_a)) <= self.params.waypoint_tolerance:
                    self.idx += 1
                if self.idx > self.subgoal_index:
                    self.subgoal_passed = True
                elif m is not None and (float(np.linalg.norm(self.subgoal - p_a)) <= self.params.waypoint_tolerance or float(np.dot(self.subgoal - p_a, m)) <= 0.0):
                    self.subgoal_passed = True
                if self.subgoal_passed:
                    if direct_path_clear(p_a, p_s, perceived, self.params.clearance):
                        self.counters["completions"] += 1
                        events.append("complete")
                        just_completed = True
                        self.lockout = self.params.replan_steps
                        self.mode, self.waypoints, self.idx, self.subgoal, self.countdown, self.progress, self.subgoal_passed = "DIRECT", (), 0, None, 0, 0.0, False
                    elif route_blocked(p_a, self.waypoints, self.idx, perceived, self.params.clearance):
                        goal = local_goal(p_a, p_s, self.params.planning_range)
                        wp = build_and_search_route(p_a, goal, perceived, self.params.clearance, self.params.max_altitude, self.params.planning_range, self.params.max_graph_nodes)
                        planned = True
                        self.counters["plans_attempted"] += 1
                        self.counters["plan_steps"].append(step_index)
                        audit = {"mode": self.mode, "obstacle_ids": tuple(sorted(o.id for o in perceived.static)), "n_perceived_static": len(perceived.static)}
                        if wp is None:
                            self._bump("invalid", "no_path")
                            self._cancel("no_path_after_subgoal", events)
                        else:
                            self.waypoints, self.idx = tuple(np.asarray(w, dtype=np.float64) for w in wp), 0
                            self.subgoal_index = -1
                            self.counters["replans"] += 1
                            events.append("replan_remainder")
                else:
                    trigger = None
                    if route_blocked(p_a, self.waypoints, self.idx, perceived, self.params.clearance):
                        trigger = "route_blocked"
                    elif self.goal_at_creation is not None and float(np.linalg.norm(self.goal_at_creation - local_goal(p_a, p_s, self.params.planning_range))) > max(
                            self.params.planning_range * 0.5, self.params.clearance * 4.0):
                        trigger = "asset_moved"
                    elif self.countdown <= 0:
                        trigger = "cadence"
                    if trigger is not None:
                        target_mode = MODES[request] if (trigger == "cadence" and 0 <= int(request) < len(MODES)) else self.mode
                        if target_mode == "DIRECT":
                            self._cancel("request_direct", events)
                            cancelled_to_direct_by_request = True
                        else:
                            cand, audit = self._plan(target_mode, p_a, p_s, p_d, perceived, step_index, events)
                            planned = True
                            if cand is None:
                                self._cancel(f"replan_invalid", events)
                            else:
                                self._commit(cand, p_a, p_s, events, replan=True)

        locked = just_completed
        if self.mode == "DIRECT" and not just_completed and self.lockout > 0:
            self.lockout -= 1
            locked = True
        if locked and self.mode == "DIRECT" and int(request) != 0:
            events.append("locked:post_completion_hold")
        if self.mode == "DIRECT" and not locked and not cancelled_to_direct_by_request and int(request) != 0 and info_available:
            cand, audit = self._plan(MODES[int(request)], p_a, p_s, p_d, perceived, step_index, events)
            planned = True
            if cand is not None:
                self._commit(cand, p_a, p_s, events, replan=False)

        direction, applied = None, False
        if self.mode != "DIRECT":
            self.countdown -= 1
            self.counters["route_steps"] += 1
            self.run_steps += 1
            self.longest_run = max(self.longest_run, self.run_steps)
            if not self.subgoal_passed and self.subgoal is not None:
                self.progress = float(np.clip(1.0 - float(np.linalg.norm(self.subgoal - p_a)) / self.initial_subgoal_distance, 0.0, 1.0))
            elif self.subgoal_passed:
                self.progress = 1.0
            # live final waypoint: the last waypoint always tracks the (local) asset goal, as in the existing RoutePlanner
            goal = local_goal(p_a, p_s, self.params.planning_range)
            if len(self.waypoints) >= 1 and self.subgoal_index != len(self.waypoints) - 1:
                self.waypoints = self.waypoints[:-1] + (goal,)
            if owner_required:
                self.counters["suppressed_by_obstacle"] += 1
                events.append("suppressed:obstacle_authority")
            else:
                target = self.waypoints[min(self.idx, len(self.waypoints) - 1)]
                offset = target - p_a
                norm = float(np.linalg.norm(offset))
                if norm > 1e-9 and np.isfinite(offset).all():
                    direction, applied = offset / norm, True
        else:
            self.run_steps = 0
        return RouteDecision(direction=direction, mode=self.mode, events=events, planned=planned, audit=audit, command_applied=applied)
