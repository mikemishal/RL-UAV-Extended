"""Route-level A1 environment: action `a1_route_action_v1` (Discrete(4) route intent), observation `a1_v2_obs_route_v1` (37-D canonical + 4-way route mode + route
progress + replan countdown = 43-D), reward `a1_v2_reward_v2` (unchanged), provisional training protocol `a1_route_training_v1`.
Authority: physical contact > frozen common obstacle layer > route follower (deterministic planner output) > nothing (DIRECT = canonical A0)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.sanitize import build_policy_info

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_a1_route as RT
from experiments import journal_v2_a1_tg as G  # defender-information gate and the frozen-layer ownership check

ACTION_VERSION, OBSERVATION_VERSION, TRAINING_PROTOCOL = RT.ACTION_VERSION, RT.OBSERVATION_VERSION, RT.TRAINING_PROTOCOL
REWARD_VERSION = R.REWARD_VERSION
OBS_ROUTE_EXTRA_SCHEMA = (("route_mode_direct", "route_state"), ("route_mode_left", "route_state"), ("route_mode_right", "route_state"),
                          ("route_mode_above", "route_state"), ("route_progress", "route_state"), ("route_replan_countdown", "route_state"))
OBS_ROUTE_SCHEMA = (*A.A1_V2_OBSERVATION_SCHEMA, *OBS_ROUTE_EXTRA_SCHEMA)
OBS_ROUTE_DIM = len(OBS_ROUTE_SCHEMA)
assert OBS_ROUTE_DIM == 43
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_training" / TRAINING_PROTOCOL
COLLISIONS = ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision")
SELECTED_CONFIG_FILE = R.SELECTED_THETA_FILE.parent.parent / "journal_v2_a1_route_level_v1" / "selected_configuration.json"


class A1RouteTrainingEnv(gym.Env):
    metadata: dict = {"render_modes": []}

    def __init__(self, ppo_seed: int, worker_index: int = 0, resume_index: int = 0, defender_bank: A.A1V2DefenderBank | None = None, scripted_a0: bool = False,
                 radius_m: float | None = None, replan_s: float | None = None, gamma: float = R.GAMMA, record_trajectory: bool = False):
        super().__init__()
        self.ppo_seed, self.worker_index, self.resume_index = int(ppo_seed), int(worker_index), int(resume_index)
        self._scripted_a0 = bool(scripted_a0)
        selected = None
        if radius_m is None or replan_s is None:
            selected = load_selected_configuration()
        self.radius_m = float(radius_m if radius_m is not None else selected["radius_m"])
        self.replan_s = float(replan_s if replan_s is not None else selected["replan_s"])
        self.gamma, self.record_trajectory = float(gamma), bool(record_trajectory)
        self._bank = defender_bank if defender_bank is not None else A.A1V2DefenderBank()
        self._stream = A.training_scenario_stream(ppo_seed, worker_index, resume_index)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_ROUTE_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._env: SoldierEnv | None = None
        self.controller: RT.RouteController | None = None
        self._pending = 0
        self.scenario: tuple[str, str, int] | None = None

    @property
    def soldier_env(self) -> SoldierEnv | None:
        return self._env

    # -- called by SoldierEnv._move_enemy before the frozen obstacle layer; the layer has the last word --
    def _override(self) -> np.ndarray:
        env = self._env
        available = G.defender_info_available(env)
        owner = G.common_layer_required(env)
        perceived = getattr(env, "_enemy_last_perceived", None) or RT.EMPTY_PERCEIVED
        decision = self.controller.step(self._pending, env._enemy_pos, env._soldier_pos, env._defender_pos, available, perceived, owner["required"])
        if decision.planned:
            ids = set(decision.audit["obstacle_ids"]) if decision.audit else set()
            perceived_ids = {o.id for o in perceived.static}
            self._audit["plan_calls"] += 1
            self._audit["plan_calls_without_info"] += int(not available)
            self._audit["ids_outside_perceived"] += len(ids - perceived_ids)
            unperceived = len(env._obstacle_layout) - len(perceived.static)  # diagnostics only: proves the planner was NOT given the full map when it was larger
            self._audit["plan_calls_with_unperceived_obstacles"] += int(unperceived > 0)
            self._audit["max_unperceived_static"] = max(self._audit["max_unperceived_static"], unperceived)
            self._audit["perceived_static_used_total"] += len(ids)
        self._step_log = {"requested": int(self._pending), "mode": decision.mode, "events": list(decision.events), "planned": bool(decision.planned),
                          "avoid_required": bool(owner["required"]), "info_available": bool(available), "applied": bool(decision.command_applied)}
        return decision.direction if decision.direction is not None else np.zeros(3)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            level, family, scenario_seed = options["scenario"]
        else:
            level, family, scenario_seed = A.sample_training_scenario(self._stream)
        self.scenario = (level, family, int(scenario_seed))
        self._family, self._level = family, level
        config = A.build_episode_config(level, family)
        self._policy = self._bank.policy(family, level, config)
        self._estimator = A.defender_estimator(family)
        self._env = SoldierEnv(config)
        self.controller = RT.RouteController(RT.RouteParams.from_config(config, self.radius_m, self.replan_s))
        self._env.enemy_tactical_override = None if self._scripted_a0 else self._override
        self._pending = 0
        self._step_log = {}
        self._audit = {"plan_calls": 0, "plan_calls_without_info": 0, "ids_outside_perceived": 0, "plan_calls_with_unperceived_obstacles": 0,
                       "max_unperceived_static": 0, "perceived_static_used_total": 0}
        self._env.reset(seed=int(scenario_seed))
        self._policy.reset()
        self._detection_range = float(config.enemy_defender_detection_range)
        self._max_altitude, self._dt = float(config.max_altitude), float(config.dt)
        self._start_pos = np.asarray(self._env._enemy_pos, dtype=np.float64).copy()
        self._prev_pos = self._start_pos.copy()
        self._range = self._range_now()
        self._initial_range = self._range
        self._path_prev = float(self._env._enemy_endurance_traveled)
        self._budget = float(self._env._enemy_endurance_budget)
        self._steps = 0
        self._sum = dict(terminal=0.0, progress=0.0, path=0.0, step=0.0)
        self._discounted = 0.0
        self._log: list[dict] = []
        self._speeds, self._alts, self._att_xyz, self._ast_xyz, self._ranges = [], [], [], [], []
        self._min_sep = float("inf")
        self._traj = [self._start_pos.round(2).tolist()] if self.record_trajectory else []
        return self._observation(), self._info({}, None)

    def _observation(self) -> np.ndarray:
        return np.concatenate([A.build_a1_v2_observation(self._env, self._detection_range), self.controller.observation()]).astype(np.float32)

    def _range_now(self) -> float:
        return float(np.linalg.norm(np.asarray(self._env._soldier_pos, dtype=np.float64) - np.asarray(self._env._enemy_pos, dtype=np.float64)))

    def _defender_action(self) -> np.ndarray:
        env = self._env
        return np.asarray(self._policy.act(env._get_obs(), build_policy_info(env._get_info(), self._estimator)), dtype=np.float32)

    def visible(self) -> bool:
        return G.defender_info_available(self._env)

    def candidate_separations(self) -> dict:
        """For the scripted max-separation policy: closest approach of the defender's CURRENT position to each valid candidate route (DIRECT = straight line).
        Pure (does not touch controller state); uses the same perceived set the controller would use."""
        env = self._env
        p_a, p_s, p_d = (np.asarray(x, dtype=np.float64) for x in (env._enemy_pos, env._soldier_pos, env._defender_pos))
        perceived = getattr(env, "_enemy_last_perceived", None) or RT.EMPTY_PERCEIVED
        out = {0: RT.polyline_min_distance(p_d, p_a, (p_s,))}
        for action, mode in ((1, "LEFT"), (2, "RIGHT"), (3, "ABOVE")):
            cand, _ = RT.plan_candidate(mode, p_a, p_s, p_d, perceived, self.controller.params)
            if cand is not None:
                out[action] = RT.polyline_min_distance(p_d, p_a, cand.waypoints)
        return out

    def step(self, action):
        self._pending = int(action)
        self._step_log = {"requested": int(action), "mode": "DIRECT", "events": [], "planned": False, "avoid_required": False, "info_available": False, "applied": False}
        _o, _r, terminated, truncated, info = self._env.step(self._defender_action())
        self._steps += 1
        pos = np.asarray(info["enemy_pos"], dtype=np.float64)
        self._speeds.append(float(np.linalg.norm(pos - self._prev_pos)) / self._dt)
        self._alts.append(float(pos[2]))
        self._min_sep = min(self._min_sep, float(info["defender_enemy_dist"]))
        self._log.append(dict(self._step_log))
        self._att_xyz.append(pos)
        self._ast_xyz.append(np.asarray(info["soldier_pos"], dtype=np.float64))
        self._ranges.append(float(info["enemy_soldier_dist"]))
        if self.record_trajectory:
            self._traj.append(pos.round(2).tolist())
        self._prev_pos = pos
        range_next = self._range_now()
        path_now = float(self._env._enemy_endurance_traveled)
        outcome = info["outcome"]
        reward = R.step_reward(range_prev_m=self._range, range_next_m=range_next, initial_range_m=self._initial_range,
                               path_step_m=path_now - self._path_prev, path_budget_m=self._budget, outcome=outcome, gamma=self.gamma)
        self._range, self._path_prev = range_next, path_now
        for key in self._sum:
            self._sum[key] += getattr(reward, key)
        self._discounted += (self.gamma ** (self._steps - 1)) * reward.total
        episode = self._episode_summary(outcome, path_now, range_next) if (terminated or truncated) else None
        return self._observation(), float(reward.total), bool(terminated), bool(truncated), self._info(info, reward, episode)

    def _episode_summary(self, outcome: str, path_now: float, range_next: float) -> dict:
        from experiments.journal_v2_validation_core import count_overshoots, orbit_revolutions

        chord = float(np.linalg.norm(self._prev_pos - self._start_pos))
        hover = R.hover_metrics(np.asarray(self._speeds), np.asarray(self._alts), self._max_altitude)
        orbit_max, _ = orbit_revolutions(np.array(self._att_xyz), np.array(self._ast_xyz))
        log = self._log
        c = self.controller.counters
        finite = bool(np.isfinite(np.asarray(self._att_xyz)).all() and np.isfinite(path_now))
        out = dict(
            outcome=outcome, level=self._level, defender=self._family, scenario_seed=self.scenario[2], steps=self._steps,
            return_undiscounted=sum(self._sum.values()), return_discounted=self._discounted, components=dict(self._sum),
            path_length_m=path_now, chord_m=chord, path_excess=(path_now / chord) if chord > 1e-9 else float("nan"),
            initial_range_m=self._initial_range, final_range_m=range_next, progress_net=1.0 - range_next / self._initial_range,
            endurance_exhausted=outcome == "enemy_endurance_exhausted", min_defender_separation_m=self._min_sep,
            mean_altitude_m=float(np.mean(self._alts)), mean_speed_mps=float(np.mean(self._speeds)),
            orbit_revolutions_max=float(orbit_max), overshoots=int(count_overshoots(np.array(self._ranges), self._dt)), finite=finite,
            requested_steps=int(sum(s["requested"] != 0 for s in log)), route_steps=int(c["route_steps"]), longest_route_run=int(self.controller.longest_run),
            plans_attempted=int(c["plans_attempted"]), plans_committed=int(c["plans_committed"]), replans=int(c["replans"]), completions=int(c["completions"]),
            invalid=dict(c["invalid"]), cancels=dict(c["cancels"]), suppressed_by_obstacle=int(c["suppressed_by_obstacle"]),
            waypoint_mean=(c["waypoint_total"] / c["plans_committed"]) if c["plans_committed"] else 0.0,
            route_length_mean=(c["length_total"] / c["plans_committed"]) if c["plans_committed"] else 0.0,
            max_plans_in_4_steps=_max_window(c["plan_steps"], 4), steps_info_available=int(sum(s["info_available"] for s in log)),
            steps_obstacle_owned=int(sum(s["avoid_required"] for s in log)),
            executed_while_owned=int(sum(s["avoid_required"] and s["applied"] for s in log)), audit=dict(self._audit), **hover,
        )
        if self.record_trajectory:
            out["trajectory"] = self._traj
            out["modes"] = [s["mode"] for s in log]
            out["events"] = [s["events"] for s in log]
            out["owned_flags"] = [bool(s["avoid_required"]) for s in log]
        return out

    def _info(self, info: dict, reward, episode: dict | None = None) -> dict:
        info = dict(info)
        info["a1_scenario"] = {"level": self._level, "defender": self._family, "scenario_seed": self.scenario[2]}
        info["a1_reward_version"], info["a1_action_version"], info["a1_observation_version"] = REWARD_VERSION, ACTION_VERSION, OBSERVATION_VERSION
        info["a1_route"] = dict(self._step_log)
        if reward is not None:
            info["a1_reward_components"] = {"terminal": reward.terminal, "progress": reward.progress, "path": reward.path, "step": reward.step}
        if episode is not None:
            info["a1_episode"] = episode
        return info

    def close(self):
        self._env = None


def _max_window(steps: list, width: int) -> int:
    best = 0
    for i, s in enumerate(steps):
        best = max(best, sum(1 for t in steps[i:] if t < s + width))
    return best


def load_selected_configuration() -> dict:
    import json

    if not SELECTED_CONFIG_FILE.exists():
        raise RuntimeError(f"no route configuration has been selected/frozen ({SELECTED_CONFIG_FILE} missing)")
    return json.loads(SELECTED_CONFIG_FILE.read_text())
