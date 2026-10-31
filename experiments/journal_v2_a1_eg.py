"""Defender-evasion GATING interface for the V2 A1 attacker: action `a1_v2_action_v6`, observation `a1_v2_obs_v1` (37-D, unchanged), reward `a1_v2_reward_v2`
(unchanged), provisional training protocol `a1_v2_training_v4`. See docs/journal_v2_a1_evasion_gate_v1/.

A1 does NOT steer. It selects the STRENGTH `b` in {0, .25, .5, .75, 1} with which the EXISTING deterministic primitive `SoldierEnv._compute_enemy_evasion`
(evasion_component = enemy_evasion_gain * alpha(d) * u_away, alpha = clip(1 - d/enemy_evasion_radius, 0, 1), used unmodified, ground-truth defender position)
is added to the mission direction, exactly as the legacy hostile model combined them (u_raw = r_hat + evasion_component), minus the legacy weave/heading-noise terms
that canonical V2 forbids:

        u_base(b) = normalize(m + b * evasion_component),      b = 0  ->  the environment's own u_base, bit-for-bit (canonical A0).

The primitive is stateless (function of the two current positions only), so no maneuver state exists and the 37-D observation is retained. When defender
information is unavailable (true range > enemy_defender_detection_range, the A1 observation's own gate) the executed blend is forced to 0. When the frozen common
stateful obstacle layer owns steering this step (its own trigger, reproduced exactly) the evasion contribution is suppressed (u_base is returned unmodified)
and the planner's command is final. Nothing here changes the simulator, A0, obstacle planner, contact, endurance, asset motion or dynamics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.sanitize import build_policy_info

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_r2 as R
from experiments import journal_v2_a1_tg as G  # reuses the exact common-layer ownership check and defender gate (no pulse state is used)

ACTION_VERSION = "a1_v2_action_v6"
OBSERVATION_VERSION = "a1_v2_obs_v1"
TRAINING_PROTOCOL = "a1_v2_training_v4"
REWARD_VERSION = R.REWARD_VERSION
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_training" / TRAINING_PROTOCOL

BLENDS = (0.0, 0.25, 0.5, 0.75, 1.0)  # pre-registered; never extended after the sweep
SELECTED_ACTION_SET_FILE = R.SELECTED_THETA_FILE.parent.parent / "journal_v2_a1_evasion_gate_v1" / "selected_action_set.json"
COLLISIONS = ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision")


def blended_direction(m: np.ndarray, evasion_component: np.ndarray, blend: float) -> np.ndarray | None:
    """normalize(m + blend * evasion_component); None (=> BASE) when blend is 0, the primitive is inactive, or the sum is degenerate."""
    if blend <= 0.0 or float(np.linalg.norm(evasion_component)) <= 1e-12:
        return None
    raw = np.asarray(m, dtype=np.float64) + float(blend) * np.asarray(evasion_component, dtype=np.float64)
    norm = float(np.linalg.norm(raw))
    return raw / norm if norm > 1e-9 else None


class A1V6TrainingEnv(gym.Env):
    """Canonical V2 A1 environment with the evasion-gating action. `blend_levels` maps action index -> blend strength (default: all five).
    `scripted_a0=True`: the unmodified canonical A0 (no override) for matched comparisons."""

    metadata: dict = {"render_modes": []}

    def __init__(self, ppo_seed: int, worker_index: int = 0, resume_index: int = 0, defender_bank: A.A1V2DefenderBank | None = None,
                 scripted_a0: bool = False, blend_levels: tuple = BLENDS, gamma: float = R.GAMMA, record_trajectory: bool = False):
        super().__init__()
        if not blend_levels or blend_levels[0] != 0.0 or any(not 0.0 <= b <= 1.0 for b in blend_levels):
            raise ValueError("blend_levels must start with BASE = 0.0 and lie in [0, 1]")
        self.ppo_seed, self.worker_index, self.resume_index = int(ppo_seed), int(worker_index), int(resume_index)
        self._scripted_a0 = bool(scripted_a0)
        self.blend_levels = tuple(float(b) for b in blend_levels)
        self.gamma, self.record_trajectory = float(gamma), bool(record_trajectory)
        self._bank = defender_bank if defender_bank is not None else A.A1V2DefenderBank()
        self._stream = A.training_scenario_stream(ppo_seed, worker_index, resume_index)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(A.A1_V2_OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.blend_levels))
        self._env: SoldierEnv | None = None
        self._pending = 0
        self.scenario: tuple[str, str, int] | None = None

    @property
    def soldier_env(self) -> SoldierEnv | None:
        return self._env

    # -- called by SoldierEnv._move_enemy BEFORE the frozen obstacle planner; the planner then has the last word --
    def _override(self) -> np.ndarray:
        env = self._env
        requested = self.blend_levels[self._pending]
        primitive = env._compute_enemy_evasion(env._defender_pos, env._enemy_pos)  # the existing primitive, unmodified (V2 zeroes it inside the command path)
        available = G.defender_info_available(env)
        owner = G.common_layer_required(env)
        suppression, executed, command = None, 0.0, np.zeros(3)
        if requested > 0.0:
            if not available:
                suppression = "no_defender_info"
            elif owner["required"]:
                suppression = "obstacle_authority"
            elif primitive["activation"] <= 1e-6:
                suppression = "evasion_inactive"
            else:
                m = G.base_mission_direction(env)
                u = None if m is None else blended_direction(m, primitive["evasion_component"], requested)
                if u is None:
                    suppression = "degenerate"
                else:
                    executed, command = requested, u
        deflection = 0.0
        if executed > 0.0:
            m = G.base_mission_direction(env)
            deflection = float(np.degrees(np.arccos(np.clip(float(np.dot(command, m)), -1.0, 1.0))))
        self._step_log = {"requested_blend": requested, "executed_blend": executed, "suppression": suppression, "avoid_required": bool(owner["required"]),
                          "info_available": bool(available), "activation": float(primitive["activation"]), "deflection_deg": deflection}
        return command  # zero vector = "no override": the environment flies its own u_base bit-for-bit

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
        self._env.enemy_tactical_override = None if self._scripted_a0 else self._override
        self._pending = 0
        self._step_log = {}
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
        self._speeds: list[float] = []
        self._alts: list[float] = []
        self._att_xyz: list[np.ndarray] = []
        self._ast_xyz: list[np.ndarray] = []
        self._ranges: list[float] = []
        self._min_sep = float("inf")
        self._traj = [self._start_pos.round(2).tolist()] if self.record_trajectory else []
        return self._observation(), self._info({}, None)

    def _observation(self) -> np.ndarray:
        return A.build_a1_v2_observation(self._env, self._detection_range).astype(np.float32)

    def _range_now(self) -> float:
        return float(np.linalg.norm(np.asarray(self._env._soldier_pos, dtype=np.float64) - np.asarray(self._env._enemy_pos, dtype=np.float64)))

    def _defender_action(self) -> np.ndarray:
        env = self._env
        return np.asarray(self._policy.act(env._get_obs(), build_policy_info(env._get_info(), self._estimator)), dtype=np.float32)

    def visible(self) -> bool:
        """Defender information availability as it will be evaluated for the NEXT decision (current positions) — for scripted switching policies."""
        return G.defender_info_available(self._env)

    def step(self, action):
        self._pending = int(action)
        self._step_log = {"requested_blend": self.blend_levels[self._pending], "executed_blend": 0.0, "suppression": None, "avoid_required": False,
                          "info_available": False, "activation": 0.0, "deflection_deg": 0.0}
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
        executed = [s["executed_blend"] > 0.0 for s in log]
        supp: dict = {}
        for s in log:
            if s["suppression"]:
                supp[s["suppression"]] = supp.get(s["suppression"], 0) + 1
        deflections = [s["deflection_deg"] for s in log if s["executed_blend"] > 0.0]
        out = dict(
            outcome=outcome, level=self._level, defender=self._family, scenario_seed=self.scenario[2], steps=self._steps,
            return_undiscounted=sum(self._sum.values()), return_discounted=self._discounted, components=dict(self._sum),
            path_length_m=path_now, chord_m=chord, path_excess=(path_now / chord) if chord > 1e-9 else float("nan"),
            initial_range_m=self._initial_range, final_range_m=range_next, progress_net=1.0 - range_next / self._initial_range,
            endurance_exhausted=outcome == "enemy_endurance_exhausted", min_defender_separation_m=self._min_sep,
            mean_altitude_m=float(np.mean(self._alts)), mean_speed_mps=float(np.mean(self._speeds)),
            orbit_revolutions_max=float(orbit_max), overshoots=int(count_overshoots(np.array(self._ranges), self._dt)),
            evasion_steps=int(sum(executed)), requested_steps=int(sum(s["requested_blend"] > 0.0 for s in log)), suppressed=supp,
            steps_primitive_active=int(sum(s["activation"] > 1e-6 for s in log)), steps_info_available=int(sum(s["info_available"] for s in log)),
            steps_obstacle_owned=int(sum(s["avoid_required"] for s in log)),
            executed_while_owned=int(sum(s["avoid_required"] and s["executed_blend"] > 0.0 for s in log)),
            max_deflection_deg=float(max(deflections)) if deflections else 0.0, mean_deflection_deg=float(np.mean(deflections)) if deflections else 0.0,
            evasion_in_last3=int(sum(executed[-3:])), **hover,
        )
        if self.record_trajectory:
            out["trajectory"] = self._traj
            out["executed_blends"] = [s["executed_blend"] for s in log]
            out["owned_flags"] = [bool(s["avoid_required"]) for s in log]
        return out

    def _info(self, info: dict, reward, episode: dict | None = None) -> dict:
        info = dict(info)
        info["a1_scenario"] = {"level": self._level, "defender": self._family, "scenario_seed": self.scenario[2]}
        info["a1_reward_version"], info["a1_action_version"], info["a1_observation_version"] = REWARD_VERSION, ACTION_VERSION, OBSERVATION_VERSION
        info["a1_evasion"] = dict(self._step_log)
        if reward is not None:
            info["a1_reward_components"] = {"terminal": reward.terminal, "progress": reward.progress, "path": reward.path, "step": reward.step}
        if episode is not None:
            info["a1_episode"] = episode
        return info

    def close(self):
        self._env = None


def load_selected_action_set() -> dict:
    if not SELECTED_ACTION_SET_FILE.exists():
        raise RuntimeError(f"no evasion-gating action set has been selected/frozen ({SELECTED_ACTION_SET_FILE} missing)")
    import json

    return json.loads(SELECTED_ACTION_SET_FILE.read_text())
