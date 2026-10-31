"""Finite defender-evasion tactical geometry for the V2 A1 attacker: action `a1_v2_action_v5`, observation `a1_v2_obs_v2`,
training protocol `a1_v2_training_v3`. See docs/journal_v2_a1_tactical_geometry_v1/.

Pre-registered hypothesis: finite, defender-triggered tactical maneuvers can provide an intelligent attacker with evasive authority without
allowing the learned policy to directly control low-level flight or compromise obstacle safety.

PPO chooses ONLY a discrete maneuver request {BASE, LEFT_PULSE, RIGHT_PULSE, CLIMB_PULSE}. Authority hierarchy (highest first):
physical/contact semantics > frozen common stateful obstacle avoidance > tactical pulse > BASE asset pursuit. The pulse is a finite,
deterministic perturbation of the CURRENT mission direction `m`; BASE (and every suppressed/rejoin/override step) executes the environment's own
`u_base` bit-for-bit. Nothing here changes the simulator, A0, the common obstacle layer, dynamics, asset motion or endurance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles import stateful_avoidance as SA
from uav_defend.policies.sanitize import build_policy_info

from experiments import journal_v2_a1 as A
from experiments import journal_v2_a1_r2 as R

# ----------------------------------------------------------------------------- identifiers
ACTION_VERSION = "a1_v2_action_v5"
OBSERVATION_VERSION = "a1_v2_obs_v2"
TRAINING_PROTOCOL = "a1_v2_training_v3"
REWARD_VERSION = R.REWARD_VERSION  # "a1_v2_reward_v2", unchanged
RESULTS_ROOT = R.V2_RESULTS_DIR / "a1_training" / TRAINING_PROTOCOL

# ----------------------------------------------------------------------------- pre-registered grid (never extended after the sweep)
THETA_GRID_DEG = (10.0, 15.0, 20.0, 30.0)
PULSE_GRID_S = (0.5, 1.0, 1.5, 2.0)
REJOIN_S = 1.0
DT_S = 0.5
REJOIN_STEPS = int(round(REJOIN_S / DT_S))  # 2
DEGENERATE_HORIZONTAL_EPS = 1e-3            # |m_h| below this => lateral axis undefined => BASE

ACTIONS = ("BASE", "LEFT_PULSE", "RIGHT_PULSE", "CLIMB_PULSE")
BASE_A, LEFT_A, RIGHT_A, CLIMB_A = 0, 1, 2, 3
MODES = ("BASE", "TACTICAL_LEFT", "TACTICAL_RIGHT", "TACTICAL_CLIMB", "REJOIN", "OBSTACLE_OVERRIDE")
TACTICAL_MODES = ("TACTICAL_LEFT", "TACTICAL_RIGHT", "TACTICAL_CLIMB")
KIND_OF_ACTION = {LEFT_A: "TACTICAL_LEFT", RIGHT_A: "TACTICAL_RIGHT", CLIMB_A: "TACTICAL_CLIMB"}

# observation v2 = the unchanged 37-D v1 layout + 5-way tactical mode one-hot + remaining fraction
OBS_V2_EXTRA_SCHEMA = (
    ("tactical_mode_base", "tactical_state"), ("tactical_mode_left", "tactical_state"), ("tactical_mode_right", "tactical_state"),
    ("tactical_mode_climb", "tactical_state"), ("tactical_mode_rejoin_or_obstacle_suppressed", "tactical_state"),
    ("tactical_remaining_fraction", "tactical_state"),
)
OBS_V2_SCHEMA = (*A.A1_V2_OBSERVATION_SCHEMA, *OBS_V2_EXTRA_SCHEMA)
OBS_V2_DIM = len(OBS_V2_SCHEMA)
assert OBS_V2_DIM == 43


def pulse_steps(duration_s: float) -> int:
    return int(round(duration_s / DT_S))


# ----------------------------------------------------------------------------- geometry
def lateral_axis(m: np.ndarray) -> np.ndarray | None:
    """l = normalize(z_hat x m_h), m_h = normalize([m_x, m_y, 0]). None when the horizontal mission projection is degenerate."""
    mh = np.array([m[0], m[1], 0.0], dtype=np.float64)
    norm = float(np.linalg.norm(mh))
    if norm < DEGENERATE_HORIZONTAL_EPS:
        return None
    mh /= norm
    lateral = np.cross(np.array([0.0, 0.0, 1.0]), mh)
    return lateral / np.linalg.norm(lateral)


def pulse_direction(kind: str, m, theta_deg: float) -> np.ndarray | None:
    """u_left = normalize(m + tan(theta) l), u_right = normalize(m - tan(theta) l), u_climb = normalize(m + tan(theta_z) z_hat).
    `m` is the CURRENT unit mission direction (recomputed every step by the environment). None => fall back to BASE."""
    m = np.asarray(m, dtype=np.float64)
    norm = float(np.linalg.norm(m))
    if norm < 1e-12:
        return None
    m = m / norm
    tangent = math.tan(math.radians(theta_deg))
    if kind == "TACTICAL_CLIMB":
        u = m + tangent * np.array([0.0, 0.0, 1.0])
    else:
        lateral = lateral_axis(m)
        if lateral is None:
            return None
        u = m + (tangent if kind == "TACTICAL_LEFT" else -tangent) * lateral
    return u / np.linalg.norm(u)


def climb_headroom_ok(altitude_m: float, max_altitude_m: float, max_climb_rate_mps: float, dt_s: float = DT_S) -> bool:
    """CLIMB resolves to BASE when there is less than one step of maximum climb below the ceiling (never requests upward motion at the ceiling)."""
    return (max_altitude_m - altitude_m) >= max_climb_rate_mps * dt_s


# ----------------------------------------------------------------------------- state machine
@dataclass
class Decision:
    executed: str                  # mode label of THIS step ("REJOIN"/"OBSTACLE_OVERRIDE"/"BASE" all execute the environment's own u_base)
    tactical_kind: str | None      # non-None only when a tactical pulse direction must be executed this step
    suppression: str | None        # why a non-BASE request did not take effect (None if honoured or BASE was requested)


@dataclass
class TacticalMachine:
    """Deterministic finite state machine: BASE -> TACTICAL_x (n steps) -> REJOIN (2 steps) -> BASE, with OBSTACLE_OVERRIDE whenever the
    frozen common layer owns steering. `mode` is the state at the START of the next step (what the observation exposes)."""

    pulse_n: int
    rejoin_n: int = REJOIN_STEPS
    mode: str = "BASE"
    remaining: int = 0
    counters: dict = field(default_factory=dict)

    def reset(self) -> None:
        self.mode, self.remaining = "BASE", 0
        self.counters = {"pulses_started": {k: 0 for k in TACTICAL_MODES}, "pulses_cancelled_by_obstacle": 0, "pulses_aborted_geometry": 0,
                         "suppressed": {}, "transitions": {}, "steps_by_mode": {m: 0 for m in MODES}}

    def __post_init__(self):
        if self.pulse_n < 1 or self.rejoin_n < 1:
            raise ValueError("pulse and rejoin must last at least one step")
        self.reset()

    def _count_suppression(self, reason: str | None) -> None:
        if reason:
            self.counters["suppressed"][reason] = self.counters["suppressed"].get(reason, 0) + 1

    def _finish(self, decision: Decision, previous: str) -> Decision:
        self.counters["steps_by_mode"][decision.executed] += 1
        key = f"{previous}->{decision.executed}"
        self.counters["transitions"][key] = self.counters["transitions"].get(key, 0) + 1
        self._count_suppression(decision.suppression)
        return decision

    def observation(self) -> np.ndarray:
        """5-way one-hot [BASE, LEFT, RIGHT, CLIMB, REJOIN-or-OBSTACLE-SUPPRESSED] + remaining fraction in [0, 1]."""
        onehot = np.zeros(5, dtype=np.float32)
        if self.mode == "BASE":
            fraction = 0.0
            onehot[0] = 1.0
        elif self.mode in TACTICAL_MODES:
            onehot[1 + TACTICAL_MODES.index(self.mode)] = 1.0
            fraction = self.remaining / self.pulse_n
        elif self.mode == "REJOIN":
            onehot[4] = 1.0
            fraction = self.remaining / self.rejoin_n
        else:  # OBSTACLE_OVERRIDE: a full rejoin is owed once the common layer releases
            onehot[4] = 1.0
            fraction = 1.0
        return np.concatenate([onehot, np.array([fraction], dtype=np.float32)])

    def step(self, requested: int, *, info_available: bool, avoid_required: bool, geometry_reason=lambda kind: None) -> Decision:
        """One decision. `geometry_reason(kind)` returns None if the pulse geometry is valid now, else a suppression reason."""
        previous = self.mode
        wants = requested != BASE_A
        if avoid_required:                              # 2: frozen common obstacle layer owns steering, immediately
            if self.mode in TACTICAL_MODES:
                self.counters["pulses_cancelled_by_obstacle"] += 1
            self.mode, self.remaining = "OBSTACLE_OVERRIDE", 0
            return self._finish(Decision("OBSTACLE_OVERRIDE", None, "obstacle_authority" if wants else None), previous)
        if self.mode == "OBSTACLE_OVERRIDE":            # released -> mandatory rejoin
            self.mode, self.remaining = "REJOIN", self.rejoin_n
        if self.mode in TACTICAL_MODES:                 # continue the active pulse; requests never restart or extend it
            kind = self.mode
            reason = geometry_reason(kind)
            if reason is None:
                self.remaining -= 1
                if self.remaining == 0:
                    self.mode, self.remaining = "REJOIN", self.rejoin_n
                return self._finish(Decision(kind, kind, "pulse_active" if wants else None), previous)
            self.counters["pulses_aborted_geometry"] += 1
            self.mode, self.remaining = "REJOIN", self.rejoin_n   # abort: fall through to a BASE rejoin this step
        if self.mode == "REJOIN":
            self.remaining -= 1
            if self.remaining == 0:
                self.mode = "BASE"
            return self._finish(Decision("REJOIN", None, "rejoin" if wants else None), previous)
        # BASE
        if not wants:
            return self._finish(Decision("BASE", None, None), previous)
        if not info_available:
            return self._finish(Decision("BASE", None, "no_defender_info"), previous)
        kind = KIND_OF_ACTION.get(int(requested))
        if kind is None:
            return self._finish(Decision("BASE", None, "invalid_action"), previous)
        reason = geometry_reason(kind)
        if reason is not None:
            return self._finish(Decision("BASE", None, reason), previous)
        self.counters["pulses_started"][kind] += 1
        self.mode, self.remaining = kind, self.pulse_n - 1
        if self.remaining == 0:
            self.mode, self.remaining = "REJOIN", self.rejoin_n
        return self._finish(Decision(kind, kind, None), previous)


def check_mode_sequence(modes: list[str], pulse_n: int, rejoin_n: int = REJOIN_STEPS) -> list[str]:
    """Invariants of the state machine on an executed-mode sequence (G5). Returns the list of violations (empty = valid)."""
    problems = []
    i, n = 0, len(modes)
    while i < n:
        m = modes[i]
        if m in TACTICAL_MODES:
            j = i
            while j < n and modes[j] == m:
                j += 1
            if j - i > pulse_n:
                problems.append(f"tactical run of {j - i} > {pulse_n} at {i}")
            # A pulse may start from BASE or directly after a COMPLETE rejoin (the rejoin's last step already returns the machine to BASE);
            # the REJOIN branch below validates that the preceding rejoin was complete.
            if i > 0 and modes[i - 1] not in ("BASE", "REJOIN"):
                problems.append(f"tactical pulse started from {modes[i - 1]} at {i}")
            if j < n and modes[j] not in ("REJOIN", "OBSTACLE_OVERRIDE"):
                problems.append(f"tactical run followed by {modes[j]} at {j}")
            i = j
            continue
        if m == "REJOIN":
            j = i
            while j < n and modes[j] == "REJOIN":
                j += 1
            run = j - i
            if run > rejoin_n:
                problems.append(f"rejoin run of {run} exceeds {rejoin_n} at {i} (rejoin reset/extension)")
            if j < n and modes[j] != "OBSTACLE_OVERRIDE" and run != rejoin_n:
                problems.append(f"rejoin of {run} steps ended by {modes[j]} instead of an obstacle override at {i}")
            i = j
            continue
        if m == "OBSTACLE_OVERRIDE" and i + 1 < n and modes[i + 1] not in ("OBSTACLE_OVERRIDE", "REJOIN"):
            problems.append(f"override followed by {modes[i + 1]} at {i + 1}")
        i += 1
    return problems


# ----------------------------------------------------------------------------- frozen common-layer ownership check
def common_layer_required(env: SoldierEnv) -> dict:
    """Will the frozen common stateful obstacle controller own steering THIS step?

    Exactly its own trigger (`StatefulAvoidanceController.plan`): the layout it will see (perceived static + threatening dynamic swept boxes via
    `SoldierEnv._combined_avoidance_layout`) is non-empty AND the frozen DIRECT rollout from the current float32 state collides. When the DIRECT rollout is
    safe the controller releases and returns `u_base` immediately, so `required` also covers the "active" state. Pure; fails closed (required)
    if the perceived set does not exist. Valid inside the override call (same step, same inputs the planner is about to use)."""
    perceived = getattr(env, "_enemy_last_perceived", None)
    if perceived is None:
        return {"required": True, "layout_size": -1}
    cfg = env.config
    horizon = cfg.enemy_obstacle_prediction_horizon_steps
    layout = env._combined_avoidance_layout(env._enemy_pos, env._enemy_vel, horizon, cfg.enemy_obstacle_clearance,
                                            perceived.to_static_layout(), perceived.dynamic)
    if len(layout) == 0:
        return {"required": False, "layout_size": 0}
    u_b = np.asarray(env._enemy_mission_command, dtype=np.float64)
    context = SA._Context(target=np.asarray(env._soldier_pos), layout=layout, limits=env._enemy_dynamics_limits(), horizon=horizon,
                          stop_radius=cfg.threat_radius, clearance=cfg.enemy_obstacle_clearance, max_altitude=cfg.max_altitude, fallback=u_b)
    safe = SA.direct_is_safe(np.asarray(env._enemy_pos, dtype=np.float32), np.asarray(env._enemy_vel, dtype=np.float32), context)
    return {"required": not safe, "layout_size": len(layout)}


def defender_info_available(env: SoldierEnv) -> bool:
    """The A1 observation's existing gate (`experiments.adaptive_attacker_env.build_attacker_observation`): true defender range <= enemy_defender_detection_range."""
    return float(np.linalg.norm(np.asarray(env._defender_pos, dtype=np.float64) - np.asarray(env._enemy_pos, dtype=np.float64))) <= float(
        env.config.enemy_defender_detection_range)


def build_obs_v2(env: SoldierEnv, detection_range: float, machine: TacticalMachine) -> np.ndarray:
    return np.concatenate([A.build_a1_v2_observation(env, detection_range), machine.observation()]).astype(np.float32)


# ----------------------------------------------------------------------------- environment
PASSIVE_DIAGNOSTIC = "passive"


class A1V5TrainingEnv(gym.Env):
    """Canonical V2 A1 environment with the finite tactical-pulse action (Discrete(4)), observation v2 (43-D) and reward v2.

    `scripted_a0=True`: the unmodified canonical A0 (no override, no machine) for matched comparisons.
    `gate_override=True` and family 'passive' are STUDY-ONLY diagnostics (mission-only geometry stress without a real defender)."""

    metadata: dict = {"render_modes": []}

    def __init__(self, ppo_seed: int, worker_index: int = 0, resume_index: int = 0, defender_bank: A.A1V2DefenderBank | None = None,
                 scripted_a0: bool = False, theta_deg: float | None = None, pulse_s: float | None = None, gamma: float = R.GAMMA,
                 gate_override: bool = False, record_trajectory: bool = False):
        super().__init__()
        self.ppo_seed, self.worker_index, self.resume_index = int(ppo_seed), int(worker_index), int(resume_index)
        self._scripted_a0 = bool(scripted_a0)
        self.theta_deg = float(theta_deg) if theta_deg is not None else load_selected_geometry()["theta_deg"]
        self.pulse_s = float(pulse_s) if pulse_s is not None else load_selected_geometry()["pulse_s"]
        self.gamma, self.gate_override, self.record_trajectory = float(gamma), bool(gate_override), bool(record_trajectory)
        self._bank = defender_bank if defender_bank is not None else A.A1V2DefenderBank()
        self._stream = A.training_scenario_stream(ppo_seed, worker_index, resume_index)
        self.machine = TacticalMachine(pulse_n=pulse_steps(self.pulse_s))
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBS_V2_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(4)
        self._env: SoldierEnv | None = None
        self._pending = BASE_A
        self.scenario: tuple[str, str, int] | None = None

    @property
    def soldier_env(self) -> SoldierEnv | None:
        return self._env

    # -- tactical stage (called by SoldierEnv._move_enemy before the frozen obstacle planner) --
    def _override(self) -> np.ndarray:
        env = self._env
        owner = common_layer_required(env)
        available = True if self.gate_override else defender_info_available(env)
        m = base_mission_direction(env)

        def geometry_reason(kind):
            if m is None or pulse_direction(kind, m, self.theta_deg) is None:
                return "degenerate_geometry"
            if kind == "TACTICAL_CLIMB" and not climb_headroom_ok(float(env._enemy_pos[2]), float(env.config.max_altitude),
                                                                  float(env.config.enemy_max_climb_rate), float(env.config.dt)):
                return "ceiling"
            return None

        decision = self.machine.step(int(self._pending), info_available=available, avoid_required=owner["required"],
                                     geometry_reason=geometry_reason)
        self._step_log = {"requested": int(self._pending), "executed": decision.executed, "suppression": decision.suppression,
                          "avoid_required": bool(owner["required"]), "info_available": bool(available)}
        if decision.tactical_kind is None:
            return np.zeros(3)  # zero vector = "no override": the environment flies its own u_base bit-for-bit
        return pulse_direction(decision.tactical_kind, m, self.theta_deg)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            level, family, scenario_seed = options["scenario"]
        else:
            level, family, scenario_seed = A.sample_training_scenario(self._stream)
        self.scenario = (level, family, int(scenario_seed))
        self._family, self._level = family, level
        if family == PASSIVE_DIAGNOSTIC:
            from experiments.journal_v2_validation_core import build_config

            config, _ = build_config(level, "V2", "passive")
            self._policy = None
        else:
            config = A.build_episode_config(level, family)
            self._policy = self._bank.policy(family, level, config)
        self._estimator = A.defender_estimator(family) if family != PASSIVE_DIAGNOSTIC else "measurement"
        self._env = SoldierEnv(config)
        self._env.enemy_tactical_override = None if self._scripted_a0 else self._override
        self._pending = BASE_A
        self._step_log = {}
        self.machine.reset()
        self._env.reset(seed=int(scenario_seed))
        if self._policy is not None:
            self._policy.reset()
        else:
            from experiments.journal_v2_validation_core import PARKED_DEFENDER

            self._env._defender_pos = np.asarray(PARKED_DEFENDER, dtype=np.float32)
            self._env._defender_vel = np.zeros(3, dtype=np.float32)
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
        self._modes: list[str] = []
        self._speeds: list[float] = []
        self._alts: list[float] = []
        self._att_xyz: list[np.ndarray] = []
        self._ast_xyz: list[np.ndarray] = []
        self._ranges: list[float] = []
        self._min_sep = float("inf")
        self._traj: list[list[float]] = [self._start_pos.round(2).tolist()] if self.record_trajectory else []
        return self._observation(), self._info({}, None)

    def _observation(self) -> np.ndarray:
        return build_obs_v2(self._env, self._detection_range, self.machine)

    def _range_now(self) -> float:
        return float(np.linalg.norm(np.asarray(self._env._soldier_pos, dtype=np.float64) - np.asarray(self._env._enemy_pos, dtype=np.float64)))

    def _defender_action(self) -> np.ndarray:
        if self._policy is None:
            return np.zeros(3, dtype=np.float32)
        env = self._env
        return np.asarray(self._policy.act(env._get_obs(), build_policy_info(env._get_info(), self._estimator)), dtype=np.float32)

    def step(self, action):
        self._pending = int(action)
        self._step_log = {"requested": int(action), "executed": "BASE", "suppression": None, "avoid_required": False, "info_available": False}
        _o, _r, terminated, truncated, info = self._env.step(self._defender_action())
        if self._scripted_a0:
            self._step_log["executed"] = "BASE"
        self._steps += 1
        pos = np.asarray(info["enemy_pos"], dtype=np.float64)
        self._speeds.append(float(np.linalg.norm(pos - self._prev_pos)) / self._dt)
        self._alts.append(float(pos[2]))
        self._min_sep = min(self._min_sep, float(info["defender_enemy_dist"]))
        self._modes.append(self._step_log["executed"])
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
        chord = float(np.linalg.norm(self._prev_pos - self._start_pos))
        hover = R.hover_metrics(np.asarray(self._speeds), np.asarray(self._alts), self._max_altitude)
        modes = self._modes
        tactical = [m in TACTICAL_MODES for m in modes]
        longest = run = 0
        for t in tactical:
            run = run + 1 if t else 0
            longest = max(longest, run)
        from experiments.journal_v2_validation_core import count_overshoots, orbit_revolutions

        orbit_max, _ = orbit_revolutions(np.array(self._att_xyz), np.array(self._ast_xyz))
        out = dict(
            outcome=outcome, level=self._level, defender=self._family, scenario_seed=self.scenario[2], steps=self._steps,
            return_undiscounted=sum(self._sum.values()), return_discounted=self._discounted, components=dict(self._sum),
            path_length_m=path_now, chord_m=chord, path_excess=(path_now / chord) if chord > 1e-9 else float("nan"),
            initial_range_m=self._initial_range, final_range_m=range_next, progress_net=1.0 - range_next / self._initial_range,
            endurance_exhausted=outcome == "enemy_endurance_exhausted", min_defender_separation_m=self._min_sep,
            theta_deg=self.theta_deg, pulse_s=self.pulse_s, tactical_steps=int(sum(tactical)),
            tactical_fraction=float(np.mean(tactical)) if modes else 0.0, longest_tactical_run=longest,
            mean_altitude_m=float(np.mean(self._alts)), mean_speed_mps=float(np.mean(self._speeds)),
            orbit_revolutions_max=float(orbit_max), overshoots=int(count_overshoots(np.array(self._ranges), self._dt)), **hover,
        )
        if not self._scripted_a0:
            c = self.machine.counters
            out.update(pulses_started=dict(c["pulses_started"]), pulses_cancelled_by_obstacle=c["pulses_cancelled_by_obstacle"],
                       pulses_aborted_geometry=c["pulses_aborted_geometry"], suppressed=dict(c["suppressed"]), transitions=dict(c["transitions"]),
                       steps_by_mode=dict(c["steps_by_mode"]), state_machine_violations=len(check_mode_sequence(modes, self.machine.pulse_n)))
        if self.record_trajectory:
            out["trajectory"] = self._traj
            out["modes"] = list(modes)
        return out

    def _info(self, info: dict, reward, episode: dict | None = None) -> dict:
        info = dict(info)
        info["a1_scenario"] = {"level": self._level, "defender": self._family, "scenario_seed": self.scenario[2]}
        info["a1_reward_version"], info["a1_action_version"], info["a1_observation_version"] = REWARD_VERSION, ACTION_VERSION, OBSERVATION_VERSION
        info["a1_tactical"] = dict(self._step_log, state_after=self.machine.mode)
        if reward is not None:
            info["a1_reward_components"] = {"terminal": reward.terminal, "progress": reward.progress, "path": reward.path, "step": reward.step}
        if episode is not None:
            info["a1_episode"] = episode
        return info

    def close(self):
        self._env = None


def base_mission_direction(env: SoldierEnv) -> np.ndarray | None:
    """Current unit mission direction m = (p_asset - p_attacker)/|.| exactly as the environment forms it this step (V2: route planning disabled)."""
    return R.base_direction(env)


# ----------------------------------------------------------------------------- frozen selection
SELECTED_GEOMETRY_FILE = R.SELECTED_THETA_FILE.parent.parent / "journal_v2_a1_tactical_geometry_v1" / "selected_geometry.json"


def load_selected_geometry() -> dict:
    if not SELECTED_GEOMETRY_FILE.exists():
        raise RuntimeError(f"no tactical geometry has been selected/frozen ({SELECTED_GEOMETRY_FILE} missing)")
    import json

    record = json.loads(SELECTED_GEOMETRY_FILE.read_text())
    if record["theta_deg"] not in THETA_GRID_DEG or record["pulse_s"] not in PULSE_GRID_S:
        raise RuntimeError("selected geometry is not on the pre-registered grid")
    return record
