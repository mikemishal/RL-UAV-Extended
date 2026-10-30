"""Corrected V2 A1 learning design: reward `a1_v2_reward_v2`, action contract `a1_v2_action_v2`,
training protocol `a1_v2_training_v2`. See docs/journal_v2_a1_training_protocol_v2.md.

The failed campaign (`a1_v2_training_v1` / `a1_v2_reward_v1`, module `journal_v2_a1`) is preserved unchanged as a
non-canonical ablation. This module reuses its frozen pieces that were NOT at fault (37-D observation,
fixed-defender bank, scenario/seed streams, PPO hyper-parameters) and replaces the reward and the action
semantics. Nothing here changes the simulator, the common obstacle/safety layer, or canonical A0.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.sanitize import build_policy_info

from experiments import journal_v2_a1 as A
from experiments.journal_v2_protocol import V2_RESULTS_DIR

# ----------------------------------------------------------------------------- identifiers
REWARD_VERSION = "a1_v2_reward_v2"
ACTION_VERSION = "a1_v2_action_v2"
TRAINING_PROTOCOL = "a1_v2_training_v2"
OBSERVATION_VERSION = A.A1_V2_OBSERVATION_VERSION  # "a1_v2_obs_v1": unchanged 37-D layout
FAILED_V1 = {"training_protocol": A.A1_V2_TRAINING_PROTOCOL, "reward_version": A.A1_V2_REWARD_VERSION,
             "label": "NON-CANONICAL — FAILED REWARD/ACTION FORMULATION"}

RESULTS_ROOT = V2_RESULTS_DIR / "a1_training" / TRAINING_PROTOCOL
FAILED_V1_RESULTS_ROOT = V2_RESULTS_DIR / "a1_training" / A.A1_V2_TRAINING_PROTOCOL
SELECTED_THETA_FILE = Path(__file__).resolve().parents[1] / "docs" / "journal_v2_a1_campaign_v2" / "selected_theta.json"

# ----------------------------------------------------------------------------- reward a1_v2_reward_v2
GAMMA = A.A1_V2_PPO_HYPERPARAMS["gamma"]  # 0.99, the PPO discount, never changed
R_F = 10.0                                # failure-terminal magnitude
TERMINAL_SUCCESS = 10.0
TERMINAL_FAILURE = -10.0
W_PROGRESS = 2.0
W_PATH = 1.0
# Discount-compensating step cost: -(1 - gamma) R_F = -0.1, so that sum_t gamma^t (-0.1) = -R_F = -10:
# surviving forever without capture is worth exactly as much as an immediate failure, never more.
STEP_COST = -0.1
assert math.isclose(STEP_COST, -(1.0 - GAMMA) * R_F, rel_tol=1e-9)
SUCCESS_OUTCOME = A.SUCCESS_OUTCOME


@dataclass(frozen=True)
class R2Reward:
    terminal: float
    progress: float
    path: float
    step: float

    @property
    def total(self) -> float:
        return self.terminal + self.progress + self.path + self.step


def terminal_reward(outcome: str) -> float:
    """+10 capture; -10 for every other terminal (interception, unsafe interception, obstacle collision,
    endurance exhaustion, timeout, any other attacker failure); 0 while ongoing."""
    if outcome == "ongoing":
        return 0.0
    return TERMINAL_SUCCESS if outcome == SUCCESS_OUTCOME else TERMINAL_FAILURE


def phi(range_m: float, initial_range_m: float) -> float:
    """Bounded normalized mission progress Phi(s) = clip(1 - d/d0, 0, 1)."""
    return float(np.clip(1.0 - float(range_m) / float(initial_range_m), 0.0, 1.0))


def step_reward(*, range_prev_m: float, range_next_m: float, initial_range_m: float, path_step_m: float,
                path_budget_m: float, outcome: str, gamma: float = GAMMA) -> R2Reward:
    """r_t = r_terminal + 2 (gamma Phi' - Phi) - dL/B_path - 0.1.

    At constant Phi the shaping term is 2 (gamma - 1) Phi <= 0, so holding a range can never earn progress
    reward. Phi is evaluated at the actual (swept-contact) terminal state and is not zeroed at termination."""
    return R2Reward(
        terminal=terminal_reward(outcome),
        progress=W_PROGRESS * (gamma * phi(range_next_m, initial_range_m) - phi(range_prev_m, initial_range_m)),
        path=-W_PATH * path_step_m / path_budget_m,
        step=STEP_COST,
    )


# ----------------------------------------------------------------------------- action a1_v2_action_v2
THETA_CANDIDATES_DEG = (15.0, 30.0, 45.0, 60.0)
PARALLEL_EPS = 1e-12


def load_selected_theta() -> float:
    if not SELECTED_THETA_FILE.exists():
        raise RuntimeError(f"theta_max has not been selected/frozen yet ({SELECTED_THETA_FILE} missing)")
    value = float(json.loads(SELECTED_THETA_FILE.read_text())["theta_max_deg"])
    if value not in THETA_CANDIDATES_DEG:
        raise RuntimeError(f"frozen theta_max {value} is not one of {THETA_CANDIDATES_DEG}")
    return value


def base_direction(env: SoldierEnv) -> np.ndarray | None:
    """The canonical base guidance direction u_base the environment is about to use for the attacker this step
    (the route command when route planning is active, else the direct pursuit direction; canonical V2 uses
    direct pursuit). Valid only inside `SoldierEnv._move_enemy`, i.e. when the tactical override is called."""
    route = getattr(env, "_enemy_route_command", None)
    vector = route if (route is not None and np.linalg.norm(route) > 1e-9) else env._enemy_mission_command
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else None


def residual_command(raw_action, u_base: np.ndarray, theta_max_deg: float) -> tuple[np.ndarray, float]:
    """Bounded residual around base guidance.

        a_perp = a - (a . u_b) u_b ;  if |a_perp| > 1: a_perp /= |a_perp|
        u_A1   = normalize(u_b + tan(theta_max) a_perp)          =>  angle(u_A1, u_b) <= theta_max

    Returns (command, deviation angle in degrees). An exactly-zero perpendicular residual returns the zero
    vector, which `SoldierEnv` treats as "no override" and therefore flies its own u_base bit-for-bit.
    The component along u_b is discarded (A1 cannot command speed or reverse the mission direction)."""
    u_b = np.asarray(u_base, dtype=np.float64)
    a = np.clip(np.asarray(raw_action, dtype=np.float64).reshape(3), -1.0, 1.0)
    a_perp = a - float(np.dot(a, u_b)) * u_b
    norm = float(np.linalg.norm(a_perp))
    if norm < PARALLEL_EPS:
        return np.zeros(3), 0.0
    if norm > 1.0:
        a_perp = a_perp / norm
    u = u_b + math.tan(math.radians(theta_max_deg)) * a_perp
    u = u / np.linalg.norm(u)
    angle = math.degrees(math.acos(float(np.clip(np.dot(u, u_b), -1.0, 1.0))))
    return u, angle


# ----------------------------------------------------------------------------- hover diagnostics (never a reward)
HOVER_SPEED_MPS = 2.0          # actual physical speed below which the vehicle is "near-hover" (v_e = 12 m/s)
HOVER_MIN_RUN_STEPS = 20       # ... for at least 20 consecutive steps (10 s)
SUSTAINED_HOVER_FRACTION = 0.25  # an episode is sustained near-hover if >= 25% of its steps lie in such runs
SUSTAINED_HOVER_MIN_STEPS = 40
CEILING_MARGIN_M = 1.0         # within 1 m of the altitude ceiling
CEILING_HOVER_SHARE = 0.5      # ceiling-hover: >= half of the near-hover steps are at the ceiling


def hover_metrics(speeds: np.ndarray, altitudes: np.ndarray, max_altitude: float) -> dict:
    """Physical-motion diagnostics from the realised trajectory (not from the PPO command)."""
    speeds = np.asarray(speeds, dtype=np.float64)
    altitudes = np.asarray(altitudes, dtype=np.float64)
    n = len(speeds)
    if n == 0:
        return {"near_hover_fraction": 0.0, "sustained_hover": False, "ceiling_hover": False, "ceiling_occupancy": 0.0,
                "longest_hover_run_steps": 0}
    slow = speeds < HOVER_SPEED_MPS
    at_ceiling = altitudes >= max_altitude - CEILING_MARGIN_M
    in_run = np.zeros(n, dtype=bool)
    longest = 0
    i = 0
    while i < n:
        if slow[i]:
            j = i
            while j < n and slow[j]:
                j += 1
            longest = max(longest, j - i)
            if j - i >= HOVER_MIN_RUN_STEPS:
                in_run[i:j] = True
            i = j
        else:
            i += 1
    hover_steps = int(in_run.sum())
    fraction = hover_steps / n
    sustained = hover_steps >= SUSTAINED_HOVER_MIN_STEPS and fraction >= SUSTAINED_HOVER_FRACTION
    ceiling_share = float(at_ceiling[in_run].mean()) if hover_steps else 0.0
    return {"near_hover_fraction": fraction, "sustained_hover": bool(sustained),
            "ceiling_hover": bool(sustained and ceiling_share >= CEILING_HOVER_SHARE),
            "ceiling_occupancy": float(at_ceiling.mean()), "longest_hover_run_steps": int(longest)}


# ----------------------------------------------------------------------------- frozen acceptance gates
GATES = {
    "capture_every_seed_positive": "every A1 seed has pooled capture rate > 0 (zero capture is an automatic failure)",
    "multi_seed_viability": "at least 2 of 3 seeds have capture > 0 in at least 5 of the 20 (defender x clutter) cells",
    "c0_endurance_max_rate": 0.05,
    "collision_max_rate_per_seed_and_level": 0.02,
    "hover_max_fraction_sustained_per_seed": 0.05,
    "ceiling_hover_max_fraction_per_seed": 0.01,
    "path_success_gt2x_max_fraction": 0.10,
    "path_success_gt5x_max_fraction": 0.01,
    "path_success_gt5x_any_requires_manual_review": True,
    "mission_advantage": ("exists a defender family such that in at least 2 of 3 seeds the paired A1 - A0 capture "
                          "difference, pooled over the 4 clutter levels (400 matched pairs), has point estimate > 0 "
                          "and a paired-bootstrap 95% CI lower bound > 0"),
    "seed_spread_max": 0.20,
    "seed_max_shortfall_vs_a0": 0.10,
}


# ----------------------------------------------------------------------------- the training environment
class A1V2R2TrainingEnv(gym.Env):
    """Canonical V2 A1 training/evaluation environment with reward v2 and the residual action contract.

    observation: 37-D `a1_v2_obs_v1`; action: Box(-1, 1, 3) raw residual `a`, mapped by `residual_command`.
    `scripted_a0=True` runs the unmodified canonical A0 (no override) for the matched comparison."""

    metadata: dict = {"render_modes": []}

    def __init__(self, ppo_seed: int, worker_index: int = 0, resume_index: int = 0,
                 defender_bank: A.A1V2DefenderBank | None = None, scripted_a0: bool = False,
                 theta_max_deg: float | None = None, gamma: float = GAMMA):
        super().__init__()
        self.ppo_seed, self.worker_index, self.resume_index = int(ppo_seed), int(worker_index), int(resume_index)
        self._scripted_a0 = bool(scripted_a0)
        self.theta_max_deg = float(theta_max_deg) if theta_max_deg is not None else load_selected_theta()
        self.gamma = float(gamma)
        self._bank = defender_bank if defender_bank is not None else A.A1V2DefenderBank()
        self._stream = A.training_scenario_stream(ppo_seed, worker_index, resume_index)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(A.A1_V2_OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(A.A1_V2_ACTION_DIM,), dtype=np.float32)
        self._env: SoldierEnv | None = None
        self._pending = np.zeros(3, dtype=np.float32)
        self._last_angle = 0.0
        self.scenario: tuple[str, str, int] | None = None

    @property
    def soldier_env(self) -> SoldierEnv | None:
        return self._env

    def _override(self) -> np.ndarray:
        u_b = base_direction(self._env)
        if u_b is None:
            self._last_angle = 0.0
            return np.zeros(3)
        command, self._last_angle = residual_command(self._pending, u_b, self.theta_max_deg)
        return command

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "scenario" in options:
            level, family, scenario_seed = options["scenario"]
        else:
            level, family, scenario_seed = A.sample_training_scenario(self._stream)
        self.scenario = (level, family, int(scenario_seed))
        config = A.build_episode_config(level, family)
        self._family, self._level = family, level
        self._policy = self._bank.policy(family, level, config)
        self._estimator = A.defender_estimator(family)
        self._env = SoldierEnv(config)
        self._env.enemy_tactical_override = None if self._scripted_a0 else self._override
        self._pending = np.zeros(3, dtype=np.float32)
        self._last_angle = 0.0
        _obs, info = self._env.reset(seed=int(scenario_seed))
        self._policy.reset()
        self._detection_range = float(config.enemy_defender_detection_range)
        self._max_altitude = float(config.max_altitude)
        self._dt = float(config.dt)
        self._start_pos = np.asarray(self._env._enemy_pos, dtype=np.float64).copy()
        self._prev_pos = self._start_pos.copy()
        self._range = self._range_now()
        self._initial_range = self._range
        self._path_prev = float(self._env._enemy_endurance_traveled)
        self._budget = float(self._env._enemy_endurance_budget)
        self._steps = 0
        self._sum = dict(terminal=0.0, progress=0.0, path=0.0, step=0.0)
        self._discounted = 0.0
        self._angles: list[float] = []
        self._speeds: list[float] = []
        self._hspeeds: list[float] = []
        self._alts: list[float] = []
        return A.build_a1_v2_observation(self._env, self._detection_range), self._info(info, None)

    def _range_now(self) -> float:
        return float(np.linalg.norm(np.asarray(self._env._soldier_pos, dtype=np.float64)
                                    - np.asarray(self._env._enemy_pos, dtype=np.float64)))

    def _defender_action(self) -> np.ndarray:
        env = self._env
        action = self._policy.act(env._get_obs(), build_policy_info(env._get_info(), self._estimator))
        return np.asarray(action, dtype=np.float32)

    def step(self, action):
        self._pending = np.asarray(action, dtype=np.float32).reshape(A.A1_V2_ACTION_DIM)
        _o, _r, terminated, truncated, info = self._env.step(self._defender_action())
        self._steps += 1
        pos = np.asarray(info["enemy_pos"], dtype=np.float64)
        displacement = pos - self._prev_pos
        self._speeds.append(float(np.linalg.norm(displacement)) / self._dt)
        self._hspeeds.append(float(np.linalg.norm(displacement[:2])) / self._dt)
        self._alts.append(float(pos[2]))
        self._angles.append(0.0 if self._scripted_a0 else self._last_angle)
        self._prev_pos = pos
        range_next = self._range_now()
        path_now = float(self._env._enemy_endurance_traveled)
        outcome = info["outcome"]
        reward = step_reward(range_prev_m=self._range, range_next_m=range_next, initial_range_m=self._initial_range,
                             path_step_m=path_now - self._path_prev, path_budget_m=self._budget, outcome=outcome,
                             gamma=self.gamma)
        self._range, self._path_prev = range_next, path_now
        for key in self._sum:
            self._sum[key] += getattr(reward, key)
        self._discounted += (self.gamma ** (self._steps - 1)) * reward.total
        episode = None
        if terminated or truncated:
            episode = self._episode_summary(outcome, path_now, range_next)
        observation = A.build_a1_v2_observation(self._env, self._detection_range)
        return observation, float(reward.total), bool(terminated), bool(truncated), self._info(info, reward, episode)

    def _episode_summary(self, outcome: str, path_now: float, range_next: float) -> dict:
        chord = float(np.linalg.norm(self._prev_pos - self._start_pos))
        angles = np.asarray(self._angles, dtype=np.float64)
        theta = self.theta_max_deg
        hover = hover_metrics(np.asarray(self._speeds), np.asarray(self._alts), self._max_altitude)
        return dict(
            outcome=outcome, level=self._level, defender=self._family, scenario_seed=self.scenario[2], steps=self._steps,
            duration_s=self._steps * self._dt, return_undiscounted=sum(self._sum.values()), return_discounted=self._discounted,
            components=dict(self._sum), path_length_m=path_now, chord_m=chord,
            path_excess=(path_now / chord) if chord > 1e-9 else float("nan"), initial_range_m=self._initial_range,
            final_range_m=range_next, progress_net=1.0 - range_next / self._initial_range,
            endurance_exhausted=outcome == "enemy_endurance_exhausted",
            endurance_fraction_final=float(np.clip(1.0 - path_now / self._budget, 0.0, 1.0)),
            residual_angle_mean_deg=float(angles.mean()), residual_angle_p95_deg=float(np.percentile(angles, 95)),
            residual_fraction_near_theta=float((angles >= 0.9 * theta).mean()), theta_max_deg=theta,
            mean_altitude_m=float(np.mean(self._alts)), mean_speed_mps=float(np.mean(self._speeds)),
            mean_horizontal_speed_mps=float(np.mean(self._hspeeds)), **hover,
        )

    def _info(self, info: dict, reward: R2Reward | None, episode: dict | None = None) -> dict:
        info = dict(info)
        info["a1_scenario"] = {"level": self._level, "defender": self._family, "scenario_seed": self.scenario[2]}
        info["a1_reward_version"] = REWARD_VERSION
        info["a1_action_version"] = ACTION_VERSION
        info["a1_residual_angle_deg"] = 0.0 if self._scripted_a0 else self._last_angle
        if reward is not None:
            info["a1_reward_components"] = {"terminal": reward.terminal, "progress": reward.progress, "path": reward.path,
                                            "step": reward.step}
        if episode is not None:
            info["a1_episode"] = episode
        return info

    def close(self):
        self._env = None


def protocol_description() -> dict:
    """Everything that defines the corrected protocol; embedded in every manifest."""
    return {
        "training_protocol": TRAINING_PROTOCOL, "reward_version": REWARD_VERSION, "action_version": ACTION_VERSION,
        "observation_version": OBSERVATION_VERSION, "observation_dim": A.A1_V2_OBSERVATION_DIM,
        "action_dim": A.A1_V2_ACTION_DIM, "theta_max_deg": load_selected_theta(),
        "reward": {"terminal_success": TERMINAL_SUCCESS, "terminal_failure": TERMINAL_FAILURE, "w_progress": W_PROGRESS,
                   "w_path": W_PATH, "step_cost": STEP_COST, "gamma": GAMMA, "R_F": R_F,
                   "potential": "Phi = clip(1 - d/d0, 0, 1)"},
        "ppo_hyperparameters": dict(A.A1_V2_PPO_HYPERPARAMS), "ppo_policy": A.A1_V2_POLICY,
        "num_envs": A.A1_V2_NUM_ENVS, "n_rollouts": 13, "timesteps_per_rollout": A.A1_V2_NUM_ENVS * A.A1_V2_PPO_HYPERPARAMS["n_steps"],
        "total_timesteps": 13 * A.A1_V2_NUM_ENVS * A.A1_V2_PPO_HYPERPARAMS["n_steps"],
        "defender_sampling": dict(zip(A.A1_V2_DEFENDER_FAMILIES, A.A1_V2_DEFENDER_PROBABILITIES)),
        "clutter_sampling": dict(zip(A.A1_V2_CLUTTER_LEVELS, A.A1_V2_CLUTTER_PROBABILITIES)),
        "endurance_path_factor": 5.0, "asset_motion_model": "legacy_gaussian_random_walk",
        "hover_diagnostic": {"speed_mps": HOVER_SPEED_MPS, "min_run_steps": HOVER_MIN_RUN_STEPS,
                             "sustained_fraction": SUSTAINED_HOVER_FRACTION, "sustained_min_steps": SUSTAINED_HOVER_MIN_STEPS,
                             "ceiling_margin_m": CEILING_MARGIN_M, "ceiling_hover_share": CEILING_HOVER_SHARE},
        "gates": GATES, "failed_v1": FAILED_V1,
    }
