"""Canonical V2 A1 (learned tactical attacker) training protocol: `a1_v2_training_v1`,
reward `a1_v2_reward_v1`. See docs/journal_v2_a1_training_protocol.md.

A1 remains a LEARNED TACTICAL attacker. The PPO action is the desired 3-D direction that replaces the
scripted tactical command through `SoldierEnv.enemy_tactical_override`; the frozen V2 obstacle/safety
layer, vehicle dynamics, swept contact, event ordering, asset motion and finite mission endurance are
untouched. This module only defines what A1 observes, how it is rewarded, which scenarios and fixed
defenders it is trained on, and how the scenario RNG streams are derived. Nothing here changes the
simulator, and nothing here trains.

Frozen A0 benchmark: tag `journal-v2-a0-canonical-v1` (docs/journal_v2_canonical_benchmark.md). Every
environment built here is asserted canonical through
`experiments.journal_v2_protocol.assert_canonical_v2_training_config`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.sector_features import (
    CHANNEL_NAMES,
    SECTOR_NAMES,
    SECTOR_OBSERVATION_V1,
    compute_sector_features,
)
from uav_defend.policies.sanitize import build_policy_info

from experiments import ppo_hyperparams as H
from experiments.journal_v2_protocol import (
    assert_canonical_v2_training_config,
    build_v2_env_config,
)

# ----------------------------------------------------------------------------- identifiers
A1_V2_REWARD_VERSION = "a1_v2_reward_v1"
A1_V2_TRAINING_PROTOCOL = "a1_v2_training_v1"
A1_V2_OBSERVATION_VERSION = "a1_v2_obs_v1"

# ----------------------------------------------------------------------------- opponents / scenarios
# Fixed, deterministic defender families that remain valid under canonical V2. Learned V1 defenders are
# never used, and `passive` is a visualization/diagnostic condition, not a training opponent.
A1_V2_DEFENDER_FAMILIES: tuple[str, ...] = ("greedy", "kalman_greedy", "pn", "lead", "rmpc")
A1_V2_DEFENDER_PROBABILITIES: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2, 0.2)
A1_V2_CLUTTER_LEVELS: tuple[str, ...] = ("C0", "C1", "C2", "C3")
A1_V2_CLUTTER_PROBABILITIES: tuple[float, ...] = (0.25, 0.25, 0.25, 0.25)

# ----------------------------------------------------------------------------- seeds
A1_V2_PPO_SEEDS: tuple[int, ...] = (201, 202, 203)  # independent PPO/model seeds, continuity with V1
A1_V2_SEEN_BY_DEFENDER_ADAPTATION = (201, 202)       # later use only; all three are independently trained
A1_V2_HELD_OUT_FROM_DEFENDER_ADAPTATION = 203

# Disjoint scenario-seed namespaces (SoldierEnv derives every stream from `reset(seed=...)`):
A1_V2_EVAL_SEED_START, A1_V2_EVAL_EPISODES = 140_000, 100            # post-training evaluation 140000-140099
A1_V2_VALIDATION_SEED_START, A1_V2_VALIDATION_EPISODES = 150_000, 40  # in-training validation 150000-150039
A1_V2_TRAIN_SCENARIO_SEED_LOW, A1_V2_TRAIN_SCENARIO_SEED_HIGH = 1_000_000, 2**31 - 1  # training scenarios
A1_V2_SCENARIO_STREAM_TAG = 0xA1D2  # entropy tag of the training scenario stream (never used elsewhere)
SEED_STRIDE = 10_000                # per-worker PPO env seed stride (same scheme as the V1 runners)

# Seed ranges used by earlier protocols that must never be reused for V2 A1 training or evaluation.
RESERVED_EXTERNAL_SEED_RANGES: dict[str, tuple[int, int]] = {
    "v2_a0_validation": (130_000, 130_099),
    "v1_a1_validation": (180_000, 180_999),
    "v1_a1_heldout_eval": (190_000, 190_999),
    "v1_clutter_eval": (95_000, 95_999),
}

# ----------------------------------------------------------------------------- PPO (inherited from V1, not retuned)
A1_V2_PPO_HYPERPARAMS: dict = dict(
    learning_rate=H.LEARNING_RATE, gamma=H.GAMMA, batch_size=H.BATCH_SIZE, n_steps=H.N_STEPS,
    n_epochs=H.N_EPOCHS, clip_range=H.CLIP_RANGE, ent_coef=H.ENT_COEF, vf_coef=H.VF_COEF,
    max_grad_norm=H.MAX_GRAD_NORM, gae_lambda=H.GAE_LAMBDA,
)
A1_V2_POLICY = "MlpPolicy"  # SB3 default: separate pi/vf nets of [64, 64], tanh; no VecNormalize (as V1)
A1_V2_TOTAL_TIMESTEPS = 400_000        # requested budget of the final V1 A1 campaign; actual = 13 x 32768 = 425,984
A1_V2_NUM_ENVS = 16
A1_V2_CHECKPOINT_FREQ = 50_000
A1_V2_VALIDATION_FREQ = 50_000

# ----------------------------------------------------------------------------- reward a1_v2_reward_v1
REWARD_TERMINAL_SUCCESS = 10.0
REWARD_TERMINAL_FAILURE = -10.0
W_PROGRESS = 2.0   # potential-based mission progress
W_PATH = 1.0       # actual integrated path / B_path
W_TIME = 0.25      # elapsed time / T_ref, T_ref = B_path / v_e
SUCCESS_OUTCOME = "soldier_caught"


@dataclass(frozen=True)
class A1Reward:
    """One step's reward and its exact decomposition."""

    terminal: float
    progress: float
    path: float
    time: float

    @property
    def total(self) -> float:
        return self.terminal + self.progress + self.path + self.time


def terminal_reward(outcome: str) -> float:
    """+10 for asset capture; -10 for EVERY other terminal outcome (defender interception, unsafe
    interception, obstacle collision of either vehicle, endurance exhaustion, timeout, ...); 0 while ongoing."""
    if outcome == "ongoing":
        return 0.0
    return REWARD_TERMINAL_SUCCESS if outcome == SUCCESS_OUTCOME else REWARD_TERMINAL_FAILURE


def potential(range_m: float, initial_range_m: float) -> float:
    """Phi(s) = -d / d0 (attacker-to-asset 3-D range normalized by the episode's initial range)."""
    return -float(range_m) / float(initial_range_m)


def a1_v2_step_reward(
    *, range_prev_m: float, range_next_m: float, initial_range_m: float, path_step_m: float,
    path_budget_m: float, dt_s: float, v_e_mps: float, outcome: str, gamma: float = H.GAMMA,
) -> A1Reward:
    """r_t = r_terminal + w_p (gamma Phi(s') - Phi(s)) - w_s dL/B_path - w_t dt/T_ref.

    Terminal transitions: Phi is evaluated at the actual terminal state (the swept contact instant that
    `SoldierEnv` reports), i.e. the potential is NOT zeroed at termination. The shaping therefore
    telescopes to w_p (1 - gamma^T d_T/d_0) in discounted return, is bounded, rewards partial progress
    even in failure, and is deliberately not claimed to be policy-invariant; the +/-10 terminal term
    dominates it. `path_step_m` is the integrated distance the attacker actually flew in the step
    (`enemy_endurance_traveled_m` difference, including a partial final step), never commanded speed.
    """
    t_ref = path_budget_m / v_e_mps
    return A1Reward(
        terminal=terminal_reward(outcome),
        progress=W_PROGRESS * (gamma * potential(range_next_m, initial_range_m) - potential(range_prev_m, initial_range_m)),
        path=-W_PATH * path_step_m / path_budget_m,
        time=-W_TIME * dt_s / t_ref,
    )


# ----------------------------------------------------------------------------- observation a1_v2_obs_v1
A1_V2_BASE_FEATURES: tuple[tuple[str, str], ...] = (
    ("attacker_pos_x", "attacker_own_state"), ("attacker_pos_y", "attacker_own_state"), ("attacker_pos_z", "attacker_own_state"),
    ("asset_pos_x", "asset_state"), ("asset_pos_y", "asset_state"), ("asset_pos_z", "asset_state"),
    ("attacker_vel_x", "attacker_own_state"), ("attacker_vel_y", "attacker_own_state"), ("attacker_vel_z", "attacker_own_state"),
    ("defender_detected_flag", "defender_state"),
    ("defender_pos_x", "defender_state"), ("defender_pos_y", "defender_state"), ("defender_pos_z", "defender_state"),
    ("defender_vel_x", "defender_state"), ("defender_vel_y", "defender_state"), ("defender_vel_z", "defender_state"),
)
A1_V2_SECTOR_FEATURES: tuple[tuple[str, str], ...] = tuple(
    (f"sector_{sector}_{channel}", "obstacle_perception_state") for sector in SECTOR_NAMES for channel in CHANNEL_NAMES
)
A1_V2_ENDURANCE_FEATURE: tuple[str, str] = ("remaining_endurance_fraction", "mission_endurance_state")
A1_V2_OBSERVATION_SCHEMA: tuple[tuple[str, str], ...] = (
    *A1_V2_BASE_FEATURES, *A1_V2_SECTOR_FEATURES, A1_V2_ENDURANCE_FEATURE,
)
A1_V2_V1_OBSERVATION_DIM = 16 + SECTOR_OBSERVATION_V1.feature_dim  # 36: the V1 layout, kept as the prefix
A1_V2_OBSERVATION_DIM = len(A1_V2_OBSERVATION_SCHEMA)               # 37
A1_V2_ACTION_DIM = 3

assert A1_V2_OBSERVATION_DIM == A1_V2_V1_OBSERVATION_DIM + 1


def remaining_endurance_fraction(env: SoldierEnv) -> float:
    """e_t = clip(1 - L_t / B_path, 0, 1) from the environment's own endurance accounting (never recomputed)."""
    budget = env._enemy_endurance_budget
    if budget is None:
        raise RuntimeError("Finite mission endurance is not enabled; this is not the canonical V2 environment.")
    return float(np.clip(1.0 - env._enemy_endurance_traveled / budget, 0.0, 1.0))


def build_a1_v2_observation(env: SoldierEnv, detection_range: float) -> np.ndarray:
    """37-D attacker observation: the 36-D V1 layout (own state, asset, defender gated by the attacker's
    detection range, 20-D obstacle sector features from the attacker's OWN perceived set) followed by
    the normalized remaining endurance fraction. No future information and no ground-truth obstacles."""
    from experiments.adaptive_attacker_env import build_attacker_observation

    base = build_attacker_observation(env, detection_range, SECTOR_OBSERVATION_V1)
    return np.concatenate([base, np.array([remaining_endurance_fraction(env)], dtype=np.float32)]).astype(np.float32)


# ----------------------------------------------------------------------------- scenario streams
def training_scenario_stream(ppo_seed: int, worker_index: int, resume_index: int = 0) -> np.random.Generator:
    """Training scenario RNG: independent of PPO's own seeding and of the global NumPy state, and a
    pure function of (ppo_seed, worker_index, resume_index)."""
    sequence = np.random.SeedSequence([A1_V2_SCENARIO_STREAM_TAG, int(ppo_seed), int(worker_index), int(resume_index)])
    return np.random.default_rng(sequence)


def sample_training_scenario(rng: np.random.Generator) -> tuple[str, str, int]:
    """(clutter level, defender family, scenario seed), levels and families uniform and independent."""
    level = A1_V2_CLUTTER_LEVELS[int(rng.choice(len(A1_V2_CLUTTER_LEVELS), p=A1_V2_CLUTTER_PROBABILITIES))]
    family = A1_V2_DEFENDER_FAMILIES[int(rng.choice(len(A1_V2_DEFENDER_FAMILIES), p=A1_V2_DEFENDER_PROBABILITIES))]
    seed = int(rng.integers(A1_V2_TRAIN_SCENARIO_SEED_LOW, A1_V2_TRAIN_SCENARIO_SEED_HIGH))
    return level, family, seed


def validation_scenarios() -> list[tuple[str, str, int]]:
    """Fixed in-training validation set (checkpoint diagnostics only): every (level, defender) cell twice."""
    return [
        (A1_V2_CLUTTER_LEVELS[i % 4], A1_V2_DEFENDER_FAMILIES[(i // 4) % 5], A1_V2_VALIDATION_SEED_START + i)
        for i in range(A1_V2_VALIDATION_EPISODES)
    ]


def evaluation_seeds() -> range:
    return range(A1_V2_EVAL_SEED_START, A1_V2_EVAL_SEED_START + A1_V2_EVAL_EPISODES)


# ----------------------------------------------------------------------------- fixed defenders
def defender_estimator(family: str) -> str:
    return "kalman" if family == "kalman_greedy" else "measurement"


class A1V2DefenderBank:
    """Fixed analytical defenders, constructed lazily per (family, clutter level) from the canonical V2
    config of that level exactly as the visualizer / research harness construct them."""

    def __init__(self) -> None:
        self._policies: dict[tuple[str, str], object] = {}

    def policy(self, family: str, level: str, config):
        key = (family, level)
        if key not in self._policies:
            from uav_defend.policies.registry import get_policy

            if family not in A1_V2_DEFENDER_FAMILIES:
                raise ValueError(f"Not an A1 V2 training defender: {family!r}")
            if family == "rmpc":
                from experiments.clutter_benchmark_protocol import _frozen_rmpc_config
                from uav_defend.policies.mpc.rmpc_policy import RMPCPolicy

                policy = RMPCPolicy(config=config, rmpc_config=_frozen_rmpc_config())
            elif family in ("greedy", "kalman_greedy"):
                policy = get_policy(family)
            else:
                policy = get_policy(family, config=config)
            self._policies[key] = policy
        return self._policies[key]


def build_episode_config(level: str, family: str):
    config = build_v2_env_config(level, defender_estimator(family))
    assert_canonical_v2_training_config(config)
    return config


# ----------------------------------------------------------------------------- the training environment
class A1V2TrainingEnv(gym.Env):
    """Canonical V2 A1 training environment.

    Each episode: sample (clutter level, defender family, scenario seed) from the worker's scenario
    stream (or take them from `options={"scenario": (level, family, seed)}` for deterministic
    validation / evaluation), build the canonical V2 config, assert it, run `SoldierEnv` with the
    attacker's tactical command supplied by the policy action and the fixed defender driven by its own
    analytical policy. Rewards follow `a1_v2_reward_v1`.
    """

    metadata: dict = {"render_modes": []}

    def __init__(self, ppo_seed: int, worker_index: int = 0, resume_index: int = 0,
                 defender_bank: A1V2DefenderBank | None = None, scripted_a0: bool = False):
        super().__init__()
        self.ppo_seed, self.worker_index, self.resume_index = int(ppo_seed), int(worker_index), int(resume_index)
        # scripted_a0=True runs the unmodified canonical A0 (no tactical override) with identical metrics;
        # used only for the matched A0 comparison in evaluation, never in training.
        self._scripted_a0 = bool(scripted_a0)
        self._bank = defender_bank if defender_bank is not None else A1V2DefenderBank()
        self._stream = training_scenario_stream(ppo_seed, worker_index, resume_index)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(A1_V2_OBSERVATION_DIM,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(A1_V2_ACTION_DIM,), dtype=np.float32)
        self._env: SoldierEnv | None = None
        self._pending_action = np.zeros(A1_V2_ACTION_DIM, dtype=np.float32)
        self.scenario: tuple[str, str, int] | None = None

    @property
    def soldier_env(self) -> SoldierEnv | None:
        return self._env

    def _action(self) -> np.ndarray:
        return self._pending_action

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        # `seed` (SB3 passes ppo_seed + i once) is accepted for gym compliance; scenario selection comes
        # only from the deterministic stream or from explicit options, never from global randomness.
        super().reset(seed=seed)
        if options and "scenario" in options:
            level, family, scenario_seed = options["scenario"]
        else:
            level, family, scenario_seed = sample_training_scenario(self._stream)
        self.scenario = (level, family, int(scenario_seed))
        config = build_episode_config(level, family)
        self._family, self._level = family, level
        self._policy = self._bank.policy(family, level, config)
        self._estimator = defender_estimator(family)
        self._env = SoldierEnv(config)
        self._env.enemy_tactical_override = None if self._scripted_a0 else self._action
        self._pending_action = np.zeros(A1_V2_ACTION_DIM, dtype=np.float32)
        _obs, info = self._env.reset(seed=int(scenario_seed))
        self._policy.reset()

        self._detection_range = float(config.enemy_defender_detection_range)
        self._start_pos = np.asarray(self._env._enemy_pos, dtype=np.float64).copy()
        self._range = self._range_now()
        self._initial_range = self._range
        self._path_prev = float(self._env._enemy_endurance_traveled)
        self._budget = float(self._env._enemy_endurance_budget)
        self._steps = 0
        self._sum = dict(terminal=0.0, progress=0.0, path=0.0, time=0.0)
        return build_a1_v2_observation(self._env, self._detection_range), self._info(info, None)

    def _range_now(self) -> float:
        return float(np.linalg.norm(np.asarray(self._env._soldier_pos, dtype=np.float64)
                                    - np.asarray(self._env._enemy_pos, dtype=np.float64)))

    def _defender_action(self) -> np.ndarray:
        env = self._env
        action = self._policy.act(env._get_obs(), build_policy_info(env._get_info(), self._estimator))
        return np.asarray(action, dtype=np.float32)

    def step(self, action):
        self._pending_action = np.asarray(action, dtype=np.float32).reshape(A1_V2_ACTION_DIM)
        _o, _r, terminated, truncated, info = self._env.step(self._defender_action())
        self._steps += 1
        range_next = self._range_now()
        path_now = float(self._env._enemy_endurance_traveled)
        outcome = info["outcome"]
        reward = a1_v2_step_reward(
            range_prev_m=self._range, range_next_m=range_next, initial_range_m=self._initial_range,
            path_step_m=path_now - self._path_prev, path_budget_m=self._budget, dt_s=float(self._env.config.dt),
            v_e_mps=float(self._env.config.v_e), outcome=outcome,
        )
        self._range, self._path_prev = range_next, path_now
        for key in self._sum:
            self._sum[key] += getattr(reward, key)
        episode = None
        if terminated or truncated:
            chord = float(np.linalg.norm(np.asarray(self._env._soldier_pos, dtype=np.float64) - self._start_pos))
            episode = dict(
                outcome=info["outcome"], level=self._level, defender=self._family, scenario_seed=self.scenario[2],
                steps=self._steps, return_undiscounted=sum(self._sum.values()), components=dict(self._sum),
                path_length_m=path_now, chord_m=chord, path_excess=(path_now / chord) if chord > 1e-9 else float("nan"),
                initial_range_m=self._initial_range, final_range_m=range_next,
                endurance_exhausted=info["outcome"] == "enemy_endurance_exhausted",
            )
        observation = build_a1_v2_observation(self._env, self._detection_range)
        return observation, float(reward.total), bool(terminated), bool(truncated), self._info(info, reward, episode)

    def _info(self, info: dict, reward: A1Reward | None, episode: dict | None = None) -> dict:
        info = dict(info)
        info["a1_scenario"] = {"level": self._level, "defender": self._family, "scenario_seed": self.scenario[2]}
        info["a1_reward_version"] = A1_V2_REWARD_VERSION
        if reward is not None:
            info["a1_reward_components"] = dataclasses.asdict(reward)
        if episode is not None:
            info["a1_episode"] = episode
        return info

    def close(self):
        self._env = None
