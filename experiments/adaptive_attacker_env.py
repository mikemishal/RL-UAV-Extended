"""
Adaptive Attacker v1 (journal extension, 29-phase adaptive-attacker
sub-task). Phases 3-15: architecture, observation contract, defender
sensing model, action space, reward, frozen defender pool, opponent
sampling, and the training-environment wrapper itself.

FIRST learned adaptive hostile-UAS policy. Replaces ONLY the hand-designed
hostile's TACTICAL command (pursuit + reactive evasion + route + weave,
see Phase 2 audit in session notes / the final report) with a learned PPO
policy; the LOW-LEVEL safety/navigation stack (obstacle-avoidance planner,
constrained-dynamics limits, boundary handling, physical-collision
outcome determination) is REUSED UNCHANGED via
`SoldierEnv.enemy_tactical_override` (see uav_defend/envs/soldier_env.py)
-- an additive, opt-in hook that is a bit-for-bit no-op when unset.

The DEFENDER is COMPLETELY FROZEN during this phase (Phase 13): a defender
instance is sampled once per episode from a frozen opponent pool (Phase
11/12) and driven by its own (unmodified) policy/model every step; no
defender network parameters are ever updated here.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.sector_features import (
    SECTOR_OBSERVATION_V1,
    SectorObservationSpec,
    compute_sector_features,
)
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info
from uav_defend.policies.residual.lead_residual_composition import DEFAULT_RESIDUAL_MAX_ANGLE_DEG
from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper

from experiments.journal_clutter_protocol import build_canonical_env_config, build_randomized_training_config

PROJECT_ROOT = Path(__file__).parent.parent
CAMPAIGN_ROOT = PROJECT_ROOT / "results" / "clutter_training_400k"

# Phase 5/6: attacker observation = 16-D base (mirrors the defender's own
# base observation layout, roles swapped -- see build_attacker_observation)
# + 20-D obstacle-sector features (Phase 6, reusing sector_features.py
# unchanged) = 36-D, matching PPO-C/PPO-Kalman-C's dimension.
ATTACKER_OBSERVATION_DIM = 16 + SECTOR_OBSERVATION_V1.feature_dim  # 36
ATTACKER_ACTION_DIM = 3  # Phase 8: normalized 3D desired direction

# Phase 15: reserved, non-overlapping seed namespaces for this sub-task
# (distinct from journal_clutter_protocol's CLUTTER_TRAINING_SEEDS
# 71-79/CLUTTER_EVAL_SEED_START=95_000 -- large gaps deliberately chosen).
ATTACKER_TRAINING_SEEDS = (201, 202, 203)
ATTACKER_DIAGNOSTIC_SEED = 200
ATTACKER_VALIDATION_SEED_START = 180_000    # Phase 18: in-training checkpoint selection only
ATTACKER_HELDOUT_EVAL_SEED_START = 190_000  # Phase 19+: final reported evaluation, never seen in training/validation

# Phase 9: attacker reward constants. Magnitudes chosen to mirror the
# defender's own existing terminal-reward scale (reward_intercept=100,
# reward_soldier_caught=-100, reward_unsafe_intercept=-150) for the two
# outcomes that DIRECTLY mirror the defender's WIN/LOSS (soldier_caught is
# the attacker's WIN, symmetric to reward_intercept; intercepted is the
# attacker's LOSS, symmetric to reward_soldier_caught). "unsafe_intercept"
# and obstacle-collision are DISTINCT failure categories the attacker
# spec's Phase 9 template did not fully disambiguate (attacker still failed
# its mission in both, but neither is "the cleanest possible defender win"
# nor "attacker's own doing" in the same way) -- given half magnitude,
# clearly documented, and revisited if diagnostic training (Phase 16)
# reveals reward-hacking around either outcome.
REWARD_ATTACKER_SUCCESS = 100.0           # soldier_caught: attacker reached the asset (WIN)
REWARD_ATTACKER_INTERCEPTED = -100.0      # intercepted: defender safely stopped the attacker (LOSS)
REWARD_ATTACKER_UNSAFE_INTERCEPT = -50.0  # defender's intercept was unsafe, but attacker still didn't reach the asset
REWARD_ATTACKER_COLLISION = -50.0         # attacker's own environmental (obstacle) failure
REWARD_ATTACKER_TIMEOUT = -20.0           # ran out the clock without reaching the asset (not dominant vs. above)
REWARD_PROGRESS_SCALE = 5.0               # mirrors defender's reward_progress_scale
REWARD_TIME_PENALTY = -0.05               # mirrors defender's reward_time_penalty


def build_attacker_observation(
    env: SoldierEnv,
    detection_range: float,
    spec: SectorObservationSpec = SECTOR_OBSERVATION_V1,
) -> np.ndarray:
    """Phase 5/6/7: 36-D attacker observation, structurally mirroring the
    defender's own existing 16-D base layout (roles swapped: attacker's
    own state + protected-asset state + GATED defender-relative state),
    plus the SAME 20-D obstacle-sector features already used by
    PPO-C/PPO-Kalman-C (Phase 6 -- "do not duplicate the observation
    implementation unnecessarily"), computed from the attacker's OWN
    `PerceivedObstacleSet` (`env._enemy_last_perceived`), never ground
    truth (Phase 5 explicit requirement).

    Defender-relative features (Phase 7 sensing contract): zeroed out
    entirely unless the TRUE 3D attacker-defender range is <=
    `detection_range` (`config.enemy_defender_detection_range`) --
    perfect measurement after "detection", consistent with the existing
    hostile model's own post-detection assumptions elsewhere in this
    codebase, and EXPLICITLY DECOUPLED from `_compute_enemy_evasion()`'s
    pre-existing, always-on, ground-truth-based reactive-evasion mechanics
    (audited, unchanged, not a bug -- see Phase 2 audit)."""
    enemy_pos_norm = np.clip(env._normalize_pos(env._enemy_pos), -1.0, 1.0).astype(np.float32)
    soldier_pos_norm = np.clip(env._normalize_pos(env._soldier_pos), -1.0, 1.0).astype(np.float32)
    v_e = env.config.v_e
    enemy_vel_norm = (
        np.clip(env._enemy_vel / v_e, -1.0, 1.0).astype(np.float32) if v_e > 0 else np.zeros(3, dtype=np.float32)
    )

    dist_defender_enemy = float(np.linalg.norm(env._defender_pos - env._enemy_pos))
    defender_detected = dist_defender_enemy <= detection_range
    detected_flag = np.array([1.0 if defender_detected else 0.0], dtype=np.float32)
    if defender_detected:
        defender_pos_info = np.clip(env._normalize_pos(env._defender_pos), -1.0, 1.0).astype(np.float32)
        v_d = env.config.v_d
        defender_vel_info = (
            np.clip(env._defender_vel / v_d, -1.0, 1.0).astype(np.float32) if v_d > 0 else np.zeros(3, dtype=np.float32)
        )
    else:
        defender_pos_info = np.zeros(3, dtype=np.float32)
        defender_vel_info = np.zeros(3, dtype=np.float32)

    base = np.concatenate([
        enemy_pos_norm, soldier_pos_norm, enemy_vel_norm,
        detected_flag, defender_pos_info, defender_vel_info,
    ])
    sector = compute_sector_features(
        defender_pos=env._enemy_pos, defender_vel=env._enemy_vel,
        perceived=env._enemy_last_perceived, sensor_range=spec.sensor_range,
    )
    return np.concatenate([base, sector]).astype(np.float32)


@dataclasses.dataclass(frozen=True)
class DefenderPoolMember:
    """One frozen defender opponent (Phase 11). `policy` is CONSTRUCTED
    ONCE (not per-episode -- dynamics/reward config fields never vary
    across randomized-clutter episodes, only obstacle layout does, so a
    single loaded model/policy instance is reused for the whole training
    run, just `.reset()` between episodes)."""

    family: str            # "Lead" | "PPO-C" | "PPO-Kalman-C" | "LR-PPO-C"
    seed: int | None       # None for the scripted Lead singleton
    estimator: str         # "measurement" | "kalman"
    needs_sector_features: bool
    is_lr_ppo: bool
    policy: object         # instance with act(obs, info[, extra_features])/reset()


def _clutter_checkpoint_path(arch: str, seed: int) -> Path:
    run_dir = CAMPAIGN_ROOT / arch / f"seed_{seed}"
    path = run_dir / "best_model.zip"
    if not path.exists():
        path = run_dir / "final_model.zip"
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint found for {arch} seed_{seed} under {run_dir}")
    return path


def build_defender_pool(
    include_ppo_c: bool = True,
    ppo_kalman_seeds: tuple[int, ...] = (74, 75, 76),
    lr_ppo_seeds: tuple[int, ...] = (77, 78, 79),
    ppo_seeds: tuple[int, ...] = (71, 72, 73),
) -> list[DefenderPoolMember]:
    """Phase 11: frozen defender training pool. Core: Lead (scripted) +
    ALL 3 clutter-trained PPO-Kalman-C seeds + ALL 3 clutter-trained
    LR-PPO-C seeds (never a single favorable checkpoint). PPO-C's 3 seeds
    included too by default (Phase 11: "if inexpensive, include PPO-C as
    well" -- an extra forward pass per step is inexpensive). RMPC
    deliberately EXCLUDED from the TRAINING pool (Phase 11: too slow to
    make attacker training practical) -- retained for EVALUATION only
    (Phase 19/20/24)."""
    # A representative canonical config for policy CONSTRUCTION only
    # (dynamics/reward fields never vary across randomized-clutter
    # episodes -- only obstacle layout does -- so any level works here).
    build_cfg_measurement = build_canonical_env_config("C2", "measurement")
    build_cfg_kalman = build_canonical_env_config("C2", "kalman")

    pool: list[DefenderPoolMember] = [
        DefenderPoolMember(
            family="Lead", seed=None, estimator="measurement",
            needs_sector_features=False, is_lr_ppo=False,
            policy=get_policy("lead", config=build_cfg_measurement),
        )
    ]
    if include_ppo_c:
        for seed in ppo_seeds:
            path = _clutter_checkpoint_path("ppo_clutter_v1", seed)
            pool.append(DefenderPoolMember(
                family="PPO-C", seed=seed, estimator="measurement",
                needs_sector_features=True, is_lr_ppo=False,
                policy=get_policy("ppo", model_path=str(path), deterministic=True),
            ))
    for seed in ppo_kalman_seeds:
        path = _clutter_checkpoint_path("ppo_kalman_clutter_v1", seed)
        pool.append(DefenderPoolMember(
            family="PPO-Kalman-C", seed=seed, estimator="kalman",
            needs_sector_features=True, is_lr_ppo=False,
            policy=get_policy("ppo_kalman", model_path=str(path), deterministic=True),
        ))
    for seed in lr_ppo_seeds:
        path = _clutter_checkpoint_path("lrppo_clutter_v1", seed)
        pool.append(DefenderPoolMember(
            family="LR-PPO-C", seed=seed, estimator="measurement",
            needs_sector_features=True, is_lr_ppo=True,
            policy=LeadResidualPPOPolicyWrapper.load(
                path, residual_max_angle_deg=DEFAULT_RESIDUAL_MAX_ANGLE_DEG,
                config=build_cfg_measurement, deterministic=True,
            ),
        ))
    return pool


def sample_defender(rng: np.random.Generator, pool: list[DefenderPoolMember]) -> DefenderPoolMember:
    """Phase 12: family-balanced-then-seed-within-family opponent sampling.
    Step 1: uniformly select a FAMILY (so e.g. "LR-PPO-C" with 3 seed
    instances does not dominate simply by having more checkpoints than
    "Lead" with 1). Step 2: if the family has >1 instance (a learned
    family), uniformly select one of its frozen seeds."""
    families = sorted(set(member.family for member in pool))
    family = families[int(rng.integers(0, len(families)))]
    candidates = [member for member in pool if member.family == family]
    return candidates[int(rng.integers(0, len(candidates)))]


class AdaptiveAttackerTrainingEnv(gym.Env):
    """Phase 3/4: Gym environment exposing the ATTACKER as the RL-controlled
    agent. Wraps a canonical `SoldierEnv` via the additive
    `enemy_tactical_override` hook (Phase 3 architecture: learned tactical
    command -> [existing, unchanged] obstacle safety -> dynamics ->
    collision). The DEFENDER is driven internally, every step, by a frozen
    policy sampled once per episode from `defender_pool` (Phase 11/12) --
    defender parameters are NEVER updated (Phase 13)."""

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        defender_pool: list[DefenderPoolMember] | None = None,
        sector_spec: SectorObservationSpec = SECTOR_OBSERVATION_V1,
        clutter_config_fn: Callable[[np.random.Generator], EnvConfig] | None = None,
    ):
        super().__init__()
        self._pool = defender_pool if defender_pool is not None else build_defender_pool()
        self._sector_spec = sector_spec
        # Phase 14/15: SAME canonical clutter protocol + randomized
        # training-clutter distribution already validated for defender
        # clutter training (no new environment distribution introduced).
        self._clutter_config_fn = (
            clutter_config_fn if clutter_config_fn is not None
            else (lambda rng: build_randomized_training_config(rng, "measurement"))
        )
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ATTACKER_OBSERVATION_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(ATTACKER_ACTION_DIM,), dtype=np.float32,
        )
        self._env: SoldierEnv | None = None
        self._defender_member: DefenderPoolMember | None = None
        self._pending_action = np.zeros(ATTACKER_ACTION_DIM, dtype=np.float32)
        self._prev_asset_distance: float | None = None
        self._episode_start_asset_distance: float | None = None
        self._step_count = 0

    def _pending_action_getter(self) -> np.ndarray:
        return self._pending_action

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        rng = np.random.default_rng(seed)
        config = self._clutter_config_fn(rng)
        self._env = SoldierEnv(config)
        self._env.enemy_tactical_override = self._pending_action_getter
        self._pending_action = np.zeros(ATTACKER_ACTION_DIM, dtype=np.float32)
        _obs, info = self._env.reset(seed=seed)

        self._defender_member = sample_defender(rng, self._pool)
        self._defender_member.policy.reset()

        self._prev_asset_distance = float(np.linalg.norm(self._env._soldier_pos - self._env._enemy_pos))
        self._episode_start_asset_distance = self._prev_asset_distance
        self._step_count = 0

        attacker_obs = build_attacker_observation(
            self._env, self._env.config.enemy_defender_detection_range, self._sector_spec,
        )
        return attacker_obs, self._augment_info(info)

    def _compute_defender_action(self) -> np.ndarray:
        env = self._env
        member = self._defender_member
        base_obs = env._get_obs()
        info = env._get_info()
        policy_info = build_policy_info(info, member.estimator)
        if not member.needs_sector_features:
            return np.asarray(member.policy.act(base_obs, policy_info), dtype=np.float32)
        sector = compute_sector_features(
            defender_pos=env._defender_pos, defender_vel=env._defender_vel,
            perceived=env._defender_last_perceived, sensor_range=self._sector_spec.sensor_range,
        )
        if member.is_lr_ppo:
            return np.asarray(member.policy.act(base_obs, policy_info, extra_features=sector), dtype=np.float32)
        return np.asarray(member.policy.act(np.concatenate([base_obs, sector]), policy_info), dtype=np.float32)

    def _compute_attacker_reward(self, outcome: str) -> float:
        asset_distance = float(np.linalg.norm(self._env._soldier_pos - self._env._enemy_pos))
        progress_reward = self._prev_asset_distance - asset_distance
        self._prev_asset_distance = asset_distance
        reward = REWARD_PROGRESS_SCALE * progress_reward + REWARD_TIME_PENALTY
        if outcome == "soldier_caught":
            reward += REWARD_ATTACKER_SUCCESS
        elif outcome == "intercepted":
            reward += REWARD_ATTACKER_INTERCEPTED
        elif outcome == "unsafe_intercept":
            reward += REWARD_ATTACKER_UNSAFE_INTERCEPT
        elif outcome in ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"):
            reward += REWARD_ATTACKER_COLLISION
        elif outcome == "timeout":
            reward += REWARD_ATTACKER_TIMEOUT
        return float(reward)

    def step(self, action):
        self._pending_action = np.asarray(action, dtype=np.float32)
        defender_action = self._compute_defender_action()
        _obs, _env_reward, terminated, truncated, info = self._env.step(defender_action)
        self._step_count += 1
        outcome = info["outcome"]
        reward = self._compute_attacker_reward(outcome)
        attacker_obs = build_attacker_observation(
            self._env, self._env.config.enemy_defender_detection_range, self._sector_spec,
        )
        return attacker_obs, reward, terminated, truncated, self._augment_info(info)

    def _augment_info(self, info: dict) -> dict:
        info = dict(info)
        info["adaptive_attacker_defender_family"] = self._defender_member.family
        info["adaptive_attacker_defender_seed"] = self._defender_member.seed
        # Phase 10 (reward-hacking diagnostics): progress toward the asset
        # since episode start, per step so far -- a learned attacker that
        # merely survives longer without making progress is visible here.
        current_asset_distance = float(np.linalg.norm(self._env._soldier_pos - self._env._enemy_pos))
        total_progress = self._episode_start_asset_distance - current_asset_distance
        info["asset_progress_total"] = total_progress
        info["asset_progress_per_step"] = total_progress / self._step_count if self._step_count else 0.0
        return info
