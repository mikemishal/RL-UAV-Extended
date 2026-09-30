"""
Defender adaptation (D1) opponent-mixture training environment (journal
extension, 34-phase defender-adaptation sub-task, Phase 5-7).

This is the DEFENDER-adaptation mirror image of
`experiments/adaptive_attacker_env.py::AdaptiveAttackerTrainingEnv`: there,
the ATTACKER is RL-controlled and the DEFENDER is a frozen opponent sampled
per episode; here, the DEFENDER is RL-controlled (via the architecture's
own EXISTING, UNCHANGED clutter-training environment -- Phase 14 randomized
clutter distribution, identical observation/action contract) and the
HOSTILE's tactical behavior is sampled per episode from a fixed mixture:

  - "A0"            (weight 0.5 ): the existing hand-designed scripted
                     hostile, reproduced EXACTLY by leaving
                     `SoldierEnv.enemy_tactical_override = None` (Phase 2
                     audit: bit-for-bit legacy behavior -- see session
                     notes / pre-launch readiness report).
  - "A1-seen-<seed>" (weight 0.25 each, seed in A1_SEEN_SEEDS): Adaptive
                     Attacker v1's own learned tactical policy for that
                     seed, loaded FROZEN/inference-only (`PPO.load(...)`,
                     `.predict(..., deterministic=True)` -- `.learn()` is
                     NEVER called on these models anywhere in this
                     module), driving the hostile via the SAME
                     `enemy_tactical_override` hook Adaptive Attacker v1's
                     own training environment uses.

A1_UNSEEN_SEED is NEVER added to the opponent pool here -- it structurally
cannot be sampled during D1 training or validation (see Phase 9/Phase-18
guarantee: `build_opponent_pool()` only ever consults `A1_SEEN_SEEDS`).

Per-episode opponent selection is DETERMINISTIC for a fixed `reset(seed=...)`
call (a dedicated RNG derived from that seed alone -- Phase 7: "deterministic
for a fixed training seed") and FIXED for the entire episode (never changes
mid-episode). Mixture weights are constants (`MIXTURE_WEIGHTS`) and are never
adjusted at runtime by this module (Phase 7: "no curriculum, weights never
change mid-run").
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Callable

import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.obstacles.sector_features import SectorObservationSpec, SECTOR_OBSERVATION_V1

from experiments.adaptive_attacker_env import (
    ATTACKER_TRAINING_SEEDS,
    build_attacker_observation,
)
from experiments.adaptive_attacker_evaluation import _attacker_checkpoint_path

# Phase 7 mandatory split: SORT the 3 Adaptive Attacker v1 seeds (201, 202,
# 203) and take the first two as "seen" (the D1 training/validation
# opponent pool) and the last as "unseen" (held out ENTIRELY from training
# and model selection -- final-generalization evaluation only). This is an
# ID-sort, NEVER a performance-based selection (explicitly required by the
# spec; do not replace with, e.g., RMPC_BEST_ATTACKER_SEED from the prior
# sub-task's evaluation, which is an unrelated cost-saving heuristic for a
# different evaluation).
_SORTED_ATTACKER_SEEDS = tuple(sorted(ATTACKER_TRAINING_SEEDS))
A1_SEEN_SEEDS: tuple[int, ...] = _SORTED_ATTACKER_SEEDS[:2]
A1_UNSEEN_SEED: int = _SORTED_ATTACKER_SEEDS[2]

MIXTURE_WEIGHTS: dict[str, float] = {"A0": 0.5}
for _seed in A1_SEEN_SEEDS:
    MIXTURE_WEIGHTS[f"A1-seen-{_seed}"] = 0.25
assert abs(sum(MIXTURE_WEIGHTS.values()) - 1.0) < 1e-9


@dataclasses.dataclass(frozen=True)
class OpponentMember:
    """One opponent in the D1 mixture. `policy` is a frozen SB3 `PPO`
    instance (inference-only) for a learned opponent, or `None` for the
    scripted A0 opponent (meaning: leave `enemy_tactical_override` unset)."""

    name: str              # "A0" | f"A1-seen-{seed}"
    kind: str              # "scripted" | "learned"
    seed: int | None
    policy: PPO | None


def build_opponent_pool(a1_seen_seeds: tuple[int, ...] = A1_SEEN_SEEDS) -> list[OpponentMember]:
    """Builds the D1 opponent pool: A0 (scripted) + one frozen, inference-
    only `PPO.load(...)` model per seed in `a1_seen_seeds`. `a1_seen_seeds`
    defaults to the mandated sorted-first-two split and must NEVER include
    `A1_UNSEEN_SEED` (asserted defensively below -- Phase 9 guarantee)."""
    if A1_UNSEEN_SEED in a1_seen_seeds:
        raise ValueError(
            f"A1_UNSEEN_SEED={A1_UNSEEN_SEED} must never appear in the D1 training/validation "
            f"opponent pool, got a1_seen_seeds={a1_seen_seeds}"
        )
    pool: list[OpponentMember] = [OpponentMember(name="A0", kind="scripted", seed=None, policy=None)]
    for seed in a1_seen_seeds:
        model = PPO.load(str(_attacker_checkpoint_path(seed)))
        pool.append(OpponentMember(name=f"A1-seen-{seed}", kind="learned", seed=seed, policy=model))
    return pool


def sample_opponent(
    rng: np.random.Generator, pool: list[OpponentMember], weights: dict[str, float] = MIXTURE_WEIGHTS,
) -> OpponentMember:
    """Deterministic (given `rng`'s state), fixed-weight opponent draw.
    `pool` order does not matter -- weights are looked up by `member.name`."""
    probs = np.array([weights[member.name] for member in pool], dtype=np.float64)
    probs = probs / probs.sum()
    idx = int(rng.choice(len(pool), p=probs))
    return pool[idx]


def opponent_pool_spec(pool: list[OpponentMember], weights: dict[str, float] = MIXTURE_WEIGHTS) -> dict:
    """JSON-serializable description of the opponent pool, for training
    manifests (Phase 13/16)."""
    return {
        "members": [{"name": m.name, "kind": m.kind, "seed": m.seed} for m in pool],
        "weights": dict(weights),
        "a1_seen_seeds": list(A1_SEEN_SEEDS),
        "a1_unseen_seed": A1_UNSEEN_SEED,
    }


class OpponentMixtureWrapper(gym.Wrapper):
    """Samples ONE opponent per episode (Phase 7) and wires
    `SoldierEnv.enemy_tactical_override` accordingly for the whole episode.

    `get_soldier_env(env) -> SoldierEnv` is architecture-specific (PPO/
    PPO-Kalman's wrapper chain exposes the base env via `.unwrapped`;
    LR-PPO's nested composition requires `env._inner._env` -- see
    `experiments/lr_ppo_clutter_training_env.py::LRPPOSectorObservationWrapper._soldier_env`)
    so this class never assumes one specific wrapper shape.

    `info["opponent_name"]` is added to both `reset()` and `step()` info
    dicts for logging/diagnostics (Phase 16/23), never consumed by the
    defender's own observation (no information leakage about which
    opponent is active)."""

    def __init__(
        self,
        env: gym.Env,
        get_soldier_env: Callable[[gym.Env], SoldierEnv],
        pool: list[OpponentMember],
        weights: dict[str, float] = MIXTURE_WEIGHTS,
        sector_spec: SectorObservationSpec = SECTOR_OBSERVATION_V1,
    ):
        super().__init__(env)
        self._get_soldier_env = get_soldier_env
        self._pool = pool
        self._weights = weights
        self._sector_spec = sector_spec
        self._current_opponent: OpponentMember | None = None

    def _make_override(self, soldier_env: SoldierEnv, model: PPO) -> Callable[[], np.ndarray]:
        detection_range = soldier_env.config.enemy_defender_detection_range

        def _override() -> np.ndarray:
            attacker_obs = build_attacker_observation(soldier_env, detection_range, self._sector_spec)
            action, _ = model.predict(attacker_obs, deterministic=True)
            return np.asarray(action, dtype=np.float64)

        return _override

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        obs, info = self.env.reset(seed=seed, options=options)
        soldier_env = self._get_soldier_env(self.env)
        rng = np.random.default_rng(seed)
        self._current_opponent = sample_opponent(rng, self._pool, self._weights)
        if self._current_opponent.kind == "scripted":
            soldier_env.enemy_tactical_override = None
        else:
            soldier_env.enemy_tactical_override = self._make_override(soldier_env, self._current_opponent.policy)
        info = dict(info)
        info["opponent_name"] = self._current_opponent.name
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info["opponent_name"] = self._current_opponent.name if self._current_opponent is not None else None
        return obs, reward, terminated, truncated, info


def _ppo_get_soldier_env(env: gym.Env) -> SoldierEnv:
    """PPO/PPO-Kalman wrapper chain: `ObstacleSectorObservationWrapper(
    RandomizedClutterWrapper(SoldierEnv))` -- `.unwrapped` recursively
    strips ALL `gym.Wrapper` layers down to the base `SoldierEnv`."""
    soldier_env = env.unwrapped
    if not isinstance(soldier_env, SoldierEnv):
        raise TypeError(f"expected SoldierEnv, got {type(soldier_env).__name__}")
    return soldier_env


def _lr_ppo_get_soldier_env(env: gym.Env) -> SoldierEnv:
    """LR-PPO wrapper chain: `LRPPOSectorObservationWrapper(
    LeadResidualClutterTrainingEnv)`; `LeadResidualClutterTrainingEnv` is
    NOT a `gym.Wrapper` (it composes an inner env directly), so the base
    `SoldierEnv` is reached via `.env._inner._env` -- the SAME path already
    used by `LRPPOSectorObservationWrapper._soldier_env()`."""
    soldier_env = env.env._inner._env
    if not isinstance(soldier_env, SoldierEnv):
        raise TypeError(f"expected SoldierEnv, got {type(soldier_env).__name__}")
    return soldier_env


def build_ppo_d1_env(
    estimator: str,
    pool: list[OpponentMember] | None = None,
    spec: SectorObservationSpec = SECTOR_OBSERVATION_V1,
) -> gym.Env:
    """D1 training/validation env for PPO ("measurement") / PPO-Kalman
    ("kalman"): the EXACT existing `build_ppo_clutter_env` (Phase 14
    randomized clutter, unchanged) wrapped with the opponent mixture."""
    from experiments.clutter_training_env import build_ppo_clutter_env

    env = build_ppo_clutter_env(estimator=estimator, spec=spec, randomized=True)
    resolved_pool = pool if pool is not None else build_opponent_pool()
    return OpponentMixtureWrapper(env, get_soldier_env=_ppo_get_soldier_env, pool=resolved_pool, sector_spec=spec)


def build_lr_ppo_d1_env(
    root_seed: int,
    pool: list[OpponentMember] | None = None,
    spec: SectorObservationSpec = SECTOR_OBSERVATION_V1,
) -> gym.Env:
    """D1 training/validation env for LR-PPO: the EXACT existing
    `build_lr_ppo_clutter_env` (Phase 14 randomized clutter, theta_max
    unchanged) wrapped with the opponent mixture."""
    from experiments.lr_ppo_clutter_training_env import build_lr_ppo_clutter_env

    env = build_lr_ppo_clutter_env(root_seed=root_seed)
    resolved_pool = pool if pool is not None else build_opponent_pool()
    return OpponentMixtureWrapper(env, get_soldier_env=_lr_ppo_get_soldier_env, pool=resolved_pool, sector_spec=spec)
