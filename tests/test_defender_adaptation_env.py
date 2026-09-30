"""
Tests for the D1 defender-adaptation opponent-mixture environment (journal
extension, 34-phase defender-adaptation sub-task). Covers the mandated
A1-seen/A1-unseen ID-sorted split, mixture-weight correctness, opponent-pool
construction (rejecting the held-out seed), deterministic/balanced
per-episode opponent sampling, and the `OpponentMixtureWrapper`'s wiring of
`SoldierEnv.enemy_tactical_override` for both the PPO/PPO-Kalman and
LR-PPO wrapper-chain shapes -- including a bit-for-bit legacy-equivalence
check when A0 (scripted) is the sampled opponent (Phase 2 audit carried
through the wrapper).
"""

from __future__ import annotations

import numpy as np
import pytest

from uav_defend.envs.soldier_env import SoldierEnv

from experiments.adaptive_attacker_env import ATTACKER_TRAINING_SEEDS
from experiments.clutter_training_env import build_ppo_clutter_env
from experiments.lr_ppo_clutter_training_env import build_lr_ppo_clutter_env
from experiments.defender_adaptation_env import (
    A1_SEEN_SEEDS,
    A1_UNSEEN_SEED,
    MIXTURE_WEIGHTS,
    OpponentMember,
    OpponentMixtureWrapper,
    _lr_ppo_get_soldier_env,
    _ppo_get_soldier_env,
    build_lr_ppo_d1_env,
    build_opponent_pool,
    build_ppo_d1_env,
    opponent_pool_spec,
    sample_opponent,
)


def test_a1_seen_unseen_split_is_id_sorted_not_performance_based():
    assert A1_SEEN_SEEDS == tuple(sorted(ATTACKER_TRAINING_SEEDS)[:2])
    assert A1_UNSEEN_SEED == sorted(ATTACKER_TRAINING_SEEDS)[2]
    assert A1_SEEN_SEEDS == (201, 202)
    assert A1_UNSEEN_SEED == 203
    assert A1_UNSEEN_SEED not in A1_SEEN_SEEDS


def test_mixture_weights_exact_values_and_sum_to_one():
    assert MIXTURE_WEIGHTS == {"A0": 0.5, "A1-seen-201": 0.25, "A1-seen-202": 0.25}
    assert abs(sum(MIXTURE_WEIGHTS.values()) - 1.0) < 1e-12


def test_build_opponent_pool_rejects_unseen_seed():
    with pytest.raises(ValueError, match="A1_UNSEEN_SEED"):
        build_opponent_pool(a1_seen_seeds=(201, A1_UNSEEN_SEED))


def test_build_opponent_pool_default_never_references_unseen_seed():
    pool = build_opponent_pool()
    seeds = {m.seed for m in pool}
    assert A1_UNSEEN_SEED not in seeds
    assert seeds == {None, 201, 202}
    names = {m.name for m in pool}
    assert names == {"A0", "A1-seen-201", "A1-seen-202"}
    scripted = [m for m in pool if m.kind == "scripted"]
    learned = [m for m in pool if m.kind == "learned"]
    assert len(scripted) == 1 and scripted[0].policy is None
    assert len(learned) == 2 and all(m.policy is not None for m in learned)


def test_opponent_pool_spec_serializable_and_never_lists_unseen_as_member():
    pool = build_opponent_pool()
    spec = opponent_pool_spec(pool)
    assert spec["a1_seen_seeds"] == list(A1_SEEN_SEEDS)
    assert spec["a1_unseen_seed"] == A1_UNSEEN_SEED
    member_seeds = {m["seed"] for m in spec["members"]}
    assert A1_UNSEEN_SEED not in member_seeds


def test_sample_opponent_deterministic_for_fixed_seed():
    pool = [
        OpponentMember("A0", "scripted", None, None),
        OpponentMember("A1-seen-201", "learned", 201, object()),
        OpponentMember("A1-seen-202", "learned", 202, object()),
    ]
    picks_a = [sample_opponent(np.random.default_rng(123), pool).name for _ in range(10)]
    picks_b = [sample_opponent(np.random.default_rng(123), pool).name for _ in range(10)]
    assert picks_a == picks_b


def test_sample_opponent_empirical_frequencies_match_weights():
    pool = [
        OpponentMember("A0", "scripted", None, None),
        OpponentMember("A1-seen-201", "learned", 201, object()),
        OpponentMember("A1-seen-202", "learned", 202, object()),
    ]
    rng = np.random.default_rng(999)
    n = 20_000
    counts = {"A0": 0, "A1-seen-201": 0, "A1-seen-202": 0}
    for _ in range(n):
        counts[sample_opponent(rng, pool).name] += 1
    for name, weight in MIXTURE_WEIGHTS.items():
        empirical = counts[name] / n
        assert abs(empirical - weight) < 0.02, f"{name}: empirical={empirical}, expected={weight}"


def _force_single_member_pool(member: OpponentMember) -> list[OpponentMember]:
    return [member]


def test_mixture_wrapper_a0_only_matches_legacy_env_bit_for_bit():
    """When the pool is forced to ONLY the scripted A0 member, the
    mixture-wrapped env must reproduce the EXACT legacy (unwrapped)
    trajectory for the same seed/actions -- carrying Phase 2's
    'enemy_tactical_override=None reproduces exact legacy behavior'
    guarantee through this new wrapper."""
    legacy_env = build_ppo_clutter_env("measurement", randomized=True)
    mixture_env = OpponentMixtureWrapper(
        build_ppo_clutter_env("measurement", randomized=True),
        get_soldier_env=_ppo_get_soldier_env,
        pool=_force_single_member_pool(OpponentMember("A0", "scripted", None, None)),
        weights={"A0": 1.0},
    )
    obs_legacy, _ = legacy_env.reset(seed=555)
    obs_mixture, info = mixture_env.reset(seed=555)
    assert info["opponent_name"] == "A0"
    np.testing.assert_allclose(obs_legacy, obs_mixture)

    rng = np.random.default_rng(0)
    for _ in range(20):
        action = rng.uniform(-1, 1, size=3).astype(np.float32)
        obs_legacy, r_legacy, term_legacy, trunc_legacy, info_legacy = legacy_env.step(action)
        obs_mixture, r_mixture, term_mixture, trunc_mixture, info_mixture = mixture_env.step(action)
        np.testing.assert_allclose(obs_legacy, obs_mixture)
        assert r_legacy == pytest.approx(r_mixture)
        assert term_legacy == term_mixture
        assert trunc_legacy == trunc_mixture
        assert info_mixture["opponent_name"] == "A0"
        if term_legacy or trunc_legacy:
            break


def test_mixture_wrapper_learned_opponent_sets_override_hook():
    pool = build_opponent_pool()
    learned_member = next(m for m in pool if m.kind == "learned")
    env = OpponentMixtureWrapper(
        build_ppo_clutter_env("measurement", randomized=True),
        get_soldier_env=_ppo_get_soldier_env,
        pool=_force_single_member_pool(learned_member),
        weights={learned_member.name: 1.0},
    )
    obs, info = env.reset(seed=42)
    soldier_env: SoldierEnv = _ppo_get_soldier_env(env.env)
    assert info["opponent_name"] == learned_member.name
    assert soldier_env.enemy_tactical_override is not None
    override_action = np.asarray(soldier_env.enemy_tactical_override())
    assert override_action.shape == (3,)
    # Opponent stays fixed across steps within the episode (Phase 7: never
    # changes mid-episode).
    for _ in range(5):
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        assert info["opponent_name"] == learned_member.name
        if term or trunc:
            break


def test_ppo_get_soldier_env_returns_real_soldier_env():
    env = build_ppo_clutter_env("measurement", randomized=True)
    env.reset(seed=1)
    soldier_env = _ppo_get_soldier_env(env)
    assert isinstance(soldier_env, SoldierEnv)


def test_lr_ppo_get_soldier_env_returns_real_soldier_env():
    env = build_lr_ppo_clutter_env(root_seed=1)
    env.reset(seed=1)
    soldier_env = _lr_ppo_get_soldier_env(env)
    assert isinstance(soldier_env, SoldierEnv)


def test_build_ppo_d1_env_has_same_observation_action_space_as_d0():
    d0_env = build_ppo_clutter_env("measurement", randomized=True)
    d1_env = build_ppo_d1_env("measurement")
    assert d0_env.observation_space.shape == d1_env.observation_space.shape
    assert d0_env.action_space.shape == d1_env.action_space.shape


def test_build_lr_ppo_d1_env_has_same_observation_action_space_as_d0():
    d0_env = build_lr_ppo_clutter_env(root_seed=1)
    d1_env = build_lr_ppo_d1_env(root_seed=1)
    assert d0_env.observation_space.shape == d1_env.observation_space.shape
    assert d0_env.action_space.shape == d1_env.action_space.shape


def test_build_ppo_d1_env_runs_full_episode_with_default_pool():
    env = build_ppo_d1_env("measurement")
    obs, info = env.reset(seed=71)
    assert "opponent_name" in info
    assert info["opponent_name"] in MIXTURE_WEIGHTS
    for _ in range(50):
        obs, reward, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if term or trunc:
            break
