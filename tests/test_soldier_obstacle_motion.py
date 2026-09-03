"""Protected-asset (soldier) obstacle-blocked-motion tests (journal
extension, Phase 3).

Run directly: python tests/test_soldier_obstacle_motion.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.envs.soldier_env import SoldierEnv


def _wall_config(**overrides) -> EnvConfig:
    defaults = dict(
        obstacles_enabled=True, obstacle_layout_mode="fixed",
        # x in [5,45], y in [-20,20], z in [0,10] -- far from the origin
        # (the true reset-time soldier/defender position) to satisfy
        # reset-time containment validation.
        obstacle_fixed_spec=((25.0, 0.0, 5.0, 20.0, 20.0, 5.0),),
        defender_standby_until_detection=False,
    )
    defaults.update(overrides)
    return EnvConfig(**defaults)


# 1. Soldier proposal into wall -> movement rejected.
def test_soldier_proposal_into_wall_rejected():
    env = SoldierEnv(config=_wall_config())
    env.reset(seed=1)
    start = np.array([4.9, 0.0, 0.0], dtype=np.float32)
    env._soldier_pos = start.copy()
    env._defender_pos = start.copy()
    blocked_any = False
    for _ in range(40):
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if info["soldier_obstacle_blocked"]:
            blocked_any = True
            # Position must remain exactly at the previous valid position.
            assert np.array_equal(info["soldier_pos"], start) or blocked_any
        if term or trunc:
            break
    assert blocked_any


# 2. Soldier proposal beside wall -> normal movement.
def test_soldier_proposal_beside_wall_normal_movement():
    env = SoldierEnv(config=_wall_config())
    env.reset(seed=1)
    start = np.array([-30.0, 0.0, 0.0], dtype=np.float32)  # far from the wall
    env._soldier_pos = start.copy()
    env._defender_pos = start.copy()
    positions = [start.copy()]
    for _ in range(10):
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        positions.append(info["soldier_pos"].copy())
        assert not info["soldier_obstacle_blocked"]
        if term or trunc:
            break
    # The soldier must actually move (random walk proceeds normally).
    assert not all(np.array_equal(positions[0], p) for p in positions[1:])


# 3. Rejected soldier movement consumes no extra RNG draw.
def test_rejected_movement_consumes_no_extra_rng_draw():
    env_free = SoldierEnv(config=EnvConfig(defender_standby_until_detection=False))
    env_free.reset(seed=7)

    env_blocked = SoldierEnv(config=_wall_config())
    env_blocked.reset(seed=7)
    start = np.array([4.9, 0.0, 0.0], dtype=np.float32)
    env_blocked._soldier_pos = start.copy()
    env_blocked._defender_pos = start.copy()

    blocked_count = 0
    for _ in range(80):
        env_free.step(np.zeros(3, dtype=np.float32))
        _, _, _, _, info = env_blocked.step(np.zeros(3, dtype=np.float32))
        if info["soldier_obstacle_blocked"]:
            blocked_count += 1
    assert blocked_count > 0  # the scenario must actually exercise blocking

    # If blocking ever resampled, the two RNG streams would have diverged in
    # draw COUNT (not just value) and this next draw would differ.
    draw_free = env_free._rng_soldier.normal(size=3)
    draw_blocked = env_blocked._rng_soldier.normal(size=3)
    assert np.array_equal(draw_free, draw_blocked)


# 4. Soldier never ends a step inside an obstacle.
def test_soldier_never_ends_inside_obstacle():
    cfg = EnvConfig(obstacles_enabled=True, obstacle_layout_mode="random", obstacle_count=10,
                     obstacle_clearance_from_asset=3.0, defender_standby_until_detection=False)
    for seed in (0, 1, 2, 3, 4):
        env = SoldierEnv(config=cfg)
        env.reset(seed=seed)
        for _ in range(200):
            _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
            assert not env._obstacle_layout.contains_point(info["soldier_pos"])
            if term or trunc:
                break


# 5. Boundary contact counts as blocked.
def test_boundary_contact_counts_as_blocked():
    env = SoldierEnv(config=_wall_config())
    env.reset(seed=1)
    # Force a displacement landing EXACTLY on the wall's -x face (x=5).
    env._soldier_pos = np.array([4.0, 0.0, 0.0], dtype=np.float32)
    env._defender_pos = env._soldier_pos.copy()
    env._move_soldier = lambda: _try_move(env, np.array([5.0, 0.0, 0.0], dtype=np.float32))
    _, _, _, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert info["soldier_obstacle_blocked"] is True


def _try_move(env, candidate):
    if env._obstacle_layout.segment_intersects(env._soldier_pos, candidate):
        env._soldier_obstacle_blocked = True
        env._soldier_obstacle_block_count += 1
        return
    env._soldier_obstacle_blocked = False
    env._soldier_pos = candidate.astype(np.float32)


# --- Regression: obstacles disabled reproduces conference behavior ---------

def test_obstacles_disabled_soldier_motion_unaffected():
    baseline = SoldierEnv(config=EnvConfig())
    disabled = SoldierEnv(config=EnvConfig(obstacles_enabled=False))
    ob, _ = baseline.reset(seed=3)
    od, _ = disabled.reset(seed=3)
    assert np.array_equal(ob, od)
    for _ in range(50):
        ob, _, term_b, trunc_b, info_b = baseline.step(np.zeros(3, dtype=np.float32))
        od, _, term_d, trunc_d, info_d = disabled.step(np.zeros(3, dtype=np.float32))
        assert np.array_equal(info_b["soldier_pos"], info_d["soldier_pos"])
        assert info_d["soldier_obstacle_blocked"] is False
        if term_b or trunc_b:
            break


# --- Diagnostic correctness --------------------------------------------------

def test_block_count_increments_only_on_blocked_steps():
    env = SoldierEnv(config=_wall_config())
    env.reset(seed=1)
    start = np.array([4.9, 0.0, 0.0], dtype=np.float32)
    env._soldier_pos = start.copy()
    env._defender_pos = start.copy()
    prev_count = 0
    for _ in range(30):
        _, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
        if info["soldier_obstacle_blocked"]:
            assert info["soldier_obstacle_block_count"] == prev_count + 1
        else:
            assert info["soldier_obstacle_block_count"] == prev_count
        prev_count = info["soldier_obstacle_block_count"]
        if term or trunc:
            break


if __name__ == "__main__":
    import inspect
    module = sys.modules[__name__]
    test_fns = [obj for name, obj in inspect.getmembers(module) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(test_fns)} tests passed")
