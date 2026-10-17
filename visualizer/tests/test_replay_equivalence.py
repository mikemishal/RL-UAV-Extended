import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
from visualizer.backend.config import load_settings
from visualizer.backend.data_access import ResearchDataStore
from visualizer.backend.replay_engine import ReplayRequest, _plain, run_replay_episode

RESEARCH_ROOT = os.environ.get("VISUALIZER_RESEARCH_DATA_ROOT")
pytestmark = pytest.mark.skipif(
    not RESEARCH_ROOT,
    reason="Set VISUALIZER_RESEARCH_DATA_ROOT to run the real simulator equivalence test.",
)


def test_lrppo_replay_matches_existing_matched_episode_runner():
    research_root = Path(RESEARCH_ROOT).resolve()
    sys.path.insert(0, str(research_root))
    try:
        from stable_baselines3 import PPO
        from experiments.defender_adaptation_evaluation import _run_matched_episode
        from experiments.journal_clutter_protocol import build_canonical_env_config
        from uav_defend.envs.soldier_env import SoldierEnv
        from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import (
            LeadResidualPPOPolicyWrapper,
        )

        request = ReplayRequest(
            defender_policy_id="lr_ppo",
            defender_generation="d1",
            defender_training_seed=77,
            attacker_condition="A1-203",
            attacker_model_seed=203,
            clutter_level="C3",
            perception_mode="map_local",
            evaluation_seed=0,
        )
        replay = run_replay_episode(request, load_settings()).episodes[0]
        repeated = run_replay_episode(request, load_settings()).episodes[0]
        assert repeated == replay
        defender_checkpoint = replay.defender.checkpoint
        attacker_checkpoint = replay.attacker.checkpoint
        assert defender_checkpoint is not None and attacker_checkpoint is not None
        store = ResearchDataStore(load_settings())
        defender_path = store.resolve_file(defender_checkpoint.research_relative_path)
        attacker_path = store.resolve_file(attacker_checkpoint.research_relative_path)
        assert store.read_bytes(defender_checkpoint.research_relative_path, defender_checkpoint.sha256)
        assert store.read_bytes(attacker_checkpoint.research_relative_path, attacker_checkpoint.sha256)

        config = build_canonical_env_config("C3", "measurement", awareness_mode="map_local")
        defender = LeadResidualPPOPolicyWrapper.load(
            str(defender_path),
            residual_max_angle_deg=30.0,
            config=config,
            deterministic=True,
            device="cpu",
        )
        attacker = PPO.load(str(attacker_path), device="cpu")
        env = SoldierEnv(config)
        captured = {"reset": None, "static": None, "steps": []}
        reset_original, step_original = env.reset, env.step

        def reset(*args, **kwargs):
            result = reset_original(*args, **kwargs)
            captured["reset"] = result[1]
            captured["static"] = [
                (np.asarray(obstacle.min_corner).copy(), np.asarray(obstacle.max_corner).copy())
                for obstacle in env._obstacle_layout.obstacles
            ]
            return result

        def step(action):
            result = step_original(action)
            info = result[4]
            captured["steps"].append({
                "action": np.asarray(action).copy(),
                "reward": float(result[1]),
                "terminated": bool(result[2]),
                "truncated": bool(result[3]),
                "info": info,
                "moving": [
                    (str(obstacle.id), np.asarray(obstacle.position).copy())
                    for obstacle in env._dynamic_obstacles
                ],
            })
            return result

        env.reset, env.step = reset, step
        instance = SimpleNamespace(estimator="measurement", needs_sector_features=True, is_lr_ppo=True)
        try:
            evaluation = _run_matched_episode(
                env, instance, defender, None, attacker, replay.simulation_seed
            )
        finally:
            env.close()

        frames = replay.frames
        initial = captured["reset"]
        assert len(captured["static"]) == len(replay.static_obstacles)
        assert all(
            np.array_equal(minimum, obstacle.min_corner_m)
            and np.array_equal(maximum, obstacle.max_corner_m)
            for (minimum, maximum), obstacle in zip(captured["static"], replay.static_obstacles)
        )
        for state_key, entity_key in (
            ("defender_pos", "defender"),
            ("enemy_pos", "attacker"),
            ("soldier_pos", "asset"),
        ):
            assert np.array_equal(initial[state_key], getattr(frames[0], entity_key).position_m)
        assert len(captured["steps"]) == replay.summary.steps == len(frames) - 1
        for index, transition in enumerate(captured["steps"], start=1):
            frame = frames[index]
            assert np.array_equal(transition["action"], frame.telemetry.requested_action)
            for state_key, entity_key in (
                ("defender_pos", "defender"),
                ("enemy_pos", "attacker"),
                ("soldier_pos", "asset"),
            ):
                assert np.array_equal(
                    transition["info"][state_key], getattr(frame, entity_key).position_m
                )
            assert transition["reward"] == frame.reward
            assert transition["terminated"] == frame.terminated
            assert transition["truncated"] == frame.truncated
            assert _plain(transition["info"], {}) == frame.privileged_diagnostics.simulator_info
            recorded_moving = [
                (obstacle.obstacle_id.removeprefix("dynamic-"), np.asarray(obstacle.position_m))
                for obstacle in frame.moving_obstacles
            ]
            assert len(recorded_moving) == len(transition["moving"])
            assert all(
                expected_id == actual_id and np.array_equal(expected_position, actual_position)
                for (expected_id, expected_position), (actual_id, actual_position)
                in zip(recorded_moving, transition["moving"])
            )

        assert evaluation["outcome"] == replay.summary.outcome == frames[-1].outcome
        assert evaluation["episode_length"] == replay.summary.steps
        assert evaluation["detection_time"] == replay.summary.detection_time_step
        assert sum(transition["reward"] for transition in captured["steps"]) == replay.summary.total_reward
        assert frames[-1].terminated or frames[-1].truncated
    finally:
        try:
            sys.path.remove(str(research_root))
        except ValueError:
            pass
