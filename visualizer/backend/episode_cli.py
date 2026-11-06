"""Simplified-API episode-generation subprocess entry point (brief section 4/14). Run via the configured
`VISUALIZER_SIMULATION_PYTHON` interpreter (the same interpreter the existing `replay_cli.py` uses), NEVER
imported directly by the FastAPI process (which may lack gymnasium/torch) -- see `episode_service.py`.

Unlike `replay_cli.py` (which replays TRAINED attacker checkpoints via the research `ReplayRequest`/registry
machinery), this CLI drives the research repository's two FROZEN, scripted/optimization-based attacker paths
directly: `enemy_control_mode="legacy"` (scripted A0, No-Clutter) and `enemy_control_mode="cbf_velocity"`
(A1-CBF, Clutter, `journal-cbf-attacker-v1` / commit a989217). Neither needs a trained-attacker checkpoint, so
there is no checkpoint registry/hash verification here -- only the DEFENDER may be a trained PPO-family
checkpoint, resolved via the SAME verified, frozen checkpoint-resolution helpers `replay_engine.py` already uses
(`_resolve_defender_checkpoint`/`_load_defender`), reusing a single, canonical (generation, seed) default per
architecture so the simplified API never asks the caller to choose one (brief section 3).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

VISUALIZER_ROOT = Path(__file__).resolve().parents[1]

# Canonical, frozen (generation, seed) default per PPO-family defender architecture -- the FIRST seed of the
# established "d1" (latest adaptation stage) frozen registry entry for each architecture. Never user-selectable
# through the simplified API (brief section 3: "Everything else comes from the preset").
_PPO_FAMILY_DEFAULT = {"ppo": ("d1", 71), "ppo_kalman": ("d1", 74), "lr_ppo": ("d1", 77)}


def _research_sys_path(research_root: Path) -> None:
    root_text = str(research_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)


def _build_policy_adapter(defender: str, cfg, research_root: Path):
    """Returns `(policy, estimator, needs_sector_features)`. Scripted/analytical defenders (passive, greedy,
    kalman_greedy, pn, lead, rmpc) are constructed directly, with no checkpoint. PPO-family defenders reuse the
    visualizer's OWN verified checkpoint-resolution pipeline (`replay_engine.py`) via a canonical frozen default,
    never re-deriving checkpoint paths/hashes here."""
    from uav_defend.policies.registry import get_policy

    estimator = "kalman" if defender == "kalman_greedy" else "measurement"
    if defender == "passive":
        class _Passive:
            def reset(self) -> None:
                return None

            def act(self, *_args, **_kwargs) -> np.ndarray:
                return np.zeros(3, dtype=np.float32)

        return _Passive(), estimator, False
    if defender in ("greedy", "kalman_greedy"):
        return get_policy(defender), estimator, False
    if defender == "rmpc":
        from experiments.clutter_benchmark_protocol import _frozen_rmpc_config
        from uav_defend.policies.mpc.rmpc_policy import RMPCPolicy

        return RMPCPolicy(config=cfg, rmpc_config=_frozen_rmpc_config()), estimator, False
    if defender in ("pn", "lead"):
        return get_policy(defender, config=cfg), estimator, False

    # PPO family: reuse the visualizer backend's OWN verified checkpoint resolution (sha256-checked, allowlisted
    # registry paths) -- never re-implemented here.
    sys.path.insert(0, str(VISUALIZER_ROOT.parent))
    from visualizer.backend.config import load_settings
    from visualizer.backend.data_access import ResearchDataStore
    from visualizer.backend.replay_engine import _load_defender, _read_registry, _research_modules
    from visualizer.backend.replay_requests import ReplayRequest

    generation, seed = _PPO_FAMILY_DEFAULT[defender]
    settings = load_settings()
    store = ResearchDataStore(settings)
    registry, _sha, _raw = _read_registry(store)
    modules = _research_modules(research_root)
    request = ReplayRequest(
        defender_policy_id=defender, defender_generation=generation, defender_training_seed=seed,
        attacker_condition="A0", clutter_level="C0", perception_mode="map_local", semantics="V2",
        evaluation_seed=0, deterministic_inference=True,
    )
    adapter = _load_defender(request, cfg, store, registry, modules)
    return adapter.policy, adapter.estimator, adapter.needs_sector_features


def _defender_observation(env, base_obs: np.ndarray, needs_sector_features: bool, research_root: Path) -> np.ndarray:
    if not needs_sector_features:
        return base_obs
    from uav_defend.obstacles import sector_features as sf

    features = sf.compute_sector_features(
        defender_pos=env._defender_pos, defender_vel=env._defender_vel,
        perceived=env._defender_last_perceived, sensor_range=sf.SECTOR_OBSERVATION_V1.sensor_range,
    )
    return np.concatenate([base_obs, features]).astype(np.float32)


def _run_no_clutter(defender: str, seed: int, research_root: Path) -> dict:
    from experiments import journal_v2_validation_core as VC
    from uav_defend.envs.soldier_env import SoldierEnv
    from uav_defend.policies.sanitize import build_policy_info

    cfg, estimator_hint = VC.build_config("C0", "V2", defender)
    policy, estimator, needs_sectors = _build_policy_adapter(defender, cfg, research_root)
    env = SoldierEnv(cfg)
    return _run_episode(env, policy, estimator, needs_sectors, seed, research_root, build_policy_info)


def _run_clutter(defender: str, seed: int, research_root: Path) -> dict:
    import experiments.journal_cbf_attacker_defender_foundation as F
    from uav_defend.envs.soldier_env import SoldierEnv
    from uav_defend.policies.sanitize import build_policy_info

    cfg, _estimator_hint = F.defender_config(defender)
    policy, estimator, needs_sectors = _build_policy_adapter(defender, cfg, research_root)
    env = SoldierEnv(cfg, enemy_control_mode="cbf_velocity", viability_conditioned_init=True, defender_cbf_enabled=True)
    return _run_episode(env, policy, estimator, needs_sectors, seed, research_root, build_policy_info)


def _run_episode(env, policy, estimator, needs_sectors, seed, research_root, build_policy_info) -> dict:
    obs, info = env.reset(seed=seed)
    if info.get("scenario_generation_failure"):
        raise RuntimeError("scenario_generation_failure")
    if policy is not None:
        policy.reset()
    start_pos = np.asarray(info["enemy_pos"], dtype=np.float64).copy()
    prev_pos = start_pos.copy()
    path_length = 0.0
    min_defender_sep = float("inf")
    min_obstacle_clearance = float("inf")
    obstacles = [
        {"min": [float(x) for x in o.min_corner], "max": [float(x) for x in o.max_corner]}
        for o in getattr(env, "_obstacle_layout", [])
    ] if hasattr(env, "_obstacle_layout") else []
    steps = [{
        "t": 0.0, "attacker_pos": [float(x) for x in info["enemy_pos"]],
        "defender_pos": [float(x) for x in info["defender_pos"]], "asset_pos": [float(x) for x in info["soldier_pos"]],
    }]
    done, step_count, info_final = False, 0, info
    while not done:
        action_obs = _defender_observation(env, np.asarray(obs, dtype=np.float32), needs_sectors, research_root)
        if policy is None:
            action = np.zeros(3, dtype=np.float32)
        else:
            action = np.asarray(policy.act(action_obs, build_policy_info(env._get_info(), estimator)), dtype=np.float32).reshape(3)
        obs, _r, te, tr, info = env.step(action)
        step_count += 1
        pos = np.asarray(info["enemy_pos"], dtype=np.float64)
        path_length += float(np.linalg.norm(pos - prev_pos))
        prev_pos = pos
        dist_de = float(np.linalg.norm(np.asarray(info["defender_pos"], dtype=np.float64) - pos))
        min_defender_sep = min(min_defender_sep, dist_de)
        clearance = info.get("min_physical_obstacle_clearance")
        if clearance is not None and np.isfinite(clearance):
            min_obstacle_clearance = min(min_obstacle_clearance, float(clearance))
        steps.append({
            "t": float(step_count * env.config.dt), "attacker_pos": [float(x) for x in info["enemy_pos"]],
            "defender_pos": [float(x) for x in info["defender_pos"]], "asset_pos": [float(x) for x in info["soldier_pos"]],
        })
        done = te or tr
        info_final = info
    outcome = info_final["outcome"]
    return {
        "replay": {
            "schema": "app_episode_replay/1", "outcome": outcome, "obstacles": obstacles, "steps": steps,
        },
        "outcome": outcome, "duration": float(step_count * env.config.dt), "steps": int(step_count),
        "asset_captured": outcome == "soldier_caught",
        "defender_interception": outcome in ("intercepted", "unsafe_intercept"),
        "obstacle_collision": outcome in ("enemy_static_obstacle_collision", "enemy_dynamic_obstacle_collision"),
        "min_defender_separation_m": (min_defender_sep if np.isfinite(min_defender_sep) else None),
        "min_obstacle_clearance_m": (min_obstacle_clearance if np.isfinite(min_obstacle_clearance) else None),
        "path_length_m": float(path_length),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one simplified-API (No Clutter / Clutter) episode.")
    parser.add_argument("--scenario", required=True, choices=("no_clutter", "clutter"))
    parser.add_argument("--defender", required=True, choices=("passive", "greedy", "kalman_greedy", "pn", "lead", "rmpc", "ppo", "ppo_kalman", "lr_ppo"))
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args(argv)

    research_root_raw = os.environ.get("VISUALIZER_RESEARCH_DATA_ROOT")
    if not research_root_raw:
        print(json.dumps({"error": {"code": "SIMULATION_DATA_ROOT_MISSING", "message": "VISUALIZER_RESEARCH_DATA_ROOT is not set."}}))
        return 1
    research_root = Path(research_root_raw).resolve()
    _research_sys_path(research_root)

    try:
        if args.scenario == "no_clutter":
            result = _run_no_clutter(args.defender, args.seed, research_root)
        else:
            result = _run_clutter(args.defender, args.seed, research_root)
    except Exception as exc:  # noqa: BLE001 -- single, safe error boundary for the subprocess contract
        print(json.dumps({"error": {"code": "EPISODE_FAILED", "message": str(exc)[:2000]}}))
        return 1

    print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
