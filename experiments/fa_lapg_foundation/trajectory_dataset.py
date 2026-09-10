"""
Supervised trajectory-prediction dataset generation (Phase 27-28).

NOT reinforcement learning. Each sample contains ONLY information that
would be causally available to a deployable controller at decision time:
estimated (measurement-derived) hostile position/velocity, defender
position/velocity, and protected-asset (soldier) position, plus simple
derived scalars (time since detection, hostile-defender range). Ground
truth hostile position is used ONLY as the supervised prediction target,
never as a model input.

Episode rollouts use the deployable `LeadInterceptPolicy` (measurement
state source) to generate realistic defender/hostile trajectories -- the
predictor is trained on the same causal information a deployable
controller would see, not on privileged states.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
from uav_defend.policies.sanitize import build_policy_info

# Fixed prediction horizons (seconds). All are exact multiples of the
# environment's fixed dt=0.5s, so targets align to existing simulation
# steps -- no interpolation is required (unlike the diagnostic study's
# retrospective CV/CA error analysis, which does need interpolation
# because intercept times are NOT generally dt-aligned).
HORIZONS = (0.5, 1.0, 2.0, 3.0, 4.0)

# Number of most-recent DETECTED steps of history fed to the predictor.
HISTORY_LENGTH = 8

FEATURE_NAMES = (
    "est_pos_x", "est_pos_y", "est_pos_z",
    "est_vel_x", "est_vel_y", "est_vel_z",
    "defender_pos_x", "defender_pos_y", "defender_pos_z",
    "defender_vel_x", "defender_vel_y", "defender_vel_z",
    "soldier_pos_x", "soldier_pos_y", "soldier_pos_z",
    "time_since_detection", "defender_hostile_range",
)
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass(frozen=True)
class EpisodeStep:
    step_index: int
    sim_time: float
    detected: bool
    features: np.ndarray  # shape (FEATURE_DIM,), only meaningful if detected
    true_position: np.ndarray  # shape (3,) -- SUPERVISED TARGET ONLY


@dataclass(frozen=True)
class EpisodeRecord:
    seed: int
    scenario: str
    dt: float
    steps: list[EpisodeStep] = field(default_factory=list)


def rollout_episode(env: SoldierEnv, seed: int, scenario: str) -> EpisodeRecord:
    """Roll out ONE episode with the deployable measurement-mode Lead
    policy, recording causal features and the true hostile position at
    every step. Does not alter env/policy behavior (pure observation)."""
    policy = LeadInterceptPolicy(state_source="measurement", config=env.config)
    obs, info = env.reset(seed=seed)
    policy.reset()

    steps: list[EpisodeStep] = []
    terminated = truncated = False
    step_index = 0
    steps_since_detection = -1

    while not (terminated or truncated):
        policy_info = build_policy_info(info, "measurement")
        detected = bool(info.get("enemy_detected", False))
        if detected:
            steps_since_detection = 0 if steps_since_detection < 0 else steps_since_detection + 1

        if detected:
            est_pos = np.asarray(info["enemy_measurement"], dtype=np.float64)
            est_vel = (
                np.asarray(info["enemy_measurement_velocity"], dtype=np.float64)
                if info.get("enemy_measurement_velocity_valid", False)
                else np.zeros(3)
            )
            defender_pos = np.asarray(info["defender_pos"], dtype=np.float64)
            defender_vel = np.asarray(info["defender_vel"], dtype=np.float64)
            soldier_pos = np.asarray(info["soldier_pos"], dtype=np.float64)
            hostile_range = float(np.linalg.norm(est_pos - defender_pos))
            features = np.concatenate([
                est_pos, est_vel, defender_pos, defender_vel, soldier_pos,
                [float(steps_since_detection) * env.config.dt, hostile_range],
            ])
        else:
            features = np.zeros(FEATURE_DIM)

        true_position = np.asarray(info["enemy_pos"], dtype=np.float64).copy()
        steps.append(EpisodeStep(step_index, step_index * env.config.dt, detected, features, true_position))

        action = policy.act(obs, policy_info)
        obs, _reward, terminated, truncated, info = env.step(action)
        step_index += 1

    return EpisodeRecord(seed=seed, scenario=scenario, dt=float(env.config.dt), steps=steps)


def build_samples(
    episode: EpisodeRecord,
    history_length: int = HISTORY_LENGTH,
    horizons: tuple[float, ...] = HORIZONS,
) -> list[dict]:
    """
    Slice an `EpisodeRecord` into fixed-length causal history windows with
    multi-horizon supervised targets.

    Samples are only created at DETECTED steps. `history` is zero-padded
    at the start of the detected window (mask marks real vs padded rows).
    Targets beyond the recorded episode length are OMITTED (None), never
    fabricated or extrapolated. `cv_pred[h]` is computed ONLY from the
    CURRENT step's features (no future leakage) so residual targets
    (`target - cv_pred`) can be verified independently.
    """
    dt = episode.dt
    samples: list[dict] = []
    last_index = len(episode.steps) - 1

    for i, step in enumerate(episode.steps):
        if not step.detected:
            continue

        window: list[np.ndarray] = []
        mask: list[int] = []
        for offset in range(history_length - 1, -1, -1):
            j = i - offset
            if j >= 0 and episode.steps[j].detected:
                window.append(episode.steps[j].features)
                mask.append(1)
            else:
                window.append(np.zeros(FEATURE_DIM))
                mask.append(0)

        est_pos_now = step.features[0:3]
        est_vel_now = step.features[3:6]

        targets: dict[float, np.ndarray | None] = {}
        cv_preds: dict[float, np.ndarray] = {}
        for h in horizons:
            step_offset = round(h / dt)
            if abs(step_offset * dt - h) > 1e-9:
                raise ValueError(f"horizon {h}s is not an exact multiple of dt={dt}s")
            cv_preds[h] = est_pos_now + est_vel_now * h
            j = i + step_offset
            targets[h] = episode.steps[j].true_position.copy() if j <= last_index else None

        samples.append({
            "seed": episode.seed,
            "scenario": episode.scenario,
            "step_index": step.step_index,
            "sim_time": step.sim_time,
            "history": np.stack(window, axis=0),
            "history_mask": np.array(mask, dtype=np.int64),
            "est_pos_now": est_pos_now,
            "est_vel_now": est_vel_now,
            "cv_pred": cv_preds,
            "targets": targets,
        })

    return samples


def build_dataset(seeds, scenario, config) -> list[dict]:
    """Roll out `seeds` under one scenario config and return the flattened
    list of per-step samples (Phase 27 dataset-generation pipeline)."""
    all_samples: list[dict] = []
    for seed in seeds:
        env = SoldierEnv(config=config)
        episode = rollout_episode(env, seed, scenario)
        all_samples.extend(build_samples(episode))
    return all_samples
