from __future__ import annotations

import math

from .application_error import ApplicationError
from .replay_contract import ReplayBundle


def validate_trace(bundle: ReplayBundle) -> None:
    if bundle.episode_count != 1 or len(bundle.episodes) != 1:
        raise ApplicationError("INVALID_REPLAY", "Replay bundles must contain exactly one episode.", 500)
    episode = bundle.episodes[0]
    if len(episode.frames) != episode.summary.steps + 1:
        raise ApplicationError("INVALID_REPLAY", "A trace must contain one initial frame plus one frame per executed step.", 500)
    if [frame.step for frame in episode.frames] != list(range(episode.summary.steps + 1)):
        raise ApplicationError("INVALID_REPLAY", "Frame indices are not contiguous.", 500)
    dt = episode.environment.dt_s
    if any(not math.isclose(frame.time_s, frame.step * dt, rel_tol=0, abs_tol=1e-12) for frame in episode.frames):
        raise ApplicationError("INVALID_REPLAY", "Frame timestamps do not match the configured simulation timestep.", 500)
    initial, final = episode.frames[0], episode.frames[-1]
    if initial.telemetry is not None or initial.reward is not None:
        raise ApplicationError("INVALID_REPLAY", "The reset frame must not contain a policy action or transition reward.", 500)
    if initial.outcome != "ongoing":
        raise ApplicationError("INVALID_REPLAY", "The reset frame must not contain a terminal outcome.", 500)
    if not (final.terminated or final.truncated):
        raise ApplicationError("INVALID_REPLAY", "The last recorded frame is not terminal or truncated.", 500)
    if final.outcome != episode.summary.outcome:
        raise ApplicationError("INVALID_REPLAY", "Final frame and episode summary outcomes disagree.", 500)
    if episode.terminal_flags != {"terminated": final.terminated, "truncated": final.truncated}:
        raise ApplicationError("INVALID_REPLAY", "Episode terminal flags disagree with the final frame.", 500)
    if not math.isclose(episode.summary.duration_s, episode.summary.steps * dt, rel_tol=0, abs_tol=1e-12):
        raise ApplicationError("INVALID_REPLAY", "Episode duration disagrees with its step count and timestep.", 500)
    if any(frame.outcome != "ongoing" for frame in episode.frames[:-1]):
        raise ApplicationError("INVALID_REPLAY", "A non-final frame contains a terminal outcome.", 500)
    detection_step = episode.summary.detection_time_step
    if detection_step is None:
        if any(frame.detection_time_step is not None for frame in episode.frames):
            raise ApplicationError("INVALID_REPLAY", "A frame records detection absent from the episode summary.", 500)
    elif any(
        frame.detection_time_step != (detection_step if frame.step >= detection_step else None)
        for frame in episode.frames
    ):
        raise ApplicationError("INVALID_REPLAY", "Frame and summary detection times disagree.", 500)
    accumulated_reward = 0.0
    for frame in episode.frames:
        if frame.step:
            if frame.reward is None or frame.cumulative_reward is None:
                raise ApplicationError("INVALID_REPLAY", "Every transition must contain reward and cumulative reward.", 500)
            accumulated_reward += frame.reward
            if not math.isclose(frame.cumulative_reward, accumulated_reward, rel_tol=0, abs_tol=1e-10):
                raise ApplicationError("INVALID_REPLAY", f"Cumulative reward is inconsistent at frame {frame.step}.", 500)
    if not math.isclose(episode.summary.total_reward, accumulated_reward, rel_tol=0, abs_tol=1e-10):
        raise ApplicationError("INVALID_REPLAY", "Episode total reward disagrees with its frame rewards.", 500)
    if any(
        any(minimum > maximum for minimum, maximum in zip(obstacle.min_corner_m, obstacle.max_corner_m))
        for obstacle in episode.static_obstacles
    ):
        raise ApplicationError("INVALID_REPLAY", "Static obstacle bounds are inverted.", 500)
    for event in episode.events:
        if event.step < 1 or event.step > episode.summary.steps:
            raise ApplicationError("INVALID_REPLAY", "An event lies outside the executed transition range.", 500)
        if not math.isclose(event.time_s, event.step * dt, rel_tol=0, abs_tol=1e-12):
            raise ApplicationError("INVALID_REPLAY", "Event timestamp disagrees with its transition index.", 500)
    if any(len(entity.position_m) != 3 for frame in episode.frames for entity in (
        frame.defender, frame.attacker, frame.asset
    )):
        raise ApplicationError("INVALID_REPLAY", "All authoritative entity positions must be native 3D.", 500)
    for frame in episode.frames[1:]:
        if frame.telemetry is None or frame.telemetry.decision_step != frame.step - 1:
            raise ApplicationError("INVALID_REPLAY", "Every transition requires telemetry for the preceding decision.", 500)
